"""S1/S-R2 — the quality gate that decides whether the speedup is allowed to be a number.

Spec: ``experiments/S1/S-R2/spec.md``

A serving rung whose only output is "FP8 is faster" has measured nothing. FP8 is *always* faster:
it reads half the weight bytes and half the cache bytes. The whole content of the rung is whether
the model that came out the other side is still the model — and that question is decided by a
threshold, against a baseline, on a named corpus and a named 200-item slice. This module assembles
the evidence and applies the threshold. It does not choose the threshold.

The evidence, and why it has this exact shape
---------------------------------------------
Two arms, same weights, same shape, same seed: bf16 (the floor) and FP8 (the rung).

  * **Perplexity.** Both arms score the *same* held-out token stream, so the comparison is paired
    at the token level and the corpus's own difficulty cancels. Stored as the summed negative
    log-likelihood and the token count rather than as a bare ppl, because ``exp(nll/n)`` is not
    linear: an average of per-window perplexities is a different number from the perplexity of the
    stream, and only the second one is comparable across window sizes.
  * **GSM8K-200, paired.** Both arms answer the *same* 200 problems, so the right summary is not
    two independent accuracies but the two **discordant counts**: how many items only bf16 got
    right (``b``), and how many only FP8 got right (``c``). Items both arms answer the same way
    carry no information about a difference between the arms, and an unpaired binomial standard
    error on 200 samples silently throws that pairing away — it asks "could these two accuracies
    come from the same coin?" when the question is "did quantization change *these* answers?".
    :class:`QualityEvidence` therefore reports ``b`` and ``c`` and takes no position on which noise
    model is right. Choosing it is the hole below.

Nothing in this module is a measurement. The arms are run by ``bench/s1_fp8_serving.py``, which
prints the speedup only when :func:`verdict` says ``PASS``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "VERDICTS",
    "GateThresholds",
    "QualityEvidence",
    "gate_thresholds",
    "verdict",
]


@dataclass(frozen=True)
class QualityEvidence:
    """Everything the gate reads, and nothing it does not.

    ``bf16_nll_nats`` / ``fp8_nll_nats`` are summed teacher-forced negative log-likelihoods over
    ``n_ppl_tokens`` transitions of ``corpus``; ``bf16_correct`` / ``fp8_correct`` are per-item
    booleans over the *same* ``gsm8k_slice``, in the same order.
    """

    corpus: str
    n_ppl_tokens: int
    bf16_nll_nats: float
    fp8_nll_nats: float
    gsm8k_slice: str
    bf16_correct: tuple[bool, ...]
    fp8_correct: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.n_ppl_tokens <= 0:
            raise ValueError("n_ppl_tokens must be positive")
        if len(self.bf16_correct) != len(self.fp8_correct):
            raise ValueError(
                f"the two arms must answer the same items: {len(self.bf16_correct)} bf16 flags vs "
                f"{len(self.fp8_correct)} fp8 flags — an unpaired comparison is not this gate"
            )
        if not self.bf16_correct:
            raise ValueError("gsm8k flags are empty")

    @classmethod
    def from_paired(
        cls,
        *,
        corpus: str,
        n_ppl_tokens: int,
        bf16_nll_nats: float,
        fp8_nll_nats: float,
        gsm8k_slice: str,
        bf16_correct: Sequence[bool],
        fp8_correct: Sequence[bool],
    ) -> QualityEvidence:
        return cls(
            corpus=corpus,
            n_ppl_tokens=n_ppl_tokens,
            bf16_nll_nats=bf16_nll_nats,
            fp8_nll_nats=fp8_nll_nats,
            gsm8k_slice=gsm8k_slice,
            bf16_correct=tuple(bool(x) for x in bf16_correct),
            fp8_correct=tuple(bool(x) for x in fp8_correct),
        )

    @property
    def n_items(self) -> int:
        return len(self.bf16_correct)

    @property
    def bf16_ppl(self) -> float:
        """``exp(total nll / tokens)`` — the perplexity of the stream, not a mean of perplexities."""
        return math.exp(self.bf16_nll_nats / self.n_ppl_tokens)

    @property
    def fp8_ppl(self) -> float:
        return math.exp(self.fp8_nll_nats / self.n_ppl_tokens)

    @property
    def ppl_rel_increase(self) -> float:
        """``fp8_ppl / bf16_ppl - 1``: the dimensionless quantity a "Δppl" threshold is about.

        Reported as a ratio rather than as an absolute ppl difference on purpose. Perplexity is
        exponential in the per-token loss, so an absolute gap means something different at ppl 6
        than at ppl 60, and the threshold would silently change meaning with the corpus.
        """
        return self.fp8_ppl / self.bf16_ppl - 1.0

    @property
    def bf16_accuracy(self) -> float:
        return sum(self.bf16_correct) / self.n_items

    @property
    def fp8_accuracy(self) -> float:
        return sum(self.fp8_correct) / self.n_items

    @property
    def accuracy_drop(self) -> float:
        """``bf16_accuracy - fp8_accuracy``, in accuracy points. Positive means FP8 lost ground."""
        return self.bf16_accuracy - self.fp8_accuracy

    @property
    def discordant(self) -> tuple[int, int]:
        """``(b, c)``: items only bf16 got right, items only FP8 got right.

        The paired statistic. ``b + c`` is the number of items that carry any information about a
        difference between the arms; the concordant items do not, however many there are.
        """
        b = sum(x and not y for x, y in zip(self.bf16_correct, self.fp8_correct, strict=True))
        c = sum(y and not x for x, y in zip(self.bf16_correct, self.fp8_correct, strict=True))
        return int(b), int(c)


@dataclass(frozen=True)
class GateThresholds:
    """The two numbers the rung is judged by. Both are Huy's; see :func:`gate_thresholds`.

    ``max_ppl_rel_increase`` bounds :attr:`QualityEvidence.ppl_rel_increase`.
    ``max_accuracy_drop`` bounds :attr:`QualityEvidence.accuracy_drop`, in accuracy points.
    """

    max_ppl_rel_increase: float
    max_accuracy_drop: float

    def __post_init__(self) -> None:
        if self.max_ppl_rel_increase < 0.0 or self.max_accuracy_drop < 0.0:
            raise ValueError(
                "a negative threshold demands that FP8 be strictly better than bf16, which is not "
                "a quality gate — it is a different experiment"
            )


# ---------------------------------------------------------------------------------------------
# THE HOLE — the tolerance and its argument. CLAUDE.md names tolerance design as never delegated,
# and here the delegation would be worse than usual: the plan (§05, quoted in the rung's spec.md)
# states this gate in words, and words are not a threshold. Turning the two phrases into two
# numbers, against this model's own baseline, is the whole intellectual content of the rung.
# ---------------------------------------------------------------------------------------------


def gate_thresholds(evidence: QualityEvidence) -> GateThresholds | None:
    """The Δppl and GSM8K-200 thresholds S-R2 is judged against. ``None`` means "not yet chosen".

    Takes the evidence, and the signature is part of the argument. A threshold on Δppl is a
    fraction *of this model's measured bf16 perplexity on this corpus*, and "within noise" on 200
    paired items is a function of the observed discordant counts — neither is knowable before the
    floor arm has run, so a function that could only return a constant would be the wrong shape for
    the decision. What comes back is still a pair of bounds, because :func:`verdict` compares
    scalars; what the evidence buys is the right to derive them.

    Four things must be decided together, and only the person who will defend the verdict can
    decide them:

      * **What "Δppl" is a Δ of.** A ratio on perplexity, an absolute difference in nats per token,
        or a difference in bits per byte? They are monotone in each other but not equal, and the
        plan's phrasing does not pick one. ``QualityEvidence`` exposes the ratio because it is
        corpus-invariant; if the argument runs through nats instead, the field to bound changes.
      * **Against which baseline number.** The threshold is a fraction *of this model's bf16
        perplexity on this corpus*. That number is not known until the floor arm has run, so the
        threshold cannot be a constant chosen in the abstract — it is chosen once the baseline is
        on the table, and the choice has to survive the corpus being swapped.
      * **What "within noise" means on 200 paired items.** The evidence gives the discordant counts
        ``(b, c)``. A paired test (McNemar / an exact binomial on ``b + c``) and an unpaired
        two-proportion test give materially different intervals at ``n = 200``, and the difference
        is largest exactly where the gate lives — small drops. Which one, at which one- or
        two-sided level, is the decision; ``max_accuracy_drop`` is where its answer lands.
      * **The false-negative budget.** This gate will be run again on every later serving rung. A
        threshold tight enough to fire on sampling jitter gets widened once and then ignored; one
        loose enough never to fire is a rubber stamp on a broken cache.

    Under ``LADDERS_STUB_HOLES=1`` this returns ``None`` so the surrounding plumbing (both arms,
    the evidence assembly, the runner, the JSON) can be exercised end to end on a CPU. ``None``
    does not mean "everything passes": :func:`verdict` maps it to ``UNJUDGED``, never to ``PASS``,
    and ``infra/bench.sh`` refuses to run at all while that flag is set.
    """
    if os.environ.get("LADDERS_STUB_HOLES") == "1":
        return None  # the stub ignores the evidence on purpose: it judges nothing
    # HUY: the Δppl and GSM8K-200 gate thresholds — spec: experiments/S1/S-R2/spec.md — fill before S-R2
    raise NotImplementedError("HUY: S-R2 gate thresholds unset — see the rung's spec.md")


VERDICTS = ("PASS", "FAIL", "UNJUDGED")


def verdict(evidence: QualityEvidence, thresholds: GateThresholds | None) -> str:
    """One evidence bundle plus one threshold pair -> one of :data:`VERDICTS`.

    ``UNJUDGED`` is a first-class answer, not an error state, and it is deliberately *not* ``PASS``:
    a run made with no threshold has produced no verdict, and letting it read as a pass is exactly
    the failure this gate exists to prevent — a speedup shipped because nobody had written down
    what quality it was allowed to cost.
    """
    if thresholds is None:
        return "UNJUDGED"
    if evidence.ppl_rel_increase > thresholds.max_ppl_rel_increase:
        return "FAIL"
    if evidence.accuracy_drop > thresholds.max_accuracy_drop:
        return "FAIL"
    return "PASS"
