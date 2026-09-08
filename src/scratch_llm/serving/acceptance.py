"""S1/S-R3 — speculative-decode acceptance measured PER DRAFT POSITION, and the model it feeds.

Two things live here, and the split is the whole point of the rung.

**The instrument** (:class:`PositionAcceptance`) counts, for every draft slot ``i`` of a round,
three different things that a single "acceptance rate" collapses into one:

    offered[i]   rounds in which a draft token existed at position i at all
    reached[i]   rounds in which position i was actually TESTED against the target
    accepted[i]  rounds in which position i's draft matched the target's greedy token

``reached`` is the one an averaged number throws away. Speculative verification is a *prefix*
accept: the first mismatch ends the round, and every later position is rejected without ever
being compared (``speculative.speculative_generate``'s accept loop breaks). So position 3 can look
terrible purely because position 0 usually failed and position 3 was never reached. The two
honest quantities are therefore

    conditional[i] = accepted[i] / reached[i]   P(accept i | the prefix 0..i-1 was accepted)
    cumulative[i]  = accepted[i] / rounds       P(the round got at least i+1 tokens)

and they answer different questions. ``cumulative`` is what vLLM logs
(``vllm/v1/spec_decode/metrics.py:46,117``: it increments ``num_accepted_tokens_per_pos[i]`` for
``i < num_accepted``, then divides by ``num_drafts``). ``conditional`` is what the analytic model
below is a function of, and it is the one this rung adds — a flat profile and a steeply decaying
profile can share a mean and imply completely different break-even draft lengths.

**The model** (:func:`spec_round_model`) turns a conditional profile plus the draft/target cost
ratio into expected accepted tokens per round and a speedup against plain decode — the `# HUY:`
hole of S-R3. Everything around it (the scan over draft lengths, the break-even search, the
instrument above, and the brute-force simulator in ``tests/serving/test_s1_s_r3.py`` that checks
it) is written; the model itself is not.

Spec: experiments/S1/S-R3/spec.md
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

# =================================================================================================
# The instrument — per-position acceptance bookkeeping
# =================================================================================================


@dataclass
class PositionAcceptance:
    """Per-draft-position acceptance counters over a run of speculative rounds.

    ``draft_len`` is the maximum draft length K the run was configured with; a drafter is allowed
    to propose fewer (``NGramDrafter`` returns ``[]`` on a miss), which is why ``offered`` is a
    counter and not simply ``rounds``.

    Feed it with :meth:`observe_round` once per target verification forward. The three vectors are
    public because a bench wants to write them to JSON verbatim: a derived rate can always be
    recomputed from counts, but counts cannot be recovered from a rate.
    """

    draft_len: int
    n_rounds: int = 0
    offered: list[int] = field(default_factory=list)
    reached: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.draft_len < 1:
            raise ValueError(f"draft_len must be >= 1, got {self.draft_len}")
        for name in ("offered", "reached", "accepted"):
            vec = getattr(self, name)
            if not vec:
                setattr(self, name, [0] * self.draft_len)
            elif len(vec) != self.draft_len:
                raise ValueError(
                    f"{name} has length {len(vec)}, expected draft_len={self.draft_len}"
                )

    def observe_round(self, n_drafted: int, n_accepted: int) -> None:
        """Record one verification round: ``n_drafted`` guesses offered, ``n_accepted`` accepted.

        The accept rule is a prefix rule, so this method encodes it exactly once, here:

          * positions ``0 .. n_accepted-1`` were reached AND accepted;
          * position ``n_accepted`` — if it exists — was reached and REJECTED. It is the mismatch
            that ended the round;
          * positions after that were offered but never reached. Counting them as rejected would
            drive their conditional rate to zero for a reason that has nothing to do with the
            drafter's quality at that position, which is the exact confound this rung is built to
            remove.

        ``n_accepted == n_drafted`` (a full round) reaches every offered position and rejects none.
        """
        if not 0 <= n_accepted <= n_drafted <= self.draft_len:
            raise ValueError(
                f"need 0 <= n_accepted <= n_drafted <= draft_len; got n_accepted={n_accepted}, "
                f"n_drafted={n_drafted}, draft_len={self.draft_len}"
            )
        self.n_rounds += 1
        for i in range(n_drafted):
            self.offered[i] += 1
        for i in range(n_accepted):
            self.reached[i] += 1
            self.accepted[i] += 1
        if n_accepted < n_drafted:
            self.reached[n_accepted] += 1  # tested, mismatched — the position that ended the round

    # -- the two rates, kept apart on purpose ----------------------------------------------------

    def conditional(self) -> list[float | None]:
        """``accepted[i] / reached[i]`` — P(accept i | prefix accepted). This is the model's input.

        ``None`` where a position was never reached, never ``0.0``: "we have no evidence about
        position i" and "position i always fails" are different statements, and silently coercing
        the first into the second is how a decaying profile gets manufactured out of a short run.
        """
        return [
            (self.accepted[i] / self.reached[i]) if self.reached[i] else None
            for i in range(self.draft_len)
        ]

    def cumulative(self) -> list[float]:
        """``accepted[i] / n_rounds`` — P(the round committed at least i+1 draft tokens).

        vLLM's ``spec_decode_num_accepted_tokens_per_pos`` divided by ``num_drafts``. Reported so
        this rung's numbers are directly comparable to a vLLM log line, not because the model
        wants it.
        """
        if self.n_rounds == 0:
            return [0.0] * self.draft_len
        return [a / self.n_rounds for a in self.accepted]

    @property
    def flat_acceptance_rate(self) -> float:
        """``sum(accepted) / sum(offered)`` — the single averaged number, for the record.

        This is what ``SpecStats.acceptance_rate`` reports and what the plan's row means by "α".
        It exists here so the spec's claim is falsifiable: if this number is equal for two runs
        whose :meth:`conditional` profiles differ, and their measured speedups differ, then the
        average is provably the wrong summary and the per-position profile is the right one.
        """
        offered = sum(self.offered)
        return sum(self.accepted) / offered if offered else 0.0

    @property
    def mean_accepted_per_round(self) -> float:
        """Measured E[accepted] per round — the empirical counterpart of the model's prediction."""
        return sum(self.accepted) / self.n_rounds if self.n_rounds else 0.0


# =================================================================================================
# The model — S-R3's one hole, and the plumbing that spends its answer
# =================================================================================================


@dataclass(frozen=True)
class SpecRound:
    """What :func:`spec_round_model` says about one speculative round.

    ``expected_accepted`` counts DRAFT tokens accepted, excluding the target's own token that
    every round commits for free (``speculative.py:167-170`` commits ``pending`` plus the accepted
    prefix). ``speedup`` is against non-speculative greedy decode of the SAME target — one target
    forward per token — which is the floor S-R3 records for the spec claim.
    """

    draft_len: int
    cost_ratio: float
    expected_accepted: float
    speedup: float


def spec_round_model(alpha: Sequence[float], cost_ratio: float) -> SpecRound:
    """Expected accepted tokens and speedup for one draft round of length ``len(alpha)``.

    ``alpha[i]`` is the **conditional** acceptance probability of draft position ``i`` — P(accept
    i | positions 0..i-1 were accepted) — i.e. :meth:`PositionAcceptance.conditional`, NOT vLLM's
    cumulative per-position vector and NOT a mean over positions. Draft positions are assumed
    conditionally independent given the prefix; that assumption is the model's, and the rung's job
    is to say where it breaks.

    ``cost_ratio`` is one draft step divided by one target decode forward, both measured on the
    box (the driver prints them: ``bench/serving/s1_ladder.py --mode spec``). It is a ratio and not
    two times so that the answer is device-independent to first order.
    """
    # HUY: the analytic spec-decode speedup model — E[accepted]/round from the per-position conditional acceptance profile and the draft/target cost ratio, hence the break-even draft length — spec: experiments/S1/S-R3/spec.md — fill before S-R3
    #
    # One physical line, deliberately: infra/holes.py matches `HUY:` and `spec:` within a single
    # line, so a marker wrapped for width drops out of `make holes` and the morning inventory
    # loses this rung.
    #
    # What the answer has to account for, none of which is settled by writing down an average:
    #   * the accept rule is a PREFIX rule. Position i is only ever tested if 0..i-1 were accepted
    #     (speculative.py:159-164). A round's outcome is therefore a function of the whole profile
    #     up to the first failure, and two profiles with the same mean are not interchangeable.
    #   * a round commits the target's own token whether or not any draft survives, so the
    #     no-acceptance case is not the zero case.
    #   * the cost side: len(alpha) draft steps at `cost_ratio` each, serial, plus ONE target
    #     forward. Whether that forward costs the same as a decode forward is a claim about the
    #     shape, not an identity: it verifies 1 + len(alpha) positions, so it is a q_len = K+1
    #     attention and a K+1-row GEMM. At small K on a memory-bound decode it is close; deciding
    #     how close, and whether the model should carry a term for it, is part of the rung.
    #   * the break-even follows: `speedup_curve` below scans this function over k = 1..len(alpha)
    #     and `break_even_draft_length` reads the crossing off it. If the model has no k at which
    #     speedup falls back below 1, that is a statement about the cost side being too generous.
    #
    # Falsify it before quoting it: tests/serving/test_s1_s_r3.py brute-forces every accept/reject
    # pattern of a random profile and compares. That simulator is independent of whatever closed
    # form goes here — it enumerates outcomes and applies the prefix rule procedurally — so it
    # will catch a formula that is right on flat profiles and wrong on shaped ones.
    #
    # The sentinel tests/conftest.py greps for is the contiguous `NotImplementedError("HUY:` — so
    # the message opens on the next line and continues by implicit concatenation.
    raise NotImplementedError(
        "HUY: the speculative-decode speedup model is unwritten. Take the conditional "
        "per-position profile from PositionAcceptance.conditional() and the measured "
        "draft/target cost ratio, derive E[accepted] per round and the speedup against "
        "one-forward-per-token decode, then check it against the brute-force simulator in "
        "tests/serving/test_s1_s_r3.py before quoting a break-even K. "
        "Spec: experiments/S1/S-R3/spec.md"
    )


def speedup_curve(alpha: Sequence[float], cost_ratio: float) -> list[SpecRound]:
    """:func:`spec_round_model` evaluated at every draft length ``k = 1 .. len(alpha)``.

    A scan over the model, not a second model: truncating the profile to its first ``k`` entries
    is what "drafting only k tokens" means, because ``alpha[i]`` is already conditional on the
    prefix and so does not change when the positions after it are dropped.
    """
    if not alpha:
        raise ValueError("alpha must be non-empty — a draft round of length 0 is plain decode")
    return [spec_round_model(alpha[:k], cost_ratio) for k in range(1, len(alpha) + 1)]


def break_even_draft_length(alpha: Sequence[float], cost_ratio: float) -> int:
    """The largest ``k`` whose round still beats plain decode; ``0`` if speculation never pays.

    "Break-even" is the crossing, not the optimum: the curve can peak at k=3 and stay above 1.0
    until k=7, and an engine that sets K to the crossing is paying draft cost for nothing. The
    optimum is ``max(speedup_curve(...), key=lambda r: r.speedup)`` and the driver prints both.
    """
    curve = speedup_curve(alpha, cost_ratio)
    winners = [r.draft_len for r in curve if r.speedup >= 1.0]
    return max(winners) if winners else 0
