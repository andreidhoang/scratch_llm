"""The arm x seed matrix: one data order per seed, one eval stream, one token budget.

A two-arm training comparison is only a comparison if the arms differ in exactly one thing. Four
things have to be nailed down for that to be true, and each of them has a way of drifting that
produces a plausible number rather than an error:

1. **Data order.** :func:`batch_schedule` draws every batch's start offsets from a generator seeded
   by the run seed alone (``np.random.Generator(PCG64(seed))``), ahead of the loop, into an array.
   ``train.get_batch`` draws from the *global* numpy RNG at the point of use, so any arm that
   consumes global randomness on the way — a quantization recipe drawing a scale, a dropout mask,
   an extra ``torch.randn`` — shifts its own data stream and the two arms stop being the same
   experiment. The schedule cannot drift that way: nothing between the seed and the array.
2. **Token budget.** Fixed in TOKENS, with ``max_steps`` derived (:func:`steps_for_tokens`), and a
   non-integral budget refused. An arm that ran 1% longer is a better model for a reason that has
   nothing to do with its dtype.
3. **Eval stream.** Fixed sequential windows, the same rule ``train._val_loss`` uses, hashed into
   :attr:`EvalStream.digest`. Both arms score the same bytes or the bpb difference is a difference
   of test sets.
4. **The arm configs themselves.** :func:`differing_fields` diffs them, and the runner asserts the
   difference is the one field the rung is about.

:func:`compare` re-checks 1-3 from the recorded results rather than trusting the runner, because
the failure this whole file exists to prevent is a matrix that was assembled correctly on Monday
and re-run with one flag changed on Tuesday.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from scratch_llm.training.bpb import (
    BpbAccounting,
    account,
    bits_per_byte,
)
from scratch_llm.training.seed_noise import SeedSigma, bpb_gate_tolerance, seed_sigma

# ---------------------------------------------------------------------------------------------
# 1. data order — a pure function of the seed
# ---------------------------------------------------------------------------------------------


def batch_schedule(
    seed: int,
    *,
    steps: int,
    batch_size: int,
    context_length: int,
    corpus_len: int,
    world_size: int = 1,
    rank: int = 0,
) -> np.ndarray:
    """``(steps, batch_size)`` window start offsets for one rank — a function of ``seed`` alone.

    The draw is for the whole GLOBAL batch and the rank takes its column slice, so the union over
    ranks is exactly the single-process stream and changing ``world_size`` re-shards the same data
    rather than resampling it. (``train.train``'s distributed branch instead offsets the global
    numpy seed by ``1000*rank``, which makes the stream a function of the topology; a bf16 run on
    8 ranks and an fp8 run on 8 ranks agree there too, but a re-run at a different world size does
    not, and neither can be compared to a single-process debug run.)
    """
    if steps <= 0 or batch_size <= 0 or world_size <= 0:
        raise ValueError(
            f"steps={steps} batch_size={batch_size} world_size={world_size} must be >0"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside world_size {world_size}")
    max_start = corpus_len - context_length - 1
    if max_start < 1:
        raise ValueError(
            f"corpus of {corpus_len} tokens too short for context_length={context_length}"
        )
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, max_start + 1, size=(steps, world_size * batch_size), dtype=np.int64)
    return np.ascontiguousarray(starts[:, rank * batch_size : (rank + 1) * batch_size])


def digest(array: np.ndarray) -> str:
    """A short content hash of an integer array — shape and dtype included, so a reshaped or
    re-typed schedule is a different digest and not a silent match."""
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str((arr.shape, arr.dtype.str)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def batches_from_schedule(
    data: np.ndarray,
    schedule: np.ndarray,
    context_length: int,
    device: str = "cpu",
) -> Callable[[int], tuple[Tensor, Tensor]]:
    """A ``batch_fn(step) -> (inputs, targets)`` reading the windows ``schedule`` names.

    Next-token alignment identical to ``train.get_batch``: ``targets[b, t] == inputs[b, t+1]``.
    """

    def batch_fn(step: int) -> tuple[Tensor, Tensor]:
        starts = schedule[step % schedule.shape[0]]
        inputs = np.stack([data[s : s + context_length] for s in starts])
        targets = np.stack([data[s + 1 : s + 1 + context_length] for s in starts])
        return (
            torch.from_numpy(inputs).long().to(device),
            torch.from_numpy(targets).long().to(device),
        )

    return batch_fn


# ---------------------------------------------------------------------------------------------
# 2. token budget
# ---------------------------------------------------------------------------------------------


def steps_for_tokens(
    tokens: int, *, batch_size: int, context_length: int, world_size: int = 1
) -> int:
    """``tokens / (batch*ctx*world)``, refusing a budget that is not a whole number of steps.

    Rounding here is how two arms end up trained on different token counts: floor one and ceil the
    other and the gate is comparing a longer run to a shorter one.
    """
    per_step = batch_size * context_length * world_size
    if tokens % per_step:
        raise ValueError(
            f"token budget {tokens} is not a whole number of steps at "
            f"batch*ctx*world = {per_step} (nearest: {tokens // per_step * per_step} or "
            f"{(tokens // per_step + 1) * per_step})"
        )
    return tokens // per_step


# ---------------------------------------------------------------------------------------------
# 3. eval stream
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalStream:
    """Fixed sequential windows of the val corpus — deliberately no RNG, so scoring consumes no
    random state and both arms score identical bytes. Same window rule as ``train._val_loss``."""

    inputs: np.ndarray
    targets: np.ndarray
    digest: str

    @property
    def n_tokens(self) -> int:
        return int(self.targets.size)


def eval_stream(val_data: np.ndarray, context_length: int, n_windows: int) -> EvalStream:
    """The first ``n_windows`` non-overlapping windows of ``val_data``."""
    available = (len(val_data) - 1) // context_length
    if available < 1:
        raise ValueError(
            f"val corpus of {len(val_data)} tokens too short for context_length={context_length}"
        )
    n = min(n_windows, available)
    starts = [i * context_length for i in range(n)]
    inputs = np.stack([np.asarray(val_data[s : s + context_length]) for s in starts])
    targets = np.stack([np.asarray(val_data[s + 1 : s + 1 + context_length]) for s in starts])
    return EvalStream(inputs=inputs, targets=targets, digest=digest(targets))


@torch.no_grad()
def eval_bpb(
    model: torch.nn.Module,
    stream: EvalStream,
    lengths: np.ndarray,
    *,
    device: str = "cpu",
    zero_byte_ids: Iterable[int] = (),
    windows_per_forward: int = 4,
) -> tuple[float, float, BpbAccounting]:
    """``(bpb, mean_ce_nats, accounting)`` for ``model`` on ``stream``.

    fp32, no autocast, ``model.eval()`` — the same regime for every arm. The arms differ in how
    they were TRAINED; scoring them under different precisions would fold an inference effect into
    a training result and neither term would be recoverable afterwards.
    """
    was_training = model.training
    model.eval()
    total_nats = 0.0
    n = 0
    try:
        for lo in range(0, stream.inputs.shape[0], windows_per_forward):
            inp = torch.from_numpy(stream.inputs[lo : lo + windows_per_forward]).long().to(device)
            tgt = torch.from_numpy(stream.targets[lo : lo + windows_per_forward]).long().to(device)
            logits = model(inp).float()
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum"
            )
            total_nats += float(nll)
            n += int(tgt.numel())
    finally:
        if was_training:
            model.train()
    acc = account(stream.targets, lengths, zero_byte_ids=zero_byte_ids)
    if acc.n_tokens != n:
        raise AssertionError(f"scored {n} transitions but accounted {acc.n_tokens}")
    return bits_per_byte(total_nats, acc), total_nats / n, acc


# ---------------------------------------------------------------------------------------------
# 4. the arms, and the one field between them
# ---------------------------------------------------------------------------------------------


def differing_fields(a: object, b: object) -> list[str]:
    """Names of the dataclass fields on which ``a`` and ``b`` disagree."""
    if not (is_dataclass(a) and is_dataclass(b)) or type(a) is not type(b):
        raise TypeError(f"differing_fields needs two dataclasses of one type, got {a!r} {b!r}")
    return [f.name for f in fields(a) if getattr(a, f.name) != getattr(b, f.name)]


def assert_one_variable(a: object, b: object, expected: str) -> None:
    """Refuse a pair of arm configs that differ anywhere but ``expected``."""
    diff = differing_fields(a, b)
    if diff != [expected]:
        raise ValueError(
            f"the arms must differ in exactly one field ({expected!r}); they differ in {diff} — "
            "any second difference is confounded with the first and the delta attributes to both"
        )


# ---------------------------------------------------------------------------------------------
# 5. timing
# ---------------------------------------------------------------------------------------------


class StepTimer:
    """Wraps a ``batch_fn`` to time the training loop from inside it.

    ``batch_fn`` is called once per step at the top of the loop, so the interval between two calls
    is one whole step. That makes it a per-step timer with no second hook into ``train()`` — at the
    cost of one detail that must be stated: interval *k* covers step *k-1*, the first call starts
    the clock and times nothing, and the final step is never closed. Run one extra step and drop
    the warm-up prefix. ``sync`` (``torch.cuda.synchronize``) closes the async gap; without it the
    intervals measure launch queueing.
    """

    def __init__(self, batch_fn: Callable[[int], tuple[Tensor, Tensor]], *, device: str = "cpu"):
        self._batch_fn = batch_fn
        self._cuda = "cuda" in str(device)
        self._last: float | None = None
        self.intervals_s: list[float] = []

    def __call__(self, step: int) -> tuple[Tensor, Tensor]:
        if self._cuda:
            torch.cuda.synchronize()
        now = perf_counter()
        if self._last is not None:
            self.intervals_s.append(now - self._last)
        self._last = now
        return self._batch_fn(step)

    def summary(self, *, tokens_per_step: int, warmup: int, min_timed: int) -> dict[str, float]:
        """Median tok/s and the p25-p75 spread, over the intervals after ``warmup``."""
        timed = self.intervals_s[warmup:]
        if len(timed) < min_timed:
            raise ValueError(
                f"{len(timed)} timed steps after {warmup} warm-ups, need {min_timed} "
                "(workspace invariant 4: >= 50 iterations, median + IQR)"
            )
        q = statistics.quantiles(timed, n=4)
        med = statistics.median(timed)
        return {
            "tok_s": tokens_per_step / med,
            "tok_s_p25": tokens_per_step / q[2],
            "tok_s_p75": tokens_per_step / q[0],
            "step_s_median": med,
            "step_s_iqr": q[2] - q[0],
            "n_timed": float(len(timed)),
        }


# ---------------------------------------------------------------------------------------------
# 6. the matrix
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmResult:
    """One cell of the matrix: one arm at one seed."""

    arm: str
    seed: int
    tokens: int
    steps: int
    bpb: float
    mean_ce: float
    tok_s: float
    tok_s_p25: float
    tok_s_p75: float
    data_digest: str
    eval_digest: str
    accounting: BpbAccounting
    converted_modules: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "tokens": self.tokens,
            "steps": self.steps,
            "bpb": self.bpb,
            "mean_ce": self.mean_ce,
            "tok_s": self.tok_s,
            "tok_s_p25": self.tok_s_p25,
            "tok_s_p75": self.tok_s_p75,
            "data_digest": self.data_digest,
            "eval_digest": self.eval_digest,
            "converted_modules": self.converted_modules,
            "accounting": self.accounting.as_dict(),
        }


@dataclass(frozen=True)
class Comparison:
    """Everything the gate needs, and nothing that decides it."""

    baseline: str
    treatment: str
    tokens: int
    seeds: tuple[int, ...]
    baseline_sigma: SeedSigma
    treatment_sigma: SeedSigma
    paired_deltas: tuple[float, ...]
    delta_bpb_means: float
    delta_bpb_paired: float
    baseline_tok_s: float
    treatment_tok_s: float

    @property
    def speedup(self) -> float:
        """Treatment tok/s over baseline tok/s at matched tokens. The rung's first claim."""
        return self.treatment_tok_s / self.baseline_tok_s

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "treatment": self.treatment,
            "tokens": self.tokens,
            "seeds": list(self.seeds),
            "baseline_sigma": self.baseline_sigma.as_dict(),
            "treatment_sigma": self.treatment_sigma.as_dict(),
            "paired_deltas": list(self.paired_deltas),
            "delta_bpb_means": self.delta_bpb_means,
            "delta_bpb_paired": self.delta_bpb_paired,
            "baseline_tok_s": self.baseline_tok_s,
            "treatment_tok_s": self.treatment_tok_s,
            "speedup": self.speedup,
        }


def compare(results: Sequence[ArmResult], *, baseline: str, treatment: str) -> Comparison:
    """Assemble the matrix, refusing every way it could stop being one experiment.

    Refusals, in the order a tired operator trips them: a missing or duplicated cell; two arms at
    different token budgets; the same seed given different data in the two arms; two eval streams;
    two byte-accounting conventions. Each is a ValueError naming both sides — a comparison that
    quietly proceeds on mismatched inputs is worse than no comparison, because it produces a
    number that looks like the one you wanted.
    """
    cells: dict[tuple[str, int], ArmResult] = {}
    for r in results:
        key = (r.arm, r.seed)
        if key in cells:
            raise ValueError(f"duplicate result for arm {r.arm!r} seed {r.seed}")
        cells[key] = r
    arms = {baseline, treatment}
    got = {a for a, _ in cells}
    if got != arms:
        raise ValueError(f"expected arms {sorted(arms)}, got {sorted(got)}")
    seeds = tuple(sorted({s for _, s in cells}))
    missing = [(a, s) for a in sorted(arms) for s in seeds if (a, s) not in cells]
    if missing:
        raise ValueError(f"incomplete matrix: no result for {missing}")

    tokens = {r.tokens for r in results}
    if len(tokens) != 1:
        raise ValueError(
            f"arms were trained on different token counts {sorted(tokens)} — at a fixed-token "
            "comparison the budget IS the control; the longer run is better for a reason that has "
            "nothing to do with the arm"
        )
    evals = {r.eval_digest for r in results}
    if len(evals) != 1:
        raise ValueError(f"arms scored different eval streams {sorted(evals)}")
    conventions = {(r.accounting.n_bytes, r.accounting.zero_byte_ids) for r in results}
    if len(conventions) != 1:
        raise ValueError(
            f"arms used different byte accounting {sorted(map(str, conventions))} — the bpb "
            "denominators differ, so the two bpb numbers are not on one scale"
        )
    for s in seeds:
        a, b = cells[(baseline, s)], cells[(treatment, s)]
        if a.data_digest != b.data_digest:
            raise ValueError(
                f"seed {s}: arms saw different data orders ({a.data_digest} vs {b.data_digest}) — "
                "the pairing is void and the delta carries a data-order term"
            )

    base = [cells[(baseline, s)] for s in seeds]
    treat = [cells[(treatment, s)] for s in seeds]
    base_sigma = seed_sigma([r.bpb for r in base])
    treat_sigma = seed_sigma([r.bpb for r in treat])
    paired = tuple(t.bpb - b.bpb for b, t in zip(base, treat, strict=True))
    return Comparison(
        baseline=baseline,
        treatment=treatment,
        tokens=next(iter(tokens)),
        seeds=seeds,
        baseline_sigma=base_sigma,
        treatment_sigma=treat_sigma,
        paired_deltas=paired,
        delta_bpb_means=treat_sigma.mean - base_sigma.mean,
        delta_bpb_paired=sum(paired) / len(paired),
        baseline_tok_s=statistics.median([r.tok_s for r in base]),
        treatment_tok_s=statistics.median([r.tok_s for r in treat]),
    )


@dataclass(frozen=True)
class Verdict:
    tolerance: float
    delta: float
    passed: bool


def verdict(comparison: Comparison, *, paired: bool = True) -> Verdict:
    """Apply Huy's tolerance. The ONLY caller of :func:`bpb_gate_tolerance`, so every statistic
    above stays computable — and printable — while the hole is open."""
    tol = bpb_gate_tolerance(comparison.baseline_sigma)
    delta = comparison.delta_bpb_paired if paired else comparison.delta_bpb_means
    return Verdict(tolerance=tol, delta=delta, passed=abs(delta) <= tol)


__all__ = [
    "ArmResult",
    "Comparison",
    "EvalStream",
    "StepTimer",
    "Verdict",
    "assert_one_variable",
    "batch_schedule",
    "batches_from_schedule",
    "compare",
    "differing_fields",
    "digest",
    "eval_bpb",
    "eval_stream",
    "steps_for_tokens",
    "verdict",
]
