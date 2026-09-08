"""Cross-rank gradient divergence — the detector for a bit flipped on one rank.

**The precondition is the whole design.** Call this on POST-reduction gradients. Before the
all-reduce every rank holds a different batch's gradients, and they differ by percent, so the only
available comparison is statistical and the natural batch-to-batch spread swamps anything short of
a sign or exponent flip. After the all-reduce every rank is *required* to hold bitwise-identical
values — measured, not assumed: gloo's and NCCL's ring algorithms are rank-symmetric, so
``all_gather``ing the reduced buffer and comparing it byte for byte is a zero-false-positive test.
That turns the detector from "is this rank's norm suspiciously large" into "do the ranks agree",
which needs no threshold at all. Nothing in this module has a tolerance.

Two channels, with different reach:

* **norm** — each rank's global grad norm, accumulated in ``accum_dtype``, compared bitwise across
  ranks. This is the detection the T-R3 row names. It is a *lossy* reduction: a flip low enough in
  the mantissa vanishes into the rounding of the sum and the ranks agree on a corrupted value. The
  boundary is derivable — :func:`min_detectable_mantissa_bit` — and it moves by ~29 bits between an
  fp32 and an fp64 accumulator, which is why the accumulator dtype is an argument and not a detail.
* **digest** — a cryptographic digest of the raw gradient bytes, compared across ranks. Lossless,
  so it catches *any* single-bit flip including mantissa bit 0 of a 70B model, where the norm
  channel is blind by ~5 bits. Costs a host copy per step, so it is the audit channel, not the
  every-step one.

**The blind spot, stated:** a flip that lands BEFORE the reduction is invisible to both channels.
The all-reduce propagates the corruption to every rank, they agree perfectly, and the run continues
with a wrong gradient. No cross-rank comparison can see it. That is what deterministic replay is
for (``oss/torchtitan/torchtitan/observability/sdc_replayer.py``), and there is a test here that
pins the blind spot so it stays a known limitation rather than becoming a surprise.

**Under ZeRO-2 there is no such thing as "the post-reduction gradient on every rank."**
``DistMuonAdamW`` reduce-scatters, so each rank owns a different shard and cross-rank comparison
would compare different data. The analogue there is the all-gathered *parameters*, which must be
identical — ``scratch_llm.utils.dist_train.assert_model_replicas_identical`` already does that.
This detector is for the DDP-shaped (all-reduce) path.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from scratch_llm.faults.inject import mantissa_bits

# Unit roundoff (half an ulp, relative) of each accumulator dtype — the whole boundary argument.
_UNIT_ROUNDOFF: dict[torch.dtype, float] = {
    torch.float64: 2.0**-53,
    torch.float32: 2.0**-24,
    torch.float16: 2.0**-11,
    torch.bfloat16: 2.0**-8,
}


def min_detectable_mantissa_bit(
    element: float,
    norm: float,
    *,
    value_dtype: torch.dtype = torch.float32,
    accum_dtype: torch.dtype = torch.float64,
) -> int:
    """Lowest mantissa bit whose flip still changes the accumulated grad norm. The mechanism:

    Flipping mantissa bit ``b`` of an element ``g`` with exponent ``e = floor(log2|g|)`` changes it
    by ``|d| = 2**(e + b - m)`` where ``m`` is the mantissa width. The squared norm
    ``S = ||g||**2`` changes by ``|2*g*d + d**2| ~ 2|g||d|``. The final rounding of ``S`` into the
    accumulator destroys any change below ``u * S`` where ``u`` is that dtype's unit roundoff, so
    the flip survives iff::

        2 * |g| * 2**(e + b - m)  >  u * ||g||**2

    Solving for ``b`` gives this function. Two consequences worth carrying around:

    * the answer scales with ``(|g| / ||g||)**2``, i.e. with **1/P** for a model of P similarly
      sized gradients — a bigger model is a *less* sensitive detector, by ``log2(P)`` bits;
    * moving the accumulator from fp32 to fp64 buys 29 bits, which for anything smaller than about
      2**30 parameters is the difference between "catches mantissa bit 0" and "blind to the low 15".

    It is a single-rounding model. Pairwise summation can keep a change that lands mid-tree, so
    treat the answer as exact to about a bit — which is what the boundary test asserts.
    """
    if element == 0.0 or norm == 0.0:
        return mantissa_bits(value_dtype) + 1  # nothing to perturb; nothing is detectable
    mantissa = mantissa_bits(value_dtype)
    u = _UNIT_ROUNDOFF[accum_dtype]
    exponent = math.floor(math.log2(abs(element)))
    # 2**b > u * norm**2 / (2 * |g| * 2**(e - m))
    threshold = u * norm**2 / (2.0 * abs(element) * 2.0 ** (exponent - mantissa))
    return max(0, math.ceil(math.log2(threshold)))


def _raw_int(t: Tensor) -> int:
    """A tensor's raw bytes as one integer — an exact comparison that works at any element size."""
    return int.from_bytes(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(), "little"
    )


def _digest(t: Tensor) -> int:
    """Lossless 56-bit digest of a tensor's raw bytes (fits an int64 for the all_gather).

    A uint8 view works for every dtype including bfloat16, which numpy cannot represent. blake2b
    rather than an XOR fold: XOR cancels a pair of identical flips, a hash does not.
    """
    raw = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=7).digest(), "little")


@dataclass(frozen=True)
class GradCheckReport:
    """The verdict for one step. ``fired`` means the ranks disagree about the reduced gradients."""

    channels: tuple[str, ...]  # which channels disagreed: "norm" and/or "digest"
    local_norms: tuple[float, ...]  # per-rank, index == rank
    disagreeing_ranks: tuple[int, ...]
    first_param: str | None  # first parameter (sorted by name) whose digest differs
    named_rank: bool  # False when world_size < 3: disagreement is visible, blame is not

    @property
    def fired(self) -> bool:
        return bool(self.channels)

    def summary(self) -> str:
        if not self.fired:
            return f"ranks agree ({len(self.local_norms)} ranks, bitwise)"
        who = (
            f"rank {self.disagreeing_ranks[0]}"
            if self.named_rank and self.disagreeing_ranks
            else f"ranks {list(self.disagreeing_ranks)} (world<3: disagreement seen, blame not assignable)"
        )
        at = f" at {self.first_param}" if self.first_param else ""
        return f"CROSS-RANK DIVERGENCE via {'+'.join(self.channels)}: {who}{at}"


class CrossRankGradCheck:
    """Compare the post-reduction gradients across ranks, exactly.

    Args:
        accum_dtype: dtype the squared-norm sum is accumulated in. fp64 by default; fp32 exists so
            the sensitivity boundary can be *measured* rather than asserted.
        channels: which of ``"norm"`` / ``"digest"`` participate. Narrowing to ``{"norm"}`` is how
            the boundary test isolates the lossy channel.
    """

    def __init__(
        self,
        *,
        accum_dtype: torch.dtype = torch.float64,
        channels: frozenset[str] = frozenset({"norm", "digest"}),
    ) -> None:
        unknown = channels - {"norm", "digest"}
        if unknown:
            raise ValueError(
                f"unknown channels {sorted(unknown)} (expected 'norm' and/or 'digest')"
            )
        if not channels:
            raise ValueError("at least one channel is required — a detector with none never fires")
        if accum_dtype not in _UNIT_ROUNDOFF:
            raise ValueError(f"accum_dtype {accum_dtype} is not a float dtype")
        self.accum_dtype = accum_dtype
        self.channels = channels
        self.reports: list[GradCheckReport] = []

    def local_norm(self, model: nn.Module) -> float:
        """Global grad l2 norm over parameters in NAME order — the same order on every rank, or
        the ranks would disagree from summation order alone and the detector would false-positive
        on every clean step."""
        total = torch.zeros((), dtype=self.accum_dtype)
        for _, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            if p.grad is None:
                continue
            total = total + (p.grad.detach().to(self.accum_dtype) ** 2).sum().cpu()
        return float(total.sqrt().item())

    @property
    def fired(self) -> bool:
        return any(r.fired for r in self.reports)

    def check(self, model: nn.Module, group: dist.ProcessGroup | None = None) -> GradCheckReport:
        """One cross-rank comparison of the CURRENT (post-reduction) gradients.

        Collective — every rank must call it, and every rank gets the same report. Single-process
        runs return a never-fired report rather than raising: the check is a no-op with one rank,
        and making the caller branch on world size is how a monitor quietly stops running.
        """
        distributed = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size(group) if distributed else 1
        names = [n for n, p in sorted(model.named_parameters(), key=lambda kv: kv[0])]
        norm = self.local_norm(model)
        if not distributed or world == 1:
            report = GradCheckReport((), (norm,), (), None, named_rank=False)
            self.reports.append(report)
            return report

        bad_channels: list[str] = []

        # NCCL has no backend for CPU tensors, so the gather buffers must live where the gradients
        # do. `_raw_int`/`_digest` copy back to host on read; only the collective needs the device.
        device = next(
            (p.grad.device for _, p in model.named_parameters() if p.grad is not None),
            torch.device("cpu"),
        )

        # -- norm channel: the reduced grads are required to be bitwise identical, so are norms.
        mine = torch.tensor([norm], dtype=self.accum_dtype, device=device)
        gathered = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(gathered, mine, group=group)
        norms = tuple(float(g[0].item()) for g in gathered)
        # Compare the BYTES, not the values: `.view(int64)` only works for an 8-byte accumulator,
        # and `==` on floats would call two different NaNs equal on the one step where that matters.
        norm_words = [_raw_int(g) for g in gathered]
        if "norm" in self.channels and len(set(norm_words)) > 1:
            bad_channels.append("norm")

        # -- digest channel: lossless, per parameter, so the first divergent tensor gets a name.
        first_param: str | None = None
        odd: tuple[int, ...] = ()
        if "digest" in self.channels:
            local = torch.tensor(
                [
                    _digest(p.grad) if p.grad is not None else 0
                    for _, p in sorted(model.named_parameters(), key=lambda kv: kv[0])
                ],
                dtype=torch.int64,
                device=device,
            )
            digests = [torch.empty_like(local) for _ in range(world)]
            dist.all_gather(digests, local, group=group)
            for i, name in enumerate(names):
                column = [int(d[i].item()) for d in digests]
                if len(set(column)) > 1:
                    first_param = name
                    odd = _minority_ranks(column)
                    if "digest" not in bad_channels:
                        bad_channels.append("digest")
                    break

        if bad_channels and not odd:
            odd = _minority_ranks(norm_words)
        report = GradCheckReport(
            channels=tuple(bad_channels),
            local_norms=norms,
            disagreeing_ranks=odd,
            first_param=first_param,
            # With two ranks a disagreement is symmetric: you learn THAT they differ, never WHICH
            # one is wrong. Naming a culprit needs a third opinion (or a replay on one rank).
            named_rank=world >= 3 and len(odd) == 1,
        )
        self.reports.append(report)
        return report


def _minority_ranks(values: list[int]) -> tuple[int, ...]:
    """Ranks holding a minority value. With a clean majority this is the corrupted rank; with two
    ranks it is both of them, which is the honest answer."""
    counts = Counter(values)
    if len(counts) < 2:
        return ()
    top = max(counts.values())
    if list(counts.values()).count(top) > 1:  # no majority (e.g. world_size == 2)
        return tuple(range(len(values)))
    majority = counts.most_common(1)[0][0]
    return tuple(i for i, v in enumerate(values) if v != majority)


__all__ = ["CrossRankGradCheck", "GradCheckReport", "min_detectable_mantissa_bit"]
