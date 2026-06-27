"""Mixed-precision numerics — why low precision needs fp32 in the right places.

A2 systems. Two load-bearing lessons that make large-batch training fit *and* stay stable:

1. **fp16 accumulation loses small addends.** A *sequential* running sum in fp16 stalls once it
   grows large enough that the increment falls below half an ulp at the current magnitude — the add
   rounds to a no-op. (e.g. summing 0.01 in fp16 freezes at 32.0: the ulp at 32 is 2⁻⁵ = 0.03125,
   so 0.01 < ulp/2 vanishes.) This is *why* optimizer state, the loss, and reductions accumulate in
   fp32 even in a bf16/fp16 run. Note this is the worst case: ``torch.sum`` uses pairwise (tree)
   reduction, which is far more accurate than this naive loop — the loop *is* the lesson.

2. **The autocast dtype rule.** Under ``torch.autocast``, matmuls run in bf16 (throughput on the
   tensor cores) while LayerNorm/softmax/reductions — and the master weights — stay fp32 (numerical
   stability). autocast encodes exactly this per-op policy; we don't pick dtypes by hand.

Interview: "you trained in fp16 and the loss plateaued / NaN'd — why?" The answer is accumulation
precision + the autocast policy (and, one precision lower, the FP8 amax/scale that overflows). This
is also the mechanistic root of the train-vs-serve logit drift (``utils/monitors.py`` kl_train_infer):
two engines reduce softmax/PV in different precisions and produce different logits on the same weights.
"""

from __future__ import annotations

import torch
from torch import Tensor


def naive_accumulate(value: float, count: int, dtype: torch.dtype) -> Tensor:
    """Add ``value`` to a running sum ``count`` times, sequentially, in ``dtype``.

    Sequential (not pairwise) on purpose: this exposes the fp16 small-addend underflow that a tree
    reduction would hide. Returns the 0-d sum tensor in ``dtype``."""
    acc = torch.zeros((), dtype=dtype)
    inc = torch.tensor(value, dtype=dtype)
    for _ in range(count):
        acc += inc
    return acc
