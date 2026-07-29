"""Stable public surface for reduction kernels (softmax, top-k) -- the dispatch layer.

Consumers above ``kernels/`` import from here; backend files are PRIVATE. Tests
and benches may target a backend directly (the softmax ladder benchmarks the
twopass/online/fused stages head-to-head; the top-k rung measures its own low
achieved-bandwidth honestly).

Routing policy:
  * CPU / non-CUDA ............ ``torch.softmax`` / ``torch.topk`` (the framework path)
  * CUDA ...................... ``softmax_triton`` (the memory-bound row kernel, ``mode``
                                selects the HBM-traffic stage) / ``topk_last_dim``
                                (iterative max-extraction) / ``fused_softmax_topk`` (the
                                traffic-saving fused path for sampling)

Single backend today (Triton). The seam exists so a fused softmax+mask+cast, a
speculative-decoding fused softmax+topk, or a FlashInfer-style batched-score
kernel lands here as a new branch, not a rewrite of every caller.

CPU-safety: nothing imported eagerly; ``__all__`` is empty so ``from dispatch
import *`` is a no-op. Every backend loads lazily via PEP 562 on first explicit
access; ``import scratch_llm.kernels.reduce.dispatch`` pulls no Triton.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__: list[str] = []

_LAZY_GPU: dict[str, tuple[str, str]] = {
    "softmax_triton": ("scratch_llm.kernels.reduce.softmax", "softmax_triton"),
    "topk_last_dim": ("scratch_llm.kernels.reduce.topk", "topk_last_dim"),
    "fused_softmax_topk": ("scratch_llm.kernels.reduce.topk", "fused_softmax_topk"),
}


def __getattr__(name: str):
    if name in _LAZY_GPU:
        import importlib

        mod_name, attr = _LAZY_GPU[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()) | set(_LAZY_GPU))


def softmax(x: Tensor, *, mode: str = "fused") -> Tensor:
    """Row softmax over the last dim of a 2-D ``(M, N)`` tensor -- drop-in for
    ``torch.softmax(x, dim=-1)`` on 2-D input.

    On CUDA routes to :func:`softmax_triton` (``mode`` selects the twopass/online/fused
    HBM-traffic stage -- the rung's whole point); on CPU falls through to ``torch.softmax``.

    A wholly ``-inf`` row yields a uniform ``1/N`` row (never NaN), matching the
    triton kernel's contract."""
    if x.is_cuda:
        from scratch_llm.kernels.reduce.softmax import softmax_triton

        return softmax_triton(x, mode=mode)
    return torch.softmax(x, dim=-1)


def topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Row-wise top-``k`` values + indices over the last dim -- drop-in for
    ``torch.topk(x, k, dim=-1)`` on 2-D input.

    On CUDA routes to :func:`topk_last_dim` (ties break to the lowest column index
    -- a *defined* rule, unlike ``torch.topk``'s unspecified tie order); on CPU
    falls through to ``torch.topk`` (whose tie order is unspecified, so callers
    depending on the lowest-index rule must stay on the CUDA path)."""
    if x.is_cuda:
        from scratch_llm.kernels.reduce.topk import topk_last_dim

        return topk_last_dim(x, k)
    return torch.topk(x, k, dim=-1)
