"""Stable public surface for normalization kernels -- the dispatch layer.

Consumers above ``kernels/`` import from here; the backend file is PRIVATE.
Tests and benches may target the backend directly.

Routing policy:
  * CPU / non-CUDA ............ ``F.rms_norm`` / ``F.layer_norm`` (the framework path)
  * CUDA ...................... ``rmsnorm_triton`` / ``layernorm_triton`` (the bandwidth-bound
                                one-row-per-block triton kernels -- the whole point of a custom
                                norm kernel is reading the row from HBM exactly once)

Single backend today (Triton). The seam exists so a future variant lands here
without touching callers: a fused norm+quant kernel (A5 quant), a fused
norm+residual, or a Blackwell tcgen05 reduction. Each would be a new branch in
:func:`rmsnorm` / :func:`layernorm`, not a rewrite of the model.

CPU-safety: nothing imported eagerly (no CPU oracle is re-exported -- the
framework ``F.rms_norm`` IS the CPU path and is imported lazily inside the routing
functions). ``import scratch_llm.kernels.norm.dispatch`` pulls no Triton; explicit
``from dispatch import rmsnorm_triton`` loads it.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__: list[str] = []

_LAZY_GPU: dict[str, tuple[str, str]] = {
    "rmsnorm_triton": ("scratch_llm.kernels.norm.normalize", "rmsnorm_triton"),
    "layernorm_triton": ("scratch_llm.kernels.norm.normalize", "layernorm_triton"),
}


def __getattr__(name: str):
    if name in _LAZY_GPU:
        import importlib

        mod_name, attr = _LAZY_GPU[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()) | set(_LAZY_GPU))


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """``x / sqrt(mean(x^2) + eps) * weight`` over the last dim -- drop-in for ``F.rms_norm``.

    Routes to the triton kernel on CUDA (one HBM read of the row, fp32 reduction
    in registers) and to ``F.rms_norm`` elsewhere. Matches ``F.rms_norm`` output."""
    if x.is_cuda:
        from scratch_llm.kernels.norm.normalize import rmsnorm_triton

        return rmsnorm_triton(x, weight, eps=eps)
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def layernorm(x: Tensor, weight: Tensor, bias: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    """``(x - mu) / sqrt(var + eps) * weight + bias`` over the last dim -- drop-in for
    ``F.layer_norm``.

    Routes to the triton kernel on CUDA (one HBM read, two fp32 reductions over the
    held tile) and to ``F.layer_norm`` elsewhere. Matches ``F.layer_norm`` output."""
    if x.is_cuda:
        from scratch_llm.kernels.norm.normalize import layernorm_triton

        return layernorm_triton(x, weight, bias, eps=eps)
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
