"""Stable public surface for attention kernels — the dispatch layer.

This is the contract every consumer above ``kernels/`` (``model.py``, ``ops/``)
imports from. Backend subpackages (``prefill/``, ``decode/``) are PRIVATE —
production code never reaches into them directly; it goes through this module.
Tests and benches may target a backend directly (they are verifying one specific
implementation, not dispatching).

Why the seam exists (the gap this closes): before dispatch.py, ``model.py`` and
``ops/attention.py`` imported ``_fa2_fwd_kernel``, ``TritonFlashAttention``, and
``paged_decode_attention`` straight from the backend files. That couples every
caller to a specific backend's file layout and kernel signature — so renaming a
backend, retuning its autotune table, or adding a second prefill path (a Hopper
WGMMA / Blackwell tcgen05 flash variant) forces edits across the model and the
custom-op. Routing through dispatch means those changes land HERE, and callers
above ``kernels/`` do not move.

CPU-safety (the ``kernels/`` package invariant — ``import scratch_llm.kernels``
must never pull Triton or compile CUDA on a CI box): the pure-PyTorch oracle is
imported eagerly and is always available; the GPU backends are loaded LAZILY via
PEP 562 ``__getattr__``. Accessing a GPU name (``TritonFlashAttention``,
``paged_decode_attention``, ``flash_attention_triton_forward``) triggers its
backend import on first use — correct, because you only ask for a GPU kernel on a
GPU box. ``from scratch_llm.kernels.attention.dispatch import flash_attention_forward``
stays CPU-safe (the oracle); ``... import TritonFlashAttention`` loads Triton.

Stable surface:
  CPU oracle (eager, pure PyTorch — the correctness ground truth):
    flash_attention_forward(q, k, v, *, is_causal, q_tile, k_tile) -> (O, L)
    FlashAttentionPyTorch                                   (torch.autograd.Function)
  GPU prefill backend (lazy — FlashAttention-2, autotuned):
    flash_attention_triton_forward(q, k, v, *, is_causal, allow_tf32) -> (O, L)
    flash_attention_triton_backward(q, k, v, o, lse, do, *, ...) -> (dQ, dK, dV)
    TritonFlashAttention                                   (autograd wrapper)
  GPU decode backend (lazy — fused paged KV, single-token query):
    paged_decode_attention(q, pool_k, pool_v, block_table, lengths) -> O

Adding a backend: register it in ``_LAZY_GPU`` (or, for a device/dtype/shape
*choice* between two prefill backends, add the routing function here — callers
above ``kernels/`` keep importing the same name).
"""

from __future__ import annotations

from typing import cast

from torch import Tensor

from scratch_llm.kernels.attention.reference import (  # CPU-safe: pure-PyTorch oracle
    FlashAttentionPyTorch,
    flash_attention_forward,
)

# `__all__` is the EAGER, CPU-safe surface — only names that exist as static
# module bindings. `from dispatch import *` therefore never triggers the lazy
# loader (it cannot pull Triton / nvcc). The full surface (including the GPU
# backends) is advertised by `__dir__()` below and documented in this module's
# docstring; the lazy names resolve through `__getattr__` on explicit import.
__all__ = [
    "FlashAttentionPyTorch",
    "flash_attention_forward",
]

# Lazy GPU backend loader (PEP 562). Each entry maps a public name to
# (backend module, attribute). The import runs on first attribute access only, so
# `import scratch_llm.kernels.attention.dispatch` stays CPU-safe; accessing a GPU
# name is the explicit act that pulls Triton / triggers nvcc JIT.
_LAZY_GPU: dict[str, tuple[str, str]] = {
    "flash_attention_triton_forward": (
        "scratch_llm.kernels.attention.prefill.fa2",
        "flash_attention_triton_forward",
    ),
    "flash_attention_triton_backward": (
        "scratch_llm.kernels.attention.prefill.fa2",
        "flash_attention_triton_backward",
    ),
    "TritonFlashAttention": (
        "scratch_llm.kernels.attention.prefill.fa2",
        "TritonFlashAttention",
    ),
    "paged_decode_attention": (
        "scratch_llm.kernels.attention.decode.paged",
        "paged_decode_attention",
    ),
    # --- Frontier rung (arch-gated; runtime deferred to H100/H200 rental day) ---
    "flash_attention_fa3_forward": (
        "scratch_llm.kernels.attention.prefill.fa3",
        "flash_attention_fa3_forward",
    ),
}


def flash_attention(q: Tensor, k: Tensor, v: Tensor, *, is_causal: bool = True) -> Tensor:
    """Self-attention ``softmax(QK^T / sqrt(D)) V`` routed by ``(device, arch)``.

    The arch-aware routing seam for attention (mirrors :func:`gemm.matmul`): on
    Hopper it routes to the FlashAttention-3 warp-specialized WGMMA+TMA kernel
    (compile-gated, runtime deferred to the rental day); on all other CUDA devices
    it routes to the Triton FA2 backend (the shipping path); on CPU it routes to
    the pure-PyTorch oracle.

    The model's hot path can stay on ``TritonFlashAttention`` (the autograd
    wrapper) directly for now — this routing fn is the *future-proofing* seam.
    When FA3 graduates with a backward pass, callers move here without touching
    the model.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        # CPU: the eager oracle (FlashAttentionPyTorch). Self-attention shape.
        from scratch_llm.kernels.attention.reference import FlashAttentionPyTorch

        return cast(Tensor, FlashAttentionPyTorch.apply(q, k, v, is_causal))

    from scratch_llm.kernels.common.arch import is_hopper

    if is_hopper():
        from scratch_llm.kernels.attention.prefill.fa3 import flash_attention_fa3_forward

        return flash_attention_fa3_forward(q, k, v, is_causal=is_causal)

    # Default CUDA path: the Triton FA2 autograd wrapper (fwd + bwd).
    from scratch_llm.kernels.attention.prefill.fa2 import TritonFlashAttention

    return cast(Tensor, TritonFlashAttention.apply(q, k, v, is_causal))


def __getattr__(name: str):
    if name in _LAZY_GPU:
        import importlib

        mod_name, attr = _LAZY_GPU[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()) | set(_LAZY_GPU))
