"""Production-grade custom operator for FlashAttention-2.

This module is a thin ``torch.library`` registration: it defines the custom_op,
its fake (meta) impl, its CUDA device impl, and the autograd formula that wires
forward to backward. The actual kernel selection lives in
``scratch_llm.kernels.attention.dispatch`` — the stable routing layer. This module
never imports a ``@triton.jit`` symbol or a backend module directly; it reaches
the GPU backend only through ``dispatch``.

History note: the CUDA impl previously launched ``_fa2_fwd_kernel`` directly with
per-head stride arguments, attempting a zero-copy 4D path. That launch was
incompatible with the current kernel signature (which flattens batch x heads and
takes 3 strides per tensor, not 4) and never ran successfully. The CUDA impl now
delegates to ``dispatch.flash_attention_triton_forward`` — the same working
forward the ``TritonFlashAttention`` autograd Function uses. For a contiguous 4-D
``(B, H, N, D)`` tensor the ``reshape(B*H, N, D)`` inside that path is a view and
``.contiguous()`` is a no-op, so the practical cost of dropping the explicit
stride plumbing is zero on the common case. Resurrecting a genuine zero-copy
non-contiguous path is a separate optimization that would land as a new entry in
``dispatch`` (e.g. ``launch_fa2_fwd_strided``), not by re-opening the backend to
``ops/``.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.library.custom_op("scratch_llm::flash_attention", mutates_args=())
def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    allow_tf32: bool = True,
) -> tuple[Tensor, Tensor]:
    """FlashAttention-2 custom operator.

    Dispatches to CPU (reference PyTorch) or CUDA (zero-copy strided Triton) backend.
    """
    # Default fallback implementation (CPU / any non-CUDA device): the oracle.
    from scratch_llm.kernels.attention.dispatch import flash_attention_forward

    return flash_attention_forward(q, k, v, is_causal=is_causal)


@flash_attention.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    allow_tf32: bool = True,
) -> tuple[Tensor, Tensor]:
    # Meta implementation for compile/symbolic shape tracing
    *lead, n, d = q.shape  # noqa: F841
    o_shape = q.shape
    lse_shape = tuple(lead) + (n,)
    return (
        torch.empty(o_shape, device=q.device, dtype=q.dtype),
        torch.empty(lse_shape, device=q.device, dtype=torch.float32),
    )


# Register CUDA implementation
@torch.library.impl("scratch_llm::flash_attention", "cuda")
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    allow_tf32: bool = True,
) -> tuple[Tensor, Tensor]:
    # CUDA path: delegate to the dispatch layer's working Triton FA2 forward.
    # Lazy import keeps the Triton dependency out of CPU/CI environments.
    from scratch_llm.kernels.attention.dispatch import (  # pyright: ignore[reportAttributeAccessIssue]
        flash_attention_triton_forward,
    )

    return flash_attention_triton_forward(q, k, v, is_causal=is_causal, allow_tf32=allow_tf32)


def flash_attention_backward(
    ctx,
    o_grad: Tensor,
    lse_grad: Tensor,  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor, None, None]:
    # Retrieve saved inputs
    q, k, v, o, lse = ctx.saved_tensors
    is_causal = ctx.is_causal
    allow_tf32 = ctx.allow_tf32

    from scratch_llm.kernels.attention.dispatch import (  # pyright: ignore[reportAttributeAccessIssue]
        flash_attention_triton_backward,
    )

    dq, dk, dv = flash_attention_triton_backward(
        q, k, v, o, lse, o_grad, is_causal=is_causal, allow_tf32=allow_tf32
    )
    return dq, dk, dv, None, None


def flash_attention_setup_context(ctx, inputs, output) -> None:
    # Save inputs/outputs for backward
    q, k, v, is_causal, allow_tf32 = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.is_causal = is_causal
    ctx.allow_tf32 = allow_tf32


# Register autograd formula using modern torch.library custom op autograd registration API
torch.library.register_autograd(
    "scratch_llm::flash_attention",
    flash_attention_backward,
    setup_context=flash_attention_setup_context,
)
