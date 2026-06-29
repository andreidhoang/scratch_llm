"""Normalization kernels (decode) — RMSNorm. Wrapper · pure-torch oracle · roofline.

Decode-time, HBM-bandwidth bound: a single read of x + write of y, so the DoD is "% of the memory
roofline" (see `rmsnorm_roofline`), not TFLOP/s. The `__global__` body lives in
`../csrc/norm/rmsnorm.cu` (your rep); this module owns everything around it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def rmsnorm_ref(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Pure-torch oracle. y = x * rsqrt(mean(x^2, -1) + eps) * w. The kernel must match this."""
    import torch  # noqa: PLC0415

    dtype = x.dtype
    x = x.float()
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x * inv_rms).to(dtype) * weight


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Fused CUDA RMSNorm over the last dim. Flattens (B,T,D) -> (N,D) and dispatches to the kernel."""
    from scratch_llm.kernels.inference._build import _ext  # noqa: PLC0415

    *lead, d = x.shape
    y = _ext().rmsnorm_cuda(x.reshape(-1, d), weight, eps)
    return y.reshape(*lead, d)


def rmsnorm_roofline(n: int = 4096, d: int = 4096, dtype: str = "bfloat16"):
    """Benchmark the fused kernel vs the torch oracle and print the one-line memory-roofline DoD.

    flops≈3·N·D (trivial); rw_bytes=(read x + write y + read w)·itemsize — the real bound. AI is
    tiny → 'memory-bound'; % of ref is vs the eager torch oracle (beat it by killing HBM trips).
    """
    import torch  # noqa: PLC0415

    from scratch_llm.kernels.bench import roofline  # noqa: PLC0415

    td = getattr(torch, dtype)
    x = torch.randn(n, d, device="cuda", dtype=td)
    w = torch.randn(d, device="cuda", dtype=td)
    rw_bytes = (2 * n * d + d) * x.element_size()
    return roofline(
        lambda: rmsnorm(x, w),
        flops=3.0 * n * d,
        rw_bytes=rw_bytes,
        label=f"rmsnorm[{dtype}]",
        ref=lambda: rmsnorm_ref(x, w),
    )
