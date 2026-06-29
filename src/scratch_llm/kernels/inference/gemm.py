"""GEMM-family kernels (decode) — GEMV (the decode Linear). Wrapper · oracle · roofline.

At batch=1/seq=1 every Linear is y = A @ x with a single activation vector: 2·M·K flops over an
M·K weight read → AI ≈ 0.5 FLOP/B, deeply HBM-bandwidth bound (one streaming read of A). This is
*why* decode is bandwidth-bound and why weight-quant is the lever. The `__global__` body lives in
`../csrc/gemm/gemv.cu` (your rep); this module owns everything around it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def gemv_ref(A: Tensor, x: Tensor) -> Tensor:
    """Pure-torch oracle. y = A @ x, A:(M,K), x:(K,) -> y:(M,). The kernel must match this.

    Reduce in fp32 (matmul of half/bf16 inputs accumulates in fp32 on tensor cores anyway) so the
    oracle and the fp32-accumulating kernel agree within atol."""
    import torch  # noqa: PLC0415

    return torch.mv(A.float(), x.float()).to(A.dtype)


def gemv(A: Tensor, x: Tensor) -> Tensor:
    """CUDA GEMV: y = A @ x for a weight A:(M,K) and a single activation vector x:(K,) -> y:(M,).

    The decode-time Linear (batch=1, seq=1). For a batched activation use matmul; this kernel is the
    bandwidth-bound vector case where AI collapses and reading A once dominates."""
    from scratch_llm.kernels.inference._build import _ext  # noqa: PLC0415

    return _ext().gemv_cuda(A, x)


def gemv_roofline(m: int = 4096, k: int = 4096, dtype: str = "bfloat16"):
    """Benchmark the GEMV kernel vs the torch oracle and print the one-line memory-roofline DoD.

    flops=2·M·K; rw_bytes=(read A + read x + write y)·itemsize. A:(M,K) dominates the read, so
    AI ≈ 2·M·K / (M·K·itemsize) → ~0.5 FLOP/B at fp32, deeply 'memory-bound' — the kernel is a
    single streaming read of A. % of ref is vs torch.mv (cuBLAS gemv); beat it by hitting peak HBM.
    """
    import torch  # noqa: PLC0415

    from scratch_llm.kernels.bench import roofline  # noqa: PLC0415

    td = getattr(torch, dtype)
    A = torch.randn(m, k, device="cuda", dtype=td)
    x = torch.randn(k, device="cuda", dtype=td)
    rw_bytes = (m * k + k + m) * A.element_size()
    return roofline(
        lambda: gemv(A, x),
        flops=2.0 * m * k,
        rw_bytes=rw_bytes,
        label=f"gemv[{dtype}]",
        ref=lambda: gemv_ref(A, x),
    )
