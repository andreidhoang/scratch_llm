"""GEMM-family kernels. Wrapper · oracle · roofline for each.

Two ops, opposite roofline regimes — the pair that teaches the whole roofline:
  * GEMV  y = A @ x  (decode Linear, batch=1): 2·M·K flops over an M·K weight read → AI ≈ 0.5 FLOP/B,
    deeply HBM-bandwidth bound (one streaming read of A) → DoD = % of HBM bandwidth.
  * GEMM  C = A @ B  (the R4 tiled-matmul rung): 2·M·N·K flops with AI that GROWS with tile size →
    a big square GEMM is compute-bound → DoD = % of cuBLAS peak TFLOP/s.
The `__global__` bodies live in `../csrc/gemm/{gemv,gemm}.cu` (your reps); this module owns the
oracle, dispatch, and roofline around them.
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


def gemm_ref(A: Tensor, B: Tensor) -> Tensor:
    """Pure-torch oracle. C = A @ B, A:(M,K), B:(K,N) -> C:(M,N). The kernel must match this.

    Reduce in fp32 (tensor cores accumulate a half/bf16 matmul in fp32 anyway) so the oracle and the
    fp32-accumulating kernel agree within atol."""
    import torch  # noqa: PLC0415

    return torch.matmul(A.float(), B.float()).to(A.dtype)


def gemm(A: Tensor, B: Tensor) -> Tensor:
    """CUDA GEMM: C = A @ B for A:(M,K), B:(K,N) -> C:(M,N).

    The compute-bound matmul rung (R4) — the counterpart to the memory-bound decode `gemv`. Grade it
    on % of cuBLAS peak TFLOP/s, not HBM bandwidth; arithmetic intensity grows with the tile size."""
    from scratch_llm.kernels.inference._build import _ext  # noqa: PLC0415

    return _ext().gemm_cuda(A, B)


def gemm_roofline(m: int = 4096, n: int = 4096, k: int = 4096, dtype: str = "bfloat16"):
    """Benchmark the GEMM kernel vs cuBLAS (torch.matmul) and print the one-line roofline DoD.

    flops=2·M·N·K; rw_bytes=(MK+KN+MN)·itemsize. AI grows with the tile, so a big square GEMM is
    'compute-bound' and the DoD is % of cuBLAS TFLOP/s — climb the tiling ladder in csrc/gemm/gemm.cu
    until you approach it (a naive first pass lands well under cuBLAS; that gap is the lesson, like
    the FA2 53%). Compared against raw torch.matmul (cuBLAS) at the native dtype.
    """
    import torch  # noqa: PLC0415

    from scratch_llm.kernels.bench import roofline  # noqa: PLC0415

    td = getattr(torch, dtype)
    A = torch.randn(m, k, device="cuda", dtype=td)
    B = torch.randn(k, n, device="cuda", dtype=td)
    rw_bytes = (m * k + k * n + m * n) * A.element_size()
    return roofline(
        lambda: gemm(A, B),
        flops=2.0 * m * n * k,
        rw_bytes=rw_bytes,
        label=f"gemm[{dtype}]",
        ref=lambda: torch.matmul(A, B),
    )
