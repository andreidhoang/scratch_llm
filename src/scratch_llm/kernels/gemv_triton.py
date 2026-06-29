"""Triton GEMV kernel (decode-time Linear) — wrapper, reference, and roofline benchmark.

This file provides the scaffolding for a custom Matrix-Vector Multiplication (y = A @ x)
Triton kernel. The kernel body itself is left empty for you to implement.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _gemv_triton_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_x,
    stride_y,
    BLOCK_K: tl.constexpr,
):
    """Triton kernel for y = A @ x.

    Grid: (M,) - each program (program_id 0) is responsible for exactly one output row.

    Your implementation contract:
    1. Identify the output row using tl.program_id(0).
    2. Stride/loop over K columns in chunks of BLOCK_K.
    3. Load values of matrix A and vector x, masking out out-of-bounds columns.
    4. Compute element-wise multiplication and reduce using tl.sum.
    5. Write the final accumulated scalar to y[row].
    """

    # 1. Identify which row this program block is processing
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return
    # 2. Initialize column offsets for the block
    cols = tl.arange(0, BLOCK_K)
    # 3. Setup accumulator in high-precision float32
    acc = 0.0
    # 4. Stride loop over columns in chunks of BLOCK_K
    for k_offset in range(0, K, BLOCK_K):
        # Calculate current column indices
        col_idx = k_offset + cols

        # Load weight elements from Row row_idx
        a_ptrs = a_ptr + row_idx * stride_am + col_idx * stride_ak
        a = tl.load(a_ptrs, mask=col_idx < K, other=0.0)

        # Load activation vector elements
        x_ptrs = x_ptr + col_idx * stride_x
        x = tl.load(x_ptrs, mask=col_idx < K, other=0.0)

        # Multiply element-wise and sum this block's results
        acc += tl.sum(a * x, axis=0)
    # 5. Write the final accumulated scalar to the output vector
    y_ptr_out = y_ptr + row_idx * stride_y
    tl.store(y_ptr_out, acc)


def gemv_triton(A: Tensor, x: Tensor) -> Tensor:
    """Launcher for the Triton GEMV kernel.

    A: (M, K) matrix
    x: (K,) vector
    y: (M,) output vector
    """
    assert A.is_cuda, "Matrix A must be on CUDA"
    assert x.is_cuda, "Vector x must be on CUDA"
    assert A.dim() == 2, "A must be 2D"
    assert x.dim() == 1, "x must be 1D"
    assert A.size(1) == x.size(0), "Inner dimensions must match"

    M, K = A.shape
    y = torch.empty((M,), device=A.device, dtype=A.dtype)

    # Grid mapping: One program/block per output row M
    grid = (M,)

    # Determine BLOCK_K (power of 2 greater than or equal to K, or tuned block size)
    # A block size of 1024 is a common default starting point.
    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K < 1024:
        BLOCK_K = 1024

    _gemv_triton_kernel[grid](
        a_ptr=A,
        x_ptr=x,
        y_ptr=y,
        M=M,
        K=K,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_x=x.stride(0),
        stride_y=y.stride(0),
        BLOCK_K=BLOCK_K,
    )
    return y


def gemv_triton_ref(A: Tensor, x: Tensor) -> Tensor:
    """Reference implementation using PyTorch eager matmul."""
    return torch.mv(A.float(), x.float()).to(A.dtype)


def gemv_triton_roofline(m: int = 4096, k: int = 4096, dtype: str = "bfloat16"):
    """Benchmark the Triton GEMV kernel against the PyTorch reference."""
    from scratch_llm.kernels.bench import roofline

    td = getattr(torch, dtype)
    A = torch.randn(m, k, device="cuda", dtype=td)
    x = torch.randn(k, device="cuda", dtype=td)

    flops = 2.0 * m * k
    rw_bytes = (m * k + k + m) * A.element_size()

    return roofline(
        fn=lambda: gemv_triton(A, x),
        flops=flops,
        rw_bytes=rw_bytes,
        label=f"gemv_triton[{dtype}]",
        ref=lambda: gemv_triton_ref(A, x),
    )
