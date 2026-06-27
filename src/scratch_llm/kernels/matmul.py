"""matmul — the canonical **naive → cuBLAS** climb (the CUDA-for-Deep-Learning spine rep).

The recurring sprint rep (BOOK_SPRINT §5): drive a matmul from a few % of cuBLAS toward ~80%+, ONE
optimization at a time, profiling after each step (`kernels/bench.py: matmul_roofline`).
`matmul_naive` is the working baseline handed to you so the profile loop is live on Day 1; the
OPTIMIZED rungs are the **reconstruct-from-blank** exercise — that is where the skill (and the
interview signal) lives. Mirror each rung in CUDA C++ from the book and benchmark it via
`bench.load_cuda` for the C++/CUDA fluency NVIDIA tests.

The CUDA C++ rung ladder (siboehm's worklog; reconstruct from the book, benchmark via load_cuda):
    R0 naive (1 thread / output)   → R1 global-memory coalescing   → R2 shared-memory blocking
    → R3 1-D block-tiling          → R4 2-D block-tiling           → R5 vectorized (float4)
    → R6 warp-tiling
Target: R6 ≈ 80-90% of cuBLAS at 4096³ (siboehm reaches ~83%).

Triton-first here (your stack). GPU-only and deliberately NOT exported from `kernels/__init__.py`,
so importing the package stays CPU/CI-safe (mirrors `flash_attention_triton.py`).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """cuBLAS via torch.matmul — the % target every rung is measured against."""
    return torch.matmul(a, b)


@triton.jit
def _matmul_naive_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    sam,
    sak,
    sbk,
    sbn,
    scm,
    scn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k0 * BLOCK_K + offs_k
        a_ptrs = a_ptr + (offs_m[:, None] * sam + kk[None, :] * sak)
        b_ptrs = b_ptr + (kk[:, None] * sbk + offs_n[None, :] * sbn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(kk[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul_naive(
    a: torch.Tensor, b: torch.Tensor, block: tuple[int, int, int] = (64, 64, 32)
) -> torch.Tensor:
    """Working baseline — fixed small blocks, no L2-reuse ordering, no autotune. Profile it FIRST,
    predict the % of cuBLAS, then climb (`matmul_tiled`)."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims must match: {a.shape} @ {b.shape}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    bm, bn, bk = block
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    _matmul_naive_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
    )
    return c


def matmul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RECONSTRUCT FROM BLANK — the rung up from `matmul_naive`. Add these one at a time, predicting
    and then re-profiling the % of cuBLAS after each (target ≥ 2× the naive %):

        1. **L2-reuse program ordering** — group program ids along M so B columns are reused across
           a tile of rows (the single biggest Triton matmul win). [Triton matmul tutorial pattern.]
        2. **Autotune** over BLOCK_M / BLOCK_N / BLOCK_K, num_warps, num_stages (software pipelining
           that overlaps the global→shared load with the tl.dot).
        3. Widen BLOCK_K / num_stages for the 4090's memory system; re-measure.

    Then mirror R2-R6 in CUDA C++ from the book and benchmark via `bench.load_cuda`.
    """
    raise NotImplementedError(
        "reconstruct matmul_tiled from blank — see this docstring + the CUDA-for-DL chapter"
    )
