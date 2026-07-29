"""A2 Rungs 5+6 — bf16 GEMM (C = A @ B) with fp32 accumulate, a three-stage Triton ladder.

The GEMM is the compute-bound anchor of the roofline: at a square N³ the minimal-traffic arithmetic
intensity is AI = 2·M·N·K / ((MN + NK + MK)·elem) = N/3 ≈ 1365 FLOP/byte at N=4096 (the A2 spec quotes
~512 under a heavier-traffic model that counts re-reads). Either figure is far past this card's ridge
(~131 FLOP/byte), so a *correct* GEMM is limited by tensor-core throughput, not HBM. The ladder shows
the climb toward that compute roof:

  (1) ``gemm_naive``  — one program per output element, a K-loop of scalar dot products through global
      memory. No blocking, no data reuse (each A row is re-read N times, each B column M times), no
      ``tl.dot`` / tensor cores. This is the siboehm "naive" analog and lands at ~1% of cuBLAS.
  (2) ``gemm_tiled``  — the classic SMEM-tiled GEMM: a BLOCK_M×BLOCK_N output tile accumulated over
      K-blocks with ``tl.dot`` (which the Triton compiler stages through shared memory and issues on the
      tensor cores). fp32 accumulator, remainder-masked. One fixed, hand-picked config.
  (3) ``gemm_autotuned`` — the same kernel body under ``triton.autotune`` over BLOCK_M/N/K, GROUP_M
      (L2 swizzle), num_warps and num_stages: the register/block-tiling + software-pipelining analog.
      This is the shipping kernel and the correctness gate.

**Triton-abstraction honesty.** In the raw-CUDA A2 ladder, the rungs *above* the SMEM tile are
float4-vectorized global loads, register block-tiling (each thread owns a TM×TN micro-tile), and an XOR
swizzle to kill shared-memory bank conflicts. In Triton **none of those are hand-written**: ``tl.load``
of a 2D block is vectorized/coalesced by the compiler, ``tl.dot`` chooses the register micro-tiling and
the SMEM layout (swizzle included), and ``num_stages`` drives the async-copy software pipeline. So stage
(3) is "autotune the compiler's knobs", not "write the swizzle" — see the notes. The measured %-of-cuBLAS
is the honest result: on Blackwell sm120, autotuned ``tl.dot`` already reaches a high fraction of cuBLAS.

Invariant (tests/test_gemm.py, gpu): matches ``torch.matmul`` at rtol 1e-2 (bf16) over a magnitude range
and on adversarial shapes — remainder tiles (M, N, K each non-multiples of the block), M=1 (GEMV), and a
single large outlier — with fp32 accumulation in every stage.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


# ==================================================================================================
# Stage 1 — naive: one output element per program, no tiling, no tl.dot, no reuse.
# ==================================================================================================
@triton.jit
def _gemm_naive_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    acc = tl.zeros((), dtype=tl.float32)
    k_off = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + k_off
        mask = k < K
        a = tl.load(a_ptr + m * stride_am + k * stride_ak, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + k * stride_bk + n * stride_bn, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(a * b, axis=0)  # fp32 accumulate, one output element
    tl.store(c_ptr + m * stride_cm + n * stride_cn, acc.to(c_ptr.dtype.element_ty))


def gemm_naive(a: Tensor, b: Tensor, *, block_k: int = 64) -> Tensor:
    """Naive C = A @ B: one program per C[m,n], scalar K-loop through global memory (no reuse)."""
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"inner dims must match: {a.shape} @ {b.shape}"
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    _gemm_naive_kernel[(m * n,)](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_K=block_k,
        num_warps=1,
    )
    return c


# ==================================================================================================
# Stages 2 & 3 — shared kernel body: SMEM-tiled tl.dot over K-blocks with a GROUP_M L2 swizzle.
# ==================================================================================================
@triton.jit
def _gemm_tiled_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Group-M swizzle: walk pids in GROUP_M×N blocks so the B columns a group touches stay L2-resident.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k0 * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        acc += tl.dot(a, b)  # bf16×bf16 → fp32 tensor-core MMA
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
        num_warps=w,
        num_stages=s,
    )
    for bm, bn, bk, w, s in [
        (128, 256, 64, 8, 3),
        (256, 128, 64, 8, 3),
        (128, 128, 64, 8, 4),
        (128, 128, 32, 4, 4),
        (128, 256, 32, 8, 4),
        (64, 256, 64, 4, 4),
        (256, 64, 64, 4, 4),
        (128, 64, 64, 4, 3),
        (64, 128, 64, 4, 4),
        (64, 64, 64, 4, 3),
    ]
]

_gemm_autotuned_kernel = triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])(
    _gemm_tiled_kernel
)


def _launch_tiled(a: Tensor, b: Tensor, meta: dict[str, int] | None) -> Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"inner dims must match: {a.shape} @ {b.shape}"
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    args = (
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    if meta is None:  # autotuned path — the compiler picks BLOCK_*/warps/stages per (M,N,K)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),
        )
        _gemm_autotuned_kernel[grid](*args)
    else:  # fixed hand-picked config
        grid = (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
        _gemm_tiled_kernel[grid](
            *args,
            BLOCK_M=meta["BLOCK_M"],
            BLOCK_N=meta["BLOCK_N"],
            BLOCK_K=meta["BLOCK_K"],
            GROUP_M=meta["GROUP_M"],
            num_warps=meta.get("num_warps", 4),
            num_stages=meta.get("num_stages", 3),
        )
    return c


def gemm_tiled(a: Tensor, b: Tensor) -> Tensor:
    """SMEM-tiled C = A @ B with one fixed config (BLOCK 128×128×64, 8 warps, 3 stages)."""
    return _launch_tiled(
        a,
        b,
        {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        },
    )


def gemm_autotuned(a: Tensor, b: Tensor) -> Tensor:
    """Autotuned C = A @ B — the shipping kernel; picks block sizes/warps/stages per (M,N,K)."""
    return _launch_tiled(a, b, None)


gemm = gemm_autotuned  # the shipping entry point
