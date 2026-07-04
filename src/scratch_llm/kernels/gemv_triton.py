"""Triton GEMV ladder — ``y = A @ x`` (A is ``(M, N)``, x is ``(N,)``) — A2 Rung 1.

GEMV reads A exactly once and does one multiply-add per element, so it is squarely **memory-bound**:
``bytes = M*N*2`` (bf16 A dominates; x and y are O(M+N)), ``FLOPs = 2*M*N`` ⇒ arithmetic intensity
``AI ≈ 1 FLOP/byte`` — two orders of magnitude below the ~131 FLOP/byte ridge of this card. The only
thing that matters is streaming A at HBM bandwidth; the deliverable is GB/s, not FLOP/s.

Three stages, each a genuinely different memory access pattern (measured in ``bench/gemv.py``):

1. :func:`gemv_naive` — one program per output row, a **single warp** walking the row in tiny
   ``BLOCK_N=64`` tiles. Correct, but under-fills each SM (1 warp/CTA) and issues narrow loads ⇒
   leaves bandwidth on the floor.
2. :func:`gemv_blockrow` — still one program per row, but a **wide, coalesced** tile
   (``BLOCK_N`` up to 4096, ``num_warps`` autotuned) with an element-wise fp32 accumulator reduced
   once at the end. The warp's lanes read contiguous A elements ⇒ fully coalesced 128-byte
   transactions; this is the stage that should reach HBM peak on a large square shape.
3. :func:`gemv_split` — **split-N two-stage**: grid ``(M, S)`` so each row is shared by ``S``
   programs that each stream an N-slice and ``atomic_add`` their partial into an fp32 ``y``. Adds
   parallelism to fill the GPU when M alone is too few CTAs (tall-skinny ``N ≫ M``); on a large
   square shape it merely matches stage 2 (already bandwidth-saturated) at the cost of the atomics.

Numerics: every stage loads A/x as their native dtype, **accumulates in fp32**, and casts the final
scalar back to A's dtype — so bf16 matches ``torch.mv`` to rtol 1e-2 and fp32 to rtol 1e-5
(tests/test_gemv.py, gpu).

Honesty (Triton vs the raw-CUDA A2 ladder): the *float4 vectorization* rung of the CUDA GEMV ladder
(loading 8 bf16 / row-lane as one 128-bit ``LDG.E.128``) is **compiler-managed** here — Triton emits
vectorized loads for a contiguous ``tl.load`` over a wide ``tl.arange`` on its own, so stages 2/3 get
that win without an explicit ``float4`` cast. Coalescing and bank-conflict avoidance are likewise the
compiler's job. The raw-CUDA variant would add: explicit ``float4``/``__nv_bfloat162`` packed loads,
a warp-shuffle tree reduction (vs ``tl.sum``), and ``__ldg`` read-only caching — none of which change
the *bound* here (we are HBM-limited, not issue-limited), only how close to it a hand kernel lands.

GPU-only (imports Triton); not re-exported by ``kernels/__init__`` so a CPU import never pulls Triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# ---------------------------------------------------------------------------------------------------
# Stage 1 — naive: one program per row, one warp, narrow tiles.
# ---------------------------------------------------------------------------------------------------


@triton.jit
def _gemv_naive_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    m_size,
    n_size,
    stride_am,
    stride_an,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= m_size:
        return
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n0 in range(0, n_size, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < n_size
        a = tl.load(a_ptr + row * stride_am + offs * stride_an, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += a * x
    y = tl.sum(acc, axis=0)
    tl.store(y_ptr + row, y.to(y_ptr.dtype.element_ty))


def gemv_naive(a: Tensor, x: Tensor) -> Tensor:
    """Naive baseline: grid=(M,), a single warp per row, ``BLOCK_N=64`` tiles. Memory-bound but
    under-utilized (1 warp/CTA, narrow loads). Fixed config — deliberately *not* autotuned."""
    m, n = a.shape
    y = torch.empty(m, device=a.device, dtype=a.dtype)
    _gemv_naive_kernel[(m,)](a, x, y, m, n, a.stride(0), a.stride(1), BLOCK_N=64, num_warps=1)
    return y


# ---------------------------------------------------------------------------------------------------
# Stage 2 — block-per-row, wide coalesced tile, autotuned width/warps.
# ---------------------------------------------------------------------------------------------------

_BLOCKROW_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=w) for bn in (512, 1024, 2048, 4096) for w in (2, 4, 8)
]


@triton.autotune(configs=_BLOCKROW_CONFIGS, key=["n_size"])
@triton.jit
def _gemv_blockrow_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    m_size,
    n_size,
    stride_am,
    stride_an,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= m_size:
        return
    # element-wise fp32 accumulator: lanes stay lane-aligned across tiles, one reduction at the end
    # (cheaper than a tl.sum per iteration). Masked lanes contribute 0.
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n0 in range(0, n_size, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < n_size
        a = tl.load(a_ptr + row * stride_am + offs * stride_an, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += a * x
    y = tl.sum(acc, axis=0)
    tl.store(y_ptr + row, y.to(y_ptr.dtype.element_ty))


def gemv_blockrow(a: Tensor, x: Tensor) -> Tensor:
    """Coalesced block-per-row: grid=(M,), wide ``BLOCK_N`` + ``num_warps`` chosen by autotune. The
    warp's lanes read contiguous A ⇒ fully coalesced loads; the target stage for a large square."""
    m, n = a.shape
    y = torch.empty(m, device=a.device, dtype=a.dtype)
    _gemv_blockrow_kernel[(m,)](a, x, y, m, n, a.stride(0), a.stride(1))
    return y


# ---------------------------------------------------------------------------------------------------
# Stage 3 — split-N two-stage: grid=(M, S), atomic_add partials into an fp32 y.
# ---------------------------------------------------------------------------------------------------


@triton.jit
def _gemv_split_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    m_size,
    n_size,
    stride_am,
    stride_an,
    n_splits,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    sid = tl.program_id(1)
    if row >= m_size:
        return
    chunk = (n_size + n_splits - 1) // n_splits  # this split owns [start, start+chunk)
    start = sid * chunk
    stop = tl.minimum(start + chunk, n_size)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n0 in range(start, stop, BLOCK_N):
        offs = n0 + tl.arange(0, BLOCK_N)
        mask = offs < stop
        a = tl.load(a_ptr + row * stride_am + offs * stride_an, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += a * x
    partial = tl.sum(acc, axis=0)
    tl.atomic_add(y_ptr + row, partial)  # S-deep contention per row; y is fp32


def gemv_split(a: Tensor, x: Tensor, n_splits: int = 8, block_n: int = 1024) -> Tensor:
    """Split-N two-stage: ``S`` programs per row each stream an N-slice and atomic_add their partial.
    Adds ``S×`` CTAs to fill the GPU when M is small (tall-skinny). y is accumulated in fp32 (atomics
    require it), then cast back to A's dtype."""
    m, n = a.shape
    y = torch.zeros(m, device=a.device, dtype=torch.float32)  # atomic target
    _gemv_split_kernel[(m, n_splits)](
        a, x, y, m, n, a.stride(0), a.stride(1), n_splits, BLOCK_N=block_n
    )
    return y.to(a.dtype)
