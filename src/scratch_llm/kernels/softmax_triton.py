"""A2 Rung 2 — row softmax over the last dim of an ``(M, N)`` tensor (the memory-wall reduction).

Softmax is the canonical *memory-bound* kernel: per element it does an exp and a couple of adds
(AI ≈ a few FLOP/byte, far left of the ~130 FLOP/byte ridge on this card), so the only thing that
sets its runtime is HBM traffic. The ladder here is the classic Milakov–Gimelshein "online softmax"
argument (arXiv:1805.02867), expressed as *passes over the row*:

  * **twopass** — safe softmax done as three streaming passes: (1) row max, (2) denominator
    ``Σ exp(x−m)``, (3) normalize + write. HBM traffic ≈ **4N** per row (3 reads + 1 write).
  * **online** — the Milakov recurrence folds max and denominator into ONE streaming pass (running
    ``m`` and ``d`` rescaled together), then a second pass normalizes + writes. HBM ≈ **3N**
    (2 reads + 1 write) → **4/3 ≈ 1.33× fewer bytes** than twopass. This is the same recurrence the
    FA2 kernel (`flash_attention_triton.py`) runs over key tiles — softmax is flash-attention with
    ``V = I``.
  * **fused** — when the whole row fits one tile (``BLOCK_N ≥ N``) the row is loaded ONCE into
    registers; max, exp and sum are register reductions, so HBM ≈ **2N** (1 read + 1 write) — the
    ideal traffic, and the peak-HBM stage this rung targets at large N.

**Numerics (adversarial-hardened, oracle-first).** All accumulation is fp32. The max subtraction
makes a ``+1e4`` outlier row overflow-safe (``exp(x−m) ≤ 1``). A wholly ``−inf`` (fully masked) row
would drive ``exp(−inf − −inf) = exp(nan)``; every exp argument is guarded with an fp32-safe max
(``m_safe = 0`` when the running max is ``−inf``) so masked lanes contribute exactly ``0``, and the
``d == 0`` empty-row test emits a **uniform ``1/N``** row instead of ``0/0`` NaN. An all-equal row
falls out as uniform automatically.

Honesty (A2 guide + kernels/CLAUDE.md): Triton owns the CUDA-level detail this rung would otherwise
drill — global-load **coalescing** (the row is unit-stride, the compiler emits coalesced/vectorized
loads), **vectorization** (float4-equivalent), and the reduction's shared-memory **bank-conflict**
swizzle are all compiler-managed here. The raw-CUDA A2 softmax ladder (a warp-shuffle tree reduction,
``float4`` row loads, a bank-conflict-free SMEM reduction) would add those by hand; this kernel does
not, and says so. ncu is blocked on this box (`ERR_NVGPUCTRPERM`) — the bound is established by
achieved-vs-measured-peak %; the ncu metric to discharge on the H100 day is named in the bench.
"""

from __future__ import annotations

import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _softmax_twopass_kernel(
    x_ptr,
    y_ptr,
    stride_row,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    """Three streaming passes (max, denom, normalize+write) → ~4N HBM per row."""
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    y_row = y_ptr + row * stride_row

    # pass 1 — running max over the streamed row
    m = tl.full((), float("-inf"), tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=0))
    m_safe = tl.where(m == float("-inf"), 0.0, m)  # all-masked row → finite shift, no exp(nan)

    # pass 2 — denominator Σ exp(x − m); masked/tail lanes: exp(−inf) = 0
    d = tl.zeros((), tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
        d += tl.sum(tl.exp(x - m_safe), axis=0)
    empty = d == 0.0  # wholly −inf row
    d_safe = tl.where(empty, 1.0, d)

    # pass 3 — normalize and write; empty row → uniform 1/N (never 0/0 NaN)
    inv_n = 1.0 / n_cols
    for start in range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=float("-inf")).to(tl.float32)
        y = tl.where(empty, inv_n, tl.exp(x - m_safe) / d_safe)
        tl.store(y_row + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_online_kernel(
    x_ptr,
    y_ptr,
    stride_row,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    """Milakov online pass (fused max+denom) + normalize pass → ~3N HBM per row."""
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    y_row = y_ptr + row * stride_row

    # pass 1 — fused running (max, denom) with rescale, one read of the row
    m = tl.full((), float("-inf"), tl.float32)
    d = tl.zeros((), tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        m_new_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        # rescale prior mass; no prior mass while m == −inf, so factor := 0 (avoids exp(nan))
        factor = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new_safe))
        p = tl.exp(x - m_new_safe)  # tail lanes exp(−inf)=0; wholly-masked tile handled below
        p_sum = tl.where(m_new == float("-inf"), 0.0, tl.sum(p, axis=0))
        d = d * factor + p_sum
        m = m_new
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    empty = d == 0.0
    d_safe = tl.where(empty, 1.0, d)

    # pass 2 — normalize and write
    inv_n = 1.0 / n_cols
    for start in range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=float("-inf")).to(tl.float32)
        y = tl.where(empty, inv_n, tl.exp(x - m_safe) / d_safe)
        tl.store(y_row + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_fused_kernel(
    x_ptr,
    y_ptr,
    stride_row,
    n_cols,
    BLOCK_N: tl.constexpr,
):
    """Row-resident single load (BLOCK_N ≥ N): max/exp/sum in registers → ideal 2N HBM per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=float("-inf")).to(tl.float32)

    m = tl.max(x, axis=0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    p = tl.exp(x - m_safe)  # masked/tail lanes: exp(−inf − 0) = 0
    d = tl.sum(p, axis=0)
    empty = d == 0.0
    y = tl.where(empty, 1.0 / n_cols, p / tl.where(empty, 1.0, d))
    tl.store(y_ptr + row * stride_row + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


_STREAM_KERNELS = {"twopass": _softmax_twopass_kernel, "online": _softmax_online_kernel}


def _num_warps(block_n: int) -> int:
    if block_n <= 2048:
        return 4
    if block_n <= 8192:
        return 8
    return 16


def softmax_triton(x: Tensor, mode: str = "fused", *, block_n: int = 1024) -> Tensor:
    """Row softmax of a 2-D ``(M, N)`` tensor over the last dim, matching ``F.softmax(x, dim=-1)``.

    ``mode``:
      * ``"twopass"`` — three streaming passes, ~4N HBM/row (the naive safe-softmax rung).
      * ``"online"``  — Milakov fused max+denom pass + write pass, ~3N HBM/row (~1.33× fewer bytes).
      * ``"fused"``   — row-resident single load, ideal 2N HBM/row (the peak-HBM stage).

    Streaming modes tile the row by ``block_n``; ``"fused"`` uses ``BLOCK_N = next_pow2(N)``.
    Accumulation is fp32; output matches ``x``'s dtype. A wholly ``−inf`` row yields a uniform
    ``1/N`` row (never NaN)."""
    if x.ndim != 2:
        raise ValueError(f"softmax_triton expects a 2-D (M, N) tensor, got shape {tuple(x.shape)}")
    x = x.contiguous()
    m_rows, n_cols = x.shape
    y = x.new_empty((m_rows, n_cols))
    grid = (m_rows,)

    if mode == "fused":
        block = triton.next_power_of_2(n_cols)
        _softmax_fused_kernel[grid](
            x, y, x.stride(0), n_cols, BLOCK_N=block, num_warps=_num_warps(block)
        )
    elif mode in _STREAM_KERNELS:
        block = min(triton.next_power_of_2(n_cols), block_n)
        _STREAM_KERNELS[mode][grid](
            x, y, x.stride(0), n_cols, BLOCK_N=block, num_warps=_num_warps(block)
        )
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'twopass', 'online', or 'fused'")
    return y
