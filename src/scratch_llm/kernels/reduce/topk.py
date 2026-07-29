"""Top-K over the last dim of ``(M, N)`` — A2 Rung 4 (the "measure the failure" rung).

Top-K is a deliberately *poor* GPU fit, and the point of this rung is to measure that honestly.
Given a ``(M, N)`` matrix we return the ``k`` largest values of each row plus their column indices
(``k`` small, e.g. 8), matching ``torch.topk(x, k, dim=-1, sorted=True)`` on tie-free inputs.

Design: one Triton program per row. The whole row is loaded once into registers, then ``k``
*iterative max-extraction* passes each do a full-row tree reduction to pull the current maximum and
mask it out for the next pass. This is a latency/occupancy-bound selection, not a streaming
bandwidth kernel — the ``k`` reduction passes form a serial dependency chain and the arithmetic
intensity is ~0 (comparisons, not FLOPs), so the achieved %-of-HBM-peak is *low by construction*.
That low number is the correct, expected result (see ``bench/topk.py`` / ``bench/RESULTS.md``).

Tie-breaking rule (DEFINED here; ``torch.topk``'s tie order is unspecified so we do not compare to
it on ties): **lowest column index wins** — among equal values the smaller index is ranked higher.
The extraction picks ``tl.min`` over the argmax candidate set, which realizes exactly that rule.

Also exposes :func:`fused_softmax_topk`: softmax is monotonic, so the top-k of the softmax equals the
top-k of the logits by *index*; fusing lets us read the row from HBM once and emit only the ``k``
probabilities, saving the softmax write + re-read round trip that a separate ``softmax`` then
``topk`` pays (traffic win measured in the bench).

GPU-only (imports Triton); not re-exported by ``kernels/__init__.py`` so a CPU import never pulls
Triton. Tests gate on the ``gpu`` marker + ``pytest.importorskip("triton")``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _topk_kernel(
    x_ptr,
    val_ptr,
    idx_ptr,
    n_cols,
    stride_xm,
    stride_om,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    # other=-inf: padded lanes can never be selected (n_cols >= K is required by the caller).
    r = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=float("-inf")).to(tl.float32)

    for i in range(K):
        maxv = tl.max(r, axis=0)  # scalar row-max
        is_max = r == maxv
        # lowest-index-wins: among the argmax lanes take the smallest column index.
        idx = tl.min(tl.where(is_max, offs, n_cols), axis=0)
        tl.store(val_ptr + row * stride_om + i, maxv)
        tl.store(idx_ptr + row * stride_om + i, idx)
        r = tl.where(offs == idx, float("-inf"), r)  # remove the winner for the next pass


@triton.jit
def _fused_softmax_topk_kernel(
    x_ptr,
    val_ptr,
    idx_ptr,
    n_cols,
    stride_xm,
    stride_om,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    r = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=float("-inf")).to(tl.float32)

    # Full-row softmax stats from the single load (denominator over ALL N, not just the top-k).
    m = tl.max(r, axis=0)
    denom = tl.sum(tl.where(mask, tl.exp(r - m), 0.0), axis=0)

    # Top-k of the softmax == top-k of the logits by index (softmax is monotone increasing).
    for i in range(K):
        maxv = tl.max(r, axis=0)
        is_max = r == maxv
        idx = tl.min(tl.where(is_max, offs, n_cols), axis=0)
        prob = tl.exp(maxv - m) / denom  # softmax probability of the selected logit
        tl.store(val_ptr + row * stride_om + i, prob)
        tl.store(idx_ptr + row * stride_om + i, idx)
        r = tl.where(offs == idx, float("-inf"), r)


def _launch(kernel, x: Tensor, k: int, val_dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    if x.ndim != 2:
        raise ValueError(f"top-k expects a 2-D (M, N) matrix, got shape {tuple(x.shape)}")
    m, n = x.shape
    if not 1 <= k <= n:
        raise ValueError(f"k={k} out of range for N={n}")
    x = x.contiguous()
    values = torch.empty((m, k), device=x.device, dtype=val_dtype)
    indices = torch.empty((m, k), device=x.device, dtype=torch.int32)
    block_n = triton.next_power_of_2(n)
    num_warps = 4 if block_n <= 2048 else (8 if block_n <= 8192 else 16)
    kernel[(m,)](
        x,
        values,
        indices,
        n,
        x.stride(0),
        values.stride(0),
        K=k,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return values, indices.to(torch.int64)


def topk_last_dim(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Row-wise top-``k`` values + indices, matching ``torch.topk(x, k, dim=-1, sorted=True)``.

    ``x``: ``(M, N)``. Returns ``(values (M, k) in x's dtype, indices (M, k) int64)``, descending by
    value. Ties break to the **lowest column index** (defined rule; see module docstring).
    """
    return _launch(_topk_kernel, x, k, x.dtype)


def fused_softmax_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Fused ``softmax`` + top-``k``: reads the row once, returns the top-``k`` **softmax
    probabilities** (fp32) + their int64 indices, descending. Equivalent to
    ``torch.topk(torch.softmax(x.float(), -1), k)`` but without materializing the full softmax to
    HBM (saves the write + re-read a two-pass path pays)."""
    return _launch(_fused_softmax_topk_kernel, x, k, torch.float32)
