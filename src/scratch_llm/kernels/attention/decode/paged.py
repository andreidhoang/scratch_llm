"""Fused paged decode attention — A1 R4.1b (the traffic-reclaim half of PagedAttention).

One query token per row, KV read directly through the block table: no gathered padded view, no
materialized ``(B, H, L)`` score matrix, no ``repeat_interleave`` GQA copies — the three
components of the R3b-measured mixed-age padding tax (P4.1.4). Grid = ``(B, H_q)``; each program
walks its row's blocks with an online softmax in fp32 (the same recurrence as the owned A2 FA2
kernel, specialized to a single query row over 16-token pages).

Invariant (tests/test_paged_kernel.py, gpu): matches the gather-path SDPA to bf16 tolerance and
is greedy-token-exact end-to-end. Write-then-mask convention: row ``b`` attends
``lengths[b] + 1`` keys (its history plus the key written this step).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _paged_decode_kernel(
    q_ptr,
    pool_k_ptr,
    pool_v_ptr,
    table_ptr,
    lengths_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_pb,
    stride_ph,
    stride_pt,
    stride_pd,
    stride_tb,
    stride_ti,
    stride_ob,
    stride_oh,
    stride_od,
    group_size,
    scale,
    BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    kv_h = h // group_size  # GQA: query head h reads its group's shared KV head

    d_off = tl.arange(0, HEAD_DIM)
    t_off = tl.arange(0, BLOCK)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d_off * stride_qd).to(tl.float32)

    n_keys = tl.load(lengths_ptr + b) + 1  # write-then-mask: self-inclusive
    n_blocks = (n_keys + BLOCK - 1) // BLOCK

    # Loop-carried scalars as explicitly fp32 (1,)-tensors: a python-float carry promotes to
    # fp64 under torch.compile's stricter Triton type inference and fails to compile.
    m = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_sum = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for i in range(n_blocks):
        phys = tl.load(table_ptr + b * stride_tb + i * stride_ti)
        base = phys * stride_pb + kv_h * stride_ph
        kv_ptrs = base + t_off[:, None] * stride_pt + d_off[None, :] * stride_pd
        k_blk = tl.load(pool_k_ptr + kv_ptrs).to(tl.float32)  # (BLOCK, D)
        # explicit fp32 cast: `scale` (runtime float arg) is inferred f64 under torch.compile's
        # Triton wrapper and would promote the whole online-softmax carry chain to f64
        scores = (tl.sum(k_blk * q[None, :], axis=1) * scale).to(tl.float32)  # (BLOCK,)
        valid = (i * BLOCK + t_off) < n_keys
        scores = tl.where(valid, scores, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=0))  # (1,)
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new)  # masked lanes: exp(-inf) = 0 exactly
        v_blk = tl.load(pool_v_ptr + kv_ptrs).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_blk, axis=0)
        l_sum = l_sum * alpha + tl.sum(p, axis=0)
        m = m_new
    out = acc / l_sum  # n_keys ≥ 1 always (write-then-mask) → l_sum > 0, no NaN row
    tl.store(out_ptr + b * stride_ob + h * stride_oh + d_off * stride_od, out)


def paged_decode_attention(
    q: Tensor, pool_k: Tensor, pool_v: Tensor, block_table: Tensor, lengths: Tensor
) -> Tensor:
    """softmax(q·Kᵀ/√d)·V for one query token per row, K/V paged behind ``block_table``.

    q: ``(B, H, 1, d)`` post-RoPE; pool_k/pool_v: ``(n_blocks, H_kv, 16, d)``;
    block_table: ``(B, max_blocks)`` long; lengths: ``(B,)`` pre-write lengths.
    Returns ``(B, H, 1, d)`` in q's dtype (fp32 accumulation in-kernel).
    """
    bsz, n_heads, s, head_dim = q.shape
    assert s == 1, "paged decode attends exactly one query token per row"
    assert head_dim & (head_dim - 1) == 0, "HEAD_DIM must be a power of two (tl.arange)"
    n_kv_heads = pool_k.shape[1]
    block = pool_k.shape[2]
    q3 = q[:, :, 0].contiguous()
    out = torch.empty((bsz, n_heads, head_dim), device=q.device, dtype=torch.float32)
    grid = (bsz, n_heads)
    _paged_decode_kernel[grid](
        q3,
        pool_k,
        pool_v,
        block_table,
        lengths,
        out,
        *q3.stride(),
        *pool_k.stride(),
        *block_table.stride(),
        *out.stride(),
        n_heads // n_kv_heads,
        1.0 / math.sqrt(head_dim),
        BLOCK=block,
        HEAD_DIM=head_dim,
        num_warps=4,
    )
    return out.unsqueeze(2).to(q.dtype)
