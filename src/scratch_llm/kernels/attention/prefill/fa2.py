"""Triton FlashAttention-2 forward kernel (Algorithm 1) — the GPU port of the oracle.

L2 Systems (A2.1). One program per (query tile, batch·head): load the Q tile once, loop over key
tiles maintaining the online-softmax running max / denominator / accumulator in fp32, write the O
tile and L = logsumexp. This is the kernel the A2.1 roofline benchmarks against
`F.scaled_dot_product_attention`; it must equal the pure-PyTorch oracle in `../reference.py`.

`triton.autotune` picks block sizes / num_warps / num_stages per sequence length — a fair-comparison
necessity, since SDPA dispatches a tuned kernel and an untuned single-config Triton kernel is not a
like-for-like roofline baseline.

GPU-only (imports Triton). Not imported by `kernels/__init__.py`, so importing the package on a
CPU/CI box never pulls Triton; tests gate on `pytest.importorskip("triton")` + the `gpu` marker.
Forward + causal only — the backward is the torch.compile recomputation path, not a hand-rolled
kernel (A2 guide §4.2.3 SKIP). Assumes self-attention (key length == query length).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

_CONFIGS = [
    triton.Config({"BLOCK_Q": bq, "BLOCK_K": bk}, num_warps=w, num_stages=s)
    for bq, bk in [(64, 64), (128, 64), (64, 128), (128, 128)]
    for w in (4, 8)
    for s in (2, 3)
]


@triton.autotune(configs=_CONFIGS, key=["n_ctx", "D"])
@triton.jit
def _fa2_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    stride_qb,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_on,
    stride_od,
    stride_lb,
    stride_ln,
    n_ctx,
    scale,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_start = pid_q * BLOCK_Q

    offs_q = q_start + tl.arange(0, BLOCK_Q)  # (BLOCK_Q,)
    offs_d = tl.arange(0, D)  # (D,)
    q_mask = offs_q < n_ctx

    q_ptrs = q_ptr + pid_b * stride_qb + (offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)  # (BLOCK_Q, D)

    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)  # running row-max
    l_i = tl.zeros((BLOCK_Q,), tl.float32)  # running denominator
    acc = tl.zeros((BLOCK_Q, D), tl.float32)  # running unnormalized output

    # Causal: a query at q_start..q_start+BLOCK_Q-1 never attends keys beyond its own index,
    # so we never loop past key index (q_start + BLOCK_Q) — skips the whole upper triangle.
    k_end = (q_start + BLOCK_Q) if IS_CAUSAL else n_ctx

    for k_start in range(0, k_end, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
        k_mask = offs_k < n_ctx
        k_ptrs = (
            k_ptr + pid_b * stride_kb + (offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        )
        v_ptrs = (
            v_ptr + pid_b * stride_vb + (offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        )
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)  # (BLOCK_K, D)
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)  # (BLOCK_K, D)

        s = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * scale  # (BLOCK_Q, BLOCK_K)
        s = tl.where(k_mask[None, :], s, -float("inf"))  # padded keys → −inf
        if IS_CAUSAL:
            s = tl.where(offs_q[:, None] >= offs_k[None, :], s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))  # (BLOCK_Q,)
        p = tl.exp(s - m_new[:, None])  # (BLOCK_Q, BLOCK_K)
        corr = tl.exp(m_i - m_new)  # (BLOCK_Q,)
        l_i = corr * l_i + tl.sum(p, axis=1)
        acc = corr[:, None] * acc + tl.dot(p, v, allow_tf32=ALLOW_TF32)
        m_i = m_new

    o = acc / l_i[:, None]
    o_ptrs = o_ptr + pid_b * stride_ob + (offs_q[:, None] * stride_on + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=q_mask[:, None])
    l_ptrs = l_ptr + pid_b * stride_lb + offs_q * stride_ln
    tl.store(l_ptrs, m_i + tl.log(l_i), mask=q_mask)


def flash_attention_triton_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    is_causal: bool = False,
    allow_tf32: bool = True,
) -> tuple[Tensor, Tensor]:
    """Triton FA2 forward. ``q,k,v``: ``(..., N, d)`` self-attention (key length == query length).

    Returns ``(O, L)`` matching :func:`scratch_llm.kernels.attention.reference.flash_attention_forward`.
    ``allow_tf32=False`` makes the matmuls bit-faithful to the fp32 oracle (used in the correctness
    test); the default ``True`` is the fast path for the roofline benchmark. Block sizes / warps /
    stages are chosen by ``triton.autotune``."""
    *lead, n, d = q.shape
    if k.shape[-2] != n:
        raise ValueError(f"kernel assumes self-attention (key len {k.shape[-2]} != query len {n})")
    b = math.prod(lead) if lead else 1

    qf = q.reshape(b, n, d).contiguous()
    kf = k.reshape(b, n, d).contiguous()
    vf = v.reshape(b, n, d).contiguous()
    o = torch.empty_like(qf)
    lse = torch.empty((b, n), dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(d)

    def grid(meta: dict) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_Q"]), b)

    _fa2_fwd_kernel[grid](
        qf,
        kf,
        vf,
        o,
        lse,
        qf.stride(0),
        qf.stride(1),
        qf.stride(2),
        kf.stride(0),
        kf.stride(1),
        kf.stride(2),
        vf.stride(0),
        vf.stride(1),
        vf.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        n,
        scale,
        IS_CAUSAL=is_causal,
        ALLOW_TF32=allow_tf32,
        D=d,
    )
    return o.reshape(*lead, n, d), lse.reshape(*lead, n)


@triton.jit
def _fa2_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    scale,
    o_ptr,
    do_ptr,
    l_ptr,
    d_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    stride_qb,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_on,
    stride_od,
    stride_dob,
    stride_dom,
    stride_dod,
    stride_lb,
    stride_ln,
    stride_db,
    stride_dn,
    stride_dqb,
    stride_dqm,
    stride_dqd,
    stride_dkb,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvn,
    stride_dvd,
    n_ctx,
    ALLOW_TF32: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D)
    k_mask = offs_k < n_ctx

    # Load K and V tiles
    k_ptrs = k_ptr + pid_b * stride_kb + (offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = v_ptr + pid_b * stride_vb + (offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

    # Initialize accumulated gradients for K and V in SRAM registers
    dk = tl.zeros((BLOCK_K, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_K, D), dtype=tl.float32)

    q_start = 0
    if IS_CAUSAL:
        q_start = (pid_k * BLOCK_K // BLOCK_Q) * BLOCK_Q

    for q_idx in range(q_start, n_ctx, BLOCK_Q):
        offs_q = q_idx + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < n_ctx

        # Load Q, dO
        q_ptrs = (
            q_ptr + pid_b * stride_qb + (offs_q[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        )
        do_ptrs = (
            do_ptr
            + pid_b * stride_dob
            + (offs_q[:, None] * stride_dom + offs_d[None, :] * stride_dod)
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        # Load LSE and D vectors
        l_ptrs = l_ptr + pid_b * stride_lb + offs_q * stride_ln
        d_ptrs = d_ptr + pid_b * stride_db + offs_q * stride_dn
        lse = tl.load(l_ptrs, mask=q_mask, other=float("-inf")).to(tl.float32)
        d_vec = tl.load(d_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        # Compute S_ij = (Q_i @ K_j.T) * scale
        s = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * scale
        s = tl.where(k_mask[None, :], s, -float("inf"))
        if IS_CAUSAL:
            s = tl.where(offs_q[:, None] >= offs_k[None, :], s, -float("inf"))

        # Compute P_ij = exp(S_ij - LSE_i)
        p = tl.exp(s - lse[:, None])
        p = tl.where(q_mask[:, None], p, 0.0)

        # Compute dP_ij = dO_i @ V_j.T
        dp = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)

        # Compute dS_ij = P_ij * (dP_ij - D_i) * scale
        ds = p * (dp - d_vec[:, None]) * scale
        ds = tl.where(q_mask[:, None], ds, 0.0)

        # Accumulate dk and dv
        dk += tl.dot(tl.trans(ds), q, allow_tf32=ALLOW_TF32)
        dv += tl.dot(tl.trans(p), do, allow_tf32=ALLOW_TF32)

        # Atomic add to global dQ_i
        dq_ptrs = (
            dq_ptr
            + pid_b * stride_dqb
            + (offs_q[:, None] * stride_dqm + offs_d[None, :] * stride_dqd)
        )
        dq_val = tl.dot(ds, k, allow_tf32=ALLOW_TF32)
        tl.atomic_add(dq_ptrs, dq_val.to(dq_ptr.dtype.element_ty), mask=q_mask[:, None])

    # Store dk and dv
    dk_ptrs = (
        dk_ptr + pid_b * stride_dkb + (offs_k[:, None] * stride_dkn + offs_d[None, :] * stride_dkd)
    )
    dv_ptrs = (
        dv_ptr + pid_b * stride_dvb + (offs_k[:, None] * stride_dvn + offs_d[None, :] * stride_dvd)
    )
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), mask=k_mask[:, None])
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), mask=k_mask[:, None])


def flash_attention_triton_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    lse: Tensor,
    do: Tensor,
    *,
    is_causal: bool = False,
    allow_tf32: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Triton FA2 backward. ``q,k,v,o,do``: ``(..., N, d)``, ``lse``: ``(..., N)``."""
    *lead, n, d = q.shape
    b = math.prod(lead) if lead else 1

    qf = q.reshape(b, n, d).contiguous()
    kf = k.reshape(b, n, d).contiguous()
    vf = v.reshape(b, n, d).contiguous()
    of = o.reshape(b, n, d).contiguous()
    dof = do.reshape(b, n, d).contiguous()
    lsef = lse.reshape(b, n).contiguous()

    # Pre-compute D vector (rowsum of dO * O)
    df = (of.to(torch.float32) * dof.to(torch.float32)).sum(dim=-1).contiguous()

    # Initialize outputs
    dq = torch.zeros_like(qf)
    dk = torch.empty_like(kf)
    dv = torch.empty_like(vf)

    scale = 1.0 / math.sqrt(d)
    BLOCK_Q = 64
    BLOCK_K = 64

    grid = (triton.cdiv(n, BLOCK_K), b)

    _fa2_bwd_kernel[grid](
        qf,
        kf,
        vf,
        scale,
        of,
        dof,
        lsef,
        df,
        dq,
        dk,
        dv,
        qf.stride(0),
        qf.stride(1),
        qf.stride(2),
        kf.stride(0),
        kf.stride(1),
        kf.stride(2),
        vf.stride(0),
        vf.stride(1),
        vf.stride(2),
        of.stride(0),
        of.stride(1),
        of.stride(2),
        dof.stride(0),
        dof.stride(1),
        dof.stride(2),
        lsef.stride(0),
        lsef.stride(1),
        df.stride(0),
        df.stride(1),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        n,
        ALLOW_TF32=allow_tf32,
        D=d,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        IS_CAUSAL=is_causal,
        num_warps=4,
    )

    return dq.reshape(*lead, n, d), dk.reshape(*lead, n, d), dv.reshape(*lead, n, d)


class TritonFlashAttention(torch.autograd.Function):
    """Triton FlashAttention-2 custom autograd Function."""

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        allow_tf32: bool = True,
    ) -> Tensor:
        o, lse = flash_attention_triton_forward(q, k, v, is_causal=is_causal, allow_tf32=allow_tf32)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        ctx.allow_tf32 = allow_tf32
        return o

    @staticmethod
    def backward(ctx, do: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None]:
        q, k, v, o, lse = ctx.saved_tensors
        is_causal = ctx.is_causal
        allow_tf32 = ctx.allow_tf32
        dq, dk, dv = flash_attention_triton_backward(
            q, k, v, o, lse, do, is_causal=is_causal, allow_tf32=allow_tf32
        )
        return dq, dk, dv, None, None
