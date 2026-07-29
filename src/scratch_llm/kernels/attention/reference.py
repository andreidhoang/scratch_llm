"""Pure-PyTorch FlashAttention-2 forward — the correctness ORACLE for the Triton port.

L2 Systems (A2.1). Implements the FA2 tiled / online-softmax recurrence (Algorithm 1) in plain
PyTorch: outer loop over query tiles, inner loop over key tiles, maintaining a running max `m`,
denominator `l`, and output accumulator `acc`, rescaling on every tile. This is the *same* recurrence
the Triton kernel implements — so it is the oracle that kernel is tested against (alongside
`F.scaled_dot_product_attention`). It runs on CPU and needs no GPU.

Falsifiable invariant (tests/test_flash_attention.py): for random Q,K,V the tiled output equals
`F.scaled_dot_product_attention` to floating-point tolerance, causal and non-causal, including ragged
tiles (sequence length not a multiple of the tile size). Kill criterion: any mismatch ⇒ the online
softmax rescale or the causal masking is wrong, and the Triton port has no trustworthy oracle.

Returns `(O, L)` where `L = logsumexp(scores)` per query — the FA2 log-normalizer the backward pass
needs (the backward itself is out of scope: torch.compile recomputation, not a hand-rolled kernel).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def flash_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    is_causal: bool = False,
    q_tile: int = 64,
    k_tile: int = 64,
) -> tuple[Tensor, Tensor]:
    """Tiled FA2 forward. ``q,k,v`` are ``(..., N, d)`` (leading dims batched). Scale = 1/√d.

    Computation is done in fp32 for numerical fidelity (the rescale subtracts a running max, so
    precision matters), then cast back to the input dtype for ``O``; ``L`` stays fp32."""
    *lead, n_q, d = q.shape
    n_k = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    b = math.prod(lead) if lead else 1
    # Accumulate in fp32 at minimum (the running-max rescale needs the headroom), but preserve
    # fp64 so this stays a high-precision oracle — `.float()` would silently cap fp64 at fp32.
    acc_dtype = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float32

    qf = q.reshape(b, n_q, d)
    kf = k.reshape(b, n_k, d)
    vf = v.reshape(b, n_k, d)
    o = torch.empty_like(qf)
    lse = torch.empty(b, n_q, dtype=acc_dtype, device=q.device)

    for qs in range(0, n_q, q_tile):
        qe = min(qs + q_tile, n_q)
        bq = qe - qs
        q_i = qf[:, qs:qe].to(acc_dtype)  # (b, bq, d)
        q_pos = torch.arange(qs, qe, device=q.device)

        m = torch.full((b, bq), float("-inf"), device=q.device, dtype=acc_dtype)  # running row-max
        ell = torch.zeros((b, bq), device=q.device, dtype=acc_dtype)  # running denominator
        acc = torch.zeros((b, bq, d), device=q.device, dtype=acc_dtype)  # unnormalized output

        for ks in range(0, n_k, k_tile):
            ke = min(ks + k_tile, n_k)
            if is_causal and ks > qe - 1:
                break  # this key tile (and all later) is entirely in the future
            k_j = kf[:, ks:ke].to(acc_dtype)  # (b, bk, d)
            v_j = vf[:, ks:ke].to(acc_dtype)
            s = torch.einsum("bqd,bkd->bqk", q_i, k_j) * scale  # (b, bq, bk)
            if is_causal:
                k_pos = torch.arange(ks, ke, device=q.device)
                disallow = k_pos[None, :] > q_pos[:, None]  # (bq, bk), True = future
                s = s.masked_fill(disallow.unsqueeze(0), float("-inf"))

            m_new = torch.maximum(m, s.amax(dim=-1))  # (b, bq)
            p = torch.exp(s - m_new.unsqueeze(-1))  # (b, bq, bk)
            corr = torch.exp(m - m_new)  # (b, bq); 0 on the first tile (m=-inf)
            ell = corr * ell + p.sum(dim=-1)
            acc = corr.unsqueeze(-1) * acc + torch.einsum("bqk,bkd->bqd", p, v_j)
            m = m_new

        o[:, qs:qe] = (acc / ell.unsqueeze(-1)).to(o.dtype)
        lse[:, qs:qe] = m + torch.log(ell)

    return o.reshape(*lead, n_q, d), lse.reshape(*lead, n_q)


def _causal_mask(n_q: int, n_k: int, device: torch.device) -> Tensor:
    """``(n_q, n_k)`` bool, True where a key is in the query's future (must be masked).

    Uses absolute positions so it matches the forward's tiled masking even when ``n_q != n_k``."""
    q_idx = torch.arange(n_q, device=device)
    k_idx = torch.arange(n_k, device=device)
    return k_idx[None, :] > q_idx[:, None]


class FlashAttentionPyTorch(torch.autograd.Function):
    """FA2 forward (tiled oracle) + the recomputation backward (Eqs 13–19), pure PyTorch.

    The point is **recompute-vs-store**: the forward persists only ``(Q,K,V,O,L)`` — all O(N·d) —
    and never the N×N probability matrix ``P``. The backward recomputes ``S`` and ``P`` from the
    saved tensors, so no O(N²) bytes ever cross the fwd→bwd boundary. The softmax-Jacobian's
    recentering term collapses to the **D-vector** ``D_i = Σ_d O_id·dO_id = Σ_j P_ij·dP_ij`` — a
    cheap d-wide reduction over tensors we already have, no ``dP`` needed to form it.

    Falsifiable invariant (tests/test_flash_attention.py): ``dQ,dK,dV`` equal plain attention's
    autograd gradients to fp tolerance, causal and non-causal. Kill criterion: any mismatch ⇒ a
    transpose in one of the backward einsums or a wrong causal recompute.
    """

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:  # type: ignore[override]
        o, lse = flash_attention_forward(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do: Tensor):  # type: ignore[override]
        q, k, v, o, lse = ctx.saved_tensors
        is_causal: bool = ctx.is_causal
        d = q.shape[-1]
        n_q, n_k = q.shape[-2], k.shape[-2]
        scale = 1.0 / math.sqrt(d)

        # Recompute in fp32 at minimum (the rescale needs headroom); keep fp64 if the input is fp64.
        acc_dtype = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float32
        qf, kf, vf = q.to(acc_dtype), k.to(acc_dtype), v.to(acc_dtype)
        of, dof = o.to(acc_dtype), do.to(acc_dtype)

        # Recompute the probability matrix P from saved (Q,K,V,L) — never stored in forward.
        s = torch.einsum("...qd,...kd->...qk", qf, kf) * scale  # (..., n_q, n_k)
        if is_causal:
            s = s.masked_fill(_causal_mask(n_q, n_k, q.device), float("-inf"))
        p = torch.exp(s - lse.to(acc_dtype).unsqueeze(-1))  # (..., n_q, n_k)

        d_vec = (of * dof).sum(dim=-1)  # D_i = O_i·dO_i  (..., n_q)
        dv = torch.einsum("...qk,...qd->...kd", p, dof)  # Pᵀ dO        (..., n_k, d)
        dp = torch.einsum("...qd,...kd->...qk", dof, vf)  # dO Vᵀ        (..., n_q, n_k)
        ds = p * (dp - d_vec.unsqueeze(-1))  # softmax backward via D    (..., n_q, n_k)
        dq = torch.einsum("...qk,...kd->...qd", ds, kf) * scale  # (..., n_q, d)
        dk = torch.einsum("...qk,...qd->...kd", ds, qf) * scale  # (..., n_k, d)

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None
