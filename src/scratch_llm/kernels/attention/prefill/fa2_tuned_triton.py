"""K2/A-R1 — the Triton kernels: block-skipped forward, exp2 path, split backward.

The GPU half of ``fa2_tuned``. That module holds every decision that can be made without silicon
— which configs an arch can afford, which blocks a query tile must visit, what the scale and the
LSE units are, how the backward splits — and this one holds the three kernels that act on those
decisions, plus the launchers.

WHAT IS ALREADY HERE (agent-written):
  * the forward's two-phase k-loop: an unmasked phase over the blocks wholly below the diagonal
    and a masked phase over the one block that straddles it, with the boundary coming from
    ``fa2_tuned.kv_span`` restated in the kernel's own arithmetic and asserted equal on CPU;
  * the arch-keyed launch path: the config list is chosen on the HOST by
    ``fa2_tuned.require_configs`` for this device and head dim, then handed to
    ``triton.autotune`` — so a config that cannot fit this arch's shared memory is never compiled,
    which is the fix for the sm_120 overflow;
  * the exp2 plumbing: a log2e-prescaled scale in, ``ln 2`` out, so LSE leaves in natural units;
  * both backward kernels — dK/dV per KV block, dQ per query block, each accumulating its own
    output in registers, no atomics anywhere;
  * every mask, bound, stride and store.

WHAT IS HUY'S (the hole, :func:`_online_softmax_step`): the running max / denominator update and
the O rescale. The plan row names "fewer O-rescales" as this rung's lesson, so the shape of that
arithmetic — how many times ``acc`` is touched per visited block, whether the correction is
applied to ``acc`` or carried — is the thing being learned and is not written here.

RUNNING IT: there is no dry dock for this rung. A ``.cu`` can be compiled in a container on a
laptop and its SASS asserted on; a Triton kernel is compiled by the driver, on the device, at first
call, so there is nothing to compile here and no ptxas report to read. Everything a dry dock would
have caught — the shared-memory budget, the block plan, the scale units — is caught by the CPU
tier of ``tests/kernels/attention/test_k2_a_r1.py`` instead, from published constants rather than
from a compiler's output. That is a weaker check than reading real SASS and is worth saying aloud.

There is also no ``LADDERS_STUB_HOLES`` placeholder: a stub body would still be Triton, and Triton
does not run on a CPU box, so it would buy nothing that the CPU tier does not already have.

Spec, floor and kill rule: ``experiments/K2/A-R1/spec.md``.
"""

from __future__ import annotations

import math
from functools import cache

import torch
import triton
import triton.language as tl
from torch import Tensor

from scratch_llm.kernels.attention.prefill.fa2_tuned import (
    LN2,
    LOG2E,
    Kind,
    require_configs,
    softmax_scale,
)
from scratch_llm.kernels.common.arch import compute_capability

# =============================================================================================
# The hole — the online-softmax rescale core
# =============================================================================================


@triton.jit
def _online_softmax_step(m_i, l_i, acc, s, v, USE_EXP2: tl.constexpr):
    """One block's contribution to the running (max, denominator, output). THE HOLE.

    Contract, frozen so that neither call site nor test moves when this is written:

    ``m_i``   (BLOCK_Q,) fp32 running row-max of the scores seen so far, in the SAME log base as
              ``s`` — log2 when ``USE_EXP2``, natural otherwise. The epilogue converts; nothing
              inside here should.
    ``l_i``   (BLOCK_Q,) fp32 running denominator. Base-free: ``sum(exp2(s2 - m2))`` and
              ``sum(exp(s - m))`` are the same number, which is why only ``m_i`` is converted at
              the end.
    ``acc``   (BLOCK_Q, D) fp32 running unnormalised output. NOT divided by ``l_i`` here — the
              epilogue does that once, which is already one rescale saved.
    ``s``     (BLOCK_Q, BLOCK_K) fp32 scores for this block, ALREADY scaled (the scale is
              log2e-prescaled on the host when ``USE_EXP2``) and ALREADY masked to ``-inf`` where
              the key is out of bounds or in the query's future. Nothing left to mask here.
    ``v``     (BLOCK_K, D) value tile in its INPUT dtype (bf16/fp16), not fp32.

    Returns ``(m_i, l_i, acc)`` updated with this block.

    Three things live in here and nowhere else:

      * the exponential: ``tl.math.exp2`` when ``USE_EXP2`` (the reason the scale was prescaled at
        all — see ``fa2_tuned.softmax_scale``), ``tl.exp`` otherwise. The backward kernels below
        show the exp2 call in context if a worked example helps.
      * the correction factor and where it is applied. ``fa2.py:101-106`` spends one full-width
        multiply on ``acc`` every block; the plan row's lesson for this rung is "fewer O-rescales",
        so how many times ``acc`` is touched per block is the variable being learned.
      * the PV product. It is inside this function, not outside, because the rescale and the
        accumulate are one pass over ``acc`` and separating them is the extra rescale. ``p`` must
        be cast to ``v.dtype`` before ``tl.dot`` — an fp32 A-operand takes the TF32 path and
        silently changes both the throughput and the numerics this rung is measured on
        (``oss/vllm/vllm/v1/attention/ops/triton_prefill_attention.py:187-188`` casts, then uses
        the three-argument accumulating ``tl.dot(p, v, acc)``).

    If a different carried state is wanted — a deferred correction, a second accumulator — change
    this signature and its two call sites in ``_fa2_tuned_fwd_kernel``. Both are in this file.
    """
    # HUY: the online-softmax rescale core — running max/denominator update, the O rescale, and the PV accumulate — spec: experiments/K2/A-R1/spec.md — fill before A-R1
    raise NotImplementedError("HUY: A-R1 online-softmax rescale core unwritten")


# =============================================================================================
# Forward
# =============================================================================================


@triton.jit
def _fa2_tuned_fwd_kernel(
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
    ln2,
    IS_CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    SKIP_MASKED_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per (query block, batch·head). ``scale`` is log2e-prescaled when ``USE_EXP2``.

    ``ln2`` arrives as a runtime scalar rather than a literal so that the constant here and the one
    in ``fa2_tuned.lse_from_running_state`` cannot drift: there is exactly one definition of it in
    the repo and the launcher passes it.
    """
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_start = pid_q * BLOCK_Q

    offs_q = q_start + tl.arange(0, BLOCK_Q)  # (BLOCK_Q,)
    offs_d = tl.arange(0, D)  # (D,)
    q_mask = offs_q < n_ctx

    q_ptrs = q_ptr + pid_b * stride_qb + (offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    # Loaded in the INPUT dtype, not up-cast to fp32 the way fa2.py:74 does. The up-cast is what
    # makes that kernel's dots take the fp32/TF32 path; here the operands stay 16-bit, the
    # accumulator stays fp32 (tl.dot's default output for 16-bit inputs), and the shared-memory
    # model in fa2_tuned.smem_terms — which counts 2 bytes per element — is a true statement about
    # this kernel rather than about half of it.
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)  # (BLOCK_Q, D)

    m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)  # running row-max, in `scale`'s log base
    l_i = tl.zeros((BLOCK_Q,), tl.float32)  # running denominator
    acc = tl.zeros((BLOCK_Q, D), tl.float32)  # running unnormalised output

    # The visit plan. fa2_tuned.kv_span is the same arithmetic in Python and the CPU suite asserts
    # the two agree at every (BLOCK_Q, BLOCK_K, n_ctx) this rung can launch.
    if IS_CAUSAL:
        masked_end = tl.minimum(n_ctx, q_start + BLOCK_Q)
        full_end = (q_start // BLOCK_K) * BLOCK_K
    else:
        masked_end = n_ctx
        full_end = (n_ctx // BLOCK_K) * BLOCK_K
    if not SKIP_MASKED_BLOCKS:
        # The lever, off: every visited block goes through the masked path, as in fa2.py:96-99.
        # A flag that cannot be disabled is a claim that cannot be falsified, and A-R1's number is
        # only interpretable if each of this rung's four levers can be measured alone.
        full_end = 0

    # Phase 1 — blocks wholly below the diagonal. No causal mask (every key is visible to every
    # query in the tile) and no bounds mask either: full_end is a multiple of BLOCK_K and is at or
    # below n_ctx, so these blocks are entirely in range. That is the whole saving — vLLM's Triton
    # prefill clamps the loop bound the same way (triton_prefill_attention.py:122) but still
    # evaluates `pos_q >= pos_k` on every tile it visits (:144).
    for k_start in range(0, full_end, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k = tl.load(
            k_ptr + pid_b * stride_kb + (offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        )
        v = tl.load(
            v_ptr + pid_b * stride_vb + (offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        )
        s = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_Q, BLOCK_K) fp32
        m_i, l_i, acc = _online_softmax_step(m_i, l_i, acc, s, v, USE_EXP2=USE_EXP2)

    # Phase 2 — the diagonal (causal) or the ragged tail (non-causal). One block for
    # BLOCK_K >= BLOCK_Q, at most BLOCK_Q/BLOCK_K + 1 otherwise.
    for k_start in range(full_end, masked_end, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < n_ctx
        k = tl.load(
            k_ptr + pid_b * stride_kb + (offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd),
            mask=k_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + pid_b * stride_vb + (offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd),
            mask=k_mask[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(k_mask[None, :], s, -float("inf"))
        if IS_CAUSAL:
            s = tl.where(offs_q[:, None] >= offs_k[None, :], s, -float("inf"))
        m_i, l_i, acc = _online_softmax_step(m_i, l_i, acc, s, v, USE_EXP2=USE_EXP2)

    o = acc / l_i[:, None]
    o_ptrs = o_ptr + pid_b * stride_ob + (offs_q[:, None] * stride_on + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=q_mask[:, None])

    # LSE leaves in NATURAL log units whatever base the mainloop ran in. Everything downstream —
    # both backward kernels, kernels/attention/reference.py — reads it that way, and the flag must
    # not leak past here. A slip is invisible in O (which never touches m_i) and wrong in the
    # gradients by exp((1 - ln2)*m) per row. fa2_tuned.lse_from_running_state is this line.
    lse = (m_i * ln2 if USE_EXP2 else m_i) + tl.log(l_i)
    tl.store(l_ptr + pid_b * stride_lb + offs_q * stride_ln, lse, mask=q_mask)


# =============================================================================================
# Backward — two kernels, no atomics
# =============================================================================================


@triton.jit
def _softmax_probs(s, lse, log2e, USE_EXP2: tl.constexpr):
    """``exp(s - lse)`` with ``s`` in whichever log base the caller scaled it in.

    Shared by both backward kernels — the forward's exponential lives inside the hole instead,
    because there it is fused with the running-max correction and cannot be factored out. ``lse``
    always arrives in NATURAL log units (that is the contract the forward's epilogue keeps), so on
    the exp2 path it is the one that has to be converted, not ``s``.

    The `if` is not a ternary on purpose: `p = a if USE_EXP2 else b` reads as a runtime select over
    two tensors, and the whole point of USE_EXP2 being tl.constexpr is that only one of the two is
    ever built.
    """
    if USE_EXP2:  # noqa: SIM108
        p = tl.math.exp2(s - lse[:, None] * log2e)
    else:
        p = tl.exp(s - lse[:, None])
    return p


@triton.jit
def _bwd_dkdv_block(
    q_ptr,
    do_ptr,
    l_ptr,
    d_ptr,
    k,
    v,
    dk,
    dv,
    pid_b,
    q_start,
    offs_k,
    offs_d,
    stride_qb,
    stride_qn,
    stride_qd,
    stride_dob,
    stride_don,
    stride_dod,
    stride_lb,
    stride_ln,
    stride_db,
    stride_dn,
    n_ctx,
    scale_exp,
    scale_grad,
    log2e,
    APPLY_CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One (query block, KV block) pair's contribution to dK/dV. Both phases call this; only
    ``APPLY_CAUSAL`` differs, which is the entire point of splitting the loop."""
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < n_ctx

    q = tl.load(
        q_ptr + pid_b * stride_qb + (offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd),
        mask=q_mask[:, None],
        other=0.0,
    )
    do = tl.load(
        do_ptr + pid_b * stride_dob + (offs_q[:, None] * stride_don + offs_d[None, :] * stride_dod),
        mask=q_mask[:, None],
        other=0.0,
    )
    # `other=inf` on LSE, not -inf: a padded row then gives s - lse = -inf and p = 0 exactly,
    # where fa2.py:262's -inf gives (-inf) - (-inf) = NaN and relies on a later `where` to erase
    # it. The `where` below is still the load-bearing guard — this only keeps a NaN from ever
    # reaching a tl.dot, where one poisoned lane contaminates a whole accumulator tile.
    lse = tl.load(l_ptr + pid_b * stride_lb + offs_q * stride_ln, mask=q_mask, other=float("inf"))
    dvec = tl.load(d_ptr + pid_b * stride_db + offs_q * stride_dn, mask=q_mask, other=0.0)

    s = tl.dot(q, tl.trans(k)) * scale_exp  # (BLOCK_Q, BLOCK_K) fp32
    if APPLY_CAUSAL:
        s = tl.where(offs_q[:, None] >= offs_k[None, :], s, -float("inf"))
    p = _softmax_probs(s, lse, log2e, USE_EXP2=USE_EXP2)
    p = tl.where(q_mask[:, None], p, 0.0)

    dp = tl.dot(do, tl.trans(v))  # (BLOCK_Q, BLOCK_K) fp32
    ds = p * (dp - dvec[:, None]) * scale_grad

    # P and dS go back to the operand dtype for the accumulating dots. That is not a shortcut: an
    # fp32 A-operand takes the TF32 path, which is a different instruction with different
    # numerics from the bf16 one the forward used, and the backward's tolerance would then be
    # measuring two things at once.
    dv += tl.dot(tl.trans(p).to(do.dtype), do)
    dk += tl.dot(tl.trans(ds).to(q.dtype), q)
    return dk, dv


@triton.jit
def _fa2_tuned_bwd_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    l_ptr,
    d_ptr,
    dk_ptr,
    dv_ptr,
    stride_qb,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vd,
    stride_dob,
    stride_don,
    stride_dod,
    stride_lb,
    stride_ln,
    stride_db,
    stride_dn,
    stride_dkb,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvn,
    stride_dvd,
    n_ctx,
    scale_exp,
    scale_grad,
    log2e,
    IS_CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    SKIP_MASKED_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per KV block. Owns dK/dV for that block outright — accumulated in registers,
    stored once, never atomically added to.

    TWO SCALES, and confusing them is this kernel's one silent-wrong-answer bug. ``scale_exp`` is
    what the score goes through before the exponential and is log2e-prescaled when ``USE_EXP2``;
    ``scale_grad`` is always the plain ``1/sqrt(d)``, because ``dS`` is a derivative with respect
    to the UNSCALED scores and knows nothing about which base the softmax was evaluated in. Using
    the prescaled one there multiplies every gradient by log2(e) = 1.4427 — a gradient that is
    uniformly 44% too large, which trains, and trains wrong.
    """
    pid_k = tl.program_id(0)
    pid_b = tl.program_id(1)
    kv_start = pid_k * BLOCK_K

    offs_k = kv_start + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D)
    k_mask = offs_k < n_ctx

    k = tl.load(
        k_ptr + pid_b * stride_kb + (offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd),
        mask=k_mask[:, None],
        other=0.0,
    )
    v = tl.load(
        v_ptr + pid_b * stride_vb + (offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd),
        mask=k_mask[:, None],
        other=0.0,
    )
    dk = tl.zeros((BLOCK_K, D), tl.float32)
    dv = tl.zeros((BLOCK_K, D), tl.float32)

    # fa2_tuned.q_span, restated. Query blocks below masked_start are entirely BEFORE this block's
    # keys, so under causality every one of their entries is masked and they contribute exactly
    # nothing to dK/dV — fa2.py:243 already skips them; what it does not do is stop masking after
    # the diagonal.
    if IS_CAUSAL:
        masked_start = (kv_start // BLOCK_Q) * BLOCK_Q
        full_start = tl.cdiv(kv_start + BLOCK_K - 1, BLOCK_Q) * BLOCK_Q
        full_start = tl.minimum(full_start, n_ctx)
    else:
        masked_start = 0
        full_start = 0
    if not SKIP_MASKED_BLOCKS:
        full_start = n_ctx  # every visited block takes the masked path

    for q_start in range(masked_start, full_start, BLOCK_Q):
        dk, dv = _bwd_dkdv_block(
            q_ptr,
            do_ptr,
            l_ptr,
            d_ptr,
            k,
            v,
            dk,
            dv,
            pid_b,
            q_start,
            offs_k,
            offs_d,
            stride_qb,
            stride_qn,
            stride_qd,
            stride_dob,
            stride_don,
            stride_dod,
            stride_lb,
            stride_ln,
            stride_db,
            stride_dn,
            n_ctx,
            scale_exp,
            scale_grad,
            log2e,
            APPLY_CAUSAL=IS_CAUSAL,
            USE_EXP2=USE_EXP2,
            D=D,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
        )
    for q_start in range(full_start, n_ctx, BLOCK_Q):
        dk, dv = _bwd_dkdv_block(
            q_ptr,
            do_ptr,
            l_ptr,
            d_ptr,
            k,
            v,
            dk,
            dv,
            pid_b,
            q_start,
            offs_k,
            offs_d,
            stride_qb,
            stride_qn,
            stride_qd,
            stride_dob,
            stride_don,
            stride_dod,
            stride_lb,
            stride_ln,
            stride_db,
            stride_dn,
            n_ctx,
            scale_exp,
            scale_grad,
            log2e,
            APPLY_CAUSAL=False,
            USE_EXP2=USE_EXP2,
            D=D,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
        )

    tl.store(
        dk_ptr + pid_b * stride_dkb + (offs_k[:, None] * stride_dkn + offs_d[None, :] * stride_dkd),
        dk.to(dk_ptr.dtype.element_ty),
        mask=k_mask[:, None],
    )
    tl.store(
        dv_ptr + pid_b * stride_dvb + (offs_k[:, None] * stride_dvn + offs_d[None, :] * stride_dvd),
        dv.to(dv_ptr.dtype.element_ty),
        mask=k_mask[:, None],
    )


@triton.jit
def _fa2_tuned_bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    l_ptr,
    d_ptr,
    dq_ptr,
    stride_qb,
    stride_qn,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vd,
    stride_dob,
    stride_don,
    stride_dod,
    stride_lb,
    stride_ln,
    stride_db,
    stride_dn,
    stride_dqb,
    stride_dqn,
    stride_dqd,
    n_ctx,
    scale_exp,
    scale_grad,
    log2e,
    IS_CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    SKIP_MASKED_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per query block. Owns dQ for that block outright.

    This is the half ``fa2.py`` does not have: there, dQ is produced inside the dK/dV kernel and
    written with ``tl.atomic_add`` (``fa2.py:293``), which costs one read-modify-write per element
    per KV block visited, forces dQ to be zeroed before the launch (``fa2.py:332``), and makes the
    result non-deterministic in the low bits because the atomics land in an arbitrary order.
    ``oss/flash-attention/flash_attn/flash_attn_triton.py:599-604`` is the same design. Upstream's
    current kernels split it instead — ``flash_attn/cute/flash_bwd_mla_dk_sm100.py`` and
    ``flash_bwd_mla_dq_dqv_sm100.py`` — which is what this does.

    It also lets the two halves take DIFFERENT configs: dK/dV holds K and V and streams Q and dO,
    this one holds Q and dO and streams K and V, so their shared-memory budgets are not the same
    function of (BLOCK_Q, BLOCK_K) and on a 99 KB arch they do not admit the same shapes.
    """
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_start = pid_q * BLOCK_Q

    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < n_ctx

    q = tl.load(
        q_ptr + pid_b * stride_qb + (offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd),
        mask=q_mask[:, None],
        other=0.0,
    )
    do = tl.load(
        do_ptr + pid_b * stride_dob + (offs_q[:, None] * stride_don + offs_d[None, :] * stride_dod),
        mask=q_mask[:, None],
        other=0.0,
    )
    lse = tl.load(l_ptr + pid_b * stride_lb + offs_q * stride_ln, mask=q_mask, other=float("inf"))
    dvec = tl.load(d_ptr + pid_b * stride_db + offs_q * stride_dn, mask=q_mask, other=0.0)
    dq = tl.zeros((BLOCK_Q, D), tl.float32)

    # Same span as the forward, for the same reason and by the same arithmetic.
    if IS_CAUSAL:
        masked_end = tl.minimum(n_ctx, q_start + BLOCK_Q)
        full_end = (q_start // BLOCK_K) * BLOCK_K
    else:
        masked_end = n_ctx
        full_end = (n_ctx // BLOCK_K) * BLOCK_K
    if not SKIP_MASKED_BLOCKS:
        full_end = 0

    for k_start in range(0, full_end, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k = tl.load(
            k_ptr + pid_b * stride_kb + (offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        )
        v = tl.load(
            v_ptr + pid_b * stride_vb + (offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        )
        s = tl.dot(q, tl.trans(k)) * scale_exp
        p = _softmax_probs(s, lse, log2e, USE_EXP2=USE_EXP2)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dvec[:, None]) * scale_grad
        dq += tl.dot(ds.to(k.dtype), k)

    for k_start in range(full_end, masked_end, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < n_ctx
        k = tl.load(
            k_ptr + pid_b * stride_kb + (offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd),
            mask=k_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + pid_b * stride_vb + (offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd),
            mask=k_mask[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * scale_exp
        s = tl.where(k_mask[None, :], s, -float("inf"))
        if IS_CAUSAL:
            s = tl.where(offs_q[:, None] >= offs_k[None, :], s, -float("inf"))
        p = _softmax_probs(s, lse, log2e, USE_EXP2=USE_EXP2)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dvec[:, None]) * scale_grad
        dq += tl.dot(ds.to(k.dtype), k)

    tl.store(
        dq_ptr + pid_b * stride_dqb + (offs_q[:, None] * stride_dqn + offs_d[None, :] * stride_dqd),
        dq.to(dq_ptr.dtype.element_ty),
        mask=q_mask[:, None],
    )


# =============================================================================================
# Launch — the arch filter is applied HERE, on the host, where the head dim is known
# =============================================================================================


@cache
def _tuned(kind: Kind, cc: tuple[int, int], d: int):  # noqa: ANN202 - returns a triton Autotuner
    """``triton.autotune`` over the configs that fit ``cc`` at head dim ``d`` — nothing else.

    Building the config list at launch instead of at import is the whole arch-keying mechanism.
    ``@triton.autotune(configs=...)`` is evaluated once, when the module is imported, and cannot
    see the device or the head dim; a single list therefore has to be legal everywhere, and
    ``fa2.py:27-32`` resolves that by being legal on Hopper and offering client Blackwell four
    configs whose shared memory it cannot allocate. Triton's ``prune_configs_by`` hook can filter
    later, but it is handed the kernel's arguments in a shape that has changed between Triton
    releases, and a filter that silently no-ops is worse here than no filter. Choosing the list on
    the host, where ``cc`` and ``d`` are ordinary Python integers, has neither problem — and
    ``require_configs`` raises a message naming the arch and the byte count when nothing fits,
    instead of leaving a launch failure to be read on a rented box.

    One ``Autotuner`` per (kernel, arch, head dim), cached, so the autotune measurements
    themselves are not repeated per call.
    """
    kernel = _KERNELS[kind]
    configs = [
        triton.Config(c.as_kwargs(), num_warps=c.num_warps, num_stages=c.num_stages)
        for c in require_configs(cc, d, kind=kind)
    ]
    return triton.autotune(configs=configs, key=["n_ctx"])(kernel)


_KERNELS = {
    "fwd": _fa2_tuned_fwd_kernel,
    "bwd_dkdv": _fa2_tuned_bwd_dkdv_kernel,
    "bwd_dq": _fa2_tuned_bwd_dq_kernel,
}


def _require_cuda(fn: str) -> tuple[int, int]:
    cc = compute_capability()
    if cc is None:
        raise RuntimeError(f"{fn} needs a CUDA device; there is none here.")
    return cc


def _flatten(t: Tensor, b: int, n: int, d: int) -> Tensor:
    return t.reshape(b, n, d).contiguous()


def forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    is_causal: bool = False,
    use_exp2: bool = True,
    skip_masked_blocks: bool = True,
) -> tuple[Tensor, Tensor]:
    """``(O, L)`` for ``(..., N, d)`` inputs. ``L`` is logsumexp in natural log units."""
    cc = _require_cuda("fa2_tuned_forward")
    *lead, n, d = q.shape
    b = math.prod(lead) if lead else 1
    qf, kf, vf = _flatten(q, b, n, d), _flatten(k, b, n, d), _flatten(v, b, n, d)
    o = torch.empty_like(qf)
    lse = torch.empty((b, n), dtype=torch.float32, device=q.device)

    def grid(meta: dict) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_Q"]), b)

    _tuned("fwd", cc, d)[grid](
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
        softmax_scale(d, use_exp2=use_exp2),
        LN2,
        IS_CAUSAL=is_causal,
        USE_EXP2=use_exp2,
        SKIP_MASKED_BLOCKS=skip_masked_blocks,
        D=d,
    )
    return o.reshape(*lead, n, d), lse.reshape(*lead, n)


def backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    lse: Tensor,
    do: Tensor,
    *,
    is_causal: bool = False,
    use_exp2: bool = True,
    skip_masked_blocks: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """``(dQ, dK, dV)`` from two kernels. ``lse`` in natural log units, as ``forward`` returns it.

    The D-vector ``D_i = sum_d O_id * dO_id`` is computed in torch, exactly as ``fa2.py:329`` does
    and as upstream's ``flash_bwd_preprocess.py`` does as its own kernel: it is an O(N·d) reduction
    over tensors that already exist, it is not on either kernel's critical path, and putting it in
    torch keeps it out of the two shapes being tuned.
    """
    cc = _require_cuda("fa2_tuned_backward")
    *lead, n, d = q.shape
    b = math.prod(lead) if lead else 1
    qf, kf, vf = _flatten(q, b, n, d), _flatten(k, b, n, d), _flatten(v, b, n, d)
    of, dof = _flatten(o, b, n, d), _flatten(do, b, n, d)
    lsef = lse.reshape(b, n).contiguous()
    dvec = (of.float() * dof.float()).sum(dim=-1).contiguous()

    # No torch.zeros for dq: nothing accumulates into it across programs, so it is written once.
    dq, dk, dv = torch.empty_like(qf), torch.empty_like(kf), torch.empty_like(vf)
    scale_exp = softmax_scale(d, use_exp2=use_exp2)
    scale_grad = softmax_scale(d, use_exp2=False)

    def dkdv_grid(meta: dict) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_K"]), b)

    def dq_grid(meta: dict) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_Q"]), b)

    _tuned("bwd_dkdv", cc, d)[dkdv_grid](
        qf,
        kf,
        vf,
        dof,
        lsef,
        dvec,
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
        dof.stride(0),
        dof.stride(1),
        dof.stride(2),
        lsef.stride(0),
        lsef.stride(1),
        dvec.stride(0),
        dvec.stride(1),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        n,
        scale_exp,
        scale_grad,
        LOG2E,
        IS_CAUSAL=is_causal,
        USE_EXP2=use_exp2,
        SKIP_MASKED_BLOCKS=skip_masked_blocks,
        D=d,
    )
    _tuned("bwd_dq", cc, d)[dq_grid](
        qf,
        kf,
        vf,
        dof,
        lsef,
        dvec,
        dq,
        qf.stride(0),
        qf.stride(1),
        qf.stride(2),
        kf.stride(0),
        kf.stride(1),
        kf.stride(2),
        vf.stride(0),
        vf.stride(1),
        vf.stride(2),
        dof.stride(0),
        dof.stride(1),
        dof.stride(2),
        lsef.stride(0),
        lsef.stride(1),
        dvec.stride(0),
        dvec.stride(1),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        n,
        scale_exp,
        scale_grad,
        LOG2E,
        IS_CAUSAL=is_causal,
        USE_EXP2=use_exp2,
        SKIP_MASKED_BLOCKS=skip_masked_blocks,
        D=d,
    )
    return dq.reshape(*lead, n, d), dk.reshape(*lead, n, d), dv.reshape(*lead, n, d)


class TunedTritonFlashAttention(torch.autograd.Function):
    """Autograd wiring for the tuned kernels — the shape ``bench``/``S1`` would call."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        use_exp2: bool = True,
    ) -> Tensor:
        o, lse = forward(q, k, v, is_causal=is_causal, use_exp2=use_exp2)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        ctx.use_exp2 = use_exp2
        return o

    @staticmethod
    def backward(ctx, do: Tensor):  # type: ignore[override]
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = backward(q, k, v, o, lse, do, is_causal=ctx.is_causal, use_exp2=ctx.use_exp2)
        return dq, dk, dv, None, None
