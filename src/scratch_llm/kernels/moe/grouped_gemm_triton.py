"""S1/S-R4 — the Triton grouped GEMM. The mainloop is the rung's hole; the launch is not.

WHAT IS ALREADY HERE (agent-written):
  * the launcher :func:`grouped_gemm_triton` — shape and contract checks, the offsets validation,
    the output allocation, the strides, the grid, and the config plumbing;
  * the grid arithmetic, both variants, with the sync each one does or does not cost
    (:func:`~scratch_llm.kernels.moe.grouped_gemm.max_m_tiles` vs ``exact_m_tiles``);
  * the frozen kernel signature and the full contract of what each argument means, so that the
    call site, the oracle and the test do not move when the body is written;
  * everything on either side of the GEMM: routing, permutation, un-permute, load accounting
    (``routing.py``), the activation and the two-GEMM pipeline (``fused.py``), the fp32 oracle
    (``grouped_gemm.reference_grouped_gemm``), and every test that gates them.

WHAT IS HUY'S (the hole, the body of :func:`_grouped_gemm_kernel`): the per-group tile mapping —
turning a program id into (which expert, which rows, which N tile) given only the group offsets —
and the accumulate/epilogue. Those two are one decision, not two: how the tile is found decides
what the K loop can assume about its row mask, and the mask decides whether the epilogue can store
a whole tile or has to guard every row. That is the arithmetic the rung exists to learn, and the
plan row's "grouped GEMM" is exactly this block.

RUNNING IT: there is no dry dock for a Triton rung — the kernel is compiled by the driver, on the
device, at first call, so nothing here can be compiled on a laptop and there is no ptxas report to
read. The CPU tier of ``tests/kernels/moe/test_s1_s_r4.py`` covers everything that can be decided
without silicon (the offsets, the permutation, the inverse, the load, the tile counts); the GEMM
itself is gated on the box against ``reference_grouped_gemm``.

Spec: ``experiments/S1/S-R4/spec.md``   ·   Map: ``experiments/S1/S-R4/map.md``
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from scratch_llm.kernels.moe.grouped_gemm import (
    GroupedGEMMConfig,
    cdiv,
    default_config,
    max_m_tiles,
    validate_group_offsets,
)
from scratch_llm.kernels.moe.routing import RUNG

#: The hole's message, kept out of the ``raise`` so the literal there stays short enough that
#: ``ruff format`` cannot wrap it onto a second line. If it wraps, tests/conftest.py stops seeing
#: the sentinel, the guard test stops xfailing, and an unwritten kernel reports green.
_HOLE = (
    "map program id -> (expert, row range, N tile) from group_offsets alone, then the K loop and "
    "the epilogue. Read experiments/S1/S-R4/spec.md and experiments/S1/S-R4/map.md first."
)


@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    group_offsets_ptr,
    M,
    N,
    K,
    E,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One tile of ``C[rows of e] = A[rows of e] @ B[e].T``. THE HOLE.

    Contract, frozen so that neither the launcher nor the test moves when this is written:

    ``a_ptr``   ``A`` — (M, K), already permuted into expert-major order by ``routing.permute``.
                Row ``m`` belongs to expert ``e`` iff ``group_offsets[e] <= m < group_offsets[e+1]``.
                Nothing in here needs to know which token row ``m`` came from; the un-permute
                owns that and runs after.
    ``b_ptr``   ``B`` — (E, N, K), vLLM's weight layout. Expert ``e``'s matrix starts at
                ``b_ptr + e * stride_be`` and the product wanted is ``A @ B[e].T``, so ``K`` is
                B's *last* axis and the contraction walks ``stride_bk``.
    ``c_ptr``   ``C`` — (M, N), allocated but NOT zeroed. Every row in ``[0, M)`` lies in exactly
                one group, so a correct mapping writes each row exactly once and a mapping that
                leaves a row unwritten shows up as garbage rather than as a plausible zero.
    ``group_offsets_ptr``  (E+1,) int32, non-decreasing, ``[0] == 0`` and ``[E] == M``. An empty
                expert has ``offsets[e] == offsets[e+1]``: a legal group with no tiles. It is the
                common case at decode (512 slots over 128 experts leaves ~2 experts empty by
                Poisson alone) and the one the whole mapping has to survive.
    ``M N K E`` runtime ints, not constexpr — M changes every step in serving, and specialising on
                it would recompile the kernel per batch size.
    grid        ``(max_m_tiles(M, E, BLOCK_M), cdiv(N, BLOCK_N))``. Axis 0 is an m-tile index over
                ALL groups concatenated, which is why the mapping is not a division: tiles are not
                evenly distributed over experts. The bound is loose on purpose (the launcher's
                docstring says by how much and why), so some axis-0 ids map past the last group and
                must return without writing.

    Three things live in here and nowhere else:

      * **the mapping.** ``pid_m`` → ``(expert, first row, rows in this tile)``. A per-tile search
        over ``group_offsets`` (linear over E, or binary over the same array, or a precomputed
        block table like vLLM's ``expert_ids`` — ``moe_align_block_size.py:50``) is the design
        choice, and its cost is paid by every one of the ~2000 tiles at prefill 4k.
      * **the row mask.** The last tile of a group is partial whenever the group's count is not a
        multiple of ``BLOCK_M`` — which, with 128 ragged groups, is almost always. vLLM avoids the
        mask by padding each group to a tile boundary and storing an out-of-range token id in the
        slack (``moe_align_block_size.py:59-72``); this design keeps the groups ragged and masks.
        Which is cheaper here is one of the rung's two questions.
      * **the accumulate and the epilogue.** fp32 accumulator, ``tl.dot`` per K step, and one
        store of the tile. The accumulator dtype is not negotiable — a bf16 accumulator over
        K=2048 loses the small addends and the error lands in the layer output — but where the
        cast back happens, and whether the store is masked or padded, is not fixed here.

    If a different decomposition is wanted — a 1-D grid with vLLM's ``GROUP_SIZE_M`` L2 raster, a
    persistent loop over tiles, ``SPLIT_K`` for the tall-thin decode GEMM — change this signature
    and the one launcher below. Both are in this file.
    """
    # HUY: the Triton grouped-GEMM mainloop — per-group tile mapping and the accumulate/epilogue — spec: experiments/S1/S-R4/spec.md — fill before S-R4
    #
    # The message must BEGIN on the raise's own line: tests/conftest.py decides whether a hole is
    # open by matching that sentinel in the source, so a formatter that wraps the string would make
    # the hole invisible and flip its guard test green. Detail lives in _HOLE above, never inline.
    raise NotImplementedError("HUY: S1/S-R4 grouped-GEMM mainloop — " + _HOLE)


def grouped_gemm_triton(
    a: Tensor,
    b: Tensor,
    group_offsets: Tensor,
    *,
    config: GroupedGEMMConfig | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """``C = A @ B[e].T`` per group. The launch; the arithmetic is :func:`_grouped_gemm_kernel`.

    Grid: ``max_m_tiles(M, E, BLOCK_M)`` by ``cdiv(N, BLOCK_N)``. Two ways to size axis 0 and the
    choice is a real one:

    * **the bound** (used here) — ``cdiv(M + E*(BLOCK_M-1), BLOCK_M)``, computed from shapes alone.
      No device→host copy, so nothing serialises against the routing kernel that produced the
      offsets. The cost is empty CTAs: at decode (M=512, E=128, BLOCK_M=32) it launches 140 tiles
      where ~126 carry rows, an 11% overshoot of CTAs that exit immediately.
    * **the exact count** — ``sum_e cdiv(count_e, BLOCK_M)`` via
      :func:`~scratch_llm.kernels.moe.grouped_gemm.exact_m_tiles`, which reads the offsets on the
      host and therefore syncs. In a 64-token decode step that sync is on the critical path and is
      worth more than the CTAs it saves; in a 4k prefill it is noise. Switch here, measure both.

    ``out`` is allocated with ``torch.empty`` and not zeroed — see the kernel's contract for why a
    row the mapping forgets should look like garbage and not like a zero.
    """
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError(
            f"{RUNG}: expected a (M, K) and b (E, N, K), got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    m_rows, k_dim = (int(x) for x in a.shape)
    n_experts, n_out, k_b = (int(x) for x in b.shape)
    if k_dim != k_b:
        raise ValueError(f"{RUNG}: a's K = {k_dim} != b's K = {k_b}")
    if a.device != b.device or a.device != group_offsets.device:
        raise ValueError(f"{RUNG}: a, b and group_offsets must share a device")
    problems = validate_group_offsets(group_offsets, n_rows=m_rows, n_experts=n_experts)
    if problems:
        raise ValueError(f"{RUNG}: bad group_offsets — " + "; ".join(problems))
    if a.stride(1) != 1 or b.stride(2) != 1:
        raise ValueError(f"{RUNG}: a and b must have a contiguous K axis (last-dim stride 1)")

    cfg = config or default_config(n_tokens=m_rows, n_experts=n_experts, n_out=n_out, k_dim=k_dim)
    bad_cfg = cfg.validate()
    if bad_cfg:
        raise ValueError(f"{RUNG}: bad config — " + "; ".join(bad_cfg))

    c = out if out is not None else torch.empty((m_rows, n_out), dtype=a.dtype, device=a.device)
    if tuple(c.shape) != (m_rows, n_out):
        raise ValueError(f"{RUNG}: out has shape {tuple(c.shape)}, expected {(m_rows, n_out)}")

    grid = (max_m_tiles(m_rows, n_experts, cfg.block_m), cdiv(n_out, cfg.block_n))
    _grouped_gemm_kernel[grid](
        a,
        b,
        c,
        group_offsets,
        m_rows,
        n_out,
        k_dim,
        n_experts,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return c


__all__ = ["grouped_gemm_triton"]
