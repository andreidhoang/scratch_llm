"""K2/A-R2 — FA3-shaped Hopper attention forward, v2 (sm_90a).

The rung the plan states as "Hopper CUDA FA fwd (FA3 §3): TMA K/V via mbarrier pipeline, 1 producer
warp + 2 consumer warpgroups; wgmma S=QK^T in RF, online softmax in RF, P->bf16 register A-operand
for O+=PV; ping-pong warpgroups" (``plan/SIXTY_DAYS_SIX_LADDERS.md`` §05 K2). Exit >= 60% of FA3
forward, kill below 40%, five-day cap.

**How this differs from** :mod:`scratch_llm.kernels.attention.prefill.fa3`, which loads the
pre-ladder skeleton ``csrc/attention/fa3_hopper.cu`` and stays exactly as it is. That kernel is
fp16, head dim 64, ONE consumer warpgroup, ``m64n64k16``, and it round-trips P through shared
memory to hand it to the second GEMM as an SS-form operand. Its own header marks five things DEFER
— the accumulator fragment map, the tensor-map coordinates, the mbarrier phase parity, the causal
mask, the epilogue mapping — and every one of them is load bearing, so v2 is a rebuild:

* bf16, head dim 128, 128x128 tiles: the shapes ``bench/kernels/attention/k2_ladder.py`` measures.
* One producer **warp** (not warpgroup) plus two consumer warpgroups — the FA3 §3 geometry.
* P never reaches shared memory. It stays in registers and enters ``O += P·V`` as a wgmma
  **register A-operand** (the RS form), which is the structural difference between an attention
  mainloop and a GEMM mainloop and the reason P must be cast to bf16 in registers.
* Rank-4 tensor maps built from the tensor's real strides, so ``[B,H,S,D]`` (SDPA's layout) and
  ``[B,S,H,D]`` (``flash_attn_func``'s native layout) both run with no transpose. A benchmark that
  transposes one side of the comparison and not the other is not a matched comparison.
* The accumulator fragment map is derived from CUTLASS's ``CLayout_64xN`` and written down.

Three things live here rather than in the ``.cu`` because they are decidable on a laptop and each
costs a rented hour if it is wrong:

* :func:`tensor_map_params` — the exact ``cuTensorMapEncodeTiled`` arguments the kernel builds, as
  data, so :func:`~scratch_llm.kernels.common.hopper_contracts.check_tma_tensor_map` can rule on
  every field by name. The driver answers every illegal field with ``CUDA_ERROR_INVALID_VALUE`` and
  names none of them.
* :func:`unsupported_reason` — the shapes this rung cannot address, and why each is a property of
  the hardware or of the mask rather than a corner that was cut.
* :func:`acc_frag_row` / :func:`acc_frag_col` — the wgmma accumulator fragment map, mirrored from
  the ``.cu`` so the CPU suite can prove it is a bijection over the 64 x N tile.

Kernel source and the ``# HUY:`` hole: ``csrc/attention/fa3_hopper_v2.cu``.
Spec, floor and kill rule: ``experiments/K2/A-R2/spec.md``. Map: ``experiments/K2/A-R2/map.md``.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from scratch_llm.kernels.attention.reference import flash_attention_forward
from scratch_llm.kernels.common.arch import compute_capability, require_arch
from scratch_llm.kernels.common.hopper_contracts import SwizzleMode

#: The K2 rung identity, in one place, so the wrapper, the tests, the bench and the dry-dock
#: registry cannot drift apart.
RUNG = "K2/A-R2"
SOURCE = "fa3_hopper_v2.cu"
SYMBOL = "fa3_hopper_v2_fwd"
ARCH = (9, 0)

#: Tile shape and warp geometry, mirrored from the defines in the ``.cu``. ``BLOCK_M`` is the one
#: that sets everything else: two consumer warpgroups exist because 128 query rows is two wgmma
#: ``M`` of 64. FlashInfer states the same dependency as one expression,
#: ``NUM_WARPS = ((CTA_Q/64)+1)*4`` (``kernel_traits.cuh:62``). The CPU tests assert the two sides
#: of this file and the ``.cu`` agree.
BLOCK_M, BLOCK_N, HEAD_DIM = 128, 128, 128
STAGES = 2
WGMMA_M, WGMMA_K = 64, 16
WG_THREADS = 128
NUM_CONSUMER_WG = BLOCK_M // WGMMA_M
NUM_CONSUMER_THREADS = NUM_CONSUMER_WG * WG_THREADS
NUM_THREADS = (NUM_CONSUMER_WG + 1) * WG_THREADS
ROWS_PER_THREAD = 2

#: ``setmaxnreg`` immediates. FA3 with two MMA warpgroups and TMA KV uses this exact pair
#: (``oss/flash-attention/hopper/flash_fwd_kernel_sm90.h:82-83``). The register file is the wall:
#: ``NUM_CONSUMER_THREADS * CONSUMER_REGS + WG_THREADS * PRODUCER_REGS`` must fit in an SM's 65536.
PRODUCER_REGS = 24
CONSUMER_REGS = 240

#: Element type of every operand, in bytes. bf16 because the K2 floor is FA3 **bf16**; the older
#: skeleton is fp16, which is a different floor and therefore a different number.
ELEM_BYTES = 2

#: The 128B swizzle atom in bf16 elements. Every TMA box's innermost extent must equal exactly
#: this. ``HEAD_DIM`` is two atoms wide, which is why every tile takes two boxes — the same split
#: K1/H-R2 makes along ``N``, here along the head dim.
SWIZZLE_ATOM_ELEMS = 128 // ELEM_BYTES
D_CHUNKS = HEAD_DIM // SWIZZLE_ATOM_ELEMS

#: TMA's own alignment rule: every ``globalStrides`` entry must be a multiple of 16 B — 8 bf16
#: elements — and it applies to each non-innermost dimension of the tensor.
TMA_STRIDE_ELEMS = 16 // ELEM_BYTES

#: The three shared-memory matrix descriptors, in bytes. Q and K are ``Major::K`` (the contracted
#: head dim is contiguous in both) and share H-R2's A-operand pair. V is ``Major::MN``: for
#: ``O += P·V`` the contracted dim is the key index and the contiguous one is the head dim, so its
#: leading offset is the stride between the two 64-wide head-dim slabs. FA3 reaches the same
#: verdict through the type system (``mainloop_fwd_sm90_tma_gmma_ws.hpp:65``: ``MmaMajorV`` is
#: ``GMMA::Major::MN`` for 16-bit V). Swapping LBO and SBO does not fault — it returns a plausible
#: wrong O — which is why the walks are asserted in the CPU suite.
Q_DESC_LBO, Q_DESC_SBO = 16, 1024
K_DESC_LBO, K_DESC_SBO = 16, 1024
V_DESC_LBO = SWIZZLE_ATOM_ELEMS * BLOCK_N * ELEM_BYTES
V_DESC_SBO = 1024

#: Shared memory. The Q tile is resident for the whole k-loop and is therefore ``extra_bytes``, not
#: part of a stage; a stage is one K tile plus one V tile. The mbarriers are static and sit outside
#: the dynamic allocation so ``SMEM_DYNAMIC_BYTES`` stays exactly the number the launcher opts into.
Q_TILE_BYTES = BLOCK_M * HEAD_DIM * ELEM_BYTES
KV_TILE_BYTES = BLOCK_N * HEAD_DIM * ELEM_BYTES
BYTES_PER_STAGE = 2 * KV_TILE_BYTES
SMEM_DYNAMIC_BYTES = Q_TILE_BYTES + BYTES_PER_STAGE * STAGES
#: Four barrier arrays (K and V, each full and empty) plus one for Q. K and V get separate
#: pipelines so a consumer can release K as soon as the QK wgmma retires while PV still reads V —
#: FA3 gives them separate pipeline params for exactly this
#: (``flash_fwd_kernel_sm90.h:226-263``).
SMEM_BARRIER_BYTES = 8 * (4 * STAGES + 1)

#: The mbarrier ``expect_tx`` byte count for one K (or V) tile: what both of its TMA boxes deliver
#: together. Too high hangs; too low lets the MMA read a half-written tile.
EXPECT_TX_BYTES_KV = KV_TILE_BYTES
EXPECT_TX_BYTES_Q = Q_TILE_BYTES

#: The two tensor layouts this rung accepts, named by their axis order. ``bhsd`` is what
#: ``torch.nn.functional.scaled_dot_product_attention`` takes; ``bshd`` is what
#: ``flash_attn_interface.flash_attn_func`` takes. Supporting both is not convenience — it is what
#: lets the rung and its floor be timed on the same bytes with no transpose inside the measured
#: region.
LAYOUTS = ("bhsd", "bshd")

_CSRC = Path(__file__).resolve().parents[5] / "csrc" / "attention"


def source_path() -> Path:
    """Absolute path to this rung's ``.cu``."""
    return _CSRC / SOURCE


def hole_is_open() -> bool:
    """True while the kernel body still carries the ``#error "HUY:`` sentinel.

    A missing source counts as open: the rung has not been written, so its kernel certainly cannot
    run, and reporting "filled" for a path typo would be the worst possible answer.
    """
    p = source_path()
    if not p.is_file():
        return True
    return '#error "HUY:' in p.read_text(encoding="utf-8", errors="replace")


# =================================================================================================
# The accumulator fragment map — CUTLASS's CLayout_64xN, in Python
# =================================================================================================


def acc_frag_row(reg: int, lane: int, warp: int) -> int:
    """Row of ``reg`` in the 64 x N wgmma accumulator held by ``(warp, lane)``.

    From ``CLayout_64xN`` (``oss/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:432-434``)::

        Layout<Shape <Shape < _4,_8, _4>, Shape <_2,_2,Int<N/8>>>,
               Stride<Stride<_128,_1,_16>, Stride<_64,_8,   _512>>>

    with the codomain indexed ``row + 64*col``. Decompose the thread id as
    ``(lane%4) + 4*(lane//4) + 32*warp`` and the value id as ``v0 + 2*v1 + 4*v2``; the strides 1 and
    16 land on the row, 128, 64 and 512 land on the column.

    The consequence the whole mainloop rests on: each thread owns exactly **two** rows, selected by
    ``v1 = (reg >> 1) & 1`` — *not* by ``reg & 1``, which selects the column. The pre-ladder
    skeleton uses ``reg & 1`` (``csrc/attention/fa3_hopper.cu:289``) and the resulting kernel runs
    at full speed while taking the softmax over the wrong axis.
    """
    return 16 * warp + (lane >> 2) + 8 * ((reg >> 1) & 1)


def acc_frag_col(reg: int, lane: int) -> int:
    """Column of ``reg`` in the 64 x N wgmma accumulator held by lane ``lane``."""
    return 2 * (lane & 3) + (reg & 1) + 8 * (reg >> 2)


def acc_frag_row_slot(reg: int) -> int:
    """Which of the thread's ``ROWS_PER_THREAD`` rows ``reg`` belongs to."""
    return (reg >> 1) & 1


# =================================================================================================
# Tensor maps and shape support
# =================================================================================================


def strides_elems(
    layout: str, *, batch: int, heads: int, seqlen: int, head_dim: int = HEAD_DIM
) -> tuple[int, int, int]:
    """``(stride_batch, stride_head, stride_seq)`` in elements for one of :data:`LAYOUTS`.

    The head dim is contiguous in both, which is what makes one tensor-map shape describe both.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if layout == "bhsd":
        return heads * seqlen * head_dim, seqlen * head_dim, head_dim
    return seqlen * heads * head_dim, head_dim, heads * head_dim


def dims_for(shape: tuple[int, ...], layout: str = "bhsd") -> tuple[int, int, int, int]:
    """``(batch, heads, seqlen, head_dim)`` for a 4-D attention tensor in one of :data:`LAYOUTS`.

    The layout is a PARAMETER and is never inferred from the strides, because it cannot be. A
    contiguous ``[B,H,S,D]`` and a contiguous ``[B,S,H,D]`` have the same stride *pattern* — dim 3
    is 1, dim 2 is ``D``, dims 1 and 0 decrease — so no predicate over strides separates them; one
    that appears to is really detecting a transposed *view*. Getting it wrong swaps heads for
    sequence, which does not fault: it makes the shape checks reject a valid tensor, or (when
    ``S == H``) accept it and compute nonsense.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if len(shape) != 4:
        raise ValueError(f"expected a 4-D attention tensor, got shape {tuple(shape)}")
    b, x, y, d = shape
    return (b, y, x, d) if layout == "bshd" else (b, x, y, d)


def tensor_map_params(
    *, batch: int, heads: int, kv_heads: int, seqlen: int, layout: str = "bhsd"
) -> dict[str, dict[str, object]]:
    """The ``cuTensorMapEncodeTiled`` arguments ``fa3_v2_encode_tensor_map`` builds, as data.

    Keyword-for-keyword what :func:`~scratch_llm.kernels.common.hopper_contracts.check_tma_tensor_map`
    takes. Extents come innermost-first — the driver's order, not the tensor's — and
    ``global_strides_bytes`` holds dimensions 1.. only, because the descriptor assumes the innermost
    stride is one element and is handed ``stride + 1``
    (``oss/fast.cu/h100/matmul/matmul_2.cuh:38,:44``).

    Rank 4, not a rank-2 map over a flattened ``(B*H*S, D)``: with the sequence collapsed into the
    row index, the last rows of one head are in bounds for the next head rather than out of bounds,
    so TMA's own bounds checking would silently read a neighbour's keys instead of zero-filling.
    """
    q_sb, q_sh, q_ss = strides_elems(layout, batch=batch, heads=heads, seqlen=seqlen)
    k_sb, k_sh, k_ss = strides_elems(layout, batch=batch, heads=kv_heads, seqlen=seqlen)
    kv = {
        "rank": 4,
        "elem_bytes": ELEM_BYTES,
        "global_dims": [HEAD_DIM, seqlen, kv_heads, batch],
        "global_strides_bytes": [k_ss * ELEM_BYTES, k_sh * ELEM_BYTES, k_sb * ELEM_BYTES],
        "box_dims": [SWIZZLE_ATOM_ELEMS, BLOCK_N, 1, 1],
        "swizzle": SwizzleMode.B128,
    }
    return {
        "q": {
            "rank": 4,
            "elem_bytes": ELEM_BYTES,
            "global_dims": [HEAD_DIM, seqlen, heads, batch],
            "global_strides_bytes": [q_ss * ELEM_BYTES, q_sh * ELEM_BYTES, q_sb * ELEM_BYTES],
            "box_dims": [SWIZZLE_ATOM_ELEMS, BLOCK_M, 1, 1],
            "swizzle": SwizzleMode.B128,
        },
        "k": kv,
        "v": dict(kv),
    }


def kv_blocks_for_tile(q_tile: int, seqlen: int, *, causal: bool = True) -> int:
    """How many ``BLOCK_N`` key blocks query tile ``q_tile`` reads. Mirrors the ``.cu``'s
    ``n_kv_blocks``.

    The producer and both consumers must agree on this number to the block. They do not derive it
    separately — the ``.cu`` computes it once, before the warp-role split, precisely because two
    derivations that disagree by one is not a wrong answer but a *hang*: the producer waits on an
    empty barrier nobody will signal, or a consumer waits on a full barrier nobody will fill. Causal
    masking makes the count depend on the query tile, which is exactly where two derivations drift.

    Mirrored here so the CPU suite can assert the property that matters — that every key the mask
    admits is inside a block that was actually loaded (:func:`causal_coverage_gap`).
    """
    total = seqlen // BLOCK_N
    if not causal:
        return total
    return min(total, (q_tile * BLOCK_M + BLOCK_M - 1) // BLOCK_N + 1)


def causal_coverage_gap(seqlen: int) -> tuple[int, int] | None:
    """The first ``(query, key)`` the causal mask admits but the k-loop never loads, or ``None``.

    An off-by-one in :func:`kv_blocks_for_tile` that loads too *few* blocks does not fault and does
    not hang: it silently drops the tail of each row's attention and returns a plausible O. Only an
    oracle catches it on a GPU — or this, on a laptop, exhaustively.
    """
    for q_tile in range(seqlen // BLOCK_M):
        loaded = kv_blocks_for_tile(q_tile, seqlen) * BLOCK_N
        last_query = q_tile * BLOCK_M + BLOCK_M - 1
        if last_query >= loaded:
            return (last_query, loaded)
    return None


def unsupported_reason(
    *, heads: int, kv_heads: int, seqlen: int, head_dim: int, seqlen_kv: int | None = None
) -> str | None:
    """Why this shape cannot run on this rung, or ``None`` if it can.

    None of these is a corner cut:

    * ``head_dim != HEAD_DIM`` — the tile shape is compiled in, and every K2 spec shape is 128
      (``bench/kernels/attention/k2_ladder.py:71-80``).
    * ``seqlen % BLOCK_M`` — **the one that differs from a GEMM.** TMA zero-fills out-of-bounds
      reads, which for a GEMM is a free tail because a zero contributes nothing to a sum. Here a
      zero-filled K row produces a *score* of zero, and a score of zero is not minus infinity: it
      enters the softmax as a real key with weight ``exp(0)``. A KV tail would be silently wrong,
      so it is refused rather than half-handled.
    * ``seqlen_kv != seqlen`` — this rung is self-attention prefill. With different lengths the
      causal diagonal's alignment (top-left or bottom-right) becomes a choice, and FA3's is
      bottom-right; making that choice belongs to the rung that needs it, not to this one.
    * ``heads % kv_heads`` — GQA maps query head ``h`` to KV head ``h // (heads//kv_heads)``, which
      requires the ratio to be exact.
    """
    if seqlen_kv is not None and seqlen_kv != seqlen:
        return (
            f"self-attention prefill only at this rung: S_q must equal S_kv; got {seqlen} vs "
            f"{seqlen_kv}"
        )
    if head_dim != HEAD_DIM:
        return f"head dim must be {HEAD_DIM} at this rung (compiled-in tile); got {head_dim}"
    if seqlen % BLOCK_M or seqlen % BLOCK_N:
        return (
            f"seqlen must be a multiple of {BLOCK_M}; TMA zero-fills out-of-bounds rows and a "
            f"zero SCORE is not -inf, so a KV tail would enter the softmax as a real key. "
            f"got S={seqlen}"
        )
    if kv_heads <= 0 or heads % kv_heads:
        return f"query heads must be a multiple of KV heads (GQA); got {heads} and {kv_heads}"
    return None


# =================================================================================================
# The floor
# =================================================================================================


def floor_unavailable_reason() -> str | None:
    """Why FA3 cannot be measured here, or ``None`` if it can.

    The floor for this rung is FA3's own forward kernel — ``flash_attn_interface``, the ``hopper/``
    build of Dao-AILab/flash-attention — at matched dtype, shape and mask. It exists on neither this
    Mac nor any non-Hopper box.

    This function returns a *reason*, and the wrapper and the bench say it out loud, because the one
    thing that must never happen is a silent fall back to FA2/SDPA. "A-R2 reached 61%" against FA2
    is not a smaller version of the claim the plan commits to (>= 60% of FA3, kill below 40%) — it
    is a different claim, roughly one and a half times easier, and nothing downstream of the number
    would show which floor produced it.
    """
    try:
        import flash_attn_interface  # noqa: F401,PLC0415
    except ImportError:
        return (
            "flash_attn_interface is not installed. FA3 is the K2 A-R2 floor and it has no CPU "
            "build; install it on the Hopper box (Dao-AILab/flash-attention, hopper/) — do NOT "
            "substitute flash_attn (FA2) or SDPA, which would measure against a different floor"
        )
    cc = compute_capability()
    if cc is None:
        return "no CUDA device — FA3 is a kernel, and a floor is a measurement on silicon"
    if cc != ARCH:
        return (
            f"FA3's forward is an sm_90a kernel and this device is sm_{cc[0] * 10 + cc[1]}; a "
            f"number against any other floor is not this rung's number"
        )
    return None


# =================================================================================================
# The oracle
# =================================================================================================


def reference_attention(
    q: Tensor, k: Tensor, v: Tensor, *, causal: bool = True, layout: str = "bhsd"
) -> tuple[Tensor, Tensor]:
    """The oracle: the same bf16 inputs, attended in fp32. Returns ``(O, LSE)``.

    It up-casts the identical operands the kernel receives rather than generating fp32 ones, so the
    only difference between this and the kernel is accumulation order and the tensor core's internal
    rounding. An fp32-*input* reference would fold the operands' own bf16 quantization error into
    the tolerance and let a real kernel bug hide underneath it.

    ``LSE`` is returned because upstream gates it too — FlashInfer checks O **and** LSE against its
    sm80 kernel (``tests/attention/test_hopper.py:55-56``) and CUTLASS's FMHA example checks both
    against a max-diff and a mean-diff threshold (``examples/88_hopper_fmha/88_hopper_fmha.cu:327,
    :334``). An O-only gate is weaker than the thing this rung is measured against, and the softmax
    state is already in registers when the epilogue runs, so returning it is nearly free.

    The tiled fp32 recurrence itself is :func:`scratch_llm.kernels.attention.reference
    .flash_attention_forward`, which the repo already tests on its own
    (``tests/kernels/test_flash_attention.py``); reimplementing it here would be a second thing to
    keep correct.
    """
    if layout == "bshd":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    heads, kv_heads = q.shape[1], k.shape[1]
    if heads != kv_heads:
        rep = heads // kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    o, lse = flash_attention_forward(q.float(), k.float(), v.float(), is_causal=causal)
    if layout == "bshd":
        o = o.transpose(1, 2)
    return o, lse.reshape(q.shape[0], heads, q.shape[2])


# =================================================================================================
# The kernel
# =================================================================================================


class HoleOpenError(NotImplementedError):
    """The rung's kernel body is still an unfilled ``# HUY:`` hole.

    A subclass of ``NotImplementedError`` so a caller that means to tolerate an unwritten rung can,
    while ``pytest.raises(NotImplementedError)`` in the hole's guard test still reads naturally.
    """


@lru_cache(maxsize=1)
def _module():  # noqa: ANN202 - opaque pybind extension module
    if hole_is_open():
        raise HoleOpenError(
            f"{SOURCE} still has an open HUY hole — the consumer mainloop is unwritten, so there "
            f"is nothing to compile. Fill it (spec: experiments/{RUNG}/spec.md), or compile the "
            f"scaffolding only with nvcc -DHUY_STUB_KERNEL_BODY=1. `make holes` lists every hole."
        )
    # 1. AOT — the extension the rented box built once, at bootstrap. A JIT compile inside a timing
    #    window is a measurement of nvcc.
    try:
        from scratch_llm import (
            _scratch_llm_kernels as _ext,  # type: ignore[attr-defined]  # noqa: PLC0415
        )

        if hasattr(_ext, SYMBOL):
            return _ext
    except ImportError:
        pass

    # 2. JIT — one source, one arch. The trailing ``a`` selects the accelerated ISA and is
    #    mandatory: base sm_90 silently omits wgmma.
    from torch.utils.cpp_extension import load  # noqa: PLC0415

    return load(
        name="k2_a_r2_fa3_v2",
        sources=[str(source_path())],
        extra_cuda_cflags=["-O3", "-lineinfo", "-arch=sm_90a"],
        verbose=False,
    )


def fa3_hopper_v2_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = True,
    return_lse: bool = False,
    layout: str = "bhsd",
) -> Tensor | tuple[Tensor, Tensor]:
    """FA3-shaped attention forward on Hopper. bf16 in, bf16 out, fp32 accumulate.

    Args:
        q, k, v: 4-D bf16 in ``layout``; the head dim must be contiguous and equal to
            :data:`HEAD_DIM`. ``k`` and ``v`` may have fewer heads than ``q`` (GQA).
        causal: apply the causal mask. The K2 spec shapes are causal except ``noncausal``, which
            exists to isolate the mask factor in the FLOP count.
        return_lse: also return the log-sum-exp per query row, ``[B, H, S]`` fp32.
        layout: ``"bhsd"`` (SDPA's, the default) or ``"bshd"`` (``flash_attn_func``'s). Both run
            with no transpose because the tensor map is built from the real strides — see
            :func:`dims_for` for why this is a parameter and not something the strides can be
            asked.

    Raises ``RuntimeError`` on a non-Hopper device and :class:`HoleOpenError` while the kernel body
    is still an unfilled hole. Never falls back to another kernel: a silent fallback would produce a
    number against a floor nobody chose.
    """
    require_arch(ARCH, fn_name="fa3_hopper_v2_fwd")
    if not (q.dtype is torch.bfloat16 and k.dtype is torch.bfloat16 and v.dtype is torch.bfloat16):
        raise TypeError(
            f"fa3_hopper_v2_fwd expects bfloat16 operands (wgmma.f32.bf16.bf16), got {q.dtype}, "
            f"{k.dtype} and {v.dtype}; the K2 floor is FA3 bf16, so another dtype measures against "
            f"no floor"
        )
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"fa3_hopper_v2_fwd expects 4-D operands, got {q.ndim}-D, {k.ndim}-D and {v.ndim}-D"
        )
    if k.shape != v.shape:
        raise ValueError(
            f"fa3_hopper_v2_fwd: k and v must have the same shape: {k.shape} vs {v.shape}"
        )
    _, heads, seqlen, head_dim = dims_for(tuple(q.shape), layout)
    _, kv_heads, seqlen_kv, _ = dims_for(tuple(k.shape), layout)
    why = unsupported_reason(
        heads=heads, kv_heads=kv_heads, seqlen=seqlen, head_dim=head_dim, seqlen_kv=seqlen_kv
    )
    if why is not None:
        raise ValueError(f"fa3_hopper_v2_fwd: {why}")
    o, lse = getattr(_module(), SYMBOL)(q, k, v, causal, layout == "bshd")
    return (o, lse) if return_lse else o


def softmax_scale_log2(head_dim: int = HEAD_DIM) -> float:
    """``log2(e)/sqrt(d)`` — the scale the kernel folds into every exponent, once, on the host.

    Every exponential in the mainloop is then a bare ``exp2`` with no multiply in front of it, and
    the running max lives in the same basis; the epilogue converts back out of it exactly once when
    it writes LSE. FA3 carries the scale identically (``softmax_scale_log2``,
    ``oss/flash-attention/hopper/flash_fwd_kernel_sm90.h:416``).
    """
    return math.log2(math.e) / math.sqrt(head_dim)
