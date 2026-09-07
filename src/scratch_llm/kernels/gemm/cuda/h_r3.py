"""K1/H-R3 — warp-specialised multistage TMA + wgmma pipeline (Hopper, sm_90a).

The third rung of the K1 GEMM ladder. One CTA is 384 threads: warpgroup 0 issues nothing but TMA
copies into a four-deep ring of shared-memory stages, warpgroups 1 and 2 issue nothing but
``wgmma.mma_async.m64n256k16`` out of it, and the two sides meet only at a pair of mbarriers per
stage. ``setmaxnreg`` is what makes the split pay: the producer hands back the registers it does not
need so each consumer can hold a 128-register fp32 accumulator.

Two facts drive every constant here, and both are checkable on a laptop:

* **The tile does not fit in 48 KB.** 49152 B per stage x 4 stages is 196608 B, so the launch must
  call ``cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)``. H-R1 (32 KB,
  one stage) did not. ptxas emits the kernel either way; the omission is a *launch* failure, hours
  after the tile shape that caused it was chosen — which is why
  :func:`~scratch_llm.kernels.common.hopper_contracts.check_smem_budget` reports the opt-in as a
  finding and the CPU suite asserts it.
* **B is MN-major.** The K1 harness hands ``b`` as ``[K, N]`` row-major, i.e. N-contiguous, and TMA
  cannot transpose: its innermost box must be exactly one 128 B swizzle atom, which for this B is 64
  *N*-elements. So the B tile lands MN-major and the wgmma issues it with ``trans_b = 1``
  (``cute::GMMA::Major::MN == 1``). That is not a preference — for a row-major x row-major GEMM on
  Hopper exactly one operand is MN-major whichever way the roles are assigned, because only one of A
  and B is K-contiguous. It changes the descriptor's leading byte offset and nothing else;
  :func:`mn_major_descriptor_offsets` derives it and the CPU suite asserts the ``.cu`` agrees.

The same TMA rule constrains the caller: N and K must each be a multiple of
:data:`TMA_INNER_ELEMS`. M is free — TMA zero-fills out-of-bounds rows, and zeros contribute nothing
to the accumulation. ``bench/kernels/gemm/k1_ladder.py``'s ``npot`` shape (N=1023) is therefore not
measurable at this rung; ``sq4096``, ``rect8192``, ``skinny16`` and ``untuned`` all are.

Kernel source and the ``// HUY:`` hole: ``csrc/gemm/h_r3_ws_bf16_sm90.cu``.
Spec, floor and kill rule: ``experiments/K1/H-R3/spec.md``. Map: ``experiments/K1/H-R3/map.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scratch_llm.kernels.common.arch import require_arch
from scratch_llm.kernels.common.hopper_contracts import ARCH as ARCH_LIMITS
from scratch_llm.kernels.common.hopper_contracts import SwizzleMode
from scratch_llm.kernels.gemm.cuda._k1_loader import load_rung

#: The K1 rung identity, in one place, so the wrapper, the tests, the bench and the dry-dock
#: registry cannot drift apart.
RUNG = "K1/H-R3"
SOURCE = "h_r3_ws_bf16_sm90.cu"
SYMBOL = "h_r3_ws_bf16"
ARCH = (9, 0)

#: Tile and pipeline shape, mirrored from the ``#define``s in the ``.cu``. The CPU contract tests
#: re-read those defines and assert they match, because nothing else would notice if they diverged:
#: the ``.cu`` generates code, these feed the shared-memory, register and descriptor arithmetic.
TILE_M, TILE_N, TILE_K = 128, 256, 64
STAGES = 4
NUM_CONSUMERS = 2
THREADS_PER_CTA = 128 * (NUM_CONSUMERS + 1)  # 1 producer + 2 consumer warpgroups = 384
PRODUCER_REGS = 24  # setmaxnreg.dec immediate
CONSUMER_REGS = 240  # setmaxnreg.inc immediate

#: TMA's innermost box must be exactly one 128 B swizzle atom = 64 bf16. Every shape constraint this
#: rung imposes on its caller descends from this one number.
TMA_INNER_ELEMS = 64
ELEM_BYTES = 2
#: Shared memory is 32 banks x 4 B, but the descriptor's unit is the 16-byte uint128 the swizzle
#: atom is built from; CUTLASS states the canonical GMMA layouts in those units and so do we.
U128_BYTES = 16

BYTES_PER_STAGE = (TILE_M * TILE_K + TILE_K * TILE_N) * ELEM_BYTES  # 49152
#: The two ``uint64_t`` mbarriers per stage. Static shared memory, deliberately outside the dynamic
#: allocation, so that no stage boundary is pushed off the 1024 B swizzle repeat.
BARRIER_BYTES = 2 * STAGES * 8
#: The dynamic shared memory the launch asks for. The extra :data:`SMEM_ALIGN` is slack to round the
#: driver's 128 B-aligned base up to a swizzle repeat, which is the precondition for every
#: descriptor carrying ``base_offset = 0`` (as both fast.cu and CUTLASS hard-code).
SMEM_ALIGN = 1024
SMEM_BYTES = STAGES * BYTES_PER_STAGE + SMEM_ALIGN

_SM90 = ARCH_LIMITS["sm_90a"]
#: What ptxas may hand every thread under ``__launch_bounds__(384, 1)``: the SM's registers split
#: evenly, floored to the 8-register granularity the hardware allocates in.
REGS_PER_THREAD_UNIFORM = (_SM90.regs_per_sm // THREADS_PER_CTA) // 8 * 8  # 168
#: What the CTA actually holds after ``setmaxnreg`` has moved registers from the producer to the
#: consumers. It comes out equal to ``REGS_PER_THREAD_UNIFORM * THREADS_PER_CTA``: warp
#: specialisation does not create registers, it only decides who gets them.
REGS_AFTER_SPLIT = PRODUCER_REGS * 128 + CONSUMER_REGS * 128 * NUM_CONSUMERS


def k_major_descriptor_offsets(*, tile_k: int = TILE_K) -> tuple[int, int]:
    """``(leading, stride)`` byte offsets for the **A** tile, K-major, 128 B swizzled.

    TMA lays A down as ``[m][k]`` with ``k`` contiguous in :data:`TMA_INNER_ELEMS`-element chunks,
    so one m-row is exactly one swizzle atom (128 B). In uint128 units CUTLASS's canonical K-major
    layout is ``((8,m),(2,k)):((8,SBO),(1,LBO))`` (``cute/atom/mma_traits_sm90_gmma.hpp:263-268``):
    the leading offset walks the two 8x8 core matrices along K — one uint128 — and the stride offset
    walks 8 core-matrix rows. Independent of ``tile_k`` while the tile is one atom wide, which is
    why the same 16 / 1024 pair appears in ``oss/fast.cu/h100/matmul/matmul_7.cuh:13-14``.
    """
    del tile_k  # independent of it while TILE_K == TMA_INNER_ELEMS; the parameter is kept so
    # the two builders read as a pair and a future BK != 64 has to face both of them.
    return U128_BYTES, 8 * TMA_INNER_ELEMS * ELEM_BYTES


def mn_major_descriptor_offsets(*, tile_k: int = TILE_K) -> tuple[int, int]:
    """``(leading, stride)`` byte offsets for the **B** tile, MN-major, 128 B swizzled.

    TMA lays B down as ``[n_outer][k][n_inner]`` — ``n_inner`` contiguous, 64 elements = 128 B — so
    in uint128 units the element at ``(n, k)`` sits at ``n_outer * 8*tile_k + k * 8 + n_inner/8``.
    CUTLASS's canonical MN-major layout is ``((8,n),(8,k)):((1,LBO),(8,SBO))``
    (``mma_traits_sm90_gmma.hpp:229-234``), so matching the two term by term gives the leading
    offset as the stride between ``n_outer`` chunks and the stride offset as the stride across 8
    k-rows. Unlike the K-major pair, the leading offset *does* depend on ``tile_k``.
    """
    per_u128 = U128_BYTES // ELEM_BYTES  # 8 bf16
    leading = (TMA_INNER_ELEMS * tile_k // per_u128) * U128_BYTES
    stride = (8 * TMA_INNER_ELEMS // per_u128) * U128_BYTES
    return leading, stride


def tma_tensor_map(*, rows: int, cols: int, box_rows: int, box_cols: int) -> dict[str, object]:
    """The ``cuTensorMapEncodeTiled`` arguments for one operand, as
    :func:`~scratch_llm.kernels.common.hopper_contracts.check_tma_tensor_map` keywords.

    A row-major ``rows x cols`` bf16 matrix, tiled ``box_rows x box_cols``. Rank 5 rather than 2
    because the 128 B swizzle forces the innermost box to be one atom: the contiguous extent is cut
    into :data:`TMA_INNER_ELEMS`-element chunks and the chunk index becomes a third rank
    (``matmul_7.cuh:38-48``). ``global_strides_bytes`` holds dimensions 1..rank-1 only — the
    innermost is implicitly contiguous — which is the driver's own convention.

    A: ``rows=M, cols=K, box=(TILE_M, TILE_K)``.  B: ``rows=K, cols=N, box=(TILE_K, TILE_N)``.
    """
    return {
        "rank": 5,
        "elem_bytes": ELEM_BYTES,
        "global_dims": [TMA_INNER_ELEMS, rows, cols // TMA_INNER_ELEMS, 1, 1],
        "global_strides_bytes": [cols * ELEM_BYTES, TMA_INNER_ELEMS * ELEM_BYTES, 0, 0],
        "box_dims": [TMA_INNER_ELEMS, box_rows, box_cols // TMA_INNER_ELEMS, 1, 1],
        "swizzle": SwizzleMode.B128,
    }


def shape_violations(m: int, n: int, k: int) -> list[str]:
    """Why ``(m, n, k)`` cannot run at this rung (empty == it can).

    Separated from :func:`h_r3_gemm` so the constraint is testable without a GPU, and so a bench or
    a sweep can skip a shape rather than discover it as a driver error code.
    """
    errs: list[str] = []
    if k % TMA_INNER_ELEMS:
        errs.append(
            f"K={k} is not a multiple of {TMA_INNER_ELEMS}: A's tensor map splits its contiguous "
            f"extent into {TMA_INNER_ELEMS}-element swizzle atoms and cannot express a remainder"
        )
    if n % TMA_INNER_ELEMS:
        errs.append(
            f"N={n} is not a multiple of {TMA_INNER_ELEMS}: B's tensor map splits its contiguous "
            f"extent into {TMA_INNER_ELEMS}-element swizzle atoms and cannot express a remainder"
        )
    if k % TILE_K:
        errs.append(
            f"K={k} is not a multiple of TILE_K={TILE_K} (no K-remainder path at this rung)"
        )
    return errs


def h_r3_gemm(a: Tensor, b: Tensor) -> Tensor:
    """``C = A @ B`` for bf16 ``a`` [M,K] and ``b`` [K,N]; fp32-accumulated, **fp32 output**.

    fp32 out, not bf16: the accumulator is fp32 in registers and rounding it on the way out would
    fold the kernel's error together with the output cast's, leaving the correctness gate unable to
    tell a descriptor bug from a rounding difference.

    ``M`` is unconstrained — TMA zero-fills the rows past the end and the epilogue predicates the
    store — but ``N`` and ``K`` must each be a multiple of :data:`TMA_INNER_ELEMS`; see
    :func:`shape_violations`.

    The shape and dtype checks run *before* :func:`require_arch`, the reverse of H-R1's order, so
    that the whole contract is reachable from a CPU test. Only the last line needs silicon.

    Raises ``RuntimeError`` on a non-Hopper device and
    :class:`~scratch_llm.kernels.gemm.cuda._k1_loader.HoleOpenError` while the kernel body is still
    an unfilled hole.
    """
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        raise TypeError(
            f"h_r3_gemm expects bfloat16 operands (wgmma.f32.bf16.bf16), got {a.dtype} and "
            f"{b.dtype}; the K1 floor is cuBLAS bf16, so another dtype measures against no floor"
        )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"h_r3_gemm expects 2-D operands, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"h_r3_gemm inner dimensions disagree: {a.shape[1]} vs {b.shape[0]}")
    errs = shape_violations(a.shape[0], b.shape[1], a.shape[1])
    if errs:
        raise ValueError("h_r3_gemm cannot run this shape: " + "; ".join(errs))
    require_arch(ARCH, fn_name="h_r3_gemm")
    return getattr(load_rung("k1_h_r3", SOURCE, SYMBOL, ARCH), SYMBOL)(a, b)


def reference_gemm(a: Tensor, b: Tensor) -> Tensor:
    """The oracle: the same bf16 inputs, multiplied in fp32.

    Slow, obviously correct, and — importantly — *not* a different problem. It up-casts the
    identical bf16 operands the kernel receives, so the only difference between this and the kernel
    is accumulation order and the tensor core's internal rounding. Comparing against an fp32 *input*
    GEMM instead would fold the operands' own quantization error into the tolerance and hide a real
    kernel bug inside it.
    """
    return torch.matmul(a.float(), b.float())
