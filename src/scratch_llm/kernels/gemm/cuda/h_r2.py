"""K1/H-R2 — TMA + 128B swizzle + mbarrier expect-tx, 2 stages (Hopper, sm_90a).

H-R1 with one thing changed: how the operand tiles reach shared memory. Same 128x128x64 tile, same
warpgroup, same ``wgmma.mma_async.m64n128k16.f32.bf16.bf16``, same fp32 accumulator. The staging
path is now the tensor memory accelerator — a host-built ``CUtensorMap``, one thread issuing
``cp.async.bulk.tensor``, an mbarrier that counts *bytes* — with two buffers, so the copy for k+1 is
in flight while the MMA for k runs. The tile shape is frozen at H-R1's precisely so the delta is the
staging path and nothing else.

Two things this module carries that H-R1's did not, because both are decidable on a laptop and both
cost a rented hour if they are wrong:

* :func:`tensor_map_params` — the exact ``cuTensorMapEncodeTiled`` arguments the ``.cu`` builds,
  as data, so ``hopper_contracts.check_tma_tensor_map`` can rule on them in the CPU suite. The
  driver answers every illegal field with ``CUDA_ERROR_INVALID_VALUE`` and names none of them.
* :func:`unsupported_reason` — the shapes TMA cannot address at all. A 16-byte row-stride rule is
  not a shortcut this rung took; it is why H-R1's ``npot`` shape (N=1023) is not measurable here.

Kernel source and the ``# HUY:`` hole: ``csrc/gemm/h_r2_tma_bf16_sm90.cu``.
Spec, floor and kill rule: ``experiments/K1/H-R2/spec.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scratch_llm.kernels.common.arch import require_arch
from scratch_llm.kernels.common.hopper_contracts import SwizzleMode
from scratch_llm.kernels.gemm.cuda._k1_loader import load_rung

#: The K1 rung identity, in one place, so the wrapper, the tests, the bench and the dry-dock
#: registry cannot drift apart.
RUNG = "K1/H-R2"
SOURCE = "h_r2_tma_bf16_sm90.cu"
SYMBOL = "h_r2_tma_bf16"
ARCH = (9, 0)

#: Tile shape, mirrored from the ``BM``/``BN``/``BK``/``STAGES`` defines in the ``.cu``. Frozen at
#: H-R1's tile on purpose: a tile change plus a staging change would leave the rung-to-rung delta
#: attributable to neither. The CPU contract tests assert the two sides agree.
TILE_M, TILE_N, TILE_K = 128, 128, 64
STAGES = 2
THREADS_PER_CTA = 128

#: Element type of both operands, in bytes. bf16 because the K1 floor is cuBLAS **bf16**.
ELEM_BYTES = 2

#: The 128B swizzle atom in bf16 elements. Every TMA box's innermost extent must equal exactly
#: this — ``cuTensorMapEncodeTiled`` rejects anything else under ``CU_TENSOR_MAP_SWIZZLE_128B``.
#: ``TILE_N`` is two atoms wide, which is why the B tile takes two boxes rather than one.
SWIZZLE_ATOM_ELEMS = 128 // ELEM_BYTES
B_CHUNKS = TILE_N // SWIZZLE_ATOM_ELEMS

#: TMA's own alignment rule: ``globalStrides`` must be a multiple of 16 B. In bf16 that is 8
#: elements, applied to the leading dimension of each operand — A's K and B's N.
TMA_STRIDE_ELEMS = 16 // ELEM_BYTES

#: The two shared-memory matrix descriptors, in bytes. They differ because the operands' smem
#: layouts differ: A arrives K-contiguous (Major::K, LBO/SBO = 16/1024, the same pair H-R1 uses),
#: B arrives N-contiguous (Major::MN, LBO = the stride between the two 64-wide N halves = 8192,
#: SBO = the stride between k-groups of 8 = 1024). Derived in the ``.cu``'s layout comment against
#: CUTLASS ``make_gmma_desc``'s two canonical forms; asserted against the source in the tests.
A_DESC_LBO, A_DESC_SBO = 16, 1024
B_DESC_LBO = SWIZZLE_ATOM_ELEMS * TILE_K * ELEM_BYTES
B_DESC_SBO = 1024

#: Shared memory, per stage and in total. One A tile + one B tile per stage; the mbarriers are
#: static and sit outside the dynamic allocation, which is why they are ``extra_bytes`` and not
#: part of the stage.
BYTES_PER_STAGE = (TILE_M * TILE_K + TILE_N * TILE_K) * ELEM_BYTES
SMEM_DYNAMIC_BYTES = BYTES_PER_STAGE * STAGES
SMEM_BARRIER_BYTES = STAGES * 8

#: The mbarrier ``expect_tx`` byte count for one stage: what all three TMAs of that stage deliver
#: together. Too high hangs, too low lets the MMA read a half-written tile.
EXPECT_TX_BYTES = BYTES_PER_STAGE


def tensor_map_params(m: int, n: int, k: int) -> dict[str, dict[str, object]]:
    """The ``cuTensorMapEncodeTiled`` arguments ``h_r2_encode_tensor_map`` builds, as data.

    Keyword-for-keyword what :func:`~scratch_llm.kernels.common.hopper_contracts.check_tma_tensor_map`
    takes, so the CPU suite can rule on every field by name. Extents are given innermost-first —
    the driver's order, not the matrix's — and ``global_strides_bytes`` holds dimensions 1.. only,
    because the descriptor assumes the innermost stride is one element and is handed ``stride + 1``
    (``oss/fast.cu/h100/matmul/matmul_2.cuh:38,:44``).

    ``a``  (M,K) row-major, innermost K; box ``{TILE_K, TILE_M}``.
    ``b``  (K,N) row-major, innermost N; box ``{SWIZZLE_ATOM_ELEMS, TILE_K}`` — one of the two
           64-wide halves the mainloop loads per stage, at N-coordinates ``tileN`` and ``tileN+64``.
    """
    return {
        "a": {
            "rank": 2,
            "elem_bytes": ELEM_BYTES,
            "global_dims": [k, m],
            "global_strides_bytes": [k * ELEM_BYTES],
            "box_dims": [TILE_K, TILE_M],
            "swizzle": SwizzleMode.B128,
        },
        "b": {
            "rank": 2,
            "elem_bytes": ELEM_BYTES,
            "global_dims": [n, k],
            "global_strides_bytes": [n * ELEM_BYTES],
            "box_dims": [SWIZZLE_ATOM_ELEMS, TILE_K],
            "swizzle": SwizzleMode.B128,
        },
    }


def unsupported_reason(m: int, n: int, k: int) -> str | None:
    """Why this shape cannot run on this rung, or ``None`` if it can.

    Two rules, and neither is a corner cut. ``K % TILE_K`` is the rung's own scope (no K-remainder
    path, same as H-R1). The 16-byte stride rule belongs to TMA: a leading dimension that is not a
    multiple of 8 bf16 elements cannot be described by a tensor map at all, so a shape like H-R1's
    ``npot`` (N=1023) is not a harder version of this rung, it is a different kernel.
    """
    if k % TILE_K:
        return f"K % {TILE_K} != 0 (no K-remainder handling at this rung); K={k}"
    if k % TMA_STRIDE_ELEMS or n % TMA_STRIDE_ELEMS:
        return (
            f"TMA requires the row stride to be a multiple of 16 B, i.e. K and N multiples of "
            f"{TMA_STRIDE_ELEMS} bf16 elements; K={k} N={n}"
        )
    return None


def h_r2_gemm(a: Tensor, b: Tensor) -> Tensor:
    """``C = A @ B`` for bf16 ``a`` [M,K] and ``b`` [K,N]; fp32-accumulated, **fp32 output**.

    Signature-identical to :func:`~scratch_llm.kernels.gemm.cuda.h_r1.h_r1_gemm`, because the two
    rungs are compared to each other and to one floor: an operand layout that differed between them
    would make the comparison meaningless.

    ``M`` and ``N`` are predicated — TMA zero-fills out-of-bounds coordinates and the epilogue drops
    them — but ``N`` must still be a multiple of 8 for the tensor map to exist at all. See
    :func:`unsupported_reason`.

    Raises ``RuntimeError`` on a non-Hopper device and
    :class:`~scratch_llm.kernels.gemm.cuda._k1_loader.HoleOpenError` while the kernel body is
    still an unfilled hole.
    """
    require_arch(ARCH, fn_name="h_r2_gemm")
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        raise TypeError(
            f"h_r2_gemm expects bfloat16 operands (wgmma.f32.bf16.bf16), got {a.dtype} and "
            f"{b.dtype}; the K1 floor is cuBLAS bf16, so another dtype measures against no floor"
        )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"h_r2_gemm expects 2-D operands, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"h_r2_gemm inner dimensions disagree: {a.shape[1]} vs {b.shape[0]}")
    why = unsupported_reason(a.shape[0], b.shape[1], a.shape[1])
    if why is not None:
        raise ValueError(f"h_r2_gemm: {why}")
    return getattr(load_rung("k1_h_r2", SOURCE, SYMBOL, ARCH), SYMBOL)(a, b)


def reference_gemm(a: Tensor, b: Tensor) -> Tensor:
    """The oracle: the same bf16 inputs, multiplied in fp32.

    Slow, obviously correct, and — importantly — *not* a different problem. It up-casts the
    identical bf16 operands the kernel receives, so the only difference between this and the kernel
    is accumulation order and the tensor core's internal rounding. Comparing against an fp32 *input*
    GEMM instead would fold the operands' own quantization error into the tolerance and hide a real
    kernel bug inside it.
    """
    return torch.matmul(a.float(), b.float())
