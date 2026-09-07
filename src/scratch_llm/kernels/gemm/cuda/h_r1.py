"""K1/H-R1 — warpgroup MMA from shared memory, single stage (Hopper, sm_90a).

The first rung of the K1 GEMM ladder: one warpgroup (128 threads) owns one 128x128 output tile,
stages A and B into shared memory, and issues ``wgmma.mma_async.m64n128k16.f32.bf16.bf16`` against
descriptors it builds by hand. No TMA, no multistage pipeline, no warp specialisation, no
persistence — each of those is a later rung, held back deliberately so that the gap this kernel
leaves is legible in a profile rather than mixed in with three other effects.

bf16, not fp16: the ladder's floor is cuBLAS **bf16** at 4096^3 (plan §04/§05), and a number
measured in one dtype against a floor in another is not a number.

Kernel source and the ``# HUY:`` hole: ``csrc/gemm/h_r1_wgmma_bf16_sm90.cu``.
Spec, floor and kill rule: ``experiments/K1/H-R1/spec.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scratch_llm.kernels.common.arch import require_arch
from scratch_llm.kernels.gemm.cuda._k1_loader import load_rung

#: The K1 rung identity, in one place, so the wrapper, the tests, the bench and the dry-dock
#: registry cannot drift apart.
RUNG = "K1/H-R1"
SOURCE = "h_r1_wgmma_bf16_sm90.cu"
SYMBOL = "h_r1_wgmma_bf16"
ARCH = (9, 0)

#: Tile shape, mirrored from the ``BM``/``BN``/``BK`` defines in the ``.cu``. The Python side needs
#: these to check the shared-memory budget and the wave quantization without a GPU
#: (``kernels.common.hopper_contracts``), and the CPU contract tests assert the two agree.
TILE_M, TILE_N, TILE_K = 128, 128, 64
THREADS_PER_CTA = 128


def h_r1_gemm(a: Tensor, b: Tensor) -> Tensor:
    """``C = A @ B`` for bf16 ``a`` [M,K] and ``b`` [K,N]; fp32-accumulated, **fp32 output**.

    fp32 out, not bf16: the accumulator is fp32 in registers and rounding it on the way out would
    fold the kernel's error together with the output cast's, leaving the correctness gate unable to
    tell a descriptor bug from a rounding difference.

    ``K`` must be a multiple of ``TILE_K`` (no K-remainder handling at this rung). ``M`` and ``N``
    are predicated, so the non-power-of-two test shape is a real test of the epilogue.

    Raises ``RuntimeError`` on a non-Hopper device and
    :class:`~scratch_llm.kernels.gemm.cuda._k1_loader.HoleOpenError` while the kernel body is
    still an unfilled hole.
    """
    require_arch(ARCH, fn_name="h_r1_gemm")
    if a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
        raise TypeError(
            f"h_r1_gemm expects bfloat16 operands (wgmma.f32.bf16.bf16), got {a.dtype} and "
            f"{b.dtype}; the K1 floor is cuBLAS bf16, so another dtype measures against no floor"
        )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"h_r1_gemm expects 2-D operands, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"h_r1_gemm inner dimensions disagree: {a.shape[1]} vs {b.shape[0]}")
    if a.shape[1] % TILE_K:
        raise ValueError(
            f"h_r1_gemm requires K % {TILE_K} == 0 at this rung (no K-remainder handling); got K={a.shape[1]}"
        )
    return getattr(load_rung("k1_h_r1", SOURCE, SYMBOL, ARCH), SYMBOL)(a, b)


def reference_gemm(a: Tensor, b: Tensor) -> Tensor:
    """The oracle: the same bf16 inputs, multiplied in fp32.

    Slow, obviously correct, and — importantly — *not* a different problem. It up-casts the
    identical bf16 operands the kernel receives, so the only difference between this and the kernel
    is accumulation order and the tensor core's internal rounding. Comparing against an fp32 *input*
    GEMM instead would fold the operands' own quantization error into the tolerance and hide a real
    kernel bug inside it.
    """
    return torch.matmul(a.float(), b.float())
