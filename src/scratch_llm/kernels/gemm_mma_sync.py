"""A3 Rung 2 — JIT loader for the mma.sync + ldmatrix + XOR-swizzled-SMEM warp-MMA GEMM.

The kernel (``gemm_mma_sync.cu``) is compiled on first import via ``torch.utils.cpp_extension.load``
(JIT nvcc, ``-arch=sm_120`` — the RTX PRO 4000 Blackwell in this box). This is Rung 1's WMMA kernel
with the abstraction lid off: the PTX warp-MMA ``mma.sync.aligned.m16n8k16.f32.f16.f16.f32`` issued by
hand, operand fragments loaded straight into the tensor-core register layout with ``ldmatrix.sync``,
and an XOR-swizzled shared-memory layout so consecutive ``ldmatrix`` rows land in distinct banks.

Inputs are float16 (the ``mma.f16.f16`` operand type); the accumulator is FP32 and the output is FP32
(mixed precision is what keeps a long-K reduction stable and lets the single-large-outlier adversarial
case not overflow). Correctness is element-wise vs ``torch.matmul`` at fp32-accumulate tolerance,
including K non-multiples of the 16/32 tiles, M/N remainder tiles, M=1, and a single large outlier.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

_CU = Path(__file__).with_suffix(".cu")


@lru_cache(maxsize=1)
def _module():  # noqa: ANN202 - opaque pybind extension module
    from torch.utils.cpp_extension import load

    return load(
        name="gemm_mma_sync",
        sources=[str(_CU)],
        extra_cuda_cflags=["-arch=sm_120", "-O3"],
        verbose=False,
    )


def gemm_mma_sync(a: Tensor, b: Tensor) -> Tensor:
    """C = A @ B for float16 ``a`` [M,K], ``b`` [K,N]; fp32-accumulated, **fp32 output**.

    Uses ``mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`` warp-MMAs fed by ``ldmatrix.sync``
    from XOR-swizzled SMEM. Remainders (K, M, N non-multiples of the tiles; M=1) are handled by
    bounds-masked zero-fill on the global->SMEM loads plus a bounds-checked epilogue.
    """
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("gemm_mma_sync expects float16 inputs (mma.f16.f16 operands)")
    return _module().gemm_mma_sync(a, b)
