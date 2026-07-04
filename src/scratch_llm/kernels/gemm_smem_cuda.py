"""A3 Rung 0 — JIT loader for the naive SMEM-blocked bf16 GEMM (CUDA C++, CUDA cores only).

The kernel (``gemm_smem_cuda.cu``) is compiled on first import via ``torch.utils.cpp_extension.load``
(JIT nvcc, targeting ``sm_120`` — the RTX PRO 4000 Blackwell in this box). This is the re-anchor floor
the A3 tensor-core ladder climbs from: a classic 32x32 shared-memory-tiled hgemm with fp32 accumulation
and no tensor cores. Expect a low-single-digit %-of-cuBLAS — that is correct; WMMA is what beats it.
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
        name="gemm_smem_cuda",
        sources=[str(_CU)],
        extra_cuda_cflags=["-arch=sm_120", "-O3"],
        verbose=False,
    )


def gemm_smem(a: Tensor, b: Tensor) -> Tensor:
    """C = A @ B for bf16 ``a`` [M,K], ``b`` [K,N]; fp32-accumulated, bf16 output.

    Correctness is defined element-wise against ``torch.matmul`` at fp32-accumulate tolerance,
    including K non-multiples of the 32-tile, M/N remainder tiles, M=1, and a single large outlier
    (the bounds-masked tile loads and the single end-of-reduction round make all of those exact).
    """
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("gemm_smem expects bfloat16 inputs")
    return _module().sgemm_smem(a, b)
