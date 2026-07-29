"""A3 Rung 0 — JIT loader for the naive SMEM-blocked bf16 GEMM (CUDA C++, CUDA cores only).

The kernel (``gemm_smem_cuda.cu``) is compiled on first import via ``torch.utils.cpp_extension.load``
(JIT nvcc). The target ISA is read at runtime from :func:`scratch_llm.kernels.common.arch.arch_name`
(so the same loader builds for sm_120 / sm_90 / sm_100 without an edit) — this is the re-anchor floor
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
    from scratch_llm.kernels.common.arch import arch_name

    # arch_name() is None on a CPU box, but this loader is only reached on the CUDA path
    # (dispatch.py::matmul gates on a.is_cuda); fall back to torch's default SASS if arch
    # detection somehow misses — never a silent wrong-arch compile.
    flags = ["-O3"]
    sm = arch_name()
    if sm is not None:
        flags.append(f"-arch={sm}")
    from torch.utils.cpp_extension import load

    return load(
        name="gemm_smem_cuda",
        sources=[str(_CU)],
        extra_cuda_cflags=flags,
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
