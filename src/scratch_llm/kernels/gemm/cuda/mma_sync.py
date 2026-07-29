"""A3 Rung 2 — JIT loader for the mma.sync + ldmatrix + XOR-swizzled-SMEM warp-MMA GEMM.

The kernel (``gemm_mma_sync.cu``) is compiled on first import via ``torch.utils.cpp_extension.load``
(JIT nvcc). The target ISA is read at runtime from :func:`scratch_llm.kernels.common.arch.arch_name`
(so the same loader builds for sm_120 / sm_90 / sm_100 without an edit). This is Rung 1's WMMA kernel
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
    # 1. AOT path — the CMake/setup.py-built extension (rental box, `pip install -e ".[gpu-aot]"`).
    #    This loader's kernel lives in csrc/pybind.cpp's symbol table; if the AOT extension was
    #    built, prefer it (no JIT cost on first call — the reason the rental benches use AOT).
    try:
        from scratch_llm import _scratch_llm_kernels as _ext  # noqa: PLC0415

        if hasattr(_ext, "gemm_mma_sync"):
            return _ext
    except ImportError:
        pass

    # 2. JIT fallback — the dev-iteration path (nvcc at first call, no build step).
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
        name="gemm_mma_sync",
        sources=[str(_CU)],
        extra_cuda_cflags=flags,
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
