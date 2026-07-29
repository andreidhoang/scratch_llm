"""Hopper WGMMA GEMM (sm_90a) — the warpgroup-MMA tensor-core frontier rung.

This is the backend the dispatch layer routes to on Hopper (H100 / H200): one
**warpgroup** (128 threads) holds a 64×64 output tile in registers and issues
``wgmma.mma_async`` straight against hand-encoded SMEM descriptors. Unlike WMMA
(A3 R1 — fragment-MMA), WGMMA reads operands *from shared memory* (not from
register fragments) via a 64-bit descriptor that encodes the start address, the
leading-byte-offset, the stride-byte-offset, and the 128B swizzle. That is the
load-bearing abstraction lift of Hopper: no ``ldmatrix``, no per-thread operand
load — the MMA reads SMEM directly, which is what lets it overlap with TMA loads
and reach ~318 TFLOPS on a 4096³ FP16 GEMM (book §7.3.1, ≈4.5× over WMMA's 71).

ISA gate: WGMMA assembles ONLY under ``-arch=sm_90a`` (the trailing ``a`` is
the *accelerated* ISA flag — base ``sm_90`` silently omits ``wgmma.*``). This
loader enforces that gate via :func:`require_cc(9, 0)` at entry AND by passing
``-arch=sm_90a`` to the JIT loader (NOT the dev-box arch — the kernel would
assemble to nothing on sm_120).

CORRECTNESS HONESTY: the kernel body in ``csrc/gemm/wgmma_sm90.cu`` is a
compile-gated structural skeleton (promoted from ``performance/rental/kernels/``)
— it assembles + the emitted PTX contains ``wgmma.mma_async``, but runtime
correctness is deferred to the H100/H200 rental day. The oracle test
(``tests/kernels/test_gemm_wgmma.py``) runs there, not here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from scratch_llm.kernels.common.arch import require_cc

# The promoted .cu source lives under csrc/ (the AOT source tree), NOT next to
# this loader — the JIT path reads it from there so the AOT and JIT paths
# compile the exact same source.
_CU = (
    Path(__file__).resolve().parents[5] / "csrc" / "gemm" / "wgmma_sm90.cu"
)  # kernels/gemm/cuda/wgmma.py → parents[5] = repo root


@lru_cache(maxsize=1)
def _module():  # noqa: ANN202 - opaque pybind extension module
    # 1. AOT path — the CMake/setup.py-built _scratch_llm_kernels extension.
    try:
        from scratch_llm import _scratch_llm_kernels as _ext  # noqa: PLC0415

        if hasattr(_ext, "wgmma_gemm_sm90"):
            return _ext
    except ImportError:
        pass

    # 2. JIT fallback — compiles csrc/gemm/wgmma_sm90.cu for sm_90a on first call.
    #    The arch is HARDCODED to sm_90a here (not read from arch_name()): WGMMA
    #    needs the 'a' suffix even on a Hopper box where arch_name() returns sm_90.
    from torch.utils.cpp_extension import load

    return load(
        name="wgmma_gemm_sm90",
        sources=[str(_CU)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90a"],
        verbose=False,
    )


def wgmma_gemm(a: Tensor, b: Tensor) -> Tensor:
    """C = A @ B for float16 ``a`` [M,K], ``b`` [K,N]; fp32-accumulated, **fp32 output**.

    Warpgroup-MMA on Hopper. ``M`` and ``N`` must be multiples of 64 (one
    warpgroup tile); no edge handling yet (the rental measurement sizes are
    4096+, so the fast path is what's measured). Raises on non-Hopper devices.
    """
    require_cc(9, 0, fn_name="wgmma_gemm")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("wgmma_gemm expects float16 inputs (wgmma.f16.f16 operands)")
    return _module().wgmma_gemm_sm90(a, b)
