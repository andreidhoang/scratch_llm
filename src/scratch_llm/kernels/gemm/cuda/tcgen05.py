"""Blackwell tcgen05 / UMMA GEMM (sm_100a) — the datacenter-Blackwell frontier rung.

tcgen05 is the *generation-5 tensor core* ISA on Blackwell datacenter (B200):
unlike Hopper's WGMMA (which accumulates in *registers*), tcgen05 accumulates
in **Tensor Memory (TMEM)** — a private per-SM SRAM region accessible only via
the ``tcgen05.alloc / mma / ld / commit / wait`` instruction family. The MMA
itself is issued by ONE thread (asynchronous, arrives on an mbarrier), and the
full warpgroup (128 threads) drains TMEM into registers in the epilogue.

This backend is the dispatch target on Blackwell datacenter (sm_100a). The
dev-box sm_120 (RTX PRO 4000) is a Blackwell *client* part — it has NO tcgen05,
NO TMEM, NO ``cta_group``. This loader enforces that via ``require_cc(10, 0)``.

Two-SM (``cta_group::2``) variant is a documented ~8% win (SMEM-bandwidth
relief); this file is the 1-SM rung only.

CORRECTNESS HONESTY: the kernel body in ``csrc/gemm/tcgen05_sm100.cu`` is a
compile-gated structural skeleton (promoted from ``performance/rental/kernels/``).
Runtime correctness is deferred to the B200 rental day.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from scratch_llm.kernels.common.arch import require_cc

_CU = Path(__file__).resolve().parents[5] / "csrc" / "gemm" / "tcgen05_sm100.cu"


@lru_cache(maxsize=1)
def _module():  # noqa: ANN202 - opaque pybind extension module
    # 1. AOT path.
    try:
        from scratch_llm import _scratch_llm_kernels as _ext  # noqa: PLC0415

        if hasattr(_ext, "tcgen05_gemm_sm100"):
            return _ext
    except ImportError:
        pass

    # 2. JIT fallback — tcgen05 assembles only under -arch=sm_100a.
    from torch.utils.cpp_extension import load

    return load(
        name="tcgen05_gemm_sm100",
        sources=[str(_CU)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100a"],
        verbose=False,
    )


def tcgen05_gemm(a: Tensor, b: Tensor) -> Tensor:
    """C = A @ B for float16 ``a`` [M,K], ``b`` [K,N]; fp32-accumulated, **fp32 output**.

    Blackwell datacenter tcgen05/UMMA. ``M`` must be a multiple of 128, ``N`` of
    256 (the 1-SM atom tile); no edge handling. Raises on non-Blackwell-DC devices.
    """
    require_cc(10, 0, fn_name="tcgen05_gemm")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("tcgen05_gemm expects float16 inputs (tcgen05.f16 operands)")
    return _module().tcgen05_gemm_sm100(a, b)
