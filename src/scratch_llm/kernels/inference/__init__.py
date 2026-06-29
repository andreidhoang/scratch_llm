"""CUDA inference-kernel wrappers (decode path), grouped by domain.

These are the *decode-time* kernels: at batch=1, seq=1 arithmetic intensity collapses and every op
is HBM-bandwidth bound, so the DoD is "% of the memory roofline", not TFLOP/s. One module per
domain owns a kernel's Python surface — the pure-torch oracle (`*_ref`), the dispatch, and the
roofline DoD entry (`*_roofline`):

    norm.py   RMSNorm           (csrc/norm/)
    gemm.py   GEMV + GEMM       (csrc/gemm/)

All built into one aggregated extension on first use (see `_build._ext`). Public names are
re-exported here so `from scratch_llm.kernels.inference import rmsnorm` stays stable across the
file split. CPU/CI-safe: torch and the CUDA build are imported lazily (importing this package never
compiles anything). The `__global__` bodies live in `../csrc/<domain>/*.cu` (the meat boundary).
"""

from scratch_llm.kernels.inference.gemm import (
    gemm,
    gemm_ref,
    gemm_roofline,
    gemv,
    gemv_ref,
    gemv_roofline,
)
from scratch_llm.kernels.inference.norm import rmsnorm, rmsnorm_ref, rmsnorm_roofline

__all__ = [
    "rmsnorm",
    "rmsnorm_ref",
    "rmsnorm_roofline",
    "gemv",
    "gemv_ref",
    "gemv_roofline",
    "gemm",
    "gemm_ref",
    "gemm_roofline",
]
