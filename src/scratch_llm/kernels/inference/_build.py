"""Build the aggregated `scratch_llm_kernels` CUDA extension — one `.so` for all inference kernels.

Shared by every domain wrapper module (norm.py, gemm.py, ...). The `__global__` bodies live in
`../csrc/<domain>/*.cu` and are reconstructed from blank (the meat boundary); this builds them.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

_CSRC_DIR = Path(__file__).parent.parent / "csrc"

# Profiling-grade device flags. -lineinfo: keep full -O3 optimization but emit SASS↔source line
# mapping so ncu/Nsight Compute show your .cu lines (essential for the roofline DoD). ptxas -v:
# print per-kernel registers / shared mem / spills at build → occupancy reasoning (the FOP
# "name the constraint: launch/latch/occupancy/bank-conflict"). NO global --use_fast_math: fast-math
# is a per-op call (use rsqrtf/__expf in-kernel), so a softmax kernel's numerics aren't silently
# degraded module-wide. Set SCRATCH_LLM_KERNEL_VERBOSE=1 to surface the ptxas -v / build log.
_CUDA_CFLAGS = ["-O3", "-lineinfo", "--ptxas-options=-v"]


@functools.cache
def _ext():
    """Build (once) and return the single aggregated `scratch_llm_kernels` extension module.

    ONE module for all inference kernels (the shippable `_C` shape), assembled from every
    `csrc/**/*.cu` launcher (recursively, across domain dirs) + the explicit `csrc/bindings.cpp`
    (its `PYBIND11_MODULE` is the public ABI: one `m.def` per kernel). Built with
    `torch.utils.cpp_extension.load` → ninja, so editing a single `.cu` recompiles only that object
    and relinks — incremental, no manual build step, cached under TORCH_EXTENSIONS_DIR. Files
    starting with `_` (e.g. `_template.cu`) are excluded.

    CPU/CI-safe: torch is imported here, lazily, so importing this module never triggers a build.
    """
    from torch.utils.cpp_extension import load  # noqa: PLC0415

    sources = [str(p) for p in sorted(_CSRC_DIR.glob("**/*.cu")) if not p.name.startswith("_")]
    sources.append(str(_CSRC_DIR / "bindings.cpp"))
    return load(
        name="scratch_llm_kernels",
        sources=sources,
        extra_cuda_cflags=_CUDA_CFLAGS,
        verbose=os.environ.get("SCRATCH_LLM_KERNEL_VERBOSE") == "1",
    )
