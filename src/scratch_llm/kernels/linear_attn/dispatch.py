"""Stable public surface for linear-attention (gated delta rule) kernels — the dispatch layer.

This is the seam the rest of the repo imports; backend subpackages will be PRIVATE, exactly as
in ``attention/`` and ``gemm/``. It exists now, before any GPU backend, for one structural
reason: the gated delta rule is this repo's North Star subsystem, and until this module existed
the recurrence had three *evidential* expressions (fp64 oracle, teaching module, hand-build
target) and no *production* one — nothing a model could call, nothing a kernel could graduate
into. Opening the seam first means the L3.2 Triton chunked-KDA kernel has a documented home and
a documented oracle on the day it is written, instead of a fourth copy of the recurrence.

Stable surface (CPU-safe, eager — the correctness ground truth):
    recurrent_reference(q, k, v, log_alpha, beta, S0=None) -> (O, S)
        The fp64 sequential recurrence. THE oracle; the human's hand (``oracle-guard.sh``).
        Raises ``NotImplementedError`` until L0.1 lands — deliberately, see ``reference.py``.
    chunked_wy(q, k, v, log_alpha, beta, chunk_size=64, S0=None,
               return_diagnostics=False, solve_dtype=None) -> (O, S) | (O, S, diag)
        The chunked WY form a training kernel must use. What the divergence map measures.
        ``solve_dtype`` holds the CxC triangular solve's precision separate from the matmul
        dtype (FLA runs the WY transform in fp32 while feeding bf16 to the tensor cores).

Backends (none yet — this is the honest state, not an omission):
    ``_LAZY_GPU`` is empty. When the first GPU backend lands it registers here and callers above
    ``kernels/`` do not move. The planned first entry is the L3.2 Triton chunked-KDA kernel
    (spec § 12.5), which will live at ``kernels/linear_attn/triton/chunk_kda.py``. That path is
    hook-blocked in ``learn`` mode (``kernel-write-guard.sh`` guards ``kernels/*/triton/*``) —
    correct by design: the kernel body is the human's rep, everything around it is delegable.

Graduation rule (``kernels/CLAUDE.md``): a backend is "shipped" only once it has (a) an
oracle-first correctness test against ``recurrent_reference`` in fp64, (b) a measured roofline
row in ``bench/RESULTS.md``, and (c) a dispatch entry here. Until (a)-(c), it is a rental
experiment, not a kernel. This family currently has zero shipped backends and says so.

CPU-safety: the oracle re-export is eager and pure-PyTorch; GPU backends will load lazily via
PEP 562 ``__getattr__``, so ``import scratch_llm.kernels.linear_attn.dispatch`` stays CPU-safe.
This is asserted by ``tests/kernels/test_kernel_dispatch_boundary.py``.
"""

from __future__ import annotations

from scratch_llm.kernels.linear_attn.reference import (  # CPU-safe: the fp64 oracle + chunked path
    chunked_wy,
    recurrent_reference,
)

# The EAGER, CPU-safe surface. `from dispatch import *` can never pull a GPU dep.
__all__ = ["chunked_wy", "recurrent_reference"]

# Lazy GPU backend loader (PEP 562), mirroring `attention/dispatch.py::_LAZY_GPU`.
# Empty until the first backend graduates — see the module docstring's graduation rule.
_LAZY_GPU: dict[str, tuple[str, str]] = {}


def __getattr__(name: str):
    if name in _LAZY_GPU:
        import importlib

        mod_name, attr = _LAZY_GPU[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()) | set(_LAZY_GPU))
