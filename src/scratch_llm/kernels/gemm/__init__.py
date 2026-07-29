"""GEMM / GEMV family -- operation-first layout with a stable dispatch surface.

Public surface: import from ``.dispatch`` -- it routes to the right backend
(triton / cuda / wmma) by ``(device, dtype, shape)`` and exposes every backend's
entry point lazily. Backend subpackages (``triton/``, ``cuda/``, ``wmma/``) are
PRIVATE; production code above ``kernels/`` reaches them only through ``dispatch``.
Tests and benches may target a backend directly (the GEMM ladder benchmarks the
rungs head-to-head).

Nothing is re-exported at the package level: every backend is GPU-only (Triton or
nvcc JIT), so eager re-export would break the ``kernels/`` CPU-safety invariant.
Use ``from scratch_llm.kernels.gemm.dispatch import matmul`` (the routed entry) or
``... import wmma_gemm`` (a specific backend, loaded lazily).
"""
