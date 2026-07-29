"""Normalization family (RMSNorm, LayerNorm) -- stable dispatch surface.

Public surface: import from ``.dispatch`` -- it routes CPU -> ``F.rms_norm`` /
``F.layer_norm`` and CUDA -> the triton one-row-per-block kernels, and re-exports
each backend lazily. The backend file is PRIVATE; production code above
``kernels/`` reaches it only through ``dispatch``.

Nothing is re-exported at the package level: the backend is GPU-only (Triton), so
eager re-export would break the ``kernels/`` CPU-safety invariant. Use
``from scratch_llm.kernels.norm.dispatch import rmsnorm`` (the routed entry) or
``... import rmsnorm_triton`` (the backend, loaded lazily).
"""
