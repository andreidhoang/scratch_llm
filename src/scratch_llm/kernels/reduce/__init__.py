"""Reduction family (softmax, top-k) -- stable dispatch surface.

Public surface: import from ``.dispatch`` -- it routes CPU -> ``torch.softmax`` /
``torch.topk`` and CUDA -> the triton kernels (incl. the fused softmax+topk
sampling path), and re-exports each backend lazily. Backend files are PRIVATE;
production code above ``kernels/`` reaches them only through ``dispatch``.

Nothing is re-exported at the package level: every backend is GPU-only (Triton),
so eager re-export would break the ``kernels/`` CPU-safety invariant. Use
``from scratch_llm.kernels.reduce.dispatch import softmax`` (the routed entry) or
``... import softmax_triton`` (the backend, loaded lazily).
"""
