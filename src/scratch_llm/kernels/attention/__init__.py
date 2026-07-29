"""Attention kernels — operation family with a stable dispatch surface.

Public symbols come through ``.dispatch`` (the routing layer); backend
subpackages (``prefill/``, ``decode/``) are PRIVATE — production code above
``kernels/`` reaches them only through ``dispatch.py``. Tests and benches may
target a backend directly.

Re-exported here: the CPU-safe oracle only, so ``import scratch_llm.kernels``
never pulls Triton (the ``kernels/`` package invariant). GPU symbols are accessed
via ``scratch_llm.kernels.attention.dispatch`` (lazy, PEP 562).
"""

from scratch_llm.kernels.attention.dispatch import (
    FlashAttentionPyTorch,
    flash_attention_forward,
)

__all__ = ["FlashAttentionPyTorch", "flash_attention_forward"]
