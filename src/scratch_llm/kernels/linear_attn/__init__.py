"""Linear attention (the gated delta rule) — the kernel family for this repo's North Star.

Re-exports only the CPU-safe oracle surface, so ``import scratch_llm.kernels.linear_attn``
never pulls Triton or compiles CUDA. Reach anything else through
:mod:`scratch_llm.kernels.linear_attn.dispatch`.
"""

from scratch_llm.kernels.linear_attn.dispatch import chunked_wy, recurrent_reference

__all__ = ["chunked_wy", "recurrent_reference"]
