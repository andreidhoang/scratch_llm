"""Production-grade custom operator dispatch layer."""

from scratch_llm.ops.attention import flash_attention

__all__ = ["flash_attention"]
