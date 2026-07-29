"""Shared kernel primitives used across operation families.

These are building blocks, not backends: CPU-safe pure-torch helpers
(:mod:`online_softmax`) and device-detection primitives (:mod:`arch`) that the
operation-family dispatch layers and the kernels themselves reuse. They are the
only ``kernels/`` submodules production code may import directly without going
through a family ``dispatch.py`` (they have no backend -- they are below the
backend layer).

Public:
  * :mod:`arch` -- GPU compute-capability detection + ISA gating (CPU-safe:
    importing it never initializes CUDA; detectors query lazily and cache).
  * :mod:`online_softmax` -- the streaming (m, d) recurrence FlashAttention fuses,
    isolated as a 1-D pure-torch oracle (CPU-safe).
"""

from scratch_llm.kernels.common.arch import (  # CPU-safe: lazy CUDA detection
    arch_name,
    compute_capability,
    is_blackwell,
    is_hopper,
    is_sm120,
    require_cc,
)
from scratch_llm.kernels.common.online_softmax import (
    online_softmax,
    online_softmax_normalizer,
    three_pass_softmax,
)

__all__ = [
    "arch_name",
    "compute_capability",
    "is_blackwell",
    "is_hopper",
    "is_sm120",
    "require_cc",
    "online_softmax",
    "online_softmax_normalizer",
    "three_pass_softmax",
]
