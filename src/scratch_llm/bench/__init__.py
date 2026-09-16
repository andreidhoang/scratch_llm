"""Measurement apparatus — the engineering that makes a kernel's speedup a *measured* result.

The unit of work is a measured artifact, gated by the Artifact DoD: correct vs an oracle, profiled to
land near a roofline you predicted first (FOP-3), the predicted-vs-measured number on record
(the campaign ledger is `ladders/ledger/ledger.py` — one instrument, not two), a one-line root
cause, and a regression guard. See
``git show 07f3de4:docs/archive/PERFORMANCE_TRACK.md`` (deleted 31/08) and
``docs/design/PERF_roofline_harness_SPEC.md``.
"""

from scratch_llm.bench.gpu_specs import GPUS, GpuSpec, measure_hbm_bandwidth
from scratch_llm.bench.harness import BenchStats, benchmark
from scratch_llm.bench.roofline import (
    Roofline,
    decode_step_flops_bytes,
    elementwise_flops_bytes,
    gemm_flops_bytes,
    pct_of_roof,
    roofline,
)

__all__ = [
    "GPUS",
    "BenchStats",
    "GpuSpec",
    "Roofline",
    "benchmark",
    "decode_step_flops_bytes",
    "elementwise_flops_bytes",
    "gemm_flops_bytes",
    "measure_hbm_bandwidth",
    "pct_of_roof",
    "roofline",
]
