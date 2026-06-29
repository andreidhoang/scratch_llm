"""Measurement apparatus — the engineering that makes a kernel's speedup a *measured* result.

The unit of work is a measured artifact, gated by the Artifact DoD: correct vs an oracle, profiled to
land near a roofline you predicted first (FOP-3), the predicted-vs-measured number on record
(:mod:`scratch_llm.bench.ledger`), a one-line root cause, and a regression guard. See
``docs/PERFORMANCE_TRACK.md`` and ``docs/design/PERF_roofline_harness_SPEC.md``.
"""

from scratch_llm.bench.gpu_specs import GPUS, GpuSpec, measure_hbm_bandwidth
from scratch_llm.bench.harness import BenchStats, benchmark
from scratch_llm.bench.ledger import Record, Regression, append, check_regressions, load
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
    "Record",
    "Roofline",
    "Regression",
    "append",
    "benchmark",
    "check_regressions",
    "decode_step_flops_bytes",
    "elementwise_flops_bytes",
    "gemm_flops_bytes",
    "load",
    "measure_hbm_bandwidth",
    "pct_of_roof",
    "roofline",
]
