"""The measurement apparatus: GPU-spec ridge points, the roofline engine's bound classification, the
predict-before-run op-counters, the timing harness, and the predict-vs-measure ledger + regression
guard. All CPU — the apparatus must be testable without a GPU; real device numbers come from bench runs.
"""

import dataclasses
import math

from scratch_llm.bench import (
    GPUS,
    Record,
    append,
    benchmark,
    check_regressions,
    decode_step_flops_bytes,
    elementwise_flops_bytes,
    gemm_flops_bytes,
    load,
    pct_of_roof,
    roofline,
)
from scratch_llm.bench.gpu_specs import BF16, FP4, FP8


def test_h100_dense_ridge_is_about_295() -> None:
    # Dense (989 TFLOP/s / 3.35 TB/s), NOT the sparse ~590 — the honesty constant.
    ridge = GPUS["h100-sxm"].ridge_point(BF16)
    assert math.isclose(ridge, 989e12 / 3.35e12, rel_tol=1e-9)
    assert 290 < ridge < 300


def test_fp4_is_twice_fp8_on_blackwell() -> None:
    for key in ("b200", "gb200"):
        spec = GPUS[key]
        assert math.isclose(spec.peak_flops[FP4], 2.0 * spec.peak_flops[FP8])


def test_big_gemm_is_compute_bound() -> None:
    flops, byts = gemm_flops_bytes(4096, 4096, 4096, dtype_bytes=2)
    rp = roofline(flops, byts, GPUS["h100-sxm"], BF16)
    assert rp.bound == "compute"
    assert rp.arithmetic_intensity > rp.ridge_point


def test_elementwise_is_memory_bound() -> None:
    flops, byts = elementwise_flops_bytes(1_000_000, dtype_bytes=2, flops_per_elem=5)
    rp = roofline(flops, byts, GPUS["h100-sxm"], BF16)
    assert rp.bound == "memory"
    assert rp.arithmetic_intensity < rp.ridge_point


def _decode_70b() -> tuple[float, float]:
    return decode_step_flops_bytes(
        70_000_000_000,
        n_layers=80,
        n_kv_heads=8,
        head_dim=128,
        context_len=2048,
        batch=1,
        weight_bytes=2,
        kv_bytes=2,
    )


def test_decode_step_is_deeply_memory_bound_with_ai_about_one() -> None:
    # The thesis, numerically: batch-1 decode AI ~ 1 FLOP/byte, ~300x below the H100 ridge.
    flops, byts = _decode_70b()
    rp = roofline(flops, byts, GPUS["h100-sxm"], BF16)
    assert rp.bound == "memory"
    assert 0.5 < rp.arithmetic_intensity < 3.0
    assert rp.ridge_point / rp.arithmetic_intensity > 100


def test_decode_predicted_seconds_equals_bytes_over_bandwidth() -> None:
    # A memory-bound op's roofline time IS bytes / HBM_BW (the decode tok/s hand-formula).
    flops, byts = _decode_70b()
    spec = GPUS["h100-sxm"]
    rp = roofline(flops, byts, spec, BF16)
    assert math.isclose(rp.predicted_seconds, byts / spec.hbm_bandwidth, rel_tol=1e-9)


def test_pct_of_roof_halves_when_twice_as_slow() -> None:
    assert math.isclose(pct_of_roof(predicted_seconds=1.0, measured_seconds=2.0), 0.5)


def test_benchmark_returns_stable_stats_on_cpu() -> None:
    stats = benchmark(lambda: sum(range(1000)), warmup=2, iters=20, use_cuda=False)
    assert stats.iters == 20
    assert stats.min <= stats.median <= stats.p90
    assert stats.median >= 0.0


def test_ledger_roundtrip_and_regression_guard(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    base = Record(
        artifact="decode:split-k",
        gpu="h100-sxm",
        dtype="bf16",
        predicted_bound="memory",
        predicted_seconds=1.0,
        measured_seconds=1.2,
        pct_of_roof=0.83,
        date="2026-06-29",
        root_cause="launch overhead",
    )
    append(base, path)
    append(dataclasses.replace(base, measured_seconds=1.1, pct_of_roof=0.91), path)  # faster: ok
    recs = load(path)
    assert len(recs) == 2
    assert recs[0].artifact == "decode:split-k"
    assert check_regressions(recs, tolerance=0.10) == []

    append(dataclasses.replace(base, measured_seconds=1.5), path)  # slower than best (1.1): flagged
    flagged = check_regressions(load(path), tolerance=0.10)
    assert len(flagged) == 1
    assert flagged[0].artifact == "decode:split-k"
    assert flagged[0].slowdown > 1.0
