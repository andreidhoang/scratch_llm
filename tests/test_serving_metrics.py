"""Failing contract for the A1 Rung-0 serving-metrics harness (bench-writer output).

Pins the *definitions* of TTFT / ITL / throughput / goodput with hand-computed numbers — the
Rung-0 understanding to own (A1 §2.3). Implement the bodies in ``serving/metrics.py`` to turn these
green. CPU-only: part of the ``-m "not gpu"`` commit gate.
"""

import math

import pytest

from scratch_llm.serving.metrics import (
    SLO,
    Percentiles,
    RequestRecord,
    ServingReport,
    percentiles,
    request_meets_slo,
    summarize,
)


def _rec(start: float, times: list[float], prompt_len: int = 8) -> RequestRecord:
    return RequestRecord(prompt_len=prompt_len, start_s=start, token_times_s=tuple(times))


def test_record_derived_quantities() -> None:
    # start=0.0; tokens emitted at 0.20, 0.25, 0.30, 0.36
    #   TTFT = 0.20 ; ITLs = [0.05, 0.05, 0.06] ; latency = 0.36 ; output_len = 4
    r = _rec(0.0, [0.20, 0.25, 0.30, 0.36])
    assert r.output_len == 4
    assert math.isclose(r.ttft_s, 0.20, abs_tol=1e-9)
    assert len(r.itls_s) == 3
    expected_itls = [0.05, 0.05, 0.06]
    assert all(
        math.isclose(a, b, abs_tol=1e-9) for a, b in zip(r.itls_s, expected_itls, strict=True)
    )
    assert math.isclose(r.latency_s, 0.36, abs_tol=1e-9)


def test_single_token_request_has_no_itls() -> None:
    r = _rec(1.0, [1.5])
    assert r.output_len == 1
    assert r.itls_s == ()
    assert math.isclose(r.ttft_s, 0.5, abs_tol=1e-9)


def test_percentiles_nearest_rank() -> None:
    # values 1..100, n=100 → idx=round(p·99): p50→50→51, p95→94→95, p99→98→99
    p = percentiles([float(i) for i in range(1, 101)])
    assert isinstance(p, Percentiles)
    assert p.p50 == 51.0
    assert p.p95 == 95.0
    assert p.p99 == 99.0


def test_percentiles_empty_raises() -> None:
    with pytest.raises(ValueError):
        percentiles([])


def test_slo_membership() -> None:
    # R1 fast: ttft 0.1, max ITL 0.1 ; R2 slow first token: ttft 0.5
    r1 = _rec(0.0, [0.1, 0.2, 0.3])
    r2 = _rec(0.0, [0.5, 1.0])
    slo = SLO(ttft_s=0.2, itl_s=0.2)
    assert request_meets_slo(r1, slo) is True
    assert request_meets_slo(r2, slo) is False


def test_summarize_throughput_goodput() -> None:
    r1 = _rec(0.0, [0.1, 0.2, 0.3])  # 3 tokens, meets SLO
    r2 = _rec(0.0, [0.5, 1.0])  # 2 tokens, misses SLO (TTFT 0.5)
    slo = SLO(ttft_s=0.2, itl_s=0.2)

    rep = summarize([r1, r2], slo)
    assert isinstance(rep, ServingReport)
    assert rep.n_requests == 2
    assert rep.total_output_tokens == 5
    assert math.isclose(rep.wall_s, 1.0, abs_tol=1e-9)  # max last-token 1.0 − min start 0.0
    assert math.isclose(rep.throughput_tok_s, 5.0, abs_tol=1e-9)  # 5 tokens / 1.0 s
    assert math.isclose(rep.goodput_tok_s, 3.0, abs_tol=1e-9)  # only R1's 3 tokens meet the SLO
    assert isinstance(rep.ttft_ms, Percentiles)


def test_summarize_empty_raises() -> None:
    with pytest.raises(ValueError):
        summarize([], SLO(ttft_s=1.0, itl_s=1.0))
