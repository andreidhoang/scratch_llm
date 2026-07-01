"""Serving metrics harness — TTFT, ITL, throughput, and goodput-under-SLO.

A1 Rung 0. The reusable measurement spine for every later rung (batching, paged KV, spec decode).
The four numbers that decide whether a serving optimization shipped (A1 §2.3):

- **TTFT** (time-to-first-token): prefill-gated responsiveness — ``token_times_s[0] − start_s``.
- **ITL / TPOT** (inter-token latency): decode-gated smoothness — successive gaps between output
  tokens.
- **Throughput**: aggregate output tok/s across all requests in a run.
- **Goodput**: the throughput that *meets an SLO* (DistServe). Raw throughput with missed SLOs is
  the number that loses customers; goodput is the one that pays.

Request latency ≈ ``TTFT + ITL × num_output_tokens`` — the identity that says which optimization
(prefill-side vs decode-side) moves which number.

**Mode-3 boundary.** This file encodes the *definitions* of the serving metrics — owning those *is*
the Rung-0 understanding. The bodies are left unimplemented on purpose; the executable spec is
``tests/test_serving_metrics.py`` (worked numbers, CPU-only, in the commit gate). Implement to green,
then teach the definitions back before wiring the GPU baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestRecord:
    """The timing trace of one decoded request. ``token_times_s[i]`` is the wall-clock time
    (seconds, same monotonic clock as ``start_s``) at which output token ``i`` was emitted;
    ``token_times_s[0]`` is the first output token. ``prompt_len`` is carried for prefill accounting.
    """

    prompt_len: int
    start_s: float
    token_times_s: tuple[float, ...]

    @property
    def output_len(self) -> int:
        """Number of output tokens emitted (``len(token_times_s)``)."""
        return len(self.token_times_s)

    @property
    def ttft_s(self) -> float:
        """Time to first token: ``token_times_s[0] − start_s``."""
        return self.token_times_s[0] - self.start_s

    @property
    def itls_s(self) -> tuple[float, ...]:
        """Inter-token latencies: successive differences of ``token_times_s`` (length
        ``output_len − 1``; empty when only one token was emitted)."""
        return tuple(
            self.token_times_s[i] - self.token_times_s[i - 1]
            for i in range(1, len(self.token_times_s))
        )

    @property
    def latency_s(self) -> float:
        """End-to-end request latency: ``token_times_s[-1] − start_s``."""
        return self.token_times_s[-1] - self.start_s


@dataclass(frozen=True)
class SLO:
    """A service-level objective. A request *meets* it iff ``ttft_s ≤ ttft_s`` **and** its worst
    inter-token gap ``max(itls_s) ≤ itl_s`` (a single-token request meets the ITL bound vacuously)."""

    ttft_s: float
    itl_s: float


@dataclass(frozen=True)
class Percentiles:
    """p50/p95/p99 of a distribution, by nearest-rank on the sorted values (the convention already
    used in ``scratch_llm.bench.harness``: ``idx = round(p·(n−1))``)."""

    p50: float
    p95: float
    p99: float


@dataclass(frozen=True)
class ServingReport:
    """The aggregate verdict over a set of requests, for one run. TTFT/ITL are reported in ms."""

    n_requests: int
    total_output_tokens: int
    wall_s: float
    throughput_tok_s: float
    goodput_tok_s: float
    ttft_ms: Percentiles
    itl_ms: Percentiles


def percentiles(values: Sequence[float]) -> Percentiles:
    """p50/p95/p99 of ``values`` by nearest-rank (see :class:`Percentiles`). Raises ``ValueError`` on
    empty input."""
    if not values:
        raise ValueError("Cannot calculate percentiles for empty sequence")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx_p50 = round(0.50 * (n - 1))
    idx_p95 = round(0.95 * (n - 1))
    idx_p99 = round(0.99 * (n - 1))
    return Percentiles(
        p50=sorted_vals[idx_p50],
        p95=sorted_vals[idx_p95],
        p99=sorted_vals[idx_p99],
    )


def request_meets_slo(record: RequestRecord, slo: SLO) -> bool:
    """Whether ``record`` satisfies both bounds of ``slo`` (see :class:`SLO`)."""
    has_bad_itl = len(record.itls_s) > 0 and max(record.itls_s) > slo.itl_s
    return record.ttft_s <= slo.ttft_s and not has_bad_itl


def summarize(records: Sequence[RequestRecord], slo: SLO) -> ServingReport:
    """Aggregate per-request traces into a :class:`ServingReport`.

    - ``wall_s = max(token_times_s[-1]) − min(start_s)`` over all records (the run's wall span).
    - ``throughput_tok_s = Σ output_len / wall_s``.
    - ``goodput_tok_s = Σ output_len over SLO-meeting requests / wall_s``.
    - ``ttft_ms``: percentiles across requests of per-request ``ttft_s`` (× 1000).
    - ``itl_ms``: percentiles across the *pool* of all inter-token gaps from all requests (× 1000).

    Raises ``ValueError`` on empty ``records``.
    """
    if not records:
        raise ValueError("Cannot summarize empty records sequence")
    max_end = max(r.token_times_s[-1] for r in records)
    min_start = min(r.start_s for r in records)
    wall_s = max_end - min_start

    total_output_tokens = sum(r.output_len for r in records)
    throughput_tok_s = total_output_tokens / wall_s

    good_output_tokens = sum(r.output_len for r in records if request_meets_slo(r, slo))
    goodput_tok_s = good_output_tokens / wall_s

    ttfts_ms = [r.ttft_s * 1000.0 for r in records]
    ttft_ms = percentiles(ttfts_ms)

    all_itls_ms = []
    for r in records:
        for itl in r.itls_s:
            all_itls_ms.append(itl * 1000.0)

    itl_ms = Percentiles(0.0, 0.0, 0.0) if not all_itls_ms else percentiles(all_itls_ms)

    return ServingReport(
        n_requests=len(records),
        total_output_tokens=total_output_tokens,
        wall_s=wall_s,
        throughput_tok_s=throughput_tok_s,
        goodput_tok_s=goodput_tok_s,
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
    )
