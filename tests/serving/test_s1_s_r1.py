"""S1/S-R1 — the instrument's own correctness gate, on CPU.

The rung's claim is a *ratio* between two engines. A ratio is only a measurement if the two arms saw
the same load and their latencies were read off the same timestamps, so those two things — not the
throughput — are what this file tests. Everything here is CPU-only and deterministic; the parts that
need an H100 skip loudly with the exact command to run on the box.

Reference oracle: ``numpy``. The arrival process is checked against ``np.random`` (the same legacy
generator ``vllm bench serve`` draws from, serve.py:2018) and against the analytic Exponential CDF;
percentiles are checked against ``np.percentile``, which is the function ``vllm bench serve`` reports
(serve.py:747, :759). Where a tolerance is a *design* choice rather than a machine-precision fact it
is a named module constant with the argument written beside it.
"""

from __future__ import annotations

import math
import os
import shutil

import numpy as np
import pytest

from scratch_llm.serving.loadgen import (
    ArmRun,
    RequestOutcome,
    RequestSpec,
    RequestStream,
    ScratchEngineAdapter,
    build_stream,
    inter_arrivals,
    percentile,
    poisson_arrivals,
    qwen3_8b_shape,
    summarize,
)

# ---------------------------------------------------------------------------------------------
# tolerances — statistical, so stated with their argument, not tuned until green
# ---------------------------------------------------------------------------------------------

#: Sample size for the distributional checks. Large enough that the KS band below is tight
#: (1.36/√20000 ≈ 0.0096) and small enough to draw in well under a second.
N_SAMPLES = 20_000

#: Kolmogorov-Smirnov critical value at α = 0.05 is 1.36/√n for a fully specified null. The null
#: here IS fully specified (Exponential with mean 1/rate — no parameter is estimated from the
#: sample), so the asymptotic constant applies unmodified.
KS_ALPHA_CONST = 1.36

#: Relative band on the sample mean of n exponential draws. The standard error of the mean is
#: (1/rate)/√n, i.e. 1/√n relative; 5/√n is a 5σ band — a test that fails once in 3.5 million runs
#: rather than one that needs a seed to stay green.
MEAN_REL_BAND = 5.0 / math.sqrt(N_SAMPLES)

#: Same 5σ argument for the coefficient of variation, whose population value is exactly 1 for an
#: Exponential and whose sampling standard error is ≈ √5/√n.
CV_REL_BAND = 5.0 * math.sqrt(5.0) / math.sqrt(N_SAMPLES)


# ---------------------------------------------------------------------------------------------
# 1. the arrival process — mean rate, Poisson shape, and vLLM's normalization
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [4.0, 8.0, 16.0])
def test_inter_arrivals_have_the_requested_mean_rate(rate: float) -> None:
    """The raw gaps average 1/rate. This is the property "Poisson at r rps" actually claims."""
    gaps = np.asarray(inter_arrivals(N_SAMPLES, rate, seed=0))
    assert gaps.mean() == pytest.approx(1.0 / rate, rel=MEAN_REL_BAND)


@pytest.mark.parametrize("rate", [4.0, 16.0])
def test_inter_arrivals_are_exponential_not_merely_random(rate: float) -> None:
    """Poisson-shaped, three ways: CV = 1, a KS test against the Exponential CDF, and a memoryless
    check. Mean alone does not distinguish a Poisson process from a uniform-jitter one; CV does
    (uniform on [0, 2/rate] has the same mean and CV = 1/√3), and KS pins the whole shape."""
    gaps = np.sort(np.asarray(inter_arrivals(N_SAMPLES, rate, seed=1)))
    assert gaps.std(ddof=1) / gaps.mean() == pytest.approx(1.0, rel=CV_REL_BAND)

    cdf = 1.0 - np.exp(-rate * gaps)
    n = gaps.size
    d_plus = np.max(np.arange(1, n + 1) / n - cdf)
    d_minus = np.max(cdf - np.arange(0, n) / n)
    assert max(d_plus, d_minus) < KS_ALPHA_CONST / math.sqrt(n)

    # Memorylessness: P(X > 2m | X > m) == P(X > m). True of the Exponential, false of e.g. Gamma(2).
    m = 1.0 / rate
    tail = gaps[gaps > m]
    assert (tail > 2 * m).mean() == pytest.approx(math.exp(-1.0), abs=4.0 / math.sqrt(tail.size))


def test_arrival_draw_reproduces_vllms_own_generator() -> None:
    """Element-for-element identity with ``vllm/benchmarks/serve.py:466-470``, which draws
    ``np.random.gamma(shape=burstiness, scale=1/(rate*burstiness))`` from the *legacy global*
    RandomState seeded at serve.py:2018. This is the test that makes "the same arrival process"
    a fact rather than a claim about two implementations of the same words."""
    rate, seed, n = 8.0, 1234, 64
    expected = np.random.RandomState(seed).gamma(shape=1.0, scale=1.0 / (rate * 1.0), size=n)
    assert np.asarray(inter_arrivals(n, rate, seed)) == pytest.approx(expected, rel=0, abs=0)


def test_normalization_pins_the_last_arrival_exactly_as_vllm_does() -> None:
    """serve.py:480-489 rescales the cumulative offsets so the last one lands at n/rate — which is
    why the *normalized* stream is not a Poisson process and must not be tested as one."""
    n, rate = 500, 16.0
    normalized = poisson_arrivals(n, rate, seed=7)
    assert normalized[-1] == pytest.approx(n / rate, rel=1e-12)
    raw = poisson_arrivals(n, rate, seed=7, normalize=False)
    assert raw[-1] != pytest.approx(n / rate, rel=1e-12)  # the 1-2% gap upstream complains about
    factor = (n / rate) / raw[-1]
    assert np.asarray(normalized) == pytest.approx(np.asarray(raw) * factor)
    assert list(normalized) == sorted(normalized)  # arrivals are monotone


def test_arrivals_are_deterministic_in_the_seed() -> None:
    assert poisson_arrivals(64, 8.0, seed=3) == poisson_arrivals(64, 8.0, seed=3)
    assert poisson_arrivals(64, 8.0, seed=3) != poisson_arrivals(64, 8.0, seed=4)


def test_arrival_rate_must_be_finite_and_positive() -> None:
    with pytest.raises(ValueError):
        poisson_arrivals(10, float("inf"), seed=0)  # rate=inf is closed loop, a different rung
    with pytest.raises(ValueError):
        poisson_arrivals(10, 0.0, seed=0)


# ---------------------------------------------------------------------------------------------
# 2. TTFT / ITL come off the right timestamps
# ---------------------------------------------------------------------------------------------


def _outcome(sent: float, first: float, gaps: list[float]) -> RequestOutcome:
    times = [first]
    for g in gaps:
        times.append(times[-1] + g)
    return RequestOutcome(
        request_id=0,
        prompt_len=1024,
        sent_s=sent,
        chunk_times_s=tuple(times),
        output_tokens=len(times),
    )


def test_ttft_includes_queueing_and_itl_does_not() -> None:
    """The whole point of the instrument. A request submitted at t=10 that waits 3 s behind other
    requests, prefills for 0.4 s, then decodes at 20 ms/token must report TTFT = 3.4 s and ITLs of
    20 ms — not TTFT = 0.4 s (queue hidden) and not a first ITL of 3.4 s (queue smeared into decode).
    Mirrors endpoint_request_func.py:238 (TTFT from ``st``) and :243 (ITL from the previous chunk)."""
    o = _outcome(sent=10.0, first=10.0 + 3.0 + 0.4, gaps=[0.020] * 5)
    assert o.ttft_s == pytest.approx(3.4)
    assert o.itls_s == pytest.approx((0.020,) * 5)
    assert max(o.itls_s) < o.ttft_s  # queueing lives in TTFT alone
    assert len(o.itls_s) == o.output_tokens - 1  # the first token yields no ITL (:236-243)
    assert o.latency_s == pytest.approx(3.5)
    assert o.tpot_s == pytest.approx((3.5 - 3.4) / 5)


def test_ttft_is_measured_from_this_requests_own_arrival_not_the_runs_start() -> None:
    """Two requests, same absolute first-token time, different arrivals: the later one must report
    the smaller TTFT. A generator that measured from a single run-wide t0 would report them equal —
    and would over-report TTFT by the whole arrival offset at 16 rps × 500 prompts (≈ 31 s)."""
    early = _outcome(sent=0.0, first=1.0, gaps=[0.01])
    late = _outcome(sent=0.9, first=1.0, gaps=[0.01])
    assert early.ttft_s == pytest.approx(1.0)
    assert late.ttft_s == pytest.approx(0.1)


def test_tpot_divides_by_tokens_not_chunks() -> None:
    """vLLM's TPOT uses the tokenizer's output_len (serve.py:616-617) while its ITL list is per
    *chunk* (:243). When a chunk bundles two tokens the two numbers legitimately differ, and the
    report must not quietly conflate them."""
    bundled = RequestOutcome(
        request_id=1, prompt_len=8, sent_s=0.0, chunk_times_s=(1.0, 1.1, 1.2), output_tokens=5
    )
    assert bundled.itls_s == pytest.approx((0.1, 0.1))
    assert bundled.tpot_s == pytest.approx(0.2 / 4)  # (latency − ttft) / (tokens − 1)


# ---------------------------------------------------------------------------------------------
# 3. percentiles are exact on a known sample, and match the floor's definition
# ---------------------------------------------------------------------------------------------


def test_percentiles_are_exact_on_a_hand_computed_sample() -> None:
    """n = 10, values 1..10. p50 → k = 9·0.5 = 4.5 → 5 + 0.5·(6−5) = 5.5. p99 → k = 8.91 →
    9 + 0.91·(10−9) = 9.91. Nearest-rank (``metrics.percentiles``) would answer 6 and 10: the
    sample is chosen so the two conventions visibly disagree."""
    vals = [float(i) for i in range(1, 11)]
    assert percentile(vals, 50.0) == pytest.approx(5.5)
    assert percentile(vals, 99.0) == pytest.approx(9.91)
    assert percentile(vals, 0.0) == 1.0
    assert percentile(vals, 100.0) == 10.0
    assert percentile([42.0], 99.0) == 42.0


@pytest.mark.parametrize("n", [3, 7, 10, 51, 200])
@pytest.mark.parametrize("p", [0.0, 25.0, 50.0, 90.0, 99.0, 100.0])
def test_percentiles_agree_with_numpy_which_is_what_the_floor_reports(n: int, p: float) -> None:
    """``vllm bench serve`` reports ``np.percentile`` (serve.py:747 TTFT, :759 ITL) — default
    method, linear interpolation. Ours must be that function, not a near neighbour."""
    vals = list(np.random.default_rng(n).random(n) * 100.0)
    assert percentile(vals, p) == pytest.approx(float(np.percentile(vals, p)), rel=1e-12)


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        percentile([], 50.0)
    with pytest.raises(ValueError):
        percentile([1.0], 101.0)


# ---------------------------------------------------------------------------------------------
# 4. both adapters are fed the identical stream
# ---------------------------------------------------------------------------------------------


class RecordingAdapter:
    """A stand-in engine that decodes nothing and only writes down what it was asked to serve."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.seen: list[tuple[int, tuple[int, ...], int, float]] = []

    def run(self, stream: RequestStream) -> ArmRun:
        self.seen = [
            (s.request_id, s.prompt_ids, s.max_new_tokens, s.arrival_s) for s in stream.specs
        ]
        outcomes = tuple(
            RequestOutcome(
                request_id=s.request_id,
                prompt_len=s.prompt_len,
                sent_s=s.arrival_s,
                chunk_times_s=tuple(s.arrival_s + 0.1 + 0.01 * i for i in range(s.max_new_tokens)),
                output_tokens=s.max_new_tokens,
            )
            for s in stream.specs
        )
        return ArmRun(outcomes=outcomes, epoch_s=0.0, end_s=stream.specs[-1].arrival_s + 10.0)


def test_the_two_adapters_receive_byte_identical_streams() -> None:
    """One stream object, two adapters, one digest. The digest is what the bench prints per arm and
    refuses to divide on when it differs (bench/s1_serve_vs_vllm.py, ``stream_digests_match``)."""
    stream = build_stream(40, 8.0, seed=11, input_len=32, output_len=6)
    a, b = RecordingAdapter("scratch_llm"), RecordingAdapter("vllm")
    ra, rb = a.run(stream), b.run(stream)
    assert a.seen == b.seen
    assert len(a.seen) == 40
    ka = summarize("a", stream, ra.outcomes, epoch_s=ra.epoch_s, end_s=ra.end_s)
    kb = summarize("b", stream, rb.outcomes, epoch_s=rb.epoch_s, end_s=rb.end_s)
    assert ka.stream_digest == kb.stream_digest


def test_digest_notices_a_perturbed_stream() -> None:
    """A digest that hashed only the count would pass the test above while both arms ran different
    prompts. Each field that changes an engine's work must move it."""
    stream = build_stream(8, 8.0, seed=2, input_len=16, output_len=4)
    base = stream.digest()
    for mutated in (
        RequestSpec(0, (1, 2, 3), 4, 0.0),  # different prompt
        RequestSpec(stream.specs[0].request_id, stream.specs[0].prompt_ids, 5, 0.0),  # budget
        RequestSpec(
            stream.specs[0].request_id,
            stream.specs[0].prompt_ids,
            stream.specs[0].max_new_tokens,
            stream.specs[0].arrival_s + 0.001,  # arrival
        ),
    ):
        perturbed = RequestStream(
            specs=(mutated, *stream.specs[1:]),
            request_rate=stream.request_rate,
            seed=stream.seed,
            burstiness=stream.burstiness,
            normalized=stream.normalized,
        )
        assert perturbed.digest() != base


def test_stream_is_reproducible_and_matches_the_floors_dataset_shape() -> None:
    """``--random-range-ratio 0.0`` (upstream default, datasets/datasets.py:575) means every request
    is exactly input_len in and output_len out — the property that makes the two arms' work equal."""
    s1 = build_stream(25, 4.0, seed=5, input_len=128, output_len=16)
    s2 = build_stream(25, 4.0, seed=5, input_len=128, output_len=16)
    assert s1.digest() == s2.digest()
    assert {sp.prompt_len for sp in s1.specs} == {128}
    assert {sp.max_new_tokens for sp in s1.specs} == {16}
    assert s1.total_prompt_tokens == 25 * 128


# ---------------------------------------------------------------------------------------------
# 5. the aggregate matches vLLM's calculate_metrics
# ---------------------------------------------------------------------------------------------


def test_summarize_uses_vllms_duration_and_throughput_definitions() -> None:
    """``duration = end − epoch`` with epoch taken *before* the first arrival (serve.py:1012, :1091),
    ``output_throughput = Σ output_tokens / duration`` (:740), and mean concurrency by Little's law
    — at a fixed open-loop rate the batch size is an outcome, so "B=32" has to be measured."""
    stream = build_stream(2, 4.0, seed=0, input_len=4, output_len=3)
    outcomes = (
        RequestOutcome(0, 4, 0.0, (1.0, 1.5, 2.0), 3),  # latency 2.0
        RequestOutcome(1, 4, 1.0, (2.0, 2.5, 3.0), 3),  # latency 2.0
    )
    rep = summarize("x", stream, outcomes, epoch_s=0.0, end_s=4.0)
    assert rep.duration_s == 4.0
    assert rep.total_output_tokens == 6
    assert rep.output_throughput == pytest.approx(6 / 4.0)
    assert rep.request_throughput == pytest.approx(2 / 4.0)
    assert rep.mean_concurrency == pytest.approx((2.0 + 2.0) / 4.0)
    assert rep.ttft_ms["p50"] == pytest.approx(1000.0)
    assert rep.itl_ms["p50"] == pytest.approx(500.0)
    assert rep.completed == 2 and rep.failed == 0


def test_failed_requests_are_excluded_but_counted() -> None:
    stream = build_stream(2, 4.0, seed=0, input_len=4, output_len=3)
    outcomes = (
        RequestOutcome(0, 4, 0.0, (1.0, 1.5), 2),
        RequestOutcome(1, 4, 1.0, (), 0, success=False, error="HTTP 500"),
    )
    rep = summarize("x", stream, outcomes, epoch_s=0.0, end_s=4.0)
    assert (rep.completed, rep.failed) == (1, 1)
    assert rep.total_output_tokens == 2


# ---------------------------------------------------------------------------------------------
# 6. the open-loop wiring, end to end, on a toy model with a fake clock
# ---------------------------------------------------------------------------------------------


class FakeClock:
    """A deterministic clock. ``sleep`` advances it; every read advances it by one tick, so the
    engine's own loop cannot stall time. Without this the arrival gate could only be tested by
    sleeping in real seconds, which is a flaky test pretending to be an integration test."""

    def __init__(self, tick: float = 1e-4) -> None:
        self.t = 0.0
        self.tick = tick

    def __call__(self) -> float:
        self.t += self.tick
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(0.0, s)


def _toy_model():
    import torch

    from scratch_llm.model import ModelConfig, TransformerLM

    cfg = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
        context_length=64,
    )
    torch.manual_seed(0)
    return TransformerLM(cfg).eval()


def test_open_loop_stamps_each_request_with_its_own_arrival() -> None:
    """The wiring change in ``serving/continuous.py`` (``arrivals_s``) end to end: a request that
    arrives late must carry ``start_s = t0 + its own arrival`` — the single timestamp the whole
    TTFT definition rests on. Also asserts the tokens are unchanged, i.e. the arrival gate moved
    *when* work happens and not *what* the engine computed."""
    pytest.importorskip("torch")
    from scratch_llm.serving.continuous import Request, serve_continuous

    model = _toy_model()
    specs = [
        RequestSpec(0, (1, 2, 3), 4, 0.0),
        RequestSpec(1, (4, 5, 6), 4, 5.0),
        RequestSpec(2, (7, 8, 9), 4, 12.0),
    ]
    stream = RequestStream(specs=tuple(specs), request_rate=1.0, seed=0)
    clock = FakeClock()
    adapter = ScratchEngineAdapter(model, n_slots=2, device="cpu", clock=clock, sleep=clock.sleep)
    run = adapter.run(stream)
    t0 = run.outcomes[0].sent_s - specs[0].arrival_s
    for spec, outcome in zip(specs, run.outcomes, strict=True):
        assert outcome.sent_s == pytest.approx(t0 + spec.arrival_s)
        assert outcome.ttft_s > 0.0
        assert outcome.output_tokens == spec.max_new_tokens

    closed = serve_continuous(
        model,
        [Request(s.request_id, s.prompt_ids, s.max_new_tokens) for s in specs],
        2,
        "cpu",
    )
    assert [c.token_ids for c in closed.completed] == [
        c.token_ids
        for c in serve_continuous(
            model,
            [Request(s.request_id, s.prompt_ids, s.max_new_tokens) for s in specs],
            2,
            "cpu",
            arrivals_s=[s.arrival_s for s in specs],
            clock=FakeClock(),
            sleep=lambda _s: None,
        ).completed
    ], "the arrival gate must change scheduling, never the tokens"


def test_arrivals_s_is_validated() -> None:
    pytest.importorskip("torch")
    from scratch_llm.serving.continuous import Request, serve_continuous

    model = _toy_model()
    reqs = [Request(0, (1, 2), 2), Request(1, (3, 4), 2)]
    with pytest.raises(ValueError, match="one offset per request"):
        serve_continuous(model, reqs, 2, "cpu", arrivals_s=[0.0])
    with pytest.raises(ValueError, match="≥ 0"):
        serve_continuous(model, reqs, 2, "cpu", arrivals_s=[0.0, -1.0])


def test_closed_loop_path_still_starts_every_request_at_t0() -> None:
    """``arrivals_s=None`` must leave every existing S1/A1 result reproducible — the closed batch
    is the old behaviour and every request keeps ``start_s = t0`` (token equality with the
    open-loop path is asserted in the test above)."""
    pytest.importorskip("torch")
    from scratch_llm.serving.continuous import Request, serve_continuous

    model = _toy_model()
    reqs = [Request(i, (1 + i, 2, 3), 3) for i in range(3)]
    res = serve_continuous(model, reqs, 2, "cpu")
    starts = {c.record.start_s for c in res.completed}
    assert len(starts) == 1, "closed loop: every request starts at t0"


# ---------------------------------------------------------------------------------------------
# 7. what only an H100 can answer — skips loudly, never passes quietly
# ---------------------------------------------------------------------------------------------

_BOX_CMD = (
    "on the box:  experiments/S1/S-R1/floor.sh   then   experiments/S1/S-R1/run.sh\n"
    "  (needs 1×H100, `pip install vllm`, and an HF cache for Qwen/Qwen3-8B's config+tokenizer)"
)


def test_qwen3_8b_shape_is_the_published_geometry() -> None:
    """Shape parity is the precondition for the whole comparison: same FLOPs/token, same KV
    bytes/token, same LM-head width. These are the numbers from Qwen3-8B's config, asserted here
    so a typo cannot silently make our arm a cheaper model than the floor's."""
    cfg = qwen3_8b_shape(1160)
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.kv_heads) == (36, 4096, 32, 8)
    assert (cfg.head_dim, cfg.ffn_dim, cfg.vocab_size) == (128, 12288, 151936)
    assert cfg.qk_norm is True and cfg.rope_theta == 1_000_000.0
    # KV bytes per token, bf16: n_layers · kv_heads · head_dim · 2 (K,V) · 2 B = 147456 B/token.
    assert cfg.n_layers * cfg.kv_heads * cfg.head_dim * 2 * 2 == 147_456


@pytest.mark.gpu
def test_qwen3_8b_shape_matches_the_hf_config_on_a_box_that_has_it() -> None:
    """The same geometry, cross-checked against the config vLLM itself will load."""
    if shutil.which("vllm") is None or os.environ.get("HF_HUB_OFFLINE") == "1":
        pytest.skip(f"needs the HF config for Qwen/Qwen3-8B — {_BOX_CMD}")
    cfg = qwen3_8b_shape(1160)
    from transformers import AutoConfig  # type: ignore[import-not-found]

    hf = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
    assert (hf.num_hidden_layers, hf.hidden_size) == (cfg.n_layers, cfg.d_model)
    assert (hf.num_attention_heads, hf.num_key_value_heads) == (cfg.n_heads, cfg.kv_heads)
    assert (hf.head_dim, hf.intermediate_size, hf.vocab_size) == (
        cfg.head_dim,
        cfg.ffn_dim,
        cfg.vocab_size,
    )


@pytest.mark.gpu
def test_our_generator_and_vllm_bench_serve_agree_on_the_same_server() -> None:
    """The validity check for the whole rung: vLLM measured by OUR client and vLLM measured by
    ``vllm bench serve`` must land on the same throughput. If they do not, the head-to-head is
    comparing clients, not engines. Nothing on a CPU box can answer this."""
    pytest.skip(
        "requires 1×H100 with vLLM serving Qwen3-8B — run both arms and diff output_throughput:\n"
        "  python bench/s1_serve_vs_vllm.py --arm vllm  --rps 16 --headline-rps 16\n"
        "  python bench/s1_serve_vs_vllm.py --arm floor --rps 16 --headline-rps 16\n"
        f"  {_BOX_CMD}"
    )
