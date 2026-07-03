"""A1 Rung 3b — continuous-batching scheduler oracle + utilization instrument (D4–D5).

Scheduling must change WHEN tokens are produced, never WHICH: every request through either policy
(continuous / static-wave) is token-identical to its standalone single-stream greedy decode, across
admission churn, ragged lengths, and multi-wave queues (R3.3 e2e). The utilization instrument is
validated against hand-computed analytics here so the GPU bench can *trust* it to explain the
R3.4 throughput gap. CPU + fixed seed; scheduling claims are asserted on step COUNTS (deterministic),
not wall time.
"""

import pytest
import torch

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.serving.continuous import (
    Request,
    ServeResult,
    serve_continuous,
    serve_static_wave,
)
from scratch_llm.serving.metrics import SLO, summarize


def _model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=64
    )
    return TransformerLM(cfg).eval()


def _assert_matches_single_stream(model: TransformerLM, result: ServeResult) -> None:
    for c in result.completed:
        expected = generate(
            model,
            list(c.request.prompt_ids),
            SamplingParams(temperature=0.0, max_tokens=c.request.max_new_tokens),
            device="cpu",
        )
        assert list(c.token_ids) == expected, (
            f"request {c.request.request_id}: {list(c.token_ids)} != {expected}"
        )


# The churn trace: ragged prompts, ragged budgets, more requests than slots → join/leave mid-run.
_CHURN = [
    Request(0, (1, 2, 3), 4),
    Request(1, (10, 20, 30, 40, 50), 9),
    Request(2, (7, 7), 2),
    Request(3, (5, 4, 3, 2), 6),
    Request(4, (100, 3, 55, 9), 3),
    Request(5, (42,), 5),
]


def test_continuous_matches_single_stream_ragged() -> None:
    """R3.3 (end-to-end): iteration-level scheduling with admission churn is token-exact."""
    model = _model()
    result = serve_continuous(model, _CHURN, n_slots=2, device="cpu")
    assert len(result.completed) == len(_CHURN)
    _assert_matches_single_stream(model, result)


def test_static_wave_matches_single_stream_ragged() -> None:
    """The wave control must be token-exact too — the bench Δ is scheduling, not numerics."""
    model = _model()
    result = serve_static_wave(model, _CHURN, n_slots=2, device="cpu")
    assert len(result.completed) == len(_CHURN)
    _assert_matches_single_stream(model, result)


def test_single_slot_degenerates_to_sequential() -> None:
    model = _model()
    result = serve_continuous(model, _CHURN[:3], n_slots=1, device="cpu")
    _assert_matches_single_stream(model, result)


def test_max_new_tokens_one_is_prefill_only() -> None:
    """A budget-1 request completes at admission — no decode step may over-generate."""
    model = _model()
    reqs = [Request(0, (1, 2, 3), 1), Request(1, (4, 5), 3)]
    result = serve_continuous(model, reqs, n_slots=2, device="cpu")
    assert len(result.completed[0].token_ids) == 1
    _assert_matches_single_stream(model, result)


def test_static_wave_utilization_matches_hand_analytics() -> None:
    """The instrument the GPU bench trusts: wave utilization == the hand-computed series.

    8 slots, 4 requests of budget 2 + 4 of budget 8 (the 16/16-trace shape scaled down): prefill
    emits token 1, so budget-m rows take m−1 decode steps → step 1 has all 8 rows active, steps
    2–7 exactly the 4 long rows → samples [1.0, 0.5×6], mean 4/7. This is the arithmetic whose
    inversion the 2026-07-03 ledger correction fixed — now pinned by a test.
    """
    model = _model()
    reqs = [Request(i, (i + 1,), 2) for i in range(4)] + [
        Request(4 + i, (10 + i,), 8) for i in range(4)
    ]
    result = serve_static_wave(model, reqs, n_slots=8, device="cpu")
    assert result.utilization == (1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert result.n_decode_steps == 7
    assert result.n_prefill_forwards == 1
    _assert_matches_single_stream(model, result)


def test_continuous_refills_beat_wave_on_step_count() -> None:
    """The scheduling claim, asserted deterministically: on a heavy-tail trace with a queue,
    continuous needs fewer lockstep decode steps and holds higher slot utilization than wave."""
    model = _model()
    reqs = [
        Request(0, (1, 2), 2),
        Request(1, (3, 4), 8),
        Request(2, (5, 6), 2),
        Request(3, (7, 8), 8),
    ]
    cont = serve_continuous(model, reqs, n_slots=2, device="cpu")
    wave = serve_static_wave(model, reqs, n_slots=2, device="cpu")
    _assert_matches_single_stream(model, cont)
    _assert_matches_single_stream(model, wave)
    assert cont.n_decode_steps < wave.n_decode_steps, (
        f"continuous {cont.n_decode_steps} steps should beat wave {wave.n_decode_steps}"
    )
    assert cont.mean_utilization > wave.mean_utilization


def test_admission_is_one_batched_prefill_per_event() -> None:
    """All simultaneously-free slots admit through ONE padded prefill forward (the admit tax is
    per event, not per request)."""
    model = _model()
    reqs = [Request(i, (i + 1, i + 2), 3) for i in range(4)]
    cont = serve_continuous(model, reqs, n_slots=4, device="cpu")
    assert cont.n_prefill_forwards == 1
    wave = serve_static_wave(model, [Request(i, (i + 1,), 2) for i in range(8)], n_slots=4)
    assert wave.n_prefill_forwards == 2  # two waves, one batched prefill each


def test_queue_wait_in_steps_continuous_below_wave() -> None:
    """Queued requests start when a slot frees (continuous) vs when the whole wave ends (wave).
    Asserted on ``admitted_at_step`` — deterministic scheduling fact — NOT wall time: a CPU-box
    scheduling hiccup inside one run must not flip the verdict (wall-time magnitude is the GPU
    bench's R3.6)."""
    model = _model()
    reqs = [
        Request(0, (1, 2), 3),
        Request(1, (3, 4), 8),
        Request(2, (5, 6), 3),
        Request(3, (7, 8), 8),
    ]
    cont = serve_continuous(model, reqs, n_slots=2, device="cpu")
    wave = serve_static_wave(model, reqs, n_slots=2, device="cpu")
    cont_wait = {c.request.request_id: c.admitted_at_step for c in cont.completed}
    wave_wait = {c.request.request_id: c.admitted_at_step for c in wave.completed}
    assert cont_wait[0] == wave_wait[0] == 0  # first wave admits at t0 either way
    assert cont_wait[1] == wave_wait[1] == 0
    for queued in (2, 3):  # the queued requests: slot-free admission strictly beats wave-end
        assert cont_wait[queued] < wave_wait[queued], (
            f"request {queued}: continuous admitted at step {cont_wait[queued]}, "
            f"wave at {wave_wait[queued]}"
        )


def test_records_are_wellformed_for_metrics() -> None:
    model = _model()
    result = serve_continuous(model, _CHURN, n_slots=3, device="cpu")
    for c in result.completed:
        r = c.record
        assert r.prompt_len == len(c.request.prompt_ids)
        assert r.output_len == c.request.max_new_tokens == len(c.token_ids)
        assert len(r.itls_s) == r.output_len - 1
        assert r.ttft_s > 0.0
        assert all(itl >= 0.0 for itl in r.itls_s)
    report = summarize([c.record for c in result.completed], SLO(ttft_s=10.0, itl_s=10.0))
    assert report.n_requests == len(_CHURN)
    assert report.total_output_tokens == sum(r.max_new_tokens for r in _CHURN)
    assert report.throughput_tok_s > 0.0


def test_validation_errors() -> None:
    model = _model()
    with pytest.raises(ValueError, match="non-empty"):
        serve_continuous(model, [], n_slots=2)
    with pytest.raises(ValueError, match="unique"):
        serve_continuous(model, [Request(0, (1,), 2), Request(0, (2,), 2)], n_slots=2)
    with pytest.raises(ValueError, match="context_length"):
        serve_continuous(model, [Request(0, (1,) * 60, 10)], n_slots=2)
    with pytest.raises(ValueError, match="≥ 1"):
        serve_continuous(model, [Request(0, (1,), 2)], n_slots=0)
