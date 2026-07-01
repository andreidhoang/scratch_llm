"""Failing contract for the A1 Rung-0 baseline — the step-timed decoder that emits RequestRecords.

Two gates:
- **token-exactness** (CPU, in the ``-m "not gpu"`` commit gate): the instrumented decode MUST match
  ``sampling.generate`` — Rung 0 instruments the decoder, it does not reimplement it.
- **reproducibility** (GPU): fixed seed + locked clocks → identical tokens and well-formed, stable
  timing on re-run (the only DoD for Rung 0).

Implement ``decode_record`` / ``run_baseline`` in ``serving/baseline.py`` to turn these green.
"""

from itertools import pairwise

import pytest
import torch

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.serving.baseline import DecodeResult, decode_record, run_baseline


def _tiny_model() -> TransformerLM:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, context_length=64)
    return TransformerLM(cfg)


def _greedy(n: int = 12) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=n)


# --- token-exactness: instrument, don't reimplement (CPU, commit gate) ---


def test_decode_record_is_token_exact_vs_generate() -> None:
    model = _tiny_model()
    prompt = [1, 2, 3, 4]
    params = _greedy()
    result = decode_record(model, prompt, params, device="cpu")
    assert isinstance(result, DecodeResult)
    assert result.token_ids == generate(model, prompt, params, device="cpu")


def test_decode_record_trace_is_well_formed() -> None:
    model = _tiny_model()
    result = decode_record(model, [5, 6, 7], _greedy(), device="cpu")
    rec = result.record
    times = rec.token_times_s
    assert len(times) == len(result.token_ids)  # one timestamp per output token
    assert rec.prompt_len == 3
    assert times[0] >= rec.start_s  # first token comes after the request start
    assert all(b >= a for a, b in pairwise(times))  # emit times are non-decreasing


# --- reproducibility: the Rung-0 DoD (GPU) ---


@pytest.mark.gpu
def test_run_baseline_reproducible_fixed_seed() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    model = _tiny_model().to("cuda")
    prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    params = _greedy()

    run1 = run_baseline(model, prompts, params, device="cuda")
    run2 = run_baseline(model, prompts, params, device="cuda")

    # Deterministic content: identical tokens across runs (fixed seed, greedy). The tight
    # reproducibility gate is token determinism; timing is "reproducible within noise" under locked
    # clocks and is asserted here only to be finite, ordered, and one-timestamp-per-token.
    assert len(run1) == len(prompts)
    assert [r.token_ids for r in run1] == [r.token_ids for r in run2]
    for r in run1:
        t = r.record.token_times_s
        assert len(t) == len(r.token_ids) > 0
        assert r.record.start_s <= t[0]
        assert all(b >= a for a, b in pairwise(t))
