"""Tests for decoding: greedy/argmax, top-p nucleus, stop tokens, and the token budget."""

import torch

from reasoning_llm.model import ModelConfig, TransformerLM
from reasoning_llm.sampling import SamplingParams, _top_p_filter, generate


def _tiny_model() -> TransformerLM:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, context_length=16)
    return TransformerLM(cfg)


def test_top_p_one_is_identity() -> None:
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    torch.testing.assert_close(_top_p_filter(probs, 1.0), probs)


def test_top_p_keeps_nucleus_and_renormalizes() -> None:
    probs = torch.tensor([0.6, 0.3, 0.08, 0.02])
    out = _top_p_filter(probs, top_p=0.85)
    # cum=[0.6,0.9,...]; nucleus keeps {0.6,0.3}, tail zeroed and renormalized.
    assert out[2].item() == 0.0 and out[3].item() == 0.0
    torch.testing.assert_close(out.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(out[0] / out[1], torch.tensor(2.0), atol=1e-5, rtol=0)


def test_greedy_equals_argmax_and_is_deterministic() -> None:
    model = _tiny_model()
    prompt = [1, 2, 3]
    params = SamplingParams(temperature=0.0, max_tokens=1)

    with torch.no_grad():
        x = torch.tensor([prompt])
        expected = int(model(x)[0, -1].argmax().item())

    out1 = generate(model, prompt, params)
    out2 = generate(model, prompt, params)
    assert out1 == [expected]
    assert out1 == out2  # greedy is deterministic


def test_respects_max_tokens_without_stop() -> None:
    model = _tiny_model()
    out = generate(model, [1, 2], SamplingParams(temperature=0.0, max_tokens=5))
    assert len(out) == 5


def test_stop_token_halts_immediately() -> None:
    model = _tiny_model()
    prompt = [1, 2, 3]
    # First, learn what greedy emits first.
    first = generate(model, prompt, SamplingParams(temperature=0.0, max_tokens=4))[0]
    # Now make that token a stop → generation should yield exactly [first].
    stopped = generate(
        model, prompt, SamplingParams(temperature=0.0, max_tokens=4, stop_ids=(first,))
    )
    assert stopped == [first]


def test_sampling_with_seed_is_reproducible() -> None:
    model = _tiny_model()
    params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=8, seed=123)
    assert generate(model, [1, 2, 3], params) == generate(model, [1, 2, 3], params)


def test_empty_prompt_rejected() -> None:
    model = _tiny_model()
    try:
        generate(model, [], SamplingParams())
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty prompt")
