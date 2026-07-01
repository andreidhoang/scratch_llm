"""A1 Rung 3a — static batched decode oracle (weight amortization must not change outputs).

The batched decode reads each weight once and applies it to all B rows; correctness demands that this
is *only* a throughput change — row ``b`` must be token-identical to a single-stream greedy decode of
prompt ``b`` (batch independence: no row leaks into another). This is the R3.3 gate before any
aggregate-throughput number is trusted. CPU + fixed seed.
"""

import pytest
import torch

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.serving.batched import batched_greedy_decode


def _model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=64
    )
    return TransformerLM(cfg).eval()


def test_batched_row_matches_single_stream() -> None:
    """R3.3 oracle: batched greedy row b == single-stream greedy decode(prompt b), token-exact."""
    model = _model()
    prompts = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [7, 7, 7, 7, 7], [100, 3, 55, 9, 2]]  # len 5
    n = 8
    batched = batched_greedy_decode(model, prompts, n, device="cpu")
    for b, p in enumerate(prompts):
        single = generate(model, p, SamplingParams(temperature=0.0, max_tokens=n), device="cpu")
        assert batched[b] == single, f"row {b}: {batched[b]} != {single}"


def test_batch_of_one_equals_single_stream() -> None:
    """Degenerate B=1: batched decode must equal the plain sampler exactly."""
    model = _model()
    p = [3, 1, 4, 1, 5, 9]
    batched = batched_greedy_decode(model, [p], 6, device="cpu")[0]
    single = generate(model, p, SamplingParams(temperature=0.0, max_tokens=6), device="cpu")
    assert batched == single


def test_requires_equal_length_prompts() -> None:
    model = _model()
    with pytest.raises(ValueError, match="equal-length"):
        batched_greedy_decode(model, [[1, 2, 3], [4, 5]], 4, device="cpu")


def test_empty_prompts_raise() -> None:
    model = _model()
    with pytest.raises(ValueError, match="non-empty"):
        batched_greedy_decode(model, [], 4, device="cpu")
