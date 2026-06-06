"""KV-cache correctness: incremental cached decoding must equal full recompute.

The make-or-break invariant (docs/design/L2_kv_cache_SPEC.md §4): logit- and token-identical
between the cached path and the recompute oracle, for MHA and GQA, single- and multi-batch.
"""

import torch

from reasoning_llm.model import KVCache, ModelConfig, TransformerLM
from reasoning_llm.sampling import SamplingParams, generate


def _model(n_kv_heads: int = 2) -> TransformerLM:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_layers=3, n_heads=4, n_kv_heads=n_kv_heads, context_length=64
    )
    m = TransformerLM(cfg)
    m.eval()
    return m


def test_prefill_matches_full_forward() -> None:
    model = _model()
    prompt = torch.tensor([[3, 1, 4, 1, 5, 9, 2]])
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        cached_last = model(prompt, cache)[0, -1]
        full_last = model(prompt)[0, -1]
    torch.testing.assert_close(cached_last, full_last)
    assert cache.length == prompt.shape[1]


def test_decode_steps_match_full_recompute() -> None:
    model = _model()
    ids = [3, 1, 4, 1, 5]
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        logits = model(torch.tensor([ids]), cache)[0, -1]  # prefill
        for _ in range(8):
            nxt = int(logits.argmax().item())
            ids.append(nxt)
            full = model(torch.tensor([ids]))[0, -1]  # recompute oracle for the same position
            logits = model(torch.tensor([[nxt]]), cache)[0, -1]  # one cached decode step
            torch.testing.assert_close(logits, full)


def _greedy(model: TransformerLM, prompt: list[int], use_cache: bool) -> list[int]:
    return generate(
        model, prompt, SamplingParams(temperature=0.0, max_tokens=12), use_cache=use_cache
    )


def test_generate_cached_equals_recompute_mha() -> None:
    model = _model(n_kv_heads=4)  # n_kv == n_heads → plain MHA
    prompt = [7, 2, 9, 1]
    assert _greedy(model, prompt, use_cache=True) == _greedy(model, prompt, use_cache=False)


def test_generate_cached_equals_recompute_gqa() -> None:
    model = _model(n_kv_heads=2)  # GQA
    prompt = [7, 2, 9, 1]
    assert _greedy(model, prompt, use_cache=True) == _greedy(model, prompt, use_cache=False)


def test_batched_prefill_matches_full() -> None:
    model = _model()
    prompts = torch.tensor([[3, 1, 4, 1, 5], [9, 2, 6, 5, 3]])
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        cached_last = model(prompts, cache)[:, -1]
        full_last = model(prompts)[:, -1]
    torch.testing.assert_close(cached_last, full_last)


def test_cache_length_advances_once_per_forward() -> None:
    model = _model()
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]), cache)
        assert cache.length == 3
        model(torch.tensor([[4]]), cache)
        assert cache.length == 4
