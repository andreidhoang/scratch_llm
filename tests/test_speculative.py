"""A1 Rung 4.3 — speculative decoding oracle (D1/D6).

Losslessness is the whole point: with a greedy target, speculative decode must be token-IDENTICAL to
plain greedy `sampling.generate`, for ANY drafter and any K — the drafter changes only WHICH tokens
are guessed (hence speed), never the committed output. The correctness gate is asserted in float64
(reduction-order-exact, hardware-independent); float32 is checked for high agreement (batched-vs-
sequential fp tie-flips are rare, like any batched inference). Also pins: rejected drafts are rolled
back cleanly (a deliberately-wrong drafter still yields the greedy output), the no-draft round is
plain decode, and the `KVCache.truncate` rollback primitive is bit-identical.
"""

import random

import pytest
import torch

from scratch_llm.model import KVCache, ModelConfig, TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.serving.speculative import (
    Drafter,
    ModelDrafter,
    NGramDrafter,
    SpecStats,
    speculative_generate,
)


def _model(seed: int = 0, dtype: torch.dtype = torch.float32) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=512, d_model=48, n_layers=3, n_heads=4, n_kv_heads=2, context_length=128
    )
    return TransformerLM(cfg).eval().to(dtype)


class _WrongDrafter:
    """Always proposes tokens the target will reject (id 0, or 1 if the target argmax is 0) — exercises
    the rejection + KV-rollback path on every round."""

    def propose(self, context_ids, k: int) -> list[int]:
        return [0 if (i % 2 == 0) else 1 for i in range(k)]


class _NoDrafter:
    def propose(self, context_ids, k: int) -> list[int]:
        return []


_PROMPTS = {
    "repetitive": list([5, 9, 13, 2, 7] * 8),
    "random": [((i * 37 + 11) % 512) for i in range(20)],
    "short": [3, 1, 4, 1, 5],
    "single-token": [42],
}


# ------------------------------------------------------------------ 1. losslessness (float64 exact)
@pytest.mark.parametrize("prompt_name", list(_PROMPTS))
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_lossless_ngram_float64(prompt_name: str, k: int) -> None:
    """P4.3.1: greedy speculative decode with the n-gram drafter is token-exact to `generate`."""
    model = _model(dtype=torch.float64)
    prompt = _PROMPTS[prompt_name]
    exp = generate(model, prompt, SamplingParams(temperature=0.0, max_tokens=40))
    got, _ = speculative_generate(model, NGramDrafter(n=3), prompt, 40, k=k)
    assert got == exp


@pytest.mark.parametrize("k", [1, 2, 4])
def test_lossless_model_drafter_float64(k: int) -> None:
    """Losslessness is drafter-independent: a (here self-)model drafter is also token-exact."""
    model = _model(dtype=torch.float64)
    prompt = _PROMPTS["random"]
    exp = generate(model, prompt, SamplingParams(temperature=0.0, max_tokens=30))
    got, _ = speculative_generate(model, ModelDrafter(model), prompt, 30, k=k)
    assert got == exp


def test_lossless_float32_high_agreement() -> None:
    """float32: speculative decode agrees with `generate` on ≥99% of tokens over many prompts; any
    residual divergence is a batched-vs-sequential argmax tie-flip (float64 test above is exact)."""
    model = _model()
    rng = random.Random(7)
    tot = agree = 0
    for _ in range(20):
        prompt = [rng.randrange(512) for _ in range(rng.randint(5, 25))]
        exp = generate(model, prompt, SamplingParams(temperature=0.0, max_tokens=40))
        got, _ = speculative_generate(model, NGramDrafter(3), prompt, 40, k=4)
        assert len(got) == len(exp)
        for a, b in zip(exp, got, strict=True):
            tot += 1
            agree += a == b
    assert agree / tot >= 0.99, f"token agreement {agree}/{tot}"


# ------------------------------------------------------------------ 2. rollback / no-draft
def test_wrong_drafts_are_corrected() -> None:
    """Every draft is wrong ⇒ every round rejects at position 0 and takes the target's token; the
    rejected drafts' KV must be rolled back cleanly, so the output is STILL the greedy sequence."""
    model = _model(dtype=torch.float64)
    prompt = _PROMPTS["random"]
    exp = generate(model, prompt, SamplingParams(temperature=0.0, max_tokens=30))
    got, stats = speculative_generate(model, _WrongDrafter(), prompt, 30, k=4)
    assert got == exp
    assert stats.n_accepted == 0  # nothing ever accepted, yet output is exact
    assert stats.mean_tokens_per_forward == pytest.approx(1.0, abs=1e-9)  # 1 token / forward


def test_no_draft_is_plain_decode() -> None:
    """A drafter that proposes nothing ⇒ speculative decode degenerates to plain 1-token decode:
    token-exact, zero accepted, exactly one decode forward per token."""
    model = _model(dtype=torch.float64)
    prompt = _PROMPTS["random"]
    exp = generate(model, prompt, SamplingParams(temperature=0.0, max_tokens=25))
    got, stats = speculative_generate(model, _NoDrafter(), prompt, 25, k=4)
    assert got == exp
    assert stats.n_drafted == 0 and stats.n_accepted == 0
    assert stats.n_target_forwards == 25 + 1  # 25 decode forwards + 1 prefill


def test_self_draft_full_acceptance_float64() -> None:
    """Draft == target ⇒ every draft matches (float64) ⇒ K+1 tokens committed per target forward."""
    model = _model(dtype=torch.float64)
    _, stats = speculative_generate(model, ModelDrafter(model), _PROMPTS["random"], 32, k=4)
    assert stats.acceptance_rate == pytest.approx(1.0)
    assert stats.mean_tokens_per_forward >= 4.0  # ~k+1 tokens/forward (draft-cost aside)


# ------------------------------------------------------------------ 3. stats + guards + primitive
def test_stats_accounting() -> None:
    model = _model()
    tokens, stats = speculative_generate(model, NGramDrafter(3), _PROMPTS["repetitive"], 40, k=4)
    assert len(tokens) == 40 == stats.n_tokens
    assert 0 <= stats.n_accepted <= stats.n_drafted
    assert isinstance(stats, SpecStats)
    assert 0.0 <= stats.acceptance_rate <= 1.0


def test_guards() -> None:
    model = _model()
    d: Drafter = NGramDrafter(3)
    with pytest.raises(ValueError):
        speculative_generate(model, d, [], 10)
    with pytest.raises(ValueError):
        speculative_generate(model, d, [1, 2], 0)
    with pytest.raises(ValueError):
        speculative_generate(model, d, [1, 2], 10, k=-1)


def test_ngram_drafter_lookup() -> None:
    """The prompt-lookup mechanism: propose the tokens that followed the last occurrence of the
    trailing n-gram; empty when there is no earlier match."""
    d = NGramDrafter(n=2)
    # last 2 = [1,2]; earlier [1,2] at index 0 is followed by [3,4]
    assert d.propose([1, 2, 3, 4, 9, 9, 1, 2], k=2) == [3, 4]
    assert d.propose([1, 2, 3, 4, 9, 9, 1, 2], k=4) == [3, 4, 9, 9]
    assert d.propose([5, 6, 7], k=2) == []  # trailing [6,7] never occurred earlier


def test_kvcache_truncate_is_slice() -> None:
    """The rollback primitive: ``truncate(M)`` yields exactly the first M positions of the cache —
    K/V bit-identical to the pre-truncate ``[:M]`` slice, length reset to M. (Comparing to a
    separately-decoded M-token cache would fail on BLAS float rounding across forward widths — that
    is fp, not a truncate bug; spec-decode losslessness above exercises the rollback end-to-end.)"""
    model = _model(dtype=torch.float64)
    seq = _PROMPTS["random"]

    def _kv(cache: KVCache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        got = cache.get(layer)
        assert got is not None
        return got

    with torch.no_grad():
        cache = KVCache(len(model.blocks))
        model(torch.tensor([seq]), cache)  # length == len(seq)
        pre = [
            (_kv(cache, la)[0][:, :, :5].clone(), _kv(cache, la)[1][:, :, :5].clone())
            for la in range(len(model.blocks))
        ]
        cache.truncate(5)
    for layer in range(len(model.blocks)):
        k, v = _kv(cache, layer)
        assert torch.equal(k, pre[layer][0]) and torch.equal(v, pre[layer][1])
    assert cache.length == 5
    with pytest.raises(ValueError):
        cache.truncate(99)  # cannot grow past current length
