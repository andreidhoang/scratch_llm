"""SDPA training-path backend (``ModelConfig.use_sdpa``) — the F1-OOM fix.

The eager path materializes (B, H, S, S) scores and retains two fp32 score-sized tensors per
layer for backward (measured: 23.9 GB at the F1 config — OOMs the 25 GB sm120 card). With
``use_sdpa=True`` the no-cache (training) forward routes through torch's fused
``scaled_dot_product_attention(is_causal=True)`` instead. Invariants tested here:

- numerics-equivalence to eager within dtype tolerance (fp32 logits; bf16-autocast loss);
- causal no-leak under sdpa (the autoregressive contract survives the backend swap);
- the F9 observer still records under sdpa (it recomputes QKᵀ independently of the backend)
  and — the F9-mixing fix — updates on TRAINING forwards only (eval forwards leave S_max
  untouched, so the reported number is attributable to one precision regime);
- default OFF is byte-identical (``use_sdpa=False`` still routes through the eager function),
  and a cache-carrying forward falls back to eager even with ``use_sdpa=True``.
"""

import math
from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import Tensor

import scratch_llm.model as model_mod
from scratch_llm.model import (
    KVCache,
    ModelConfig,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerLM,
    cross_entropy,
)


def _cfg(**overrides: object) -> ModelConfig:
    base: dict[str, object] = dict(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,  # exercise GQA: sdpa runs AFTER the kv repeat_interleave
        context_length=64,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _same_seed_pair(seed: int = 0, **overrides: object) -> tuple[TransformerLM, TransformerLM]:
    """Two models with identical weights: the use_sdpa flag consumes no RNG."""
    torch.manual_seed(seed)
    eager = TransformerLM(_cfg(**overrides))
    torch.manual_seed(seed)
    sdpa = TransformerLM(_cfg(use_sdpa=True, **overrides))
    return eager, sdpa


def _attn(model: TransformerLM, layer: int = 0) -> MultiHeadSelfAttention:
    return cast(TransformerBlock, model.blocks[layer]).attn


def _spying_eager(calls: list[int]) -> Callable[..., Tensor]:
    """Wrap the module-level eager attention so routing can be asserted, math unchanged."""
    real = model_mod.scaled_dot_product_attention

    def spy(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        calls.append(1)
        return real(q, k, v, mask)

    return spy


# ---------------------------------------------------------------------------
# (a) numerics-equivalence: sdpa is the same math, minus the materialized scores
# ---------------------------------------------------------------------------


def test_sdpa_logits_match_eager_fp32() -> None:
    eager, sdpa = _same_seed_pair()
    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        y_eager = eager(ids)
        y_sdpa = sdpa(ids)
    torch.testing.assert_close(y_sdpa, y_eager, atol=1e-5, rtol=1e-5)


def test_sdpa_bf16_autocast_loss_close_to_eager() -> None:
    # The F1 training regime: bf16 autocast. Same weights/batch ⇒ the two backends' CE must
    # agree to bf16 rounding, and both must sit at the uniform-prediction init (≈ log V).
    eager, sdpa = _same_seed_pair(seed=1)
    ids = torch.randint(0, 512, (2, 32))
    targets = torch.randint(0, 512, (2, 32))
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        loss_eager = cross_entropy(eager(ids), targets).item()
        loss_sdpa = cross_entropy(sdpa(ids), targets).item()
    assert abs(loss_eager - loss_sdpa) < 1e-2, f"eager {loss_eager:.4f} vs sdpa {loss_sdpa:.4f}"
    assert abs(loss_sdpa - math.log(512)) < 0.5


def test_sdpa_backward_produces_grads() -> None:
    # The point of the fix is the TRAINING forward: backward through the fused kernel must
    # populate every gradient (fused SDPA recomputes attention in backward — nothing retained).
    torch.manual_seed(0)
    model = TransformerLM(_cfg(use_sdpa=True))
    ids = torch.randint(0, 512, (2, 16))
    targets = torch.randint(0, 512, (2, 16))
    cross_entropy(model(ids), targets).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing grad: {name}"


# ---------------------------------------------------------------------------
# (b) causal no-leak under sdpa
# ---------------------------------------------------------------------------


def test_sdpa_causal_no_leak() -> None:
    torch.manual_seed(0)
    model = TransformerLM(_cfg(use_sdpa=True))
    model.eval()
    ids = torch.randint(0, 512, (2, 20))
    cut = 10
    with torch.no_grad():
        base = model(ids)
        perturbed = ids.clone()
        perturbed[:, cut + 1 :] = (perturbed[:, cut + 1 :] + 7) % 512
        after = model(perturbed)
    torch.testing.assert_close(base[:, : cut + 1], after[:, : cut + 1])
    assert not torch.allclose(base[:, cut + 1 :], after[:, cut + 1 :])


# ---------------------------------------------------------------------------
# (c) F9 observer: records under sdpa, training forwards only
# ---------------------------------------------------------------------------


def test_observer_records_under_sdpa_and_matches_eager_oracle() -> None:
    # The observer recomputes QKᵀ itself (backend-independent): under use_sdpa it must record
    # exactly what the eager-path model records on the same weights/batch.
    eager, sdpa = _same_seed_pair(seed=2, track_attn_logits=True)
    ids = torch.randint(0, 512, (2, 16))
    eager(ids)
    sdpa(ids)
    for layer in range(2):
        e, s = _attn(eager, layer), _attn(sdpa, layer)
        assert e.last_max_logits is not None and s.last_max_logits is not None
        torch.testing.assert_close(s.last_max_logits, e.last_max_logits)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_observer_does_not_update_in_eval_mode(use_sdpa: bool) -> None:
    # F9-mixing fix: train.py's _val_loss runs fp32 eval forwards ~50×/arm — those must not
    # pollute the training-regime S_max. Only training-mode forwards update the observer.
    torch.manual_seed(3)
    model = TransformerLM(_cfg(track_attn_logits=True, use_sdpa=use_sdpa))
    ids = torch.randint(0, 512, (2, 16))

    model.eval()
    with torch.no_grad():
        model(ids)
    assert _attn(model).last_max_logits is None  # an eval forward records nothing
    assert _attn(model).max_logits_running is None

    model.train()
    model(ids)
    last = _attn(model).last_max_logits
    running = _attn(model).max_logits_running
    assert last is not None and running is not None
    last, running = last.clone(), running.clone()

    model.eval()
    with torch.no_grad():
        model(torch.randint(0, 512, (2, 16)))  # a different batch would move last_max_logits
    after_last = _attn(model).last_max_logits
    after_running = _attn(model).max_logits_running
    assert after_last is not None and after_running is not None
    assert torch.equal(after_last, last)  # untouched by the eval forward
    assert torch.equal(after_running, running)


# ---------------------------------------------------------------------------
# (d) default path unchanged + cache fallback routing
# ---------------------------------------------------------------------------


def test_default_off_is_bitwise_identical_and_routes_through_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ModelConfig(vocab_size=8, d_model=8, n_layers=1, n_heads=2).use_sdpa is False
    calls: list[int] = []
    monkeypatch.setattr(model_mod, "scaled_dot_product_attention", _spying_eager(calls))

    torch.manual_seed(0)
    default = TransformerLM(_cfg())  # use_sdpa unset — the current-HEAD construction
    torch.manual_seed(0)
    explicit_off = TransformerLM(_cfg(use_sdpa=False))
    ids = torch.randint(0, 512, (2, 16))
    with torch.no_grad():
        y_default = default(ids)
        assert len(calls) == 2  # one eager call per layer — the default path is untouched
        y_off = explicit_off(ids)
    assert torch.equal(y_default, y_off)  # bit-identical, not merely close

    calls.clear()
    torch.manual_seed(0)
    sdpa_model = TransformerLM(_cfg(use_sdpa=True))
    with torch.no_grad():
        sdpa_model(ids)
    assert calls == []  # the sdpa forward never enters the score-materializing function


def test_sdpa_model_cache_path_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    # sdpa covers exactly cache=None; prefill/decode forwards (cache present) take the eager
    # path and must still match the full no-cache (sdpa) recompute within backend tolerance.
    torch.manual_seed(0)
    model = TransformerLM(_cfg(use_sdpa=True))
    model.eval()
    calls: list[int] = []
    monkeypatch.setattr(model_mod, "scaled_dot_product_attention", _spying_eager(calls))

    prompt = torch.tensor([[3, 1, 4, 1, 5, 9, 2]])
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        cached_last = model(prompt, cache)[0, -1]  # prefill: cache present ⇒ eager
        assert len(calls) == 2  # one per layer
        full_last = model(prompt)[0, -1]  # no cache ⇒ sdpa
        assert len(calls) == 2  # unchanged — sdpa forward added no eager calls
        decode_logits = model(torch.tensor([[7]]), cache)[0, -1]  # decode step: eager again
        assert len(calls) == 4
        full_decode = model(torch.tensor([[3, 1, 4, 1, 5, 9, 2, 7]]))[0, -1]
    torch.testing.assert_close(cached_last, full_last, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decode_logits, full_decode, atol=1e-5, rtol=1e-5)
