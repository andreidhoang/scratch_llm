"""F9 — MuonClip / QK-Clip guard (Kimi-K2, arXiv:2507.20534).

Two additive pieces, both default-off:

- **Observer** (`ModelConfig.track_attn_logits`): `MultiHeadSelfAttention` records each head's
  max pre-softmax logit per forward (`last_max_logits`, shape `(n_heads,)`) plus a running max.
  OFF must be byte-identical — same outputs, no RNG consumption, nothing recorded.
- **Clip** (`apply_qk_clip`): post-step, weight-space — every head whose observed S_max exceeds
  τ gets its W_q and W_k head slices scaled by exactly `sqrt(τ/S_max)` (MHA 1:1 layout), so a
  re-forward's logits land at τ. Under-τ heads stay byte-identical; all-under-τ is a no-op.

These invariants ARE the F9 DoD: (a) qk_norm bounds the max logit (<30) below the un-normed
max; (b) exact-factor rescale of over-τ heads only; (c) no-op under τ; (d) re-forward ≤ τ(+slack).
"""

import math
from typing import cast

import numpy as np
import torch
from torch import Tensor

from scratch_llm.model import (
    ModelConfig,
    MultiHeadSelfAttention,
    RotaryPositionalEmbedding,
    TransformerBlock,
    TransformerLM,
)
from scratch_llm.optim import apply_qk_clip
from scratch_llm.train import TrainConfig, train


def _cfg(**overrides: object) -> ModelConfig:
    # Full MHA by default: QK-Clip's per-head W_q/W_k slices are then 1:1 (the exact-factor DoD).
    base: dict[str, object] = dict(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        context_length=32,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _attn(model: TransformerLM, layer: int = 0) -> MultiHeadSelfAttention:
    return cast(TransformerBlock, model.blocks[layer]).attn


def _fresh_attn(cfg: ModelConfig, seed: int = 0) -> MultiHeadSelfAttention:
    torch.manual_seed(seed)
    rope = RotaryPositionalEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)
    return MultiHeadSelfAttention(cfg, rope)


def _last_max(attn: MultiHeadSelfAttention) -> Tensor:
    assert attn.last_max_logits is not None
    return attn.last_max_logits


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


def test_observer_off_is_byte_identical_and_records_nothing() -> None:
    # Default path (track_attn_logits=False) must be byte-identical to a model that never had the
    # feature: identical outputs, no RNG consumed by the observer, and no maxima recorded.
    torch.manual_seed(0)
    m_off = TransformerLM(_cfg())
    torch.manual_seed(0)
    m_on = TransformerLM(_cfg(track_attn_logits=True))  # flag consumes no RNG ⇒ same weights
    ids = torch.randint(0, 128, (2, 16))

    rng_before = torch.random.get_rng_state()
    y_off = m_off(ids)
    assert torch.equal(torch.random.get_rng_state(), rng_before)  # forward consumes no RNG
    y_on = m_on(ids)
    assert torch.equal(torch.random.get_rng_state(), rng_before)  # observer consumes no RNG
    assert torch.equal(y_off, y_on)  # observer must not change the math

    for layer in range(2):
        assert _attn(m_off, layer).last_max_logits is None
        assert _attn(m_off, layer).max_logits_running is None
        assert isinstance(_attn(m_on, layer).last_max_logits, Tensor)


def test_observer_records_per_layer_head_maxima_with_running_max() -> None:
    cfg = _cfg(track_attn_logits=True)
    torch.manual_seed(0)
    model = TransformerLM(cfg)
    model(torch.randint(0, cfg.vocab_size, (2, 16)))
    for layer in range(cfg.n_layers):
        maxima = _last_max(_attn(model, layer))
        assert maxima.shape == (cfg.n_heads,)
        assert maxima.dtype == torch.float32
        assert bool(torch.isfinite(maxima).all())

    # Running max across forwards is the elementwise max of the per-forward maxima.
    attn = _attn(model)
    first = _last_max(attn).clone()
    model(torch.randint(0, cfg.vocab_size, (2, 16)))  # RNG advanced ⇒ a different batch
    second = _last_max(attn)
    assert attn.max_logits_running is not None
    torch.testing.assert_close(attn.max_logits_running, torch.maximum(first, second))


def test_observer_matches_hand_computed_masked_scores() -> None:
    # Exact-math oracle at the module level: recompute per-head max pre-softmax logit from the
    # module's own projections/norm/rope and the causal mask — must match what forward recorded.
    cfg = _cfg(track_attn_logits=True)
    attn = _fresh_attn(cfg)
    b, s = 2, 8
    x = torch.randn(b, s, cfg.d_model)
    pos = torch.arange(s)
    attn(x, pos)

    q = attn.q_proj(x).view(b, s, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(b, s, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
    q = attn.rope(attn.q_norm(q), pos)
    k = attn.rope(attn.k_norm(k), pos)
    scores = q @ k.transpose(-2, -1) / math.sqrt(cfg.head_dim)
    causal = torch.arange(s).unsqueeze(0) <= torch.arange(s).unsqueeze(1)  # (s, s) True=attend
    expected = scores.masked_fill(~causal, float("-inf")).amax(dim=(0, 2, 3)).float()
    torch.testing.assert_close(_last_max(attn), expected)


def test_qk_norm_bounds_max_logit_below_unnormed_max() -> None:
    # DoD (a): on the same scaled input, qk_norm=True caps the max logit (< 30, the F9/KILL
    # threshold) while qk_norm=False lets it blow up with input scale.
    cfg_on = _cfg(track_attn_logits=True, qk_norm=True)
    cfg_off = _cfg(track_attn_logits=True)
    attn_on = _fresh_attn(cfg_on, seed=0)
    attn_off = _fresh_attn(cfg_off, seed=0)  # RMSNorm init consumes no RNG ⇒ same projections
    torch.manual_seed(1)
    x = torch.randn(2, 16, cfg_on.d_model) * 100.0  # large scale — the bf16 blow-up regime
    pos = torch.arange(16)
    attn_on(x, pos)
    attn_off(x, pos)
    max_on = float(_last_max(attn_on).max())
    max_off = float(_last_max(attn_off).max())
    assert max_on < max_off, f"qk_norm max {max_on:.2f} not below un-normed {max_off:.2f}"
    assert max_on < 30.0, f"qk_norm max logit {max_on:.2f} breaches the <30 falsifier"


# ---------------------------------------------------------------------------
# apply_qk_clip
# ---------------------------------------------------------------------------


def test_apply_qk_clip_exact_rescale_over_tau_heads_only() -> None:
    # DoD (b): hand-built S_max — head 0 over τ, head 1 AT τ (strict >, so untouched), rest under.
    cfg = _cfg(track_attn_logits=True)
    attn = _fresh_attn(cfg)
    attn.last_max_logits = torch.tensor([250.0, 100.0, 40.0, 99.9])
    wq = attn.q_proj.weight.detach().clone()
    wk = attn.k_proj.weight.detach().clone()
    wv = attn.v_proj.weight.detach().clone()

    n_clipped = apply_qk_clip(attn, tau=100.0)

    assert n_clipped == 1
    hd = cfg.head_dim
    factor = math.sqrt(100.0 / 250.0)
    assert torch.equal(attn.q_proj.weight[:hd], wq[:hd] * factor)
    assert torch.equal(attn.k_proj.weight[:hd], wk[:hd] * factor)
    assert torch.equal(attn.q_proj.weight[hd:], wq[hd:])  # under/at-τ heads byte-identical
    assert torch.equal(attn.k_proj.weight[hd:], wk[hd:])
    assert torch.equal(attn.v_proj.weight, wv)  # V is never part of the logit — never touched


def test_apply_qk_clip_noop_when_all_heads_under_tau() -> None:
    # DoD (c): every head ≤ τ ⇒ zero clips, weights byte-identical. Also: a model that never
    # ran a tracked forward (last_max_logits is None) is a no-op, not an error.
    cfg = _cfg(track_attn_logits=True)
    attn = _fresh_attn(cfg)
    attn.last_max_logits = torch.tensor([10.0, 99.0, 100.0, 3.0])
    wq = attn.q_proj.weight.detach().clone()
    wk = attn.k_proj.weight.detach().clone()
    assert apply_qk_clip(attn, tau=100.0) == 0
    assert torch.equal(attn.q_proj.weight, wq)
    assert torch.equal(attn.k_proj.weight, wk)

    fresh = _fresh_attn(cfg)  # no forward ⇒ no observations
    assert apply_qk_clip(fresh, tau=100.0) == 0


def test_reforward_after_clip_is_bounded_by_tau() -> None:
    # DoD (d): observe → clip → re-forward on the SAME input: every head ≤ τ(+slack); the heads
    # that were over land at ≈ τ exactly (logit ∝ q·k scales by exactly τ/S_max). qk_norm=False —
    # with qk_norm on, the clip is inert (see test below), so this is the path where it must bite.
    cfg = _cfg(track_attn_logits=True)
    attn = _fresh_attn(cfg)
    torch.manual_seed(2)
    x = torch.randn(2, 16, cfg.d_model) * 50.0
    pos = torch.arange(16)
    attn(x, pos)
    before = _last_max(attn).clone()
    tau = 0.25 * float(before.max())
    over = before > tau
    assert bool(over.any())

    n_clipped = apply_qk_clip(attn, tau=tau)
    assert n_clipped == int(over.sum())

    attn(x, pos)
    after = _last_max(attn)
    assert bool((after <= tau * (1 + 1e-4)).all()), f"post-clip maxima {after} exceed τ={tau}"
    torch.testing.assert_close(after[over], torch.full_like(after[over], tau), rtol=1e-4, atol=0)
    torch.testing.assert_close(after[~over], before[~over])  # under-τ heads unchanged


def test_qk_clip_is_inert_under_qk_norm() -> None:
    # Kimi-K2 composition claim: with qk_norm on, RMSNorm renormalizes q,k AFTER the projection,
    # so a weight-space rescale cannot move the logits — the clip fires but changes (almost)
    # nothing. Slack is the RMSNorm eps only.
    cfg = _cfg(track_attn_logits=True, qk_norm=True)
    attn = _fresh_attn(cfg)
    torch.manual_seed(3)
    x = torch.randn(2, 16, cfg.d_model)
    pos = torch.arange(16)
    attn(x, pos)
    before = _last_max(attn).clone()
    tau = 0.5 * float(before.max())  # force some heads over τ
    assert bool((before > tau).any())
    assert apply_qk_clip(attn, tau=tau) > 0  # it DOES rescale weights…
    attn(x, pos)
    torch.testing.assert_close(_last_max(attn), before, rtol=1e-3, atol=1e-3)  # …to no effect


def test_gqa_clip_scales_q_full_factor_and_leaves_shared_k_untouched() -> None:
    # GQA deviation (documented): a kv head is SHARED across a query-head group, so rescaling
    # W_k rows would perturb under-τ sibling heads (violating byte-identity). The clip instead
    # puts the FULL factor τ/S_max on the over-τ head's W_q rows — same bound, siblings intact.
    cfg = _cfg(track_attn_logits=True, n_kv_heads=2)  # heads 0,1 share kv 0; heads 2,3 share kv 1
    attn = _fresh_attn(cfg)
    attn.last_max_logits = torch.tensor([200.0, 50.0, 50.0, 50.0])
    wq = attn.q_proj.weight.detach().clone()
    wk = attn.k_proj.weight.detach().clone()

    assert apply_qk_clip(attn, tau=100.0) == 1
    hd = cfg.head_dim
    assert torch.equal(attn.q_proj.weight[:hd], wq[:hd] * (100.0 / 200.0))  # full factor on q
    assert torch.equal(attn.q_proj.weight[hd:], wq[hd:])  # sibling q heads untouched
    assert torch.equal(attn.k_proj.weight, wk)  # shared K entirely untouched


# ---------------------------------------------------------------------------
# Train-loop wiring
# ---------------------------------------------------------------------------


def _run_train(model_cfg: ModelConfig, **overrides: object) -> list[tuple[int, float]]:
    torch.manual_seed(0)
    model = TransformerLM(model_cfg)
    data = np.tile(np.arange(16, dtype=np.int64), 200)
    cfg = TrainConfig(
        max_steps=3,
        batch_size=4,
        context_length=8,
        log_every=1,
        seed=0,
        **overrides,  # type: ignore[arg-type]
    )
    return train(cfg, data, model)


def test_train_default_path_untouched_and_qk_clip_runs() -> None:
    # The knobs must be invisible by default: no-knob run == explicit-off run == observer-on run
    # with a τ no head reaches (loss histories bitwise identical — observer & no-op clip change
    # neither the math nor the RNG stream). And with a tiny τ the clip fires every step and the
    # 3-step train still runs to finite losses.
    base = _run_train(_cfg())
    explicit_off = _run_train(_cfg(), qk_clip=False, qk_clip_tau=42.0)
    assert base == explicit_off

    inert = _run_train(_cfg(track_attn_logits=True), qk_clip=True, qk_clip_tau=1e9)
    assert base == inert

    clipped = _run_train(_cfg(track_attn_logits=True), qk_clip=True, qk_clip_tau=0.05)
    assert len(clipped) == len(base)
    assert all(math.isfinite(loss) for _, loss in clipped)


def test_train_qk_clip_requires_tracking() -> None:
    # Fail fast: qk_clip without the observer would silently no-op every step.
    try:
        _run_train(_cfg(), qk_clip=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError: qk_clip=True needs track_attn_logits=True")
