"""Discipline tests for the from-scratch Transformer LM.

These are the cheapest, highest-leverage correctness oracles in the stack: loss-at-init,
causal no-leak, and the RoPE relative-position property. If any fails, the model is wrong
in a way that would silently corrupt every downstream measurement.
"""

import math

import torch

from reasoning_llm.model import (
    ModelConfig,
    RotaryPositionalEmbedding,
    TransformerLM,
    cross_entropy,
    scaled_dot_product_attention,
)


def _small_cfg(**overrides: object) -> ModelConfig:
    base: dict[str, object] = dict(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,  # exercise GQA by default
        context_length=64,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def test_forward_shape() -> None:
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (3, 16))
    logits = model(ids)
    assert logits.shape == (3, 16, cfg.vocab_size)


def test_loss_at_init_is_log_vocab() -> None:
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (8, 32))
    targets = torch.randint(0, cfg.vocab_size, (8, 32))
    loss = cross_entropy(model(ids), targets).item()
    expected = math.log(cfg.vocab_size)
    assert abs(loss - expected) < 0.3, f"loss {loss:.3f} vs log V {expected:.3f}"


def test_causal_attention_does_not_leak_future() -> None:
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TransformerLM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 20))
    cut = 10

    with torch.no_grad():
        base = model(ids)
        perturbed = ids.clone()
        # change every token strictly after the cut
        perturbed[:, cut + 1 :] = (perturbed[:, cut + 1 :] + 7) % cfg.vocab_size
        after = model(perturbed)

    # logits at positions 0..cut must be identical (they cannot see the future)
    torch.testing.assert_close(base[:, : cut + 1], after[:, : cut + 1])
    # sanity: something downstream actually changed
    assert not torch.allclose(base[:, cut + 1 :], after[:, cut + 1 :])


def test_rope_score_depends_only_on_offset() -> None:
    torch.manual_seed(0)
    head_dim, max_len = 16, 128
    rope = RotaryPositionalEmbedding(head_dim, max_len)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def score(i: int, j: int) -> float:
        qi = rope(q, torch.tensor([i]))
        kj = rope(k, torch.tensor([j]))
        return (qi * kj).sum().item()

    # same offset (i - j = 5) at different absolute positions → same score
    assert abs(score(5, 0) - score(20, 15)) < 1e-4
    # different offset → different score (guards against a no-op RoPE)
    assert abs(score(5, 0) - score(6, 0)) > 1e-4


def test_sdpa_causal_mask_zeros_future() -> None:
    torch.manual_seed(0)
    s, d = 4, 8
    q, k, v = (torch.randn(1, 1, s, d) for _ in range(3))
    causal = torch.tril(torch.ones(s, s, dtype=torch.bool))
    out_full = scaled_dot_product_attention(q, k, v, causal)
    # position 0 attends only to itself → output equals v[0]
    torch.testing.assert_close(out_full[0, 0, 0], v[0, 0, 0])


def test_full_mha_when_kv_equals_heads() -> None:
    torch.manual_seed(0)
    cfg = _small_cfg(n_kv_heads=4)  # n_kv == n_heads ⇒ plain MHA path
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    assert model(ids).shape == (1, 8, cfg.vocab_size)


def test_seed_reproducibility() -> None:
    def run() -> torch.Tensor:
        torch.manual_seed(1234)
        model = TransformerLM(_small_cfg())
        ids = torch.arange(16).unsqueeze(0) % 512
        with torch.no_grad():
            return model(ids)

    torch.testing.assert_close(run(), run())


def test_config_rejects_bad_shapes() -> None:
    for bad in (
        dict(d_model=65, n_heads=4),  # not divisible
        dict(n_heads=4, n_kv_heads=3),  # 4 % 3 != 0
        dict(d_model=4, n_heads=4),  # head_dim=1, odd → RoPE needs even
    ):
        try:
            _small_cfg(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
