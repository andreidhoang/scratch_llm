"""Discipline tests for the from-scratch Transformer LM.

These are the cheapest, highest-leverage correctness oracles in the stack: loss-at-init,
causal no-leak, and the RoPE relative-position property. If any fails, the model is wrong
in a way that would silently corrupt every downstream measurement.
"""

import math
from typing import cast

import torch

from scratch_llm.model import (
    ModelConfig,
    MultiHeadSelfAttention,
    RotaryPositionalEmbedding,
    TransformerBlock,
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


def _block0_attn(model: TransformerLM) -> MultiHeadSelfAttention:
    """White-box accessor: torch types Module attributes as Tensor|Module, so cast for the checker."""
    return cast(TransformerBlock, model.blocks[0]).attn


def test_qk_norm_off_is_identity() -> None:
    # Default (qk_norm=False) MUST be byte-identical to the no-QK-norm model: the norms are Identity.
    attn = _block0_attn(TransformerLM(_small_cfg()))
    assert isinstance(attn.q_norm, torch.nn.Identity)
    assert isinstance(attn.k_norm, torch.nn.Identity)


def test_qk_norm_loss_at_init_holds() -> None:
    # QK-norm must not break the uniform-prediction init: loss is still ≈ log(vocab).
    torch.manual_seed(0)
    cfg = _small_cfg(qk_norm=True)
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (8, 32))
    targets = torch.randint(0, cfg.vocab_size, (8, 32))
    loss = cross_entropy(model(ids), targets).item()
    assert abs(loss - math.log(cfg.vocab_size)) < 0.3, f"loss {loss:.3f} vs log V"


def test_qk_norm_bounds_query_magnitude() -> None:
    # WHY QK-norm exists: it bounds each head's q,k vector magnitude so the attention logit
    # q·kᵀ/√d cannot explode with input scale (the #1 bf16 instability). With QK-norm the per-head
    # RMS is ≈ 1 regardless of input scale; without it (Identity) the RMS scales WITH the input.
    cfg_on = _small_cfg(qk_norm=True)
    q_on = _block0_attn(TransformerLM(cfg_on)).q_norm  # RMSNorm(head_dim)
    q_off = _block0_attn(TransformerLM(_small_cfg())).q_norm  # Identity
    torch.manual_seed(0)
    x = torch.randn(2, cfg_on.n_heads, 6, cfg_on.head_dim)  # (B, H, S, head_dim)
    for scale in (1.0, 100.0):  # scale-invariant: each q vector is renormalized to unit RMS
        rms = q_on(x * scale).pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-2, rtol=0)
    # Identity path: magnitude scales linearly with the input, so logits would blow up at scale.
    rms1 = q_off(x).pow(2).mean(dim=-1).sqrt()
    rms100 = q_off(x * 100).pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms100, rms1 * 100, rtol=1e-4, atol=0)


def test_sdpa_loss_at_init_is_log_vocab() -> None:
    # The fused-SDPA training path (use_sdpa=True, the F1-OOM fix) must preserve the
    # uniform-prediction init: CE ≈ log(vocab). Backend equivalence lives in test_sdpa_backend.py.
    torch.manual_seed(0)
    cfg = _small_cfg(use_sdpa=True)
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (8, 32))
    targets = torch.randint(0, cfg.vocab_size, (8, 32))
    loss = cross_entropy(model(ids), targets).item()
    expected = math.log(cfg.vocab_size)
    assert abs(loss - expected) < 0.3, f"loss {loss:.3f} vs log V {expected:.3f}"


def test_triton_attention_forward_backward() -> None:
    # Verify that model runs forward/backward with custom Triton FlashAttention on CUDA
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA required for the Triton attention model test")

    # Check if triton is present
    try:
        import triton  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("Triton required for the Triton attention model test")

    torch.manual_seed(0)
    cfg = _small_cfg(use_triton_attention=True)
    model = TransformerLM(cfg).cuda()
    ids = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
    targets = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")

    # Forward
    logits = model(ids)
    loss = cross_entropy(logits, targets)
    assert logits.shape == (2, 16, cfg.vocab_size)

    # Backward
    loss.backward()

    # Assert gradients exist
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
