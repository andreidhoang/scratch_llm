"""A1 Rung 4.5 — MLA weight-absorption oracle.

The load-bearing identity: attending in latent space with W_UK folded into the query and W_UV into
the output is NUMERICALLY IDENTICAL to reconstructing K,V per head and attending normally. That
identity is what lets a serving engine cache the low-rank latent c_KV (+ the decoupled RoPE key)
instead of the full per-head K,V — a large KV reduction at ZERO quality change. Also pins causality
and the KV-byte accounting.
"""

import pytest
import torch

from scratch_llm.mla import MLAConfig, MultiHeadLatentAttention


def _mla(dtype: torch.dtype = torch.float64) -> MultiHeadLatentAttention:
    torch.manual_seed(0)
    cfg = MLAConfig(d_model=128, n_heads=8, d_head=16, d_latent=64, d_rope=16, max_seq_len=256)
    return MultiHeadLatentAttention(cfg).to(dtype).eval()


def test_absorption_identity_float64() -> None:
    """Naive reconstruct-then-attend == absorbed latent-space attention, to float64 precision."""
    mla = _mla()
    h = torch.randn(2, 24, 128, dtype=torch.float64)
    pos = torch.arange(24)
    with torch.no_grad():
        naive = mla.forward_naive(h, pos)
        absorbed = mla.forward_absorbed(h, pos)
    assert torch.allclose(naive, absorbed, atol=1e-10), (naive - absorbed).abs().max().item()


def test_absorption_identity_float32() -> None:
    """The identity survives float32 (the deployment dtype) within rounding."""
    mla = _mla(torch.float32)
    h = torch.randn(2, 24, 128)
    pos = torch.arange(24)
    with torch.no_grad():
        naive = mla.forward_naive(h, pos)
        absorbed = mla.forward_absorbed(h, pos)
    assert torch.allclose(naive, absorbed, atol=1e-4), (naive - absorbed).abs().max().item()


def test_causal_no_future_leak() -> None:
    """Truncating the sequence after position t must not change outputs at ≤ t (causal mask holds
    in both paths — a future token cannot leak into a past one)."""
    mla = _mla()
    h = torch.randn(1, 16, 128, dtype=torch.float64)
    pos = torch.arange(16)
    with torch.no_grad():
        full = mla.forward_absorbed(h, pos)
        prefix = mla.forward_absorbed(h[:, :10], pos[:10])
    assert torch.allclose(full[:, :10], prefix, atol=1e-10)


def test_kv_byte_accounting() -> None:
    """MLA caches d_latent + d_rope values/token vs 2·n_heads·d_head for MHA — the capacity win."""
    cfg = MLAConfig(d_model=7168, n_heads=128, d_head=128, d_latent=512, d_rope=64)
    assert cfg.kv_bytes_per_token(2) == (512 + 64) * 2 == 1152
    assert cfg.mha_kv_bytes_per_token(2) == 2 * 128 * 128 * 2
    # MLA is a small fraction of full MHA and several× smaller than GQA-8
    assert cfg.kv_bytes_per_token(2) / cfg.mha_kv_bytes_per_token(2) < 0.05
    assert cfg.gqa_kv_bytes_per_token(8, 2) / cfg.kv_bytes_per_token(2) > 3.0


def test_d_rope_must_be_even() -> None:
    with pytest.raises(ValueError):
        MultiHeadLatentAttention(MLAConfig(d_model=64, n_heads=4, d_head=16, d_latent=32, d_rope=7))
