"""A5 Rung 4 — FP8 E4M3 KV cache oracle tests (CPU fake-quant; NOT gpu-marked).

The numeric law, not ``allclose``, is the oracle:

* **E4M3 format** — max magnitude ±448, values above it clamped (never NaN), representable
  values reproduced exactly, deterministic round-trip.
* **FP8 ≫ INT4 fidelity** — on Gaussian data FP8 SQNR is far above INT4's (measured ~31.8 vs
  ~18.6 dB): 8 bits vs 4 bits.
* **Per-channel-K law (integer quant)** — with channel outliers (transformer "massive
  activations"), a per-*token* integer scale is wrecked by the outlier channel; per-*channel*
  gives ~2.5× smaller MSE (measured 2.49×). FP8's constant relative precision makes it
  granularity-insensitive — that is *why* the FP8 KV cache is robust.
* **End-to-end** — greedy/teacher-forced decode with an FP8 KV cache reproduces the BF16-KV
  logits within noise (SQNR ~24.5 dB) at **half the KV bytes** (FP8 codes are 1 byte vs BF16 2).
* **INT4 stress** — an INT4 KV cache visibly degrades (logit MSE ~2.9× larger, SQNR ~4.6 dB
  lower). If it did not degrade, the stress would not be stressing.
"""

from __future__ import annotations

import torch

from scratch_llm.model import KVCache, ModelConfig, TransformerLM
from scratch_llm.quant.fp8_kv import (
    FP8_E4M3_MAX,
    QuantizedKVCache,
    dequantize_fp8_e4m3,
    dequantize_int4,
    greedy_generate_with_cache,
    quantize_fp8_e4m3,
    quantize_int4,
    sqnr_db,
)


def _fp8(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return dequantize_fp8_e4m3(*quantize_fp8_e4m3(x, axis))


def _int4(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return dequantize_int4(*quantize_int4(x, axis))


# --------------------------------------------------------------------------------------------
# E4M3 format
# --------------------------------------------------------------------------------------------
def test_e4m3_storage_is_one_byte() -> None:
    """The byte-halving is real: float8_e4m3fn is a 1-byte type (vs BF16's 2)."""
    assert torch.empty(1, dtype=torch.float8_e4m3fn).element_size() == 1


def test_e4m3_clamps_overflow_without_nan() -> None:
    """E4M3 turns >448 into NaN on a raw cast; our quantizer clamps first, so no NaN escapes and
    the top of the range saturates at ±448."""
    x = torch.tensor([[500.0, -500.0, 448.0, 0.0]])
    # axis=-1 amax=500 → scale=500/448; the clamp keeps the largest code at ±448*scale ≈ ±500.
    deq = _fp8(x)
    assert torch.isfinite(deq).all()
    assert deq.abs().max().item() <= 500.0 + 1e-3
    # A direct code at exactly the max magnitude round-trips finite (no NaN top).
    codes, scale = quantize_fp8_e4m3(torch.tensor([[FP8_E4M3_MAX]]), -1)
    assert torch.isfinite(dequantize_fp8_e4m3(codes, scale)).all()


def test_e4m3_grid_values_exact_at_unit_scale() -> None:
    """The E4M3 grid itself (bias 7, 3 mantissa bits) reproduces its representable values exactly
    under a unit scale. 1.0, 1.5, 1.75, 2.0, 3.0, 4.0 are all on the grid; 448 is the max. (Note:
    amax-to-448 scaling is not a power of two, so it *does* perturb these — tested elsewhere via
    SQNR; here we pin the underlying format.)"""
    x = torch.tensor([1.0, 1.5, 1.75, 2.0, 3.0, 4.0, -2.0, FP8_E4M3_MAX])
    grid = x.to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(grid, x)


def test_roundtrip_deterministic() -> None:
    torch.manual_seed(7)
    x = torch.randn(3, 5, 9)
    assert torch.equal(_fp8(x), _fp8(x))


# --------------------------------------------------------------------------------------------
# FP8 vs INT4 tensor-level SQNR (8 bits ≫ 4 bits)
# --------------------------------------------------------------------------------------------
def test_fp8_sqnr_beats_int4_and_clears_floor() -> None:
    torch.manual_seed(3)
    g = torch.randn(4, 64, 128)
    fp8_db = sqnr_db(g, _fp8(g))
    int4_db = sqnr_db(g, _int4(g))
    # FP8 E4M3 (3 mantissa bits) on Gaussian data measures ~31.8 dB; INT4 ~18.6 dB.
    assert fp8_db > 28.0, fp8_db
    assert fp8_db > int4_db + 8.0, (fp8_db, int4_db)


# --------------------------------------------------------------------------------------------
# Per-channel-K law (the granularity choice) — integer quant, channel outliers
# --------------------------------------------------------------------------------------------
def test_per_channel_k_beats_per_token_under_channel_outliers() -> None:
    """K has a few persistently-large channels (massive activations). A per-token integer scale is
    dominated by the outlier channel and crushes the rest; per-channel adapts → ~2.5× smaller MSE.
    """
    torch.manual_seed(1)
    k = torch.randn(1, 4, 256, 32)  # (B, n_kv, T, head_dim)
    outlier = torch.randperm(32)[:2]
    k[..., outlier] *= 10.0
    per_channel = _int4(k, axis=2)  # scale reduces over tokens → one scale per channel
    per_token = _int4(k, axis=3)  # scale reduces over head_dim → one scale per token
    mse_pc = torch.mean((k - per_channel) ** 2).item()
    mse_pt = torch.mean((k - per_token) ** 2).item()
    ratio = mse_pt / mse_pc
    assert ratio > 2.0, ratio  # measured ~2.49×

    # FP8 is granularity-insensitive (constant relative precision) — per-channel gives no gain,
    # yet FP8's SQNR stays far above INT4's regardless of granularity.
    assert sqnr_db(k, _fp8(k, axis=2)) > sqnr_db(k, per_channel) + 8.0


# --------------------------------------------------------------------------------------------
# End-to-end: FP8 KV cache wired into the A1 decoder — fidelity, halved bytes, INT4 stress
# --------------------------------------------------------------------------------------------
def _toy_model() -> tuple[TransformerLM, ModelConfig]:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=256, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, context_length=256
    )
    model = TransformerLM(cfg)
    model.eval()
    return model, cfg


@torch.no_grad()
def _teacher_forced_logits(model: TransformerLM, ids: list[int], prompt_len: int, cache: object):
    """Feed the SAME fixed token sequence through ``cache`` (prefill prompt, then one real token
    per step) and collect the next-token logits. Identical inputs across caches isolate KV-quant
    error from trajectory divergence — the clean SQNR oracle."""
    x = torch.tensor([ids[:prompt_len]])
    logits = [model(x, cache)[0, -1]]  # type: ignore[arg-type]
    for t in range(prompt_len, len(ids)):
        x = torch.tensor([[ids[t]]])
        logits.append(model(x, cache)[0, -1])  # type: ignore[arg-type]
    return torch.stack(logits[:-1])


def test_fp8_kv_matches_bf16_and_int4_stresses() -> None:
    model, cfg = _toy_model()
    ids = list(range(1, 49))
    prompt_len = 16

    ref = _teacher_forced_logits(model, ids, prompt_len, QuantizedKVCache(cfg.n_layers, "bf16"))
    fp8 = _teacher_forced_logits(model, ids, prompt_len, QuantizedKVCache(cfg.n_layers, "fp8"))
    int4 = _teacher_forced_logits(model, ids, prompt_len, QuantizedKVCache(cfg.n_layers, "int4"))

    fp8_sqnr = sqnr_db(ref, fp8)
    int4_sqnr = sqnr_db(ref, int4)
    fp8_mse = torch.mean((ref - fp8) ** 2).item()
    int4_mse = torch.mean((ref - int4) ** 2).item()

    # Fidelity: FP8 KV reproduces BF16-KV logits within noise (measured SQNR ~24.5 dB, MSE ~2.4e-3).
    assert fp8_sqnr > 22.0, fp8_sqnr
    assert fp8_mse < 5e-3, fp8_mse
    # Token agreement (per-position argmax) high on the FP8 path.
    agree = (fp8.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert agree >= 0.85, agree

    # STRESS: INT4 KV visibly degrades — materially larger error, materially lower SQNR.
    assert int4_mse > 2.0 * fp8_mse, (int4_mse, fp8_mse)  # measured ~2.9×
    assert fp8_sqnr > int4_sqnr + 3.0, (fp8_sqnr, int4_sqnr)  # measured ~4.6 dB better


def test_fp8_kv_halves_bytes() -> None:
    model, _ = _toy_model()
    prompt = list(range(1, 17))
    n = 32

    bf16 = QuantizedKVCache(3, "bf16")
    fp8 = QuantizedKVCache(3, "fp8")
    int4 = QuantizedKVCache(3, "int4")
    for cache in (bf16, fp8, int4):
        gen, _ = greedy_generate_with_cache(model, prompt, n, cache)
        assert len(gen) == n  # the cache duck-types KVCache: the A1 decoder drove it end-to-end

    # FP8 codes are 1 byte vs BF16 2 → the cache is ~half the bytes (amortized scales are small).
    assert fp8.kv_bytes < 0.62 * bf16.kv_bytes, (fp8.kv_bytes, bf16.kv_bytes)  # measured 0.552×
    assert int4.kv_bytes < fp8.kv_bytes  # INT4 smaller still (~0.30×), at the cost of fidelity


def test_quantized_cache_duck_types_kvcache() -> None:
    """The QuantizedKVCache is a drop-in for model.KVCache: same length/append/get/advance
    contract, so model.py needs no change to decode against it."""
    model, cfg = _toy_model()
    prompt = list(range(1, 17))
    plain, _ = greedy_generate_with_cache(model, prompt, 8, KVCache(cfg.n_layers))
    fake, _ = greedy_generate_with_cache(model, prompt, 8, QuantizedKVCache(cfg.n_layers, "fp8"))
    assert len(plain) == len(fake) == 8
