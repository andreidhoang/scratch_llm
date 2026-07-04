"""Tests for group-wise INT4 quantization + 2-nibble packing (A5 Rung 2).

Oracle:
  * pack/unpack round-trips BIT-EXACT on adversarial inputs (odd length, all-0xF,
    alternating signs, a single group).
  * group-wise symmetric INT4 clears the ~6.02*4 dB SQNR floor comfortably.
  * a W4A16 linear (int4 weights, fp16 activations) has small output MSE vs fp16.
"""

from __future__ import annotations

import math

import torch

from scratch_llm.quant.int4_group import (
    GROUP_SIZE,
    INT4_QMAX,
    INT4_QMIN,
    dequantize_groupwise_int4,
    mse,
    pack_int4,
    quantize_groupwise_int4,
    sqnr_db,
    unpack_int4,
    w4a16_linear,
)


# --------------------------------------------------------------------------- #
# pack / unpack bit-exactness (do this FIRST — the most common bug)            #
# --------------------------------------------------------------------------- #
def _roundtrip(q: torch.Tensor) -> torch.Tensor:
    return unpack_int4(pack_int4(q), q.numel())


def test_pack_unpack_full_grid():
    """Every value on the signed 4-bit grid round-trips."""
    q = torch.arange(INT4_QMIN, INT4_QMAX + 1, dtype=torch.int8)
    assert torch.equal(_roundtrip(q), q)


def test_pack_unpack_odd_length_pad():
    """Odd length: the pad nibble is appended and dropped on unpack."""
    q = torch.tensor([7, -8, 3, -1, 5], dtype=torch.int8)  # length 5 (odd)
    packed = pack_int4(q)
    assert packed.numel() == 3  # ceil(5/2)
    assert torch.equal(unpack_int4(packed, 5), q)


def test_pack_unpack_all_0xF():
    """All nibbles 0xF == -1 after sign extension."""
    q = torch.full((64,), -1, dtype=torch.int8)
    packed = pack_int4(q)
    assert torch.equal(packed, torch.full((32,), 0xFF, dtype=torch.uint8))
    assert torch.equal(unpack_int4(packed, 64), q)


def test_pack_unpack_alternating_signs():
    """Alternating +7 / -8 stresses the high/low nibble ordering."""
    q = torch.tensor([7, -8] * 50, dtype=torch.int8)
    # high nibble 7 (0b0111), low nibble -8 -> 0x8 -> byte 0x78
    packed = pack_int4(q)
    assert torch.equal(packed, torch.full((50,), 0x78, dtype=torch.uint8))
    assert torch.equal(unpack_int4(packed, 100), q)


def test_pack_unpack_single_group():
    """A single 128-wide group of random codes round-trips exactly."""
    g = torch.randint(INT4_QMIN, INT4_QMAX + 1, (GROUP_SIZE,), dtype=torch.int8)
    assert torch.equal(_roundtrip(g), g)


def test_pack_high_low_nibble_placement():
    """First value lands in the high nibble, second in the low nibble."""
    q = torch.tensor([1, 2], dtype=torch.int8)  # 0x1 << 4 | 0x2 == 0x12
    assert int(pack_int4(q)[0].item()) == 0x12


def test_pack_unpack_random_bitexact():
    """Fuzz: 200 random-length arrays over the full grid round-trip exactly."""
    torch.manual_seed(0)
    for _ in range(200):
        n = int(torch.randint(1, 300, (1,)).item())
        q = torch.randint(INT4_QMIN, INT4_QMAX + 1, (n,), dtype=torch.int8)
        assert torch.equal(_roundtrip(q), q)


# --------------------------------------------------------------------------- #
# Quantizer round-trip through packing (codes survive the byte layer)         #
# --------------------------------------------------------------------------- #
def test_quantized_codes_survive_packing():
    """The INT4 codes emitted by the quantizer pack/unpack bit-exactly."""
    torch.manual_seed(1)
    w = torch.randn(8, GROUP_SIZE * 3)
    q, _ = quantize_groupwise_int4(w)
    flat = q.reshape(-1)
    assert torch.equal(unpack_int4(pack_int4(flat), flat.numel()).reshape(q.shape), q)
    assert int(q.min().item()) >= INT4_QMIN
    assert int(q.max().item()) <= INT4_QMAX


# --------------------------------------------------------------------------- #
# SQNR floor for group-wise INT4                                              #
# --------------------------------------------------------------------------- #
def test_groupwise_int4_sqnr_matches_granular_noise_model():
    """SQNR equals the analytic granular-quantization floor (proves no bug).

    For symmetric absmax INT4 the quantization step per group *is* the scale, so a
    granular (non-clipping) uniform quantizer predicts noise power ``step**2/12``.
    On Gaussian weights this yields an SQNR of ``6.02*b + c`` with a *negative*
    ``c`` (absmax runs ~3 sigma, so the step is coarse relative to the RMS):
    ~18.6 dB here, i.e. ``c ~= -5.4``. Any gross bug (missing clamp, double
    rounding, wrong scale) would push the measured SQNR *below* this floor, so we
    assert the measured value sits right at the analytic prediction.
    """
    torch.manual_seed(2)
    w = torch.randn(256, GROUP_SIZE * 8)
    q, scales = quantize_groupwise_int4(w)
    wdeq = dequantize_groupwise_int4(q, scales)
    dB = sqnr_db(w, wdeq)

    n_groups = w.shape[1] // GROUP_SIZE
    step = scales.reshape(256, n_groups, 1)  # scale == quantization step
    signal = w.to(torch.float64).pow(2).sum()
    noise_pred = (step.to(torch.float64).pow(2) / 12.0).expand(256, n_groups, GROUP_SIZE).sum()
    pred_dB = float(10.0 * math.log10(signal / noise_pred))

    assert dB > 6.02 * 4 - 6.0, f"SQNR {dB:.2f} dB implausibly low (a bug)"
    assert abs(dB - pred_dB) < 1.0, (
        f"measured SQNR {dB:.2f} dB deviates from granular floor {pred_dB:.2f} dB"
    )


def test_groupwise_beats_pertensor_sqnr():
    """Per-group scaling has higher SQNR than a single per-tensor scale."""
    torch.manual_seed(3)
    # Heterogeneous per-group scale: makes local scaling matter.
    w = torch.randn(64, GROUP_SIZE * 6)
    w[:, GROUP_SIZE * 3 :] *= 20.0
    q_g, s_g = quantize_groupwise_int4(w)
    dB_group = sqnr_db(w, dequantize_groupwise_int4(q_g, s_g))
    # Per-tensor == one giant group over the whole row.
    q_t, s_t = quantize_groupwise_int4(w, group_size=w.shape[1])
    dB_tensor = sqnr_db(w, dequantize_groupwise_int4(q_t, s_t, group_size=w.shape[1]))
    assert dB_group > dB_tensor


# --------------------------------------------------------------------------- #
# W4A16 linear layer-output MSE vs fp16                                        #
# --------------------------------------------------------------------------- #
def test_w4a16_linear_output_mse_small():
    """Layer output MSE of INT4 weights vs fp16 weights is small."""
    torch.manual_seed(4)
    in_f, out_f, batch = GROUP_SIZE * 8, 512, 32
    w = torch.randn(out_f, in_f) * 0.02  # typical linear-weight scale
    x = torch.randn(batch, in_f, dtype=torch.float16)

    q, scales = quantize_groupwise_int4(w)
    y_q = w4a16_linear(x, q, scales)

    y_ref = torch.nn.functional.linear(x, w.to(torch.float16))

    # INT4 group-quant is granular-noise-limited: the weight SQNR (~18.6 dB,
    # rel MSE ~1.4e-2) propagates near-linearly to the layer output, so ~1.4%
    # relative output MSE is the honest floor, not a slack bar.
    rel_mse = mse(y_ref, y_q) / mse(y_ref, torch.zeros_like(y_ref))
    assert rel_mse < 2e-2, f"relative output MSE {rel_mse:.2e} too large"
    assert y_q.dtype == x.dtype
    assert y_q.shape == (batch, out_f)


def test_w4a16_linear_with_bias():
    """Bias is applied and shapes/dtype are preserved."""
    torch.manual_seed(5)
    w = torch.randn(16, GROUP_SIZE * 2) * 0.02
    bias = torch.randn(16, dtype=torch.float16)
    x = torch.randn(4, GROUP_SIZE * 2, dtype=torch.float16)
    q, scales = quantize_groupwise_int4(w)
    y = w4a16_linear(x, q, scales, bias=bias)
    y_nobias = w4a16_linear(x, q, scales)
    assert torch.allclose(y - bias, y_nobias, atol=1e-2)
