"""Tests for A5 Rung 0+1 — INT8 symmetric per-tensor and asymmetric affine + per-channel.

Oracle = the stated numeric laws, NOT allclose:
  * R0 symmetric round-trip SQNR on a clean Gaussian sits near the INT8 floor
    (~6.02*7 dB); the book example [1.2,-0.8,2.5,-1.7] reconstructs with the amax
    value exact and the rest off by ~0.01.
  * R1 asymmetric affine BEATS symmetric on a skewed (all-positive, heavy-tail) tensor.
  * Per-channel BEATS per-tensor.
  * A full W8A8 linear (INT8 W + INT8 X, integer matmul, dequant) matches fp32 within
    ~1e-2 relative.
Each test prints the ACTUAL measured dB / MSE / relative error.
"""

from __future__ import annotations

import math

import torch

from scratch_llm.quant.int8 import (
    dequantize_affine,
    dequantize_symmetric,
    fake_quant_affine,
    fake_quant_symmetric,
    mse,
    quantize_affine,
    quantize_symmetric,
    sqnr_db,
    w8a8_linear,
)

# INT8 symmetric uses 127 levels per side (~7.99 effective bits). The ideal full-scale floor
# is ~6.02*7.99 + 1.76 ~ 49 dB, but a Gaussian whose range is set by its amax pays a
# peak-to-average penalty and lands near ~40 dB. We assert a defensible floor and print actual.
_INT8_SQNR_FLOOR_DB = 38.0


def test_symmetric_round_trip_sqnr_near_floor() -> None:
    """Clean Gaussian symmetric round-trip should clear the INT8 SQNR floor."""
    torch.manual_seed(0)
    x = torch.randn(4096)
    x_hat = fake_quant_symmetric(x)
    db = sqnr_db(x, x_hat)
    print(f"\n[R0 sym] Gaussian round-trip SQNR = {db:.2f} dB (floor {_INT8_SQNR_FLOOR_DB})")
    assert db > _INT8_SQNR_FLOOR_DB, f"SQNR {db:.2f} dB below floor -> scale/clamp/round bug"
    # Sanity: 6.02*bits growth law — dropping to ~4-bit codes must lose ~ (7-4)*6 ~ 18 dB.
    q4 = torch.round(x / (x.abs().max() / 7)).clamp(-7, 7)
    db4 = sqnr_db(x, q4 * (x.abs().max() / 7))
    print(f"[R0 sym] 3-bit-equivalent SQNR = {db4:.2f} dB (delta {db - db4:.1f} dB)")
    assert db - db4 > 12.0


def test_symmetric_book_example() -> None:
    """Reproduce the textbook example: amax value exact, others off ~0.01."""
    x = torch.tensor([1.2, -0.8, 2.5, -1.7])
    q, s = quantize_symmetric(x)
    x_hat = dequantize_symmetric(q, s)
    err = (x - x_hat).abs()
    print(f"\n[R0 book] codes={q.tolist()} s={s.item():.6f}")
    print(f"[R0 book] x_hat={[round(v, 4) for v in x_hat.tolist()]}")
    print(f"[R0 book] |err|={[round(v, 4) for v in err.tolist()]}")
    # amax value (2.5) maps to code 127 -> reconstructed exactly.
    assert err[2].item() == 0.0
    # 1.2 lands almost exactly on a grid point (~exact); its error is far below one step.
    assert err[0].item() < 1e-3
    # The remaining two are genuinely quantized, off by ~0.01 (< half a step ~0.0098).
    assert 1e-3 < err[1].item() < 1.5e-2
    assert 1e-3 < err[3].item() < 1.5e-2
    # No error may exceed half a quantization step.
    assert err.max().item() <= s.item() / 2 + 1e-9


def test_affine_beats_symmetric_on_skewed() -> None:
    """On an all-positive heavy-tailed (post-activation-like) tensor, asym beats sym."""
    torch.manual_seed(1)
    x = torch.relu(torch.randn(8192)) ** 1.5  # all-positive, heavy right tail

    sym_hat = fake_quant_symmetric(x)
    aff_hat = fake_quant_affine(x)
    db_sym = sqnr_db(x, sym_hat)
    db_aff = sqnr_db(x, aff_hat)
    print(f"\n[R1 skew] sym SQNR = {db_sym:.2f} dB | asym SQNR = {db_aff:.2f} dB")
    print(f"[R1 skew] asym advantage = {db_aff - db_sym:.2f} dB")
    # Symmetric wastes the negative half of the range on data that is never negative.
    assert db_aff > db_sym + 3.0
    assert mse(x, aff_hat) < mse(x, sym_hat)


def test_affine_represents_zero_exactly() -> None:
    """The affine zero-point makes real 0.0 reconstruct exactly (ReLU/pad sparsity)."""
    torch.manual_seed(2)
    x = torch.relu(torch.randn(1000))  # lots of exact zeros
    q, s, z = quantize_affine(x)
    x_hat = dequantize_affine(q, s, z)
    zeros = x == 0.0
    max_zero_err = (x_hat[zeros]).abs().max().item()
    print(
        f"\n[R1 zero] z={z.item():.0f} max reconstruction error at true-zero = {max_zero_err:.2e}"
    )
    assert max_zero_err == 0.0


def test_per_channel_beats_per_tensor() -> None:
    """Per-channel (per-row) scales beat a single per-tensor scale when channel ranges differ."""
    torch.manual_seed(3)
    # 32 output channels whose amax spans two orders of magnitude -> per-tensor is hostage to
    # the largest channel; small channels get crushed.
    scales = torch.logspace(-2, 1, 32).unsqueeze(1)  # [32,1]
    w = torch.randn(32, 512) * scales

    per_tensor = fake_quant_symmetric(w, axis=None)
    per_channel = fake_quant_symmetric(w, axis=0)  # one scale per output row
    db_pt = sqnr_db(w, per_tensor)
    db_pc = sqnr_db(w, per_channel)
    print(f"\n[R1 per-ch] per-tensor SQNR = {db_pt:.2f} dB | per-channel SQNR = {db_pc:.2f} dB")
    print(f"[R1 per-ch] per-channel advantage = {db_pc - db_pt:.2f} dB")
    assert db_pc > db_pt + 6.0
    assert mse(w, per_channel) < mse(w, per_tensor)


def test_w8a8_linear_matches_fp32() -> None:
    """Integer-GEMM W8A8 linear matches the fp32 reference within ~1e-2 relative error."""
    torch.manual_seed(4)
    m, k, n = 64, 256, 128
    x = torch.randn(m, k)
    w = torch.randn(n, k) * torch.logspace(-1, 1, n).unsqueeze(1)  # varied channel ranges

    y_ref = x @ w.t()
    y_q = w8a8_linear(x, w)

    rel = (y_q - y_ref).norm() / y_ref.norm()
    db = sqnr_db(y_ref, y_q)
    print(f"\n[W8A8] relative error = {rel.item():.4e} | output SQNR = {db:.2f} dB")
    # ~1e-2 is the calibrated expectation for INT8xINT8 over a K=256 contraction; measured
    # sits right at ~1.0e-2, so we allow a small band around it (this is genuine quant noise,
    # not a scale/round bug — the output SQNR ~40 dB confirms the accumulator dequant is right).
    assert rel.item() < 1.5e-2, f"W8A8 relative error {rel.item():.2e} exceeds ~1e-2 band"
    # Batched input should carry through the leading dims unchanged.
    xb = x.reshape(8, 8, k)
    yb = w8a8_linear(xb, w)
    assert yb.shape == (8, 8, n)
    assert torch.allclose(yb.reshape(m, n), y_q, atol=0, rtol=0)


def test_sqnr_growth_law() -> None:
    """SQNR grows ~6.02 dB per bit: 6-bit codes beat 3-bit codes by ~3*6 dB on a Gaussian."""
    torch.manual_seed(5)
    x = torch.randn(16384)
    amax = x.abs().max()

    def sqnr_at_bits(bits: int) -> float:
        qmax = 2 ** (bits - 1) - 1
        s = amax / qmax
        q = torch.round(x / s).clamp(-qmax, qmax)
        return sqnr_db(x, q * s)

    db6 = sqnr_at_bits(6)
    db3 = sqnr_at_bits(3)
    per_bit = (db6 - db3) / 3.0
    print(f"\n[law] SQNR@6bit={db6:.2f} @3bit={db3:.2f} -> {per_bit:.2f} dB/bit (ideal ~6.02)")
    assert math.isclose(per_bit, 6.02, abs_tol=1.5)
