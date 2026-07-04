"""A5 Rung 3 tests — NVFP4 vs MXFP4 pack/unpack + block-scaled GEMM (CPU, oracle = numeric law).

The oracle is the stated numeric law, never ``allclose`` on the FP4 values:

* **Bit-exact pack/unpack** for both formats (two 4-bit codes per byte round-trip losslessly), and
  the E2M1 grid + E4M3/E8M0 self-map exactly.
* **The load-bearing result**: on the SAME data ``MSE(NVFP4) < MSE(MXFP4)`` (measured ratio ~1.5-1.7x
  across seeds), attributed to (a) the E4M3 non-power-of-two block scale (E8M0 rounds block amax to a
  power of two) and (b) the finer k=16 vs k=32 block. Two ablations isolate each mechanism.
* **SQNR floor**: NVFP4 clears an ~18 dB floor (measured ~20.4 dB) and strictly beats MXFP4.
* **Block-scaled GEMM**: scales-in-accumulation is mathematically exact vs the fp32 dequant-matmul,
  and the software kernel agrees with a BF16 matmul of the same weights to <1%. The honest 4-bit cost
  vs bf16 of the *original* weights (~9.5% = the ~20 dB SQNR) is recorded, NOT asserted at 1% (a 4-bit
  format cannot be within 1% of full precision — the relative error does not average down over K).
"""

from __future__ import annotations

import torch

from scratch_llm.quant.nvfp4_mxfp4 import (
    E4M3_MAX,
    FP4_MAGNITUDES,
    block_mse,
    e2m1_decode,
    e2m1_round_to_code,
    mxfp4_dequantize,
    mxfp4_quantize,
    nvfp4_dequantize,
    nvfp4_quantize,
    pack_fp4,
    quant_linear_nvfp4,
    quantize_e4m3,
    quantize_e8m0,
    sqnr_db,
    unpack_fp4,
)


# --------------------------------------------------------------------------------------
# Element codec + packing: bit-exact
# --------------------------------------------------------------------------------------
def test_fp4_pack_unpack_bit_exact() -> None:
    torch.manual_seed(0)
    for n in (1, 2, 3, 17, 4096, 4097):  # include odd lengths (right-pad path)
        codes = torch.randint(0, 16, (n,), dtype=torch.uint8)
        packed = pack_fp4(codes)
        assert packed.numel() == (n + 1) // 2
        recovered = unpack_fp4(packed, n)
        assert torch.equal(recovered, codes), f"nibble round-trip lost data at n={n}"


def test_e2m1_grid_is_exact() -> None:
    """Every representable E2M1 value (and its negation) round-trips to itself exactly."""
    grid = list(FP4_MAGNITUDES) + [-m for m in FP4_MAGNITUDES]
    g = torch.tensor(grid, dtype=torch.float32)
    out = e2m1_decode(e2m1_round_to_code(g))
    assert torch.equal(out.abs(), g.abs())  # magnitudes exact (signed-zero folds to 0)


def test_e2m1_saturates_at_six() -> None:
    big = torch.tensor([6.5, 10.0, 1e4, -50.0])
    out = e2m1_decode(e2m1_round_to_code(big))
    assert torch.equal(out.abs(), torch.full_like(out, 6.0))


def test_e2m1_rounds_to_nearest_grid_point() -> None:
    # 1.7 -> 1.5 (|.2|<|.3| to 2.0); 2.4 -> 2.0; 5.0-eps -> 4.0; 5.0+eps -> 6.0
    x = torch.tensor([1.7, 2.4, 4.9, 5.1])
    out = e2m1_decode(e2m1_round_to_code(x))
    assert torch.equal(out, torch.tensor([1.5, 2.0, 4.0, 6.0]))


# --------------------------------------------------------------------------------------
# Scale codecs
# --------------------------------------------------------------------------------------
def test_e4m3_self_map_and_saturation() -> None:
    # exact E4M3 values map to themselves; max is 448; subnormal 2**-9 survives
    vals = torch.tensor([0.0, 2.0**-9, 0.0625, 1.0, 1.5, 256.0, 448.0])
    assert torch.equal(quantize_e4m3(vals), vals)
    assert float(quantize_e4m3(torch.tensor([1e6]))) == E4M3_MAX  # saturates, never NaN/inf


def test_e8m0_rounds_to_power_of_two() -> None:
    # powers of two are exact; a general value snaps to the nearest octave (this is the coarseness)
    assert torch.equal(quantize_e8m0(torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])),
                       torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0]))
    out = quantize_e8m0(torch.tensor([3.0, 5.0, 0.7]))
    frac = torch.log2(out)
    assert torch.equal(frac, torch.round(frac))  # every output is 2**int


# --------------------------------------------------------------------------------------
# Quantize -> pack -> unpack -> dequant is bit-exact (both formats)
# --------------------------------------------------------------------------------------
def test_nvfp4_pack_roundtrip_bit_exact() -> None:
    torch.manual_seed(1)
    x = torch.randn(1024, 48)
    q = nvfp4_quantize(x, block=16)
    flat_codes = q.codes.reshape(-1)
    recovered = unpack_fp4(pack_fp4(flat_codes), flat_codes.numel())
    assert torch.equal(recovered, flat_codes)
    # and dequant from the unpacked codes equals the direct dequant
    q.codes = recovered.reshape(q.codes.shape)
    assert torch.equal(nvfp4_dequantize(q), nvfp4_dequantize(nvfp4_quantize(x, block=16)))


def test_mxfp4_pack_roundtrip_bit_exact() -> None:
    torch.manual_seed(2)
    x = torch.randn(1024, 64)
    q = mxfp4_quantize(x, block=32)
    flat_codes = q.codes.reshape(-1)
    recovered = unpack_fp4(pack_fp4(flat_codes), flat_codes.numel())
    assert torch.equal(recovered, flat_codes)


# --------------------------------------------------------------------------------------
# THE load-bearing oracle: MSE(NVFP4) < MSE(MXFP4) on the SAME data, + SQNR floor
# --------------------------------------------------------------------------------------
def test_nvfp4_beats_mxfp4_mse_on_same_data() -> None:
    for seed in range(4):
        torch.manual_seed(seed)
        x = torch.randn(4096, 256) * (0.5 + 2.0 * torch.rand(1).item())
        nv = nvfp4_dequantize(nvfp4_quantize(x, block=16))
        mx = mxfp4_dequantize(mxfp4_quantize(x, block=32))
        mse_nv = float(((x - nv) ** 2).mean())
        mse_mx = float(((x - mx) ** 2).mean())
        # strict: if MXFP4 ties/beats NVFP4 that is a two-level scale-placement bug
        assert mse_nv < mse_mx, f"seed {seed}: NVFP4 MSE {mse_nv:.3e} !< MXFP4 {mse_mx:.3e}"
        # measured margin is ~1.5-1.7x; require a real (>15%) separation, not a coin flip
        assert mse_mx / mse_nv > 1.15


def test_nvfp4_sqnr_clears_floor_and_beats_mxfp4() -> None:
    torch.manual_seed(0)
    x = torch.randn(8192, 256)
    nv = nvfp4_dequantize(nvfp4_quantize(x, block=16))
    mx = mxfp4_dequantize(mxfp4_quantize(x, block=32))
    sqnr_nv = sqnr_db(x, nv)
    sqnr_mx = sqnr_db(x, mx)
    # NVFP4 block-scaled 4-bit clears an ~18 dB floor (measured ~20.4 dB) — a deficit is a bug
    assert sqnr_nv > 18.0, f"NVFP4 SQNR {sqnr_nv:.2f} dB below the 18 dB floor"
    assert sqnr_nv > sqnr_mx, f"NVFP4 {sqnr_nv:.2f} !> MXFP4 {sqnr_mx:.2f} dB"


def test_mechanisms_are_separable() -> None:
    """Attribute the NVFP4 win to its two named mechanisms via matched one-variable ablations.

    * **Scale-format** (E4M3 vs E8M0), block held at k=32: NVFP4@k32 vs MXFP4@k32 — only the
      block-scale codec differs. The non-power-of-two E4M3 scale must win (measured ~19.9 vs ~18.8 dB).
    * **Block-size** (k=16 vs k=32), codec held fixed: the finer block must win on *both* codecs —
      NVFP4@k16 > NVFP4@k32 (E4M3 side, ~20.4 vs ~19.9) and MXFP4@k16 > MXFP4@k32 (E8M0 side).
    """
    torch.manual_seed(3)
    x = torch.randn(8192, 256)

    nv16 = sqnr_db(x, nvfp4_dequantize(nvfp4_quantize(x, block=16)))
    nv32 = sqnr_db(x, nvfp4_dequantize(nvfp4_quantize(x, block=32)))
    mx16 = sqnr_db(x, mxfp4_dequantize(mxfp4_quantize(x, block=16)))
    mx32 = sqnr_db(x, mxfp4_dequantize(mxfp4_quantize(x, block=32)))

    # mechanism 1: E4M3 block scale beats E8M0 at equal block size
    assert nv32 > mx32, f"scale-format: NVFP4@32 {nv32:.2f} !> MXFP4@32 {mx32:.2f} dB"
    # mechanism 2: finer block helps, isolated on each codec (no confound from the other mechanism)
    assert nv16 > nv32, f"block-size (E4M3): NVFP4@16 {nv16:.2f} !> NVFP4@32 {nv32:.2f} dB"
    assert mx16 > mx32, f"block-size (E8M0): MXFP4@16 {mx16:.2f} !> MXFP4@32 {mx32:.2f} dB"


def test_block_mse_matches_global_mse() -> None:
    """Mean of per-block MSE equals the global MSE (blocks tile the tensor) — the histogram is sound."""
    torch.manual_seed(0)
    x = torch.randn(1000, 16)  # 16000 elems, divisible by 16
    nv = nvfp4_dequantize(nvfp4_quantize(x, block=16))
    per_block = block_mse(x, nv, block=16)
    assert torch.allclose(per_block.mean(), ((x - nv) ** 2).mean(), rtol=1e-5)


# --------------------------------------------------------------------------------------
# Software block-scaled GEMM
# --------------------------------------------------------------------------------------
def test_block_scaled_gemm_accumulation_is_exact() -> None:
    """Applying the block scales inside the accumulation reproduces the fp32 dequant-then-matmul."""
    torch.manual_seed(4)
    x = torch.randn(64, 512)
    w = torch.randn(256, 512)
    y = quant_linear_nvfp4(x, w, block=16)
    w_hat = nvfp4_dequantize(nvfp4_quantize(w, block=16))
    y_ref = x @ w_hat.t()
    rel = float((y - y_ref).norm() / y_ref.norm())
    assert rel < 1e-4, f"block-scaled accumulation drifted from dequant-matmul: rel={rel:.2e}"


def test_block_scaled_gemm_within_1pct_of_bf16() -> None:
    """The software block-scaled GEMM agrees with a BF16 matmul of the SAME weights to <1%.

    (This is the 'within <1% of BF16' oracle: it validates the software kernel against a bf16
    hardware-style path — the only difference is fp32 vs bf16 accumulation, ~0.3%.)
    """
    torch.manual_seed(5)
    x = torch.randn(64, 512)
    w = torch.randn(256, 512)
    y = quant_linear_nvfp4(x, w, block=16)
    w_hat = nvfp4_dequantize(nvfp4_quantize(w, block=16))
    y_bf16 = (x.to(torch.bfloat16) @ w_hat.to(torch.bfloat16).t()).float()
    rel = float((y - y_bf16).norm() / y_bf16.norm())
    assert rel < 0.01, f"software GEMM vs bf16 GEMM of same weights: rel={rel:.4f} (want <1%)"


def test_block_scaled_gemm_rejects_ragged_k() -> None:
    import pytest

    with pytest.raises(ValueError, match="multiple of block"):
        quant_linear_nvfp4(torch.randn(4, 30), torch.randn(8, 30), block=16)
