"""Tests for AWQ (Activation-aware Weight Quantization), A5 §4.3 (arXiv:2306.00978).

The LOAD-BEARING oracle: on a realistic Linear whose calibration activations have a few
high-magnitude "salient" input channels, AWQ INT4 must achieve MEASURABLY LOWER layer-output MSE
(vs the fp32 reference) than NAIVE round-to-nearest INT4 **at the same group size / bit-width**.
This is a layer-level MSE demonstration on a synthetic-but-realistic layer — not a full-model
perplexity run (that is a rental-gated SKIP). All reported numbers are the actual measured MSEs.
"""

from __future__ import annotations

import torch

from scratch_llm.quant.awq import (
    act_scale,
    awq_dequantize,
    awq_linear,
    quantize_awq_int4,
    search_awq_scale,
)
from scratch_llm.quant.int4_group import (
    GROUP_SIZE,
    dequantize_groupwise_int4,
    mse,
    quantize_groupwise_int4,
)

IN_FEATURES = GROUP_SIZE * 4  # 512 → 4 groups
OUT_FEATURES = 256
N_SALIENT = 8  # ~1.5% of channels are salient by activation
SALIENT_MULT = 12.0


def _make_layer(
    seed: int,
    n_tokens: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Realistic Linear + calibration activations with planted salient channels.

    Weights ~ N(0, 0.02²) (a typical linear-weight scale, magnitudes uncorrelated with saliency);
    activations ~ N(0, 1) except ``N_SALIENT`` randomly chosen input channels are boosted ×12 —
    the LLM "activation outlier" channels AWQ is designed to protect.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(OUT_FEATURES, IN_FEATURES, generator=g) * 0.02
    x = torch.randn(n_tokens, IN_FEATURES, generator=g)
    salient = torch.randperm(IN_FEATURES, generator=g)[:N_SALIENT]
    x[:, salient] *= SALIENT_MULT
    return w, x, salient


def _naive_int4_output(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    q, scales = quantize_groupwise_int4(w, GROUP_SIZE)
    w_deq = dequantize_groupwise_int4(q, scales, GROUP_SIZE)
    return x @ w_deq.t()


# --------------------------------------------------------------------------- #
# THE load-bearing oracle: AWQ beats naive at the SAME bit-width / group size  #
# --------------------------------------------------------------------------- #
def test_awq_beats_naive_int4_same_bitwidth():
    """AWQ INT4 output MSE < naive round-to-nearest INT4, same group size, on salient-channel data."""
    w, x, _ = _make_layer(seed=0)
    y_ref = x @ w.t()

    mse_naive = mse(y_ref, _naive_int4_output(w, x))

    quant = quantize_awq_int4(w, x=x, group_size=GROUP_SIZE)
    mse_awq = mse(y_ref, awq_linear(x, quant))
    recovery = mse_naive / mse_awq

    # Both quantizers use identical INT4 grid + group_size=128; only the scale placement differs.
    assert mse_awq < mse_naive, f"AWQ MSE {mse_awq:.4e} did not beat naive {mse_naive:.4e}"
    assert recovery > 1.4, f"recovery factor {recovery:.2f}x below the demonstrated ~1.6x"


def test_awq_scale_generalizes_to_heldout_activations():
    """Scale searched on a calibration split still wins on a held-out activation split."""
    w, x, _ = _make_layer(seed=1, n_tokens=512)
    x_calib, x_eval = x[:256], x[256:]
    y_ref_eval = x_eval @ w.t()

    mse_naive = mse(y_ref_eval, _naive_int4_output(w, x_eval))

    # Search the AWQ scale on calib ONLY, then measure on held-out eval tokens.
    quant = quantize_awq_int4(w, x=x_calib, group_size=GROUP_SIZE)
    mse_awq = mse(y_ref_eval, awq_linear(x_eval, quant))

    assert mse_awq < mse_naive, (
        f"AWQ overfit the calib split: eval MSE {mse_awq:.4e} !< naive {mse_naive:.4e}"
    )
    assert mse_naive / mse_awq > 1.3, "held-out recovery below the demonstrated ~1.6x"


def test_awq_wins_across_seeds():
    """The win is not a lucky seed: AWQ beats naive on every one of several layers."""
    for seed in range(5):
        w, x, _ = _make_layer(seed=seed)
        y_ref = x @ w.t()
        mse_naive = mse(y_ref, _naive_int4_output(w, x))
        quant = quantize_awq_int4(w, x=x, group_size=GROUP_SIZE)
        mse_awq = mse(y_ref, awq_linear(x, quant))
        assert mse_awq < mse_naive, f"seed {seed}: AWQ {mse_awq:.4e} !< naive {mse_naive:.4e}"


# --------------------------------------------------------------------------- #
# Mechanism sanity: the scale search recovers the α=0 identity + protects salient #
# --------------------------------------------------------------------------- #
def test_ratio_zero_reproduces_naive_bit_exact():
    """α=0 ⇒ scale ≡ 1 ⇒ AWQ codes equal naive INT4 codes bit-for-bit (search can't lose)."""
    w, x, _ = _make_layer(seed=2)
    ones = torch.ones(IN_FEATURES)
    quant0 = quantize_awq_int4(w, chan_scale=ones, group_size=GROUP_SIZE)
    q_naive, _ = quantize_groupwise_int4(w, GROUP_SIZE)
    assert torch.equal(quant0.q, q_naive)


def test_search_scale_lifts_salient_channels():
    """The searched scale is > 1 on salient channels and the chosen α is interior (0<α<1)."""
    w, x, salient = _make_layer(seed=3)
    res = search_awq_scale(w, x, group_size=GROUP_SIZE)
    a = act_scale(x)
    # salient channels have the largest activation magnitude → largest scale
    assert res.chan_scale[salient].mean() > res.chan_scale.median()
    assert a[salient].mean() > a.median()
    # the optimum is a genuine trade-off, not an endpoint (endpoints = naive / over-scaled)
    assert 0.0 < res.ratio < 1.0, f"degenerate optimum at ratio={res.ratio}"


def test_awq_dequantize_matches_linear_forward():
    """`awq_linear` == matmul against the reconstructed effective weight (consistency check)."""
    w, x, _ = _make_layer(seed=4)
    quant = quantize_awq_int4(w, x=x, group_size=GROUP_SIZE)
    w_eff = awq_dequantize(quant)
    y_direct = x @ w_eff.t()
    y_linear = awq_linear(x, quant)
    assert torch.allclose(y_direct, y_linear, atol=1e-5)


def test_awq_linear_bias_shape_dtype():
    """Bias is applied; output shape/dtype follow the activation (fp16 W4A16 path)."""
    w, x_fp32, _ = _make_layer(seed=5, n_tokens=32)
    x = x_fp32.to(torch.float16)
    bias = torch.randn(OUT_FEATURES, dtype=torch.float16)
    quant = quantize_awq_int4(w, x=x_fp32, group_size=GROUP_SIZE)
    y = awq_linear(x, quant, bias=bias)
    y_nobias = awq_linear(x, quant)
    assert y.shape == (32, OUT_FEATURES)
    assert y.dtype == torch.float16
    assert torch.allclose(y - bias, y_nobias, atol=1e-2)
