"""AWQ — Activation-aware Weight Quantization for W4A16 linears (A5 §4.3, arXiv:2306.00978).

This is a *real* post-training quantization (PTQ) method: it recovers accuracy over naive
round-to-nearest INT4 **at the identical bit-width and group size**, by choosing *where* to spend
the grid's dynamic range instead of adding bits.

Mechanism (why protecting salient-by-**activation** channels recovers accuracy)
------------------------------------------------------------------------------
A linear computes ``y = x @ Wᵀ``. Weight-quantization error propagates to the output as
``y_q - y = x @ (W_deq - W)ᵀ``; the contribution of input channel ``j`` is *weighted by the
activation magnitude* ``x_j``. So a handful of channels driven by large-magnitude activations
(the LLM "activation outlier" channels) dominate the output error even when their *weights* are
perfectly ordinary — round-to-nearest, which treats every channel identically, spends its error
budget in the wrong place. AWQ exploits an exact algebraic identity: for any positive per-input-
channel scale ``s``, ``(x / s) @ (W · s)ᵀ == x @ Wᵀ``. It scales the salient weight *columns* UP
before rounding (and folds the matching ``1/s`` into the activations). Scaling a column up moves
it higher on the group's signed-INT4 grid, so its **relative** rounding error shrinks — while the
group step ``absmax/7`` barely moves, because the many non-salient channels in the group still set
the group's absmax. After the ``1/s`` is folded back, that column's quantization noise is divided
by ``s``, cutting its (activation-amplified) share of the output error by ``~s²``. Non-salient
columns are scaled slightly *down* by the geometric-mean normalization and carry a little more
error, but their tiny activations make that negligible. The single knob is the exponent ``α`` in
``s = act_scaleᵅ``: too small protects nothing, too large inflates every group's step and hurts the
whole group. AWQ grid-searches ``α ∈ [0, 1]`` to minimize the *measured* calibration output MSE —
no gradients, no retraining. ``α = 0`` is exactly naive INT4 (``s ≡ 1``), so the search can never
do worse than the baseline on the calibration set.

Oracle / invariants (the tests)
-------------------------------
1. **AWQ beats naive at the SAME group size / bit-width.** On a Linear with realistic weights and a
   calibration set that has a few high-magnitude "salient" activation channels, AWQ INT4 achieves
   *measurably lower* layer-output MSE (vs the fp32 reference output) than round-to-nearest INT4.
   The recovery factor ``MSE_naive / MSE_awq`` is reported and asserted ``> 1``.
2. **Generalization.** The scale is searched on a calibration split and the win still holds on a
   *held-out* activation split — the protection transfers, it does not overfit the calib tokens.
3. **α = 0 identity.** With ``ratio = 0`` the AWQ scale is all-ones and reproduces naive INT4
   bit-for-bit, so the grid search is a strict improvement over the baseline.

Honesty note: this is a **layer-level MSE demonstration** on a synthetic-but-realistic Linear
(realistic weight scale + planted activation outliers). It is *not* a full-model perplexity run —
that is a rental-gated SKIP per the curriculum. The reported numbers are the actual measured MSEs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from scratch_llm.quant.int4_group import (
    GROUP_SIZE,
    dequantize_groupwise_int4,
    mse,
    quantize_groupwise_int4,
)

__all__ = [
    "AWQQuant",
    "AWQSearchResult",
    "act_scale",
    "awq_dequantize",
    "awq_linear",
    "quantize_awq_int4",
    "search_awq_scale",
]


@dataclass(frozen=True)
class AWQQuant:
    """An AWQ-quantized W4A16 weight.

    ``q``/``group_scales`` are the group-INT4 codes of the *scaled* weight ``W · chan_scale``;
    ``chan_scale`` is the per-input-channel AWQ scale ``s`` (shape ``[in]``) that must be folded
    back as ``1/s`` at inference. The effective dequantized weight is
    ``dequant(q, group_scales) / chan_scale`` (see :func:`awq_dequantize`).
    """

    q: Tensor
    group_scales: Tensor
    chan_scale: Tensor
    group_size: int


@dataclass(frozen=True)
class AWQSearchResult:
    """Result of the AWQ scale search: the chosen scale, its grid exponent, and calib MSE."""

    chan_scale: Tensor
    ratio: float
    calib_mse: float


def act_scale(x: Tensor) -> Tensor:
    """Per-input-channel activation magnitude ``E[|x_j|]`` — AWQ's saliency signal.

    ``x`` is ``[..., in]`` (any number of leading token dims); returns a ``[in]`` fp32 vector of
    the mean absolute activation per input channel. Large entries mark the salient channels whose
    weight columns AWQ protects.
    """
    return x.to(torch.float32).abs().reshape(-1, x.shape[-1]).mean(dim=0)


def _normalize_scale(s: Tensor) -> Tensor:
    """Geometric-mean-center a positive scale so it neither inflates nor deflates overall magnitude.

    ``s / sqrt(max(s) · min(s))`` pins the geometric midpoint of the scale to 1, keeping the group
    absmaxes (and hence the average INT4 step) roughly unchanged: salient channels move *up*,
    non-salient channels move *down*, and the search only decides how far.
    """
    s = s.clamp_min(1e-4)
    return s / (s.max() * s.min()).sqrt()


def _awq_dequant_weight(w: Tensor, chan_scale: Tensor, group_size: int) -> Tensor:
    """Quantize ``W · s`` to group-INT4, dequantize, and fold ``1/s`` back → effective fp weight."""
    w_scaled = w.to(torch.float32) * chan_scale[None, :]
    q, group_scales = quantize_groupwise_int4(w_scaled, group_size)
    return dequantize_groupwise_int4(q, group_scales, group_size) / chan_scale[None, :]


def search_awq_scale(
    w: Tensor,
    x: Tensor,
    group_size: int = GROUP_SIZE,
    grid: int = 20,
) -> AWQSearchResult:
    """Grid-search the AWQ per-input-channel scale ``s = act_scale(x)ᵅ`` minimizing output MSE.

    For each ``α = i/grid`` over ``i ∈ [0, grid]`` it builds the normalized scale, quantizes the
    scaled weight to group-INT4, folds ``1/s`` back, and measures the layer-output MSE against the
    fp32 reference ``x @ Wᵀ`` on the calibration activations ``x``. Returns the best-scoring scale.
    ``α = 0`` (``s ≡ 1``) is naive INT4, so the result never scores worse than the baseline here.
    """
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(w.shape)}")
    wf = w.to(torch.float32)
    xf = x.to(torch.float32)
    y_ref = xf @ wf.t()
    a = act_scale(xf)

    best: AWQSearchResult | None = None
    for i in range(grid + 1):
        ratio = i / grid
        s = _normalize_scale(a.pow(ratio))
        w_eff = _awq_dequant_weight(wf, s, group_size)
        cur = mse(y_ref, xf @ w_eff.t())
        if best is None or cur < best.calib_mse:
            best = AWQSearchResult(chan_scale=s, ratio=ratio, calib_mse=cur)
    assert best is not None  # grid >= 0 guarantees at least one iteration
    return best


def quantize_awq_int4(
    w: Tensor,
    x: Tensor | None = None,
    chan_scale: Tensor | None = None,
    group_size: int = GROUP_SIZE,
    grid: int = 20,
) -> AWQQuant:
    """AWQ-quantize a weight ``[out, in]`` to W4A16.

    Provide calibration activations ``x`` (``[..., in]``) to search the scale, or pass a
    precomputed ``chan_scale`` (``[in]``) to reuse one. Returns an :class:`AWQQuant` whose codes are
    the group-INT4 quantization of the *scaled* weight ``W · s``.
    """
    if chan_scale is not None:
        scale = chan_scale.to(torch.float32)
    elif x is not None:
        scale = search_awq_scale(w, x, group_size=group_size, grid=grid).chan_scale
    else:
        raise ValueError("provide either calibration activations `x` or a `chan_scale`")
    w_scaled = w.to(torch.float32) * scale[None, :]
    q, group_scales = quantize_groupwise_int4(w_scaled, group_size)
    return AWQQuant(q=q, group_scales=group_scales, chan_scale=scale, group_size=group_size)


def awq_dequantize(quant: AWQQuant) -> Tensor:
    """Reconstruct the effective fp32 weight: ``dequant(q, group_scales) / chan_scale``."""
    w_scaled_deq = dequantize_groupwise_int4(quant.q, quant.group_scales, quant.group_size)
    return w_scaled_deq / quant.chan_scale[None, :]


def awq_linear(x: Tensor, quant: AWQQuant, bias: Tensor | None = None) -> Tensor:
    """W4A16 linear with an AWQ-quantized weight: ``y = x @ W_effᵀ (+ bias)``.

    Reconstructs the effective weight (INT4-scaled-dequant with ``1/s`` folded back) in ``x``'s
    dtype and runs the matmul there — the accumulation semantics an AWQ/Marlin kernel emulates.
    ``x`` is ``[..., in]``; returns ``[..., out]`` in ``x``'s dtype.
    """
    w_eff = awq_dequantize(quant).to(x.dtype)
    return torch.nn.functional.linear(x, w_eff, bias)
