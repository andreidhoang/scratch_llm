"""A5 Rung 0+1 — INT8 fake-quant: symmetric per-tensor and asymmetric affine.

This module implements the two foundational quantization schemes of the A5 ladder as
*fake-quant* (quantize then immediately dequantize back to float, so downstream code stays
in fp32). Everything is CPU-buildable torch — no GPU, no custom kernels.

Numeric law (the oracle these routines are graded against)
-----------------------------------------------------------
Quantization replaces a real value with the nearest point on a uniform grid of step ``s``.
The round-off is bounded by ``s/2`` and behaves like additive noise with variance ``s**2/12``
(uniform over one step). The signal-to-quantization-noise ratio is

    SQNR_dB = 10 * log10( sum(x**2) / sum((x - x_hat)**2) )

Each extra bit halves ``s`` and so quadruples the SNR power ratio, i.e. buys ``10*log10(4) =
6.02 dB``. Hence the classic floor ``SQNR ~ 6.02 * bits + c``. The constant ``c`` depends on
how the full-scale range is chosen relative to the signal: the textbook ``+1.76 dB`` assumes a
full-scale sinusoid, while a Gaussian tensor whose range is set by its (outlier) amax pays a
peak-to-average penalty and lands a few dB lower. So for symmetric INT8 (levels -127..127, an
effective ~7.99 bits) a clean Gaussian sits near ~40 dB, not the ideal ~44 dB — we report the
*measured* dB and only ever treat a value that falls *below* the achievable floor as a bug
(bad scale, missing clamp, wrong rounding), never as "acceptable loss".

Symmetric (R0) vs. asymmetric affine (R1)
-----------------------------------------
Symmetric maps ``[-amax, +amax]`` onto ``[-127, +127]`` with zero-point pinned at 0. On data
that is *not* centered — e.g. a post-GELU / post-ReLU activation that is all-positive with a
heavy tail — symmetric wastes the entire negative half of the code range, doubling the step and
losing ~6 dB. Asymmetric affine fits ``[min, max]`` onto the full ``[0, 255]`` with a learned
integer zero-point ``z``, recovering that range and beating symmetric on skewed tensors.

Per-channel / per-token
-----------------------
A single per-tensor scale is hostage to the worst channel's amax. Giving each output row (weight)
or each token row (activation) its own scale shrinks every step to that row's true range, so
per-channel strictly beats per-tensor in MSE / SQNR.

Why the per-channel weight scale factors OUT of the GEMM
--------------------------------------------------------
A linear layer computes ``Y[m, n] = sum_k X[m, k] * W[n, k]`` (contraction over ``k``). With
per-token activation scales ``s_X[m]`` (constant along ``k``) and per-channel weight scales
``s_W[n]`` (also constant along ``k``, since each output channel ``n`` owns one scale shared by
all its ``k`` weights):

    Y[m, n] = sum_k (q_X[m, k] * s_X[m]) * (q_W[n, k] * s_W[n])
            = s_X[m] * s_W[n] * sum_k q_X[m, k] * q_W[n, k]
                                [ pure INT8 x INT8 -> INT32 accumulate ]

Because neither scale depends on the contraction index ``k``, both pull straight out of the
sum. The hardware runs one integer matmul into an INT32 accumulator and applies the rank-1
outer-product of scales ``s_X[m] * s_W[n]`` exactly *once* per output element at the end
(dequant the accumulator). Were the weight scale to vary *along* ``k`` (a "per-input-channel"
scheme on the contraction axis) it could not be hoisted, and the fast integer GEMM would be
impossible — this is precisely why weights are quantized per *output* channel.
"""

from __future__ import annotations

import torch

__all__ = [
    "sqnr_db",
    "mse",
    "quantize_symmetric",
    "dequantize_symmetric",
    "fake_quant_symmetric",
    "quantize_affine",
    "dequantize_affine",
    "fake_quant_affine",
    "w8a8_linear",
]

# INT8 code range for the *symmetric* scheme: we deliberately use a symmetric [-127, 127]
# (dropping -128) so that 0.0 maps exactly to code 0 and +/- are treated identically.
_SYM_QMAX = 127
# Asymmetric affine uses the full unsigned byte [0, 255].
_AFF_QMIN = 0
_AFF_QMAX = 255


def sqnr_db(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in decibels: ``10*log10(P_signal / P_noise)``.

    Computed in float64 for a trustworthy oracle. Returns ``+inf`` if reconstruction is exact.
    """
    sig = (x.double() ** 2).sum()
    noise = ((x.double() - x_hat.double()) ** 2).sum()
    if noise.item() == 0.0:
        return float("inf")
    return (10.0 * torch.log10(sig / noise)).item()


def mse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Mean squared reconstruction error (float64)."""
    return ((x.double() - x_hat.double()) ** 2).mean().item()


# --------------------------------------------------------------------------------------------
# R0 — symmetric (per-tensor or per-channel)
# --------------------------------------------------------------------------------------------
def quantize_symmetric(
    x: torch.Tensor, axis: int | None = None, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantization.

    ``s = amax / 127``; ``q = round(x / s).clamp(-127, 127)``. The scale is a positive float.

    Args:
        x: tensor to quantize.
        axis: if ``None``, a single per-tensor scale; otherwise one scale per slice along
            ``axis`` (amax is reduced over *all other* axes), i.e. per-channel / per-token.
            The returned scale keeps its dims (broadcastable against ``x``).

    Returns:
        ``(q, s)`` where ``q`` is int8 codes and ``s`` is the float scale (broadcastable).
    """
    if axis is None:
        amax = x.detach().abs().max()
    else:
        reduce_dims = [d for d in range(x.dim()) if d != axis]
        amax = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
    s = (amax / _SYM_QMAX).clamp_min(eps)
    q = torch.round(x / s).clamp(-_SYM_QMAX, _SYM_QMAX).to(torch.int8)
    return q, s


def dequantize_symmetric(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Invert :func:`quantize_symmetric`: ``x_hat = q * s``."""
    return q.to(s.dtype) * s


def fake_quant_symmetric(x: torch.Tensor, axis: int | None = None) -> torch.Tensor:
    """Round-trip ``x -> quantize -> dequantize`` under the symmetric scheme."""
    q, s = quantize_symmetric(x, axis=axis)
    return dequantize_symmetric(q, s)


# --------------------------------------------------------------------------------------------
# R1 — asymmetric affine (per-tensor or per-channel)
# --------------------------------------------------------------------------------------------
def quantize_affine(
    x: torch.Tensor, axis: int | None = None, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric affine (unsigned) INT8 quantization onto ``[0, 255]``.

    ``s = (max - min) / 255``; ``z = round(-min / s).clamp(0, 255)``;
    ``q = round(x / s + z).clamp(0, 255)``. The zero-point ``z`` is an integer code so that the
    real value ``0.0`` is represented exactly (important for zero-padding / ReLU sparsity).

    Args:
        x: tensor to quantize.
        axis: ``None`` for one affine map over the whole tensor, else per-slice along ``axis``.

    Returns:
        ``(q, s, z)`` — uint8 codes, float scale, and float integer-valued zero-point (both
        ``s`` and ``z`` are broadcastable against ``x``).
    """
    if axis is None:
        xmin = x.detach().min()
        xmax = x.detach().max()
    else:
        reduce_dims = [d for d in range(x.dim()) if d != axis]
        xmin = x.detach().amin(dim=reduce_dims, keepdim=True)
        xmax = x.detach().amax(dim=reduce_dims, keepdim=True)
    # Always include 0 in the represented range so the zero-point is a valid code.
    xmin = torch.minimum(xmin, torch.zeros_like(xmin))
    xmax = torch.maximum(xmax, torch.zeros_like(xmax))
    s = ((xmax - xmin) / _AFF_QMAX).clamp_min(eps)
    z = torch.round(-xmin / s).clamp(_AFF_QMIN, _AFF_QMAX)
    q = torch.round(x / s + z).clamp(_AFF_QMIN, _AFF_QMAX).to(torch.uint8)
    return q, s, z


def dequantize_affine(q: torch.Tensor, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Invert :func:`quantize_affine`: ``x_hat = (q - z) * s``."""
    return (q.to(s.dtype) - z) * s


def fake_quant_affine(x: torch.Tensor, axis: int | None = None) -> torch.Tensor:
    """Round-trip ``x -> quantize -> dequantize`` under the asymmetric affine scheme."""
    q, s, z = quantize_affine(x, axis=axis)
    return dequantize_affine(q, s, z)


# --------------------------------------------------------------------------------------------
# W8A8 linear — the payoff: integer GEMM with scales hoisted out of the contraction
# --------------------------------------------------------------------------------------------
def w8a8_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Fake-quant W8A8 linear: ``Y ~ X @ W.T`` with INT8 X (per-token) and INT8 W (per-channel).

    Weights ``w`` have shape ``[out, in]`` and are quantized *symmetrically per output channel*
    (one scale per row, over the contraction axis ``in``). Activations ``x`` have shape
    ``[..., in]`` and are quantized *symmetrically per token* (one scale per row). The core is a
    genuine INT8 x INT8 -> INT32 accumulate; the per-token and per-channel scales are applied as
    a single rank-1 outer product on the accumulator at the very end (see module docstring).

    Returns the dequantized fp32 output, which matches the fp32 reference ``x @ w.T`` to within
    ~1e-2 relative error.
    """
    *batch, in_features = x.shape
    x2d = x.reshape(-1, in_features)

    # Per-token activation scale (row = token, contraction axis = features -> axis 1).
    q_x, s_x = quantize_symmetric(x2d, axis=0)  # q_x:[M,K] int8, s_x:[M,1]
    # Per-channel weight scale (row = output channel, contraction axis = in -> axis 0).
    q_w, s_w = quantize_symmetric(w, axis=0)  # q_w:[N,K] int8, s_w:[N,1]

    # Integer matmul into an int32 accumulator: acc[m,n] = sum_k q_x[m,k] * q_w[n,k].
    acc = q_x.to(torch.int32) @ q_w.to(torch.int32).t()  # [M, N]

    # Dequant once: outer product of the token scale (M,1) and channel scale (N,) -> (M,N).
    y2d = acc.to(torch.float32) * (s_x * s_w.reshape(1, -1))
    return y2d.reshape(*batch, w.shape[0])
