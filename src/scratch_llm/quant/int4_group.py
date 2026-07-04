"""Group-wise symmetric INT4 quantization with 2-nibble/byte packing (A5 Rung 2).

This module implements the classic W4A16 weight-only recipe used by GPTQ / AWQ /
Marlin-style kernels:

  * A signed 4-bit integer grid with levels ``{-8, ..., 7}`` (two's-complement
    nibbles). Two nibbles are packed into one ``uint8``: the first value lands in
    the *high* nibble, the second in the *low* nibble::

        byte = (q1 & 0xF) << 4 | (q2 & 0xF)

    Unpacking sign-extends each nibble back to the range ``[-8, 7]``.

  * A *group-wise symmetric* quantizer: the weight is split into contiguous groups
    of ``g = 128`` values along the input (contraction) dimension, and each group
    gets its own fp scale ``s = absmax(group) / 7``. Per-group scales let the grid
    track local dynamic range, which is what makes 4-bit weights usable.

Numeric invariants (the oracle for the tests):

  * **pack / unpack is BIT-EXACT** for any input in ``[-8, 7]``, including odd
    length (a zero pad nibble is appended and dropped on unpack), all-``0xF``
    (i.e. ``-1``), alternating signs, and a single group.

  * **SQNR floor**: uniform b-bit quantization has an SQNR floor of roughly
    ``6.02 * b + c`` dB. For symmetric 4-bit weights this lands well above the
    ``~24 dB`` (``6.02 * 4``) mid-tread bound once per-group scaling is applied;
    the test asserts a conservative floor and reports the *measured* dB.

  * **W4A16 layer output MSE**: dequantizing INT4 weights to fp16 and running the
    linear against fp16 activations produces an output whose MSE vs the fp16
    reference is small (a few 1e-4 relative), the layer-level analogue of the
    "within ~0.3-0.5 perplexity of fp16" GPTQ/Marlin bar.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "GROUP_SIZE",
    "INT4_QMAX",
    "INT4_QMIN",
    "dequantize_groupwise_int4",
    "mse",
    "pack_int4",
    "quantize_groupwise_int4",
    "sqnr_db",
    "unpack_int4",
    "w4a16_linear",
]

GROUP_SIZE = 128
INT4_QMIN = -8
INT4_QMAX = 7


# --------------------------------------------------------------------------- #
# Bit-exact nibble packing                                                     #
# --------------------------------------------------------------------------- #
def pack_int4(q: Tensor) -> Tensor:
    """Pack a 1-D signed 4-bit tensor into ``uint8``, two nibbles per byte.

    ``q`` holds integer values in ``[-8, 7]``. The first element of each pair is
    stored in the high nibble, the second in the low nibble. If ``q`` has odd
    length, a single ``0`` nibble is appended so the last byte is well-formed;
    :func:`unpack_int4` drops it given the original length.

    Returns a ``uint8`` tensor of length ``ceil(len(q) / 2)``.
    """
    if q.dim() != 1:
        raise ValueError(f"pack_int4 expects a 1-D tensor, got shape {tuple(q.shape)}")
    n = q.numel()
    # Mask to the low 4 bits (this maps -8..-1 -> 8..15, 0..7 -> 0..7).
    nibbles = (q.to(torch.int64) & 0xF).to(torch.int64)
    if n % 2 == 1:
        nibbles = torch.cat([nibbles, nibbles.new_zeros(1)])
    hi = nibbles[0::2]
    lo = nibbles[1::2]
    packed = (hi << 4) | lo
    return packed.to(torch.uint8)


def unpack_int4(packed: Tensor, n: int) -> Tensor:
    """Inverse of :func:`pack_int4`: unpack ``n`` sign-extended 4-bit values.

    Each byte yields the high nibble first, then the low nibble. Every nibble is
    sign-extended from 4 bits, so ``0xF`` maps back to ``-1`` and ``0x8`` to
    ``-8``. Only the first ``n`` values are returned (dropping any odd-length pad).
    Result dtype is ``int8``.
    """
    if packed.dim() != 1:
        raise ValueError(f"unpack_int4 expects a 1-D tensor, got shape {tuple(packed.shape)}")
    b = packed.to(torch.int64)
    hi = (b >> 4) & 0xF
    lo = b & 0xF
    out = torch.stack([hi, lo], dim=1).reshape(-1)  # interleave hi, lo
    out = out[:n]
    # Sign-extend 4-bit two's complement: values >= 8 become negative.
    out = torch.where(out >= 8, out - 16, out)
    return out.to(torch.int8)


# --------------------------------------------------------------------------- #
# Group-wise symmetric INT4 quantizer                                          #
# --------------------------------------------------------------------------- #
def quantize_groupwise_int4(w: Tensor, group_size: int = GROUP_SIZE) -> tuple[Tensor, Tensor]:
    """Symmetric group-wise INT4 quantization of a 2-D weight ``[out, in]``.

    The input (last) dimension is split into contiguous groups of ``group_size``;
    each group gets a symmetric scale ``s = absmax / 7`` and rounds to the signed
    4-bit grid ``[-8, 7]``. ``in`` must be divisible by ``group_size``.

    Returns ``(q, scales)`` where ``q`` is ``int8`` in ``[-8, 7]`` with the same
    shape as ``w`` and ``scales`` has shape ``[out, in // group_size]`` (fp32).
    """
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(w.shape)}")
    out_features, in_features = w.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by group_size={group_size}")
    n_groups = in_features // group_size
    wf = w.to(torch.float32).reshape(out_features, n_groups, group_size)
    absmax = wf.abs().amax(dim=-1, keepdim=True)  # [out, n_groups, 1]
    # Avoid divide-by-zero on all-zero groups; a zero scale would still give q=0.
    scales = (absmax / INT4_QMAX).clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.round(wf / scales).clamp_(INT4_QMIN, INT4_QMAX).to(torch.int8)
    return q.reshape(out_features, in_features), scales.reshape(out_features, n_groups)


def dequantize_groupwise_int4(q: Tensor, scales: Tensor, group_size: int = GROUP_SIZE) -> Tensor:
    """Reconstruct fp32 weights from INT4 codes and per-group scales.

    Inverse of :func:`quantize_groupwise_int4`. ``q`` is ``[out, in]``, ``scales``
    is ``[out, in // group_size]``.
    """
    out_features, in_features = q.shape
    n_groups = in_features // group_size
    qf = q.to(torch.float32).reshape(out_features, n_groups, group_size)
    deq = qf * scales.to(torch.float32).reshape(out_features, n_groups, 1)
    return deq.reshape(out_features, in_features)


# --------------------------------------------------------------------------- #
# W4A16 linear                                                                 #
# --------------------------------------------------------------------------- #
def w4a16_linear(
    x: Tensor,
    q: Tensor,
    scales: Tensor,
    bias: Tensor | None = None,
    group_size: int = GROUP_SIZE,
) -> Tensor:
    """W4A16 linear: INT4 group-quantized weights, fp16 activations.

    Weights are dequantized to the activation dtype and the matmul is done in that
    dtype (the accumulation semantics a Marlin/GPTQ kernel emulates). ``x`` is
    ``[..., in]``, ``q``/``scales`` describe a weight ``[out, in]``; returns
    ``[..., out]`` in ``x``'s dtype.
    """
    wdeq = dequantize_groupwise_int4(q, scales, group_size).to(x.dtype)
    out = torch.nn.functional.linear(x, wdeq, bias)
    return out


# --------------------------------------------------------------------------- #
# Error metrics (the oracle)                                                   #
# --------------------------------------------------------------------------- #
def sqnr_db(ref: Tensor, approx: Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB: ``10*log10(P_signal/P_noise)``."""
    ref = ref.to(torch.float64)
    approx = approx.to(torch.float64)
    signal = ref.pow(2).sum()
    noise = (ref - approx).pow(2).sum()
    if noise.item() == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(signal / noise).item())


def mse(ref: Tensor, approx: Tensor) -> float:
    """Mean squared error between two tensors (fp64 accumulation)."""
    return float((ref.to(torch.float64) - approx.to(torch.float64)).pow(2).mean().item())
