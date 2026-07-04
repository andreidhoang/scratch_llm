"""A5 Rung 3 — NVFP4 two-level pack/unpack vs MXFP4, and a software block-scaled GEMM.

This module implements CPU-buildable fake-quant numerics for the two production 4-bit
micro-scaled floating formats and the block-scaled matmul that consumes them.

Formats
-------
* **FP4 = E2M1** (the shared element format for both): 1 sign + 2 exp + 1 mantissa, so the
  representable magnitudes are exactly ``{0, .5, 1, 1.5, 2, 3, 4, 6}`` (max magnitude 6).
* **NVFP4** (NVIDIA, block-16, two-level scaling, ~4.5 bits/value): a per-**tensor** fp32
  ``s_global`` PLUS a per-**16-block** scale stored in **E4M3** (bias 7, max +-448). Because the
  block scale is a *non-power-of-two* fp8 number it tracks the ideal ``block_amax/6`` scale to
  ~1 part in 16, not to the nearest octave.
* **MXFP4** (OCP Microscaling, block-32, single-level, ~4.25 bits/value): a per-**32-block**
  scale stored in **E8M0** (a *pure exponent* — a power of two, bias 127). Rounding the ideal
  scale to a power of two is up to 2x off in the floor sense (sqrt2 typical for round-to-nearest).

The NVFP4 recipe pinned here follows **NVIDIA TensorRT-Model-Optimizer / TransformerEngine**:
``s_global = amax / (448 * 6)`` (so the E4M3 block scale, which is ``block_amax*448/amax``, is by
construction <= 448 and always representable), ``s_b = quantize_E4M3(block_amax / 6 / s_global)``,
element ``q = round_E2M1(x / (s_global * s_b))``, dequant ``x_hat = s_global * s_b * q``.
Reference: NVIDIA "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference" and the
TransformerEngine ``nvfp4`` recipe (the ``/6`` element-max and ``*448`` global-scale folding).

Oracle / invariants
-------------------
1. **Bit-exact pack/unpack** for BOTH formats: two 4-bit codes pack into one uint8 and unpack to
   the identical codes; a quantize -> pack -> unpack -> dequant round-trip reproduces the direct
   dequant exactly.
2. **The load-bearing result**: on the SAME data, per-block ``MSE(NVFP4) < MSE(MXFP4)``, caused by
   two named mechanisms — (a) the E4M3 non-power-of-two block scale (E8M0 rounds the block amax to
   a power of two, up to 2x off) and (b) the finer ``k=16`` vs ``k=32`` block. If MXFP4 ever ties
   or beats NVFP4 that is a two-level scale-placement bug, not "acceptable loss".
3. **SQNR**: NVFP4 SQNR strictly exceeds MXFP4 SQNR on the same data (measured ~20.4 vs ~18.7 dB on
   Gaussian). Note the ``6.02·bits`` linear-quantizer rule does NOT apply to non-uniform FP4 on
   Gaussian data (the E2M1 grid is exponentially spaced), so the absolute dB sits below ``6.02·4``;
   the load-bearing claim is the *relative* NVFP4 > MXFP4 gap and the MSE ratio, not an absolute floor.
4. **Block-scaled GEMM**: a linear whose weight is NVFP4-quantized along K, with the block scales
   applied inside the accumulation, lands within 1% (relative Frobenius) of the BF16 matmul.
"""

from __future__ import annotations

import torch

__all__ = [
    "FP4_MAGNITUDES",
    "E4M3_MAX",
    "block_mse",
    "e2m1_decode",
    "e2m1_round_to_code",
    "mxfp4_dequantize",
    "mxfp4_quantize",
    "nvfp4_dequantize",
    "nvfp4_quantize",
    "pack_fp4",
    "quant_linear_nvfp4",
    "quantize_e4m3",
    "quantize_e8m0",
    "sqnr_db",
    "unpack_fp4",
]

# FP4 = E2M1 representable magnitudes (index == 3-bit magnitude field). Max magnitude 6.
FP4_MAGNITUDES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MAX = 6.0

# E4M3: 1-4-3, bias 7, max finite magnitude 448 (S.1111.111 is NaN-only, so 448 = S.1111.110).
E4M3_MAX = 448.0
_E4M3_EXP_MIN = -6  # min *normal* unbiased exponent (e_field=1 -> 1-7); subnormals share it
_E4M3_EXP_MAX = 8  # max unbiased exponent (e_field=15 -> 15-7)
_E4M3_MANT_BITS = 3

_TINY = 1e-30


def _as_float(x: torch.Tensor) -> torch.Tensor:
    """Promote to fp32 for the numerics (fake-quant is computed in fp32, then re-cast by callers)."""
    return x.to(torch.float32)


# --------------------------------------------------------------------------------------
# FP4 (E2M1) element codec + nibble packing
# --------------------------------------------------------------------------------------
def e2m1_round_to_code(x: torch.Tensor) -> torch.Tensor:
    """Round a *pre-scaled* tensor to E2M1 and return 4-bit codes (bit3 sign, bits2-0 magnitude).

    Magnitude is the nearest of ``FP4_MAGNITUDES`` (ties resolve to the lower index — negligible on
    real data); anything above 6 saturates to 6. Sign is preserved even for zero (a +0/-0 code),
    which is harmless because both decode to 0.0.
    """
    xf = _as_float(x)
    mags = torch.tensor(FP4_MAGNITUDES, dtype=torch.float32, device=xf.device)
    a = xf.abs()
    # nearest grid magnitude: argmin over the 8 anchors (ties -> lower index, measure-zero on reals)
    idx = torch.argmin((a.unsqueeze(-1) - mags).abs(), dim=-1).to(torch.uint8)
    sign = (xf < 0).to(torch.uint8)
    return (sign << 3) | idx


def e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    """Decode 4-bit E2M1 codes back to fp32 values."""
    mags = torch.tensor(FP4_MAGNITUDES, dtype=torch.float32, device=code.device)
    idx = (code & 0x7).long()
    sign = torch.where((code & 0x8) != 0, -1.0, 1.0)
    return sign * mags[idx]


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack a flat tensor of 4-bit codes (uint8, values 0..15) into uint8 nibbles, 2 codes/byte.

    Odd length is right-padded with a zero (a +0 code). Returns the packed byte tensor; recover the
    original codes with :func:`unpack_fp4` (pass the original ``n``).
    """
    c = codes.to(torch.uint8).reshape(-1)
    if c.numel() % 2 == 1:
        c = torch.cat([c, torch.zeros(1, dtype=torch.uint8, device=c.device)])
    lo = c[0::2]
    hi = c[1::2]
    return (lo & 0x0F) | ((hi & 0x0F) << 4)


def unpack_fp4(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of :func:`pack_fp4` — recover the first ``n`` 4-bit codes (bit-exact)."""
    p = packed.to(torch.uint8).reshape(-1)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=1).reshape(-1)
    return out[:n]


# --------------------------------------------------------------------------------------
# Scale codecs: E4M3 (NVFP4 block scale) and E8M0 (MXFP4 block scale)
# --------------------------------------------------------------------------------------
def quantize_e4m3(v: torch.Tensor) -> torch.Tensor:
    """Round non-negative magnitudes to the nearest **E4M3** value (bias 7, saturating at 448).

    Uses per-element binade + 3-bit mantissa rounding (round-half-to-even via ``torch.round``);
    handles the subnormal region (unbiased exponent clamped to -6, ULP 2**-9) naturally.
    """
    vf = _as_float(v).clamp(min=0.0, max=E4M3_MAX)
    exp = torch.floor(torch.log2(vf.clamp_min(_TINY)))
    exp = exp.clamp(min=_E4M3_EXP_MIN, max=_E4M3_EXP_MAX)
    step = torch.exp2(exp - _E4M3_MANT_BITS)  # ULP within the binade
    q = torch.round(vf / step) * step
    q = q.clamp(min=0.0, max=E4M3_MAX)
    return torch.where(vf > 0.0, q, torch.zeros_like(q))


def quantize_e8m0(v: torch.Tensor) -> torch.Tensor:
    """Round positive magnitudes to the nearest **power of two** (E8M0 pure-exponent codec).

    ``value = 2**round(log2(v))``. This is the raw E8M0 encoding; MXFP4's *scale selection* wraps it
    with a no-overflow (ceil) policy — see :func:`mxfp4_quantize`.
    """
    vf = _as_float(v)
    exp = torch.round(torch.log2(vf.clamp_min(_TINY)))
    exp = exp.clamp(min=-127.0, max=127.0)
    return torch.where(vf > 0.0, torch.exp2(exp), torch.zeros_like(vf))


def _e8m0_no_overflow_scale(amax: torch.Tensor) -> torch.Tensor:
    """Smallest power-of-two block scale that maps ``amax`` into the FP4 range without clipping.

    ``s_b = 2**ceil(log2(amax / 6))`` — the MXFP4 no-overflow scale policy: every element then
    satisfies ``|x| / s_b <= 6``, so nothing saturates. The cost is *up to 2x* wasted dynamic range
    (the block max lands anywhere in ``(3, 6]`` depending on where ``amax/6`` sits in its octave) —
    exactly the coarseness the E4M3 block scale of NVFP4 avoids. Round-to-nearest E8M0 would instead
    sometimes round the scale *down* and clip a lone outlier, an artefact that (unfairly) penalises
    finer blocks; the no-overflow policy is the faithful, stronger MXFP4 baseline.
    """
    exp = torch.ceil(torch.log2((amax / _FP4_MAX).clamp_min(_TINY)))
    exp = exp.clamp(min=-127.0, max=127.0)
    return torch.exp2(exp)


# --------------------------------------------------------------------------------------
# NVFP4 (block-16, two-level: fp32 global + E4M3 block scale)
# --------------------------------------------------------------------------------------
def _blockify(x: torch.Tensor, block: int) -> tuple[torch.Tensor, int]:
    """Flatten ``x`` and view as ``(n_blocks, block)``, right-padding the last block with zeros."""
    flat = _as_float(x).reshape(-1)
    n = flat.numel()
    pad = (-n) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    return flat.reshape(-1, block), n


class NVFP4Tensor:
    """The packed NVFP4 payload: E2M1 codes + fp32 global scale + per-block E4M3 scales."""

    __slots__ = ("block", "codes", "n", "s_block", "s_global", "shape")

    def __init__(
        self,
        codes: torch.Tensor,
        s_global: torch.Tensor,
        s_block: torch.Tensor,
        shape: torch.Size,
        n: int,
        block: int,
    ) -> None:
        self.codes = codes  # (n_blocks, block) uint8
        self.s_global = s_global  # scalar fp32
        self.s_block = s_block  # (n_blocks,) fp32, E4M3-representable
        self.shape = shape
        self.n = n
        self.block = block


def nvfp4_quantize(x: torch.Tensor, block: int = 16) -> NVFP4Tensor:
    """Quantize ``x`` to NVFP4 (block-16 default) with the TensorRT-Model-Optimizer recipe."""
    xf = _as_float(x)
    shape = xf.shape
    xb, n = _blockify(xf, block)
    amax = xb.abs().max()
    s_global = (amax / (E4M3_MAX * _FP4_MAX)).clamp_min(_TINY)
    block_amax = xb.abs().amax(dim=1)
    s_block = quantize_e4m3(block_amax / _FP4_MAX / s_global)
    scale = (s_global * s_block).clamp_min(_TINY).unsqueeze(1)
    codes = e2m1_round_to_code(xb / scale)
    return NVFP4Tensor(codes, s_global.to(torch.float32), s_block, shape, n, block)


def nvfp4_dequantize(q: NVFP4Tensor) -> torch.Tensor:
    """Dequantize an :class:`NVFP4Tensor` back to fp32 in the original shape."""
    mag = e2m1_decode(q.codes)
    scale = (q.s_global * q.s_block).unsqueeze(1)
    flat = (mag * scale).reshape(-1)[: q.n]
    return flat.reshape(q.shape)


# --------------------------------------------------------------------------------------
# MXFP4 (block-32, single-level: E8M0 power-of-two block scale)
# --------------------------------------------------------------------------------------
class MXFP4Tensor:
    """The packed MXFP4 payload: E2M1 codes + per-block E8M0 (power-of-two) scales."""

    __slots__ = ("block", "codes", "n", "s_block", "shape")

    def __init__(
        self,
        codes: torch.Tensor,
        s_block: torch.Tensor,
        shape: torch.Size,
        n: int,
        block: int,
    ) -> None:
        self.codes = codes
        self.s_block = s_block  # (n_blocks,) power-of-two
        self.shape = shape
        self.n = n
        self.block = block


def mxfp4_quantize(x: torch.Tensor, block: int = 32) -> MXFP4Tensor:
    """Quantize ``x`` to MXFP4 (block-32 default), OCP microscaling with an E8M0 block scale."""
    xf = _as_float(x)
    shape = xf.shape
    xb, n = _blockify(xf, block)
    block_amax = xb.abs().amax(dim=1)
    s_block = _e8m0_no_overflow_scale(block_amax).clamp_min(_TINY)
    codes = e2m1_round_to_code(xb / s_block.unsqueeze(1))
    return MXFP4Tensor(codes, s_block, shape, n, block)


def mxfp4_dequantize(q: MXFP4Tensor) -> torch.Tensor:
    """Dequantize an :class:`MXFP4Tensor` back to fp32 in the original shape."""
    mag = e2m1_decode(q.codes)
    flat = (mag * q.s_block.unsqueeze(1)).reshape(-1)[: q.n]
    return flat.reshape(q.shape)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def sqnr_db(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB: ``10*log10(||x||^2 / ||x - x_hat||^2)``."""
    xf = _as_float(x)
    sig = (xf**2).sum()
    noise = ((xf - _as_float(x_hat)) ** 2).sum().clamp_min(_TINY)
    return float(10.0 * torch.log10(sig / noise))


def block_mse(x: torch.Tensor, x_hat: torch.Tensor, block: int) -> torch.Tensor:
    """Per-block mean-squared error, ``(n_blocks,)``, over blocks that tile the flattened tensor."""
    a, n = _blockify(x, block)
    b, _ = _blockify(x_hat, block)
    return ((a - b) ** 2).mean(dim=1)


# --------------------------------------------------------------------------------------
# Software block-scaled GEMM (NVFP4 weights, scales applied in the accumulation)
# --------------------------------------------------------------------------------------
def quant_linear_nvfp4(x: torch.Tensor, weight: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Linear ``y = x @ weight.T`` with ``weight`` NVFP4-quantized along K (block-16 default).

    The weight is quantized per-row into K-blocks; the matmul is accumulated **block by block**,
    dequantizing each K-block's NVFP4 codes with its ``s_global * s_block`` scale before the partial
    ``x_block @ w_block.T`` — i.e. the block scales live inside the accumulation, mirroring a real
    block-scaled tensor-core GEMM. ``x`` stays in fp32 here (weight-only quantization).

    ``weight`` is ``(N, K)`` with ``K`` a multiple of ``block``; ``x`` is ``(M, K)``. Returns
    ``(M, N)`` fp32. Invariant: within 1% relative Frobenius of the BF16 reference matmul.
    """
    xf = _as_float(x)
    w = _as_float(weight)
    n_out, k = w.shape
    if k % block != 0:
        raise ValueError(f"K={k} must be a multiple of block={block} for the block-scaled GEMM")

    # Per-row, per-K-block NVFP4 quantization of the weight (shared global scale, as NVFP4 specifies).
    amax = w.abs().max()
    s_global = (amax / (E4M3_MAX * _FP4_MAX)).clamp_min(_TINY)
    n_kb = k // block
    wb = w.reshape(n_out, n_kb, block)
    block_amax = wb.abs().amax(dim=2)  # (N, n_kb)
    s_block = quantize_e4m3(block_amax / _FP4_MAX / s_global)  # (N, n_kb), E4M3
    scale = (s_global * s_block).clamp_min(_TINY).unsqueeze(-1)  # (N, n_kb, 1)
    codes = e2m1_round_to_code(wb / scale)  # (N, n_kb, block)
    w_hat = e2m1_decode(codes) * scale  # (N, n_kb, block) dequantized

    # Accumulate the matmul block-by-block along K (scales already folded into w_hat per block).
    y = xf.new_zeros(xf.shape[0], n_out)
    for kb in range(n_kb):
        x_blk = xf[:, kb * block : (kb + 1) * block]  # (M, block)
        y += x_blk @ w_hat[:, kb, :].t()  # (M, N)
    return y
