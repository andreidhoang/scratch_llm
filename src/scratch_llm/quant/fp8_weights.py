"""S1/S-R2 — per-output-channel FP8 (E4M3) weight quantization, driving ``torch._scaled_mm``.

Spec: ``experiments/S1/S-R2/spec.md``   ·   Map: ``experiments/S1/S-R2/map.md``

What this is for
----------------
At decode time a dense LM is a weight-streaming problem: batch 32 against an 8B model reads every
weight once per step and does two FLOPs per byte of it. Storing the weights as E4M3 (one byte per
weight instead of two) is therefore not a compute optimisation — it halves the *bytes* the step
must pull through HBM. This module is the storage-and-GEMM half of that; ``fp8_kv_per_head.py`` is
the cache half.

The scale derivation (the load-bearing choice)
----------------------------------------------
A linear layer computes ``Y[m, n] = sum_k X[m, k] * W[n, k]`` — the contraction is over ``k``.
Write ``X[m, k] = q_X[m, k] * s_X[m]`` and ``W[n, k] = q_W[n, k] * s_W[n]``. Neither scale depends
on ``k``, so both pull straight out of the sum:

    Y[m, n] = s_X[m] * s_W[n] * sum_k q_X[m, k] * q_W[n, k]

The hardware runs one FP8 x FP8 -> FP32 matmul and applies the rank-1 outer product of scales once
per output element. ``quant/int8.py``'s module docstring derives this in full for INT8; the algebra
is identical here and is *why* weights are quantized per **output** channel and never along the
contraction axis. A scale that varied along ``k`` could not be hoisted and there would be no fast
GEMM to call.

So:

  * **weight** ``W`` is ``(N, K)``; scale is ``amax_k |W[n, k]| / 448``, shape ``(N, 1)`` — one
    scale per output channel, computed once at load, static thereafter.
  * **activation** ``X`` is ``(M, K)``; scale is ``amax_k |X[m, k]| / 448``, shape ``(M, 1)`` —
    one scale per token, computed **dynamically** on every forward. An activation's dynamic range
    moves with the prompt; a static activation scale needs a calibration set and buys a saturation
    risk that a per-token amax does not have. (vLLM makes the same split — see map.md.)

Both scales are floored at :data:`MIN_SCALE`. An all-zero row would otherwise give scale 0 and a
0/0 in the division; the floor is vLLM's ``_FP8_MIN_SCALING_FACTOR`` (``input_quant_fp8.py:25``),
reproduced so the two implementations round the same degenerate case the same way.

E4M3 facts, measured on torch 2.12 rather than recalled (see ``tests/quant/test_s1_s_r2.py``)
---------------------------------------------------------------------------------------------
``float8_e4m3fn`` is 4 exponent bits, 3 mantissa bits, bias 7, no infinities, one NaN encoding:

  * largest finite magnitude **448**; the top binade's step is 32, so a *raw* cast round-to-nearest
    maps everything in ``[448, 464]`` down onto 448 and everything **above 464 to NaN** — it does
    not saturate. That is why every quantizer here clamps to +-448 *before* the cast: a value that
    overflows a stale calibration must become the largest representable number, not a NaN that
    poisons a whole attention row.
  * spacing at 1.0 is ``eps = 2**-3 = 0.125``; round-to-nearest therefore has unit roundoff
    ``2**-4 = 0.0625`` — a normal value's relative quantization error cannot exceed 1/16.
  * smallest normal ``2**-6``; smallest subnormal ``2**-9``; anything under ``2**-10`` flushes to
    zero. Below the min normal the error bound is absolute (``2**-10``), not relative — a channel
    whose amax is much larger than its typical element loses its small elements entirely, which is
    the mechanism behind "outlier channels ruin per-tensor quantization".

Correctness oracle
------------------
:func:`fp8_linear_reference` takes the *same* codes and the *same* scales as :func:`fp8_linear` and
does the matmul in float64. The only differences that remain are the accumulation order inside the
GEMM and the rounding of the output to ``out_dtype`` — so a test comparing the two is a test of the
kernel path, not of the quantizer, and its tolerance is a function of ``K`` and the output dtype's
eps rather than a number anyone had to choose.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor, nn

__all__ = [
    "E4M3_MAX",
    "E4M3_MIN_NORMAL",
    "E4M3_MIN_SUBNORMAL",
    "E4M3_NAN_ABOVE",
    "E4M3_UNIT_ROUNDOFF",
    "FP8_DTYPE",
    "MIN_SCALE",
    "Fp8Linear",
    "convert_linears_to_fp8",
    "dequantize",
    "fp8_linear",
    "fp8_linear_reference",
    "quantize_per_channel",
    "quantize_per_token",
    "rowwise_scaled_mm_supported",
    "saturating_cast_e4m3",
]

FP8_DTYPE = torch.float8_e4m3fn

#: Largest finite E4M3 magnitude.
E4M3_MAX = 448.0
#: A raw cast round-to-nearest saturates up to this value and yields NaN strictly above it
#: (448 + ulp/2, ulp = 32 in the top binade). Never rely on it: clamp first.
E4M3_NAN_ABOVE = 464.0
#: Unit roundoff, 2**-4: the bound on a *normal* value's relative quantization error.
E4M3_UNIT_ROUNDOFF = 0.0625
#: Smallest normal (2**-6) and smallest subnormal (2**-9). Below half the latter, values flush to 0.
E4M3_MIN_NORMAL = 0.015625
E4M3_MIN_SUBNORMAL = 0.001953125

#: FP8 accumulation mode. False = fp32 accumulator. ``use_fast_accum=True`` accumulates in the
#: tensor core at reduced precision: faster, and a different kernel numerically — so the oracle test
#: and the benchmark must both use this value or they are measuring two different things.
DEFAULT_FAST_ACCUM = False

#: Scale floor for an all-zero group — vLLM's ``_FP8_MIN_SCALING_FACTOR`` (input_quant_fp8.py:25).
MIN_SCALE = 1.0 / (E4M3_MAX * 512.0)

# ``torch._scaled_mm`` is private by name and public by use: it is the FP8 GEMM entry point, and
# vLLM calls it directly (``scaled_mm/pytorch.py:117``). Bound once here so the type-checker
# suppression lives in exactly one place instead of at every call site.
_SCALED_MM = torch._scaled_mm  # pyright: ignore[reportPrivateImportUsage]


def saturating_cast_e4m3(v: Tensor) -> Tensor:
    """Cast to E4M3 with **saturation**, never NaN.

    ``v.to(float8_e4m3fn)`` alone is not this function: torch's cast rounds to nearest, which
    happens to saturate on ``[448, 464]`` but returns NaN above 464. Clamping first makes the
    overflow policy explicit and makes it the same on every torch version and every device.
    """
    return v.clamp(-E4M3_MAX, E4M3_MAX).to(FP8_DTYPE)


def _amax_scale(x: Tensor, dim: int) -> Tensor:
    """``amax(|x|) / 448`` along ``dim``, kept as fp32 and floored — the symmetric scale rule."""
    amax = x.detach().abs().amax(dim=dim, keepdim=True).to(torch.float32)
    return (amax / E4M3_MAX).clamp_min(MIN_SCALE)


def quantize_per_channel(w: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a weight ``(N, K)`` to E4M3 codes plus a per-output-channel scale ``(N, 1)``.

    The reduction is over ``K`` — the contraction axis — precisely so the resulting scale does
    *not* depend on it and can be hoisted out of the GEMM (see the module docstring).
    """
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D (N, K), got shape {tuple(w.shape)}")
    scale = _amax_scale(w, dim=1)
    return saturating_cast_e4m3(w.to(torch.float32) / scale), scale


def quantize_per_token(x: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize an activation ``(M, K)`` to E4M3 codes plus a per-token scale ``(M, 1)``.

    Dynamic: the amax is recomputed every forward. Cheap (one pass over an activation that is
    about to be read by the GEMM anyway) and immune to a calibration set that did not contain the
    prompt actually being served.
    """
    if x.ndim != 2:
        raise ValueError(f"activation must be 2-D (M, K), got shape {tuple(x.shape)}")
    scale = _amax_scale(x, dim=1)
    return saturating_cast_e4m3(x.to(torch.float32) / scale), scale


def dequantize(codes: Tensor, scale: Tensor) -> Tensor:
    """``codes * scale`` in fp32 — the exact inverse map of the quantizer's dequant step."""
    return codes.to(torch.float32) * scale.to(torch.float32)


@lru_cache(maxsize=8)
def rowwise_scaled_mm_supported(device_index: int = 0) -> bool:
    """Probe once whether ``torch._scaled_mm`` accepts ``(M,1) x (1,N)`` scales on this device.

    Probed rather than assumed: rowwise FP8 scaling landed in ``_scaled_mm`` at different torch
    versions for CUDA and ROCm, and vLLM's own rowwise ``_scaled_mm`` kernel is gated to ROCm
    (``scaled_mm/pytorch.py:127-135``) precisely because the CUDA answer is a moving target. If the
    probe fails, :func:`fp8_linear` uses the unfused path — the same fallback vLLM's
    ``ChannelWiseTorchFP8ScaledMMLinearKernel`` (``scaled_mm/pytorch.py:200-266``) uses.
    """
    if not torch.cuda.is_available():
        return False
    dev = torch.device("cuda", device_index)
    try:
        a = torch.zeros((16, 32), dtype=FP8_DTYPE, device=dev)
        b = torch.zeros((32, 16), dtype=FP8_DTYPE, device=dev).t().contiguous().t()
        sa = torch.ones((16, 1), dtype=torch.float32, device=dev)
        sb = torch.ones((1, 16), dtype=torch.float32, device=dev)
        _SCALED_MM(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    except Exception:
        return False
    return True


def _scaled_mm(
    a: Tensor,
    b: Tensor,
    sa: Tensor,
    sb: Tensor,
    out_dtype: torch.dtype,
    bias: Tensor | None,
    fast_accum: bool,
) -> Tensor:
    out = _SCALED_MM(
        a, b, scale_a=sa, scale_b=sb, bias=bias, out_dtype=out_dtype, use_fast_accum=fast_accum
    )
    # torch < 2.5 returned (out, amax); >= 2.5 returns the tensor. vLLM carries the same shim
    # (scaled_mm/pytorch.py:120-123) — keep it so a downgrade on the box is a wrong number, not a
    # crash three frames away.
    if isinstance(out, tuple):
        out = out[0]
    return out


def fp8_linear(
    x: Tensor,
    weight_codes: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    use_fast_accum: bool = DEFAULT_FAST_ACCUM,
) -> Tensor:
    """``x @ dequant(weight)^T + bias`` through ``torch._scaled_mm``. CUDA only.

    ``x`` is ``(..., K)`` in any float dtype and is quantized per token here. ``weight_codes`` is
    ``(N, K)`` E4M3 with ``weight_scale`` ``(N, 1)`` fp32 — the pair returned by
    :func:`quantize_per_channel`.

    Operand contract (each item is a real constraint of the op, not defensive coding):
      * both operands must be FP8 and 2-D, so a ``(B, T, K)`` activation is flattened and restored;
      * ``b`` must be **column-major**: ``weight_codes`` is ``(N, K)`` row-major, so ``.t()`` is
        ``(K, N)`` with unit stride down the rows — exactly what the op wants, with no copy;
      * ``K`` must be a multiple of 16;
      * scales are fp32 on the same device, ``(M, 1)`` and ``(1, N)`` for the rowwise path;
      * ``use_fast_accum`` changes the accumulator, so the correctness test and the benchmark must
        pass the same value or they are measuring two different kernels.
    """
    if x.is_cpu:
        raise RuntimeError(
            "fp8_linear needs CUDA (torch._scaled_mm has no CPU kernel). Use "
            "fp8_linear_reference on a laptop; measure on the box: "
            "infra/rent.sh sync-up <user@host> && ssh in && "
            "bash ladders/experiments/S1/S-R2/run.sh"
        )
    n, k = weight_codes.shape
    if x.shape[-1] != k:
        raise ValueError(f"activation last dim {x.shape[-1]} != weight K {k}")
    if k % 16 != 0:
        raise ValueError(f"torch._scaled_mm requires K % 16 == 0, got K={k}")
    out_shape = (*x.shape[:-1], n)
    x_codes, x_scale = quantize_per_token(x.reshape(-1, k))
    b = weight_codes.t()  # (K, N), column-major by construction
    if rowwise_scaled_mm_supported(x.device.index or 0):
        out = _scaled_mm(x_codes, b, x_scale, weight_scale.t(), out_dtype, bias, use_fast_accum)
        return out.view(out_shape)
    # Unfused fallback: run the GEMM with unit scales into fp32 and apply the rank-1 outer product
    # of scales afterwards. Mathematically identical (the scales were always hoistable); it costs
    # one extra pass over the output. vLLM's channelwise torch path does exactly this.
    ones = torch.ones(1, dtype=torch.float32, device=x.device)
    acc = _scaled_mm(x_codes, b, ones, ones, torch.float32, None, use_fast_accum)
    acc = acc * x_scale * weight_scale.t()
    if bias is not None:
        acc = acc + bias
    return acc.to(out_dtype).view(out_shape)


def fp8_linear_reference(
    x: Tensor,
    weight_codes: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """The slow, obviously-correct oracle for :func:`fp8_linear`: float64, no fused kernel.

    Quantizes ``x`` with the *same* per-token rule, dequantizes both operands exactly, and
    contracts in float64. Everything that differs between this and :func:`fp8_linear` is therefore
    attributable to the GEMM's accumulation order and the final cast — which is what makes the
    comparison a test of the kernel rather than a re-test of the quantizer.
    """
    n, k = weight_codes.shape
    if x.shape[-1] != k:
        raise ValueError(f"activation last dim {x.shape[-1]} != weight K {k}")
    out_shape = (*x.shape[:-1], n)
    x_codes, x_scale = quantize_per_token(x.reshape(-1, k))
    xd = x_codes.to(torch.float64) * x_scale.to(torch.float64)
    wd = weight_codes.to(torch.float64) * weight_scale.to(torch.float64)
    acc = xd @ wd.t()
    if bias is not None:
        acc = acc + bias.to(torch.float64)
    return acc.to(out_dtype).view(out_shape)


class Fp8Linear(nn.Module):
    """A drop-in replacement for ``nn.Linear`` holding E4M3 codes and a per-channel scale.

    Storage is a real 1-byte tensor, so the weight-byte halving is a measured property of the
    module (``weight_codes.element_size() == 1``), not a claim in a docstring. ``bias`` stays in
    the compute dtype: it is ``N`` numbers against ``N*K`` weights, and quantizing it would buy
    nothing and cost a rounding.

    On CUDA the forward runs :func:`fp8_linear`; on CPU it runs :func:`fp8_linear_reference`, so
    the surrounding engine is exercisable on a laptop at laptop speed while being bit-comparable
    to nothing but itself. A number never comes off the CPU path — ``bench/s1_fp8_serving.py``
    refuses to run without CUDA.
    """

    weight_codes: Tensor
    weight_scale: Tensor
    bias: Tensor | None

    def __init__(
        self,
        weight_codes: Tensor,
        weight_scale: Tensor,
        bias: Tensor | None = None,
        *,
        out_dtype: torch.dtype = torch.bfloat16,
        use_fast_accum: bool = DEFAULT_FAST_ACCUM,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_codes", weight_codes)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("bias", bias)
        self.out_dtype = out_dtype
        self.use_fast_accum = use_fast_accum
        self.out_features, self.in_features = weight_codes.shape

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        out_dtype: torch.dtype | None = None,
        use_fast_accum: bool = DEFAULT_FAST_ACCUM,
    ) -> Fp8Linear:
        codes, scale = quantize_per_channel(linear.weight.data)
        bias = None if linear.bias is None else linear.bias.data.clone()
        return cls(
            codes,
            scale,
            bias,
            out_dtype=out_dtype or linear.weight.dtype,
            use_fast_accum=use_fast_accum,
        )

    def dequantized_weight(self) -> Tensor:
        """``(N, K)`` fp32 — what the GEMM sees. The input to any error accounting."""
        return dequantize(self.weight_codes, self.weight_scale)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_cpu:
            return fp8_linear_reference(
                x, self.weight_codes, self.weight_scale, self.bias, out_dtype=self.out_dtype
            )
        return fp8_linear(
            x,
            self.weight_codes,
            self.weight_scale,
            self.bias,
            out_dtype=self.out_dtype,
            use_fast_accum=self.use_fast_accum,
        )


#: Layers left in bf16 by default. The LM head is one matmul per step against a vocab-sized output
#: and it is the one whose error lands directly on the sampled token; embeddings are a gather, not
#: a GEMM, so quantizing them buys no bandwidth on the decode path at all.
DEFAULT_SKIP = ("lm_head", "embed", "wte", "output")


def convert_linears_to_fp8(
    root: nn.Module,
    *,
    skip: tuple[str, ...] = DEFAULT_SKIP,
    min_in_features: int = 16,
    use_fast_accum: bool = DEFAULT_FAST_ACCUM,
) -> list[str]:
    """Replace every ``nn.Linear`` under ``root`` with :class:`Fp8Linear`, in place.

    Returns the qualified names replaced, so a caller can assert *which* layers were converted
    rather than trusting a count. Modules whose qualified name contains any entry of ``skip``, and
    any layer whose ``in_features`` is not a multiple of 16 (``_scaled_mm``'s K constraint), are
    left alone — silently quantizing a layer the GEMM cannot take would fail at the first forward
    on the box instead of here.
    """
    replaced: list[str] = []
    for name, child in list(root.named_children()):
        if isinstance(child, nn.Linear):
            qualified = name
            if any(s in qualified for s in skip) or child.in_features % min_in_features != 0:
                continue
            setattr(root, name, Fp8Linear.from_linear(child, use_fast_accum=use_fast_accum))
            replaced.append(qualified)
        else:
            replaced.extend(
                f"{name}.{sub}"
                for sub in convert_linears_to_fp8(
                    child,
                    skip=skip,
                    min_in_features=min_in_features,
                    use_fast_accum=use_fast_accum,
                )
            )
    return replaced
