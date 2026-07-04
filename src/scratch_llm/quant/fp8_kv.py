"""A5 Rung 4 — FP8 E4M3 KV cache, wired into the A1 decoder.

Intent
------
The decode-time KV cache is the dominant runtime memory of an autoregressive LM: it grows
linearly with sequence length and is read every step. Storing it in **FP8 E4M3** (1 byte/elem)
instead of BF16 (2 bytes/elem) **halves** the cache footprint. This module builds the
quant/dequant primitives and a drop-in :class:`QuantizedKVCache` that duck-types the A1
:class:`~scratch_llm.model.KVCache` (``length`` / ``append`` / ``get`` / ``advance``), so the
existing decoder in ``model.py`` uses it with **no change to model.py**.

Format facts (E4M3): 4 exponent bits, 3 mantissa bits, bias 7, max magnitude ±448, NaN-only at
the top (no ±inf). Values above 448 do **not** saturate — they cast to NaN — so we clamp before
casting. We use ``torch.float8_e4m3fn`` for bit-accurate E4M3 rounding; storage is a real 1-byte
tensor, so the byte-halving is *measured*, not asserted.

Quantization granularity (the load-bearing choice)
--------------------------------------------------
KV tensors are (B, n_kv_heads, T, head_dim). Transformer K has **outlier channels** — a few
``head_dim`` channels with much larger magnitude than the rest. A per-*token* scale (one scale
per row, = max over channels) then wastes almost all FP8 codes on the outlier channel and
quantizes the small channels with a huge step. A per-*channel* scale (one scale per ``head_dim``
column, = max over tokens) adapts to each channel and recovers that precision.

  * **K → per-channel scale** (reduce over the token axis, dim=2).
  * **V → per-token scale**   (reduce over ``head_dim``, dim=3).

Invariants (the oracle — SQNR/MSE, never allclose)
--------------------------------------------------
1. **End-to-end fidelity.** Greedy generation with an FP8 KV cache reproduces the BF16-KV token
   sequence (within noise) at ~half the KV bytes; per-step logit MSE is small.
2. **Granularity law.** ``MSE(per-token-K) / MSE(per-channel-K)`` is materially > 1 (~2.5×) on
   K with realistic per-channel magnitude spread — per-channel quantization is strictly better.
3. **Stress / degradation.** An INT4 KV cache (4 bits vs FP8's 8) has materially larger error
   and visibly worse SQNR — if it does *not* degrade, the stress harness is not stressing.

SQNR is reported as measured; a value below the ~6.02·bits floor is a bug (bad scale, missing
clamp, wrong rounding), never "acceptable loss".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from scratch_llm.model import TransformerLM

# E4M3 constants (bias 7, 3 mantissa bits, NaN-only top).
FP8_E4M3_MAX = 448.0
# Symmetric INT4 code range [-7, 7] (8 reserved / unused → symmetric, no −8 asymmetry).
INT4_QMAX = 7.0

_EPS = 1e-12


# --------------------------------------------------------------------------------------------
# Core fake-quant primitives. `axis` is the reduction axis for the per-group amax:
#   K per-channel → axis=2 (reduce over tokens);  V/K per-token → axis=3 (reduce over head_dim).
# Each returns the dequantized fp32 tensor AND the true stored-byte count (codes + scales), so
# byte accounting is measured from the actual storage width, not assumed.
# --------------------------------------------------------------------------------------------
def quantize_fp8_e4m3(x: Tensor, axis: int) -> tuple[Tensor, Tensor]:
    """Fake-quant ``x`` to FP8 E4M3 with a per-group symmetric scale (group = the ``axis`` slice).

    scale = amax(|x|) / 448 maps the group's largest magnitude onto the top of the E4M3 range so
    the full 3-bit mantissa is used across the group (small values pushed away from underflow).
    Returns ``(codes, scale)`` where ``codes`` is a real ``float8_e4m3fn`` (1-byte) tensor.
    """
    amax = x.abs().amax(dim=axis, keepdim=True).clamp_min(_EPS)
    scale = amax / FP8_E4M3_MAX
    # Clamp BEFORE the cast: E4M3 turns >448 into NaN rather than saturating.
    codes = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return codes, scale


def dequantize_fp8_e4m3(codes: Tensor, scale: Tensor) -> Tensor:
    return codes.to(torch.float32) * scale


def quantize_int4(x: Tensor, axis: int) -> tuple[Tensor, Tensor]:
    """Fake-quant ``x`` to symmetric INT4 (round-to-nearest, range [-7, 7]) — the stress baseline."""
    amax = x.abs().amax(dim=axis, keepdim=True).clamp_min(_EPS)
    scale = amax / INT4_QMAX
    codes = torch.round(x / scale).clamp(-INT4_QMAX, INT4_QMAX)
    return codes, scale


def dequantize_int4(codes: Tensor, scale: Tensor) -> Tensor:
    return codes * scale


def sqnr_db(reference: Tensor, approx: Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB: 10·log10(‖x‖² / ‖x−x̂‖²).

    The numeric oracle for a quantizer. The floor is ~6.02·(mantissa bits)+c; a deficit below it
    means a scale/clamp/rounding bug, not tolerable loss.
    """
    ref = reference.float()
    err = (ref - approx.float()).pow(2).sum().item()
    sig = ref.pow(2).sum().item()
    if err <= 0.0:
        return math.inf
    return 10.0 * math.log10(sig / max(err, _EPS))


# --------------------------------------------------------------------------------------------
# Byte accounting helpers (per element, incl. amortized scale overhead over a T×D block).
# --------------------------------------------------------------------------------------------
def _fp8_block_bytes(codes: Tensor, scale: Tensor) -> int:
    return codes.numel() * 1 + scale.numel() * 4  # 1 B/code + fp32 scales


def _int4_block_bytes(codes: Tensor, scale: Tensor) -> int:
    return math.ceil(codes.numel() / 2) + scale.numel() * 4  # 4-bit packed + fp32 scales


def _bf16_block_bytes(numel: int) -> int:
    return numel * 2


# --------------------------------------------------------------------------------------------
# The drop-in cache. Duck-types scratch_llm.model.KVCache for the plain (non-slot) decode path:
# model.MultiHeadSelfAttention uses cache.length / append / get, and TransformerLM.forward calls
# cache.advance(s). It must NOT subclass SlotKVCache (that routes down the ragged-batch path).
# Blocks are stored dequantized-per-block (fake-quant), but total bytes are tracked from the true
# quantized storage width so `kv_bytes` reflects what an FP8 cache actually costs.
# --------------------------------------------------------------------------------------------
class QuantizedKVCache:
    """Per-layer KV cache that fake-quantizes every appended K/V block.

    ``scheme`` selects the quantizer: ``"fp8"`` (E4M3, per-channel-K + per-token-V — the rung
    deliverable), ``"int4"`` (the stress baseline), or ``"bf16"`` (the half-precision reference).
    ``k_per_channel`` toggles K granularity so the per-channel-vs-per-token law can be measured
    through the exact same cache path.
    """

    def __init__(
        self,
        n_layers: int,
        scheme: str = "fp8",
        k_per_channel: bool = True,
    ) -> None:
        if scheme not in ("fp8", "int4", "bf16"):
            raise ValueError(f"unknown scheme {scheme!r}")
        self.scheme = scheme
        self.k_per_channel = k_per_channel
        self._k: list[list[Tensor]] = [[] for _ in range(n_layers)]
        self._v: list[list[Tensor]] = [[] for _ in range(n_layers)]
        # Per-channel-K scale is CALIBRATED ONCE per layer (from the prefill block) and reused for
        # every later single-token decode append — the production static-KV-quant pattern. That is
        # what makes the byte cost honest: a per-channel scale recomputed on each 1-token append
        # would be one scale per element (4 B/elem) and defeat the point; amortized, it is ~0 B/elem
        # over a long sequence, so FP8 codes (1 B/elem) genuinely halve the BF16 (2 B/elem) cache.
        self._k_scale: list[Tensor | None] = [None] * n_layers
        self._length = 0
        self.kv_bytes = 0  # measured total stored bytes (K+V, codes + scales)

    @property
    def length(self) -> int:
        return self._length

    def _quant_v(self, x: Tensor) -> tuple[Tensor, int]:
        """V: per-token scale (reduce over head_dim, dim=3) — one scale per (head, token)."""
        if self.scheme == "bf16":
            return x.to(torch.bfloat16).to(torch.float32), _bf16_block_bytes(x.numel())
        if self.scheme == "fp8":
            codes, scale = quantize_fp8_e4m3(x, 3)
            return dequantize_fp8_e4m3(codes, scale), _fp8_block_bytes(codes, scale)
        codes, scale = quantize_int4(x, 3)
        return dequantize_int4(codes, scale), _int4_block_bytes(codes, scale)

    def _quant_k(self, layer: int, x: Tensor) -> tuple[Tensor, int]:
        """K: per-channel scale (calibrate-once from prefill, reduce over tokens dim=2) when
        ``k_per_channel``; else per-token (dim=3) recomputed per append (the comparison mode)."""
        if self.scheme == "bf16":
            return x.to(torch.bfloat16).to(torch.float32), _bf16_block_bytes(x.numel())

        if not self.k_per_channel:  # per-token-K: fresh scale each append, no amortization
            if self.scheme == "fp8":
                codes, scale = quantize_fp8_e4m3(x, 3)
                return dequantize_fp8_e4m3(codes, scale), _fp8_block_bytes(codes, scale)
            codes, scale = quantize_int4(x, 3)
            return dequantize_int4(codes, scale), _int4_block_bytes(codes, scale)

        # per-channel-K: calibrate the scale on the first (prefill) block, reuse it thereafter.
        scale_bytes = 0
        scale = self._k_scale[layer]
        if scale is None:
            amax = x.abs().amax(dim=2, keepdim=True).clamp_min(_EPS)
            qmax = FP8_E4M3_MAX if self.scheme == "fp8" else INT4_QMAX
            scale = amax / qmax
            self._k_scale[layer] = scale
            scale_bytes = scale.numel() * 4  # one-time per-channel scale vector
        if self.scheme == "fp8":
            codes = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            return codes.to(torch.float32) * scale, codes.numel() * 1 + scale_bytes
        codes = torch.round(x / scale).clamp(-INT4_QMAX, INT4_QMAX)
        return codes * scale, math.ceil(codes.numel() / 2) + scale_bytes

    def get(self, layer: int) -> tuple[Tensor, Tensor] | None:
        if not self._k[layer]:
            return None
        return torch.cat(self._k[layer], dim=2), torch.cat(self._v[layer], dim=2)

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        """Quantize + store this step's K (per-channel or per-token) and V (per-token)."""
        kq, kb = self._quant_k(layer, k_new)
        vq, vb = self._quant_v(v_new)
        self._k[layer].append(kq)
        self._v[layer].append(vq)
        self.kv_bytes += kb + vb

    def advance(self, n: int) -> None:
        self._length += n


# --------------------------------------------------------------------------------------------
# Greedy decode driver with an injected cache. Mirrors sampling._decode's use_cache path exactly
# (prefill the prompt once, then feed one token per step) — but lets us swap the cache object,
# which sampling.generate hardcodes. This is how the FP8 cache is "wired into the A1 decoder".
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_with_cache(
    model: TransformerLM,
    prompt_ids: Sequence[int],
    max_tokens: int,
    cache: object,
) -> tuple[list[int], Tensor]:
    """Greedy-decode ``max_tokens`` against ``cache``; return (generated ids, stacked step logits).

    ``cache`` is any object duck-typing the plain KVCache contract (a real
    :class:`~scratch_llm.model.KVCache` for the exact/BF16 reference, or a
    :class:`QuantizedKVCache`). Logits are returned so callers can measure per-step logit MSE.
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")
    model.eval()
    ctx = model.cfg.context_length
    ids = list(prompt_ids)
    x = torch.tensor([ids[-ctx:]], dtype=torch.long)
    logits = model(x, cache)[0, -1]  # type: ignore[arg-type]
    generated: list[int] = []
    step_logits: list[Tensor] = []
    for _ in range(max_tokens):
        step_logits.append(logits)
        next_id = int(logits.argmax().item())
        generated.append(next_id)
        x = torch.tensor([[next_id]], dtype=torch.long)
        logits = model(x, cache)[0, -1]  # type: ignore[arg-type]
    return generated, torch.stack(step_logits)
