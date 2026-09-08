"""S1/S-R2 — E4M3 KV cache with **per-head** scales, calibrated once and static thereafter.

Spec: ``experiments/S1/S-R2/spec.md``   ·   Map: ``experiments/S1/S-R2/map.md``

Why per-head, when ``quant/fp8_kv.py`` already argued for per-channel
--------------------------------------------------------------------
The A5 rung (``quant/fp8_kv.py``) chose a per-*channel* scale for K, reducing over tokens, because
transformer K has outlier channels and a per-token scale wastes the code range on them. That
argument is still true — and it is not available to a serving engine.

A fused attention kernel reads the cache inside the QK^T loop. A per-channel K scale multiplies
along the head-dim axis, which is the axis being contracted, so it cannot be hoisted out of the
dot product: it has to be folded into Q before the kernel runs (possible, but it changes the Q
path and the kernel's operand contract), or applied element-wise inside the inner loop (which
gives back the bandwidth the quantization bought). A per-**head** scale is one number per
``(layer, head)`` that is constant across every element the kernel touches for that head, so it
multiplies the accumulator once at the end — the same hoisting argument that makes per-output-
channel weight scales free (``quant/fp8_weights.py``).

That is the trade this rung is making explicit: per-head is coarser than per-channel and finer
than the per-tensor scalar production actually ships (vLLM stores one ``k_scale`` and one
``v_scale`` per attention layer — ``kv_cache.py:18-39,124-127``, see map.md). Whether the coarsening
from per-channel costs measurable perplexity is not a matter of opinion; it is the Δppl the rung's
quality gate reads.

The scale derivation
--------------------
For a KV block of shape ``(B, H, T, D)``:

    scale[h] = amax over {batch, tokens, head_dim} of |x[:, h, :, :]|  /  448      -> (1, H, 1, 1)

Three decisions are packed into that line, and each has a failure mode:

  * **Reduce over batch, not per (b, h).** A per-``(b, h)`` scale is not shareable: in a paged
    cache one physical block can be read for a different sequence after a prefix share, and a
    scale keyed on the slot that wrote it would dequantize another request's tokens. Reducing over
    B makes the scale a property of the *head*, which is what the layout says it is.
  * **Calibrate once, on the prefill block; static afterwards.** A scale recomputed on each
    one-token append is one scale per token per head — that is a per-token scale wearing a
    per-head name, it costs 4 bytes per head per token of extra cache, and it makes the stored
    codes depend on when they were written. Static per-head scales cost ``H`` floats per layer,
    amortized to nothing, so the 1-byte codes genuinely halve a bf16 cache.
  * **Saturate, do not wrap.** Static calibration's failure mode is a later token whose magnitude
    exceeds the calibrated amax. E4M3's raw cast returns **NaN** above 464 (see
    ``fp8_weights.py``), and one NaN in K is one whole softmax row of NaN. So the quantizer clamps
    to +-448 and :class:`PerHeadFp8KVCache` *counts* the clamped elements. The count is the
    evidence for whether the calibration window was long enough; it is not a warning to ignore.

Invariants the tests pin (``tests/quant/test_s1_s_r2.py``), all on CPU:
  1. the round trip lands within E4M3's own quantum, relative above the min normal and absolute
     below it — no chosen constant appears, only the format's;
  2. a value past the calibrated range saturates to ``448 * scale`` and is counted, never NaN;
  3. the scales are per head: a tensor whose heads differ by orders of magnitude is quantized far
     better by ``H`` scales than by one, and the test fails if a single global scale would do;
  4. quantize(dequantize(codes)) == codes at a fixed scale — a static cache re-reads its own bytes
     forever, so the map must be idempotent or the cache drifts.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scratch_llm.quant.fp8_weights import E4M3_MAX, MIN_SCALE, saturating_cast_e4m3

__all__ = [
    "PerHeadFp8KVCache",
    "bf16_reference_bytes",
    "dequantize_per_head",
    "per_head_amax",
    "per_head_scale",
    "quantize_per_head",
    "saturated_count",
]


def per_head_amax(x: Tensor) -> Tensor:
    """``(B, H, T, D) -> (1, H, 1, 1)``: the max magnitude anywhere in each head.

    Reduced over batch as well as tokens and head-dim; see the module docstring for why the batch
    axis must go into the reduction rather than stay in the scale's shape.
    """
    if x.ndim != 4:
        raise ValueError(f"KV block must be (B, H, T, D), got shape {tuple(x.shape)}")
    return x.detach().abs().amax(dim=(0, 2, 3), keepdim=True).to(torch.float32)


def per_head_scale(x: Tensor) -> Tensor:
    """The symmetric per-head scale ``amax / 448``, floored so an all-zero head cannot divide by 0."""
    return (per_head_amax(x) / E4M3_MAX).clamp_min(MIN_SCALE)


def quantize_per_head(x: Tensor, scale: Tensor) -> Tensor:
    """``(B, H, T, D)`` -> E4M3 codes, given a ``(1, H, 1, 1)`` scale. Saturating, never NaN."""
    _check_scale(x, scale)
    return saturating_cast_e4m3(x.to(torch.float32) / scale)


def dequantize_per_head(codes: Tensor, scale: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Codes back to ``dtype``. Exact in fp32: one multiply, no rounding of the code itself."""
    return (codes.to(torch.float32) * scale).to(dtype)


def saturated_count(x: Tensor, scale: Tensor) -> int:
    """How many elements of ``x`` the clamp had to move — the static-calibration overflow counter."""
    _check_scale(x, scale)
    return int((x.detach().abs() > E4M3_MAX * scale).sum().item())


def _check_scale(x: Tensor, scale: Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"KV block must be (B, H, T, D), got shape {tuple(x.shape)}")
    if tuple(scale.shape) != (1, x.shape[1], 1, 1):
        raise ValueError(
            f"per-head scale must have shape (1, H, 1, 1) = {(1, x.shape[1], 1, 1)}, "
            f"got {tuple(scale.shape)} — a scale of any other shape is not per-head"
        )


class PerHeadFp8KVCache:
    """Per-layer KV cache storing E4M3 codes with static per-head scales.

    Duck-types the plain-decode cache contract that ``model.MultiHeadSelfAttention`` uses
    (``length`` / ``append`` / ``get`` / ``advance``), like ``quant/fp8_kv.QuantizedKVCache``, so an
    engine takes it without any change to ``model.py``. It must not subclass ``SlotKVCache`` —
    that routes down the ragged-batch path, which has a different write contract.

    Storage is real ``float8_e4m3fn``: one byte per element. :attr:`kv_bytes` is accumulated from
    the actual storage width (codes + the one-time scale vectors), so the halving against bf16 is
    measured rather than asserted.

    ``k_saturated`` / ``v_saturated`` count elements the clamp moved after calibration. A non-zero
    count is not a failure — it is the number that says whether the calibration window covered the
    distribution, and it belongs in the rung's evidence next to Δppl.
    """

    def __init__(self, n_layers: int, *, dtype: torch.dtype = torch.float32) -> None:
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        self.n_layers = n_layers
        self.dtype = dtype
        self._k: list[list[Tensor]] = [[] for _ in range(n_layers)]
        self._v: list[list[Tensor]] = [[] for _ in range(n_layers)]
        self._k_scale: list[Tensor | None] = [None] * n_layers
        self._v_scale: list[Tensor | None] = [None] * n_layers
        self._length = 0
        self.kv_bytes = 0
        self.k_saturated = 0
        self.v_saturated = 0
        self.n_elements = 0

    @property
    def length(self) -> int:
        return self._length

    def scales(self, layer: int) -> tuple[Tensor | None, Tensor | None]:
        """This layer's ``(k_scale, v_scale)``, each ``(1, H, 1, 1)`` or ``None`` before calibration."""
        return self._k_scale[layer], self._v_scale[layer]

    def set_scales(self, layer: int, k_scale: Tensor, v_scale: Tensor) -> None:
        """Install scales from outside — the checkpoint-calibrated deployment path.

        vLLM loads ``k_scale``/``v_scale`` from the checkpoint and warns loudly when they are
        absent (``kv_cache.py:146-151``), because an uncalibrated FP8 cache silently running at
        scale 1.0 is a quality bug that looks like a working system. Same here: an engine that has
        offline scales installs them before the first append and never enters the calibrate path.
        """
        for s in (k_scale, v_scale):
            if s.ndim != 4 or s.shape[0] != 1 or s.shape[2] != 1 or s.shape[3] != 1:
                raise ValueError(f"scale must be (1, H, 1, 1), got {tuple(s.shape)}")
        self._k_scale[layer] = k_scale.to(torch.float32)
        self._v_scale[layer] = v_scale.to(torch.float32)
        self.kv_bytes += (k_scale.numel() + v_scale.numel()) * 4

    def saturated_fraction(self) -> float:
        """Clamped elements over all stored elements. 0.0 before anything is appended."""
        if self.n_elements == 0:
            return 0.0
        return (self.k_saturated + self.v_saturated) / self.n_elements

    def _scale_for(self, layer: int, x: Tensor, which: str) -> Tensor:
        store = self._k_scale if which == "k" else self._v_scale
        scale = store[layer]
        if scale is None:
            scale = per_head_scale(x)
            store[layer] = scale
            self.kv_bytes += scale.numel() * 4  # one-time (1, H, 1, 1) fp32 vector
        return scale

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        """Quantize and store this step's K and V at the layer's (calibrated) per-head scales."""
        k_scale = self._scale_for(layer, k_new, "k")
        v_scale = self._scale_for(layer, v_new, "v")
        self.k_saturated += saturated_count(k_new, k_scale)
        self.v_saturated += saturated_count(v_new, v_scale)
        k_codes = quantize_per_head(k_new, k_scale)
        v_codes = quantize_per_head(v_new, v_scale)
        self._k[layer].append(k_codes)
        self._v[layer].append(v_codes)
        self.kv_bytes += k_codes.numel() + v_codes.numel()  # 1 B per code, measured
        self.n_elements += k_codes.numel() + v_codes.numel()

    def get(self, layer: int) -> tuple[Tensor, Tensor] | None:
        """Dequantized ``(K, V)`` for this layer, concatenated along the token axis."""
        if not self._k[layer]:
            return None
        k_scale, v_scale = self._k_scale[layer], self._v_scale[layer]
        assert k_scale is not None and v_scale is not None  # set by the first append
        k = torch.cat(self._k[layer], dim=2)
        v = torch.cat(self._v[layer], dim=2)
        return (
            dequantize_per_head(k, k_scale, self.dtype),
            dequantize_per_head(v, v_scale, self.dtype),
        )

    def codes(self, layer: int) -> tuple[Tensor, Tensor] | None:
        """The stored bytes themselves — ``float8_e4m3fn``, ``element_size() == 1``."""
        if not self._k[layer]:
            return None
        return torch.cat(self._k[layer], dim=2), torch.cat(self._v[layer], dim=2)

    def advance(self, n: int) -> None:
        self._length += n


def bf16_reference_bytes(n_elements: int) -> int:
    """What the same cache would cost in bf16 — the floor :attr:`PerHeadFp8KVCache.kv_bytes` is read
    against. Two bytes per element and no scales."""
    return n_elements * 2
