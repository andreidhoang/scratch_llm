"""k3.checkpoint — the released Kimi K3 checkpoint (HF names, MXFP4 routed experts) into K3Model.

The checkpoint is Moonshot's ``KimiK3ForConditionalGeneration`` (HF modeling_kimi_k3.py :893-919):
a vision tower, a projector, and the text model at ``language_model`` — ``KimiLinearForCausalLM``
(modeling_kimi_linear.py :1236-1247), decoder under ``language_model.model.``, head at
``language_model.lm_head``. K3Model's names are the text model's with that prefix stripped
(model.py), so a load is a rename plus two format conversions.

Names. The release has 497,220 tensors in 96 shards (``model.safetensors.index.json``); every
family is one of these (checked against the real index by the network gate):

    language_model.model.{embed_tokens, layers.{i}.*, norm, output_attn_res_*}  →  prefix stripped
    language_model.lm_head.weight                                                →  lm_head.weight
    language_model.layers.{i}.*                  vLLM's spelling (nvidia/model.py:1761-1767) → same
    vision_tower.*, mm_projector.*               skipped: K3Model is the text model only
    layers.{i}.* with i ≥ num_layers             skipped: MTP draft layers (vLLM :1724-1738); the
                                                 release has none (num_nextn_predict_layers = 0)

Layer indices are 0-based, as in HF and ``K3Model.layers``: checkpoint ``layers.{i}`` is
``layer_id = i + 1``. Anything else keeps its name and so reaches ``load_state_dict`` as an
unexpected key. The MTP skip is unconditional: vLLM skips ``layers.{L + i}`` only when
``num_nextn_predict_layers > 0`` (nvidia/model.py:1721-1737), and ``K3Config`` carries no such
field. A ``layers.{i ≥ num_layers}`` tensor therefore lands in ``LoadReport.skipped``, never in
``unexpected``, so ``strict=True`` does not catch a checkpoint deeper than the config.

Formats. The shard headers store three kinds of tensor:

* F32 — ``A_log``, ``dt_bias``, the three ``*_conv1d.weight``, KDA's ``o_norm.weight`` and the QB
  bias ``gate.e_score_correction_bias``. They pass through as F32, and the target parameter must
  hold them exactly: fp32 or fp64. vLLM allocates A_log, dt_bias, the convs and the bias in fp32
  whatever the model dtype (kimi_k3/nvidia/kda.py:463-464, :474, :514-515; model.py:624-626), as
  HF does A_log and dt_bias (:520-527). A bf16 copy would move every KDA decay and, through
  s + b, the top-k set itself. ``model.to(torch.bfloat16)`` casts these too, so the load refuses
  such a model; a bf16 model casts every other parameter and leaves these in fp32 (the KDA and
  router math already runs in fp32 on bf16 activations).
* BF16 — every other dense tensor; converted to ``dtype``.
* MXFP4 — the routed experts ``experts.{e}.w{1,2,3}``: compressed-tensors
  ``mxfp4-pack-quantized`` (config.json ``quantization_config``: group 32, E8M0 scales). Each
  weight is two uint8 tensors, ``weight_packed`` [out, in/2] and ``weight_scale`` [out, in/32],
  decoded into one ``.weight`` [out, in] in ``dtype`` (:func:`dequantize_mxfp4`). The pairing is
  by suffix, as vLLM's (:1512-1513); in the release only the routed experts carry it.

``A_log`` is stored [a_log_size] (num_heads live entries and a zero tail). It passes through
unchanged: ``KDALayer._load_from_state_dict`` keeps [0:num_heads] and refuses a nonzero tail.

Not implemented: the vision tower and projector, MTP layers, lazy per-shard loading (the whole
translated state dict is built before ``load_state_dict``), tied embeddings.

Gates: ``tests/test_k3_checkpoint.py`` — a tiny K3Model exported in the release's layout (HF names,
F32/BF16 as the real headers, experts MXFP4-packed) loads strictly and bit-exactly; the MXFP4
decode ≡ an independent bit-level decoder; the name rules and refusals; and, network-only, the
real index and shard headers against ``k3_full`` on meta.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from scratch_llm.k3.config import K3Config
from scratch_llm.k3.model import K3Model
from scratch_llm.quant.nvfp4_mxfp4 import e2m1_decode, unpack_fp4

__all__ = [
    "MXFP4_GROUP",
    "LoadReport",
    "dequantize_mxfp4",
    "hf_to_k3_name",
    "hf_to_k3_state_dict",
    "load_hf_checkpoint",
]

#: Elements per E8M0 scale: config.json ``quantization_config`` group_size, the OCP MX block.
MXFP4_GROUP = 32

_E8M0_NAN = 0xFF  # OCP MX v1.0 E8M0: no zero, no infinity, 0xFF is NaN
_PREFIXES = ("language_model.model.", "language_model.")  # longest first
_SKIPPED = ("vision_tower.", "mm_projector.")
_PACKED, _SCALE = ".weight_packed", ".weight_scale"
_LAYER = re.compile(r"layers\.(\d+)\.")


# ---------------------------------------------------------------------------
# MXFP4
# ---------------------------------------------------------------------------


def dequantize_mxfp4(
    weight_packed: Tensor, weight_scale: Tensor, *, dtype: torch.dtype = torch.float32
) -> Tensor:
    """One compressed-tensors MXFP4 weight → [out, in] in ``dtype``:

        w[r, c] = E2M1(code[r, c]) · 2^(weight_scale[r, c // 32] − 127)

    weight_packed  uint8 [out, in/2]: byte j of a row holds element 2j in its LOW nibble and
                   2j + 1 in its high nibble (vLLM ``break_fp4_bytes``, nvfp4_emulation_utils.py
                   :303-320; ``quant.nvfp4_mxfp4.unpack_fp4`` is the same order)
    E2M1 code      bit 3 the sign, bits 2..0 an index into (0, ½, 1, 1½, 2, 3, 4, 6); code 0x8 is
                   −0 (``quant.nvfp4_mxfp4.e2m1_decode``, break_fp4_bytes :313-318)
    weight_scale   uint8 [out, in/32], E8M0: the value 2^(s − 127) for s ∈ [0, 254]; s = 0xFF is
                   NaN (OCP MX v1.0, torch ``float8_e8m0fnu``) and is refused

    The E8M0 scale is an exponent field with no mantissa. fp64 (bias 1023) holds every finite
    code as a normal number, so writing s − 127 + 1023 into fp64's exponent bits is exact. vLLM's
    MXFP4 linear kernels decode it the same way, code 0 as 2^−127: they view ``weight_scale`` as
    ``torch.float8_e8m0fnu`` (kernels/linear/mxfp4/xpu.py:39, humming.py:53). Only vLLM's
    DeepSeek-V4 DeepGEMM helper differs: it bit-casts ``s << 23`` into fp32
    (deepseek_v4/nvidia/model.py:331-332), where code 0 lands on +0.0 and 0xFF on +inf.

    Each product is ±0 or a two-significant-bit number of magnitude 2^−128 … 1.5 · 2^129, exact
    in fp64. The cast to ``dtype`` is then the only rounding point, and it must be exact: fp32
    and bf16 hold every value below 2^128 (2^−128 is a subnormal in both), so only the largest
    codes under scales 253-254 overflow them; fp16 fails far earlier. A cast that changes a value
    raises.
    """
    if weight_packed.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
        raise ValueError(
            f"MXFP4 halves must be uint8; got {weight_packed.dtype}, {weight_scale.dtype}"
        )
    if weight_packed.ndim != 2:
        raise ValueError(f"weight_packed must be [out, in/2]; got {tuple(weight_packed.shape)}")
    rows, cols = weight_packed.shape[0], 2 * weight_packed.shape[1]
    groups = cols // MXFP4_GROUP
    if cols % MXFP4_GROUP or tuple(weight_scale.shape) != (rows, groups):
        raise ValueError(
            f"weight_packed {tuple(weight_packed.shape)} encodes [{rows}, {cols}], which needs "
            f"in % {MXFP4_GROUP} == 0 and weight_scale [{rows}, {groups}]; "
            f"got weight_scale {tuple(weight_scale.shape)}"
        )
    if bool((weight_scale == _E8M0_NAN).any()):
        raise ValueError("E8M0 scale 0xFF is NaN; refusing to load")

    codes = unpack_fp4(weight_packed, rows * cols).reshape(rows, cols)
    values = e2m1_decode(codes).to(torch.float64)
    scale = ((weight_scale.to(torch.int64) + (1023 - 127)) << 52).view(torch.float64)
    exact = (values.view(rows, groups, MXFP4_GROUP) * scale[..., None]).view(rows, cols)
    out = exact.to(dtype)
    if not torch.equal(out.to(torch.float64), exact):
        raise ValueError(f"MXFP4 values are not exact in {dtype} (overflow or underflow)")
    return out


# ---------------------------------------------------------------------------
# Names and state dict
# ---------------------------------------------------------------------------


def hf_to_k3_name(name: str, cfg: K3Config) -> str | None:
    """The K3Model name of one checkpoint tensor, or None for a deliberately skipped one (vision,
    projector, MTP). Both MXFP4 halves map to the ``.weight`` they encode. A name outside the
    known families comes back unchanged, so the load reports it as unexpected."""
    if name.startswith(_SKIPPED):
        return None
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name.removeprefix(prefix)
            break
    layer = _LAYER.match(name)
    if layer is not None and int(layer.group(1)) >= cfg.num_layers:
        return None
    for suffix in (_PACKED, _SCALE):
        if name.endswith(suffix):
            return name.removesuffix(suffix) + ".weight"
    return name


def _holds(dtype: torch.dtype, source: torch.dtype) -> bool:
    """Every ``source`` value is exact in ``dtype``: promoting the two adds nothing (bf16 ⊂ fp32 ⊂
    fp64; fp16 holds neither bf16's range nor fp32)."""
    return torch.promote_types(source, dtype) == dtype


def _dense(name: str, tensor: Tensor, dtype: torch.dtype) -> Tensor:
    """BF16 → ``dtype``; F32 unchanged (module docstring). The release stores nothing else."""
    if tensor.dtype == torch.bfloat16:
        if not _holds(dtype, torch.bfloat16):
            raise ValueError(f"{name}: BF16 does not convert to {dtype} exactly")
        return tensor.to(dtype)
    if tensor.dtype == torch.float32:
        return tensor
    raise ValueError(
        f"{name}: {tensor.dtype} — the checkpoint stores BF16, F32 and MXFP4 uint8 pairs only"
    )


def _translate(
    named_tensors: Iterable[tuple[str, Tensor]], cfg: K3Config, dtype: torch.dtype
) -> tuple[dict[str, Tensor], list[str]]:
    """(K3 state dict, skipped checkpoint names). The two halves of an MXFP4 weight may arrive in
    either order and apart; each is held until its partner arrives."""
    state: dict[str, Tensor] = {}
    halves: dict[str, dict[str, Tensor]] = {}  # K3 name -> the MXFP4 halves seen so far
    skipped: list[str] = []
    for hf_name, tensor in named_tensors:
        name = hf_to_k3_name(hf_name, cfg)
        if name is None:
            skipped.append(hf_name)
            continue
        suffix = next((s for s in (_PACKED, _SCALE) if hf_name.endswith(s)), None)
        if suffix is None:
            value = _dense(hf_name, tensor, dtype)
        else:
            pair = halves.setdefault(name, {})
            if suffix in pair:
                raise ValueError(f"{hf_name}: the same MXFP4 half appears twice")
            pair[suffix] = tensor
            if len(pair) < 2:
                continue
            del halves[name]
            try:
                value = dequantize_mxfp4(pair[_PACKED], pair[_SCALE], dtype=dtype)
            except ValueError as err:
                raise ValueError(f"{name} (MXFP4): {err}") from err
        if name in state:
            raise ValueError(f"{hf_name}: a second checkpoint tensor for {name}")
        state[name] = value
    if halves:
        raise ValueError(f"MXFP4 halves without their partner: {sorted(halves)}")
    return state, skipped


def hf_to_k3_state_dict(
    named_tensors: Iterable[tuple[str, Tensor]],
    cfg: K3Config,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Checkpoint (name, tensor) pairs → a K3Model state dict: names stripped, skipped names
    dropped, BF16 and dequantized MXFP4 in ``dtype``, F32 as F32, ``A_log`` at its stored
    [a_log_size] (module docstring). Raises on an unknown dtype, a ``dtype`` that cannot hold
    BF16 (or an MXFP4 value) exactly, a duplicate name, an MXFP4 half without its partner or a
    malformed MXFP4 pair, naming the tensor."""
    return _translate(named_tensors, cfg, dtype)[0]


@dataclass(frozen=True)
class LoadReport:
    """What :func:`load_hf_checkpoint` did not load, by kind."""

    missing: tuple[str, ...]  # model parameters the checkpoint did not provide
    unexpected: tuple[str, ...]  # translated checkpoint names with no model parameter
    skipped: tuple[str, ...]  # checkpoint names left out on purpose: vision, projector, MTP


def load_hf_checkpoint(
    model: K3Model, named_tensors: Iterable[tuple[str, Tensor]], *, strict: bool = True
) -> LoadReport:
    """Translate (:func:`hf_to_k3_state_dict`, ``dtype`` = the model's embedding dtype) and
    ``model.load_state_dict``, which copies every tensor into its parameter's own dtype.

    Nothing is rounded. Before any parameter is written, every translated tensor must be exact in
    its parameter's dtype, or the load raises naming each one that is not. That refuses the F32
    families in a bf16 model, which is what ``model.to(torch.bfloat16)`` makes; a bf16 model
    must keep them fp32 (module docstring). The MXFP4 decode raises rather than round.

    ``strict=True`` raises ``load_state_dict``'s RuntimeError, naming every missing and unexpected
    key; ``strict=False`` loads what matches and reports the rest. In both modes a shape mismatch
    raises with the tensor's name, as does a nonzero ``A_log`` tail (KDALayer). As with
    ``load_state_dict``, a raise at that stage can leave the model partly loaded.
    """
    state, skipped = _translate(named_tensors, model.cfg, model.embed_tokens.weight.dtype)
    targets = {name: t.dtype for name, t in model.state_dict().items()}
    rounded = [
        f"{name} ({t.dtype} into {targets[name]})"
        for name, t in state.items()
        if name in targets and not _holds(targets[name], t.dtype)
    ]
    if rounded:
        raise ValueError(
            f"{len(rounded)} checkpoint tensors would round into narrower parameters: "
            + ", ".join(rounded)
        )
    result = model.load_state_dict(state, strict=strict)
    return LoadReport(tuple(result.missing_keys), tuple(result.unexpected_keys), tuple(skipped))
