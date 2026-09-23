"""K3/checkpoint gates (spec §7, loader): the released checkpoint's names and formats into K3Model.

1. Round trip: a tiny K3Model's own weights, exported in the release's layout — HF names, F32 and
   BF16 exactly where the real shard headers have them, routed experts MXFP4-packed by the packer
   below, A_log zero-padded to a_log_size — load strictly into a fresh model. Every parameter
   equals the source bit for bit and the logits are identical. Both layer spellings (HF's
   ``language_model.model.layers.`` and vLLM's ``language_model.layers.``) and both MXFP4 half
   orders are exercised.
2. ``dequantize_mxfp4`` ≡ a decoder written here from the bit layouts alone (a Python loop over
   bytes; E2M1 from its exponent and mantissa fields, E8M0 through ``math.ldexp``): all 256 bytes
   under every finite scale code, fp64 over the whole range and fp32/bf16 over the range they
   hold, compared bitwise (so −0 counts); named values (±0, ±6, nibble order); the refusals.
3. Names: strict mode reports a missing and an unexpected key; vision, projector and MTP names are
   skipped (MTP synthetically: the release has none); a lone MXFP4 half, a duplicate, an unknown
   dtype and a shape mismatch raise with the tensor's name.
4. A_log: the padded storage passes the translation unchanged and loads; a nonzero tail is
   refused by KDALayer.
5. Nothing rounds: a blanket bf16 model (its F32 families in bf16) is refused before any write,
   naming every such tensor; a bf16 model that keeps them fp32 loads them bit-exactly and runs.
6. (network + slow, self-skips unless SCRATCH_LLM_NETWORK_TESTS=1) The real index at a pinned
   commit: every text tensor maps onto a ``k3_full`` meta parameter and every parameter is
   covered; the shard headers of layers 0 (KDA + dense MLP), 1 (KDA + MoE), 3 (MLA + MoE) and of
   the head equal the meta shapes (experts dequantized); layer 0's real A_log loads through
   KDALayer's tail check; one real expert decodes exactly to bf16.

The packer below quantizes like a minmax observer (per row and 32-group, the smallest power-of-two
scale that saturates nothing, nearest E2M1 code); the source model then takes the decoded values,
so the round trip needs no tolerance. It is written independently of ``quant.nvfp4_mxfp4``, which
the loader reuses.
"""

from __future__ import annotations

import json
import math
import os
import struct
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor

from scratch_llm.k3.checkpoint import (
    MXFP4_GROUP,
    dequantize_mxfp4,
    hf_to_k3_name,
    hf_to_k3_state_dict,
    load_hf_checkpoint,
)
from scratch_llm.k3.config import (
    K3Config,
    KDAConfig,
    MLAConfig,
    MoEConfig,
    build_layer_pattern,
    k3_full,
)
from scratch_llm.k3.core.kda import KDALayer
from scratch_llm.k3.model import K3Model

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # E2M1 magnitudes, for the packer only
_INT_VIEW = {torch.float64: torch.int64, torch.float32: torch.int32, torch.bfloat16: torch.int16}
# F32 in the release's shard headers (model-00001/00002); every other dense tensor is BF16.
_F32_SUFFIXES = (
    "A_log",
    "dt_bias",
    "conv1d.weight",
    "o_norm.weight",
    "e_score_correction_bias",
)


def _tiny_cfg() -> K3Config:
    """Layers 1-3 KDA, 4 MLA (period 4, terminal); layer 1 dense, 2-4 MoE. Expert widths are 32
    so each expert matrix is one MXFP4 group wide; a_log_size 4 > num_heads 2 stores a tail."""
    kda_layers, mla_layers = build_layer_pattern(4)
    assert kda_layers == (1, 2, 3) and mla_layers == (4,)
    return K3Config(
        hidden_size=32,
        num_layers=4,
        kda_layers=kda_layers,
        mla_layers=mla_layers,
        vocab_size=64,
        kda=KDAConfig(num_heads=2, head_dim=8, decay_rank=4, a_log_size=4),
        mla=MLAConfig(
            num_heads=2,
            q_lora_rank=12,
            kv_lora_rank=10,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=6,
        ),
        moe=MoEConfig(
            num_experts=4,
            top_k=2,
            num_shared_experts=1,
            expert_intermediate=32,
            latent_size=32,
            dense_intermediate=48,
        ),
        attn_res_block_size=2,
        max_position_embeddings=64,
    )


def _bits(t: Tensor) -> Tensor:
    return t.view(_INT_VIEW[t.dtype])


# ---------------------------------------------------------------------------
# Independent MXFP4 encoder / decoder (plain Python over the bit layouts).
# ---------------------------------------------------------------------------


def _e2m1_value(code: int) -> float:
    """E2M1 from its fields: sign | 2 exponent bits (bias 1) | 1 mantissa bit; exponent 0 is
    subnormal (0 or ½)."""
    sign, exponent, mantissa = code >> 3, (code >> 1) & 0b11, code & 1
    normal = (1.0 + 0.5 * mantissa) * 2.0 ** (exponent - 1)
    magnitude = 0.5 * mantissa if exponent == 0 else normal
    return -magnitude if sign else magnitude


def _decode_reference(packed: Tensor, scale: Tensor) -> Tensor:
    """fp64 [rows, 2·bytes]: element 2j is byte j's LOW nibble (vLLM break_fp4_bytes), scaled by
    2^(s − 127) of its 32-group. Python floats are fp64, and every product is exact there."""
    rows = []
    for byte_row, scale_row in zip(packed.tolist(), scale.tolist(), strict=True):
        row = []
        for j, byte in enumerate(byte_row):
            for c, code in ((2 * j, byte & 0xF), (2 * j + 1, byte >> 4)):
                row.append(math.ldexp(_e2m1_value(code), scale_row[c // MXFP4_GROUP] - 127))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float64)


def _pack_mxfp4(w: Tensor) -> tuple[Tensor, Tensor]:
    """w [out, in] → (weight_packed uint8 [out, in/2], weight_scale uint8 [out, in/32]). Scale
    2^e with e the smallest integer such that amax / 2^e ≤ 6; codes round to the nearest E2M1
    magnitude; the sign bit is set for negative inputs, so a small negative rounds to −0."""
    packed, scales = [], []
    for row in w.tolist():
        codes, row_scales = [], []
        for g in range(0, len(row), MXFP4_GROUP):
            group = row[g : g + MXFP4_GROUP]
            amax = max(abs(x) for x in group)
            mantissa, exponent = math.frexp(amax / 6.0)  # amax/6 = mantissa · 2^exponent
            e = -127 if amax == 0 else exponent - 1 if mantissa == 0.5 else exponent
            row_scales.append(e + 127)
            for x in group:
                q = abs(x) / 2.0**e
                index = min(range(8), key=lambda i: abs(_E2M1[i] - q))
                codes.append(index | (0x8 if x < 0 else 0))
        packed.append([lo | (hi << 4) for lo, hi in zip(codes[0::2], codes[1::2], strict=True)])
        scales.append(row_scales)
    return torch.tensor(packed, dtype=torch.uint8), torch.tensor(scales, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# A tiny model in the release's checkpoint layout.
# ---------------------------------------------------------------------------


def _hf_name(name: str, layer_prefix: str) -> str:
    if name.startswith("lm_head."):
        return "language_model." + name
    if name.startswith("layers."):
        return layer_prefix + name.removeprefix("layers.")
    return "language_model.model." + name


@torch.no_grad()
def _source_model(cfg: K3Config) -> tuple[K3Model, dict[str, tuple[Tensor, Tensor]]]:
    """A model whose every parameter is off its init (norms off 1, A_log and the QB bias off 0)
    and exactly representable in the release's storage: BF16 values where the checkpoint stores
    BF16, MXFP4 values in the experts. Returns it and each expert's packed halves by K3 name."""
    torch.manual_seed(0)
    model = K3Model(cfg)
    gen = torch.Generator().manual_seed(1)
    packed: dict[str, tuple[Tensor, Tensor]] = {}
    for name, p in model.named_parameters():
        p.add_(0.1 * torch.randn(p.shape, generator=gen))
        if ".experts." in name:
            packed[name] = _pack_mxfp4(p)
            p.copy_(_decode_reference(*packed[name]))
        elif not name.endswith(_F32_SUFFIXES):
            p.copy_(p.to(torch.bfloat16))
    return model, packed


def _export(
    model: K3Model,
    packed: dict[str, tuple[Tensor, Tensor]],
    layer_prefix: str = "language_model.model.layers.",
) -> list[tuple[str, Tensor]]:
    """(HF name, tensor) in the release's dtypes; A_log zero-padded to a_log_size."""
    kda = model.cfg.kda
    out: list[tuple[str, Tensor]] = []
    for name, p in model.state_dict().items():
        hf = _hf_name(name, layer_prefix)
        if name in packed:
            stem = hf.removesuffix(".weight")
            out += [
                (stem + ".weight_packed", packed[name][0]),
                (stem + ".weight_scale", packed[name][1]),
            ]
        elif name.endswith("A_log"):
            out.append((hf, torch.cat([p, p.new_zeros(kda.a_log_size - kda.num_heads)])))
        elif name.endswith(_F32_SUFFIXES):
            out.append((hf, p.clone()))
        else:
            out.append((hf, p.to(torch.bfloat16)))
    return out


def _ids(cfg: K3Config) -> Tensor:
    return torch.randint(0, cfg.vocab_size, (2, 9), generator=torch.Generator().manual_seed(2))


# ---------------------------------------------------------------------------
# 1. Round trip.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layer_prefix", ["language_model.model.layers.", "language_model.layers."], ids=["hf", "vllm"]
)
def test_round_trip_loads_the_release_layout_bit_exactly(layer_prefix: str) -> None:
    cfg = _tiny_cfg()
    source, packed = _source_model(cfg)
    exported = _export(source, packed, layer_prefix)
    assert {t.dtype for _, t in exported} == {torch.float32, torch.bfloat16, torch.uint8}

    torch.manual_seed(3)
    fresh = K3Model(cfg)
    before = dict(fresh.named_parameters())
    for name, p in source.named_parameters():  # nothing matches before the load
        assert not torch.equal(before[name], p), name

    # reversed: every MXFP4 scale arrives before its packed half
    report = load_hf_checkpoint(fresh, reversed(exported))
    assert report.missing == report.unexpected == report.skipped == ()
    loaded = dict(fresh.named_parameters())
    for name, p in source.named_parameters():
        assert torch.equal(_bits(loaded[name].detach()), _bits(p.detach())), name

    ids = _ids(cfg)
    with torch.no_grad():
        assert torch.equal(fresh(ids), source(ids))


# ---------------------------------------------------------------------------
# 2. MXFP4 decode ≡ the bit-level reference.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "max_scale"),
    [(torch.float64, 254), (torch.float32, 252), (torch.bfloat16, 252)],
    ids=["fp64", "fp32", "bf16"],
)
def test_mxfp4_decode_matches_the_bit_level_reference(dtype: torch.dtype, max_scale: int) -> None:
    """Row r holds all 256 bytes (16 groups of 16 bytes); group g takes scale code
    (r + g) mod (max_scale + 1). Over the rows every byte meets every scale code, and the scale
    varies inside a row, so a wrong group index or nibble order cannot pass. max_scale is the
    largest code whose largest value (6 · 2^(s − 127)) the dtype holds: fp32 and bf16 top out
    below 2^128, so 252 (1.5 · 2^127); fp64 takes every finite code."""
    n = max_scale + 1
    groups = 256 // (MXFP4_GROUP // 2)
    packed = torch.arange(256, dtype=torch.uint8).repeat(n, 1)
    scale = ((torch.arange(n)[:, None] + torch.arange(groups)[None, :]) % n).to(torch.uint8)
    got = dequantize_mxfp4(packed, scale, dtype=dtype)
    want = _decode_reference(packed, scale).to(dtype)  # exact: in range by construction
    assert got.dtype == dtype and got.shape == (n, 512)
    assert torch.equal(_bits(got), _bits(want))


def test_mxfp4_named_values() -> None:
    """Byte 0x80 = (+0, −0), 0x7F = (−6, +6), 0x21 = (½, 1): the LOW nibble is the even element.
    Scale 0 is 2^−127, as torch's float8_e8m0fnu reads it (not the DeepSeek-V4 DeepGEMM helper's
    bitcast 0), scale 254 is 2^127."""
    packed = torch.zeros(3, 16, dtype=torch.uint8)
    packed[:, :3] = torch.tensor([0x80, 0x7F, 0x21], dtype=torch.uint8)
    scale = torch.tensor([[127], [0], [254]], dtype=torch.uint8)
    w = dequantize_mxfp4(packed, scale, dtype=torch.float64)
    assert w[0, :6].tolist() == [0.0, -0.0, -6.0, 6.0, 0.5, 1.0]
    assert not w[0, 0].signbit() and w[0, 1].signbit()
    assert w[1, 4].item() == 2.0**-128 and w[1, 3].item() == 6 * 2.0**-127
    assert w[2, 2].item() == -6 * 2.0**127


@pytest.mark.parametrize(
    ("byte", "scale", "dtype", "match"),
    [
        (0x01, 0xFF, torch.float64, "0xFF is NaN"),
        (0x07, 254, torch.float32, "not exact"),  # 6 · 2^127 overflows fp32
        (0x07, 254, torch.bfloat16, "not exact"),
        (0x01, 0, torch.float16, "not exact"),  # 2^−128 underflows fp16
    ],
)
def test_mxfp4_refuses_nan_scales_and_inexact_casts(
    byte: int, scale: int, dtype: torch.dtype, match: str
) -> None:
    packed = torch.full((1, 16), byte, dtype=torch.uint8)
    with pytest.raises(ValueError, match=match):
        dequantize_mxfp4(packed, torch.tensor([[scale]], dtype=torch.uint8), dtype=dtype)


@pytest.mark.parametrize(
    ("packed", "scale", "match"),
    [
        (torch.zeros(2, 16, dtype=torch.int8), torch.zeros(2, 1, dtype=torch.uint8), "uint8"),
        (torch.zeros(2, 16, dtype=torch.uint8), torch.zeros(1, 2, dtype=torch.uint8), "needs"),
        (torch.zeros(2, 8, dtype=torch.uint8), torch.zeros(2, 1, dtype=torch.uint8), "needs"),
        (torch.zeros(32, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8), r"\[out, in/2\]"),
    ],
    ids=["dtype", "scale-shape", "partial-group", "rank"],
)
def test_mxfp4_refuses_malformed_pairs(packed: Tensor, scale: Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        dequantize_mxfp4(packed, scale)


# ---------------------------------------------------------------------------
# 3. Names, strictness and refusals.
# ---------------------------------------------------------------------------


def test_name_rules() -> None:
    cfg = _tiny_cfg()  # 4 layers: checkpoint layers.0 .. layers.3
    table = {
        "language_model.model.embed_tokens.weight": "embed_tokens.weight",
        "language_model.model.layers.3.self_attn.kv_b_proj.weight": "layers.3.self_attn.kv_b_proj.weight",
        "language_model.layers.3.self_attn.kv_b_proj.weight": "layers.3.self_attn.kv_b_proj.weight",
        "language_model.model.output_attn_res_proj.weight": "output_attn_res_proj.weight",
        "language_model.lm_head.weight": "lm_head.weight",
        "language_model.model.layers.1.block_sparse_moe.experts.3.w2.weight_packed": (
            "layers.1.block_sparse_moe.experts.3.w2.weight"
        ),
        "language_model.model.layers.1.block_sparse_moe.experts.3.w2.weight_scale": (
            "layers.1.block_sparse_moe.experts.3.w2.weight"
        ),
        "language_model.model.layers.4.self_attn.q_proj.weight": None,  # MTP: index ≥ num_layers
        "language_model.layers.12.enorm.weight": None,
        "vision_tower.encoder.blocks.0.wqkv.weight": None,
        "mm_projector.proj.0.weight": None,
        "some.other.tensor": "some.other.tensor",  # unknown: kept, so the load reports it
    }
    for hf, k3 in table.items():
        assert hf_to_k3_name(hf, cfg) == k3, hf


def test_strict_reports_missing_and_unexpected_and_skips_vision_and_mtp() -> None:
    cfg = _tiny_cfg()
    source, packed = _source_model(cfg)
    extra = torch.zeros(2, 2, dtype=torch.bfloat16)
    skipped = [
        "vision_tower.encoder.blocks.0.wqkv.weight",
        "mm_projector.proj.0.weight",
        f"language_model.model.layers.{cfg.num_layers}.self_attn.q_proj.weight",
    ]
    exported = _export(source, packed) + [(n, extra) for n in skipped]

    report = load_hf_checkpoint(K3Model(cfg), exported)  # skips are not errors in strict mode
    assert report.missing == report.unexpected == () and report.skipped == tuple(skipped)

    broken = [(n, t) for n, t in exported if n != "language_model.model.norm.weight"]
    broken.append(("language_model.model.layers.0.self_attn.bogus.weight", extra))
    with pytest.raises(
        RuntimeError, match=r'(?s)Missing key.*"norm\.weight".*Unexpected key.*bogus'
    ):
        load_hf_checkpoint(K3Model(cfg), broken)
    report = load_hf_checkpoint(K3Model(cfg), broken, strict=False)
    assert report.missing == ("norm.weight",)
    assert report.unexpected == ("layers.0.self_attn.bogus.weight",)
    assert report.skipped == tuple(skipped)


def test_refusals_name_the_tensor() -> None:
    cfg = _tiny_cfg()
    source, packed = _source_model(cfg)
    exported = _export(source, packed)
    expert = "language_model.model.layers.1.block_sparse_moe.experts.0.w1"
    by_name = dict(exported)

    lone = [(n, t) for n, t in exported if n != expert + ".weight_scale"]
    with pytest.raises(ValueError, match=r"without their partner.*experts\.0\.w1\.weight"):
        hf_to_k3_state_dict(lone, cfg)

    norm = "language_model.model.norm.weight"
    both_spellings = [*exported, ("language_model.norm.weight", by_name[norm])]
    with pytest.raises(ValueError, match=r"second checkpoint tensor for norm\.weight"):
        hf_to_k3_state_dict(both_spellings, cfg)

    packed_half = (expert + ".weight_packed", by_name[expert + ".weight_packed"])
    twice = [packed_half, *exported]  # the duplicate arrives before the pair completes
    with pytest.raises(ValueError, match="same MXFP4 half appears twice"):
        hf_to_k3_state_dict(twice, cfg)
    with pytest.raises(ValueError, match="without their partner"):  # ... or after it
        hf_to_k3_state_dict([*exported, packed_half], cfg)

    fp16 = [(n, t.half() if n == norm else t) for n, t in exported]
    with pytest.raises(ValueError, match=r"model\.norm\.weight: torch\.float16"):
        hf_to_k3_state_dict(fp16, cfg)
    with pytest.raises(ValueError, match=r"embed_tokens\.weight: BF16 does not convert"):
        hf_to_k3_state_dict(exported, cfg, dtype=torch.float16)  # fp16 lacks bf16's range

    bad_scale = [(n, t[:, :0] if n == expert + ".weight_scale" else t) for n, t in exported]
    with pytest.raises(ValueError, match=r"experts\.0\.w1\.weight \(MXFP4\): .*needs"):
        hf_to_k3_state_dict(bad_scale, cfg)

    wide = [(n, torch.cat([t, t[:1]]) if n == norm else t) for n, t in exported]
    with pytest.raises(RuntimeError, match=r"size mismatch for norm\.weight"):
        load_hf_checkpoint(K3Model(cfg), wide, strict=False)


def test_load_decodes_in_the_model_dtype() -> None:
    """The MXFP4 exactness check runs against the model's dtype: with scale code 254 and code 7,
    group 0 of one expert row holds 6 · 2^127. An fp64 model receives it exactly; an fp32 model
    refuses it instead of storing inf."""
    cfg = _tiny_cfg()
    source, packed = _source_model(cfg)
    name = "layers.1.block_sparse_moe.experts.0.w1.weight"
    weight_packed, weight_scale = (t.clone() for t in packed[name])
    weight_packed[0, 0], weight_scale[0, 0] = 0x77, 254
    exported = _export(source, {**packed, name: (weight_packed, weight_scale)})

    model = K3Model(cfg).double()
    load_hf_checkpoint(model, exported)
    assert model.get_parameter(name)[0, :2].tolist() == [6 * 2.0**127] * 2
    with pytest.raises(ValueError, match=r"experts\.0\.w1\.weight \(MXFP4\): .*not exact"):
        load_hf_checkpoint(K3Model(cfg), exported)


# ---------------------------------------------------------------------------
# 4. A_log storage tail.
# ---------------------------------------------------------------------------


def test_a_log_padding_passes_through_and_a_nonzero_tail_is_refused() -> None:
    cfg = _tiny_cfg()
    source, packed = _source_model(cfg)
    exported = _export(source, packed)
    a_log = "language_model.model.layers.0.self_attn.A_log"
    stored = dict(exported)[a_log]
    assert stored.shape == (cfg.kda.a_log_size,) and cfg.kda.a_log_size > cfg.kda.num_heads

    state = hf_to_k3_state_dict(exported, cfg, dtype=torch.float64)
    assert torch.equal(state["layers.0.self_attn.A_log"], stored)  # unchanged, still F32
    assert state["layers.0.self_attn.A_log"].dtype == torch.float32
    assert state["norm.weight"].dtype == torch.float64  # BF16 → dtype
    assert state["layers.1.block_sparse_moe.experts.0.w1.weight"].dtype == torch.float64

    tail = stored.clone()
    tail[-1] = 1e-3
    with pytest.raises(RuntimeError, match="nonzero padding"):
        load_hf_checkpoint(K3Model(cfg), [(n, tail if n == a_log else t) for n, t in exported])


# ---------------------------------------------------------------------------
# 5. Nothing rounds: the F32 families in a bf16 model.
# ---------------------------------------------------------------------------


def test_a_bf16_model_must_keep_the_f32_families_in_fp32() -> None:
    """The source's F32-family values are generic fp32 (off init by 0.1 · randn), so a bf16
    parameter could not hold them."""
    cfg = _tiny_cfg()
    source, packed = _source_model(cfg)
    exported = _export(source, packed)
    families = {n for n, _ in source.named_parameters() if n.endswith(_F32_SUFFIXES)}
    assert len(families) == 6 * cfg.num_kda_layers + cfg.num_moe_layers

    blanket = K3Model(cfg).to(torch.bfloat16)
    before = {n: t.clone() for n, t in blanket.state_dict().items()}
    refused = rf"^{len(families)} checkpoint tensors would round into narrower parameters"
    with pytest.raises(ValueError, match=refused) as err:
        load_hf_checkpoint(blanket, exported)
    for name in families:
        assert f"{name} (torch.float32 into torch.bfloat16)" in str(err.value), name
    for name, t in blanket.state_dict().items():  # refused before anything was written
        assert torch.equal(t, before[name]), name

    served = K3Model(cfg)
    with torch.no_grad():
        for name, p in served.named_parameters():
            if name not in families:
                p.data = p.data.to(torch.bfloat16)
    load_hf_checkpoint(served, exported)
    for name, p in source.named_parameters():  # the BF16 values are exact in bf16 by construction
        got = served.get_parameter(name)
        assert got.dtype == (torch.float32 if name in families else torch.bfloat16), name
        assert torch.equal(got.float(), p), name
    with torch.no_grad():
        logits = served(_ids(cfg))
    assert logits.dtype == torch.bfloat16 and bool(torch.isfinite(logits).all())


# ---------------------------------------------------------------------------
# 6. The real checkpoint (network).
# ---------------------------------------------------------------------------

# main's x-repo-commit when this gate was written (2026-09-22); pinned so the gate is reproducible.
_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
_BASE = f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{_REVISION}"
_CACHE = Path.home() / ".cache" / "scratch_llm" / "kimi_k3" / _REVISION[:12]
_SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    """GET (bytes [start, end] inclusive when given); urllib keeps the Range across HF's 302."""
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r:
        return r.read()


def _cached(path: Path, fetch: Callable[[], bytes]) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(fetch())
        part.replace(path)
    return path.read_bytes()


def _header(shard: str) -> tuple[dict[str, dict], int]:
    """safetensors: 8-byte little-endian header length N, N bytes of JSON, then the data.
    Returns (header without __metadata__, byte offset of the data)."""
    url = f"{_BASE}/{shard}"

    def fetch() -> bytes:
        (n,) = struct.unpack("<Q", _get(url, 0, 7))
        return _get(url, 8, 8 + n - 1)

    raw = _cached(_CACHE / f"{shard}.header.json", fetch)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header, 8 + len(raw)


def _tensor(weight_map: dict[str, str], name: str) -> Tensor:
    shard = weight_map[name]
    header, base = _header(shard)
    meta = header[name]
    start, end = meta["data_offsets"]
    raw = _cached(
        _CACHE / "tensors" / name, lambda: _get(f"{_BASE}/{shard}", base + start, base + end - 1)
    )
    dtype = _SAFETENSORS_DTYPES[meta["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=dtype).view(meta["shape"])  # little-endian host


@pytest.mark.network
@pytest.mark.slow
def test_real_checkpoint_names_and_shapes() -> None:
    if not os.environ.get("SCRATCH_LLM_NETWORK_TESTS"):
        pytest.skip("network test: set SCRATCH_LLM_NETWORK_TESTS=1 to run")
    cfg = k3_full()
    with torch.device("meta"):
        model = K3Model(cfg)
    params = {name: tuple(p.shape) for name, p in model.state_dict().items()}

    index = _cached(
        _CACHE / "model.safetensors.index.json",
        lambda: _get(f"{_BASE}/model.safetensors.index.json"),
    )
    weight_map: dict[str, str] = json.loads(index)["weight_map"]
    mapped = {hf: hf_to_k3_name(hf, cfg) for hf in weight_map}
    skipped = {hf for hf, k3 in mapped.items() if k3 is None}
    assert skipped and all(hf.startswith(("vision_tower.", "mm_projector.")) for hf in skipped)
    assert {k3 for k3 in mapped.values() if k3 is not None} == set(params)

    heads = (
        "language_model.model.layers.{}.input_layernorm.weight",
        "language_model.lm_head.weight",
    )
    shards = {weight_map[heads[0].format(i)] for i in (0, 1, 3)} | {weight_map[heads[1]]}
    checked: set[str] = set()
    for shard in sorted(shards):
        header, _ = _header(shard)
        for hf, meta in header.items():
            k3, shape = mapped[hf], tuple(meta["shape"])
            assert k3 is not None, hf
            if hf.endswith(".weight_packed"):
                assert meta["dtype"] == "U8", hf
                shape = (shape[0], 2 * shape[1])
            elif hf.endswith(".weight_scale"):
                assert meta["dtype"] == "U8", hf
                shape = (shape[0], MXFP4_GROUP * shape[1])
            elif hf.endswith("self_attn.A_log"):  # stored [a_log_size]; KDALayer keeps num_heads
                assert shape == (cfg.kda.a_log_size,), hf
                shape = (cfg.kda.num_heads,)
            else:
                assert meta["dtype"] in ("BF16", "F32"), hf
            assert shape == params[k3], hf
            checked.add(k3)
    assert any(".mlp." in n for n in checked) and any(".kv_b_proj." in n for n in checked)
    assert any(".experts." in n for n in checked) and any(".A_log" in n for n in checked)

    a_log = _tensor(weight_map, "language_model.model.layers.0.self_attn.A_log")
    layer = KDALayer(cfg.kda, hidden_size=8)  # the real head layout; a tiny hidden size
    layer.load_state_dict({"A_log": a_log}, strict=False)  # refuses a nonzero tail
    assert torch.equal(layer.A_log.detach(), a_log[: cfg.kda.num_heads])

    expert = "language_model.model.layers.1.block_sparse_moe.experts.0.w2"
    w = dequantize_mxfp4(
        _tensor(weight_map, expert + ".weight_packed"),
        _tensor(weight_map, expert + ".weight_scale"),
        dtype=torch.bfloat16,
    )
    assert tuple(w.shape) == params["layers.1.block_sparse_moe.experts.0.w2.weight"]
