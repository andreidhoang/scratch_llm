"""S1/S-R2 — FP8 weights + E4M3 per-head KV, gated in two tiers.

  CPU   the arithmetic, which is all of the rung that is decidable without silicon: the scale
        derivation, E4M3's dynamic range and its subnormal floor, the saturation policy, whether
        the per-head scales are per-head, idempotence of the stored map, and the quality gate's
        bookkeeping. Milliseconds.
  gpu   ``torch._scaled_mm`` against the float64 oracle at the same codes and the same scales.
        Needs an FP8-capable card; skips with the box command otherwise.

The CPU tier carries the weight here, because every way this rung can be wrong produces a working
system. A per-tensor scale where a per-head one was intended still decodes — it just quietly zeroes
the quiet heads. A cast without a clamp still runs — until one token overflows the calibration and
a NaN eats a softmax row. A scale recomputed per append still "works" — it just stops the cache
from being 1 byte per element and makes the byte-halving a lie. None of those faults, none of them
are slow, and all of them are decidable on a laptop.

Spec: experiments/S1/S-R2/spec.md   ·   Map: experiments/S1/S-R2/map.md
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from scratch_llm.quant.fp8_kv_per_head import (
    PerHeadFp8KVCache,
    bf16_reference_bytes,
    dequantize_per_head,
    per_head_scale,
    quantize_per_head,
    saturated_count,
)
from scratch_llm.quant.fp8_weights import (
    E4M3_MAX,
    E4M3_MIN_NORMAL,
    E4M3_MIN_SUBNORMAL,
    E4M3_NAN_ABOVE,
    E4M3_UNIT_ROUNDOFF,
    FP8_DTYPE,
    dequantize,
    fp8_linear,
    fp8_linear_reference,
    quantize_per_channel,
    quantize_per_token,
)
from scratch_llm.quant.s_r2_gate import (
    GateThresholds,
    QualityEvidence,
    gate_thresholds,
    verdict,
)

BOX = (
    "no CUDA device — measure on the rung's box: infra/rent.sh sync-up <user@host> && "
    "infra/rent.sh ssh <user@host> && bash ladders/experiments/S1/S-R2/run.sh"
)

_REPO = Path(__file__).resolve().parents[2]


def _evidence(
    *, ppl_ratio: float, bf16_flags: list[bool], fp8_flags: list[bool]
) -> QualityEvidence:
    """Evidence whose FP8 perplexity is ``ppl_ratio`` times the bf16 one, by construction.

    ppl = exp(nll/n), so multiplying the perplexity by r means adding ``n·ln r`` nats. Building the
    fixture from that identity rather than from a stored ppl keeps the test honest about which
    quantity the gate actually reads.
    """
    n = 4096
    base = 2.0 * n
    return QualityEvidence.from_paired(
        corpus="fixture",
        n_ppl_tokens=n,
        bf16_nll_nats=base,
        fp8_nll_nats=base + n * math.log(ppl_ratio),
        gsm8k_slice="fixture[:8]",
        bf16_correct=bf16_flags,
        fp8_correct=fp8_flags,
    )


# =============================================================================================
# THE HOLE — exactly one guard (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("S1/S-R2", "src/scratch_llm/quant/s_r2_gate.py")
def test_the_quality_gate_thresholds_decide_the_rung() -> None:
    """Fails while ``gate_thresholds()`` is unwritten; passes when Huy has chosen both numbers.

    The two thresholds are the only thing S-R2 produces that an agent must not produce. Everything
    else in this rung — the quantizers, the cache, the oracle, the runner, the floor — exists to
    put one question in front of a person: how much perplexity, and how much GSM8K, is the speedup
    allowed to cost. This test asserts only value-free properties of whatever he chooses, because
    asserting a value here would *be* choosing it.
    """
    identical = _evidence(ppl_ratio=1.0, bf16_flags=[True] * 8, fp8_flags=[True] * 8)
    thresholds = gate_thresholds(identical)

    if thresholds is None:
        # LADDERS_STUB_HOLES=1: the placeholder exists so the plumbing can run, and it judges
        # NOTHING. A stub that returned PASS would ship a speedup against no quality bound at all.
        assert verdict(identical, thresholds) == "UNJUDGED"
        return

    assert thresholds.max_ppl_rel_increase >= 0.0, (
        "a negative Δppl bound demands FP8 beat bf16 on perplexity; that is a different experiment"
    )
    assert thresholds.max_accuracy_drop >= 0.0

    assert verdict(identical, thresholds) == "PASS", (
        "an FP8 arm identical to the bf16 arm must pass any gate — otherwise the threshold is "
        "tighter than 'no change at all' and nothing can ever pass it"
    )

    broken = _evidence(ppl_ratio=10.0, bf16_flags=[True] * 8, fp8_flags=[False] * 8)
    assert verdict(broken, thresholds) == "FAIL", (
        "a 10x perplexity blow-up with zero GSM8K accuracy must fail; a gate that passes it is a "
        "rubber stamp"
    )


# =============================================================================================
# CPU — the weight quantizer. Per output channel, and within E4M3's own quantum.
# =============================================================================================


def test_per_channel_scale_round_trips_within_the_formats_own_quantum() -> None:
    """The round trip's error is bounded by E4M3's rounding, not by a constant anyone picked.

    Two regimes, because E4M3 has two. Above the smallest normal (2**-6) the spacing is
    proportional to the value, so the bound is *relative* and equals the unit roundoff 2**-4 — one
    half of ``eps = 2**-3``. In the subnormal range the spacing is fixed at 2**-9, so the bound is
    *absolute* and equals half of that. A single "rtol" would be wrong in both directions: too
    loose for normals, unsatisfiable for subnormals.
    """
    torch.manual_seed(0)
    w = torch.randn(32, 128) * torch.logspace(-3, 3, 32).unsqueeze(1)  # 6 decades across channels
    codes, scale = quantize_per_channel(w)

    assert codes.dtype is FP8_DTYPE and codes.element_size() == 1
    assert tuple(scale.shape) == (32, 1), "one scale per OUTPUT channel, reduced over K"
    assert not torch.isnan(codes.float()).any(), "the clamp exists so this can never be NaN"

    v = w.to(torch.float32) / scale  # what the cast actually rounds
    q = codes.float()
    err = (v - q).abs()
    normal = v.abs() >= E4M3_MIN_NORMAL
    assert torch.all(err[normal] <= E4M3_UNIT_ROUNDOFF * v[normal].abs()), (
        "a normal E4M3 value cannot be more than one unit roundoff (2**-4) off in relative terms"
    )
    assert torch.all(err[~normal] <= E4M3_MIN_SUBNORMAL / 2), (
        "in the subnormal range the grid is uniform with step 2**-9, so the error is absolute"
    )

    # The scale is amax/448 exactly so that each channel's largest element lands on the top of the
    # range: if it did not, the channel would be wasting code space it paid a byte per element for.
    assert torch.all(q.abs().amax(dim=1) == E4M3_MAX)


def test_per_channel_beats_per_tensor_on_a_weight_with_channel_spread() -> None:
    """A per-tensor scale is hostage to the loudest channel; this is that hostage, measured.

    The quiet channels are six decades below the loud ones, so a single global scale pushes them
    under E4M3's flush-to-zero floor and reconstructs them as literally nothing. This is a test the
    per-channel implementation passes and a per-tensor one cannot.
    """
    torch.manual_seed(1)
    w = torch.randn(8, 64) * torch.logspace(-3, 3, 8).unsqueeze(1)
    per_channel = dequantize(*quantize_per_channel(w))

    global_scale = w.abs().amax() / E4M3_MAX
    per_tensor = (w / global_scale).clamp(-E4M3_MAX, E4M3_MAX).to(FP8_DTYPE).float() * global_scale

    quiet = w[0]
    assert torch.all(per_tensor[0] == 0.0), (
        "the premise of the test: one global scale annihilates the quiet channel entirely"
    )
    rel = (per_channel[0] - quiet).abs() / quiet.abs()
    assert torch.all(rel <= E4M3_UNIT_ROUNDOFF), "per-channel keeps the quiet channel to 2**-4"


def test_per_token_activation_scale_reduces_over_the_contraction_axis() -> None:
    """``(M, K) -> (M, 1)``: one scale per token, constant along K so the GEMM can hoist it."""
    torch.manual_seed(2)
    x = torch.randn(6, 32) * torch.tensor([1e-2, 1.0, 1e2, 1e-2, 1.0, 1e2]).unsqueeze(1)
    codes, scale = quantize_per_token(x)
    assert tuple(scale.shape) == (6, 1)
    assert torch.all(codes.float().abs().amax(dim=1) == E4M3_MAX)
    # Rows are independent: scaling one row must not move any other row's codes.
    x2 = x.clone()
    x2[0] *= 1000.0
    codes2, _ = quantize_per_token(x2)
    assert torch.equal(codes.float()[1:], codes2.float()[1:])


def test_the_raw_cast_would_produce_nan_which_is_why_the_quantizer_clamps() -> None:
    """The reason :func:`saturating_cast_e4m3` exists, made executable.

    ``float8_e4m3fn`` has no infinity: round-to-nearest carries everything in ``[448, 464]`` down
    onto 448 (464 is the tie, and 480 is the NaN encoding), and everything strictly above 464
    becomes NaN. A KV cache that let that happen would turn one over-range token into a whole row
    of NaN after the softmax.
    """
    raw = torch.tensor([E4M3_MAX, E4M3_NAN_ABOVE, E4M3_NAN_ABOVE + 1.0, 1e5]).to(FP8_DTYPE).float()
    assert raw[0].item() == E4M3_MAX
    assert raw[1].item() == E4M3_MAX, "the round-to-nearest band reaches 464"
    assert math.isnan(raw[2].item()) and math.isnan(raw[3].item())

    clamped = dequantize(*quantize_per_channel(torch.tensor([[1e5, 1.0, 0.0, -1e5]])))
    assert not torch.isnan(clamped).any()


# =============================================================================================
# CPU — the KV cache. Per HEAD, and a test that one global scale could not pass.
# =============================================================================================


def _kv(batch: int = 2, heads: int = 4, tokens: int = 8, dim: int = 16) -> torch.Tensor:
    """A KV block whose heads span six decades — the spread per-head scales exist for."""
    torch.manual_seed(3)
    x = torch.randn(batch, heads, tokens, dim)
    x = x * torch.logspace(-3, 3, heads).view(1, heads, 1, 1)
    x[1] *= 2.0  # the batch rows differ, so a per-(b, h) scale would be visible
    return x


def test_per_head_scales_are_per_head_and_a_global_scale_cannot_pass_this() -> None:
    """H distinct scales, and the quiet head survives — which under one global scale it does not.

    The discriminator is not "per-head is a bit better". With six decades between the loudest and
    quietest head, one global scale maps every element of the quiet head below E4M3's
    flush-to-zero floor (2**-10 of the scale), so its reconstruction is exactly the zero tensor.
    Per-head keeps it to one unit roundoff. A test that merely compared MSEs would also pass for a
    per-tensor implementation with a lucky distribution; this one cannot.
    """
    x = _kv()
    scale = per_head_scale(x)
    assert tuple(scale.shape) == (1, x.shape[1], 1, 1)
    assert len(set(scale.flatten().tolist())) == x.shape[1], "H heads, H distinct scales"

    per_head = dequantize_per_head(quantize_per_head(x, scale), scale)
    global_scale = x.abs().amax() / E4M3_MAX
    per_tensor = (x / global_scale).clamp(-E4M3_MAX, E4M3_MAX).to(FP8_DTYPE).float() * global_scale

    quiet = x[:, 0]
    assert torch.all(per_tensor[:, 0] == 0.0), (
        "premise: one global scale flushes the quiet head to zero (its elements land under 2**-10 "
        "of the shared scale, E4M3's smallest subnormal / 2)"
    )
    rel = (per_head[:, 0] - quiet).abs() / quiet.abs()
    assert torch.all(rel <= E4M3_UNIT_ROUNDOFF)


def test_the_per_head_scale_reduces_over_batch_not_per_batch_row() -> None:
    """A scale keyed on ``(b, h)`` cannot be shared by a paged block; this pins it to ``(1, H, 1, 1)``."""
    x = _kv()
    scale = per_head_scale(x)

    flipped = per_head_scale(x.flip(0))
    assert torch.equal(scale, flipped), "the scale is a property of the head, not of the row order"

    row0 = per_head_scale(x[:1])
    assert not torch.equal(scale, row0), (
        "the batch rows have different magnitudes, so a scale that ignored the batch axis would "
        "equal the first row's — the reduction must cover B"
    )
    with pytest.raises(ValueError, match="not per-head"):
        quantize_per_head(x, scale.expand(x.shape[0], -1, -1, -1))


def test_a_value_past_the_calibrated_range_saturates_and_is_counted() -> None:
    """Static calibration's failure mode: saturation to ``448 * scale``, never NaN, always counted."""
    prefill = _kv()
    cache = PerHeadFp8KVCache(n_layers=1)
    cache.append(0, prefill, prefill)
    cache.advance(prefill.shape[2])
    k_scale, _ = cache.scales(0)
    assert k_scale is not None
    assert cache.k_saturated == 0, "the calibration block cannot overflow its own amax"

    over = prefill[:, :, :1].clone()
    over[:, 0] = over[:, 0].abs().clamp_min(1.0) * 1e4  # head 0 blows past its calibrated range
    n_over = saturated_count(over, k_scale)
    assert n_over > 0

    cache.append(0, over, over)
    cache.advance(1)
    assert cache.k_saturated == n_over and cache.v_saturated == n_over
    assert cache.saturated_fraction() > 0.0

    got = cache.get(0)
    assert got is not None
    k_out, _ = got
    assert not torch.isnan(k_out).any(), (
        "saturation, not NaN — that is the whole point of the clamp"
    )
    tail = k_out[:, 0, -1]
    ceiling = (E4M3_MAX * k_scale[0, 0, 0, 0]).item()
    assert torch.allclose(tail.abs(), torch.full_like(tail, ceiling)), (
        "an over-range element must land exactly on the top of the head's range"
    )
    # And the raw cast, on the same numbers, would have produced NaN instead.
    raw = (over[:, 0] / k_scale[0, 0]).to(FP8_DTYPE).float()
    assert torch.isnan(raw).any()


def test_values_below_the_subnormal_floor_dequantize_to_exactly_zero() -> None:
    """E4M3's dynamic range has a bottom as well as a top, and it is 2**-10 of the scale.

    Below half the smallest subnormal, round-to-nearest-even carries a value to zero. Expressed in
    the head's own units that is a dynamic range of 448 / 2**-10 ≈ 4.6e5 between the head's amax
    and the smallest element it can still represent — the number that decides whether per-head is
    fine enough for a head with an outlier channel.
    """
    x = torch.zeros(1, 1, 4, 2)
    x[0, 0, 0, 0] = 1.0  # sets the head's amax, hence the scale
    scale = per_head_scale(x)
    unit = scale[0, 0, 0, 0].item()
    x[0, 0, 1, 0] = unit * (E4M3_MIN_SUBNORMAL / 2) * 0.9  # under the flush-to-zero floor
    x[0, 0, 2, 0] = unit * E4M3_MIN_SUBNORMAL  # the smallest representable step

    out = dequantize_per_head(quantize_per_head(x, scale), scale)
    assert out[0, 0, 1, 0].item() == 0.0
    assert out[0, 0, 2, 0].item() == pytest.approx(unit * E4M3_MIN_SUBNORMAL, rel=1e-6)
    assert (E4M3_MAX / (E4M3_MIN_SUBNORMAL / 2)) > 4e5  # the range the docstring quotes


def test_quantize_dequantize_is_idempotent_at_a_fixed_scale() -> None:
    """Re-quantizing already-quantized values at the same scale is the identity, bit for bit.

    A static cache re-reads its own bytes for the whole life of a request, and any engine that
    re-quantizes on a copy, a page migration, or a prefix share does exactly this round trip. If it
    were not idempotent the cache would drift a little every time it moved. The scale is *held
    fixed* on purpose: recomputing the amax from the dequantized tensor can differ by an fp32 ulp,
    and that would test the amax, not the map.
    """
    x = _kv()
    scale = per_head_scale(x)
    once = quantize_per_head(x, scale)
    twice = quantize_per_head(dequantize_per_head(once, scale), scale)
    assert torch.equal(once.view(torch.uint8), twice.view(torch.uint8))


def test_the_cache_stores_one_byte_per_element_and_says_so() -> None:
    """The byte halving is read off the storage, not asserted in prose."""
    x = _kv()
    cache = PerHeadFp8KVCache(n_layers=2)
    for layer in (0, 1):
        cache.append(layer, x, x)
    cache.advance(x.shape[2])

    stored = cache.codes(0)
    assert stored is not None and stored[0].dtype is FP8_DTYPE
    assert stored[0].element_size() == 1

    n_elements = 2 * 2 * x.numel()  # K and V, two layers
    scale_bytes = 2 * 2 * x.shape[1] * 4  # one fp32 per head, per tensor, per layer — once
    assert cache.kv_bytes == n_elements + scale_bytes
    assert cache.kv_bytes < bf16_reference_bytes(n_elements)


def test_the_cache_duck_types_the_decode_contract() -> None:
    """``length``/``append``/``get``/``advance`` — the interface ``model.py`` calls, unchanged."""
    cache = PerHeadFp8KVCache(n_layers=1)
    assert cache.length == 0 and cache.get(0) is None
    x = _kv()
    cache.append(0, x, x)
    cache.advance(x.shape[2])
    assert cache.length == x.shape[2]
    got = cache.get(0)
    assert got is not None and got[0].shape == x.shape


def test_installed_scales_take_precedence_over_calibration() -> None:
    """The checkpoint-calibrated path: scales installed before the first append are the ones used."""
    x = _kv()
    cache = PerHeadFp8KVCache(n_layers=1)
    scale = torch.full((1, x.shape[1], 1, 1), 1.0)
    cache.set_scales(0, scale, scale)
    cache.append(0, x, x)
    k_scale, v_scale = cache.scales(0)
    assert k_scale is not None and v_scale is not None
    assert torch.equal(k_scale, scale) and torch.equal(v_scale, scale)
    assert cache.k_saturated > 0, (
        "scale 1.0 saturates the loud heads — the vLLM default warns for a reason"
    )


# =============================================================================================
# CPU — the reference oracle itself, checked against a definition anyone can read.
# =============================================================================================


def test_the_oracle_agrees_with_the_definition_of_a_matmul() -> None:
    """``fp8_linear_reference`` is the oracle, so it is checked against the sum, not against a peer."""
    torch.manual_seed(4)
    x = torch.randn(3, 4)
    w = torch.randn(5, 4)
    codes, scale = quantize_per_channel(w)
    bias = torch.randn(5)
    got = fp8_linear_reference(x, codes, scale, bias, out_dtype=torch.float64)

    xq, xs = quantize_per_token(x)
    xd = (xq.to(torch.float64) * xs.to(torch.float64)).tolist()
    wd = (codes.to(torch.float64) * scale.to(torch.float64)).tolist()
    for m in range(3):
        for n in range(5):
            expect = sum(xd[m][k] * wd[n][k] for k in range(4)) + float(bias[n])
            assert got[m][n].item() == pytest.approx(expect, rel=1e-12, abs=1e-12)


def test_fp8_linear_refuses_the_cpu_and_names_the_box() -> None:
    """No silent CPU fallback for the measured path: ``_scaled_mm`` has no CPU kernel."""
    w = torch.randn(16, 32)
    codes, scale = quantize_per_channel(w)
    with pytest.raises(RuntimeError, match="rent.sh"):
        fp8_linear(torch.randn(4, 32), codes, scale)


# =============================================================================================
# CPU — the gate's bookkeeping. Not the thresholds (those are the hole), the arithmetic under them.
# =============================================================================================


def test_the_gsm8k_comparison_is_paired_and_reports_discordant_counts() -> None:
    """Two arms with the *same* accuracy can still have changed every answer; ``(b, c)`` sees it."""
    bf16 = [True, True, False, False, True, True, False, False]
    fp8 = [True, False, True, False, True, False, True, False]
    ev = _evidence(ppl_ratio=1.0, bf16_flags=bf16, fp8_flags=fp8)
    assert ev.bf16_accuracy == ev.fp8_accuracy
    assert ev.accuracy_drop == 0.0
    assert ev.discordant == (2, 2), (
        "identical accuracies, four items changed — an unpaired binomial on 200 samples cannot "
        "see this, which is why the gate is handed b and c"
    )


def test_perplexity_is_read_as_a_ratio_of_the_streams_own_perplexity() -> None:
    ev = _evidence(ppl_ratio=1.5, bf16_flags=[True], fp8_flags=[True])
    assert ev.fp8_ppl / ev.bf16_ppl == pytest.approx(1.5)
    assert ev.ppl_rel_increase == pytest.approx(0.5)


def test_evidence_refuses_an_unpaired_comparison() -> None:
    with pytest.raises(ValueError, match="same items"):
        QualityEvidence.from_paired(
            corpus="c",
            n_ppl_tokens=8,
            bf16_nll_nats=1.0,
            fp8_nll_nats=1.0,
            gsm8k_slice="s",
            bf16_correct=[True, True],
            fp8_correct=[True],
        )


def test_an_unwritten_threshold_is_unjudged_and_never_a_pass() -> None:
    """The seal, as a test: no thresholds means no verdict, and no verdict means no number."""
    ev = _evidence(ppl_ratio=1.0, bf16_flags=[True], fp8_flags=[True])
    assert verdict(ev, None) == "UNJUDGED"


def test_a_negative_threshold_is_rejected_as_a_different_experiment() -> None:
    with pytest.raises(ValueError, match="not"):
        GateThresholds(max_ppl_rel_increase=-1.0, max_accuracy_drop=0.0)


# =============================================================================================
# CPU — the runner's contract: the number is gated, and the floor is S-R1's engine or nothing.
# =============================================================================================


def _runner():
    path = _REPO / "bench" / "s1_fp8_serving.py"
    spec = importlib.util.spec_from_file_location("_s1_fp8_serving", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_runner_will_not_invent_the_bf16_engine() -> None:
    """The floor is S-R1's engine at matched model and shape; a stand-in would be a fake floor."""
    runner = _runner()
    with pytest.raises(RuntimeError, match="S-R1"):
        runner.resolve_engine_factory("scratch_llm.serving.does_not_exist:build_engine")
    assert runner.DEFAULT_ENGINE.startswith("scratch_llm.serving.")


def test_the_runner_declares_the_shape_both_arms_are_compared_at() -> None:
    runner = _runner()
    shape = runner.SHAPES["qwen3_8b_b32"]
    assert shape.batch == 32 and "Qwen3-8B" in shape.model
    assert runner.main(["--rung", "S-R2", "--dry-run"]) == 0


# =============================================================================================
# GPU — torch._scaled_mm against the float64 oracle, at the same codes and the same scales.
# =============================================================================================


@pytest.mark.gpu
def test_scaled_mm_matches_the_float64_oracle() -> None:
    """The only differences left are the GEMM's accumulation order and the output cast.

    The bound is therefore *derived*, not chosen: half an ulp of the output dtype on the result,
    plus a sqrt(K) random-walk of fp32 accumulation roundoff on the magnitude sum. Nothing here is
    a tolerance anyone had to defend — the rung's one defended constant is the quality gate's.
    """
    if not torch.cuda.is_available():
        pytest.skip(BOX)
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("E4M3 _scaled_mm wants sm89+; route this rung to H100 (sm90) — " + BOX)

    torch.manual_seed(5)
    m, k, n = 128, 512, 256
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    codes, scale = quantize_per_channel(w)

    got = fp8_linear(x, codes, scale, out_dtype=torch.bfloat16).to(torch.float64)
    ref = fp8_linear_reference(x, codes, scale, out_dtype=torch.float64)

    xq, xs = quantize_per_token(x)
    magnitude = (xq.to(torch.float64) * xs.to(torch.float64)).abs() @ (
        codes.to(torch.float64) * scale.to(torch.float64)
    ).abs().t()
    bound = ref.abs() * (torch.finfo(torch.bfloat16).eps / 2) + magnitude * (
        math.sqrt(k) * torch.finfo(torch.float32).eps
    )
    assert torch.all((got - ref).abs() <= bound)


@pytest.mark.gpu
def test_the_fp8_cache_really_is_half_the_bytes_on_the_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip(BOX)
    x = _kv(batch=4, heads=8, tokens=64, dim=128).cuda()
    cache = PerHeadFp8KVCache(n_layers=1)
    cache.append(0, x, x)
    stored = cache.codes(0)
    assert stored is not None
    device_bytes = stored[0].numel() * stored[0].element_size() * 2
    assert device_bytes * 2 == bf16_reference_bytes(2 * x.numel())
