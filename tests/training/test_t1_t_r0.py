"""T1/T-R0 — the floor's arithmetic, as known-answer assertions on a laptop.

The measurement needs 8 H100s. The *accounting* needs none, and it is where a training floor
actually goes wrong: a tok/s number is hard to fake, but an MFU number quoted under an unstated
convention is wrong by 30% and looks fine. So everything except the stopwatch is pinned here.

  1. 6ND against a hand-derived parameter count for Llama-3 1B — every weight matrix multiplied
     out by hand below, so a change in torchtitan's flavor definition breaks a test, not a claim.
  2. The attention term: linear in S per token (quadratic per sequence), and the causal factor
     torchtitan deliberately does not apply.
  3. The AC multiplier is exactly what the selected policy implies — no fitted constants.
  4. MFU is dimensionless and returns 1.0 at peak.
  5. Our accounting agrees with torchtitan's own formula, and the harness refuses the run if the
     log says otherwise.

Spec: experiments/T1/T-R0/spec.md   ·   Map: experiments/T1/T-R0/map.md
"""

from __future__ import annotations

import re
import shutil

import pytest

from scratch_llm._workspace import workspace_root
from scratch_llm.training.flops import (
    AC_FULL,
    AC_NONE,
    AC_POLICIES,
    AC_SELECTIVE_OP,
    ATTENTION_FLOPS_PER_ELEMENT,
    FLOPS_PER_PARAM,
    LLAMA3_1B,
    TORCHTITAN_PEAK_BF16_DENSE,
    DecoderShape,
    StepReport,
    flop_breakdown,
    recomputed_linears,
    step_report,
    utilization,
)
from scratch_llm.training.titan_floor import (
    GLOBAL_TOKENS_PER_STEP,
    LOG_FREQ,
    PRESETS,
    SEQ_LEN,
    STEPS,
    TOKENS_PER_MICROBATCH_PER_DP_RANK,
    Preset,
    llama3_1b_t_r0,
    llama3_1b_t_r0_tp2,
    parse_log,
    reduce_log,
    summarize,
    torchrun_command,
)

# --- the hand-derived Llama-3 1B parameter count -------------------------------------------------
# oss/torchtitan/torchtitan/models/llama3/__init__.py:151-195 — dim 2048, 16 layers, 32 q-heads,
# 8 kv-heads, head_dim = 2048/32 = 64, ffn_hidden 8192, vocab 128256, weight tying ON, no biases
# (models/common/linear.py:38 `bias: bool = False`).
_WQKV = 2048 * (32 * 64 + 2 * 8 * 64)  # 2048 x 3072  = 6,291,456
_WO = (32 * 64) * 2048  # 2048 x 2048  = 4,194,304
_W1 = 2048 * 8192  # 16,777,216
_W3 = 2048 * 8192  # 16,777,216
_W2 = 8192 * 2048  # 16,777,216
_BLOCK_LINEARS = _WQKV + _WO + _W1 + _W3 + _W2  # 60,817,408
_BLOCK_NORMS = 2 * 2048  # attention_norm + ffn_norm
_BLOCK = _BLOCK_LINEARS + _BLOCK_NORMS  # 60,821,504
_TIED_TABLE = 128256 * 2048  # 262,668,288, counted once
_TOTAL = 16 * _BLOCK + _TIED_TABLE + 2048  # 1,235,814,400
_ATTN_PER_TOKEN_PER_S = 16 * 6 * 32 * (64 + 64)  # 393,216 FLOPs per token per unit of seq_len


class TestParameterCount:
    def test_hand_derived_block_and_total(self) -> None:
        assert _BLOCK_LINEARS == 60_817_408
        assert _TOTAL == 1_235_814_400
        assert LLAMA3_1B.block_linear_params == _BLOCK_LINEARS
        assert LLAMA3_1B.block_params == _BLOCK
        assert LLAMA3_1B.total_params == _TOTAL

    def test_block_linears_are_in_forward_order(self) -> None:
        # The order is not cosmetic: selective AC recomputes by position (see TestAcMultiplier).
        # attention.py:765 (fused wqkv) -> wo -> feed_forward.py:54 `w2(silu(w1(x)) * w3(x))`.
        assert [spec.name for spec in LLAMA3_1B.block_linears()] == ["wqkv", "wo", "w1", "w3", "w2"]
        assert [spec.params for spec in LLAMA3_1B.block_linears()] == [_WQKV, _WO, _W1, _W3, _W2]

    def test_tied_embedding_is_counted_once_and_stays_active(self) -> None:
        # models/utils.py:519-523: an embedding table is dropped from the active count *unless* it
        # is the lm_head's parameter. With tying it is, so active == total here.
        assert LLAMA3_1B.tie_embeddings
        assert LLAMA3_1B.active_params == LLAMA3_1B.total_params == _TOTAL

    def test_untying_moves_262M_params_out_of_the_flop_count(self) -> None:
        untied = DecoderShape(**{**vars(LLAMA3_1B), "tie_embeddings": False})
        assert untied.total_params == _TOTAL + _TIED_TABLE
        assert untied.active_params == _TOTAL  # the embedding lookup is not a matmul
        assert untied.total_params - untied.active_params == _TIED_TABLE

    def test_active_minus_matmul_is_exactly_the_norm_weights(self) -> None:
        # The 6N convention weights *every* remaining parameter by 1, RMSNorm included. That is
        # 67,584 of 1.24B — 0.005%, kept only so our number matches torchtitan's bit for bit.
        assert LLAMA3_1B.active_params - LLAMA3_1B.matmul_params == 16 * _BLOCK_NORMS + 2048
        assert LLAMA3_1B.active_params - LLAMA3_1B.matmul_params == 67_584


class TestSixNd:
    def test_6nd_is_six_times_the_hand_count(self) -> None:
        assert FLOPS_PER_PARAM == 6.0
        assert LLAMA3_1B.dense_flops_per_token() == 6 * _TOTAL == 7_414_886_400

    def test_6nd_has_no_sequence_length_in_it(self) -> None:
        # The whole point of separating the terms: 6ND is flat in S, the attention term is not.
        for seq in (512, 2048, 8192, 131072):
            got = flop_breakdown(LLAMA3_1B, seq)
            assert got.dense_per_token == 6 * _TOTAL


class TestAttentionTerm:
    def test_matches_torchtitans_closed_form(self) -> None:
        # models/utils.py:447 — 6 * H * (qk_head_dim + v_head_dim) * S, summed over layers.
        assert ATTENTION_FLOPS_PER_ELEMENT == 6.0
        for seq in (1024, 4096, 8192):
            expected = 16 * 6 * 32 * (64 + 64) * seq
            assert LLAMA3_1B.attention_flops_per_token(seq) == expected
            assert expected == _ATTN_PER_TOKEN_PER_S * seq

    def test_linear_per_token_and_quadratic_per_sequence(self) -> None:
        per_token_4k = LLAMA3_1B.attention_flops_per_token(4096)
        per_token_8k = LLAMA3_1B.attention_flops_per_token(8192)
        assert per_token_8k == pytest.approx(2 * per_token_4k)  # linear in S, per token
        assert per_token_8k * 8192 == pytest.approx(4 * per_token_4k * 4096)  # S^2 per sequence

    def test_uses_query_heads_not_kv_heads(self) -> None:
        # GQA (8 kv heads here) shrinks the KV cache, not the number of score matrices.
        mqa = DecoderShape(**{**vars(LLAMA3_1B), "n_kv_heads": 1})
        assert mqa.attention_flops_per_token(8192) == LLAMA3_1B.attention_flops_per_token(8192)

    def test_causal_factor_halves_it_and_torchtitan_does_not_apply_it(self) -> None:
        full = LLAMA3_1B.attention_flops_per_token(8192, causal_factor=1.0)
        half = LLAMA3_1B.attention_flops_per_token(8192, causal_factor=0.5)
        assert half == pytest.approx(full / 2)
        # default == torchtitan's convention (models/utils.py:437: no sparsity credit)
        assert LLAMA3_1B.attention_flops_per_token(8192) == full
        with pytest.raises(ValueError):
            LLAMA3_1B.attention_flops_per_token(8192, causal_factor=0.0)

    def test_attention_is_a_third_of_the_pinned_config_not_a_rounding_error(self) -> None:
        at_2k = flop_breakdown(LLAMA3_1B, 2048).attention_share
        at_8k = flop_breakdown(LLAMA3_1B, SEQ_LEN).attention_share
        assert at_2k == pytest.approx(0.0980, abs=5e-4)
        assert at_8k == pytest.approx(0.3029, abs=5e-4)
        # Quoting a 6ND-only MFU at the pinned seq_len understates by this factor.
        assert flop_breakdown(LLAMA3_1B, SEQ_LEN).model_per_token / (6 * _TOTAL) == pytest.approx(
            1.4344, abs=5e-4
        )


class TestAcMultiplier:
    def test_no_ac_is_exactly_one(self) -> None:
        got = flop_breakdown(LLAMA3_1B, SEQ_LEN, ac_policy=AC_NONE)
        assert got.recompute_per_token == 0.0
        assert got.ac_multiplier == 1.0
        assert got.executed_per_token == got.model_per_token

    def test_selective_op_recomputes_the_even_positioned_mms(self) -> None:
        # activation_checkpoint.py:272-278 increments the mm counter then recomputes on even parity
        # => positions 2 and 4 of [wqkv, wo, w1, w3, w2].
        names = [spec.name for spec in recomputed_linears(LLAMA3_1B, AC_SELECTIVE_OP)]
        assert names == ["wo", "w3"]
        recomputed = _WO + _W3
        assert recomputed == 20_971_520
        # 2 of 5 mms, but 34% of the block's linear FLOPs: parity does not care about shape.
        assert recomputed / _BLOCK_LINEARS == pytest.approx(0.3448, abs=5e-4)

    def test_selective_op_multiplier_is_the_policy_arithmetic(self) -> None:
        got = flop_breakdown(LLAMA3_1B, SEQ_LEN, ac_policy=AC_SELECTIVE_OP)
        # one extra FORWARD (2 FLOPs/param) of wo and w3, in every one of the 16 blocks
        expected_extra = 2 * 16 * (_WO + _W3)
        assert got.recompute_per_token == expected_extra == 671_088_640
        model = 6 * _TOTAL + _ATTN_PER_TOKEN_PER_S * SEQ_LEN
        assert got.model_per_token == model
        assert got.ac_multiplier == pytest.approx(1 + expected_extra / model)
        # attention is NEVER recomputed under SAC: SDPA/flex are in the MUST_SAVE set (:42-49)
        assert got.recompute_per_token == pytest.approx(
            flop_breakdown(LLAMA3_1B, 1024, ac_policy=AC_SELECTIVE_OP).recompute_per_token
        )

    def test_full_ac_reruns_the_whole_block_attention_included(self) -> None:
        got = flop_breakdown(LLAMA3_1B, SEQ_LEN, ac_policy=AC_FULL)
        attention = _ATTN_PER_TOKEN_PER_S * SEQ_LEN
        expected_extra = 2 * 16 * _BLOCK + attention / 3  # fwd is 2 of the 6, i.e. a third
        assert got.recompute_per_token == pytest.approx(expected_extra)
        assert got.ac_multiplier == pytest.approx(1 + expected_extra / got.model_per_token)
        # and unlike SAC, it grows with sequence length, because the attention op re-runs too
        short = flop_breakdown(LLAMA3_1B, 1024, ac_policy=AC_FULL)
        assert got.recompute_per_token > short.recompute_per_token

    def test_full_ac_on_the_checkpointed_region_alone_is_the_textbook_four_thirds(self) -> None:
        # "Full recompute = 8ND against 6ND = 4/3" is true *of the checkpointed region*. The
        # whole-model multiplier is smaller, and the difference is not rounding: the tied lm_head
        # (262M of 1.24B params) and the final norm live outside `model.layers`
        # (activation_checkpoint.py:156-161) and are never recomputed.
        block_params = 16 * _BLOCK
        region_only = (6 * block_params + 2 * block_params) / (6 * block_params)
        assert region_only == pytest.approx(4 / 3)
        got = flop_breakdown(LLAMA3_1B, SEQ_LEN, ac_policy=AC_FULL)
        assert got.ac_multiplier < region_only
        assert got.ac_multiplier == pytest.approx(1.2839, abs=5e-4)
        assert flop_breakdown(
            LLAMA3_1B, SEQ_LEN, ac_policy=AC_SELECTIVE_OP
        ).ac_multiplier == pytest.approx(1.0631, abs=5e-4)

    def test_policies_are_ordered_and_unknown_ones_raise(self) -> None:
        mults = {
            p: flop_breakdown(LLAMA3_1B, SEQ_LEN, ac_policy=p).ac_multiplier for p in AC_POLICIES
        }
        assert mults[AC_NONE] == 1.0
        assert mults[AC_NONE] < mults[AC_SELECTIVE_OP] < mults[AC_FULL]
        with pytest.raises(ValueError):
            flop_breakdown(LLAMA3_1B, SEQ_LEN, ac_policy="layerwise_every_other")
        with pytest.raises(ValueError):
            recomputed_linears(LLAMA3_1B, "nope")


class TestUtilization:
    def test_mfu_is_one_at_peak(self) -> None:
        peak = TORCHTITAN_PEAK_BF16_DENSE["H100 SXM"]
        flops_per_token = flop_breakdown(LLAMA3_1B, SEQ_LEN).model_per_token
        at_peak = peak / flops_per_token  # tok/s/GPU that would saturate the tensor cores
        assert utilization(flops_per_token, at_peak, peak) == pytest.approx(1.0)

    def test_mfu_is_dimensionless(self) -> None:
        # Scale FLOPs and peak by the same factor (i.e. change the unit) — the ratio must not move.
        peak = TORCHTITAN_PEAK_BF16_DENSE["H100 SXM"]
        base = utilization(1e10, 9000, peak)
        assert utilization(1e10 * 1e-12, 9000, peak * 1e-12) == pytest.approx(base)
        assert 0.0 < base < 1.0

    def test_mfu_scales_linearly_with_throughput(self) -> None:
        peak = TORCHTITAN_PEAK_BF16_DENSE["H100 SXM"]
        assert utilization(1e10, 18000, peak) == pytest.approx(2 * utilization(1e10, 9000, peak))

    def test_bad_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            utilization(1e10, 0.0, 1e15)
        with pytest.raises(ValueError):
            utilization(1e10, 9000, 0.0)
        with pytest.raises(ValueError):
            utilization(-1.0, 9000, 1e15)

    def test_the_four_conventions_are_ordered_and_differ_by_known_factors(self) -> None:
        report = step_report(
            LLAMA3_1B,
            seq_len=SEQ_LEN,
            tokens_per_s_per_device=9000.0,
            peak_flops_per_device=TORCHTITAN_PEAK_BF16_DENSE["H100 SXM"],
            ac_policy=AC_SELECTIVE_OP,
        )
        assert report.mfu_6nd < report.mfu_executed_attention < report.mfu_model < report.hfu
        assert report.hfu / report.mfu_model == pytest.approx(report.breakdown.ac_multiplier)
        assert report.mfu_model / report.mfu_6nd == pytest.approx(1.4344, abs=5e-4)
        assert report.achieved_tflops_per_device == pytest.approx(
            report.breakdown.model_per_token * 9000.0 / 1e12
        )

    def test_hfu_equals_mfu_iff_no_recompute(self) -> None:
        def report(policy: str) -> StepReport:
            return step_report(
                LLAMA3_1B,
                seq_len=SEQ_LEN,
                tokens_per_s_per_device=9000.0,
                peak_flops_per_device=TORCHTITAN_PEAK_BF16_DENSE["H100 SXM"],
                ac_policy=policy,
            )

        none = report(AC_NONE)
        assert none.hfu == pytest.approx(none.mfu_model)
        full = report(AC_FULL)
        assert full.hfu > full.mfu_model


# --- the harness: log parsing and the refusal ----------------------------------------------------
def _log_line(step: int, tps: float, flops_per_token: float) -> str:
    """A torchtitan metrics line, formatted exactly as components/metrics.py:532-541 does it."""
    tflops = flops_per_token * tps / 1e12
    mfu = 100 * flops_per_token * tps / TORCHTITAN_PEAK_BF16_DENSE["H100 SXM"]
    return (
        f"[titan] 2026-09-08 12:00:00,000 - root - INFO - step: {step:2}  loss:  7.12345  "
        f"grad_norm:  1.2345  memory: 40.12GiB(50.00%)  tps: {round(tps):,}  "
        f"tflops: {tflops:,.2f}  mfu: {mfu:.2f}%"
    )


def _synthetic_log(*, steps: int = 120, tps: float = 9000.0, seq_len: int = SEQ_LEN) -> list[str]:
    flops_per_token = flop_breakdown(LLAMA3_1B, seq_len).model_per_token
    head = [
        "[titan] - root - INFO - Model llama3 1B size: 1,235,814,400 total parameters",
        "[titan] - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14",
    ]
    # a slow first step (compile) and a GC hiccup, so the median/IQR have something to survive
    body = [
        _log_line(s, tps * (0.25 if s == 1 else 0.9 if s % 50 == 0 else 1.0), flops_per_token)
        for s in range(1, steps + 1)
    ]
    return head + body


class TestLogHarness:
    def test_parses_a_titan_metrics_line(self) -> None:
        parsed = parse_log(_synthetic_log(steps=3))
        assert parsed.total_params == 1_235_814_400
        assert parsed.peak_flops == pytest.approx(9.89e14)
        assert [row.step for row in parsed.steps] == [1, 2, 3]
        assert parsed.steps[1].tps == 9000.0
        assert parsed.steps[1].loss == pytest.approx(7.12345)
        assert parsed.steps[1].mfu_pct is not None

    def test_implied_flops_per_token_round_trips_through_the_log(self) -> None:
        model = flop_breakdown(LLAMA3_1B, SEQ_LEN).model_per_token
        row = parse_log(_synthetic_log(steps=2)).steps[1]
        # the log rounds tps to an int and tflops to 2dp; that is the only loss
        assert row.implied_flops_per_token == pytest.approx(model, rel=1e-4)

    def test_reduce_log_drops_the_warmup_and_reports_median_iqr(self) -> None:
        result = reduce_log(_synthetic_log(), preset=PRESETS["fsdp8"], warmup_steps=20, nproc=8)
        assert result["measured_steps"] == 100
        assert result["tps_per_device"]["median"] == pytest.approx(9000.0, rel=1e-3)
        # the compile-slow step 1 is inside the warm-up, so it cannot drag the median
        assert result["tps_per_device"]["min"] > 8000.0
        assert result["metric"] == "titan_llama3_1b_fsdp8_tps"
        assert result["flops_per_token_agreement"] == pytest.approx(1.0, abs=1e-3)
        # loss is recorded, never gated: the threshold that separates "converging" from
        # "diverged" is a numerics call, and this rung does not make it.
        assert result["loss"]["median"] == pytest.approx(7.12345)
        assert len(result["loss_first_last"]) == 2

    def test_reduce_log_refuses_a_short_window(self) -> None:
        with pytest.raises(SystemExit, match=">= 50"):
            reduce_log(_synthetic_log(steps=60), preset=PRESETS["fsdp8"], warmup_steps=20, nproc=8)

    def test_reduce_log_refuses_when_the_flop_model_disagrees(self) -> None:
        # A log from a seq_len=2048 job: same tok/s, 26% fewer FLOPs per token. Recording it as
        # this rung's floor would compare T-R1 against a different quantity.
        with pytest.raises(SystemExit, match="REFUSED"):
            reduce_log(
                _synthetic_log(seq_len=2048), preset=PRESETS["fsdp8"], warmup_steps=20, nproc=8
            )

    def test_summarize_is_median_and_interquartile(self) -> None:
        got = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
        assert got["median"] == 3.0
        assert got["q25"] == 2.0
        assert got["q75"] == 4.0
        assert got["iqr_pct"] == pytest.approx(100 * 2 / 3)
        assert got["n"] == 5.0

    def test_both_presets_are_wired_and_named_apart(self) -> None:
        assert set(PRESETS) == {"fsdp8", "fsdp4_tp2"}
        metrics = {p.metric for p in PRESETS.values()}
        assert len(metrics) == 2  # T-R1's 85% must say which one it means
        assert PRESETS["fsdp4_tp2"].tensor_parallel_degree == 2

    def test_torchrun_command_is_the_documented_invocation(self) -> None:
        cmd = torchrun_command(PRESETS["fsdp8"], nproc=8, steps=None)
        assert cmd[:2] == ["torchrun", "--nproc_per_node=8"]
        assert "torchtitan.train" in cmd
        assert cmd[cmd.index("--module") + 1] == "scratch_llm.training.titan_floor"
        assert cmd[cmd.index("--config") + 1] == "llama3_1b_t_r0"
        assert "--training.steps" not in cmd  # steps live in the config, not the CLI
        assert torchrun_command(PRESETS["fsdp8"], nproc=8, steps=7)[-2:] == [
            "--training.steps",
            "7",
        ]


class TestPinnedConfig:
    """The config function itself — the single most likely way this rung dies on the box.

    Every kwarg below has to exist at pin d263ca0a. Building it costs milliseconds and no GPU, so
    it runs before any silicon is rented; it skips here only because torchtitan's own deps
    (tyro, grain, ...) are not installed on the laptop.
    """

    def test_builds_and_pins_what_the_spec_says_it_pins(self) -> None:
        pytest.importorskip("tyro", reason="pip install -r oss/torchtitan/requirements.txt")
        pytest.importorskip("torchtitan", reason="infra/bootstrap.sh installs it at the pin")
        cfg = llama3_1b_t_r0()
        assert cfg.model_spec is not None and cfg.model_spec.flavor == "1B"
        assert cfg.training.max_context_length == SEQ_LEN
        assert (
            cfg.training.num_tokens_per_microbatch_per_dp_rank == TOKENS_PER_MICROBATCH_PER_DP_RANK
        )
        assert cfg.training.num_tokens_per_train_step == GLOBAL_TOKENS_PER_STEP
        assert cfg.training.steps == STEPS
        assert cfg.metrics.log_freq == LOG_FREQ and cfg.metrics.disable_color_printing
        assert cfg.compile.enable and "model" in cfg.compile.components
        assert cfg.parallelism.tensor_parallel_degree == 1
        assert cfg.debug.seed == 0

    def test_tp2_changes_the_parallelism_and_nothing_else(self) -> None:
        pytest.importorskip("tyro", reason="pip install -r oss/torchtitan/requirements.txt")
        pytest.importorskip("torchtitan", reason="infra/bootstrap.sh installs it at the pin")
        base, tp2 = llama3_1b_t_r0(), llama3_1b_t_r0_tp2()
        assert tp2.parallelism.tensor_parallel_degree == 2
        assert base.parallelism.tensor_parallel_degree == 1
        # same global batch, so TP=2 buys its halved DP degree with 2 grad-accum steps, not with
        # half the tokens per optimizer step
        assert tp2.training.num_tokens_per_train_step == base.training.num_tokens_per_train_step
        assert (
            tp2.training.num_tokens_per_microbatch_per_dp_rank
            == base.training.num_tokens_per_microbatch_per_dp_rank
        )
        assert tp2.training.max_context_length == base.training.max_context_length

    def test_global_batch_is_the_same_tokens_per_gpu_in_both_presets(self) -> None:
        # CPU arithmetic, no torchtitan needed: 131072 global / 8 GPUs = 16384 tok/GPU/step, with
        # grad_accum = global / (microbatch * dp_degree) = 1 at TP=1 and 2 at TP=2
        # (`trainer.py:462-474`).
        assert GLOBAL_TOKENS_PER_STEP == 131072
        for preset in PRESETS.values():
            dp_degree = 8 // preset.tensor_parallel_degree
            grad_accum = GLOBAL_TOKENS_PER_STEP // (TOKENS_PER_MICROBATCH_PER_DP_RANK * dp_degree)
            assert GLOBAL_TOKENS_PER_STEP % (TOKENS_PER_MICROBATCH_PER_DP_RANK * dp_degree) == 0
            assert grad_accum == preset.tensor_parallel_degree
            assert GLOBAL_TOKENS_PER_STEP / 8 == 16384


# --- the map, made executable --------------------------------------------------------------------
_TITAN = workspace_root() / "oss" / "torchtitan"


@pytest.mark.skipif(not _TITAN.is_dir(), reason=f"no torchtitan checkout at {_TITAN}; infra/oss.sh")
class TestTorchtitanConventionHasNotDrifted:
    """The map claims three things about torchtitan's FLOP convention. Assert them against source.

    If any of these fails, `experiments/T1/T-R0/map.md` is stale and the floor's MFU column is
    being compared to a different formula than the one it was written against.
    """

    def test_flops_per_token_is_6_active_plus_attention(self) -> None:
        src = (_TITAN / "torchtitan/models/llama3/model.py").read_text()
        assert "return nparams, 6 * active_nparams + attention_op_flops" in src

    def test_attention_term_is_6_h_dqk_dv_s_with_no_causal_discount(self) -> None:
        src = (_TITAN / "torchtitan/models/utils.py").read_text()
        assert "return 6 * num_heads * (qk_head_dim + v_head_dim) * attended_tokens" in src
        assert "do not account for sparsity in causal attention" in src
        assert "recomputation should not be counted in calculating MFU" in src

    def test_mfu_is_model_flops_per_device_over_spec_peak(self) -> None:
        src = (_TITAN / "torchtitan/components/metrics.py").read_text()
        assert "tps = self.ntokens_since_last_log / (" in src
        assert "time_delta * self.parallel_dims.non_data_parallel_size" in src
        assert "mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops" in src
        # no HFU anywhere: torchtitan never adds the recompute term
        assert "hfu" not in src.lower()

    def test_h100_peak_matches_the_constant_we_divide_by(self) -> None:
        src = (_TITAN / "torchtitan/tools/utils.py").read_text()
        assert "return 989e12" in src
        assert TORCHTITAN_PEAK_BF16_DENSE["H100 SXM"] == 989e12

    def test_the_1b_flavor_still_has_the_shape_we_hand_counted(self) -> None:
        src = (_TITAN / "torchtitan/models/llama3/__init__.py").read_text()
        body = src[src.index("def _1b(") : src.index("def _3b(")]
        for expected in ("dim = 2048", "n_heads = 32", "n_kv_heads = 8", "n_layers = 16"):
            assert expected in body, expected
        assert "enable_weight_tying=True" in body
        assert re.search(r"multiple_of=1024,\s*ffn_dim_multiplier=1\.5", body)
        assert '"1B": (_1b, 131072)' in src

    def test_selective_ac_still_recomputes_every_second_mm(self) -> None:
        src = (_TITAN / "torchtitan/distributed/activation_checkpoint.py").read_text()
        assert "if func in mm_ops and meta[mm_count_key] % 2 == 0:" in src
        assert "return CheckpointPolicy.PREFER_RECOMPUTE" in src
        assert "torch.ops.aten._scaled_dot_product_cudnn_attention.default" in src

    def test_fully_shard_granularity_is_one_unit_per_block(self) -> None:
        src = (_TITAN / "torchtitan/distributed/fsdp.py").read_text()
        assert "for layer_id, transformer_block in model.layers.items():" in src
        assert "Group them together in one FSDP unit to avoid duplicate all-gathers." in src

    def test_there_is_still_no_llama3_1b_recipe_upstream(self) -> None:
        # The reason titan_floor.py has to define one. If upstream adds `llama3_1b`, prefer it and
        # re-pin — a floor should be someone else's config wherever that is possible.
        src = (_TITAN / "torchtitan/models/llama3/config_registry.py").read_text()
        assert "def llama3_1b(" not in src


# --- the measurement itself: 8 GPUs, never a silent pass -----------------------------------------
@pytest.mark.gpu
def test_floor_measurement_requires_eight_gpus() -> None:
    torch = pytest.importorskip("torch")
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n < 8 or shutil.which("torchrun") is None:
        pytest.skip(
            f"T1/T-R0 is an 8xH100 measurement; this box has {n} visible GPU(s). On the box:\n"
            "  infra/rent.sh sync-up <user@host> && ssh <host>\n"
            "  make predict L=T1 R=T-R0 M=titan_llama3_1b_fsdp8_tps V=<tok/s/GPU> "
            'NOTE="<mechanism>"\n'
            "  bash ladders/experiments/T1/T-R0/floor.sh\n"
            "  bash ladders/experiments/T1/T-R0/run.sh"
        )
    preset: Preset = PRESETS["fsdp8"]
    assert preset.tensor_parallel_degree == 1
    pytest.skip(
        "8 GPUs present: run the floor through the harness, not through pytest — "
        "bash experiments/T1/T-R0/floor.sh (it locks clocks and writes provenance)."
    )
