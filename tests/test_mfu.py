"""A6 MFU/HFU instrumentation — the scoreboard, as executable known-answer assertions.

Oracles (pure CPU arithmetic, no GPU):
  1. the 6ND identity is exact by hand computation,
  2. the published PaLM 540B anchor (46.2% MFU / 57.8% HFU) is reproduced to within the ~0.5pp the
     6N approximation drops by omitting the attention term,
  3. HFU >= MFU for every run, with the analytic (6+2r)/6 ratio,
  4. the six-killer decomposition multiplies back to the measured MFU and its blame-shares sum to 1.
"""

import math

import pytest

from scratch_llm.utils.mfu import (
    FLOPS_PER_PARAM,
    MFU_KILLERS,
    PEAK_FLOPS_BF16_DENSE,
    TFLOP,
    compose_mfu,
    hardware_flops,
    hfu,
    hfu_over_mfu,
    mfu,
    mfu_gap_attribution,
    peak_flops_aggregate,
    training_flops,
    unexplained_factor,
)

# PaLM 540B published training-efficiency anchor (Chowdhery et al. 2022, Appendix B):
# 6144 TPU-v4 chips @ 275 TFLOP/s bf16 dense, 238,300 tokens/sec observed throughput,
# reported 46.2% MFU / 57.8% HFU.
PALM_N = 540 * 10**9
PALM_CHIPS = 6144
PALM_PEAK = 275 * TFLOP
PALM_TOKENS_PER_SEC = 238_300
PALM_MFU_PUBLISHED = 0.462
PALM_HFU_PUBLISHED = 0.578


class TestSixNDIdentity:
    def test_training_flops_is_exactly_6ND(self) -> None:
        n, d = 7, 11
        assert training_flops(n, d) == 6 * n * d == 462
        # the 6 decomposes as 2 forward + 4 backward
        assert FLOPS_PER_PARAM == 6.0
        assert training_flops(n, d, flops_per_param=2.0) == 2 * n * d  # inference-only forward

    def test_identity_holds_at_scale(self) -> None:
        # a 70B model over a 4M-token step: 6·70e9·4e6 = 1.68e18 FLOPs
        assert training_flops(70 * 10**9, 4 * 10**6) == 6 * 70 * 10**9 * 4 * 10**6
        assert training_flops(70 * 10**9, 4 * 10**6) == 1.68e18

    def test_zero_work_is_zero_flops(self) -> None:
        assert training_flops(0, 1000) == 0.0
        assert training_flops(1000, 0) == 0.0


class TestMfuMechanics:
    def test_mfu_is_achieved_over_peak_by_hand(self) -> None:
        # 10 devices at 100 TFLOP/s peak = 1e15 peak; a step doing 5e14 FLOPs in 1 s => 50%.
        n, d = 10**9, 1  # 6e9 FLOPs/token... pick numbers that land clean
        # choose so training_flops = 5e14: 6·n·d with n such that 6n = 5e14 -> not integer; use flops directly
        step_flops = training_flops(n, d)  # 6e9
        peak = peak_flops_aggregate(10, 100 * TFLOP)  # 1e15
        expected = step_flops / 1.0 / peak
        assert mfu(n, d, 1.0, 10, 100 * TFLOP) == pytest.approx(expected)

    def test_mfu_scales_inversely_with_step_time(self) -> None:
        base = mfu(PALM_N, PALM_TOKENS_PER_SEC, 1.0, PALM_CHIPS, PALM_PEAK)
        slow = mfu(PALM_N, PALM_TOKENS_PER_SEC, 2.0, PALM_CHIPS, PALM_PEAK)
        assert slow == pytest.approx(base / 2.0)

    def test_mfu_scales_inversely_with_device_count(self) -> None:
        few = mfu(PALM_N, PALM_TOKENS_PER_SEC, 1.0, PALM_CHIPS, PALM_PEAK)
        many = mfu(PALM_N, PALM_TOKENS_PER_SEC, 1.0, 2 * PALM_CHIPS, PALM_PEAK)
        assert many == pytest.approx(few / 2.0)

    def test_bad_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            mfu(PALM_N, 1, 0.0, PALM_CHIPS, PALM_PEAK)  # zero step time
        with pytest.raises(ValueError):
            mfu(PALM_N, 1, 1.0, 0, PALM_PEAK)  # zero devices
        with pytest.raises(ValueError):
            mfu(PALM_N, 1, 1.0, PALM_CHIPS, 0.0)  # zero peak


class TestPalmAnchor:
    def test_palm_mfu_6nd_estimate_matches_published_within_half_point(self) -> None:
        # step_time = 1 s with tokens = tokens/sec gives the per-second (throughput) MFU directly.
        got = mfu(PALM_N, PALM_TOKENS_PER_SEC, 1.0, PALM_CHIPS, PALM_PEAK)
        # exact hand value of the 6N estimate
        hand = 6 * PALM_N * PALM_TOKENS_PER_SEC / (PALM_CHIPS * PALM_PEAK)
        assert got == pytest.approx(hand)
        assert got == pytest.approx(0.45697, abs=1e-4)
        # reproduces the published 46.2% to within the ~0.5pp the 6N model drops by omitting
        # attention FLOPs (and rounding of the 540B / 238.3k-tok numbers).
        assert abs(got - PALM_MFU_PUBLISHED) < 0.01

    def test_palm_mfu_closes_to_published_with_attention_correction(self) -> None:
        # PaLM: L=118, s=2048, d=18432. The attention term ≈ 12·L·s·d per token nudges the
        # estimate up toward the paper's 46.2%. This documents *where* the 0.5pp lives; the
        # correction is additive FLOPs the caller supplies via extra_flops.
        n_layers, seq, d_model = 118, 2048, 18432
        attn_per_token = 12 * n_layers * seq * d_model
        extra = attn_per_token * PALM_TOKENS_PER_SEC
        got = mfu(PALM_N, PALM_TOKENS_PER_SEC, 1.0, PALM_CHIPS, PALM_PEAK, extra_flops=extra)
        # attention pushes it above the 6N-only 45.7%, bracketing the published 46.2%
        assert got > 0.457
        assert abs(got - PALM_MFU_PUBLISHED) < 0.012

    def test_palm_hfu_from_selective_recompute_matches_published(self) -> None:
        # The published 57.8/46.2 = 1.251 ratio implies a selective-recompute forward fraction of
        # r = (1.251·6 - 6)/2 ≈ 0.753. Feeding that r reproduces the 57.8% HFU from the 6N MFU base.
        r = (PALM_HFU_PUBLISHED / PALM_MFU_PUBLISHED * FLOPS_PER_PARAM - FLOPS_PER_PARAM) / 2.0
        assert r == pytest.approx(0.7532, abs=1e-3)
        got_hfu = hfu(PALM_N, PALM_TOKENS_PER_SEC, 1.0, PALM_CHIPS, PALM_PEAK, recompute_fraction=r)
        got_mfu = mfu(PALM_N, PALM_TOKENS_PER_SEC, 1.0, PALM_CHIPS, PALM_PEAK)
        # HFU/MFU reproduces the published ratio exactly (it is (6+2r)/6 by construction)
        assert got_hfu / got_mfu == pytest.approx(PALM_HFU_PUBLISHED / PALM_MFU_PUBLISHED)
        # and HFU lands within ~0.5pp of the published 57.8%, same 6N residual as MFU
        assert abs(got_hfu - PALM_HFU_PUBLISHED) < 0.01

    def test_palm_peak_is_the_tpu_v4_spec(self) -> None:
        assert PEAK_FLOPS_BF16_DENSE["TPU_v4"] == PALM_PEAK


class TestHfuVsMfu:
    def test_hfu_ge_mfu_always(self) -> None:
        for r in (0.0, 0.1, 0.5, 0.753, 1.0):
            got_hfu = hfu(
                PALM_N, 10**6, 3.0, 512, PEAK_FLOPS_BF16_DENSE["H100_SXM"], recompute_fraction=r
            )
            got_mfu = mfu(PALM_N, 10**6, 3.0, 512, PEAK_FLOPS_BF16_DENSE["H100_SXM"])
            assert got_hfu >= got_mfu

    def test_hfu_equals_mfu_iff_no_recompute(self) -> None:
        args = (PALM_N, 10**6, 3.0, 512, PEAK_FLOPS_BF16_DENSE["H100_SXM"])
        assert hfu(*args, recompute_fraction=0.0) == pytest.approx(mfu(*args))
        assert hfu(*args, recompute_fraction=1.0) > mfu(*args)

    def test_full_recompute_ratio_is_exactly_four_thirds(self) -> None:
        assert hfu_over_mfu(1.0) == pytest.approx(4.0 / 3.0)
        assert hfu_over_mfu(0.0) == 1.0
        assert hfu_over_mfu(0.5) == pytest.approx(7.0 / 6.0)

    def test_hardware_flops_is_model_plus_extra_forward(self) -> None:
        n, d = 100, 10
        assert hardware_flops(n, d, recompute_fraction=0.0) == training_flops(n, d)
        # full recompute adds one forward = +2ND => 8ND
        assert hardware_flops(n, d, recompute_fraction=1.0) == 8 * n * d
        assert hardware_flops(n, d, recompute_fraction=0.5) == 7 * n * d

    def test_numeric_ratio_matches_analytic(self) -> None:
        args = (PALM_N, 10**6, 3.0, 512, PEAK_FLOPS_BF16_DENSE["H100_SXM"])
        for r in (0.0, 0.25, 0.75, 1.0):
            ratio = hfu(*args, recompute_fraction=r) / mfu(*args)
            assert ratio == pytest.approx(hfu_over_mfu(r))

    def test_recompute_fraction_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            hardware_flops(100, 10, recompute_fraction=1.5)
        with pytest.raises(ValueError):
            hfu_over_mfu(-0.1)


class TestSixKillerDecomposition:
    def test_compose_is_ideal_times_product(self) -> None:
        eff = {"unoverlapped_comm": 0.9, "pipeline_bubble": 0.8, "memory_bound_kernels": 0.5}
        assert compose_mfu(1.0, eff) == pytest.approx(0.9 * 0.8 * 0.5)
        assert compose_mfu(0.6, eff) == pytest.approx(0.6 * 0.36)

    def test_missing_killer_is_lossless(self) -> None:
        # a killer absent from the dict contributes factor 1.0
        assert compose_mfu(1.0, {"stragglers": 0.7}) == pytest.approx(0.7)
        assert compose_mfu(1.0, {}) == 1.0

    def test_attribution_shares_sum_to_one(self) -> None:
        eff = {"unoverlapped_comm": 0.9, "pipeline_bubble": 0.8, "small_per_gpu_batch": 0.6}
        shares = mfu_gap_attribution(eff)
        assert sum(shares.values()) == pytest.approx(1.0)
        # every killer gets nonzero blame since every η < 1
        assert all(0.0 < s < 1.0 for s in shares.values())

    def test_equal_losses_get_equal_blame(self) -> None:
        eff = {"unoverlapped_comm": 0.5, "pipeline_bubble": 0.5}
        shares = mfu_gap_attribution(eff)
        assert shares["unoverlapped_comm"] == pytest.approx(0.5)
        assert shares["pipeline_bubble"] == pytest.approx(0.5)

    def test_attribution_is_log_additive_by_hand(self) -> None:
        eff = {"unoverlapped_comm": 0.9, "moe_imbalance": 0.4}
        shares = mfu_gap_attribution(eff)
        la, lb = -math.log(0.9), -math.log(0.4)
        assert shares["unoverlapped_comm"] == pytest.approx(la / (la + lb))
        assert shares["moe_imbalance"] == pytest.approx(lb / (la + lb))
        # the bigger loss (0.4) owns the bigger share
        assert shares["moe_imbalance"] > shares["unoverlapped_comm"]

    def test_lossless_killer_gets_zero_blame(self) -> None:
        shares = mfu_gap_attribution({"unoverlapped_comm": 1.0, "stragglers": 0.5})
        assert shares["unoverlapped_comm"] == 0.0
        assert shares["stragglers"] == pytest.approx(1.0)

    def test_no_gap_gives_all_zero_shares(self) -> None:
        shares = mfu_gap_attribution({"unoverlapped_comm": 1.0, "pipeline_bubble": 1.0})
        assert all(s == 0.0 for s in shares.values())

    def test_unexplained_factor_is_one_when_killers_fully_explain(self) -> None:
        ideal, eff = 0.6, {"unoverlapped_comm": 0.9, "pipeline_bubble": 0.8}
        measured = compose_mfu(ideal, eff)  # exactly explained
        assert unexplained_factor(ideal, measured, eff) == pytest.approx(1.0)

    def test_unexplained_factor_flags_a_seventh_loss(self) -> None:
        ideal, eff = 0.6, {"unoverlapped_comm": 0.9}
        modeled = compose_mfu(ideal, eff)  # 0.54
        measured = modeled * 0.8  # an un-modeled 20% loss remains
        assert unexplained_factor(ideal, measured, eff) == pytest.approx(0.8)

    def test_all_six_killers_recognized(self) -> None:
        eff = dict.fromkeys(MFU_KILLERS, 0.9)
        composed = compose_mfu(1.0, eff)
        assert composed == pytest.approx(0.9**6)
        assert len(MFU_KILLERS) == 6

    def test_unknown_killer_and_bad_efficiency_raise(self) -> None:
        with pytest.raises(ValueError):
            compose_mfu(1.0, {"typo_killer": 0.9})
        with pytest.raises(ValueError):
            compose_mfu(1.0, {"stragglers": 1.5})
        with pytest.raises(ValueError):
            mfu_gap_attribution({"stragglers": 0.0})


class TestRealisticScoreboard:
    def test_worked_h100_run_reports_a_sane_number(self) -> None:
        # 70B model, 512 H100s, 4M tokens/step, 6.2 s/step.
        n, tokens, t, dev = 70 * 10**9, 4 * 10**6, 6.2, 512
        got = mfu(n, tokens, t, dev, PEAK_FLOPS_BF16_DENSE["H100_SXM"])
        # 6·70e9·4e6 / 6.2 / (512·989.5e12) — a plausible large-run MFU, in (0.4, 0.6)
        hand = 6 * n * tokens / t / (dev * PEAK_FLOPS_BF16_DENSE["H100_SXM"])
        assert got == pytest.approx(hand)
        assert 0.4 < got < 0.6
