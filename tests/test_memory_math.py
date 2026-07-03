"""A2 `optimizer_state_sharding_accounting` — the 100B memory one-pager as executable assertions.

Every headline number in `docs/design/A2_100B_MEMORY_ONEPAGER.md` is reproduced here by calling
the same `scratch_llm.utils.memory_math` function the doc cites; if a formula drifts, the doc is
provably stale. Pure CPU integer math, no torch.
"""

import math

import pytest

from scratch_llm.utils.memory_math import (
    GB,
    TB,
    activation_bytes_per_layer,
    gpus_needed,
    optimizer_state_bytes,
    param_bytes,
    residual_stream_bytes,
    training_state_breakdown,
    zero_shard_breakdown,
    zero_shard_bytes,
)

N_100B = 100 * 10**9
HBM_80GB = 80 * GB
# The one-pager's concrete 100B shape: 12·L·d² = 12·80·10240² ≈ 100.7B non-embedding params.
LAYERS, D_MODEL, N_HEADS, SEQ = 80, 10240, 80, 4096


class TestPerParamStories:
    def test_fp32_story_is_exactly_16_bytes_per_param(self) -> None:
        assert optimizer_state_bytes(1, mixed_precision=False) == 16
        breakdown = training_state_breakdown(1, mixed_precision=False)
        assert breakdown == {
            "fp32_weights": 4,
            "fp32_grads": 4,
            "adam_m_fp32": 4,
            "adam_v_fp32": 4,
        }

    def test_mixed_story_is_18_to_20_bytes_per_param(self) -> None:
        assert optimizer_state_bytes(1, mixed_precision=True, bf16_grad_copy=True) == 20
        assert optimizer_state_bytes(1, mixed_precision=True, bf16_grad_copy=False) == 18
        breakdown = training_state_breakdown(1, mixed_precision=True)
        assert breakdown["fp32_master_weights"] == 4
        assert breakdown["bf16_weights"] == 2
        assert breakdown["bf16_grads"] == 2

    def test_100b_headline_state_is_1_6_to_2_0_tb(self) -> None:
        fp32_total = optimizer_state_bytes(N_100B, mixed_precision=False)
        mixed_lo = optimizer_state_bytes(N_100B, mixed_precision=True, bf16_grad_copy=False)
        mixed_hi = optimizer_state_bytes(N_100B, mixed_precision=True, bf16_grad_copy=True)
        assert fp32_total == 1600 * GB  # 1.6 TB
        assert mixed_lo == 1800 * GB  # 1.8 TB
        assert mixed_hi == 2000 * GB  # 2.0 TB
        assert 1.6 * TB <= fp32_total <= mixed_hi <= 2.0 * TB

    def test_param_bytes_one_pager_rows(self) -> None:
        assert param_bytes(N_100B, 4) == 400 * GB  # each fp32 term: master/grad/m/v
        assert param_bytes(N_100B, 2) == 200 * GB  # each bf16 copy: weights/grads


class TestZeroLadder:
    def test_world_size_1_matches_total_state_for_every_stage(self) -> None:
        for stage in (0, 1, 2, 3):
            for mixed in (False, True):
                assert zero_shard_bytes(stage, N_100B, 1, mixed) == optimizer_state_bytes(
                    N_100B, mixed
                )

    def test_zero1_at_w64_cuts_optimizer_state_64x_but_replicates_params_and_grads(self) -> None:
        full = zero_shard_breakdown(0, N_100B, 64, mixed_precision=False)
        z1 = zero_shard_breakdown(1, N_100B, 64, mixed_precision=False)
        assert z1["optimizer"] * 64 == full["optimizer"]  # m+v sharded exactly 64x
        assert z1["weights"] == full["weights"] == 400 * GB  # replicated
        assert z1["grads"] == full["grads"] == 400 * GB  # replicated

    def test_zero3_divides_everything_by_world_size(self) -> None:
        for mixed in (False, True):
            total = optimizer_state_bytes(N_100B, mixed)
            for world in (8, 64, 1000):  # all divide 100e9
                assert zero_shard_bytes(3, N_100B, world, mixed) * world == total

    def test_per_rank_bytes_monotone_in_stage(self) -> None:
        sizes = [zero_shard_bytes(s, N_100B, 64, mixed_precision=True) for s in (0, 1, 2, 3)]
        assert sizes == sorted(sizes, reverse=True)
        assert sizes[0] > sizes[1] > sizes[2] > sizes[3]

    def test_one_pager_ladder_rows_mixed_w64(self) -> None:
        rows = [zero_shard_bytes(s, N_100B, 64, mixed_precision=True) for s in (0, 1, 2, 3)]
        assert rows[0] == 2000 * GB  # DDP: full 2.0 TB on every rank
        assert rows[1] == 425 * GB  # ZeRO-1: 400 GB replicated + 1.6 TB / 64
        assert rows[2] == 228_125_000_000  # ZeRO-2: 228.125 GB
        assert rows[3] == 31_250_000_000  # ZeRO-3: 31.25 GB — first rung under 80 GB

    def test_uneven_world_charges_ceil_of_owned_params(self) -> None:
        got = zero_shard_breakdown(3, 10, 3, mixed_precision=False)  # ceil(10/3)=4 owned
        assert got == {"weights": 16, "grads": 16, "optimizer": 32}

    def test_stage_and_world_validation(self) -> None:
        with pytest.raises(ValueError):
            zero_shard_bytes(4, 10, 2, mixed_precision=False)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            zero_shard_bytes(1, 10, 0, mixed_precision=False)


class TestActivations:
    def test_formula_matches_korthikanti_hand_computation(self) -> None:
        b, s, h, a = 2, 8, 16, 4
        assert (
            activation_bytes_per_layer(b, s, h, a, flash=False)
            == 34 * s * b * h + 5 * a * s * s * b
        )
        assert activation_bytes_per_layer(b, s, h, a, flash=True) == 34 * s * b * h

    def test_flash_removes_exactly_the_quadratic_term(self) -> None:
        b, s, h, a = 1, SEQ, D_MODEL, N_HEADS
        diff = activation_bytes_per_layer(b, s, h, a, flash=False) - activation_bytes_per_layer(
            b, s, h, a, flash=True
        )
        assert diff == 5 * a * s * s * b

    def test_one_pager_seq4k_numbers(self) -> None:
        no_flash = activation_bytes_per_layer(1, SEQ, D_MODEL, N_HEADS, flash=False)
        flash = activation_bytes_per_layer(1, SEQ, D_MODEL, N_HEADS, flash=True)
        assert no_flash == 8_136_949_760  # ~8.14 GB / layer
        assert flash == 1_426_063_360  # ~1.43 GB / layer
        assert LAYERS * no_flash == 650_955_980_800  # ~651 GB for the 80-layer stack
        assert LAYERS * flash == 114_085_068_800  # ~114 GB
        # Checkpointing keeps one residual tensor per block boundary: ~84 MB * 80 ≈ 6.7 GB.
        assert residual_stream_bytes(1, SEQ, D_MODEL) == 83_886_080
        assert LAYERS * residual_stream_bytes(1, SEQ, D_MODEL) == 6_710_886_400
        # The doc's "~82%": the quadratic term's share of a vanilla layer's activation bytes.
        assert round(100 * (no_flash - flash) / no_flash) == 82

    def test_quadratic_term_grows_with_seq_squared(self) -> None:
        f = lambda s: activation_bytes_per_layer(1, s, D_MODEL, N_HEADS, flash=False)  # noqa: E731
        quad = lambda s: 5 * N_HEADS * s * s  # noqa: E731
        assert f(8192) - f(4096) == (quad(8192) - quad(4096)) + 34 * 4096 * D_MODEL

    def test_one_pager_config_is_really_100b(self) -> None:
        nonembed = 12 * LAYERS * D_MODEL * D_MODEL
        assert nonembed == 100_663_296_000  # ≈ 100.7B — the doc's 100B shape
        assert math.isclose(nonembed, N_100B, rel_tol=0.01)


class TestGpusNeeded:
    def test_zero3_state_only_gpu_counts(self) -> None:
        assert gpus_needed(N_100B, HBM_80GB, stage=3, mixed_precision=False) == 20
        assert gpus_needed(N_100B, HBM_80GB, stage=3, mixed_precision=True) == 25

    def test_zero1_and_zero2_cannot_fit_100b_on_80gb_at_any_world_size(self) -> None:
        for stage in (0, 1, 2):
            with pytest.raises(ValueError):
                gpus_needed(N_100B, HBM_80GB, stage=stage, mixed_precision=True)  # type: ignore[arg-type]

    def test_result_actually_fits_and_is_minimal(self) -> None:
        w = gpus_needed(N_100B, HBM_80GB, stage=3, mixed_precision=True)
        assert zero_shard_bytes(3, N_100B, w, mixed_precision=True) <= HBM_80GB
        assert zero_shard_bytes(3, N_100B, w - 1, mixed_precision=True) > HBM_80GB

    def test_activation_budget_raises_gpu_count(self) -> None:
        act = LAYERS * activation_bytes_per_layer(1, SEQ, D_MODEL, N_HEADS, flash=True)
        w_with = gpus_needed(N_100B, HBM_80GB, stage=3, activation_bytes_per_gpu=act // 16)
        assert w_with > gpus_needed(N_100B, HBM_80GB, stage=3)

    def test_small_model_fits_one_gpu(self) -> None:
        n_1b = 10**9
        assert gpus_needed(n_1b, HBM_80GB, stage=0, mixed_precision=True) == 1
