"""S3 scaling-sweep wiring: grid construction, steps math, results IO, and the fit gates.

Pure/CPU-level: the grid's exact N is locked against ``model_config_for_depth`` instantiation,
the token-budget → steps conversion is exact, results.json round-trips, and ``fit_scaling_law``
is exercised on planted power-law points — the a+b gate passes on a consistent planted law and
raises on a planted C=6ND violation, the R² gate raises on an off-law point, the D:N decision
rule flips both ways, and the oracle overlay carries nanochat's published constants.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from scratch_llm.model import TransformerLM
from scratch_llm.scaling.isoflop import isoflop_min
from scratch_llm.scaling.s3_sweep import (
    CONTEXT_LENGTH,
    D20_COMPUTE_FLOPS,
    GRID_SPEC,
    GRID_SPEC_V2,
    VOCAB_SIZE,
    SweepRecord,
    append_result,
    average_seed_replicates,
    build_grid,
    build_grid_v2,
    fit_scaling_law,
    load_results,
    nanochat_core_fit,
    select_points,
    steps_for_budget,
    write_fit_outputs,
)
from scratch_llm.speedrun import model_config_for_depth

# Measured once by instantiation (vocab 32768, ctx 2048, untied embeddings, qk_norm=True),
# then frozen. qk_norm adds per-layer parameters that the real GPU training path enables.
# Depths 6/14/16 are the v2 grid's additions — model_config_for_depth is generic over depth
# (d_model = 64·depth, head_dim 128), no registry extension needed.
EXPECTED_N = {
    4: 19_991_808,
    6: 35_789_184,
    8: 59_255_296,
    12: 135_288_576,
    14: 195_228_544,
    16: 269_521_920,
}


def _record(
    point: str,
    depth: int,
    n_params: int,
    ratio: float,
    tokens: int,
    compute: float,
    val_bpb: float,
    *,
    seed: int = 0,
    target_compute: float | None = None,
) -> SweepRecord:
    rec = SweepRecord(
        point=point,
        depth=depth,
        n_params=n_params,
        ratio=ratio,
        tokens=tokens,
        compute=compute,
        val_loss=2.0,  # nats/token — carried in the record, not the fit axis
        val_bpb=val_bpb,
        wall_s=60.0,
        seed=seed,
    )
    if target_compute is not None:
        rec["target_compute"] = target_compute
    return rec


def _planted_isoflop_records(
    k_n: float,
    a: float,
    budgets: tuple[float, ...] = (1e16, 1e17, 1e18),
    *,
    off_law_budget: float | None = None,
    freeze_tokens: bool = False,
) -> list[SweepRecord]:
    """Plant a textbook IsoFLOP sweep: at each budget C, three runs at N_opt·{0.5, 1, 2} with
    bpb = 1 + log2(N/N_opt)² — the per-budget argmin lands exactly on the planted
    N_opt = k_n·C^a. Recorded tokens are C/(6N) (the C=6ND-consistent driver behavior) unless
    ``freeze_tokens`` plants the "steps conversion ignored the ratio" bug (D constant across
    budgets ⇒ b ≈ 0 ⇒ the a+b gate must fire)."""
    records: list[SweepRecord] = []
    for i, c in enumerate(budgets):
        n_opt = k_n * c**a
        if c == off_law_budget:
            n_opt *= 3.0  # one off-law point: C=6ND stays consistent, the power law breaks
        for mult in (0.5, 1.0, 2.0):
            n = n_opt * mult
            tokens = 1_000_000_000 if freeze_tokens else int(c / (6.0 * n))
            bpb = 1.0 + math.log2(n / n_opt) ** 2
            records.append(_record(f"p{i}x{mult}", 4, int(n), 20, tokens, c, bpb))
    return records


def test_grid_matches_instantiated_model() -> None:
    grid = build_grid()
    assert [g.point for g in grid] == [s[0] for s in GRID_SPEC] == [f"s{i}" for i in range(1, 9)]
    assert [(g.depth, g.ratio) for g in grid] == [
        (4, 8),
        (4, 20),
        (4, 40),
        (8, 8),
        (8, 20),
        (8, 40),
        (12, 8),
        (12, 20),
    ]  # depths {4,8,12} x ratios {8,20,40} minus (12, 40)
    for g in grid:
        cfg = model_config_for_depth(g.depth, VOCAB_SIZE, CONTEXT_LENGTH)
        cfg = replace(cfg, qk_norm=True, use_sdpa=True)
        expected = sum(p.numel() for p in TransformerLM(cfg).parameters())
        assert g.n_params == expected == EXPECTED_N[g.depth]  # exact, never the ~20M table values
        assert g.tokens == g.ratio * g.n_params  # D = ratio × N
        assert g.compute == pytest.approx(6.0 * g.n_params * g.tokens, rel=1e-12)  # C = 6ND


def test_steps_for_budget_conversion() -> None:
    # steps = D / (batch × ctx), ceil so the run covers the full token budget.
    assert steps_for_budget(10 * 32 * 2048, 32, 2048) == 10  # exact division stays exact
    assert steps_for_budget(10 * 32 * 2048 + 1, 32, 2048) == 11  # one extra token ⇒ one extra step
    n4 = EXPECTED_N[4]
    assert steps_for_budget(8 * n4, 32, 2048) == math.ceil(8 * n4 / (32 * 2048))
    with pytest.raises(ValueError):
        steps_for_budget(1000, 0, 2048)


def test_select_points() -> None:
    grid = build_grid()
    assert [g.point for g in select_points(grid, "s1, s3,s8")] == ["s1", "s3", "s8"]
    with pytest.raises(ValueError, match="unknown grid points"):
        select_points(grid, "s1,s9")


def test_results_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    assert load_results(path) == []  # absent file ⇒ empty, the first run creates it
    rec1 = _record("s1", 4, EXPECTED_N[4], 8, 8 * EXPECTED_N[4], 6.0 * 8 * EXPECTED_N[4] ** 2, 0.9)
    rec2 = _record(
        "s2", 4, EXPECTED_N[4], 20, 20 * EXPECTED_N[4], 6.0 * 20 * EXPECTED_N[4] ** 2, 0.8
    )
    append_result(path, rec1)
    append_result(path, rec2)
    loaded = load_results(path)
    assert loaded == [rec1, rec2]
    # The on-disk schema is the spec'd record shape (seed included — dual-seed fits average
    # replicates of one (point, budget) on the bpb axis before min-picking).
    row = json.loads(path.read_text())[0]
    assert set(row) == {
        "point",
        "depth",
        "n_params",
        "ratio",
        "tokens",
        "compute",
        "val_loss",
        "val_bpb",
        "wall_s",
        "seed",
    }


def test_fit_recovers_planted_powerlaw() -> None:
    # k_n = sqrt(1/(6·8)) ⇒ D_opt/N_opt = 8 at every budget (a = b = 0.5 exactly).
    records = _planted_isoflop_records(k_n=math.sqrt(1.0 / 48.0), a=0.5)
    report = fit_scaling_law(records)
    assert report.n_law.exponent == pytest.approx(0.5, abs=1e-6)
    assert report.d_law.exponent == pytest.approx(0.5, abs=1e-6)
    assert report.n_law.exponent + report.d_law.exponent == pytest.approx(1.0, abs=1e-6)
    assert report.r2_n == pytest.approx(1.0, abs=1e-9)
    assert report.r2_d == pytest.approx(1.0, abs=1e-9)
    # Min-pick landed on the planted N_opt per budget, and recorded D tracks the C=6ND bridge.
    assert report.budgets == [1e16, 1e17, 1e18]
    assert len(report.n_opts) == 3
    assert report.d_opts == pytest.approx(report.d_opts_bridge, rel=1e-6)
    # Decision rule: optimal ratio 8 < 15 ⇒ recommend re-registering the d20's D.
    assert report.optimal_ratio_at_d20 == pytest.approx(8.0, rel=1e-6)
    assert report.recommend_reregister is True


def test_fit_exponent_gate_fires_on_planted_violation() -> None:
    # The bug class the gate exists for: recorded D frozen across budgets (a steps-conversion
    # or record-misalignment bug) ⇒ b ≈ 0 ⇒ a+b ≈ 0.5, far outside [0.95, 1.05].
    records = _planted_isoflop_records(k_n=math.sqrt(1.0 / 48.0), a=0.5, freeze_tokens=True)
    with pytest.raises(ValueError, match="exponent-sum gate"):
        fit_scaling_law(records)


def test_fit_r2_gate_fires_on_off_law_point() -> None:
    # One budget's argmin N planted 3x off the power law (C=6ND still consistent ⇒ the a+b
    # gate passes and it is the R² gate that must catch the broken law).
    records = _planted_isoflop_records(k_n=math.sqrt(1.0 / 48.0), a=0.5, off_law_budget=1e17)
    with pytest.raises(ValueError, match="R² gate"):
        fit_scaling_law(records)


def test_decision_rule_holds_at_ratio_20() -> None:
    # k_n = sqrt(1/(6·20)) ⇒ compute-optimal D:N = 20 ≥ 15 ⇒ the registered 9.6B stands.
    records = _planted_isoflop_records(k_n=math.sqrt(1.0 / 120.0), a=0.5)
    report = fit_scaling_law(records)
    assert report.optimal_ratio_at_d20 == pytest.approx(20.0, rel=1e-6)
    assert report.recommend_reregister is False


def test_fit_needs_two_budgets() -> None:
    records = _planted_isoflop_records(k_n=0.1, a=0.5, budgets=(1e16,))
    with pytest.raises(ValueError):
        fit_scaling_law(records)


def test_oracle_overlay_and_fit_outputs(tmp_path: Path) -> None:
    # nanochat's CORE fit at our d20's C: the pre-registered cross-check lands ≈0.195.
    assert nanochat_core_fit(D20_COMPUTE_FLOPS) == pytest.approx(0.195, abs=5e-3)

    records = _planted_isoflop_records(k_n=math.sqrt(1.0 / 48.0), a=0.5)
    report = fit_scaling_law(records)
    json_path, md_path = write_fit_outputs(report, records, tmp_path)

    fit = json.loads(json_path.read_text())
    oracle = fit["oracle"]
    assert oracle["nanochat_d20_miniseries"] == {
        "n_params": 477e6,
        "tokens": 3.82e9,
        "core": 0.1708,
    }
    assert oracle["nanochat_leaderboard_d24_ratio8"] == {"core": 0.2626, "val_bpb": 0.718}
    assert oracle["gpt2_xl"] == {"core": 0.256525}
    assert oracle["nanochat_core_fit"] == {"coeff": 3.7555, "exponent": -0.0344}
    assert oracle["nanochat_core_fit_at_d20"] == pytest.approx(0.195, abs=5e-3)
    rule = fit["decision_rule"]
    assert rule["threshold"] == 15.0
    assert rule["recommend_reregister"] is True
    assert rule["d20_registered_tokens"] == 9.6e9

    md = md_path.read_text()
    assert "RE-REGISTER" in md  # the planted ratio-8 law trips the <15 decision rule
    assert "0.1708" in md and "0.2626" in md and "0.718" in md and "0.256525" in md


# -----------------------------------------------------------------------------------------------
# v2 amendment: exact-C budget planner, depths 6/14/16, quadratic min-pick, dual-seed averaging.
# -----------------------------------------------------------------------------------------------


def test_v2_depths_instantiate_near_policy_counts() -> None:
    """Depths 6/14/16 — the v2 grid's additions — instantiate through the generic
    ``model_config_for_depth`` (d_model = 64·depth, head_dim 128; no registry extension).
    Counts are frozen from instantiation and within ±5% of the policy table's d14 ≈ 193.6M /
    d16 ≈ 268.4M."""
    for depth in (6, 14, 16):
        cfg = model_config_for_depth(depth, VOCAB_SIZE, CONTEXT_LENGTH)
        assert cfg.d_model == 64 * depth and cfg.n_layers == depth
        cfg = replace(cfg, qk_norm=True, use_sdpa=True)
        n = sum(p.numel() for p in TransformerLM(cfg).parameters())
        assert n == EXPECTED_N[depth]
    assert EXPECTED_N[14] == pytest.approx(193.6e6, rel=0.05)
    assert EXPECTED_N[16] == pytest.approx(268.4e6, rel=0.05)


def test_build_grid_v2_exact_c_planner() -> None:
    batch = 32
    grid = build_grid_v2(batch_size=batch)
    # Every budget has exactly its three planned sizes, systematically named b<budget>_d<depth>.
    assert (
        [g.point for g in grid]
        == [f"b{i}_d{d}" for i, (_, depths) in enumerate(GRID_SPEC_V2, start=1) for d in depths]
        == [
            "b1_d4",
            "b1_d6",
            "b1_d8",
            "b2_d6",
            "b2_d8",
            "b2_d12",
            "b3_d8",
            "b3_d12",
            "b3_d14",
            "b4_d8",
            "b4_d12",
            "b4_d14",
            "b5_d12",
            "b5_d14",
            "b5_d16",
        ]
    )  # regression lock on the spec'd budget × size assignment
    flat_spec = [(c, d) for c, depths in GRID_SPEC_V2 for d in depths]
    for (target_c, depth), g in zip(flat_spec, grid, strict=True):
        assert g.depth == depth and g.target_compute == target_c
        assert g.n_params == EXPECTED_N[depth]  # exact instantiated N, never table values
        # D is a whole number of optimizer steps (batch × ctx tokens each), nearest-step
        # rounding of the ideal D = C_target/(6N).
        assert g.tokens == g.steps * batch * CONTEXT_LENGTH
        assert g.steps == max(1, round(target_c / (6.0 * g.n_params) / (batch * CONTEXT_LENGTH)))
        # The effective C is RECOMPUTED from the rounded D — exactly self-consistent...
        assert g.compute == 6.0 * g.n_params * g.tokens
        # ...and coincides with the target within one step's worth of FLOPs (nearest-step
        # rounding actually keeps it within half a step; the gate is the full step).
        assert abs(g.compute - target_c) <= 6.0 * g.n_params * batch * CONTEXT_LENGTH
        # The plan-frozen steps are what the run path must use; another batch re-plans.
        assert g.planned_steps(batch, CONTEXT_LENGTH) == g.steps
        with pytest.raises(ValueError, match="planned for batch"):
            g.planned_steps(2 * batch, CONTEXT_LENGTH)
    # Cross-size coincidence within a budget: the per-size effective C's spread is under one
    # step's worth of the largest size's C.
    for target_c, _ in GRID_SPEC_V2:
        at_budget = [g for g in grid if g.target_compute == target_c]
        assert len(at_budget) == 3
        spread = max(g.compute for g in at_budget) - min(g.compute for g in at_budget)
        one_step_c = 6.0 * max(g.n_params for g in at_budget) * batch * CONTEXT_LENGTH
        assert spread <= one_step_c
    assert set(grid[0].to_dict()) == {
        "point",
        "depth",
        "n_params",
        "target_compute",
        "tokens",
        "steps",
        "batch_size",
        "compute",
    }


def test_average_seed_replicates() -> None:
    b1d4_s0 = _record("b1_d4", 4, 100, 20.0, 2000, 1.2e6, 1.00, seed=0)
    b1d4_s1 = _record("b1_d4", 4, 100, 20.0, 2000, 1.2e6, 1.10, seed=1)
    solo = _record("b1_d8", 8, 200, 20.0, 4000, 4.8e6, 0.95, seed=1)
    # A pre-amendment record without a seed field is treated as seed 0 and left as-is.
    legacy = cast(SweepRecord, {k: v for k, v in solo.items() if k != "seed"})
    avg = average_seed_replicates([b1d4_s0, b1d4_s1, legacy])
    assert len(avg) == 2  # the (b1_d4, budget) seeds collapse; the singleton stands
    merged = next(r for r in avg if r["point"] == "b1_d4")
    assert merged["val_bpb"] == pytest.approx(1.05)  # the seed MEAN on the loss axis
    assert merged["n_params"] == 100 and merged["tokens"] == 2000  # replicates share scaffolding
    assert next(r for r in avg if r["point"] == "b1_d8") == legacy


def test_fit_averages_seeds_before_min_pick() -> None:
    # Two budgets x three sizes x two seeds. Each single seed's raw argmin is off the planted
    # per-budget winner; the seed-mean bpb picks it at both budgets.
    ns = (1e7, 2e7, 4e7)
    bpb = {  # (budget, seed) -> per-size bpb
        (1e16, 0): (1.00, 1.03, 1.05),  # seed 0 alone would argmin N1
        (1e16, 1): (1.06, 1.01, 1.05),  # seed 1 alone would argmin N2
        (1e17, 0): (1.00, 1.02, 1.04),  # seed 0 alone would argmin N1
        (1e17, 1): (1.08, 1.06, 1.00),  # seed 1 alone would argmin N3
    }  # seed means: [1.03, 1.02, 1.05] -> N2 at 1e16; [1.04, 1.04, 1.02] -> N3 at 1e17
    records = [
        _record(
            f"b{bi}_n{ni}",
            4,
            int(n),
            20.0,
            int(c / (6.0 * n)),
            c,
            bpb[(c, seed)][ni],
            seed=seed,
        )
        for bi, c in enumerate((1e16, 1e17), start=1)
        for seed in (0, 1)
        for ni, n in enumerate(ns)
    ]
    report = fit_scaling_law(records, smooth=False)
    assert report.n_opts == [2e7, 4e7]  # the seed-mean pick at BOTH budgets
    # Proof the averaging did it: seed 0's records alone argmin to N1 at both budgets.
    runs0 = [
        {
            "compute_budget": float(r["compute"]),
            "final_loss": r["val_bpb"],
            "parameters": float(r["n_params"]),
            "tokens": float(r["tokens"]),
        }
        for r in records
        if r["seed"] == 0
    ]
    assert [n for _, n in isoflop_min(runs0)] == [1e7, 1e7]


def test_fit_v2_records_group_by_target_budget() -> None:
    """v2 records: the sizes of one budget share ``target_compute`` while their effective C
    differs by the step rounding. The fit must group budgets on target_compute — keyed on
    effective C, every size would become its own one-point budget and the smooth pick would
    degenerate to the raw fallback."""
    k_n = math.sqrt(1.0 / 48.0)  # vertex D:N = 8 — the same planted law as the v1 tests
    budgets = (1e17, 4e17, 8.5e17)
    records: list[SweepRecord] = []
    for bi, c in enumerate(budgets, start=1):
        n_opt = k_n * c**0.5
        for di, mult in enumerate((0.5, 1.0, 2.0)):
            n = int(n_opt * mult)
            tokens = int(c / (6.0 * n)) - 7 * di  # per-size step-rounding jitter: C_eff ≠ C_k
            bpb = 1.0 + math.log2(n / n_opt) ** 2
            records.append(
                _record(
                    f"b{bi}_m{di}",
                    4,
                    n,
                    tokens / n,
                    tokens,
                    6.0 * n * tokens,
                    bpb,
                    target_compute=c,
                )
            )
    effs_at_b1 = {r["compute"] for r in records if r.get("target_compute") == 1e17}
    assert len(effs_at_b1) == 3  # the jitter really does make effective C distinct per size

    report = fit_scaling_law(records)  # smooth by default
    assert report.budgets == list(budgets)  # grouped on target C, not the jittered effective C
    assert report.clamped_budgets == []  # the planted vertex is interior at every budget
    assert report.smooth is True
    assert report.n_law.exponent == pytest.approx(0.5, abs=1e-6)
    assert report.n_law.exponent + report.d_law.exponent == pytest.approx(1.0, abs=1e-6)
    assert report.r2_n == pytest.approx(1.0, abs=1e-9)
    assert report.to_dict()["min_pick"] == {"smooth": True, "clamped_budgets": []}
