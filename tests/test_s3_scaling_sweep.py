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

import pytest

from scratch_llm.model import TransformerLM
from scratch_llm.scaling.s3_sweep import (
    CONTEXT_LENGTH,
    D20_COMPUTE_FLOPS,
    GRID_SPEC,
    VOCAB_SIZE,
    SweepRecord,
    append_result,
    build_grid,
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
EXPECTED_N = {4: 19_991_808, 8: 59_255_296, 12: 135_288_576}


def _record(
    point: str,
    depth: int,
    n_params: int,
    ratio: int,
    tokens: int,
    compute: float,
    val_bpb: float,
) -> SweepRecord:
    return SweepRecord(
        point=point,
        depth=depth,
        n_params=n_params,
        ratio=ratio,
        tokens=tokens,
        compute=compute,
        val_loss=2.0,  # nats/token — carried in the record, not the fit axis
        val_bpb=val_bpb,
        wall_s=60.0,
    )


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
    # The on-disk schema is the spec'd record shape.
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
