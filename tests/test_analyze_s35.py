"""`scripts/analyze_s35.py` — synthetic-tree tests for the S3.5 analysis driver.

Builds a fake artifacts tree in tmp_path (P5 LR sweep with a planted tie, the batch
de-confound pair, and an 11-point fit set generated from a planted Chinchilla law) and
exercises the driver's importable logic functions plus the graceful-missing-data paths.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _mod():
    """Import the CLI module (scripts/ is not a package)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import analyze_s35
    finally:
        sys.path.pop(0)
    return analyze_s35


# Planted law: L(N, D) = E + A·N^(−α) + B·D^(−β) — the fit should recover roughly these.
PLANTED = {"E": 0.8, "A": 400.0, "alpha": 0.30, "B": 500.0, "beta": 0.30}
N_BY_DEPTH = {4: 19_991_808, 8: 59_255_296, 12: 135_288_576, 14: 193_600_000}
WINNER_LR = 3e-3  # the planted tie-break winner is ×1.0 of the 3e-3 base


def _planted_bpb(n_params: float, tokens: float, noise: float = 0.0) -> float:
    return (
        PLANTED["E"]
        + PLANTED["A"] * n_params ** (-PLANTED["alpha"])
        + PLANTED["B"] * tokens ** (-PLANTED["beta"])
        + noise
    )


def _record(point: str, depth: int, ratio: float, bpb: float, lr: float | None) -> dict:
    n = N_BY_DEPTH[depth]
    tokens = int(ratio * n)
    rec = {
        "point": point,
        "depth": depth,
        "n_params": n,
        "ratio": ratio,
        "tokens": tokens,
        "compute": 6.0 * n * tokens,
        "val_loss": bpb * 0.7,
        "val_bpb": bpb,
        "wall_s": 60.0,
        "seed": 0,
    }
    if lr is not None:
        rec["lr"] = lr
    return rec


def _write_run(root: Path, rel: str, records: list[dict]) -> None:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(records))


def _build_tree(root: Path) -> None:
    """The full fake tree: P5 sweep with a planted tie, the pair, and the fit set. The sweep
    bpbs are the planted law at d12@r4 plus a bowl offset, so the winner point doubles as a
    fit-set point without any overwrite."""
    n12, n8 = N_BY_DEPTH[12], N_BY_DEPTH[8]
    # P5 d12 sweep: min bpb at ×1.4, but ×1.0 sits inside the 0.003 tie band → the
    # tie-break must pick ×1.0 (the LOWER LR). Dir names match the pod layout literally.
    law_d12r4 = _planted_bpb(n12, 4 * n12)
    offsets = {0.5: 0.020, 0.7: 0.010, 1.0: 0.002, 1.4: 0.000, 2.0: 0.015}
    for dirname, mult in mod_dirs().items():
        _write_run(
            root,
            f"p5/{dirname}",
            [_record("p5_d12r4", 12, 4, law_d12r4 + offsets[mult], WINNER_LR * mult)],
        )
    # Width probe at {0.7, 1.0, 1.4}×winner, argmin planted at ×1.0 (no drift); ×1.0 is the
    # planted-law d8@r4 point and doubles as a fit-set point.
    law_d8r4 = _planted_bpb(n8, 4 * n8)
    for mult, off in {0.7: 0.010, 1.0: 0.000, 1.4: 0.004}.items():
        _write_run(
            root,
            f"p5/width_lr{mult}",
            [_record("p5_d8r4", 8, 4, law_d8r4 + off, WINNER_LR * mult)],
        )
    _write_run(root, "s3_rerun_pair/batch4", [_record("s7", 12, 8, 0.950, WINNER_LR)])
    _write_run(root, "s3_rerun_pair/batch8", [_record("s7", 12, 8, 0.945, WINNER_LR)])
    _write_run(root, "s3_scaling_sweep", [_record("s7", 12, 8, 0.9402, None)])

    rng = np.random.default_rng(0)
    ladder = [(1, 4, 8), (2, 4, 20), (3, 4, 40), (4, 8, 8), (5, 8, 20), (6, 8, 40)]
    for i, depth, ratio in ladder:
        n = N_BY_DEPTH[depth]
        bpb = _planted_bpb(n, ratio * n, float(rng.normal(0, 5e-4)))
        _write_run(root, f"s35/ladder_s{i}", [_record(f"s{i}", depth, ratio, bpb, WINNER_LR)])
    for rel, point, depth, ratio in [
        ("p5/confirm_r8", "confirm_r8", 12, 8),
        ("s35/d12_r20", "d12_r20", 12, 20),
        ("s35/d14_r8", "d14_r8", 14, 8),
    ]:
        n = N_BY_DEPTH[depth]
        bpb = _planted_bpb(n, ratio * n, float(rng.normal(0, 5e-4)))
        _write_run(root, rel, [_record(point, depth, ratio, bpb, WINNER_LR)])


def mod_dirs() -> dict[str, float]:
    """The pod's literal P5 sweep dir names (NOT f\"{mult:g}\" — 1.0 must render \"lr1.0\")."""
    return {"lr0.5": 0.5, "lr0.7": 0.7, "lr1.0": 1.0, "lr1.4": 1.4, "lr2.0": 2.0}


def test_p5_winner_uses_lower_lr_tiebreak(tmp_path: Path) -> None:
    mod = _mod()
    _build_tree(tmp_path)
    out = mod.analyze_p5(tmp_path, tmp_path / "s35_analysis")
    assert out is not None
    # Min bpb sits at ×1.4 but ×1.0 is within the 0.003 tie band → lower LR wins.
    assert out["winner_mult"] == pytest.approx(1.0)
    assert out["winner_lr"] == pytest.approx(WINNER_LR)
    assert (tmp_path / "s35_analysis" / "p5_lr_bowl.png").exists()


def test_pair_verdict_flags_confound(tmp_path: Path) -> None:
    mod = _mod()
    _build_tree(tmp_path)
    out = mod.analyze_pair(tmp_path)
    assert out is not None
    assert out["delta"] == pytest.approx(-0.005)
    assert out["batch8"] < out["batch4"]


def test_fit_recovers_planted_law_and_emits_artifacts(tmp_path: Path) -> None:
    mod = _mod()
    _build_tree(tmp_path)
    out_dir = tmp_path / "s35_analysis"
    payload = mod.analyze_fit(tmp_path, out_dir, WINNER_LR, 1.0, n_boot=30, seed=0)
    assert payload is not None
    params = payload["params"]
    assert params["alpha"] == pytest.approx(PLANTED["alpha"], abs=0.1)
    assert params["beta"] == pytest.approx(PLANTED["beta"], abs=0.1)
    gates = payload["gates"]
    assert set(gates["gates"]) == {
        "r2_log_l",
        "bootstrap_pi_half_width",
        "loo_ratio_swing",
        "residual_trend",
    }
    assert payload["recipe_v1_violations"] == []
    d20 = payload["d20"]
    assert d20["predicted_bpb"] == pytest.approx(_planted_bpb(480.4e6, 9.6e9), abs=0.05)
    assert (out_dir / "s35_fit.json").exists()
    for png in ("s35_fit.png", "s35_residuals.png", "s35_optimals.png"):
        assert (out_dir / png).exists()
    on_disk = json.loads((out_dir / "s35_fit.json").read_text())
    assert on_disk["winner_lr"] == pytest.approx(WINNER_LR)
    assert len(on_disk["fit_set"]) == len(payload["fit_set"])


def test_recipe_violation_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    mod = _mod()
    _build_tree(tmp_path)
    # Break one fit point's LR — the check must name it.
    path = tmp_path / "s35" / "ladder_s3" / "results.json"
    recs = json.loads(path.read_text())
    recs[0]["lr"] = 6e-3
    path.write_text(json.dumps(recs))
    payload = mod.analyze_fit(
        tmp_path, tmp_path / "s35_analysis", WINNER_LR, 1.0, n_boot=10, seed=0
    )
    assert payload is not None
    assert any("s3" in v for v in payload["recipe_v1_violations"])
    assert "recipe-v1 VIOLATIONS" in capsys.readouterr().out


def test_missing_tree_never_crashes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    mod = _mod()
    assert mod.analyze_p5(tmp_path, tmp_path / "out") is None
    assert mod.analyze_pair(tmp_path) is None
    assert mod.analyze_fit(tmp_path, tmp_path / "out", WINNER_LR, 1.0, n_boot=5) is None
    assert "WARNING" in capsys.readouterr().out


def test_cli_all_autopicks_winner(tmp_path: Path) -> None:
    mod = _mod()
    _build_tree(tmp_path)
    mod.main(["all", "--artifacts-root", str(tmp_path)])
    assert (tmp_path / "s35_analysis" / "s35_fit.json").exists()
