"""IsoFLOP/Chinchilla fitter invariants on the real course data + planted synthetic laws.

The load-bearing claims, each falsifiable: 9 budgets yield exactly 9 (C, N_opt) pairs; an exact
power law is recovered to fp precision; the fit lives in log-log space (a raw-space fit on the
same heteroscedastic data lands provably elsewhere); C=6ND round-trips; the a+b≈1 gate fires on
corrupted data; and the course-JSON exponents/extrapolations are locked as a regression with the
predicted N_opt/D_opt at 1e23 and 1e24 FLOPs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scratch_llm.scaling.isoflop import (
    PowerLaw,
    check_exponent_sum,
    compute_from_params_tokens,
    fit_powerlaw,
    isoflop_min,
    nonembed_params,
    propose_shape,
    tokens_from_compute_params,
)
from scratch_llm.utils.seeding import seed_everything

FIXTURE = Path(__file__).parent / "fixtures" / "isoflops_curves.json"


def _course_runs() -> list[dict[str, float]]:
    return json.loads(FIXTURE.read_text())


def _raw_space_powerlaw_exponent(xs: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares power fit in RAW space (what fit_powerlaw must NOT do): grid over the
    exponent with the closed-form optimal coefficient k(p) = <y,x^p>/<x^p,x^p> per candidate."""
    best_sse, best_p = np.inf, np.nan
    for p in np.arange(0.0, 1.5, 1e-4):
        xp = xs**p
        k = float(ys @ xp) / float(xp @ xp)
        sse = float(((ys - k * xp) ** 2).sum())
        if sse < best_sse:
            best_sse, best_p = sse, float(p)
    return best_p


def test_isoflop_min_nine_budgets_on_course_data() -> None:
    runs = _course_runs()
    pairs = isoflop_min(runs)
    assert len(pairs) == 9  # 72 runs across exactly 9 distinct compute budgets
    budgets = [c for c, _ in pairs]
    assert budgets == sorted(budgets)
    assert budgets[0] == pytest.approx(6e18) and budgets[-1] == pytest.approx(3e21)
    # Spot-check against a manual argmin at the smallest budget.
    at_6e18 = [r for r in runs if r["compute_budget"] == 6e18]
    manual = min(at_6e18, key=lambda r: r["final_loss"])["parameters"]
    assert pairs[0][1] == manual == 762093419


def test_fit_powerlaw_recovers_exact_law() -> None:
    xs = np.logspace(0, 10, 50)
    law_true = PowerLaw(exponent=0.42, coeff=3.7)
    ys = law_true.coeff * xs**law_true.exponent
    law = fit_powerlaw(xs.tolist(), ys.tolist())
    assert law.exponent == pytest.approx(0.42, abs=1e-10)
    assert law.coeff == pytest.approx(3.7, rel=1e-10)
    assert law.predict(1e12) == pytest.approx(3.7 * 1e12**0.42, rel=1e-9)


def test_fit_is_log_space_not_raw_space() -> None:
    """Heteroscedastic discrimination: plant y = 2·x^0.5 with ONE downward multiplicative outlier
    at the largest x. In log space every point carries equal weight, so the fit barely moves; in
    raw space the largest point dominates the residual and drags the exponent far off. If
    fit_powerlaw ever regresses to a raw-space fit, this test fails."""
    seed_everything(1234)
    p_true = 0.5
    xs = np.logspace(0, 6, 20)
    ys = 2.0 * xs**p_true
    ys[-1] /= 3.0

    log_exp = fit_powerlaw(xs.tolist(), ys.tolist()).exponent
    raw_exp = _raw_space_powerlaw_exponent(xs, ys)

    log_gap = abs(log_exp - p_true)
    raw_gap = abs(raw_exp - p_true)
    assert log_gap < 0.03  # one outlier in 20 equal-weight points barely moves the log fit
    assert raw_gap > 0.15  # the raw fit chases the dominant point (lands near 0.31)
    assert raw_gap > 5 * log_gap
    assert abs(raw_exp - log_exp) > 0.1  # the two estimators are provably different fits


def test_fit_powerlaw_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        fit_powerlaw([1.0], [2.0])
    with pytest.raises(ValueError):
        fit_powerlaw([1.0, 2.0], [3.0, -1.0])


def test_compute_bridge_round_trip() -> None:
    n_params, n_tokens = 7.0e10, 2.4e11
    compute = compute_from_params_tokens(n_params, n_tokens)
    assert compute == pytest.approx(6.0 * n_params * n_tokens, rel=1e-12)
    assert tokens_from_compute_params(compute, n_params) == pytest.approx(n_tokens, rel=1e-12)
    assert compute_from_params_tokens(
        n_params, tokens_from_compute_params(compute, n_params)
    ) == pytest.approx(compute, rel=1e-12)


def test_nonembed_params_and_inverse() -> None:
    assert nonembed_params(24, 2048) == 12 * 24 * 2048**2
    for target in (1e8, 7.0e10, 2.1e11):
        n_layer, d_model = propose_shape(target)
        # Round-trip: the proposed shape realizes the target N to within rounding error.
        assert nonembed_params(n_layer, d_model) == pytest.approx(target, rel=0.02)
        # The stated assumption holds: d_model/n_layer tracks the default 128 aspect ratio.
        assert 0.6 < (d_model / n_layer) / 128.0 < 1.5
    with pytest.raises(ValueError):
        propose_shape(-1.0)


def test_exponent_sum_gate_fires_on_corrupted_data() -> None:
    check_exponent_sum(0.469, 0.531)  # clean course-data exponents pass
    with pytest.raises(ValueError, match="exponent-sum gate"):
        check_exponent_sum(0.469, 0.62)

    # Corruption of the bug class the gate exists for: D derived from MISALIGNED rows
    # (each budget paired with the wrong N_opt). Both fits look plausible alone; the sum screams.
    pairs = isoflop_min(_course_runs())
    budgets = np.array([c for c, _ in pairs])
    n_opts = np.array([n for _, n in pairs])
    a = fit_powerlaw(budgets.tolist(), n_opts.tolist()).exponent
    d_corrupt = budgets / (6.0 * n_opts[::-1])
    b_corrupt = fit_powerlaw(budgets.tolist(), d_corrupt.tolist()).exponent
    with pytest.raises(ValueError, match="exponent-sum gate"):
        check_exponent_sum(a, b_corrupt)


def test_course_json_regression_locked() -> None:
    """The full pipeline on the vendored course JSON, with exponents and 1e23/1e24 extrapolations
    locked. Predict-before-run (pre-registered): a ≈ b ≈ 0.5 and a+b ∈ [0.95, 1.05]."""
    pairs = isoflop_min(_course_runs())
    budgets = [c for c, _ in pairs]
    n_opts = [n for _, n in pairs]
    n_law = fit_powerlaw(budgets, n_opts)
    d_opts = [tokens_from_compute_params(c, n) for c, n in pairs]
    d_law = fit_powerlaw(budgets, d_opts)
    a, b = n_law.exponent, d_law.exponent

    # Pre-registered predictions.
    assert 0.4 < a < 0.6 and 0.4 < b < 0.6
    assert 0.95 <= a + b <= 1.05
    check_exponent_sum(a, b)

    # Regression locks (measured once on the vendored fixture, then frozen).
    assert a == pytest.approx(0.46868, abs=5e-4)
    assert b == pytest.approx(0.53132, abs=5e-4)
    assert a + b == pytest.approx(1.0, abs=1e-9)  # exact: D is derived from the same (C, N) rows

    assert n_law.predict(1e23) == pytest.approx(7.005e10, rel=1e-3)
    assert d_law.predict(1e23) == pytest.approx(2.379e11, rel=1e-3)
    assert n_law.predict(1e24) == pytest.approx(2.061e11, rel=1e-3)
    assert d_law.predict(1e24) == pytest.approx(8.086e11, rel=1e-3)
    # Internal consistency: the predicted (N, D) at each target burns the target compute.
    for target in (1e23, 1e24):
        implied = compute_from_params_tokens(n_law.predict(target), d_law.predict(target))
        assert implied == pytest.approx(target, rel=0.02)
