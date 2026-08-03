"""Tests for the S3.5 joint scaling-law fitter (Chinchilla Approach 3 + the four §1.4 gates).

The strategy is planted-law recovery: generate records from a KNOWN L(N, D) with tiny seeded
noise, then check the fitter gets the parameters back, shrugs off an outlier (Huber's whole
reason for existing — this repo got bitten by one), passes all four gates on clean data, and
fails loudly when the data stops being a clean law. The clean fit is a module-scoped fixture:
one 48-start multi-start fit is ~2 s and most tests can share it.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from scratch_llm.scaling.joint_fit import (  # noqa: E402
    D20_COMPUTE_FLOPS,
    JointLaw,
    bootstrap_prediction_interval,
    evaluate_gates,
    fit_joint_law,
    loo_ratio_swing,
    residual_trend,
)
from scratch_llm.scaling.s3_sweep import SweepRecord  # noqa: E402

# The planted law, in the bpb regime our sweep actually lives in (~0.6–1.0 bpb).
TRUE_E, TRUE_A, TRUE_B, TRUE_ALPHA, TRUE_BETA = 0.6, 400.0, 800.0, 0.45, 0.50
# The d20 prediction point (480.4M params × 9.6B registered tokens).
D20_N_PARAMS, D20_TOKENS = 480.4e6, 9.6e9


def true_bpb(n_params: float, tokens: float) -> float:
    return TRUE_E + TRUE_A * n_params**-TRUE_ALPHA + TRUE_B * tokens**-TRUE_BETA


def make_records(noise: float = 1e-3, seed: int = 0) -> list[SweepRecord]:
    """12 records on a 4-size × 3-ratio grid with tiny log-space noise. The geometry is the
    §1.2 prescription: N spans 40× (10M→400M) and C spans ~8000× with the top end at the
    d20's C — a narrow span leaves A and α trading off and no tolerance can be met."""
    rng = np.random.default_rng(seed)
    sizes = [(4, 10_000_000), (8, 50_000_000), (12, 135_000_000), (16, 400_000_000)]
    ratios = [8.0, 20.0, 40.0]
    records: list[SweepRecord] = []
    for depth, n_params in sizes:
        for ratio in ratios:
            tokens = int(ratio * n_params)
            bpb = true_bpb(n_params, tokens) * float(np.exp(rng.normal(0.0, noise)))
            records.append(
                SweepRecord(
                    point=f"d{depth}r{int(ratio)}",
                    depth=depth,
                    n_params=n_params,
                    ratio=ratio,
                    tokens=tokens,
                    compute=6.0 * n_params * tokens,
                    val_loss=bpb,
                    val_bpb=bpb,
                    wall_s=1.0,
                    seed=0,
                )
            )
    return records


@pytest.fixture(scope="module")
def clean_records() -> list[SweepRecord]:
    return make_records()


@pytest.fixture(scope="module")
def clean_law(clean_records: list[SweepRecord]) -> JointLaw:
    return fit_joint_law(clean_records)


def test_recovers_planted_law(clean_law: JointLaw) -> None:
    for fitted, truth, name in [
        (clean_law.E, TRUE_E, "E"),
        (clean_law.A, TRUE_A, "A"),
        (clean_law.B, TRUE_B, "B"),
        (clean_law.alpha, TRUE_ALPHA, "alpha"),
        (clean_law.beta, TRUE_BETA, "beta"),
    ]:
        assert abs(fitted - truth) / truth < 0.05, (
            f"{name}: fitted {fitted:.4g} vs true {truth:.4g}"
        )
    # Multi-start degeneracy diagnostic: a healthy fit has many starts in the best basin.
    assert clean_law.n_starts == 48
    assert clean_law.n_converged > 1


def test_huber_is_outlier_robust_vs_least_squares(clean_records: list[SweepRecord]) -> None:
    # One bad run: bpb 10% high (+0.095 in log space — way past Huber's δ = 1e-3 knee).
    records = [
        SweepRecord({**r, "val_bpb": r["val_bpb"] * 1.10, "val_loss": r["val_loss"] * 1.10})
        if i == 5
        else r
        for i, r in enumerate(clean_records)
    ]
    huber = fit_joint_law(records)  # δ = 1e-3
    lsq = fit_joint_law(records, huber_delta=1e6)  # δ → ∞ IS quadratic loss

    probe_n, probe_d = 3.0e8, 6.0e9
    truth = true_bpb(probe_n, probe_d)
    err_huber = abs(huber.predict(probe_n, probe_d) - truth) / truth
    err_lsq = abs(lsq.predict(probe_n, probe_d) - truth) / truth
    assert err_huber < 0.02, f"Huber fit degraded: {err_huber:.4f} relative error"
    assert err_huber < err_lsq, (
        f"Huber ({err_huber:.4f}) should degrade less than LSQ ({err_lsq:.4f})"
    )


def test_predict_optimal_exponent_sanity(clean_law: JointLaw) -> None:
    # With α = β = 0.5 the closed form gives N_opt ∝ C^(β/(α+β)) = C^0.5: 4× C ⇒ 2× N_opt.
    law = JointLaw(
        A=500.0, B=500.0, E=0.6, alpha=0.5, beta=0.5, objective=0.0, n_starts=1, n_converged=1
    )
    n1, d1, l1 = law.predict_optimal(1.0e18)
    n2, _, _ = law.predict_optimal(4.0e18)
    assert abs(n2 / n1 - 2.0) < 1e-9
    # The bridge identity holds: C = 6·N_opt·D_opt, and L_pred is the law at that point.
    assert abs(6.0 * n1 * d1 - 1.0e18) < 1e6
    assert l1 == pytest.approx(law.predict(n1, d1))

    # And the FITTED law keeps the sane exponent end to end.
    m1, _, _ = clean_law.predict_optimal(1.0e18)
    m2, _, _ = clean_law.predict_optimal(4.0e18)
    expected = 4.0 ** (clean_law.beta / (clean_law.alpha + clean_law.beta))
    assert abs(m2 / m1 - expected) / expected < 1e-6


def test_bootstrap_interval_deterministic_and_tight_on_clean_data(
    clean_records: list[SweepRecord], clean_law: JointLaw
) -> None:
    iv1 = bootstrap_prediction_interval(
        clean_records, D20_N_PARAMS, D20_TOKENS, n_boot=100, seed=0, parent=clean_law
    )
    iv2 = bootstrap_prediction_interval(
        clean_records, D20_N_PARAMS, D20_TOKENS, n_boot=100, seed=0, parent=clean_law
    )
    assert (iv1.lo, iv1.hi) == (iv2.lo, iv2.hi)  # seeded ⇒ reproducible
    assert iv1.n_failed_fits == 0
    assert iv1.half_width < 0.02
    assert not iv1.fit_failure_red_flag


def test_residual_trend_catches_structured_misfit(
    clean_records: list[SweepRecord], clean_law: JointLaw
) -> None:
    rho, p_value = residual_trend(clean_records, clean_law)
    assert abs(rho) < 0.7 or p_value >= 0.05  # clean data: no strong significant trend

    # Same records plus a drift term in log C the E + A/N^α + B/D^β form cannot express —
    # the residual-vs-compute trend must come back strong AND significant.
    bent: list[SweepRecord] = []
    mid = np.log(6.0 * 50e6 * 20 * 50e6)
    for r in clean_records:
        drift = 0.10 * (np.log(r["compute"]) - mid)
        bpb = float(r["val_bpb"]) * float(np.exp(drift))
        bent.append(SweepRecord({**r, "val_bpb": bpb, "val_loss": bpb}))
    bent_law = fit_joint_law(bent)
    rho, p_value = residual_trend(bent, bent_law)
    assert abs(rho) >= 0.7 and p_value < 0.05


def test_all_gates_pass_on_clean_data(
    clean_records: list[SweepRecord], clean_law: JointLaw
) -> None:
    # n_boot=100 keeps the suite under 30 s; the module default is the registered 200.
    report = evaluate_gates(
        clean_records, clean_law, n_params=D20_N_PARAMS, tokens=D20_TOKENS, n_boot=100
    )
    by_name = {g.name: g for g in report.gates}
    assert by_name["r2_log_l"].value >= 0.98
    assert by_name["bootstrap_pi_half_width"].value <= 0.02
    assert by_name["loo_ratio_swing"].value <= 2.0
    assert by_name["residual_trend"].passed
    assert report.all_passed


def test_gates_fail_with_a_bad_outlier(clean_records: list[SweepRecord]) -> None:
    # A catastrophic point (+30% bpb): even the Huber fit can't explain it away, and at
    # least one of the four gates must fire — that is the entire point of the gate bundle.
    records = [
        SweepRecord({**r, "val_bpb": r["val_bpb"] * 1.30, "val_loss": r["val_loss"] * 1.30})
        if i == 5
        else r
        for i, r in enumerate(clean_records)
    ]
    law = fit_joint_law(records)
    report = evaluate_gates(records, law, n_params=D20_N_PARAMS, tokens=D20_TOKENS, n_boot=50)
    assert not report.all_passed
    assert not report.gates[0].passed  # the R² gate is the one that fires here


def test_loo_swing_near_one_on_clean_data(
    clean_records: list[SweepRecord], clean_law: JointLaw
) -> None:
    swing = loo_ratio_swing(clean_records, D20_COMPUTE_FLOPS, parent=clean_law)
    assert 1.0 <= swing < 1.5
