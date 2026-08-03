"""S3.5 joint scaling-law fitter — Chinchilla Approach 3 with the replication fixes (§1.1/§1.4).

The S3 fit failed honestly: per-budget min-picking (Approach 2) on a grid never built for it
produced the D-zigzag, the bare R² gate fired, and the lab bought 12 GPU-h of measurements and
zero law. This module is the S3.5 replacement estimator, specified in
``docs/FRONTIER_2026_SCALING_PROGRAM.md`` §1.1:

* **Joint parametric fit on ALL points** — ``L(N, D) = E + A·N^(−α) + B·D^(−β)`` (Hoffmann
  et al. 2022, Approach 3), no per-budget selection step that can zigzag. The fit axis is
  ``val_bpb`` (tokenizer-invariant), N = ``n_params``, D = the RECORDED ``tokens`` — never the
  ``C/(6N)`` bridge, which would make a+b ≡ 1 identically and mute driver bugs.
* **Besiroglu-et-al. replication corrections** (arXiv:2404.10102 — the paper that made
  Chinchilla's Method 3 reproducible): Huber loss (δ = 1e-3) on LOG-space residuals — one
  outlier run must not drag five parameters, and this lab has already been bitten by one —
  plus **multi-start L-BFGS-B** from a grid of initializations, because the objective has
  near-degenerate basins and a single start silently lands wherever it started. The objective
  is the MEAN Huber (not Besiroglu's sum): identical argmin, and the 1e-6 convergence
  comparison across starts stays interpretable at any grid size.
* **UQ is part of the fit, not an afterthought** (§1.4): nonparametric bootstrap over runs for
  a prediction interval at the target point, leave-one-out swing of the compute-optimal D:N
  ratio, and a Spearman residual-trend test — a high R² with structured residuals is still a
  rejected fit (R² alone blessed the failed S3 fit; that is exactly why gates ii–iv exist).

Acceptance gates (§1.4, pre-registered): (i) R² ≥ 0.98 in log-L space, retained for continuity
but necessary-not-sufficient; (ii) bootstrap 90% PI half-width ≤ 0.02 bpb at the d20 point;
(iii) LOO swing of D_opt/N_opt at the d20's C ≤ 2× (S3's drop-s7 swing was 2.9×); (iv) no
strong AND significant monotone residual trend vs log compute.

scipy is an optional dependency (the ``scaling`` extra), so every scipy import is lazy,
inside the function that needs it — the module itself stays numpy-only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from scratch_llm.scaling.isoflop import FLOPS_PER_PARAM_TOKEN

if TYPE_CHECKING:
    from scratch_llm.scaling.s3_sweep import SweepRecord

# The d20 the law extrapolates to: 480.4M params × 9.6B registered tokens ≈ 2.77e19 FLOPs.
# Duplicated from s3_sweep.py on purpose: importing s3_sweep pulls in the torch training
# stack, and this module stays numpy-only so the fitter runs anywhere.
D20_COMPUTE_FLOPS = 2.77e19

# The estimator's knobs, pinned by FRONTIER_2026_SCALING_PROGRAM.md §1.1.
HUBER_DELTA = 1e-3  # Huber transition on log-L residuals — outlier-robust, nearly LSQ inlier
MULTISTART_ALPHAS = (0.2, 0.4, 0.6, 0.8)  # exponent init grid (Besiroglu et al.)
MULTISTART_LOG_E = (-2.0, -1.0, 0.0)  # irreducible-loss init grid, e = ln E
CONVERGENCE_TOL = 1e-6  # starts within this objective of the best count as the same basin

# The four pre-registered acceptance gates (§1.4).
R2_THRESHOLD = 0.98  # (i) log-L R² — necessary, not sufficient
PI_HALF_WIDTH_THRESHOLD = 0.02  # (ii) bpb, 90% bootstrap PI at the d20 point
LOO_SWING_THRESHOLD = 2.0  # (iii) max/min D_opt:N_opt ratio dropping one point at a time
RESIDUAL_RHO_THRESHOLD = 0.7  # (iv) |Spearman rho| of residuals vs log C ...
RESIDUAL_P_THRESHOLD = 0.05  # ... must be < 0.7 OR insignificant at this level to pass
BOOT_FAILURE_FRACTION = 0.2  # >20% failed bootstrap refits is itself a red flag


def _arrays(records: Sequence[SweepRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(log N, log D, log L) from sweep records — fit axis val_bpb, D is the recorded tokens."""
    ln_n = np.log(np.asarray([float(r["n_params"]) for r in records], dtype=np.float64))
    ln_d = np.log(np.asarray([float(r["tokens"]) for r in records], dtype=np.float64))
    ln_l = np.log(np.asarray([float(r["val_bpb"]) for r in records], dtype=np.float64))
    return ln_n, ln_d, ln_l


def _huber(r: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Huber value and derivative: quadratic for |r| ≤ δ, linear beyond (outlier-robust)."""
    a = np.abs(r)
    quad = a <= delta
    value = np.where(quad, 0.5 * r * r, delta * (a - 0.5 * delta))
    deriv = np.where(quad, r, delta * np.sign(r))
    return value, deriv


def _objective(
    theta: np.ndarray, ln_n: np.ndarray, ln_d: np.ndarray, ln_l: np.ndarray, delta: float
) -> tuple[float, np.ndarray]:
    """Mean Huber of log-space residuals, with analytic gradient for L-BFGS-B.

    θ = (a, b, e, α, β) with a = ln A, b = ln B, e = ln E. The prediction is
    ``log L = LSE(a − α·ln N, b − β·ln D, e)`` — the log of E + A·N^(−α) + B·D^(−β) computed
    without leaving log space (Hoffmann et al.'s parameterization; LSE keeps it overflow-free).
    The LSE gradient is the softmax over the three terms, which makes the θ-gradient cheap:
    e.g. ∂logL/∂α = −softmax_weight(N-term)·ln N.
    """
    a, b, e, alpha, beta = theta
    u = a - alpha * ln_n
    v = b - beta * ln_d
    m = np.maximum(np.maximum(u, v), e)
    exp_u, exp_v, exp_e = np.exp(u - m), np.exp(v - m), np.exp(e - m)
    total = exp_u + exp_v + exp_e
    log_pred = m + np.log(total)
    value, deriv = _huber(log_pred - ln_l, delta)
    w_u, w_v, w_e = exp_u / total, exp_v / total, exp_e / total
    grad = np.array(
        [
            (deriv * w_u).mean(),
            (deriv * w_v).mean(),
            (deriv * w_e).mean(),
            -(deriv * w_u * ln_n).mean(),
            -(deriv * w_v * ln_d).mean(),
        ]
    )
    return float(value.mean()), grad


@dataclass(frozen=True)
class JointLaw:
    """The fitted law ``L(N, D) = E + A·N^(−α) + B·D^(−β)`` plus fit-quality forensics.

    ``n_converged`` counts starts landing within ``CONVERGENCE_TOL`` objective of the best —
    the degeneracy diagnostic: the Chinchilla objective has near-flat directions (A and α
    trade off), so a healthy fit has MANY starts in the best basin; ``n_converged == 1`` means
    the answer depends on where the optimizer happened to start and must not be trusted.
    """

    A: float
    B: float
    E: float
    alpha: float
    beta: float
    objective: float  # final mean Huber loss of the winning start
    n_starts: int
    n_converged: int

    def predict(self, n_params: float, tokens: float) -> float:
        """``L = E + A·N^(−α) + B·D^(−β)`` — exp of the LSE prediction, in bpb."""
        return float(
            self.E
            + self.A * float(n_params) ** (-self.alpha)
            + self.B * float(tokens) ** (-self.beta)
        )

    def predict_optimal(self, compute: float) -> tuple[float, float, float]:
        """Closed-form compute-optimal (N_opt, D_opt, L_pred) under C = 6·N·D.

        Minimizing L(N, D) s.t. 6ND = C: substitute D = C/(6N) and set dL/dN = 0, giving
        ``N_opt^(α+β) = (αA/(βB))·(C/6)^β`` — with G = (α·A)/(β·B),
        ``N_opt = G^(1/(α+β)) · (C/6)^(β/(α+β))`` and D_opt from the bridge
        (Hoffmann et al. 2022, Approach 3, the D.3 derivation; the fitted exponents give
        N_opt ∝ C^(β/(α+β)) — for α = β = 0.5 that's the Chinchilla-familiar C^0.5).
        """
        s = self.alpha + self.beta
        g = (self.alpha * self.A) / (self.beta * self.B)
        flops = compute / FLOPS_PER_PARAM_TOKEN
        n_opt = g ** (1.0 / s) * flops ** (self.beta / s)
        d_opt = compute / (FLOPS_PER_PARAM_TOKEN * n_opt)
        return float(n_opt), float(d_opt), self.predict(n_opt, d_opt)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "A": self.A,
            "B": self.B,
            "E": self.E,
            "alpha": self.alpha,
            "beta": self.beta,
            "objective": self.objective,
            "n_starts": self.n_starts,
            "n_converged": self.n_converged,
        }


def fit_joint_law(
    records: Sequence[SweepRecord],
    *,
    huber_delta: float = HUBER_DELTA,
    warm_start: JointLaw | None = None,
) -> JointLaw:
    """Fit ``L(N, D) = E + A·N^(−α) + B·D^(−β)`` jointly on all records (Approach 3).

    Multi-start L-BFGS-B over α, β ∈ {0.2, 0.4, 0.6, 0.8} × e ∈ {−2, −1, 0} (48 starts);
    a, b initialize from the data as ``ln(mean L) + α·mean(ln N)`` (resp. D) — the value that
    would make each loss term exactly explain the mean loss. The best final objective wins;
    ``warm_start`` (used by the bootstrap/LOO refits) replaces the grid with a single start
    from a parent fit's θ — resampled data sits near the parent's basin, so the warm refit is
    the same estimator at ~1/48th of the cost. ``huber_delta`` is exposed so tests can run
    δ → ∞ as an effective least-squares control. Raises RuntimeError if no start converges.
    """
    from scipy.optimize import minimize

    if len(records) < 5:
        raise ValueError(f"need >=5 points to fit 5 parameters, got {len(records)}")
    ln_n, ln_d, ln_l = _arrays(records)
    if warm_start is not None:
        w = warm_start
        inits = [np.array([np.log(w.A), np.log(w.B), np.log(w.E), w.alpha, w.beta])]
    else:
        ln_mean_l = float(np.log(np.exp(ln_l).mean()))
        inits = [
            np.array(
                [
                    ln_mean_l + alpha0 * ln_n.mean(),
                    ln_mean_l + beta0 * ln_d.mean(),
                    e0,
                    alpha0,
                    beta0,
                ]
            )
            for alpha0 in MULTISTART_ALPHAS
            for beta0 in MULTISTART_ALPHAS
            for e0 in MULTISTART_LOG_E
        ]
    bounds = [(-50.0, 50.0), (-50.0, 50.0), (-50.0, 10.0), (1e-3, 2.0), (1e-3, 2.0)]

    finals: list[tuple[float, np.ndarray]] = []
    for x0 in inits:
        res = minimize(
            _objective,
            x0,
            args=(ln_n, ln_d, ln_l, huber_delta),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            # The objective's valley is nearly flat (A trades off against α), so the default
            # ftol/gtol stop orders of magnitude early — tighten them to actually reach the
            # basin floor (verified: recovers a planted θ to machine precision at zero noise).
            options={"ftol": 1e-18, "gtol": 1e-12, "maxiter": 100_000},
        )
        if np.isfinite(res.fun) and np.all(np.isfinite(res.x)):
            finals.append((float(res.fun), np.asarray(res.x, dtype=np.float64)))
    if not finals:
        raise RuntimeError("joint fit failed: no L-BFGS-B start produced a finite optimum")

    best_fun, best_theta = min(finals, key=lambda t: t[0])
    n_converged = sum(1 for fun, _ in finals if fun <= best_fun + CONVERGENCE_TOL)
    a, b, e, alpha, beta = (float(x) for x in best_theta)
    return JointLaw(
        A=float(np.exp(a)),
        B=float(np.exp(b)),
        E=float(np.exp(e)),
        alpha=alpha,
        beta=beta,
        objective=best_fun,
        n_starts=len(inits),
        n_converged=n_converged,
    )


@dataclass(frozen=True)
class PredictionInterval:
    """Bootstrap prediction interval at one (N, D) point.

    ``n_failed_fits`` is surfaced, not hidden: a fit that only works on SOME resamplings of
    the data is not a certified law, so >20% failures (``fit_failure_red_flag``) fails the
    gate no matter how tight the interval looks.
    """

    lo: float
    hi: float
    half_width: float
    n_failed_fits: int
    n_boot: int
    level: float

    @property
    def fit_failure_red_flag(self) -> bool:
        return self.n_failed_fits > BOOT_FAILURE_FRACTION * self.n_boot

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "half_width": self.half_width,
            "n_failed_fits": self.n_failed_fits,
            "n_boot": self.n_boot,
            "level": self.level,
            "fit_failure_red_flag": self.fit_failure_red_flag,
        }


def bootstrap_prediction_interval(
    records: Sequence[SweepRecord],
    n_params: float,
    tokens: float,
    n_boot: int = 200,
    seed: int = 0,
    level: float = 0.90,
    parent: JointLaw | None = None,
) -> PredictionInterval:
    """Nonparametric bootstrap over runs: resample records with replacement (seeded), refit,
    and take the central ``level`` quantiles of the predicted bpb at (``n_params``, ``tokens``).

    Refits warm-start from the full-data fit — the resampled objective is a perturbation of
    the parent one, so the warm refit lands in the same basin at a fraction of the multi-start
    cost (200 × 48 cold starts would make the gate the slowest test in the repo). ``parent``
    lets a caller that already fitted the same records (``evaluate_gates``) skip the refit.
    """
    if parent is None:
        parent = fit_joint_law(records)
    rng = np.random.default_rng(seed)
    n = len(records)
    preds: list[float] = []
    n_failed = 0
    for _ in range(n_boot):
        sample = [records[i] for i in rng.integers(0, n, size=n)]
        try:
            law = fit_joint_law(sample, warm_start=parent)
        except (RuntimeError, ValueError):
            n_failed += 1
            continue
        pred = law.predict(n_params, tokens)
        if np.isfinite(pred):
            preds.append(pred)
        else:
            n_failed += 1
    if not preds:
        raise RuntimeError(f"all {n_boot} bootstrap refits failed — the fit is not estimable")
    q_lo = (1.0 - level) / 2.0
    lo, hi = (float(x) for x in np.quantile(np.asarray(preds), [q_lo, 1.0 - q_lo]))
    return PredictionInterval(
        lo=lo,
        hi=hi,
        half_width=(hi - lo) / 2.0,
        n_failed_fits=n_failed,
        n_boot=n_boot,
        level=level,
    )


def loo_ratio_swing(
    records: Sequence[SweepRecord], compute: float, *, parent: JointLaw | None = None
) -> float:
    """Leave-one-out stability: refit dropping each record once, take D_opt/N_opt at
    ``compute`` per refit, return max/min — the generalized form of S3's drop-s7 alarm
    (ratio@d20 swung 26.9 → 78.2 = 2.9×, which is why the gate is ≤ 2×). A law whose answer
    hinges on one run is a leverage artifact, not a measurement. Any failed refit raises:
    silently skipping it would understate the swing. ``parent`` skips the full-data refit
    when the caller already has it (the refits only use it as their warm start).
    """
    if parent is None:
        parent = fit_joint_law(records)
    ratios: list[float] = []
    for i in range(len(records)):
        subset = [r for j, r in enumerate(records) if j != i]
        law = fit_joint_law(subset, warm_start=parent)
        n_opt, d_opt, _ = law.predict_optimal(compute)
        ratios.append(d_opt / n_opt)
    return float(max(ratios) / min(ratios))


def residual_trend(records: Sequence[SweepRecord], law: JointLaw) -> tuple[float, float]:
    """Structured-residual check: Spearman rank correlation of (log L_true − log L_pred)
    vs log compute. Returns (rho, p_value).

    Rank correlation, not Pearson: the failure mode is a MONOTONE (or zigzag) drift of
    residuals along the compute axis — the signature S3 exhibited — and Spearman catches any
    monotone misfit regardless of shape. Strong AND significant trend ⇒ the functional form is
    wrong (missing term, wrong regime) and no gate count rescues the fit.
    """
    from scipy.stats import spearmanr

    ln_n, ln_d, ln_l = _arrays(records)
    residuals = np.array(
        [
            ln_l[i] - np.log(law.predict(np.exp(ln_n[i]), np.exp(ln_d[i])))
            for i in range(len(records))
        ]
    )
    log_c = np.log(np.asarray([float(r["compute"]) for r in records], dtype=np.float64))
    # scipy's SignificanceResult stubs don't expose .statistic/.pvalue to pyright; it IS the
    # (statistic, pvalue) namedtuple, so cast and unpack it instead of touching attributes.
    rho, p_value = cast("tuple[float, float]", spearmanr(log_c, residuals))
    return float(rho), float(p_value)


def r2_log_l(records: Sequence[SweepRecord], law: JointLaw) -> float:
    """R² in log-L space — the space the fit lives in (a raw-space R² would be dominated by
    the largest losses and bless a bad exponent, same lesson as ``r2_loglog`` in s3_sweep)."""
    ln_n, ln_d, ln_l = _arrays(records)
    ln_pred = np.array(
        [np.log(law.predict(np.exp(ln_n[i]), np.exp(ln_d[i]))) for i in range(len(records))]
    )
    ss_res = float(np.square(ln_l - ln_pred).sum())
    ss_tot = float(np.square(ln_l - ln_l.mean()).sum())
    return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class GateResult:
    """One acceptance gate's verdict: the measured value, its threshold, and pass/fail."""

    name: str
    value: float
    threshold: float
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateReport:
    """The four pre-registered §1.4 gates. ``all_passed`` — and ONLY ``all_passed`` — licenses
    quoting the law within its span; extrapolation to the d20 carries its interval or is not
    quoted at all."""

    gates: tuple[GateResult, ...]
    all_passed: bool
    compute: float
    n_params: float
    tokens: float

    def to_dict(self) -> dict[str, object]:
        return {
            "all_passed": self.all_passed,
            "compute": self.compute,
            "n_params": self.n_params,
            "tokens": self.tokens,
            "gates": {
                g.name: {
                    "value": g.value,
                    "threshold": g.threshold,
                    "passed": g.passed,
                    "detail": g.detail,
                }
                for g in self.gates
            },
        }


def evaluate_gates(
    records: Sequence[SweepRecord],
    law: JointLaw,
    *,
    compute: float = D20_COMPUTE_FLOPS,
    n_params: float,
    tokens: float,
    n_boot: int = 200,
    seed: int = 0,
) -> GateReport:
    """Evaluate the four pre-registered acceptance gates (§1.4) at the target point.

    Why four: R² alone blessed the failed S3 fit — a high R² can coexist with a luck-dependent
    basin (ii), single-point leverage (iii), and structured residuals (iv). Each gate exists
    because one of those failure modes actually happened. ``(n_params, tokens)`` is the
    prediction point for the bootstrap interval (the d20's 480.4M params × 9.6B tokens);
    ``compute`` is where the LOO ratio swing is read.
    """
    r2 = r2_log_l(records, law)
    interval = bootstrap_prediction_interval(
        records, n_params, tokens, n_boot=n_boot, seed=seed, parent=law
    )
    swing = loo_ratio_swing(records, compute, parent=law)
    rho, p_value = residual_trend(records, law)

    gates = (
        GateResult(
            name="r2_log_l",
            value=r2,
            threshold=R2_THRESHOLD,
            passed=r2 >= R2_THRESHOLD,
            detail="R² in log-L space ≥ 0.98 — necessary, not sufficient (retained for continuity)",
        ),
        GateResult(
            name="bootstrap_pi_half_width",
            value=interval.half_width,
            threshold=PI_HALF_WIDTH_THRESHOLD,
            passed=interval.half_width <= PI_HALF_WIDTH_THRESHOLD
            and not interval.fit_failure_red_flag,
            detail=(
                f"90% bootstrap PI half-width ≤ 0.02 bpb at the target point "
                f"({interval.n_failed_fits}/{interval.n_boot} refits failed; "
                f">{BOOT_FAILURE_FRACTION:.0%} failing is itself a red flag)"
            ),
        ),
        GateResult(
            name="loo_ratio_swing",
            value=swing,
            threshold=LOO_SWING_THRESHOLD,
            passed=swing <= LOO_SWING_THRESHOLD,
            detail="max/min D_opt:N_opt leaving each point out once, ≤ 2× (S3 measured 2.9×)",
        ),
        GateResult(
            name="residual_trend",
            value=abs(rho),
            threshold=RESIDUAL_RHO_THRESHOLD,
            passed=abs(rho) < RESIDUAL_RHO_THRESHOLD or p_value >= RESIDUAL_P_THRESHOLD,
            detail=(
                f"Spearman residual-vs-log-C trend: fails only when strong AND significant "
                f"(|rho| ≥ {RESIDUAL_RHO_THRESHOLD} and p < {RESIDUAL_P_THRESHOLD}); "
                f"p = {p_value:.4f}"
            ),
        ),
    )
    return GateReport(
        gates=gates,
        all_passed=all(g.passed for g in gates),
        compute=compute,
        n_params=n_params,
        tokens=tokens,
    )
