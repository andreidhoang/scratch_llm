"""IsoFLOP / Chinchilla fitter — compute-optimal N and D from small sweeps (A3 scaling).

The method is three moves. (1) **IsoFLOP min-picking**: at a fixed budget C, final loss is
quadratic-ish in N (too-small N can't absorb the compute; too-large N can't take enough steps), so
the per-budget argmin-loss run gives one ``(C, N_opt)`` point. (2) **Power-law fit in log-log
space**: ``log N_opt = a·log C + const`` by linear regression — never a raw-space fit, because raw
least squares is heteroscedastic (the largest values dominate the residual and drag the exponent).
(3) The **compute identity** ``C ≈ 6·N·D`` (2 FLOPs/param/token forward + 4 backward) bridges the
two laws: ``D_opt = C / (6·N_opt)`` is derived, not separately measured.

Falsifiable invariant: the two exponents in ``N_opt ∝ C^a`` and ``D_opt ∝ C^b`` must satisfy
``a + b ≈ 1`` (the ``C = 6ND`` identity forces it); ``check_exponent_sum`` raises loudly on
violation — it catches min-picking and bridge bugs *before* any extrapolation. On the CS336 course
data a ≈ b ≈ 0.5 (Chinchilla: scale params and data about equally).

Interview question this answers: "derive and defend a scaling law" — including why the fit is done
in log space, why the exponents must sum to one, and how far past the data an extrapolation
honestly reaches (state the factor; the fit only holds while the regime does).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

FLOPS_PER_PARAM_TOKEN = 6.0


@dataclass(frozen=True)
class PowerLaw:
    """``y = coeff · x^exponent`` — fitted in log-log space, predicted by exponentiating."""

    exponent: float
    coeff: float

    def predict(self, x: float) -> float:
        return self.coeff * x**self.exponent


def isoflop_min(runs: Sequence[Mapping[str, float]]) -> list[tuple[float, float]]:
    """Group runs by ``compute_budget``, take the min-``final_loss`` run per budget.

    Returns one ``(C, N_opt)`` pair per distinct budget, sorted by C ascending — the measurement
    primitive of the whole method (per the CS336 PDF: per-budget argmin, not Hoffmann's quadratic).
    """
    by_budget: dict[float, Mapping[str, float]] = {}
    for run in runs:
        budget = run["compute_budget"]
        best = by_budget.get(budget)
        if best is None or run["final_loss"] < best["final_loss"]:
            by_budget[budget] = run
    return [(c, by_budget[c]["parameters"]) for c in sorted(by_budget)]


def fit_powerlaw(xs: Sequence[float], ys: Sequence[float]) -> PowerLaw:
    """Fit ``y = coeff · x^exponent`` by linear regression on ``(log x, log y)``.

    Log space, never raw: raw least squares on a power law is heteroscedastic — the largest y
    values dominate the residual and bias the exponent (see tests for the planted counterexample).
    """
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if x_arr.size < 2 or x_arr.size != y_arr.size:
        raise ValueError(f"need >=2 paired points, got {x_arr.size} xs and {y_arr.size} ys")
    if np.any(x_arr <= 0) or np.any(y_arr <= 0):
        raise ValueError("power-law fit needs strictly positive xs and ys (log-log space)")
    slope, intercept = np.polyfit(np.log(x_arr), np.log(y_arr), 1)
    return PowerLaw(exponent=float(slope), coeff=float(np.exp(intercept)))


def compute_from_params_tokens(n_params: float, n_tokens: float) -> float:
    """``C = 6·N·D`` — total training FLOPs for N params over D tokens."""
    return FLOPS_PER_PARAM_TOKEN * n_params * n_tokens


def tokens_from_compute_params(compute: float, n_params: float) -> float:
    """``D = C / (6·N)`` — the bridge that derives the D_opt law from the N_opt law."""
    return compute / (FLOPS_PER_PARAM_TOKEN * n_params)


def nonembed_params(n_layer: int, d_model: int) -> int:
    """Non-embedding parameter count ≈ ``12·L·d²`` (attention 4d² + MLP 8d² per block)."""
    return 12 * n_layer * d_model**2


def propose_shape(n_params: float, aspect_ratio: float = 128.0) -> tuple[int, int]:
    """Invert ``N ≈ 12·L·d²`` to a concrete ``(n_layer, d_model)`` for a target N.

    Assumption: a fixed aspect ratio ``d_model / n_layer`` (default 128, the GPT-2/3-family
    width-to-depth regime Kaplan et al. found loss is flat across). With ``d = r·L`` the target
    becomes ``N = 12·r²·L³``; solve L from the cube root, then recover d exactly from
    ``d = sqrt(N / (12·L))`` so rounding error lands in one place.
    """
    if n_params <= 0 or aspect_ratio <= 0:
        raise ValueError("n_params and aspect_ratio must be positive")
    n_layer = max(1, round((n_params / (12.0 * aspect_ratio**2)) ** (1.0 / 3.0)))
    d_model = max(1, round((n_params / (12.0 * n_layer)) ** 0.5))
    return n_layer, d_model


def check_exponent_sum(a: float, b: float, tol: float = 0.05) -> None:
    """The ``a + b ≈ 1`` gate — ``C = 6ND`` forces the two exponents to sum to one.

    Raises loudly on violation so a min-picking or bridge bug is caught *before* extrapolating.
    """
    if abs(a + b - 1.0) > tol:
        raise ValueError(
            f"exponent-sum gate failed: a={a:.4f}, b={b:.4f}, a+b={a + b:.4f} is outside "
            f"1±{tol} — the C=6ND identity is violated; suspect min-picking or the bridge"
        )
