"""A3 chinchilla_isoflops, end to end: fit the course IsoFLOP data, gate on a+b≈1, extrapolate.

Runs the scaling/isoflop.py fitter on the vendored ``tests/fixtures/isoflops_curves.json``
(72 runs, 9 budgets), fits ``N_opt ∝ C^a``, derives ``D_opt = C/(6N)`` and fits ``D_opt ∝ C^b``,
runs the exponent-sum gate, predicts N_opt/D_opt at 1e23 and 1e24 FLOPs, prints the two
one-sentence answers plus the extrapolation factor, and saves the log-log plot (data, both fit
lines, extrapolation region shaded) to ``bench/a3_isoflop.png``.

Usage: ``python scripts/a3_isoflop.py`` (matplotlib Agg — headless-safe).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scratch_llm.scaling.isoflop import (
    PowerLaw,
    check_exponent_sum,
    fit_powerlaw,
    isoflop_min,
    propose_shape,
    tokens_from_compute_params,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "isoflops_curves.json"
PLOT_PATH = REPO / "bench" / "a3_isoflop.png"
TARGETS = (1e23, 1e24)

N_COLOR = "#2a78d6"  # categorical slot 1 (blue)
D_COLOR = "#1baf7a"  # categorical slot 2 (aqua) — direct-labeled per the relief rule
INK = "#333333"


def _plot(
    budgets: list[float],
    n_opts: list[float],
    d_opts: list[float],
    n_law: PowerLaw,
    d_law: PowerLaw,
    extrap_factor: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_max = max(budgets)
    grid = np.logspace(np.log10(min(budgets)), np.log10(max(TARGETS)), 200)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)
    ax.axvspan(c_max, max(TARGETS), color="#000000", alpha=0.05, zorder=0)
    ax.plot(grid, [n_law.predict(float(c)) for c in grid], color=N_COLOR, lw=2, zorder=2)
    ax.plot(grid, [d_law.predict(float(c)) for c in grid], color=D_COLOR, lw=2, zorder=2)
    ax.scatter(budgets, n_opts, s=36, color=N_COLOR, zorder=3, label="N_opt (params)")
    ax.scatter(budgets, d_opts, s=36, color=D_COLOR, zorder=3, label="D_opt (tokens)")
    for law, color in ((n_law, N_COLOR), (d_law, D_COLOR)):
        ax.scatter(
            TARGETS,
            [law.predict(t) for t in TARGETS],
            s=46,
            marker="D",
            color=color,
            edgecolors="white",
            linewidths=1.0,
            zorder=4,
        )
    ax.text(2e20, 1.1e9, f"N_opt = {n_law.coeff:.3f}·C^{n_law.exponent:.3f}", color=INK, fontsize=9)
    ax.text(
        2e19, 1.1e11, f"D_opt = {d_law.coeff:.3f}·C^{d_law.exponent:.3f}", color=INK, fontsize=9
    )
    ax.text(
        np.sqrt(c_max * max(TARGETS)),
        9e8,
        f"extrapolation\n(x{extrap_factor:,.0f} past data)",
        color="#666666",
        fontsize=8,
        ha="center",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training compute C (FLOPs)")
    ax.set_ylabel("Count (parameters / tokens)")
    ax.set_title("Chinchilla IsoFLOP fit — compute-optimal N and D vs training compute")
    ax.grid(True, which="major", color="#e0e0e0", linewidth=0.6, zorder=1)
    ax.legend(loc="upper left", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.text(
        0.01,
        0.005,
        f"Log-log linear fit on per-budget argmin-loss runs (9 IsoFLOP profiles, 72 runs).\n"
        f"Shaded = extrapolation, x{extrap_factor:,.0f} past the largest fitted budget "
        f"({c_max:.0e} -> {max(TARGETS):.0e} FLOPs).",
        fontsize=7,
        color="#666666",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(PLOT_PATH)
    plt.close(fig)


def main() -> None:
    runs = json.loads(FIXTURE.read_text())
    pairs = isoflop_min(runs)
    budgets = [c for c, _ in pairs]
    n_opts = [n for _, n in pairs]
    d_opts = [tokens_from_compute_params(c, n) for c, n in pairs]

    n_law = fit_powerlaw(budgets, n_opts)
    d_law = fit_powerlaw(budgets, d_opts)
    a, b = n_law.exponent, d_law.exponent
    check_exponent_sum(a, b)  # raises before any extrapolation if the fit is inconsistent

    c_max = max(budgets)
    extrap_factor = max(TARGETS) / c_max

    print(
        f"IsoFLOP fit: {len(pairs)} budgets ({min(budgets):.0e} ... {c_max:.0e} FLOPs), "
        f"{len(runs)} runs"
    )
    print(f"  N_opt = {n_law.coeff:.4f} * C^{a:.4f}")
    print(f"  D_opt = {d_law.coeff:.4f} * C^{b:.4f}")
    print(f"  gate: a + b = {a + b:.4f} in 1 +/- 0.05 -> PASS")

    n23, n24 = n_law.predict(1e23), n_law.predict(1e24)
    d23, d24 = d_law.predict(1e23), d_law.predict(1e24)
    shape23, shape24 = propose_shape(n23), propose_shape(n24)
    print(
        f"Answer (a): the compute-optimal model size is N_opt ~= {n23:.2e} params "
        f"(~{n23 / 1e9:.0f}B, e.g. n_layer={shape23[0]}, d_model={shape23[1]}) at 1e23 FLOPs "
        f"and {n24:.2e} (~{n24 / 1e9:.0f}B, n_layer={shape24[0]}, d_model={shape24[1]}) "
        f"at 1e24 FLOPs."
    )
    print(
        f"Answer (b): the compute-optimal data budget is D_opt = C/(6*N_opt) ~= {d23:.2e} "
        f"tokens (~{d23 / 1e9:.0f}B) at 1e23 FLOPs and {d24:.2e} (~{d24 / 1e9:.0f}B) at "
        f"1e24 FLOPs."
    )
    print(
        f"Extrapolation factor: 1e24 reaches {extrap_factor:,.0f}x past the largest fitted "
        f"budget ({c_max:.0e}) — the fit holds only while the training regime does."
    )

    _plot(budgets, n_opts, d_opts, n_law, d_law, extrap_factor)
    print(f"Saved plot: {PLOT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
