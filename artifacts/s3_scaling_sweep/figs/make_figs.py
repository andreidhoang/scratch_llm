"""S3 scaling-law fit — the four teaching figures (live data + planned grid).

Fig 1: why a fixed compute budget has a best model size (the isoFLOP U-curve).
Fig 2: our s1–s7 ladder in (N, D) space — ratio rays, iso-compute diagonals, the d20 star.
Fig 3: the loss frontier — measured val_bpb vs compute (live results.json) + oracles.
Fig 4: what the fitter does with our grid — N*(C), D*(C) power laws, the gates, the
       extrapolation to the d20's compute.

Run: uv run python artifacts/s3_scaling_sweep/figs/make_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scratch_llm.scaling.isoflop import fit_powerlaw
from scratch_llm.scaling.s3_sweep import D20_COMPUTE_FLOPS, build_grid

OUT = Path(__file__).parent
RESULTS = OUT.parent / "results.json"

# --------------------------------------------------------------------------------------
# Fig 1 — the isoFLOP U-curve (synthetic Chinchilla-form loss, Hoffmann Approach-3
# constants: E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.28 — illustration only).
# --------------------------------------------------------------------------------------
E, A, B, AL, BE = 1.69, 406.4, 410.7, 0.34, 0.28


def chinchilla_L(n: np.ndarray, c: float) -> np.ndarray:
    d = c / (6.0 * n)
    return E + A / n**AL + B / d**BE


fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
n = np.logspace(6.5, 10, 400)
mins = []
for c, color in [(1e17, "tab:blue"), (1e18, "tab:orange"), (1e19, "tab:green")]:
    L = chinchilla_L(n, c)
    ax.plot(n, L, color=color, label=f"C = {c:.0e}")
    i = int(np.argmin(L))
    mins.append((c, n[i]))
    ax.plot(n[i], L[i], "o", color=color, ms=9, mec="k")
    ax.annotate("N*(C)", (n[i], L[i]), textcoords="offset points", xytext=(8, 10))
ax.set_xscale("log")
ax.set_xlabel("model size N (params) — bigger N eats the budget, D = C/6N shrinks")
ax.set_ylabel("loss L(N, D)")
ax.set_title("Fig 1 · At fixed compute, loss is U-shaped in N — the argmin is N*(C)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig1_isoflop_U.png")
plt.close(fig)

# --------------------------------------------------------------------------------------
# Fig 2 — the ladder in (N, D) space.
# --------------------------------------------------------------------------------------
grid = build_grid()
done = set()
if RESULTS.exists():
    done = {r["point"] for r in json.loads(RESULTS.read_text())}

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
nn = np.logspace(7, 9.2, 100)
for c in [1.92e16, 4.8e16, 9.59e16, 1.69e17, 4.21e17, 8.43e17, 2.77e19]:
    ax.plot(nn, c / (6 * nn), color="0.85", lw=1, zorder=0)
    ax.text(nn[-1] * 1.05, c / (6 * nn[-1]), f"C={c:.0e}", fontsize=7, color="0.5")
for r in [8, 20, 40]:
    ax.plot(nn, r * nn, ":", color="0.6", lw=1)
    ax.text(nn[6], r * nn[6] * 1.15, f"D = {r}·N", fontsize=7, color="0.4", rotation=38)
colors = {4: "tab:blue", 8: "tab:orange", 12: "tab:green"}
for g in grid:
    if g.point == "s8":
        continue
    marker = "o" if g.point in done else "s"
    fill = colors[g.depth] if g.point in done else "none"
    ax.plot(g.n_params, g.tokens, marker, ms=10, mfc=fill, mec=colors[g.depth], mew=2)
    ax.annotate(g.point, (g.n_params, g.tokens), textcoords="offset points", xytext=(9, 4),
                fontsize=9, weight="bold")
ax.plot(480.4e6, 9.6e9, "*", ms=18, color="crimson", mec="k")
ax.annotate("d20 (480M, 9.6B)", (480.4e6, 9.6e9), textcoords="offset points", xytext=(10, -3),
            fontsize=9, color="crimson", weight="bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("N — params (log)")
ax.set_ylabel("D — tokens (log)")
ax.set_title("Fig 2 · The S3 ladder: filled = measured, hollow = running/pending\n"
             "grey diagonals = iso-compute C=6ND · dotted rays = D:N ratio")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=colors[d], label=f"depth {d}")
                   for d in (4, 8, 12)], loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig2_ladder.png")
plt.close(fig)

# --------------------------------------------------------------------------------------
# Fig 3 — the loss frontier (live).
# --------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
if RESULTS.exists():
    recs = sorted(json.loads(RESULTS.read_text()), key=lambda r: r["compute"])
    by_depth: dict[int, list] = {}
    for r in recs:
        by_depth.setdefault(r["depth"], []).append(r)
    for d, rs in by_depth.items():
        ax.plot([r["compute"] for r in rs], [r["val_bpb"] for r in rs], "o-",
                color=colors[d], label=f"depth {d} (measured)")
        for r in rs:
            ax.annotate(f"{r['point']}\n{r['val_bpb']:.4f}", (r["compute"], r["val_bpb"]),
                        textcoords="offset points", xytext=(6, 8), fontsize=8)
    pending = [g for g in grid if g.point not in {r["point"] for r in recs} and g.point != "s8"]
    ax.plot([g.compute for g in pending], [1.05] * len(pending), "v", color="0.5", ms=9)
    for i, g in enumerate(pending):
        ax.annotate(g.point, (g.compute, 1.05), textcoords="offset points",
                    xytext=(-4 + 14 * (i % 2), -16), fontsize=8, color="0.5")
# oracles: F12 arms (35M / 700M tok) and nanochat leaderboard d24 ratio-8 (~1.38B params)
c_f12 = 6 * 35e6 * 700e6
ax.plot(c_f12, 1.30205, "D", ms=9, mfc="none", mec="tab:red", mew=2)
ax.annotate("F12 ClimbMix 1.3021", (c_f12, 1.30205), textcoords="offset points",
            xytext=(8, -4), fontsize=8, color="tab:red")
ax.plot(c_f12, 1.19197, "D", ms=9, mfc="none", mec="tab:purple", mew=2)
ax.annotate("F12 FineWeb-EDU 1.1920", (c_f12, 1.19197), textcoords="offset points",
            xytext=(-160, -16), fontsize=8, color="tab:purple")
c_d24 = 6 * 1.38e9 * (8 * 1.38e9)
ax.plot(c_d24, 0.718, "P", ms=12, mfc="none", mec="k", mew=2)
ax.annotate("nanochat d24 leaderboard 0.718 (FP8)", (c_d24, 0.718),
            textcoords="offset points", xytext=(-190, 8), fontsize=8)
ax.set_xscale("log")
ax.set_ylim(0.68, 1.42)
ax.set_xlabel("compute C = 6ND (FLOPs, log)")
ax.set_ylabel("val bpb (held-out)")
ax.set_title("Fig 3 · The loss frontier: bpb falls as a power law in compute\n"
             "(triangles = queued points; diamonds/plus = external oracles)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig3_frontier.png")
plt.close(fig)

# --------------------------------------------------------------------------------------
# Fig 4 — the N*(C) / D*(C) fits on the planned grid + extrapolation to the d20.
# --------------------------------------------------------------------------------------
pts = [g for g in grid if g.point != "s8"]
cs = [g.compute for g in pts]
ns = [float(g.n_params) for g in pts]
ds = [float(g.tokens) for g in pts]
n_law = fit_powerlaw(cs, ns)
d_law = fit_powerlaw(cs, ds)


def r2(law, xs, ys):
    lx, ly = np.log(xs), np.log(ys)
    pred = np.log(law.coeff) + law.exponent * lx
    return 1 - float(np.square(ly - pred).sum()) / float(np.square(ly - ly.mean()).sum())


r2n, r2d = r2(n_law, cs, ns), r2(d_law, cs, ds)
cc = np.logspace(np.log10(min(cs)), np.log10(D20_COMPUTE_FLOPS), 200)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
for ax, law, vals, name, r2v in [
    (axes[0], n_law, ns, "N*(C) params", r2n),
    (axes[1], d_law, ds, "D*(C) tokens", r2d),
]:
    ax.plot(cs, vals, "o", ms=9, color="tab:blue", label="ladder (min-pick)")
    ax.plot(cc, law.predict(cc), "-", color="tab:orange",
            label=f"fit: {law.coeff:.3g}·C^{law.exponent:.3f}  (R²={r2v:.3f})")
    ax.axvline(D20_COMPUTE_FLOPS, color="crimson", ls="--", lw=1)
    ax.text(D20_COMPUTE_FLOPS * 0.7, ax.get_ylim()[0], "d20 C", rotation=90, fontsize=8,
            color="crimson", va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compute C (log)")
    ax.set_title(name)
    ax.legend(fontsize=8)
ratio = d_law.predict(D20_COMPUTE_FLOPS) / n_law.predict(D20_COMPUTE_FLOPS)
fig.suptitle(
    f"Fig 4 · The fitter on our grid: a={n_law.exponent:.3f}, b={d_law.exponent:.3f}, "
    f"a+b={n_law.exponent + d_law.exponent:.3f} (PASS) — but min R²={min(r2n, r2d):.3f} < 0.98 "
    f"⇒ gate RAISES (T1)\nD zigzags by design (ratios 8/20/40 cycle) ⇒ D*(C) is no clean power "
    f"law; extrapolated ratio at d20 = {ratio:.1f}",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(OUT / "fig4_fit_gates.png")
plt.close(fig)

print("wrote fig1_isoflop_U.png fig2_ladder.png fig3_frontier.png fig4_fit_gates.png")
print(f"a={n_law.exponent:.4f} b={d_law.exponent:.4f} a+b={n_law.exponent + d_law.exponent:.4f} "
      f"R2n={r2n:.4f} R2d={r2d:.4f} ratio@d20={ratio:.2f}")
