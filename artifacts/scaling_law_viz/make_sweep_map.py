"""Visual map of every sweep point: S3 (done), P5 (planned), S3.5 (planned), d20 target.

Data: artifacts/s3_scaling_sweep/{results,grid}.json + pinned values from
docs/FRONTIER_2026_SCALING_PROGRAM.md / FRONTIER_2026_D20_CERTAINTY_PLAN.md §6.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- real S3 data -----------------------------------------------------------
res = json.load(open("artifacts/s3_scaling_sweep/results.json"))
s3 = {r["point"]: r for r in res}

# ---- pinned planned values (SCALING_PROGRAM.md / CERTAINTY_PLAN §6) ---------
N = {"d4": 19_991_808, "d8": 59_255_296, "d12": 135_288_576, "d14": 193_600_000,
     "d20": 480_400_000}
C_d20 = 2.77e19


def flops(n_params, ratio):
    return 6 * n_params * n_params * ratio  # 6*N*D, D = ratio*N (tokens/param)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 8.2),
                              gridspec_kw={"width_ratios": [1.55, 1]})

# ================= Panel A: compute plane ====================================
ax.set_xscale("log")
ax.set_yscale("log")

# S3 measured points
s3_offsets = {"s1": (8, 6), "s2": (8, 6), "s3": (8, -16), "s4": (-60, 10),
              "s5": (10, 2), "s6": (8, -16), "s7": (-8, 12)}
for p, r in s3.items():
    fam = f'd{r["depth"]}'
    ax.scatter(r["compute"], r["n_params"], s=130, zorder=5,
               color={"d4": "#4C9BD6", "d8": "#2E8B57", "d12": "#D64545"}[fam],
               marker="o", edgecolor="k", linewidth=0.7)
    ax.annotate(f'{p}\nbpb={r["val_bpb"]:.3f}', (r["compute"], r["n_params"]),
                textcoords="offset points", xytext=s3_offsets[p], fontsize=8)

# s7 batch-confound callout
s7 = s3["s7"]
ax.annotate("s7 batch-4 confound\n(rerun @ batch-8 pending)",
            xy=(s7["compute"], s7["n_params"]), xytext=(s7["compute"] * 0.13, s7["n_params"] * 4.6),
            fontsize=8.5, color="#8B0000",
            arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.2))

# P5 planned — all 5 LR runs sit at ONE (C, N) point; draw a single square
p5_lr = flops(N["d12"], 4)
ax.scatter([p5_lr], [N["d12"]], s=190, marker="s", color="#E8A33D",
           edgecolor="k", zorder=6)
ax.annotate("P5 core sweep: 5 runs at THIS point,\nLR ×{0.5,0.7,1.0,1.4,2.0}, d12 @ ratio-4\n(same compute — only LR moves; see Panel B)",
            (p5_lr, N["d12"]), textcoords="offset points", xytext=(-12, -60),
            fontsize=8.5, color="#8a5a00", ha="right")
# P5 confirmation run reuses s7's exact (C, N) — draw it as a halo around s7
ax.scatter([flops(N["d12"], 8)], [N["d12"]], s=460, marker="s",
           facecolors="none", edgecolor="#E8A33D", linewidth=2.2, zorder=6)
ax.annotate("P5 confirm: d12@r8 with winning LR\n(same (C,N) as s7 — halo; doubles as T2 anchor)",
            (flops(N["d12"], 8), N["d12"]), textcoords="offset points",
            xytext=(-16, 32), fontsize=8.5, color="#8a5a00", ha="right")
p5_w = flops(N["d8"], 4)
ax.scatter([p5_w], [N["d8"]], s=190, marker="s", color="#F2C14E",
           edgecolor="k", zorder=6)
ax.annotate("P5 width probe: 3 runs here,\nd8 LR ×{0.7,1.0,1.4}",
            (p5_w, N["d8"]), textcoords="offset points", xytext=(-12, -38),
            fontsize=8.5, color="#8a5a00", ha="right")

# S3.5 re-fit points
for name, fam, ratio, style in [
    ("d12-r20 (=s8, re-homed)", "d12", 20, "D"),
    ("d14-r8", "d14", 8, "D"),
    ("d14-r20 (optional)", "d14", 20, "D"),
]:
    c = flops(N[fam], ratio)
    ax.scatter(c, N[fam], s=150, marker=style, color="#7B5EA7", edgecolor="k", zorder=6)
    ax.annotate(name, (c, N[fam]), textcoords="offset points", xytext=(12, -18),
                fontsize=8.5, color="#4B2E83")

# R1/R2 ablation arms share the d12-r20 horizon
ax.annotate("R1/R2 arch-ablation arms also live here\n(d12 @ ratio-20, recipe-v1)",
            xy=(flops(N["d12"], 20), N["d12"]), xytext=(flops(N["d12"], 20) * 0.28, N["d12"] * 0.14),
            fontsize=8, color="#4B2E83", style="italic",
            arrowprops=dict(arrowstyle="->", color="#4B2E83", lw=1))

# d20 target
ax.scatter(C_d20, N["d20"], s=340, marker="*", color="#FFD700", edgecolor="k",
           linewidth=1.2, zorder=7)
ax.annotate("d20 flagship target\n2.77e19 FLOPs · 480.4M\nband 0.89–0.92 bpb",
            (C_d20, N["d20"]), textcoords="offset points", xytext=(-8, 26),
            fontsize=9, fontweight="bold", ha="right")

# iso-ratio guide lines (tokens/param = const → C ∝ N²)
ns = np.logspace(7.2, 8.85, 40)
for r in (4, 8, 20, 40):
    ax.plot(6 * ns * ns * r, ns, ":", color="gray", lw=0.9, alpha=0.65)
    ax.text(6 * ns[-1] ** 2 * r * 1.1, ns[-1], f"ratio {r}", fontsize=7.5,
            color="gray", rotation=42, va="bottom")

# extrapolation-distance bracket
ax.annotate("", xy=(C_d20, 3.3e7), xytext=(flops(N["d12"], 20), 3.3e7),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
ax.text(np.sqrt(C_d20 * flops(N["d12"], 20)), 2.35e7,
        "extrapolation ≈ 12.6× compute (within 1 order of magnitude)",
        fontsize=8, ha="center")

ax.set_xlabel("Compute C = 6ND (FLOPs, log)", fontsize=11)
ax.set_ylabel("Model size N (params, log)", fontsize=11)
ax.set_title("Panel A — Every run on the (C, N) plane\n"
             "blue/green/red = S3 measured · orange squares = P5 (recipe-v1 sweep) · "
             "purple diamonds = S3.5 re-fit · star = d20", fontsize=10)
ax.set_xlim(1.2e16, 4e20)
ax.set_ylim(1.2e7, 1.3e9)
ax.grid(True, which="both", alpha=0.18)

# ================= Panel B: P5 LR sweep design ===============================
mult = np.array([0.5, 0.7, 1.0, 1.4, 2.0])
# schematic U-curve (bpb vs LR multiplier), typical near-optimum shape
x = np.linspace(0.42, 2.15, 300)
curve = 0.958 + 0.012 * (np.log(x) ** 2) / (np.log(1.5) ** 2) + 0.004 * np.maximum(x - 1.55, 0) ** 2
ax2.plot(x, curve, "-", color="#888", lw=2, label="expected loss-vs-LR bowl (schematic)")
ypts = 0.958 + 0.012 * (np.log(mult) ** 2) / (np.log(1.5) ** 2) + 0.004 * np.maximum(mult - 1.55, 0) ** 2
ax2.scatter(mult, ypts, s=140, color="#E8A33D", edgecolor="k", zorder=5,
            label="P5 grid: ×{0.5, 0.7, 1.0, 1.4, 2.0}")
for m, y in zip(mult, ypts):
    ax2.annotate(f"×{m}", (m, y), textcoords="offset points", xytext=(0, 9),
                 ha="center", fontsize=9, fontweight="bold")

# tie-break zone
ax2.axhspan(0.958, 0.961, color="#2E8B57", alpha=0.18)
ax2.text(1.46, 0.9586, "tie zone < 0.003 bpb\n→ pick the LOWER LR\n(safe for scale-up)",
         fontsize=8.5, color="#1d5c39")
# instability region
ax2.axvspan(1.7, 2.15, color="#D64545", alpha=0.12)
ax2.text(1.74, 0.985, "divergence /\ninstability risk\n(grows with scale)", fontsize=8.5,
         color="#8B0000")

ax2.set_xlabel("LR multiplier (log-ish spacing around current LR)", fontsize=11)
ax2.set_ylabel("final bpb (schematic)", fontsize=11)
ax2.set_title("Panel B — Why 5 LR points (P5 core sweep)\n"
              "bowl shape → 5 log-spaced points resolve both slopes + the floor",
              fontsize=10)
ax2.set_xlim(0.42, 2.15)
ax2.legend(fontsize=8.5, loc="upper left")
ax2.grid(alpha=0.2)

fig.tight_layout()
out = "artifacts/scaling_law_viz/sweep_map.png"
fig.savefig(out, dpi=150)
print("wrote", out)
