"""Scaling-law pedagogy figures — a planted universe with Chinchilla's fitted constants.

L(N, D) = E + A/N^alpha + B/D^beta   (E=1.69, A=406.4, alpha=0.34, B=410.7, beta=0.2849)

Analytic optimum at fixed C (D = C/6N):
    dL/dN = 0  =>  N_opt = [(alpha*A / beta*B) * (C/6)^beta]^(1/(alpha+beta))
    =>  N_opt ∝ C^a with a = beta/(alpha+beta) ≈ 0.456
    =>  D_opt ∝ C^b with b = alpha/(alpha+beta) ≈ 0.544   (a + b = 1 by construction)

The four figures: (1) the iso-FLOP U-curves and their minima, (2) raw vs log-log axes,
(3) the raw-space least-squares trap (heteroscedasticity), (4) the extrapolation gap
from our s8 point to the d20 with the d14/d16 trigger positions marked.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E, A, B, AL, BE = 1.69, 406.4, 410.7, 0.34, 0.2849
OUT = Path(__file__).parent


def loss_at_budget(N: np.ndarray, C: float) -> np.ndarray:
    """L(N | C): D is forced by the budget via the C = 6ND identity."""
    D = C / (6.0 * N)
    return E + A / N**AL + B / D**BE


def n_opt_exact(C: float) -> float:
    return ((AL * A / (BE * B)) * (C / 6.0) ** BE) ** (1.0 / (AL + BE))


A_EXACT = BE / (AL + BE)
B_EXACT = AL / (AL + BE)

# ---------------------------------------------------------------- fig 1: the U-curves
budgets = [1e17, 1e18, 1e19]
Ns = np.logspace(6, 10.5, 4000)
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#1f77b4", "#d62728", "#2ca02c"]
mins_n, mins_l = [], []
for C, col in zip(budgets, colors, strict=True):
    L = loss_at_budget(Ns, C)
    i = int(np.argmin(L))
    mins_n.append(Ns[i])
    mins_l.append(L[i])
    ax.plot(Ns, L, color=col, lw=2, label=f"C = {C:.0e} FLOPs")
    ax.plot(Ns[i], L[i], "o", color="black", ms=9, zorder=5)
    ax.annotate(
        f"đáy: N* = {Ns[i]:.1e}\nL = {L[i]:.3f}",
        (Ns[i], L[i]),
        textcoords="offset points",
        xytext=(14, 26),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
ax.plot(mins_n, mins_l, "k--", lw=1.5, alpha=0.7)
ax.text(2.2e6, mins_l[0] + 0.06, "đường nối các đáy = N_opt(C)", fontsize=9, style="italic")
ax.set_xscale("log")
ax.set_xlabel("N — số params của model (log scale)")
ax.set_ylabel("loss L(N | C)")
ax.set_title("Hình 1 — Iso-FLOP: ở ngân sách CỐ ĐỊNH, loss theo N là một chữ U")
ax.text(
    0.02, 0.96,
    "nhánh trái: model quá nhỏ → hết chỗ chứa pattern\n(A/N^α dominate: từ điển bé, luyện bao nhiêu cũng plateau)",
    transform=ax.transAxes, fontsize=9, va="top",
)
ax.text(
    0.98, 0.96,
    "nhánh phải: model quá lớn → không đủ tokens để train\n(B/D^β dominate: từ điển khổng lồ, mới đọc 3 trang)",
    transform=ax.transAxes, fontsize=9, va="top", ha="right",
)
ax.legend(loc="center right")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(OUT / "fig1_isoflop_U.png", dpi=140)
plt.close(fig)

# ------------------------------------------------- fig 2: raw axes vs log-log axes
Cs = np.logspace(16.5, 19.5, 8)
Nopts = np.array([n_opt_exact(C) for C in Cs])
slope, intercept = np.polyfit(np.log10(Cs), np.log10(Nopts), 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(Cs, Nopts, "o-", color="#1f77b4")
ax1.set_xlabel("C — compute budget (FLOPs)")
ax1.set_ylabel("N_opt — params tối ưu")
ax1.set_title("Trục thường: một đường cong — khó đọc số mũ")
ax1.grid(alpha=0.3)

ax2.plot(np.log10(Cs), np.log10(Nopts), "o", color="#1f77b4", label="các đáy chữ U đo được")
xx = np.array([16.5, 19.5])
ax2.plot(xx, intercept + slope * xx, "r-", lw=2,
         label=f"đường thẳng fit: slope a = {slope:.3f}")
ax2.set_xlabel("log₁₀ C")
ax2.set_ylabel("log₁₀ N_opt")
ax2.set_title("Trục log-log: power law = ĐƯỜNG THẲNG, slope = exponent")
ax2.annotate(
    f"a = {slope:.3f}  (lý thuyết: β/(α+β) = {A_EXACT:.3f})",
    (16.8, intercept + slope * 16.8 + 0.12), fontsize=11, color="red",
)
ax2.legend()
ax2.grid(alpha=0.3)
fig.suptitle("Hình 2 — Cùng một dữ liệu, hai cách nhìn: log-log biến power law thành bài toán fit đường thẳng")
fig.tight_layout()
fig.savefig(OUT / "fig2_loglog.png", dpi=140)
plt.close(fig)

# ------------------------------------- fig 3: the raw-space least-squares trap
rng = np.random.default_rng(0)
Cs3 = np.logspace(15, 19, 7)
Nexact = np.array([n_opt_exact(C) for C in Cs3])
Nnoisy = Nexact * np.exp(rng.normal(0, 0.12, size=Cs3.size))  # ~12% multiplicative noise

# log-space fit (the right way): polyfit on (log C, log N)
a_log, logk_log = np.polyfit(np.log10(Cs3), np.log10(Nnoisy), 1)
k_log = 10**logk_log

# raw-space fit (the trap): minimize Σ(N − k·C^a)² over a, with optimal k in closed form
def best_k_raw(a: float) -> float:
    Ca = Cs3**a
    return float((Nnoisy * Ca).sum() / (Ca**2).sum())


grid_a = np.linspace(0.2, 0.7, 501)
sse_raw = [np.sum((Nnoisy - best_k_raw(a) * Cs3**a) ** 2) for a in grid_a]
a_raw = float(grid_a[int(np.argmin(sse_raw))])
k_raw = best_k_raw(a_raw)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(np.log10(Cs3), np.log10(Nnoisy), "ko", ms=9, label="điểm đo (nhiễu ~12% theo tỷ lệ)")
ax.plot(xx := np.log10([1e15, 1e19]), np.log10(k_log) + a_log * xx, "g-", lw=2.5,
        label=f"fit LOG-LOG (đúng): a = {a_log:.3f}")
ax.plot(xx, np.log10(k_raw) + a_raw * xx, "r--", lw=2.5,
        label=f"fit RAW-space (sai): a = {a_raw:.3f}")
ax.set_xlabel("log₁₀ C")
ax.set_ylabel("log₁₀ N_opt")
ax.set_title("Hình 3 — Cái bẫy raw-space: điểm budget lớn nhất giữ 99% 'quyền biểu quyết'")
ax.annotate(
    "fit raw kéo đường thẳng lệch về phía\nđiểm lớn nhất và bỏ rơi các điểm nhỏ\n"
    f"→ exponent lệch ({a_raw:.3f} thay vì ~{A_EXACT:.2f})",
    (15.6, np.log10(Nnoisy[-1]) - 0.9), fontsize=10, color="darkred",
)
ax.legend(loc="upper left")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT / "fig3_raw_trap.png", dpi=140)
plt.close(fig)

# ------------------------------------- fig 4: the extrapolation gap to the d20
S8_C, D14_C, D16_C, D20_C = 2.2e18, 4.5e18, 8.6e18, 2.77e19
Cs_fit = np.logspace(16, np.log10(S8_C), 6)           # chỉ fit tới s8 (điểm free lớn nhất)
N_fit = np.array([n_opt_exact(C) for C in Cs_fit]) * np.exp(rng.normal(0, 0.05, Cs_fit.size))
a4, logk4 = np.polyfit(np.log10(Cs_fit), np.log10(N_fit), 1)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(np.log10(Cs_fit), np.log10(N_fit), "o", color="#1f77b4", ms=9,
        label="các điểm sweep MIỄN PHÍ (s1–s8, standing box)")
xx = np.log10([1e16, D20_C])
ax.plot(xx[np.log10([1e16, D20_C]) <= np.log10(S8_C)],
        (logk4 + a4 * xx)[np.log10([1e16, D20_C]) <= np.log10(S8_C)], "b-", lw=2.5,
        label=f"fit: N_opt ∝ C^{a4:.3f}")
mask = np.log10([1e16, D20_C]) > np.log10(S8_C)
ax.plot(np.log10([S8_C, D20_C]), logk4 + a4 * np.log10([S8_C, D20_C]), "r--", lw=2.5,
        label="EXTRAPOLATION — không có điểm đo nào ở đây")
ax.axvspan(np.log10(S8_C), np.log10(D20_C), color="red", alpha=0.07)
for Cx, name, col in [(D14_C, "d14\n(T1/T3)", "orange"), (D16_C, "d16\n(T2)", "purple"),
                      (D20_C, "d20 — $100", "red")]:
    ax.axvline(np.log10(Cx), color=col, ls=":", lw=1.8)
    ax.text(np.log10(Cx), ax.get_ylim()[0] + 0.15, name, rotation=0, ha="center",
            fontsize=9, color=col)
ax.set_xlabel("log₁₀ C (FLOPs)")
ax.set_ylabel("log₁₀ N_opt")
ax.set_title("Hình 4 — Khoảng cách extrapolation: s8 → d20 là 12.6×; d14/d16 cắt còn ~6× / ~3.2×")
ax.legend(loc="upper left")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT / "fig4_extrapolation.png", dpi=140)
plt.close(fig)

# ----------------------------------------------------------------- numeric summary
d20_n = float(10 ** (logk4 + a4 * np.log10(D20_C)))
d20_d = D20_C / (6 * d20_n)
print(f"a_exact = {A_EXACT:.4f}  b_exact = {B_EXACT:.4f}  a+b = {A_EXACT + B_EXACT:.4f}")
print(f"log-log fit on clean points: a = {slope:.4f}")
print(f"noisy points: a_log = {a_log:.4f} (đúng)  vs  a_raw = {a_raw:.4f} (bẫy raw-space)")
print(f"fig4 fit (tới s8): a = {a4:.4f}")
print(f"extrapolate tới d20 C=2.77e19: N_opt = {d20_n:.2e}, D_opt = {d20_d:.2e}, "
      f"ratio D:N = {d20_d / d20_n:.1f}")
print("figures:", [p.name for p in sorted(OUT.glob('fig*.png'))])
