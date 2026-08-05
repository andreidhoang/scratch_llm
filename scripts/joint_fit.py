"""
S3.5 joint scaling-law fitter — fits L(N,D) = E + A/N^alpha + B/D^beta on
CLEAN eta*-calibrated bpb points only.

Methodology (why this is not noise):
  1. Full Chinchilla fit via scipy curve_fit (TRF) with PHYSICAL bounds:
       E in [0.0, 0.50]   irreducible floor (bpb)
       A in [1e-2, 1e5]   N-prefactor
       B in [1e-2, 1e5]   D-prefactor
       alpha in [0.05, 0.80]   N exponent (literature 0.30-0.50)
       beta  in [0.05, 0.80]   D exponent (literature 0.25-0.40)
  2. Bootstrap residuals (N_boot=2000) -> CI on every param and on any
     extrapolation (e.g. d20 prediction). With few points the CI is WIDE;
     we report it honestly rather than hiding it.
  3. Local-slope diagnostics that do NOT need the 4-param global fit:
       beta_local  = d ln(L-E) / d ln(D)   at fixed N=135M (two D points)
       alpha_local = d ln(L-E) / d ln(N)   across the ladder (normalized D:N)
     These are the robust checks when the global fit is under-identified.
  4. Regime/residual gate: flags any point whose residual exceeds 2 sigma_boot.

The clean dataset is assembled by load_clean() which reads ONLY:
  - artifacts/p5/lr0.7/results.json   (d12 r4, the eta* winner)
  - artifacts/p5_phase2/d8r4_widthprobe/results.json
  - artifacts/p5_phase2/s7_confirm_eta/results.json
  - artifacts/p5_phase2/d14r8/results.json
  - artifacts/p5_phase2/d14r20/results.json   (if present)
Old s1-s6 and s7-batch-pair (default-LR) are EXCLUDED by construction.
"""
from __future__ import annotations
import json, math, os, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr
except ImportError:
    sys.exit("need scipy: pip install scipy")


# ---- functional form ---------------------------------------------------------
def chinchilla(ND, E, A, alpha, B, beta):
    """L(N,D) = E + A*N^-alpha + B*D^-beta.  ND = array shape (2, n) or (n,2)."""
    ND = np.atleast_2d(np.asarray(ND))
    if ND.shape[0] != 2:          # accept (n,2) too
        ND = ND.T
    N, D = ND[0], ND[1]
    return E + A * np.power(N, -alpha) + B * np.power(D, -beta)


BOUNDS = ([0.0, 1e-2, 0.05, 1e-2, 0.05], [0.50, 1e5, 0.80, 1e5, 0.80])
P0_DEFAULT = [0.33, 40.0, 0.34, 20.0, 0.28]   # E,A,alpha,B,beta — lit-anchored


# ---- data loading ------------------------------------------------------------
@dataclass
class Pt:
    name: str
    N: float
    D: float
    bpb: float
    lr: float

def load_clean(root: str = ".") -> list[Pt]:
    """Assemble the CLEAN eta*-calibrated dataset. Excludes old tokenizer +
    default-LR runs by construction (never reads those paths)."""
    root = Path(root)
    sources = [
        ("d12r4_p5winner", root/"artifacts/p5/lr0.7/results.json"),
        ("d8r4_widthprobe", root/"artifacts/p5_phase2/d8r4_widthprobe/results.json"),
        ("s7_confirm_eta", root/"artifacts/p5_phase2/s7_confirm_eta/results.json"),
        ("d14r8",          root/"artifacts/p5_phase2/d14r8/results.json"),
        ("d14r20",         root/"artifacts/p5_phase2/d14r20/results.json"),
    ]
    pts: list[Pt] = []
    for name, path in sources:
        if not path.exists():
            continue
        recs = json.loads(path.read_text())
        if not recs:
            continue
        r = recs[0]
        pts.append(Pt(name, r["n_params"], r["tokens"], r["val_bpb"], r.get("lr", 0.0021)))
    return pts


# ---- local-slope diagnostics (robust at few points) --------------------------
def local_beta(pts: list[Pt], E_guess: float = 0.33) -> dict | None:
    """beta at fixed N: need two D points at the same N. We have d12r4 & d12r8
    (both N=135M). beta = -d ln(L-E) / d ln(D)."""
    byN: dict[int, list[Pt]] = {}
    for p in pts:
        byN.setdefault(round(p.N), []).append(p)
    for N, grp in byN.items():
        if len(grp) < 2:
            continue
        grp.sort(key=lambda p: p.D)
        a, b = grp[0], grp[-1]
        LE_a = math.log(max(a.bpb - E_guess, 1e-9))
        LE_b = math.log(max(b.bpb - E_guess, 1e-9))
        lnD_a, lnD_b = math.log(a.D), math.log(b.D)
        slope = (LE_b - LE_a) / (lnD_b - lnD_a)
        return {"N": N, "beta_local": -slope,
                "D_a": a.D, "D_b": b.D, "bpb_a": a.bpb, "bpb_b": b.bpb}
    return None


def local_alpha(pts: list[Pt], E_guess: float = 0.33) -> dict | None:
    """alpha across N at comparable D:N ratio. Use the r4 / r8 rungs.
    alpha = -d ln(L-E) / d ln(N). Needs >=2 distinct N values."""
    distinct = sorted({round(p.N) for p in pts})
    if len(distinct) < 2:
        return None
    series = []
    for p in sorted(pts, key=lambda q: q.N):
        if p.bpb > E_guess:
            series.append((p.N, p.D, p.bpb, math.log(p.bpb - E_guess)))
    if len(series) < 2:
        return None
    xs = np.array([math.log(s[0]) for s in series])
    ys = np.array([s[3] for s in series])
    slope, _ = np.polyfit(xs, ys, 1)
    return {"alpha_local": -slope, "n_points": len(series),
            "Ns": [s[0] for s in series]}


# ---- main fitter -------------------------------------------------------------
def fit(pts: list[Pt], n_boot: int = 2000, seed: int = 0):
    """Full Chinchilla fit + bootstrap CIs. Returns (point_est, boot_samples, diag)."""
    if len(pts) < 4:
        return None, None, {"error": f"need >=4 pts for 4-param fit, got {len(pts)}"}
    N = np.array([p.N for p in pts])
    D = np.array([p.D for p in pts])
    y = np.array([p.bpb for p in pts])
    ND = np.vstack([N, D])
    # point estimate
    try:
        popt, pcov = curve_fit(chinchilla, ND, y, p0=P0_DEFAULT,
                                bounds=BOUNDS, maxfev=20000)
    except Exception as e:
        return None, None, {"error": f"curve_fit failed: {e}"}
    resid = y - chinchilla(ND, *popt)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    # bootstrap: resample residuals, refit
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        yb = chinchilla(ND, *popt) + rng.choice(resid - resid.mean(), size=len(y))
        try:
            pb, _ = curve_fit(chinchilla, ND, yb, p0=popt, bounds=BOUNDS, maxfev=20000)
            boots.append(pb)
        except Exception:
            pass
    boots = np.array(boots) if boots else np.empty((0, 5))
    diag = {
        "n_pts": len(pts), "r2": r2, "rmse": math.sqrt(ss_res/len(y)),
        "residuals": resid.tolist(),
        "point_est": dict(zip(["E","A","alpha","B","beta"], [round(v,5) for v in popt])),
        "boot_n": len(boots),
    }
    if len(boots):
        for i, k in enumerate(["E","A","alpha","B","beta"]):
            col = boots[:, i]
            diag[f"{k}_ci95"] = [round(float(np.percentile(col, 2.5)), 5),
                                  round(float(np.percentile(col, 97.5)), 5)]
            diag[f"{k}_median"] = round(float(np.median(col)), 5)
    return popt, boots, diag


def predict(popt, N, D):
    """Point prediction; pair with bootstrap for CI."""
    return float(chinchilla(np.array([[N],[D]]), *popt)[0])


# ---- CLI ---------------------------------------------------------------------
def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    pts = load_clean(root)
    print(f"=== CLEAN dataset ({len(pts)} points, eta*-calibrated) ===")
    for p in pts:
        print(f"  {p.name:20s} N={p.N/1e6:7.2f}M  D={p.D/1e9:.3f}B  bpb={p.bpb:.4f}  lr={p.lr}")
    if not pts:
        print("No clean points found yet."); return
    print("\n=== Local-slope diagnostics (robust at few points) ===")
    lb = local_beta(pts)
    if lb: print(f"  beta_local @ N={lb['N']/1e6:.0f}M: {lb['beta_local']:.3f}  "
                 f"(bpb {lb['bpb_a']:.4f}@{lb['D_a']/1e9:.2f}B -> {lb['bpb_b']:.4f}@{lb['D_b']/1e9:.2f}B)")
    else: print("  beta_local: need 2 D-points at same N (will compute after s7_confirm)")
    la = local_alpha(pts)
    if la: print(f"  alpha_local: {la['alpha_local']:.3f}  ({la['n_points']} N-rungs)")
    else: print("  alpha_local: need >=2 distinct N (will compute as ladder fills)")
    print("\n=== Full Chinchilla fit ===")
    if len(pts) < 5:
        print(f"  [NOTE: {len(pts)} pts -> global fit weakly identified; local-slope diagnostics above are primary]")
    popt, boots, diag = fit(pts)
    if diag.get("error"):
        print(f"  {diag['error']}"); return
    print(f"  n_pts={diag['n_pts']}  R2={diag['r2']:.4f}  RMSE={diag['rmse']:.5f}")
    pe = diag["point_est"]
    print(f"  point:   E={pe['E']:.4f}  A={pe['A']:.2f}  alpha={pe['alpha']:.4f}  B={pe['B']:.2f}  beta={pe['beta']:.4f}")
    if diag.get("boot_n"):
        print(f"  bootstrap ({diag['boot_n']} resamples), 95% CI:")
        for k in ["E","alpha","beta"]:
            lo, hi = diag[f"{k}_ci95"]
            print(f"    {k:6s}: {lo:.4f} .. {hi:.4f}  (median {diag[f'{k}_median']:.4f})")
    # d20 extrapolation with HONEST uncertainty (bootstrap CI propagated)
    if popt is not None and len(boots):
        # N(d20): d_model=64*20=1280. Extrapolate from d14 (N=195.23M, dm=896).
        # Body params ~ dm^2 * depth. d20/d14 ratio = (1280/896)^2 * (20/14).
        N_d20 = 195.23e6 * (1280/896)**2 * (20/14)
        print(f"\n=== d20 prediction (EXTRAPOLATION, N~{N_d20/1e6:.0f}M) ===")
        print(f"  (caveat: {len(pts)}-pt fit, CI is wide — report range not point)")
        for rname, Dratio in [("r8",8),("r20",20)]:
            D = Dratio * N_d20
            pt_pred = predict(popt, N_d20, D)
            boot_pred = [float(chinchilla(np.array([[N_d20],[D]]), *b)[0]) for b in boots]
            lo, hi = np.percentile(boot_pred, [2.5, 97.5])
            print(f"  d20 {rname}: {pt_pred:.4f} bpb  (95% CI {lo:.4f}..{hi:.4f})")
    print("\n=== Residuals (gate: |resid| > 2*RMSE flagged) ===")
    for p, r in zip(pts, diag["residuals"]):
            flag = " *** FLAG" if abs(r) > 2*diag["rmse"] and diag["rmse"]>0 else ""
            print(f"  {p.name:20s} pred={p.bpb-r:.4f} obs={p.bpb:.4f} resid={r:+.5f}{flag}")


if __name__ == "__main__":
    main()
