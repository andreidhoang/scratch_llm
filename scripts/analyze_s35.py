"""S3.5 analysis driver — P5 LR verdict, batch de-confound pair, joint scaling-law fit.

Consumes the results synced back from the remote pod (per-run ``results.json`` files, each a
list of ``SweepRecord`` dicts) and emits the three pre-registered products of
``docs/FRONTIER_2026_SCALING_PROGRAM.md`` §1/§3 + ``FRONTIER_2026_D20_CERTAINTY_PLAN.md`` §6:

    python scripts/analyze_s35.py p5    # LR-sweep bowl, tie-break winner pick, width probe
    python scripts/analyze_s35.py pair  # batch4-vs-batch8 s7 de-confound verdict
    python scripts/analyze_s35.py fit   # joint law + the four §1.4 gates + d20 prediction
    python scripts/analyze_s35.py all   # p5 + pair + fit (winner auto-picked from the sweep)

Options: ``--artifacts-root`` redirects the artifact tree (default ``artifacts/`` — the
synthetic-tree smoke tests point it at a tmp dir); ``--winner-lr`` pins the P5 winner for
``fit``/``all`` instead of auto-picking it from the sweep with the §6 tie-break.

Data layout under the artifacts root (every run dir holds one ``results.json``):
``s3_rerun_pair/{batch4,batch8}``, ``p5/lr{0.5,0.7,1.0,1.4,2.0}`` (d12 @ ratio-4 sweep),
``p5/confirm_r8`` (d12 @ ratio-8 at the winner), ``p5/width_lr{0.7,1.0,1.4}`` (d8 @ ratio-4
probe at multipliers OF THE WINNER), ``s35/ladder_s1..s6``, ``s35/d12_r20``, ``s35/d14_r8``,
and optionally ``s35/d14_r20``. Missing dirs are skipped with a warning — the driver analyzes
what is there and never crashes on partial sync-backs.

The fit/gates machinery is ``scratch_llm.scaling.joint_fit`` (Hoffmann Approach 3 + the
Besiroglu replication fixes + the four pre-registered §1.4 gates) — this script owns loading,
verdicts, and charts only; nothing is reimplemented here. ``s3_sweep`` itself is never
imported at runtime (it pulls in the torch training stack); the record type is TYPE_CHECKING
-only, the same discipline as ``joint_fit.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow `python scripts/analyze_s35.py` without an editable install.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scratch_llm.scaling.joint_fit import (
    JointLaw,
    bootstrap_prediction_interval,
    evaluate_gates,
    fit_joint_law,
    r2_log_l,
    residual_trend,
)

if TYPE_CHECKING:
    from scratch_llm.scaling.s3_sweep import SweepRecord

RESULTS_FILENAME = "results.json"
OUT_DIRNAME = "s35_analysis"

# Recipe constants (FRONTIER_2026_D20_CERTAINTY_PLAN.md §6): the base recipe peak LR, the
# pre-registered tie band (single-seed val_bpb noise), and the d12 sweep grid multipliers.
BASE_LR = 3e-3
TIE_BAND_BPB = 0.003
P5_MULTIPLIERS = (0.5, 0.7, 1.0, 1.4, 2.0)
WIDTH_MULTIPLIERS = (0.7, 1.0, 1.4)
# Run-dir names as written by the pod driver — pinned literally (f"{1.0:g}" would give "lr1",
# but the on-disk dirs are "lr1.0").
P5_SWEEP_DIRS = {0.5: "lr0.5", 0.7: "lr0.7", 1.0: "lr1.0", 1.4: "lr1.4", 2.0: "lr2.0"}
WIDTH_PROBE_DIRS = {0.7: "width_lr0.7", 1.0: "width_lr1.0", 1.4: "width_lr1.4"}

# The d20 target (s3_sweep.py pins the same numbers; duplicated here so this script — like
# joint_fit.py — never imports the torch-pulling s3_sweep module at runtime).
D20_COMPUTE_FLOPS = 2.77e19
D20_PARAMS = 480.4e6
D20_TOKENS = 9.6e9

# The historical s7 reference (old tokenizer instance — reference only, never in the fit).
HISTORICAL_S7_BPB = 0.9402
HISTORICAL_RESULTS = Path("s3_scaling_sweep") / RESULTS_FILENAME

DEPTH_COLORS = {
    4: "#4C9BD6",
    8: "#2E8B57",
    12: "#D64545",
    14: "#7B5EA7",
    16: "#E8A33D",
    20: "#FFD700",
}


# ---------------------------------------------------------------------------
# Loading (graceful: missing/empty/multi-record files warn, never crash)
# ---------------------------------------------------------------------------


def load_last_record(run_dir: Path) -> SweepRecord | None:
    """The last record of ``<run_dir>/results.json`` (append semantics), or None with a
    printed warning when the file is absent or empty."""
    path = run_dir / RESULTS_FILENAME
    if not path.exists():
        print(f"WARNING: {path} not found — skipping this run")
        return None
    records = json.loads(path.read_text())
    if not records:
        print(f"WARNING: {path} is empty — skipping this run")
        return None
    if len(records) > 1:
        print(f"WARNING: {path} holds {len(records)} records — taking the last (append semantics)")
    return cast("SweepRecord", records[-1])


def record_lr(record: SweepRecord) -> float:
    """Effective peak LR: the record's ``lr`` override, else the recipe default 3e-3."""
    return float(record.get("lr", BASE_LR))


def record_multiplier(record: SweepRecord, base_lr: float = BASE_LR) -> float:
    return record_lr(record) / base_lr


def load_lr_sweep(root: Path, names: dict[float, str]) -> list[tuple[float, SweepRecord]]:
    """Load an LR sweep as (multiplier-of-base, record) pairs, sorted by multiplier."""
    points: list[tuple[float, SweepRecord]] = []
    for mult, dirname in names.items():
        rec = load_last_record(root / dirname)
        if rec is not None:
            points.append((mult, rec))
    return sorted(points, key=lambda t: t[0])


# ---------------------------------------------------------------------------
# p5 — the LR-sweep bowl, the §6 tie-break winner pick, the width probe
# ---------------------------------------------------------------------------


def pick_winner(
    sweep: list[tuple[float, SweepRecord]],
) -> tuple[float, SweepRecord, list[float]]:
    """The §6 tie-break, exactly: every grid point within 0.003 bpb of the minimum is a tie;
    ties break to the LOWER LR (cheap insurance against the horizon shift to the d20's
    18,311 steps). Returns (winner multiplier, winner record, tied multipliers)."""
    if not sweep:
        raise ValueError("empty sweep — no winner to pick")
    min_bpb = min(r["val_bpb"] for _, r in sweep)
    tied = sorted((m, r) for m, r in sweep if r["val_bpb"] <= min_bpb + TIE_BAND_BPB)
    winner_mult, winner_rec = min(tied, key=lambda t: record_lr(t[1]))
    return winner_mult, winner_rec, [m for m, _ in tied]


def bowl_is_well_formed(sweep: list[tuple[float, SweepRecord]], min_mult: float) -> bool:
    """Well-formed bowl: the argmin is interior (both grid sides higher than the min).
    A minimum at a grid edge means the basin was not bracketed."""
    mults = [m for m, _ in sweep]
    return min(mults) < min_mult < max(mults)


def _fmt_mult(x: float) -> str:
    return f"{x:g}"


def plot_p5_bowl(
    sweep: list[tuple[float, SweepRecord]],
    width: list[tuple[float, SweepRecord]],
    winner_lr: float | None,
    out_path: Path,
) -> None:
    """Two panels: the d12 sweep (bpb vs LR, log-x, tie band shaded) and the d8 width probe
    on the multiplier-of-winner axis."""
    fig, (ax, axw) = plt.subplots(1, 2, figsize=(12, 4.8))

    lrs = [record_lr(r) for _, r in sweep]
    bpbs = [r["val_bpb"] for _, r in sweep]
    ax.set_xscale("log")
    ax.plot(lrs, bpbs, "o-", color="#E8A33D", mec="k", zorder=5)
    for (mult, _), lr, bpb in zip(sweep, lrs, bpbs, strict=True):
        ax.annotate(
            f"×{_fmt_mult(mult)}",
            (lr, bpb),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    min_bpb = min(bpbs)
    ax.axhspan(min_bpb, min_bpb + TIE_BAND_BPB, color="#2E8B57", alpha=0.18)
    ax.text(
        lrs[0],
        min_bpb + TIE_BAND_BPB * 0.25,
        f"tie zone < {TIE_BAND_BPB} bpb → pick the LOWER LR",
        fontsize=8.5,
        color="#1d5c39",
    )
    if winner_lr is not None:
        ax.axvline(winner_lr, color="#8B0000", ls="--", lw=1.2)
        ax.text(
            winner_lr,
            max(bpbs),
            f" winner {winner_lr:g}",
            rotation=90,
            fontsize=8.5,
            color="#8B0000",
            va="top",
        )
    ax.set_xlabel("peak LR (log)")
    ax.set_ylabel("val bpb")
    ax.set_title("P5 core sweep — d12 @ ratio-4")
    ax.grid(alpha=0.2)

    if width and winner_lr is not None:
        wmults = [record_lr(r) / winner_lr for _, r in width]
        wbpbs = [r["val_bpb"] for _, r in width]
        axw.plot(wmults, wbpbs, "s-", color="#F2C14E", mec="k", zorder=5)
        for m, b in zip(wmults, wbpbs, strict=True):
            axw.annotate(
                f"×{_fmt_mult(m)}",
                (m, b),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        axw.axvline(1.0, color="#8B0000", ls="--", lw=1.2)
        axw.set_xlabel("LR multiplier of the d12 winner")
        axw.set_ylabel("val bpb")
        axw.set_title("P5 width probe — d8 @ ratio-4")
    else:
        axw.text(
            0.5,
            0.5,
            "width probe data not synced yet",
            ha="center",
            va="center",
            transform=axw.transAxes,
            color="0.5",
        )
        axw.set_title("P5 width probe — d8 @ ratio-4 (pending)")
    axw.grid(alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def analyze_p5(root: Path, out_dir: Path) -> dict[str, float] | None:
    """The P5 verdict: sweep table, tie-break winner, bowl-shape check, width-probe drift.
    Returns {"winner_lr": ..., "winner_mult": ...} or None when the sweep is too partial."""
    sweep = load_lr_sweep(root / "p5", P5_SWEEP_DIRS)
    if len(sweep) < 2:
        print(
            f"WARNING: P5 sweep too partial ({len(sweep)}/{len(P5_MULTIPLIERS)} points) — "
            "no winner pick"
        )
        return None
    if len(sweep) < len(P5_MULTIPLIERS):
        print(
            f"WARNING: P5 sweep incomplete — {len(sweep)}/{len(P5_MULTIPLIERS)} grid points "
            "synced; the verdict is provisional"
        )

    print("\n== P5 LR sweep (d12 @ ratio-4) ==")
    print(f"{'mult':>6} {'lr':>9} {'val_bpb':>8}")
    for mult, rec in sweep:
        print(f"×{_fmt_mult(mult):>5} {record_lr(rec):>9.4g} {rec['val_bpb']:>8.4f}")

    winner_mult, winner_rec, tied = pick_winner(sweep)
    winner_lr = record_lr(winner_rec)
    min_bpb = min(r["val_bpb"] for _, r in sweep)
    argmin_mult = min(sweep, key=lambda t: t[1]["val_bpb"])[0]
    print(
        f"min bpb {min_bpb:.4f} at ×{_fmt_mult(argmin_mult)}; "
        f"tie band (+{TIE_BAND_BPB}) holds ×{', ×'.join(_fmt_mult(m) for m in tied)}"
    )
    print(
        f"P5 WINNER (tie-break = lower LR within {TIE_BAND_BPB} bpb): "
        f"×{_fmt_mult(winner_mult)} of base {BASE_LR:g} → lr = {winner_lr:g}"
    )

    if not bowl_is_well_formed(sweep, argmin_mult):
        print(
            "WARNING: optimum at grid edge ⇒ extend the grid — the bowl is NOT bracketed, "
            "the winner is not certified (FRONTIER_2026_D20_CERTAINTY_PLAN §6)"
        )
    else:
        print("bowl well-formed: argmin is interior, both grid sides higher than the min")

    drift_verdict = "width probe not synced — drift check pending"
    width = load_lr_sweep(root / "p5", WIDTH_PROBE_DIRS)
    if width:
        # Width points were run at multipliers OF THE WINNER, so normalize by the winner LR —
        # a perfect µP transfer puts the d8 bowl's argmin at multiplier 1.0.
        w_argmin_mult, _, _ = pick_winner(width)
        w_argmin_lr = record_lr(dict((m, r) for m, r in width)[w_argmin_mult])
        rel = w_argmin_lr / winner_lr
        drift_verdict = f"d8 bowl argmin at ×{_fmt_mult(rel)} of the d12 winner — " + (
            "µP LR transfer holds at d8 (no drift)"
            if abs(rel - 1.0) < 1e-6
            else f"DRIFT: d8 optimum sits at ×{rel:.2f}, not ×1.0 — re-check before d20"
        )
    print(f"width-probe verdict: {drift_verdict}")

    plot_p5_bowl(sweep, width, winner_lr, out_dir / "p5_lr_bowl.png")
    return {"winner_lr": winner_lr, "winner_mult": winner_mult}


# ---------------------------------------------------------------------------
# pair — the s7 batch de-confound (batch4 vs batch8, same point/tokenizer/seed)
# ---------------------------------------------------------------------------


def analyze_pair(root: Path) -> dict[str, float] | None:
    """The batch de-confound verdict: if batch8 ≤ batch4 meaningfully, the batch-size
    confound explains the original s7 anomaly. Also quotes the 0.9402 historical s7 record
    (old tokenizer instance — reference only)."""
    b4 = load_last_record(root / "s3_rerun_pair" / "batch4")
    b8 = load_last_record(root / "s3_rerun_pair" / "batch8")
    if b4 is None or b8 is None:
        print("WARNING: batch de-confound pair incomplete — verdict pending")
        return None

    bpb4, bpb8 = float(b4["val_bpb"]), float(b8["val_bpb"])
    delta = bpb8 - bpb4
    print("\n== s7 batch de-confound (d12 @ ratio-8, fresh tokenizer/seed, only batch differs) ==")
    print(f"batch4 val_bpb = {bpb4:.4f}")
    print(f"batch8 val_bpb = {bpb8:.4f}")
    print(f"delta (batch8 − batch4) = {delta:+.4f} bpb")
    if bpb8 <= bpb4 - TIE_BAND_BPB:
        verdict = (
            f"batch8 beats batch4 by {-delta:.4f} bpb (> {TIE_BAND_BPB} noise band) — "
            "the batch-size confound EXPLAINS the original s7 anomaly"
        )
    elif abs(delta) <= TIE_BAND_BPB:
        verdict = (
            f"within the {TIE_BAND_BPB} bpb noise band — batch size is NOT the "
            "driver of the s7 anomaly (or its effect is below single-seed noise)"
        )
    else:
        verdict = (
            f"batch8 is WORSE by {delta:.4f} bpb — the batch-size confound does NOT "
            "explain the s7 anomaly; investigate before the re-fit leans on s7"
        )
    print(f"verdict: {verdict}")

    hist_path = root / HISTORICAL_RESULTS
    if hist_path.exists():
        hist = [r for r in json.loads(hist_path.read_text()) if r.get("point") == "s7"]
        if hist:
            print(
                f"historical s7 reference: val_bpb = {float(hist[-1]['val_bpb']):.4f} "
                "(OLD tokenizer instance — reference only, never comparable head-to-head)"
            )
    else:
        print(
            f"WARNING: {hist_path} not found — historical s7 reference unavailable "
            f"(expected ≈{HISTORICAL_S7_BPB}, old tokenizer)"
        )
    return {"batch4": bpb4, "batch8": bpb8, "delta": delta}


# ---------------------------------------------------------------------------
# fit — the S3.5 joint law, the four §1.4 gates, the d20 prediction
# ---------------------------------------------------------------------------


def _p5_record_at_lr(root: Path, winner_lr: float) -> SweepRecord | None:
    """The P5 sweep record whose peak LR matches the winner — found by LR match, not by
    dir-name arithmetic, so a user-pinned ``--winner-lr`` works regardless of grid naming."""
    for dirname in P5_SWEEP_DIRS.values():
        path = root / "p5" / dirname / RESULTS_FILENAME
        if not path.exists():
            continue
        rec = load_last_record(root / "p5" / dirname)
        if rec is not None and abs(record_lr(rec) - winner_lr) <= 1e-9 * max(1.0, winner_lr):
            return rec
    print(
        f"WARNING: no p5 sweep record at the winner LR {winner_lr:g} — the d12@r4 winner "
        "point is absent from the fit set"
    )
    return None


def assemble_fit_set(root: Path, winner_lr: float) -> list[SweepRecord]:
    """The pre-registered fit set (SCALING_PROGRAM §3 stage 2): the s1–s6 ladder refreshed at
    the winner LR, the P5 winner point itself (d12@r4), the d8 width probe at ×1.0 of the
    winner, the d12@r8 confirmation (T2 anchor), and the d12-r20 / d14-r8 [/ d14-r20]
    extension points. Missing dirs skip with a warning."""
    slots: list[tuple[str, Path]] = [
        (f"ladder_s{i}", root / "s35" / f"ladder_s{i}") for i in range(1, 7)
    ]
    slots += [
        ("width_lr1.0 (d8@r4)", root / "p5" / WIDTH_PROBE_DIRS[1.0]),
        ("confirm_r8 (d12@r8)", root / "p5" / "confirm_r8"),
        ("d12_r20", root / "s35" / "d12_r20"),
        ("d14_r8", root / "s35" / "d14_r8"),
        ("d14_r20 (optional)", root / "s35" / "d14_r20"),
    ]
    records: list[SweepRecord] = []
    winner_rec = _p5_record_at_lr(root, winner_lr)
    if winner_rec is None:
        print("  fit-set slot missing: p5_winner (d12@r4)")
    for label, run_dir in slots:
        rec = load_last_record(run_dir)
        if rec is not None:
            records.append(rec)
        else:
            print(f"  fit-set slot missing: {label}")
    if winner_rec is not None:
        # Keep the fit set in compute order: the winner point (C ≈ 4.4e17) slots after the
        # ladder's small points and before confirm_r8.
        records.append(winner_rec)
        records.sort(key=lambda r: float(r["compute"]))
    return records


def check_recipe_v1(records: list[SweepRecord], winner_lr: float) -> list[str]:
    """Every fit point must sit at the winner LR — records without an ``lr`` field are at the
    recipe default 3e-3 (fine iff the winner IS 3e-3). Returns the violation messages."""
    violations: list[str] = []
    for r in records:
        eff = record_lr(r)
        if abs(eff - winner_lr) > 1e-9 * max(1.0, winner_lr):
            origin = "default 3e-3 (no lr field)" if "lr" not in r else f"lr={eff:g}"
            violations.append(f"{r['point']}: {origin} ≠ winner {winner_lr:g}")
    return violations


def print_fit_table(records: list[SweepRecord]) -> None:
    print("\n| point | depth | N | ratio | val_bpb | lr |")
    print("|---|---|---|---|---|---|")
    for r in records:
        ratio = float(r["ratio"])
        ratio_s = str(int(ratio)) if ratio.is_integer() else f"{ratio:.2f}"
        print(
            f"| {r['point']} | {r['depth']} | {r['n_params']:,} | {ratio_s} "
            f"| {r['val_bpb']:.4f} | {record_lr(r):g} |"
        )


def plot_fit(
    records: list[SweepRecord],
    law: JointLaw,
    pred_d20: float,
    pi_lo: float,
    pi_hi: float,
    out_path: Path,
) -> None:
    """bpb vs compute (log-x): points colored by depth, the fitted law as an iso-N line per
    measured depth (D = C/6N), and the d20 target star carrying its bootstrap PI."""
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set_xscale("log")
    c_lo = min(float(r["compute"]) for r in records)
    c_grid = np.logspace(np.log10(c_lo), np.log10(D20_COMPUTE_FLOPS), 200)

    by_depth: dict[int, list[SweepRecord]] = {}
    for r in records:
        by_depth.setdefault(int(r["depth"]), []).append(r)
    for depth, rs in sorted(by_depth.items()):
        color = DEPTH_COLORS.get(depth, "0.4")
        n = float(rs[0]["n_params"])
        ax.plot(
            c_grid,
            [law.predict(n, c / (6.0 * n)) for c in c_grid],
            "-",
            color=color,
            lw=1.2,
            alpha=0.7,
        )
        for r in rs:
            ax.scatter(
                float(r["compute"]),
                float(r["val_bpb"]),
                s=90,
                color=color,
                edgecolor="k",
                linewidth=0.7,
                zorder=5,
            )
            ax.annotate(
                f"{r['point']}\n{r['val_bpb']:.3f}",
                (float(r["compute"]), float(r["val_bpb"])),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=8,
            )
        ax.text(
            c_grid[-1] * 0.9,
            law.predict(n, D20_COMPUTE_FLOPS / (6.0 * n)),
            f"d{depth}",
            fontsize=8,
            color=color,
            ha="right",
        )
    # The d20's own N as a dashed guide the target star sits on.
    ax.plot(
        c_grid,
        [law.predict(D20_PARAMS, c / (6.0 * D20_PARAMS)) for c in c_grid],
        "--",
        color="0.5",
        lw=1.0,
    )
    ax.text(
        c_grid[0],
        law.predict(D20_PARAMS, c_lo / (6.0 * D20_PARAMS)),
        "d20 N (extrapolated)",
        fontsize=8,
        color="0.5",
    )

    ax.errorbar(
        [D20_COMPUTE_FLOPS],
        [pred_d20],
        yerr=[[pred_d20 - pi_lo], [pi_hi - pred_d20]],
        fmt="*",
        ms=20,
        color="#FFD700",
        mec="k",
        ecolor="k",
        elinewidth=1.4,
        capsize=5,
        zorder=7,
    )
    ax.annotate(
        f"d20 target\n{pred_d20:.3f} [{pi_lo:.3f}, {pi_hi:.3f}]",
        (D20_COMPUTE_FLOPS, pred_d20),
        textcoords="offset points",
        xytext=(-10, 22),
        fontsize=9,
        fontweight="bold",
        ha="right",
    )
    ax.set_xlabel("compute C = 6ND (FLOPs, log)")
    ax.set_ylabel("val bpb")
    ax.set_title("S3.5 joint fit — L(N,D) = E + A·N^(−α) + B·D^(−β), iso-N lines per depth")
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_residuals(records: list[SweepRecord], law: JointLaw, out_path: Path) -> None:
    """Log-residuals vs log-compute — the structured-residual visual behind gate (iv): a
    certified fit shows scatter around zero, not a monotone/zigzag drift."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xscale("log")
    for r in records:
        res = np.log(float(r["val_bpb"])) - np.log(
            law.predict(float(r["n_params"]), float(r["tokens"]))
        )
        color = DEPTH_COLORS.get(int(r["depth"]), "0.4")
        ax.scatter(
            float(r["compute"]), res, s=90, color=color, edgecolor="k", linewidth=0.7, zorder=5
        )
        ax.annotate(
            r["point"],
            (float(r["compute"]), res),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.axhline(0.0, color="k", lw=1)
    rho, p_value = residual_trend(records, law)
    ax.set_xlabel("compute C = 6ND (FLOPs, log)")
    ax.set_ylabel("log residual  ln L_true − ln L_pred")
    ax.set_title(
        f"S3.5 residuals vs compute — Spearman ρ = {rho:+.2f} (p = {p_value:.3f}); "
        "no strong+significant monotone trend allowed"
    )
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_optimals(records: list[SweepRecord], law: JointLaw, out_path: Path) -> None:
    """N_opt(C) and D_opt/N_opt(C) from the fitted law's closed form, out to the d20's C."""
    c_lo = min(float(r["compute"]) for r in records)
    c_grid = np.logspace(np.log10(c_lo), np.log10(D20_COMPUTE_FLOPS), 120)
    n_opts, ratios = [], []
    for c in c_grid:
        n_opt, d_opt, _ = law.predict_optimal(float(c))
        n_opts.append(n_opt)
        ratios.append(d_opt / n_opt)

    fig, (axn, axr) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True)
    axn.plot(c_grid, n_opts, "-", color="#4C9BD6", lw=1.8)
    axn.set_yscale("log")
    axn.set_ylabel("N_opt(C) — params (log)")
    axr.plot(c_grid, ratios, "-", color="#D64545", lw=1.8)
    axr.set_ylabel("D_opt / N_opt at C")
    axr.set_xlabel("compute C (FLOPs, log)")
    for ax in (axn, axr):
        ax.set_xscale("log")
        ax.axvline(D20_COMPUTE_FLOPS, color="crimson", ls="--", lw=1.2)
        ax.grid(True, which="both", alpha=0.18)
    n20, d20, _ = law.predict_optimal(D20_COMPUTE_FLOPS)
    axn.scatter(
        [D20_COMPUTE_FLOPS], [n20], marker="*", s=220, color="#FFD700", edgecolor="k", zorder=6
    )
    axn.annotate(
        f"d20: N_opt = {n20:.2e}",
        (D20_COMPUTE_FLOPS, n20),
        textcoords="offset points",
        xytext=(-10, -18),
        ha="right",
        va="top",
        fontsize=9,
    )
    axr.scatter(
        [D20_COMPUTE_FLOPS],
        [d20 / n20],
        marker="*",
        s=220,
        color="#FFD700",
        edgecolor="k",
        zorder=6,
    )
    axr.annotate(
        f"ratio {d20 / n20:.1f}",
        (D20_COMPUTE_FLOPS, d20 / n20),
        textcoords="offset points",
        xytext=(-10, -16),
        ha="right",
        va="top",
        fontsize=9,
    )
    axn.set_title("S3.5 compute-optimal frontier from the joint fit (d20 marked)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def analyze_fit(
    root: Path,
    out_dir: Path,
    winner_lr: float,
    winner_mult: float | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, object] | None:
    """The S3.5 joint fit: recipe-v1 check, fit_joint_law, the four §1.4 gates, the d20
    prediction with its bootstrap PI, the three charts, and the machine-readable JSON."""
    mult = winner_mult if winner_mult is not None else winner_lr / BASE_LR
    records = assemble_fit_set(root, winner_lr)
    if len(records) < 5:
        print(
            f"WARNING: only {len(records)} fit points synced (need ≥ 5 for 5 parameters) — "
            "fit skipped"
        )
        return None

    violations = check_recipe_v1(records, winner_lr)
    if violations:
        print(
            "WARNING: recipe-v1 VIOLATIONS in the fit set — these points are not at the "
            f"P5 winner LR {winner_lr:g}:"
        )
        for v in violations:
            print(f"  - {v}")
    else:
        print(f"recipe-v1 check: all {len(records)} fit points at the winner LR {winner_lr:g}")

    print_fit_table(records)

    law = fit_joint_law(records)
    report = evaluate_gates(
        records, law, n_params=D20_PARAMS, tokens=D20_TOKENS, n_boot=n_boot, seed=seed
    )
    pi = bootstrap_prediction_interval(
        records, D20_PARAMS, D20_TOKENS, n_boot=n_boot, seed=seed, parent=law
    )
    pred_d20 = law.predict(D20_PARAMS, D20_TOKENS)
    n_opt, d_opt, l_opt = law.predict_optimal(D20_COMPUTE_FLOPS)

    print("\n== S3.5 joint fit (Approach 3 + Besiroglu fixes) ==")
    print(
        f"L(N,D) = {law.E:.4f} + {law.A:.4g}·N^(−{law.alpha:.4f}) + {law.B:.4g}·D^(−{law.beta:.4f})"
    )
    print(
        f"objective {law.objective:.3e}; {law.n_converged}/{law.n_starts} starts in the "
        "best basin (1 would mean the answer depends on where the optimizer started)"
    )
    print(f"log-L R² = {r2_log_l(records, law):.4f}")
    print("\n-- §1.4 acceptance gates --")
    for g in report.gates:
        print(
            f"[{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.value:.4g} vs {g.threshold:g} "
            f"— {g.detail}"
        )
    verdict = (
        "CERTIFIED — quote within span; extrapolation carries its interval"
        if report.all_passed
        else "REJECTED — report the non-fit; the D:N decision stays HOLD-by-economics"
    )
    print(f"\nOVERALL: {verdict}")
    print(
        f"\npredict_optimal(C = {D20_COMPUTE_FLOPS:.2e}): N_opt = {n_opt:.3e}, "
        f"D_opt = {d_opt:.3e} (ratio {d_opt / n_opt:.1f}), L_pred = {l_opt:.4f}"
    )
    print(
        f"d20 point (N = {D20_PARAMS:.3e}, D = {D20_TOKENS:.2e}): predicted bpb = "
        f"{pred_d20:.4f}, {pi.level:.0%} bootstrap PI [{pi.lo:.4f}, {pi.hi:.4f}] "
        f"(half-width {pi.half_width:.4f}; {pi.n_failed_fits}/{pi.n_boot} refits failed)"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_fit(records, law, pred_d20, pi.lo, pi.hi, out_dir / "s35_fit.png")
    plot_residuals(records, law, out_dir / "s35_residuals.png")
    plot_optimals(records, law, out_dir / "s35_optimals.png")

    payload: dict[str, object] = {
        "winner_lr": winner_lr,
        "winner_mult": mult,
        "recipe_v1_violations": violations,
        "fit_set": [dict(r) for r in records],
        "params": law.to_dict(),
        "gates": report.to_dict(),
        "d20": {
            "compute_flops": D20_COMPUTE_FLOPS,
            "n_params": D20_PARAMS,
            "tokens": D20_TOKENS,
            "predicted_bpb": pred_d20,
            "bootstrap_pi": pi.to_dict(),
            "n_opt": n_opt,
            "d_opt": d_opt,
            "optimal_ratio": d_opt / n_opt,
            "l_pred_at_optimal": l_opt,
        },
    }
    json_path = out_dir / "s35_fit.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {json_path}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _auto_winner(root: Path) -> tuple[float, float] | None:
    """Auto-pick the winner from the p5 sweep data (same §6 tie-break), without plotting."""
    sweep = load_lr_sweep(root / "p5", P5_SWEEP_DIRS)
    if len(sweep) < 2:
        return None
    winner_mult, winner_rec, _ = pick_winner(sweep)
    return record_lr(winner_rec), winner_mult


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO / "artifacts",
        help="artifact tree root (default: artifacts/)",
    )
    common.add_argument(
        "--winner-lr",
        type=float,
        default=None,
        help="pin the P5 winner LR for fit/all instead of auto-picking it",
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("p5", parents=[common], help="LR-sweep bowl + winner pick + width probe")
    sub.add_parser("pair", parents=[common], help="batch4/batch8 s7 de-confound verdict")
    sub.add_parser("fit", parents=[common], help="joint fit + §1.4 gates + d20 prediction")
    sub.add_parser("all", parents=[common], help="p5 + pair + fit in sequence")
    args = parser.parse_args(argv)

    root: Path = args.artifacts_root
    out_dir = root / OUT_DIRNAME

    if args.command == "p5":
        analyze_p5(root, out_dir)
        return
    if args.command == "pair":
        analyze_pair(root)
        return

    # fit / all need the winner LR.
    winner_lr: float | None = args.winner_lr
    winner_mult: float | None = None
    if args.command == "all":
        p5_out = analyze_p5(root, out_dir)
        analyze_pair(root)
        if winner_lr is None and p5_out is not None:
            winner_lr = p5_out["winner_lr"]
            winner_mult = p5_out["winner_mult"]
    if winner_lr is None:
        auto = _auto_winner(root)
        if auto is None:
            print(
                "ERROR: no --winner-lr given and the p5 sweep is too partial to auto-pick "
                "one — fit skipped"
            )
            return
        winner_lr, winner_mult = auto
        print(
            f"auto-picked P5 winner from the sweep data: lr = {winner_lr:g} "
            f"(×{_fmt_mult(winner_mult)})"
        )
    else:
        winner_mult = winner_lr / BASE_LR
    analyze_fit(root, out_dir, winner_lr, winner_mult)


if __name__ == "__main__":
    main()
