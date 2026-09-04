"""E001 figure + ledger table. Reads the sweep JSON, writes the public figure.

    python -m experiments.plot_e001                      # after --run
    python -m experiments.plot_e001 --in results/x.json --out figs/e001

Three panels, because three panels IS the hypothesis test:

  [1] rel_err vs gate, one line per dtype, machine-eps floor drawn.
      H1 says the curves rise toward log_gate -> 0 (left).
      H2 says they rise toward log_gate -> -inf (right).
      H3 says the float64 curve is FLAT ON THE FLOOR at every gate.
  [2] rel_err vs cond(T), log-log, Pearson r annotated.   <- H1's discriminator
  [3] rel_err vs max(1/a), log-log, Pearson r annotated.  <- H2's discriminator

Panel 1 alone is the figure that goes in the write-up. Panels 2-3 are what turn
"the curve goes up" into "the curve goes up BECAUSE", which is the only version
of this result anyone will cite.
"""

from __future__ import annotations

import argparse
import json
import math
import os

DTYPE_STYLE = {
    "float64": dict(color="#1f77b4", marker="o", label="float64"),
    "float32": dict(color="#2ca02c", marker="s", label="float32"),
    "bfloat16": dict(color="#d62728", marker="^", label="bfloat16"),
}
EPS = {"float64": 2.22e-16, "float32": 1.19e-7, "bfloat16": 7.81e-3}


def _pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _lg(v, floor=1e-30):
    return math.log10(max(v, floor))


def markdown_table(pts):
    """Paste-ready ledger rows -- no retyping numbers by hand, which is how
    transcription errors get into a public result."""
    gates = sorted({p["log_gate"] for p in pts}, reverse=True)
    dts = [d for d in ("float64", "float32", "bfloat16") if any(p["dtype"] == d for p in pts)]
    out = [
        "| log_gate | " + " | ".join(dts) + " | cond(T) max | max 1/a |",
        "|---" * (len(dts) + 3) + "|",
    ]
    for g in gates:
        row = [f"`{g:g}`"]
        for d in dts:
            m = [p for p in pts if p["log_gate"] == g and p["dtype"] == d]
            row.append(f"{m[0]['rel_err_o']:.3e}" if m else "—")
        anyp = [p for p in pts if p["log_gate"] == g]
        row += [f"{anyp[0]['cond_T_max']:.3e}", f"{anyp[0]['max_inv_a']:.3e}"]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/e001_gate_sweep.json")
    ap.add_argument("--out", default="figs/e001_divergence")
    a = ap.parse_args()

    with open(a.inp) as f:
        blob = json.load(f)
    pts, meta = blob["points"], blob.get("meta", {})

    print("\n" + markdown_table(pts) + "\n")
    tbl = a.out + "_table.md"
    os.makedirs(os.path.dirname(tbl) or ".", exist_ok=True)
    with open(tbl, "w") as f:
        f.write(
            f"<!-- oracle: {meta.get('oracle')}  solve_dtype: {meta.get('solve_dtype')} -->\n\n"
        )
        f.write(markdown_table(pts) + "\n")
    print(f"  ledger table -> {tbl}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed -- table written, figure skipped)")
        print("  uv pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # --- Panel 1: the money figure -------------------------------------------
    ax = axes[0]
    gates = sorted({p["log_gate"] for p in pts}, reverse=True)
    xs = list(range(len(gates)))
    for d, st in DTYPE_STYLE.items():
        ys = []
        for g in gates:
            m = [p for p in pts if p["log_gate"] == g and p["dtype"] == d]
            ys.append(m[0]["rel_err_o"] if m else float("nan"))
        if all(y != y for y in ys):
            continue
        ax.plot(xs, ys, lw=1.8, ms=5, **st)
        ax.axhline(EPS[d], color=st["color"], ls=":", lw=0.9, alpha=0.55)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g:g}" for g in gates], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("log_gate      (left: α→1 'remember'   →   right: α<1 'forget')")
    ax.set_ylabel("‖O_chunked − O_ref‖ / ‖O_ref‖   (vs fp64 oracle)")
    ax.set_title(
        "Chunked vs recurrent divergence\n(dotted = machine eps for that dtype)", fontsize=10
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)

    # --- Panels 2 & 3: mechanism ---------------------------------------------
    for ax, key, name, hyp in (
        (axes[1], "cond_T_max", "cond(T)  — triangular system", "H1"),
        (axes[2], "max_inv_a", "max(1/a)  — gate rescaling", "H2"),
    ):
        allx, ally = [], []
        for d, st in DTYPE_STYLE.items():
            sel = [
                p for p in pts if p["dtype"] == d and math.isfinite(p[key]) and p["rel_err_o"] > 0
            ]
            if not sel:
                continue
            x = [p[key] for p in sel]
            y = [p["rel_err_o"] for p in sel]
            ax.scatter(
                x,
                y,
                s=34,
                alpha=0.8,
                edgecolors="none",
                color=st["color"],
                marker=st["marker"],
                label=st["label"],
            )
            if d == "bfloat16":
                allx += [_lg(t) for t in x]
                ally += [_lg(t) for t in y]
        r = _pearson(allx, ally)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("rel_err_o")
        ax.set_title(f"{hyp} discriminator — bf16 log-log r = {r:.3f}", fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"E001 — gated delta rule, chunked (WY) vs recurrent    "
        f"oracle: fp64 sequential    solve_dtype: {meta.get('solve_dtype')}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{a.out}.{ext}", dpi=160)
        print(f"  figure -> {a.out}.{ext}")


if __name__ == "__main__":
    main()
