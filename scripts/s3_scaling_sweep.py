"""S3 scaling-law sweep CLI — plan / run / fit for the real-corpus IsoFLOP grid (E2E §S3).

    python scripts/s3_scaling_sweep.py plan [--grid v1|v2] [--batch 32] [--out PATH]
    python scripts/s3_scaling_sweep.py run --data-dir DATA [--grid v1|v2] [--points s1,s2,...]
        [--out-dir DIR] [--batch 32] [--device cuda] [--bf16] [--compile] [--seed 0]
    python scripts/s3_scaling_sweep.py fit [--out-dir DIR] [--results PATH] [--smooth|--no-smooth]

``plan`` emits the grid without running anything — v1 (default, the s1–s8 depth × ratio grid:
exact instantiated N, D = ratio × N, C = 6ND) or v2 (the budget × size repair grid: exact N,
D = C_target/(6N) rounded to whole optimizer steps, target vs effective C per point). ``run``
executes the selected points on the shard-backed pretrain path and appends each finished record
to ``<out-dir>/results.json`` (records carry the seed — dual-seed coverage is one invocation
per seed). ``fit`` loads results.json, averages replicate seeds on the bpb axis, fits
``N_opt ∝ C^a`` / ``D_opt ∝ C^b`` on val_bpb via scaling/isoflop.py (quadratic-in-log-N min-pick
by default; ``--no-smooth`` keeps the v1 raw argmin), runs the pre-registered gates
(a+b ∈ [0.95, 1.05], log-log R² ≥ 0.98), applies the D:N decision rule, and writes
``fit.json`` + ``fit.md`` with the nanochat oracle overlay.

The sweep runs on the standing GPU box; everything is CPU-safe for plan/fit and the wiring
tests. Logic lives in ``scratch_llm.scaling.s3_sweep`` (importable by tests) — this is the
thin argparse front-end, same pattern as scripts/f6_moe_ablation.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/s3_scaling_sweep.py` without an editable install.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scratch_llm.scaling.s3_sweep import (
    RESULTS_FILENAME,
    build_grid,
    build_grid_v2,
    fit_scaling_law,
    load_results,
    run_sweep,
    select_points,
    write_fit_outputs,
)

DEFAULT_OUT_DIR = REPO / "artifacts" / "s3_scaling_sweep"


def main() -> None:
    parser = argparse.ArgumentParser(description="S3 scaling-law sweep (E2E §S3)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="print/emit the grid as JSON (runs nothing)")
    p_plan.add_argument(
        "--grid",
        choices=["v1", "v2"],
        default="v1",
        help="v1: s1-s8 depth x ratio grid (default); v2: budget x size exact-C grid",
    )
    p_plan.add_argument(
        "--batch",
        type=int,
        default=32,
        help="batch size — v2 rounds D to whole steps of batch x ctx (default 32)",
    )
    p_plan.add_argument("--out", type=Path, default=None, help="also write the grid JSON here")

    p_run = sub.add_parser("run", help="execute grid points on the real-corpus pretrain path")
    p_run.add_argument(
        "--grid",
        choices=["v1", "v2"],
        default="v1",
        help="which grid to run (default v1; v2 points are b1_d4 .. b5_d16)",
    )
    p_run.add_argument(
        "--points",
        default=None,
        help="comma-separated grid points, e.g. s1,s2 (default: all s1-s8)",
    )
    p_run.add_argument(
        "--data-dir",
        required=True,
        help="shard dataset dir from data/shards.py (tokenizer.json + *.bin)",
    )
    p_run.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_run.add_argument(
        "--batch", type=int, default=32, help="batch size (steps = D / (batch x ctx))"
    )
    p_run.add_argument("--device", default="cpu")
    p_run.add_argument("--bf16", action="store_true", help="bf16 autocast (GPU)")
    p_run.add_argument(
        "--compile", action="store_true", help="torch.compile (not with --bf16 on sm120)"
    )
    p_run.add_argument("--seed", type=int, default=0)

    p_fit = sub.add_parser("fit", help="fit the scaling law from results.json + emit fit.json/md")
    p_fit.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p_fit.add_argument(
        "--results",
        type=Path,
        default=None,
        help="results JSON path (default: <out-dir>/results.json)",
    )
    p_fit.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="quadratic-in-log-N min-pick per budget (Chinchilla A2, default); "
        "--no-smooth keeps the v1 raw argmin",
    )

    args = parser.parse_args()

    if args.command == "plan":
        if args.grid == "v2":
            grid = [g.to_dict() for g in build_grid_v2(batch_size=args.batch)]
        else:
            grid = [g.to_dict() for g in build_grid()]
        text = json.dumps(grid, indent=2)
        print(text)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
            print(f"Saved grid: {args.out}")
        return

    if args.command == "run":
        grid = build_grid_v2(batch_size=args.batch) if args.grid == "v2" else build_grid()
        points = select_points(grid, args.points) if args.points else grid
        run_sweep(
            points,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            batch_size=args.batch,
            device=args.device,
            seed=args.seed,
            bf16=args.bf16,
            compile_model=args.compile,
        )
        return

    # fit
    results_path = args.results if args.results is not None else args.out_dir / RESULTS_FILENAME
    records = load_results(results_path)
    if not records:
        raise SystemExit(f"no finished points in {results_path} — run grid points first")
    report = fit_scaling_law(records, smooth=args.smooth)
    json_path, md_path = write_fit_outputs(report, records, args.out_dir)
    print(report.to_markdown(records))
    print(f"Saved fit: {json_path} + {md_path}")


if __name__ == "__main__":
    main()
