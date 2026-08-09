"""P5.5 d20 probe CLI — plan / run / select for the target-scale LR gate (ADR-0020).

    python scripts/d20_probe.py plan
    torchrun --standalone --nproc_per_node=8 scripts/d20_probe.py run --mult 1.0 \
        --data-dir DATA --out-root artifacts/d20_probe
    python scripts/d20_probe.py select --out-root artifacts/d20_probe

``plan`` prints the three arms (LRs from the composite rule's √B transfer of η*=0.0021 to the
d20's 524,288-tok batch) plus the exact per-arm torchrun commands — runs nothing. ``run``
executes ONE arm (idempotent: a finished arm's results.json means skip); on the 8×H100 node it
is launched once per arm under torchrun (world size is read from the env — the same command is
a single-process run off torchrun). ``select`` applies the pre-registered rule (min val_bpb;
ties within 0.003 bpb break to the lower LR) and writes ``selection.json`` — its ``winner_lr``
is the ``--lr`` of the full-budget d20 launch.

Logic lives in ``scratch_llm.scaling.d20_probe`` (importable by tests) — this is the thin
argparse front-end, same pattern as scripts/s3_scaling_sweep.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/d20_probe.py` without an editable install.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scratch_llm.scaling.d20_probe import (  # noqa: E402
    D20_CONTEXT_LENGTH,
    D20_GLOBAL_BATCH_TOKENS,
    D20_PER_GPU_BATCH,
    PROBE_TOKENS,
    build_arms,
    center_lr,
    load_selection,
    run_arm,
    steps_for_probe,
    write_selection,
)

DEFAULT_OUT_ROOT = REPO / "artifacts" / "d20_probe"


def main() -> None:
    parser = argparse.ArgumentParser(description="P5.5 d20 target-scale LR probe (ADR-0020)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="print the arms + exact torchrun commands (runs nothing)")

    p_run = sub.add_parser("run", help="run ONE arm (idempotent; torchrun-aware)")
    p_run.add_argument("--mult", type=float, required=True, choices=[a.mult for a in build_arms()])
    p_run.add_argument(
        "--data-dir",
        required=True,
        help="shard dataset dir (the staged d20 corpus — ClimbMix per the F12 final decision)",
    )
    p_run.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p_run.add_argument("--device", default="cuda")
    p_run.add_argument("--batch", type=int, default=D20_PER_GPU_BATCH, help="per-rank sequences")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast")
    p_run.add_argument(
        "--compile", action="store_true", help="torch.compile (only if the sm90 re-test passed)"
    )

    p_sel = sub.add_parser("select", help="pick the winner from finished arms")
    p_sel.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)

    args = parser.parse_args()
    arms = build_arms()

    if args.command == "plan":
        steps = steps_for_probe(PROBE_TOKENS, D20_GLOBAL_BATCH_TOKENS)
        print(
            f"P5.5 d20 probe — center lr = η*·√32 = {center_lr():.6f} "
            f"(η*=0.0021, batch 16,384 → 524,288 tok/step)"
        )
        print(
            f"per arm: {PROBE_TOKENS:,} tokens = {steps} steps @ global batch "
            f"{D20_GLOBAL_BATCH_TOKENS:,} (8×{D20_PER_GPU_BATCH}×{D20_CONTEXT_LENGTH})\n"
        )
        for arm in arms:
            print(f"  {arm.name}: mult={arm.mult} lr={arm.lr:.6f}")
            print(
                f"    torchrun --standalone --nproc_per_node=8 scripts/d20_probe.py run "
                f"--mult {arm.mult} --data-dir <D20_CORPUS> --out-root {DEFAULT_OUT_ROOT}"
            )
        print(f"\n  then: python scripts/d20_probe.py select --out-root {DEFAULT_OUT_ROOT}")
        return

    if args.command == "run":
        arm = next(a for a in arms if a.mult == args.mult)
        record = run_arm(
            arm,
            out_root=args.out_root,
            data_dir=args.data_dir,
            device=args.device,
            batch_size=args.batch,
            seed=args.seed,
            bf16=not args.no_bf16,
            compile_model=args.compile,
        )
        # Every rank ran; rank 0's line is the audit trail (all ranks hold the same record).
        print(
            f"[{arm.name}] lr={arm.lr:.6f} steps={record['steps']} "
            f"val_bpb={record['val_bpb']:.4f} val_loss={record['val_loss']:.4f} "
            f"wall={record['wall_s'] / 60:.1f} min world={record['world_size']}"
        )
        return

    # select
    selection = load_selection(args.out_root)
    path = write_selection(selection, args.out_root)
    for r in selection.records:
        marker = " <-- WINNER" if r["arm"] == selection.winner.name else ""
        tied = " (tied)" if r["arm"] in selection.tied else ""
        print(f"  {r['arm']}: lr={r['lr']:.6f} val_bpb={r['val_bpb']:.4f}{tied}{marker}")
    tie_note = (
        f" — tie within band, broke to lower LR (tied: {selection.tied})" if selection.tied else ""
    )
    print(
        f"\nP5.5 verdict: commit the d20 at --lr {selection.winner.lr:.6f} "
        f"(val_bpb {selection.winner_bpb:.4f}){tie_note}"
    )
    print(f"Saved selection: {path}")


if __name__ == "__main__":
    main()
