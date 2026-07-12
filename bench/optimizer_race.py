"""F1-run — the iso-FLOP Muon-vs-tuned-AdamW race, on real shards (GPU day driver).

Pre-registered in bench/RESULTS.md §Frontier ablations (07-04, recalibrated 07-09 BEFORE the run):
- Muon tokens-to-match in the **1.1–1.4× band vs an independently LR-TUNED AdamW baseline**
  (≈15–25% saving expected at 30–50M params, shrinking with N), or ≥0.02 nats lower at iso-FLOP.
- NS overhead <1% of wall (analytic bound); **KILL:** saving <5% vs the tuned baseline, divergence
  at the reused LR, or NS overhead >3%.
The tuned baseline is MANDATORY (arXiv 2509.02046: the 1.4–2× Muon headlines came from under-tuned
baselines) — this driver runs the sweep first, then the race, in one pre-committed protocol.

Run (sm120 box; bf16 EAGER — bf16+compile NaNs on this inductor, see RESULTS.md F4):
  python bench/optimizer_race.py --data-dir data/fineweb_edu --depth 8 --tokens 7e8 \
      --batch-size 32 --context-length 1024 --amp bf16 --device cuda --out f1_race.json
Resume past the sweep with a known winner:  --baseline-lr 6e-4
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from scratch_llm.data.shards import load_dataset_tokens, load_shard
from scratch_llm.eval.optimizer_race import RaceResult, run_race, sweep_lr
from scratch_llm.speedrun import model_config_for_depth
from scratch_llm.train import TrainConfig


def _detect_vocab_size(data_dir: Path) -> int:
    """Largest ``max_token_id + 1`` across shard sidecars — the safe embedding size."""
    paths = sorted(data_dir.glob("*.bin"))
    if not paths:
        raise FileNotFoundError(f"no shards in {data_dir}")
    return max(load_shard(p)[1].max_token_id for p in paths) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="shard dir (A1 data/shards.py layout)")
    ap.add_argument("--depth", type=int, default=8, help="nanochat scaling: d_model = 64·depth")
    ap.add_argument("--tokens", type=float, default=7e8, help="D — total training tokens per arm")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--context-length", type=int, default=1024)
    ap.add_argument("--vocab-size", type=int, default=0, help="0 = auto-detect from shard sidecars")
    ap.add_argument("--sweep-lrs", default="1e-4,2e-4,3e-4,6e-4,1e-3")
    ap.add_argument(
        "--sweep-frac", type=float, default=0.2, help="sweep horizon as a fraction of D"
    )
    ap.add_argument("--baseline-lr", type=float, default=0.0, help="skip the sweep, use this LR")
    ap.add_argument("--muon-lr", type=float, default=0.0, help="0 = reuse the tuned baseline LR")
    ap.add_argument("--val-frac", type=float, default=0.005, help="tail fraction held out for val")
    ap.add_argument("--eval-every", type=int, default=0, help="0 = auto (~50 evals per run)")
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="f1_race.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tokens = load_dataset_tokens(data_dir)
    vocab_size = args.vocab_size or _detect_vocab_size(data_dir)
    val_n = max(int(len(tokens) * args.val_frac), args.eval_batches * args.context_length + 1)
    train_data, val_data = tokens[:-val_n], tokens[-val_n:]

    tokens_per_step = args.batch_size * args.context_length
    max_steps = math.ceil(args.tokens / tokens_per_step)
    model_cfg = model_config_for_depth(args.depth, vocab_size, args.context_length)

    def _train_cfg(steps: int) -> TrainConfig:
        return TrainConfig(
            max_steps=steps,
            batch_size=args.batch_size,
            context_length=args.context_length,
            warmup_steps=max(10, steps // 100),
            eval_every=args.eval_every or max(1, steps // 50),
            eval_batches=args.eval_batches,
            amp_dtype=None if args.amp == "none" else args.amp,
            device=args.device,
            seed=args.seed,
            verbose=True,
        )

    # --- Arm 0: the tuned baseline (mandatory — an untuned baseline is a fake win) ------------
    if args.baseline_lr > 0:
        baseline_lr = args.baseline_lr
        sweep_rows = []
        print(f"sweep SKIPPED — using --baseline-lr {baseline_lr:g}")
    else:
        lrs = [float(x) for x in args.sweep_lrs.split(",")]
        sweep_steps = max(1, math.ceil(max_steps * args.sweep_frac))
        print(f"LR sweep: {lrs} at {sweep_steps} steps ({args.sweep_frac:.0%} horizon)")
        sweep = sweep_lr(model_cfg, _train_cfg(sweep_steps), train_data, val_data, lrs)
        sweep_rows = [
            {"lr": arm.lr, "final_val": arm.val_curve[-1][1], "wall_s": arm.wall_seconds}
            for arm in sweep.arms
        ]
        for row in sweep_rows:
            print(f"  lr {row['lr']:.1e} → val {row['final_val']:.4f} ({row['wall_s']:.0f}s)")
        baseline_lr = sweep.best_lr
        print(f"tuned baseline LR = {baseline_lr:g}")

    # --- The race: identical N, D, seed, data stream; only the optimizer differs ---------------
    muon_lr = args.muon_lr or baseline_lr  # Moonlight RMS-match: one LR band serves both
    race: RaceResult = run_race(
        model_cfg,
        _train_cfg(max_steps),
        train_data,
        val_data,
        baseline_lr=baseline_lr,
        challenger_lr=muon_lr,
        profile_ns=True,
    )

    saving = race.token_saving
    row = {
        "n_params": race.n_params,
        "total_tokens": race.total_tokens,
        "compute_flops": race.compute_flops,
        "baseline_lr": baseline_lr,
        "muon_lr": muon_lr,
        "baseline_final_val": race.baseline.val_curve[-1][1],
        "muon_final_val": race.challenger.val_curve[-1][1],
        "tokens_to_match": race.tokens_to_match,
        "token_saving_fraction": saving,
        "nats_delta": race.nats_delta,
        "ns_overhead": race.challenger.ns_overhead,
        "ns_calls": race.challenger.ns_calls,
        "baseline_wall_s": race.baseline.wall_seconds,
        "muon_wall_s": race.challenger.wall_seconds,
    }
    Path(args.out).write_text(
        json.dumps(
            {
                "args": vars(args),
                "sweep": sweep_rows,
                "race": row,
                "baseline_curve": race.baseline.val_curve,
                "challenger_curve": race.challenger.val_curve,
                "model_config": asdict(model_cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # The ledger row (paste into bench/RESULTS.md §Frontier ablations · F1 iso-FLOP):
    saving_txt = f"{saving:.1%}" if saving is not None else "NEVER MATCHED"
    verdict = (
        "KILL"
        if (saving is None or saving < 0.05) and race.nats_delta > -0.02
        else ("PASS" if race.challenger.ns_overhead < 0.01 else "PASS (NS overhead ⚠)")
    )
    print(
        f"\n| F1 iso-FLOP ({race.n_params / 1e6:.0f}M, D={race.total_tokens / 1e6:.0f}M, "
        f"C={race.compute_flops:.2e}) | tuned-AdamW lr {baseline_lr:g} vs Muon | "
        f"predicted 1.1–1.4× band | <5% saving | "
        f"saving {saving_txt} · Δ {race.nats_delta:+.4f} nats · NS {race.challenger.ns_overhead:.2%} "
        f"→ **{verdict}** |"
    )
    print(f"full curves + config → {args.out}")


if __name__ == "__main__":
    main()
