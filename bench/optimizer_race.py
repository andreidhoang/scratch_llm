"""F1-run — the iso-FLOP Muon-vs-tuned-AdamW race, on real shards (GPU day driver).

Pre-registered in bench/RESULTS.md §Frontier ablations (07-04, recalibrated 07-09 BEFORE the run):
- Muon tokens-to-match in the **1.1–1.4× band vs an independently LR-TUNED AdamW baseline**
  (≈15–25% saving expected at 30–50M params, shrinking with N), or ≥0.02 nats lower at iso-FLOP.
- NS overhead <1% of wall (analytic bound); **KILL:** saving <5% vs the tuned baseline, divergence
  at the reused LR, or NS overhead >3%.
The tuned baseline is MANDATORY (arXiv 2509.02046: the 1.4–2× Muon headlines came from under-tuned
baselines) — this driver runs the sweep first, then the race, in one pre-committed protocol.

Crash containment (a 5–7 h run must never lose a finished stage): every completed stage — sweep,
baseline arm, challenger arm, final metrics — is atomically persisted to --out (tmp + os.replace)
the moment it exists; ``status`` in the JSON names the newest stage on disk. A diverged challenger
is DATA (the pre-registered KILL), not a crash. An epochs guard (--max-epochs, default 1.5) aborts
multi-epoch memorization runs BEFORE burning compute — the pre-registered science assumes ~1 epoch.

Run (sm120 box; bf16 EAGER — bf16+compile NaNs on this inductor, see RESULTS.md F4):
  python bench/optimizer_race.py --data-dir data/fineweb_edu --depth 8 --tokens 7e8 \
      --batch-size 32 --context-length 1024 --amp bf16 --device cuda --out results/f1_race.json
Resume past the sweep with a known winner:  --baseline-lr 6e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from scratch_llm.data.shards import load_dataset_tokens, load_shard
from scratch_llm.eval.optimizer_race import ArmResult, RaceResult, run_race, sweep_lr
from scratch_llm.model import ModelConfig
from scratch_llm.speedrun import model_config_for_depth
from scratch_llm.train import TrainConfig


def _detect_vocab_size(data_dir: Path) -> int:
    """Largest ``max_token_id + 1`` across shard sidecars — the safe embedding size."""
    paths = sorted(data_dir.glob("*.bin"))
    if not paths:
        raise FileNotFoundError(f"no shards in {data_dir}")
    return max(load_shard(p)[1].max_token_id for p in paths) + 1


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument(
        "--max-epochs",
        type=float,
        default=1.5,
        help="abort if D/corpus exceeds this — the pre-registered science needs ~1 epoch "
        "(a multi-epoch run measures memorization, not optimizer signal); raise it explicitly "
        "for smoke runs on tiny corpora",
    )
    ap.add_argument(
        "--no-track-logits",
        action="store_true",
        help="disable the F9 ride-along max-attn-logit observer (on by default)",
    )
    ap.add_argument(
        "--no-qk-norm",
        dest="qk_norm",
        action="store_false",
        help="disable per-head QK RMSNorm (ON by default — the pre-registered F9 falsifier "
        "'S_max < 30 under qk_norm' is conditioned on it being on)",
    )
    ap.add_argument(
        "--attention",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="training attention path: sdpa = fused scaled_dot_product_attention (never "
        "materializes the S×S scores — the F1-OOM fix); eager = the reference path",
    )
    ap.add_argument("--amp", choices=["none", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="f1_race.json")
    return ap


def model_cfg_from_args(args: argparse.Namespace, vocab_size: int) -> ModelConfig:
    """The one place CLI flags become architecture — unit-testable without shards or training."""
    cfg = model_config_for_depth(args.depth, vocab_size, args.context_length)
    cfg = replace(cfg, qk_norm=args.qk_norm, use_sdpa=(args.attention == "sdpa"))
    if not args.no_track_logits:
        # F9 ride-along: the per-head max-logit observer costs one amax per forward and lets this
        # run double as F9's falsifier (S_max < 30 under qk_norm => QK-Clip gamma==1 sub-1B).
        cfg = replace(cfg, track_attn_logits=True)
    return cfg


def verdict_string(
    *,
    diverged: bool,
    saving: float | None,
    nats_delta: float | None,
    ns_overhead: float,
) -> str:
    """The full pre-registered F1 ternary (module docstring), pure so every KILL branch is
    unit-testable without a race.

    Precedence: divergence at the reused LR ≻ saving <5% without the ≥0.02-nats rescue ≻
    NS overhead >3%. PASS carries a warn tag from 1% (the analytic <1% bound) up to the 3% KILL.
    ``nats_delta=None`` (only possible alongside ``diverged``) never rescues a missing saving.
    """
    if diverged:
        return "KILL (divergence at reused LR)"
    if (saving is None or saving < 0.05) and (nats_delta is None or nats_delta > -0.02):
        return "KILL"
    if ns_overhead > 0.03:
        return "KILL (NS overhead >3%)"
    if ns_overhead >= 0.01:
        return "PASS (NS overhead ⚠)"
    return "PASS"


def _write_json_atomic(payload: dict[str, Any], out_path: Path) -> None:
    """tmp file + os.replace in the same directory — a crash mid-write never truncates --out."""
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)


def _finite_or_none(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def _arm_row(arm: ArmResult) -> dict[str, Any]:
    """Everything worth keeping if the process dies right after this arm finishes."""
    return {
        "label": arm.label,
        "optimizer": arm.optimizer,
        "lr": arm.lr,
        "diverged": arm.diverged,
        "final_val": _finite_or_none(arm.val_curve[-1][1] if arm.val_curve else None),
        "wall_s": arm.wall_seconds,
        "ns_seconds": arm.ns_seconds,
        "ns_calls": arm.ns_calls,
        "max_attn_logit": arm.max_attn_logit,
        "val_curve": arm.val_curve,
    }


def main() -> None:
    args = build_parser().parse_args()

    # Fail on an unwritable --out NOW, not after 5-7 h of arms (the runbook pre-commits
    # --out results/f1_race.json with no results/ dir).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    tokens = load_dataset_tokens(data_dir)
    vocab_size = args.vocab_size or _detect_vocab_size(data_dir)
    val_n = max(int(len(tokens) * args.val_frac), args.eval_batches * args.context_length + 1)
    train_data, val_data = tokens[:-val_n], tokens[-val_n:]

    # Epochs guard: a --tokens budget far beyond the corpus silently turns the race into a
    # memorization benchmark and the PASS/KILL row into noise. Always print the number.
    epochs = args.tokens / len(train_data)
    print(f"epochs = D / corpus = {args.tokens:.3g} / {len(train_data):,} = {epochs:.2f}")
    if epochs > args.max_epochs:
        raise SystemExit(
            f"ABORT: --tokens {args.tokens:.3g} over a {len(train_data):,}-token corpus is "
            f"{epochs:.1f} epochs (> --max-epochs {args.max_epochs:g}). The pre-registered F1 "
            "science assumes ~1 epoch — a multi-epoch run measures memorization, not optimizer "
            "signal. Build more shards or shrink --tokens; if you really mean a multi-epoch run "
            f"(e.g. a smoke test), override explicitly with --max-epochs {math.ceil(epochs)}."
        )

    tokens_per_step = args.batch_size * args.context_length
    max_steps = math.ceil(args.tokens / tokens_per_step)
    model_cfg = model_cfg_from_args(args, vocab_size)

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

    # Incremental atomic persistence: `payload` grows a stage at a time and hits disk after each
    # one — a crash at any point leaves the newest completed stage in --out under `status`.
    payload: dict[str, Any] = {"status": "started", "args": vars(args), "epochs": epochs}

    def _persist(status: str) -> None:
        payload["status"] = status
        _write_json_atomic(payload, out_path)

    # --- Arm 0: the tuned baseline (mandatory — an untuned baseline is a fake win) ------------
    if args.baseline_lr > 0:
        baseline_lr = args.baseline_lr
        sweep_rows: list[dict[str, Any]] = []
        print(f"sweep SKIPPED — using --baseline-lr {baseline_lr:g}")
    else:
        lrs = [float(x) for x in args.sweep_lrs.split(",")]
        sweep_steps = max(1, math.ceil(max_steps * args.sweep_frac))
        print(f"LR sweep: {lrs} at {sweep_steps} steps ({args.sweep_frac:.0%} horizon)")
        sweep = sweep_lr(model_cfg, _train_cfg(sweep_steps), train_data, val_data, lrs)
        sweep_rows = [
            {
                "lr": arm.lr,
                "final_val": _finite_or_none(arm.val_curve[-1][1] if arm.val_curve else None),
                "wall_s": arm.wall_seconds,
                "diverged": arm.diverged,
            }
            for arm in sweep.arms
        ]
        for row in sweep_rows:
            val_txt = f"val {row['final_val']:.4f}" if row["final_val"] is not None else "DIVERGED"
            print(f"  lr {row['lr']:.1e} → {val_txt} ({row['wall_s']:.0f}s)")
        baseline_lr = sweep.best_lr
        print(f"tuned baseline LR = {baseline_lr:g}")
    payload["sweep"] = sweep_rows
    payload["baseline_lr"] = baseline_lr
    _persist("sweep_done")

    # --- The race: identical N, D, seed, data stream; only the optimizer differs ---------------
    muon_lr = args.muon_lr or baseline_lr  # Moonlight RMS-match: one LR band serves both
    profile_ns = True  # NS wall share is a headline metric; it perturbs ONLY muon_wall_s

    def _on_arm(role: str, arm: ArmResult) -> None:
        # run_race's per-arm seam: each finished arm hits disk before the next one starts, and a
        # diverged baseline is persisted BEFORE run_race raises on it.
        payload[f"{role}_arm"] = _arm_row(arm)
        _persist(f"{role}_arm_done")

    race: RaceResult = run_race(
        model_cfg,
        _train_cfg(max_steps),
        train_data,
        val_data,
        baseline_lr=baseline_lr,
        challenger_lr=muon_lr,
        profile_ns=profile_ns,
        arm_hook=_on_arm,
    )

    saving = race.token_saving
    row = {
        "n_params": race.n_params,
        "total_tokens": race.total_tokens,
        "compute_flops": race.compute_flops,
        "baseline_lr": baseline_lr,
        "muon_lr": muon_lr,
        "baseline_final_val": race.baseline.val_curve[-1][1],
        "muon_final_val": _finite_or_none(
            race.challenger.val_curve[-1][1] if race.challenger.val_curve else None
        ),
        "tokens_to_match": race.tokens_to_match,
        "token_saving_fraction": saving,
        "nats_delta": race.nats_delta,
        "diverged": race.challenger.diverged,
        "epochs": epochs,
        "ns_overhead": race.challenger.ns_overhead,
        "ns_calls": race.challenger.ns_calls,
        "baseline_wall_s": race.baseline.wall_seconds,
        "muon_wall_s": race.challenger.wall_seconds,
        # profile_ns adds a sync fence per NS call to the CHALLENGER's clock only — muon_wall_s
        # is not comparable to baseline_wall_s without this flag.
        "muon_wall_perturbed_by_profiler": profile_ns,
        "max_attn_logit_baseline": race.baseline.max_attn_logit,
        "max_attn_logit_muon": race.challenger.max_attn_logit,
        "qk_norm": model_cfg.qk_norm,
        "attention": args.attention,
    }
    payload["race"] = row
    payload["baseline_curve"] = race.baseline.val_curve
    payload["challenger_curve"] = race.challenger.val_curve
    payload["model_config"] = asdict(model_cfg)
    _persist("complete")

    # The ledger row (paste into bench/RESULTS.md §Frontier ablations · F1 iso-FLOP):
    verdict = verdict_string(
        diverged=race.challenger.diverged,
        saving=saving,
        nats_delta=race.nats_delta,
        ns_overhead=race.challenger.ns_overhead,
    )
    if race.challenger.diverged:
        saving_txt = "DIVERGED"
    elif saving is None:
        saving_txt = "NEVER MATCHED"
    else:
        saving_txt = f"{saving:.1%}"
    nats_txt = f"{race.nats_delta:+.4f}" if race.nats_delta is not None else "n/a"
    print(
        f"\n| F1 iso-FLOP ({race.n_params / 1e6:.0f}M, D={race.total_tokens / 1e6:.0f}M, "
        f"C={race.compute_flops:.2e}) | tuned-AdamW lr {baseline_lr:g} vs Muon | "
        f"predicted 1.1–1.4× band | <5% saving | "
        f"saving {saving_txt} · Δ {nats_txt} nats · NS {race.challenger.ns_overhead:.2%} "
        f"→ **{verdict}** |"
    )
    if race.challenger.max_attn_logit is not None and race.baseline.max_attn_logit is not None:
        print(
            f"| F9 ride-along (qk_norm={model_cfg.qk_norm}) | max per-head attn logit | <30 ⇒ "
            f"QK-Clip γ≡1 sub-1B | ≥30 sustained | adamw {race.baseline.max_attn_logit:.1f} · "
            f"muon {race.challenger.max_attn_logit:.1f} |"
        )
    print(f"full curves + config → {args.out}")


if __name__ == "__main__":
    main()
