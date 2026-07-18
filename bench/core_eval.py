"""CORE eval CLI — score a config-carrying checkpoint on the official DCLM CORE 22-task suite.

The d20 headline-metric harness (P3): per-task accuracy + centered accuracy + the CORE mean,
JSON out + a paste-ready ledger row for bench/RESULTS.md. Published anchors for calibration:
nanochat original d20 CORE 0.2219 · GPT-2 XL 0.2565 (comparable in *recipe*; tokenizer differs —
see src/scratch_llm/eval/core_suite.py docstring; the validation run is P5).

Run (full suite; downloads the 26 MB eval bundle to ~/.cache/scratch_llm on first use):
  python bench/core_eval.py --ckpt runs/d20/model_final.pt --data-dir data/fineweb_edu \
      --device cuda --out core_eval.json

Smoke run (2 tasks, 20 examples each):
  python bench/core_eval.py --ckpt ckpt.pt --data-dir data/ --tasks arc_easy,winograd --limit 20
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections.abc import Sequence
from pathlib import Path

from scratch_llm.data.shards import DOC_SEPARATOR, load_tokenizer
from scratch_llm.eval.core_suite import (
    CORE_TASK_NAMES,
    CoreTaskResult,
    TaskSpec,
    core_task_specs,
    evaluate_core_suite,
)
from scratch_llm.train import build_model_from_checkpoint

_ANCHORS = "anchors: nanochat d20 0.2219 · GPT-2 XL 0.2565"


def _ledger_row(core: float, n_tasks: int, limit: int | None, ckpt: str, step: int) -> str:
    """One markdown row, paste-ready for bench/RESULTS.md."""
    scope = f"{n_tasks}/{len(CORE_TASK_NAMES)} tasks" + (f", limit={limit}" if limit else ", full")
    date = _dt.date.today().isoformat()
    return (
        f"| CORE (DCLM 22-task) | {_ANCHORS} | measured {core:.4f} "
        f"| ckpt={ckpt} step={step} · {scope} | {date} |"
    )


def main(argv: Sequence[str] | None = None, specs: Sequence[TaskSpec] | None = None) -> None:
    """CLI entry. ``specs`` overrides the official suite (tests register fixture tasks through
    the same TaskSpec framework; default None = the 22 official tasks)."""
    ap = argparse.ArgumentParser(description="DCLM CORE evaluation of a scratch_llm checkpoint.")
    ap.add_argument(
        "--ckpt", required=True, help="config-carrying checkpoint (train.save_checkpoint)"
    )
    ap.add_argument("--data-dir", required=True, help="dir holding the staged tokenizer.json")
    ap.add_argument("--tasks", default="", help="comma-separated task-name subset (default: all)")
    ap.add_argument(
        "--limit", type=int, default=-1, help="max examples per task, post-shuffle (-1 = all)"
    )
    ap.add_argument("--batch-size", type=int, default=16, help="scoring rows per forward")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--cache-dir", default=None, help="eval-bundle cache dir (default ~/.cache/scratch_llm)"
    )
    ap.add_argument("--out", default="core_eval.json")
    args = ap.parse_args(argv)

    model, step = build_model_from_checkpoint(args.ckpt, map_location=args.device)
    model = model.to(args.device).eval()
    tokenizer = load_tokenizer(args.data_dir)

    # Our BOS analog: the doc-separator special every pretraining document starts after
    # (nanochat prepends its <|bos|>; core_suite docstring documents the delta).
    bos_id: int | None = None
    if DOC_SEPARATOR in tokenizer.special_tokens:
        bos_id = tokenizer.encode(DOC_SEPARATOR)[0]
    else:
        print(f"note: tokenizer has no {DOC_SEPARATOR!r} special — scoring without a BOS prepend")

    names = [t.strip() for t in args.tasks.split(",") if t.strip()] or None
    if specs is None:
        specs = core_task_specs(cache_dir=args.cache_dir, names=names)
    elif names is not None:
        known = {s.name for s in specs}
        unknown = sorted(set(names) - known)
        if unknown:
            raise SystemExit(f"unknown task(s) {unknown}; registered: {sorted(known)}")
        specs = [s for s in specs if s.name in set(names)]
    limit = None if args.limit <= 0 else args.limit

    def report(res: CoreTaskResult) -> None:
        print(
            f"{res.name:<35} accuracy: {res.accuracy:.4f} | centered: {res.centered:.4f} "
            f"| n: {res.n}"
        )

    suite = evaluate_core_suite(
        model,
        tokenizer,
        specs,
        device=args.device,
        limit=limit,
        batch_size=args.batch_size,
        bos_id=bos_id,
        max_seq_len=model.cfg.context_length,
        progress=report,
    )

    core = suite.core
    print(f"\nCORE metric: {core:.4f}  ({_ANCHORS})")
    row = _ledger_row(core, len(specs), limit, args.ckpt, step)
    print("\nledger row (paste into bench/RESULTS.md):")
    print(row)

    payload = {
        "args": vars(args),
        "ckpt_step": step,
        "bos_id": bos_id,
        **suite.to_dict(),
        "ledger_row": row,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
