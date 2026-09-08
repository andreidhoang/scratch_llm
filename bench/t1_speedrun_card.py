"""T1/T-R4 — the per-checkpoint report card + vibe-eval CLI for the speedrun spine.

One command over a finished ``speedrun.py`` work-dir: for **every** stage checkpoint it finds
(``pretrain.pt`` → ``midtrain.pt`` → ``sft.pt``) it computes val bits-per-byte on one held-out
split, the DCLM CORE aggregate on one task subset, and the fixed vibe set under one decoding
config — run **twice**, so ``vibe_deterministic`` is a measured field of the card and not a hope.
The cards land in one JSON array; the d20 gate (``--target``) prints PASS/FAIL clause by clause;
the last line of stdout is a single bare number, the ledger's metric.

    python bench/t1_speedrun_card.py --work-dir runs/d20 --data-dir data/fineweb_edu \\
        --val-shards data/fineweb_edu_val --device cuda --limit 500 \\
        --target experiments/T1/T-R4/d20_target.json --out cards.json

    python bench/t1_speedrun_card.py --dry-run --work-dir runs/d20 --data-dir data/x

The floor is a *reference card*: a production checkpoint scored on the same CORE task list, same
subsample limit and same eval bundle, ingested from its own harness's JSON report —

    python bench/t1_speedrun_card.py --from-report nanochat_base_eval.json \\
        --reference-id nanochat-d20 --n-params <N> --tokens-seen <T> --out floor_card.json

— which is why ``EvalProtocol`` fingerprints *core*, *val* and *vibe* separately: a floor shares
the CORE recipe with us and nothing else, and one fingerprint over everything would refuse the
only comparison that matters. Model-side facts (tokenizer, BOS analog, context length) are
provenance, not comparability blockers; they are documented deltas
(see ``src/scratch_llm/eval/core_suite.py``'s "Known deltas from nanochat").

This bench times nothing, so it does not import ``bench/_harness.py``: there is no L2 to flush and
no CUDA-event median to take. Its measurement hygiene is of a different kind — fixed prompts,
fixed decode, fixed split, fingerprints on all three.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from scratch_llm.data.shards import DOC_SEPARATOR, load_dataset_tokens, load_tokenizer
from scratch_llm.eval.checkpoint_card import (
    CheckpointCard,
    D20Target,
    EvalProtocol,
    IncomparableCards,
    MissingMetric,
    Stage,
    compare,
    d20_gate,
    save_cards,
    split_id,
)
from scratch_llm.eval.core_suite import EVAL_BUNDLE_URL, core_task_specs, evaluate_core_suite
from scratch_llm.eval.metrics import bits_per_byte
from scratch_llm.eval.vibe import VibeDecode, run_vibe_evals, runs_identical, vibe_metrics
from scratch_llm.tokenizer import Tokenizer
from scratch_llm.train import build_model_from_checkpoint

# The stage spine as speedrun.py writes it (stage-boundary, optimizer-free artifacts).
_STAGE_FILES: tuple[tuple[str, str], ...] = (
    ("pretrain", "pretrain.pt"),
    ("midtrain", "midtrain.pt"),
    ("sft", "sft.pt"),
)

# A base checkpoint has no turn structure to terminate, so it is probed raw; the SFT checkpoint is
# probed through the chat template and scored on whether it stops. The two are deliberately not
# comparable — vibe.diff_runs refuses across templates.
_STAGE_TEMPLATE: dict[str, str] = {"pretrain": "raw", "midtrain": "raw", "sft": "chat"}


def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _provenance(device: str) -> dict[str, str]:
    return {
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "commit_scratch_llm": _sh("git rev-parse --short HEAD"),
        "gpu": _sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
    }


def _load_val_text(
    tokenizer: Tokenizer, *, shards: str | None, text_file: str | None, glob: str, max_tokens: int
) -> tuple[np.ndarray, str, int]:
    """The held-out split: token ids, its stable id, and the UTF-8 byte count bpb divides by.

    The id hashes the **text bytes**, not the token ids: bpb is bits over bytes and is tokenizer-
    invariant by construction, so two models with different tokenizers scored on the same text are
    legitimately comparable — and the same text truncated differently is not.
    """
    if shards is not None:
        tokens = np.asarray(load_dataset_tokens(shards, glob), dtype=np.int64)
        if max_tokens > 0:
            tokens = tokens[:max_tokens]  # prefix, not a sample: no RNG in a split definition
        text = tokenizer.decode(tokens.tolist())
        name = f"shards:{Path(shards).name}:{glob}:{tokens.size}"
    elif text_file is not None:
        text = Path(text_file).read_text(encoding="utf-8")
        tokens = np.asarray(tokenizer.encode(text), dtype=np.int64)
        if max_tokens > 0:
            tokens = tokens[:max_tokens]
            text = tokenizer.decode(tokens.tolist())
        name = f"text:{Path(text_file).name}:{tokens.size}"
    else:
        raise SystemExit(
            "one of --val-shards / --val-text is required (bpb needs a held-out split)"
        )
    if tokens.size < 2:
        raise SystemExit(f"held-out split {name} has {tokens.size} tokens — nothing to score")
    text_bytes = text.encode("utf-8")
    return tokens, split_id(text_bytes, name), len(text_bytes)


def _find_checkpoints(args: argparse.Namespace) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    if args.work_dir:
        work = Path(args.work_dir)
        for stage, fname in _STAGE_FILES:
            path = work / fname
            if path.is_file():
                found.append((stage, path))
    for item in args.ckpt or []:
        if "=" not in item:
            raise SystemExit(f"--ckpt expects stage=path, got {item!r}")
        stage, path = item.split("=", 1)
        found.append((stage, Path(path)))
    if not found:
        raise SystemExit(
            "no stage checkpoints found — pass --work-dir <speedrun work_dir> (it holds "
            "pretrain.pt / sft.pt) or --ckpt stage=path."
        )
    return found


def _card_for_checkpoint(
    stage: str,
    path: Path,
    tokenizer: Tokenizer,
    *,
    args: argparse.Namespace,
    val_tokens: np.ndarray,
    protocol_base: EvalProtocol,
    bos_id: int | None,
) -> CheckpointCard:
    model, step = build_model_from_checkpoint(path, map_location=args.device)
    model = model.to(args.device).eval()

    bpb = bits_per_byte(
        model,
        val_tokens,
        protocol_base.val_num_bytes,
        context_length=model.cfg.context_length,
        device=args.device,
    )

    specs = core_task_specs(cache_dir=args.cache_dir, names=list(protocol_base.core_tasks))
    suite = evaluate_core_suite(
        model,
        tokenizer,
        specs,
        device=args.device,
        limit=protocol_base.core_limit,
        batch_size=args.batch_size,
        bos_id=bos_id,
        max_seq_len=model.cfg.context_length,
    )

    template = _STAGE_TEMPLATE.get(stage, "raw")
    decode = VibeDecode(
        template=template,  # type: ignore[arg-type]
        temperature=args.vibe_temperature,
        top_p=args.vibe_top_p,
        max_tokens=args.vibe_max_tokens,
        seed=args.vibe_seed,
    )
    first = run_vibe_evals(model, tokenizer, decode=decode, device=args.device)
    metrics: dict[str, float] = {
        "val_bpb": bpb.bits_per_byte,
        "nats_per_token": bpb.nats_per_token,
        "core": suite.core,
        **vibe_metrics(first),
    }
    if args.vibe_repeat > 1:
        # The determinism claim, measured: the same checkpoint on the same prompts and the same
        # decode must return the same token ids. Anything else means the comparison between two
        # checkpoints is reading sampler noise.
        again = run_vibe_evals(model, tokenizer, decode=decode, device=args.device)
        metrics["vibe_deterministic"] = 1.0 if runs_identical(first, again) else 0.0

    protocol = EvalProtocol(
        core_tasks=protocol_base.core_tasks,
        core_limit=protocol_base.core_limit,
        core_bundle=protocol_base.core_bundle,
        val_split_id=protocol_base.val_split_id,
        val_num_bytes=protocol_base.val_num_bytes,
        vibe_set_id=first.set_id,
        vibe_prompts_hash=first.prompts_hash,
        vibe_decode_id=first.decode.id,
    )
    provenance = _provenance(args.device)
    provenance.update(
        {
            "checkpoint": str(path),
            "context_length": str(model.cfg.context_length),
            "bos_id": "none" if bos_id is None else str(bos_id),
            "tokenizer": f"{args.data_dir}:{len(tokenizer.vocab)}",
        }
    )
    return CheckpointCard(
        checkpoint_id=f"{stage}@{path.name}",
        stage=stage,  # type: ignore[arg-type]
        step=step,
        n_params=sum(p.numel() for p in model.parameters()),
        tokens_seen=args.tokens_seen,
        protocol=protocol,
        metrics=metrics,
        core_tasks={t.name: t.accuracy for t in suite.tasks},
        vibe=first,
        provenance=provenance,
    )


def _canonical_tasks(args: argparse.Namespace) -> tuple[str, ...]:
    """The CORE task list in manifest order — the core-scope fingerprint, for BOTH paths.

    ``core_task_specs`` performs no I/O (its loaders are lazy), so this is hermetic; it also
    rejects an unknown task name here rather than 40 GPU-minutes later.
    """
    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    return tuple(s.name for s in core_task_specs(cache_dir=args.cache_dir, names=names or None))


def _reference_card(args: argparse.Namespace) -> CheckpointCard:
    """Ingest a floor harness's own JSON report into a ``reference`` card.

    Deliberately dumb: read one key out of a JSON file, refuse loudly if it is not there. The floor
    is scored by the floor's own harness — substituting our scorer for theirs would make the
    comparison ours, not theirs — so the only thing we do here is give their number a card, with
    the CORE protocol fields filled in by hand from the run they did.
    """
    report = json.loads(Path(args.from_report).read_text(encoding="utf-8"))
    node: object = report
    for key in args.core_key.split("."):
        if not isinstance(node, dict) or key not in node:
            available = sorted(node) if isinstance(node, dict) else type(node).__name__
            raise SystemExit(
                f"{args.from_report}: key path {args.core_key!r} not found (at {key!r}); "
                f"available: {available}"
            )
        node = node[key]
    if args.n_params <= 0 or args.tokens_seen <= 0:
        raise SystemExit(
            "--n-params and --tokens-seen are required for a reference card: a floor at an "
            "unknown parameter count and an unknown token budget is not a matched floor."
        )
    # The task list must be canonicalised through the SAME function the rung path uses, or the
    # core fingerprint differs on task ORDER alone and the floor card refuses the one comparison
    # it exists for. Empty --tasks means the whole 22-task suite here exactly as it does there.
    protocol = EvalProtocol(
        core_tasks=_canonical_tasks(args),
        core_limit=None if args.limit <= 0 else args.limit,
        core_bundle=args.bundle,
    )
    return CheckpointCard(
        checkpoint_id=args.reference_id,
        stage="reference",
        step=int(args.reference_step),
        n_params=args.n_params,
        tokens_seen=args.tokens_seen,
        protocol=protocol,
        metrics={"core": float(node)},  # type: ignore[arg-type]
        provenance={"report": str(args.from_report), "harness": args.reference_harness},
    )


def _print_plan(args: argparse.Namespace, tasks: Sequence[str]) -> None:
    print("# t1_speedrun_card [dry-run] — nothing measured")
    print(f"#   work-dir      {args.work_dir}")
    print(f"#   data-dir      {args.data_dir}")
    print(
        f"#   val split     shards={args.val_shards} text={args.val_text} max={args.val_max_tokens}"
    )
    print(f"#   core tasks    {len(tasks)} · limit={args.limit if args.limit > 0 else 'all'}")
    print(f"#   vibe          template per stage {_STAGE_TEMPLATE} · repeat={args.vibe_repeat}")
    print(
        f"#   vibe decode   t={args.vibe_temperature} p={args.vibe_top_p} "
        f"n={args.vibe_max_tokens} seed={args.vibe_seed}"
    )
    print(f"#   target        {args.target or '(none — gate not evaluated)'}")
    print(f"#   headline      {args.headline}.{args.headline_metric}  ← the bare last line")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="T1/T-R4 per-checkpoint report card + vibe-evals.")
    ap.add_argument("--work-dir", default=None, help="speedrun work_dir (pretrain.pt / sft.pt)")
    ap.add_argument("--ckpt", action="append", default=[], help="stage=path (repeatable)")
    ap.add_argument("--data-dir", default=None, help="dir holding the staged tokenizer.json")
    ap.add_argument("--val-shards", default=None, help="held-out shard dir (data/shards.py)")
    ap.add_argument("--val-glob", default="*.bin")
    ap.add_argument("--val-text", default=None, help="held-out UTF-8 text file (alternative)")
    ap.add_argument("--val-max-tokens", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--tasks", default="", help="comma-separated CORE task subset (default: all)")
    ap.add_argument("--limit", type=int, default=-1, help="max examples per task (-1 = all)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache-dir", default=None, help="eval-bundle cache (default ~/.cache/...)")
    ap.add_argument("--bundle", default=EVAL_BUNDLE_URL, help="eval bundle id (core fingerprint)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--tokens-seen",
        type=int,
        default=0,
        help="pretraining tokens behind these checkpoints — NOT recoverable from the checkpoint "
        "file, so it comes from the run manifest and the card refuses to exist without it.",
    )
    ap.add_argument("--vibe-temperature", type=float, default=0.0)
    ap.add_argument("--vibe-top-p", type=float, default=1.0)
    ap.add_argument("--vibe-max-tokens", type=int, default=96)
    ap.add_argument("--vibe-seed", type=int, default=0)
    ap.add_argument("--vibe-repeat", type=int, default=2, help="2 = measure vibe_deterministic")
    ap.add_argument("--target", default=None, help="d20 target JSON (Huy's numbers)")
    ap.add_argument("--floor-card", default=None, help="reference card JSON to compare against")
    ap.add_argument("--headline", default="pretrain", help="stage whose metric is the bare number")
    ap.add_argument("--headline-metric", default="core")
    ap.add_argument("--out", default="t1_t_r4_cards.json")
    ap.add_argument("--dry-run", action="store_true")
    # reference-card ingest (the floor path)
    ap.add_argument("--from-report", default=None, help="floor harness JSON → a reference card")
    ap.add_argument("--core-key", default="core", help="dotted key path to CORE in that JSON")
    ap.add_argument("--reference-id", default="reference")
    ap.add_argument("--reference-step", type=int, default=0)
    ap.add_argument("--reference-harness", default="upstream")
    ap.add_argument("--n-params", type=int, default=0)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.from_report:
        card = _reference_card(args)
        print(card.to_table())
        save_cards([card], args.out)
        print(f"# wrote {args.out}")
        print(f"{card.metric('core'):.6f}")
        return

    tasks = _canonical_tasks(args)

    if args.dry_run:
        _print_plan(args, tasks)
        return

    if not args.data_dir:
        raise SystemExit("--data-dir is required (the staged tokenizer defines the token axis)")
    if args.vibe_repeat < 2:
        raise SystemExit(
            "--vibe-repeat must be >= 2: an sft card requires the measured `vibe_deterministic`, "
            "and one vibe run cannot show that the second one matched it."
        )
    if args.tokens_seen <= 0:
        raise SystemExit(
            "--tokens-seen is required: 'at the d20 target' is a statement about compute, and the "
            "checkpoint file does not carry it (train.save_checkpoint stores model+config+step)."
        )

    tokenizer = load_tokenizer(args.data_dir)
    bos_id = (
        tokenizer.encode(DOC_SEPARATOR)[0] if DOC_SEPARATOR in tokenizer.special_tokens else None
    )
    val_tokens, val_id, val_bytes = _load_val_text(
        tokenizer,
        shards=args.val_shards,
        text_file=args.val_text,
        glob=args.val_glob,
        max_tokens=args.val_max_tokens,
    )
    protocol_base = EvalProtocol(
        core_tasks=tasks,
        core_limit=None if args.limit <= 0 else args.limit,
        core_bundle=args.bundle,
        val_split_id=val_id,
        val_num_bytes=val_bytes,
    )

    cards: list[CheckpointCard] = []
    for stage, path in _find_checkpoints(args):
        card = _card_for_checkpoint(
            stage,
            path,
            tokenizer,
            args=args,
            val_tokens=val_tokens,
            protocol_base=protocol_base,
            bos_id=bos_id,
        )
        print(card.to_table())
        cards.append(card)

    by_stage: dict[str, CheckpointCard] = {c.stage: c for c in cards}
    save_cards(cards, args.out)
    print(f"# wrote {args.out}")

    base = by_stage.get("pretrain")
    chat = by_stage.get("sft")
    if base is not None and chat is not None:
        for scope in ("core", "val"):
            try:
                deltas = compare(base, chat, scope=scope)  # type: ignore[arg-type]
                for name, (a, b, d) in deltas.items():
                    print(f"# Δ{name:<18} pretrain {a:.6f} → sft {b:.6f}  ({d:+.6f})")
            except (IncomparableCards, MissingMetric) as exc:
                print(f"# no {scope} comparison: {exc}")

    if args.floor_card and not Path(args.floor_card).is_file():
        # A missing floor is not a reason to lose the measurement — but it IS a reason to say so.
        print(f"# no floor card at {args.floor_card} — run experiments/T1/T-R4/floor.sh first")
    elif args.floor_card:
        from scratch_llm.eval.checkpoint_card import load_cards

        for floor in load_cards(args.floor_card):
            try:
                deltas = compare(base or cards[0], floor, scope="core")
                for name, (a, b, d) in deltas.items():
                    print(f"# vs floor {floor.checkpoint_id}: {name} {a:.6f} vs {b:.6f} ({d:+.6f})")
            except (IncomparableCards, MissingMetric) as exc:
                print(f"# no floor comparison: {exc}")

    if args.target:
        if base is None or chat is None:
            print("# gate: UNEVALUATED — needs both a pretrain and an sft card")
        else:
            try:
                result = d20_gate(base, chat, D20Target.from_json(args.target))
                print(result.to_table())
            except ValueError as exc:
                print(f"# gate: UNSET — {exc}")

    stage_key: Stage = args.headline
    headline = by_stage.get(stage_key)
    if headline is None:
        raise SystemExit(f"no {args.headline} card — cannot print the headline metric")
    print(f"{headline.metric(args.headline_metric):.6f}")


if __name__ == "__main__":
    main()
