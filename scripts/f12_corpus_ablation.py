"""F12 corpus ablation CLI — FineWeb-EDU vs ClimbMix at iso-FLOP, each arm its own BPE.

Thin launcher over ``scratch_llm.eval.corpus_ablation`` (the library owns the design
invariants; this script owns staging, defaults, and persistence). The pre-registered run:

    python scripts/f12_corpus_ablation.py --device cuda --bf16 --num-workers 8 --core

cheap standing-box smoke (depth 4, small budget) / hermetic CPU smoke (no network):

    python scripts/f12_corpus_ablation.py --depth 4 --steps 200 --held-out-docs 128
    python scripts/f12_corpus_ablation.py --toy

Real mode stages each arm from the HF hub parquet bulk path (shards.py): FineWeb-EDU
sample-10BT directly, ClimbMix via ``nvidia/Nemotron-ClimbMix`` + the stdlib GPT-2
detokenizer. Downloads are lazy per arm and resumable (``--parquet-cache``); an already
staged arm (``<out-dir>/<arm>/`` from a previous run, or banked shards) is adopted via
``--fineweb-data-dir`` / ``--climbmix-data-dir`` instead of re-staging. The held-out set is
a fixed FineWeb-EDU rows-API sample from a deep offset; BOTH arms are 13-gram-decontaminated
against it at build time and the drop rate is logged per arm (``build_stats.json``).

Persistence: each arm's result JSON is written the moment the arm finishes (``arm_hook``);
the final verdict lands in ``<out-dir>/f12_verdict.json`` with the pre-registered falsifier
and kill-criterion fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

# Allow `python scripts/f12_corpus_ablation.py` without an editable install.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scratch_llm.data.shards import (
    DEFAULT_BPE_TRAIN_BYTES,
    download_fineweb_slice,
    download_parquet_files,
    list_hub_parquet_files,
    plan_parquet_download,
)
from scratch_llm.eval.core_suite import TaskSpec, core_task_specs
from scratch_llm.eval.corpus_ablation import (
    CORPUS_SOURCES,
    CorpusAblationArm,
    CorpusArmResult,
    CorpusSource,
    HeldOutSet,
    arm_result_to_dict,
    hub_doc_batches,
    run_corpus_ablation,
    split_held_out,
)
from scratch_llm.train import TrainConfig

DEFAULT_OUT_DIR = REPO / "artifacts" / "f12_corpus_ablation"

# Held-out rows-API offset, deep into sample-10BT: the staged parquet prefix (which starts
# at file 0) is unlikely to even contain these docs — and the 13-gram gate guarantees
# exclusion regardless, logging the per-arm drop rate.
HELD_OUT_OFFSET = 500_000

DocBatches = Callable[[], Iterable[Sequence[str]]]


# ---------------------------------------------------------------------------
# Toy mode — hermetic CPU smoke: built-in tiny corpora, held-out by construction
# ---------------------------------------------------------------------------


def _toy_docs(prefix: str, n: int, start: int = 0) -> list[str]:
    # The per-doc index lands at least every 9 words, so no two toy docs share a 13-gram —
    # the A0 gate (which guards the held-out set + default eval texts) drops nothing here,
    # and the toy held-out is excluded purely by construction (split_held_out). Arm-specific
    # docs start their indices past the pool so a same-index collision can't sneak a 13-gram
    # past the prefix difference.
    return [
        f"{prefix} document {i} tells a story. the model reads tokens of document {i} and "
        f"learns to predict. good data helps the {prefix} model on task {i}."
        for i in range(start, start + n)
    ]


def _toy_inputs(n_held_out: int) -> tuple[dict[CorpusAblationArm, DocBatches], HeldOutSet]:
    """Both toy arms share a pool (held-out split off BEFORE training) plus arm-specific docs,
    so the smoke exercises the full pipeline with no network."""
    train_pool, held_out = split_held_out(_toy_docs("shared", 24), n_held_out)
    corpora = {
        CorpusAblationArm.FINEWEB_EDU: train_pool + _toy_docs("fineweb", 8, start=100),
        CorpusAblationArm.CLIMBMIX: train_pool + _toy_docs("climbmix", 8, start=200),
    }
    return ({arm: (lambda docs=docs: [docs]) for arm, docs in corpora.items()}, held_out)


def _toy_core_specs() -> tuple[TaskSpec, ...]:
    """Fixture CORE task for ``--toy --core``: the real 22-task suite is network-backed, so
    the hermetic smoke scores a tiny built-in MC task through the same CORE recipe."""
    examples = [
        {"query": "which animal barks", "choices": ["the dog", "the cat"], "gold": 0},
        {"query": "which animal meows", "choices": ["the dog", "the cat"], "gold": 1},
    ]
    return (
        TaskSpec(
            name="toy_mc",
            task_type="multiple_choice",
            loader=lambda: examples,
            random_baseline=0.5,
        ),
    )


# ---------------------------------------------------------------------------
# Real mode — lazy per-arm hub staging (download starts when the arm builds, not before)
# ---------------------------------------------------------------------------


def _hub_factory(source: CorpusSource, target_tokens: int, cache_dir: Path) -> DocBatches:
    staged: dict[str, list[Path]] = {}

    def factory() -> Iterable[Sequence[str]]:
        if "paths" not in staged:
            files = list_hub_parquet_files(dataset=source.repo_id, subdir=source.subdir)
            plan = plan_parquet_download(files, target_tokens)
            print(plan.describe())
            staged["paths"] = download_parquet_files(plan.files, cache_dir, progress=print)
        return hub_doc_batches(source, staged["paths"], cache_dir)()

    return factory


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="F12 corpus ablation: FineWeb-EDU vs ClimbMix at iso-FLOP, per-arm BPE."
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Model depth (aspect-ratio scaling). Default 6 = the pre-registered ~35.8M "
        "run; 4 = cheap smoke (toy: 2).",
    )
    p.add_argument(
        "--target-tokens",
        type=float,
        default=700_000_000,
        help="Token budget D per arm (iso-FLOP: both arms share it) and the shard-build stop.",
    )
    p.add_argument("--vocab-size", type=int, default=None, help="Default 32768 (toy: 512).")
    p.add_argument("--context", type=int, default=None, help="Default 2048 (toy: 64).")
    p.add_argument("--batch", type=int, default=None, help="Default 32 (toy: 8).")
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Training steps per arm; default derives D from --target-tokens/(batch·ctx).",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Val-curve cadence (must be >0 — the DoD wants a curve); default steps//20.",
    )
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--optimizer", choices=["adamw", "muon_adamw"], default="muon_adamw")
    p.add_argument("--device", default="cpu")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (GPU).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--max-train-bytes",
        type=int,
        default=DEFAULT_BPE_TRAIN_BYTES,
        help="BPE training-sample byte budget — SAME for both arms (the per-corpus tokenizer "
        "fairness condition).",
    )
    p.add_argument(
        "--held-out-docs",
        type=int,
        default=None,
        help="Shared held-out set size (default 2048; toy: 4).",
    )
    p.add_argument("--held-out-offset", type=int, default=HELD_OUT_OFFSET)
    p.add_argument(
        "--core",
        action="store_true",
        help="Also score CORE (real mode: the network-backed 22-task suite; toy: a fixture "
        "task). Off by default — CORE is secondary to bpb and noisy single-seed.",
    )
    p.add_argument("--core-limit", type=int, default=None, help="Max examples per CORE task.")
    p.add_argument(
        "--toy",
        action="store_true",
        help="Hermetic CPU smoke: built-in tiny corpora for both arms, no network.",
    )
    p.add_argument(
        "--fineweb-data-dir",
        default=None,
        help="Adopt an already-staged FineWeb-EDU shard dir (banked shards or a previous "
        "run's <out-dir>/fineweb_edu) instead of hub staging.",
    )
    p.add_argument(
        "--climbmix-data-dir",
        default=None,
        help="Same, for the ClimbMix arm (<out-dir>/climbmix from a previous run).",
    )
    p.add_argument(
        "--parquet-cache",
        default=None,
        help="Directory for downloaded parquet + the GPT-2 encoder (default <out-dir>/parquet).",
    )
    args = p.parse_args(argv)

    toy = args.toy
    depth = args.depth if args.depth is not None else (2 if toy else 6)
    vocab_size = args.vocab_size if args.vocab_size is not None else (512 if toy else 32768)
    context = args.context if args.context is not None else (64 if toy else 2048)
    batch = args.batch if args.batch is not None else (8 if toy else 32)
    target = int(args.target_tokens)
    steps = (
        args.steps
        if args.steps is not None
        else (20 if toy else max(1, target // (batch * context)))
    )
    eval_every = args.eval_every if args.eval_every is not None else max(1, steps // 20)
    held_n = args.held_out_docs if args.held_out_docs is not None else (4 if toy else 2048)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.parquet_cache) if args.parquet_cache else out_dir / "parquet"

    # Adopted (prebuilt) arms need no factory; the rest stage from the hub (or the toy pool).
    prebuilt: dict[CorpusAblationArm, str] = {}
    if args.fineweb_data_dir:
        prebuilt[CorpusAblationArm.FINEWEB_EDU] = args.fineweb_data_dir
    if args.climbmix_data_dir:
        prebuilt[CorpusAblationArm.CLIMBMIX] = args.climbmix_data_dir

    corpora: dict[CorpusAblationArm, DocBatches] = {}
    held_out: HeldOutSet
    if toy:
        toy_corpora, held_out = _toy_inputs(held_n)
        corpora = {a: f for a, f in toy_corpora.items() if a not in prebuilt}
    else:
        print(f"held-out: {held_n} FineWeb-EDU docs @ rows offset {args.held_out_offset}")
        held_out = HeldOutSet(
            docs=tuple(download_fineweb_slice(n_docs=held_n, offset=args.held_out_offset))
        )
        for arm in CorpusAblationArm:
            if arm not in prebuilt:
                corpora[arm] = _hub_factory(CORPUS_SOURCES[arm], target, cache)

    core_specs = None
    if args.core:
        core_specs = _toy_core_specs() if toy else core_task_specs()

    train_cfg = TrainConfig(
        max_steps=steps,
        batch_size=batch,
        context_length=context,
        max_lr=args.lr,
        warmup_steps=max(1, steps // 20),
        eval_every=eval_every,
        optimizer=args.optimizer,
        amp_dtype="bf16" if args.bf16 else None,
        device=args.device,
        seed=args.seed,
    )

    def _persist_arm(result: CorpusArmResult) -> None:
        path = out_dir / f"{result.arm.value}_result.json"
        path.write_text(json.dumps(arm_result_to_dict(result), indent=2), encoding="utf-8")
        print(
            f"arm {result.arm.value}: val_bpb={result.bpb.bits_per_byte:.4f} "
            f"(overlap rate {result.overlap_rate}) → {path}"
        )

    result = run_corpus_ablation(
        corpora,
        held_out,
        out_dir,
        train_cfg,
        depth=depth,
        vocab_size=vocab_size,
        max_train_bytes=args.max_train_bytes,
        target_tokens=None if toy else target,
        core_specs=core_specs,
        core_limit=args.core_limit,
        num_workers=args.num_workers,
        prebuilt=prebuilt,
        arm_hook=_persist_arm,
        progress=print,
    )

    verdict = {
        "falsifier": "ClimbMix val_bpb < FineWeb-EDU val_bpb at iso-FLOP "
        "(predicted Δ ≈ −0.010..−0.030)",
        "falsifier_confirmed": result.bpb_delta < 0,
        "kill_criterion": "ClimbMix bpb ≥ FineWeb-EDU bpb at iso-FLOP ⇒ keep the banked "
        "FineWeb-EDU corpus; do not stage ClimbMix",
        "kill_triggered": result.bpb_delta >= 0,
        "config": {
            "depth": depth,
            "vocab_size": vocab_size,
            "context": context,
            "batch": batch,
            "steps": steps,
            "target_tokens": target,
            "max_train_bytes": args.max_train_bytes,
            "seed": args.seed,
            "optimizer": args.optimizer,
            "toy": toy,
        },
        **result.to_dict(),
    }
    verdict_path = out_dir / "f12_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(
        f"\nF12 verdict: {result.verdict} (bpb Δ = {result.bpb_delta:+.4f}; "
        f"FWE {result.baseline.bpb.bits_per_byte:.4f} vs ClimbMix "
        f"{result.challenger.bpb.bits_per_byte:.4f}) → {verdict_path}"
    )


if __name__ == "__main__":
    main()
