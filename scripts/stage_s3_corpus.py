"""Stage the F12-winning corpus for the S3 scaling sweep.

After F12 finishes, this script reads the verdict, adopts the winning arm's tokenizer,
and streams enough tokens to cover the s1-s7 grid (largest point needs ~2.37B tokens).
The staged dir is directly consumable by ``scripts/s3_scaling_sweep.py run --data-dir``.

Usage:
    python scripts/stage_s3_corpus.py --f12-dir artifacts/f12_corpus_ablation \
        --out-dir artifacts/s3_scaling_sweep/data --target-tokens 2_500_000_000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/stage_s3_corpus.py` without an editable install.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scratch_llm.data.shards import (
    DEFAULT_TOKENS_PER_SHARD,
    DOC_SEPARATOR,
    ShardMeta,
    StreamStats,
    build_gpt2_detokenizer,
    download_parquet_files,
    iter_parquet_doc_batches,
    list_hub_parquet_files,
    load_tokenizer,
    plan_parquet_download,
    save_tokenizer_files,
    stream_tokenize_to_shards,
)
from scratch_llm.eval.corpus_ablation import CORPUS_SOURCES, CorpusAblationArm, CorpusSource


def _hub_doc_batches(
    source: CorpusSource,
    parquet_paths: list[Path],
    cache_dir: Path,
    batch_size: int = 1024,
):
    decode = build_gpt2_detokenizer(cache_dir) if source.gpt2_detokenize else None
    return iter_parquet_doc_batches(
        parquet_paths, text_column=source.text_column, batch_size=batch_size, decode=decode
    )


def stage_s3_corpus(
    f12_dir: Path,
    out_dir: Path,
    target_tokens: int,
    parquet_cache: Path | None = None,
    num_workers: int = 8,
    override_corpus: str | None = None,
) -> tuple[list[ShardMeta], StreamStats]:
    """Read F12 verdict, copy the winning tokenizer, and stream ~target_tokens of that corpus.

    ``override_corpus`` lets the operator deviate from the measured verdict (e.g. follow an
    external corpus choice such as nanochat's) while keeping the verdict file honest.
    """
    if override_corpus is not None:
        arm = CorpusAblationArm(override_corpus)
        print(f"Operator override: staging {arm.value} (ignoring F12 verdict)")
    else:
        verdict_path = f12_dir / "f12_verdict.json"
        if not verdict_path.exists():
            raise FileNotFoundError(f"F12 verdict not found: {verdict_path}")
        verdict = json.loads(verdict_path.read_text())
        winner = verdict["verdict"]
        if winner == "climbmix_wins":
            arm = CorpusAblationArm.CLIMBMIX
        elif winner == "keep_fineweb_edu":
            arm = CorpusAblationArm.FINEWEB_EDU
        else:
            raise ValueError(f"unknown F12 verdict: {winner}")

    source = CORPUS_SOURCES[arm]
    arm_dir = f12_dir / arm.value
    tokenizer = load_tokenizer(arm_dir)
    print(f"F12 winner: {arm.value} — loading tokenizer from {arm_dir}")

    cache_dir = parquet_cache or out_dir / "parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = list_hub_parquet_files(dataset=source.repo_id, subdir=source.subdir)
    plan = plan_parquet_download(files, target_tokens)
    print(plan.describe())
    paths = download_parquet_files(plan.files, cache_dir, progress=print)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer_files(tokenizer, out_dir)
    eot_ids = tokenizer.encode(DOC_SEPARATOR)
    if len(eot_ids) != 1:
        raise RuntimeError(f"{DOC_SEPARATOR!r} must encode to 1 id, got {eot_ids}")

    return stream_tokenize_to_shards(
        _hub_doc_batches(source, paths, cache_dir),
        tokenizer,
        eot_ids[0],
        out_dir,
        tokens_per_shard=DEFAULT_TOKENS_PER_SHARD,
        target_tokens=target_tokens,
        num_workers=num_workers,
        progress=print,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage the F12-winning corpus for S3 scaling sweep")
    p.add_argument("--f12-dir", type=Path, default=REPO / "artifacts" / "f12_corpus_ablation")
    p.add_argument("--out-dir", type=Path, default=REPO / "artifacts" / "s3_scaling_sweep" / "data")
    p.add_argument(
        "--target-tokens",
        type=int,
        default=2_500_000_000,
        help="Token budget to stage (default 2.5B covers s6, the largest s1-s7 point).",
    )
    p.add_argument("--parquet-cache", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--corpus",
        choices=[a.value for a in CorpusAblationArm],
        default=None,
        help="Override the F12-verdict corpus choice (logs a deviation).",
    )
    args = p.parse_args(argv)

    metas, stats = stage_s3_corpus(
        args.f12_dir,
        args.out_dir,
        args.target_tokens,
        parquet_cache=args.parquet_cache,
        num_workers=args.num_workers,
        override_corpus=args.corpus,
    )
    print(f"\nStaged {stats.n_tokens:,} tokens in {stats.n_shards} shards → {args.out_dir}")


if __name__ == "__main__":
    main()
