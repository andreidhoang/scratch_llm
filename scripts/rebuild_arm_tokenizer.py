"""Rebuild an F12 arm's BPE tokenizer (recovery tool).

DO NOT USE THIS FOR ``climbmix``. That arm's tokenizer is pinned and checked into git at
``assets/tokenizers/climbmix/tokenizer.json`` (md5 ``4fc61379fc4bbaee73842a4aa8752a02``); the
pipeline installs it via ``require_pinned_tokenizer`` in ``scripts/_tokenizer_guard.sh``, which
copies and verifies rather than rebuilding. Rebuilding it is how the s1-s7 sweep was invalidated
(bytes/token 4.08 -> 1.93; ``docs/S35_DATA_PROVENANCE.md``). This tool remains correct for a NEW
arm that has no pinned copy, and as a last-resort recovery path — see the determinism caveat below.

The per-arm tokenizers live under ``artifacts/f12_corpus_ablation/<arm>/`` — gitignored,
so they die with the pod that produced them. This script reproduces the deterministic BPE
phase of ``data.shards.build_dataset_streaming`` for one arm: stream the arm's parquet docs
in order, stage doc-by-doc up to ``DEFAULT_BPE_TRAIN_BYTES`` (doc-granular cap), train the
byte-level BPE (vocab 32768, DOC_SEPARATOR the only special), save ``tokenizer.json`` where
``stage_s3_corpus.py`` expects it.

Determinism: same parquet files + same doc order + same byte cap + same trainer ⇒ same
vocab/merges. Identity with a previously built tokenizer is NOT verifiable once the original
is lost — any experiment straddling a rebuild must rerun its control arm (the S3.5 batch
de-confound runs BOTH batch-4 and batch-8 fresh for exactly this reason).

Usage:
    python scripts/rebuild_arm_tokenizer.py --arm climbmix
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow `python scripts/rebuild_arm_tokenizer.py` without an editable install.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scratch_llm.data.shards import (
    DEFAULT_BPE_TRAIN_BYTES,
    DOC_SEPARATOR,
    build_gpt2_detokenizer,
    download_parquet_files,
    iter_parquet_doc_batches,
    list_hub_parquet_files,
    plan_parquet_download,
    save_tokenizer_files,
)
from scratch_llm.eval.corpus_ablation import CORPUS_SOURCES, CorpusAblationArm
from scratch_llm.tokenizer import Tokenizer, train_bpe

# The BPE sample is 16 MiB of text (~4-5M tokens); one small download slice is plenty.
_DOWNLOAD_TARGET_TOKENS = 50_000_000


def rebuild_arm_tokenizer(arm: CorpusAblationArm, out_dir: Path, parquet_cache: Path) -> Path:
    source = CORPUS_SOURCES[arm]
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_cache.mkdir(parents=True, exist_ok=True)

    files = list_hub_parquet_files(dataset=source.repo_id, subdir=source.subdir)
    plan = plan_parquet_download(files, _DOWNLOAD_TARGET_TOKENS)
    paths = download_parquet_files(plan.files, parquet_cache, progress=print)

    decode = build_gpt2_detokenizer(parquet_cache) if source.gpt2_detokenize else None
    batches = iter_parquet_doc_batches(
        paths, text_column=source.text_column, batch_size=1024, decode=decode
    )

    staged_bytes = 0
    staged_docs = 0
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        staged = Path(fh.name)
        for batch in batches:
            for doc in batch:
                if staged_docs:
                    fh.write(DOC_SEPARATOR)
                fh.write(doc)
                staged_docs += 1
                staged_bytes += len(doc.encode("utf-8"))
                if staged_bytes >= DEFAULT_BPE_TRAIN_BYTES:
                    break
            if staged_bytes >= DEFAULT_BPE_TRAIN_BYTES:
                break
    try:
        print(
            f"BPE sample: {staged_docs:,} docs, {staged_bytes / 2**20:.1f} MiB "
            f"(cap {DEFAULT_BPE_TRAIN_BYTES / 2**20:.0f} MiB) → train_bpe(vocab_size=32768)"
        )
        vocab, merges = train_bpe(staged, 32768, special_tokens=[DOC_SEPARATOR])
    finally:
        staged.unlink(missing_ok=True)

    tokenizer = Tokenizer(vocab, merges, special_tokens=[DOC_SEPARATOR])
    saved = save_tokenizer_files(tokenizer, out_dir)
    print(f"saved tokenizer → {saved}")
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=[a.value for a in CorpusAblationArm], required=True)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: artifacts/f12_corpus_ablation/<arm>",
    )
    p.add_argument(
        "--parquet-cache",
        type=Path,
        default=None,
        help="default: artifacts/s3_scaling_sweep/data/parquet (shared with staging)",
    )
    args = p.parse_args()
    out_dir = args.out_dir or REPO / "artifacts" / "f12_corpus_ablation" / args.arm
    cache = args.parquet_cache or REPO / "artifacts" / "s3_scaling_sweep" / "data" / "parquet"
    rebuild_arm_tokenizer(CorpusAblationArm(args.arm), out_dir, cache)


if __name__ == "__main__":
    main()
