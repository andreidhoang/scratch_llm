"""Real-corpus token shards — the pretraining data path (A1, close-the-loop front).

Turns a document stream (FineWeb-EDU or any list of texts) into durable, memmap-able token
shards that ``train.py::get_batch`` consumes unchanged — replacing the in-RAM toy corpus as
the substrate every downstream rung (A2 checkpoint chaining, F1-run, midtrain/SFT, the d20)
builds on.

Format (nanoGPT/nanochat convention, headerless): raw native-little-endian token ids, one
``<|eot|>`` id appended after every document, dtype uint16 while every id < 2¹⁶ — with the
pre-registered kill-switch to uint32 the moment any id crosses it (no silent wraparound).
Metadata lives in a ``<stem>.meta.json`` sidecar so the ``.bin`` stays exactly
``itemsize·n_tokens`` bytes. The tokenizer that produced the shards is staged alongside them
(``tokenizer.json``) — shards are meaningless without the vocab that indexed them.

Invariants (tested in tests/test_shards.py, pre-registered in bench/RESULTS.md):
- **Round-trip:** shard bytes → ids == ``encode(d0)+[eot]+encode(d1)+[eot]``.
- **Size:** file bytes == ``itemsize·n_tokens``, zero header/padding.
- **Alignment:** ``get_batch`` on the loaded memmap is next-token-aligned.

Scope note (§D): the base spec is nano-scale — documents are buffered in RAM per shard. The
~11B-token shuffled multi-shard streamer for the d20 is a deferred extension, not this file.

Interview question this answers: "why memmap uint16 shards with an EOT separator, and what
breaks if you train across document boundaries without one?"
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from scratch_llm.tokenizer import Tokenizer, train_bpe

# The document separator. Matches the `<|eot|>` special the A3 chat template will formalize,
# so shards built today stay valid when chat specials land.
DOC_SEPARATOR = "<|eot|>"

_UINT16_MAX = 2**16 - 1


class TokenEncoder(Protocol):
    """The minimal tokenizer surface a shard builder needs."""

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class ShardMeta:
    """Sidecar metadata for one shard (`<stem>.meta.json`). The .bin itself is headerless."""

    n_tokens: int
    n_docs: int
    dtype: str  # "uint16" | "uint32" — the pre-registered kill-switch axis
    eot_id: int
    max_token_id: int
    # Docs rejected by the A0 ``doc_filter`` seam (decontamination). Defaults to 0 so pre-A0
    # sidecars (missing the key) keep loading unchanged.
    n_docs_filtered: int = 0


def _meta_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".meta.json")


def tokenize_to_shard(
    docs: Iterable[str],
    tokenizer: TokenEncoder,
    eot_id: int,
    out_path: str | Path,
    *,
    doc_filter: Callable[[str], bool] | None = None,
) -> ShardMeta:
    """Encode ``docs``, appending ``eot_id`` after every document, and write the shard.

    The EOT after *every* doc (not between) makes concatenated shards seamless and gives the
    model an unambiguous document boundary — without it, windows sampled across two unrelated
    documents teach spurious long-range dependencies.

    ``doc_filter`` (A0 seam, optional): a keep-predicate (True = keep) applied to each doc
    BEFORE tokenization — e.g. :func:`scratch_llm.data.decontaminate.decontam_doc_filter`.
    ``None`` is byte-identical to the pre-seam behavior; rejected docs are counted in
    ``ShardMeta.n_docs_filtered``.
    """
    out_path = Path(out_path)
    ids: list[int] = []
    n_docs = 0
    n_docs_filtered = 0
    for doc in docs:
        if doc_filter is not None and not doc_filter(doc):
            n_docs_filtered += 1
            continue
        ids.extend(tokenizer.encode(doc))
        ids.append(eot_id)
        n_docs += 1
    if not ids:
        if n_docs_filtered:
            raise ValueError(
                f"tokenize_to_shard: doc_filter rejected all {n_docs_filtered} document(s)"
            )
        raise ValueError("tokenize_to_shard needs at least one document")

    max_token_id = max(max(ids), eot_id)
    # Kill-switch (pre-registered): any id ≥ 2¹⁶ silently wrapping in uint16 would corrupt the
    # corpus — switch the whole shard to uint32 instead.
    dtype = np.uint16 if max_token_id <= _UINT16_MAX else np.uint32
    arr = np.asarray(ids, dtype=np.int64)
    arr.astype(dtype).tofile(out_path)

    meta = ShardMeta(
        n_tokens=len(ids),
        n_docs=n_docs,
        dtype=np.dtype(dtype).name,
        eot_id=eot_id,
        max_token_id=int(max_token_id),
        n_docs_filtered=n_docs_filtered,
    )
    _meta_path(out_path).write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return meta


def load_shard(path: str | Path) -> tuple[np.ndarray, ShardMeta]:
    """Load one shard as a read-only memmap plus its sidecar metadata.

    Validates the headerless-size invariant (file bytes == itemsize·n_tokens) so a truncated
    or foreign file fails loudly instead of training on garbage.
    """
    path = Path(path)
    meta_raw = json.loads(_meta_path(path).read_text(encoding="utf-8"))
    meta = ShardMeta(**meta_raw)
    itemsize = np.dtype(meta.dtype).itemsize
    actual = path.stat().st_size
    if actual != itemsize * meta.n_tokens:
        raise ValueError(
            f"{path} size mismatch: {actual} bytes != {itemsize}·{meta.n_tokens} "
            "(truncated shard or wrong sidecar?)"
        )
    tokens = np.memmap(path, dtype=np.dtype(meta.dtype), mode="r")
    return tokens, meta


def load_dataset_tokens(data_dir: str | Path, shard_glob: str = "*.bin") -> np.ndarray:
    """Load every shard in ``data_dir`` (sorted by name) as one flat token array.

    A single shard stays a zero-copy memmap; multiple shards are concatenated in RAM — fine at
    nano scale, and exactly the boundary where the §D d20 streamer takes over.
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob(shard_glob))
    if not paths:
        raise FileNotFoundError(f"no shards matching {shard_glob!r} in {data_dir}")
    if len(paths) == 1:
        return load_shard(paths[0])[0]
    return np.concatenate([load_shard(p)[0] for p in paths])


# ---------------------------------------------------------------------------------------------
# Tokenizer staging — shards are only meaningful with the vocab that produced them.
# ---------------------------------------------------------------------------------------------

_TOKENIZER_FILE = "tokenizer.json"


def save_tokenizer_files(tokenizer: Tokenizer, out_dir: str | Path) -> Path:
    """Stage the tokenizer next to the shards as one JSON file.

    Since A2, a thin delegate to :meth:`Tokenizer.save` (which canonicalized this module's
    JSON format — same payload, so pre-A2 ``tokenizer.json`` files keep loading). The
    two-file ``Tokenizer.from_files`` format stays unused here: its ``A B``-per-line merges
    file is ambiguous when the left symbol itself contains a space.
    """
    return tokenizer.save(Path(out_dir) / _TOKENIZER_FILE)


def load_tokenizer(data_dir: str | Path) -> Tokenizer:
    """Rebuild the exact tokenizer staged by :func:`save_tokenizer_files`."""
    return Tokenizer.load(Path(data_dir) / _TOKENIZER_FILE)


# ---------------------------------------------------------------------------------------------
# One-call dataset build: docs → tokenizer.json + shard_%05d.bin (+ sidecars)
# ---------------------------------------------------------------------------------------------


def build_dataset(
    docs: Sequence[str],
    out_dir: str | Path,
    vocab_size: int,
    docs_per_shard: int | None = None,
    *,
    doc_filter: Callable[[str], bool] | None = None,
) -> list[ShardMeta]:
    """Train a byte-level BPE on ``docs`` (with the EOT special), stage it, and shard the corpus.

    ``docs_per_shard=None`` writes one shard; otherwise documents are chunked in order. The
    tokenizer is trained on the same documents it shards — so ``doc_filter`` (A0's
    decontamination gate, True = keep) is applied ONCE, up front, before BPE training: eval
    text must not shape the merges either. ``None`` is byte-identical to the pre-seam
    behavior. For dropped-doc accounting, run
    :func:`scratch_llm.data.decontaminate.decontaminate_docs` first and pass ``result.kept``.
    """
    if not docs:
        raise ValueError("build_dataset needs at least one document")
    if doc_filter is not None:
        n_before = len(docs)
        docs = [doc for doc in docs if doc_filter(doc)]
        if not docs:
            raise ValueError(f"build_dataset: doc_filter rejected all {n_before} document(s)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train_bpe reads a file; stage the corpus, separators included (the trainer splits them
    # out pre-tokenization, so they only mark boundaries — they are never merged across).
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(DOC_SEPARATOR.join(docs))
        staged = Path(fh.name)
    try:
        vocab, merges = train_bpe(staged, vocab_size, special_tokens=[DOC_SEPARATOR])
    finally:
        staged.unlink(missing_ok=True)
    tokenizer = Tokenizer(vocab, merges, special_tokens=[DOC_SEPARATOR])
    save_tokenizer_files(tokenizer, out_dir)

    eot_ids = tokenizer.encode(DOC_SEPARATOR)
    if len(eot_ids) != 1:
        raise RuntimeError(f"{DOC_SEPARATOR!r} must encode to exactly one id, got {eot_ids}")

    chunk = len(docs) if docs_per_shard is None else docs_per_shard
    metas: list[ShardMeta] = []
    for shard_idx, start in enumerate(range(0, len(docs), chunk)):
        metas.append(
            tokenize_to_shard(
                docs[start : start + chunk],
                tokenizer,
                eot_ids[0],
                out_dir / f"shard_{shard_idx:05d}.bin",
            )
        )
    return metas


# ---------------------------------------------------------------------------------------------
# FineWeb-EDU slice — network path (stdlib only; tests self-skip without the opt-in env var)
# ---------------------------------------------------------------------------------------------


def download_fineweb_slice(
    n_docs: int = 64,
    *,
    dataset: str = "HuggingFaceFW/fineweb-edu",
    config: str = "sample-10BT",
    split: str = "train",
    offset: int = 0,
    timeout: float = 30.0,
) -> list[str]:
    """Fetch ``n_docs`` document texts via the HF datasets-server rows API (no extra deps).

    Nano-scale on purpose: the API pages 100 rows at a time — the d20's ~11B tokens come from
    the parquet shards + streamer (§D extension), not this endpoint.
    """
    docs: list[str] = []
    while len(docs) < n_docs:
        length = min(100, n_docs - len(docs))
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={urllib.parse.quote(dataset, safe='')}"
            f"&config={config}&split={split}&offset={offset + len(docs)}&length={length}"
        )
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (https, fixed host)
            payload = json.load(resp)
        rows = payload.get("rows", [])
        if not rows:
            raise RuntimeError(f"datasets-server returned no rows at offset {offset + len(docs)}")
        docs.extend(str(r["row"]["text"]) for r in rows)
    return docs[:n_docs]


def main() -> None:
    p = argparse.ArgumentParser(description="Build token shards (A1): FineWeb/corpus → memmap.")
    p.add_argument("--out", required=True, help="Output dataset directory.")
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--n-docs", type=int, default=64, help="FineWeb-EDU docs to download.")
    p.add_argument("--corpus", default=None, help="Local text file (blank-line-separated docs).")
    p.add_argument("--docs-per-shard", type=int, default=None)
    p.add_argument(
        "--decontaminate",
        action="store_true",
        help="A0 gate: strip train docs that 13-gram-overlap the eval sets before sharding.",
    )
    p.add_argument(
        "--eval-file",
        action="append",
        default=[],
        help="Extra eval-text file (one item per line) to guard against; repeatable.",
    )
    args = p.parse_args()

    if args.corpus is not None:
        text = Path(args.corpus).read_text(encoding="utf-8")
        docs = [d for d in text.split("\n\n") if d.strip()]
    else:
        docs = download_fineweb_slice(n_docs=args.n_docs)

    doc_filter = None
    if args.decontaminate:
        from scratch_llm.data.decontaminate import (
            build_eval_ngrams,
            decontam_doc_filter,
            default_eval_texts,
        )

        eval_ngrams = build_eval_ngrams(default_eval_texts(args.eval_file))
        doc_filter = decontam_doc_filter(eval_ngrams)
        kept = [d for d in docs if doc_filter(d)]
        print(
            f"A0 decontamination: kept {len(kept)}/{len(docs)} docs ({len(docs) - len(kept)} dropped)"
        )

    metas = build_dataset(
        docs, args.out, args.vocab_size, args.docs_per_shard, doc_filter=doc_filter
    )
    total = sum(m.n_tokens for m in metas)
    print(
        f"{len(metas)} shard(s), {total:,} tokens ({metas[0].dtype}), {len(docs)} docs → {args.out}"
    )


if __name__ == "__main__":
    main()
