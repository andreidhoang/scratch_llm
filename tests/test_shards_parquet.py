"""F1-data parquet ingest — executable spec for the streaming bulk path (shards.py).

Invariants:
- **Batch-wise streaming:** ``iter_parquet_doc_batches`` yields the parquet rows in file
  order, ≤ ``batch_size`` docs at a time, text column only; null/empty texts are dropped.
- **Round-trip:** parquet → streaming shards is byte-identical to the buffered path
  (:func:`tokenize_to_shard`) on the same documents.
- **End-to-end:** ``build_dataset_streaming`` over a parquet source stages a tokenizer and
  shards that ``load_dataset_tokens`` round-trips.

The parquet files are WRITTEN HERE via pyarrow — no network. pyarrow is the optional
``data`` extra, so the whole module skips without it (the CPU commit gate stays hermetic).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scratch_llm.data.shards import (
    DOC_SEPARATOR,
    build_dataset_streaming,
    iter_parquet_doc_batches,
    load_dataset_tokens,
    load_tokenizer,
    stream_tokenize_to_shards,
    tokenize_to_shard,
)
from scratch_llm.tokenizer import Tokenizer, train_bpe

pa = pytest.importorskip("pyarrow", reason="parquet tests need the data extra (pyarrow)")
pq = pytest.importorskip("pyarrow.parquet")

_DOCS = [
    "the quick brown fox jumps over the lazy dog. " * 6,
    "a language model learns to predict the next token from the previous ones. " * 6,
    "we own every layer from the byte to the reinforcement update. " * 6,
    "streaming shards must never hold the corpus in memory at once. " * 6,
]


def _train_tokenizer(tmp_path: Path, docs: list[str], vocab_size: int = 300) -> Tokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(DOC_SEPARATOR.join(docs), encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size, special_tokens=[DOC_SEPARATOR])
    return Tokenizer(vocab, merges, special_tokens=[DOC_SEPARATOR])


def _write_parquet(path: Path, texts: list[str | None]) -> None:
    # A second column proves the reader projects text_column only.
    table = pa.table({"text": texts, "url": [f"u{i}" for i in range(len(texts))]})
    pq.write_table(table, path)


# ---------------------------------------------------------------------------------------------
# iter_parquet_doc_batches — the streaming reader contract
# ---------------------------------------------------------------------------------------------


def test_iter_parquet_doc_batches_streams_in_order(tmp_path: Path) -> None:
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _write_parquet(p1, ["d0", "d1", "d2"])
    _write_parquet(p2, ["d3", None, "", "d4"])  # null + empty rows carry no tokens: dropped

    batches = list(iter_parquet_doc_batches([p1, p2], batch_size=2))

    assert [d for b in batches for d in b] == ["d0", "d1", "d2", "d3", "d4"]
    assert all(len(b) <= 2 for b in batches)


def test_iter_parquet_doc_batches_custom_text_column(tmp_path: Path) -> None:
    p = tmp_path / "c.parquet"
    pq.write_table(pa.table({"content": ["x", "y"]}), p)
    assert list(iter_parquet_doc_batches([p], text_column="content")) == [["x", "y"]]


# ---------------------------------------------------------------------------------------------
# parquet → shards — byte-identical to the buffered path, then end-to-end
# ---------------------------------------------------------------------------------------------


def test_parquet_round_trip_matches_buffered_shard(tmp_path: Path) -> None:
    parquet = tmp_path / "docs.parquet"
    _write_parquet(parquet, list(_DOCS))
    tok = _train_tokenizer(tmp_path, _DOCS)
    eot = tok.encode(DOC_SEPARATOR)[0]

    ref = tmp_path / "ref.bin"
    ref_meta = tokenize_to_shard(_DOCS, tok, eot, ref)

    out_dir = tmp_path / "stream"
    metas, stats = stream_tokenize_to_shards(
        iter_parquet_doc_batches([parquet], batch_size=2), tok, eot, out_dir
    )

    assert len(metas) == 1 and metas[0] == ref_meta
    assert (out_dir / "shard_00000.bin").read_bytes() == ref.read_bytes()
    assert stats.n_docs == len(_DOCS) and stats.n_tokens == ref_meta.n_tokens


def test_build_dataset_streaming_from_parquet_end_to_end(tmp_path: Path) -> None:
    parquet = tmp_path / "docs.parquet"
    _write_parquet(parquet, list(_DOCS))

    out_dir = tmp_path / "data"
    metas, stats = build_dataset_streaming(
        lambda: iter_parquet_doc_batches([parquet], batch_size=2), out_dir, vocab_size=300
    )

    tok = load_tokenizer(out_dir)
    eot = tok.encode(DOC_SEPARATOR)
    assert len(eot) == 1 and eot[0] == metas[0].eot_id

    expected: list[int] = []
    for doc in _DOCS:
        expected.extend(tok.encode(doc))
        expected.append(eot[0])
    assert load_dataset_tokens(out_dir).tolist() == expected
    assert stats.n_docs == len(_DOCS) and stats.n_tokens == len(expected)
