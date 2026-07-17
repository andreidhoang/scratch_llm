"""A1 real-corpus shards — executable spec (docs/FRONTIER_2026_TASKSPEC.md §A·A1).

DoD invariants (pre-registered in bench/RESULTS.md §Frontier ablations):
- **Round-trip:** the shard is exactly ``encode(d0)+[eot]+encode(d1)+[eot]``.
- **Size:** file bytes == ``itemsize·n_tokens`` — headerless (nanoGPT/nanochat convention).
- **Alignment:** ``get_batch`` on the loaded memmap yields (inputs, targets) shifted by one.
- **Kill-switch:** any token id ≥ 2¹⁶ switches the shard dtype to uint32 (no silent wraparound).

F1-scale streaming path (this file too; parquet ingest itself in test_shards_parquet.py):
- **Byte-identity:** the streaming writer produces the same bytes as the buffered path.
- **MP == SP:** multiprocess tokenization output equals single-process, byte for byte.
- **Budget:** ``target_tokens`` stops within one doc-batch of the budget.
- **Plan:** the parquet download plan is dry-runnable and covers the token budget — no network.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scratch_llm.data.shards import (
    DOC_SEPARATOR,
    ParquetFileInfo,
    build_dataset,
    build_dataset_streaming,
    download_fineweb_slice,
    download_parquet_files,
    hub_tree_parquet_infos,
    iter_parquet_doc_batches,
    list_hub_parquet_files,
    load_dataset_tokens,
    load_shard,
    load_tokenizer,
    plan_parquet_download,
    stream_tokenize_to_shards,
    tokenize_to_shard,
)
from scratch_llm.tokenizer import Tokenizer, train_bpe
from scratch_llm.train import get_batch

# ---------------------------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------------------------

_DOCS = [
    "the quick brown fox jumps over the lazy dog. " * 8,
    "a language model learns to predict the next token from the previous ones. " * 8,
    "we own every layer from the byte to the reinforcement update. " * 8,
]


def _train_real_tokenizer(tmp_path: Path, vocab_size: int = 300) -> Tokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(DOC_SEPARATOR.join(_DOCS), encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size, special_tokens=[DOC_SEPARATOR])
    return Tokenizer(vocab, merges, special_tokens=[DOC_SEPARATOR])


class _StubTokenizer:
    """Deterministic encoder emitting unique consecutive ids — makes stream positions
    identifiable for the alignment test, and lets the uint32 kill-switch be forced."""

    def __init__(self, start: int = 1) -> None:
        self._next = start

    def encode(self, text: str) -> list[int]:
        ids = list(range(self._next, self._next + len(text)))
        self._next += len(text)
        return ids


# ---------------------------------------------------------------------------------------------
# tokenize_to_shard / load_shard — the format contract
# ---------------------------------------------------------------------------------------------


def test_shard_round_trips_docs_with_eot(tmp_path: Path) -> None:
    tok = _train_real_tokenizer(tmp_path)
    eot_id = tok.encode(DOC_SEPARATOR)
    assert len(eot_id) == 1  # a special must be exactly one id
    eot = eot_id[0]

    out = tmp_path / "shard_00000.bin"
    meta = tokenize_to_shard(_DOCS[:2], tok, eot, out)
    tokens, loaded_meta = load_shard(out)

    expected = tok.encode(_DOCS[0]) + [eot] + tok.encode(_DOCS[1]) + [eot]
    assert tokens.tolist() == expected
    assert meta.n_tokens == len(expected)
    assert meta.n_docs == 2
    assert meta.eot_id == eot
    assert loaded_meta == meta


def test_shard_file_is_headerless_uint16(tmp_path: Path) -> None:
    tok = _train_real_tokenizer(tmp_path)
    eot = tok.encode(DOC_SEPARATOR)[0]
    out = tmp_path / "shard_00000.bin"
    meta = tokenize_to_shard(_DOCS[:1], tok, eot, out)

    assert meta.dtype == "uint16"
    assert out.stat().st_size == 2 * meta.n_tokens  # DoD: file size == 2·n_tokens, no header


def test_uint32_kill_switch(tmp_path: Path) -> None:
    """Pre-registered kill criterion: any id ≥ 2¹⁶ must switch the dtype, not wrap around."""
    tok = _StubTokenizer(start=2**16 - 2)  # ids straddle the uint16 boundary
    out = tmp_path / "big.bin"
    meta = tokenize_to_shard(["abcd"], tok, 0, out)

    assert meta.dtype == "uint32"
    assert out.stat().st_size == 4 * meta.n_tokens
    tokens, _ = load_shard(out)
    assert tokens.tolist() == [2**16 - 2, 2**16 - 1, 2**16, 2**16 + 1, 0]


def test_empty_docs_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tokenize_to_shard([], _StubTokenizer(), 0, tmp_path / "empty.bin")


def test_load_shard_rejects_truncated_file(tmp_path: Path) -> None:
    tok = _StubTokenizer()
    out = tmp_path / "shard.bin"
    tokenize_to_shard(["hello world"], tok, 0, out)
    out.write_bytes(out.read_bytes()[:-2])  # corrupt: drop one token
    with pytest.raises(ValueError, match="size"):
        load_shard(out)


# ---------------------------------------------------------------------------------------------
# get_batch consumes the shard — the training-loop contract (DoD)
# ---------------------------------------------------------------------------------------------


def test_get_batch_is_next_token_aligned_on_memmap(tmp_path: Path) -> None:
    tok = _StubTokenizer()
    out = tmp_path / "shard.bin"
    tokenize_to_shard(["x" * 150, "y" * 149], tok, 0, out)
    tokens, meta = load_shard(out)
    assert isinstance(tokens, np.memmap)  # zero-copy load
    stream = np.asarray(tokens, dtype=np.int64)

    batch_size, context = 4, 8
    inputs, targets = get_batch(tokens, batch_size, context)
    assert inputs.dtype.is_floating_point is False and inputs.dtype.is_signed  # int64 via .long()

    for b in range(batch_size):
        row = inputs[b].numpy()
        # Locate the sampled window in the stream (ids are unique except the two eots).
        candidates = [
            s
            for s in np.flatnonzero(stream[: len(stream) - context] == row[0])
            if np.array_equal(stream[s : s + context], row)
        ]
        assert len(candidates) == 1, "sampled window must exist uniquely in the token stream"
        s = candidates[0]
        assert np.array_equal(targets[b].numpy(), stream[s + 1 : s + 1 + context])


# ---------------------------------------------------------------------------------------------
# build_dataset / load_dataset_tokens / load_tokenizer — the directory contract
# ---------------------------------------------------------------------------------------------


def test_build_dataset_writes_tokenizer_and_shards(tmp_path: Path) -> None:
    out_dir = tmp_path / "data"
    metas = build_dataset(_DOCS, out_dir, vocab_size=300)

    assert (out_dir / "tokenizer.json").exists()
    bins = sorted(out_dir.glob("*.bin"))
    assert len(bins) == len(metas) == 1
    assert (out_dir / f"{bins[0].stem}.meta.json").exists()

    tok = load_tokenizer(out_dir)
    eot = tok.encode(DOC_SEPARATOR)
    assert len(eot) == 1 and eot[0] == metas[0].eot_id

    tokens = load_dataset_tokens(out_dir)
    expected: list[int] = []
    for doc in _DOCS:
        expected.extend(tok.encode(doc))
        expected.append(eot[0])
    assert tokens.tolist() == expected

    # The reloaded tokenizer round-trips text, including the special.
    text = f"the quick brown fox{DOC_SEPARATOR}a language model"
    assert tok.decode(tok.encode(text)) == text


def test_multi_shard_concat_in_sorted_order(tmp_path: Path) -> None:
    out_dir = tmp_path / "data"
    metas = build_dataset(_DOCS, out_dir, vocab_size=300, docs_per_shard=1)
    assert len(metas) == 3
    assert sum(m.n_tokens for m in metas) == load_dataset_tokens(out_dir).size

    parts = [load_shard(p)[0] for p in sorted(out_dir.glob("*.bin"))]
    assert load_dataset_tokens(out_dir).tolist() == np.concatenate(parts).tolist()


def test_load_dataset_tokens_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset_tokens(tmp_path)


# ---------------------------------------------------------------------------------------------
# speedrun wiring — the shard-backed pretrain path
# ---------------------------------------------------------------------------------------------


def test_speedrun_pretrains_from_shards(tmp_path: Path) -> None:
    from scratch_llm.speedrun import SpeedrunConfig, run_speedrun

    out_dir = tmp_path / "data"
    metas = build_dataset(_DOCS, out_dir, vocab_size=300)

    result = run_speedrun(
        SpeedrunConfig(
            depth=2,
            context_length=32,
            train_steps=3,
            batch_size=4,
            data_dir=str(out_dir),
            sample_tokens=8,
        )
    )
    assert result.stages == ["shards", "pretrain", "eval", "sample"]
    assert result.n_tokens == sum(m.n_tokens for m in metas)
    assert isinstance(result.sample, str) and result.sample
    assert result.report_card.val_bpb is not None and result.report_card.val_bpb > 0


# ---------------------------------------------------------------------------------------------
# Streaming builder (F1-data) — bytes identical to the buffered path, MP == SP, budget stop
# ---------------------------------------------------------------------------------------------


def _batched(docs: list[str], n: int) -> list[list[str]]:
    return [docs[i : i + n] for i in range(0, len(docs), n)]


def test_streaming_single_shard_bytes_match_buffered_path(tmp_path: Path) -> None:
    tok = _train_real_tokenizer(tmp_path)
    eot = tok.encode(DOC_SEPARATOR)[0]
    ref = tmp_path / "ref.bin"
    ref_meta = tokenize_to_shard(_DOCS, tok, eot, ref)

    out_dir = tmp_path / "stream"
    metas, stats = stream_tokenize_to_shards(iter(_batched(_DOCS, 2)), tok, eot, out_dir)

    assert len(metas) == 1
    assert (out_dir / "shard_00000.bin").read_bytes() == ref.read_bytes()
    assert metas[0] == ref_meta  # sidecar contract unchanged, field for field
    assert stats.n_docs == len(_DOCS) and stats.n_shards == 1
    assert stats.n_tokens == ref_meta.n_tokens
    assert stats.n_text_bytes == sum(len(d.encode("utf-8")) for d in _DOCS)
    assert stats.bytes_per_token == pytest.approx(stats.n_text_bytes / stats.n_tokens)


def test_streaming_multi_shard_splits_at_doc_boundaries(tmp_path: Path) -> None:
    tok = _train_real_tokenizer(tmp_path)
    eot = tok.encode(DOC_SEPARATOR)[0]
    ref = tmp_path / "ref.bin"
    tokenize_to_shard(_DOCS, tok, eot, ref)

    out_dir = tmp_path / "stream"
    metas, stats = stream_tokenize_to_shards(
        iter(_batched(_DOCS, 1)), tok, eot, out_dir, tokens_per_shard=10
    )

    assert len(metas) == 3  # every doc exceeds 10 tokens → flush at each doc boundary
    assert stats.n_shards == 3
    assert load_dataset_tokens(out_dir).tolist() == load_shard(ref)[0].tolist()
    for p in sorted(out_dir.glob("*.bin")):
        tokens, meta = load_shard(p)  # each shard honors the size invariant on load
        assert tokens[-1] == meta.eot_id  # doc-boundary split: a shard always ends on EOT


def test_multiprocess_tokenization_matches_single_process(tmp_path: Path) -> None:
    tok = _train_real_tokenizer(tmp_path)
    eot = tok.encode(DOC_SEPARATOR)[0]
    docs = [f"{d} variant {i}" for i, d in enumerate(_DOCS * 4)]
    batches = _batched(docs, 3)

    sp_dir, mp_dir = tmp_path / "sp", tmp_path / "mp"
    sp_metas, sp_stats = stream_tokenize_to_shards(iter(batches), tok, eot, sp_dir, num_workers=0)
    mp_metas, mp_stats = stream_tokenize_to_shards(iter(batches), tok, eot, mp_dir, num_workers=2)

    assert mp_metas == sp_metas
    assert mp_stats == sp_stats
    sp_bins, mp_bins = sorted(sp_dir.glob("*.bin")), sorted(mp_dir.glob("*.bin"))
    for sp_bin, mp_bin in zip(sp_bins, mp_bins, strict=True):
        assert mp_bin.read_bytes() == sp_bin.read_bytes()


def test_unpicklable_tokenizer_falls_back_to_single_process(tmp_path: Path) -> None:
    class LocalEncoder:  # function-local class: pickle-by-reference fails → must fall back
        def encode(self, text: str) -> list[int]:
            return [len(text)]

    metas, stats = stream_tokenize_to_shards(
        iter([["ab", "cd"]]), LocalEncoder(), 1, tmp_path / "fb", num_workers=4
    )
    assert stats.n_docs == 2 and len(metas) == 1  # fell back in-process, no worker crash
    assert load_shard(tmp_path / "fb" / "shard_00000.bin")[0].tolist() == [2, 1, 2, 1]


def test_target_tokens_stops_within_one_batch(tmp_path: Path) -> None:
    tok = _StubTokenizer()
    docs_per_batch, doc_len = 4, 50  # each batch = 4·(50+1) = 204 tokens
    batches = [["x" * doc_len] * docs_per_batch for _ in range(10)]
    target = 500

    metas, stats = stream_tokenize_to_shards(
        iter(batches), tok, 0, tmp_path / "budget", target_tokens=target
    )

    batch_tokens = docs_per_batch * (doc_len + 1)
    assert stats.n_tokens >= target  # budget reached …
    assert stats.n_tokens - batch_tokens < target  # … within ONE doc-batch of it
    assert stats.n_tokens == sum(m.n_tokens for m in metas)  # accounting == written shards
    assert stats.n_docs == 3 * docs_per_batch  # ceil(500/204) = 3 batches consumed


def test_build_dataset_streaming_writes_tokenizer_and_shards(tmp_path: Path) -> None:
    out_dir = tmp_path / "data"
    # Cap of 256 B is crossed by the first doc (doc-granular) — the tokenizer trains on a
    # truncated sample, but the FULL stream is sharded.
    metas, stats = build_dataset_streaming(
        lambda: iter(_batched(_DOCS, 2)), out_dir, vocab_size=300, max_train_bytes=256
    )

    assert (out_dir / "tokenizer.json").exists()
    tok = load_tokenizer(out_dir)
    eot = tok.encode(DOC_SEPARATOR)
    assert len(eot) == 1 and eot[0] == metas[0].eot_id

    expected: list[int] = []
    for doc in _DOCS:
        expected.extend(tok.encode(doc))
        expected.append(eot[0])
    assert load_dataset_tokens(out_dir).tolist() == expected
    assert stats.n_docs == len(_DOCS) and stats.n_tokens == len(expected)


def test_build_dataset_streaming_matches_buffered_when_uncapped(tmp_path: Path) -> None:
    buf_dir, stream_dir = tmp_path / "buffered", tmp_path / "stream"
    build_dataset(_DOCS, buf_dir, vocab_size=300)
    build_dataset_streaming(
        lambda: iter(_batched(_DOCS, 2)), stream_dir, vocab_size=300, max_train_bytes=1 << 30
    )

    # Same staged BPE bytes → same tokenizer → byte-identical shards.
    tok_json = (stream_dir / "tokenizer.json").read_bytes()
    assert tok_json == (buf_dir / "tokenizer.json").read_bytes()
    shard = (stream_dir / "shard_00000.bin").read_bytes()
    assert shard == (buf_dir / "shard_00000.bin").read_bytes()


def test_stream_empty_input_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one document"):
        stream_tokenize_to_shards(iter([]), _StubTokenizer(), 0, tmp_path / "empty")


def test_stream_all_docs_filtered_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rejected all"):
        stream_tokenize_to_shards(
            iter([["a", "b"]]), _StubTokenizer(), 0, tmp_path / "filt", doc_filter=lambda _: False
        )


# ---------------------------------------------------------------------------------------------
# Parquet bulk-download planning + guarded import — pure functions, no network
# ---------------------------------------------------------------------------------------------


def _fake_files(n: int, size: int) -> list[ParquetFileInfo]:
    return [
        ParquetFileInfo(
            path=f"sample/10BT/{i:03d}_00000.parquet",
            size=size,
            url=f"https://huggingface.co/datasets/x/resolve/main/sample/10BT/{i:03d}_00000.parquet",
        )
        for i in range(n)
    ]


def test_hub_tree_parquet_infos_filters_and_resolves() -> None:
    entries = [
        {"type": "file", "path": "sample/10BT/001_00000.parquet", "size": 2},
        {"type": "file", "path": "sample/10BT/000_00000.parquet", "size": 1},
        {"type": "directory", "path": "sample/10BT/sub", "size": 0},
        {"type": "file", "path": "sample/10BT/README.md", "size": 3},
    ]
    infos = hub_tree_parquet_infos(entries, dataset="HuggingFaceFW/fineweb-edu", revision="main")

    # Parquet files only, sorted by path — directories and non-parquet entries dropped.
    assert [f.path for f in infos] == [
        "sample/10BT/000_00000.parquet",
        "sample/10BT/001_00000.parquet",
    ]
    assert infos[0].size == 1
    assert infos[0].url == (
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"
        "/resolve/main/sample/10BT/000_00000.parquet"
    )


def test_plan_parquet_download_covers_budget_with_minimal_prefix() -> None:
    files = _fake_files(10, size=1000)
    plan = plan_parquet_download(files, target_tokens=2000, bytes_per_token=2.0, safety=1.0)

    # need = 2000 tokens · 2.0 B/token = 4000 bytes → exactly 4 files; a 5th would be waste.
    assert plan.files == tuple(files[:4])  # prefix, in hub-sorted order
    assert plan.total_bytes == 4000
    assert plan.est_tokens == 2000


def test_plan_parquet_download_caps_at_all_files() -> None:
    files = _fake_files(2, size=100)
    plan = plan_parquet_download(files, target_tokens=10**9)
    assert plan.files == tuple(files)  # short corpus: plan = everything …
    assert plan.est_tokens < 10**9  # … and the shortfall is visible, not hidden


def test_download_plan_describe_is_the_dry_run_report() -> None:
    files = _fake_files(3, size=1 << 20)
    plan = plan_parquet_download(files, target_tokens=1_572_864, bytes_per_token=2.0, safety=1.0)
    report = plan.describe()

    assert "3 file(s)" in report
    assert "002_00000.parquet" in report  # every planned file listed by name …
    assert "1.0 MiB" in report and "GiB" in report  # … with sizes, before any download


def test_download_parquet_files_skips_exact_size_cached(tmp_path: Path) -> None:
    # url is unreachable on purpose: an exact-size cache hit must never touch the network.
    info = ParquetFileInfo(
        path="sample/10BT/000_00000.parquet", size=4, url="https://invalid.example/nope"
    )
    dest = tmp_path / "000_00000.parquet"
    dest.write_bytes(b"abcd")

    msgs: list[str] = []
    paths = download_parquet_files([info], tmp_path, progress=msgs.append)

    assert paths == [dest]
    assert any(m.startswith("cached") for m in msgs)


def test_missing_pyarrow_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)  # None entry → ImportError on import
    with pytest.raises(ImportError, match=r"\[data\]"):
        next(iter_parquet_doc_batches(["never-opened.parquet"]))


# ---------------------------------------------------------------------------------------------
# FineWeb slice + hub listing — network, CI-skipped
# ---------------------------------------------------------------------------------------------


@pytest.mark.network
def test_list_hub_parquet_files() -> None:
    if not os.environ.get("SCRATCH_LLM_NETWORK_TESTS"):
        pytest.skip("network test: set SCRATCH_LLM_NETWORK_TESTS=1 to run")
    files = list_hub_parquet_files()
    assert files and all(f.path.endswith(".parquet") and f.size > 0 for f in files)


@pytest.mark.network
def test_download_fineweb_slice() -> None:
    if not os.environ.get("SCRATCH_LLM_NETWORK_TESTS"):
        pytest.skip("network test: set SCRATCH_LLM_NETWORK_TESTS=1 to run")
    docs = download_fineweb_slice(n_docs=8)
    assert len(docs) == 8
    assert all(isinstance(d, str) and d.strip() for d in docs)
