"""A1 real-corpus shards — executable spec (docs/FRONTIER_2026_TASKSPEC.md §A·A1).

DoD invariants (pre-registered in bench/RESULTS.md §Frontier ablations):
- **Round-trip:** the shard is exactly ``encode(d0)+[eot]+encode(d1)+[eot]``.
- **Size:** file bytes == ``itemsize·n_tokens`` — headerless (nanoGPT/nanochat convention).
- **Alignment:** ``get_batch`` on the loaded memmap yields (inputs, targets) shifted by one.
- **Kill-switch:** any token id ≥ 2¹⁶ switches the shard dtype to uint32 (no silent wraparound).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scratch_llm.data.shards import (
    DOC_SEPARATOR,
    build_dataset,
    download_fineweb_slice,
    load_dataset_tokens,
    load_shard,
    load_tokenizer,
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
# FineWeb slice — network, CI-skipped
# ---------------------------------------------------------------------------------------------


@pytest.mark.network
def test_download_fineweb_slice() -> None:
    if not os.environ.get("SCRATCH_LLM_NETWORK_TESTS"):
        pytest.skip("network test: set SCRATCH_LLM_NETWORK_TESTS=1 to run")
    docs = download_fineweb_slice(n_docs=8)
    assert len(docs) == 8
    assert all(isinstance(d, str) and d.strip() for d in docs)
