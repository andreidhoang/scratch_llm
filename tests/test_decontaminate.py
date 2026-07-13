"""A0 decontamination gate — executable spec (docs/FRONTIER_2026_TASKSPEC.md §A · A0).

DoD invariants (pre-registered):
- **Verbatim catch / paraphrase pass:** a train doc embedding a verbatim 13+-word eval sentence
  is dropped at threshold=0; the same doc with a few words changed passes.
- **Accounting:** overlap_rate is exact (hand-computed on a 4-doc set with 1 planted).
- **Short-item rule:** eval items shorter than n words guard via their full word-tuple.
- **Normalization symmetry:** UPPERCASE / extra-whitespace plantings are still caught.
- **Shards seam:** ``doc_filter=None`` is byte-identical to the pre-seam behavior; with the
  decontam filter the planted doc is absent from the shard.
- **Kill check surfaced:** >20% of a slice flagged sets ``suspicious=True`` (never auto-fails).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scratch_llm.data.decontaminate import (
    SUSPICIOUS_OVERLAP_RATE,
    DecontamResult,
    build_eval_ngrams,
    decontam_doc_filter,
    decontaminate_docs,
    default_eval_texts,
    ngram_overlap,
)
from scratch_llm.data.shards import build_dataset, load_shard, load_tokenizer, tokenize_to_shard

# ---------------------------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------------------------

# A 27-word (normalized) eval sentence — the first GSM fixture question, verbatim.
EVAL_SENTENCE = (
    "Natalia sold 48 clips in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?"
)

CLEAN_DOCS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the quiet forest "
    "until it reaches the river bank at dawn",
    "a language model learns to predict the next token from the previous ones by minimizing "
    "cross entropy over a large corpus of text",
    "we own every layer from the byte to the reinforcement update and measure each one "
    "against a pre-registered falsifiable prediction",
]


def _planted_doc(sentence: str = EVAL_SENTENCE) -> str:
    return f"{CLEAN_DOCS[0]} {sentence} {CLEAN_DOCS[1]}"


class _StubTokenizer:
    """Deterministic encoder emitting unique consecutive ids (mirrors tests/test_shards.py)."""

    def __init__(self, start: int = 1) -> None:
        self._next = start

    def encode(self, text: str) -> list[int]:
        ids = list(range(self._next, self._next + len(text)))
        self._next += len(text)
        return ids


# ---------------------------------------------------------------------------------------------
# build_eval_ngrams — the guard-set contract
# ---------------------------------------------------------------------------------------------


def test_build_eval_ngrams_shapes() -> None:
    grams = build_eval_ngrams([EVAL_SENTENCE], n=13)
    assert isinstance(grams, frozenset)
    assert grams and all(len(g) == 13 for g in grams)
    # 27 normalized words -> 27 - 13 + 1 = 15 sliding windows.
    assert len(grams) == 15

    short = build_eval_ngrams(["356"], n=13)
    assert short == frozenset({("356",)})  # full-tuple rule for short items

    assert build_eval_ngrams(["", "   ", "?!"]) == frozenset()  # nothing normalizable


# ---------------------------------------------------------------------------------------------
# ngram_overlap — hand-computed fractions
# ---------------------------------------------------------------------------------------------

_WORDS13 = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike"


def test_ngram_overlap_exact_fraction() -> None:
    grams = build_eval_ngrams([_WORDS13], n=13)
    # 14-word doc -> 2 windows; the first is the eval 13-gram, the second is shifted off it.
    doc = _WORDS13 + " november"
    assert ngram_overlap(doc, grams, n=13) == pytest.approx(0.5)
    assert ngram_overlap(_WORDS13, grams, n=13) == pytest.approx(1.0)
    assert ngram_overlap(CLEAN_DOCS[0], grams, n=13) == 0.0
    assert ngram_overlap("", grams, n=13) == 0.0


def test_ngram_overlap_short_doc_full_tuple_rule() -> None:
    grams = build_eval_ngrams(["356"], n=13)  # a Countdown-target-like short eval item
    assert ngram_overlap("356", grams) == 1.0
    assert ngram_overlap("  356! ", grams) == 1.0  # normalization: punctuation/whitespace
    # Documented limitation: short eval items only guard (normalized-)identical short docs.
    assert ngram_overlap("356 extra", grams) == 0.0
    # A long doc's 13-grams can never equal a shorter tuple.
    assert ngram_overlap(CLEAN_DOCS[0] + " 356", grams) == 0.0


# ---------------------------------------------------------------------------------------------
# decontaminate_docs — verbatim catch, paraphrase pass, accounting, kill check
# ---------------------------------------------------------------------------------------------


def test_verbatim_planting_dropped_paraphrase_passes() -> None:
    grams = build_eval_ngrams([EVAL_SENTENCE])
    planted = _planted_doc()

    # Paraphrase: words changed every <13 positions, so no 13-word window survives verbatim.
    paraphrased_sentence = (
        "Natalia offered 48 clips in April, and then she sold roughly half as many clips "
        "in May. Approximately how many clips did Natalia sell altogether in April and May?"
    )
    assert ngram_overlap(paraphrased_sentence, grams) == 0.0  # self-check on the paraphrase
    paraphrased = _planted_doc(paraphrased_sentence)

    result = decontaminate_docs([CLEAN_DOCS[0], planted, paraphrased, CLEAN_DOCS[2]], grams)
    assert isinstance(result, DecontamResult)
    assert result.kept == [CLEAN_DOCS[0], paraphrased, CLEAN_DOCS[2]]
    assert result.n_dropped == 1 and result.n_kept == 3


def test_overlap_rate_hand_computed_on_four_docs() -> None:
    grams = build_eval_ngrams([EVAL_SENTENCE])
    planted = _planted_doc()
    result = decontaminate_docs([*CLEAN_DOCS, planted], grams)

    assert result.n_kept == 3 and result.n_dropped == 1
    assert result.overlap_rate == pytest.approx(1 / 4)
    assert result.suspicious is True  # 25% > 20% — surfaced, not auto-failed
    assert len(result.dropped_samples) == 1
    assert result.dropped_samples[0] == planted[:80]


def test_threshold_semantics_keep_at_or_below() -> None:
    grams = build_eval_ngrams([_WORDS13], n=13)
    half_overlap_doc = _WORDS13 + " november"  # overlap exactly 0.5

    at_zero = decontaminate_docs([half_overlap_doc], grams, n=13, threshold=0.0)
    assert at_zero.n_dropped == 1
    at_half = decontaminate_docs([half_overlap_doc], grams, n=13, threshold=0.5)
    assert at_half.n_dropped == 0 and at_half.kept == [half_overlap_doc]


def test_suspicious_kill_check_boundary(caplog: pytest.LogCaptureFixture) -> None:
    assert pytest.approx(0.20) == SUSPICIOUS_OVERLAP_RATE
    grams = build_eval_ngrams([EVAL_SENTENCE])
    planted = _planted_doc()

    # Exactly 20% (1 of 5): NOT suspicious (strictly-greater rule).
    ok = decontaminate_docs([*CLEAN_DOCS, CLEAN_DOCS[0], planted], grams)
    assert ok.overlap_rate == pytest.approx(0.2) and ok.suspicious is False

    # 50% (2 of 4): suspicious — logged and marked, docs still dropped (investigate, don't trust).
    with caplog.at_level("WARNING", logger="scratch_llm.data.decontaminate"):
        sus = decontaminate_docs([CLEAN_DOCS[0], planted, planted, CLEAN_DOCS[1]], grams)
    assert sus.suspicious is True and sus.n_dropped == 2
    assert any("suspicious" in rec.message.lower() for rec in caplog.records)


def test_empty_input_is_clean() -> None:
    result = decontaminate_docs([], build_eval_ngrams([EVAL_SENTENCE]))
    assert result.n_kept == 0 and result.n_dropped == 0
    assert result.overlap_rate == 0.0 and result.suspicious is False


# ---------------------------------------------------------------------------------------------
# Normalization symmetry — the gate silently fails if the two sides normalize differently
# ---------------------------------------------------------------------------------------------


def test_uppercase_and_whitespace_plantings_still_caught() -> None:
    grams = build_eval_ngrams([EVAL_SENTENCE])

    upper = _planted_doc(EVAL_SENTENCE.upper())
    spaced = _planted_doc(EVAL_SENTENCE.replace(" ", "  \t").replace(". ", ".\n\n"))
    unpunctuated = _planted_doc(EVAL_SENTENCE.replace(",", "").replace(".", "").replace("?", ""))

    for doc in (upper, spaced, unpunctuated):
        assert ngram_overlap(doc, grams) > 0.0
    result = decontaminate_docs([upper, spaced, unpunctuated, CLEAN_DOCS[0]], grams)
    assert result.n_dropped == 3 and result.kept == [CLEAN_DOCS[0]]


# ---------------------------------------------------------------------------------------------
# default_eval_texts — in-repo guard corpus + extra_paths for non-vendored sets
# ---------------------------------------------------------------------------------------------


def test_default_eval_texts_covers_in_repo_sets(tmp_path: Path) -> None:
    texts = default_eval_texts()
    assert EVAL_SENTENCE in texts  # GSM fixture questions, verbatim
    assert "72" in texts  # ... and their short answers (full-tuple guards)
    assert any("arithmetic expression" in t for t in texts)  # Countdown questions

    # extra_paths: one text per line, blank lines skipped (real GSM8K/MMLU supplied this way).
    extra = tmp_path / "mmlu.txt"
    extra.write_text("first mmlu question text\n\nsecond mmlu question text\n", encoding="utf-8")
    with_extra = default_eval_texts(extra_paths=[extra])
    assert "first mmlu question text" in with_extra
    assert "second mmlu question text" in with_extra
    assert len(with_extra) == len(texts) + 2


def test_default_guard_catches_countdown_template_for_any_seed() -> None:
    # The Countdown question template's tail (>=13 words) is seed-independent, so the default
    # guard set catches a rendered question even for numbers/targets outside the default pool.
    from scratch_llm.envs.countdown import render_countdown_question

    grams = build_eval_ngrams(default_eval_texts())
    foreign_question = render_countdown_question([97, 89, 83], 269069)
    assert ngram_overlap(_planted_doc(foreign_question), grams) > 0.0

    gsm_planted = _planted_doc()  # GSM fixture question through the default guard set
    result = decontaminate_docs([CLEAN_DOCS[0], gsm_planted], grams)
    assert result.n_dropped == 1 and result.kept == [CLEAN_DOCS[0]]


# ---------------------------------------------------------------------------------------------
# Shards seam — doc_filter threading (strictly additive)
# ---------------------------------------------------------------------------------------------


def test_tokenize_to_shard_doc_filter_none_is_byte_identical(tmp_path: Path) -> None:
    docs = ["ab", "cd"]
    out_default = tmp_path / "default.bin"
    out_none = tmp_path / "none.bin"
    meta_default = tokenize_to_shard(docs, _StubTokenizer(), 0, out_default)
    meta_none = tokenize_to_shard(docs, _StubTokenizer(), 0, out_none, doc_filter=None)

    # Pinned expected bytes: encode("ab")+eot+encode("cd")+eot as headerless uint16.
    expected = np.array([1, 2, 0, 3, 4, 0], dtype=np.uint16).tobytes()
    assert out_default.read_bytes() == expected
    assert out_none.read_bytes() == expected
    assert meta_default == meta_none
    assert meta_default.n_docs_filtered == 0


def test_tokenize_to_shard_doc_filter_drops_and_counts(tmp_path: Path) -> None:
    grams = build_eval_ngrams([EVAL_SENTENCE])
    keep = decontam_doc_filter(grams)
    docs = [CLEAN_DOCS[0], _planted_doc(), CLEAN_DOCS[1]]

    out_filtered = tmp_path / "filtered.bin"
    meta = tokenize_to_shard(docs, _StubTokenizer(), 0, out_filtered, doc_filter=keep)
    out_clean = tmp_path / "clean.bin"
    tokenize_to_shard([CLEAN_DOCS[0], CLEAN_DOCS[1]], _StubTokenizer(), 0, out_clean)

    assert out_filtered.read_bytes() == out_clean.read_bytes()
    assert meta.n_docs == 2 and meta.n_docs_filtered == 1
    tokens, loaded = load_shard(out_filtered)
    assert loaded == meta and tokens.size == meta.n_tokens


def test_tokenize_to_shard_all_docs_filtered_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tokenize_to_shard(
            ["a", "b"], _StubTokenizer(), 0, tmp_path / "empty.bin", doc_filter=lambda _: False
        )


def test_build_dataset_doc_filter_removes_planted_doc(tmp_path: Path) -> None:
    grams = build_eval_ngrams([EVAL_SENTENCE])
    keep = decontam_doc_filter(grams)
    planted = _planted_doc()
    dirty = [CLEAN_DOCS[0], planted, CLEAN_DOCS[1]]
    clean = [CLEAN_DOCS[0], CLEAN_DOCS[1]]

    filtered_dir = tmp_path / "filtered"
    clean_dir = tmp_path / "clean"
    metas = build_dataset(dirty, filtered_dir, vocab_size=300, doc_filter=keep)
    build_dataset(clean, clean_dir, vocab_size=300)

    # The filter runs BEFORE BPE training, so the whole dataset (vocab + shards) is
    # byte-identical to one built from the clean corpus alone.
    assert sum(m.n_docs for m in metas) == 2
    filtered_bins = sorted(filtered_dir.glob("*.bin"))
    clean_bins = sorted(clean_dir.glob("*.bin"))
    assert [p.read_bytes() for p in filtered_bins] == [p.read_bytes() for p in clean_bins]

    # And the planted eval sentence is absent from the decoded shard text.
    tok = load_tokenizer(filtered_dir)
    decoded = tok.decode(load_shard(filtered_bins[0])[0].tolist())
    assert EVAL_SENTENCE not in decoded


def test_build_dataset_doc_filter_none_is_byte_identical(tmp_path: Path) -> None:
    docs = [CLEAN_DOCS[0], CLEAN_DOCS[1]]
    dir_default = tmp_path / "default"
    dir_none = tmp_path / "none"
    build_dataset(docs, dir_default, vocab_size=300)
    build_dataset(docs, dir_none, vocab_size=300, doc_filter=None)

    default_bins = sorted(dir_default.glob("*.bin"))
    none_bins = sorted(dir_none.glob("*.bin"))
    assert [p.read_bytes() for p in default_bins] == [p.read_bytes() for p in none_bins]
    assert (dir_default / "tokenizer.json").read_bytes() == (
        dir_none / "tokenizer.json"
    ).read_bytes()


def test_pre_a0_sidecar_without_filtered_key_still_loads(tmp_path: Path) -> None:
    import json

    out = tmp_path / "old.bin"
    tokenize_to_shard(["ab"], _StubTokenizer(), 0, out)
    sidecar = out.with_suffix(".meta.json")
    meta_raw = json.loads(sidecar.read_text(encoding="utf-8"))
    meta_raw.pop("n_docs_filtered")  # simulate a shard written before the A0 seam existed
    sidecar.write_text(json.dumps(meta_raw), encoding="utf-8")

    _, meta = load_shard(out)
    assert meta.n_docs_filtered == 0


def test_build_dataset_all_docs_filtered_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filter"):
        build_dataset([CLEAN_DOCS[0]], tmp_path / "d", vocab_size=300, doc_filter=lambda _: False)
