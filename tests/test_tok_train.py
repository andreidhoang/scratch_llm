"""F12 `scripts/tok_train.py` — byte-budget determinism + tokenizer staging tests (CPU).

The load-bearing properties for the corpus ablation: the byte budget truncates at a UTF-8
boundary (never splits a multibyte char), and two corpora capped at the same budget train
byte-identical tokenizers when their capped prefixes match — that is what "both arms train
the tokenizer on the same byte budget" means operationally.
"""

import sys
from pathlib import Path

import pytest

from scratch_llm.data.shards import DOC_SEPARATOR, load_tokenizer


def _tok_train():
    """Import the CLI module (scripts/ is not a package)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import tok_train
    finally:
        sys.path.pop(0)
    return tok_train


_CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 40
    + "a language model learns to predict the next token. " * 40
)


def test_byte_budget_truncation_is_deterministic(tmp_path: Path) -> None:
    """Same file + same budget ⇒ identical staged bytes, valid UTF-8, within budget."""
    tt = _tok_train()
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(_CORPUS, encoding="utf-8")

    text_a, staged_a = tt.read_corpus_capped(corpus_file, 1000)
    text_b, staged_b = tt.read_corpus_capped(corpus_file, 1000)
    assert text_a == text_b
    assert staged_a == staged_b <= 1000
    text_a.encode("utf-8")  # would raise on a split sequence


def test_byte_budget_never_splits_a_multibyte_char(tmp_path: Path) -> None:
    """A budget landing mid-codepoint drops the partial char instead of corrupting it."""
    tt = _tok_train()
    corpus_file = tmp_path / "utf8.txt"
    corpus_file.write_text("héllo wörld — éèê", encoding="utf-8")
    full = corpus_file.read_bytes()
    for budget in range(1, len(full)):
        text, _ = tt.read_corpus_capped(corpus_file, budget)
        assert text.encode("utf-8") == full[: len(text.encode("utf-8"))]


def test_same_budget_same_tokenizer(tmp_path: Path) -> None:
    """Two corpora with a shared prefix, capped at the prefix length ⇒ identical vocabs."""
    tt = _tok_train()
    short = tmp_path / "short.txt"
    long = tmp_path / "long.txt"
    short.write_text(_CORPUS, encoding="utf-8")
    long.write_text(_CORPUS + "extra text only the long corpus carries. " * 40, encoding="utf-8")

    budget = len(_CORPUS.encode("utf-8"))
    tok_a = tt.train_tokenizer(short, tmp_path / "a", vocab_size=320, byte_budget=budget)
    tok_b = tt.train_tokenizer(long, tmp_path / "b", vocab_size=320, byte_budget=budget)
    assert tok_a.vocab == tok_b.vocab
    assert tok_a.merges == tok_b.merges


def test_directory_input_reads_txt_sorted(tmp_path: Path) -> None:
    tt = _tok_train()
    (tmp_path / "b.txt").write_text("second part. ", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first part. ", encoding="utf-8")
    text, _ = tt.read_corpus_capped(tmp_path, None)
    assert text == "first part. \nsecond part. "


def test_train_tokenizer_stages_loadable_tokenizer(tmp_path: Path) -> None:
    """The staged tokenizer.json round-trips through shards.load_tokenizer, carries <|eot|>."""
    tt = _tok_train()
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(_CORPUS, encoding="utf-8")
    out_dir = tmp_path / "tok"

    tokenizer = tt.train_tokenizer(corpus_file, out_dir, vocab_size=320, byte_budget=2048)
    assert (out_dir / "tokenizer.json").exists()
    reloaded = load_tokenizer(out_dir)

    doc = "the quick brown fox learns."
    assert reloaded.decode(reloaded.encode(doc)) == doc
    assert len(tokenizer.encode(DOC_SEPARATOR)) == 1  # the shard doc-separator contract


def test_cli_main_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tt = _tok_train()
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(_CORPUS, encoding="utf-8")
    out_dir = tmp_path / "cli_out"
    tt.main(
        [
            str(corpus_file),
            "--vocab-size",
            "300",
            "--byte-budget",
            "1500",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert (out_dir / "tokenizer.json").exists()
    assert "budget 1,500" in capsys.readouterr().out
