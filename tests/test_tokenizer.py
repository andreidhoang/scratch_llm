"""Tests for the byte-level BPE tokenizer.

The headline test reproduces the stylized CS336/Sennrich ``bpe_example`` exactly — it is
the cheapest correctness oracle for the merge core (counting + lexicographic tie-break).
"""

from pathlib import Path

from scratch_llm.tokenizer import (
    Tokenizer,
    _compute_merges,
    _pretokenize_counts,
    train_bpe,
)


def _word(s: bytes) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in s)


def test_compute_merges_reproduces_bpe_example() -> None:
    # Whitespace-split frequencies from the PDF: {low:5, lower:2, widest:3, newest:6}.
    word_freqs = {
        _word(b"low"): 5,
        _word(b"lower"): 2,
        _word(b"widest"): 3,
        _word(b"newest"): 6,
    }
    merges = _compute_merges(word_freqs, num_merges=6)
    assert merges == [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]


def test_tie_break_prefers_lexicographically_greater_pair() -> None:
    # All four pairs occur once; PDF says the merge is ('BA', 'A').
    word_freqs = {(b"A", b"B"): 1, (b"A", b"C"): 1, (b"B", b"ZZ"): 1, (b"BA", b"A"): 1}
    merges = _compute_merges(word_freqs, num_merges=1)
    assert merges == [(b"BA", b"A")]


def test_train_bpe_vocab_ordering_and_size(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("low low low low low\nlower lower\n", encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=256 + 1 + 5, special_tokens=["<|endoftext|>"])

    assert vocab[0] == b"<|endoftext|>"  # specials first
    assert vocab[1] == bytes([0]) and vocab[256] == bytes([255])  # then 256 bytes
    assert len(vocab) == 256 + 1 + len(merges)
    assert len(merges) == 5


def test_train_bpe_rejects_too_small_vocab(tmp_path: Path) -> None:
    corpus = tmp_path / "c.txt"
    corpus.write_text("hello world", encoding="utf-8")
    try:
        train_bpe(corpus, vocab_size=100, special_tokens=["<|endoftext|>"])
    except ValueError:
        return
    raise AssertionError("expected ValueError for vocab_size < 256 + n_special")


def _round_trip_tokenizer(tmp_path: Path) -> Tokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "the cat sat on the mat. the dog ran! greetings, 世界 🌍\n" * 20,
        encoding="utf-8",
    )
    vocab, merges = train_bpe(corpus, vocab_size=400, special_tokens=["<|endoftext|>"])
    return Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


def test_encode_decode_round_trip_ascii_and_unicode(tmp_path: Path) -> None:
    tok = _round_trip_tokenizer(tmp_path)
    for s in ["the cat sat", "hello, world!", "世界 🌍 emoji", "  leading spaces", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_special_token_is_single_id_and_not_split(tmp_path: Path) -> None:
    tok = _round_trip_tokenizer(tmp_path)
    ids = tok.encode("hello<|endoftext|>world")
    eot_id = tok._special_to_id["<|endoftext|>"]
    assert ids.count(eot_id) == 1
    assert tok.decode(ids) == "hello<|endoftext|>world"


def test_encode_iterable_matches_encode(tmp_path: Path) -> None:
    tok = _round_trip_tokenizer(tmp_path)
    lines = ["the cat\n", "sat on the mat\n"]
    assert list(tok.encode_iterable(lines)) == tok.encode("".join(lines))


def test_pretokenize_does_not_cross_special_boundary() -> None:
    counts = _pretokenize_counts("ab<|endoftext|>cd", ["<|endoftext|>"])
    # The special token is removed; "ab" and "cd" are counted, nothing spans the boundary.
    assert _word(b"ab") in counts
    assert _word(b"cd") in counts
    assert all(b"<" not in b"".join(w) for w in counts)
