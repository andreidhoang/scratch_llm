"""Byte-level BPE tokenizer — train it, then encode/decode with it.

L1 substrate (A1). The tokenizer defines the vocabulary axis of every logit and every
KL we ever compute, so owning it is the precondition for measuring
`true_quality_gap = reward - true_quality (+ hack_rate, kl_train_infer)`.

Two correctness invariants this module must satisfy (tested in tests/test_tokenizer.py):
- **Round-trip:** ``decode(encode(s)) == s`` for any valid UTF-8 string.
- **Reference merges:** the stylized Sennrich/CS336 ``bpe_example`` reproduces the merge
  sequence ``[st, est, ow, low, west, ne]`` exactly (catches tie-break / counting bugs).

Scope note (ADR-0001): the v0.1.0 *served* policy uses an HF model's native tokenizer;
this from-scratch BPE is the owned-substrate artifact and powers the tiny-LM smoke run.
Kill criterion (A1 guide): if BPE training on TinyStories blows the time budget, reach
for ``multiprocessing`` pretokenization before anything fancier.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

# GPT-2 pre-tokenization pattern (Radford et al. 2019). Splits text into pre-tokens so
# that merges never cross word/punctuation boundaries; a leading space attaches to the
# following word (" the" is one pre-token). Requires the `regex` module: stdlib `re`
# has no \p{L}/\p{N} Unicode-property classes.
_GPT2_PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# A symbol is a `bytes` object (one or more bytes); a "word" is a tuple of symbols.
Symbol = bytes
Word = tuple[Symbol, ...]
Pair = tuple[Symbol, Symbol]


def _pretokenize_counts(text: str, special_tokens: list[str]) -> Counter[Word]:
    """Pre-token frequency table, split on special tokens first so no merge crosses a
    document boundary. Special tokens are NOT counted here — they get their own vocab ids.
    """
    if special_tokens:
        # Longest-first so an overlapping special (e.g. "<|eot|>" vs "<|eot|><|eot|>")
        # matches greedily. No capture group -> the delimiters are dropped from `chunks`.
        delimiter = "|".join(regex.escape(s) for s in sorted(special_tokens, key=len, reverse=True))
        chunks = regex.split(delimiter, text)
    else:
        chunks = [text]

    counts: Counter[Word] = Counter()
    for chunk in chunks:
        for match in _GPT2_PAT.finditer(chunk):
            token_bytes = match.group().encode("utf-8")
            counts[tuple(bytes([b]) for b in token_bytes)] += 1
    return counts


def _merge_word(word: list[Symbol], pair: Pair, merged: Symbol) -> list[Symbol]:
    """Replace every non-overlapping left-to-right occurrence of `pair` with `merged`."""
    out: list[Symbol] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _compute_merges(word_freqs: dict[Word, int], num_merges: int) -> list[Pair]:
    """The core BPE training loop.

    Greedily merge the most frequent adjacent symbol pair, breaking ties by the
    **lexicographically greater pair** (CS336 convention). Pair counts are maintained
    incrementally: only words that contained the merged pair are re-scanned each round,
    so cost scales with affected words, not the whole corpus.
    """
    words: list[list[Symbol]] = [list(w) for w in word_freqs]
    freqs: list[int] = list(word_freqs.values())

    pair_counts: Counter[Pair] = Counter()
    pair_to_words: dict[Pair, set[int]] = defaultdict(set)
    for i, word in enumerate(words):
        f = freqs[i]
        for a, b in zip(word, word[1:], strict=False):
            pair_counts[(a, b)] += f
            pair_to_words[(a, b)].add(i)

    merges: list[Pair] = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        # max by (count, then the pair itself) -> highest count, ties to the
        # lexicographically greater pair (bytes compare lexicographically).
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged = best[0] + best[1]
        merges.append(best)

        for i in list(pair_to_words[best]):
            f = freqs[i]
            old = words[i]
            new = _merge_word(old, best, merged)
            words[i] = new

            old_pairs = Counter(zip(old, old[1:], strict=False))
            new_pairs = Counter(zip(new, new[1:], strict=False))
            for p in old_pairs.keys() | new_pairs.keys():
                delta = (new_pairs[p] - old_pairs[p]) * f
                if delta:
                    pair_counts[p] += delta
                if new_pairs[p] > 0:
                    pair_to_words[p].add(i)
                else:
                    pair_to_words[p].discard(i)
                if pair_counts.get(p, 0) <= 0:
                    pair_counts.pop(p, None)

        pair_counts.pop(best, None)
        pair_to_words.pop(best, None)

    return merges


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a byte-level BPE tokenizer on the text at `input_path`.

    Returns ``(vocab, merges)`` where vocab ids are assigned in the order
    (special tokens, then the 256 byte values, then one id per merge) — the ordering the
    CS336 spec uses. ``merges`` is the ordered list of merged byte pairs.
    """
    special_tokens = special_tokens or []
    base_size = len(special_tokens) + 256
    if vocab_size < base_size:
        raise ValueError(
            f"vocab_size={vocab_size} too small: need >= {base_size} "
            f"({len(special_tokens)} special tokens + 256 bytes)."
        )

    vocab: dict[int, bytes] = {}
    idx = 0
    for token in special_tokens:
        vocab[idx] = token.encode("utf-8")
        idx += 1
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1

    text = Path(input_path).read_text(encoding="utf-8")
    word_freqs = _pretokenize_counts(text, special_tokens)
    merges = _compute_merges(dict(word_freqs), vocab_size - base_size)

    for a, b in merges:
        vocab[idx] = a + b
        idx += 1

    return vocab, merges


class Tokenizer:
    """Encode text to token ids and back, given a trained vocab + merges.

    Encoding mirrors training: pre-tokenize with the GPT-2 regex (splitting out special
    tokens, which map straight to their id), then apply the learned merges in training
    order to each pre-token.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self._bytes_to_id: dict[bytes, int] = {b: i for i, b in vocab.items()}
        self._ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(merges)}
        self._special_to_id: dict[str, int] = {}
        for token in self.special_tokens:
            encoded = token.encode("utf-8")
            if encoded not in self._bytes_to_id:
                # Allow specials that weren't in the trained vocab (e.g. added later).
                new_id = max(vocab) + 1 if vocab else 0
                self.vocab[new_id] = encoded
                self._bytes_to_id[encoded] = new_id
            self._special_to_id[token] = self._bytes_to_id[encoded]
        self._cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load a tokenizer from a JSON vocab ({id: latin-1 string}) and a merges file
        (one ``A B`` pair per line, latin-1 encoded so every byte round-trips)."""
        raw_vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        vocab = {int(i): s.encode("latin-1") for i, s in raw_vocab.items()}
        merges: list[Pair] = []
        for line in Path(merges_path).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            a, b = line.split(" ", 1)
            merges.append((a.encode("latin-1"), b.encode("latin-1")))
        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: bytes) -> list[int]:
        """Apply learned merges to one pre-token's bytes, return its token ids."""
        cached = self._cache.get(token_bytes)
        if cached is not None:
            return cached

        parts: list[bytes] = [bytes([b]) for b in token_bytes]
        while len(parts) >= 2:
            best_rank: int | None = None
            best_i = -1
            for i in range(len(parts) - 1):
                rank = self._ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]

        ids = [self._bytes_to_id[p] for p in parts]
        self._cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for segment, is_special in self._split_on_specials(text):
            if is_special:
                ids.append(self._special_to_id[segment])
            else:
                for match in _GPT2_PAT.finditer(segment):
                    ids.extend(self._bpe(match.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode a stream of text (e.g. file lines) without buffering it all."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def _split_on_specials(self, text: str) -> list[tuple[str, bool]]:
        """Split text into (segment, is_special) pieces, keeping the specials."""
        if not self.special_tokens:
            return [(text, False)]
        delimiter = (
            "("
            + "|".join(regex.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True))
            + ")"
        )
        out: list[tuple[str, bool]] = []
        for piece in regex.split(delimiter, text):
            if not piece:
                continue
            out.append((piece, piece in self._special_to_id))
        return out
