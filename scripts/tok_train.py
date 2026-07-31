"""Per-corpus BPE tokenizer training CLI (F12 corpus ablation).

F12 retrains a byte-level BPE (vocab 32,768) on EACH arm's corpus — comparing bpb with a
foreign tokenizer would conflate the corpus with tokenizer fit. Both arms train on the same
BYTE budget (``--byte-budget``) so the tokenizers differ only in the corpus they fit.

The tokenizer is saved via ``data/shards.save_tokenizer_files`` (one ``tokenizer.json`` in
``--out-dir``), so the output dir is directly consumable by the shard builders — the
tokenizer that indexes a corpus is staged next to its shards (shards.py convention).

Usage:
    python scripts/tok_train.py CORPUS.txt --out-dir data/tok_fineweb
    python scripts/tok_train.py corpus_dir/ --vocab-size 32768 --byte-budget 16777216 \
        --out-dir data/tok_climbmix
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow `python scripts/tok_train.py` without an editable install.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scratch_llm.data.shards import DOC_SEPARATOR, save_tokenizer_files
from scratch_llm.tokenizer import Tokenizer, train_bpe


def read_corpus_capped(input_path: str | Path, byte_budget: int | None) -> tuple[str, int]:
    """Read the corpus (one text file, or every ``*.txt`` in a directory, sorted) as text,
    truncated to ``byte_budget`` UTF-8 bytes.

    Truncation is utf-8-boundary safe (a partial trailing multibyte char is dropped, never
    split) and DETERMINISTIC: same input + same budget ⇒ identical bytes, so both ablation
    arms train their tokenizer on exactly the same byte budget. ``None`` reads the corpus
    uncapped. Returns ``(text, n_bytes_staged)``.
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        paths = sorted(input_path.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(f"no *.txt files in {input_path}")
    elif input_path.is_file():
        paths = [input_path]
    else:
        raise FileNotFoundError(f"{input_path} is neither a file nor a directory")

    chunks: list[bytes] = []
    staged = 0
    for path in paths:
        data = path.read_bytes()
        if byte_budget is not None:
            remaining = byte_budget - staged
            if remaining <= 0:
                break
            data = data[:remaining]
        # Drop a trailing partial UTF-8 sequence (decode/encode round-trip at the boundary).
        text = data.decode("utf-8", errors="ignore")
        data = text.encode("utf-8")
        chunks.append(data)
        staged += len(data)
    corpus = b"\n".join(chunks).decode("utf-8")
    return corpus, staged


def train_tokenizer(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    vocab_size: int = 32768,
    byte_budget: int | None = None,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """Train a byte-level BPE on the (byte-capped) corpus and stage ``tokenizer.json``.

    ``special_tokens`` defaults to the shard ``<|eot|>`` document separator so the tokenizer
    is directly compatible with the shard builders (train_bpe splits specials out before
    pre-tokenization — they mark boundaries and are never merged across). Staging follows
    ``speedrun._train_tokenizer``: train_bpe reads a file, so the corpus lands on a temp file.
    """
    special_tokens = [DOC_SEPARATOR] if special_tokens is None else special_tokens
    corpus, staged = read_corpus_capped(input_path, byte_budget)
    if not corpus.strip():
        raise ValueError(f"{input_path} yielded no training text (byte_budget={byte_budget})")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(corpus)
        tmp = Path(fh.name)
    try:
        vocab, merges = train_bpe(tmp, vocab_size, special_tokens=special_tokens)
    finally:
        tmp.unlink(missing_ok=True)

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_tokenizer_files(tokenizer, out_dir)
    budget = "uncapped" if byte_budget is None else f"budget {byte_budget:,}"
    print(
        f"trained BPE on {staged / 2**20:.1f} MiB ({budget}): {len(vocab):,} vocab, "
        f"{len(merges):,} merges → {out_path}"
    )
    return tokenizer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Train a per-corpus byte-level BPE tokenizer (F12) → tokenizer.json."
    )
    p.add_argument("input", help="Corpus text file, or a directory of .txt files (sorted).")
    p.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        help="Default 32768 = the repo-wide 2^15 convention (SpeedrunConfig.vocab_size).",
    )
    p.add_argument(
        "--byte-budget",
        type=int,
        default=None,
        help="Cap the training sample at this many UTF-8 bytes (utf-8-boundary-safe, "
        "deterministic). Both F12 arms MUST pass the same budget. Default: uncapped.",
    )
    p.add_argument("--out-dir", required=True, help="Output dir for tokenizer.json.")
    p.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="Special token (repeatable); default: the shard <|eot|> document separator.",
    )
    args = p.parse_args(argv)
    train_tokenizer(
        args.input,
        args.out_dir,
        vocab_size=args.vocab_size,
        byte_budget=args.byte_budget,
        special_tokens=args.special_token,
    )


if __name__ == "__main__":
    main()
