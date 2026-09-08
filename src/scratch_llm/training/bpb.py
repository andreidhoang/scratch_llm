"""Bits per byte — the quality axis of T1/T-R2, computed against the tokenizer's BYTE count.

    bpb = ( sum_t -ln p(x_t | x_<t) ) / ln 2 / n_bytes

The denominator is BYTES. Divide the same numerator by the number of TOKENS and you get bits per
token, which is a different and incomparable number: larger by the corpus's bytes-per-token ratio
(~4 for a byte-level BPE on English text). The two differ by a factor, not an epsilon, so the
mistake never looks like noise — it looks like a much worse model, uniformly, in both arms, and it
cancels in the delta, which is exactly why nobody notices until someone quotes the absolute number
next to a published one.

Two further rules this module enforces. Both are few-percent errors, and a gate that fires at the
1e-3 level cannot absorb a few percent:

* **The bytes counted are the bytes of the tokens that were actually scored.** A teacher-forced
  pass over n tokens scores n-1 transitions; the first token of a window is never predicted, so
  its bytes are not in the denominator. :func:`scratch_llm.eval.metrics.bits_per_byte` takes
  ``num_bytes`` from its caller and cannot check this; here the loss and the accounting come out
  of one object, so they cannot disagree.
* **Byte length is ``len(vocab[id])``** — the same bytes :meth:`Tokenizer.decode` would emit.
  Under that rule a special token carries its literal bytes (``<|endoftext|>`` is 13 of them),
  which inflates the denominator and so LOWERS bpb; pass its id in ``zero_byte_ids`` to score it
  at zero bytes instead. Either convention is defensible on its own; mixing them across two arms
  is not, so :class:`BpbAccounting` carries the one that was used and
  :func:`scratch_llm.training.run_matrix.compare` refuses a matrix whose arms disagree.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

#: nats -> bits.
LN2 = math.log(2.0)


@dataclass(frozen=True)
class BpbAccounting:
    """The denominator of one bpb number, and the convention that produced it.

    ``n_tokens`` is transitions SCORED (not tokens in the stream) and ``n_bytes`` is the byte
    length of exactly those transitions' target tokens.
    """

    n_tokens: int
    n_bytes: int
    zero_byte_ids: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.n_tokens <= 0:
            raise ValueError(f"n_tokens must be positive, got {self.n_tokens}")
        if self.n_bytes <= 0:
            raise ValueError(
                f"n_bytes must be positive, got {self.n_bytes} — a zero-byte denominator is the "
                "signature of byte lengths taken from the wrong vocab, or of every scored token "
                "being in zero_byte_ids"
            )

    @property
    def bytes_per_token(self) -> float:
        """The corpus's compression ratio under this tokenizer — the factor between bpb and
        bits/token, and the number to sanity-check first when a bpb looks 4x wrong."""
        return self.n_bytes / self.n_tokens

    def as_dict(self) -> dict[str, object]:
        return {
            "n_tokens": self.n_tokens,
            "n_bytes": self.n_bytes,
            "bytes_per_token": self.bytes_per_token,
            "zero_byte_ids": list(self.zero_byte_ids),
        }


def byte_lengths(vocab: Mapping[int, bytes], *, zero_byte_ids: Iterable[int] = ()) -> np.ndarray:
    """``lengths[id] = len(vocab[id])`` as an int64 array, with ``zero_byte_ids`` forced to 0.

    Indexable by token id, so the byte count of a stream is one gather + one sum. Ids absent from
    the vocab keep length 0, which would silently shrink the denominator — so the array is only as
    long as ``max(vocab)+1`` and :func:`account` bounds-checks against it.
    """
    if not vocab:
        raise ValueError("empty vocab: byte lengths would be all-zero and every bpb infinite")
    lengths = np.zeros(max(vocab) + 1, dtype=np.int64)
    for token_id, raw in vocab.items():
        lengths[token_id] = len(raw)
    for token_id in zero_byte_ids:
        lengths[token_id] = 0
    return lengths


def account(
    target_ids: np.ndarray, lengths: np.ndarray, *, zero_byte_ids: Iterable[int] = ()
) -> BpbAccounting:
    """Accounting for a scored stream: ``target_ids`` is the tokens the model PREDICTED.

    Pass the targets, never the inputs. They differ by one shift, so on a 4096-token window the
    two denominators differ by ~0.02% — invisible, and wrong in the same direction forever.
    """
    ids = np.asarray(target_ids).reshape(-1)
    if ids.size == 0:
        raise ValueError("no scored targets")
    hi = int(ids.max())
    if hi >= lengths.size:
        raise ValueError(
            f"token id {hi} is outside the vocab's byte-length table (size {lengths.size}) — "
            "the corpus and the tokenizer are not the same pair"
        )
    return BpbAccounting(
        n_tokens=int(ids.size),
        n_bytes=int(lengths[ids].sum()),
        zero_byte_ids=tuple(sorted(set(zero_byte_ids))),
    )


def bits_per_byte(total_nats: float, acc: BpbAccounting) -> float:
    """bpb from the SUMMED negative log-likelihood in nats over ``acc.n_tokens`` transitions."""
    if not math.isfinite(total_nats):
        raise ValueError(f"total_nats is {total_nats} — the run diverged; there is no bpb")
    return total_nats / LN2 / acc.n_bytes


def bits_per_byte_from_mean_loss(mean_nats_per_token: float, acc: BpbAccounting) -> float:
    """bpb from a MEAN cross-entropy in nats/token — what a training loop actually reports.

    ``bpb = mean_ce / ln2 * (n_tokens / n_bytes)``: the mean is converted to bits per token and
    then divided by the bytes-per-token ratio. If you stop after the first factor you have bits
    per token; :func:`bits_per_token` is that number, named, so the two can never be confused at
    a call site.
    """
    return bits_per_byte(mean_nats_per_token * acc.n_tokens, acc)


def bits_per_token(mean_nats_per_token: float) -> float:
    """Mean CE in nats/token -> bits/token. NOT bpb. Present so the wrong number has a name."""
    return mean_nats_per_token / LN2


__all__ = [
    "LN2",
    "BpbAccounting",
    "account",
    "bits_per_byte",
    "bits_per_byte_from_mean_loss",
    "bits_per_token",
    "byte_lengths",
]
