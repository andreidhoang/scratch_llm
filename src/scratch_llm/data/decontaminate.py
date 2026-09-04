"""A0 decontamination gate — strip train documents that n-gram-collide with the eval sets.

Intent (git show 07f3de4:docs/archive/FRONTIER_2026_TASKSPEC.md §A · A0): every scored run's ablation numbers are only
meaningful if the pretraining shards do not contain the eval text. This module builds a guard
set of word n-grams from the eval corpora and drops any train document whose n-grams collide
with it — BEFORE the document reaches BPE training or shard tokenization
(:func:`scratch_llm.data.shards.build_dataset` threads a ``doc_filter`` for exactly this).

Rule: **n=13 exact-collision n-grams**, the GPT-3 appendix-C / nanochat practice — 13
consecutive words is long enough that an accidental match is vanishingly rare in natural text,
and short enough that a verbatim leaked eval item almost surely contains one. At the default
``threshold=0.0`` a SINGLE colliding 13-gram drops the document; a paraphrase (a few words
changed, breaking every 13-window) passes. Eval items shorter than ``n`` words (e.g. a
Countdown target) contribute their FULL word-tuple as one guard entry, matched against the
full tuple of (normalized-)short train docs — a documented, deliberately narrow rule: short
items only guard identical short docs, never substrings of long ones.

Normalization (documented choice, applied IDENTICALLY on both sides — the gate silently fails
if the two sides normalize differently): lowercase, replace every character that is not a
Unicode alphanumeric with a space, split on whitespace (collapsing runs). Simple, symmetric,
and robust to case/punctuation/whitespace variation between the eval file and a crawled copy.

Invariant (tested in tests/test_decontaminate.py): a train doc with a verbatim 13+-word eval
sentence planted inside — uppercased or re-whitespaced or stripped of punctuation — is caught;
clean and paraphrased docs pass.

Kill check (surfaced, never auto-failed): if more than :data:`SUSPICIOUS_OVERLAP_RATE` of a
slice is flagged, the eval set has probably leaked wholesale into the pretraining corpus —
``DecontamResult.suspicious`` is set and a warning logged so a human investigates instead of
silently discarding 20%+ of the data.

Guard-corpus coverage — what is in-repo vs supplied at shard-build time:
- **In-repo** (:func:`default_eval_texts`): the hermetic GSM fixture questions + answers
  (``envs/gsm_math.py::DEFAULT_FIXTURE``) and the default-seeded Countdown pool's questions +
  targets (``envs/countdown.py``, seed=0 · 16 tasks). The Countdown question template's tail is
  seed-independent and ≥13 words, so these entries guard rendered Countdown questions for ANY
  seed (for docs ≥ n words).
- **NOT in-repo** — must be passed via ``extra_paths`` (one text per line) when building shards
  for a scored run: the real GSM8K test split, MMLU, and any external report-card val texts.
  (The report card's val_bpb split is carved from the training corpus at run time — that is a
  train/val-split concern, not this gate's.)

Interview question this module answers: "How do you decontaminate pretraining data against
your eval sets, and why n=13 exact-match n-grams?"
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The GPT-3/nanochat collision length: long enough that accidental natural-text matches are
# vanishingly rare, short enough that a verbatim leaked eval item almost surely contains one.
DEFAULT_NGRAM_N = 13

# Kill check (pre-registered): flagging more than this fraction of a slice means the eval set
# likely leaked into the pretraining corpus — surface for investigation, don't silently drop.
SUSPICIOUS_OVERLAP_RATE = 0.20

_MAX_DROPPED_SAMPLES = 5
_SAMPLE_PREFIX_CHARS = 80

# Everything that is not a Unicode alphanumeric becomes a space (``\w`` minus underscore).
_NON_WORD = re.compile(r"[^\w]|_")


def _normalize_words(text: str) -> list[str]:
    """The ONE normalization, shared by both sides: lowercase → non-alphanumerics → space →
    whitespace-split (collapses runs). Symmetry is load-bearing — see the module docstring."""
    return _NON_WORD.sub(" ", text.lower()).split()


def build_eval_ngrams(
    eval_texts: Iterable[str], n: int = DEFAULT_NGRAM_N
) -> frozenset[tuple[str, ...]]:
    """The guard set: every word ``n``-gram of every eval text, post-normalization.

    Texts shorter than ``n`` words contribute their FULL word-tuple as one entry (so short
    eval items like Countdown targets still guard — against identical short docs only).
    Empty/punctuation-only texts contribute nothing.
    """
    grams: set[tuple[str, ...]] = set()
    for text in eval_texts:
        words = _normalize_words(text)
        if not words:
            continue
        if len(words) < n:
            grams.add(tuple(words))
        else:
            grams.update(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return frozenset(grams)


def ngram_overlap(
    doc: str, eval_ngrams: AbstractSet[tuple[str, ...]], n: int = DEFAULT_NGRAM_N
) -> float:
    """Fraction of ``doc``'s word ``n``-grams present in ``eval_ngrams`` (0.0..1.0).

    Docs shorter than ``n`` words return 0.0 unless their full word-tuple hits a short-eval
    entry (then 1.0). Empty docs are 0.0.
    """
    words = _normalize_words(doc)
    if not words:
        return 0.0
    if len(words) < n:
        return 1.0 if tuple(words) in eval_ngrams else 0.0
    windows = len(words) - n + 1
    hits = sum(1 for i in range(windows) if tuple(words[i : i + n]) in eval_ngrams)
    return hits / windows


@dataclass(frozen=True)
class DecontamResult:
    """Outcome of one decontamination pass — kept docs plus the accounting the log needs."""

    kept: list[str]
    n_kept: int
    n_dropped: int
    overlap_rate: float  # n_dropped / (n_kept + n_dropped); 0.0 on empty input
    dropped_samples: tuple[str, ...]  # first few dropped-doc prefixes, for the log
    suspicious: bool  # overlap_rate > SUSPICIOUS_OVERLAP_RATE — investigate, don't trust


def decontaminate_docs(
    docs: Iterable[str],
    eval_ngrams: AbstractSet[tuple[str, ...]],
    n: int = DEFAULT_NGRAM_N,
    threshold: float = 0.0,
) -> DecontamResult:
    """Keep docs with :func:`ngram_overlap` ``<= threshold``; drop (and account for) the rest.

    The default ``threshold=0.0`` is the standard exact-collision rule: ANY colliding
    ``n``-gram drops the document. The kill check never auto-fails — it marks the result
    ``suspicious`` and logs, because a >20% drop rate means the eval set leaked wholesale and
    a human must look before any scored run proceeds.
    """
    kept: list[str] = []
    dropped_samples: list[str] = []
    n_dropped = 0
    for doc in docs:
        if ngram_overlap(doc, eval_ngrams, n) <= threshold:
            kept.append(doc)
        else:
            n_dropped += 1
            if len(dropped_samples) < _MAX_DROPPED_SAMPLES:
                dropped_samples.append(doc[:_SAMPLE_PREFIX_CHARS])
    total = len(kept) + n_dropped
    overlap_rate = n_dropped / total if total else 0.0
    suspicious = overlap_rate > SUSPICIOUS_OVERLAP_RATE
    if suspicious:
        logger.warning(
            "decontamination flagged %d/%d docs (%.1f%%) — suspicious: above the %.0f%% kill "
            "check, the eval set may have leaked into the pretraining corpus; investigate "
            "before any scored run. Sample dropped prefixes: %s",
            n_dropped,
            total,
            100 * overlap_rate,
            100 * SUSPICIOUS_OVERLAP_RATE,
            dropped_samples,
        )
    return DecontamResult(
        kept=kept,
        n_kept=len(kept),
        n_dropped=n_dropped,
        overlap_rate=overlap_rate,
        dropped_samples=tuple(dropped_samples),
        suspicious=suspicious,
    )


def decontam_doc_filter(
    eval_ngrams: AbstractSet[tuple[str, ...]],
    n: int = DEFAULT_NGRAM_N,
    threshold: float = 0.0,
) -> Callable[[str], bool]:
    """A keep-predicate (True = keep) for the ``doc_filter`` seam in
    :func:`scratch_llm.data.shards.tokenize_to_shard` / ``build_dataset``.

    For dropped-doc accounting beyond the per-shard count, run :func:`decontaminate_docs`
    first and pass ``result.kept`` instead.
    """

    def keep(doc: str) -> bool:
        return ngram_overlap(doc, eval_ngrams, n) <= threshold

    return keep


def default_eval_texts(extra_paths: Sequence[str | Path] = ()) -> list[str]:
    """Assemble the guard corpus from what is vendored in-repo, plus optional external files.

    In-repo (always included): the GSM fixture questions + answers
    (``envs/gsm_math.py::DEFAULT_FIXTURE``) and the default-seeded Countdown pool's questions +
    target strings (seed=0, 16 tasks — matching ``CountdownEnv`` defaults; the question
    template's ≥13-word tail is seed-independent, so these guard ANY seed's rendered
    questions). NOT in-repo — supply via ``extra_paths`` (one text per line, blank lines
    skipped): real GSM8K/MMLU files and any external report-card val texts.

    The envs imports are lazy: they transitively import torch (via ``rollout.types`` →
    ``sampling``), and :mod:`scratch_llm.data` stays numpy-only at import time.
    """
    from scratch_llm.envs.countdown import generate_countdown_tasks
    from scratch_llm.envs.gsm_math import DEFAULT_FIXTURE

    texts: list[str] = []
    for question, answer in DEFAULT_FIXTURE:
        texts.append(question)
        texts.append(answer)
    for task in generate_countdown_tasks(16, seed=0):
        texts.append(task.metadata["question"])
        texts.append(task.metadata["target"])
    for path in extra_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(line)
    return texts
