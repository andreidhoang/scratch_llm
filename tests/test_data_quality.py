"""W7b tests — quality classifier (`scratch_llm.data.quality`).

Fully hermetic: fastText is a library dependency (importorskip), training runs on synthetic
trusted-positive vs random-negative corpora built from disjoint vocabularies — no downloads.
The load-bearing assertions: canonical `wiki`/`cc` labels, pos/neg separation, and the
threshold dial — keep-fraction monotone non-increasing as the threshold rises.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest

pytest.importorskip("fasttext")

from scratch_llm.data.filters import classifier_keep  # noqa: E402
from scratch_llm.data.quality import (  # noqa: E402
    DEFAULT_QUALITY_MODEL_FILENAME,
    QualityClassifier,
    _default_classifier,
    classify_quality,
    train_quality_classifier,
)

# Disjoint vocabularies: the "trusted source" register vs the "random web spam" register.
# Quality is DEFINED by this choice of label source — the classifier only amortizes it.
POS_VOCAB = [
    "encyclopedia",
    "reference",
    "citation",
    "historical",
    "research",
    "university",
    "published",
    "journal",
    "science",
    "theory",
    "chapter",
    "documented",
    "evidence",
    "analysis",
    "literature",
    "archive",
]
NEG_VOCAB = [
    "click",
    "free",
    "winner",
    "casino",
    "pills",
    "cheap",
    "jackpot",
    "subscribe",
    "offer",
    "limited",
    "deal",
    "buy",
    "now",
    "bonus",
    "prize",
    "unsubscribe",
]


def make_texts(vocab: list[str], n_docs: int, seed: int, words_per_doc: int = 20) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choices(vocab, k=words_per_doc)) for _ in range(n_docs)]


@pytest.fixture(scope="module")
def clf() -> QualityClassifier:
    pos = make_texts(POS_VOCAB, 150, seed=0)
    neg = make_texts(NEG_VOCAB, 150, seed=1)
    # NOTE: word_ngrams=1 reliably hits "RuntimeError: Encountered NaN" in fasttext 0.9.3 on
    # tiny-vocab corpora (bucket=0 path); the module default word_ngrams=2 is stable.
    return train_quality_classifier(pos, neg, dim=16, epoch=10)


def test_labels_and_separation(clf: QualityClassifier) -> None:
    """Held-out pos/neg docs get the canonical labels with confident scores."""
    for text in make_texts(POS_VOCAB, 10, seed=2):
        label, score = clf.classify(text)
        assert label == "wiki"
        assert 0.8 < score <= 1.0
    for text in make_texts(NEG_VOCAB, 10, seed=3):
        label, score = clf.classify(text)
        assert label == "cc"
        assert 0.8 < score <= 1.0


def test_threshold_keep_fraction_monotone_non_increasing(clf: QualityClassifier) -> None:
    """The threshold is the precision/recall dial: raising it can only shrink the kept corpus.

    Eval mix includes half-and-half vocabulary docs so scores spread across (0.5, 1.0) and the
    sweep exercises interior thresholds, not just the endpoints.
    """
    rng = random.Random(4)
    mixed_vocab = POS_VOCAB + NEG_VOCAB
    eval_docs = (
        make_texts(POS_VOCAB, 15, seed=5)
        + make_texts(NEG_VOCAB, 15, seed=6)
        + [" ".join(rng.choices(mixed_vocab, k=20)) for _ in range(30)]
    )
    classifications = [clf.classify(t) for t in eval_docs]

    thresholds = [i / 20 for i in range(1, 20)]  # 0.05 … 0.95
    keep_fracs = [
        sum(classifier_keep(c, "wiki", t) for c in classifications) / len(classifications)
        for t in thresholds
    ]
    assert all(a >= b for a, b in itertools.pairwise(keep_fracs))
    # The dial actually moves on this eval set (not a degenerate flat sweep).
    assert keep_fracs[0] > keep_fracs[-1]
    assert 0.0 < keep_fracs[0] <= 1.0


def test_save_load_roundtrip(clf: QualityClassifier, tmp_path: Path) -> None:
    path = clf.save(tmp_path / "quality_test.bin")
    reloaded = QualityClassifier.load(path)
    for text in make_texts(POS_VOCAB, 3, seed=7) + make_texts(NEG_VOCAB, 3, seed=8):
        label_a, score_a = clf.classify(text)
        label_b, score_b = reloaded.classify(text)
        assert label_a == label_b
        assert score_a == pytest.approx(score_b, abs=1e-6)


def test_classify_quality_with_explicit_model(clf: QualityClassifier) -> None:
    label, score = classify_quality("documented evidence in the published journal archive", clf)
    assert label == "wiki"
    assert score > 0.5


def test_classify_quality_default_model_missing_and_present(
    clf: QualityClassifier, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCRATCH_LLM_DATA_MODELS_DIR", str(tmp_path))
    _default_classifier.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="ADR-0016"):
            classify_quality("some text")
        clf.save(tmp_path / DEFAULT_QUALITY_MODEL_FILENAME)
        label, _ = classify_quality("cheap pills casino jackpot click now free bonus prize")
        assert label == "cc"
    finally:
        _default_classifier.cache_clear()  # do not leak the tmp model into other tests


def test_train_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="positive and one negative"):
        train_quality_classifier([], ["negative doc"])
