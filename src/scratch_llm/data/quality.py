"""A4 quality classifier — where *signal design* is the deliverable (ADR-0016).

Intent: "quality" has no ground truth. This classifier defines it **operationally by the label
source**: positives are text from trusted-source-linked pages (CS336 A4 uses Wikipedia
external-reference URLs — the GPT-2/WebText "linked-by-Reddit-karma" trick with a different
trust anchor), negatives are random CommonCrawl. The fastText model merely *amortizes that
judgment* over the whole corpus; choosing the positive set IS the modeling decision. Change the
label source and you have defined a different corpus — no amount of model capacity undoes that.

Invariants: ``classify`` returns the same ``(label, score)`` shape as every other A4 filter
(:func:`scratch_llm.data.filters.classifier_keep` applies unchanged), with canonical labels
``"wiki"`` (high quality) vs ``"cc"`` (random web); the keep-fraction of any corpus is monotone
non-increasing in the score threshold — the threshold is the precision/recall dial (higher ⇒
smaller, cleaner corpus).

Interview question this module answers: "how do you define 'high quality' for pretraining data
when no labels exist?" — you don't; you *pick a trusted positive source*, train a cheap
classifier against random negatives, and sweep the threshold. DataComp-LM (2024) is the
scaled-up receipt: the filtering recipe beats more tokens.
"""

from __future__ import annotations

import functools
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scratch_llm.data.filters import fasttext_classify, models_dir

#: Canonical labels — match the official A4 fixture semantics (`wiki` positive, `cc` negative).
POSITIVE_LABEL = "wiki"
NEGATIVE_LABEL = "cc"

#: Default persisted model consumed by :func:`classify_quality` when no model is passed.
DEFAULT_QUALITY_MODEL_FILENAME = "quality_wiki_cc.bin"


def _one_line(text: str) -> str:
    """fastText's file format is one document per line — collapse all whitespace."""
    return " ".join(text.split())


class QualityClassifier:
    """A supervised fastText model behind the shared ``(label, score)`` filter interface."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def train(
        cls,
        pos_texts: Sequence[str],
        neg_texts: Sequence[str],
        *,
        pos_label: str = POSITIVE_LABEL,
        neg_label: str = NEGATIVE_LABEL,
        dim: int = 64,
        epoch: int = 10,
        lr: float = 0.5,
        word_ngrams: int = 2,
        min_count: int = 1,
        seed: int = 0,
    ) -> QualityClassifier:
        """Train trusted-positive vs random-negative — the signal-design step (ADR-0016).

        ``pos_texts`` carry the entire definition of "quality"; ``neg_texts`` should be drawn
        from the same distribution the filter will run on (random CC), otherwise the classifier
        learns the domain gap instead of the quality gap. Single-threaded so a fixed input
        yields reproducible training.
        """
        import fasttext

        if not pos_texts or not neg_texts:
            raise ValueError("need at least one positive and one negative text")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for label, texts in ((pos_label, pos_texts), (neg_label, neg_texts)):
                for text in texts:
                    line = _one_line(text)
                    if line:
                        f.write(f"__label__{label} {line}\n")
            train_path = Path(f.name)
        try:
            model = fasttext.train_supervised(
                input=str(train_path),
                dim=dim,
                epoch=epoch,
                lr=lr,
                wordNgrams=word_ngrams,
                minCount=min_count,
                thread=1,  # single-threaded + fixed seed => reproducible training
                seed=seed,
                verbose=0,
            )
        finally:
            train_path.unlink(missing_ok=True)
        return cls(model)

    @classmethod
    def load(cls, path: str | Path) -> QualityClassifier:
        """Load a previously saved classifier."""
        import fasttext

        # setattr: the eprint hook is added dynamically by fasttext; pyright can't see it.
        setattr(fasttext.FastText, "eprint", lambda *args, **kwargs: None)  # noqa: B010
        return cls(fasttext.load_model(str(path)))

    def save(self, path: str | Path) -> Path:
        """Persist the model (fastText binary format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        return path

    def classify(self, text: str) -> tuple[str, float]:
        """``(label, score)`` — argmax label with clamped confidence; thresholdable via
        :func:`scratch_llm.data.filters.classifier_keep`."""
        return fasttext_classify(self._model, text)


def train_quality_classifier(
    pos_texts: Sequence[str], neg_texts: Sequence[str], **kwargs: Any
) -> QualityClassifier:
    """Functional alias for :meth:`QualityClassifier.train` (the spec-named entry point)."""
    return QualityClassifier.train(pos_texts, neg_texts, **kwargs)


@functools.lru_cache(maxsize=4)
def _default_classifier(path_str: str) -> QualityClassifier:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"no quality model at {path}; train one with "
            "train_quality_classifier(pos, neg).save(path) — the label source you pick for "
            "`pos` defines what 'quality' means (ADR-0016)"
        )
    return QualityClassifier.load(path)


def classify_quality(text: str, model: QualityClassifier | None = None) -> tuple[str, float]:
    """``("wiki" | "cc", confidence)`` for one document.

    Uses ``model`` when given, else the persisted default at
    ``models_dir()/quality_wiki_cc.bin`` (raises ``FileNotFoundError`` with training
    instructions when absent — there is deliberately no bundled model: shipping one would hide
    the signal-design decision).
    """
    if model is None:
        model = _default_classifier(str(models_dir() / DEFAULT_QUALITY_MODEL_FILENAME))
    return model.classify(text)


def run_classify_quality(text: str) -> tuple[str, float]:
    """Official-adapter alias for :func:`classify_quality` (default persisted model)."""
    return classify_quality(text)
