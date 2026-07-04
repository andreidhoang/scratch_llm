"""A4 `filter_data` driver — the canonical pipeline order + per-filter discard accounting.

Intent: encode the one ordering that survives scrutiny (A4 guide §2(6)):

    extract → language-ID → Gopher → NSFW → toxic → PII-mask (transform) → quality → dedup LAST

Cheap/destructive filters run first (language ID is the cheapest, highest-volume cut; Gopher is
transparent structure), classifiers next, the most expensive classifier (quality) last among the
per-doc stages, and dedup runs on the *survivors only*: deduping earlier wastes signatures on
documents later filters discard anyway, and (for exact line dedup) pre-filter line counts would
be corrupted by documents that never reach the corpus. PII masking sits between the harmful
classifiers and quality because it *transforms* rather than drops — harmful classifiers see the
raw page (masks would hide signal), quality and everything downstream see the masked text that
will actually be trained on.

Invariant (the audit trail): for every run,
``total_in == kept + sum(discarded per stage)`` — every input document is accounted for exactly
once, and the per-stage discard table is logged so you can say *which* filter shaped the corpus.
PII masking never changes the document count.

Interview question this module answers: "in what order do you run corpus filters and why does
dedup go last?" — cheap/destructive early, transform-don't-drop for PII, dedup on survivors;
and "show me the discard accounting" is how you audit any data recipe.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scratch_llm.data import dedup, filters

logger = logging.getLogger(__name__)

#: Canonical stage order — discard accounting is keyed by these names.
STAGE_ORDER = ("extract", "language", "gopher", "nsfw", "toxic", "quality", "dedup")

#: ``text -> (label, score)`` — the one classifier shape (filters.py / quality.py).
TextClassifier = Callable[[str], tuple[str, float]]


@dataclass
class FilterConfig:
    """Thresholds + classifier hooks for one pipeline run.

    Classifier stages are *opt-in*: a ``None`` classifier disables its stage, so the default
    config is fully hermetic (regex + Gopher + dedup only — no model files needed). Wire the
    real fastText models with :meth:`with_default_models`. Every threshold uses
    :func:`scratch_llm.data.filters.classifier_keep` semantics: keep iff the argmax label is
    the keep-label and its confidence clears the threshold.
    """

    language: str = "en"
    language_threshold: float = 0.65
    nsfw_keep_label: str = "non-nsfw"
    nsfw_threshold: float = 0.5
    toxic_keep_label: str = "non-toxic"
    toxic_threshold: float = 0.5
    quality_keep_label: str = "wiki"
    quality_threshold: float = 0.5
    mask_pii: bool = True
    gopher: bool = True
    dedup: bool = True
    minhash: Mapping[str, Any] = field(default_factory=lambda: dict(dedup.DEFAULT_MINHASH_PARAMS))
    language_classifier: TextClassifier | None = None
    nsfw_classifier: TextClassifier | None = None
    toxic_classifier: TextClassifier | None = None
    quality_classifier: TextClassifier | None = None

    @classmethod
    def with_default_models(cls, **overrides: Any) -> FilterConfig:
        """Config wired to the real fastText classifiers (auto-downloads models on first use).

        The quality stage needs the persisted default model (``classify_quality``) — pass
        ``quality_classifier=None`` to skip it, or a trained
        :class:`~scratch_llm.data.quality.QualityClassifier`'s ``.classify``.
        """
        from scratch_llm.data import quality

        settings: dict[str, Any] = {
            "language_classifier": filters.identify_language,
            "nsfw_classifier": filters.classify_nsfw,
            "toxic_classifier": filters.classify_toxic_speech,
            "quality_classifier": quality.classify_quality,
        }
        settings.update(overrides)
        return cls(**settings)


@dataclass(frozen=True)
class DocResult:
    """One surviving document: its id and final (extracted, PII-masked) text."""

    doc_id: str
    text: str


@dataclass
class FilterReport:
    """The pipeline's audit trail: survivors + per-stage discard accounting.

    Invariant: ``total_in == kept + sum(discarded.values())``.
    """

    total_in: int
    discarded: dict[str, int]
    pii_masked: dict[str, int]
    survivors: list[DocResult]

    @property
    def kept(self) -> int:
        return len(self.survivors)

    def discard_table(self) -> str:
        """The per-filter discard table (the `filter_data` written deliverable's shape)."""
        rows = [f"{'stage':<10} {'in':>8} {'discarded':>10} {'kept':>8} {'discard%':>9}"]
        entering = self.total_in
        for stage in STAGE_ORDER:
            dropped = self.discarded.get(stage, 0)
            kept = entering - dropped
            pct = 100.0 * dropped / entering if entering else 0.0
            rows.append(f"{stage:<10} {entering:>8} {dropped:>10} {kept:>8} {pct:>8.1f}%")
            entering = kept
        total_dropped = self.total_in - self.kept
        total_pct = 100.0 * total_dropped / self.total_in if self.total_in else 0.0
        rows.append(
            f"{'TOTAL':<10} {self.total_in:>8} {total_dropped:>10} {self.kept:>8} "
            f"{total_pct:>8.1f}%"
        )
        rows.append(
            "PII masked (transform, not drop): "
            + " ".join(f"{k}={v}" for k, v in self.pii_masked.items())
        )
        return "\n".join(rows)


def _extract(raw: str | bytes) -> str | None:
    if isinstance(raw, bytes | bytearray):
        return filters.extract_text_from_html_bytes(bytes(raw))
    return raw


def _dedup_survivors(
    alive: list[DocResult], minhash_params: Mapping[str, Any], work_dir: Path | None
) -> tuple[list[DocResult], int]:
    """Run MinHash dedup (dedup.py, W7a) over the surviving docs via a scratch directory."""
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()
        paths = []
        for i, doc in enumerate(alive):
            path = in_dir / f"{i:06d}.txt"
            path.write_text(doc.text, encoding="utf-8")
            paths.append(path)
        surviving = dedup.minhash_dedup(
            paths,
            num_hashes=int(minhash_params["num_hashes"]),
            num_bands=int(minhash_params["num_bands"]),
            ngrams=int(minhash_params["ngrams"]),
            jaccard_threshold=float(minhash_params["jaccard_threshold"]),
            out_dir=out_dir,
        )
        keep_indices = {int(p.stem) for p in surviving}
    kept = [doc for i, doc in enumerate(alive) if i in keep_indices]
    return kept, len(alive) - len(kept)


def filter_data(
    docs: Sequence[str | bytes],
    config: FilterConfig | None = None,
    *,
    doc_ids: Sequence[str] | None = None,
    work_dir: Path | None = None,
) -> FilterReport:
    """Run the canonical filter pipeline over ``docs``; return survivors + discard accounting.

    ``docs`` items are raw HTML ``bytes`` (extracted via resiliparse) or already-extracted
    ``str`` text. Stages run in :data:`STAGE_ORDER`; each document is charged to the *first*
    stage that discards it. The discard table is logged at INFO — the audit trail is a
    deliverable, not a debug aid.
    """
    config = config or FilterConfig()
    ids = list(doc_ids) if doc_ids is not None else [f"doc{i:05d}" for i in range(len(docs))]
    if len(ids) != len(docs):
        raise ValueError(f"doc_ids length {len(ids)} != docs length {len(docs)}")
    if len(set(ids)) != len(ids):
        raise ValueError("doc_ids must be unique")

    discarded: dict[str, int] = dict.fromkeys(STAGE_ORDER, 0)
    pii_masked = {"emails": 0, "phones": 0, "ips": 0}
    alive: list[DocResult] = []

    for doc_id, raw in zip(ids, docs, strict=True):
        # 1. extract — the corpus entry point.
        text = _extract(raw)
        if text is None or not text.strip():
            discarded["extract"] += 1
            continue
        # 2. language — cheapest, highest-volume cut.
        if config.language_classifier is not None and not filters.classifier_keep(
            config.language_classifier(text), config.language, config.language_threshold
        ):
            discarded["language"] += 1
            continue
        # 3. Gopher — transparent structural heuristics.
        if config.gopher and not filters.gopher_quality_filter(text):
            discarded["gopher"] += 1
            continue
        # 4. harmful content — classifiers see the RAW page (masks would hide signal).
        if config.nsfw_classifier is not None and not filters.classifier_keep(
            config.nsfw_classifier(text), config.nsfw_keep_label, config.nsfw_threshold
        ):
            discarded["nsfw"] += 1
            continue
        if config.toxic_classifier is not None and not filters.classifier_keep(
            config.toxic_classifier(text), config.toxic_keep_label, config.toxic_threshold
        ):
            discarded["toxic"] += 1
            continue
        # 5. PII mask — TRANSFORM, never drops; downstream stages see the masked text.
        if config.mask_pii:
            text, n = filters.mask_emails(text)
            pii_masked["emails"] += n
            text, n = filters.mask_phone_numbers(text)
            pii_masked["phones"] += n
            text, n = filters.mask_ips(text)
            pii_masked["ips"] += n
        # 6. quality — the most expensive classifier runs on the fewest documents.
        if config.quality_classifier is not None and not filters.classifier_keep(
            config.quality_classifier(text), config.quality_keep_label, config.quality_threshold
        ):
            discarded["quality"] += 1
            continue
        alive.append(DocResult(doc_id, text))

    # 7. dedup LAST — signatures only for survivors (W7a machinery).
    if config.dedup and alive:
        alive, n_dropped = _dedup_survivors(alive, config.minhash, work_dir)
        discarded["dedup"] = n_dropped

    report = FilterReport(
        total_in=len(docs), discarded=discarded, pii_masked=pii_masked, survivors=alive
    )
    logger.info("filter_data discard accounting:\n%s", report.discard_table())
    return report
