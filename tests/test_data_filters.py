"""W7b tests — filter family + pipeline (`scratch_llm.data.{filters,pipeline}`).

Hermetic (no network, no model files): PII masks, Gopher heuristics, `classifier_keep`,
pipeline order / PII placement / accounting / dedup-last via injected fake classifiers.
Conditional: resiliparse extraction (`importorskip`), fastText model tests skip unless the
model file is already present in `models_dir()` — tests never trigger a download.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scratch_llm.data import filters
from scratch_llm.data.filters import (
    classifier_keep,
    gopher_quality_filter,
    mask_emails,
    mask_ips,
    mask_phone_numbers,
    model_path,
)
from scratch_llm.data.pipeline import STAGE_ORDER, DocResult, FilterConfig, filter_data

OFFICIAL_FIXTURES = Path("/workspace/lectures/assignment4-data/tests/fixtures")
needs_official_fixtures = pytest.mark.skipif(
    not OFFICIAL_FIXTURES.is_dir(), reason="official A4 fixtures not present on this box"
)


def needs_model(name: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not model_path(name).exists(), reason=f"{name} not present in models_dir()"
    )


needs_lid = needs_model(filters.LID_MODEL)
needs_nsfw = needs_model(filters.NSFW_MODEL)
needs_toxic = needs_model(filters.TOXIC_MODEL)


# ---------------------------------------------------------------------------
# PII masks (hermetic) — transforms with (new_text, n_masked) returns
# ---------------------------------------------------------------------------


def test_mask_emails_multiple_and_existing_token_untouched() -> None:
    text = (
        "Prior datasets used |||EMAIL_ADDRESS||| as a mask token. "
        "Contact alice@example.com or bob.smith+tag@sub.example.co.uk for details."
    )
    masked, n = mask_emails(text)
    assert n == 2
    assert masked == (
        "Prior datasets used |||EMAIL_ADDRESS||| as a mask token. "
        "Contact |||EMAIL_ADDRESS||| or |||EMAIL_ADDRESS||| for details."
    )
    # Idempotent: masking the masked text changes nothing.
    again, n_again = mask_emails(masked)
    assert (again, n_again) == (masked, 0)


@pytest.mark.parametrize(
    "number",
    ["2831823829", "(283)-182-3829", "(283) 182 3829", "283-182-3829", "+1 283.182.3829"],
)
def test_mask_phone_number_formats(number: str) -> None:
    masked, n = mask_phone_numbers(f"call me at {number} tomorrow")
    assert n == 1
    assert masked == "call me at |||PHONE_NUMBER||| tomorrow"


def test_mask_phones_ignores_long_digit_runs() -> None:
    text = "order id 123456789012345 shipped"
    masked, n = mask_phone_numbers(text)
    assert (masked, n) == (text, 0)


def test_mask_ips_valid_quad_masked_invalid_octets_ignored() -> None:
    masked, n = mask_ips("server at 192.0.2.146. backup at 999.999.999.999 stays")
    assert n == 1
    assert masked == "server at |||IP_ADDRESS|||. backup at 999.999.999.999 stays"


# ---------------------------------------------------------------------------
# Gopher heuristics (hermetic) — the four structural checks
# ---------------------------------------------------------------------------

GOOD_SENTENCE = "The curious engineer measured the system throughput very carefully today. "


def test_gopher_accepts_ordinary_prose() -> None:
    assert gopher_quality_filter(GOOD_SENTENCE * 10)  # 110 words, mean len ~5.5


def test_gopher_word_count_bounds() -> None:
    assert not gopher_quality_filter(GOOD_SENTENCE * 4)  # 44 words < 50
    assert not gopher_quality_filter("word " * 100_001)  # > 100k words
    assert gopher_quality_filter("word " * 100)  # in range, mean len 4


def test_gopher_mean_word_length_bounds() -> None:
    assert not gopher_quality_filter("an is " * 50)  # mean 2 < 3
    assert not gopher_quality_filter("extraordinarily incomprehensibilities " * 50)  # mean > 10
    assert gopher_quality_filter("the with " * 50)  # mean 3.5


def test_gopher_ellipsis_line_fraction() -> None:
    ellipsis = [f"teaser line number {i} keeps going and going..." for i in range(40)]
    normal = [f"regular line number {i} states one complete fact." for i in range(60)]
    assert not gopher_quality_filter("\n".join(ellipsis + normal))  # 40% > 30%
    assert gopher_quality_filter("\n".join(ellipsis[:20] + normal + normal))  # ~14%


def test_gopher_alpha_word_fraction() -> None:
    numeric = "1729 " * 45
    alpha = "meaningful " * 15
    assert not gopher_quality_filter(numeric + alpha)  # 25% alpha words < 80%
    assert gopher_quality_filter(alpha * 4)  # 100% alpha


# ---------------------------------------------------------------------------
# The shared threshold rule (hermetic)
# ---------------------------------------------------------------------------


def test_classifier_keep_truth_table() -> None:
    assert classifier_keep(("en", 0.9), "en", 0.65)
    assert not classifier_keep(("de", 0.9), "en", 0.65)  # wrong label
    assert not classifier_keep(("en", 0.5), "en", 0.65)  # confident enough? no
    assert classifier_keep(("en", 0.65), "en", 0.65)  # boundary keeps


# ---------------------------------------------------------------------------
# Extraction (needs resiliparse; official fixture additionally needs the scaffold)
# ---------------------------------------------------------------------------


def test_extract_text_utf8_and_encoding_fallback() -> None:
    pytest.importorskip("resiliparse")
    html = "<html><body><h1>Tiêu đề</h1><p>Xin chào <b>thế giới</b></p></body></html>"
    out = filters.extract_text_from_html_bytes(html.encode("utf-8"))
    assert out is not None
    assert "Xin chào thế giới" in out

    latin1 = "<html><body><p>caf\xe9 menu</p></body></html>".encode("latin-1")
    with pytest.raises(UnicodeDecodeError):
        latin1.decode("utf-8")  # proves the fallback path is exercised
    out = filters.extract_text_from_html_bytes(latin1)
    assert out is not None
    assert "menu" in out


@needs_official_fixtures
def test_extract_text_matches_official_moby_fixture() -> None:
    pytest.importorskip("resiliparse")
    raw = (OFFICIAL_FIXTURES / "moby.html").read_bytes()
    expected = (OFFICIAL_FIXTURES / "moby_extracted.txt").read_text()
    assert filters.extract_text_from_html_bytes(raw) == expected


# ---------------------------------------------------------------------------
# Classifier filters (conditional on model files; never download in tests)
# ---------------------------------------------------------------------------


@needs_lid
def test_identify_language_english_and_chinese() -> None:
    pytest.importorskip("fasttext")
    lang, score = filters.identify_language(
        "The quick brown fox jumps over the lazy dog near the river bank."
    )
    assert lang == "en"
    assert 0.5 < score <= 1.0
    lang, score = filters.identify_language("欢迎来到我们的网站，这里有很多有趣的内容")
    assert lang == "zh"
    assert 0.5 < score <= 1.0


@needs_nsfw
def test_classify_nsfw_labels() -> None:
    pytest.importorskip("fasttext")
    label, score = filters.classify_nsfw(
        "hardcore porn xxx explicit sex video suck my cock you filthy slut"
    )
    assert label == "nsfw"
    assert 0.5 < score <= 1.0
    label, score = filters.classify_nsfw(
        "have a nice day at the library with your family and friends"
    )
    assert label == "non-nsfw"
    assert 0.5 < score <= 1.0


@needs_toxic
def test_classify_toxic_speech_labels() -> None:
    pytest.importorskip("fasttext")
    label, score = filters.classify_toxic_speech(
        "you are a worthless idiot and everyone hates you, shut up you disgusting moron you fucker"
    )
    assert label == "toxic"
    assert 0.5 < score <= 1.0
    label, score = filters.classify_toxic_speech(
        "thank you for the thoughtful review, I will update the section accordingly"
    )
    assert label == "non-toxic"
    assert 0.5 < score <= 1.0


# ---------------------------------------------------------------------------
# Pipeline: canonical order, PII placement, accounting, dedup-last (hermetic)
# ---------------------------------------------------------------------------


class RecordingClassifier:
    """Fake `(label, score)` classifier that records every (stage, text) call."""

    def __init__(self, stage: str, log: list[tuple[str, str]], verdict) -> None:
        self.stage = stage
        self.log = log
        self.verdict = verdict

    def __call__(self, text: str) -> tuple[str, float]:
        self.log.append((self.stage, text))
        return self.verdict(text)


def _doc(marker: str) -> str:
    """A synthetic doc that passes Gopher (>=50 words, sane structure) carrying a marker word."""
    return f"marker {marker} appears right here. " + GOOD_SENTENCE * 8


def test_pipeline_runs_stages_in_canonical_order_and_charges_first_dropping_stage() -> None:
    calls: list[tuple[str, str]] = []
    config = FilterConfig(
        dedup=False,
        language_classifier=RecordingClassifier(
            "language", calls, lambda t: ("de", 0.99) if "germanic" in t else ("en", 0.99)
        ),
        nsfw_classifier=RecordingClassifier(
            "nsfw", calls, lambda t: ("nsfw", 0.99) if "lewd" in t else ("non-nsfw", 0.99)
        ),
        toxic_classifier=RecordingClassifier(
            "toxic", calls, lambda t: ("toxic", 0.99) if "hostile" in t else ("non-toxic", 0.99)
        ),
        quality_classifier=RecordingClassifier(
            "quality", calls, lambda t: ("cc", 0.99) if "lowqual" in t else ("wiki", 0.99)
        ),
    )
    docs = [_doc("clean"), _doc("germanic"), _doc("lewd"), _doc("hostile"), _doc("lowqual")]
    report = filter_data(docs, config)

    assert report.kept == 1
    assert report.survivors[0].doc_id == "doc00000"
    assert report.discarded == {
        "extract": 0, "language": 1, "gopher": 0, "nsfw": 1, "toxic": 1, "quality": 1, "dedup": 0,
    }  # fmt: skip
    # The surviving doc visits classifier stages in canonical order.
    clean_calls = [stage for stage, text in calls if "clean" in text]
    assert clean_calls == ["language", "nsfw", "toxic", "quality"]
    # A doc dropped at nsfw never reaches later stages (first-dropping-stage charging).
    lewd_calls = [stage for stage, text in calls if "lewd" in text]
    assert lewd_calls == ["language", "nsfw"]
    # Global stage ordering is the canonical one.
    assert list(report.discarded) == list(STAGE_ORDER)


def test_pipeline_masks_pii_after_harmful_before_quality() -> None:
    calls: list[tuple[str, str]] = []
    email = "leaker@example.com"
    config = FilterConfig(
        dedup=False,
        gopher=False,
        nsfw_classifier=RecordingClassifier("nsfw", calls, lambda t: ("non-nsfw", 0.99)),
        quality_classifier=RecordingClassifier("quality", calls, lambda t: ("wiki", 0.99)),
    )
    report = filter_data([f"please write to {email} about the incident report"], config)

    (nsfw_seen,) = [text for stage, text in calls if stage == "nsfw"]
    (quality_seen,) = [text for stage, text in calls if stage == "quality"]
    assert email in nsfw_seen  # harmful classifiers see the RAW page
    assert email not in quality_seen  # quality sees the masked text
    assert "|||EMAIL_ADDRESS|||" in quality_seen
    assert report.pii_masked["emails"] == 1
    assert report.kept == 1 and "|||EMAIL_ADDRESS|||" in report.survivors[0].text


def test_pipeline_accounting_identity_dedup_last_and_discard_table_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[tuple[str, str]] = []
    base = GOOD_SENTENCE * 8  # 80 words: passes Gopher
    docs = [
        base,  # A — survivor, first of the near-dup pair
        base.upper().replace(".", "!!"),  # B — near-dup of A after normalization → dedup drop
        "way too short to pass gopher",  # C — gopher drop
        _doc("germanic"),  # D — language drop
        _doc("hostile"),  # E — toxic drop
        # F, G — survivors with fully distinct bodies (no shared 5-grams with A or each other,
        # otherwise the shared base text would make them true near-dups and dedup would merge).
        "Economic historians debate the causes of the rapid postwar expansion in detail. " * 8,
        "Travel notes from the northern valley describe quiet mornings beside the river. " * 8,
    ]
    config = FilterConfig(
        language_classifier=RecordingClassifier(
            "language", calls, lambda t: ("de", 0.99) if "germanic" in t else ("en", 0.99)
        ),
        toxic_classifier=RecordingClassifier(
            "toxic", calls, lambda t: ("toxic", 0.99) if "hostile" in t else ("non-toxic", 0.99)
        ),
    )
    with caplog.at_level(logging.INFO, logger="scratch_llm.data.pipeline"):
        report = filter_data(docs, config)

    assert report.discarded == {
        "extract": 0, "language": 1, "gopher": 1, "nsfw": 0, "toxic": 1, "quality": 0, "dedup": 1,
    }  # fmt: skip
    # The audit-trail identity: every input charged exactly once.
    assert report.total_in == report.kept + sum(report.discarded.values())
    assert report.kept == 3
    # Dedup keeps the FIRST of the near-dup pair (deterministic input-order survivor).
    assert [d.doc_id for d in report.survivors] == ["doc00000", "doc00005", "doc00006"]
    # The discard table is logged at INFO with all stages + the total row.
    table = "\n".join(r.message for r in caplog.records)
    for stage in STAGE_ORDER:
        assert stage in table
    assert "TOTAL" in table and "PII masked" in table


def test_pipeline_extract_stage_discards_empty_html() -> None:
    pytest.importorskip("resiliparse")
    report = filter_data(
        [b"<html><body></body></html>", "plain text already extracted " * 10],
        FilterConfig(gopher=False, dedup=False),
    )
    assert report.discarded["extract"] == 1
    assert report.kept == 1


def test_pipeline_doc_ids_validated() -> None:
    with pytest.raises(ValueError, match="unique"):
        filter_data(["a", "b"], FilterConfig(dedup=False, gopher=False), doc_ids=["x", "x"])
    with pytest.raises(ValueError, match="length"):
        filter_data(["a", "b"], FilterConfig(dedup=False, gopher=False), doc_ids=["x"])


@needs_lid
@needs_nsfw
@needs_toxic
def test_pipeline_with_real_models_slice() -> None:
    """End-to-end slice with the real fastText classifiers (quality stage skipped — the
    persisted default quality model is a deliberate non-bundle, see quality.py)."""
    pytest.importorskip("fasttext")
    config = FilterConfig.with_default_models(quality_classifier=None)
    english = (
        "The public library reopened after extensive renovation this spring. "
        "Visitors can now browse the expanded local history collection every afternoon. "
    ) * 5
    chinese = "欢迎来到我们的网站，这里有很多有趣的内容和文章。" * 20
    rant = "you are a worthless idiot and everyone hates you, shut up you disgusting moron. " * 12
    report = filter_data([english, chinese, rant], config)
    assert report.kept == 1
    assert report.discarded["language"] == 1
    # The rant trips BOTH Jigsaw models (measured: nsfw 1.0, toxic 1.0); first-dropping-stage
    # charging books it to nsfw, the earlier harmful stage in the canonical order.
    assert report.discarded["nsfw"] == 1
    assert report.discarded["toxic"] == 0


def test_doc_result_is_frozen() -> None:
    doc = DocResult("id", "text")
    with pytest.raises(AttributeError):
        doc.text = "other"  # type: ignore[misc]
