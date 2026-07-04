"""A4 filter family — HTML→text extraction, language ID, PII masking, harmful-content
classifiers, and the Gopher structural heuristics.

Intent: every *classifier* filter here (language ID, NSFW, toxic — and the quality classifier in
:mod:`scratch_llm.data.quality`) shares ONE shape: ``text -> (label, score)``, thresholded by
:func:`classifier_keep`. Internalize the shape once and four filters become one; a single
threshold-sweep harness covers them all. The PII masks are *transforms* (they never drop a
document); Gopher is a transparent ``bool`` heuristic; extraction is the corpus entry point.

Invariants:
- Classifier outputs are the argmax label (fastText ``__label__`` prefix stripped, language codes
  remapped to their primary subtag) with confidence clamped to [0, 1].
- ``mask_*`` return ``(new_text, n_masked)`` and never touch an already-masked ``|||…|||`` token.
- Model files live in :func:`models_dir` (``$WORKSPACE/.data_models`` by default) and
  auto-download from :data:`MODEL_URLS` on first use; nothing in this module imports
  ``fasttext``/``resiliparse`` at module-import time, so the module stays importable — and the
  regex/Gopher paths fully usable — on a box without the ``[data]`` extras.

Interview question this module answers: "you have raw CommonCrawl — what filters do you run and
why do they all look the same?" — extract to text, then cheap/destructive first (language,
structure), classifiers behind one ``(label, score) + threshold`` interface, PII as a transform,
and the threshold on each classifier is the same cost/quality dial everywhere.
"""

from __future__ import annotations

import functools
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

#: Documented download sources (CS336 A4 handout §2.3 / §2.5). lid.176 is fastText's 176-language
#: identifier; the two Jigsaw models are the Dolma project's fastText bigram classifiers trained
#: on the Jigsaw Toxic Comments dataset.
MODEL_URLS = {
    "lid.176.bin": ("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"),
    "jigsaw_fasttext_bigrams_nsfw_final.bin": (
        "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/"
        "jigsaw_fasttext_bigrams_nsfw_final.bin"
    ),
    "jigsaw_fasttext_bigrams_hatespeech_final.bin": (
        "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/"
        "jigsaw_fasttext_bigrams_hatespeech_final.bin"
    ),
}

LID_MODEL = "lid.176.bin"
NSFW_MODEL = "jigsaw_fasttext_bigrams_nsfw_final.bin"
TOXIC_MODEL = "jigsaw_fasttext_bigrams_hatespeech_final.bin"

#: lid.176 emits ISO-639 codes; a few Chinese/English variants are remapped to the primary
#: subtag so downstream thresholds compare against one canonical code (the A4 "en/zh" remap).
_LANG_ALIASES = {"eng": "en", "zho": "zh", "cmn": "zh", "yue": "zh", "wuu": "zh"}


# ---------------------------------------------------------------------------
# Model files: location, auto-download, cached load
# ---------------------------------------------------------------------------


def models_dir() -> Path:
    """Directory holding the fastText model files.

    ``$SCRATCH_LLM_DATA_MODELS_DIR`` if set, else ``$WORKSPACE/.data_models``
    (``/workspace/.data_models`` when ``WORKSPACE`` is unset).
    """
    env = os.environ.get("SCRATCH_LLM_DATA_MODELS_DIR")
    if env:
        return Path(env)
    return Path(os.environ.get("WORKSPACE", "/workspace")) / ".data_models"


def model_path(name: str) -> Path:
    """Where ``name`` lives (or would live) on this box — tests skip when this is absent."""
    return models_dir() / name


def ensure_model(name: str, *, auto_download: bool = True) -> Path:
    """Return the local path of a registered model, downloading it on first use.

    Downloads go to a ``.partial`` file first and are renamed only on success, so a killed
    download never leaves a truncated model that ``load_model`` would choke on.
    """
    path = model_path(name)
    if path.exists():
        return path
    if name not in MODEL_URLS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODEL_URLS)}")
    if not auto_download:
        raise FileNotFoundError(f"{name} not found at {path}; download it from {MODEL_URLS[name]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(MODEL_URLS[name], partial)  # noqa: S310 — pinned https URLs
    partial.replace(path)
    return path


@functools.cache
def _load_fasttext(path: str) -> Any:
    """Load (once per path) a fastText model; silences the spurious load_model warning."""
    import fasttext

    # setattr: the eprint hook is added dynamically by fasttext; pyright can't see it.
    setattr(fasttext.FastText, "eprint", lambda *args, **kwargs: None)  # noqa: B010
    return fasttext.load_model(path)


def fasttext_classify(model: Any, text: str) -> tuple[str, float]:
    """Argmax ``(label, score)`` from any fastText supervised model — the one shared shape.

    Calls the pybind layer (``model.f.predict``) directly: the Python wrapper's
    ``np.array(probs, copy=False)`` raises under numpy>=2. Input is collapsed to a single line
    (fastText predicts one line at a time); scores are clamped to [0, 1] (fastText can emit
    1.0000x). Empty input returns ``("unknown", 0.0)``.
    """
    line = " ".join(text.split())
    predictions = model.f.predict(line + "\n", 1, 0.0, "strict")
    if not predictions:
        return ("unknown", 0.0)
    prob, label = predictions[0]
    return label.removeprefix("__label__"), min(max(float(prob), 0.0), 1.0)


def classifier_keep(classification: tuple[str, float], keep_label: str, threshold: float) -> bool:
    """The one threshold rule every filter uses: keep iff argmax label is the keep label AND its
    confidence clears the threshold (an uncertain "good" verdict is treated as a discard —
    corpus-building errs toward dropping)."""
    label, score = classification
    return label == keep_label and score >= threshold


# ---------------------------------------------------------------------------
# Extraction (the corpus entry point)
# ---------------------------------------------------------------------------


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """HTML bytes → plain text via resiliparse; UTF-8 first, sniffed encoding as fallback.

    CommonCrawl bytes are not reliably UTF-8: try strict UTF-8, then
    ``resiliparse.parse.encoding.detect_encoding`` and decode with ``errors="replace"`` (a few
    mojibake chars beat dropping the document). Returns ``None`` only when no decoding is
    possible. Matches the official ``run_extract_text_from_html_bytes`` adapter
    (``extract_plain_text`` defaults reproduce the moby fixture byte-for-byte).
    """
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding

    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        if encoding is None:
            return None
        try:
            html = html_bytes.decode(encoding, errors="replace")
        except LookupError:
            return None
    return extract_plain_text(html)


# ---------------------------------------------------------------------------
# Classifier filters: language ID, NSFW, toxic speech
# ---------------------------------------------------------------------------


def identify_language(text: str) -> tuple[str, float]:
    """``(language_code, confidence)`` from fastText lid.176 — the first instance of the
    ``(label, score) + threshold`` pattern.

    Codes are remapped to the primary subtag (``zh-…``/``yue``/``wuu`` → ``zh``, ``eng`` → ``en``)
    so a pipeline threshold like ``keep if ("en", score ≥ 0.65)`` needs exactly one code.
    """
    model = _load_fasttext(str(ensure_model(LID_MODEL)))
    label, score = fasttext_classify(model, text)
    base = label.split("-")[0].split("_")[0].lower()
    return _LANG_ALIASES.get(base, base), score


def classify_nsfw(text: str) -> tuple[str, float]:
    """``("nsfw" | "non-nsfw", confidence)`` from the Dolma Jigsaw NSFW fastText model."""
    model = _load_fasttext(str(ensure_model(NSFW_MODEL)))
    return fasttext_classify(model, text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """``("toxic" | "non-toxic", confidence)`` from the Dolma Jigsaw hate-speech fastText model."""
    model = _load_fasttext(str(ensure_model(TOXIC_MODEL)))
    return fasttext_classify(model, text)


# ---------------------------------------------------------------------------
# PII masking (transforms — never drop a document)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# US-shaped numbers: optional +1/1 prefix, 3-3-4 groups with (), spaces, dots or dashes.
# Digit lookarounds stop matches inside longer digit runs (order ids, timestamps).
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")

_IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
# Known cost (documented, accepted): a 4-part version string like "1.2.3.4" also matches.
_IP_RE = re.compile(rf"(?<!\d)(?:{_IPV4_OCTET}\.){{3}}{_IPV4_OCTET}(?!\d)")


def mask_emails(text: str) -> tuple[str, int]:
    """Replace email addresses with ``|||EMAIL_ADDRESS|||``; returns ``(new_text, n_masked)``.

    Existing ``|||EMAIL_ADDRESS|||`` tokens are inert (no ``@``), so masking is idempotent.
    """
    return _EMAIL_RE.subn("|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Replace US-format phone numbers with ``|||PHONE_NUMBER|||``; ``(new_text, n_masked)``."""
    return _PHONE_RE.subn("|||PHONE_NUMBER|||", text)


def mask_ips(text: str) -> tuple[str, int]:
    """Replace dotted-quad IPv4 addresses (octets 0–255) with ``|||IP_ADDRESS|||``;
    ``(new_text, n_masked)``."""
    return _IP_RE.subn("|||IP_ADDRESS|||", text)


# ---------------------------------------------------------------------------
# Gopher structural heuristics (Rae et al. 2021, Appendix A subset)
# ---------------------------------------------------------------------------


def gopher_quality_filter(text: str) -> bool:
    """The A4 Gopher subset — ``True`` iff the document passes all four structural checks:

    1. 50 ≤ word count ≤ 100 000 (whitespace tokens);
    2. 3 ≤ mean word length ≤ 10 characters;
    3. ≤ 30% of lines end with an ellipsis (``...`` or ``…``);
    4. ≥ 80% of words contain at least one alphabetic character.

    Cheap, transparent, and destructive — it runs *early* in the pipeline, right after
    language ID, so the expensive classifiers never see trivially unusable pages.
    """
    words = text.split()
    n_words = len(words)
    if not 50 <= n_words <= 100_000:
        return False

    mean_len = sum(len(w) for w in words) / n_words
    if not 3.0 <= mean_len <= 10.0:
        return False

    lines = text.splitlines()
    if lines:
        n_ellipsis = sum(1 for line in lines if line.rstrip().endswith(("...", "…")))
        if n_ellipsis / len(lines) > 0.30:
            return False

    n_alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    return n_alpha / n_words >= 0.80


# ---------------------------------------------------------------------------
# Official A4 adapter names (W9 wiring is `from scratch_llm.data.filters import ...`)
# ---------------------------------------------------------------------------


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """Official-adapter alias for :func:`extract_text_from_html_bytes`."""
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[str, float]:
    """Official-adapter alias for :func:`identify_language`."""
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    """Official-adapter alias for :func:`mask_emails`."""
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    """Official-adapter alias for :func:`mask_phone_numbers`."""
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    """Official-adapter alias for :func:`mask_ips`."""
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[str, float]:
    """Official-adapter alias for :func:`classify_nsfw`."""
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[str, float]:
    """Official-adapter alias for :func:`classify_toxic_speech`."""
    return classify_toxic_speech(text)


def run_gopher_quality_filter(text: str) -> bool:
    """Official-adapter alias for :func:`gopher_quality_filter`."""
    return gopher_quality_filter(text)
