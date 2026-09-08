"""S1/S-R2 — CPU tests for the FP8 quality arms.

What is checkable on a laptop is the arithmetic and the refusals, and that is what this
file checks. The HF load itself is not: `transformers` is a declared dependency
(pyproject.toml) that bootstrap.sh installs on the box, so `build_quality_arms` is exercised
there and the two tests that need it skip here rather than pretending.

The load-bearing test is `test_score_nll_alignment_*`: a model that puts all its mass on the
NEXT token must score ~0 nats. Off-by-one in either direction turns that into a large number,
so it pins the one line of this module that has no other way to be wrong.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from scratch_llm.quant.fp8_weights import convert_linears_to_fp8
from scratch_llm.serving import quality_arms as hq
from scratch_llm.serving.quality_arms import (
    QualityArm,
    extract_final_number,
    gsm8k_gold,
)


# ── fakes: the smallest things satisfying what QualityArm actually calls ────────────
class _Out:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _OracleLM(nn.Module):
    """Puts all mass on the token that actually follows. Perfect predictor => NLL -> 0."""

    def __init__(self, vocab: int, *, shift: int = 1) -> None:
        super().__init__()
        self.vocab = vocab
        self.shift = shift
        self._p = nn.Parameter(torch.zeros(1))  # so .parameters() has a device

    def forward(self, ids: torch.Tensor) -> _Out:
        logits = torch.full((*ids.shape, self.vocab), -30.0)
        for b in range(ids.shape[0]):
            for t in range(ids.shape[1]):
                nxt = t + self.shift
                if nxt < ids.shape[1]:
                    logits[b, t, int(ids[b, nxt])] = 30.0
        return _Out(logits)


class _Tok:
    pad_token_id, eos_token_id = 0, 0

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def __call__(self, text: str, return_tensors: str | None = None):
        return {"input_ids": torch.tensor([self._ids])}


def _engine(ids: list[int], vocab: int = 16, shift: int = 1) -> QualityArm:
    return QualityArm("bf16", _OracleLM(vocab, shift=shift), _Tok(ids))


# ── the answer graders ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text,want",
    [
        ("first 12 then 5, so 17", "17"),  # last, not first: CoT ends in the answer
        ("the total is 1,234 dollars", "1234"),  # thousands separators stripped
        ("answer: -8", "-8"),
        ("no digits here", None),
        ("it costs 3.50.", "3.50"),  # trailing sentence period is not the decimal
    ],
)
def test_extract_final_number(text: str, want: str | None) -> None:
    assert extract_final_number(text) == want


def test_gsm8k_gold_reads_the_marker() -> None:
    assert gsm8k_gold("Jane has 3 apples ... #### 18") == "18"
    with pytest.raises(ValueError, match="no gold number"):
        gsm8k_gold("a narrative answer with no number at all")


# ── the scoring arithmetic ──────────────────────────────────────────────────────────────
def test_score_nll_alignment_perfect_predictor_scores_zero(monkeypatch) -> None:
    """Predict t+1 from t. If the slice were off by one this would be ~30 nats/transition."""
    monkeypatch.setattr(hq, "NLL_WINDOW", 4)
    monkeypatch.setitem(hq.CORPORA, "toy", ("toy", None, "test"))
    monkeypatch.setattr(hq, "_load_dataset", lambda *a, **k: [{"text": "unused"}])
    nats, n = _engine([1, 2, 3, 4, 5, 6, 7, 8]).score_nll("toy")
    assert n == 2 * (4 - 1)  # two whole windows, three transitions each
    assert nats == pytest.approx(0.0, abs=1e-3)


def test_score_nll_misaligned_predictor_is_not_free(monkeypatch) -> None:
    """The negative control: a model predicting the CURRENT token must not score ~0."""
    monkeypatch.setattr(hq, "NLL_WINDOW", 4)
    monkeypatch.setitem(hq.CORPORA, "toy", ("toy", None, "test"))
    monkeypatch.setattr(hq, "_load_dataset", lambda *a, **k: [{"text": "unused"}])
    nats, _ = _engine([1, 2, 3, 4, 5, 6, 7, 8], shift=0).score_nll("toy")
    assert nats > 10.0


def test_score_nll_drops_the_trailing_partial_window(monkeypatch) -> None:
    """Both arms must see an identical stream; a ragged tail is where that breaks."""
    monkeypatch.setattr(hq, "NLL_WINDOW", 4)
    monkeypatch.setitem(hq.CORPORA, "toy", ("toy", None, "test"))
    monkeypatch.setattr(hq, "_load_dataset", lambda *a, **k: [{"text": "unused"}])
    _, n_exact = _engine([1, 2, 3, 4, 5, 6, 7, 8]).score_nll("toy")
    _, n_ragged = _engine([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).score_nll("toy")
    assert n_exact == n_ragged == 6


def test_score_nll_refuses_a_corpus_shorter_than_one_window(monkeypatch) -> None:
    monkeypatch.setattr(hq, "NLL_WINDOW", 4)
    monkeypatch.setitem(hq.CORPORA, "toy", ("toy", None, "test"))
    monkeypatch.setattr(hq, "_load_dataset", lambda *a, **k: [{"text": "unused"}])
    with pytest.raises(RuntimeError, match="shorter than one"):
        _engine([1, 2]).score_nll("toy")


# ── the refusals ────────────────────────────────────────────────────────────────────────
def test_decode_batch_refuses_to_produce_a_throughput_number() -> None:
    """The whole reason this engine is separate: it must never look like S-R2's speedup."""
    with pytest.raises(NotImplementedError, match="comparison to nothing"):
        _engine([1, 2, 3, 4]).decode_batch(object())


def test_unknown_corpus_and_slice_raise_rather_than_substitute() -> None:
    eng = _engine([1, 2, 3, 4])
    with pytest.raises(KeyError, match="unknown corpus"):
        eng.score_nll("wikitext103-test")
    with pytest.raises(KeyError, match="unknown GSM8K slice"):
        eng.gsm8k_correct("gsm8k-test")


def test_fp8_conversion_is_a_silent_noop_on_a_non_nn_Linear_model() -> None:
    """Pins the trap `build_quality_arms` guards against, on a model shaped like ours.

    convert_linears_to_fp8 dispatches on isinstance(m, nn.Linear). A projection class that
    merely quacks like Linear is skipped and the "fp8" arm stays bf16 — which is why
    build_quality_arms raises on an empty conversion list rather than trusting the call.
    """

    class Linear(nn.Module):  # deliberately NOT nn.Linear — mirrors model.py:125
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(32, 32))

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = Linear()

    assert convert_linears_to_fp8(Block()) == []

    class Real(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(32, 32)

    assert convert_linears_to_fp8(Real()) == ["proj"]


def test_build_quality_arms_needs_transformers_and_says_so() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="transformers"):
            hq.build_quality_arms(device="cpu")
    else:
        pytest.skip("transformers present — the HF load is exercised on the box, not here")
