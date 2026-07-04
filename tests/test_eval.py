"""Tests for the report-card eval harness (F-front): val_bpb, MC scoring, generative, aggregate."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from scratch_llm.eval import (
    MCTask,
    bits_per_byte,
    build_report_card,
    core_style_score,
    evaluate_generative,
    evaluate_multiple_choice,
    option_logprob,
    predict_choice,
)
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.sampling import SamplingParams
from scratch_llm.train import TrainConfig, train


class UniformLM(nn.Module):
    """Returns all-equal logits → a uniform next-token distribution (the entropy oracle)."""

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab

    def forward(self, ids: Tensor) -> Tensor:
        return torch.zeros(*ids.shape, self.vocab)


class FavorLM(nn.Module):
    """Always boosts a single token id — a controllable model for the selection logic."""

    def __init__(self, vocab: int, favored: int, boost: float = 12.0) -> None:
        super().__init__()
        self.vocab, self.favored, self.boost = vocab, favored, boost

    def forward(self, ids: Tensor) -> Tensor:
        out = torch.zeros(*ids.shape, self.vocab)
        out[..., self.favored] = self.boost
        return out


class CharTokenizer:
    """A trivial char-level tokenizer (id = ord) for wiring tests — satisfies TextTokenizer."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_bits_per_byte_uniform_equals_log2_vocab() -> None:
    """A uniform model scores exactly log2(V) bits/token and ln(V) nats/token; with 1 byte/token
    that is log2(V) bpb. (Also pins the corrected metric: NLL summed, not meaned.)"""
    v = 8
    ids = [i % v for i in range(17)]  # 17 tokens (all < vocab) → 16 scored transitions
    res = bits_per_byte(UniformLM(v), ids, num_bytes=16, context_length=8)
    assert res.n_tokens == 16
    assert math.isclose(res.nats_per_token, math.log(v), rel_tol=1e-5)
    assert math.isclose(res.bits_per_byte, math.log2(v), rel_tol=1e-5)  # 16·log2(8)/16 = 3.0


def test_predict_choice_and_option_logprob_prefer_favored_token() -> None:
    model = FavorLM(vocab=10, favored=5)
    prompt = [1, 2, 3]
    assert predict_choice(model, prompt, [[4], [5], [6]]) == 1  # option with the favored token
    assert option_logprob(model, prompt, [5]) > option_logprob(model, prompt, [4])


def test_option_logprob_length_normalization_divides_by_length() -> None:
    model = FavorLM(vocab=10, favored=5)
    summed = option_logprob(model, [1], [5, 5, 5], length_normalize=False)
    averaged = option_logprob(model, [1], [5, 5, 5], length_normalize=True)
    assert math.isclose(averaged, summed / 3, rel_tol=1e-5)


def test_evaluate_multiple_choice_accuracy_is_deterministic() -> None:
    # 'a'==97 is the favored token; the correct option is always the one that encodes to it.
    model = FavorLM(vocab=128, favored=ord("a"))
    tok = CharTokenizer()
    examples = [("q", ["a", "b"], 0), ("r", ["c", "a"], 1)]  # correct option contains 'a'
    assert evaluate_multiple_choice(model, tok, examples).accuracy == 1.0


def test_core_style_score_centers_on_random_baseline() -> None:
    acc = {"t1": 0.5, "t2": 1.0}
    base = {"t1": 0.25, "t2": 0.5}
    expected = (((0.5 - 0.25) / 0.75) + ((1.0 - 0.5) / 0.5)) / 2
    assert math.isclose(core_style_score(acc, base), expected, rel_tol=1e-6)
    assert core_style_score({}, {}) == 0.0


@pytest.fixture(scope="module")
def trained() -> tuple[TransformerLM, np.ndarray]:
    # Trained ONCE for the module (shared across the three composition tests). vocab 128 so
    # CharTokenizer's ord()-based ids fit; the corpus uses only ids 0–15 (a period-16 cycle) the
    # model memorizes → bpb ≪ the uniform log2(128) = 7 bits.
    data = np.tile(np.arange(16, dtype=np.int64), 400)
    model = TransformerLM(
        ModelConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=4, context_length=32)
    )
    train(
        TrainConfig(max_steps=150, batch_size=16, context_length=16, max_lr=3e-3, seed=0),
        data,
        model,
    )
    return model, data


def test_bits_per_byte_drops_below_uniform_after_training(
    trained: tuple[TransformerLM, np.ndarray],
) -> None:
    model, data = trained
    held_out = data[:64]  # a slice of the same structured cycle
    res = bits_per_byte(model, held_out, num_bytes=len(held_out) - 1, context_length=16)
    assert res.bits_per_byte < 1.0, f"did not learn: {res.bits_per_byte:.3f} bpb (uniform = 7.0)"


def test_build_report_card_composes_bpb_and_mc(
    trained: tuple[TransformerLM, np.ndarray],
) -> None:
    model, data = trained
    tok = CharTokenizer()
    mc = MCTask(name="toy", examples=[("q", ["a", "b"], 0)], random_baseline=0.5)
    card = build_report_card(
        model, tok, val_tokens=list(data[:48]), val_num_bytes=47, mc_tasks=[mc], context_length=16
    )
    assert card.val_bpb is not None and math.isfinite(card.val_bpb)
    assert "toy" in card.mc and 0.0 <= card.mc["toy"] <= 1.0
    assert card.core_style is not None
    assert "val_bpb" in card.to_markdown()


def test_evaluate_generative_runs_and_grades(
    trained: tuple[TransformerLM, np.ndarray],
) -> None:
    model, _ = trained
    tok = CharTokenizer()
    # max_tokens kept within context_length (2 + 8 < 32) so RoPE stays in range; the trivial grader
    # verifies the generate→decode→grade plumbing composes.
    res = evaluate_generative(
        model,
        tok,
        examples=[("\x01\x02", "ref")],
        grade_fn=lambda comp, ref: len(comp) >= 0,
        params=SamplingParams(temperature=0.0, max_tokens=8),
    )
    assert res.n == 1 and 0.0 <= res.accuracy <= 1.0
