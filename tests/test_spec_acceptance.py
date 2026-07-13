"""F3 — de-confounded speculative-decoding acceptance harness (CPU mechanism suite).

The earlier serving measurement ran the n-gram drafter against RANDOM-weight targets and saw ~0
acceptance on every prompt — which falsified nothing about domains: a random target's greedy
continuation has no structure for any drafter to track. These tests pin the F3 harness itself:
(1) report shape/arithmetic on deterministic drafters (no-draft ⇒ rate 0, tok/fwd 1; self-draft
float64 ⇒ rate 1, tok/fwd ≥ k), (2) the MECHANISM — overfit a tiny model on a structured corpus
and show structured-prompt acceptance ≫ shuffled-prompt acceptance (the domain signal EXISTS once
the target is trained), (3) losslessness carried per prompt (committed == greedy, float64-exact),
(4) loud failure on degenerate inputs. The measured GPU falsifier (prose <10%, code/JSON 40–60%
on the F1 checkpoint) runs later via bench/f3_deconfound_acceptance.py.
"""

import random

import numpy as np
import pytest
import torch

from scratch_llm.eval.spec_acceptance import (
    AcceptanceReport,
    DomainPrompt,
    measure_domain_acceptance,
    report_to_markdown,
    tokenize_domain_prompts,
)
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.serving.speculative import ModelDrafter, NGramDrafter
from scratch_llm.train import TrainConfig, train


def _model(seed: int = 0, dtype: torch.dtype = torch.float64) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=128
    )
    return TransformerLM(cfg).eval().to(dtype)


class _NoDrafter:
    """Proposes nothing — speculative decode degenerates to plain 1-token decode."""

    def propose(self, context_ids, k: int) -> list[int]:
        return []


class _ByteTokenizer:
    """Minimal TextTokenizer stub: one token per UTF-8 byte."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


_PROMPTS = [
    DomainPrompt("repetitive", tuple([5, 9, 13, 2, 7] * 8)),
    DomainPrompt("repetitive", tuple([11, 4] * 12)),
    DomainPrompt("random", tuple((i * 37 + 11) % 256 for i in range(20))),
]


# ------------------------------------------------------------- 1. report shape + arithmetic
def test_no_draft_report_arithmetic() -> None:
    """No drafts ⇒ per-domain: 0 drafted/accepted, rate 0, exactly 1 token per decode forward,
    n_tokens = n_prompts · T; rows keep first-appearance domain order; float64 ⇒ lossless."""
    model = _model()
    report = measure_domain_acceptance(model, _PROMPTS, _NoDrafter(), max_new_tokens=12, k=4)
    assert isinstance(report, AcceptanceReport)
    assert [row.domain for row in report.rows] == ["repetitive", "random"]
    assert len(report.prompts) == 3
    by = report.by_domain()
    assert by["repetitive"].n_prompts == 2 and by["random"].n_prompts == 1
    for row in report.rows:
        assert row.n_drafted == 0 and row.n_accepted == 0
        assert row.acceptance_rate == 0.0
        assert row.n_tokens == row.n_prompts * 12
        assert row.n_decode_forwards == row.n_tokens  # plain decode: 1 token per forward
        assert row.tokens_per_forward == pytest.approx(1.0)
        assert row.lossless and row.n_lossless == row.n_prompts
    assert report.lossless


def test_self_draft_full_acceptance() -> None:
    """Draft == target (float64) ⇒ every draft accepted: rate 1.0, ≥ k tokens per decode forward."""
    model = _model()
    prompts = [DomainPrompt("any", tuple((i * 37 + 11) % 256 for i in range(20)))]
    report = measure_domain_acceptance(model, prompts, ModelDrafter(model), max_new_tokens=20, k=4)
    row = report.rows[0]
    assert row.acceptance_rate == pytest.approx(1.0)
    assert row.tokens_per_forward >= 4.0
    assert report.lossless


def test_domain_rows_sum_prompt_stats() -> None:
    """Aggregation invariant: every domain row is the exact field-wise sum of its prompt results."""
    model = _model()
    report = measure_domain_acceptance(model, _PROMPTS, NGramDrafter(n=3), max_new_tokens=16, k=4)
    for row in report.rows:
        mine = [r for r in report.prompts if r.prompt.domain == row.domain]
        assert row.n_prompts == len(mine)
        assert row.n_drafted == sum(r.stats.n_drafted for r in mine)
        assert row.n_accepted == sum(r.stats.n_accepted for r in mine)
        assert row.n_tokens == sum(r.stats.n_tokens for r in mine)
        assert row.n_decode_forwards == sum(r.stats.n_target_forwards - 1 for r in mine)
        assert row.n_lossless == sum(r.lossless for r in mine)


# ------------------------------------------------------------- 2. losslessness (float64 exact)
def test_lossless_per_prompt_ngram_float64() -> None:
    """Committed ids == plain greedy `generate` output for EVERY prompt, token-exact (float64)."""
    model = _model()
    report = measure_domain_acceptance(model, _PROMPTS, NGramDrafter(n=3), max_new_tokens=32, k=4)
    for res in report.prompts:
        assert res.committed_ids == res.greedy_ids
        assert res.lossless and res.token_agreement == pytest.approx(1.0)
    assert report.lossless


# ------------------------------------------------------------- 3. the OVERFIT mechanism test
def test_overfit_structured_prompt_beats_shuffled() -> None:
    """The F3 mechanism, end-to-end on CPU: overfit a tiny model on a repeating 32-token cycle,
    then n-gram acceptance on a structured prompt (two periods of the cycle — the trailing 3-gram
    recurs one period back, and the memorized target follows the proposed continuation) must beat
    acceptance on the SAME tokens shuffled (no earlier n-gram match ⇒ nothing useful to draft).
    This is the trained-target counterpart of the random-weights null result."""
    period = 32
    rng = random.Random(0)
    cycle = list(range(period))
    rng.shuffle(cycle)
    data = np.tile(np.array(cycle, dtype=np.int64), 128)  # 4096-token corpus, pure structure

    cfg = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=128
    )
    torch.manual_seed(0)
    model = TransformerLM(cfg)
    history = train(
        TrainConfig(
            max_steps=150,  # measured: loss ≈ 0.10 by step 50, ≈ 0.02 by step 100 (gate: < 0.2)
            batch_size=8,
            context_length=64,
            max_lr=3e-3,
            min_lr=3e-4,
            warmup_steps=10,
            log_every=10,
            device="cpu",
            seed=0,
        ),
        data,
        model,
    )
    final_loss = history[-1][1]
    assert final_loss < 0.2, (
        f"overfit failed (final loss {final_loss:.3f}) — the mechanism test needs a memorized "
        "corpus; raise max_steps/max_lr"
    )

    model = model.double().eval()  # float64 measurement ⇒ losslessness is reduction-order-exact
    structured = cycle * 2  # 64 tokens: trailing 3-gram occurs one period earlier
    shuffled = list(structured)
    random.Random(3).shuffle(shuffled)  # same tokens, structure destroyed
    prompts = [
        DomainPrompt("structured", tuple(structured)),
        DomainPrompt("shuffled", tuple(shuffled)),
    ]
    report = measure_domain_acceptance(model, prompts, NGramDrafter(n=3), max_new_tokens=24, k=4)
    by = report.by_domain()
    s, r = by["structured"], by["shuffled"]

    assert report.lossless  # committed == greedy for BOTH domains — acceptance is never bought
    assert s.acceptance_rate > r.acceptance_rate, (
        f"structured {s.acceptance_rate:.1%} ≤ shuffled {r.acceptance_rate:.1%}: "
        "domain signal missing on a memorized corpus"
    )
    assert s.acceptance_rate >= 0.5  # memorized cycle: drafts are the true continuation
    assert r.acceptance_rate <= 0.2  # no earlier n-gram match ⇒ (almost) nothing drafted
    assert s.tokens_per_forward > 1.5  # the acceptance actually buys tokens per target forward


# ------------------------------------------------------------- 4. degenerate inputs + helpers
def test_degenerate_inputs_raise() -> None:
    model = _model()
    drafter = NGramDrafter(n=3)
    with pytest.raises(ValueError):
        measure_domain_acceptance(model, [], drafter, max_new_tokens=8)
    with pytest.raises(ValueError):
        measure_domain_acceptance(model, [DomainPrompt("empty", ())], drafter, max_new_tokens=8)
    with pytest.raises(ValueError):
        measure_domain_acceptance(model, _PROMPTS, drafter, max_new_tokens=0)
    with pytest.raises(ValueError):
        measure_domain_acceptance(model, _PROMPTS, drafter, max_new_tokens=8, k=-1)


def test_tokenize_domain_prompts() -> None:
    prompts = tokenize_domain_prompts(_ByteTokenizer(), {"prose": ["hi", "yo"], "code": ["x=1"]})
    assert [p.domain for p in prompts] == ["prose", "prose", "code"]
    assert prompts[0].ids == tuple(b"hi") and prompts[0].text == "hi"
    assert prompts[2].ids == tuple(b"x=1")


def test_report_to_markdown() -> None:
    model = _model()
    report = measure_domain_acceptance(model, _PROMPTS, _NoDrafter(), max_new_tokens=8, k=4)
    md = report_to_markdown(report)
    lines = md.splitlines()
    assert lines[0].startswith("| domain |")
    assert any("| repetitive |" in line for line in lines)
    assert any("| random |" in line for line in lines)
    assert "0.0%" in md and "1.00" in md  # rate + tokens/forward formatting
