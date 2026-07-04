"""W8d — per-instance DPO loss + Bradley-Terry RM loss (`scratch_llm.algos.dpo`).

Hermetic, CPU-only, seeded. The analytic backbone: for a *constant-logit* LM (every position
emits the same logits row) with uniform logits everywhere except a boost δ on the chosen
response's word (policy) and ρ on the same word (ref), the four sequence log-prob sums collapse
so that policy_margin = δ, ref_margin = ρ, hence

    loss = −log σ(β·(δ − ρ))    exactly.

That one closed form drives the hand-computed case, both monotonicity tests, and the
DPO ≡ Bradley-Terry(implicit reward) identity. The official tiny-gpt2 fixture test (0.9104,
β=0.5) runs conditionally against the local scaffold checkout — no network, skips if absent.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from scratch_llm.algos.dpo import (
    ALPACA_PROMPT_TEMPLATE,
    bradley_terry_rm_loss,
    per_instance_dpo_loss,
)
from scratch_llm.model import ModelConfig, TransformerLM

# --------------------------------------------------------------------------------------------
# Stubs: word-level tokenizer (official-fixture semantics) + fully-controlled logit models
# --------------------------------------------------------------------------------------------

_VOCAB = {
    "<pad>": 0,
    "<eos>": 1,
    "<unk>": 2,
    "good": 3,
    "bad": 4,
    "fix": 5,
    "it": 6,
}
_VOCAB_SIZE = 32  # model vocab; ids above the word list stay unused
_WORD_RE = re.compile(r"\w+|[^\w\s]")  # mirrors the official fixture's Whitespace pre-tokenizer


class WordStubTokenizer:
    """Plain-`encode(text)` word-level stub; unknown words (the Alpaca boilerplate) -> <unk>."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        return [_VOCAB.get(tok, 2) for tok in _WORD_RE.findall(text)]


class EOSFreeTokenizer(WordStubTokenizer):
    """Same vocab but no EOS attribute — exercises the no-EOS-to-append path."""

    eos_token_id = None  # pyright: ignore[reportAssignmentType]


class ConstantLogitLM(nn.Module):
    """Context-free LM: every position emits the same learnable logits row (B, L, V).

    Makes every DPO term analytic: log p(token t) = logits[t] − logsumexp(logits), independent of
    position and prefix — the sharpest possible oracle for margin arithmetic.
    """

    def __init__(self, logits: Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone())

    def forward(self, token_ids: Tensor) -> Tensor:
        batch, seq = token_ids.shape
        return self.logits.view(1, 1, -1).expand(batch, seq, -1)


class LogitNudge(nn.Module):
    """Wrap a base LM and add +delta to one token's logit — 'policy upweights chosen' knob."""

    def __init__(self, base: nn.Module, token_id: int, delta: float) -> None:
        super().__init__()
        self.base = base
        self.token_id = token_id
        self.delta = delta

    def forward(self, token_ids: Tensor) -> Tensor:
        logits = self.base(token_ids)
        assert isinstance(logits, Tensor)
        bumped = logits.clone()
        bumped[..., self.token_id] = bumped[..., self.token_id] + self.delta
        return bumped


def _boosted_lm(token_id: int, delta: float) -> ConstantLogitLM:
    logits = torch.zeros(_VOCAB_SIZE)
    logits[token_id] = delta
    return ConstantLogitLM(logits)


def _tiny_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=_VOCAB_SIZE, d_model=32, n_layers=2, n_heads=2, context_length=64)
    return TransformerLM(cfg)


TOK = WordStubTokenizer()
PROMPT = "fix it"
CHOSEN = "good"  # -> ids [3, eos]: the 2-token response of the hand-computed case
REJECTED = "bad"  # -> ids [4, eos]

LOG2 = math.log(2.0)


def _sigma_loss(beta: float, margin: float) -> float:
    """Hand formula: −log σ(β·margin), computed with math.* (independent of torch)."""
    return -math.log(1.0 / (1.0 + math.exp(-beta * margin)))


# --------------------------------------------------------------------------------------------
# policy == ref  ⇒  loss == log 2, for every β
# --------------------------------------------------------------------------------------------


def test_policy_equals_ref_gives_log2_exactly() -> None:
    model = _tiny_model(seed=0)
    ref = copy.deepcopy(model)
    for beta in (0.1, 0.5, 1.0, 2.0):
        loss = per_instance_dpo_loss(model, ref, TOK, beta, PROMPT, CHOSEN, REJECTED)
        assert loss.ndim == 0
        # Identical weights ⇒ the four sums cancel pairwise bit-exactly ⇒ σ(0) ⇒ log 2.
        assert math.isclose(loss.item(), LOG2, rel_tol=0.0, abs_tol=1e-7)


def test_same_module_as_both_policy_and_ref_gives_log2() -> None:
    model = _tiny_model(seed=1)
    loss = per_instance_dpo_loss(model, model, TOK, 0.5, PROMPT, CHOSEN, REJECTED)
    assert math.isclose(loss.item(), LOG2, rel_tol=0.0, abs_tol=1e-7)


# --------------------------------------------------------------------------------------------
# Hand-computed 2-token case (constant-logit oracle): loss == −log σ(β(δ−ρ)) to 1e-6
# --------------------------------------------------------------------------------------------


def test_hand_computed_two_token_case_uniform_ref() -> None:
    """Chosen = [good, eos], rejected = [bad, eos]; policy boosts 'good' by δ, ref uniform.

    Under a constant-logit model: logZ_θ = log(V−1+e^δ); logp_θ(good) = δ − logZ_θ, and every
    other token (bad, eos) scores −logZ_θ. The shared eos term cancels in the chosen−rejected
    difference, so policy_margin = δ exactly; the uniform ref's margin is 0. Loss = −log σ(βδ).
    """
    delta, beta = 1.3, 0.7
    policy = _boosted_lm(token_id=3, delta=delta)
    ref = ConstantLogitLM(torch.zeros(_VOCAB_SIZE))
    loss = per_instance_dpo_loss(policy, ref, TOK, beta, PROMPT, CHOSEN, REJECTED)
    assert math.isclose(loss.item(), _sigma_loss(beta, delta), rel_tol=0.0, abs_tol=1e-6)


def test_hand_computed_case_ref_margin_subtracts() -> None:
    """Ref also boosts 'good' (ρ): loss = −log σ(β(δ−ρ)) — the ref margin must subtract.

    With ρ > δ the *ref* prefers chosen more strongly than the policy does, so the loss goes
    ABOVE log 2 even though the policy itself prefers chosen — the KL anchor at work.
    """
    delta, rho, beta = 0.9, 1.7, 0.5
    policy = _boosted_lm(token_id=3, delta=delta)
    ref = _boosted_lm(token_id=3, delta=rho)
    loss = per_instance_dpo_loss(policy, ref, TOK, beta, PROMPT, CHOSEN, REJECTED)
    assert math.isclose(loss.item(), _sigma_loss(beta, delta - rho), rel_tol=0.0, abs_tol=1e-6)
    assert loss.item() > LOG2


# --------------------------------------------------------------------------------------------
# Monotonicity: upweighting chosen strictly decreases the loss; β scales the margin
# --------------------------------------------------------------------------------------------


def test_loss_strictly_decreases_as_policy_upweights_chosen() -> None:
    ref = ConstantLogitLM(torch.zeros(_VOCAB_SIZE))
    losses = [
        per_instance_dpo_loss(
            _boosted_lm(token_id=3, delta=d), ref, TOK, 1.0, PROMPT, CHOSEN, REJECTED
        ).item()
        for d in (0.0, 0.5, 1.0, 2.0, 4.0)
    ]
    assert math.isclose(losses[0], LOG2, rel_tol=0.0, abs_tol=1e-7)  # δ=0 is the coin flip
    assert all(a > b for a, b in zip(losses, losses[1:], strict=False))


def test_loss_strictly_decreases_with_nudged_a1_model() -> None:
    """Same monotonicity through a real tiny TransformerLM, nudging the 'good' logit."""
    base = _tiny_model(seed=2)
    ref = copy.deepcopy(base)
    losses = [
        per_instance_dpo_loss(
            LogitNudge(base, token_id=3, delta=d), ref, TOK, 1.0, PROMPT, CHOSEN, REJECTED
        ).item()
        for d in (0.0, 0.5, 1.0, 2.0)
    ]
    assert math.isclose(losses[0], LOG2, rel_tol=0.0, abs_tol=1e-7)
    assert all(a > b for a, b in zip(losses, losses[1:], strict=False))


def test_beta_scaling_monotonicity() -> None:
    """Positive margin: larger β pushes the loss toward 0. Negative margin: toward +∞."""
    ref = ConstantLogitLM(torch.zeros(_VOCAB_SIZE))
    prefers_chosen = _boosted_lm(token_id=3, delta=1.0)
    prefers_rejected = _boosted_lm(token_id=4, delta=1.0)
    betas = (0.1, 0.5, 1.0, 2.0, 5.0)
    good = [
        per_instance_dpo_loss(prefers_chosen, ref, TOK, b, PROMPT, CHOSEN, REJECTED).item()
        for b in betas
    ]
    bad = [
        per_instance_dpo_loss(prefers_rejected, ref, TOK, b, PROMPT, CHOSEN, REJECTED).item()
        for b in betas
    ]
    assert all(a > b for a, b in zip(good, good[1:], strict=False))
    assert all(a < b for a, b in zip(bad, bad[1:], strict=False))
    assert all(g < LOG2 < w for g, w in zip(good, bad, strict=True))


# --------------------------------------------------------------------------------------------
# Gradients: policy-only (π_ref is a frozen anchor, never a gradient path)
# --------------------------------------------------------------------------------------------


def test_gradient_flows_to_policy_only() -> None:
    policy = _tiny_model(seed=3)
    ref = copy.deepcopy(policy)  # weights identical; still must receive NO grad
    loss = per_instance_dpo_loss(
        policy,
        ref,
        TOK,
        beta=1.0,
        prompt=PROMPT,
        response_chosen=CHOSEN,
        response_rejected=REJECTED,
    )
    assert loss.requires_grad
    loss.backward()
    # At margin 0 dℓ/dmargin = −β/2 ≠ 0, so real gradient mass must land on the policy…
    policy_grad_norm = sum(
        p.grad.abs().sum().item() for p in policy.parameters() if p.grad is not None
    )
    assert policy_grad_norm > 0.0
    assert policy.lm_head.weight.grad is not None
    # …and none on the reference — its params never enter the graph (grad is None, not just 0).
    assert all(p.grad is None for p in ref.parameters())
    assert all(p.requires_grad for p in ref.parameters())  # frozen by no_grad, not by mutation


# --------------------------------------------------------------------------------------------
# The whiteboard pairing: Bradley-Terry, and DPO == BT at the implicit reward
# --------------------------------------------------------------------------------------------


def test_bradley_terry_equal_rewards_is_log2() -> None:
    loss = bradley_terry_rm_loss(torch.tensor(1.5), torch.tensor(1.5))
    assert math.isclose(loss.item(), LOG2, rel_tol=0.0, abs_tol=1e-7)


def test_bradley_terry_hand_formula_and_monotonicity() -> None:
    margins = (-2.0, -0.5, 0.0, 0.5, 2.0)
    losses = [bradley_terry_rm_loss(torch.tensor(m), torch.tensor(0.0)).item() for m in margins]
    for m, actual in zip(margins, losses, strict=True):
        assert math.isclose(actual, _sigma_loss(1.0, m), rel_tol=0.0, abs_tol=1e-6)
    assert all(a > b for a, b in zip(losses, losses[1:], strict=False))  # better chosen ⇒ lower


def test_bradley_terry_elementwise_shape() -> None:
    rc, rr = torch.tensor([0.0, 1.0, -1.0]), torch.tensor([0.0, 0.0, 0.0])
    out = bradley_terry_rm_loss(rc, rr)
    assert out.shape == (3,)


def test_dpo_equals_bradley_terry_at_implicit_reward() -> None:
    """ℓ_DPO == ℓ_BT(β·log(π_θ/π_ref)(y_w), β·log(π_θ/π_ref)(y_l)) — the closed-form reduction.

    Constant-logit setup: implicit rewards are r̂_w = β·δ_w-effect, r̂_l = 0-effect; concretely
    with policy boosting 'good' by δ and uniform ref, log-ratio(chosen) = δ − Δ and
    log-ratio(rejected) = −Δ where Δ = 2·(logZ_θ − logZ_ref) over the two response tokens.
    The Δ cancels inside BT exactly as Z(x) cancels in the DPO derivation.
    """
    delta, beta = 1.1, 0.6
    v = float(_VOCAB_SIZE)
    log_z_theta = math.log((v - 1.0) + math.exp(delta))
    log_z_ref = math.log(v)
    # chosen = [good, eos]: logp_θ = (δ − logZ_θ) + (−logZ_θ); logp_ref = −2·logZ_ref
    logratio_w = (delta - 2.0 * log_z_theta) - (-2.0 * log_z_ref)
    # rejected = [bad, eos]: logp_θ = −2·logZ_θ; logp_ref = −2·logZ_ref
    logratio_l = (-2.0 * log_z_theta) - (-2.0 * log_z_ref)
    bt = bradley_terry_rm_loss(
        torch.tensor(beta * logratio_w), torch.tensor(beta * logratio_l)
    ).item()
    dpo = per_instance_dpo_loss(
        _boosted_lm(token_id=3, delta=delta),
        ConstantLogitLM(torch.zeros(_VOCAB_SIZE)),
        TOK,
        beta,
        PROMPT,
        CHOSEN,
        REJECTED,
    ).item()
    assert math.isclose(dpo, bt, rel_tol=0.0, abs_tol=1e-6)


# --------------------------------------------------------------------------------------------
# Formatting seam + guards
# --------------------------------------------------------------------------------------------


def test_alpaca_template_matches_official_prompt_file() -> None:
    """The hardcoded template must stay verbatim with the scaffold's alpaca_sft.prompt."""
    prompt_file = (
        Path(__file__).resolve().parents[2]
        / "lectures/assignment5-alignment/cs336_alignment/prompts_safety/alpaca_sft.prompt"
    )
    if not prompt_file.is_file():
        pytest.skip("official A5 scaffold checkout not present")
    official = prompt_file.read_text()
    assert official == ALPACA_PROMPT_TEMPLATE + "{response}\n"


def test_eos_free_tokenizer_still_works_and_shifts_the_number() -> None:
    """No EOS attr ⇒ nothing appended; the loss is computed over the bare response tokens."""
    ref = ConstantLogitLM(torch.zeros(_VOCAB_SIZE))
    policy = _boosted_lm(token_id=3, delta=1.0)
    loss = per_instance_dpo_loss(policy, ref, EOSFreeTokenizer(), 1.0, PROMPT, CHOSEN, REJECTED)
    # Same closed form: single-token responses, margin still δ.
    assert math.isclose(loss.item(), _sigma_loss(1.0, 1.0), rel_tol=0.0, abs_tol=1e-6)


def test_empty_response_without_eos_raises() -> None:
    policy = _tiny_model(seed=4)
    with pytest.raises(ValueError, match="zero tokens"):
        per_instance_dpo_loss(policy, policy, EOSFreeTokenizer(), 1.0, PROMPT, "", REJECTED)


# --------------------------------------------------------------------------------------------
# Conditional: the official tiny-gpt2 fixture (local scaffold checkout; no network)
# --------------------------------------------------------------------------------------------

_A5_FIXTURES = Path(__file__).resolve().parents[2] / "lectures/assignment5-alignment/tests/fixtures"


@pytest.mark.skipif(
    not (_A5_FIXTURES / "tiny-gpt2").is_dir() or not (_A5_FIXTURES / "tiny-gpt2-ref").is_dir(),
    reason="official A5 tiny-gpt2 fixtures not present",
)
def test_official_tiny_gpt2_fixture_value() -> None:
    """Replicates the scaffold's test_per_instance_dpo_loss verbatim: β=0.5 ⇒ loss ≈ 0.9104."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {
        "<pad>": 0, "<eos>": 1, "<unk>": 2, "###": 3, "Instruction": 4, ":": 5,
        "Response": 6, "The": 7, "quick": 8, "brown": 9, "fox": 10, "jumps": 11,
        "over": 12, "the": 13, "lazy": 14, "dog": 15, ".": 16, "their": 17,
        "crazy": 18, "frog": 19,
    }  # fmt: skip
    word_tokenizer = tokenizers.Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    word_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=word_tokenizer, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(_A5_FIXTURES / "tiny-gpt2")
    ref = transformers.AutoModelForCausalLM.from_pretrained(_A5_FIXTURES / "tiny-gpt2-ref")

    loss = per_instance_dpo_loss(
        model,
        ref,
        tokenizer,
        beta=0.5,
        prompt="The quick brown fox jumps over",
        response_chosen="the lazy dog.",
        response_rejected="their crazy frog.",
    )
    assert torch.isclose(loss, torch.tensor(0.9104), atol=1e-4)
