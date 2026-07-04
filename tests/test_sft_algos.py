"""W8a — SFT + masking primitives (`scratch_llm.algos.sft`).

Hermetic, CPU-only, seeded. The mask-exactness expectations are the *official A5 scaffold
snapshot values* hand-transcribed (same word-level tokenizer semantics), so passing here means
the official `test_tokenize_prompt_and_output` snapshot passes at the W9 acceptance run.
"""

from __future__ import annotations

import math
import re

import pytest
import torch

from scratch_llm.algos.sft import (
    aggregate_loss_across_microbatch,
    compute_entropy,
    get_response_log_probs,
    masked_mean,
    masked_normalize,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import AdamW
from scratch_llm.tokenizer import Tokenizer as BPETokenizer

# --------------------------------------------------------------------------------------------
# Stub tokenizers (duck-typed seam: HF-style kwarg encode / plain encode / repo BPE)
# --------------------------------------------------------------------------------------------

_VOCAB = {
    "<pad>": 0,
    "<eos>": 1,
    "<unk>": 2,
    "Hello": 3,
    "world": 4,
    "This": 5,
    "is": 6,
    "a": 7,
    "test": 8,
    "another": 9,
}
_WORD_RE = re.compile(r"\w+|[^\w\s]")  # mirrors the official fixture's Whitespace pre-tokenizer


class WordStubTokenizer:
    """Plain-`encode(text)` tokenizer replicating the official A5 word-level test fixture."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        return [_VOCAB.get(tok, 2) for tok in _WORD_RE.findall(text)]


class HFStyleStubTokenizer(WordStubTokenizer):
    """HF-style `encode(text, add_special_tokens=...)`; injects a poison BOS unless disabled."""

    BOS = 99

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = super().encode(text)
        return [self.BOS, *ids] if add_special_tokens else ids


PROMPTS = ["Hello, world!", "This is a test.", "This is another test."]
OUTPUTS = ["Hello, world!", "This is a test.", "This is another test."]

# Official scaffold snapshot (test_tokenize_prompt_and_output.npz), transcribed verbatim.
EXPECTED_INPUT_IDS = [
    [3, 2, 4, 2, 3, 2, 4, 2, 0],
    [5, 6, 7, 8, 2, 5, 6, 7, 8],
    [5, 6, 9, 8, 2, 5, 6, 9, 8],
]
EXPECTED_LABELS = [
    [2, 4, 2, 3, 2, 4, 2, 0, 0],
    [6, 7, 8, 2, 5, 6, 7, 8, 2],
    [6, 9, 8, 2, 5, 6, 9, 8, 2],
]
EXPECTED_MASK = [
    [0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1],
]


# --------------------------------------------------------------------------------------------
# tokenize_prompt_and_output — the everything-downstream-depends-on-it test
# --------------------------------------------------------------------------------------------


def test_tokenize_mask_exactness_official_semantics() -> None:
    out = tokenize_prompt_and_output(PROMPTS, OUTPUTS, WordStubTokenizer())
    assert out["input_ids"].dtype == torch.long
    assert out["labels"].dtype == torch.long
    assert out["response_mask"].dtype == torch.bool
    assert torch.equal(out["input_ids"], torch.tensor(EXPECTED_INPUT_IDS))
    assert torch.equal(out["labels"], torch.tensor(EXPECTED_LABELS))
    assert torch.equal(out["response_mask"], torch.tensor(EXPECTED_MASK, dtype=torch.bool))


def test_tokenize_shape_is_max_concat_len_minus_one() -> None:
    out = tokenize_prompt_and_output(PROMPTS, OUTPUTS, WordStubTokenizer())
    # Longest concat is 5 + 5 = 10 tokens -> width 9.
    assert out["input_ids"].shape == (3, 9)
    assert out["labels"].shape == (3, 9)
    assert out["response_mask"].shape == (3, 9)


def test_tokenize_mask_alignment_hand_derived() -> None:
    """Row 0: prompt=4 toks, response=4 toks, concat=8, padded to 10, width 9.

    Label index j scores full-sequence position j+1, so response positions [4, 8) map to mask
    indices [3, 7): exactly 4 ones — one per response token — and the padded tail is 0.
    """
    out = tokenize_prompt_and_output(PROMPTS, OUTPUTS, WordStubTokenizer())
    mask_row = out["response_mask"][0]
    assert mask_row.sum().item() == 4
    assert mask_row[3:7].all() and not mask_row[:3].any() and not mask_row[7:].any()
    # Masked labels are exactly the response token ids (the response re-states the prompt here).
    assert out["labels"][0][mask_row].tolist() == [3, 2, 4, 2]


def test_tokenize_hf_style_gets_no_special_tokens() -> None:
    """The HF path must pass add_special_tokens=False — a BOS mid-concat corrupts the mask."""
    out = tokenize_prompt_and_output(PROMPTS, OUTPUTS, HFStyleStubTokenizer())
    assert (out["input_ids"] != HFStyleStubTokenizer.BOS).all()
    assert torch.equal(out["input_ids"], torch.tensor(EXPECTED_INPUT_IDS))


def test_tokenize_duck_types_repo_bpe_tokenizer() -> None:
    """The repo byte-BPE Tokenizer (plain encode, no pad attr) works via explicit pad id."""
    tok = BPETokenizer(vocab={i: bytes([i]) for i in range(256)}, merges=[])
    out = tokenize_prompt_and_output(["ab", "a"], ["cd", "bcd"], tok, pad_token_id=0)
    # Row 0: concat=[97,98,99,100]; row 1: concat=[97,98,99,100]; max_len=4, width 3.
    assert torch.equal(out["input_ids"], torch.tensor([[97, 98, 99], [97, 98, 99]]))
    assert torch.equal(out["labels"], torch.tensor([[98, 99, 100], [98, 99, 100]]))
    expected_mask = torch.tensor([[False, True, True], [True, True, True]])
    assert torch.equal(out["response_mask"], expected_mask)


def test_tokenize_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError):
        tokenize_prompt_and_output(["a"], [], WordStubTokenizer())


# --------------------------------------------------------------------------------------------
# compute_entropy
# --------------------------------------------------------------------------------------------


def test_entropy_of_uniform_logits_is_log_vocab() -> None:
    vocab = 128
    logits = torch.zeros(2, 5, vocab)
    torch.testing.assert_close(
        compute_entropy(logits), torch.full((2, 5), math.log(vocab)), rtol=0, atol=1e-6
    )


def test_entropy_shift_invariant_and_stable() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 50)
    base = compute_entropy(logits)
    shifted = compute_entropy(logits + 10_000.0)  # naive exp(logits) would overflow to inf
    assert torch.isfinite(shifted).all()
    # fp32 carries ~1e-3 absolute noise at magnitude 1e4; the invariant is finiteness + closeness.
    torch.testing.assert_close(base, shifted, rtol=1e-3, atol=5e-3)
    # Entropy is bounded: 0 <= H <= log V.
    assert (base >= 0).all() and (base <= math.log(50) + 1e-5).all()


def test_entropy_of_peaked_logits_near_zero() -> None:
    logits = torch.full((1, 4, 10), -100.0)
    logits[..., 3] = 100.0
    assert compute_entropy(logits).abs().max().item() < 1e-4


# --------------------------------------------------------------------------------------------
# get_response_log_probs + loss-at-init oracle (tiny A1 model)
# --------------------------------------------------------------------------------------------


def _tiny_model(vocab_size: int = 64, seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=2, context_length=32)
    return TransformerLM(cfg)


def test_get_response_log_probs_matches_manual_gather() -> None:
    model = _tiny_model()
    torch.manual_seed(1)
    input_ids = torch.randint(0, 64, (2, 10))
    labels = torch.randint(0, 64, (2, 10))
    out = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
    with torch.no_grad():
        logits = model(input_ids)
        assert isinstance(logits, torch.Tensor)
        manual = torch.log_softmax(logits.float(), dim=-1)
        manual = manual.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(out["log_probs"], manual)
    assert out["log_probs"].shape == (2, 10)
    assert out["token_entropy"].shape == (2, 10)
    assert not out["token_entropy"].requires_grad  # diagnostic, detached
    assert out["log_probs"].requires_grad  # the grad-bearing policy term (ADR-0006)


def test_get_response_log_probs_entropy_key_optional() -> None:
    model = _tiny_model()
    torch.manual_seed(2)
    ids = torch.randint(0, 64, (1, 6))
    assert "token_entropy" not in get_response_log_probs(model, ids, ids)


def test_loss_at_init_is_log_vocab() -> None:
    """Discipline #1: fresh LM ≈ uniform prediction, masked mean NLL ≈ log V."""
    vocab = 64
    model = _tiny_model(vocab)
    torch.manual_seed(3)
    input_ids = torch.randint(0, vocab, (8, 16))
    labels = torch.randint(0, vocab, (8, 16))
    mask = torch.rand(8, 16) > 0.3  # any mask: init loss is position-independent
    with torch.no_grad():
        log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
    nll = masked_mean(-log_probs, mask).item()
    assert abs(nll - math.log(vocab)) < 0.3, f"init NLL {nll:.3f} vs log V {math.log(vocab):.3f}"


# --------------------------------------------------------------------------------------------
# masked ops vs hand-computed
# --------------------------------------------------------------------------------------------

T = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
M = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)


def test_masked_normalize_hand_computed() -> None:
    # Global: (1 + 3 + 5 + 6) / 5 = 3.0
    torch.testing.assert_close(masked_normalize(T, M, 5.0), torch.tensor(3.0))
    # dim=-1: [4/2, 11/2]
    torch.testing.assert_close(masked_normalize(T, M, 2.0, dim=-1), torch.tensor([2.0, 5.5]))
    # dim=0: [1, 5, 9] / 3
    torch.testing.assert_close(
        masked_normalize(T, M, 3.0, dim=0), torch.tensor([1.0, 5.0, 9.0]) / 3.0
    )


def test_masked_mean_hand_computed() -> None:
    # Global: 15 / 4
    torch.testing.assert_close(masked_mean(T, M), torch.tensor(3.75))
    # dim=-1: [4/2, 11/2]
    torch.testing.assert_close(masked_mean(T, M, dim=-1), torch.tensor([2.0, 5.5]))
    # dim=0: [1/1, 5/1, 9/2]
    torch.testing.assert_close(masked_mean(T, M, dim=0), torch.tensor([1.0, 5.0, 4.5]))


def test_masked_ops_gradients_flow_only_through_mask() -> None:
    t = T.clone().requires_grad_(True)
    masked_normalize(t, M, 2.0).backward()
    assert t.grad is not None
    torch.testing.assert_close(t.grad, M.to(torch.float32) / 2.0)


def test_aggregate_loss_across_microbatch_modes() -> None:
    torch.manual_seed(42)
    loss_tokens = torch.randn(2, 10)
    torch.manual_seed(42)
    mask = torch.rand(2, 10) > 0.5
    seq = aggregate_loss_across_microbatch(loss_tokens, mask, "sequence")
    torch.testing.assert_close(seq, masked_mean(loss_tokens, mask, dim=-1).mean())
    const = aggregate_loss_across_microbatch(loss_tokens, mask, "constant", 42)
    torch.testing.assert_close(const, masked_normalize(loss_tokens, mask, 42.0))
    with pytest.raises(ValueError):
        aggregate_loss_across_microbatch(loss_tokens, mask, "constant")


# --------------------------------------------------------------------------------------------
# sft_microbatch_train_step — grad-accum invariance + overfit-one-batch
# --------------------------------------------------------------------------------------------


def test_sft_step_loss_value_and_metadata() -> None:
    torch.manual_seed(4)
    log_probs = (-torch.rand(2, 5)).requires_grad_(True)  # leaf, so .grad is populated
    mask = torch.tensor([[1, 1, 0, 0, 0], [0, 1, 1, 1, 0]], dtype=torch.bool)
    loss, meta = sft_microbatch_train_step(log_probs, mask, 4, normalize_constant=2.0)
    expected_micro = masked_normalize(-log_probs.detach(), mask, 2.0, dim=-1).mean()
    torch.testing.assert_close(loss.detach(), expected_micro / 4)
    torch.testing.assert_close(meta["microbatch_loss"], expected_micro)
    assert meta["num_response_tokens"].item() == 5
    assert log_probs.grad is not None  # backward() ran inside the step


def test_grad_accum_scaling_invariance() -> None:
    """k microbatches with gradient_accumulation_steps=k == one full-batch gradient."""
    torch.manual_seed(5)
    base = torch.randn(4, 6)
    mask = torch.rand(4, 6) > 0.4
    mask[:, 0] = True  # every row keeps >= 1 response token

    full = base.clone().requires_grad_(True)
    sft_microbatch_train_step(full, mask, 1, normalize_constant=3.0)

    accum = base.clone().requires_grad_(True)
    for rows in (slice(0, 2), slice(2, 4)):
        sft_microbatch_train_step(accum[rows], mask[rows], 2, normalize_constant=3.0)

    assert full.grad is not None and accum.grad is not None
    torch.testing.assert_close(accum.grad, full.grad)


def test_sft_step_rejects_bad_accum_steps() -> None:
    with pytest.raises(ValueError):
        sft_microbatch_train_step(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), 0)


def test_overfit_one_batch() -> None:
    """Discipline #2: SFT steps drive per-token NLL < 0.1 on one tiny fixed batch (CPU)."""
    vocab = 32
    model = _tiny_model(vocab, seed=6)
    torch.manual_seed(7)
    input_ids = torch.randint(0, vocab, (2, 8))
    labels = torch.randint(0, vocab, (2, 8))
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 2:] = True  # positions 0-1 play the prompt, the rest is "response"
    n_response = float(mask.sum())

    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    nll = float("inf")
    for _ in range(200):
        opt.zero_grad()
        log_probs = get_response_log_probs(model, input_ids, labels)["log_probs"]
        _, meta = sft_microbatch_train_step(log_probs, mask, 1, normalize_constant=n_response)
        opt.step()
        nll = meta["mean_token_nll"].item()
        if nll < 0.05:
            break
    assert nll < 0.1, f"failed to overfit one batch: final per-token NLL {nll:.4f}"
