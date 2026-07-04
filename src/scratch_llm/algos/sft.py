"""A5 SFT + masking primitives — the per-token log-prob machinery every post-training loss shares.

CS336 A5 §4.2 (``tokenize_prompt_and_output`` → ``compute_entropy`` → ``get_response_log_probs``
→ ``masked_normalize`` / ``masked_mean`` → ``sft_microbatch_train_step``). These six functions are
the substrate: SFT consumes them directly, GRPO/Dr.GRPO reuses the exact same masking + scoring
path for its policy term, and DPO reuses ``get_response_log_probs`` for both π_θ and π_ref. Get
the ``response_mask`` off by one here and every downstream RL number is silently corrupt.

Pinned semantics (matched against the official A5 scaffold snapshots in
``lectures/assignment5-alignment/tests/_snapshots``):

- **Tokenize/shift:** concatenate prompt+output token ids per row, right-pad the *concatenated*
  sequence to the batch max length, then ``input_ids = padded[:, :-1]`` and
  ``labels = padded[:, 1:]`` — the shift happens *after* padding, so a short row keeps its own
  final token inside ``input_ids`` (snapshot-pinned; shift-then-pad is the classic wrong answer).
- **Mask alignment:** ``response_mask`` lives in *label* coordinates: label index ``j`` scores the
  token at full-sequence position ``j+1``, so response positions ``[len(prompt), len(concat))``
  map to mask indices ``[len(prompt)−1, len(concat)−1)``. Prompt and padding are 0.
- **Aggregation is a hyperparameter, not a detail:** ``masked_mean`` (per-sequence mean → batch
  mean) vs ``masked_normalize`` (sum / fixed constant) is exactly the GRPO-vs-Dr.GRPO
  length-normalization lever — a per-sequence mean up-weights tokens in short responses; a fixed
  constant gives every token the same gradient weight regardless of response length.

Invariants (tested in ``tests/test_sft_algos.py``):
- entropy of uniform logits == log V, and ``compute_entropy(logits + c) == compute_entropy(logits)``
  (logsumexp form — no exp overflow);
- masked mean NLL of a fresh A1 ``TransformerLM`` ≈ log V (loss-at-init oracle, discipline #1);
- summing ``sft_microbatch_train_step`` gradients over k equal microbatches with
  ``gradient_accumulation_steps=k`` reproduces the single full-batch gradient bit-for-bit-close;
- 200 SFT steps overfit one tiny batch to per-token NLL < 0.1 (discipline #2).

Grad convention (ADR-0006): the *grad-bearing* current-policy log-probs are recomputed here via
``get_response_log_probs`` inside the training step; any log-prob stored on a rollout is a frozen
diagnostic, never a loss input. Token entropy is returned detached (a monitor signal, not a loss
term).

Interview question this module answers: "You SFT on (prompt, response) pairs — which tokens
contribute to the loss, how does the mask line up after the causal shift, and why does the choice
of length normalization change what the gradient optimizes?"
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@runtime_checkable
class TokenizerLike(Protocol):
    """Minimal duck-typed tokenizer contract: ``encode(text) -> list[int]``.

    Both HF tokenizers (whose ``encode`` also accepts ``add_special_tokens``) and the repo BPE
    :class:`scratch_llm.tokenizer.Tokenizer` satisfy it structurally. ``pad_token_id`` /
    ``eos_token_id`` attributes are read opportunistically when present.
    """

    def encode(self, text: str) -> list[int]: ...


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode without special tokens: BOS/EOS injected mid-concat would corrupt the mask."""
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:  # plain `encode(text)` tokenizers (e.g. the repo BPE)
        return list(tokenizer.encode(text))


def _resolve_pad_id(tokenizer: Any, pad_token_id: int | None) -> int:
    """Explicit argument wins; else tokenizer's pad, else EOS, else 0 (mask makes it inert)."""
    if pad_token_id is not None:
        return pad_token_id
    for attr in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: TokenizerLike,
    *,
    pad_token_id: int | None = None,
) -> dict[str, Tensor]:
    """Tokenize prompts and outputs separately, concat, pad, shift — and build ``response_mask``.

    Returns ``input_ids``/``labels`` of shape ``(B, max_concat_len − 1)`` (long) and a bool
    ``response_mask`` aligned with ``labels``: True exactly where the label token is part of the
    response (prompt and padding are False). Padding uses the tokenizer's pad id (override with
    ``pad_token_id=`` for pad-less tokenizers like the repo BPE); padded label positions hold the
    pad id but are never scored because the mask is 0 there.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError(f"{len(prompt_strs)} prompts vs {len(output_strs)} outputs")
    if not prompt_strs:
        raise ValueError("empty batch")

    pad_id = _resolve_pad_id(tokenizer, pad_token_id)
    prompt_ids = [_encode(tokenizer, p) for p in prompt_strs]
    concat_ids = [p + _encode(tokenizer, o) for p, o in zip(prompt_ids, output_strs, strict=True)]

    batch = len(concat_ids)
    max_len = max(len(c) for c in concat_ids)
    padded = torch.full((batch, max_len), pad_id, dtype=torch.long)
    response_mask_full = torch.zeros((batch, max_len), dtype=torch.bool)
    for i, (concat, prompt) in enumerate(zip(concat_ids, prompt_ids, strict=True)):
        padded[i, : len(concat)] = torch.tensor(concat, dtype=torch.long)
        response_mask_full[i, len(prompt) : len(concat)] = True

    # Shift AFTER padding (snapshot-pinned). Label index j scores full-sequence position j+1,
    # so the mask aligned with labels is the full-sequence mask shifted left by one.
    return {
        "input_ids": padded[:, :-1],
        "labels": padded[:, 1:],
        "response_mask": response_mask_full[:, 1:],
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Per-position entropy of the next-token distribution, H = logZ − Σ p·logit (Eq. 1).

    ``logits: (..., V) → (...)``. The logsumexp form never exponentiates raw logits, so it is
    stable under any additive shift: ``compute_entropy(logits + c) == compute_entropy(logits)``.
    Uniform logits give exactly ``log V``. This is the entropy-collapse monitor for RL runs.
    """
    log_z = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return log_z - (probs * logits).sum(dim=-1)


def get_response_log_probs(
    model: nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score ``labels`` under the model: per-token ``log p_θ(labels_t | input_ids_{≤t})``.

    Works with both the A1 :class:`~scratch_llm.model.TransformerLM` (returns raw logits) and HF
    causal LMs (return an object with ``.logits``) — the one duck-typed seam SFT, GRPO and DPO all
    score through. ``log_probs`` keeps the autograd graph (this IS the grad-bearing policy term,
    ADR-0006); ``token_entropy`` (present only when ``return_token_entropy=True``) is detached —
    it is a monitor signal, not a loss input.
    """
    out = model(input_ids)
    logits: Tensor = out.logits if hasattr(out, "logits") else out
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": log_probs}
    if return_token_entropy:
        with torch.no_grad():
            result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Masked sum divided by a *fixed* constant: ``Σ (tensor·mask) / normalize_constant``.

    ``dim=None`` reduces everything; otherwise reduces ``dim`` only. The constant does not depend
    on the mask — that is the Dr.GRPO side of the length-norm lever: every response token carries
    the same gradient weight no matter how long its sequence is (contrast :func:`masked_mean`).
    """
    masked = tensor * mask.to(tensor.dtype)
    total = masked.sum() if dim is None else masked.sum(dim=dim)
    return total / normalize_constant


def masked_mean(tensor: Tensor, mask: Tensor, dim: int | None = None) -> Tensor:
    """Mean of ``tensor`` over positions where ``mask`` is 1: ``Σ(tensor·mask) / Σ mask``.

    ``dim=None`` reduces everything. Per-sequence (``dim=-1``) this normalizes by *that row's*
    response length — the GRPO-style aggregation that up-weights tokens in short responses.
    Caller invariant: the mask must have ≥1 active element per reduced slice (else division by 0).
    """
    mask_f = mask.to(tensor.dtype)
    if dim is None:
        return (tensor * mask_f).sum() / mask_f.sum()
    return (tensor * mask_f).sum(dim=dim) / mask_f.sum(dim=dim)


def aggregate_loss_across_microbatch(
    per_token_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: float | None = None,
) -> Tensor:
    """Reduce a per-token loss ``(B, L)`` to a scalar under the chosen length-normalization.

    ``"sequence"``: mean over each sequence's masked tokens, then mean over sequences (GRPO
    default). ``"constant"``: total masked sum / ``normalization_constant`` (Dr.GRPO / SFT with a
    fixed Z). Thin composition of the two masked ops so the choice stays a one-word toggle.
    """
    if loss_normalization == "sequence":
        return masked_mean(per_token_loss, mask, dim=-1).mean()
    if normalization_constant is None:
        raise ValueError("normalization_constant is required when loss_normalization='constant'")
    return masked_normalize(per_token_loss, mask, normalization_constant)


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One SFT microbatch: masked NLL, scaled for grad accumulation, ``backward()`` called.

    Loss per example = ``Σ_t −log p_θ(o_t)·mask_t / normalize_constant`` (a *fixed* constant, not
    the row's token count); microbatch loss = batch mean; the returned (and backwarded) loss is
    that divided by ``gradient_accumulation_steps``, so accumulating k equal microbatches
    reproduces the full-batch gradient exactly (tested invariance). Caller owns
    ``optimizer.step()`` / ``zero_grad()`` cadence.

    Returns ``(scaled_loss, metadata)`` with ``microbatch_loss`` (unscaled, detached),
    ``mean_token_nll`` (the log-V-comparable overfit/underfit signal) and
    ``num_response_tokens`` in metadata.
    """
    if gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be ≥ 1, got {gradient_accumulation_steps}"
        )
    nll = -policy_log_probs
    per_example = masked_normalize(nll, response_mask, normalize_constant, dim=-1)
    microbatch_loss = per_example.mean()
    loss = microbatch_loss / gradient_accumulation_steps
    loss.backward()
    metadata = {
        "microbatch_loss": microbatch_loss.detach(),
        "mean_token_nll": masked_mean(nll, response_mask).detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }
    return loss, metadata
