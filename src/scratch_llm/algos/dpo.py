"""A5 supplement — the per-instance DPO loss (Eq. 3) beside its Bradley-Terry parent.

CS336 A5 supplement §6.1–6.3 (Rafailov et al. 2023). Bradley-Terry reward modeling says a pairwise
preference is a logistic comparison of scalar rewards: ``ℓ_RM = −log σ(r(x,y_w) − r(x,y_l))``
(supp Eq. 1). RLHF learns that ``r`` explicitly, then PPO-optimizes the policy against it online.
DPO collapses the reward model *into the policy*: under the KL-regularized RLHF objective the
optimal policy satisfies ``r(x,y) = β·log(π_θ(y|x)/π_ref(y|x)) + β·log Z(x)``, and plugging that
implicit reward into Bradley-Terry (the intractable ``Z(x)`` cancels in the pairwise difference)
gives Eq. 3::

    ℓ_DPO = −log σ(β·[log π_θ(y_w|x) − log π_ref(y_w|x)] − β·[log π_θ(y_l|x) − log π_ref(y_l|x)])

No sampling, no explicit reward model, no online RL — just four conditional log-probabilities per
preference pair, with ``π_ref`` frozen. That is why :func:`per_instance_dpo_loss` and
:func:`bradley_terry_rm_loss` live side by side: the DPO loss *is* the BT loss evaluated at the
policy's implicit reward (tested as an identity in ``tests/test_dpo_algos.py``).

Pinned semantics (matched against the official A5 scaffold ``tests/test_dpo.py``):

- **Alpaca template**: the prompt is wrapped with ``prompts_safety/alpaca_sft.prompt`` (verbatim
  text in :data:`ALPACA_PROMPT_TEMPLATE`), the response follows ``### Response:\\n``.
- **EOS appended** after each response (as a token id — string-level EOS can retokenize).
- **Sequence-level sums**: each of the four terms is ``Σ_t log p(response_t | prefix)`` over the
  response(+EOS) tokens only. The prompt log-prob cancels pairwise (both within π_θ and within
  π_ref), so conditional and full-concat sums yield the *identical* loss; we use the conditional
  form because it reuses :func:`~scratch_llm.algos.sft.get_response_log_probs` — the one scoring
  seam SFT, GRPO and DPO all share.
- **Devices**: the models may live on different devices (the supplement's 2-GPU recipe); the loss
  is returned on the policy model's device.

Invariants (tested):
- at ``π_θ == π_ref`` the margin is exactly 0 and the loss is ``log 2`` for every β;
- upweighting the chosen response's logits under π_θ strictly decreases the loss; for a
  constant-logit model with the chosen token boosted by δ (policy) and ρ (ref), the loss is
  exactly ``−log σ(β(δ−ρ))``;
- gradients reach only the policy (π_ref is scored under ``torch.no_grad``);
- ``ℓ_DPO == ℓ_BT(β·logratio_w, β·logratio_l)`` — the closed-form reduction, as an identity.

Interview question this module answers: "RLHF (reward model + PPO) vs DPO — derive how the reward
model disappears, state what β controls, and name what you give up by going offline." (β scales
the implicit reward, i.e. the strength of the KL tether to π_ref; you give up on-policy
exploration — DPO can only reweight behaviours present in the preference data.)

DPO stays a tested *primitive* here: a training pipeline (RMSprop, HH data, grad accumulation) is
ADR-gated per the A5 build guide §7 — the loss is the load-bearing mastery artifact.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_llm.algos.sft import TokenizerLike, _encode, get_response_log_probs

# Verbatim from the official scaffold's cs336_alignment/prompts_safety/alpaca_sft.prompt, cut at
# the response slot: the formatted prompt *ends* with "### Response:\n" and the response (+EOS)
# supplies the rest of the sequence.
ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def bradley_terry_rm_loss(reward_chosen: Tensor, reward_rejected: Tensor) -> Tensor:
    """Bradley-Terry pairwise preference NLL: ``−log σ(r(x,y_w) − r(x,y_l))`` (supp Eq. 1).

    Elementwise over any matching/broadcastable reward shapes — one loss per preference pair
    (scalar inputs give a scalar); batch reduction is the caller's choice. Equal rewards give
    exactly ``log 2`` (the coin-flip preference). This is the loss an explicit RLHF reward model
    trains under; DPO is this same loss evaluated at the policy's implicit reward
    ``β·log(π_θ/π_ref)`` — see the module docstring for the cancellation argument.
    """
    return -F.logsigmoid(reward_chosen - reward_rejected)


def _model_device(model: nn.Module) -> torch.device:
    """Device the model computes on — DPO must tolerate π_θ and π_ref on different devices."""
    return next(model.parameters()).device


def _response_log_prob(model: nn.Module, prompt_ids: list[int], response_ids: list[int]) -> Tensor:
    """Sequence-level ``log p(response | prompt) = Σ_t log p(response_t | prefix)`` (0-dim).

    Scores the concatenated ids with the standard shift: label index ``j`` scores full-sequence
    position ``j+1``, so the response tokens (full positions ``[len(prompt), len(concat))``) live
    at label indices ``[len(prompt)−1, len(concat)−1)``. Keeps the autograd graph — the caller
    decides which model is frozen.
    """
    device = _model_device(model)
    full = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    per_token = get_response_log_probs(model, full[:, :-1], full[:, 1:])["log_probs"]
    return per_token[0, len(prompt_ids) - 1 :].sum()


def per_instance_dpo_loss(
    policy_model: nn.Module,
    ref_model: nn.Module,
    tokenizer: TokenizerLike,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Tensor:
    """Per-instance DPO loss (supp Eq. 3): ``−log σ(β·(policy_margin − ref_margin))``.

    The prompt is Alpaca-wrapped (:data:`ALPACA_PROMPT_TEMPLATE`); each response gets the
    tokenizer's EOS id appended, then all four sequence log-probs are summed over response(+EOS)
    tokens only. π_ref is evaluated under ``torch.no_grad`` — the reference is a frozen KL anchor,
    never a gradient path — and the scalar loss lands on ``policy_model``'s device.

    Works with both the A1 :class:`~scratch_llm.model.TransformerLM` and HF causal LMs via the
    duck-typed logits seam of :func:`~scratch_llm.algos.sft.get_response_log_probs`.
    """
    prompt_ids = _encode(tokenizer, ALPACA_PROMPT_TEMPLATE.format(instruction=prompt))
    if not prompt_ids:
        raise ValueError("tokenizer encoded the Alpaca-wrapped prompt to zero tokens")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_suffix = [int(eos_id)] if eos_id is not None else []
    chosen_ids = _encode(tokenizer, response_chosen) + eos_suffix
    rejected_ids = _encode(tokenizer, response_rejected) + eos_suffix
    if not chosen_ids or not rejected_ids:
        raise ValueError(
            "a response encoded to zero tokens (and the tokenizer has no EOS to append) — "
            "its sequence log-prob would be a vacuous 0"
        )

    logp_theta_w = _response_log_prob(policy_model, prompt_ids, chosen_ids)
    logp_theta_l = _response_log_prob(policy_model, prompt_ids, rejected_ids)
    with torch.no_grad():
        logp_ref_w = _response_log_prob(ref_model, prompt_ids, chosen_ids)
        logp_ref_l = _response_log_prob(ref_model, prompt_ids, rejected_ids)

    policy_margin = logp_theta_w - logp_theta_l
    ref_margin = (logp_ref_w - logp_ref_l).to(policy_margin.device)
    return -F.logsigmoid(beta * (policy_margin - ref_margin))
