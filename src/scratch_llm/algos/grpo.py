"""A5 GRPO / Dr.GRPO — group-relative policy optimization on verifiable rewards (Alg. 3).

CS336 A5 §7.2 (Eq. 28/31 advantage · Eq. 32 REINFORCE · Eq. 33 GRPO-clip). This is the RL
*optimizer* of ``algos/``: it turns a batch of graded rollouts into a policy update, reusing the
exact per-token log-prob + masking machinery SFT built (:mod:`scratch_llm.algos.sft`) so the
gradient path is identical — only the per-token *weight* changes (an advantage instead of a −1).

The two senior toggles this module makes first-class (ADR-0017):

- **GRPO (Eq. 28) vs Dr.GRPO (Eq. 31)** — ``normalize_by_std``. GRPO divides the group-centered
  reward by the group std (+ε); Dr.GRPO subtracts the mean *only*. Dividing by std up-weights
  low-variance (easy / nearly-solved) groups — a **question-difficulty bias** — so Dr.GRPO
  (``normalize_by_std=False``) is the repo default for the *loop* (the primitive keeps GRPO as its
  named default to match the Eq. 28 adapter snapshot).
- **``masked_mean`` vs ``masked_normalize`` aggregation** — ``length_normalization``. Per-sequence
  mean (GRPO) normalizes each response by *its own length*, so a token in a long response gets a
  smaller gradient than a token in a short one — the **length bias** Dr.GRPO removes by dividing by
  a *fixed* constant instead. The batch-of-2 worked example in ``tests/test_grpo_algos.py`` pins
  the two gradients apart.

Pinned numeric semantics (verified against the official A5 scaffold snapshots in
``lectures/assignment5-alignment/tests/_snapshots``):

- group std uses the **unbiased** estimator (``unbiased=True``, ddof=1): raw ``[1,0,0,1]`` with
  ``group_size=2`` gives GRPO advantages ``±0.7071`` (÷0.7071), Dr.GRPO ``±0.5``;
- naive per-token loss ``−A·logπ``; GRPO-clip ``−min(ρA, clip(ρ,1−ε,1+ε)A)`` with
  ``ρ = exp(logπ − logπ_old)``; ``was_clipped`` = the ratio left the ``1±ε`` trust region.

Grad convention (ADR-0006): the grad-bearing current-policy log-probs are **recomputed** each
microbatch via :func:`get_response_log_probs`; the log-probs stored on a rollout are the frozen
``π_old`` used only for the IS-ratio and the clip — never differentiated.

Mandatory RL logging (discipline #4, wired *into* the loop, not bolted on): every step emits
entropy, ``KL(cur‖ref)``, ``KL(cur‖old)``, IS-ratio mean+ESS, reward-distribution stats, and
length-by-correctness — all through :mod:`scratch_llm.utils.monitors`. A GRPO run without these is
an uninterpretable run.

Interview question this module answers: "Walk me from a batch of graded rollouts to a policy
gradient step — where does the group baseline come from, what does the clip buy you, and what two
biases does Dr.GRPO remove?"
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from scratch_llm.algos.sft import (
    compute_entropy,
    get_response_log_probs,
    masked_mean,
    masked_normalize,
)
from scratch_llm.envs.protocol import DecodeFn, Graded, Task, VerifiableEnv
from scratch_llm.rollout.local import LocalBackend
from scratch_llm.rollout.types import Rollout
from scratch_llm.sampling import SamplingParams
from scratch_llm.utils import monitors

LossType = Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]
LengthNorm = Literal["mean", "constant"]

# A text-in grader: reward_fn(response, ground_truth) -> {"reward","format_reward","answer_reward"}.
# Positional, matching the official drgrpo_grader.r1_zero_reward_fn / adapter reward_fn shape.
RewardFn = Callable[[str, str], dict[str, float]]

# The generation seam: produce `group_size` rollouts for one task from the *current* policy.
SampleFn = Callable[[Task, int], list[Rollout]]


# =============================================================================================
# Advantage estimation — the group baseline (Eq. 28 GRPO / Eq. 31 Dr.GRPO)
# =============================================================================================


def _group_normalize(
    raw_rewards: Tensor,
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tensor:
    """Center each group's rewards by the group mean; optionally divide by the group std.

    ``raw_rewards`` is ``(N,)`` with ``N = n_groups · group_size`` laid out group-contiguously.
    ``normalize_by_std=True`` ⇒ Eq. 28 (GRPO, ÷(std+ε), unbiased std); ``False`` ⇒ Eq. 31
    (Dr.GRPO, subtract mean only). Returns advantages ``(N,)``.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be ≥ 1, got {group_size}")
    if raw_rewards.ndim != 1 or raw_rewards.numel() % group_size != 0:
        raise ValueError(
            f"raw_rewards {tuple(raw_rewards.shape)} not divisible into groups of {group_size}"
        )
    groups = raw_rewards.view(-1, group_size)
    centered = groups - groups.mean(dim=1, keepdim=True)
    if normalize_by_std:
        std = groups.std(dim=1, keepdim=True, unbiased=True)  # ddof=1 — snapshot-pinned
        centered = centered / (std + advantage_eps)
    return centered.reshape(-1)


def compute_group_normalized_rewards(
    reward_fn: RewardFn,
    rollout_responses: Sequence[str],
    repeated_ground_truths: Sequence[str],
    group_size: int,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Grade rollouts, then group-normalize into advantages (Eq. 28 / Eq. 31).

    ``rollout_responses`` and ``repeated_ground_truths`` both have length ``rollout_batch_size =
    n_prompts · group_size`` (the ground truth repeated ``group_size`` times per prompt), laid out
    group-contiguously. ``normalize_by_std`` toggles GRPO (True, ÷group-std) ↔ Dr.GRPO (False).

    Returns ``(advantages, raw_rewards, metadata)`` — both tensors ``(rollout_batch_size,)``,
    ``metadata`` the reward-distribution stats (total/format/answer means + std/max/min) a run logs.
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            f"{len(rollout_responses)} responses vs {len(repeated_ground_truths)} ground truths"
        )
    reward_dicts = [
        reward_fn(resp, gt)
        for resp, gt in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.tensor([d["reward"] for d in reward_dicts], dtype=torch.float32)
    advantages = _group_normalize(raw_rewards, group_size, advantage_eps, normalize_by_std)
    metadata = {
        "reward_mean": float(raw_rewards.mean()),
        "reward_std": float(raw_rewards.std(unbiased=True)) if raw_rewards.numel() > 1 else 0.0,
        "reward_max": float(raw_rewards.max()),
        "reward_min": float(raw_rewards.min()),
        "format_reward_mean": float(
            sum(d["format_reward"] for d in reward_dicts) / len(reward_dicts)
        ),
        "answer_reward_mean": float(
            sum(d["answer_reward"] for d in reward_dicts) / len(reward_dicts)
        ),
    }
    return advantages, raw_rewards, metadata


# =============================================================================================
# The per-token loss family (Eq. 32 REINFORCE · Eq. 33 GRPO-clip) + the dispatcher
# =============================================================================================


def _as_column(scores: Tensor) -> Tensor:
    """Reshape a per-response ``(B,)`` or ``(B,1)`` score to ``(B,1)`` for broadcasting over ``L``."""
    if scores.ndim == 1:
        return scores.unsqueeze(-1)
    return scores


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
) -> Tensor:
    """REINFORCE per-token loss ``−A_t · logπ_θ(o_t)`` (Eq. 32), advantage broadcast over tokens.

    ``raw_rewards_or_advantages`` is ``(B,)`` or ``(B,1)`` (one scalar per response);
    ``policy_log_probs`` is ``(B, L)``. Returns ``(B, L)``. The advantage is constant across a
    response's tokens — GRPO's group baseline lives in ``A``, not in the per-token term.
    """
    return -_as_column(raw_rewards_or_advantages) * policy_log_probs


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """GRPO-clip per-token loss ``−min(ρ_t A, clip(ρ_t, 1−ε, 1+ε) A)`` (Eq. 33), ``ρ = exp(Δlogπ)``.

    The clip is the PPO trust region: it caps how far one update can chase a rollout it was not
    sampled under, which is what makes >1 epoch per rollout batch safe. Returns ``(loss, metadata)``
    with ``metadata["was_clipped"]`` a per-token bool = the ratio left the ``[1−ε, 1+ε]`` region
    (so ``clip`` actually changed it) and ``metadata["clip_fraction"]`` its mean.
    """
    adv = _as_column(advantages)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    loss = -torch.minimum(ratio * adv, clipped_ratio * adv)
    was_clipped = (ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)
    metadata = {
        "was_clipped": was_clipped,
        "clip_fraction": was_clipped.float().mean().detach(),
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: Tensor,
    loss_type: LossType,
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Dispatch the three policy-gradient variants — the baselining ablation in one switch.

    ``no_baseline`` ⇒ REINFORCE on ``raw_rewards`` (no variance reduction); ``reinforce_with_baseline``
    ⇒ REINFORCE on group ``advantages``; ``grpo_clip`` ⇒ Eq. 33 clip (needs ``old_log_probs`` +
    ``cliprange``). Returns ``(per_token_loss, metadata)`` — metadata empty except for ``grpo_clip``.
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("no_baseline requires raw_rewards")
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("reinforce_with_baseline requires advantages")
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("grpo_clip requires advantages, old_log_probs, and cliprange")
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    raise ValueError(f"unknown loss_type {loss_type!r}")


# =============================================================================================
# The microbatch update step — per-token loss → aggregate → scale → backward
# =============================================================================================


def _aggregate(
    per_token_loss: Tensor,
    response_mask: Tensor,
    length_normalization: LengthNorm,
    normalize_constant: float,
) -> Tensor:
    """Reduce a per-token loss ``(B, L)`` to a scalar, mean-over-sequences at the top level.

    ``"mean"`` (GRPO): per-sequence masked mean (÷ that row's response length), then mean over
    sequences. ``"constant"`` (Dr.GRPO): per-sequence masked sum ÷ *fixed* ``normalize_constant``,
    then mean over sequences. Both average over sequences last, so k equal microbatches scaled by
    1/k reproduce the full-batch gradient exactly (grad-accum invariance).
    """
    if length_normalization == "mean":
        per_sequence = masked_mean(per_token_loss, response_mask, dim=-1)
    elif length_normalization == "constant":
        per_sequence = masked_normalize(per_token_loss, response_mask, normalize_constant, dim=-1)
    else:  # pragma: no cover - Literal guards this
        raise ValueError(f"unknown length_normalization {length_normalization!r}")
    return per_sequence.mean()


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    loss_type: LossType,
    *,
    raw_rewards: Tensor | None = None,
    advantages: Tensor | None = None,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    length_normalization: LengthNorm = "constant",
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One GRPO microbatch: per-token loss → aggregate → ÷ grad-accum → ``backward()``.

    Computes the ``loss_type`` per-token loss, aggregates under ``length_normalization`` (Dr.GRPO
    ``"constant"`` by default, ADR-0017), scales by ``1/gradient_accumulation_steps`` so accumulating
    k equal microbatches reproduces the full-batch gradient, and calls ``backward()``. The caller
    owns ``optimizer.step()`` / ``zero_grad()`` cadence. Returns ``(scaled_loss, metadata)``.
    """
    if gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be ≥ 1, got {gradient_accumulation_steps}"
        )
    per_token_loss, meta = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    microbatch_loss = _aggregate(
        per_token_loss, response_mask, length_normalization, normalize_constant
    )
    loss = microbatch_loss / gradient_accumulation_steps
    loss.backward()
    metadata: dict[str, Tensor] = {
        **meta,
        "microbatch_loss": microbatch_loss.detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }
    return loss, metadata


# =============================================================================================
# The full training loop (Alg. 3): rollout → grade → advantage → microbatch update + logging
# =============================================================================================


@dataclass(frozen=True)
class GRPOStepMetrics:
    """One GRPO step's ledger + the mandatory-log :class:`~scratch_llm.utils.monitors.MonitorSnapshot`.

    ``mean_reward`` is measured on the rollouts sampled *before* this step's update (the learning
    curve the loop must move up); the KL / entropy / IS-ratio channels are measured *after* the
    update (how far the policy moved). ``length_*_mean`` is response length split by correctness —
    the verbosity-reward-hacking tell.
    """

    step: int
    mean_reward: float
    mean_format_reward: float
    mean_answer_reward: float
    mean_loss: float
    entropy: float
    kl_current_ref: float
    kl_current_old: float
    is_ratio_mean: float
    is_ratio_ess: float
    length_correct_mean: float
    length_incorrect_mean: float
    snapshot: monitors.MonitorSnapshot


def make_rollout_sampler(
    model: nn.Module,
    params: SamplingParams,
    device: str = "cpu",
) -> SampleFn:
    """A temperature :data:`SampleFn` over the from-scratch model via :class:`LocalBackend`.

    Draws ``group_size`` independent continuations per task. Pass ``params`` with ``seed=None`` so
    the decode does not reseed per call — the loop seeds the global RNG once, and each rollout then
    advances it, giving a *diverse* group (the reward variance GRPO needs) while staying
    reproducible. The sampler holds the live ``model`` reference, so it always samples the current
    (updated) policy.
    """
    backend = LocalBackend(model, device)  # type: ignore[arg-type]  # from-scratch TransformerLM

    def sample_fn(task: Task, group_size: int) -> list[Rollout]:
        return [backend.generate(task.prompt_ids, params) for _ in range(group_size)]

    return sample_fn


def _collate_rollouts(
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    old_logprobs: Sequence[Sequence[float]],
    pad_id: int,
) -> dict[str, Tensor]:
    """Batch ragged (prompt, response, per-token π_old logprob) triples into label-aligned tensors.

    Same pinned masking as :func:`scratch_llm.algos.sft.tokenize_prompt_and_output` (pad the
    concatenated sequence, *then* shift; mask + old-logprobs live in label coordinates).
    """
    concat = [list(p) + list(r) for p, r in zip(prompt_ids, response_ids, strict=True)]
    batch, max_len = len(concat), max(len(c) for c in concat)
    padded = torch.full((batch, max_len), pad_id, dtype=torch.long)
    mask_full = torch.zeros((batch, max_len), dtype=torch.bool)
    old_full = torch.zeros((batch, max_len), dtype=torch.float32)
    for i, (row, prompt, lp) in enumerate(zip(concat, prompt_ids, old_logprobs, strict=True)):
        ps, cs = len(prompt), len(row)
        padded[i, :cs] = torch.tensor(row, dtype=torch.long)
        mask_full[i, ps:cs] = True
        if cs > ps:
            old_full[i, ps:cs] = torch.tensor(lp, dtype=torch.float32)
    return {
        "input_ids": padded[:, :-1],
        "labels": padded[:, 1:],
        "response_mask": mask_full[:, 1:],
        "old_log_probs": old_full[:, 1:],
    }


def _grade(
    env: VerifiableEnv,
    task: Task,
    rollout: Rollout,
    reward_fn: RewardFn | None,
    decode: DecodeFn,
) -> Graded:
    """Grade one rollout: the env's grader (correct default — it has task metadata) unless an
    explicit text-in ``reward_fn`` overrides it."""
    if reward_fn is None:
        return env.grade(task, rollout)
    text = decode(rollout.response_ids)
    rd = reward_fn(text, task.ground_truth)
    return Graded(
        reward=rd["reward"],
        format_reward=rd["format_reward"],
        answer_reward=rd["answer_reward"],
        response_text=text,
    )


@torch.no_grad()
def _response_rows(
    model: nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    response_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-response-token full log-softmax rows ``(M, V)``, taken-token log-probs ``(M,)`` and
    entropy ``(M,)`` — the inputs the KL / IS-ratio / entropy monitors consume."""
    model.eval()
    out = model(input_ids)
    logits: Tensor = out.logits if hasattr(out, "logits") else out
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    rows = log_probs[response_mask]
    taken = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[response_mask]
    entropy = compute_entropy(logits)[response_mask]
    return rows, taken, entropy


def grpo_train_loop(
    policy: nn.Module,
    env: VerifiableEnv,
    sample_fn: SampleFn,
    optimizer: torch.optim.Optimizer,
    *,
    reward_fn: RewardFn | None = None,
    decode_fn: DecodeFn | None = None,
    n_grpo_steps: int = 10,
    group_size: int = 8,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = False,
    loss_type: LossType = "reinforce_with_baseline",
    length_normalization: LengthNorm = "constant",
    normalize_constant: float = 1.0,
    cliprange: float = 0.2,
    epochs_per_rollout_batch: int = 1,
    microbatch_size: int | None = None,
    max_grad_norm: float | None = None,
    pad_id: int = 0,
    seed: int = 0,
) -> list[GRPOStepMetrics]:
    """Alg. 3: (sample G/prompt → grade → group-normalize advantage → microbatch update) × steps.

    On-policy by default (``epochs_per_rollout_batch=1`` ⇒ ``reinforce_with_baseline``, no clip, no
    ``π_old`` needed for the loss). ``epochs_per_rollout_batch>1`` switches to ``grpo_clip`` and
    caches the rollout-time log-probs as ``π_old`` (mild off-policy behind the trust-region clip).
    Dr.GRPO defaults (``normalize_by_std=False`` + ``length_normalization="constant"``, ADR-0017).

    Grading uses ``env.grade`` (it has the task metadata verifiable envs need) unless a text-in
    ``reward_fn`` overrides it. **Before returning**, every step wires the mandatory RL logs
    (entropy, KL(cur‖ref), KL(cur‖old), IS-ratio mean+ESS, reward stats, length-by-correctness)
    through :mod:`scratch_llm.utils.monitors`. Returns one :class:`GRPOStepMetrics` per step.
    """
    if epochs_per_rollout_batch > 1 and loss_type != "grpo_clip":
        loss_type = "grpo_clip"  # >1 epoch is off-policy ⇒ the clip is required for stability
    torch.manual_seed(seed)
    decode: DecodeFn = decode_fn if decode_fn is not None else env.decode()
    tasks = env.tasks()
    if not tasks:
        raise ValueError("env has no tasks")

    ref_model = copy.deepcopy(policy)  # frozen π_ref for KL(cur‖ref)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    history: list[GRPOStepMetrics] = []
    for step in range(n_grpo_steps):
        # --- rollout + grade (mean_reward measures the policy BEFORE this step's update) ---------
        prompt_ids: list[Sequence[int]] = []
        response_ids: list[Sequence[int]] = []
        old_logprobs: list[Sequence[float]] = []
        responses_text: list[str] = []
        ground_truths: list[str] = []
        graded: list[Graded] = []
        for task in tasks:
            rollouts = sample_fn(task, group_size)
            if len(rollouts) != group_size:
                raise ValueError(f"sample_fn returned {len(rollouts)} rollouts, want {group_size}")
            for rollout in rollouts:
                g = _grade(env, task, rollout, reward_fn, decode)
                prompt_ids.append(rollout.prompt_ids)
                response_ids.append(rollout.response_ids)
                old_logprobs.append(rollout.logprobs)
                responses_text.append(g.response_text)
                ground_truths.append(task.ground_truth)
                graded.append(g)

        raw_rewards = torch.tensor([g.reward for g in graded], dtype=torch.float32)
        advantages = _group_normalize(raw_rewards, group_size, advantage_eps, normalize_by_std)
        batch = _collate_rollouts(prompt_ids, response_ids, old_logprobs, pad_id)
        n = raw_rewards.numel()
        mb = microbatch_size if microbatch_size is not None else n
        chunks = [slice(s, min(s + mb, n)) for s in range(0, n, mb)]
        grad_accum = len(chunks)

        old_model = copy.deepcopy(policy)  # π_old snapshot (pre-update) for KL(cur‖old) + IS ratio
        old_model.eval()

        # --- microbatch update (optionally multiple epochs behind the clip) ---------------------
        policy.train()
        losses: list[float] = []
        for _epoch in range(epochs_per_rollout_batch):
            optimizer.zero_grad()
            for chunk in chunks:
                out = get_response_log_probs(
                    policy, batch["input_ids"][chunk], batch["labels"][chunk]
                )
                _, meta = grpo_microbatch_train_step(
                    out["log_probs"],
                    batch["response_mask"][chunk],
                    grad_accum,
                    loss_type,
                    raw_rewards=_as_column(raw_rewards[chunk]),
                    advantages=_as_column(advantages[chunk]),
                    old_log_probs=batch["old_log_probs"][chunk],
                    cliprange=cliprange,
                    length_normalization=length_normalization,
                    normalize_constant=normalize_constant,
                )
                losses.append(float(meta["microbatch_loss"]))
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        history.append(
            _log_step(
                step=step,
                policy=policy,
                ref_model=ref_model,
                old_model=old_model,
                batch=batch,
                raw_rewards=raw_rewards,
                graded=graded,
                response_ids=response_ids,
                mean_loss=sum(losses) / len(losses) if losses else 0.0,
            )
        )
    return history


def _log_step(
    *,
    step: int,
    policy: nn.Module,
    ref_model: nn.Module,
    old_model: nn.Module,
    batch: dict[str, Tensor],
    raw_rewards: Tensor,
    graded: Sequence[Graded],
    response_ids: Sequence[Sequence[int]],
    mean_loss: float,
) -> GRPOStepMetrics:
    """Assemble one step's mandatory RL logs (discipline #4) into a :class:`GRPOStepMetrics`."""
    input_ids, labels, mask = batch["input_ids"], batch["labels"], batch["response_mask"]
    cur_rows, cur_taken, cur_entropy = _response_rows(policy, input_ids, labels, mask)
    ref_rows, _, _ = _response_rows(ref_model, input_ids, labels, mask)
    old_rows, old_taken, _ = _response_rows(old_model, input_ids, labels, mask)

    cur_np, ref_np, old_np = cur_rows.numpy(), ref_rows.numpy(), old_rows.numpy()
    is_ratios = monitors.importance_ratios(cur_taken.numpy(), old_taken.numpy())
    rewards_np = raw_rewards.numpy()
    lengths = [len(r) for r in response_ids]
    correct = [len for len, g in zip(lengths, graded, strict=True) if g.reward > 0]
    incorrect = [len for len, g in zip(lengths, graded, strict=True) if g.reward <= 0]

    # Single train/infer engine here (LocalBackend both roles) ⇒ kl_train_infer == 0 by construction;
    # the channel becomes load-bearing the moment a distinct serving engine (SGLang) is wired in.
    snapshot = monitors.build_snapshot(
        kl_current_ref=monitors.mean_kl(cur_np, ref_np),
        kl_current_old=monitors.mean_kl(cur_np, old_np),
        kl_train_infer=0.0,
        is_ratios=is_ratios,
        rewards=rewards_np,
        lengths=lengths,
    )
    return GRPOStepMetrics(
        step=step,
        mean_reward=float(rewards_np.mean()),
        mean_format_reward=float(sum(g.format_reward for g in graded) / len(graded)),
        mean_answer_reward=float(sum(g.answer_reward for g in graded) / len(graded)),
        mean_loss=mean_loss,
        entropy=float(cur_entropy.mean()),
        kl_current_ref=snapshot.kl_current_ref,
        kl_current_old=snapshot.kl_current_old,
        is_ratio_mean=snapshot.is_ratio_mean,
        is_ratio_ess=snapshot.is_ratio_ess,
        length_correct_mean=float(sum(correct) / len(correct)) if correct else float("nan"),
        length_incorrect_mean=float(sum(incorrect) / len(incorrect)) if incorrect else float("nan"),
        snapshot=snapshot,
    )


def expected_reward(
    model: nn.Module,
    env: VerifiableEnv,
    answer_token_of: Callable[[Task], int],
) -> float:
    """Deterministic ``E[reward]`` for a single-response-token env: mean over tasks of the policy's
    probability of that task's answer token. The noise-free convergence oracle the toy end-to-end
    asserts strictly rises (the sampled ``mean_reward`` is its Monte-Carlo estimate)."""
    model.eval()
    total = 0.0
    tasks = env.tasks()
    with torch.no_grad():
        for task in tasks:
            out: Any = model(torch.tensor([task.prompt_ids], dtype=torch.long))
            logits: Tensor = out.logits if hasattr(out, "logits") else out
            probs = torch.softmax(logits[0, -1].float(), dim=-1)
            total += float(probs[answer_token_of(task)])
    return total / len(tasks)
