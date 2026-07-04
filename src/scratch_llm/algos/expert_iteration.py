"""Expert Iteration / STaR (A5 §5, Alg. 2) — reinforcement learning without a policy gradient.

Intent: the cheapest RL baseline and the conceptual bridge from SFT to GRPO. Each step: sample
``G`` rollouts per task from the current policy, grade them with the env's verifiable grader,
**keep only the correct ones**, and SFT on that filtered set — then repeat. Filtering + SFT
amplifies the probability mass the model *already* places on correct behavior; no advantage, no
importance ratio, no value network.

**Why EI plateaus — the no-credit-assignment note (DoD).** EI's only learning signal is binary
keep/drop at the *sequence* level: every token of a kept rollout is reinforced equally (SFT), and
a dropped rollout contributes nothing — there is no per-token credit assignment (a mostly-right
solution with one bad step teaches exactly as much as random noise: zero), no negative gradient
pushing probability *away* from wrong answers, and no exploration pressure beyond sampling
temperature. Once the policy stops producing *new* correct rollouts on the still-unsolved tasks,
the filtered set stops changing and the fixed point is reached — the tasks it cannot yet solve
stay unsolved. That plateau is precisely the motivation for GRPO: group-relative *advantages*
give wrong rollouts negative weight and per-token gradients scale with how much better a rollout
is than its group.

Invariants (tested in ``tests/test_expert_iteration.py``): per-step metrics are computed on the
rollouts sampled *before* that step's SFT update (so ``kept_fraction`` at step *t* measures the
policy after *t* updates' worth of prior steps); with zero kept rollouts the model's parameters
are untouched that step; the whole loop is deterministic given a deterministic ``SampleFn``.

Interview question this module answers: "Why does filtering-then-SFT (STaR/EI) improve a
reasoning model at all, and what structural limitation makes policy-gradient methods (GRPO)
eventually necessary?"
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import nan
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from scratch_llm.algos.sft import get_response_log_probs, sft_microbatch_train_step
from scratch_llm.envs.protocol import Task, VerifiableEnv
from scratch_llm.rollout.types import Rollout


@runtime_checkable
class SampleFn(Protocol):
    """The sampling seam: produce ``group_size`` rollouts for one task from the *current* policy.

    A Protocol (not a backend) so CI can inject a deterministic scripted sampler while real runs
    plug a ``RolloutClient``-backed one — the EI loop itself stays free of GPU sampling.
    """

    def __call__(self, task: Task, group_size: int) -> list[Rollout]: ...


@dataclass(frozen=True)
class EIStepMetrics:
    """One EI step's ledger, measured on the rollouts sampled at the *start* of the step."""

    step: int
    n_rollouts: int
    n_correct: int
    kept_fraction: float  # n_correct / n_rollouts — the learning curve EI must move up
    n_kept: int  # after de-duplication; the actual SFT set size
    mean_reward: float
    mean_format_reward: float
    mean_answer_reward: float
    mean_sft_loss: float  # NaN when nothing was kept (no update that step)
    num_sft_examples: int


def collate_prompt_response_ids(
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    pad_id: int = 0,
) -> dict[str, Tensor]:
    """Build ``input_ids``/``labels``/``response_mask`` from already-tokenized id sequences.

    Same pinned semantics as :func:`scratch_llm.algos.sft.tokenize_prompt_and_output` (pad the
    concatenated sequence, *then* shift; mask lives in label coordinates) — but starting from ids,
    which is what EI has: rollouts carry token ids, not strings.
    """
    if len(prompt_ids) != len(response_ids):
        raise ValueError(f"{len(prompt_ids)} prompts vs {len(response_ids)} responses")
    if not prompt_ids:
        raise ValueError("empty batch")
    concat = [list(p) + list(r) for p, r in zip(prompt_ids, response_ids, strict=True)]
    batch, max_len = len(concat), max(len(c) for c in concat)
    padded = torch.full((batch, max_len), pad_id, dtype=torch.long)
    mask_full = torch.zeros((batch, max_len), dtype=torch.bool)
    for i, (row, prompt) in enumerate(zip(concat, prompt_ids, strict=True)):
        padded[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask_full[i, len(prompt) : len(row)] = True
    return {
        "input_ids": padded[:, :-1],
        "labels": padded[:, 1:],
        "response_mask": mask_full[:, 1:],
    }


def expert_iteration(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    env: VerifiableEnv,
    sample_fn: SampleFn,
    *,
    n_ei_steps: int,
    group_size: int,
    sft_epochs_per_step: int = 1,
    sft_batch_size: int = 8,
    keep_threshold: float = 1.0,
    normalize_constant: float = 1.0,
    pad_id: int = 0,
    deduplicate: bool = True,
) -> list[EIStepMetrics]:
    """Run Alg. 2: (sample ``G`` per task → grade → keep correct → SFT on kept) × ``n_ei_steps``.

    Keeps a rollout when ``Graded.reward >= keep_threshold`` (default 1.0 = format AND answer
    correct under the r1-zero composition). ``deduplicate`` drops exact repeats of
    ``(task_id, response_ids)`` within a step — G identical greedy samples should count once in
    the SFT set (their multiplicity is not a learning signal). SFT reuses the W8a primitives:
    :func:`get_response_log_probs` for the grad-bearing log-probs and
    :func:`sft_microbatch_train_step` for the masked-NLL update (one optimizer step per
    minibatch, ``gradient_accumulation_steps=1``). Returns one :class:`EIStepMetrics` per step.
    """
    if n_ei_steps < 1 or group_size < 1:
        raise ValueError(f"need n_ei_steps ≥ 1 and group_size ≥ 1, got {n_ei_steps}, {group_size}")
    tasks = env.tasks()
    if not tasks:
        raise ValueError("env has no tasks")

    history: list[EIStepMetrics] = []
    for step in range(n_ei_steps):
        # --- sample + grade (metrics measure the policy BEFORE this step's update) -----------
        kept: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        n_rollouts = n_correct = 0
        sum_reward = sum_format = sum_answer = 0.0
        for task in tasks:
            rollouts = sample_fn(task, group_size)
            if len(rollouts) != group_size:
                raise ValueError(
                    f"sample_fn returned {len(rollouts)} rollouts for group_size={group_size}"
                )
            for rollout in rollouts:
                graded = env.grade(task, rollout)
                n_rollouts += 1
                sum_reward += graded.reward
                sum_format += graded.format_reward
                sum_answer += graded.answer_reward
                if graded.reward >= keep_threshold:
                    n_correct += 1
                    key = (task.task_id, rollout.response_ids)
                    if deduplicate and key in seen:
                        continue
                    seen.add(key)
                    kept.append((task.prompt_ids, rollout.response_ids))

        # --- SFT on the filtered-correct set --------------------------------------------------
        losses: list[float] = []
        if kept:
            model.train()
            for _ in range(sft_epochs_per_step):
                for start in range(0, len(kept), sft_batch_size):
                    chunk = kept[start : start + sft_batch_size]
                    batch = collate_prompt_response_ids(
                        [p for p, _ in chunk], [r for _, r in chunk], pad_id=pad_id
                    )
                    out = get_response_log_probs(model, batch["input_ids"], batch["labels"])
                    _, meta = sft_microbatch_train_step(
                        out["log_probs"],
                        batch["response_mask"],
                        gradient_accumulation_steps=1,
                        normalize_constant=normalize_constant,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    losses.append(float(meta["microbatch_loss"]))

        history.append(
            EIStepMetrics(
                step=step,
                n_rollouts=n_rollouts,
                n_correct=n_correct,
                kept_fraction=n_correct / n_rollouts,
                n_kept=len(kept),
                mean_reward=sum_reward / n_rollouts,
                mean_format_reward=sum_format / n_rollouts,
                mean_answer_reward=sum_answer / n_rollouts,
                mean_sft_loss=sum(losses) / len(losses) if losses else nan,
                num_sft_examples=len(kept),
            )
        )
    return history
