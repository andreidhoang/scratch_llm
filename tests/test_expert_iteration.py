"""W8b — Expert Iteration (`scratch_llm.algos.expert_iteration`) on a toy env + tiny A1 model.

The sampler stub is *deterministic inverse-CDF sampling*: rollout g of G is correct iff the
current policy's probability of the answer token exceeds the fixed quantile g/G. That makes the
"sampled" correct-fraction a step function of p(answer|prompt) — so EI's learning (SFT on kept →
p rises → more quantiles crossed) is provable with zero sampling flakiness. Threshold g=0 is 0,
so at least one rollout per task is always correct: the EI bootstrap can never stall at step 0.
Hermetic, CPU-only, seeded.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from scratch_llm.algos.expert_iteration import (
    EIStepMetrics,
    SampleFn,
    collate_prompt_response_ids,
    expert_iteration,
)
from scratch_llm.envs.protocol import DecodeFn, Graded, Task, VerifiableEnv
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import AdamW
from scratch_llm.rollout.types import Rollout

# Toy universe: vocab 8. Prompts use tokens {0..3}, answers are {4,5,6}, 7 is the wrong token.
VOCAB = 8
N_TASKS = 3
WRONG_TOKEN = 7
GROUP_SIZE = 8


class ToyTokenEnv:
    """Single-token-answer env: task i has prompt (0, i+1) and answer token 4+i."""

    def __init__(self) -> None:
        self._tasks = [
            Task(task_id=f"toy-{i}", prompt_ids=(0, i + 1), ground_truth=str(4 + i))
            for i in range(N_TASKS)
        ]

    def decode(self) -> DecodeFn:
        return lambda ids: " ".join(str(int(i)) for i in ids)

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def grade(self, task: Task, rollout: Rollout) -> Graded:
        text = self.decode()(rollout.response_ids)
        ok = float(text == task.ground_truth)
        return Graded(reward=ok, format_reward=1.0, answer_reward=ok, response_text=text)


class QuantileSampler:
    """Deterministic 'sampler': rollout g emits the answer token iff p(answer|prompt) > g/G."""

    def __init__(self, model: TransformerLM) -> None:
        self.model = model

    def _p_answer(self, task: Task) -> float:
        with torch.no_grad():
            logits = self.model(torch.tensor([task.prompt_ids], dtype=torch.long))
            assert isinstance(logits, torch.Tensor)
            return float(F.softmax(logits[0, -1], dim=-1)[int(task.ground_truth)])

    def __call__(self, task: Task, group_size: int) -> list[Rollout]:
        p = self._p_answer(task)
        rollouts = []
        for g in range(group_size):
            token = int(task.ground_truth) if p > g / group_size else WRONG_TOKEN
            rollouts.append(
                Rollout(
                    prompt_ids=task.prompt_ids,
                    response_ids=(token,),
                    logprobs=(math.log(max(p, 1e-9)),),
                    stop_reason="stop",
                )
            )
        return rollouts


class AlwaysWrongSampler:
    """Never produces a correct rollout — the EI no-signal edge case."""

    def __call__(self, task: Task, group_size: int) -> list[Rollout]:
        return [
            Rollout(
                prompt_ids=task.prompt_ids,
                response_ids=(WRONG_TOKEN,),
                logprobs=(0.0,),
                stop_reason="stop",
            )
            for _ in range(group_size)
        ]


def _fresh_model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=1, n_heads=2, context_length=8)
    return TransformerLM(cfg)


def _run_ei(seed: int = 0) -> list[EIStepMetrics]:
    model = _fresh_model(seed)
    optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    env = ToyTokenEnv()
    sampler = QuantileSampler(model)
    return expert_iteration(
        model,
        optimizer,
        env,
        sampler,
        n_ei_steps=6,
        group_size=GROUP_SIZE,
        sft_epochs_per_step=1,
        sft_batch_size=4,
    )


# ------------------------------------------------------------------------------------------
# The headline invariant: EI provably improves kept-fraction over steps
# ------------------------------------------------------------------------------------------


def test_ei_kept_fraction_rises_over_steps():
    metrics = _run_ei(seed=0)
    fractions = [m.kept_fraction for m in metrics]
    # Strictly increasing every step, solved by the last (measured curve for this seed:
    # 0.167 → 0.458 → 0.708 → 0.875 → 0.958 → 1.0).
    assert all(b > a for a, b in zip(fractions, fractions[1:]))  # noqa: B905 — offset zip
    assert fractions[0] < 0.25
    assert fractions[-1] >= 0.9
    # The bootstrap guarantee: quantile g=0 has threshold 0, so step 0 always keeps something.
    assert metrics[0].n_correct >= N_TASKS
    assert metrics[0].n_kept >= 1


def test_ei_metrics_bookkeeping():
    metrics = _run_ei(seed=0)
    assert [m.step for m in metrics] == list(range(6))
    for m in metrics:
        assert m.n_rollouts == N_TASKS * GROUP_SIZE
        assert m.kept_fraction == m.n_correct / m.n_rollouts
        # Toy env: reward == answer_reward, format always well-formed.
        assert m.mean_reward == m.mean_answer_reward == m.kept_fraction
        assert m.mean_format_reward == 1.0
        # Dedup collapses each task's identical correct rollouts to one SFT example.
        assert m.n_kept <= N_TASKS
        assert m.num_sft_examples == m.n_kept
        assert not math.isnan(m.mean_sft_loss)


def test_ei_is_deterministic_given_seed():
    assert _run_ei(seed=0) == _run_ei(seed=0)  # EIStepMetrics is a frozen dataclass → value eq


def test_ei_zero_kept_leaves_model_untouched():
    model = _fresh_model(seed=1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = AdamW(model.parameters(), lr=1e-2)
    metrics = expert_iteration(
        model,
        optimizer,
        ToyTokenEnv(),
        AlwaysWrongSampler(),
        n_ei_steps=2,
        group_size=4,
    )
    for m in metrics:
        assert m.kept_fraction == 0.0 and m.n_kept == 0
        assert math.isnan(m.mean_sft_loss)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_sampler_stub_satisfies_sample_fn_protocol():
    assert isinstance(QuantileSampler(_fresh_model()), SampleFn)
    assert isinstance(AlwaysWrongSampler(), SampleFn)
    assert isinstance(ToyTokenEnv(), VerifiableEnv)


# ------------------------------------------------------------------------------------------
# The id-collate helper (same pad-then-shift semantics as tokenize_prompt_and_output)
# ------------------------------------------------------------------------------------------


def test_collate_prompt_response_ids_hand_example():
    batch = collate_prompt_response_ids([[1, 2], [3]], [[4, 5], [6]], pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 4], [3, 6, 0]]
    assert batch["labels"].tolist() == [[2, 4, 5], [6, 0, 0]]
    assert batch["response_mask"].tolist() == [
        [False, True, True],  # labels 4,5 are the response of row 0
        [True, False, False],  # label 6 is the response of row 1; padding masked out
    ]
