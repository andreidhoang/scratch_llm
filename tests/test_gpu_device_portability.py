"""GPU device-portability regression tests for the main-track A1/A5 code (sm120+).

The A5 engine (SFT · GRPO/Dr.GRPO · DPO) was built and tested CPU-first, so device bugs — batch
tensors built on CPU then fed to a CUDA model, or the numpy-native RL-logging bridge — cannot be
caught by the CPU/gloo gate. These ``@pytest.mark.gpu`` tests exercise each piece **on the GPU it
is ultimately deployed to** (the A5 runbooks target a rented CUDA box). They locked in the fix for
the ``grpo_train_loop`` device mismatch found 2026-07-04; keep them green on any GPU box.

Self-contained (inline toy env + tokenizer) so they never couple to another test module.
"""

from __future__ import annotations

import math

import pytest
import torch

from scratch_llm.algos.dpo import per_instance_dpo_loss
from scratch_llm.algos.grpo import (
    expected_reward,
    grpo_train_loop,
    make_rollout_sampler,
)
from scratch_llm.algos.sft import get_response_log_probs, sft_microbatch_train_step
from scratch_llm.envs.protocol import Graded, Task
from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy
from scratch_llm.optim import AdamW
from scratch_llm.sampling import SamplingParams

pytestmark = pytest.mark.gpu

DEV = "cuda"

# --- inline learnable single-response-token env (mirrors the CPU toy in test_grpo_algos) ---------
_VOCAB = 11
_ANSWER_BASE = 8
_N_TASKS = 3


class _SingleTokenEnv:
    def __init__(self) -> None:
        self._tasks = [
            Task(
                task_id=f"ct-{i}", prompt_ids=(1, 2 + i, 2 + i), ground_truth=str(_ANSWER_BASE + i)
            )
            for i in range(_N_TASKS)
        ]

    def decode(self):
        return lambda ids: " ".join(str(int(i)) for i in ids)

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def grade(self, task: Task, rollout) -> Graded:
        ok = float(tuple(rollout.response_ids) == (int(task.ground_truth),))
        return Graded(reward=ok, format_reward=1.0, answer_reward=ok, response_text="")


def _answer_token_of(task: Task) -> int:
    return int(task.ground_truth)


class _WordTokenizer:
    """Minimal ``.encode``/pad/eos tokenizer for the DPO device check."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        return [2 + (hash(w) % 20) for w in text.split()] or [2]


def _tiny(vocab: int, ctx: int, seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, d_model=32, n_layers=2, n_heads=2, context_length=ctx)
    return TransformerLM(cfg).to(DEV)


def test_a1_forward_backward_loss_at_init_on_gpu() -> None:
    model = _tiny(vocab=256, ctx=128)
    x = torch.randint(0, 256, (4, 64), device=DEV)
    logits = model(x)
    assert logits.device.type == "cuda"
    loss = cross_entropy(
        logits.reshape(-1, 256), torch.randint(0, 256, (4, 64), device=DEV).reshape(-1)
    )
    assert abs(loss.item() - math.log(256)) < 0.5  # loss-at-init ≈ log V
    loss.backward()
    g = next(p.grad for p in model.parameters() if p.grad is not None)
    assert g.device.type == "cuda"


def test_sft_overfit_one_batch_on_gpu() -> None:
    model = _tiny(vocab=64, ctx=32)
    opt = AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 64, (2, 12), device=DEV)
    labels = torch.randint(0, 64, (2, 12), device=DEV)
    mask = torch.ones_like(labels, dtype=torch.float32)
    for _ in range(250):
        out = get_response_log_probs(model, ids, labels)
        assert out["log_probs"].device.type == "cuda"
        sft_microbatch_train_step(out["log_probs"], mask, 1)
        opt.step()
        opt.zero_grad()
    final = sft_microbatch_train_step(
        get_response_log_probs(model, ids, labels)["log_probs"], mask, 1
    )[0]
    assert final.item() < 0.1  # drove a single batch to ~0 on device


def test_grpo_train_loop_runs_and_learns_on_gpu() -> None:
    """The device-portability regression: the RL engine must run end-to-end on CUDA (rollout →
    grade → advantage → update → the numpy-native RL logs) and demonstrably learn. Exact CPU
    convergence magnitude is not reproduced (CUDA sampling RNG differs), so the learning signal is
    asserted RNG-robustly: exact reward rises + the windowed sampled reward rises + entropy falls."""
    model = _tiny(vocab=_VOCAB, ctx=8)
    env = _SingleTokenEnv()
    opt = AdamW(model.parameters(), lr=0.05)
    sampler = make_rollout_sampler(model, SamplingParams(temperature=1.2, max_tokens=1, seed=None))

    r0 = expected_reward(model, env, _answer_token_of)
    hist = grpo_train_loop(model, env, sampler, opt, n_grpo_steps=30, group_size=12, seed=0)
    r1 = expected_reward(model, env, _answer_token_of)

    assert next(model.parameters()).device.type == "cuda"
    assert r1 > r0  # exact (noise-free) policy competence rose on device
    first = sum(h.mean_reward for h in hist[:5]) / 5
    last = sum(h.mean_reward for h in hist[-5:]) / 5
    assert last > first + 0.05  # windowed sampled-reward rise (noise-robust)
    assert hist[-1].entropy < hist[0].entropy  # policy sharpened
    for h in hist:  # mandatory RL logs finite on device (discipline #4)
        assert math.isfinite(h.kl_current_ref) and h.kl_current_ref >= 0.0
        assert math.isfinite(h.kl_current_old) and 0.0 <= h.is_ratio_ess <= 1.0


def test_dpo_loss_policy_equals_ref_on_gpu() -> None:
    model = _tiny(vocab=32, ctx=32)
    tok = _WordTokenizer()
    loss = per_instance_dpo_loss(model, model, tok, 0.5, "solve it", "the answer", "wrong answer")
    assert loss.device.type == "cuda"
    assert abs(loss.item() - math.log(2)) < 1e-4  # π_θ == π_ref ⇒ −log σ(0) = log 2


def test_bf16_autocast_forward_backward_on_gpu() -> None:
    model = _tiny(vocab=256, ctx=128)
    x = torch.randint(0, 256, (4, 64), device=DEV)
    tgt = torch.randint(0, 256, (4, 64), device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = cross_entropy(model(x).reshape(-1, 256), tgt.reshape(-1))
    loss.backward()
    g = next(p.grad for p in model.parameters() if p.grad is not None)
    assert math.isfinite(loss.item()) and not torch.isnan(g).any()  # fp32-master grads finite
