"""W8c — GRPO / Dr.GRPO engine (`scratch_llm.algos.grpo`).

Hermetic, CPU-only, seeded, < 60 s. Six invariant groups, one per DoD line:

1. group-normalization vs hand-computed for BOTH toggles (Eq. 28 GRPO ÷std ↔ Eq. 31 Dr.GRPO
   subtract-mean), including the difficulty-bias case: a low-variance group is up-weighted under
   GRPO but not Dr.GRPO;
2. the GRPO clip activates EXACTLY outside 1±ε (``was_clipped`` metadata; boundary is the
   off-by-one mutation guard);
3. the ``compute_policy_gradient_loss`` dispatcher equals its three parts;
4. the batch-of-2 length-normalization worked example — ``masked_mean`` vs ``masked_normalize`` give
   DIFFERENT, hand-derivable gradients (the Dr.GRPO length-bias argument in code);
5. gradient-accumulation invariance (k microbatches == 1 batch);
6. toy end-to-end — ``grpo_train_loop`` makes mean reward strictly rise on a learnable single-token
   Countdown-style env, plus a real ``CountdownEnv`` integration smoke.

The group-norm / loss numerics are pinned to the official A5 scaffold snapshots
(``lectures/assignment5-alignment/tests/_snapshots``): raw ``[1,0,0,1]`` ⇒ GRPO ``±0.7071``
(unbiased std), Dr.GRPO ``±0.5``; naive loss ``−A·logπ``; clip ``−min(ρA, clip(ρ)A)``.
"""

from __future__ import annotations

import math

import pytest
import torch

from scratch_llm.algos.grpo import (
    GRPOStepMetrics,
    compute_group_normalized_rewards,
    compute_grpo_clip_loss,
    compute_naive_policy_gradient_loss,
    compute_policy_gradient_loss,
    expected_reward,
    grpo_microbatch_train_step,
    grpo_train_loop,
    make_rollout_sampler,
)
from scratch_llm.algos.sft import masked_mean, masked_normalize
from scratch_llm.envs.protocol import DecodeFn, Graded, Task
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import AdamW
from scratch_llm.rollout.types import Rollout
from scratch_llm.sampling import SamplingParams
from scratch_llm.utils import monitors


def _reward_from_response(response: str, ground_truth: str) -> dict[str, float]:
    """A reward_fn that reads the reward straight off the response string (``"0.6"`` → 0.6)."""
    del ground_truth
    r = float(response)
    return {"reward": r, "format_reward": 1.0, "answer_reward": r}


# =============================================================================================
# (1) group normalization — GRPO (Eq. 28) vs Dr.GRPO (Eq. 31) + the difficulty-bias case
# =============================================================================================


def test_group_norm_matches_oracle_snapshot_both_toggles() -> None:
    responses = ["1.0", "0.0", "0.0", "1.0"]
    gts = ["x"] * 4

    adv_grpo, raw_grpo, meta = compute_group_normalized_rewards(
        _reward_from_response, responses, gts, group_size=2, normalize_by_std=True
    )
    adv_drgrpo, raw_dr, _ = compute_group_normalized_rewards(
        _reward_from_response, responses, gts, group_size=2, normalize_by_std=False
    )

    # GRPO ÷(unbiased std) → ±0.7071 (snapshot-pinned); Dr.GRPO subtract mean only → ±0.5.
    assert adv_grpo.tolist() == pytest.approx([0.70710, -0.70710, -0.70710, 0.70710], abs=1e-4)
    assert adv_drgrpo.tolist() == pytest.approx([0.5, -0.5, -0.5, 0.5], abs=1e-6)
    # raw rewards are returned unchanged; metadata logs the reward distribution.
    assert raw_grpo.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert raw_dr.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert meta["reward_mean"] == pytest.approx(0.5)
    assert meta["reward_max"] == 1.0 and meta["reward_min"] == 0.0


def test_group_norm_difficulty_bias_direction() -> None:
    # Group L (low variance): [0.6, 0.4]; Group H (high variance): [1.0, 0.0]. Same group means'
    # spread differs 5×. Dr.GRPO preserves that spread; GRPO's ÷std erases it → the low-variance
    # group is up-weighted. Indices 0 (L) and 2 (H) are the two "better than group" responses.
    responses = ["0.6", "0.4", "1.0", "0.0"]
    gts = ["x"] * 4
    adv_grpo, _, _ = compute_group_normalized_rewards(
        _reward_from_response, responses, gts, group_size=2, normalize_by_std=True
    )
    adv_drgrpo, _, _ = compute_group_normalized_rewards(
        _reward_from_response, responses, gts, group_size=2, normalize_by_std=False
    )

    # Dr.GRPO: the low-variance group's advantage magnitude is the *raw* edge (0.1) — much smaller
    # than the high-variance group's (0.5). The response is only slightly better than its group.
    assert abs(adv_drgrpo[0]) == pytest.approx(0.1, abs=1e-6)
    assert abs(adv_drgrpo[2]) == pytest.approx(0.5, abs=1e-6)
    assert abs(adv_drgrpo[0]) < abs(adv_drgrpo[2])  # low-variance group NOT up-weighted

    # GRPO: ÷std normalizes both groups to the same magnitude (~0.7071) — the low-variance group
    # has been up-weighted to parity with the high-variance one (the difficulty bias).
    assert abs(adv_grpo[0]) == pytest.approx(0.70710, abs=1e-4)
    assert abs(adv_grpo[2]) == pytest.approx(0.70710, abs=1e-4)

    # The predicted direction: GRPO amplifies the low-variance group's advantage (0.1 → 0.707) and
    # equalizes the L/H ratio, whereas Dr.GRPO keeps it at 0.1/0.5 = 0.2.
    assert abs(adv_grpo[0]) > abs(adv_drgrpo[0])
    ratio_grpo = abs(adv_grpo[0]) / abs(adv_grpo[2])
    ratio_drgrpo = abs(adv_drgrpo[0]) / abs(adv_drgrpo[2])
    assert ratio_grpo > ratio_drgrpo
    assert ratio_grpo == pytest.approx(1.0, abs=1e-3)
    assert ratio_drgrpo == pytest.approx(0.2, abs=1e-3)


# =============================================================================================
# (2) GRPO clip — activates exactly outside 1±ε; the boundary is the off-by-one guard
# =============================================================================================


def test_grpo_clip_activates_exactly_outside_band() -> None:
    cliprange = 0.1  # trust region [0.9, 1.1]
    ratios = torch.tensor([[0.8, 0.9, 1.0, 1.1, 1.2]])  # 0.9 and 1.1 sit ON the boundary
    policy_log_probs = torch.log(ratios)
    old_log_probs = torch.zeros_like(policy_log_probs)
    advantages = torch.ones((1, 1))  # A = +1 for every token

    loss, meta = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    was_clipped = meta["was_clipped"]

    # Only ratios STRICTLY outside [0.9, 1.1] are clipped. Boundary ratios 0.9 and 1.1 are NOT —
    # an off-by-one that used <=/>= (or clamped the band a step early) flips these two and fails.
    assert was_clipped.tolist() == [[True, False, False, False, True]]
    assert meta["clip_fraction"] == pytest.approx(2 / 5)

    # For A > 0 the upper clip caps the surrogate: ratio 1.2 → −clip(1.2)=−1.1; lower side keeps
    # the unclipped value (the PPO min asymmetry): ratio 0.8 → −0.8; in-band → −ratio.
    assert loss[0].tolist() == pytest.approx([-0.8, -0.9, -1.0, -1.1, -1.1], abs=1e-6)


# =============================================================================================
# (3) the dispatcher equals its three parts
# =============================================================================================


def test_policy_gradient_dispatcher_equals_parts() -> None:
    torch.manual_seed(0)
    policy_log_probs = torch.randn(3, 4)
    old_log_probs = torch.randn(3, 4)
    raw_rewards = torch.rand(3, 1)
    advantages = raw_rewards - raw_rewards.mean(0)

    no_base, m0 = compute_policy_gradient_loss(
        policy_log_probs, "no_baseline", raw_rewards=raw_rewards
    )
    assert torch.equal(no_base, compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs))
    assert m0 == {}

    reinforce, m1 = compute_policy_gradient_loss(
        policy_log_probs, "reinforce_with_baseline", advantages=advantages
    )
    assert torch.equal(reinforce, compute_naive_policy_gradient_loss(advantages, policy_log_probs))
    assert m1 == {}

    clip, m2 = compute_policy_gradient_loss(
        policy_log_probs,
        "grpo_clip",
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=0.2,
    )
    ref_loss, ref_meta = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, 0.2)
    assert torch.equal(clip, ref_loss)
    assert torch.equal(m2["was_clipped"], ref_meta["was_clipped"])


def test_dispatcher_rejects_missing_args_and_unknown_type() -> None:
    logp = torch.zeros(2, 3)
    adv = torch.zeros(2, 1)
    with pytest.raises(ValueError):
        compute_policy_gradient_loss(logp, "no_baseline")  # missing raw_rewards
    with pytest.raises(ValueError):
        compute_policy_gradient_loss(logp, "reinforce_with_baseline")  # missing advantages
    with pytest.raises(ValueError):
        compute_policy_gradient_loss(logp, "grpo_clip", advantages=adv)  # missing old/cliprange
    with pytest.raises(ValueError):
        compute_policy_gradient_loss(logp, "bogus")  # type: ignore[arg-type]


# =============================================================================================
# (4) the length-normalization worked example — masked_mean vs masked_normalize gradients differ
# =============================================================================================


def _length_norm_grad(agg: str) -> torch.Tensor:
    """Gradient of the batch-of-2 length-norm example under one aggregation.

    Two responses: row 0 has 1 response token, row 1 has 3. Advantage = +1 everywhere, so the
    per-token loss is −logπ. ``masked_mean`` divides each row by *its own* length (GRPO); a fixed
    ``masked_normalize`` constant (=3) weights every token identically (Dr.GRPO).
    """
    logp = torch.zeros(2, 3, requires_grad=True)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    per_token = compute_naive_policy_gradient_loss(torch.ones(2, 1), logp)  # −logp
    if agg == "mean":
        loss = masked_mean(per_token, mask, dim=-1).mean()
    else:
        loss = masked_normalize(per_token, mask, 3.0, dim=-1).mean()
    loss.backward()
    assert logp.grad is not None
    return logp.grad


def test_length_normalization_changes_gradient() -> None:
    grad_mean = _length_norm_grad("mean")
    grad_norm = _length_norm_grad("constant")

    # masked_mean (GRPO): the short response's token carries 3× the gradient of a long-response
    # token — 1/(row length) weighting. Concrete: −0.5 (len-1 row) vs −1/6 (len-3 row).
    assert grad_mean[0, 0].item() == pytest.approx(-0.5, abs=1e-6)
    assert grad_mean[1, 0].item() == pytest.approx(-1 / 6, abs=1e-6)

    # masked_normalize (Dr.GRPO): every response token gets the SAME weight −1/6 regardless of its
    # sequence length — the length bias is gone.
    assert grad_norm[0, 0].item() == pytest.approx(-1 / 6, abs=1e-6)
    assert grad_norm[1, 0].item() == pytest.approx(-1 / 6, abs=1e-6)

    # The two aggregations therefore produce DIFFERENT gradients — the Dr.GRPO de-biasing argument.
    assert grad_mean[0, 0].item() != pytest.approx(grad_norm[0, 0].item())
    assert abs(grad_mean[0, 0]) > abs(grad_mean[1, 0])  # GRPO up-weights the short response
    assert grad_norm[0, 0].item() == pytest.approx(grad_norm[1, 0].item())  # Dr.GRPO does not


# =============================================================================================
# (5) gradient-accumulation invariance: k microbatches == 1 full batch
# =============================================================================================


@pytest.mark.parametrize("length_normalization", ["mean", "constant"])
def test_grad_accum_invariance(length_normalization: str) -> None:
    torch.manual_seed(0)
    base = torch.randn(4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)
    advantages = torch.randn(4, 1)

    def run(chunks: list[slice], grad_accum: int) -> torch.Tensor:
        w = base.clone().requires_grad_(True)
        for chunk in chunks:
            grpo_microbatch_train_step(
                w[chunk],
                mask[chunk],
                grad_accum,
                "reinforce_with_baseline",
                advantages=advantages[chunk],
                length_normalization=length_normalization,  # type: ignore[arg-type]
                normalize_constant=5.0,
            )
        assert w.grad is not None
        return w.grad

    full = run([slice(0, 4)], grad_accum=1)
    accumulated = run([slice(0, 2), slice(2, 4)], grad_accum=2)
    assert torch.allclose(full, accumulated, atol=1e-6)


# =============================================================================================
# (6) toy end-to-end — grpo_train_loop makes mean reward strictly rise + real-Countdown smoke
# =============================================================================================

_VOCAB = 11
_ANSWER_BASE = 8
_N_TASKS = 3


class SingleTokenCountdownEnv:
    """A learnable single-response-token Countdown-style env: task ``i`` primes a distinct 3-token
    prompt and its verifiably-correct answer is the single token ``_ANSWER_BASE + i``. Small enough
    that a tiny TransformerLM samples correct answers at init (so groups have reward variance), yet
    verifiable (binary exact-match) — the CPU stand-in for the GPU Countdown capstone."""

    def __init__(self) -> None:
        self._tasks = [
            Task(
                task_id=f"ct-{i}", prompt_ids=(1, 2 + i, 2 + i), ground_truth=str(_ANSWER_BASE + i)
            )
            for i in range(_N_TASKS)
        ]

    def decode(self) -> DecodeFn:
        return lambda ids: " ".join(str(int(i)) for i in ids)

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def grade(self, task: Task, rollout: Rollout) -> Graded:
        ok = float(tuple(rollout.response_ids) == (int(task.ground_truth),))
        return Graded(
            reward=ok,
            format_reward=1.0,
            answer_reward=ok,
            response_text=self.decode()(rollout.response_ids),
        )


def _answer_token_of(task: Task) -> int:
    return int(task.ground_truth)


def test_grpo_train_loop_reward_strictly_rises() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=_VOCAB, d_model=32, n_layers=2, n_heads=2, context_length=8)
    model = TransformerLM(cfg)
    env = SingleTokenCountdownEnv()
    optimizer = AdamW(model.parameters(), lr=0.05)
    params = SamplingParams(temperature=1.2, max_tokens=1, seed=None)
    sampler = make_rollout_sampler(model, params)

    reward_before = expected_reward(model, env, _answer_token_of)
    history = grpo_train_loop(
        model,
        env,
        sampler,
        optimizer,
        n_grpo_steps=30,
        group_size=12,
        normalize_by_std=False,
        seed=0,
    )
    reward_after = expected_reward(model, env, _answer_token_of)

    # The exact (noise-free) policy competence rises clearly past its start, then the group-variance
    # signal vanishes as it masters a task and entropy collapses (deterministic at seed 0: 0.065 →
    # 0.333, i.e. it learns 1 of 3 tasks before the plateau). The exact task count / reward ceiling
    # is RNG-stream-sensitive, so assert "learned meaningfully", not a pinned ceiling.
    assert reward_after > reward_before + 0.15  # clear rise (measured +0.27)
    assert reward_after >= 0.3  # mastered at least one task

    # The sampled learning curve rises end-to-end and on a windowed basis (direction, not magnitude).
    assert history[-1].mean_reward > history[0].mean_reward
    first_window = sum(h.mean_reward for h in history[:5]) / 5
    last_window = sum(h.mean_reward for h in history[-5:]) / 5
    assert last_window > first_window

    # Learning sharpens the policy: response-token entropy falls over the run (collapse signal).
    assert history[-1].entropy < history[0].entropy

    # The mandatory RL logs are wired and finite every step (discipline #4).
    for h in history:
        assert isinstance(h.snapshot, monitors.MonitorSnapshot)
        # KL ≥ 0 analytically; allow fp slack — once the policy collapses (entropy → 0) π_old ≈
        # π_current so the k3 estimate is ≈ 0 and rounds microscopically negative (measured −4e-8).
        assert math.isfinite(h.kl_current_ref) and h.kl_current_ref >= -1e-6
        assert math.isfinite(h.kl_current_old) and h.kl_current_old >= -1e-6
        assert 0.0 <= h.is_ratio_ess <= 1.0
        assert math.isfinite(h.is_ratio_mean)
    # At step 0 the pre-update π_old IS π_ref (both the initial policy) → the two KLs coincide.
    assert history[0].kl_current_ref == pytest.approx(history[0].kl_current_old, abs=1e-6)


def test_grpo_train_loop_runs_on_real_countdown_env() -> None:
    """Integration smoke: the loop composes with the real byte-level ``CountdownEnv`` (rollout →
    grade → advantage → update → log) end-to-end. Reward stays ~0 for a random tiny model (real
    Countdown needs a real model + GPU scale, per the capstone runbook) — this asserts the wiring,
    not convergence."""
    from scratch_llm.envs.countdown import CountdownEnv

    env = CountdownEnv(n_tasks=2, seed=0, n_numbers=2, max_value=6)
    max_prompt = max(len(t.prompt_ids) for t in env.tasks())
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=256, d_model=16, n_layers=1, n_heads=2, context_length=max_prompt + 16
    )
    model = TransformerLM(cfg)
    optimizer = AdamW(model.parameters(), lr=0.01)
    params = SamplingParams(temperature=1.0, max_tokens=6, seed=None)
    sampler = make_rollout_sampler(model, params)

    history = grpo_train_loop(
        model, env, sampler, optimizer, n_grpo_steps=2, group_size=2, normalize_by_std=False, seed=0
    )
    assert len(history) == 2
    for h in history:
        assert isinstance(h, GRPOStepMetrics)
        assert math.isfinite(h.mean_reward)
        assert math.isfinite(h.kl_current_ref) and math.isfinite(h.entropy)
    # A fresh 256-vocab head is ~uniform → response-token entropy ≈ log V (loss-at-init sanity).
    assert history[0].entropy == pytest.approx(math.log(256), abs=0.3)
