# ADR-0017 — A5 GRPO: Dr.GRPO is the default, and loss aggregation is a pinned hyperparameter

**Status:** accepted · 2026-07-04 (sprint node W8c, ADR-0014)
**Scope:** `src/scratch_llm/algos/grpo.py` — the two "senior" toggles the A5 guide §7 names as ADR
triggers: (1) GRPO (Eq. 28) vs Dr.GRPO (Eq. 31) as the repo default, and (2) `masked_mean` vs
`masked_normalize` for loss aggregation. Both are load-bearing hyperparameters, not details — each
changes what the gradient optimizes.

## Context

GRPO (DeepSeek-Math `2402.03300`) drops the PPO value network and uses the *group* of `G` sampled
responses as a free baseline: `A^(i) = (r^(i) − mean(r)) / (std(r) + ε)` (Eq. 28). Dr.GRPO
(`2503.20783`) shows the two normalizations GRPO stacks on top of REINFORCE each inject a bias:

- **÷ group std (Eq. 28 → Eq. 31).** Dividing the centered reward by the group std rescales *every*
  group to unit spread, so a group with a small reward spread (an easy / nearly-solved question, or
  a hard one the policy fails consistently) is **up-weighted** relative to a high-variance group —
  a **question-difficulty bias**. Dr.GRPO keeps `A^(i) = r^(i) − mean(r)` and lets the raw edge
  stand.
- **per-response length normalization (`masked_mean`).** Averaging the per-token loss over *each
  response's own length* (÷|o_i|) means a token in a long response carries a smaller gradient than
  a token in a short one. For negative-advantage (wrong) responses this *under-penalizes* long
  wrong answers → the policy is pushed to make wrong answers longer — the observed **length
  inflation**. Dr.GRPO divides by a **fixed constant** instead, so every response token carries the
  same weight regardless of sequence length.

Both are exposed as first-class toggles in `grpo.py`: `normalize_by_std` and `length_normalization`
(`"mean"` = GRPO per-sequence mean, `"constant"` = Dr.GRPO fixed divisor).

## Decision

1. **Dr.GRPO is the repo default for the *loop*.** `grpo_train_loop(..., normalize_by_std=False,
   length_normalization="constant")`. Both bias corrections are on by default; GRPO is a one-flag
   opt-in (`normalize_by_std=True`, `length_normalization="mean"`). The R1-Zero / Countdown capstone
   and every A5 run start from the de-biased objective.

2. **The primitive `compute_group_normalized_rewards` keeps `normalize_by_std=True` as its *named*
   default.** This is deliberate and narrow: that default makes the standalone function reproduce
   the **Eq. 28 (GRPO) adapter snapshot** — raw `[1,0,0,1]`, `group_size=2` → `±0.7071` (unbiased
   std, ddof=1) — so the W9 acceptance wiring for `test_compute_group_normalized_rewards` is exact.
   The *policy* (Dr.GRPO) is expressed where policy belongs — the loop's default — not by flipping a
   primitive's default and desyncing it from the oracle.

3. **Aggregation is pinned, mean-over-sequences at the top level.** Both modes reduce a per-token
   loss to a **per-sequence** scalar first (`masked_mean(dim=-1)` or `masked_normalize(dim=-1, C)`),
   then take the **mean over sequences**. This is required for **gradient-accumulation invariance**:
   only a top-level mean-over-sequences makes k equal microbatches scaled by `1/k` reproduce the
   full-batch gradient. (The `dim=None` global `masked_normalize` in `sft.aggregate_loss_across_
   microbatch` is *not* accum-invariant under `/k` scaling and is therefore not used inside the
   GRPO microbatch step.)

4. **`std` uses the unbiased (ddof=1) estimator**, matching the official A5 snapshot (`0.7071`, not
   the population `0.5`). Pinned in the module docstring and `tests/test_grpo_algos.py`.

## Justification

**Predict-before-run (the difficulty bias, stated before the assert).** Two groups, L = `[0.6,0.4]`
(low variance) and H = `[1.0,0.0]` (high variance). Dr.GRPO advantages: L's better response gets
`0.1`, H's gets `0.5` — the raw edge is preserved, so `|A_L| < |A_H|`. GRPO ÷std: **both** collapse
to `≈0.7071` — the low-variance group has been up-weighted to parity. The predicted direction
(GRPO up-weights the low-variance group, Dr.GRPO does not) is asserted in
`test_group_norm_difficulty_bias_direction` (ratio `|A_L|/|A_H|`: GRPO `≈1.0` vs Dr.GRPO `0.2`).

**Predict-before-run (the length bias).** Batch of two responses, lengths 1 and 3, advantage `+1`
everywhere. Under `masked_mean` the len-1 token gets gradient `−0.5` and the len-3 token `−1/6`:
tokens in the longer response are systematically down-weighted (1/length). Under `masked_normalize`
(fixed `C`) both get `−1/6`: length-independent. The two concrete gradients are pinned in
`test_length_normalization_changes_gradient` — the Dr.GRPO de-biasing argument written out in code.

**Engine validation (predict → measured).** `grpo_train_loop` with the Dr.GRPO defaults on a
learnable single-token Countdown-style env + a tiny A1 `TransformerLM` and a temperature sampler
makes the policy's exact expected reward rise **0.19 → 0.67** (sampled mean reward `0.19 → 0.67`)
over 30 CPU steps, deterministic at seed 0 (`tests/test_grpo_algos.py::
test_grpo_train_loop_reward_strictly_rises`; ledgered in `bench/RESULTS.md`). It proves the loop
composes (rollout → grade → advantage → update) and learns — nothing about a real model, per the
capstone framing.

## Consequences

- **The two other §7 triggers are resolved inline, not deferred.** *Off-policy epochs>1:*
  in-scope but gated — `epochs_per_rollout_batch>1` auto-switches to `grpo_clip` and caches the
  rollout-time log-probs as `π_old` (the clip is the trust region that makes multi-epoch safe);
  the default loop is on-policy single-epoch (`reinforce_with_baseline`, no clip). *Env backend
  scope:* confirmed — the CPU `LocalBackend` is sufficient to test `grpo_train_loop` end-to-end;
  it plays both the train- and the (stand-in) infer-engine role, so `kl_train_infer == 0` by
  construction until a distinct serving engine (SGLang) is wired in, at which point the channel
  (already logged) becomes load-bearing.
- **Mandatory RL logging is wired into the loop, not bolted on** (discipline #4): every
  `GRPOStepMetrics` carries entropy, `KL(cur‖ref)`, `KL(cur‖old)`, IS-ratio mean+ESS,
  reward-distribution stats, and length-by-correctness, all via `utils/monitors.py`.
- **Signature divergence from the on-disk oracle is a known, deliberate gap.** The
  `lectures/assignment5-alignment/tests/adapters.py` checked out on this box is a *newer* A5
  revision (`run_compute_group_normalized_rewards(raw_rewards, …, baseline, advantage_normalizer)`,
  `run_compute_policy_gradient_loss(…, importance_reweighting_method)`,
  `run_aggregate_loss_across_microbatch`, `run_grpo_train_step`) than the CS336 A5 build guide and
  this node's contract (the classic `compute_group_normalized_rewards(reward_fn, …, normalize_by_std)`
  / `compute_grpo_clip_loss` / `no_baseline|reinforce_with_baseline|grpo_clip` family). `grpo.py`
  implements the classic family the guide + DoD specify; the **numerics are matched to the oracle
  snapshots** (unbiased std `0.7071`, `−min(ρA, clip(ρ)A)`), so W9 wiring is thin adapters over
  matching semantics, not a re-derivation. Recorded so W9 doesn't mistake the gap for a bug.
