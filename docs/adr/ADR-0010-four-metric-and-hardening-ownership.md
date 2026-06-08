# ADR-0010 — Four-metric definitions + HardeningLevel/RewardFn ownership (the L5 keystone)

- **Status:** Accepted (2026-06-08)
- **Layer:** L5 RLVR engine — the keystone leaf modules (`envs/levels.py`, `envs/protocol.py`)
- **Decides:** the cross-cutting contracts every L5 reward/env/algo module builds into

## Context

A parallel design pass over the L5 packages, with an adversarial review, surfaced five
blocker-grade contract divergences (`docs/IMPLEMENTATION_PLAN.md` §3): `HardeningLevel` was defined
twice with a real `reward.py ↔ exploitability.py` import cycle; `RewardFn` had four incompatible
signatures (text vs token-ids); `hack_rate` had two definitions; the group-advantage fn forked
(reward_fn-in vs array-in); and the `Rollout → TokenizedBatch` seam was missing. These cannot be
left to whichever module lands first — they must be pinned before any L5 code.

## Decision

Two leaf modules own the contracts; everything else imports from them, making the L5 graph acyclic.

1. **`envs/levels.py` owns one `HardeningLevel`** (`L1_EXTENSIONAL` … `L5_PERTURBATION`) and
   `SMOKE_LEVELS = (L1, L3, L5)`. It imports nothing. (Exact L2–L4 matcher semantics are finalized
   when `exploitability.py` lands; names + ordering are pinned now.)
2. **`envs/protocol.py` owns `RewardFn` (text-in / dict-out, keyword-only), `DecodeFn`, `Task`,
   `Graded`, `VerifiableEnv`.** It imports only `levels` (and `Rollout` under `TYPE_CHECKING`). The
   env decodes `Rollout.response_ids` once and grades on **text** — a grader never sees token ids
   (BPE vs HF boundaries differ → token-id graders are not backend-agnostic).
3. **`hack_rate` is signal-based:** `(# rewarded rollouts firing ≥1 hack signal) / (# rewarded
   rollouts)` — computable from the batch alone, not conflated with `true_quality`.
4. **Group-advantage is array-in:** `compute_group_normalized_rewards(raw_rewards, group_size, *,
   config)` — the env grades, the trainer passes a raw-reward array; advantage carries no strings.
5. **`true_quality_gap = mean(reward) − mean(true_quality)` per HardeningLevel**; `kl_train_infer`
   stays `monitors.mean_kl`, HALT @ 0.10.

## Consequences

- (+) All four through-line metrics have one unambiguous definition and home; the smoke run can
  compute them from these modules.
- (+) The import cycle is structurally impossible (the leaves depend on nothing / only on `levels`).
- (+) The engine is backend-agnostic: text-in grading + array-in advantage means the same code runs
  on the CPU `LocalBackend` and a GPU HF/SGLang backend.
- (−) A small `tokenized_batch_from_rollouts` seam (id-based, no re-tokenization) must be added in
  `algos/sft.py` (the fifth blocker) — tracked in the plan, not this ADR.
