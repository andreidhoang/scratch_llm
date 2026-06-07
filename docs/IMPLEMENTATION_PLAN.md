# reasoningLLM v0.1.0 → VERA — Master Implementation Plan

> **What this doc is.** The *integration spine* the per-layer build guides cannot provide. The
> guides (`docs/assignment_guides/A*.md`) each design **one** layer in isolation; this plan owns
> what only a whole-stack view can: the **reconciled cross-layer contracts**, the **dependency
> DAG from current repo state**, the **concrete CPU smoke run** that ships v0.1.0, the **CPU/GPU
> resourcing order**, and the **scope tiers**. It does **not** restate the guides — it cites them
> (`A5 §7`, `A3 §6`, …). Per-module derivations live there; the *seams* live here.
>
> **Provenance.** Authored 2026-06-07 from (a) a full read of the five build guides + `STATUS.md`
> + the L1/L2 source, and (b) a 7-package parallel design pass with an adversarial principal
> review. The review caught **five blocker-grade contract divergences** between the
> independently-designed packages; §3 is their reconciliation and is the most load-bearing part
> of this plan. **Scope** (locked, per `CLAUDE.md`): build the engine + CPU smoke run to the
> **Day-14 ship test (v0.1.0)**, then **VERA (v0.1.x)** as the post-sprint finding. MoE×RL and
> multimodal stay **v0.2.0, ADR-stub only** — never in v0.1.0.

---

## 0. The one number every line of this plan serves

```
true_quality_gap = reward − true_quality          (+ hack_rate, kl_train_infer)
```

Owned end-to-end, token → reward. The four metrics the **Day-14 ship test** demands, logged
across **3 of 5 HardeningLevels**: `reward / hack_rate / true_quality_gap / kl_train_infer`. If a
module doesn't help *compute* or *protect* one of these four, it is off the v0.1.0 critical path.

---

## 1. Current state (what is green, what is greenfield)

From `docs/STATUS.md` (76 tests green, ruff/pyright clean, ~12 s CPU suite):

| Layer | Built & green | Greenfield (this plan) |
|---|---|---|
| **L1** Substrate | ✅ tokenizer · model (GQA/RoPE/SwiGLU/RMSNorm) · optim (AdamW+clip+cosine) · train · sampling · seeding · **opt-in MoE** | — (complete) |
| **L2** Systems | ✅ `utils/monitors.py` (three-KL + `kl_train_infer` HALT@0.10) · KV-cache · **rollout seam** (`Rollout`/`RolloutClient` + `LocalBackend`) · FA2 (oracle+Triton, 53 % SDPA) · `kl_train_infer` bridge (4090) | ⬜ DDP/ZeRO-1 (gloo) · 100B one-pager · real SGLang (Hopper) |
| **L5** RLVR engine | — | ⬜ **all of `algos/`, `rewards/`, `envs/`** (the crown) |
| **L4** Data | — | ⬜ `data/{curation,contamination,ablation}.py` |
| **L3** Scaling | — | ⬜ `scaling/hack_rate_fit.py` |
| Ship | — | ⬜ smoke run · VERA |

Every `algos/`, `rewards/`, `envs/`, `scaling/`, `data/` package is a **1-line stub** today. The
DAG below starts from exactly this state — **not** the documented A2-first Days-1–14 schedule
(execution already diverged to L1-first; following the old schedule now is wrong).

---

## 2. The four load-bearing design invariants

1. **Backend-agnostic over `Rollout`.** The L5 engine consumes `Rollout` objects
   (`rollout/types.py`), reward functions, and `utils/monitors.py`. The **same `algos/` code**
   runs on (a) the tiny from-scratch CPU `TransformerLM` via `LocalBackend` (green-CI + smoke
   run) and (b) a real HF model (Qwen2.5-Math-1.5B) on a GPU via an HF/SGLang backend (VERA).
   This is what makes the $0 smoke run a *real* proof of plumbing, not a separate toy. **Every
   contract in §3 is shaped to preserve this.**
2. **The keystone is a pair of leaf modules.** Four of the five blockers (§3) all resolve in one
   place: a single `envs/levels.py` (owns `HardeningLevel`) + `envs/protocol.py` (owns
   `RewardFn`/`DecodeFn`/`Task`/`Graded`/`VerifiableEnv`). Nothing in L5 is reorderable around
   them — they land **first**.
3. **The smoke run is the sequencing engine.** It is Phase-1 "lock the eval" made executable: it
   pulls the *minimum viable slice* of every L5 module and must compute all four metrics on CPU.
   It is the anti-gold-plating guard — if a module isn't needed by the smoke run or VERA, defer it.
4. **Scope tiers are frozen.** v0.1.0 = engine + smoke run (**ships**). VERA = v0.1.x (the
   finding). v0.2.0 = MoE×RL + multimodal (**ADR-stub only**). "Entire spec" = build to the
   Day-14 test, then VERA — *not* build v0.2.0.

---

## 3. The reconciled contract layer (resolves the 5 blockers) — **build these first**

The seven parallel designs diverged on exactly the cross-cutting contracts. These are pinned here
and become **ADR-0009** (the four-metric + ownership ADR), which lands *with* the keystone leaves.

### 3.1 Keystone leaves (zero upstream deps — Tier 0)

**`src/reasoning_llm/envs/levels.py`** — the *single* `HardeningLevel`, one agreed semantics
(resolves Blocker 1: the enum was double-defined in `exploitability.py` *and* `levels.py` with
conflicting level-2/3/4 meanings **and** a real top-level import cycle `reward.py ↔
exploitability.py`):

```python
class HardeningLevel(IntEnum):
    L1_EXTENSIONAL = 1   # answer-set / label match — maximally gameable
    L2_FORMAT      = 2   # requires well-formed <answer>…</answer> + exact-string match
    L3_NORMALIZED  = 3   # normalized-answer match (latex \boxed, whitespace, sign, trailing-zero)
    L4_HARDENED    = 4   # + consistency/distractor checks (unit/sign/range; majority-consistency)
    L5_PERTURBATION = 5  # isomorphic-perturbation hardened (oracle-grade matcher)

SMOKE_LEVELS = (HardeningLevel.L1_EXTENSIONAL, HardeningLevel.L3_NORMALIZED, HardeningLevel.L5_PERTURBATION)
```

> The exact L2–L4 matcher semantics are finalized against the grader in **ADR-0009**; the names +
> ordering + the 3-of-5 `SMOKE_LEVELS` span are pinned now so every package imports one truth.

**`src/reasoning_llm/envs/protocol.py`** — the contracts every reward/env/algo builds into;
imports `HardeningLevel` from `levels` and **nothing from `reward.py`** (this kills the cycle):

```python
RewardDict = dict[str, float]                 # {"reward", "format_reward", "answer_reward"}
class RewardFn(Protocol):
    def __call__(self, *, response_text: str, ground_truth: str) -> RewardDict: ...
class DecodeFn(Protocol):
    def __call__(self, ids: Sequence[int]) -> str: ...
@dataclass(frozen=True)
class Task:   ...   # task_id, prompt_ids, ground_truth, metadata
@dataclass(frozen=True)
class Graded: ...   # reward, format_reward, answer_reward, response_text
class VerifiableEnv(Protocol):
    def decode(self) -> DecodeFn: ...                          # the env OWNS the tokenizer
    def tasks(self) -> Sequence[Task]: ...
    def grade(self, task: Task, rollout: Rollout) -> Graded: ...  # decode → reward_fn under its level
```

### 3.2 The five pinned contracts (the ADR-0009 body)

| # | Blocker the review caught | **Pinned contract** |
|---|---|---|
| B1 | `HardeningLevel` double-defined + `reward↔exploitability` import cycle | §3.1: one `levels.py` + one `protocol.py`; `RewardFn` is a protocol leaf, so `exploitability.py` imports `HardeningLevel`+`RewardFn` from leaves, never from `reward.py`. **Acyclic.** |
| B2 | **four** incompatible `RewardFn` signatures (text vs token-ids) | **text-in, dict-out, keyword-only**: `(*, response_text: str, ground_truth: str) -> RewardDict`. The **env decodes `Rollout.response_ids` once** (`DecodeFn`) and calls the grader on text. A grader must never see token ids (boundaries differ between the from-scratch BPE and an HF tokenizer → not backend-agnostic). |
| B3 | `hack_rate` had two **definitions** | **signal-based**: `hack_rate = (#rewarded rollouts firing ≥1 hack signal) / (#rewarded rollouts)`. Computable from the batch alone; does **not** conflate with `true_quality`. The `true_quality<0.5` numerator is deleted. |
| B4 | `compute_group_normalized_rewards` forked (reward_fn-in vs array-in) | **array-in**: `compute_group_normalized_rewards(raw_rewards, group_size, *, config)`. Advantage **never** touches strings or a `reward_fn` — the *env* grades, the *trainer* passes the raw-reward array down. Preserves backend-agnosticism. |
| B5 | **missing** `Rollout → TokenizedBatch` seam (re-tokenization silently shifts the mask on the HF path) | **add** `tokenized_batch_from_rollouts(rollouts, *, pad_token_id) -> TokenizedBatch` in `algos/sft.py`, building `input_ids/labels/response_mask` directly from `Rollout.prompt_ids + response_ids` — **mask boundary = `len(prompt_ids)`, exact, no re-tokenization**. The string-based `tokenize_prompt_and_output` stays for the SFT warm-start path. |
| — | `true_quality_gap` definition | `true_quality_gap = mean(reward) − mean(true_quality)` **per HardeningLevel**. |
| — | `kl_train_infer` | already wired (`monitors.mean_kl`, HALT@0.10); on CPU the smoke run logs the positive-control value (drift is GPU-measured). |

### 3.3 The three single-owner resolutions (Majors 6–8)

- **One train loop.** `algos/grpo_trainer.py :: grpo_train_loop(...)` is the *only* Alg-3 loop.
  The smoke runner calls it via an `on_snapshot` callback that assembles a `LevelRecord`. The
  duplicate `algos/grpo_loop.py` / `run_grpo` from the smoke-run design is **dropped**.
- **One toy env.** `envs/toy_verifiable.py :: ToyVerifiableEnv` (L5-rewards owner) is the single
  CPU integration vehicle; the smoke run *consumes* it. The smoke-run `ToyProblem` free-function
  duplicate is **dropped**. One oracle, not two.
- **One fitter.** `scaling/hack_rate_fit.py` is L3's sole, **v0.1.0** CPU/numpy deliverable; only
  the GPU *sweep that feeds it points* is v0.1.x. The smoke-run `FitResult` redefinition is
  **dropped**.

---

## 4. The cross-layer DAG + critical path (from current state)

Tags: **[CPU]** green-CI-buildable now · **[GPU]** resource on vast.ai (per `CLAUDE.md` "resource,
never drop") · **★** = on the smoke-run critical path.

```
Tier 0  ★ envs/levels.py · envs/protocol.py                         [CPU]  + commit ADR-0009
            │  (keystone — 4 blockers resolve here; nothing precedes it)
Tier 1  ★ utils/run_record.py (LevelRecord, frozen Day-14 schema)   [CPU]
        ★ rewards/parsing.py                                        [CPU]   ┐ no cross-deps:
          scaling/hack_rate_fit.py (L3, v0.1.0)                     [CPU]   │ build in parallel
          data/types.py · data/ngram.py (L4 primitives)            [CPU]   ┘
Tier 2  ★ rewards/reward.py (imports levels+parsing)                [CPU]
        → ★ envs/exploitability.py (levels+reward+parsing; acyclic) [CPU]
        → ★ rewards/hack_detector.py (signal-based hack_rate)       [CPU]
Tier 3  ★ algos/sft.py (+ tokenized_batch_from_rollouts seam)       [CPU]  ◆ loss-at-init ≈ logV
        → ★ algos/advantage.py (array-in; GRPO↔Dr.GRPO toggle)      [CPU]  ◆ overfit-one-batch
Tier 4  ★ algos/off_policy.py → algos/policy_loss.py                [CPU]  ◆ overfit-one-batch
Tier 5  ★ envs/true_quality.py (oracle, leakage guard)             [CPU]
        ★ envs/toy_verifiable.py (single owner; the CPU vehicle)    [CPU]
Tier 6  ★ algos/grpo_trainer.py (the ONE loop)                      [CPU]  ◆ snapshot every step
        ★ scripts/smoke_run.py + tests/test_smoke_run.py            [CPU]  ◆◆ DAY-14 SHIP GATE
─────────────────────────────  v0.1.0 SHIPS HERE  ─────────────────────────────
Tier 7    TinyZero Countdown-1.5B "aha" repro (~$30)                [GPU]  ◆ GO/NO-GO for VERA
Tier 8    VERA (v0.1.x): math env · contamination gate · 5×3 grid · H2/H3 · plots · paper  [GPU]

Off-critical-path (parallel, do not let these block the spine):
  L4   data/curation.py · data/contamination.py · data/ablation.py  [CPU]  (contamination = a VERA precondition, NOT a smoke-run one)
  L2-finish  utils/distributed.py (DDP/gloo) · utils/zero_optimizer.py (ZeRO-1/gloo)
             · utils/memory_accounting.py · docs/design/L2_distributed_SPEC.md (100B one-pager)  [CPU]
             · bench/sglang_kl_train_infer.py  [GPU/Hopper]   (credentialing, not a ship dep)
```

**Why this order is forced:** the keystone (Tier 0) pins the four contracts that four blockers
need; reward/env/algo code is unbuildable before it. Everything from Tier 1 on is then a clean
topological walk to the smoke run. The contamination gate and L2-finish are genuinely off the
*smoke-run* path (the smoke run uses a toy env with no contamination surface), so they run in
parallel and never gate the ship.

---

## 5. The reconciled module map (post-review signatures)

Citing the build guide for per-module derivation; the **pinned public API** is the contract. All
follow repo conventions (frozen dataclasses, keyword-only required args, numpy for engine-agnostic
math, every docstring names §3-brief / falsifiable-prediction / kill-criterion). `◆` = discipline gate.

### L5 — `algos/` (the RL engine) · guide `A5 §1–§3, §7`

| File | Pinned public API (contract) | Tests / ◆gates | Tag |
|---|---|---|---|
| `algos/sft.py` | `tokenize_prompt_and_output(...)→TokenizedBatch` · **`tokenized_batch_from_rollouts(rollouts,*,pad_token_id)→TokenizedBatch`** (B5) · `compute_entropy(logits)` · `get_response_log_probs(model,input_ids,labels,*,return_token_entropy=False)→dict` (grad-bearing twin of `LocalBackend.score`) · `masked_normalize` · `masked_mean` · `sft_microbatch_train_step(...)` | ◆`sft_loss_at_init≈logV` · ◆`overfit_one_batch` · `entropy_uniform=logV`/`onehot=0` · `logprobs_match_local_score` · `mask_boundary_eq_string_path` | CPU |
| `algos/advantage.py` | **`compute_group_normalized_rewards(raw_rewards,group_size,*,config)`** (B4, array-in) — GRPO Eq28 ↔ Dr.GRPO Eq31 via `normalize_by_std` toggle; RLOO baseline; sign-robust clip | `grpo_vs_drgrpo_differ` · `zero_variance_group_safe` · `seed_repro` | CPU |
| `algos/off_policy.py` | `compute_grpo_clip_loss(...)→(loss,meta{was_clipped})` (Eq33) · `truncated_importance_ratios(...)` (reuses `monitors.importance_ratios`) · `ess_guard(is_ratios,*,warn_fraction)` (reuses `monitors.effective_sample_size`) · `kl_train_infer_guard(...)` (HALT@0.10) | `clip_matches_unclamped_when_in_band` · `near_singular_ratio_truncated` · `ess_collapse_warns` | CPU |
| `algos/policy_loss.py` | `compute_naive_policy_gradient_loss(...)` (Eq32) · `compute_policy_gradient_loss(...,loss_type∈{no_baseline,reinforce_with_baseline,grpo_clip})` · `grpo_microbatch_train_step(...)` | ◆`overfit_one_batch` (the 3 variants converge) · `dispatcher_routes` | CPU |
| `algos/grpo_trainer.py` | `@dataclass GRPOTrainConfig` · **`grpo_train_loop(policy, env, client, *, config, on_snapshot)`** — pulls `Rollout`s, env-grades → raw rewards, advantages, clipped step, **emits a `MonitorSnapshot` every step** (all 12 fields, discipline #4) | `emits_snapshot_per_step` · `halt_on_kl_gt_0.10` · `runs_on_LocalBackend` | CPU |

> **Module-placement note (resolves a guide-vs-reality tension):** `A5 §3` maps
> `compute_entropy`/`get_response_log_probs` → `utils/monitors.py`. That is the *logging* mapping;
> the **grad-bearing** scorer cannot live in `monitors.py` (which is pure-numpy by invariant). It
> lives in `algos/sft.py`. `monitors.py` is untouched.

### L5 — `rewards/` + `envs/` (the differentiator) · guide `A5 §3 net-new rows, §5`

| File | Pinned public API | Tests / ◆gates | Tag |
|---|---|---|---|
| `rewards/parsing.py` | `extract_answer` · `has_well_formed_answer` · `normalize_answer` (latex `\boxed` unwrap, ws/sign/trailing-zero) · `split_think_answer` | `extract_last_block` · `normalize_idempotent` · `boxed_unwrap` | CPU |
| `rewards/reward.py` | `answer_reward(response_text,ground_truth)→float` · `format_reward(response_text)→float` · `make_reward_fn(level,...)→RewardFn` · `r1_zero_reward(...)→RewardDict` (CS336 parity, the L2 default) | `r1_zero_matches_cs336_grader` · `reward_dict_keys` | CPU |
| `envs/exploitability.py` | `matcher_for(level)→Callable[[str,str],float]` · `harden(level, base_reward)→RewardFn` · `@dataclass Perturbation` · `perturb_task(...)` — **the 5-level dial as a controlled independent variable** | `monotone_gameability` (L1 most-gameable→L5 least) · `harden_is_acyclic` | CPU |
| `rewards/hack_detector.py` | `@dataclass HackSignals` · `detect(response_text,ground_truth,*,reasoning_text=None)→HackSignals` · `detect_batch(...)` · **`hack_rate(signals,rewards)→float`** (B3, signal-based) | `verbosity_signal_fires` · `format_gaming_signal` · `hack_rate_denominator_is_rewarded` | CPU |
| `envs/true_quality.py` | `@dataclass OracleSplit` · `class TrueQualityOracle(split, train_task_ids, *, decode)` → `assert_no_leakage()` · `true_quality(backend,params,*,seed)→float` · `true_quality_gap(reward_mean,*,backend,params,seed)→float` | ◆`assert_no_leakage` (held-out ∩ train = ∅) · `memorized_answer_scores_0_on_perturbation` | CPU |
| `envs/toy_verifiable.py` | `class ToyVerifiableEnv(tokenizer, level, *, n_tasks, seed)` (satisfies `VerifiableEnv`) · `oracle_split(*,n_held_out,seed)→OracleSplit` (disjoint by construction) · `build_toy_smoke_env(...)` | `problems_seed_reproducible` · `random_policy_gets_positive_gap` (non-degeneracy) · `test_integration_l5` (end-to-end) | CPU |

### L4 — `data/` (curation + the contamination gate) · guide `A4 §3, §6`

| File | Pinned public API | Tests | Tag |
|---|---|---|---|
| `data/types.py` | `@dataclass RewardRecord` · `EvalItem` · `CurationReport` (`.discard_rate`) | dataclass round-trips | CPU |
| `data/ngram.py` | `normalize_text` · `word_ngrams(text,n,*,normalize)` · `jaccard` · `minhash_signature(items,num_hashes,*,seed)` · `estimate_jaccard` · `lsh_bands` · `collision_probability` | ◆`P[minhash match]≈Jaccard` on a hand-built pair · `lsh_scurve` | CPU |
| `data/curation.py` | `exact_deduplication` · `minhash_deduplication(...)` · `gopher_quality_filter(...)` · `quality_signal(record)` · `curate(...)` | `known_dups_collapse_to_unique_count` | CPU |
| `data/contamination.py` | `build_train_ngram_index(...)` · `detect_contamination(...)→ContaminationReport` (`.is_clean`) · **`assert_eval_set_clean(...)`** (raises `ContaminationError`) — **the hard Phase-3 precondition: no `true_quality_gap` on an unchecked eval set** | `detects_planted_overlap` · `clean_set_passes` | CPU |
| `data/ablation.py` (A4.1) | `@dataclass NoiseConfig` · `inject_noise(...)` · `run_clean_vs_noisy(...)→AblationResult` (`.hack_rate_delta`) | ◆predict-first: `noisy − clean > 0` | CPU |

### L3 — `scaling/hack_rate_fit.py` (the reframed fitter) · guide `A3 §3, §6`

`compute_from_params_tokens(N,D)` / `tokens_from_compute_params` (the `C=6ND` bridge) ·
`@dataclass PowerLaw{a,const}.predict(x)` · `fit_powerlaw(xs,ys)` (log-log linfit) · `r_squared` ·
`isoflop_min(points,*,budget_key,metric_key)` (**generic over y ∈ {loss, hack_rate,
true_quality_gap}**) · `check_complementary_exponents(a,b,tol=0.05)` (the `a+b≈1` sanity check) ·
`h3_verdict(...)→SlopeVerdict` · `class QueryPlanner` (predeclare a `(N,C_infer)` grid, never
exceed budget; `reserve`/`BudgetExceeded`). **Tests** [CPU, pure-numpy]: `recovers_known_slope` ·
`a_plus_b_approx_1` · `slope_by_hardening_level`. Docstring names **VERA axis-3 / H3** (positive
slope at L1–L2, flat/negative at L5; kill = no monotone trend ⇒ axis-3 null, report it). The
Stanford training-API client is **course-only / dropped** — the VERA analog queries our own
rollout engine.

### Ship — `utils/run_record.py` + `scripts/smoke_run.py`

- `utils/run_record.py`: `@dataclass LevelRecord{reward_mean, hack_rate, true_quality_mean,
  true_quality_gap, kl_train_infer, level, n_problems, seed}` with `.halt` (delegates to HALT@0.10)
  · `build_level_record(...)` · `record_to_json` · `write_run_log(records, path)` (JSONL) ·
  `read_run_log(path)`. **This file freezes the Day-14 log schema.**
- `scripts/smoke_run.py`: wires the tiny `TransformerLM` + `ToyVerifiableEnv` through
  `grpo_train_loop` over `SMOKE_LEVELS (L1/L3/L5) × ≤100 problems × 1 seed`, assembling a
  `LevelRecord` per level via the `on_snapshot` callback, writing a JSONL log. `main()` =
  argparse → run → `write_run_log`.

### L2-finish — distributed credentialing · guide `A2 §1 (15–25), §6`

| File | API / deliverable | Tag |
|---|---|---|
| `utils/distributed.py` + `tests/test_ddp.py` | DDP container: naive → flat-bucket all-reduce → overlap (`register_post_accumulate_grad_hook`); **2-rank gloo equivalence test ×5** | CPU (gloo) |
| `utils/zero_optimizer.py` + `tests/test_sharded_optimizer.py` | ZeRO-1 wrapping the existing `AdamW`: each rank owns ~1/world params, broadcast after step; gloo test ×5 | CPU (gloo) |
| `utils/memory_accounting.py` + test | pure-python params+grads+Adam ≈ 16–20 B/param → the numbers behind the one-pager | CPU |
| `docs/design/L2_distributed_SPEC.md` | the **100B memory one-pager** (verbatim "train a 100B model" answer: 1.6 TB state → DP+TP+PP) | docs |
| `bench/sglang_kl_train_infer.py` (+ optional `tests/test_sglang_kl.py`, `gpu`-marked, sm90 guard) | real SGLang on **Hopper**: fp8/INT4-KV drift → `kl_train_infer` toward HALT; honors ADR-0008 | **GPU/Hopper** |

---

## 6. The smoke run — the v0.1.0 ship artifact (the 4-metric ↔ source contract)

The Day-14 test is mechanical: *does it run, is it public, is the smoke run logged with all four
metrics across 3 HardeningLevels?* This table is the contract — **each metric names the exact
module that computes it**, and all are CPU-computable:

| Metric | Computed by | On CPU? |
|---|---|---|
| `reward` | `ToyVerifiableEnv.grade` → `rewards/reward.py` (under the level's matcher) | ✅ |
| `hack_rate` | `rewards/hack_detector.hack_rate(signals, rewards)` (signal-based) | ✅ |
| `true_quality_gap` | `envs/true_quality.TrueQualityOracle.true_quality_gap` (held-out toy oracle) − reward_mean | ✅ |
| `kl_train_infer` | `rollout/local.distribution_logprobs` → `monitors.mean_kl` (already green via `test_rollout.py`); CPU logs the positive-control value | ✅ |

**`tests/test_smoke_run.py` is the green-CI Day-14 gate** — it asserts *plumbing only*: the run
emits one `LevelRecord` per level, all four metrics are finite and in range
(`reward∈[0,1]`, `hack_rate∈[0,1]`, `gap` finite, `kl≥0`), the JSONL reloads with the frozen
schema, and `len(levels)==3`. **No GPU, $0, in the CI suite.** Naming guard: the trivial import
gate is `tests/test_smoke.py` — the RL smoke test is `tests/test_smoke_run.py` (do not collide).

---

## 7. The TinyZero go/no-go gate (Tier 7, ~$30 GPU)

Placed **after the GRPO core lands and before any VERA build-out** (a non-converging RL engine
makes the exploitability dial measure noise). Reproduce the R1-Zero "aha" on **Countdown with
Qwen2.5-1.5B**. **Pass criterion:** emergent self-verification / rising reward curve at 1.5B
(the documented floor — 0.5B fails). **Kill / fallback:** if cross-family RL is too finicky,
collapse VERA to the single-family Qwen-1.5B grid (still publishable). This is the cheapest
validation that the whole backend-agnostic engine works on a *real* model before spending the
VERA budget. vast.ai: a single A100/H100, **destroy (not stop)** the community pod when done.

---

## 8. VERA experiment plan (v0.1.x, post-sprint, GPU)

The first scientific finding *on* the shipped engine — the "replicate-then-break" artifact.

**Two new code seams** (CPU-testable parsing/fit; GPU sweeps):
- `envs/math_verifiable.py` — MATH-500 / AIME held-out env (`MathProblem`, `load_math_problems`,
  `make_math_reward_fn(level)→RewardFn` *same protocol as the toy env*,
  `MathTrueQualityOracle(TrueQualityOracle)` = held-out + isomorphic perturbation). Grader matches
  `drgrpo_grader` on known pairs; **perturbation oracle is leakage-free** (a memorized answer
  scores 0 on the perturbed isomorph).
- the H3 fit reuses `scaling/hack_rate_fit.py` (`fit_hack_rate_vs_compute`,
  `slope_by_hardening_level`) — already built v0.1.0.

**Pre-register H1–H3 before any run** (discipline #5):
- **H1** spurious gains replicate on **Qwen, not Llama/OLMo** — predict random ≈ **+21**, GT ≈
  **+29** on Qwen-Math; **near-zero transfer** to Llama-3.2-1B / OLMo-2-1B. If random helps Llama
  equally, H1 is falsified — *report it* (a well-analyzed negative is the win).
- **H2** `hack_rate` falls **monotonically** as `HardeningLevel` rises.
- **H3** `hack_rate` **grows** with inference compute under a weak verifier; slope **positive at
  L1–L2, flat/negative at L5**.

**The grid:** spurious-rewards battery = 5 reward conditions (ground-truth / format-only / random
/ incorrect-label / majority-vote) × 3 families (Qwen2.5-1.5B / Llama-3.2-1B / OLMo-2-1B);
HardeningLevel sweep (H2); inference-compute sweep (H3, best-of-n **or** CoT length — pin in
ADR-0014).

**Hard precondition (Phase 3, non-negotiable):** `data/contamination.assert_eval_set_clean(...)`
on MATH-500 / AIME held-out **before any `true_quality_gap` number is reported**.

**Budget ~$260–390**, line-itemed: TinyZero Countdown sanity ~$30 → spurious-rewards grid (bulk)
→ cross-family runs → HardeningLevel + inference-compute sweeps. **vast.ai resourcing** (per the
project's GPU-ops notes): community pods, **destroy not stop**; filter *verified + reliable +
price-cap*; SSH-probe readiness; **SGLang needs Hopper (no Ada sm89)** — rent H100 for any
SGLang-served rollouts or the `kl_train_infer`-drift demonstration.

**Output:** the honest `true_quality_gap` across families × hardening levels + the H3 scaling plot
→ blog post + paper draft (the `ml-paper-writing` skill).

---

## 9. The ADR block to write (contiguous, no collisions)

The review caught all six packages claiming ADR-0009. Assigned block (0009 lands *with* the
keystone; the rest as their layer lands):

| ADR | Decision | Default (per guide §8) |
|---|---|---|
| **0009** | four-metric definitions + `HardeningLevel` ownership/semantics (**the keystone**) | §3.2 as pinned |
| **0010** | Dr.GRPO default + the two levers (`normalize_by_std` *and* `masked_mean` vs `masked_normalize`) | Dr.GRPO default, toggle exposed |
| **0011** | train loop in `algos/grpo_trainer.py`, **synchronous-only** for v0.1.0 | one-step staleness + epochs>1 deferred |
| **0012** | `true_quality` leakage guard (held-out + isomorphic perturbation, `assert_no_leakage`) | disjoint-by-construction |
| **0013** | contamination n-gram size for short/templated MATH/AIME | pin n + exact-vs-fuzzy before VERA |
| **0014** | VERA axis-3 `(N, C_infer)` grid + `C_infer` operationalization (best-of-n vs CoT length) | fix grid when budget-bound |
| note | **DPO**: build `dpo_loss` to mastery (interview staple) but **keep it out of the shipped engine** | GRPO-only ships |

**v0.2.0 ADR stubs (do not build):** MoE×RL collapse (GSPO/R3) and multimodal-VLM remain
stub-only under `docs/adr/` (ADR-0005, ADR-0007 already exist) — never smuggled into v0.1.0.

---

## 10. Discipline gates & green-CI (enforced every commit)

Per `CLAUDE.md` + each guide §7. Apply where marked `◆` in §4–§5:
- **loss-at-init ≈ log V** — `algos/sft.py` (fresh head's masked NLL == `cross_entropy` ≈ logV).
- **overfit-one-batch** — `algos/sft.py`, `algos/advantage.py`, `algos/policy_loss.py`.
- **fixed-seed reproducibility** — every stochastic module (reuse `utils/seeding.py`).
- **mandatory three-KL logging** — `grpo_train_loop` emits a full `MonitorSnapshot` every step
  (`KL(cur‖ref)`, `KL(cur‖old)`, **`kl_train_infer`**, IS-ratio histogram, reward/length stats);
  HALT@0.10. Absence = uninterpretable run.
- **contamination check before any eval number** — `data/contamination.assert_eval_set_clean`.
- **predict-before-you-run** — the falsifiable number committed first (H1–H3; the A4.1 delta sign).

**Green-CI** (blocks red commits via `.claude/hooks/green-ci-gate.sh`): `ruff check` +
`ruff format --check` + `pyright` + `pytest -m "not gpu"`. Commit messages: `<area>: <imperative>`.
GPU items are `@pytest.mark.gpu` (skipped in CI), exercised on rented hardware.

---

## 11. Risk register (from the review §issues + guides §8)

| Risk | Severity | Mitigation |
|---|---|---|
| Building reward/env/algo before the keystone → cycle + forked contracts | **blocker** | Tier 0 first, full stop. §3 is the law. |
| Smoke run can't log a metric (esp. missing toy `true_quality` oracle) | **blocker** | §6 table verified CPU-computable; `envs/true_quality.py` is Tier 5, before the runner. |
| HF-path mask shift from re-tokenization | **blocker** | `tokenized_batch_from_rollouts` (B5) — id-based, boundary = `len(prompt_ids)`. |
| GRPO doesn't converge on a real model | high | TinyZero gate (Tier 7) **before** VERA; single-family Qwen fallback. |
| `true_quality` oracle leakage → fake-small gap | high | `assert_no_leakage` gate (ADR-0012); isomorphic perturbation. |
| VERA eval contamination → dishonest number | high | `assert_eval_set_clean` precondition (ADR-0013). |
| L2-finish / contamination block the spine | medium | Both are off the smoke-run critical path (§4); run in parallel. |
| Cross-family RL finicky (Llama/OLMo LR/KL) | medium | ADR fallback to single-family Qwen-1.5B grid (<$120, still the artifact). |
| Scope creep (v0.2.0 into v0.1.0) | medium | §2.4 + §9 stubs; the smoke run is the anti-gold-plating guard. |

---

## 12. Definition of done

- **v0.1.0 (ships):** Tiers 0–6 green; `tests/test_smoke_run.py` passes in CI; `scripts/smoke_run.py`
  writes a JSONL log with `reward / hack_rate / true_quality_gap / kl_train_infer` across L1/L3/L5;
  public repo, green CI; ADRs 0009–0011 logged. **= the Day-14 mechanical test.**
- **v0.1.x (VERA):** TinyZero gate passed; contamination gate clean; H1–H3 pre-registered and run;
  the cross-family × hardening `true_quality_gap` table + H3 plot; blog + paper draft. ADRs 0012–0014.
- **Credentialing (parallel):** DDP/ZeRO-1 gloo tests green; 100B one-pager; SGLang-on-Hopper drift
  measured.

---

## 13. Source-of-truth pointers

- Per-layer *how-to* (the load-bearing 20 %): `docs/assignment_guides/A{1..5}_*_BUILD_GUIDE.md`
  (start at `INDEX.md`).
- Positioning / job-market map: `../UNIFIED_FRONTIER_PROJECT_SPEC.md` (§3 briefs, §5 ledger).
- Capstone scope + VERA (H1–H3, 5 levels, budget): `../CAPSTONE_AND_STUDY_PLAN.md`.
- Build status (single source of truth): `docs/STATUS.md`.
- Constitution + disciplines + frozen scope: `CLAUDE.md`.

> **The plan in one line:** land the two keystone leaves + ADR-0009, walk the CPU DAG to a $0
> green-CI smoke run that logs all four metrics across three HardeningLevels (v0.1.0 ships), gate
> on a $30 TinyZero "aha," then run VERA across three families and publish the honest
> `true_quality_gap`.
