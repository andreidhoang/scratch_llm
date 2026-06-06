# A5 — Alignment · BUILD GUIDE (CS336 → reasoningLLM L5 RLVR Engine)

> **One-liner.** A5 is the **crown layer (L5)** — the only assignment where you build the RLVR engine itself (SFT → Expert Iteration → GRPO/Dr.GRPO, plus the optional supplement's DPO/safety). It is the scarcest, highest-value 2026 skill cluster (RLVR × reward modeling × verifier-exploitability). **Source files it feeds:** `algos/advantage.py` (RLOO + sign-robust clip), `algos/off_policy.py` (truncated-IS + ESS), `rewards/reward.py` (composition under the 5 `HardeningLevel`s), `rewards/hack_detector.py`, `envs/exploitability.py` (the 5-level dial), `envs/true_quality.py` (held-out + isomorphic-perturbation oracle), `rollout/async_orchestrator.py`. **Through-line tie:** L5 *is* the engine that measures `true_quality_gap = reward − true_quality`, plus `hack_rate` and `kl_train_infer`. CS336 hands you the *optimizer* (GRPO); the add-ons (A5.2) hand you the *honest measurement* CS336 never asks for — that delta is the whole reason `reasoningLLM` exists.

---

## 1. What CS336 actually requires (every deliverable — main + supplement)

A5 main (`Version 1.0.2`) covers MATH reasoning RL: zero-shot baseline → SFT → Expert Iteration → GRPO. Only `tests/test_sft.py` and `tests/test_grpo.py` are mandatory; EI/baseline are non-mandatory. The supplement (`Version 1.0.1`, **entirely optional**) covers generalist instruction-tuning + DPO + safety/red-teaming on Llama 3.1 8B.

**Priority legend:** `[LB]` LOAD-BEARING (engine path), `[CR]` COURSE-ROTE (build once, won't ship), `[SKIP]` (informs nothing on the engine path).

### Main assignment — Reasoning RL on Qwen 2.5 Math 1.5B

| Deliverable (verbatim Problem name) | PDF section | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|
| `math_baseline` — zero-shot MATH eval; `evaluate_vllm(model, reward_fn, prompts, params)`; categorize format×answer reward; commentary | main §3.2 (pp. 5–6) | `drgrpo_grader.r1_zero_reward_fn` (provided); no test | 2–3 h | **[CR]** (the r1_zero prompt + grader pattern is reused everywhere; the *eval-categorization* commentary is rote) |
| `tokenize_prompt_and_output` — tokenize Q + O separately, concat, build `response_mask` (1 on response, 0 on prompt/pad) | main §4.2 (p. 9) | `run_tokenize_prompt_and_output` / `test_tokenize_prompt_and_output` | 1.5 h | **[LB]** masking primitive — every per-token loss depends on it |
| `compute_entropy` — per-token next-token entropy, numerically stable (logsumexp) | main §4.2 (p. 10), Eq. 1 | `run_compute_entropy` / `test_compute_entropy` | 0.5 h | **[LB]** mandatory RL log (entropy collapse signal) |
| `get_response_log_probs` — per-token `log p_θ(x_t \| x_<t)`, optional token-entropy | main §4.2 (pp. 10–11), Eq. 2 | `run_get_response_log_probs` / `test_get_response_log_probs` | 1 h | **[LB]** the scoring primitive (SFT + RL + DPO all call it) |
| `masked_normalize` — masked sum / constant, optional `dim` | main §4.2 (pp. 11–12) | `run_masked_normalize` / `test_masked_normalize` | 0.5 h | **[LB]** the length-normalization lever (Dr.GRPO vs GRPO hinges on this) |
| `sft_microbatch_train_step` — masked NLL, grad-accum scaling, returns `(loss, metadata)` | main §4.2 (pp. 12–13) | `run_sft_microbatch_train_step` / `test_sft_microbatch_train_step` | 1.5 h | **[LB]** the SFT update; reused as the warm-start path |
| `log_generations` — log prompt, response, gt, reward(format/answer/total), avg entropy, avg/length stats by correctness | main §4.2 (p. 13) | no test | 1 h | **[LB]** length-vs-correctness logging is the verbosity-hack detector |
| `sft_experiment` — run SFT on `sft.jsonl`, sweep {128,256,512,1024,full}; ≥15% val; filtered-correct variant | main §4.3 (pp. 13–15) | run only (2 H100 hrs) | GPU | **[CR]** (the data-size ablation is course-rote; the *filtered-correct* idea is the EI seed) |
| `expert_iteration_experiment` — EI (Alg. 2): sample G, keep correct, SFT, repeat `n_ei_steps=5`; sweep G∈{4,8,16}, batch∈{512,1024,2048} | main §5 (pp. 15–16), Alg. 2 | run only (6 H100 hrs) | GPU | **[CR]** (STaR/EI is the cheapest RL-without-gradients baseline; build to understand, won't ship as a core algo) |
| `compute_group_normalized_rewards` — raw rewards, group-normalize (Eq. 28 GRPO **or** Eq. 31 Dr.GRPO), return adv/raw/metadata | main §7.2 (pp. 22–23), Eq. 28/31 | `run_compute_group_normalized_rewards` / `test_compute_group_normalized_rewards` | 1.5 h | **[LB]** ← `algos/advantage.py` core |
| `compute_naive_policy_gradient_loss` — `−A_t · log p_θ(o_t)`, broadcast adv over seq | main §7.2 (p. 23), Eq. 32 | `run_compute_naive_policy_gradient_loss` / `test_…` | 0.5 h | **[LB]** REINFORCE term |
| `compute_grpo_clip_loss` — Eq. 33 per-token clip, return `(loss, metadata{was_clipped})` | main §7.2 (p. 24), Eq. 33 | `run_compute_grpo_clip_loss` / `test_compute_grpo_clip_loss` | 1.5 h | **[LB]** ← off-policy stability; the clip *is* the PPO trust region |
| `compute_policy_gradient_loss` — wrapper dispatching `no_baseline` / `reinforce_with_baseline` / `grpo_clip` | main §7.2 (pp. 24–25) | `run_compute_policy_gradient_loss` / `test_…` | 0.5 h | **[LB]** the ablation switch (the three loss variants are the baseline experiment) |
| `masked_mean` — masked mean over `dim` | main §7.2 (p. 26) | `run_masked_mean` / `test_masked_mean` | 0.5 h | **[LB]** the per-sequence aggregator (contrast with `masked_normalize`) |
| `grpo_microbatch_train_step` — per-token loss → `masked_mean` → grad-accum → backward | main §7.2 (pp. 26–27) | `run_grpo_microbatch_train_step` / `test_…` | 1.5 h | **[LB]** the GRPO update step |
| `grpo_train_loop` — full Alg. 3 loop; on-policy defaults (`n_grpo_steps=200`, `rollout_batch=256`, `group_size=8`); ≥sensible val rewards | main §7.2 (pp. 27–29), Alg. 3 + Eq. 29 | run only (5 pts) | GPU | **[LB]** ← `rollout/async_orchestrator.py` + the whole engine integration |
| `grpo_learning_rate` / `grpo_baselines` — LR sweep; baseline vs no-baseline; ≥25% MATH | main §8 (p. 29) | run only | GPU | **[LB]** the baselining ablation is the first real `reward` curve |
| Length-normalization study (`masked_mean` vs `masked_normalize`, batch-size-2 worked example) | main §8 (p. 30) | conceptual | 0.5 h | **[LB]** this is the Dr.GRPO de-biasing argument in code |

### Supplement — Instruction Tuning + RLHF (DPO) on Llama 3.1 8B (all optional)

| Deliverable (verbatim Problem name) | PDF section | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|
| `mmlu_baseline` (parse + eval), `gsm8k_baseline`, `alpaca_eval_baseline`, `sst_baseline` (SimpleSafetyTests) | supp §2.1–2.4 (pp. 3–7) | `run_parse_mmlu_response`, `run_parse_gsm8k_response`; rest run-only | 3–4 h | **[CR]** generalist-eval harness; useful pattern, not engine path |
| `look_at_sft` — inspect 10 instruction-tuning examples | supp §3.1 (p. 8) | no test | 0.5 h | **[SKIP]** |
| `data_loading` — packed SFT `Dataset` + `iterate_batches` (Alpaca template, packed sequences) | supp §3.2.1 (pp. 9–10) | `get_packed_sft_dataset`, `run_iterate_batches` / `test_packed_sft_dataset`, `test_iterate_batches` | 2 h | **[CR]** (packing is a real systems skill; lives in A2/data path, not L5) |
| `sft_script` + `sft` — instruction-tune Llama 3.1 8B, 1 epoch, ctx 512, lr 2e-5 | supp §3.2.2 (pp. 12) | run only (24 H100 hrs) | GPU | **[CR]** vanilla SFT — commoditized |
| `mmlu_sft`/`gsm8k_sft`/`alpaca_eval_sft`/`sst_sft` — re-eval the tuned model | supp §4.1–4.4 (pp. 13–15) | run only | GPU | **[SKIP]** (eval-delta reporting; not engine) |
| `red_teaming` — 3 misuse vectors; try to break the tuned model | supp §4.5 (p. 15) | no test | 1 h | **[CR]** (conceptually feeds `hack_detector.py` intuition; keep the *catalog*, skip the writeup) |
| `dpo_loss` — per-instance DPO loss (Eq. 3), `π_θ` vs `π_ref`, Alpaca template, EOS appended | supp §5.3 (p. 18), Eq. 3 | `run_per_instance_dpo_loss` / `test_per_instance_dpo_loss` | 1.5 h | **[CR→decide]** (clean preference-optimization primitive; see §8 ADR — keep iff DPO stays in v0.1.0) |
| `dpo_training` — DPO loop on Anthropic HH, β=0.1, lr 1e-6, RMSprop, track classification accuracy | supp §5.4 (pp. 18–19) | run only (4 pts) | GPU | **[SKIP]** for v0.1.0 (HH preference RLHF is off the verifiable-reward thesis) |
| `look_at_hh` — load + inspect Anthropic HH (chosen/rejected) | supp §5.2 (p. 17) | no test | 0.5 h | **[SKIP]** |

---

## 2. The equations/algorithms that matter (senior extraction)

Seven load-bearing pieces. Everything else in A5 is plumbing around these.

1. **SFT masked cross-entropy** — `L_SFT = − Σ_t mask_t · log p_θ(o_t | q, o_<t) / Z`. Only response tokens contribute (the `response_mask`); prompt + padding are zeroed. *Why it matters:* it is the warm-start and the *exact same per-token log-prob machinery* GRPO reuses — get the masking wrong here and every downstream RL number is corrupt. The `normalize_constant` / `dim` choice is not cosmetic: it is the length-normalization lever that distinguishes GRPO from Dr.GRPO.

2. **GRPO group-relative advantage (Eq. 28)** — `A^(i) = (r^(i) − mean(r^(1..G))) / (std(r^(1..G)) + ε)`. Sample G outputs per question, use the *group* as the baseline — no learned value network. Constant across tokens of a response (`A_t^(i) = A^(i)`). *Why it matters:* this is the entire reason GRPO is cheap and stable on verifiable rewards — the variance-reduction baseline is free. It is the heart of `algos/advantage.py`.

3. **Dr.GRPO de-biasing (Eq. 31)** — `A^(i) = r^(i) − mean(r^(1..G))` (drop the `/std`). *Why it matters:* dividing by per-group std up-weights low-variance (easy or already-solved) questions — a **question-difficulty bias**; combined with response-length normalization it inflates output length. Dr.GRPO removes both. This is `Spurious_Rewards`/`understand-r1-zero`'s correction and the *single most important "senior" toggle* in the file — VERA's H1 depends on running both and reporting the gap.

4. **Importance-sampling ratio + GRPO-Clip (Eq. 27, 29, 33)** — off-policy gradient reweights by `ρ_t = π_θ(o_t) / π_θ_old(o_t)`; the per-token objective is `min(ρ_t A, clip(ρ_t, 1−ε, 1+ε) A)`. *Why it matters:* the clip is the PPO trust region — it lets you take multiple gradient steps per rollout batch (epochs > 1, off-policy) without the policy running away from the rollout distribution. Naive IS ratios are catastrophic for MoE (R3: ~94% of tokens select divergent experts) — this is where `algos/off_policy.py`'s **truncated-IS + ESS monitor** earns its keep and where `kl_train_infer` is measured.

5. **REINFORCE / policy-gradient loss (Eq. 20, 32)** — `∇J = E[Σ_t ∇log π_θ(a_t|s_t) R(τ)]`; per-token loss `−A_t · log p_θ(o_t)`. *Why it matters:* the irreducible core — every variant (no-baseline, reinforce+baseline, grpo-clip) is this with a different `A_t` and an optional clip. `compute_policy_gradient_loss` is the ablation dispatcher.

6. **DPO / Bradley-Terry loss (supp Eq. 1, 3)** — reward model: `ℓ_RM = −log σ(r(x,y_w) − r(x,y_l))` (Bradley-Terry pairwise); DPO collapses the reward model into the policy: `ℓ_DPO = −log σ(β log [π_θ(y_w)/π_ref(y_w)] − β log [π_θ(y_l)/π_ref(y_l)])`. *Why it matters:* DPO needs no sampling and no reward model — just conditional log-probs under `π_θ` and `π_ref`. It is the *preference*-based alternative to verifiable-reward RL; conceptually important (interview-load-bearing for "explain RLHF vs DPO"), but **off** the verifiable-reward engine thesis (see §8).

7. **The verifiable-reward grader** — `r_T = 1 if normalize(answer) == normalize(gt) else 0`, with `format_reward` (well-formed `<answer>…</answer>`) factored separately. *Why it matters:* this binary, rule-based reward is what makes the whole loop *verifiable* (no human, no RM) — and it is exactly the object the add-ons attack. The `reward_fn` returning `{"reward", "format_reward", "answer_reward"}` is the seam where `rewards/reward.py` composition and `envs/exploitability.py` hardening plug in.

---

## 3. Map to reasoningLLM_scratch source files

| CS336 deliverable | → `src/reasoning_llm/…` | keep / adapt / course-only |
|---|---|---|
| `compute_group_normalized_rewards` (Eq. 28 + Dr.GRPO Eq. 31) | **`algos/advantage.py`** — RLOO/group baseline + sign-robust clip; expose `normalize_by_std` toggle = GRPO↔Dr.GRPO | **keep verbatim as the core**; CS336 gives you both variants — wire both |
| `compute_grpo_clip_loss` + IS ratio (Eq. 27, 33) | **`algos/off_policy.py`** — truncated-IS + **ESS monitor**; add the edge tests (zero-variance group, near-singular ratios) named in STUDY_PLAN Day 7 | **adapt**: CS336 stops at plain clip; the add-on adds truncation + ESS + the `kl_train_infer` hook |
| `reward_fn` / `r1_zero_reward_fn` (format + answer composition) | **`rewards/reward.py`** — compose `format_reward` × `answer_reward` under each of the 5 `HardeningLevel`s | **adapt**: CS336 has *one* grader; you parameterize it over the dial |
| `log_generations` length-by-correctness + `red_teaming` catalog | **`rewards/hack_detector.py`** — verbosity / format-gaming / answer-leak detectors | **adapt**: CS336's logging fields *are* the detector signals; productionize them |
| (no CS336 equivalent — the gap A5.2 fills) | **`envs/exploitability.py`** — the 5 `HardeningLevel`s (extensional/answer-match → format-gameable → perturbation-hardened) | **net-new** — this is the differentiator CS336 never asks for |
| (no CS336 equivalent) | **`envs/true_quality.py`** — held-out + isomorphic-perturbation oracle the policy never trains on | **net-new** — the honest denominator of `true_quality_gap` |
| `compute_entropy`, `get_response_log_probs`, three-KL | **`utils/monitors.py`** (L2) — entropy, IS-ratio histogram, the three KLs incl. `kl_train_infer` HALT@0.10 | **keep**: CS336's `compute_entropy` + logging spec is exactly the mandatory-log discipline |
| `grpo_train_loop` (Alg. 3) + on-policy/off-policy epochs | **`rollout/async_orchestrator.py`** (L2/L5) — experience buffer, one-step staleness | **adapt**: CS336's loop is synchronous; the ship target adds async staleness |
| `sft_microbatch_train_step`, `tokenize_prompt_and_output`, `masked_normalize`, `masked_mean` | shared primitives under `algos/` + `utils/` (the SFT warm-start path) | **keep** as primitives |
| `dpo_loss` (supp Eq. 3) | **decision-gated** — only enters the repo if DPO stays in v0.1.0 (§8) | **course-only by default** |
| All supplement evals (MMLU/GSM8K/AlpacaEval/SST) + packed `Dataset` + Llama SFT/DPO training | — | **course-only / skip** — generalist-chat path, off the L5 thesis |

---

## 4. Map to core context docs

- **`UNIFIED_FRONTIER_PROJECT_SPEC.md §3 · L5·A5 brief:** the three add-ons that turn CS336-A5 into the crown. **A5.1** = your own GRPO + Dr.GRPO (`algos/advantage.py` RLOO+sign-robust clip, `algos/off_policy.py` truncated-IS+ESS) + the **TinyZero Countdown-1.5B "aha" repro** (~$30; emerges at 1.5B, fails at 0.5B). **A5.2** = the verifier-exploitability dial (`envs/exploitability.py`, 5 levels) + the true-quality oracle (`envs/true_quality.py`) + the **spurious-rewards battery** (ground-truth / format-only / random / incorrect-label / majority-vote) across Qwen2.5-1.5B / Llama-3.2-1B / OLMo-2-1B — *"the differentiator that fills CS336's gap."* **A5.3** = MoE×RL collapse (`kl_train_infer` under GRPO, fixed by GSPO/R3) — **v0.2.0-B, do not pull into v0.1.0.**
- **`UNIFIED_FRONTIER_PROJECT_SPEC.md §5** (evidence ledger): the evidence ledger — VERA is the canonical *"replicate-then-break a paper"* artifact (Neel Nanda's endorsed on-ramp); well-analyzed negatives ("random reward did *not* transfer to Llama — here's the family-dependence") are valued *above* weak positives. A5 is what carries the **Post-Training RE** role (Anthropic *"novel training environments for RL agents,"* OpenAI *"reward models, graders, evals"*).
- **`CAPSTONE_AND_STUDY_PLAN.md §2** (VERA): the engine reuses A5's `algos/`, `rewards/`, `envs/exploitability.py`, `envs/true_quality.py`, `utils/monitors.py` wholesale; VERA adds only `envs/math_verifiable.py`. **The 5 `HardeningLevel`s** (extensional/answer-match → format-gameable → perturbation-hardened/isomorphic). **Pre-registered H1–H3** (§2.4): H1 spurious gains replicate on Qwen, *not* Llama/OLMo; H2 `hack_rate` falls monotonically as hardening rises; H3 `hack_rate` grows with inference compute under a weak verifier. **Spurious-rewards grid** = the 5 reward conditions × 3 families. **Budget** ~$260–390 (Countdown sanity ~$30 → grid → cross-family → hardening sweeps). **Floor: 1.5B** (0.5B fails the TinyZero learning-signal test).
- **`CAPSTONE_AND_STUDY_PLAN.md §3 · row A5**: "the RLVR engine itself" → `algos/advantage.py` (RLOO+sign-robust clip), `algos/off_policy.py` (truncated-IS+ESS), `rewards/reward.py` (composition under all 5 levels), `rewards/hack_detector.py`; lecture backing **L15 (SFT/RLHF), L16 (RLVR), L17 (RL systems)**.
- **`STUDY_PLAN_2026.md §0.8** (Days 5–8, ship targets unchanged): **Day 5** A5 GRPO objective + **L16 RLVR** → `rewards/reward.py`; **Day 6** A5 advantage (RLOO/Dr.GRPO) → `algos/advantage.py`; **Day 7** A5 off-policy IS + ESS + **L17** → `algos/off_policy.py`; **Day 8** A5 SFT loop + **L15** → `rollout/async_orchestrator.py`. G2 (EOD 6) and G6 ship-cadence (EOD 7) both bite during A5 — keep the daily green commit.
- **repo `CLAUDE.md` L5 row + discipline #4**: L5 = `algos/`, `rewards/`, `envs/exploitability.py`, `envs/true_quality.py`. **Discipline #4 (mandatory, absence = uninterpretable run):** log the **three KLs separately** — `KL(current‖ref)`, `KL(current‖old)`, **`kl_train_infer`** — plus IS-ratio histograms, reward-distribution stats, and length stats; `kl_train_infer` HALT@**0.10**. CS336's `compute_entropy` + `log_generations` spec is the minimum; the repo discipline extends it.

---

## 5. The frontier 2026 lens

**Commoditized (do not lead with these):** vanilla SFT (supplement's `sft`), masked CE, gradient accumulation, packed datasets, generalist evals. These are table-stakes — `torch.compile`-era table-stakes — and every candidate has them. The supplement is almost entirely here.

**Scarce (the low-supply, high-demand corner):**
- **GRPO stability** — making RL not diverge is rare. The progression to internalize: **GRPO** (DeepSeek-R1, `2501.12948`, RL stage ~$294K, displaced PPO+DPO) → **Dr.GRPO** (`2503.20783`, removes the std-normalization difficulty-bias and length inflation) → **GSPO** (`2507.18071`, sequence-level ratios that fix MoE-RL collapse). Knowing *why* each step exists is the senior signal.
- **Non-gameable verifiers** — *"almost nobody can build a verifier a strong policy won't game."* This is `envs/exploitability.py` + `envs/true_quality.py`. The 2026 named failure mode: **"LLMs Gaming Verifiers"** (`2604.15149` ⚠) — RLVR models pass *extensional* verifiers by enumerating instance labels, and this **grows with inference compute**; isomorphic-perturbation testing detects it (→ HardeningLevel 5).
- **The spurious-rewards phenomenon** (`2506.10947`) — the field-shaking result A5.2 red-teams. **Stated precisely:** on Qwen2.5-Math, *random* reward gives **+21.4** pts vs *ground-truth* **+29.1** pts on MATH-500; format-only and *incorrect-label* rewards also give large gains. But this **fails to transfer to Llama-3 / OLMo-2** — it is partly GRPO amplifying a pretrained prior ("code reasoning" rose **65% → >90%**), not new capability. **R3 router replay** (`2510.11370`): even with identical weights, training and inference engines select divergent experts — **~94% of tokens differ in ≥1 layer** — collapsing 3/3 GRPO runs (this is what `kl_train_infer` HALT@0.10 catches). **TinyZero**: the R1-Zero "aha" reproduces for **~$30** at **1.5B**, and **0.5B fails** (the floor).

**The differentiator:** CS336 teaches you to *optimize* a reward (GRPO) but never to *audit* it. **A5.2 — verifier-exploitability as a controlled independent variable** — is precisely the gap. Treating `HardeningLevel` as a dial and `true_quality` as a held-out oracle you never train against is the thing no homework and few candidates have built. That is the entire reason `reasoningLLM` is on-frontier.

---

## 6. Prioritization verdict — what matters / what to skip

**The ~20% that is load-bearing (build these with care, they ship):**
1. **GRPO + Dr.GRPO** — `compute_group_normalized_rewards` with the `normalize_by_std` toggle (Eq. 28 ↔ Eq. 31), the clip loss, the policy-gradient wrapper, the microbatch step, the train loop. → `algos/advantage.py`, `algos/off_policy.py`.
2. **The exploitability dial + true-quality oracle** — net-new, no CS336 deliverable. → `envs/exploitability.py`, `envs/true_quality.py`. This is the single highest-value artifact in the whole 14 days.
3. **The three-KL logging discipline** — `compute_entropy` + `get_response_log_probs` + IS-ratio/reward/length stats + `kl_train_infer` HALT@0.10. → `utils/monitors.py`. Without it the run is uninterpretable (repo discipline #4).
4. **The spurious-rewards battery** — the 5 reward conditions, run with the *predict-before-you-run* numbers (random +21 / GT +29 on Qwen; near-zero transfer). This is VERA's H1.
5. **The SFT/masking primitives** — `tokenize_prompt_and_output`, `masked_normalize`, `masked_mean`, `sft_microbatch_train_step` — small but everything depends on correct masking.

**Course-rote (build once to understand, then move on):**
- `math_baseline` eval-categorization, `expert_iteration_experiment` (EI is the no-gradient baseline — cheap intuition, not a shipped algo), the supplement's `sft_script`/`sft`, packed `data_loading`, the generalist eval harness.
- **`dpo_loss`** — *build the loss* (it's a clean 1.5 h primitive and an interview staple), but whether it enters the repo is an open decision (§8). Treat as course-rote-plus.

**SKIP (informs nothing on the engine path):**
- Supplement `look_at_sft`, `look_at_hh`, all `*_sft` eval-delta re-runs, `dpo_training` on Anthropic HH (HH preference RLHF is off the verifiable-reward thesis), the `red_teaming` *writeup* (keep only the misuse *catalog* as `hack_detector.py` seeds), AlpacaEval/SST annotator runs.
- Rationale: the supplement's center of gravity is *generalist-chat alignment on Llama 3.1 8B*. The L5 thesis is *verifiable-reward reasoning RL on 1.5B math*. Anything that only advances the chat path is skip-on-the-engine-path (still fine to skim for the interview "RLHF vs DPO" answer).

---

## 7. Build checklist (ordered, with discipline gates)

1. **Lock the metric first (Phase 1).** Before any code: name `reward`, `true_quality`, `true_quality_gap`, `hack_rate`, `kl_train_infer`. Each must be monotonic, low-noise, above-random. Write the falsifiable numbers (H1–H3) into a committed doc — **predict before you run** (discipline #5).
2. **Primitives + loss-at-init.** `tokenize_prompt_and_output` → `compute_entropy` → `get_response_log_probs` → `masked_normalize` / `masked_mean`. Assert loss-at-init ≈ `log(vocab_size)` on a fresh head; run `pytest -k test_tokenize_prompt_and_output …` green. (Discipline #1.)
3. **SFT step + overfit-one-batch.** `sft_microbatch_train_step`; drive train loss → ~0 on a single batch before trusting anything (discipline #2). This is the warm-start path.
4. **Advantage + losses (the L5 core).** `compute_group_normalized_rewards` with **both** GRPO (Eq. 28) and Dr.GRPO (Eq. 31) variants; `compute_naive_policy_gradient_loss`; `compute_grpo_clip_loss`; the `compute_policy_gradient_loss` wrapper; `grpo_microbatch_train_step`. → seeds `algos/advantage.py`, `algos/off_policy.py`.
5. **TinyZero Countdown-1.5B "aha" repro (EARLY DE-RISK).** Get GRPO running on Countdown with Qwen2.5-1.5B (~$30) — reproduce the R1-Zero "aha" by ~day 3–4. This validates the *entire engine* before you build the dial. **Kill criterion:** if cross-family RL is too finicky, collapse to single-family Qwen-1.5B (still publishable).
6. **Mandatory RL logging in, not reactively (discipline #4).** Wire the three KLs separately + IS-ratio histograms + reward-dist stats + length-by-correctness *before* the first real GRPO run. Set `kl_train_infer` HALT@0.10.
7. **The exploitability dial + oracle (A5.2).** `envs/exploitability.py` (5 levels), `envs/true_quality.py` (held-out + isomorphic perturbation — guard against leakage, it's make-or-break), `rewards/hack_detector.py`. Compose `rewards/reward.py` over the dial.
8. **Pre-register H1–H3, then run the spurious-rewards battery.** 5 reward conditions × 3 families. **Predict first:** random ≈ +21 / GT ≈ +29 on Qwen-Math; **near-zero transfer to Llama/OLMo** — if random reward helps Llama equally, H1 is falsified; *report it* (well-analyzed negative = a win).
9. **Daily green commit (G6, EOD 7).** Each of Days 5–8 closes with a green-CI commit on its §0.8 ship target. G2 (EOD 6) checks you can connect the credit-assignment seam.

---

## 8. Open questions / ADR triggers

- **Keep DPO in v0.1.0, or GRPO-only?** The supplement's `dpo_loss` is a clean primitive and an interview staple, but DPO is *preference*-based, not *verifiable-reward*-based — arguably off the L5 thesis. **ADR trigger:** decide whether `algos/` carries a DPO path or whether DPO stays a course-rote exercise that never enters the repo. Default lean: build the loss to mastery, keep it *out* of the shipped engine unless a preference-data env materializes.
- **Single-family fallback if cross-family RL is too finicky.** Llama-3.2-1B / OLMo-2-1B may need different LR/KL than Qwen (CAPSTONE §2.7 honest risk). **ADR trigger:** if >1–2 debug runs per family burn, collapse VERA to the single-family Qwen-1.5B spurious-rewards grid + the `HardeningLevel` dial (<$120, still the "I red-teamed spurious-rewards at 1.5B" artifact).
- **Supplement safety scope.** How much of the safety/red-teaming supplement feeds `hack_detector.py` vs. is pure skip? **ADR trigger:** keep only the *misuse catalog* as detector seeds; explicitly scope *out* AlpacaEval/SST/HH-DPO runs (generalist-chat, off-thesis).
- **GRPO vs Dr.GRPO as the default loss.** `normalize_by_std=True` (GRPO) vs `False` (Dr.GRPO). The length-normalization worked example (main §8, p. 30) shows the gradient differs. **ADR trigger:** which is the repo default, and is the toggle a first-class config for VERA's ablation? (Lean: Dr.GRPO default, toggle exposed.)
- **`masked_mean` vs `masked_normalize` for loss aggregation.** Per-token mean vs sum/constant changes credit assignment across response lengths (the Lambert/Liu/Yu length-norm debate). **ADR trigger:** pin the aggregation choice and document why — this is a load-bearing hyperparameter, not a detail.
- **Async staleness for `rollout/async_orchestrator.py`.** Day-8 target adds one-step staleness over CS336's synchronous Alg. 3. **ADR trigger:** confirm one-step default and whether off-policy epochs>1 (requiring cached old log-probs, main §7.2 p. 28) are in v0.1.0 scope or deferred.
