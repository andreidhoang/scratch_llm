# A5 — Alignment · BUILD GUIDE (CS336 post-training → `algos/` · `rewards/` · `envs/`)

> **📖 Read first (slides → this build):** Lectures **15 → 16** — mid/post-training SFT (pdf) · RL
> from verifiable rewards / GRPO (pdf); then **10** (inference/rollout, py) and **12** (evaluation,
> py). Full map + read-order: [`../LECTURE_MAP.md`](../LECTURE_MAP.md).

> **One-liner.** A5 is the **post-training crown**: the one assignment where you build the RL
> fine-tuning engine itself — **SFT → Expert Iteration (STaR) → GRPO/Dr.GRPO**, optimized against a
> **verifiable reward** (rule-based format + answer grading), plus the optional supplement's **DPO /
> reward-modeling** and **safety / red-teaming** concepts. This is the scarcest, highest-value 2026
> skill cluster — *making RL post-training not diverge* — and the interview-load-bearing one (every
> post-training screen is some subset of GRPO/Dr.GRPO/PPO, reward modeling, RLHF-vs-DPO, RL
> stability). It builds the three stub packages in this repo: **`algos/`** (the SFT/RL update steps
> + advantage estimation), **`rewards/`** (the r1-zero format+answer grader), and **`envs/`** (the
> verifiable math / Countdown environment behind the existing `envs/protocol.py`).

---

## 1. What CS336 actually requires (every deliverable — main + supplement)

A5 main (`Version 1.0.2`) is MATH-reasoning RL on **Qwen2.5-Math-1.5B**: zero-shot baseline → SFT →
Expert Iteration → GRPO. Of the unit tests, `tests/test_sft.py` and `tests/test_grpo.py` are
mandatory; baseline/EI are non-mandatory-but-built. The supplement (`Version 1.0.1`, **entirely
optional in the course**, but core-for-mastery here) covers generalist instruction-tuning + DPO +
safety/red-teaming on **Llama-3.1-8B**. Adapter names below are quoted from
`../../../lectures/assignment5-alignment/tests/adapters.py`; equations from the two PDFs.

**Priority legend** (matches `INDEX.md`): **LOAD-BEARING** = master it cold, it carries the
assignment *and* the interview; **COURSE-ROTE** = implement correctly to pass the tests, don't
over-invest; **SKIP** = a GPU re-run / dollar-sink with no mastery carry (build the *concept*, skip
the *run*).

### Main assignment — Reasoning RL on Qwen2.5-Math-1.5B

| Deliverable (PDF `Problem` name) | PDF § | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|
| `math_baseline` — zero-shot MATH eval; `evaluate_vllm(model, reward_fn, prompts, params)`; categorize by (format × answer) reward; commentary | §3.2 (pp. 5–6) | `drgrpo_grader.r1_zero_reward_fn` (provided); no adapter | 2–3 h | **COURSE-ROTE** (the eval-categorization commentary is rote; the r1-zero prompt + grader *pattern* it exercises is reused everywhere) |
| `tokenize_prompt_and_output` — tokenize Q + O separately, concat, build `response_mask` (1 on response tokens in `labels`, 0 on prompt/pad) | §4.2 (p. 9) | `run_tokenize_prompt_and_output` / `test_tokenize_prompt_and_output` | 1.5 h | **LOAD-BEARING** — the masking primitive; every per-token loss (SFT, GRPO, DPO) depends on it being right |
| `compute_entropy` — per-token next-token entropy, numerically stable (logsumexp) | §4.2 (p. 10), Eq. 1 | `run_compute_entropy` / `test_compute_entropy` | 0.5 h | **LOAD-BEARING** — the mandatory entropy log (entropy-collapse signal) |
| `get_response_log_probs` — per-token `log p_θ(x_t \| x_<t)`, optional token-entropy | §4.2 (pp. 10–11), Eq. 2 | `run_get_response_log_probs` / `test_get_response_log_probs` | 1 h | **LOAD-BEARING** — the scoring primitive (SFT, RL, *and* DPO all call it) |
| `masked_normalize` — masked **sum / constant**, optional `dim` | §4.2 (pp. 11–12) | `run_masked_normalize` / `test_masked_normalize` | 0.5 h | **LOAD-BEARING** — the length-normalization lever (the GRPO-vs-Dr.GRPO aggregation hinges on this) |
| `sft_microbatch_train_step` — masked NLL, grad-accum scaling, returns `(loss, metadata)` | §4.2 (pp. 12–13) | `run_sft_microbatch_train_step` / `test_sft_microbatch_train_step` | 1.5 h | **LOAD-BEARING** — the SFT update; reused as the warm-start path for RL |
| `log_generations` — log prompt, response, gt, reward (format/answer/total), avg entropy, response-length stats split by correctness | §4.2 (p. 13) | no adapter | 1 h | **LOAD-BEARING** — length-by-correctness logging is how you catch verbosity reward-hacking (discipline #4) |
| `sft_experiment` — run SFT on `sft.jsonl`; sweep size {128,256,512,1024,full}; ≥~15% val; filtered-correct variant | §4.3 (pp. 13–15) | run only (~2 H100 hrs) | GPU | **COURSE-ROTE** (the data-size ablation is rote; the *filtered-correct* idea is exactly the EI seed) |
| `expert_iteration_experiment` — EI / STaR (Alg. 2): sample G, keep correct, SFT, repeat `n_ei_steps=5`; sweep G∈{4,8,16}, batch∈{512,1024,2048} | §5 (pp. 15–16), Alg. 2 | run only (~6 H100 hrs) | GPU | **LOAD-BEARING** — the cheapest RL-without-gradients baseline; the conceptual bridge from SFT to GRPO (you *must* understand why filtering-then-SFT already improves reasoning) |
| `compute_group_normalized_rewards` — raw rewards, group-normalize (Eq. 28 GRPO **or** Eq. 31 Dr.GRPO via `normalize_by_std`), return `(advantages, raw_rewards, metadata)` | §7.2 (pp. 22–23), Eq. 28/31 | `run_compute_group_normalized_rewards` / `test_compute_group_normalized_rewards` | 1.5 h | **LOAD-BEARING** — the heart of `algos/` advantage estimation; the GRPO↔Dr.GRPO toggle |
| `compute_naive_policy_gradient_loss` — `−A_t · log p_θ(o_t)`, broadcast adv over the sequence | §7.2 (p. 23), Eq. 32 | `run_compute_naive_policy_gradient_loss` / `test_…` | 0.5 h | **LOAD-BEARING** — the REINFORCE term, the irreducible core |
| `compute_grpo_clip_loss` — Eq. 33 per-token clip on the IS ratio; return `(loss, metadata{was_clipped})` | §7.2 (p. 24), Eq. 33 | `run_compute_grpo_clip_loss` / `test_compute_grpo_clip_loss` | 1.5 h | **LOAD-BEARING** — the clip *is* the PPO trust region; the thing that makes >1 epoch per rollout batch safe |
| `compute_policy_gradient_loss` — wrapper dispatching `no_baseline` / `reinforce_with_baseline` / `grpo_clip` | §7.2 (pp. 24–25) | `run_compute_policy_gradient_loss` / `test_…` | 0.5 h | **LOAD-BEARING** — the 3-variant dispatcher *is* the baselining ablation |
| `masked_mean` — masked **mean** over `dim` | §7.2 (p. 26) | `run_masked_mean` / `test_masked_mean` | 0.5 h | **LOAD-BEARING** — the per-sequence aggregator (contrast with `masked_normalize`; this pairing is the length-norm study) |
| `grpo_microbatch_train_step` — per-token loss → aggregate → grad-accum → backward; returns `(loss, metadata)` | §7.2 (pp. 26–27) | `run_grpo_microbatch_train_step` / `test_…` | 1.5 h | **LOAD-BEARING** — the GRPO update step |
| `grpo_train_loop` — full Alg. 3 loop; on-policy defaults (`n_grpo_steps=200`, `rollout_batch=256`, `group_size=8`); sensible val rewards | §7.2 (pp. 27–29), Alg. 3 + Eq. 29 | run only (5 pts) | GPU | **LOAD-BEARING** — the whole-engine integration: rollout → grade → advantage → update |
| `grpo_learning_rate` / `grpo_baselines` — LR sweep; baseline-vs-no-baseline; ≥~25% MATH | §8 (p. 29) | run only | GPU | **LOAD-BEARING** — the baselining ablation is the first real reward curve |
| Length-normalization study (`masked_mean` vs `masked_normalize`, batch-size-2 worked example) | §8 (p. 30) | conceptual | 0.5 h | **LOAD-BEARING** — this is the Dr.GRPO de-biasing argument written out in code |

### Supplement — Instruction Tuning + RLHF (DPO) + Safety on Llama-3.1-8B (optional in the course; the loss + concepts are core here)

| Deliverable (PDF `Problem` name) | PDF § | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|
| `mmlu_baseline` / `gsm8k_baseline` (parse + eval), `alpaca_eval_baseline`, `sst_baseline` (SimpleSafetyTests) | §2.1–2.4 (pp. 3–7) | `run_parse_mmlu_response`, `run_parse_gsm8k_response`; rest run-only | 3–4 h | **COURSE-ROTE** — the generalist-eval harness; useful parsing pattern (last-number / option-letter), not the engine |
| `look_at_sft` — inspect 10 instruction-tuning examples | §3.1 (p. 8) | no adapter | 0.5 h | **SKIP** |
| `data_loading` — packed SFT `Dataset` + `iterate_batches` (Alpaca template, packed sequences) | §3.2.1 (pp. 9–10) | `get_packed_sft_dataset`, `run_iterate_batches` / `test_packed_sft_dataset`, `test_iterate_batches` | 2 h | **COURSE-ROTE** — sequence packing is a real data-path skill; build it, don't gold-plate |
| `sft_script` + `sft` — instruction-tune Llama-3.1-8B, 1 epoch, ctx 512, lr 2e-5 | §3.2.2 (p. 12) | run only (~24 H100 hrs) | GPU | **COURSE-ROTE** — vanilla SFT, commoditized; the loss is already the main-assignment `sft_microbatch_train_step` |
| `mmlu_sft` / `gsm8k_sft` / `alpaca_eval_sft` / `sst_sft` — re-eval the tuned model | §4.1–4.4 (pp. 13–15) | run only | GPU | **SKIP** — eval-delta reporting on the generalist-chat model; no mastery carry |
| `red_teaming` — 3 misuse vectors; try to break the tuned model | §4.5 (p. 15) | no adapter | 1 h | **LOAD-BEARING (concept)** — the misuse catalog + *why alignment matters*; the safety story you must be able to tell. Skip the polished writeup, keep the catalog |
| `dpo_loss` — per-instance DPO loss (Eq. 3), `π_θ` vs `π_ref`, Alpaca template, EOS appended | §5.3 (p. 18), Eq. 3 | `run_compute_per_instance_dpo_loss` / `test_per_instance_dpo_loss` | 1.5 h | **LOAD-BEARING** — the preference-optimization primitive; the "explain RLHF vs DPO" interview staple lives here (build it beside Bradley-Terry reward modeling) |
| `dpo_training` — DPO loop on Anthropic-HH, β=0.1, lr 1e-6, RMSprop, track classification accuracy | §5.4 (pp. 18–19) | run only (4 pts) | GPU | **SKIP** (the heavy run) — build the loss + understand the loop; the multi-hour HH run carries no extra mastery |
| `look_at_hh` — load + inspect Anthropic-HH (chosen/rejected) | §5.2 (p. 17) | no adapter | 0.5 h | **SKIP** |

---

## 2. The equations / algorithms that matter (senior extraction)

Eight load-bearing pieces. Everything else in A5 is plumbing around these.

1. **SFT masked cross-entropy** — `L_SFT = − Σ_t mask_t · log p_θ(o_t | q, o_<t) / Z`. Only response
   tokens contribute (the `response_mask`); prompt + padding are zeroed. *Why it matters:* it is the
   warm-start *and* the exact per-token log-prob machinery GRPO reuses — get the masking wrong here
   and every downstream RL number is silently corrupt. The `normalize_constant` / `dim` choice is not
   cosmetic: it is the length-normalization lever that separates GRPO from Dr.GRPO.

2. **Expert Iteration / STaR (Alg. 2)** — sample `G` rollouts per question, **keep only the
   verifiably-correct ones**, SFT on that filtered set, repeat for `n_ei_steps`. *Why it matters:*
   it is reinforcement learning *without a policy gradient* — the cheapest possible RL baseline and
   the conceptual bridge from SFT to GRPO. Internalize why it works (the model already can solve some
   problems; filtering + SFT amplifies that mass) and why it plateaus (no credit assignment, no
   exploration pressure beyond temperature) — that contrast is the motivation for GRPO.

3. **GRPO group-relative advantage (Eq. 28)** — `A^(i) = (r^(i) − mean(r^(1..G))) / (std(r^(1..G)) +
   ε)`. Sample `G` outputs per question; use the *group* as the baseline — **no learned value
   network**. Constant across the tokens of a response (`A_t^(i) = A^(i)`). *Why it matters:* this is
   the entire reason GRPO is cheap and stable on verifiable rewards — the variance-reduction baseline
   is free (you already sampled the group). It is the core of `compute_group_normalized_rewards`.

4. **Dr.GRPO de-biasing (Eq. 31)** — `A^(i) = r^(i) − mean(r^(1..G))` (drop the `/ std`). *Why it
   matters:* dividing by the per-group std up-weights low-variance (easy / nearly-solved) questions —
   a **question-difficulty bias**; combined with per-response length normalization it inflates output
   length. Dr.GRPO removes both. This is the single most important "senior" toggle in the file
   (`normalize_by_std=False`), and the `masked_mean`-vs-`masked_normalize` length-norm study below is
   the second half of the same correction.

5. **Importance-sampling ratio + GRPO-Clip (Eq. 27, 29, 33)** — the off-policy gradient reweights by
   `ρ_t = π_θ(o_t) / π_θ_old(o_t)`; the per-token objective is `min(ρ_t · A, clip(ρ_t, 1−ε, 1+ε) ·
   A)`. *Why it matters:* the clip is the PPO trust region — it lets you take **multiple gradient
   steps per rollout batch** (epochs > 1, mild off-policy) without the policy running away from the
   distribution it was sampled under. CS336 stops exactly here: a plain per-token clip on `ρ_t` with
   one `cliprange`. The IS-ratio histogram you log around it is a mandatory RL-stability signal.

6. **REINFORCE / policy-gradient loss (Eq. 20, 32)** — `∇J = E[Σ_t ∇log π_θ(a_t|s_t) · R(τ)]`;
   per-token loss `−A_t · log p_θ(o_t)`. *Why it matters:* the irreducible core — every variant
   (`no_baseline`, `reinforce_with_baseline`, `grpo_clip`) is this with a different `A_t` and an
   optional clip. `compute_policy_gradient_loss` is the dispatcher; running all three is the baseline
   ablation that shows *why* the group baseline reduces variance.

7. **DPO / Bradley-Terry (supp Eq. 1, 3)** — reward model: `ℓ_RM = −log σ(r(x,y_w) − r(x,y_l))`
   (Bradley-Terry pairwise preference). DPO collapses the reward model *into the policy*: `ℓ_DPO =
   −log σ(β [log π_θ(y_w)/π_ref(y_w) − log π_θ(y_l)/π_ref(y_l)])`. *Why it matters:* DPO needs **no
   sampling and no reward model** — just conditional log-probs under `π_θ` and `π_ref` on a fixed
   preference pair. It is the *preference*-based alternative to verifiable-reward RL, and the answer
   to the standard "RLHF (PPO + reward model) vs DPO" interview question: PPO learns an explicit RM
   and optimizes it online (flexible, unstable, expensive); DPO is the closed-form offline reduction
   (stable, cheap, but tied to the preference dataset and the reference policy).

8. **The verifiable-reward grader** — `answer_reward = 1 if normalize(answer) == normalize(gt) else
   0`, with `format_reward` (well-formed `<answer>…</answer>`) factored separately, and `reward` the
   scalar the update optimizes. *Why it matters:* this binary, rule-based reward is what makes the
   whole loop *verifiable* — no human, no learned RM, no annotator drift. The provided
   `drgrpo_grader.r1_zero_reward_fn` returns exactly `{"reward", "format_reward", "answer_reward"}`,
   which is the `RewardDict` / `Graded` shape pinned in `envs/protocol.py` — that dict is the seam
   where `rewards/` plugs into `envs/`.

---

## 3. Map to `src/scratch_llm/{algos,rewards,envs}/`

A5 builds three stub packages. Inner filenames are *not* committed yet — map each deliverable to its
package and the real `envs/protocol.py` seam; choose module names when you build (don't pre-commit a
layout). Logging lands in the already-built `utils/monitors.py`. The build plan that orders this is
[`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md); the spec + test oracle is the official
scaffold at `../../../lectures/assignment5-alignment/` (`tests/adapters.py`, `tests/test_sft.py`,
`tests/test_grpo.py`, `tests/test_dpo.py`, and the provided `cs336_alignment/drgrpo_grader.py`).

| CS336 deliverable | → `src/scratch_llm/…` | Keep / adapt / course-only |
|---|---|---|
| `tokenize_prompt_and_output`, `compute_entropy`, `get_response_log_probs`, `masked_normalize`, `masked_mean`, `sft_microbatch_train_step` | **`algos/`** — the SFT/train-step module + masking helpers (the warm-start path) | **keep** as the shared primitives every loss reuses |
| `expert_iteration_experiment` (Alg. 2) | **`algos/`** — sample → filter-correct → SFT loop (reuses `rewards/` to grade, `algos/` to SFT) | **keep** — the no-gradient RL baseline |
| `compute_group_normalized_rewards` (Eq. 28 + Dr.GRPO Eq. 31) | **`algos/`** — group-relative advantage; expose the `normalize_by_std` toggle = GRPO↔Dr.GRPO | **keep — the core**; wire both variants, the toggle is first-class |
| `compute_naive_policy_gradient_loss`, `compute_grpo_clip_loss`, `compute_policy_gradient_loss` | **`algos/`** — the three loss variants + the dispatcher (REINFORCE / +baseline / grpo-clip) | **keep** — the loss family and its ablation switch |
| `grpo_microbatch_train_step`, `grpo_train_loop` (Alg. 3) | **`algos/`** — the GRPO update step + the full rollout→grade→advantage→update loop | **keep** — the engine integration |
| `r1_zero_reward_fn` (format + answer composition → `RewardDict`) | **`rewards/`** — the verifiable grader; conforms to `envs/protocol.py` `RewardFn` (text-in / dict-out) | **keep** — the verifiable reward; the math grader can wrap the provided `drgrpo_grader` |
| the verifiable math / Countdown task pool + decode + grade | **`envs/`** — a `VerifiableEnv` behind the existing **`envs/protocol.py`** (`Task` / `Graded` / `DecodeFn`) | **keep** — the env the loop samples from; CPU `LocalBackend` for CI, GPU backend for real rollouts |
| `compute_entropy` + `get_response_log_probs` + the KL / IS-ratio / reward / length logs | **`utils/monitors.py`** (already built) — entropy, the two training-KL channels, IS-ratio histogram, reward + length stats | **keep** — discipline #4; CS336's `compute_entropy` + `log_generations` spec *is* the mandatory-log minimum |
| `dpo_loss` (supp Eq. 3) | **`algos/`** — per-instance DPO loss (`π_θ` vs `π_ref`); see the ADR on whether it ships as a path or stays a studied primitive | **keep the loss** (LOAD-BEARING for mastery); a shipped DPO *pipeline* is ADR-gated (§7) |
| supplement evals (MMLU / GSM8K / AlpacaEval / SST), packed `data_loading`, Llama SFT/DPO training runs | — | **course-only / skip** — generalist-chat path; build the parsers + packing once, skip the runs |

The reframe in one sentence: **`algos/` is the optimizer (SFT → EI → GRPO/Dr.GRPO + the loss
family), `rewards/` is the verifiable grader, `envs/` is the task pool — and `envs/protocol.py` is
the seam that keeps them decoupled** so the same engine runs on a CPU backend in CI and a GPU backend
for real rollouts.

---

## 4. The frontier-2026 lens

**Commoditized (do not lead with these):** vanilla SFT, masked cross-entropy, gradient
accumulation, packed datasets, generalist eval harnesses. These are table-stakes — every candidate
has them. The supplement's SFT path is almost entirely here.

**Scarce (the low-supply, high-demand corner):**

- **RL stability — making post-training not diverge.** The progression to internalize cold, and to
  be able to explain *why each step exists*:
  - **GRPO** (DeepSeek-Math `2402.03300`, DeepSeek-R1 `2501.12948`) — drop the PPO value network; use
    the group mean as a free baseline. Cheaper, fewer moving parts, stable on verifiable rewards.
  - **Dr.GRPO** (`2503.20783`) — remove the std-normalization difficulty-bias *and* the
    per-response length normalization that together inflate output length. The `normalize_by_std`
    toggle + the `masked_mean`/`masked_normalize` choice in this assignment *are* this correction.
  - Knowing the GRPO → Dr.GRPO step (what bias each piece introduces, and the worked length-norm
    example) is the senior signal an interviewer probes for.
- **Verifiable rewards (RLVR).** A rule-based, binary grader (format + answer) is what makes the loop
  trustworthy with no human and no learned reward model — this is why reasoning RL took off on math /
  code, where correctness is checkable. `rewards/` + `envs/` is exactly this.
- **RLHF vs DPO** — the post-training decision tree: explicit reward model + online PPO (flexible,
  unstable, expensive) vs DPO's offline closed-form reduction (stable, cheap, dataset-bound). The
  supplement's `dpo_loss` + Bradley-Terry reward modeling is the artifact that lets you answer this
  from first principles rather than from a blog post.
- **Why alignment matters / safety.** The supplement's red-teaming (misuse vectors, why an
  instruction-tuned model needs a safety pass) is the concept layer behind every production
  post-training stack — interview-relevant even when you skip the eval runs.

**The mandatory RL-logging discipline (load-bearing, CS336-legitimate):** an RL run with no
instrumentation is uninterpretable. Before the first GRPO step, wire **entropy**, the two training-KL
channels **`KL(current‖ref)`** and **`KL(current‖old)`**, the **IS-ratio histogram**, and
**reward-distribution + response-length stats** (length-by-correctness is the verbosity-hack
detector). When the rollout (inference) engine differs from the training engine, also log the
**train↔inference KL drift** — a real production signal. This is repo discipline #4 and the single
highest-leverage habit in the whole assignment.

---

## 5. Prioritization verdict — what matters / what to skip

**The ~20% that is load-bearing (master these cold):**
1. **The SFT + masking primitives** — `tokenize_prompt_and_output` (+ `response_mask`),
   `compute_entropy`, `get_response_log_probs`, `masked_normalize`, `masked_mean`,
   `sft_microbatch_train_step`. Small, but *everything* downstream depends on correct masking.
2. **Expert Iteration (STaR)** — the cheapest RL-without-gradients baseline and the conceptual bridge
   to GRPO; build it and be able to say why it works and where it plateaus.
3. **GRPO + Dr.GRPO** — `compute_group_normalized_rewards` with the `normalize_by_std` toggle (Eq. 28
   ↔ Eq. 31), `compute_grpo_clip_loss` (Eq. 33), the `compute_policy_gradient_loss` 3-variant
   dispatcher, `grpo_microbatch_train_step`, `grpo_train_loop`. The core of `algos/`.
4. **The length-normalization study** — `masked_mean` vs `masked_normalize`, the batch-size-2 worked
   example. This *is* the Dr.GRPO de-biasing argument in code.
5. **The verifiable-reward grader** — `r1_zero_reward_fn` (format + answer → `RewardDict`); the
   `rewards/` ↔ `envs/protocol.py` seam.
6. **The DPO loss + reward modeling (supplement)** — `run_compute_per_instance_dpo_loss` (Eq. 3) +
   Bradley-Terry; the RLHF-vs-DPO interview staple.
7. **The safety / red-teaming concepts (supplement)** — the misuse catalog + why alignment matters.
8. **The mandatory RL-logging discipline** — entropy + the two training-KLs + IS-ratio histogram +
   reward/length stats, wired *before* the first GRPO run (discipline #4).

**Course-rote (build once to pass the tests, then move on):**
- `math_baseline` eval-categorization; the supplement's `sft_script`/`sft` (the loss is already the
  main-assignment SFT step); packed `data_loading`; the generalist eval parsers
  (`parse_mmlu_response`, `parse_gsm8k_response`).

**SKIP (no mastery carry — build the concept, skip the run):**
- Supplement `look_at_sft`, `look_at_hh`; the `*_sft` eval-delta re-runs
  (`mmlu_sft`/`gsm8k_sft`/`alpaca_eval_sft`/`sst_sft`); the full `dpo_training` run on Anthropic-HH;
  the AlpacaEval / SST annotator runs. Rationale: the supplement's center of gravity is
  *generalist-chat alignment on Llama-3.1-8B*; the load-bearing post-training mastery is the
  **algorithms** (GRPO/Dr.GRPO, DPO loss) + the **concepts** (RLHF-vs-DPO, safety), not the
  multi-hour generalist-chat eval/training grind.

**Optional GPU capstone (not required, high signal):** reproduce the **R1-Zero "aha" on Countdown
with Qwen2.5-1.5B** (~$30 on a rented GPU). It is the cheapest end-to-end validation that the RL
engine actually works on a real model — the emergent multi-step reasoning behavior shows up at
**1.5B and fails to emerge at 0.5B**, so 1.5B is the floor. Frame it as engine-validation: it proves
`grpo_train_loop` + `rewards/` + `envs/` compose into a working learner, nothing more.

**Interview leverage (why this assignment is the crown):** the post-training family — GRPO / Dr.GRPO
/ PPO, reward modeling, RLHF vs DPO, RL stability and its logging — is the scarcest, most-screened
2026 skill cluster, and A5 is where you own all of it from scratch.

---

## 6. Build checklist (ordered, with discipline gates)

1. **Primitives + loss-at-init.** Build `tokenize_prompt_and_output` → `compute_entropy` →
   `get_response_log_probs` → `masked_normalize` / `masked_mean`. Assert **loss-at-init ≈
   log(vocab_size)** on a fresh head (discipline #1 — the cheapest correctness oracle). Run
   `pytest -k "test_tokenize_prompt_and_output or test_compute_entropy or test_masked"` green against
   the official snapshots.
2. **SFT step + overfit-one-batch.** Build `sft_microbatch_train_step`; **drive train loss → ~0 on a
   single batch** before trusting anything downstream (discipline #2). This is the warm-start path
   and `test_sft.py`'s 10-step gradient check.
3. **Expert Iteration.** Wire sample → filter-correct → SFT → repeat (Alg. 2). Predict the
   direction (filtered-correct val accuracy rises over `n_ei_steps`, then plateaus) before running.
4. **Advantage + the loss family (the `algos/` core).** `compute_group_normalized_rewards` with
   **both** GRPO (Eq. 28) and Dr.GRPO (Eq. 31, `normalize_by_std=False`);
   `compute_naive_policy_gradient_loss`; `compute_grpo_clip_loss`; the `compute_policy_gradient_loss`
   dispatcher; `grpo_microbatch_train_step`. Green `test_grpo.py` against the snapshots.
5. **Mandatory RL logging — wired in, not bolted on (discipline #4).** *Before* the first real GRPO
   run, instrument **entropy + `KL(current‖ref)` + `KL(current‖old)` + IS-ratio histogram +
   reward-distribution + length-by-correctness stats** via `utils/monitors.py`. An RL run without
   these is an uninterpretable run.
6. **`grpo_train_loop` integration.** Compose `envs/` (sample) → `rewards/` (grade) → `algos/`
   (advantage → loss → update). **Predict before you run** (discipline #5): write the falsifiable
   number first — e.g. GRPO val reward rising above the zero-shot `math_baseline`, ≥~25% MATH for
   `grpo_baselines`. The prediction is the debugging anchor.
7. **DPO loss (supplement).** `run_compute_per_instance_dpo_loss` (Eq. 3) against `test_dpo.py`'s
   `tiny-gpt2` fixture (expected loss ≈ 0.5785). Be able to whiteboard it next to Bradley-Terry.
8. **Optional GPU capstone.** If resourcing a GPU: the Countdown / Qwen2.5-1.5B "aha" repro (~$30) as
   the end-to-end engine check. Keep the run green-logged (the §5 logging applies).
9. **Green-CI throughout.** Every commit: `ruff check` + `ruff format --check` + `pyright` +
   `pytest -m "not gpu"`. CPU-buildable steps (1–4, 7) gate on the snapshot tests; GPU steps (6, 8)
   are resourced on rented hardware, never silently dropped.

---

## 7. ADR triggers (log in `../adr/`, don't relitigate inline)

- **GRPO vs Dr.GRPO as the repo default.** `normalize_by_std=True` (GRPO) vs `False` (Dr.GRPO); the
  length-normalization worked example (§8, p. 30) shows the gradient differs. **Trigger:** which is
  the default, and is the toggle a first-class config? (Lean: Dr.GRPO default, toggle exposed.)
- **`masked_mean` vs `masked_normalize` for loss aggregation.** Per-token mean vs sum/constant
  changes credit assignment across response lengths. **Trigger:** pin the aggregation and document
  why — it is a load-bearing hyperparameter, not a detail.
- **Does DPO ship as a path, or stay a studied primitive?** The `dpo_loss` is LOAD-BEARING for
  mastery, but DPO is preference-based (needs a preference dataset + a frozen reference), distinct
  from the verifiable-reward RL path. **Trigger:** decide whether `algos/` carries a DPO training
  pipeline or whether the loss stays a tested primitive until a preference-data env materializes.
- **`envs/` backend scope.** `envs/protocol.py` already supports a CPU `LocalBackend` (CI) and a GPU
  backend (real rollouts). **Trigger:** confirm the CPU backend is enough to test `grpo_train_loop`
  end-to-end on a toy env, and which GPU inference backend (HF generate / SGLang / vLLM) the rollout
  path targets.
- **Off-policy epochs > 1.** `grpo_train_loop` defaults are on-policy; epochs > 1 require caching the
  old log-probs (`π_θ_old`) per rollout batch (§7.2, p. 28). **Trigger:** is multi-epoch off-policy
  in scope, or on-policy-only for the first working loop?
