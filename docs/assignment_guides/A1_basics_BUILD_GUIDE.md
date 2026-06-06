# A1 — Basics · BUILD GUIDE (CS336 → reasoningLLM L1 Substrate)

> **STATUS: ✅ L1 substrate complete** — `tokenizer.py`, `model.py`, `optim.py`, `train.py`,
> `sampling.py`, `utils/seeding.py` built, tested, green (see [`../STATUS.md`](../STATUS.md)).
> Discipline tests baked in: loss-at-init≈logV, causal-no-leak, overfit-one-batch, seed-repro.

> **One-liner.** CS336 A1 ("Building a Transformer LM") makes you build, from scratch, the *entire policy substrate* — a byte-level BPE tokenizer, a pre-norm decoder Transformer (RMSNorm · RoPE · SwiGLU · causal MHA), cross-entropy, AdamW, the cosine+warmup schedule, gradient clipping, a memory-mapped training loop, checkpointing, and a temperature/top-p decoder — and train a tiny LM on TinyStories. This is **Layer L1 (Substrate)** of the reasoningLLM stack. It feeds **"the policy served in rollouts"**: the model whose logits the RLVR engine inspects, the tokenizer the env encodes with, the sampler the rollout client calls. **The through-line tie:** you cannot measure `true_quality_gap = reward − true_quality` (or `hack_rate`, or `kl_train_infer`) on a policy you do not own end-to-end. `kl_train_infer` is *literally* a KL between two engines' next-token distributions over your tokenizer's vocab; `hack_rate` is read off length/logit statistics; sampling temperature/top-p is the exploration knob of every rollout. Owning A1 is the precondition for every diagnosis downstream. L1 is table-stakes ("implement a Transformer from scratch in ~45 min") — necessary but, on its own, commoditized. The scarce part is what you bake *into* your own substrate (the discipline harness, and the MoE add-on), not the vanilla forward pass.

---

## 1. What CS336 actually requires (every deliverable)

Priority key: **LOAD-BEARING** = the engine depends on it (logits/sampling/tokenizer/optimizer that L5 inspects). **COURSE-ROTE** = required to pass tests, but a means to the substrate, not a differentiator — implement correctly, do not over-invest. **SKIP** = do the minimum to get a working LM; do not chase.

All adapter functions live in `tests/adapters.py` (the assignment's glue layer; named per the PDF "implement the test adapter at [`adapters.X`]"). Point totals are the PDF's.

| # | Deliverable (PDF task name) | PDF section | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|---|
| 1 | `unicode1` / `unicode2` — Unicode & UTF-8 written Qs (3 short answers) | §2.1–2.2 | — (written) | 0.5 h | COURSE-ROTE |
| 2 | **`train_bpe`** — train byte-level BPE (vocab+merges, special-token boundaries, fast incremental pair-count) | §2.4–2.5 | `run_train_bpe` | 6–10 h | **LOAD-BEARING** |
| 3 | `train_bpe_tinystories` — train BPE on TinyStories, vocab 10K, `<\|endoftext\|>` (≤30 min, ≤30 GB) | §2.5 | — (run + write-up) | 1–2 h | COURSE-ROTE |
| 4 | `train_bpe_expts_owt` — train BPE on OpenWebText, vocab 32K (≤12 h, ≤100 GB) | §2.5 | — (run + write-up) | 1 h (mostly wall-clock) | SKIP (run only if compute is free) |
| 5 | **`tokenizer`** — `Tokenizer` class: `encode` / `encode_iterable` / `decode` / `from_files`, special-token handling, U+FFFD on malformed bytes | §2.6 | `get_tokenizer` | 4–6 h | **LOAD-BEARING** |
| 6 | `tokenizer_experiments` — compression ratio, cross-tokenizer test, throughput, encode datasets → `uint16` `.npy` | §2.7 | — (run + write-up) | 1–2 h | COURSE-ROTE |
| 7 | **`linear`** — `Linear` (no bias, `W` not `Wᵀ`, trunc-normal init) | §3.3.2 | `run_linear` | 0.5 h | **LOAD-BEARING** |
| 8 | **`embedding`** — `Embedding` lookup module | §3.3.3 | `run_embedding` | 0.5 h | **LOAD-BEARING** |
| 9 | **`rmsnorm`** — RMSNorm (upcast→fp32→downcast) | §3.4.1 | `run_rmsnorm` | 0.5 h | **LOAD-BEARING** |
| 10 | **`positionwise_feedforward`** — SwiGLU FFN (W1,W2,W3; d_ff≈⅛/3·d_model rounded to ×64) | §3.4.2 | `run_swiglu` | 1 h | **LOAD-BEARING** |
| 11 | **`rope`** — `RotaryPositionalEmbedding` (precomputed cos/sin buffer, `persistent=False`) | §3.4.3 | `run_rope` | 1.5 h | **LOAD-BEARING** |
| 12 | **`softmax`** — numerically-stable softmax over a given dim | §3.4.4 | `run_softmax` | 0.25 h | **LOAD-BEARING** |
| 13 | **`scaled_dot_product_attention`** — SDPA with optional boolean mask (−∞ on False) | §3.4.4 | `run_scaled_dot_product_attention` | 1.5 h | **LOAD-BEARING** |
| 14 | **`multihead_self_attention`** — causal MHA w/ RoPE; head = batch dim; 3 projections (Q,K,V) + O | §3.4.5 | `run_multihead_self_attention` | 2–3 h | **LOAD-BEARING** |
| 15 | **`transformer_block`** — pre-norm block: `x + MHA(RMSNorm(x))`, `x + FFN(RMSNorm(x))` | §3.5 | `run_transformer_block` | 1 h | **LOAD-BEARING** |
| 16 | **`transformer_lm`** — full LM: embed → N blocks → final RMSNorm → LM head (no weight tying required) | §3.5 | `run_transformer_lm` | 1.5 h | **LOAD-BEARING** |
| 17 | `transformer_accounting` — params + memory + FLOPs accounting for GPT-2 XL & friends (written) | §3.5 | — (written) | 2 h | COURSE-ROTE (but high-signal — it *is* the A2/interview memory-math drill) |
| 18 | **`cross_entropy`** — stable CE from logits (cancel log-exp, average over batch) | §4.1 | `run_cross_entropy` | 0.5 h | **LOAD-BEARING** |
| 19 | `sgd` / `learning_rate_tuning` — toy SGD optimizer + LR-divergence written Q | §4.2 | (SGD class; written) | 0.5 h | COURSE-ROTE |
| 20 | **`adamw`** — AdamW as `torch.optim.Optimizer` subclass (decoupled wd, bias-corrected α_t) | §4.3 | `get_adamw_cls` | 1.5 h | **LOAD-BEARING** |
| 21 | `adamw_accounting` — peak-memory & FLOPs algebra for training (written) | §4.3 | — (written) | 1.5 h | COURSE-ROTE (high-signal; same memory math as A2.2 "train a 100B model") |
| 22 | **`learning_rate_schedule`** — cosine schedule w/ linear warmup (3-phase) | §4.4 | `get_lr_cosine_schedule` | 0.5 h | **LOAD-BEARING** |
| 23 | **`gradient_clipping`** — clip by global ℓ₂-norm (ε=1e-6) | §4.5 | `run_gradient_clipping` | 0.5 h | **LOAD-BEARING** |
| 24 | **`data_loading`** — sample (input, next-token) batches from a token array, to device | §5.1 | `run_get_batch` | 0.5 h | **LOAD-BEARING** |
| 25 | **`checkpointing`** — `save_checkpoint` / `load_checkpoint` (model+optim+iter) | §5.2 | `run_save_checkpoint`, `run_load_checkpoint` | 0.5 h | **LOAD-BEARING** |
| 26 | **`training_together`** — full training script (config, `np.memmap`, checkpoint, logging) | §5.3 | — (script) | 2–3 h | **LOAD-BEARING** |
| 27 | `experiment_log` — experiment-tracking infra (curves vs steps & wall-clock) | §7.1 | — (infra) | 1 h | COURSE-ROTE |
| 28 | **`decoding`** — generate: prompt completion, max-tokens, temperature, top-p (nucleus), stop on `<\|endoftext\|>` | §6 | — (function) | 1.5 h | **LOAD-BEARING** |
| 29 | `learning_rate` (TinyStories) — LR sweep; reach **val loss ≤ 1.45** (≤2.00 on CPU/MPS) | §7.2 | — (run + curves) | compute-bound | COURSE-ROTE |
| 30 | OpenWebText leaderboard — perplexity chase under a time budget, submit PR | §1, §7 | — (run) | open-ended | **SKIP** |

---

## 2. The equations/algorithms that matter (senior extraction)

Six load-bearing primitives. State them cold; this is the 45-minute whiteboard set.

1. **BPE merge rule (greedy, deterministic).** Initialize vocab = 256 bytes (+ specials). Repeat until `|vocab| = vocab_size`: count every adjacent byte-pair frequency *within pre-tokens* (never across pre-token or special-token boundaries); merge the most frequent pair, **breaking ties by the lexicographically greater pair**; append the new token; record the merge. Pre-tokenize with the GPT-2 regex and **split on special tokens first** so no merge crosses a document boundary. *Why it matters for the engine:* the tokenizer defines the vocab axis of every logit and every KL; `kl_train_infer` and `true_quality` are computed over exactly these token IDs. A wrong merge order = a different policy.

2. **RoPE rotation.** For query/key at position `i`, rotate each 2-D coordinate pair `k` by `θ_{i,k} = i / Θ^{(2k−2)/d}` using the 2×2 block `[[cos, −sin],[sin, cos]]`; apply to Q and K only (not V), independently per head. No learnable parameters; cos/sin are a precomputed `persistent=False` buffer sliced by `token_positions`. *Why:* relative positional encoding is the 2026 default; getting the per-head batching and the position slice right is the most common subtle bug, and position handling is what makes KV-cache / multi-turn rollouts correct.

3. **Scaled dot-product attention.** `Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask)V`, where the boolean mask adds `−∞` to disallowed `(i,j)` (causal: `j ≤ i`). Subtract row-max before `exp` for stability. *Why:* the core mixing operation; causal masking *is* the autoregressive pretraining objective — a leak trivializes next-token prediction and silently inflates any reward.

4. **SwiGLU FFN.** `FFN(x) = W2 · (SiLU(W1 x) ⊙ W3 x)`, `SiLU(x) = x·σ(x)`, `d_ff ≈ (8/3)·d_model` rounded to a multiple of 64, no bias. *Why:* the 2026-default activation (Llama/Qwen/PaLM); the gating is where most of the non-attention FLOPs and params live, which the FLOPs-accounting and `adamw_accounting` problems force you to quantify.

5. **AdamW update (decoupled weight decay).** `m ← β₁m + (1−β₁)g`; `v ← β₂v + (1−β₂)g²`; bias-corrected step `α_t = α·√(1−β₂ᵗ)/(1−β₁ᵗ)`; `θ ← θ − α_t · m/(√v + ε)`; **then** `θ ← θ − αλθ` (decay decoupled from the gradient, *not* added to `g`). Defaults: `β₂=0.95, wd=0.1` for LMs. *Why:* the optimizer you own; its state is 2×params of memory (the dominant term in the training memory budget) and the thing whose moments you'd checkpoint/restore in any resumed RL run.

6. **Cross-entropy / loss-at-init.** `ℓ_i = −log softmax(o_i)[x_{i+1}]`, computed by cancelling log and exp (`logsumexp(o) − o[target]`), averaged over batch. **At initialization, a correct LM has CE ≈ log(vocab_size)** (uniform prediction). For vocab 10K, that is `log(10000) ≈ 9.21` nats. *Why:* this single number is the cheapest correctness oracle in the whole stack (discipline #1); it catches head/embedding/masking bugs before you waste a training run, and it is the baseline against which every reward/quality number is read.

---

## 3. Map to reasoningLLM_scratch source files

reasoningLLM's clean-room layout is `src/reasoning_llm/{algos,rewards,envs,rollout,scaling,data,utils}/`. A1 has **no dedicated `model/` package in the v0.1.0 frozen scope** — L1's deliverable is "the policy served in rollouts," i.e. the substrate other layers consume. The mapping below states where each A1 piece *touches* the engine; "course-only" pieces stay in your A1 scratch package and are not promoted into `src/reasoning_llm/`.

| CS336 deliverable | `src/reasoning_llm/...` touch-point | Keep / adapt for the engine vs course-only |
|---|---|---|
| BPE `train_bpe` + `Tokenizer` | the vocab/encode layer every `envs/*` task and `rollout/*` uses; `utils/monitors.py` computes KLs over this vocab | **Keep.** The tokenizer defines the token axis of `kl_train_infer` and `true_quality`. Adapt: ensure `<\|endoftext\|>` and any reasoning special tokens are stable across train/infer engines. |
| `transformer_lm` (+ block, MHA, RoPE, SwiGLU, RMSNorm, Linear, Embedding) | **the policy object** the rollout engine serves; the thing whose logits `rewards/hack_detector.py` and `utils/monitors.py` inspect | **Keep the architecture knowledge; in v0.1.0 the served policy is typically a Qwen3-class HF model, not your hand-rolled LM.** Your from-scratch build is the *mastery artifact* that lets you read/patch logits, add GQA/MoE, and trust the sampler. Adapt: GQA + MoE are the v0.2.0 extensions of this same file. |
| `softmax`, `decoding` (temperature, top-p) | the **sampler** behind `rollout/sglang_client.py`; exploration knob of every rollout | **Keep.** Temperature/top-p directly shape the rollout distribution and therefore `reward` and `hack_rate`; the sampler must match between train and infer engines or `kl_train_infer` is polluted by a sampling mismatch, not real drift. |
| `cross_entropy`, `adamw`, `lr_schedule`, `gradient_clipping` | training-time only; in RL these become the policy-gradient update in `algos/` | **Adapt.** CE → the SFT/log-prob term; AdamW + clip + schedule → the optimizer of the RL fine-tune. The loss-at-init and overfit-one-batch *disciplines* (below) port directly into `algos/` tests. |
| `data_loading`, `checkpointing`, `training_together` | the harness pattern reused by `algos/` training scripts; manifests in `utils/` | **Adapt.** `np.memmap` loading + checkpoint/restore is the same plumbing an RL run needs to resume; `experiment_log` → `utils/monitors.py` logging conventions. |
| `transformer_accounting`, `adamw_accounting` | feeds the A2.2 "100B memory-math one-pager" (`utils/monitors.py` memory budget) | **Keep the method, course-only artifact.** The FLOP/memory algebra is the verbatim interview answer; not a runtime file. |

---

## 4. Map to core context docs

- **`UNIFIED_FRONTIER_PROJECT_SPEC.md §3 (L1 · A1 brief).** L1 = "you own the policy." Core = BPE + Transformer from scratch + AdamW + CE. Two senior add-ons defined there, both **v0.2.0 / additive — do not pull into the 14-day v0.1.0**:
  - **A1.1 — MoE layer from scratch:** top-k gating + **aux-loss-free load balancing** (DeepSeek-style) + z-loss + per-expert token histogram + router-entropy logging. Falsifiable prediction: on a tiny 8-of-32 MoE, router entropy stays `> 0.9·log(K)`; collapse to a few experts ⇒ balancer broken. Kill: >2 debug days ⇒ fall back to dense+GQA (MoE becomes v0.2.0-B).
  - **A1.2 — the discipline harness baked into your own code:** loss-at-init ≈ log V, overfit-one-batch, fixed-seed reproducibility. Spec calls this "the single highest-leverage debugging test in ML" and the engineering-quality signal labs silently screen for.
  - Interview leverage (spec §3): "implement MHA / a Transformer layer," tensor-shape & masking fluency, "explain MoE routing and load balancing," "what's the loss at init?"
- **`UNIFIED_FRONTIER_PROJECT_SPEC.md §2` (the stack diagram).** L1 SUBSTRATE feeds "the model the rollout engine serves"; the through-line `true_quality_gap = reward − true_quality (+ hack_rate, kl_train_infer)` is owned end-to-end *because* L1 gives you the logits/tokenizer/sampler.
- **`UNIFIED_FRONTIER_PROJECT_SPEC.md §5` (evidence ledger / role-targeting).** L1 + L2 ⇒ "Pretraining / Core Modeling RE" lead artifact = MoE-from-scratch + memory math. Public GitHub + green CI from day one (clean engineering = the differentiator). A1 alone is parity; the add-ons + capstone are the low-supply corner.
- **`CAPSTONE_AND_STUDY_PLAN.md §3` (row A1).** "A1 — Basics → the policy substrate: attention, tokenizer, sampling, AdamW → you *own* the model the rollout engine serves; you can't diagnose hacking without understanding logits/sampling end-to-end. Lecture backing: CS336 L2, L3, L4." Threading order: **A1 → A2 → A5 → A3 + A4 → VERA**.
- **`STUDY_PLAN_2026.md §0.8` (reconciled schedule).** A1 lands on **Day 9** (A1 Transformer substrate + batched generation + **L10 inference**, ship target `rollout/sglang_client.py`) and **Day 11** (A1 BPE + AdamW + tiny LM; wire the `kl_train_infer` monitor, ship target `utils/monitors.py` HALT@0.10). Companion table: Day 9 → "implement a Transformer"; KV-cache (xAI). Day 11 → "implement MHA/Transformer in ~45 min" (Generalist RE). Note: BPE training is split to Day 11; the architecture + sampler land Day 9 because they front-feed the rollout/serving layer.
- **Repo `CLAUDE.md` (L1 row + disciplines).** L1 row: "BPE · Transformer · GQA/RoPE/SwiGLU · (MoE) → the policy served in rollouts." Disciplines #1 (loss-at-init ≈ log V), #2 (overfit-one-batch), #3 (fixed-seed) are **A1.2 made native** — write each invariant as a test, then make it pass; green-CI rule (ruff/pyright/pytest).

---

## 5. The frontier 2026 lens

**Commoditized vs scarce.** A vanilla dense decoder Transformer is *commoditized* — `torch.compile` already emits competent kernels, every candidate can produce a forward pass, and "implement a Transformer" is table-stakes, not a differentiator. What is **scarce**: (a) MoE routing + load-balancing that does not collapse; (b) GQA done correctly (KV-cache economics); (c) the **discipline harness** — a repo where loss-at-init and overfit-one-batch are *native tests*, not afterthoughts. Per the spec (§1.5, Neel Nanda), weak engineering is the most common silent rejection of promising researchers; clean, reproducible substrate code is disproportionately valuable.

**The 2026 default decoder (baseline, don't be a hero — copy it):** **GQA + RoPE + SwiGLU + RMSNorm + AdamW(β₂=0.95, wd=0.1)**, pre-norm, no bias, untied or tied embeddings. CS336 A1 hands you all of this except GQA (it builds full MHA; GQA = K/V head-sharing, a small adaptation). **MoE is the frontier default** — every open-weight flagship (DeepSeek-V3/R1, Qwen3, Llama 4, Kimi K2) is sparse; the sweep found zero dense flagships.

**The add-on that turns rote A1 into a research signal:**
- **A1.1 (MoE-from-scratch)** with aux-loss-free balancing + router-entropy logging — this is where L1 connects to the hottest 2025-26 failure mode: **MoE breaks GRPO via routing mismatch** (GSPO measured ~10% of activated experts flipping after one gradient step; R3 measured ~94% of tokens differing in ≥1 layer between train and infer engines, causing 3/3 GRPO collapse). That collapse is diagnosed by your `kl_train_infer` pillar — so the MoE substrate you build in A1.1 is the policy for the v0.2.0-B headline result. **Still v0.2.0; do not smuggle into v0.1.0.**
- **A1.2 (discipline harness)** — loss-at-init + overfit-one-batch baked into your own test suite. This is the cheapest, highest-leverage "engineering quality" signal in the whole project and it is *free* once you've written the model.

---

## 6. Prioritization verdict — what matters / what to skip

**The ~20% that is load-bearing for the through-line (build these to mastery, whiteboard-cold):**
the tokenizer (`train_bpe` + `Tokenizer`), the full architecture chain (`linear` → `embedding` → `rmsnorm` → `rope` → `swiglu` → `softmax` → `scaled_dot_product_attention` → `multihead_self_attention` → `transformer_block` → `transformer_lm`), `cross_entropy`, `adamw`, `lr_schedule`, `gradient_clipping`, `data_loading`, `checkpointing`, `training_together`, and **`decoding` (temperature + top-p)**. These *are* the policy/logits/sampler/optimizer the RLVR engine inspects. Bake the **A1.2 disciplines** (loss-at-init ≈ log V, overfit-one-batch, fixed seed) directly into their tests.

**Course-rote (implement correctly for tests, then move on — do not over-invest):**
the `unicode1/2` written Qs; the SGD toy optimizer + `learning_rate_tuning`; `experiment_log`; the BPE training *runs* on TinyStories/OWT and `tokenizer_experiments` write-ups; `learning_rate` sweep to val-loss ≤ 1.45. **Exception — treat `transformer_accounting` and `adamw_accounting` as high-signal even though they're "written":** they *are* the A2.2 "100B memory-math one-pager" and a verbatim interview gate ("how would you train a 100B model?"). Do them properly once.

**Explicit SKIP list:**
1. **The OpenWebText leaderboard / perplexity hyperparameter-chase** beyond a single working LM. It is pure compute-spend with zero engine signal; a tuned val-loss number does not move `true_quality_gap`.
2. **`train_bpe_expts_owt`** (32K-vocab OWT BPE) unless compute is already free — TinyStories 10K is sufficient to prove the tokenizer.
3. **C++/Rust BPE speedups** (the PDF's optional `cppy`/`nanobind`/PyO3 path) — `multiprocessing` pre-tokenization is enough for the resource budget; the systems win belongs in A2, not here.
4. **GQA/MoE in v0.1.0** — keep them as the documented v0.2.0 axes (ADR stubs), per frozen scope; building them now is a G6 ship-cadence risk.

---

## 7. Build checklist (ordered, with discipline gates)

Build order respects data dependencies; discipline gates (**◆**) are inserted where they catch the most bugs for the least effort.

1. **BPE training** (`train_bpe`): pre-tokenize with the GPT-2 regex via `re.finditer`; split on special tokens *first* (`re.split` on `"|".join(re.escape(s))`); incremental pair-count cache; deterministic lexicographic tie-break. **◆ predict-before-you-run:** assert the `bpe_example` from the PDF (corpus → `st, est, ow, low, west, ne`) reproduces exactly before touching TinyStories.
2. **Tokenizer** (`Tokenizer`): `encode` / `decode` round-trip; `encode_iterable` (lazy, memory-bounded); `from_files`; `errors='replace'` (U+FFFD) on decode. **◆ fixed-seed reproducibility:** assert `decode(encode(s)) == s` on a Unicode-heavy fixture.
3. **`Linear`, `Embedding`** (no bias, `W` stored not `Wᵀ`, trunc-normal init).
4. **`RMSNorm`** (upcast→fp32→compute→downcast to original dtype).
5. **`softmax`** (subtract max) → **`scaled_dot_product_attention`** (boolean mask, −∞ on False; test 3-D and 4-D).
6. **`RoPE`** (precomputed `persistent=False` cos/sin buffer; slice by `token_positions`; apply to Q,K only, head as batch dim).
7. **SwiGLU `positionwise_feedforward`** (d_ff = round-to-×64 of (8/3)·d_model).
8. **`multihead_self_attention`** (causal mask via `torch.triu`; RoPE per head; 3 projections + O).
9. **`transformer_block`** (pre-norm: `x + MHA(RMSNorm(x))`, then `x + FFN(RMSNorm(x))`).
10. **`transformer_lm`** (embed → N blocks → final RMSNorm → LM head). **◆ loss-at-init:** a fresh LM's `cross_entropy` over random data must be `≈ log(vocab_size)` (e.g. ≈9.21 for 10K). If not, the head/embedding/masking is wrong — fix before proceeding. *(This is repo discipline #1 and add-on A1.2.)*
11. **`cross_entropy`** (logsumexp form; average over batch) — needed by the loss-at-init check, so co-develop with step 10.
12. **`adamw`** (decoupled wd; bias-corrected α_t; `self.state` per-param moments) → **`gradient_clipping`** (global ℓ₂, ε=1e-6) → **`lr_cosine_schedule`** (warmup/anneal/post).
13. **`data_loading`** (`np.memmap` mode, dtype match, to-device) → **`checkpointing`** (model+optim+iter).
14. **`training_together`**: wire it all. **◆ overfit-one-batch:** before any real run, drive train loss to ~0 on a single batch. If it won't, the optimizer/data/loss wiring is broken — *not* the data. *(Repo discipline #2; add-on A1.2.)*
15. **TinyStories run** (vocab 10K, ctx 256, d_model 512, d_ff 1344, 4 layers, 16 heads, Θ=10000, ~17M params). **◆ predict-before-you-run:** write the target val-loss (≤1.45 GPU / ≤2.00 CPU-MPS) *first*; it's your debugging anchor.
16. **`decoding`** (temperature + top-p nucleus + stop on `<\|endoftext\|>`); sanity-read generated TinyStories text. This is the sampler the rollout engine will reuse — verify temperature/top-p semantics match what the serving engine expects.
17. **CONNECT:** one sentence — which `reasoning_llm` file did this move? (e.g., "the sampler now matches `rollout/sglang_client.py`'s decode contract"). Green tests → atomic commit (`<area>: <imperative>`).

---

## 8. Open questions / ADR triggers

Candidates for `docs/adr/` (log a stub; do not relitigate in code):

1. **Dense vs MoE substrate for v0.1.0.** Frozen decision: dense (+GQA-ready). MoE-from-scratch (A1.1) is **v0.2.0-B**, gated by the ">2 debug days ⇒ fall back to dense" kill criterion. ADR should record *why* MoE is deferred (G6 ship-cadence) and what would promote it (the `kl_train_infer`-collapse headline). → `ADR: dense-vs-MoE policy substrate (MoE = v0.2.0-B)`.
2. **Tokenizer choice for the served policy.** A1 builds a byte-level BPE from scratch, but the v0.1.0 rollout policy is a Qwen3-class HF model with *its own* tokenizer. ADR: does the engine use the HF model's native tokenizer (recommended — required for `kl_train_infer` to compare like-for-like vocabs) or the hand-rolled one (only for the from-scratch tiny-LM smoke)? Mismatched vocabs make `kl_train_infer` meaningless. → `ADR: tokenizer of record for the served policy`.
3. **GQA vs full MHA in the substrate.** 2026 default is GQA (KV-cache economics); A1 builds full MHA. ADR: when does the served policy adopt GQA, and does the from-scratch build add it (cheap: K/V head-sharing) as the GQA-fluency artifact? → `ADR: GQA adoption in the policy substrate`.
4. **Sampler parity (train vs infer).** Temperature/top-p must be identical between the training-time sampler and `rollout/sglang_client.py`, or `kl_train_infer` measures a sampling-config mismatch rather than real engine drift. ADR: pin the canonical decode config and assert parity in `utils/monitors.py`. → `ADR: train/infer sampler parity contract`.
5. **Weight tying (embedding ↔ LM head).** A1 does not require it; many 2026 small models tie. Low-stakes, but record the choice so checkpoint shapes are stable across the smoke run. → optional ADR note.
