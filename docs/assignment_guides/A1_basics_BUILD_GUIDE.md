# A1 — Basics · BUILD GUIDE (CS336 → the LM substrate)

> **📖 Read first (slides → this build):** Lectures **1 → 2 → 3 → 4** — tokenization (py) · resource
> accounting (py) · architecture (pdf) · attention alternatives + MoE (pdf). Full map + read-order:
> [`../LECTURE_MAP.md`](../LECTURE_MAP.md).

> **STATUS: ✅ A1 substrate complete** — `tokenizer.py`, `model.py`, `moe.py`, `optim.py`,
> `train.py`, `sampling.py`, `utils/seeding.py` built, tested, green (see [`../STATUS.md`](../STATUS.md)).
> Discipline tests baked in: loss-at-init≈logV, causal-no-leak, overfit-one-batch, seed-repro.

> **One-liner.** CS336 A1 ("Building a Transformer LM") makes you build, from scratch, the *entire
> language-model substrate* — a byte-level BPE tokenizer, a pre-norm decoder Transformer (RMSNorm ·
> RoPE · SwiGLU · causal MHA), cross-entropy, AdamW, the cosine+warmup schedule, gradient clipping,
> a memory-mapped training loop, checkpointing, and a temperature/top-p decoder — and train a tiny
> LM on TinyStories. This is the **foundation layer**: every later assignment (systems kernels,
> scaling laws, data pipelines, RL post-training) sits on top of a model, a tokenizer, and an
> optimizer you own line by line. **The organizing principle:** own every primitive to production
> standard and be able to whiteboard it cold — "implement a Transformer from scratch in ~45 min" is
> table-stakes for a frontier-lab interview, so the bar is not "it passes tests" but "you can derive
> it, predict its loss at init, and defend every shape and mask." The commoditized part is the
> vanilla forward pass; the scarce part is what you bake *around* it — the discipline harness
> (loss-at-init, overfit-one-batch, fixed-seed) and the completed extensions (GQA, MoE with
> aux-loss-free balancing) that turn a rote build into an engineering signal.

---

## 1. What CS336 actually requires (every deliverable)

Priority key: **LOAD-BEARING** = a primitive every later assignment depends on; you must own it to
mastery and be able to whiteboard it cold (the model, tokenizer, optimizer, sampler). **COURSE-ROTE**
= required to pass tests or the handout, but a means to the substrate, not a differentiator —
implement correctly, do not over-invest. **SKIP** = do the minimum to get a working LM; do not chase.

All adapter functions live in `tests/adapters.py` (the assignment's glue layer; named per the PDF
"implement the test adapter at [`adapters.X`]"). Point totals are the PDF's.

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
| 21 | `adamw_accounting` — peak-memory & FLOPs algebra for training (written) | §4.3 | — (written) | 1.5 h | COURSE-ROTE (high-signal; same memory math as the A2 "train a 100B model" interview answer) |
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

Seven load-bearing primitives. State them cold; this is the 45-minute whiteboard set. Each "why"
clause names the subtle bug an interviewer probes for — that is the actual signal.

1. **BPE merge rule (greedy, deterministic).** Initialize vocab = 256 bytes (+ specials). Repeat
   until `|vocab| = vocab_size`: count every adjacent byte-pair frequency *within pre-tokens* (never
   across pre-token or special-token boundaries); merge the most frequent pair, **breaking ties by
   the lexicographically greater pair**; append the new token; record the merge. Pre-tokenize with
   the GPT-2 regex and **split on special tokens first** so no merge crosses a document boundary.
   *Why it matters:* the merge order is the one place where a "looks correct" tokenizer silently
   diverges from the reference — a wrong tie-break or a merge that crosses a boundary yields a
   different vocab and fails the determinism test. The tokenizer defines the token axis of every
   logit downstream, so a wrong merge order is a wrong model.

2. **RoPE rotation.** For query/key at position `i`, rotate each 2-D coordinate pair `k` by
   `θ_{i,k} = i / Θ^{(2k−2)/d}` using the 2×2 block `[[cos, −sin],[sin, cos]]`; apply to Q and K
   only (not V), independently per head. No learnable parameters; cos/sin are a precomputed
   `persistent=False` buffer sliced by `token_positions`. *Why:* relative positional encoding is the
   2026 default, and the per-head batching plus the `token_positions` slice are the single most
   common subtle bug — the classic interview trap. Correct position handling is also what makes a
   KV-cache (incremental decoding, the A2 deliverable) produce identical logits to a full forward.

3. **Scaled dot-product attention.** `Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask)V`, where the
   boolean mask adds `−∞` to disallowed `(i,j)` (causal: `j ≤ i`). Subtract row-max before `exp` for
   stability. *Why:* the core mixing operation, and the canonical "your training blew up" bug —
   masked positions must become `−∞` *before* softmax (not zeroed after), and the mask must broadcast
   to the score shape. A causal leak trivializes next-token prediction: the loss falls implausibly
   fast because the model can see the answer. (This is also the kernel A2's FlashAttention reimplements
   — owning the naïve form is the prerequisite for owning the tiled one.)

4. **SwiGLU FFN.** `FFN(x) = W2 · (SiLU(W1 x) ⊙ W3 x)`, `SiLU(x) = x·σ(x)`,
   `d_ff ≈ (8/3)·d_model` rounded to a multiple of 64, no bias. *Why:* the 2026-default activation
   (Llama/Qwen/PaLM); the gating is where most of the non-attention FLOPs and params live, which the
   FLOPs-accounting (#17) and `adamw_accounting` (#21) problems force you to quantify — and which the
   "how many params/FLOPs in your model?" interview question expects you to derive on the spot.

5. **AdamW update (decoupled weight decay).** `m ← β₁m + (1−β₁)g`; `v ← β₂v + (1−β₂)g²`;
   bias-corrected step `α_t = α·√(1−β₂ᵗ)/(1−β₁ᵗ)`; `θ ← θ − α_t · m/(√v + ε)`; **then**
   `θ ← θ − αλθ` (decay decoupled from the gradient, *not* added to `g`). Defaults: `β₂=0.95, wd=0.1`
   for LMs. *Why:* the optimizer you own; "what does the W in AdamW change vs Adam, and why?" is a
   stock question. Its state is 2×params of memory (`m` and `v`), the dominant term in the training
   memory budget you compute in #21 and re-derive for the A2 100B-model answer.

6. **Cross-entropy / loss-at-init.** `ℓ_i = −log softmax(o_i)[x_{i+1}]`, computed by cancelling log
   and exp (`logsumexp(o) − o[target]`), averaged over batch. **At initialization, a correct LM has
   CE ≈ log(vocab_size)** (uniform prediction). For vocab 10K, that is `log(10000) ≈ 9.21` nats.
   *Why:* this single number is the cheapest correctness oracle in the whole stack (discipline #1);
   it catches head/embedding/masking bugs before you waste a training run, and "what's the loss at
   init of a freshly initialized LM?" is a fast filter interviewers use to test whether you actually
   understand the objective.

7. **MoE routing + aux-loss-free load balancing** *(completed extension, `moe.py`).* A token routes
   to its top-`k` experts by router logits; the layer output is the gate-weighted sum of those
   experts' FFNs. Naïve top-k collapses (a few experts win every token), so balance the load
   **without an auxiliary loss** (DeepSeek-style): maintain a per-expert bias added to the router
   logits *for selection only*, nudged up for under-used experts and down for over-used ones each
   step, so the gradient path stays clean. Log a **per-expert token histogram** and **router
   entropy**. *Why:* the falsifiable check — on a small 8-of-32 MoE, router entropy should stay
   `> 0.9·log(K)`; collapse to a few experts means the balancer is broken. "Explain MoE routing and
   how you keep experts balanced" is the sparse-model interview question, and every 2026 open-weight
   flagship (DeepSeek-V3, Qwen3, Llama 4, Kimi K2) is sparse — so this is the differentiating
   primitive, not a footnote.

---

## 3. Map to `src/scratch_llm/` (what you built, where it lives)

The A1 substrate is a flat set of modules under `src/scratch_llm/` (the layered packages
`{algos,rewards,envs,rollout,scaling,data,utils,kernels}/` are for A2–A5). Each CS336 deliverable
maps to a concrete symbol you can open and read:

| CS336 deliverable | `src/scratch_llm/...` symbol(s) | Notes |
|---|---|---|
| `train_bpe` + `Tokenizer` | `tokenizer.py` — `train_bpe()`, `Tokenizer` class (`_pretokenize_counts`, `_compute_merges`) | Deterministic merges (lexicographic tie-break), special-token-first splitting, `encode`/`encode_iterable`/`decode`/`from_files`, U+FFFD on malformed bytes. |
| The full architecture chain | `model.py` — `Linear`, `Embedding`, `RMSNorm`, `softmax`, `scaled_dot_product_attention`, `RotaryPositionalEmbedding`, `silu`, `SwiGLU`, `MultiHeadSelfAttention`, `TransformerBlock`, `TransformerLM` | Pre-norm decoder; `ModelConfig` holds the shape knobs. **GQA is built in** — `MultiHeadSelfAttention` supports K/V head sharing and `KVCache` does incremental decoding (see ADR-0002). |
| `cross_entropy` | `model.py` — `cross_entropy()` | logsumexp form; the loss-at-init oracle (≈ log V) is a test against it. |
| MoE (completed extension) | `moe.py` — `Router`, `MoEFeedForward`, `MoEConfig`, `MoEStats`, `AuxOutput` | Top-k routing + aux-loss-free balancing + per-expert histogram + router-entropy logging (see ADR-0007). |
| `adamw`, `gradient_clipping`, `learning_rate_schedule` | `optim.py` — `AdamW`, `gradient_clipping()`, `cosine_lr()` | AdamW = `torch.optim.Optimizer` subclass, decoupled wd, bias-corrected α_t; clip by global ℓ₂; 3-phase cosine. |
| `decoding`, `softmax` (sampler side) | `sampling.py` — `generate()`, `generate_with_logprobs()`, `SamplingParams`, `_top_p_filter`, `_sample_next` | Temperature + top-p nucleus + stop-on-`<\|endoftext\|>`; `generate_with_logprobs` exposes per-token log-probs (see ADR-0006). |
| `data_loading`, `checkpointing`, `training_together` | `train.py` — `get_batch()`, `save_checkpoint()`, `load_checkpoint()`, `train()`, `TrainConfig` | `np.memmap` batch sampling, model+optim+iter checkpoint/restore, the wired training loop. |
| `transformer_accounting`, `adamw_accounting` | *(written deliverables — no runtime file)* | The params/memory/FLOPs algebra; keep the worked answers, they recur as the A2 memory-math drill. |

For the full build spine (A1→A5, build order, discipline gates) see
[`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md). The authoritative spec and test oracle is
the official scaffold at `../../../lectures/assignment1-basics/` (the handout PDF + `tests/adapters.py`);
implement against those adapters, do not copy solutions.

---

## 4. The frontier 2026 lens

**Commoditized vs scarce.** A vanilla dense decoder Transformer is *commoditized* — `torch.compile`
already emits competent kernels, every candidate can produce a forward pass, and "implement a
Transformer" is table-stakes, not a differentiator. What is **scarce** and what this build deliberately
owns: (a) **MoE routing + load-balancing that does not collapse** — built in `moe.py` with
aux-loss-free balancing and router-entropy logging; (b) **GQA done correctly** (KV-cache economics) —
built in `model.py`; (c) the **discipline harness** — a repo where loss-at-init and overfit-one-batch
are *native tests*, not afterthoughts. Weak engineering is the most common silent rejection of
otherwise-strong candidates; clean, reproducible substrate code is disproportionately valuable.

**The 2026 default decoder (baseline, don't be a hero — copy it):** **GQA + RoPE + SwiGLU + RMSNorm +
AdamW(β₂=0.95, wd=0.1)**, pre-norm, no bias, untied or tied embeddings. CS336 A1 builds full MHA;
this repo extends it to **GQA** (K/V head-sharing, a small adaptation already in `model.py`). **MoE is
the frontier default** — every open-weight flagship (DeepSeek-V3/R1, Qwen3, Llama 4, Kimi K2) is
sparse; the sweep found zero dense flagships, which is why the MoE extension is built rather than
deferred.

**The completed extensions that turn rote A1 into a research signal:**
- **MoE-from-scratch** (`moe.py`) with aux-loss-free balancing + router-entropy logging. This is where
  the substrate connects to the hottest 2025–26 sparse-model failure mode: routing can flip between a
  training step and an inference engine, and a candidate who has *implemented* top-k routing and a
  balancer can reason about it concretely instead of hand-waving. Falsifiable: router entropy
  `> 0.9·log(K)` on a tiny 8-of-32 MoE, else the balancer is broken.
- **The discipline harness** — loss-at-init ≈ log V, overfit-one-batch, fixed-seed reproducibility,
  baked into the test suite. The cheapest, highest-leverage "engineering quality" signal in the whole
  project, and free once the model exists.

**Interview leverage this assignment buys you:** "implement MHA / a Transformer layer in ~45 min,"
tensor-shape & masking fluency, "what's the loss at init?", "what does the W in AdamW change?",
"explain MoE routing and load balancing," and the params/memory/FLOPs accounting that becomes the
"how would you train a 100B model?" answer.

---

## 5. Prioritization verdict — what matters / what to skip

**The ~20% that is load-bearing (build these to mastery, whiteboard-cold):**
the tokenizer (`train_bpe` + `Tokenizer`), the full architecture chain (`linear` → `embedding` →
`rmsnorm` → `rope` → `swiglu` → `softmax` → `scaled_dot_product_attention` →
`multihead_self_attention` → `transformer_block` → `transformer_lm`), `cross_entropy`, `adamw`,
`lr_schedule`, `gradient_clipping`, `data_loading`, `checkpointing`, `training_together`, and
**`decoding` (temperature + top-p)**. These are the model/optimizer/sampler every later assignment
builds on. Bake the **disciplines** (loss-at-init ≈ log V, overfit-one-batch, fixed seed) directly
into their tests.

**Completed extensions (built, not deferred):** **GQA** (K/V head-sharing + KV-cache in `model.py`)
and **MoE** (top-k routing + aux-loss-free balancing in `moe.py`). Present these as what you built
beyond the CS336 core — they are the scarce, differentiating part of the substrate.

**Course-rote (implement correctly for tests, then move on — do not over-invest):**
the `unicode1/2` written Qs; the SGD toy optimizer + `learning_rate_tuning`; `experiment_log`; the
BPE training *runs* on TinyStories and `tokenizer_experiments` write-ups; the TinyStories
`learning_rate` sweep to val-loss ≤ 1.45. **Exception — treat `transformer_accounting` and
`adamw_accounting` as high-signal even though they're "written":** they are the params/memory/FLOPs
math that recurs as the A2 "how would you train a 100B model?" interview answer. Do them properly
once.

**Explicit SKIP list (3 items):**
1. **The OpenWebText leaderboard / perplexity hyperparameter-chase** beyond a single working LM. Pure
   compute-spend with no mastery carry; a tuned val-loss number teaches nothing the working LM didn't.
2. **`train_bpe_expts_owt`** (32K-vocab OWT BPE) unless compute is already free — TinyStories 10K is
   sufficient to prove the tokenizer.
3. **C++/Rust BPE speedups** (the PDF's optional `cppy`/`nanobind`/PyO3 path) —
   `multiprocessing` pre-tokenization is enough for the resource budget; the systems-optimization win
   belongs in A2, not here.

---

## 6. Build checklist (ordered, with discipline gates)

Build order respects data dependencies; discipline gates (**◆**) are inserted where they catch the
most bugs for the least effort.

1. **BPE training** (`train_bpe`): pre-tokenize with the GPT-2 regex via `re.finditer`; split on
   special tokens *first* (`re.split` on `"|".join(re.escape(s))`); incremental pair-count cache;
   deterministic lexicographic tie-break. **◆ predict-before-you-run:** assert the `bpe_example` from
   the PDF (corpus → `st, est, ow, low, west, ne`) reproduces exactly before touching TinyStories.
2. **Tokenizer** (`Tokenizer`): `encode` / `decode` round-trip; `encode_iterable` (lazy,
   memory-bounded); `from_files`; `errors='replace'` (U+FFFD) on decode. **◆ round-trip
   reproducibility:** assert `decode(encode(s)) == s` on a Unicode-heavy fixture.
3. **`Linear`, `Embedding`** (no bias, `W` stored not `Wᵀ`, trunc-normal init).
4. **`RMSNorm`** (upcast→fp32→compute→downcast to original dtype).
5. **`softmax`** (subtract max) → **`scaled_dot_product_attention`** (boolean mask, −∞ on False; test
   3-D and 4-D).
6. **`RoPE`** (precomputed `persistent=False` cos/sin buffer; slice by `token_positions`; apply to
   Q,K only, head as batch dim).
7. **SwiGLU `positionwise_feedforward`** (d_ff = round-to-×64 of (8/3)·d_model).
8. **`multihead_self_attention`** (causal mask via `torch.triu`; RoPE per head; 3 projections + O).
9. **`transformer_block`** (pre-norm: `x + MHA(RMSNorm(x))`, then `x + FFN(RMSNorm(x))`).
10. **`transformer_lm`** (embed → N blocks → final RMSNorm → LM head). **◆ loss-at-init:** a fresh
    LM's `cross_entropy` over random data must be `≈ log(vocab_size)` (e.g. ≈9.21 for 10K). If not,
    the head/embedding/masking is wrong — fix before proceeding. *(Discipline #1.)*
11. **`cross_entropy`** (logsumexp form; average over batch) — needed by the loss-at-init check, so
    co-develop with step 10.
12. **`adamw`** (decoupled wd; bias-corrected α_t; `self.state` per-param moments) →
    **`gradient_clipping`** (global ℓ₂, ε=1e-6) → **`cosine_lr`** (warmup/anneal/post).
13. **`data_loading`** (`np.memmap` mode, dtype match, to-device) → **`checkpointing`**
    (model+optim+iter).
14. **`training_together`**: wire it all. **◆ overfit-one-batch:** before any real run, drive train
    loss to ~0 on a single batch. If it won't, the optimizer/data/loss wiring is broken — *not* the
    data. *(Discipline #2.)*
15. **TinyStories run** (vocab 10K, ctx 256, d_model 512, d_ff 1344, 4 layers, 16 heads, Θ=10000,
    ~17M params). **◆ predict-before-you-run:** write the target val-loss (≤1.45 GPU / ≤2.00 CPU-MPS)
    *first*; it's your debugging anchor.
16. **`decoding`** (temperature + top-p nucleus + stop on `<\|endoftext\|>`); sanity-read generated
    TinyStories text. Verify temperature/top-p semantics behave as expected (temperature→1, top-p→1
    recovers plain sampling; top-p→0 / temperature→0 recovers greedy).
17. **Extensions (built):** **GQA** in `MultiHeadSelfAttention` (K/V head-sharing; ◆ assert it matches
    full-MHA logits when `n_kv_heads == n_heads`) and **MoE** in `moe.py` (◆ router entropy
    `> 0.9·log(K)` on a tiny 8-of-32 config — the balancer-not-collapsed check).
18. Green tests (ruff/ruff-format/pyright/`pytest -m "not gpu"`) → atomic commit
    (`<area>: <imperative>`).

---

## 7. Open questions / ADR triggers

Each of these is already recorded in [`../adr/`](../adr/) — read the ADR, don't relitigate in code:

1. **GQA vs full MHA in the substrate.** 2026 default is GQA (KV-cache economics); A1's spec builds
   full MHA. → **[`ADR-0002-gqa-in-substrate.md`](../adr/ADR-0002-gqa-in-substrate.md)** (GQA built as
   K/V head-sharing, the KV-cache-fluency artifact).
2. **MoE in the substrate.** A1's spec is dense; the frontier default is sparse. → decision to build
   it: **[`ADR-0007-moe-pulled-forward.md`](../adr/ADR-0007-moe-pulled-forward.md)** (top-k routing +
   aux-loss-free balancing in `moe.py`), with the dense-baseline rationale in
   **[`ADR-0004-dense-substrate-v0.1.0.md`](../adr/ADR-0004-dense-substrate-v0.1.0.md)**.
3. **Tokenizer of record.** Whether the from-scratch BPE or a pretrained model's tokenizer is used
   when serving a real (HF) policy in later assignments. →
   **[`ADR-0001-tokenizer-of-record.md`](../adr/ADR-0001-tokenizer-of-record.md)**.
4. **Sampler parity (train vs infer).** Temperature/top-p must be identical between the training-time
   sampler and any inference engine, or generation diverges for a config reason, not a model reason. →
   **[`ADR-0003-sampler-parity.md`](../adr/ADR-0003-sampler-parity.md)**.
5. **Log-prob convention.** How `generate_with_logprobs` defines per-token log-probs (pre/post
   temperature, which dim). → **[`ADR-0006-policy-logprob-convention.md`](../adr/ADR-0006-policy-logprob-convention.md)**.
6. **Weight tying (embedding ↔ LM head).** A1 does not require it; many 2026 small models tie. Low-
   stakes; record the choice so checkpoint shapes stay stable. → optional ADR note.
