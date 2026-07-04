# Close the Loop → Frontier Ablations (2026) — engineering spec + execution DAG

> **What this is.** The master spec for a **third front** on `scratch_llm`: adopt Karpathy's
> **nanochat** end-to-end integration spine to produce a *trained, evaluated, chat-capable* model
> from our own code, then run an **EV-ranked, pre-registered, iso-FLOP frontier-technique ablation
> study** on that working baseline. It is the repo's `PERF_ENGINEERING_SPEC`-style contract for
> this front: every rung carries a falsifiable prediction, a kill criterion, a definition-of-done,
> a file map, and primary-source citations. Runs **in parallel** with the perf-kernel curriculum
> (`performance/PERF_PLAN.md`) and DELTA (`../../DELTA.md`) — see §9.
>
> **Provenance.** Produced 2026-07-04 by a multi-agent research pass (workflow `wjzztlnz7`, 10
> agents: nanochat + repo deep-dive, four frontier streams — Muon / DeepSeek / GLM-Qwen-Kimi /
> MTP-serving — and an adversarial verification stage). Verifier result: **54 claims CONFIRMED,
> 1 REFUTED, 4 UNCERTAIN**; corrections are propagated in-line and listed in §8. Approved plan of
> record: `~/.claude/plans/misty-sniffing-cerf.md`. Claims-honesty (FOP-4) applies: a number here
> is `[FACT]` only once *measured* in `bench/RESULTS.md`; everything below is a **pre-registered
> prediction** until then.

---

## §0 — The through-line (why this front exists)

> **Own every layer of a language model AND run the whole loop end-to-end to a model that talks —
> then prove which 2026 frontier techniques carry their weight with measured, pre-registered,
> iso-FLOP ablations and a public report card.**

This is **additive** to the CS336 mandate, not a teardown. CS336 A1–A5 gave us a component museum
of the highest quality (461 CPU tests, an advanced serving stack). The load-bearing gap, read as a
principal researcher (FOP-4, *implemented ≠ measured*): **the loop has never closed.** No model has
been trained end-to-end into something that chats; every real pretrain/eval/RL run is
"rental-gated," and every serving number was measured against a **random-weight toy** (the
`bench/RESULTS.md:383` acceptance-rate confound is the proof). nanochat is the antidote: one
minimal `speedrun.sh` that runs tokenizer → pretrain → midtrain → SFT → RL → eval report card →
chat, to a talking model for ~$100. We adopt that spine over the components we already own, then
use the working baseline as the substrate for the ablation study — the part that demonstrates
production-RE skill and can contribute a real, cited result.

## §1 — Real vs toy (first-hand inventory)

| Piece | File | State (verified by reading) |
|---|---|---|
| Dense decoder (RMSNorm-fp32 · RoPE · SwiGLU · GQA · opt-in QK-norm · **tied** emb by default) | `src/scratch_llm/model.py` | Real, tested. No logit softcap, no LM z-loss, no MTP head. |
| MLA (weight-absorption identity) | `src/scratch_llm/mla.py` | **Toy** — 1 layer, not wired to attention/KV cache; float64 identity only. |
| MoE (DeepSeek fine-grained + shared + aux-loss-free bias γ=1e-3 + seq aux + z-loss) | `src/scratch_llm/moe.py` | Integrated + unit-tested; **never trained on real data**. |
| Optimizer | `optim.py` / `train.py:118` | **AdamW only** (β₂=0.95, decoupled WD, cosine). **No Muon, no param groups.** |
| Training loop | `train.py:136` | **plain fp32, no autocast/compile**; memmap batches; no real-corpus run. |
| Serving stack | `serving/` | Real + measured, but against random-weight toys (confounded — `RESULTS.md:383`). |
| RL (SFT/EI/GRPO/Dr.GRPO/DPO + r1_zero grader + envs) | `algos/`,`rewards/`,`envs/` | Built + toy-CPU; **no real-model "aha" on record**. |

**Deltas nanochat has that we lack:** Muon optimizer; **untied** embeddings (separate
`embedding_lr`/`unembedding_lr`); an integrated `speedrun.sh`; a **report card** (DCLM CORE +
`val_bpb` + ARC/MMLU/GSM8K/HumanEval); a midtraining stage + tool-use + chat UI.

## §2 — Model-size tiers (Karpathy-style)

`C = 6·N·D` used to cross-check tokens. Decision of record: **straight to the $100 d20** as the
headline artifact, with a free nano pre-flight to protect the paid run.

| Tier | Depth / size | Tokens | Compute | Cost | Purpose |
|---|---|---|---|---|---|
| **Pre-flight (nano)** | depth ~4–6, ~5–20M | ~0.2–1B | 1 GPU, minutes | ~$0–5 | Smoke-test the whole `speedrun.sh` before spending the $100. Protects against a plumbing bug. |
| **Headline = nanochat d20** | depth 20, **~561M** (n_embd≈1280) | **≈11–12B** (C≈4e19 ⇒ D≈11.9B, ~21 tok/param, Chinchilla-optimal) | 8×H100, ~2–4 h | **~$48–100** | The talking model + public report card. Target **CORE ≈ 0.256–0.269** (GPT-2 grade). |
| **Ablation / science** | ~30–300M on the standing Blackwell (24 GB) | Chinchilla-optimal/point | 1 GPU, free | $0 | Many seeds of the iso-FLOP ablations — where the science lives. |
| **Stretch (opt-in)** | d26 (~$70) → d32 (~1.9B, ~$800) | — | 8×H100 | ~$70–800 | Only if an ablation result justifies scale. d26 is a *passing suggestion* in nanochat, not a benchmarked tier (verifier UNCERTAIN). |

Tokenizer to match: GPT-4-style **Rust BPE**, vocab **65,536 = 2¹⁶** *(confirm against
`nanochat/tokenizer.py` at build time)*.

## §3 — The ranked ablation program (the heart of the front)

EV-ranked (leverage ÷ effort, weighted by frontier-lab signal). **#1–#4 are the 80/20.** Each rung
is a pre-registered, falsifiable experiment; the discipline *is* the hireable skill.

**Node pointer (current):** `F1 — MuonAdamW` (Phase 1). Advance the pointer as rungs ship.

| # | Rung | Effort | Pre-registered result + KILL | Primary sources |
|---|---|---|---|---|
| **1** | **MuonAdamW** | M | Iso-FLOP: Muon reaches AdamW's val loss with **≥15% fewer tokens** (or ≥0.02 nats lower at equal FLOPs); NS overhead **<1%** (bound T·m/B). **KILL** if saving <5% or divergence at the reused AdamW LR. | Jordan (Muon); Moonlight 2502.16982; Kimi-K2 2507.20534 |
| **2** | **MTP draft head** (V3 D=1, sequential) | L | On the trained model: **≥1.5× tokens/target-forward** on open-ended text where n-gram ≈1.0×; lossless. **KILL** if MTP aux degrades val loss >0.01 nats. | Gloeckle 2404.19737; V3 2412.19437; GLM-4.5 2508.06471 |
| **3** | **De-confound serving on a real model** | S | n-gram acceptance **collapses <10% on open text**, stays ~40–60% on code/JSON — reproducing the pre-registered prompt-dependence. | repo ledger; FA-3 2407.08608 |
| **4** | **bf16-autocast + torch.compile** (→FP8 GEMM gated by train↔serve KL) | M | compile+bf16 **+10–20% MFU**; FP8 **~1.3–1.5×** step throughput while train↔serve KL under tolerance (slower than bf16 with compile OFF). | Databricks FP8; Llama-3 MFU; SDPA≈FA-2 |
| **5** | **MLA for real** (wire `mla.py` → `ModelConfig`/block + KV cache) | M | MLA within **+0.02 val loss** of iso-param GQA-8; KV/token **~3.5× smaller**; decode uplift **≥1.2× at 16k**. **KILL** if gap >0.05 nats. | DeepSeek-V2 2405.04434 |
| **6** | **DeepSeekMoE balancing ablation** (code exists) | S | aux-loss-free ≤ large-aux val loss while router entropy **>0.9·log N**; no-balancing collapses to a few experts (kill signal). | V3 2412.19437; 2408.15664 |
| **7** | **GRPO/Dr.GRPO RLVR "aha"** (code exists) | L | Mean reward ↑ monotonically; **correct-answer response length grows**; KL(cur‖ref) bounded; a neural-RM control reward-hacks. | R1 2501.12948 |
| **8** | **DSA long-context lab** (stretch) | L | KL-aligned indexer recovers **≥95%** of dense attention mass; within **+0.03 val loss** at 8k; FLOP cut at long ctx. **KILL** if gap >0.1 nats. | V3.2-Exp 2512.02556 |
| **9** | **Logit-stability guard** | S | With `qk_norm` on, max per-head logit stays **<~30** through the ablations — confirming QK-Clip unnecessary sub-1B. | Kimi-K2 2507.20534 |

**Convergent-defaults signal:** MTP (V3 **and** GLM-4.5), Muon (nanochat + Moonlight + Kimi-K2),
aux-loss-free MoE balancing (V3 + GLM + Qwen3), QK-norm (Qwen3/Gemma3/OLMo2 — already in `model.py`)
are adopted across *independent* labs → strongest evidence they're load-bearing, not fashion.

### Rung cards (build contracts)

Each card: **Hypothesis · Prediction (falsifiable) · Kill · DoD · Files · Interview question.**

**F1 · MuonAdamW.**
- *Hypothesis.* Orthogonalizing the 2D-matrix momentum update (Newton–Schulz) + Moonlight
  RMS-matching gives a strictly better loss-per-FLOP than AdamW, reusing AdamW's LR band.
- *Prediction.* At iso-FLOP (`C=6ND` held constant via `scaling/isoflop.py`), Muon-hybrid reaches
  all-AdamW's final val loss with **≥15% fewer tokens** (or ≥0.02 nats lower at equal FLOPs), NS
  wall-clock overhead **<1%** (analytic bound `T·m/B`).
- *Kill.* Token saving <5%, OR divergence at the reused AdamW LR, OR NS overhead >3%.
- *DoD.* Muon class + `build_muon_adamw_groups()` in `optim.py`; unit tests (NS singular values ∈
  [0.7,1.3] after 5 steps; hybrid overfit-one-batch <1e-2; param-partition: no overlap, **tied
  embed/head tensor and every 1-D param in the AdamW group**); iso-FLOP curve logged to
  `bench/RESULTS.md`; green-CI.
- *Files.* `optim.py`, `train.py:118`, `tests/test_optim.py`, `scaling/isoflop.py`.
- *Interview.* "Derive the Muon update; why orthogonalize the momentum, and why does RMS-matching
  let you reuse AdamW's learning rate?"
- *Corrected math (verifier).* A full-rank orthogonalized update on an `[A,B]` matrix has RMS
  **`1/√max(A,B)`** (not `1/max(A,B)`); scale by `0.2·√max(A,B)` to land in AdamW's 0.2–0.4 band;
  WD 0.1. **Route the weight-TIED 2D tensor to AdamW despite it being 2D** (`model.py:917`).

**F2 · MTP draft head.**
- *Hypothesis.* A DeepSeek-V3-style sequential MTP module (D=1) densifies the training signal and
  doubles as a *learned* speculative-decode draft that beats n-gram on open text.
- *Prediction.* ≥1.5× tokens/target-forward at K=2 on open-ended generation (n-gram ≈1.0× there);
  2nd-token acceptance directionally toward the V3 85–90% regime; greedy output token-exact.
- *Kill.* MTP aux loss (λ=0.3→0.1) degrades base val loss by >0.01 nats.
- *DoD.* MTP module from existing primitives (shared `token_emb`+`lm_head`, RMSNorm×2, `eh_proj`
  Linear(2d→d), 1 `TransformerBlock`, aux CE on the +2 token); `MTPDrafter` implementing the
  existing `Drafter` protocol (`serving/speculative.py:59`); acceptance vs n-gram measured on
  prose vs code.
- *Files.* `model.py:910-944` (the `x`-before-`lm_head` seam at `:938`), `serving/speculative.py`.
- *Interview.* "Why does MTP both improve pretraining and give a free draft head — and why
  sequential (V3) over parallel independent heads (Gloeckle)?"

**F3 · De-confound serving.** Train the toy to a real checkpoint (short pretrain until loss ≪ log V),
re-run `bench/speculative.py` unchanged. *Prediction:* open-text n-gram acceptance collapses from
54–72% toward <10% while code/JSON stays ~40–60%. *DoD:* the falsified `P4.3.3` prompt-dependence
restored + logged. *Files:* `train.py`, `bench/speculative.py`, `RESULTS.md`.

**F4 · Cheap MFU.** `torch.autocast(bf16)` + `torch.compile` on the forward (train.py is fp32 at
`:136`); then an FP8 GEMM ablation **gated by the existing train↔serve logit-KL check**.
*Prediction:* +10–20% MFU from compile+bf16; FP8 ~1.3–1.5× while KL under tolerance (and slower
than bf16 with compile OFF). *Files:* `train.py`, `quant/`, `bench/`.

**F5 · MLA for real.** Wire `mla.py` as `attn='gqa'|'mla'` into `ModelConfig`/`TransformerBlock` +
KV cache; iso-param MLA-vs-GQA-8 at 0.2–0.5B. *Prediction:* +0.02 val loss, KV/token ~3.5× smaller
(measured 1152 B vs 4096 B), decode ≥1.2× at 16k. *Files:* `mla.py`, `model.py`, `serving/`.

**F6 · MoE balancing.** aux-loss-free (γ=1e-3) vs seq-aux (α=1e-4) vs none; fine- vs coarse-grained.
*Prediction:* aux-loss-free ≤ large-aux val loss, router entropy >0.9·log N; no-balancing collapses.
*Files:* `moe.py`, `model.py:959` (`moe_update_biases`), a new ablation harness.

**F7 · GRPO "aha".** Run `algos/grpo.py` + `envs/{countdown,gsm_math}` + `rewards/r1_zero` on the
trained base at ~0.5B; log the R1-Zero self-evolution curve (reward↑, correct-answer length growth,
bounded KL). *Neural-RM control* reward-hacks to make the rule-based point. *Files:* `algos/`,
`rewards/`, `envs/`, `utils/monitors.py`.

**F8 · DSA (stretch).** KL-aligned lightning-indexer + top-k gather; attention-recall + long-context
perplexity vs full attention. *Prediction:* ≥95% mass recovered, +0.03 val loss at 8k. *Files:* new.

**F9 · Logit guard.** Keep `qk_norm` (model.py:769) as the sub-1B guard; add MuonClip QK-Clip
(γ=min(1,τ/S_max), τ≈100) *only if* a >1B run shows logits climbing past ~100. No gold-plating.

## §4 — nanochat adoption map (best implementations → our files)

| Adopt from nanochat | Maps to |
|---|---|
| `speedrun.sh` integration spine (tokenizer→pretrain→midtrain→SFT→RL→eval→serve; scale by `--depth`) | **NEW** `scripts/speedrun.sh` orchestrating existing modules |
| Report card (DCLM CORE + `val_bpb` + ARC-E/C, MMLU, GSM8K, HumanEval, ChatCORE) | **NEW** `src/scratch_llm/eval/` |
| MuonAdamW split (matrix_lr 0.02, embedding_lr 0.2, unembedding_lr 0.004, momentum 0.95, ns_steps 5, LRs ×(n_embd/768)^-0.5) | `optim.py` + `train.py` (= F1/F4) |
| Untied embeddings | `model.py:917` (`tie_embeddings=False` default) |
| Midtraining stage (conversations + MC + tool-use) before SFT | NEW mid-train data path in `train.py` |
| KV-cached inference engine + chat UI | `serving/` (already stronger — point it at a real checkpoint + thin chat loop) |

**Philosophy to copy:** minimal, hackable, dependency-lite, one readable file per concern, one
`--depth` knob — with our green-CI + adapter-oracle rigor on top.

## §5 — Execution DAG (phases, node pointer)

- **Phase 0 — Pre-flight (local, ~$0).** Stand up `scripts/speedrun.sh` + `eval/` report card; run
  the **nano** tier (depth ~4) end-to-end on the Blackwell in minutes. **Gate:** a (bad but real)
  report card + the chat loop replies. *This protects the $100.*
- **Phase 1 — Close the loop + Muon ($100 d20).** Land **F1 (MuonAdamW)** + **F4 (bf16/compile)**
  first (they change the run you pay for), untie embeddings, then launch the **d20 8×H100**
  speedrun. **Deliverable:** a talking model + a public report card (target CORE ≈ GPT-2). Highest-EV node.
- **Phase 2 — Ablations on the real base.** **F3** de-confound → **F2** MTP → **F5** MLA-real →
  **F6** MoE balancing → **F7** GRPO "aha." Each pre-registered in `bench/RESULTS.md`, iso-FLOP,
  one variable, kill criterion checked.
- **Phase 3 — Frontier edges (opt-in).** **F8** DSA long-context; MTP/MLA decode kernels feed the
  perf curriculum + DELTA (now on a real model, not a toy).

**Rental gating:** the d20/d26/d32 runs are `deploy/runbooks/`-style rental steps (8×H100); nano +
all ablation science run on the standing sm120 card. Follow the repo's "develop on the GPU, rent
only for scale" discipline.

## §6 — Frontier landscape (2026), ranked for transfer

| Line | State (verified) | Transfer |
|---|---|---|
| **DeepSeek** V3 (671B/37B) · R1 (GRPO/RLVR) · **V3.2-Exp DSA** (2512.02556) | MLA, DeepSeekMoE, aux-loss-free bias, MTP, R1 GRPO, DSA | **Spine of §3** (F2/F5/F6/F7/F8) |
| **GLM** — "GLM5-2" = **GLM-5.2** (Z.ai, 2026-06-16, ~744B/40B, 1M ctx); lineage GLM-5 (Feb 2026, 745B/44B), GLM-4.5 "ARC" (2508.06471, 355B/32B), GLM-4.6 (357B/32B, 200K ctx) | MTP + agentic/coding RL (ARC) + long-context as flagship priorities; deep-narrow MoE | Reinforces F2 (MTP) + the agentic-RL edge; long-ctx → F8 |
| **Kimi K2** (1.04T/32B, **MuonClip**, 15.5T tok) | Muon at trillion scale + QK-Clip; agentic tool-RL | Validates F1 at scale; QK-Clip = F9 |
| **Qwen3** (dense + MoE; QK-norm; no QKV bias) | Clean modern recipe; QK-norm already in `model.py` | Recipe confirmation |
| **OLMo 2/3, SmolLM3** | Fully-open recipes (norm placement, z-loss, stability) | Pretrain-stability reference |

## §7 — Verification (acceptance oracle)

- **The report card is the loop's oracle.** `speedrun.sh` emits CORE/`val_bpb`/ARC/MMLU/GSM8K/
  HumanEval; d20 target **CORE ≥ ~0.256** (GPT-2 grade); the chat UI answers a held-out prompt.
- **Each ablation** ships only when its pre-registered falsifier (§3) is measured in
  `bench/RESULTS.md` (predict-before-run), one variable at iso-FLOP, with the kill criterion checked.
- **Floors unchanged.** green-CI (`ruff` + `pyright` + `pytest -m "not gpu"`) + the official
  adapter-test oracles remain the commit gate; new numeric claims are `[FACT]` only when measured.

## §8 — Honesty ledger (verifier corrections propagated)

- **[REFUTED→fixed]** Full-rank orthogonalized Muon update RMS = **1/√max(A,B)** (not 1/max(A,B));
  `0.2·√max(A,B)` scale + WD 0.1 correct (Moonlight 2502.16982, Lemma 1).
- **[UNCERTAIN→softened]** "Muon subsumes muP" is an overstatement — Essential AI (2505.02222)
  transfers muP **with** Muon up to ~4B params; treat as "Muon eases, not replaces, muP."
- **[UNCERTAIN]** nanochat **d26** is a passing suggestion, not a benchmarked tier; solid tiers are
  d20 (~561M/~$100) and d32 (~1.9B/~$800), plus a d34 release.
- **Open confirmations (do at build time, don't assert now):** nanochat exact per-stage token
  counts; ReLU²-vs-SwiGLU MLP; logit-softcap; value-embeddings; vocab=65536 — all **to be confirmed
  against `nanochat/gpt.py` / `tokenizer.py`**. Two deep-dive research agents failed (schema-retry);
  their content was recovered first-hand (repo audit) + direct README/GLM fetches.

## §9 — Coexistence with the perf + DELTA fronts

Three fronts, one checkout, zone-disciplined (ADR-0014 model). **Synergies, not competition:** F2
(MTP) and F5 (MLA) produce the exact decode kernels the perf/DELTA fronts optimize; F3/F4 give
DELTA a real model to roofline instead of a random-weight toy. This front's zone:
`src/scratch_llm/{eval,}`, `scripts/`, `optim.py`, `train.py`, `model.py` (additive: MTP head +
`attn` variant + untie), `mla.py` (promote), `moe.py` (ablation harness), the F-rung sections of
`bench/RESULTS.md`. Shared files (`CLAUDE.md`, `STATUS.md`, `pyproject.toml`, `bench/RESULTS.md`):
pull-rebase before commit, additive edits, precise `git add` (never `-A`).

## Source-of-truth pointers

- Approved plan of record: `~/.claude/plans/misty-sniffing-cerf.md`.
- Decision record: `docs/adr/ADR-0018-close-the-loop-nanochat-front.md`.
- Measurement ledger (pre-register + measure every rung): `bench/RESULTS.md` (§Frontier ablations).
- Build status: `docs/STATUS.md`. Frontier-defaults context: `docs/FRONTIER_PRACTICE_2026.md`.
- Parallel fronts: `performance/PERF_PLAN.md` (perf), `../../DELTA.md` (capstone).
- Reference repo (oracle, re-own — do not copy): karpathy/nanochat (`nanochat/{gpt,optim,tokenizer,engine}.py`, `speedrun.sh`).
