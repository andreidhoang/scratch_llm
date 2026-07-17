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
>
> ---
>
> ### ⚠ Re-verification — 2026-07-09 (frontier moved; re-weight before you build)
>
> A second deep-research pass (workflow `wf_b5d7adde-3bf`; 8 angles, primary-source adversarial
> verification, July-2026 cutoff) re-checked the 9 bets against the *actual* mid-2026 frontier. **The
> program's spine is sound and its honesty discipline is its most hireable feature — but the frontier
> has moved under three load-bearing bets, and two 2026-defining directions are missing entirely.**
> Full per-rung verdicts + citations in **§10**; the actionable deltas:
>
> 1. **F1 (Muon) — REFRAME, don't cut.** The "Muon beats AdamW" headline is **substantially deflated**
>    in 2026. Stanford/Marin *Fantastic Optimizers I* (`2509.02046`, Sep 2025): a *well-tuned* AdamW
>    shrinks Muon's edge from **1.4× at 0.1B → 1.1× at 1.2B**; the 1.4–2× claims came from *under-tuned
>    baselines* + intermediate-checkpoint eval. *Optimizers II: Hyperball* (`2606.16899`, Jun 2026):
>    plain **MuonW ≈10%** at 1.2B; their **MuonH sustains 20–30%** and supersedes plain Muon. → The
>    **≥15% pre-registration is too aggressive** and, worse, will produce a *fake win* unless the AdamW
>    baseline is independently LR-tuned. Recalibrate the kill band + make "properly-tuned baseline" the
>    DoD — **reproducing the deflation *is* the 2026 skill.** (§10 F1.)
> 2. **F7 (GRPO "aha") — REFRAME from "reproduce" to "debunk".** The reward↑ + length-growth "aha" is a
>    **known GRPO-bias artifact**: random rewards recover ~21.4 of 29.1 pts of the real-reward MATH gain
>    on Qwen (clipping bias amplifies pretraining priors; base-model-dependent). DeepSeek-V3.2 itself now
>    **penalizes length** as an artifact. → Drop "length-growth = success" as the oracle; the honest 2026
>    artifact is the **random-reward control that shows the aha is spurious**, with **Dr.GRPO/GSPO** as
>    the bias-corrected objective. This is a *stronger* signal than the naive reproduction. (§10 F7.)
> 3. **F8 (DSA) — PROMOTE from stretch to core.** Sparse/linear attention is *the* 2026 attention
>    frontier (DeepSeek V3.2 DSA, GLM-5 ships DSA in production "lossless by construction"). It is the
>    freshest, scarcest technique in the program and should not be gated last.
> 4. **MISSING #1 — hybrid *linear* attention (add F10).** Kimi Linear (Oct 2025) + Qwen3-Next/Qwen3.5
>    (Gated DeltaNet : full-attn 3:1) now **beat full attention** and position **MLA as "the baseline to
>    beat"** — 75% KV cut, 6× decode @1M. The program covers MLA (F5) + DSA (F8) but **not the linear
>    family at all.** Attention design is *contested, not settled* in 2026 → high ablation signal.
> 5. **MISSING #2 — agentic / tool-use RL (add F11).** The **#1 stated priority** of DeepSeek, Moonshot
>    (K2), and Qwen mid-2026 is agentic RL + long-horizon tool use; 2026 eval shifted from chatbot to
>    agentic (SWE-bench Verified). A small verifiable-tool-use env + multi-turn RL is higher-signal than
>    F6/F9.
> 6. **F4 precision — retarget FP8 → NVFP4.** FP8 pretraining is **settled/table-stakes** (V3 <0.25% vs
>    bf16). The 2026 precision frontier is **FP4/NVFP4** (validated 12B/10T, val-loss = FP8; Blackwell
>    7× FP4 GEMM — the standing sm120 card is Blackwell). Keep bf16+compile as the cheap-MFU win; move
>    the *stretch* to an NVFP4 fake-quant numerics study.
> 7. **Settled-science → fast table-stakes, not headline science:** **F6** (sharpen to ablate balancing
>    *granularity* batch-vs-seq — the actually-open question — or downweight); **F9** (cheap guard, keep
>    small); **F3/F5** (necessary hygiene + DSA-substrate, ship the first-slice, don't over-invest).
> 8. **nanochat factual drift — corrected in §2/§8:** vocab **65,536 → 32,768** and D:N **20 → 8**
>    (commit `ccf4b7f9`, 2026-01-07); original d20 base CORE **0.22** (0.256 is a later/bigger target);
>    original $92.4/3h51m is Oct-2025, current README ~$48/~2h at ~d26 for GPT-2-grade.
>
> **Re-weighted EV order (2026 hiring signal):** **F8+F10** (attention efficiency — scarce) · **F7-reframed
> + F11** (RL honesty + agentic — scarce) · then **F1** (methodology) · **F2** (convergent free-draft)
> · **F4** (bf16 + NVFP4) as table-stakes-done-well · **F3/F5/F6/F9** as necessary-but-minimize. The
> *discipline* (pre-registration, tuned baselines, kill criteria, adversarial controls) outranks any
> single result — it is what a frontier RE is actually screened on.

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
| Optimizer | `optim.py` / `train.py:118` | AdamW (β₂=0.95, decoupled WD, cosine) **+ Muon + CombinedOptimizer** (`optim.py:170,318`) shipped and wired into the F1 race harness (`eval/optimizer_race.py`); the default dense-decoder `train.py:118` path is still AdamW-only pending F1's verdict. |
| Training loop | `train.py:136` | **plain fp32, no autocast/compile**; memmap batches; no real-corpus run. |
| Serving stack | `serving/` | Real + measured, but against random-weight toys (confounded — `RESULTS.md:383`). |
| RL (SFT/EI/GRPO/Dr.GRPO/DPO + r1_zero grader + envs) | `algos/`,`rewards/`,`envs/` | Built + toy-CPU; **no real-model "aha" on record**. |

**Deltas nanochat has that we lack:** ~~Muon optimizer~~ (shipped `optim.py:170,318`, live in the F1
race harness — the *default* dense-decoder path (`train.py:118`) stays AdamW-only until F1 verdicts
it in); **untied** embeddings (separate `embedding_lr`/`unembedding_lr`) outside the F1 harness (which
already forces `tie_embeddings=False`, `speedrun.py:120`); an integrated `speedrun.sh`; a **report
card** (DCLM CORE + `val_bpb` + ARC/MMLU/GSM8K/HumanEval); a midtraining stage + tool-use + chat UI.

## §2 — Model-size tiers (Karpathy-style)

`C = 6·N·D` used to cross-check tokens. Decision of record: **straight to the $100 d20** as the
headline artifact, with a free nano pre-flight to protect the paid run.

> **⚠ Numbers updated 2026-07-09 (§10, verified).** nanochat changed its defaults in commit
> `ccf4b7f9` (2026-01-07, Karpathy — *"vocab size down to 32K. D:N ratio from 20 to 8"*): **vocab
> 65,536 → 32,768 (2¹⁵)** and **tokens:params 20 → 8**. So a *current* d20 (~561M) speedrun targets
> **~4.5B tokens** (D:N 8), not ~11.2B — **cheaper, not dearer.** The **≈11.2B / 20-tok/param / vocab
> 65,536** figures below describe the **original Oct-2025 d20** (560,988,160 params, 11.22B tok,
> $92.40 / 3h51m, **base CORE 0.22**) — kept for provenance but superseded. Current README: GPT-2-grade
> (CORE ≈ 0.256) is now **~$48 / ~2h at ~d26**, a *later, bigger* tier — not the d20. Confirm the exact
> D:N/vocab you run against `scripts/tok_train.py` at HEAD.

| Tier | Depth / size | Tokens | Compute | Cost | Purpose |
|---|---|---|---|---|---|
| **Pre-flight (nano)** | depth ~4–6, ~5–20M | ~0.2–1B | 1 GPU, minutes | ~$0–5 | Smoke-test the whole `speedrun.sh` before spending the $100. Protects against a plumbing bug. |
| **Headline = nanochat d20** | depth 20, **~561M** (n_embd≈1280) | **~4.5B @ D:N 8** *(current default; orig. Oct-25 = 11.22B @ D:N 20)* | 8×H100, ~2–4 h | **~$48–100** | The talking model + public report card. Original base **CORE 0.22**; GPT-2-grade (0.256) is a later ~d26 tier, not the d20. |
| **Ablation / science** | ~30–300M on the standing Blackwell (24 GB) | Chinchilla-optimal/point | 1 GPU, free | $0 | Many seeds of the iso-FLOP ablations — where the science lives. |
| **Stretch (opt-in)** | d26 (~$48, GPT-2-grade) → d32 (~1.9B, ~$800) | — | 8×H100 | ~$48–800 | Only if an ablation result justifies scale. d26 is now the README's GPT-2-CORE tier; d32/d34 are the large releases. |

Tokenizer to match: GPT-4-style **Rust BPE**, vocab **32,768 = 2¹⁵** *(current nanochat default since
`ccf4b7f9`; the original 2¹⁶=65,536 is superseded — confirm against `scripts/tok_train.py` at HEAD)*.

## §3 — The ranked ablation program (the heart of the front)

EV-ranked (leverage ÷ effort, weighted by frontier-lab signal). Each rung is a pre-registered,
falsifiable experiment; the discipline *is* the hireable skill.

> **⚠ EV re-ranked 2026-07-09 (§10).** The original 1–9 order below is preserved for the rung-card
> cross-refs, but the **scarce-2026 EV order is F8+F10 · F7-reframed+F11 · F1 · F2 · F4 · then
> F3/F5/F6/F9**. Attention-efficiency (F8 DSA + the new **F10** linear-hybrid) and RL-honesty/agentic
> (F7-reframed + the new **F11**) carry the differentiating hiring signal; F1/F2/F4 are table-stakes done
> well; F3/F5/F6/F9 are necessary-but-minimize. The **KILL bands on F1 and F7 are recalibrated** to the
> verified 2026 evidence (below, marked ⚠). Rung cards for F10/F11 are in §10.

**Node pointer (current):** `F1-run — iso-FLOP Muon vs AdamW` (A1 shards + A2 checkpoint chaining
✅ 2026-07-09; F1 unit level ✅). *Ordering call (Mode-2, human-owned): F1 stays the pending headline for
loop-closure, but consider pulling **F8/F10** first as the higher-2026-signal science.* Advance the
pointer as rungs ship.

| # | Rung | Effort | Pre-registered result + KILL | Primary sources |
|---|---|---|---|---|
| **1** | **MuonAdamW** ⚠*recalibrated* | M | Iso-FLOP with a **separately LR-tuned AdamW baseline** (mandatory — an untuned baseline is a *fake win*, `2509.02046`): Muon **1.1–1.4× band, scale-dependent** (predict ~1.3× / ≈15–25% saving @ 30–50M, shrinking with N); or ≥0.02 nats lower @ iso-FLOP; NS overhead **<1%**. **KILL** if saving <5% vs the *tuned* AdamW, or divergence at reused LR. *Muon is deflated + superseded by MuonH (`2606.16899`) — the tuned-baseline methodology is the artifact, not the win.* | Jordan; Moonlight 2502.16982; **Wen 2509.02046 (I) + 2606.16899 (II/Hyperball); Essential AI 2505.02222** |
| **2** | **MTP draft head** (V3 D=1, sequential) | L | On the trained model: **≥1.5× tokens/target-forward** on open text (n-gram ≈1.0×); lossless. Val-loss: **expect NEUTRAL bpb at ≤561M** (V3 ablation: Pile bpb 0.729→0.729; gains are downstream, not bpb) — so falsifier is Δbpb ∈ [−0.02,+0.01], **payoff is the free draft head, not densification**. **KILL** if MTP aux degrades val loss >0.01 nats. | Gloeckle 2404.19737; V3 2412.19437; GLM-5, Qwen3-Next (8 prod. models 2026) |
| **3** | **De-confound serving on a real model** | S | n-gram acceptance **collapses <10% on open text**, stays ~40–60% on code/JSON. *Hygiene fix, low frontier signal — keep small; it unblocks F2b.* | repo ledger; FA-3 2407.08608 |
| **4** | **bf16-autocast + torch.compile** (**stretch → NVFP4**, not FP8) | M | compile+bf16 **+10–20% MFU**. *FP8 is settled/table-stakes (V3 <0.25% vs bf16); retarget the stretch to an **NVFP4** fake-quant numerics study — the Blackwell-native 2026 precision frontier (sm120 supports it).* | Databricks FP8; **NVFP4 12B/10T (2025)**; V3 FP8 2412.19437 |
| **5** | **MLA for real** (wire `mla.py` → block + KV cache) | M | MLA within **+0.02 val loss** of iso-param GQA-8; KV/token **~3.5× smaller**; decode **≥1.2× at 16k**. **KILL** if gap >0.05. *MLA is now "the baseline to beat" (Kimi Linear) — ship the **first-slice** as the DSA/F10 substrate, don't over-invest as a headline.* | DeepSeek-V2 2405.04434 |
| **6** | **MoE balancing** — sharpen to **granularity** (batch-vs-seq) | S | The *open* question (per `2408.15664`'s own finding: batch-wise aux ≈ bias-free): **batch-wise balancing ≤ seq-wise val loss**, entropy **>0.9·log N**; no-balancing collapses. *Bias-free-vs-seq alone is near-settled — ablate granularity or downweight.* | V3 2412.19437; 2408.15664; Qwen3 (still uses aux-loss) |
| **7** | **RLVR — debunk the "aha", don't reproduce it** ⚠*reframed* | L | Reward↑ + KL bounded; **random-reward control recovers most of the gain** (aha is a GRPO clipping-bias artifact — `2506` spurious-rewards); use **Dr.GRPO/GSPO** (bias-corrected). **Drop "length-growth = success"** (V3.2 *penalizes* length). **KILL** if the control *fails to* reward-hack (then the effect was real learning — good, but re-baseline). | R1 2501.12948; **Dr.GRPO; GSPO (Qwen); spurious-rewards; V3.2 2512.02556** |
| **8** | **DSA sparse attention** ⚠*PROMOTED to core* | L | KL-aligned lightning-indexer recovers **≥95%** of dense mass; within **+0.03 val loss** at 8k; O(L²)→O(Lk) FLOP crossover. **KILL** if gap >0.1. *Freshest, scarcest technique; GLM-5 ships it in production. Do NOT gate last.* | **V3.2-Exp 2512.02556; GLM-5** |
| **9** | **Logit-stability guard** | S | `qk_norm` on ⇒ max per-head logit **<~30**, QK-Clip γ==1 sub-1B. *Cheap guard, keep small — QK-norm is settled.* | Kimi-K2 2507.20534 |
| **10** | **Hybrid linear attention** 🆕*(GDN/KDA-style)* | M/L | iso-param 3:1 (Gated-DeltaNet : full-attn) hybrid within **+0.03 val loss** of full attention at ≤300M; **KV/token ≥2× smaller**, decode uplift at long ctx. *The 2026 attention frontier the program lacked — MLA is now the baseline it beats.* **KILL** if gap >0.1 nats or no KV win. | **Kimi Linear (Moonshot, Oct-25); Qwen3-Next/Qwen3.5; Gated DeltaNet; NSA 2502.11089** |
| **11** | **Agentic / tool-use RL** 🆕*(the #1 2026 priority)* | L | A verifiable multi-turn tool env (calculator/code-exec): success-rate ↑ with turns-used bounded; a **format-only reward hacks** (control). *DeepSeek/K2/Qwen's stated #1 direction; 2026 eval = agentic (SWE-bench).* **KILL** if success-rate flat or reward-hacked without detection. | **K2 2507.20534; DeepSeek V3.2; Qwen3.5 agentic evals** |

**Convergent-defaults signal (verified 2026-07-09):** **MTP** (V3 + GLM-5 + Qwen3-Next + 6 more prod.
models — *strongest* convergent signal) and **QK-norm** (Qwen3/Gemma3/OLMo2, already in `model.py`) are
genuinely convergent → load-bearing, not fashion. **But three "defaults" are contested, not settled:**
**Muon** is deflated + superseded (MuonH); **MoE balancing** is *not* converged (Qwen3 still uses
aux-loss; batch-vs-seq granularity is the real question); **attention** is actively contested
(MLA vs DSA vs linear-hybrid vs — MiniMax-M2.5 — plain GQA). Contested ≠ settled ⇒ *higher* ablation
signal, not lower.

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

**F4 · Cheap MFU.** `torch.autocast(bf16)` + `torch.compile` on the forward (train.py was fp32);
then an FP8 GEMM ablation **gated by the existing train↔serve logit-KL check**. *Prediction:*
+10–20% MFU from compile+bf16; FP8 ~1.3–1.5× while KL under tolerance (and slower than bf16 with
compile OFF). *Files:* `train.py`, `quant/`, `bench/`. **Status (2026-07-04, wired + GPU-verified):**
bf16-autocast ✓ and `torch.compile` ✓ each train to loss 8e-4 on sm120; their **combination NaNs on
this box's torch-2.12 inductor** (reproduces with plain AdamW — a codegen bug, not our logic), caught
now by a loud NaN guard in `train.py`. bf16+compile is the intended **H100-rental** path; the MFU
delta is measured there. See `bench/RESULTS.md` §Frontier ablations F1/F4.

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

- **Phase 0 — Pre-flight (local, ~$0). ✅ DONE 2026-07-04.** `scripts/speedrun.sh` +
  `scratch_llm.speedrun` + `eval/` report card run **tokenizer → pretrain (MuonAdamW) → eval →
  sample** end-to-end; verified on the sm120 Blackwell (depth 4, 8.3 s) — `val_bpb 0.02` + a
  coherent sample (`bench/RESULTS.md` §Phase 0). One `--depth` knob (20 ⇒ the d20 headline). *This
  gate protects the $100.* Follow-ons (midtrain/SFT/RL) reuse `train`/`algos` (F2/F7).
- **Phase 1 — Close the loop + Muon ($100 d20).** Land **F1 (MuonAdamW)** + **F4 (bf16/compile)**
  first (they change the run you pay for), untie embeddings, then launch the **d20 8×H100**
  speedrun. **Deliverable:** a talking model + a public report card (target CORE ≈ GPT-2). Highest-EV node.
- **Phase 2 — Ablations on the real base.** **F3** de-confound → **F2** MTP → **F5** MLA-real →
  **F6** MoE balancing → **F7** GRPO "aha." Each pre-registered in `bench/RESULTS.md`, iso-FLOP,
  one variable, kill criterion checked. *(2026-07-09 §10: **F7 reframed** to a random-reward debunk
  control; **F6 sharpened** to batch-vs-seq granularity.)*
- **Phase 3 — Frontier edges (the scarce-2026 cluster — PROMOTED from opt-in, §10).** **F8** DSA +
  **F10** hybrid **linear** attention (Gated-DeltaNet — the 2026 attention frontier that beats MLA;
  the **model-side twin of the DELTA GDN-2 decode kernel**) → **F11** agentic/tool-use RL (the #1
  stated lab priority). MTP/MLA/GDN decode kernels feed the perf curriculum + **DELTA** (now on a real
  model, not a toy). These carry the differentiating hiring signal — do not gate them last.

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
  counts; ReLU²-vs-SwiGLU MLP; logit-softcap; value-embeddings — all **to be confirmed
  against `nanochat/gpt.py` / `tokenizer.py`**. Two deep-dive research agents failed (schema-retry);
  their content was recovered first-hand (repo audit) + direct README/GLM fetches.

**Corrections added 2026-07-09 (second research pass `wf_b5d7adde-3bf`, primary-source-verified — see §10):**
- **[CORRECTED]** nanochat **vocab 65,536 → 32,768 (2¹⁵)** and **D:N 20 → 8** (commit `ccf4b7f9`,
  2026-01-07). The `vocab=65536` "open confirmation" above is now *resolved to 32768*. §2 updated.
- **[CORRECTED]** original d20 **base CORE = 0.22** (not 0.256); CORE 0.256 (GPT-2-grade) is a *later,
  bigger* tier (~d26, ~$48/~2h in the current README). §2 updated.
- **[DEFLATED]** Muon's AdamW edge is **1.4× @0.1B → 1.1× @1.2B** vs a *tuned* baseline (`2509.02046`);
  plain MuonW ≈10% @1.2B, superseded by **MuonH/Hyperball** (`2606.16899`, Jun 2026). The prior ledger's
  "Muon eases muP" softening stands; **F1's ≥15% pre-registration was recalibrated** (§3, §10 F1).
- **[ARTIFACT]** the R1-Zero "aha" (reward↑ + length-growth) is a **GRPO clipping-bias artifact** —
  random rewards recover most of the Qwen MATH gain; DeepSeek-V3.2 *penalizes* length. **F7 reframed**
  from "reproduce" to "debunk-with-control" (§3, §10 F7).
- **[MOVED-ON]** FP8 pretraining is **settled** (V3 <0.25% vs bf16); the 2026 precision frontier is
  **NVFP4** (validated 12B/10T). **F4 stretch retargeted** FP8→NVFP4 (§3, §10 F4).
- **[GAP]** the program had **no linear-hybrid attention** (Kimi Linear / Gated-DeltaNet — the 2026
  frontier that beats MLA) and **no agentic RL** (the #1 stated lab priority). Added as **F10 / F11**.

## §9 — Coexistence with the perf + DELTA fronts

Three fronts, one checkout, zone-disciplined (ADR-0014 model). **Synergies, not competition:** F2
(MTP) and F5 (MLA) produce the exact decode kernels the perf/DELTA fronts optimize; F3/F4 give
DELTA a real model to roofline instead of a random-weight toy. This front's zone:
`src/scratch_llm/{eval,}`, `scripts/`, `optim.py`, `train.py`, `model.py` (additive: MTP head +
`attn` variant + untie), `mla.py` (promote), `moe.py` (ablation harness), the F-rung sections of
`bench/RESULTS.md`. Shared files (`CLAUDE.md`, `STATUS.md`, `pyproject.toml`, `bench/RESULTS.md`):
pull-rebase before commit, additive edits, precise `git add` (never `-A`).

## §10 — 2026-07-09 frontier re-verification (per-rung verdict + citations)

> **Method.** Deep-research workflow `wf_b5d7adde-3bf`: 8 research angles (optimizers · attention/KV ·
> MTP · RLVR · precision · MoE · nanochat ground-truth · lab direction), fan-out web search → source
> fetch → **3-vote adversarial verification** against primary sources with a July-2026 cutoff. The
> optimizer-deflation and nanochat-cost facts each carried **high-confidence, multiply-verified** verdicts;
> the attention/RL/precision findings are single-source-verified against the named primary papers. As
> with the first pass, the final synthesis agents stalled on schema-retry — the verdicts below are
> recovered first-hand from the verified claim-set (FOP-4: these are `[FACT]`-about-the-literature, the
> *ablation* numbers stay pre-registered predictions until measured in `bench/RESULTS.md`).

**Verdict legend:** ✅ CONFIRMED still-valuable · ✏️ NEEDS-UPDATE (reframe/recalibrate) · ✂️ SETTLED /
minimize · 🆕 MISSING (add).

| Rung | Verdict | Why (verified 2026) | Primary sources |
|---|---|---|---|
| **F1 Muon** | ✏️ | Muon **deflated**: 1.4×@0.1B→1.1×@1.2B vs *tuned* AdamW; the 1.4–2× headline came from *under-tuned baselines* + mid-checkpoint eval. Plain MuonW ≈10%@1.2B; **MuonH/Hyperball supersedes it** (20–30%). The rung stays valuable **only if** the AdamW baseline is independently LR-tuned — reproducing the deflation *is* the 2026 skill. Recalibrate ≥15%→the 1.1–1.4× scale-dependent band. | Wen *Fantastic Optimizers I* **2509.02046** (Sep-25); *II Hyperball* **2606.16899** (Jun-26); Essential AI **2505.02222** |
| **F2 MTP** | ✅ | **Strongest convergent signal** — 8 prod. models by 2026 (V3, GLM-5, Qwen3-Next, Nemotron-3, MiniMax, Xiaomi MiMo, Step-3.5, Tencent Hy3). V3 D=1 sequential, self-spec draft 85–90% 2nd-tok / 1.8× decode, EAGLE-framed. **Caveat:** at ≤561M the pretrain-densifier bpb gain is ≈0 (V3: 0.729→0.729) — pre-register **neutral bpb**; the payoff is the free draft head. | Gloeckle **2404.19737**; V3 **2412.19437**; GLM-5 / Qwen3-Next (2026) |
| **F3 de-confound** | ✅ (small) | Correct + necessary honesty fix (the RESULTS.md acceptance confound), **low frontier signal**. Keep minimal; it unblocks F2b. | repo ledger; FA-3 2407.08608 |
| **F4 precision** | ✏️ | bf16+compile MFU = real & keep. **FP8 is settled/table-stakes** (V3 <0.25% vs bf16). Retarget the *stretch* **FP8 → NVFP4**: validated at 12B/10T (val-loss = FP8), Blackwell 7× FP4 GEMM, sm120 is Blackwell. NVFP4 > MXFP4 (16-elem blocks, E4M3 scales). | V3 FP8 2412.19437; **NVFP4 12B/10T (Aug-25)** |
| **F5 MLA** | ✏️ (substrate) | MLA was the converged 2024–25 choice (V2/V3, K2) **but is now "the baseline to beat"** (Kimi Linear). Value in 2026 = **substrate for DSA (F8) + F10**, not a standalone headline. Ship the first-slice; don't over-invest. | DeepSeek-V2 **2405.04434**; Kimi Linear (Oct-25) |
| **F6 MoE balancing** | ✂️→✏️ | **Not fully converged** (Qwen3 still uses global-batch aux-loss; Cerebras downplays the "aux-loss-free" novelty) — so *an* ablation is informative, but the bias-free-vs-seq framing is near-settled. **Sharpen the variable to balancing *granularity* (batch-wise vs seq-wise)** — `2408.15664`'s own result is *batch-wise aux ≈ bias-free*. Else downweight. | V3 **2412.19437**; **2408.15664**; Qwen3 (2501.15383) |
| **F7 RLVR "aha"** | ✏️ (reframe) | The reward↑ + length-growth "aha" is a **GRPO clipping-bias artifact**: random rewards recover ~21.4/29.1 pts of the Qwen MATH gain; base-model-dependent (fails on Llama3/OLMo2); DeepSeek-V3.2 now *penalizes* length. **Reframe: debunk with a random-reward control**, use **Dr.GRPO / GSPO** (bias-corrected). Drop length-growth as the oracle. Stronger signal than naive reproduction. | R1 **2501.12948**; Dr.GRPO; **GSPO (Qwen)**; spurious-rewards; V3.2 **2512.02556** |
| **F8 DSA** | ✅ **PROMOTE** | **Freshest, scarcest technique in the program.** DSA O(L²)→O(Lk), ReLU lightning-indexer (FP8, few heads), k=2048, built *on* MLA; continued-pretrain recipe (freeze all but indexer). **GLM-5 ships DSA in production**, "lossless by construction." De-gate from stretch → core. | **V3.2-Exp 2512.02556**; GLM-5 |
| **F9 QK-clip** | ✂️ | QK-norm is settled (Qwen3/Gemma3/OLMo2, already in `model.py`); MuonClip is K2's trillion-scale fix. Sub-1B → qk_norm suffices. Cheap guard, keep small — not a headline. | Kimi-K2 **2507.20534** |
| **F10 linear-hybrid** | 🆕 **ADD** | The **2026 attention frontier the program lacked.** Kimi Linear (KDA/Gated-DeltaNet, Oct-25) **beats full attention** across short/long/RL, 75% KV cut, 6× decode @1M; Qwen3-Next/Qwen3.5 = GDN:full-attn 3:1. Attention is **contested, not settled** (MLA vs DSA vs linear-hybrid vs GQA) ⇒ high ablation signal. | **Kimi Linear (Moonshot, Oct-25); Qwen3-Next/3.5; Gated DeltaNet; NSA 2502.11089** |
| **F11 agentic RL** | 🆕 **ADD** | The **#1 stated priority** of DeepSeek, Moonshot (K2), Qwen mid-2026; 2026 eval shifted chatbot→agentic (SWE-bench Verified 76.4). A small verifiable-tool-use env + multi-turn RL + a format-only reward-hack control is higher-signal than F6/F9. | K2 **2507.20534**; V3.2; Qwen3.5 agentic evals |

### The two new rung cards

**F10 · Hybrid linear attention (Gated-DeltaNet / KDA-style).**
- *Hypothesis.* A layerwise hybrid (linear-attention blocks : full/MLA blocks ≈ 3:1) matches full-attention
  quality while cutting KV memory and giving near-linear long-context scaling — the Kimi-Linear / Qwen3-Next
  result, reproduced small.
- *Prediction (falsifiable).* At ≤300M iso-param: hybrid within **+0.03 val loss** of a full-attention
  baseline; **KV/token ≥2× smaller**; measurable decode-throughput uplift as ctx grows. **KILL** if val gap
  >0.1 nats, OR no KV/decode win (then the linear block is dead weight at this scale).
- *DoD.* A `GatedDeltaNet` (or KDA) block from primitives (chunkwise recurrent scan, gating); a hybrid
  `TransformerBlock` interleave knob; loss-at-init ≈ log V; the recurrent scan matches a reference torch
  loop (float64); KV-bytes measured vs full attention. **Zone:** new `linear_attn.py` + additive `model.py`
  `attn='gdn_hybrid'` — coordinate with the perf front on any shared attention seam.
- *Interview.* "Why did 2026 labs move to linear-attention *hybrids* over pure MLA — what does the 3:1
  ratio buy, and why keep *any* full-attention layers instead of going fully linear?"

**F11 · Agentic / tool-use RL (the scarce 2026 cluster).**
- *Hypothesis.* Multi-turn RL over a *verifiable* tool environment (calculator / code-exec / retrieval)
  teaches tool-use policy that a single-turn RLVR "aha" cannot — the DeepSeek/K2/Qwen agentic direction,
  at from-scratch scale.
- *Prediction (falsifiable).* On a synthetic verifiable env: task success-rate ↑ monotonically with
  **turns-used bounded** (no tool-spam), KL(cur‖ref) bounded; a **format-only reward** (answer-blind,
  rewards tool-call *shape* not correctness) provably reward-hacks — the control that proves the verifiable
  reward is load-bearing. **KILL** if success-rate is flat, OR the format-only control is *not* caught by the
  guardrail logging (then the reward/monitoring is uninterpretable).
- *DoD.* Reuses `algos/grpo.py` + `envs/` + `rewards/` (a `ToolEnv` implementing the `VerifiableEnv`
  Protocol + a multi-turn rollout); the mandatory RL logging (entropy, KL-separately, IS-ratio, reward
  stats, **turns/length stats**) present and within thresholds (the `rl-run-auditor` gate). **Zone:**
  `algos/`, `envs/`, `rewards/`, `utils/monitors.py` — F-front-owned.
- *Interview.* "Why is agentic/tool-use RL the 2026 frontier over single-turn RLVR — and how do you keep a
  multi-turn agent from reward-hacking the tool-call *format* instead of solving the task?"

### Direction ranking (2026 hiring/contribution signal)

Verified lab priorities mid-2026: **DeepSeek** (Dec-25) — (a) sparse attention for long context, (b) scaled
RL post-training, (c) agentic task synthesis; **Moonshot K2** — agentic data synthesis + joint RL (SOTA
open agentic); **Qwen3.5** (Feb-26) — hybrid-linear attention + agentic (SWE-bench 76.4 > GPT-5.2 75.4).
The throughline: **agentic capability, with attention efficiency/sparsity as the architectural
differentiator.** Mapped onto the program:

- **Tier 1 — scarce, differentiating (do first / do best):** **F8 (DSA) + F10 (linear-hybrid)** — attention
  efficiency, the contested 2026 frontier · **F7-reframed + F11 (agentic)** — RL honesty + the #1 lab
  priority. These are where a from-scratch artifact *contributes*, not just reproduces.
- **Tier 2 — table-stakes, done well:** **F1** (the *tuned-baseline methodology* is the signal, not the win)
  · **F2** (convergent MTP + free draft) · **F4** (bf16+compile, then NVFP4).
- **Tier 3 — necessary but minimize (don't headline):** **F3** (hygiene) · **F5** (MLA first-slice as F8/F10
  substrate) · **F6** (sharpen to granularity or downweight) · **F9** (cheap guard).

**One-line thesis for the front:** the program's *discipline* — pre-registration, independently-tuned
baselines, kill criteria, adversarial controls — is now worth more than most individual results, precisely
because 2026's biggest optimizer and RL findings were **methodology corrections** (under-tuned baselines;
spurious-reward artifacts). Lean into that: the hireable artifact is *"I measured the frontier claims
honestly and three of them didn't survive a tuned baseline / a proper control."*

## Source-of-truth pointers

- **Buildable task breakdown (the next-phase DAG):** [`FRONTIER_2026_TASKSPEC.md`](FRONTIER_2026_TASKSPEC.md)
  — 25 rungs (A-loop-to-chat + B-ablation, incl. F10 linear-hybrid + F11 agentic-RL added 2026-07-09),
  EV-ranked, each with exact interfaces / tests / falsifier /
  kill / zone note; grounded per-rung in the real code by workflow `w77bbp4pb`.
- Approved plan of record: `~/.claude/plans/misty-sniffing-cerf.md`.
- Decision record: `docs/adr/ADR-0018-close-the-loop-nanochat-front.md`.
- Measurement ledger (pre-register + measure every rung): `bench/RESULTS.md` (§Frontier ablations).
- Build status: `docs/STATUS.md`. Frontier-defaults context: `docs/FRONTIER_PRACTICE_2026.md`.
- Parallel fronts: `performance/PERF_PLAN.md` (perf), `../../DELTA.md` (capstone).
- Reference repo (oracle, re-own — do not copy): karpathy/nanochat (`nanochat/{gpt,optim,tokenizer,engine}.py`, `speedrun.sh`).
