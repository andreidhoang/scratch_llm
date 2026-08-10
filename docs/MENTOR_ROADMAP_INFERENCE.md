# Inference + Kernel Engineering — the Merged Mentor Roadmap (one spine, one capstone)

> **What this is — MERGED 2026-08-06.** The Vizuara Inference workshop and the Vizuara Kernel workshop
> are **one discipline** at the 2026 frontier: both attack the single physical fact that *decode is
> memory-bandwidth-bound (AI≈1) while prefill is compute-bound* (`PERFORMANCE_TRACK.md` §0). Every
> "inference" optimization *is* a kernel (paged decode, split-K, NVFP4 GEMM); every LLM "kernel" *is*
> an inference optimization (FA2/3/4, the DELTA/KDA decode kernel). The 2026 JD cluster lists them in
> *one* role ("GPU Performance Engineer": CUDA/Triton/CUTLASS/PTX/NCU/roofline/FP8/vLLM-SGLang-internals
> /KV-cache-kernels). So this is the **single merged mentor roadmap** — the inference and kernel lanes
> are not twins; they are one ladder: `performance/` A1(serving)→A2(kernels)→A3(TC)→A4(FA)→A5(quant)
> →A6(dist)→A7(capstone), ending in the **K3/DELTA-KDA co-designed artifact**.
>
> **Provenance.** Synthesized 2026-08-06 from (a) the live inference.vizuara.ai + kernel-workshop
> curricula and (b) this repo's measured ledger (`bench/RESULTS.md`, `docs/STATUS.md`, the
> `performance/` A-files). Claims tagged `[MEASURED]` = our number off our box; `[REPORTED]` = external
> source; `[GAP]` = not yet owned.
>
> **Relationship to existing docs.** Supersedes the "twin" framing — this is now the single inference+
> kernel mentor roadmap. It does **not** reopen any plan; it maps the merged syllabus to owned code and
> names the gaps. **Lever 2** (on-demand). The kernel lane's operating spec stays
> [`performance/KERNEL_ROADMAP_2026.md`](../performance/KERNEL_ROADMAP_2026.md) §3/§4/§6; this doc is
> the *what-where-why* map over both workshops at once.

---

## 0a. Capstone decision (recorded 2026-08-06, user-directed) — the K3 capstone, not Vizuara's

**From the principal-RE-at-a-frontier-lab view, the scratch_llm capstone is strictly more valuable
than both Vizuara capstones. Both Vizuara capstones are discarded as portfolio targets.**

- **Vizuara Capstone 1** ("Speed-Optimized 7B Server") = integration of solved techniques (vLLM owns
  it). Your `PERFORMANCE_TRACK.md` §4 calls a serving engine "well-trodden." It teaches you to *use*
  tools, not *make* them. → **Demoted to a checkpoint/rung**: the H100 rental day that validates your
  serving substrate and produces the honest vLLM-gap diagnosis. It is a *gate feeding K3*, not a
  capstone.
- **Vizuara Capstone 2** (OpenClaw-RL WhatsApp agent) = a product/app; frontier labs don't hire for
  chatbot deployment. Breadth noise. → **Discarded** as a portfolio target (the RL-loop-close learning
  value still folds into the A5/F11 ablation, but the WhatsApp surface is not built).
- **scratch_llm capstone = the K3 track, anchored by DELTA→KDA (K10.2)** = **architecture↔kernel
  co-design**, the scarcest class your research identified. A fused **FP8/NVFP4 recurrent-state decode
  kernel** for GDN/KDA-class linear attention, married to a model you rebuilt (Kimi K3: KDA · Gated
  MLA-NoPE · LatentMoE). **Niche verified OPEN** in two prior primary-source passes
  (`performance/KERNEL_ROADMAP_2026.md` §1 row 7: 2026-07-14 and 2026-07-30 re-verification) — no public
  library ships it (FLA = standard precision only; FlashQLA = chunked-prefill only; FlashInfer/TK/
  DeepGEMM/vendor = none). The people who do this (Songlin Yang → Thinking Machines; Tri Dao; the
  FlashInfer team) are who labs hire for the premium kernel/perf roles.

**The barbell (honest caveat — repo's open question RQ6):** the niche co-design is *higher-ceiling,
higher-variance*; no verified 2026 comp data proves it out-earns a broad OSS PR for top bands. So the
principal's move is the barbell: **DELTA/KDA = the spike** (high ceiling) **+ upstream PR + public
worklog = the base** (low variance). Keep both — never bet only on the niche.

---

## 0. The one honest finding (read this first)

**You have already built, from scratch and measured, nearly every mechanism the Vizuara Inference
workshop teaches at survey level.** This is the same verdict the Kernel roadmap reached (V2: "skip the
workshop, strip-mine the syllabus"), and it holds *a fortiori* for the inference track:

| Vizuara inference concept | This repo's measured status | Verdict |
|---|---|---|
| KV-cache, decode-is-memory-bound | `bench/RESULTS.md` R1: AI≈1, 136× below ridge, 77% of memory wall w/ graphs | ✅ **MEASURED deeper** |
| PagedAttention | R4.1: fused Triton paged decode, frag 5%, capacity ×9.3, ×3.52 vs wave | ✅ **MEASURED deeper** |
| Continuous batching | R3b: 2.30× wall / 2.93× steps, TTFT 4.9×, analytic model proven | ✅ **MEASURED deeper** |
| Chunked prefill | R4.2: token-exact, 44 tests (mechanism ✓; spike-reduction falsified) | ✅ **MEASURED** |
| Speculative decoding | R4.3: lossless, 27 tests, ×1.2–1.4 via n-gram drafting | ✅ **MEASURED** |
| CUDA-graph decode | R4.4: B=1 −74% step, 253 tok/s = 77% memory wall | ✅ **MEASURED deeper** |
| MLA latent cache | `mla.py` toy + weight-absorption identity (A1 §4.5) | ✅ **MEASURED** (toy) |
| Prefill/decode disaggregation | demo-level (A1 §4.6) | 🟡 demonstrated |
| FlashAttention-2 | `kernels/`: fwd+bwd Triton, ~50% SDPA, 44× leaner | ✅ **MEASURED** |
| Quantization (INT4/8, FP8-KV, NVFP4, AWQ) | `quant/`: NVFP4 1.48×<MXFP4, FP8-KV, AWQ 1.71× | ✅ **MEASURED** |
| Distributed (TP/PP/EP/MoE) | `utils/{tp_mlp,pipeline_schedule,ep_moe,mfu}.py`: gloo-verified, 112 tests | ✅ **code-complete** (rental-gated) |

**The binding constraint is NOT missing content — it is the mastery deficit.** Per
[`learning/PROGRESS.md`](learning/PROGRESS.md) (the live ledger): **12 / 89 Bài cold-defended** as of
2026-07-14 (M1 1.1–1.4 · M2 2.1–2.6 · M3 3.1–3.2) — with M2 2.4–2.6 and M3 3.1–3.2 flagged
"Driver-explained, Navigator-closed" (need re-own for interview-grade cold defense), and the entire
PERF half (S1–S6, 41 Bài) still ⬜. The code was built under `delegate` mode (agents typed it); the
human's generative ownership has not caught up. Interviews at frontier labs are **AI-prohibited,
live** (KERNEL_ROADMAP §2). Memory forms only from what *you* generate, not what you read (Anthropic
skill-formation RCT, arXiv 2601.20245: AI-assisted learners scored 50% vs 67%, worst on debugging).

**So the mentorship question is not "teach me these concepts" — it is:**
1. *Convert built inventory into defensible mastery* (the rebuild ladder + daily quarry protocol).
2. *Fill the genuine gaps* (real vLLM/SGLang/TRT-LLM usage, Ray Serve, edge deployment).
3. *Extract public artifacts* (upstream PRs, the worklog) — the thing that actually gets you hired.

---

## 1. The workshop curriculum, mapped

The Inference workshop is **2 phases, 14 sessions, 2 capstones** (Apr 27 – May 25 2026 cohort).
Below: each block → what it teaches → what you own → the gap → the mentor action.

### Phase 1 — Foundations & Optimization (7 sessions L1–L7)

**Block A: The decode memory-wall thesis (≈ L1–L2)**
- *Teaches:* prefill vs decode roofline; KV-cache; why decode is bandwidth-bound.
- *You own it:* `performance/A1_transformer_inference.md` §2.1–2.2 (the hand-roofline derivation),
  `bench/RESULTS.md` R1 (measured AI≈1, 136× below ridge). You can predict decode tok/s from
  `HBM_BW/2P` *before* measuring — the principal-grade skill (A1 §9 rubric).
- *Gap:* none at the concept level. *Mentor action:* **P0.5-L0/L1 rebuild** — re-derive the roofline
  and KV-sizing formula from blank, timed, no AI. Gate: the formula on paper matches the code.

**Block B: Attention as the system — GQA/MQA/MLA (≈ L3–L4)**
- *Teaches:* KV-cache compression, GQA, MQA, MLA weight-absorption.
- *You own it:* `model.py` (GQA), `mla.py` (MLA toy + absorption identity). A1 §4.5.
- *Gap:* the *production* DeepSeek FlashMLA reading (you have the toy; the fused kernel is reading-tier).
- *Mentor action:* Feynman the absorption trick: *why* does `(W^UQ c^Q)·(W^UK c^KV)` let you attend in
  latent space, and *why* does RoPE break it (the decoupled-RoPE pathway)? Teach it back cold.

**Block C: FlashAttention (≈ L5)**
- *Teaches:* online softmax, tiling, IO-awareness.
- *You own it:* `kernels/attention/prefill/fa2.py` (Triton fwd+bwd), `kernels/common/online_softmax.py`.
  A4 note covers FA2→FA3→FA4.
- *Gap:* FA3/FA4 are compile-verified only (rental-gated: P1 H100 / P2 B200 days).
- *Mentor action:* the quarry Q1 rep — blank-page online-softmax → tiled GEMM → FA2 fwd inner loop.

**Block D: Continuous batching + PagedAttention (≈ L6–L7) — the core**
- *Teaches:* Orca iteration-level scheduling; vLLM paged KV; the throughput wins.
- *You own it:* `serving/continuous.py` (490 LOC, the one-engine-two-policies design),
  `model.py:PagedKVCache` (block table, CoW, admission guard). **Measured: 2.30× wall, frag 5%, ×3.52.**
  This is the *headline* — you've built the thing the workshop only surveys.
- *Gap:* you've never run *production vLLM* on a real model and compared against your engine.
- *Mentor action:* **Gap-G1** (§3 below) — run vLLM on Llama/Qwen, measure goodput, write the honest
  gap-to-vLLM diagnosis (A1 §9 principal rubric: "we're 1.7× off because…").

**Phase 1 Capstone:** "Build a Speed-Optimized LLM Inference Server" (7B, raw weights → optimized →
benchmark). → *You have built the substrate; the capstone for you is the P1 H100 rental day* (run the
real 7B through your paged+continuous+spec+graph engine on a real GPU, then vs vLLM).

### Phase 2 — Production & Edge (7 sessions)

**Block E: Production frameworks (≈ L8–L9)**
- *Teaches:* vLLM internals (scheduler, PagedAttention), SGLang (RadixAttention, structured gen).
- *You own it:* the *mechanisms* (built them). You do **NOT** own: running vLLM/SGLang as production
  tools on real models, reading their scheduler source, or contributing to them.
- *Gap:* **G1 (vLLM real-run + source read) and G2 (SGLang RadixAttention — you have it as an
  unbuilt A1 §7 stretch goal).** These are the workshop's genuine value-add for you.
- *Mentor action:* source-read sprints: vLLM `scheduler.py` + `block_manager.py`; SGLang
  `radix_cache.py`. Map their abstractions to your `continuous.py` / `PagedKVCache`.

**Block F: Distributed serving — Ray Serve + Megatron (≈ L10–L11)**
- *Teaches:* Ray Serve (distributed serving), Megatron-LM (model parallelism, GRPO online RL).
- *You own it:* toy TP/1F1B/EP-MoE (`utils/`, gloo-verified, 112 tests). The comms algebra
  (`utils/comms_calc.py`, verified vs the official PDF).
- *Gap:* **G3 — Ray Serve is entirely absent from your repo.** Megatron you have at toy level.
- *Mentor action:* Ray Serve is a `[GAP]` — either a 2–3 day focused build (a Ray Serve wrapper around
  your engine, or deploying vLLM on Ray) or a deliberate reading-tier decision (your
  `PERFORMANCE_TRACK.md` §4 says "subtract: reimplementing MLA/Megatron from scratch → understand +
  cite"). Decide explicitly.

**Block G: TensorRT-LLM + quantization at the engine level (≈ L12)**
- *Teaches:* TRT-LLM (INT4/INT8, kernel fusion, the engine layer).
- *You own it:* `quant/` (INT8/INT4/NVFP4/FP8-KV/AWQ, measured) — the *numerics* layer, deeper than
  the workshop. You do NOT own: TRT-LLM as a deployment engine.
- *Gap:* **G4 — TRT-LLM deployment** (reading/usage tier; building a TRT-LLM engine is a NVIDIA-internal
  tool, not a from-scratch target).
- *Mentor action:* reading-tier: the TRT-LLM plugin architecture, how kernel fusion works at the
  engine level. You already own the quant numerics; this is "how a commercial engine packages them."

**Block H: Edge deployment (≈ L13–L14 + hardware labs)**
- *Teaches:* llama.cpp (laptop), Raspberry Pi (ARM quant), Android (SmolChat), Jetson (CUDA edge).
- *You own it:* **nothing** — edge deployment is entirely outside this repo's scope.
- *Gap:* **G5 — edge deployment is a genuine 0%-owned track.**
- *Mentor decision:* is edge a target? Your repo is datacenter-inference-focused. Edge is a *different
  career vector* (on-device AI, mobile). If not targeting it → reading-tier awareness only. If
  targeting it → a focused llama.cpp + one-device-deploy sprint (1–2 weeks).

**Phase 2 Capstone:** "OpenClaw-RL: Self-Improving WhatsApp AI Assistant" (RL-improving agent).
→ Overlaps your A5 alignment track (`algos/`, `rewards/`, `envs/`) + the F11 agentic-RL frontier
ablation. **You own the RL substrate; the capstone value for you is the deployment + RL-loop-close.**

---

## 2. The four genuine gaps (ranked by hiring-EV for *your* target roles)

| # | Gap | Cost | Why it matters | Mentor action |
|---|---|---|---|---|
| **G1** | **Run production vLLM/SGLang on a real model, compare to your engine** | ~1 H100 day ($25–60) | A1 §9 principal rubric requires you to diagnose "the gap to vLLM and why." You can't without running it. This is the single highest-EV gap. | P1 rental day: serve Llama-3.1-8B on vLLM, measure goodput-under-SLO, then run the same model on your engine, write the honest gap postmortem. |
| **G2** | **SGLang RadixAttention prefix cache** | 2–3 days, standing box | Your A1 §7 stretch goal, unbuilt. Cross-request prefix reuse (multi-turn/few-shot) is a 6.4× win (SGLang). It's the natural next serving feature after paged-KV. | Build a radix-tree prefix cache over your `PagedKVCache`; benchmark on a multi-turn trace. |
| **G3** | **Ray Serve distributed serving** | reading: 2h; build: 3–5 days | Entirely absent. But per your `PERFORMANCE_TRACK.md` §4, distributed *reimplementation* is a subtract-item. Decide: reading-tier (the likely answer) or a focused deploy. | Read Ray Serve docs + one deploy; decide explicitly whether to build. Don't let it drift. |
| **G4/G5** | **TRT-LLM + edge (llama.cpp/Pi/Android/Jetson)** | reading: 2h each; deploy: 1–2 wk each | Different career vectors. TRT-LLM = NVIDIA-internal tool (reading-tier). Edge = on-device AI (only if you're targeting mobile/embedded roles). | Reading-tier unless you're explicitly pivoting to edge. Don't dilute the datacenter focus. |

**The gaps that DON'T matter (you already own them deeper):** KV-cache, roofline, GQA/MLA,
FlashAttention-2, continuous batching, PagedAttention, speculative decoding, CUDA graphs, quantization
numerics, TP/PP/EP comms algebra. The workshop teaches these at survey level; you've *built and
measured* them. Re-deriving them from blank (the P0.5 rebuild ladder) is the mastery path — not
re-learning them.

---

## 3. The mentorship protocol (how we run this together)

This repo already has the machinery — this section wires the inference track into it.

### 3.1 The two-track mandate (hold both, per `CONTEXT_ENGINEERING.md` §3.8)

1. **Forced first-principles mastery** — derive → visualize 3 ways → predict → test-first → **teach-back
   gate** → connect-to-frontier. Run via `/master <concept>`.
2. **Relentless execution** — always build the next load-bearing step; never idle. Run via `/next`.

The teach-back is a **concept gate** (don't advance until you can explain it), **never a commit gate**
(green-CI is the only commit blocker). Hold both: understanding is forced; bureaucracy is not.

### 3.2 The Feynman teaching loop (the mentor's core method)

For each inference concept, we run this 6-step loop (it's the `GPU_FROM_ZERO` protocol, already proven
on the kernel track):

```
DERIVE   — you derive the math from first principles, on paper, no AI (I ask Socratic questions only)
VISUALIZE — three lenses: (1) the tensor-shape diagram, (2) the memory-traffic diagram, (3) the timeline
PREDICT  — write the number and the bound BEFORE the run (roofline-first; FOP-3)
BUILD    — test-first: the failing oracle test exists before the implementation (FOP-3 DoD = a profile)
TEACH-BACK — explain it to me as if I'm a smart junior; I play confused-student and ask "why?"
CONNECT  — tie it to the frontier (the paper, the production system, the interview question)
```

**Why I refuse to hand you derivations or kernel bodies** (the evidence, `KERNEL_ROADMAP_2026.md` §6.5):
the Anthropic skill-formation RCT (2601.20245) found AI-assisted learners scored 50% vs 67%, with the
worst gap on **debugging** — but *conceptual-inquiry-only* interaction scored ≥65%. Retrieval practice
(AI-quizzing) lifted accuracy 73%→89% (arXiv 2507.05629). **So: I quiz, I ask, I sabotage — I do not
lecture or type the answer.** My five jobs: Socratic tutor · scaffolder (failing tests/benches) ·
profiler analyst · saboteur · adversarial reviewer.

### 3.3 The inference-specific quarry reps (pick 1/day, 30–45 min, ledger every rep)

These extend the kernel-track P0 quarry (`KERNEL_ROADMAP_2026.md` §6) to inference. Each is retrieval
practice, not re-learning.

| Rep | Drill | Gate |
|---|---|---|
| **I1 Blank-page decode roofline** | From blank: derive AI of decode vs prefill, compute the HBM_BW/2P ceiling for a 7B BF16 model on H100, state the bound. Timed 15'. | number within ±10% of `bench/RESULTS.md` R1; correct bound = memory |
| **I2 KV-sizing from memory** | Write `KV_bytes = 2·L·H_kv·d·dtype·seq·batch` from blank; compute for Llama-3-8B at seq 8k, batch 32; state when KV > weights. | formula correct; crossover seq identified |
| **I3 Continuous-batching model** | Derive `speedup ≈ max_len/(mean_len + B·τ_p/τ)` from blank; predict continuous vs wave on a named trace. | matches the measured 2.30× within the model's band |
| **I4 Paged-attention block-table** | Draw the logical→physical block map for 3 requests (one shared-prefix, one scattered, one mid-block); explain CoW. | the diagram is correct; CoW trigger named |
| **I5 Spec-decode losslessness** | Derive the rejection-sampling rule that makes spec-decode distribution-identical; name the classic bug. | the rule is correct; "output changed ⇒ rejection rule wrong" named |
| **I6 Teach-back: MLA absorption** | Explain why `(W^UQ c^Q)·(W^UK c^KV)` lets you attend in latent space, and why RoPE breaks it. | ≥B grade; the decoupled-RoPE pathway named |
| **I7 Sabotage drill** | I plant a bug in `continuous.py` / `PagedKVCache` (wrong cache slice, off-by-one position, CoW not triggering, admission guard leak); you get the symptom, fix under 25'. | root-cause named *before* the fix |

### 3.4 The rebuild ladder for inference (P0.5, the deep-work slot)

The from-scratch inference engine was agent-built. Before the P1 rental day, re-derive and re-type the
core serving ladder from blank in `mastery/src/mastery_llm/` (the mirror skeleton that already exists),
via `/rebuild`. Rungs (each: derive on paper → study the agent version timed → blank rebuild vs oracle
→ predict-then-bench vs the ledgered number → diff-defend → `/feynman`):

```
IL0  decode roofline + the HBM_BW/2P ceiling           (≈ 1 block)
IL1  KV-cache (contiguous) + the sizing formula          (≈ 1 block)
IL2  GQA/MQA cache reduction                             (≈ 0.5 block)
IL3  static → continuous batching scheduler              (≈ 2 blocks)  [the headline]
IL4  PagedKVCache (block table, CoW, admission guard)    (≈ 2 blocks)  [the headline]
IL5  paged decode kernel (Triton)                        (≈ 2 blocks)
IL6  speculative decoding (rejection sampling)           (≈ 1.5 blocks)
IL7  CUDA-graph decode capture                           (≈ 1 block)
IL8  MLA latent cache + absorption identity              (≈ 1.5 blocks)
```

≈ 12.5 blocks. **Consequence, accepted:** the P1 H100 rental day slides ~2 weeks — measuring the
agent-built engine on a real GPU before the human owns the sm120 ladder would mint indefensible
inventory (same logic as the kernel P0.5 resequence).

---

## 4. The phased plan (dated, subordinate to the existing fronts)

This integrates with — does not replace — the four active fronts in `STATUS.md`. Inference mastery is
the *overlay* on the perf front.

| When | Milestone | Evidence |
|---|---|---|
| Now → +2 wk | **P0.5 inference rebuild ladder IL0–IL8** (`/rebuild`, mastery/, from blank) + daily I-rep quarry | every core serving component re-typed by the human vs oracle + ledger; teach-backs climb from 12/89 (PERF half 0→~10) |
| +2 wk (post-ladder) | **P1 H100 rental day** (the perf-track Phase-2 day, ADR-0012; serves as the Vizuara "7B server" checkpoint for you): serve real 7B on your engine + on vLLM, write the gap postmortem | **G1 closed**: goodput-vs-vLLM diagnosis with numbers the human defends cold |
| +2.5 wk | **G2 RadixAttention** build (standing box): radix-tree prefix cache over your paged-KV; multi-turn benchmark | 6.4×-class prefix-reuse measurement on a few-shot trace |
| +3 wk | **G3 Ray Serve decision** (explicit: reading-tier or build) + the perf-track **A6 8×H200 serving day** (ADR-0012 Phase 4 — the frontier-MoE serving day: DeepSeek-R1 FP8, TP×EP, MLA KV, PD-disagg) | either a Ray Serve deploy or a documented reading-tier decision (no drift) |
| parallel (K3 track) | **K9 = 8×B300 Modal rental** (~$120–170, per `docs/k3/ROADMAP.md`) — the **separate** K3-track serving day to re-measure the hosting book's numbers on a rebuilt Kimi K3. *Distinct from the perf-track 8×H200 day above.* | K3 ROADMAP K9 runbook executed; measured numbers ledgered |
| extraction | **P4.1 upstream PR** to vLLM/SGLang/FlashInfer (a fix from your ncu/gap findings, or your RadixAttention implementation) | the hiring artifact (pathway #2 in every stream) |

**Budget:** ≈ $25–60 (perf P1 H100) + $0 (standing box for G2) + ~$50–120 (perf P2 B200) + ~$22–24/hr
(perf A6 8×H200) + ~$120–170 (K3 K9 8×B300). **$0 tuition** — the workshop is strictly dominated for
your level (same verdict as the kernel track, V2).

---

## 5. The interview surface (what "defend cold" means for inference)

By the end of this overlay, you can walk into a frontier-lab inference loop and, with no AI, no notes:

1. **Draw the decode roofline** for any model/GPU and predict tok/s before measuring (FOP-3).
2. **Size the KV cache** for any config and state when it exceeds weights (the capacity constraint).
3. **Derive the continuous-batching speedup** and explain the ITL cost (the throughput↔latency trade).
4. **Explain PagedAttention** as virtual memory: block table, CoW, the 60–80%→<4% waste reduction.
5. **Prove spec-decode is lossless** (the rejection rule) and name the classic bug.
6. **Explain MLA's weight-absorption trick** and why RoPE needs the decoupled pathway.
7. **Diagnose "why is my engine 1.7× off vLLM"** with the specific evidence (the principal-rubric skill).
8. **Name the 2026 production stack**: DeepEP/EPLB/DBO, PD-disagg (architecturally required for wide EP),
   NIXL KV transfer, AFD, RadixAttention, FlashInfer, the deterministic-inference frontier.
9. **Explain the spec-decode batch-erosion curve** (the senior-vs-principal insight: gains erode past B≈8).
10. **Articulate goodput-under-SLO** as the metric that pays the bills (DistServe).

This list is your self-test. Any item you can't deliver cold → that's the next quarry rep.

---

## 6. Pointers

- **The spine:** [`performance/A1_transformer_inference.md`](../performance/A1_transformer_inference.md) (the
  from-scratch build ladder + frontier-2026 core + principal rubric).
- **The kernel twin:** [`performance/KERNEL_ROADMAP_2026.md`](../performance/KERNEL_ROADMAP_2026.md) §3
  (the Vizuara *Kernel* workshop dissection — same verdict, V2).
- **The thesis layer:** [`docs/PERFORMANCE_TRACK.md`](PERFORMANCE_TRACK.md) §0 (the decode memory-wall,
  owned end-to-end) + §2 (the 2026 findings).
- **The mastery deficit:** [`docs/learning/PROGRESS.md`](learning/PROGRESS.md) (the live ledger:
  12/89 Bài cold-defended as of 2026-07-14, PERF half 0/41 — the binding constraint).
- **The evidence for learn-mode:** [`performance/KERNEL_ROADMAP_2026.md`](../performance/KERNEL_ROADMAP_2026.md)
  §6.5 (the Claude-stack RCT evidence — why I refuse to type derivations/kernel bodies).
- **The serving engine:** `src/scratch_llm/serving/continuous.py` (490 LOC, the headline artifact).
- **The measurement ledger:** [`bench/RESULTS.md`](../bench/RESULTS.md) (every number, predict-vs-measured).
