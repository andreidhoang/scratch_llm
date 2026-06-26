# STRATEGY — frontier-lab readiness, GDM signals & the ship-next

> **Single source of truth** for where this project stands against frontier-lab hiring, how it maps to the
> GDM/Feinberg signals, the serving-stack OSS lane, and the one prioritized action list. Synthesized from
> three now-removed working docs (the frontier-readiness verdict, the GDM signal-alignment note, and the
> serving/OSS lane) — this doc is their canonical merge. Last synthesis: 2026-06-18.
> Re-verify any version/number in §6–§7 before quoting it live.

---

## 1. The verdict (where this stands)

**Planning, design and direction are frontier-grade; the gap to an offer is execution, not taste.** A job is
landed on shipped, defensible, differentiating artifacts, and the two differentiators — the A5 RL "aha" and
the DELTA kernel — are **0% built**, while what *is* shipped is A1 (commoditized) plus the easy half of A2.
The git history shows code froze on Jun 8 and the days since produced only strategy documents, at a **5.3:1
doc-to-code byte ratio**. The one-liner: **you are planning like a frontier lab; you are not yet shipping
like one.** The plan→artifact conversion is the entire remaining job.

On the planning axis the self-grade of "~80% aligned" is fair (closer to ~90%). On the **execution axis it is
~25%**, and that is the axis a committee hires on.

---

## 2. What's verified (grounding — not assumed)

| Claim under test | Method | Result |
|---|---|---|
| GatedDeltaNet-2 (arXiv 2605.22791) exists & matches the RFC | web → arXiv + NVlabs repo + HF | **TRUE.** NVIDIA (Hatamizadeh, Choi, Kautz); channel-wise erase/write decoupling; chunkwise Triton *training* kernels; 1.3B / 100B FineWeb-Edu. The "training-code-only, no checkpoint" premise the whole scope rests on holds. |
| USC FPGA paper (arXiv 2603.05931) exists & matches | web → arXiv | **TRUE.** Persistent-state, 5-phase→**1 read / 1 write** per token, GVA paired-head, 63 µs/token, **4.5× over H100 PCIe**. The "GPU analog of the 2-pass restructuring" framing is sound and non-redundant. |
| Shipped code is production-grade, not glued together | read `flash_attention_triton.py`, `monitors.py` | **TRUE.** FA2 fwd is a correct online-softmax kernel (fp32 accumulators, causal triangle-skip, autotuned for a fair SDPA comparison). `monitors.py` separates the three KLs, logs IS/ESS, and makes omitting a guardrail a *type error*. |
| Execution pace | `git log` | **14 commits, all Jun 6–8. Nothing since.** |
| Doc:code ratio | byte count | **~491 KB md : 93 KB py (5.3:1)**; `FRONTIER_PRACTICE_2026.md` alone is ~175 KB. |

The premise being real is the single most important finding — most "novel kernel" capstones die because the
paper was misremembered. This one isn't.

---

## 3. Frontier-practice & GDM-signal scorecard

Each row is a real frontier-ML phase or a Feinberg signal, mapped to the artifact. This is the strong layer.

| Frontier practice / Feinberg signal | Current location | Grade / verdict |
|---|---|---|
| **Spec before code** (hypothesis → falsifiable numeric predictions → scope → kill criteria) | DELTA RFC §2–§4, §10 (P1–P7 with explicit falsifiers; "what this is NOT"; kill criteria up front) | **A+** — exceeds the bar; most candidates never write this |
| **Lock evals before architecture** | DELTA RFC §6 (correctness oracle + Nsight roofline harness before any kernel opt) | **A** |
| **Overfit/sanity gates** | loss-at-init ≈ log V, overfit-one-batch (`optim.py` tests), the 512-step drift test | **A** |
| **Mandatory RL logging** | `monitors.py`: KL(cur‖ref), KL(cur‖old), **KL(train‖infer) w/ 0.10 HALT**, IS/ESS, reward+length | **A** — built & tested; scarce skill shipped |
| **Ablate one variable, kill fast, document negatives** | DELTA RFC §8 + §10 (P2 fail = publishable negative, cf. DeepSeek-R1 appendix) | **A** |
| **GPU perf discipline** (profile/roofline first, lock clocks, measured not theoretical peak) | DELTA plan §1; P1 denominator = STREAM microbench on the exact SKU | **A** |
| **Research-as-stochastic-MDP / taste** | the barbell = EV bet on a stochastic DAG; spike+base sharing infra | **A** — strategically correct, not just buzzword-aligned |
| **Epistemic honesty** | [FACT]/[INFERENCE]/[UNCERTAIN] labels; FlashQLA flagged unverified; the 512 KB/head→32 KB/head self-correction | **A+** — this *is* the forensic-honesty hiring signal |
| **Kernels / low-level opt / backend-at-scale** | A2 FA2-Triton, DELTA fused decode, DDP/ZeRO/FSDP labs (⬜) | **Aligned (build pending)** |
| **Scaling-law recipes + prediction rules** | A3 IsoFLOP+Chinchilla, Huber+bootstrap, extrapolation reporting (⬜) | **Aligned (build pending)** |
| **MFU / inference co-design** | A2 roofline; DELTA "batch-1 decode is memory-bound" | **Partial** — roofline measured, but *MFU-as-anti-goal* + "select matrix topologies to saturate units" not yet named |
| **Quantization** (INT4 weights **+ activation quant**; TCO≈electricity) | A2 FP8 fake-quant lab; INT4/activation/KV-quant are know-it only | **Partial → gap** (see §5.2) |
| **Distillation** (teacher→student; the infra multiplier behind Flash) | only data-label distillation (FineWeb-Edu); no model-KD | **Major gap** (see §5.1) |
| **MoE pipeline-prefill** (expert-parallel vs pipeline layers) | A1 MoE; A2 PP-schedule lab + expert-parallel know-it — not synthesized | **Partial** (see §5.4) |
| **Production serving-stack contribution** (the green flag a hiring manager verifies) | not in the curriculum at all | **Gap → the OSS lane, §6** |

---

## 4. The execution gap + risk register

What a committee actually opens the repo and sees:

| Layer | Claimed | Shipped reality | For the job |
|---|---|---|---|
| A1 substrate | ✅ | real, tested, clean | good — but commoditized; table stakes |
| A2 systems | "partial" | FA2 (53% of SDPA), KV-cache, monitors, rollout seam | the **easy half**; the scarce A2 skill — **DDP / ZeRO-1 / FSDP** — is ⬜; the "100B memory" one-pager is ⬜ |
| A3 / A4 / A5 | "to build" | **empty stubs** | **A5 RL is the scarcest 2026 cluster and it does not exist** |
| **DELTA kernel** | designed, fact-checked | **no kernel code; gated on Step-0** | the headline differentiator; currently a document |

**Risk register (what breaks the plan):**

1. **P3 is the riskiest single deliverable** — "match or beat `fla` recurrent (≥90%)" gates the entire GDN-2
   result, and `fla` is Songlin Yang's kernel. **Derisk P3 on plain GDN first; do not touch GDN-2 until it
   clears.** If it won't clear in budget, the honest pivot is the roofline-characterization + free-decoupling
   thesis, not a speed multiplier.
2. **Three unbuilt high-effort arms, one solo engineer, 4 weeks, bursty rental** (finish-A2-distributed +
   build-A5-RL + build-DELTA). Default failure = three things half-done, none portfolio-grade. Honor the
   barbell kill-criterion now by sequencing base-first.
3. **FA2 at 53% of SDPA** is a double-edged headline — fine *with* the roofline writeup explaining *why*,
   a weakness naked.
4. **Doc-drift is active, not hypothetical** — it is what the days since Jun 8 were. (See §9.)
5. **Step-0 is overdue** (was Jun 14–15: H100 access + checkpoint decision). Everything downstream blocks on it.

---

## 5. GDM / Feinberg technical additions to adopt

The few high-weight signals the specs currently treat as awareness-only or omit. Distillation is the one true
scope addition; the rest are framing/elevation. Sequencing: 5.3 + framing first (cheap), 5.2 next (green
substrate today), 5.1 interleaved with A5.

### 5.1 Model knowledge-distillation — **build-lab** (the major gap)
Feinberg's single most-emphasized infra lever: distillation transfers teacher statistics to a cheap student;
DeepMind rewrote distillation infra 3–4 generations, and one 4-month rewrite "uncovered new scaling laws that
directly enabled Gemini Flash." Serve-cheap students are the product. Currently **zero** model-KD.
*Lands in:* `src/scratch_llm/algos/distill.py` + an A3 distillation-scaling fit.

- **Logit KD (Hinton 2015):** `L = α·CE(y, z_s) + (1−α)·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T))`. The `T²`
  restores gradient scale. This is **forward KL** (teacher = p): mass-covering.
- **Reverse-KL / on-policy KD (GKD, Agarwal 2024):** `KL(student ‖ teacher)` on **student-sampled** sequences
  — mode-seeking, fixes the train/inference mismatch. Reuses the rollout seam.
- **Sequence-level KD (Kim & Rush 2016):** SFT the student on teacher outputs — reuses the A5 SFT step.
- **Distillation scaling-law mini-fit:** student loss vs {student N, distillation tokens, teacher quality};
  reuses the A3 power-law fitter. Deliverable = the compute-allocation insight (when is distilling cheaper).
- **Invariants (test-first):** `T=1, α=1` ⇒ KD loss byte-identical to plain `cross_entropy`; overfit-one-batch
  drives student logits → teacher logits; loss-at-init ≈ log V holds; forward-KL mass-covering vs reverse-KL
  mode-seeking on a 2-mode toy (predict the difference first).

### 5.2 Quantization — **build-lab** (elevate from know-it)
"~99% of TCO is the electricity to power the chips" — dropping operand size cuts power, latency, and $/request.
The "true miracle" is **quantizing runtime activations**, not just weights.
*Lands in:* extend the A2 FP8 fake-quant lab to INT8/INT4 weight **+ activation**.

- Symmetric affine quant `q = round(clip(x/s, qmin, qmax)); x̂ = q·s`; per-tensor → per-channel scale `s`.
- Weight-only INT8 → INT4, then **per-token activation** quant (the hard, valuable part).
- Report **perplexity delta vs bit-width**; tie to TCO (decode is memory-bound → KV/activation bits dominate $/token).
- **Invariants:** bits→16 recovers fp16 perplexity; INT8 weight-only ⇒ small Δ; INT4 **activation** exposes the
  outlier problem ⇒ motivates per-channel / SmoothQuant-AWQ (state the predicted failure first).

### 5.3 GDM signaling deliverable (cheapest, highest-signal)
Feinberg explicitly invited candidates to do the **Scaling Book handwritten exercises** + a
**transformer-from-scratch** exercise, and send a **video walkthrough** — "GDM routinely interviews and advances
candidates who submit these." `scratch_llm/model.py` already *is* the transformer exercise.
*Deliverable:* (a) Scaling Book part-by-part handwritten sheets, (b) the impl mapped to `model.py`, (c) a recorded walkthrough.

### 5.4 Disciplines to name GDM's way (the pieces exist; frame them)
- **Research as a stochastic MDP / taste** — every build-lab pick is an a-priori **EV(success-rate ÷ time)**
  call on a stochastic DAG whose nodes fail, vs the deterministic DAG of SWE.
- **Citation-tree traversal** — root paper → who-cites-it → triage worth in <N min without reading cover-to-cover.
- **MFU as an anti-goal** — 100% MFU = matmul-only, no HBM reads; pick layer shapes/topologies that saturate units.
- **MoE pipeline-prefill (the Flash-2.0 trick)** — pipeline *layers* across chips so experts stay resident and
  the all-to-all hides behind compute, instead of sharding N experts across N chips (latency scales terribly in N).
- **Strategic notes:** stay GPU-pragmatic but add a TPU/XLA/JAX know-it ("golfing the XLA compiler" is the GDM
  serving reality); the internal-transfer route (become the definitive LLM-integration expert for a product area)
  and the FUD rebuttal ("you cannot disbar an AI" → RE roles that close the research↔product↔reliability gap are durable).

---

## 6. The serving-stack OSS lane (the production axis CS336 doesn't cover)

CS336 trains you to **build the stack from scratch** (FA2, KV-cache, quant numerics, distillation, the DELTA
decode kernel). That's the *knowledge*. The green flag a hiring manager actually verifies is **contributing to
the production serving stack** — "optimize vLLM or SGLang, demonstrate disaggregated serving via TensorRT, or fix
distributed KV-cache load-balancing." An **upstream merged PR** proves *useful-to-others on the real frontier
stack* and routes you to maintainers who work **at** NVIDIA / the labs / the serving startups. This lane spends
the curriculum's ability in public; it is a **PR ladder, not an essay.**

### 6.1 Crosswalk — your CS336 artifact → the PR it unlocks
| CS336 artifact | Serving-stack analog | The PR it unlocks |
|---|---|---|
| **A2 FA2-Triton** (53% SDPA + roofline) | vLLM/SGLang attention & custom kernels | a kernel/perf PR, or tighten one with your roofline method |
| **A2 KV-cache** | vLLM PagedAttention / `KVCacheManager`; SGLang RadixAttention | a **KV-connector** PR (`KVConnectorBase_V1`) or paged/prefix-cache fix |
| **DELTA decode kernel** (GDN-2) | FlashInfer / paged-decode; linear-attn decode | a flash-decoding / linear-attn decode-kernel PR (your differentiator, upstreamed) |
| **A2 quantization lab** (FP8→INT4/activation) | `llm-compressor` / compressed-tensors / NVFP4 | a **quantization-method/config** PR (cleanest entry — dedicated SIG) |
| **A5 spec-decode adjacency** (rollout seam, monitors) | EAGLE-3 / MTP in vLLM/SGLang | a speculative-decoding improvement or eval |
| **A2 DDP/ZeRO/FSDP** (⬜) | distributed serving (TP/PP/EP), Dynamo PD | a multi-GPU/backend parity or disaggregation PR |

### 6.2 The 2026 serving landscape (awareness — discuss, don't build)
- **TensorRT-LLM** — now PyTorch-native (`LLM` API, `trtllm-serve`/`trtllm-bench`); FP8/NVFP4/INT4-AWQ, in-flight batching, EAGLE-3/MTP, disaggregated serving + wide-EP.
- **NVIDIA Dynamo** — datacenter orchestration *above* TRT-LLM/vLLM/SGLang: disaggregated prefill/decode, KV-aware routing (~2× TTFT), KV Block Manager (GPU→CPU→SSD), NIXL (RDMA KV transfer). GA Mar 16 2026.
- **NIM** — containers bundling weights + auto-selected backend + OpenAI API.
- **vLLM (V1)** — EngineCore token-budget scheduler; **PagedAttention**; chunked prefill + prefix caching default; disaggregation via **`KVConnectorBase_V1`**; spec-decode; FP8/INT4/AWQ/GPTQ/compressed-tensors/NVFP4.
- **SGLang** — **RadixAttention** (radix-tree prefix reuse) + HiCache; overlap scheduling; jump-forward decoding; PD disaggregation; EAGLE-2/3.
- **Low precision** — FP8 = safe default; **NVFP4** (16-elem blocks, two-level scaling) production-ready on Blackwell. Ties to §5.2.
- **Research currency** — EAGLE-3 (`2503.01840`), DistServe (`2401.09670`), Sarathi-Serve (`2403.02310`), KIVI (`2402.02750`), NSA (`2502.11089`).
- **Extended landscape (reference, 2026-06-22)** — a fuller serving-stack head-to-head (vLLM V1 / SGLang / TRT-LLM / Dynamo), the disaggregation + KV-systems map, the Hopper→Blackwell→Rubin hardware roadmap, the frontier-model table, and the hiring/take-home signals are collected in [`reference/Frontier_Inference_2026_Research_Brief.md`](reference/Frontier_Inference_2026_Research_Brief.md). Reference only — this §6 remains the canonical OSS lane.

### 6.3 The OSS PR ladder (pick ONE primary repo — vLLM *or* SGLang — go deep)
- **Rung 1 — learn the flow (start NOW, parallel with A2; ~2 small merges):** a `recipes` entry; a `[Doc]` PR
  (SGLang recommends this to learn the codebase); a good-first-issue bugfix **+ unit test**, or a quant config/fix.
- **Rung 2 — the substantive PR (after A2 KV-cache / DELTA decode is real):** a **KV connector** against
  `KVConnectorBase_V1`; **or** a quantization method/kernel (NVFP4/compressed-tensors gap on a MoE); **or** a
  **flash-decoding / linear-attn decode** kernel (the DELTA work, upstreamed); **or** a scheduler/perf fix.
- **Rung 3 — sustained presence:** review others' PRs, answer issues, ship a second. Maintainers are the referral pipeline.
- **Process:** vLLM needs **DCO sign-off** (`git commit -s`) + PR-title prefixes (`[Bugfix]`/`[Kernel]`/`[Model]`/`[Doc]`), RFC for >500 LOC. SGLang CI is maintainer-gated. Build from source + run tests first; keep PRs small + single-purpose.

---

## 7. Role targets & sequencing

Goal: **inference/systems research engineer, 6–12 months** (offer window ≈ Dec 2026 – Jun 2027). The PR *is* the
application — it bypasses résumé screens.

- **Tier 1 (OSS-driven, faster, visa-friendlier) — start applying ~when the Rung-2 PR is open:** **NVIDIA**
  (*AI Inference Performance Engineer* / *LLM Inference Frameworks* — JDs literally list "contributing to
  TensorRT-LLM/vLLM/SGLang"), plus the **NVIDIA Vietnam R&D Center** (a possible no-relocation path — get on the
  req list early); **Baseten** (*GPU Kernels / Model Performance*), **Fireworks**, **Together**, Modal, Groq.
- **Tier 2 (reach; visa + relocation, higher bar):** OpenAI (*RE*, inference/scaling), Anthropic (*Performance
  Engineer, Inference Systems / GPU*), Google DeepMind. Entry = a substantive merged PR + the DELTA artifact +
  a maintainer referral + the GDM signaling video (§5.3).

---

## 8. The ship-next (the one canonical action list)

The review's ruthless cut, with the OSS lane interleaved (parallel + long-lead, ~2 hrs/wk). **Base-first** —
the base is higher-EV *and* lower-variance, and it derisks the spike.

1. **Freeze the docs. Today.** No new strategy/practice docs until something compiles that didn't before. (§9.)
2. **Close DELTA Step-0 this week** — H100 provider + price + locked-clock test run; commit to architecture-only
   + projected e2e. Half a day, blocking.
3. **Ship the A5 RL "aha" first** — GRPO/Dr.GRPO on Countdown with Qwen2.5-1.5B (~$30–100; emerges at 1.5B,
   fails at 0.5B). `monitors.py` is already built, so the guardrails are free. **Highest-EV move on the board.**
4. **OSS Rung-1 in parallel, from Week 1** (~2 hrs/wk) — one `recipes`/`[Doc]`/quant-config PR to vLLM **or**
   SGLang. Long review latency = start now.
5. **Then DELTA, P3-first** — match `fla` recurrent on plain GDN before a line of GDN-2. The decode kernel is
   also the OSS Rung-2 target (a flash-decoding PR).
6. **Bank the free wins** (days each, all currently ⬜): the **100B-memory one-pager**; **tighten FA2 past SDPA
   or ship the roofline writeup** explaining the 53%; the **GDM signaling video** (§5.3).

**Definition of "landed":** ≥3 small merged PRs + 1 substantive PR, on top of a converged, explainable RL run +
an honest decode-kernel result (speed *or* characterization) + the 100B one-pager + the GDM video. That package
clears the NVIDIA/Baseten/Fireworks bar and is a credible reach at OpenAI/Anthropic/DeepMind. The current
portfolio (A1 + half A2 + docs) does not.

---

## 9. Anti-bloat (honor your own review)

- **No more strategy/practice docs.** Doc-to-code is already 5.3:1; the marginal planning doc has ~zero hiring EV.
- **Don't duplicate the A2/DELTA/L2 specs** — point at them, don't re-spec them.
- **Don't go wide** across all four serving frameworks — one primary repo, deep.
- **Don't let the OSS lane displace the #1 move** (the A5 RL run). OSS is the parallel long-lead lane, not the headline.

---

## Sources & currency caveats

**Foundational premise (verified):** [arXiv 2605.22791 — Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) ·
[NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) ·
[arXiv 2603.05931 — USC persistent-state decode (FPGA)](https://arxiv.org/pdf/2603.05931).
**Serving stack:** `github.com/NVIDIA/TensorRT-LLM` · `github.com/ai-dynamo/dynamo` · `developer.nvidia.com/nim`
· `github.com/vllm-project/vllm` (+ `llm-compressor`, `recipes`) · `github.com/sgl-project/sglang` ·
`roadmap.vllm.ai` · `docs.sglang.ai`. **Papers:** EAGLE-3 `2503.01840` · DistServe `2401.09670` · Sarathi-Serve
`2403.02310` · KIVI `2402.02750` · NSA `2502.11089`. **Roles:** `jobs.nvidia.com` · `jobs.ashbyhq.com/baseten` ·
`job-boards.greenhouse.io/anthropic` · `openai.com/careers`.

*Caveats: serving-stack versions drift (vLLM ≈ v0.20.x / SGLang ≈ v0.5.x at synthesis time; TRT-LLM stable tag
may have moved); some Blackwell/Rubin figures were vendor/secondary. Verify before quoting live. The FlashQLA
prefill multipliers remain unverified (see DELTA.md §1).*
