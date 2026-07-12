# The Frontier Inference Engineering Curriculum (2026) — v2, research-grounded

> **Status — reference / external research, added 2026-06-22.** Imported into `cs336/` as on-demand reference, **subordinate to the repo's canon**. The inference *plan of record* is [`../DELTA.md`](../DELTA.md) (the decode-kernel capstone spec) + [`../STRATEGY.md`](../STRATEGY.md) §6 (the serving-stack OSS lane); the canonical **2026 frontier layer** is [`../scratch_llm/docs/FRONTIER_PRACTICE_2026.md`](../scratch_llm/docs/FRONTIER_PRACTICE_2026.md). This file is broad landscape/learning context — **not** a competing source of truth. Index + overlap map: [`README.md`](README.md). Evidence base: [`Frontier_Inference_2026_Research_Brief.md`](Frontier_Inference_2026_Research_Brief.md).

### A first-principles path to contributing real value at NVIDIA-, Anthropic-, OpenAI-, DeepMind-caliber inference teams

*Mentor framing: I'm writing this as a principal inference engineer would brief a high-potential hire. No cargo-culting, no hand-waving. Every module exists because it changes a number you can measure.*

*This v2 is grounded in a five-agent deep-research sweep of the June-2026 frontier. The numbers, names, and dates here are sourced in the companion **`Frontier_Inference_2026_Research_Brief.md`** — read it alongside this. Where I cite a figure, the brief has the URL.*

---

## 0. How to use this

Three rules, and they are the whole method:

1. **Derive before you read.** For every technique, first work out from physics why it *should* help — how many bytes or FLOPs it removes — then read the paper and check your prediction against their numbers. If they disagree, you misunderstood the technique; find out why. This is how you build the intuition to invent, not just reproduce.
2. **Build to understand.** You don't understand PagedAttention until you've written a fragmenting KV allocator and watched it stall. You don't understand FlashAttention until your naive kernel OOMs on the attention matrix and the tiled one doesn't. The capstone ladder (§4) is the spine, not an afterthought.
3. **Measure everything, trust nothing.** A profiler trace is how careers advance; "it feels faster" is how they stall.

**The single mental model that subsumes the field:** LLM inference is a **bytes-and-FLOPs problem on a memory-bandwidth-bound machine.** Decode streams weights + KV out of HBM (bandwidth-bound); prefill saturates tensor cores (compute-bound). Every technique here is one of: *move fewer bytes* (quantization, GQA/MLA, KV compression), *move them once for many requests* (batching, prefix caching), *do more FLOPs per byte* (FlashAttention, fusion), *skip work* (speculative decoding, sparsity), or *put the work on the right machine* (disaggregation, parallelism). Hold that taxonomy and the field stops being a list of tricks and becomes a design space.

**What changed since the v1 draft (and why it matters):** the workload flipped to **decode-dominant, agentic, reasoning-heavy**; the binding constraint moved from FLOPs to **memory (KV bytes, expert residency, HBM bandwidth)**; **NVFP4/Blackwell FP4** displaced INT4 as the quantization frontier; **disaggregation went mainstream** (the live debate is now *when to re-aggregate*); **FlashInfer/FA4** became the Blackwell attention backend; **TGI was archived (March 2026)**; and the talent bar crystallized around **"make this kernel faster" in CUDA** + a **published beat-a-baseline benchmark**. The curriculum below is rebuilt around those facts.

---

## 1. North star: what "frontier inference value" actually is in 2026

**The economics.** Inference is now the dominant compute cost of any deployed model, and two shifts amplified it: models reached hundreds of millions of users, and **test-time compute** (reasoning models emitting 5K–50K-token chains of thought) turned short answers into long ones — NVIDIA frames reasoning as **~100× more inference compute** per query, and reasoning tokens are billed even though the user never sees them. A 20% cut in $/token at fixed quality multiplies across the fleet and beats most model-quality tweaks. **That is where your value is created.**

**The physics.** A decode step reads every weight from HBM to produce one token, at **arithmetic intensity ≈ 1 FLOP/byte** — against an H100 ridge point of ~295 (BF16) / ~591 (FP8) FLOPs/byte, i.e. ~150–600× below the knee. **Decode is profoundly memory-bandwidth-bound.** Prefill processes the whole prompt in parallel against the same weights, so it's **compute-bound**. This one dichotomy explains chunked prefill, disaggregation, why batching is the throughput lever, and why quantization helps decode (fewer bytes) and prefill (faster tensor cores) for *different* reasons. Concrete ceiling to memorize: a 70B FP16 model on one H100 tops out near **`3.35 TB/s ÷ 140 GB ≈ 24 tok/s` at batch 1** — pure bandwidth.

**The metrics you live by:** **TTFT** (prefill-bound), **TPOT/ITL** (decode-bound), **throughput** (cost driver), **goodput** (throughput that meets SLOs — the only kind worth money), **$/M tokens**, and **MBU/MFU** (bandwidth/FLOPs utilization — decode is an MBU game, prefill an MFU game). You are paid to push **goodput up and $/token down at a fixed quality bar and latency SLO**. Everything below serves that sentence.

---

## 2. Critical teardown of the workshop curriculum you pasted

The Vizuara outline is a good survey — better than most. You asked me to keep only what compounds toward frontier value, so:

**Keep — core and well-chosen.** KV cache (correctly placed early — it's the central data structure), attention variants Parts 1 & 2 (MHA/MQA/GQA/MLA/sparse, then sliding-window/SSM/Mamba), FlashAttention 1/2/3, "anatomy of a vLLM step" + SGLang RadixAttention, quantization, speculative decoding + MTP, parallelism, and disaggregated serving. The final lectures (guided decoding, eval harness, cold starts, canary, cache-aware routing, guardrails, million-user system) are the most job-relevant material in the whole syllabus — most courses never reach them.

**Cut or demote — noise for *your* goal (frontier datacenter inference).** The edge/embedded track (phone, Raspberry Pi Zero 2W, Jetson) teaches a *different* specialization; it won't transfer to an NVIDIA datacenter-inference or Anthropic-serving interview — keep one lab as breadth, drop the rest. "Inference and YCombinator" / "Ideas Journal" is startup ideation, not engineering — cut from the technical track. "Finetuning + Distillation" is mostly a *training* topic — keep only the *distillation-for-inference* slice. "Embodied AI / World Models" and the "Voice Agents" capstone are *applications* — optional, not core. "Multimodal" stays but lighter.

**Add — missing for 2026 (the research made these non-negotiable).** (1) A real **GPU kernel track** — Triton **and** CUDA/CUTLASS/CuTe, Nsight roofline profiling; the workshop treats FlashAttention as a concept, but the job is "make this kernel faster" in CUDA. (2) **FP4 / NVFP4 on Blackwell** — the precision frontier moved past FP8. (3) **Large-scale Expert Parallelism** (DeepEP, EPLB, the DeepSeek prefill-EP32/decode-EP144 recipe) as its own deep module — *the* 2025–26 systems problem. (4) **FlashInfer** as the real backend, not just the FA papers. (5) **Agentic / KV-reuse serving** — 100:1 input:output, >95% prefix reuse, cache-hit-rate as the dominant cost lever. (6) **Test-time-compute / reasoning economics** as a first-class module. (7) **A rigorous benchmark/eval discipline** (TTFT/TPOT/goodput done right, MLPerf methodology). (8) An **OSS-contribution playbook** — the bridge from study to a job.

---

## 3. The curriculum: a first-principles dependency chain

Ordered so each module is the prerequisite for the next. The later modules are unintelligible without the earlier physics. Each lists *why it matters → core concepts → what you build → the grounded 2026 anchor*.

**Module 0 — The forward pass, exactly: FLOPs, bytes, roofline.** *Why:* you can't optimize what you can't account for. *Concepts:* per-token decode ≈ `2·P` FLOPs and ≈ `P·bytes/param + KV_bytes`; prefill ≈ `2·P·T` + attention `O(T²)`; arithmetic intensity, ridge point, MBU vs MFU; why batch amortizes weight bytes but not KV. *Build:* a roofline model for Llama-3-8B — predict TTFT/TPOT on H100, then measure and reconcile (→ Capstone 0). *Anchor:* decode AI ≈ 1 FLOP/byte; H100 ridge ~295/591; 70B@H100 ≈ 24 tok/s batch-1.

**Module 1 — The KV cache: the object the system is built around.** *Why:* it's why decode is cheap-per-step and why *memory, not FLOPs*, caps batch size — the highest-leverage systems topic. *Concepts:* `KV_bytes = 2·L·n_kv_heads·d_head·dtype·seqlen·batch`; fragmentation; **PagedAttention** (KV as virtual memory — blocks, block tables, copy-on-write; waste 60–80% → <4%, 2–4× throughput); **prefix caching / RadixAttention** (radix tree over KV; longest-prefix match); KV **quantization** (FP8 KV is now production-default, 2× capacity) and **offloading** (LMCache: GPU→DRAM→SSD→S3). *Build:* a paged KV allocator with block table, prefix sharing, eviction (→ Capstone 1). *Anchor:* vLLM hash-based APC vs SGLang radix tree; agentic traces ~94% cacheable.

**Module 2 — Attention efficiency (the algorithm): shrinking KV and compute.** *Why:* the cheapest byte is the one you never store; this sets the serving-cost ceiling. *Concepts:* the ladder **MHA→MQA→GQA** (the G=H/4…H/8 sweet spot) **→ MLA** (DeepSeek: latent compression, **~576 elements/token, ~57× smaller than MHA, *better* quality**); **sliding window + attention sinks** (Gemma 3 5:1, GPT-OSS 128-token); **Native Sparse Attention → DSA** (training-native top-k=2048, **~11.6× decode at 64K**, ~50% long-context price cut); **linear/SSM hybrids** at 3:1–7:1 (**Kimi Linear beat full attention under fair comparison**; Qwen3-Next ~10× throughput at 32K+; MiniMax-M1 ~25% of R1 FLOPs at 100K). *Build:* implement GQA + toy MLA; plot the memory/quality frontier. *Anchor:* sparsity-and-hybrid is the year's biggest architectural shift.

**Module 3 — Attention kernels (the systems): FlashAttention & FlashInfer.** *Why:* where algorithm meets silicon, and where NVIDIA-caliber engineers separate from API users. *Concepts:* the naive `O(T²)` attention matrix is the enemy; **online softmax** (single streaming pass); **tiling / IO-awareness** (FA-1); work partitioning (FA-2); **FA-3** (Hopper: warp specialization, TMA, FP8 — ~740 TFLOP/s, ~75% util); **FA-4** (Blackwell, CuTeDSL — **~1,605 TFLOP/s on B200, first attention kernel past a petaflop**); **FlashInfer** (MLSys-2025 best paper, unified block-sparse KV + JIT — the *de-facto backend*, hardware-selected). *Build:* a Triton attention kernel (online softmax + tiling); benchmark vs SDPA; profile in Nsight Compute and read the roofline (→ Capstone 2). *Anchor:* the "default backend" is now FA4-on-Blackwell / FA3-on-Hopper / FlashInfer.

**Module 4 — Numerical precision & quantization: fewer, faster bytes.** *Why:* attacks decode (fewer bytes → higher MBU) and prefill (faster low-precision tensor cores → higher MFU) at once — top ROI after batching. *Concepts:* the ladder FP16/BF16 → **FP8** (standard) → **FP4** (**NVFP4** 16-elem blocks + E4M3 scale, lower error than **MXFP4** 32-elem + E8M0; Blackwell-native, "**same accuracy at 2.3× throughput vs FP8**"); weight-only (AWQ/GPTQ — W4A16 for memory-bound batch-1) vs weight+activation (SmoothQuant, FP8 — for compute-bound batch); KV quant; **MoE quant** (per-expert scales); **QAD** (quantization-aware distillation) for FP4 accuracy recovery; the sub-4-bit wall (ParetoQ 2–3-bit learning transition). *Build:* W4A16 with a **fused dequant-GEMM Triton kernel**; perplexity + task accuracy vs BF16; stretch FP8/NVFP4 (→ Capstone 2). *Anchor:* GPT-OSS ships MXFP4 (120B on one 80GB H100); FP4 training within <1.5% of FP8 loss. *Tooling:* TensorRT Model Optimizer, llm-compressor.

**Module 5 — Batching & scheduling: the throughput engine.** *Why:* amortizes weight-byte movement across requests — the 1× → 20×+ lever. *Concepts:* static batching's waste; **continuous/in-flight batching** (Orca — up to ~23×); **chunked prefill** (Sarathi — interleave compute-bound prefill with memory-bound decode, exploiting the split); the scheduler maximizing goodput under KV-memory and SLO constraints; preemption, recompute-vs-swap. *Build:* add a continuous-batching scheduler to your Module-1 engine; plot throughput-vs-latency vs static (→ Capstone 1). *Anchor:* vLLM V1's **unified token-budget scheduler** erases the prefill/decode distinction in code.

**Module 6 — Speculative decoding & MTP: breaking the sequential barrier.** *Why:* decode is sequential and memory-bound, so the GPU idles on compute; speculation fills it for multiple tokens per memory-bound step. *Concepts:* draft-then-verify (lossless via rejection sampling); expected accepted tokens ≈ `(1−α^{k+1})/(1−α)`; **Medusa → EAGLE-3** (NeurIPS 2025, **up to 6.5×**, acceptance 0.75–0.85, *sustains at higher batch*); **MTP** (DeepSeek-V3 native, >80% acceptance → ~1.8× at batch 1); **P-EAGLE** (parallel draft, **1.05–1.69× over EAGLE-3 on B200**); why gains decay as batch fills but **recover under long context** (the reasoning regime). *Build:* EAGLE-style or draft speculation; measure acceptance + end-to-end speedup on a reasoning workload; show the high-batch crossover (→ Capstone 3). *Anchor:* vLLM V1 dropped the generic draft-model path; EAGLE-3/MTP/P-EAGLE are the live set.

**Module 7 — Parallelism for inference: the interconnect decides.** *Why:* frontier models don't fit on one GPU, and the network dictates the split. *Concepts:* **TP** (intra-layer; all-reduce every layer → needs NVLink; intra-node), **PP** (cross-layer; bubbles), **EP** (MoE; all-to-all), **DP/replicas**, **SP/CP** (long context). The governing rule: **match parallelism to interconnect topology**, and pick differently for prefill vs decode. *Build:* run TP across 2–4 GPUs; measure the all-reduce tax and TP-degree effect. *Anchor:* "5D parallelism" is just the product of these axes; the NVL72 130 TB/s fabric makes the *rack* the unit.

**Module 8 — MoE inference & large-scale Expert Parallelism.** *Why:* frontier open models are sparse MoE; serving them is *the* 2025–26 systems problem and a top hiring signal. *Concepts:* router → top-k experts → combine; **sparsity cuts FLOPs but not VRAM** (all experts resident); EP at scale with **all-to-all dispatch/combine**; **DeepEP** (FP8 all-to-all), **EPLB** (hot-expert load balancing), **DeepGEMM**; the recipe **prefill EP32 / decode EP144**. *Build:* serve a small MoE with EP across GPUs; measure all-to-all cost + load imbalance; add EPLB-style redundant experts (→ Capstone 4). *Anchor:* SGLang's open reproduction — **52.3k in + 22.3k out tok/s/node on 96×H100, ~$0.20/1M output, 5× vs TP**; sparsity trend Mixtral 27% → Kimi K2 3.1%.

**Module 9 — Disaggregated serving: each phase on the right machine.** *Why:* prefill and decode have opposite hardware appetites; co-locating starves one. *Concepts:* separate P/D pools; KV transfer over the fastest link (**NIXL**, not contiguous-only NCCL); the **transfer tax** (Llama-70B KV ≈ 320 KB/token → 3.2 GB for 10k tokens → ~250 ms over 100 Gbps); when it wins (tight TPOT SLOs, scale, bursty/lopsided traffic) and when to *re-aggregate* (DuetServe: only above ~60–70% util). Reference systems: **DistServe, Splitwise, Mooncake, NVIDIA Dynamo**. *Build:* a two-pool P/D split with KV handoff; find the goodput crossover vs chunked prefill (→ Capstone 4). *Anchor:* Dynamo 1.0 GA (March 2026), >30× on DeepSeek-R1 on GB200 NVL72; **Rubin CPX physically splits prefill into dedicated silicon**.

**Module 10 — The production stack & ops.** *Why:* real value ships inside these systems; know one cold, read the others. *Concepts:* read **vLLM V1** or **SGLang** source end-to-end (scheduler, block manager, model runner, FlashInfer backend, sampler); **TensorRT-LLM** for the compiled NVIDIA-max path; the production surface — autoscaling on **queue depth/KV fill** (KEDA), **cold starts** (Run:ai Model Streamer ~4.88 s; Modal snapshot 70→12 s), **cache-aware routing** (the dominant cost lever — llm-d P90 TTFT 0.5 s vs 90 s+ random), **multi-LoRA** (S-LoRA ~2,000 adapters), **guided decoding** (XGrammar/Outlines + its kernel cost), **SLO-aware scheduling/goodput**, **guardrails** (<50 ms/check), the **K8s Gateway API Inference Extension**. *Build:* deploy on vLLM/SGLang on a rented GPU; add prefix caching + a guided-decoding schema; build a proper benchmark harness; **then read the scheduler source and write up exactly what one step does**. *Anchor:* TGI is archived — use vLLM V1 / SGLang.

**Module 11 — Frontier workloads: reasoning, agentic, long context, multimodal.** *Why:* the workload mix moved, and the cost structure moved with it. *Concepts:* **reasoning** (decode-dominant, low-batch, KV grows with CoT; DeepSeek-V3 needs ~18K aggregate batch to be compute-bound on H100; "overthinking" → adaptive budgets, s1 budget-forcing); **agentic** (>95% prefix reuse, 100:1 input:output → KV as a persistent datastore; caching took a real coding task $1.35 → $0.54); **long context** (KV eviction/compression H2O→SnapKV→**R-KV** reasoning-aware: ~100% perf at 10% cache; RULER reality — models use only 50–65% of advertised context); **multimodal** (vision/video inflate tokens 1–3 orders → **disaggregated Encode-Prefill-Decode**, in vLLM 0.11.1, up to 71% TTFT cut; realtime voice sub-250 ms first-packet). *Build:* measure TPOT/cost vs CoT length; implement a KV-eviction policy; quantify the quality/memory tradeoff. *Anchor:* RAG isn't dead — 8–82× cheaper, hybrid consensus.

**Module 12 — Hardware deep dive: Hopper → Blackwell → Rubin, from the metal.** *Why:* a principal engineer reasons in HBM TB/s, NVLink domains, and tensor-core formats. *Concepts:* **H100** (3.35 TB/s) → **H200** (4.8 TB/s, *same die* — the clean proof decode scales with bandwidth) → **B200** (8 TB/s, native FP4) → **GB200 NVL72** (72 GPUs/one NVLink domain, 130 TB/s fabric, ~30× H100) → **GB300** (288 GB HBM3e, +50% for reasoning) → **Rubin** (HBM4 ~22 TB/s; **Rubin CPX** = dedicated prefill GPU on GDDR7). Map each spec to a phase: HBM bandwidth → decode TPOT; tensor-core FLOPs/format → prefill; NVLink-domain size → feasible TP/EP degree. Competitors: AMD MI355X (288 GB @ 8 TB/s, ROCm now first-class in vLLM), TPU v7 Ironwood (inference-focused), Groq/Cerebras (SRAM extremists), AWS Trainium3 (Anthropic's Rainier ≈ 500k chips). *Build:* re-run your Module-0 roofline for two GPUs; predict then verify the TPOT delta from bandwidth alone.

**Cross-cutting track — Kernels & profiling (run from Module 3 on).** CUDA + **Triton** for kernels; **CUTLASS/CuTe** literacy for GEMM/MoE (CuTe DSL, ThunderKittens); **DeepGEMM/FlashMLA** as references. Profiling: **Nsight Systems** (find the stall) → **Nsight Compute** (per-kernel roofline, warp-stall sampling) → torch.profiler/Kineto/Perfetto. The habit: never optimize without a profile, never claim a speedup without a before/after trace. **This track is what makes you NVIDIA-caliber rather than framework-fluent** — and it's exactly the interview ("make this kernel faster," naive-matmul→cuBLAS).

---

## 4. The capstone ladder (compute-tiered, rising signal)

Each capstone is a portfolio artifact: a repo + a benchmark report + a short writeup of what you measured and why. The research sharpened this ladder — three additions (★) are the highest-signal-per-effort items for 2026.

**Capstone 0 — "Inference from first principles" (free Colab T4 / Kaggle, a weekend).** Pure-PyTorch decode loop for a small model, KV-cache on/off, instrumented TTFT/TPOT, and a roofline model proving decode is bandwidth-bound and batching amortizes weights. The deliverable is the *arithmetic-intensity plot + writeup*, not the loop. *Signal: you understand the physics — the thing most candidates fake.*

**Capstone 1 — "Mini-vLLM: PagedAttention + continuous batching" (T4/A100).** A small engine: paged KV allocator + block table, prefix sharing, continuous-batching scheduler, preemption. Benchmark vs naive static batching; show fragmentation collapse and concurrency gains. *Signal: the core of every modern serving system — built, not configured. This is the "design an inference batching system for one GPU" interview, made real.*

**Capstone 2 — "Quantization + a custom kernel" (A100 / local 4090).** W4A16 with a **fused dequant-GEMM Triton kernel**; perplexity + task accuracy vs BF16; **Nsight Compute roofline before/after**. *Signal: kernel-level skill — the rarest, most NVIDIA-relevant signal.*

**★ Capstone 2.5 — "Real CUDA/CUTLASS kernel" (A100/H100; B200 rental for FP4).** Triton alone stops one rung short of kernel roles (benchmarks show hand-CUDA still wins correctness/perf). Write one real CUDA or CUTLASS/CuTe kernel — a tiled GEMM marched toward cuBLAS (coalescing → shared-memory tiling → vectorized loads → Tensor Cores/WGMMA), or an NVFP4 path on Blackwell. *Signal: closes the gap C2 leaves; this is literally the take-home.*

**Capstone 3 — "Speculative decoding that works" (A100/H100).** EAGLE-style or draft speculation in your engine or as a vLLM/SGLang plugin; report acceptance rate + end-to-end TPOT speedup on a reasoning workload; show the high-batch crossover and the long-context recovery. Higher-signal variant: reproduce/analyze **P-EAGLE**. *Signal: algorithm reasoning landed in a real system.*

**★ Capstone 3.5 — "Agentic KV-reuse / prefix-cache win" (cheap — T4/A100).** Currently absent from most curricula and arguably the highest-leverage *application* optimization. Build a cache-aware gateway over ≥4 replicas, replay a realistic agent/multi-turn trace, and drive cache hit rate from ~12% → >90% (report TTFT and $/task vs a no-cache baseline). *Signal: you understand the 2026 cost structure; cheap to run, very legible.*

**Capstone 4 — "Disaggregation + expert parallelism" (multi-GPU H100, stretch).** A two-pool prefill/decode split with KV handoff (NIXL), and/or a small MoE with expert parallelism + EPLB. Measure goodput under an SLO and the all-to-all / KV-transfer tax. **Scope tightly** — 2 GPUs, a tiny MoE, one clean before/after number; this is the capstone most likely to sprawl. *Signal: frontier serving — the staff-level flex.*

**★ Capstone 5 — "Ship it: a published beat-a-baseline benchmark + a merged OSS PR."** Two artifacts the research flagged as the cheapest separators of "studied" vs "ships": (a) a **reproducible benchmark that honestly beats a baseline** (e.g., "X% faster than vLLM on H100 at this trace") — proving you can *measure honestly*, the rarest skill; and (b) a **substantive merged PR** to **SGLang** (leaner queue → smarter first target), **vLLM**, **FlashInfer** (highest-leverage kernel repo; there's an MLSys-2026 contest on-ramp), or **TensorRT-LLM**. Avoid docs/typo PRs as a credibility play. *Signal: the only one that fully counts — you contributed value to the actual frontier stack.*

The workshop's "InferTutor" / voice-agent capstones are fine *applied* projects, but the ladder above is strictly higher-signal for frontier-inference roles.

**Compute strategy (under ~$150 for the whole ladder).** Stack free Colab (T4) + Kaggle (~30 hr/wk) ≈ 60 GPU-hr/week for C0–C3 at small scale. Rent in short bursts for profiling/FP4/multi-GPU: Vast.ai H100 spot ~$1/hr, RunPod H100 ~$2–2.7/hr, B200 from ~$2.1/hr, Modal H100 ~$3.95/hr zero-idle. **NVFP4 is Blackwell-exclusive** (B200/B300/RTX 5090) — H100/H200/A100 top out at FP8, so rent Blackwell only for the FP4 capstones.

---

## 5. Building capstones with AI agents in Colab (the workflow)

You asked to build these with AI agents — here's the discipline that makes that produce *understanding*, not just code you can't defend in an interview.

**Spec first.** Write a one-page spec before any code: exact data structures (the block-table layout), invariants (no two sequences share a writable block), and the metrics you'll report. Your leverage over an agent is the precision of the spec.

**Test- and measurement-in-the-loop.** Have the agent write the benchmark and correctness tests *first* (e.g., "paged engine output must match the unbatched reference token-for-token"), then the implementation. An agent with a failing test to turn green produces far better code.

**Profile, then ask "why."** When the kernel is slow, *you* read the Nsight trace and hand the agent the diagnosis ("bandwidth-bound at 40% MBU from uncoalesced KV reads"). The agent types; you must be the one who can read the roofline — that's the skill the interview tests.

**Force the writeup.** End each capstone by having the agent draft the report, then *you* rewrite the "why it works" section from first principles. If you can't write it, you haven't learned it — go back.

---

## 6. Sequencing & cadence

**Recommended order (research-refined):** C0 → C1 → C2 → **C2.5 (CUDA kernel)** → **C3.5 (agentic KV)** → C3 → **C5 (beat-a-baseline benchmark)** → C4 → **C5 (OSS PR, ongoing)**.

**Fast track (8–10 weeks, if already strong in PyTorch/transformers):** Modules 0–1 + C0–C1 (wk 1–2) → Modules 2–4 + C2/C2.5 (wk 3–5) → Modules 5–6 + C3 + C3.5 (wk 6–7) → Modules 7–10 + C4 (wk 8–9) → Modules 11–12 + start C5 (wk 10, ongoing). **Deep track (4–6 months):** same order, read every primary paper, derive before reading, and don't leave a module until its capstone reproduces the paper's qualitative result; the kernels/profiling track runs in parallel from week 3 (a kernel a week).

**Weekly rhythm:** Mon derive + read; Tue–Thu build + profile; Fri measure, write up, and (from week 6) scan OSS issue trackers. **Ship the writeup publicly each week — the public artifact trail *is* the portfolio.**

---

## 7. Talent & interview reality (grounded), and converting study into value

Studying gets you knowledge; **shipping** gets you the role.

**The bar.** Table stakes: strong **C++ and Python**, architecture + profiling fundamentals, and — for kernel roles — **CUDA is gating, not preferred** (Anthropic's Performance Engineer GPU role names CUDA, Triton, CUTLASS, FlashAttention, Nsight, NCCL/NVLink, INT8/FP8 explicitly). Hands-on vLLM/SGLang/TensorRT-LLM is near-mandatory at startups. The field bifurcates into a **CUDA-kernel track** (NVIDIA, OpenAI, GPU startups, vLLM/Red Hat) and a **compiler/accelerator track** (Groq/Cerebras/SambaNova MLIR; DeepMind TPU + JAX/Pallas/Mosaic); Anthropic straddles both.

**What interviews probe.** The signature round is **"make this kernel faster"** — Anthropic open-sourced its performance take-home (optimize a kernel using a hot-reloading Perfetto trace); Together gives 4–8h CUDA-optimization take-homes. The canonical exercise is **naive matmul → cuBLAS**. **Roofline/arithmetic-intensity reasoning is ubiquitous** ("prefill compute-bound, decode memory-bound; compute the batch crossover"). Systems-design centers on serving stacks — Anthropic's most-reported prompt is **"design an inference batching system for one GPU"** (extending to KV eviction, paged attention, spec decoding, P/D disaggregation). Groq and several startups **skip LeetCode** in favor of GitHub/shipped work.

**The conversion playbook.** OSS contributions to the serving stack are the **single strongest differentiator** — start by reproducing a benchmark and filing a precise issue, then a good-first-issue, then a real one (a kernel, scheduler improvement, quant path, missing benchmark). Target **SGLang** first (leaner review queue), then vLLM/FlashInfer. A **published, reproducible beat-a-baseline benchmark** is the credibility multiplier that makes the PR land. Know the canon cold (the Research Brief's reading list). The four things hiring managers actually test, and what this curriculum is built backward from: (1) can you do the roofline estimate live; (2) do you know *why* a technique works; (3) have you made a real system faster, with a before/after trace; (4) can you read unfamiliar kernel/scheduler code and reason about it.

---

## 8. Future directions (2026 → 2028) — build toward these

The research is unanimous on the trajectory; bias your later projects here.

1. **Memory/bandwidth, not FLOPs, is the scaling axis.** Sub-quadratic attention (DSA-style learned sparsity; linear/full hybrids) becomes the default; HBM and EP all-to-all are the gating resources.
2. **Sparsity falls toward ~1–2% active** → **wide EP + P/D/encoder disaggregation across hundreds of GPUs** is the default topology, and disaggregation goes *finer* (attention–FFN split).
3. **KV becomes a managed storage tier** (cluster-wide pools, SSD/CXL, zero-copy NIXL); the open problem is a **standard cross-engine KV interchange format**. The economic unit shifts from **$/token to $/task with cache-hit-rate as the dominant lever.**
4. **Hardware co-designs around the prefill/decode split** (Rubin CPX = dedicated prefill silicon); **FP4 becomes the default numeric format** for inference *and* training, with QAD the standard accuracy-recovery recipe; sub-3-bit stays gated by the ParetoQ learning transition.
5. **Inference-time compute becomes adaptive and learned** — budget-aware effort control and model routers make *dynamic per-request compute* a serving-layer scheduling problem.
6. **New model classes pull serving toward streaming + hard latency walls** — diffusion LLMs (KV-caching solved via Block Diffusion) for high-throughput code/agent paths; hybrid-SSM for long-context/high-batch; realtime multimodal and world models making sub-250 ms first-packet and fixed per-frame budgets standard SLOs.
7. **Kernels go Python-first and compiler-generated** (tile IRs, autotuning, agentic kernel generation) — but **roofline reasoning and memory-hierarchy mastery remain the durable human skills.** Invest there.

**Portfolio-grade open problems** (a strong project on any of these carries real 2026 signal): agentic/multi-turn KV reuse beating SGLang's radix cache on a real trace; the disaggregation re-aggregation crossover (when does it stop paying?); adaptive test-time-compute that curbs overthinking without accuracy loss; NVFP4 accuracy recovery on real workloads; three-tier HBM→DRAM→NVMe KV offloading with honest latency numbers; diffusion-LLM serving (the quality/parallelism tradeoff is barely charted).

---

*Companion file: `Frontier_Inference_2026_Research_Brief.md` holds the cited evidence base (frameworks, numbers, hardware specs, model table, talent signals, full reading list with URLs). Built to be torn apart and rebuilt as you measure — if a claim here disagrees with your profiler on your hardware, the profiler is right. Chase the discrepancy; that's where the real value is.*
