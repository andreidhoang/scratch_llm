# Performance & Inference Engineering — the track (2026)

> ⚠️ **ERRATA-G binding (2026-08-26).** This document is a **pre-audit planning generation** and is
> NOT current law. The sealed measure-first 60-day corpus governs (Desktop `plan/` docs + the memory
> progress-tracker; stamp precedence F > E > D > C > B > body). Where this document assigns
> sequencing, gates, or flagship artifacts, the sealed corpus supersedes it — the 2026-08-13 audit
> falsified the KDA-decode-kernel flagship (five shipped implementations), and the K3-from-scratch
> rebuild runs as the **post-gate season W9–W16**, with E001/E002 as its numerics CI. Full binding:
> the ERRATA-G blocks in `docs/k3/ROADMAP.md` and `docs/k3/MERGED_KERNELS_K3_ROADMAP.md`.

> **What this is.** The cross-cutting **performance/inference-systems spine** over the A1→A5→DELTA build:
> the one storyline an L8/L9 performance principal screens for, the 2026 frontier findings that justify
> it, and the EV-ranked build list that turns it into measured artifacts. It is the *systems/perf* twin
> of [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) (the assignment spine) and
> [`FRONTIER_PRACTICE_2026.md`](FRONTIER_PRACTICE_2026.md) (per-pillar defaults). The **on-ramp** for
> someone with no GPU background is [`GPU_FROM_ZERO.md`](GPU_FROM_ZERO.md); this doc is *why* that
> ladder is pointed where it is.
>
> **Provenance.** Synthesized 2026-06-29 from a six-stream deep-research pass (inference serving · low-
> precision/quant · MoE & parallelism · kernels & roofline · RL-systems · internal-doc analysis), each
> stream returning cited, confidence-tagged findings + an EV-ranked build list. Claims here are tagged
> `[FACT]` (primary source) / `[INFERENCE]` (derived). Verify a specific number against the cited
> primary before quoting it in an interview.
>
> **Superseded as the operating plan (2026-07-01).** This doc's build list spawned the
> `performance/` curriculum two days after it was written; the perf build now runs on
> **`performance/PERF_PLAN.md`** (phases, current node, rentals) + **`PERF_ENGINEERING_SPEC.md`**
> (predictions, DoD, kill criteria). What stays live here is the **reference layer**: the thesis
> (§0), the 2026 findings (§2), the subtract-list (§4), and the honesty constants (§5 — cited by
> `bench/RESULTS.md`). §1 and §3 are retired below with their resolutions.
>
> **Frontier layer refreshed (2026-07-14).** The findings in §2/§2.1/§6 now have a fresher,
> primary-source-verified successor: [`../performance/KERNEL_ROADMAP_2026.md`](../performance/KERNEL_ROADMAP_2026.md)
> — §1 (frontier delta incl. corrections to the 07-09 pass) + §2 (verified hiring bar) + §3 (Vizuara
> curriculum dissection). **Where they conflict, the roadmap wins.** This doc stays the thesis layer (§0/§5).
> **Re-verified 2026-07-30:** the roadmap's §1 07-30 block + `performance/references.md` §8 carry the
> post-07-14 deltas (CUTLASS 4.6 · B300/Rubin hardware · AFD · DSA · FA4 caveats · NCCL device API);
> the curriculum A-files were refreshed in place. The roadmap still wins on conflicts.

---

## 0. The thesis — "the decode memory-wall, owned end-to-end"

All five external research streams converged, independently, on the **same #1 highest-EV artifact**:

> **A roofline / "predict-the-number" discipline — predict the bound (compute vs memory vs overhead)
> and the number *before* the run, then show the measured profile that confirms or refutes it.**

That is not a coincidence — it is the hiring signal, and it is already this repo's constitution (FOP-3:
"DoD is a profile, not a green test"). The frontier-practice research and the repo's operating principles
point at the *identical* core skill.

**The one physical fact everything orbits** `[FACT]`:

> **Decode is memory-bandwidth-bound; prefill is compute-bound.** At batch-1, decode arithmetic
> intensity ≈ 1 FLOP/byte — **~200–300× below the H100 dense ridge point (~295 FLOP/byte)**. Decode
> tok/s is set by HBM bandwidth, not FLOPs (H100 3.35 → H200 4.8 → B200 8.0 TB/s ≈ the decode-speed ladder).

**Every** 2026 inference technique is one lever to raise decode AI back toward the bandwidth ceiling:

| Lever | Mechanism | Anchor (dated primary) |
|---|---|---|
| **Batching** | amortize weight loads across the batch | continuous batching 8–23× (Orca OSDI'22, vLLM); spec-decode *erodes* past B≈8 |
| **KV reduction** | fewer bytes/token re-read | MLA −93% KV (DeepSeek-V2 2405.04434) · GQA −8× (2305.13245) · PagedAttention waste 60–80%→<4% (2309.06180) |
| **KV / weight quant** | fewer bytes/element | FP8 KV ~free (vLLM 2026-04) · KIVI 2-bit, 2.35–3.47× (2402.02750) · INT4 weight-only ≈4× decode |
| **The kernel** | split-K decode, GQA grouping | Flash-Decoding 6–36× (PyTorch 2023-10) · FlashInfer −29–69% ITL (MLSys'25 2501.01005) |
| **Speculative decode** | verify K drafts per weight-load | EAGLE-3 τ≈6.6, ~2.5× prod (2503.01840) — *1-batch ≠ throughput* |
| **Linear-attn state** | constant-size state replaces growing KV | Mamba-2 (2405.21060), GatedDeltaNet (2412.06464) → decode flat in context = **DELTA** |
| **RL twin** | rollout *generation* is the RL bottleneck (~66–91%); train↔infer logprob drift breaks GRPO | "secretly off-policy" (Feng Yao 2025-08); Thinking Machines 2025-09 |

This single storyline threads through **A2** (kernels/systems), the **inference runtime**, **DELTA**
(GDN-2 decode kernel), *and* **A5** (RL rollout throughput + drift). "Start small, demonstrate large
value" = the small artifact (the roofline harness, [`design/PERF_roofline_harness_SPEC.md`](design/PERF_roofline_harness_SPEC.md))
is the seed of a coherent decode-performance portfolio.

---

## 1. Where the repo stands — RETIRED (2026-07-03)

Build state lives in [`STATUS.md`](STATUS.md); the current perf node lives in
`performance/PERF_PLAN.md`. The 2026-06-29 snapshot that sat here aged out within days and is kept
only in git history: its "live blocker" (the missing `rollout/` package) was resolved the same day
(green-CI restored, see STATUS), and its "wedge" (the KV-cache **never timed**) was measured
2026-07-01 as perf-curriculum **A1 R1** (51 → 173 tok/s eager→compiled, 15%→53% HBM —
`bench/RESULTS.md`).

---

## 2. The 2026 frontier findings to internalize (distilled, cited)

The load-bearing facts from each stream. Full reports are in the session synthesis; the durable numbers:

**Kernels & roofline** `[FACT]` — FA3: **740 TFLOP/s FP16 ≈ 75% of H100 peak** via warp-specialization
(TMA producer / WGMMA consumer) + GEMM-softmax pingpong + FP8 incoherent processing (arXiv 2407.08608).
Skipping WGMMA caps a GEMM at ~63% of peak; with it ≥94%. Decode is a *different kernel* (query-len 1 →
batch-1 <1% GPU util); Flash-Decoding split-K → 6.2×@256 to 35.7×@65k (PyTorch 2023-10). Grade decode on
**% HBM bandwidth, not % FLOPs.** *Honesty:* dense H100 ridge ≈ **295** (not the sparse ~590); the FA4
"1605 TFLOP/s" figure is directional until checked vs arXiv 2603.05451.

**Low-precision** `[FACT]` — DeepSeek-V3 FP8: tile-wise 1×128 act / block-wise 128×128 weight scaling,
E4M3 both directions, **FP32 promotion every 128 accumulations**; norms/attention/embeddings/head stay
BF16; loss-rel-error <0.25%. **NVFP4** (block-16 + fractional E4M3 inner + per-tensor FP32 outer) ≈ 88%
lower quant error than MXFP4 (block-32 + E8M0); NVFP4 pretraining validated at 12B/10T, <1% gap. **FP4 =
2× FP8.** KV asymmetry: Keys per-channel, Values per-token (KIVI 2402.02750). *Cautionary:* naive FP8
attention *accumulation* dropped 128k needle 91%→13%; a two-level fix restored 89% — the bug was
accumulation precision, not storage format.

**MoE & parallelism** `[FACT]` — DeepSeek-V3 = 671B/37B, **2.788M H800-hrs ($5.576M final-run rent, not
total R&D)**, first openly-documented frontier FP8 run. Deployed prefill EP32 / decode **EP320, 1 expert/GPU**
— wide EP exists to *fit* experts in aggregate HBM (memory-capacity-bound). Comms cost ordering: **TP
all-reduce > EP all-to-all > PP P2P > DP all-reduce** → nest TP innermost. 16 B/param Adam math →
671B ≈ 10.7 TB optimizer state. DualPipe ≈ halves the bubble; H800 NVLink:IB = 3.2×. Real MFU: Llama-3
405B 41% (→38% at 128k from CP comms); FSDP GPT-175B ≈ 55–60%.

**Inference serving** `[FACT]` — Disaggregated prefill/decode is standard (DistServe 7.4× / 12.6× SLO,
OSDI'24); KV transfer <0.1% of latency on RDMA/NVLink but the TTFT bottleneck on TCP. Chunked prefill +
decode-priority scheduling is the vLLM-V1 default; `max_num_batched_tokens` is the **TTFT↔ITL** knob.
RadixAttention prefix-cache 6.4× (always-on, <0.3% overhead). EAGLE-3 τ rises 3.98→6.62; **spec-decode
gains erode as batch grows — can be *slower* than vanilla at B≥8** (the senior-vs-principal insight).

**RL-systems** `[FACT]/[INFERENCE]` — rollout generation ~66–91% of step time. Train↔infer logprob
mismatch (batch-invariance, per Thinking Machines 2025-09) makes GRPO secretly off-policy → silent
reward collapse (e.g. 0.574→0.255 over 280 steps uncorrected). Fix = truncated importance sampling
(ρ=π_θ/π_rollout); **contested** — Group-Relative-REINFORCE camp (2509.24203) finds clipping matters
more; MiniMax CISPO clips the IS weight. Mandatory dashboard: entropy · KL(cur‖ref) **and** KL(cur‖old)
**separately** · IS-ratio histogram · reward+`frac_reward_zero_std` · **length** (verbosity hacking:
1171→2343 tokens). Dr.GRPO drops the length/std norm that causes "wrong answers get longer."

### 2.1 — 2026-07-09 refresh (perf/kernel/serving re-verification `wf_f3af3987-949`, primary-source)

The §2 findings above (2026-06-29) hold; four things moved since, and they **re-point the highest-paying
signal toward architecture↔kernel co-design.** Full per-rung verdict below in §6.

- **Kernel *authoring* moved to Python DSLs — but the PTX understanding did not become optional.** `[FACT]`
  **FlashAttention-4** (arXiv **2603.05451**, 5 Mar 2026, MLSys-26 oral; Tri Dao et al.) is **written
  entirely in CuTe-DSL (Python), zero CUDA C++** — it lowers to PTX → ptxas → SASS — and hits **1605
  TFLOP/s B200 BF16 = 71%** (Blackwell-native: TMEM accumulators, fully-async 5th-gen tensor cores, 2-CTA
  MMA), **20–30× faster compile** than C++ templates. **CUTLASS 4.0** (2025-06-03) shipped the **CuTe DSL**
  ("write high-perf GPU kernels in Python"); 4.3 added SM100 FMHA-bwd + MLA in the DSL. **Nuance the
  verifier flagged:** this is NOT "PTX is dead" — C++ CUTLASS is still developed and the DSL *lowers to
  PTX*, so TMEM/2-CTA/async-tensor-core understanding is the prerequisite for writing good CuTe DSL. →
  **Curriculum update:** the hand-PTX WGMMA/tcgen05 rungs are the *understanding layer* (keep — they're
  the SASS you must read); **ADD a CuTe-DSL authoring rung** (author the DELTA/FA kernel in CuTe DSL,
  diff against the hand-PTX version). Hand-sm90a-WGMMA-at-80%-cuBLAS is now closer to *table-stakes
  understanding*; the 2026 differentiator is authoring in the DSL **and** explaining the PTX it emits.
- **FA3 is one generation behind; FA4/Blackwell/CuTe-DSL is the headline.** The A4 "FA3-class Hopper
  kernel" stays as *understand the Hopper warp-spec generation*, but the **headline attention artifact is
  FA4-class on Blackwell in CuTe DSL** — and the standing sm120 card + the B200 rental are Blackwell.
- **Linear-attention decode is a live problem — but DELTA's scarcity is NARROWER than first framed
  (corrected).** `[FACT]` **GDN-2** (Gated DeltaNet-2, NVIDIA, arXiv **2605.22791**, 2026-05-22 —
  Hatamizadeh/Choi/Kautz; channel-wise erase gate `b_t` + write gate `w_t`, generalizing Gated DeltaNet
  and KDA) is real and current and beats Mamba-2/GDN/KDA/Mamba-3. At batch-1 GDN decode is **memory-bound**
  — the full fixed-size recurrent state round-trips HBM every token (arXiv 2603.05931). **⚠ Correction to
  the first draft of this note:** GDN-2 is **already in flash-linear-attention (FLA)** at BOTH the layer
  (`fla/layers/gdn2.py`) AND the kernel level — **including a *standard-precision* `fused_recurrent` DECODE
  kernel** (`fla/ops/gdn2/fused_recurrent.py`). Qwen's **FlashQLA** (TileLang, v0.1.2 2026-07-09) is
  chunked-**prefill only**. So DELTA is **NOT** greenfield at the architecture level *or* the plain-decode
  level — **its ONLY defensible scarcity is the FUSED LOW-PRECISION (FP8/NVFP4) recurrent-*state*
  materialization decode path**, which is absent across all surveyed libraries. Low precision directly cuts
  the HBM round-trip that bounds batch-1 decode → it's a real win, not a bandwidth dead-end (the stronger
  "HBM-bounded, fusion/precision can't help" framing was **refuted 0-3**). → **Re-scope DELTA to exactly
  the fp8/nvfp4 state-read/write decode kernel on the GDN-2 two-gate recurrence, benchmarked against FLA's
  `fused_recurrent` (standard-precision) as the baseline oracle.** **Re-scoped again per the K3 ROADMAP
  (2026-07-31, K10.2):** the flagship is now the **KDA decode kernel** (per-channel decay, Kimi K3's
  linear-attention recurrence), built on DELTA's GDN-2 base and benchmarked vs the FLA oracle — DELTA's
  correctness contract inherits from `core/kda.py`'s three-path equivalence (`docs/k3/ROADMAP.md`). It stays
  the program's **#1 co-designed artifact** married to F10 (see §6) — author it in a DSL (TileLang / CuTe DSL).
- **Serving: wide-EP disaggregated is now the *required* production shape.** `[FACT]` Large-scale MoE
  serving in 2026 = **DeepEP all-to-all + EPLB (expert-parallel load balancing) + Dual-Batch Overlap
  (DBO) for decode + PD-disaggregation + CUDA-graph FULL_AND_PIECEWISE.** PD-disagg is **architecturally
  required** (not just an optimization): DeepEP runs two dispatch modes — Normal (prefill) and
  Low-Latency/CUDA-Graph (decode) — that can't coexist in one engine. SGLang reproduces DeepSeek's PD +
  wide-EP on 96×H100 (~52k in / 22k out tok/s/node); vLLM ~2.2k tok/s/H200 for R1. The A1 primitives are
  the right foundation; the A6 serving day should name the wide-EP stack. **Freshness:** the canonical
  frontier-MoE serving target moved to **DeepSeek-V4** (1.6T Pro / 285B Flash, 1M ctx, Blackwell 8×B200/
  B300) by Apr 2026 — R1-671B's *physics* still teaches (MLA KV, EP-vs-TP, routing), but flag it as the
  baseline, V4-Flash/Kimi-K2/GLM-5 as the current flag. **Update (2026-07-27):** **Kimi K3** is now the
  repo's **rebuild-and-serve flagship** (K3 track chartered 2026-07-31, `docs/k3/ROADMAP.md`; K9 = 8×B300
  Modal rental, ~$120–170). EAGLE-3.1 + Kimi-K2.6 draft models (TorchSpec)
  make draft-model spec-decode a maintained 2026 workflow (→ F2b MTP drafter).
- **Precision: NVFP4 confirmed the right bet — but the *vendor GEMM* now exists.** `[FACT]` NVFP4 ≫ MXFP4
  (arXiv 2603.08747: Qwen2.5-0.5B WikiText PPL **21.63 vs 36.71**; 7B 6.47 vs 7.31). **Nuance (don't
  over-claim):** the gap **shrinks with scale** (~70% at 0.5B → ~13% at 7B — "≫" is strongest for small
  models), calibration (MR-GPTQ/OAS) closes most of MXFP4's gap, and MXFP4 has broader ecosystem adoption
  (OpenAI gpt-oss). NVFP4-W4A4 ~6–8k tok/s (1.7–2× A100 on Blackwell). **CUTLASS 4.0 already ships NVFP4/
  MXFP4/MXFP6/MXFP8 block-scaled GEMM via tcgen05** — so a hand-rolled tcgen05/NVFP4 GEMM is
  **table-stakes-vs-vendor**; reframe A3 §4.3 / A5 R3 as *understand the two-level scaling + beat MXFP4 for
  the documented reason + use the vendor primitive well*, not "reimplement the kernel." NVFP4 was not yet
  the vLLM inference default as of late 2025 (FP4 kernels still maturing).
- **Hiring signal (honest, `[INFERENCE]` on comp).** Frontier labs carry **dedicated** kernel/perf roles
  (Anthropic May-2026: "TPU Kernel Engineer", "GPU Performance Engineer") screening for exactly this
  cluster — serving, batching, quantization, **hardware↔software co-design.** Caveat (FOP-4): infra/perf
  is a *minority of headcount* (Anthropic ~26 infra vs 71 research/eng vs 87 sales of 346+), and this
  pass **could not verify specific comp bands** (job pages are JS-SPAs; salary didn't render) — so treat
  "$X total-comp" claims as unverified. The verified signal is *skill-demand + scarcity*, not a number.
  **The scarcest, highest-leverage artifact class = architecture↔kernel co-design** (own the model *and*
  the kernel), which is precisely DELTA+F10 — not a reproduced GEMM (vendor-served) or a serving engine
  (well-trodden).

---

## 3. The 80/20 build spine — RETIRED; each item's resolution (2026-07-03)

The EV-ranked list that sat here became the `performance/` curriculum; the original tables are in
git history (`1a0900c`). Where every item landed:

| Was (tier · item) | Landed as | Status |
|---|---|---|
| T0 · unblock `rollout/` import | rebuilt same day (STATUS "Green-CI restored 2026-06-29") | ✅ done |
| T1 #1 · roofline + predict-the-number harness | `src/scratch_llm/bench/` (`gpu_specs` · `roofline` · `harness` · `ledger`) | ✅ built 06-29, in daily use |
| T1 #2 · time the KV-cache decode | perf A1 **R1** (`bench/decode_roofline.py` + `decode_overhead_strip.py`) — on the standing GPU, no rental | ✅ measured 07-01 |
| T2 #3 · RL train↔infer drift + IS correction | A5 track — **gated behind the perf mandate** (2026-06-30) | ⬜ queued |
| T2 #4 · continuous-batching scheduler ("CPU-simulatable") | perf A1 **R3a/R3b** — real engine, measured 2.30× wall / 2.93× steps vs static-wave | ✅ measured 07-03 |
| T2 #5 · quant-error-vs-bits harness | perf **A5/Phase 1e** (INT8→INT4→NVFP4 numerics rungs) | ⬜ queued (planned) |
| T3 · decode-attention kernel (paged-KV) | perf A1 **R4.1** — fused Triton paged decode, 5.90 ms/step = ×3.52 vs wave | ✅ measured 07-03 |
| T3 · spec-decode batch-erosion curve | perf A1 **R4.3** (lossless spec-decode rung) | ⬜ queued (planned) |
| T3 · A2 distributed finish (ZeRO-1 → FSDP2 → comms) | DDP ✅ (665b6e6); ZeRO-1/FSDP queued behind the mandate | 🟡 partial |

---

## 4. Subtract-before-add (what NOT to build)

- ❌ Full multi-node RDMA P/D disaggregation deployment → a **CPU simulator** captures ~80% of the
  mastery signal for ~10% of the GPU spend.
- ❌ Reimplementing MLA / Megatron-TP from scratch → understand + cite; building them is an architecture
  project, not a perf one. Toy TP only (already tiered SKIP in `FRONTIER_PRACTICE_2026.md`).
- ❌ Chasing vendor best-case multipliers (H2O 29×, LMCache 15×, FA4 1605 TFLOP/s) → workload-specific;
  measure your own.
- ❌ FA2 *backward* Triton kernel → the `torch.compile` recompute path suffices (already SKIP).

## 5. Honesty constants (bake into the harness; the research flagged these)

- Dense **~295 FLOP/byte** H100 ridge, **not** the sparse ~590. State which peak (MFU denominators swing ~2×).
- **FP4 = 2× FP8**; B200 ≈ 9 PF dense FP4, GB200 ≈ 10 dense / 20 sparse (don't quote the marketing 20).
- DeepSeek-V3 **2.788M H800-hrs / $5.576M = final-run rent-equivalent, not total R&D.**
- Decode tok/s lower bound = `(weights + KV bytes) / HBM_BW`.

## 6. Per-rung perf verdict (2026-07-09 re-verification — §2.1 sources)

Verdict legend: ✅ CONFIRMED high-value · ⚖️ TABLE-STAKES (necessary, do-well-but-not-a-headline) ·
✏️ NEEDS-UPDATE (reframe to the 2026 generation) · ⬇️ COMMODITY / vendor-served.

| Rung | Verdict | 2026 reality → action |
|---|---|---|
| **A1 serving** (paged KV · continuous batch · PD-disagg · spec-decode · CUDA-graph decode · MLA toy) | ✅ | Correct 2026 foundation. Add the wide-EP names (DeepEP · EPLB · DBO) and that **PD-disagg is now *architecturally required*** for wide EP (two DeepEP dispatch modes can't co-exist). |
| **A2 CUDA-core ladder** (GEMV→GEMM 134% cuBLAS-proxy) | ⚖️ | Table-stakes kernel fluency. Necessary floor; not a differentiator. Keep, don't over-polish. |
| **A3 tensor cores** (WMMA→mma.sync→WGMMA→tcgen05, hand-PTX) | ✏️ | The hand-PTX is the *understanding layer* (keep — the SASS you must read). **ADD a CuTe-DSL authoring rung** — FA4 & CUTLASS 4.0 are Python-DSL-authored; 2026 kernels ship in the DSL. |
| **A4 FlashAttention** (FA2 done, FA3-Hopper deferred) | ✏️ | FA2 = table-stakes. FA3 = *understand the Hopper generation*. **Headline = FA4-class on Blackwell in CuTe DSL** (arXiv 2603.05451; TMEM/2-CTA; 1605 TF/s/71%) — the standing sm120 + B200 are Blackwell. |
| **A5 quantization** (NVFP4/MXFP4/FP8-KV/AWQ) | ✅→⚖️ | NVFP4 numerics = the right Blackwell bet (✅, NVFP4≫MXFP4 verified). But **the tcgen05/NVFP4 *GEMM* is vendor-served (CUTLASS 4.0)** → reframe A3§4.3/A5R3 as understand+beat-MXFP4+use-the-primitive, not reimplement (⚖️). |
| **A6 8×H200 R1-serving day** | ✅✏️ | Physics still teaches (MLA KV · EP-vs-TP · routing · PD-disagg). But **R1-671B is dated** — name **DeepSeek-V4-Flash (285B, Blackwell) / Kimi-K2 / GLM-5** as the current flag; R1 = the teaching baseline. **Kimi K3 (released 2026-07-27) is now the repo's rebuild-and-serve flagship** (`docs/k3/ROADMAP.md`; K9 = 8×B300 Modal rental, ~$120–170). Add the DeepEP/EPLB/DBO stack. |
| **A7 capstone** (WGMMA GEMM · FP8 FA3 attn · NVFP4 GEMM + OSS PR) | ⚖️✅ | The three kernels reproduce vendor-served frontier (⚖️) — the **OSS PR is the hiring artifact** (a merged PR to FlashInfer/vLLM/CUTLASS). Reframe two of the three toward CuTe-DSL/FA4-class + point the capstone at **DELTA as the payload** (DELTA → **KDA** per K3 ROADMAP K10.2). |
| **DELTA → KDA** (GDN-2-based **low-precision** fused *decode* kernel) | ✅ **#1 — re-scoped** | GDN-2 is a live NVIDIA arch (2605.22791) beating all linear-attn baselines. **⚠ FLA already ships a standard-precision GDN-2 `fused_recurrent` decode kernel** — so DELTA is **only** scarce as the **fp8/nvfp4 recurrent-*state* decode path** (absent everywhere). **Re-scoped per K3 ROADMAP K10.2 (2026-07-31): the flagship is now the KDA decode kernel (per-channel decay) on DELTA's GDN-2 base, benchmarked vs the FLA oracle.** So-scoped it's the #1 co-designed artifact with F10 — author in a DSL (TileLang/CuTe). |

**Blackwell / sm120 correctness note (verified):** on datacenter Blackwell (sm100) the Hopper **WGMMA
(`wgmma.mma_async`) is deprecated**, replaced by **`tcgen05.mma` (UMMA) + Tensor Memory (TMEM, 256 KB/SM)**
— so a hand-sm90a-WGMMA kernel targets a superseded ISA on the newest silicon, and **FA3 does not run on
B200** (Hopper-bound). Crucially, **the standing sm120 consumer card has NO tcgen05/TMEM** (ptxas rejects
`tcgen05.mma` for sm_120 → extended `mma.sync`), which *validates* the curriculum's decision to keep
tcgen05/WGMMA as compile-verify-only, gated to the H100/B200 rentals. The FA3-on-H100 rung's transferable
value is the **primitives** (TMA · warp-spec · pipelining · online-softmax), not the WGMMA *kernel*.

**One-line perf thesis (2026):** the durable signal is the *roofline/predict-the-number discipline* (§0)
plus **architecture↔kernel co-design** `[INFERENCE — comp evidence did not survive verification, RQ6
unanswered]` — the **KDA decode kernel** (per-channel decay; re-scoped from DELTA's GDN-2 low-precision
base, benchmarked vs the FLA `fused_recurrent` oracle — K3 ROADMAP K10.2, `docs/k3/ROADMAP.md`) married
to the F10 model-side Gated-DeltaNet is the one artifact here that no surveyed library
ships. Everything else is table-stakes fluency (A2/A5-GEMM, now vendor-served by CUTLASS 4.x) or a
reproduction whose hiring value is the landed **OSS PR**, not the kernel. **Honest caveat:** whether this
niche co-design out-signals a broadly-useful OSS PR (FP8 attention / NVFP4 GEMM / Wide-EP) for the
highest-paying roles is an *open question* — the research found **no verifiable 2026 comp data** (RQ6).

## 7. Pointers

- **The operating perf plan (current node, phases, rentals): `../performance/PERF_PLAN.md` +
  `../performance/PERF_ENGINEERING_SPEC.md`** — this doc is its reference layer.
- The from-zero on-ramp (rung 0→9): [`GPU_FROM_ZERO.md`](GPU_FROM_ZERO.md)
- Per-deliverable specs: `docs/design/PERF_*_SPEC.md` (+ the existing `L2_*_SPEC.md`)
- Per-pillar frontier defaults: [`FRONTIER_PRACTICE_2026.md`](FRONTIER_PRACTICE_2026.md)
- The assignment spine + DELTA: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7, `../../DELTA.md`
- **The perf critical path going forward: [`k3/ROADMAP.md`](k3/ROADMAP.md)** — K9 8×B300 Modal rental
  (~$120–170) and K10.2 KDA decode kernel (per-channel decay, on DELTA's GDN-2 base, vs the FLA oracle).
- Build status: [`STATUS.md`](STATUS.md) · Constitution + FOP + Mode boundary: [`../CLAUDE.md`](../CLAUDE.md)
