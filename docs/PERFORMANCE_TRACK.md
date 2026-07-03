# Performance & Inference Engineering — the track (2026)

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

## 6. Pointers

- **The operating perf plan (current node, phases, rentals): `../performance/PERF_PLAN.md` +
  `../performance/PERF_ENGINEERING_SPEC.md`** — this doc is its reference layer.
- The from-zero on-ramp (rung 0→9): [`GPU_FROM_ZERO.md`](GPU_FROM_ZERO.md)
- Per-deliverable specs: `docs/design/PERF_*_SPEC.md` (+ the existing `L2_*_SPEC.md`)
- Per-pillar frontier defaults: [`FRONTIER_PRACTICE_2026.md`](FRONTIER_PRACTICE_2026.md)
- The assignment spine + DELTA: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7, `../../DELTA.md`
- Build status: [`STATUS.md`](STATUS.md) · Constitution + FOP + Mode boundary: [`../CLAUDE.md`](../CLAUDE.md)
