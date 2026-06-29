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

## 1. Where the repo stands (perf/inference subset)

From [`STATUS.md`](STATUS.md) + a direct disk check (2026-06-29):

**✅ Already frontier-aligned:** FA2 oracle + Triton fwd + the **honest 53%-of-SDPA roofline negative**
(a principal-grade claims-honesty artifact); KV-cache; `utils/monitors.py` (entropy + the three KLs +
IS/ESS); `utils/mixed_precision.py`; selective checkpointing; the `kl_train_infer` design spec.

**🔴 Live blocker:** `tests/test_rollout.py:5` imports `scratch_llm.rollout`, which is **absent on disk**
(STATUS claims "rollout seam ✅" — a doc/disk discrepancy). pytest collection fails → the green-CI hook
blocks every commit. **Tier-0: resolve before any commit.**

**🕳️ The wedge (highest-value first result):** the KV-cache is **correctness-tested but never *timed*.**
You own the asset that *is* the decode memory-wall and have never measured it against a roofline.

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

## 3. The 80/20 build spine (EV-ranked, Mode-aware)

Collapsed from the five ~7-item research lists (~35 items) for *this* repo, weighting EV/effort ×
scarcity × builds-on-existing × Mode-3-respecting. **Every deliverable is a measured profile/curve, not
a green test.** Mode tag: *(you)* = Mode-3 human body; *(AI)* = agent-buildable harness/scaffold.

### Tier 0 — Unblock *(today, ~30 min, AI)*
Resolve the `rollout` import → green CI. Prerequisite to every commit. **Do not blindly stub** — confirm
why `rollout/` is missing (uncommitted? checkout state?) before restoring or skipping.

### Tier 1 — The spine seed *(highest EV, CPU/cheap)*
| # | Build | Deliverable | Predict anchor | Mode |
|---|---|---|---|---|
| 1 | **Roofline + predict-the-number harness** (`bench/roofline.py`) — generalizes the one-off FA2 roofline. Spec: [`design/PERF_roofline_harness_SPEC.md`](design/PERF_roofline_harness_SPEC.md) | a roofline-PNG generator + per-token (weights+KV) breakdown, reused by every item below | "decode B=1 ≈ 300× below the H100 ridge" | you count FLOPs/bytes; AI scaffolds plot+table |
| 2 | **Time the existing KV-cache decode** (cheap vast.ai burst). Spec: [`design/PERF_decode_roofline_SPEC.md`](design/PERF_decode_roofline_SPEC.md) | measured-vs-predicted tok/s + nsys profile: tensor cores idle, HBM saturated → *proves* memory-bound | "70B BF16 ≈ 24 tok/s = 140GB ÷ 3.35TB/s" | you read the profile; AI scaffolds the bench |

### Tier 2 — Scarce differentiators *(map to the stated ship-order)*
| # | Build | Why scarce | Mode |
|---|---|---|---|
| 3 | **A5 RL train↔infer logprob-drift + IS correction** (builds on `monitors.py` + the `kl_train_infer` spec) | THE hot 2025–26 RL-systems result; A5 is the repo's highest-EV next ship | GRPO loss body *(you)*; drift bench + dashboard *(AI)* |
| 4 | **Continuous-batching scheduler** (Orca-style, CPU-simulatable on the A1 model) | the core 2026 serving primitive; reproduces 8–23× on your stack | *(AI)* |
| 5 | **Quant-error-vs-bits harness** (CPU) — sweep granularity × format | cheapest "thinks in numbers" quant artifact; feeds DELTA's NVFP4-on-state | *(AI)*; the FP8 GEMM body *(you)* |

### Tier 3 — Kernel reps + the A2 distributed finish *(your Mode-3 reps; AI scaffolds/profiles)*
- **Decode-attention kernel** (split-K / Flash-Decoding + GQA + paged-KV gather) — graded on **% HBM
  bandwidth**. The scarce decode-kernel skill, *directly the DELTA axis*. *(you write the Triton body)*
- **Spec-decode batch-erosion curve** — acceptance-τ + the crossover where spec-decode loses at B≥8.
- **A2 finish** (already the stated parallel ship): ZeRO-1 → FSDP2 → **comms-roofline harness** (reuses
  Tier-1) + the 100B/671B memory one-pager; **MoE all-to-all** dispatch/combine as the stretch.

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

- The from-zero ladder (rung 0→9): [`GPU_FROM_ZERO.md`](GPU_FROM_ZERO.md)
- Per-deliverable specs: `docs/design/PERF_*_SPEC.md` (+ the existing `L2_*_SPEC.md`)
- Per-pillar frontier defaults: [`FRONTIER_PRACTICE_2026.md`](FRONTIER_PRACTICE_2026.md)
- The assignment spine + DELTA: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7, `../../DELTA.md`
- Build status: [`STATUS.md`](STATUS.md) · Constitution + FOP + Mode boundary: [`../CLAUDE.md`](../CLAUDE.md)
