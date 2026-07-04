# Performance Curriculum Implementation Plan

> The ordered execution plan for `performance/A1–A7`. The spec (PERF_ENGINEERING_SPEC.md) defines
> WHAT and WHY; this file defines WHEN and HOW — the phase sequencing, hardware batching strategy,
> time/cost estimates, and the "current node" pointer.
>
> **Update this file** whenever a phase or rung completes: mark it done, advance "Current node."
> This is the orientation file for every session that opens this curriculum.

---

## Current Node

```
Phase: 1a — **A1 COMPLETE (R0–R4.6 + design note, 2026-07-04)** → **A2 R0 (profiler+roofline harness)
  NEXT**. R4.2 mechanism-only (spike falsified, R4.2b deferred); R4.3 lossless spec decode; R4.4
  CUDA-graph −74% step / 77% of wall; R4.5 MLA identity to eps; R4.6 disagg ITL p99 3× + KV transfer
  characterized.  [delegate mode, ADR-0013]
Hardware: Standing GPU (sm_120)
Done (2026-07-04, R4.5+R4.6, A1 CLOSED): MLA toy (mla.py, weight-absorption identity 1.4e-15, KV
  3.56× < GQA-8, 5 tests) + PD-disagg demo (bench/disagg.py: decode-worker ITL p99 3× better 20.2 vs
  61.5 ms for a 0.18 ms one-time 16.8 MB KV transfer; goodput@SLO wins). A1 design note written
  (performance/notes/A1_design_note.md): the 16%→53%→77%-of-wall arc, every rung's before/after, the
  honest ~1.3× (latency) / ~1.5–2× (throughput) gap to vLLM = chunked-prefill piggyback + prefix cache.
Next action (A2 R0 — profiler + roofline harness): ncu automation is BLOCKED on this box
  (ERR_NVGPUCTRPERM, unprivileged) → re-base on nsys traces + CUDA-event timing + analytic
  bytes/FLOPs vs measured peaks (0.55 TB/s, 72 TF/s); reproduce the bench/RESULTS.md R0 baseline; a
  per-kernel "ncu debt" list feeds the H100 rental runbook. Then A2 R1 GEMV ladder (naive→coalesced→
  vectorized float4, target >80% of 0.55 TB/s) → R2 softmax → R3 RMSNorm → R4 TopK → R5/R6 GEMM.
Done (2026-07-04, R4.4): CUDA-graph decode over the fixed-address paged pool — serving/cudagraph.py
  CudaGraphDecoder (manual torch.cuda.CUDAGraph capture; the paged kernel's fixed (B,H) grid +
  device-side length loop is the only capturable substrate — dense view_len shape grows, cat-cache
  addresses grow, torch.compile reduce-overhead refuses the in-place advance). gpu test 4/4 token-exact.
  MEASURED (RESULTS.md 2026-07-04): B=1 −74.3% step-time (15.38→3.96 ms), 65→253 tok/s = 77% of the
  327 tok/s memory wall (eager 20% → compiled 53% → cudagraph 77% — R1 gap CLOSED); nsys 54–68
  cudaLaunchKernel/step → 1 cudaGraphLaunch/step. Pred −20–28% FALSIFIED good (our eager is far more
  launch-bound than vLLM's H100 path).
Done (2026-07-04, R4.3): lossless speculative decoding — serving/speculative.py (Drafter protocol +
  NGramDrafter prompt-lookup + ModelDrafter; speculative_generate with the pending-token invariant) +
  KVCache.truncate rollback primitive. 27 tests (float64 token-exact all drafters/K; wrong-drafter
  still exact ⇒ clean rollback; float32 ≥99%). MEASURED (RESULTS.md 2026-07-04): token-exact
  losslessness ✓ (P4.3.1); ×1.2–1.4 wall / 1.3–1.5 tok/target-forward via zero-cost n-gram drafting;
  P4.3.3/P4.3.5 FALSIFIED — the untrained model's degenerate-repetitive greedy makes n-gram acceptance
  54–72% even on RANDOM prompts (acceptance tracks model output entropy, not the prompt). Mechanism
  writeup (Medusa/EAGLE/MTP) in the note.
Next action (R4.4 — CUDA-graph decode): capture the decode step as a CUDA graph, replay per step to
  eliminate the ~955 per-token launches (R1: eager 51 → compiled 173 tok/s; graph should climb toward
  the ~275–327 memory ceiling). Substrate = the R4.1 paged Triton kernel: fixed (B,H) grid + fixed
  block-pool addresses + device-side length loop ⇒ CUDA-graph-capturable (the dense path's view_len
  shape grows; R1's cat-cache broke capture — both avoided). DoD = outputs identical (oracle R4.1),
  step-time reduction on nsys (CPU-gap before/after). Then R4.5 (MLA toy) → R4.6 (PD-disagg).
Done (2026-07-04, R4.2): chunked-prefill MECHANISM shipped & token-exact — model.py ChunkPrefillView
  (single-slot chunk at a running offset; RoPE absolute-pos ⇒ KV bit-identical to one-shot) + storage
  hooks _write_prefill_kv_at/slot_kv_view + serving/continuous.py prefill_chunk_size (dense+continuous).
  44 tests (single-chunk torch.equal KV; float64 token-exact all sizes; float32 divergences = argmax
  tie-flips). MEASURED (RESULTS.md 2026-07-04, bench/chunked_prefill.py): the sequential-interleave
  design REGRESSES all serving metrics (ITL p50 ×6.6–14.4, p99 did NOT fall, agg 547→145 tok/s) —
  pre-reg P4.2.2/3/5 FALSIFIED. Diagnosed: serialized 1-prefill/iteration admission (14 batched → 160–550
  UNBATCHED prefills) starves decode; un-piggybacked chunk cost lands in every gap. R4.2b (deferred, NOT
  an R4.3 prereq): fused mixed-query-len prefill+decode kernel (Sarathi piggyback; extend R4.1 paged
  kernel) + batched admission + shallow trace. Lesson: chunked prefill's win is a kernel/batching
  property, not scheduling-only.
Next action (R4.3 — speculative decoding, lossless): prompt-lookup (n-gram) drafter → target verifies
  K drafts in one forward → accept longest greedy-matching prefix → KV rollback (new KVCache.truncate).
  DoD = greedy output token-exact to sampling.generate (oracle); measure acceptance + speedup on
  repetitive vs random prompts. Spec: performance/notes/A1_R43_speculative_decoding.md. Then R4.4
  (CUDA-graph decode over the fixed-address paged pool) → R4.5 (MLA toy) → R4.6 (PD-disagg).
Done (2026-07-03, R4.1): PagedKVCache (16-tok block pool, block table, on-demand alloc, trash-block
  write safety, refcounted prefix sharing/CoW, committed-blocks admission guard) + fused Triton
  paged decode kernel, behind the new SlotKVCache base (dense + paged, one slot contract). Measured
  (RESULTS.md P4.1.1–P4.1.4): frag 5.0% (pred 4–8%), capacity ×9.3 peak-provisioned; gather-only
  control +6.3%/step (paged storage alone LOSES — as registered); **kernel 5.90 ms/step = the whole
  R3b mixed-age tax reclaimed and better: ×3.52 vs wave-dense, 3,528 tok/s (+55% over dense-cont)**.
  Tax decomposition: ~3.2 ms padded-SDPA compute, ~0.5 ms bytes. 13 new tests; 157 CPU green.
Next action (R4.2 — chunked prefill): interleave prefill chunks between decode steps; DoD =
  TTFT/ITL curve vs chunk size, no output divergence (oracle = R3b greedy). Motivation measured:
  ITL p99 ~29 ms admission spikes (prefill-in-the-decode-stream) in every R4.1 arm vs ~7 ms p50.
  Then R4.3 (spec decode) → R4.4 (CUDA-graph decode over the fixed-address block pool).
Done (2026-07-03): R3b continuous scheduler SHIPPED & MEASURED. model.py: BatchedKVCache static slot
  buffer (write-then-mask, per-row RoPE positions + key masks, NaN-free by construction) + PrefillView
  (admission prefill duck-types the uniform causal path); serving/continuous.py: one engine, two policies
  (continuous vs static-wave — measured Δ is pure scheduling). Ownership contract: device state is
  graph-owned, python mirror is scheduler-owned (in-graph list reads bake Dynamo ordering guards →
  recompile storm; measured, fixed, unique_graphs=2). Oracle: 21 CPU tests (ragged == single-stream,
  poisoned-slot invariance, utilization == hand analytics). Measured (RESULTS.md 2026-07-03):
  R3.4 PASS 2.30× wall / 2.93× by steps on heavy-tail ×8 waves (util 24.9%→72.8%); R3.4s 1.11×/1.46×;
  R3.6 PASS TTFT p95 4.9× (shallow queue); R3.5 ITL 5.4→9.6ms (the honest cost). Wall<steps gap = the
  dense-buffer PADDING TRAFFIC, measured at ~1.27× (9.6 vs 6.3 ms/step) — R4.1's motivation, quantified.
Next action (R4.1 — PagedAttention, Triton): 16-token blocks + block table over the R3b slot buffer;
  contiguous-match oracle (paged == contiguous bit-for-bit on non-contiguous allocations); DoD <4% waste
  + reclaim a measured share of the 1.27× padding tax on the R3b heavy-tail bench. R4.4 (cudagraph decode,
  closes R1's 53%→wall gap) shares this buffer — order per plan is 4.1 → 4.2 → 4.3 → 4.4.
LINCHPIN realized: the static buffer + serve() harness from R3b is the substrate every R4.x rung
  measures against (bench/continuous.py is the standing workload).
```

> **Reset 2026-07-01.** Built from scratch: the exploratory Jun-29 perf kernels were removed (tag
> `pre-perf-kernel-reset`). Foundations reused: the `scratch_llm.bench` measurement apparatus,
> `sampling.generate` (token-exact decode), CS336 A2 FA2.
>
> **Rung 0 shipped 2026-07-01** (`34f739e` serving, `f3907fb` bench): `serving/metrics.py`
> (TTFT/ITL/throughput/goodput at p50/p95/p99) + `serving/baseline.py` (instrumented decode —
> token-exact vs `sampling.generate`, under `no_grad`, fixed-seed reproducible). `GpuSpec` for the
> standing card added so `roofline()` scores decode on sm_120. Ship-reviewer ACCEPT; 10 serving
> tests green; carried-forward Rung-1 hardening: discard warmup, sync-once-for-throughput.

---

## Phase Map

```
Phase 1a  A1 Rung 0–3 + frontier 4.1–4.6          sm_120 standing   ~1.5–2 weeks
Phase 1b  A2 Rungs 0–6 (CUDA-core ladder)           sm_120 standing   ~2–2.5 weeks
Phase 1c  A3 Rungs 0–2 (WMMA + mma.sync)            sm_120 standing   ~0.5 week
Phase 1d  A4 Rungs 0–3 (naive → FA2 WMMA)           sm_120 standing   ~1 week
Phase 1e  A5 Rungs R0–R4 (numerics + NVFP4 math)    sm_120 standing   ~1 week
───────────────────────────────────────────────────────────────────────────────────
Phase 2   H100 batch  (A2§4, A3 rungs3-4+§4.1, A4 rung4/FA3, A5 FP8-WGMMA)
          + single-GPU frontier serving block (70B-FP8 · gpt-oss-120b)
          Rent 1× H100 SXM (~$2–2.5/hr)                              ~12–15 H100-hrs
───────────────────────────────────────────────────────────────────────────────────
Phase 3   B200 batch  (A3 §4.2–4.3 tcgen05/NVFP4-MMA, A5 §7 NVFP4 B200)
          Rent 1× B200 (~$4–7/hr)                                     ~4–6 B200-hrs
───────────────────────────────────────────────────────────────────────────────────
Phase 4   THE SERVING DAY — a frontier MoE on one NVLink domain (ADR-0012)
          A6 R0–R1 + DeepSeek-R1 FP8: TP×EP · MLA KV · PD-disagg
          Rent 8× H200 SXM node, ON-DEMAND (~$20–32/node-hr)          ~6–10 node-hrs
───────────────────────────────────────────────────────────────────────────────────
Phase 5   OPTIONAL (training-leaning): A6 multi-node EP + 1F1B pipeline
          + A7 capstone kernel polish (H100/B200 hours, not cluster)  ~0–10 hrs
───────────────────────────────────────────────────────────────────────────────────
Total est. (inference track, ADR-0012): ~$25–45 (P2) + ~$25–45 (P3) + ~$150–320 (P4)
          + ~$30–50 (A7 polish) [+ ~$50–100 optional P5] ≈ **$230–460**  (old ~$700 retired)
```

---

## Phase 1a — A1: Transformer Inference as a System

**Hardware:** sm_120 standing GPU  
**Sequential order within A1:** Rung 0 → 1 → 2 → 3 → 4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6  
**Prerequisites:** A1 spec registered in PERF_ENGINEERING_SPEC.md §4/A1 ✅

| Rung | Status | DoD gate | Hardware |
|---|---|---|---|
| 0: metrics harness + PyTorch eager baseline | ✅ | metrics reproducible, fixed-seed (34f739e) | sm_120 |
| 1: KV-cache decoder (contiguous) | 🔵 | token-exact ✅; decode measured + overhead-stripped (15%→53% HBM via fusion); wall-close deferred to R4.4 | sm_120 |
| 2: GQA/MQA | ✅ | KV/token 128/32/4 KB (32:8:1); compiled decode MQA 1.92× MHA @16K; eager control ~1× (da64bfb→R2) | sm_120 |
| 3: continuous batching (Orca-style) | ✅ | R3a agg 66× B=1, flip @B≈128; R3b continuous 2.30× wall / 2.93× steps vs static-wave (R3.4 PASS), TTFT p95 4.9× (R3.6), oracle green — padding tax 1.27× measured → R4.1 | sm_120 |
| 4.1: PagedAttention (16-tok blocks, Triton) | ✅ | frag 5.0%, capacity ×9.3; contiguous-match bit-exact (scattered/boundary/poison/CoW); **kernel 5.90 ms/step, ×3.52 vs wave, +55% vs dense-cont** — whole padding tax reclaimed | sm_120 |
| 4.2: chunked prefill | 🟡 mechanism ✓ (token-exact, 44 tests) + measured; **spike-reduction FALSIFIED** (sequential-interleave regresses: ITL p50 ×6.6–14.4, agg 547→145 tok/s) → **R4.2b piggyback deferred** (fused prefill+decode kernel) | sm_120 |
| 4.3: speculative decoding (lossless) | ✅ | lossless token-exact (27 tests); ×1.2–1.4 wall / 1.3–1.5 tok/forward (n-gram); acceptance tracks model output-entropy not prompt | sm_120 |
| 4.4: CUDA graphs decode | ✅ | token-exact (gpu 4/4); **B=1 −74.3% step, 253 tok/s = 77% of wall** (eager 20%→compiled 53%→graph 77%, R1 gap closed); nsys 54–68→1 launch/step; manual capture over paged pool | sm_120 |
| 4.5: MLA latent cache (toy scale) | ✅ | weight-absorption identity to machine eps (1.4e-15, 5 tests); MLA KV 3.56× < GQA-8 / 1.8% of MHA per layer | sm_120 |
| 4.6: prefill/decode disaggregation | ✅ | demonstrated: decode-worker ITL p99 3× better (20.2 vs 61.5 ms) for a 0.18 ms one-time KV transfer; goodput@SLO wins | sm_120 |
| Design note | ✅ | `performance/notes/A1_design_note.md` — the roofline spine + every rung's before/after + honest vLLM gap | — |

**How to start rung 0:**
```
1. /standup (orient: confirm current node = A1 Rung 0)
2. Invoke bench-writer agent: "Write the failing correctness + metrics harness tests for A1 Rung 0
   in a new src/scratch_llm/serving/ module (reuse the kept scratch_llm.bench apparatus —
   decode_step_flops_bytes, measure_hbm_bandwidth, roofline). Target: (a) greedy generation output
   reproducible fixed-seed, (b) TTFT/ITL/throughput/goodput at p50/p95/p99 from a batch of requests."
3. Implement the PyTorch eager baseline + metrics harness (recruit sampling.generate — it already
   exists and is token-exact; this rung instruments it, it does not rebuild the decoder).
4. Confirm pytest -m gpu passes.
5. Log baseline numbers to bench/RESULTS.md.
```

---

## Phase 1b — A2: Kernel Optimization CUDA-Core Ladder

**Hardware:** sm_120 standing GPU  
**Order:** Rung 0 → 1 → 2 → 3 → 4 → 5 → 6 (→ Phase 2 for §4.1–4.5 on H100)  
**Prerequisites:** A1 complete (reuse metrics harness; GEMV intuition from A1 decode)

| Rung | Status | DoD gate | Hardware |
|---|---|---|---|
| 0: profiler + roofline harness (ncu automation, CSV, matplotlib) | ✅ | `bench/kernel_roofline.py`: peaks reproduce baseline (0.551 TB/s · 72.1 TF/s · ridge 131); copy 100.1% mem / gemm 100.0% cmp; ncu BLOCKED → re-based on %-of-peak + nsys, 5 metrics as H100 debt | sm_120 |
| 1: GEMV ladder (naive→coalesced→two-stage→vectorized float4) | ✅ | **blockrow 528.8 GB/s = 95.9% HBM** (>80% MET), 107% torch.mv; float4/swizzle Triton-managed | sm_120 |
| 2: softmax (online→warp-shuffle→fused) | ✅ | **fused ~100% HBM**; online 1.33× fewer bytes / 1.17× faster; adversarial +1e4/−inf NaN-free | sm_120 |
| 3: RMSNorm + LayerNorm (Welford) | ✅ | **both ~100% HBM @N≥4096**; RMS vs LN within ±1% (small-N edge = noise, honest) | sm_120 |
| 4: Top-K ladder (naive→min-heap→parallel); fused softmax+TopK | ✅ | **46.9% peak** (honest poor-GPU-fit); **fused softmax+topk 3.4× faster / 3.0× less traffic** | sm_120 |
| 5: GEMM part 1 (naive→coalesced→SMEM tile); run at ≥4096³ | ✅ | naive 0.2% → tiled 128.5% of cuBLAS-proxy (siboehm shape) | sm_120 |
| 6: GEMM part 2 (1D→2D blocktiling→vectorized); Memory%→Compute% flip | ✅ | autotuned **134.3% of cuBLAS-proxy** (101.9 TF/s); compute-bound AI 1365 | sm_120 |
| §4.1–4.5: WGMMA+TMA+warp-spec+FP8 | ⬜ | → Phase 2 (H100 required) | H100 |
| Design note | ⬜ | 2–3 pages | — |

---

## Phase 1c — A3: Tensor Cores (WMMA + mma.sync on sm_120)

**Hardware:** sm_120 standing GPU (rungs 0–2 only; rungs 3–4 + §4 → Phase 2/3)  
**Prerequisites:** A2 Rung 6 complete (CUDA-core ceiling established; A3 starts from that floor)

| Rung | Status | DoD gate | Hardware |
|---|---|---|---|
| 0: naive SMEM GEMM baseline (re-anchor from A2) | ⬜ | ~43 TF/s class result; book's regime | sm_120 |
| 1: WMMA GEMM + cp.async double-buffer | ⬜ | ≥40% of sm_120 peak; FP32 accum justified | sm_120 |
| 2: mma.sync + ldmatrix + XOR-swizzled SMEM | ⬜ | bank-conflict count ≈0; ≥60% of peak | sm_120 |
| PTX artifact (hand-decode wgmma, hand-encode descriptor) | ⬜ | every qualifier annotated | paper / any GPU |
| 3: WGMMA+TMA 3-stage (4 sub-rungs) | ⬜ | → Phase 2 (H100 required) | H100 |
| 4: WGMMA FP8 + wait_group overlap | ⬜ | → Phase 2 | H100 |
| §4.1: warp-specialized persistent | ⬜ | → Phase 2 | H100 |
| §4.2–4.3: tcgen05/NVFP4 | ⬜ | → Phase 3 (B200 required) | B200 |
| Design note | ⬜ | 2–3 pages | — |

**PTX artifact:** can be done now, without GPU. Read NVIDIA PTX ISA §wgmma.mma_async. Decode every
qualifier of `wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16`. Hand-encode a 64-bit SMEM
descriptor (address/leading-dim/stride/swizzle/validity). Pair with the `nvcc -ptx` output you'll
compare against in Phase 2. File: `performance/artifacts/wgmma_descriptor_manual.md`.

---

## Phase 1d — A4: Flash Attention (FA1/FA2 on sm_120)

**Hardware:** sm_120 standing GPU (rungs 0–3; rung 4 → Phase 2 H100)  
**Prerequisites:** A3 Rung 2 (WMMA for GEMM tiles); A2 Rung 2 (online softmax internalized)

| Rung | Status | DoD gate | Hardware |
|---|---|---|---|
| 0: naive 3-kernel attention oracle | ⬜ | matches PyTorch SDPA to <1e-3; OOM at ~16K seqlen | sm_120 |
| 1: single-row online softmax (Numba/1-thread CUDA) | ⬜ | <1e-6 vs 3-pass; +50 outlier test | sm_120 |
| 2: fused tiled FA1 (WMMA QKᵀ, online-softmax SRAM, deferred ÷d, causal) | ⬜ | <1e-3 vs R0; constant SMEM; no OOM at N=64K | sm_120 |
| 3: FA2 work-partitioning (fewer non-matmul FLOPs, seq-dim parallel, split-Q) | ⬜ | ~50–73% of sm_120 peak; occupancy rose | sm_120 |
| §4.3: one variant (GQA or paged or MLA) | ⬜ | oracle-verified; KV-memory consequence measured | sm_120 |
| backward pass | ⬜ | gradient-check <1e-2 FP16 | sm_120 |
| 4: FA3-class Hopper kernel | ⬜ | → Phase 2 (H100) | H100 |
| Design note | ⬜ | 3–4 pages | — |

---

## Phase 1e — A5: Quantization (Numerics on sm_120)

**Hardware:** sm_120 standing GPU (all numerics; NVFP4 native MMA → Phase 3 B200)  
**Prerequisites:** A4 Rung 0 (understand dynamic range issues); A1 Rung 4.4 (FP8 KV context)

| Rung | Status | DoD gate | Hardware |
|---|---|---|---|
| R0: INT8 sym per-tensor | ✅ | SQNR 40.5 dB Gaussian; book example reproduced; 6.75 dB/bit law | sm_120 / CPU |
| R1: INT8 asym + per-channel/token | ✅ | asym 43.5 > sym 37.4 (+6.1); per-ch 42.3 > per-tensor 33.3 (+9.1); W8A8 rel 1e-2 | sm_120 / CPU |
| R2: group-wise INT4 (g=128), packed | ✅ | pack/unpack bit-exact (500-case fuzz); SQNR 18.64 vs 18.60 floor; group > per-tensor | sm_120 / CPU |
| R3: NVFP4 two-level + MXFP4 comparison + block-scaled GEMM | ✅ | NVFP4 MSE 1.48× < MXFP4 (stronger baseline confirmed); GEMM 0.28% vs bf16; histogram PNG | sm_120 / CPU |
| R4: FP8 E4M3 KV cache wired into A1 decoder | ✅ | E2E 24.45 dB vs BF16-KV; per-ch-K 2.49×; INT4-KV degrades; bytes 0.552× | sm_120 |
| §4.3: PTQ method (AWQ/GPTQ/QuaRot) on real linear layer | ✅ | AWQ INT4 1.71× MSE recovery vs naive (held-out 1.69×); salient-col err ↓2.87× | sm_120 / CPU |
| §7 (stretch): NVFP4 native MMA on B200 | ⬜ | → Phase 3 (B200) | B200 |
| Design note | ⬜ | 2–3 pages | — |

---

## Phase 2 — H100 Batch (A2§4 + A3 Rung3-4+§4.1 + A4 Rung4 + A5 FP8-WGMMA)

**Gate to enter Phase 2:** All Phase 1a–1e DoD boxes checked; bench/RESULTS.md has a logged
measurement for every sm_120-runnable rung; the PTX artifact (A3) is written.

**Pre-rental checklist:**
```
□ All sm_120 rungs oracle-correct (pytest -m gpu passes for A1–A5 sm_120 tests)
□ bench/RESULTS.md has an entry for every rung in Phase 1 (predicted + measured)
□ Compiled and cross-checked: A2 §4.1 (cp.async/TMA) code is syntactically correct
  but NOT compiled for sm_120 (will fail); ready to swap -arch=sm_90a on H100
□ PTX artifact written (A3 wgmma descriptor manual)
□ Triton FA3 skeleton written (A4 Rung 4 starter test scaffolded by bench-writer)
□ vastai or RunPod command ready to spin up 1× H100 SXM instance
```

**H100 session plan (batch everything):**

| Order | What | Est. hrs |
|---|---|---|
| 1st | Boot instance; install toolchain; verify `nvcc -arch=sm_90a` + `ncu` + CUTLASS | 0.5 hr |
| 2nd | A2 §4.1: cp.async → multistage pipeline (TMA variant on H100) | 1 hr |
| 3rd | A2 §4.2: WGMMA tensor-core mainloop (the ~10× jump; target ~318 TF/s) | 1.5 hr |
| 4th | A2 §4.3: warp-specialized + persistent + Stream-K | 1.5 hr |
| 5th | A2 §4.4: epilogue fusion | 0.5 hr |
| 6th | A2 §4.5: FP8 fine-grained GEMM (two-level promotion) | 1 hr |
| 7th | A3 Rung 3.1: basic WGMMA | 0.5 hr |
| 8th | A3 Rung 3.2–3.4: larger tiles → TMA → 3-stage | 1 hr |
| 9th | A3 Rung 4: WGMMA FP8 + wait_group overlap | 0.5 hr |
| 10th | A3 §4.1: warp-specialized persistent (target ≥85% of H100 dense) | 1 hr |
| 11th | A4 Rung 4: FA3-class kernel (start from CUTLASS/CuTe; warp-spec + TMA + ping-pong + FP8) | 2 hr |
| 12th | A5 FP8-WGMMA (DeepGEMM two-level accumulation on Hopper) | 1 hr |
| 13th | **Serving block S1** — Llama-3.3-70B FP8 single-GPU: B=1 decode vs roofline; batch sweep; KV-headroom audit | 1.5 hr |
| 14th | **Serving block S2** — gpt-oss-120b (MXFP4-native MoE, single-GPU by design): B=1 + aggregate; MoE-vs-dense decode contrast | 1 hr |
| **Total** | | **~14–15 H100-hrs ≈ ~$30–45** |

**Critical: log EVERY measurement to bench/RESULTS.md BEFORE the instance is released.**

**Serving-block pre-registrations (S1–S3 — copy into bench/RESULTS.md before the session):**

| # | experiment | predicted | derivation / bound |
|---|---|---|---|
| S1 | Llama-3.3-70B FP8, B=1 decode tok/s | **35–45** (roofline ceiling **47.9**) | `3.35 TB/s ÷ 70 GB weights`; engines reach 75–90% of the wall (our sm_120 compiled path measured 53–71% — Hopper fused paths do better) → memory-bound |
| S2 | KV headroom, 70B-FP8 on 80 GB | **~3–5 GB free ⇒ only ~10–15 K cached tokens** (GQA-8 BF16 = 327.7 KB/tok) ⇒ tiny max batch | ~74 GB usable − 70 GB weights. THE single-GPU capacity lesson: a dense 70B on H100 barely batches — this is why FP8-KV, H200 capacity, and MLA exist (Phase-4 P5 completes the argument) |
| S3 | gpt-oss-120b (5.1B active, MXFP4), B=1 decode | **100–250 tok/s — latency-floor-bound**, NOT the naive ~1,100 | active weights ≈ 2.7–3 GB/tok ÷ 3.35 TB/s ≈ 0.9 ms; per-layer launch/attention floors dominate — the small-active-MoE preview of Phase-4 P2's node-scale version |

---

## Phase 3 — B200 Batch (A3 §4.2–4.3 + A5 §7)

**Gate to enter Phase 3:** Phase 2 complete; A3 §4.1 warp-specialized WGMMA hitting ≥85% of H100
dense. The jump to B200 should only happen once you've exhausted H100 headroom.

**B200 pre-rental checklist:**
```
□ Read Colfax tutorials Part 1–4 (WGMMA+UMMA+TMEM+NVFP4) — all four
□ Read NVIDIA PTX ISA tcgen05 section: alloc/ld/st/cp/commit/fence
□ Read gau-nernst tcgen05 kernel code (reference implementation)
□ Prepared: A3 §4.2 code with tcgen05 PTX — syntactically complete, un-compiled
□ Prepared: NVFP4 quantizer (A5 R3) verified on sm_120 in software
```

| Order | What | Est. hrs |
|---|---|---|
| 1st | Boot instance; verify sm_100a compile; trivial tcgen05 PTX compiles | 0.5 hr |
| 2nd | A3 §4.2 (1-SM): tcgen05/TMEM BF16 GEMM; target ~1200 TF/s (60%) early | 1.5 hr |
| 3rd | A3 §4.2 (warp-spec): climb to ~1300–1476 TF/s (~65%); record 2-SM ~8% result | 1 hr |
| 4th | A3 §4.3: NVFP4 block-scaled MMA; target placement on 9,000 TF/s tier; accuracy vs BF16 | 1.5 hr |
| 5th | A5 §7: NVFP4 native MMA element-rate vs FP8; record hardware ratio | 0.5 hr |
| **Total** | | **~5 B200-hrs ≈ ~$25** |

---

## Phase 4 — THE SERVING DAY: a real frontier MoE on one NVLink domain (8× H200)

> **Re-aimed 2026-07-03 ([ADR-0012](../docs/adr/ADR-0012-inference-rental-tiers.md)).** Was: 8×H100
> training-parallelism ladders. Now: the single most instructive rental for frontier inference —
> bring up **DeepSeek-R1 671B FP8** on one 8×H200 NVLink node and measure every serving-native
> distributed behavior against pre-registered numbers. A6 Rung 0 (topology/busbw) and Rung 1
> (TP micro) execute inside this day; the training-leaning depth (1F1B pipeline, multi-node EP,
> elastic ckpt) moves to optional Phase 5.

**Gate:** Phases 1–3 complete; every sm_120/H100/B200 rung ledgered.
**Hardware:** 8× H200 SXM (1,128 GB HBM3e, 4.8 TB/s/GPU, NVLink4 900 GB/s/GPU) — **on-demand only**
(spec §2 rule 5). **Fallback** if H200 unavailable/mispriced: 8× H100 serving **Qwen3-235B-A22B
FP8** (235 GB fits with 365 GB headroom) — loses the R1 flag, keeps every primitive.

### Why R1 is the teacher (first principles)

1. **Fit forces multi-GPU:** 671 GB FP8 weights exceed any single GPU; even the TP-8 shard is
   83.9 GB/GPU (> H100's 80) — the model *is* the reason the node exists (ADR-0012 fit table).
2. **MoE routing is THE 2026 serving problem:** 256 routed experts, top-8/token, 37 B active of
   671 B — EP-vs-TP is a live measured tradeoff here, not a slide.
3. **MLA is THE 2026 KV story:** `(512+64) × 61 L × 2 B = 70.3 KB/token` — 4.7× less than a 70B
   GQA-8 (327.7 KB). Long-context serving economics from architecture, completing Phase 2's S2.
4. **Honesty anchor:** SGLang/vLLM publish single-node R1-FP8 configs + numbers — our measurements
   have a public reference to be checked against.

### Pre-rental checklist (extends spec §2 rules 1–6)

```
□ Engine image pinned; the EXACT launch command smoke-tested beforehand on Tier 0/1 (small model)
□ nccl-tests built or install rehearsed; `nvidia-smi topo -m` parse rehearsed
□ Listing: ≥5 Gbps down, ≥1.5 TB disk, NVLink SXM (not PCIe) — verify inside the first 10 min or kill
□ P1–P7 pre-registration rows copied into bench/RESULTS.md (from the table below)
□ Two-load trace (bench/continuous.py make_trace: heavy-tail, saturated + shallow) adapted to the
  engine's benchmark client — R3b finding: one trace cannot measure both throughput and TTFT
□ On-demand instance; hard budget alarm at 12 node-hrs
```

### The day (≈ 8–8.5 node-hrs)

| Hr | Block | What / gate |
|---|---|---|
| 0–0.75 | Pre-flight | topo verify (all pairs NV*); **start the 671 GB weight pull immediately** (ETA gate ≤ 90 min); run the nccl-tests busbw sweep 1 KB→1 GB *while downloading* → P1 (A6 Rung 0) |
| 0.75–1.75 | Bring-up | R1-FP8 TP-8 up; correctness smoke (5 greedy prompts + logprob sanity); B=1 decode tok/s → P2 |
| 1.75–3.25 | Throughput | concurrency sweep 1→256: aggregate tok/s + ITL/TTFT percentiles; locate the aggregate knee; experts-hit-vs-B curve → P3; cross-check published engine numbers |
| 3.25–4.75 | EP vs TP | expert-parallel vs TP-8 at B∈{32,64,128}: throughput + per-expert load histogram → P4; A6 Rung-1 TP micro (20 min: the 2-AllReduce/layer count check) |
| 4.75–5.5 | MLA KV audit | engine-reported KV/token vs the 70.3 KB analytic; max concurrent×ctx vs the ~6.4 M-token pool → P5 |
| 5.5–7.0 | PD-disagg | prefill/decode split vs co-located on the two-load trace: TTFT p95 + goodput under SLO → P6 |
| 7.0–7.75 | Stretch | MTP spec-decode acceptance + speedup → P7; re-run anything noisy |
| 7.75–8.25 | Ledger | every number → bench/RESULTS.md; postmortem skeleton → `performance/notes/A6_serving_day.md`; release |

### Pre-registered predictions P1–P7 (derivations inline; copy to RESULTS.md before the session)

| # | experiment | predicted | derivation / bound |
|---|---|---|---|
| P1 | AllReduce busbw, large msg (≥256 MB) | **≥720 GB/s** (80% of the 900 GB/s line rate); small msg (1–8 MB, decode-sized): **latency floor 15–40 μs** | ring efficiency at line rate; the small-msg floor is the input to P2 |
| P2 | R1 B=1 decode tok/s | **30–60 — LATENCY-bound, not bandwidth-bound** | naive active-weight roofline: ~37 GB active FP8 ÷ 8 GPUs ÷ 4.8 TB/s ≈ 0.96 ms → ~1,000 tok/s. Real bound: 61 layers × (2 AllReduces × 20–40 μs + launches + routing) ≈ 4–8 ms/token. THE node-scale B=1 lesson (contrast the sm_120 arc: overhead → memory; here: latency) |
| P3 | aggregate decode @ concurrency ≥128 | **≥3,000 tok/s**; the scaling knee arrives LATER than dense | MoE amortization is weaker than dense: E[experts hit] = 256·(1−(1−8/256)^B) → B=32 hits ≈163/256, so expert-weight reads keep growing with B until B ≫ E/k = 32 — dense R3a saw AI≈B; register the bent curve |
| P4 | EP-8 vs TP-8, B ≥ 64 | **EP ≥1.2×**, with the imbalance histogram logged | wire/token: EP a2a ≈ top-k·h·(1 B fp8 dispatch + 2 B bf16 combine)·(7/8) ≈ **150 KB** vs TP-MoE AllReduce ≈ 61 × 2·(7/8)·7168·2 B ≈ **1.5 MB** — ~10× less wire and no replicated expert reads; risk = hot experts (per-expert load ties to `moe.py`'s aux-loss-free balancing) |
| P5 | MLA KV/token (engine-reported) | **≈70 KB/token BF16 (±10%)** | (512+64)×61×2 B; capacity: free-KV-GB ÷ 70.3 KB ≈ the ~6 M-token pool (e.g. 128 × 50 K ctx) |
| P6 | PD-disagg vs co-located | **TTFT p95 ≥2× better at matched throughput** (or goodput ≥1.3× under SLO) | co-located prefill bursts evict decode from the batch (R4.2/R4.6 at scale); measured on the two-load trace — shallow for TTFT, saturated for throughput (R3b lesson) |
| P7 | MTP spec-decode (stretch) | acceptance **60–80%**, decode **×1.5–2** | R1 ships an MTP head; acceptance is domain-dependent — register the band, measure |

### DoD (all `[FACT]`-ledgered BEFORE release)

- [ ] P1–P6 measured or explicitly killed with the observed blocker (P7 stretch)
- [ ] The six headline numbers in RESULTS.md: busbw large/small · B=1 tok/s + its bound · aggregate
      peak + knee-B · EP:TP ratio + balance histogram · KV/token + pool size · disagg delta
- [ ] Postmortem note (1 page, peer-review quality): what the node taught that sm_120 could not

### Kill criteria

- Bring-up > 2 h → swap to Qwen3-235B-A22B FP8 (battle-tested, fits everywhere); the day's physics
  survives the model swap.
- Topo shows PIX/PHB, or weight-pull ETA > 90 min → kill the listing inside hour 1 (sunk ≤ $30).
- An engine bug blocks EP or disagg → do NOT debug the engine on node-time; measure the TP-8
  surface completely, file the gap in the postmortem.

**Cost: 6–10 node-hrs × $20–32 ≈ $150–320.**

---

## Phase 5 — OPTIONAL multi-node A6 (training-leaning) + A7 Capstone kernel polish

> **Re-scoped 2026-07-03 (ADR-0012): the multi-node block is CUT from the inference track.** The
> one inference-relevant fact it adds — the **~18× NVLink→IB cliff** (§4.2, 900 → ~50 GB/s/dir) —
> is legible from Phase-4 single-node numbers + nccl-tests documentation. Production multi-node
> serving (prefill/decode fleets, cross-node DeepEP) reuses the primitives Phase 4 measures; what
> it adds is RDMA plumbing and ops practice — a job, not a rental. Rent only for A6 completeness
> or A7 Track C. **A7 Track B (kernel suite) stays committed** — it needs H100/B200 hours, not a
> cluster.

**Optional multi-node scope (2-node, 16× H100 over IB), if pursued:**
- A6 Rung 2 (1F1B pipeline — moved from Phase 4; PP is a cross-node serving tool, a single-node
  training exercise) + Rung 3: EP all-to-all (NCCL AllToAll → DeepEP); DeviceMesh DP×TP×PP×EP
- §4.1 EP overlap under compute (nsys zero-SM proof) · §4.2 the IB segment of the cliff ·
  §4.3 MFU at 16/32/64 · §4.4 DeviceMesh re-mapping · §4.5 async DCP + elastic restart

**A7 Capstone (Track B — Kernel Suite, committed):**
- Integrate A2§4 WGMMA GEMM + A4 FA3 attention + A5 NVFP4
- Benchmark all three vs their ceilings; write the design doc
- PR to FlashInfer / CUTLASS / vLLM (strongly encouraged)

| Item | Est. cost |
|---|---|
| A7 capstone (H100/B200 kernel polish) — committed | ~$30–50 |
| Multi-node A6 (2-node 16× H100) — optional | ~$50–100 |
| **Total Phase 5** | **~$30–50 (+$50–100 optional)** |

---

## Total Cost and Timeline Estimate

| Phase | Duration | Compute cost (est.) |
|---|---|---|
| 1a–1e (sm_120) | 6–8 weeks part-time | $0 |
| 2 (H100 batch + serving block) | 1 session (~12–15 hrs) | ~$30–45 |
| 3 (B200 batch) | 1 session (~4–6 hrs) | ~$25–45 |
| 4 (**8× H200 serving day**) | 1 session (~6–10 node-hrs) | ~$150–320 |
| 5 (A7 kernel polish; multi-node optional) | 0–2 sessions | ~$30–50 (+$50–100 opt.) |
| **Total (inference track)** | **~8–12 weeks** | **~$235–460** |

**Three rental sessions, peak 8 GPUs concurrent** (ADR-0012). Budget-cut order if constrained:
drop B200 first (defer — DELTA's NVFP4 headline eventually needs it) → fold the Phase-2 single-GPU
session into the node day (H200 is sm_90a too; the kernels compile there) → **the node is
irreducible**: a single GPU cannot teach frontier serving, because frontier serving is
definitionally a multi-GPU problem (the weights don't fit). Phases 1+2 alone still deliver A1–A5
with Hopper frontier kernels for ~$30–45.

---

## Rung-to-Rung Workflow (session template)

Every session working a rung:

```
1. /standup  →  orient: read "Current node" in this file; read the rung's spec in PERF_ENGINEERING_SPEC.md
2. Write prediction in bench/RESULTS.md BEFORE running  (predicted | — | — | — | next hypothesis)
3. Invoke bench-writer: "Write the failing correctness test + bench harness for [rung], oracle = [X]"
4. Implement the kernel — per `.claude/execution-mode` (ADR-0013): `delegate` (current) → Claude
   implements end-to-end, in a different context than the bench-writer; `learn` → human implements (Mode-3)
5. pytest -m gpu confirms oracle-correct
6. Run bench; log measurement to bench/RESULTS.md (measured | bound | root-cause | next)
7. Invoke roofline-analyst if the number doesn't match prediction
8. Invoke kernel-ship-reviewer before committing
9. Update "Current node" in this file
10. Update docs/STATUS.md §"Performance Curriculum"
11. /eod (close session, persist state)
```

---

## Design Notes Tracker

| Assignment | File | Status |
|---|---|---|
| A1 design note | `performance/notes/A1_design_note.md` | ✅ (2026-07-04) |
| A2 design note | `performance/notes/A2_design_note.md` | ✅ (2026-07-04) |
| A3 design note | `performance/notes/A3_design_note.md` | ⬜ |
| A4 design note | `performance/notes/A4_design_note.md` | ⬜ |
| A5 design note | `performance/notes/A5_design_note.md` | ⬜ |
| A6 design note | `performance/notes/A6_design_note.md` | ⬜ |
| A7 capstone design doc | `performance/notes/A7_capstone_design.md` | ⬜ |

Notes live in `performance/notes/` (create the dir when writing the first note).
