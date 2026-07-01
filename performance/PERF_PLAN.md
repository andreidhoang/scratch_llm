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
Phase: 1a — A1 Rung 1 (KV-cache decoder: prove decode is memory-bound)
Hardware: Standing GPU (sm_120)
Next action: size a ~1B bf16 model, run serving.run_baseline (locked clocks, discard warmup),
  place decode on the roofline three ways (hand AI≈1 · Nsight SoL Memory%≫Compute% · measured
  tok/s vs ~266 ceiling) and log the first bench/RESULTS.md row (predicted vs measured + gap).
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
          Rent 1× H100 SXM (~$2/hr)                                  ~8–12 H100-hrs
───────────────────────────────────────────────────────────────────────────────────
Phase 3   B200 batch  (A3 §4.2–4.3 tcgen05/NVFP4-MMA, A5 §7 NVFP4 B200)
          Rent 1× B200 (~$4/hr)                                       ~4–6 B200-hrs
───────────────────────────────────────────────────────────────────────────────────
Phase 4   A6 single-node  (8× H100 SXM node, rungs 0–2)
          Rent 8× H100 node (~$23/node-hr)                            ~4–6 node-hrs
───────────────────────────────────────────────────────────────────────────────────
Phase 5   A6 multi-node + A7 capstone
          Rent 2-node, 16× H100 + H100/B200 for capstone kernels     ~6–10 hrs
───────────────────────────────────────────────────────────────────────────────────
Total est. cost (rough): <$300 for A1–A5 (single-GPU); $100–200 H100 batch;
                          $50–100 B200 batch; $150–200 multi-GPU (A6). Budget: ~$700.
```

---

## Phase 1a — A1: Transformer Inference as a System

**Hardware:** sm_120 standing GPU  
**Sequential order within A1:** Rung 0 → 1 → 2 → 3 → 4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6  
**Prerequisites:** A1 spec registered in PERF_ENGINEERING_SPEC.md §4/A1 ✅

| Rung | Status | DoD gate | Hardware |
|---|---|---|---|
| 0: metrics harness + PyTorch eager baseline | ✅ | metrics reproducible, fixed-seed (34f739e) | sm_120 |
| 1: KV-cache decoder (contiguous) | ⬜ | token-exact; GEMM→GEMV shift measured | sm_120 |
| 2: GQA/MQA | ⬜ | KV memory reduction measured | sm_120 |
| 3: continuous batching (Orca-style) | ⬜ | ≥2× aggregate throughput | sm_120 |
| 4.1: PagedAttention (16-tok blocks, Triton) | ⬜ | <4% waste; contiguous-match test | sm_120 |
| 4.2: chunked prefill | ⬜ | TTFT/ITL curve vs chunk size | sm_120 |
| 4.3: speculative decoding (lossless) | ⬜ | greedy output token-exact | sm_120 |
| 4.4: CUDA graphs decode | ⬜ | step-time reduction on nsys | sm_120 |
| 4.5: MLA latent cache (toy scale) | ⬜ | weight-absorption identity verified | sm_120 |
| 4.6: prefill/decode disaggregation | ⬜ | goodput vs co-located baseline | sm_120 |
| Design note | ⬜ | 2–3 pages, peer-review quality | — |

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
| 0: profiler + roofline harness (ncu automation, CSV, matplotlib) | ⬜ | reproduces bench/RESULTS.md baseline | sm_120 |
| 1: GEMV ladder (naive→coalesced→two-stage→vectorized float4) | ⬜ | >80% of 0.55 TB/s; Nsight SoL Memory%≈100% | sm_120 |
| 2: softmax (online→warp-shuffle→fused) | ⬜ | adversarial tests pass; traffic win measured | sm_120 |
| 3: RMSNorm + LayerNorm (Welford) | ⬜ | RMSNorm speedup vs LN quantified | sm_120 |
| 4: Top-K ladder (naive→min-heap→parallel); fused softmax+TopK | ⬜ | honest failure mode documented | sm_120 |
| 5: GEMM part 1 (naive→coalesced→SMEM tile); run at ≥4096³ | ⬜ | %-of-cuBLAS shape: ~1%→8%→13% | sm_120 |
| 6: GEMM part 2 (1D→2D blocktiling→vectorized); Memory%→Compute% flip | ⬜ | %-of-cuBLAS: ~37%→68%→78%; flip confirmed | sm_120 |
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
| R0: INT8 sym per-tensor | ⬜ | SQNR ≈44 dB; reproduce book example | sm_120 / CPU |
| R1: INT8 asym + per-channel/token | ⬜ | asym > sym on post-GELU; per-channel > per-tensor | sm_120 / CPU |
| R2: group-wise INT4 (g=128), packed | ⬜ | bit-exact pack/unpack on adversarial inputs | sm_120 / CPU |
| R3: NVFP4 two-level + MXFP4 comparison + block-scaled GEMM | ⬜ | NVFP4 block MSE < MXFP4; per-block histogram | sm_120 / CPU |
| R4: FP8 E4M3 KV cache wired into A1 decoder | ⬜ | quality within noise of BF16; 2× capacity | sm_120 |
| §4.3: PTQ method (AWQ/GPTQ/QuaRot) on real linear layer | ⬜ | measured accuracy recovery on real model | sm_120 / CPU |
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
| **Total** | | **~12 H100-hrs ≈ ~$24** |

**Critical: log EVERY measurement to bench/RESULTS.md BEFORE the instance is released.**

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

## Phase 4 — A6 Single-Node (8× H100 SXM)

**Gate:** Phases 1–3 complete; A1–A5 all DoD boxes checked.

**Single-node work (can be done without multi-node):**
- Rung 0: nccl-tests, topology map, Ring/Tree/NVLS comparison
- Rung 1: Megatron TP MLP (TP-8 single-node, ~100% scaling efficiency target)
- Rung 2: 1F1B pipeline (naive → streams+events → 1F1B → interleaved)
- §4.2: Bandwidth cliff (NVLink busbw at message sizes)
- §4.5: Async DCP checkpoint + elastic restart

**Multi-node (Rung 3 + §4.1, §4.3, §4.4) → Phase 5.**

| Order | What | Est. hrs (node-hrs @ $23/hr) |
|---|---|---|
| 1st | Rung 0: topology + nccl-tests busbw sweep + NVLS | 1 node-hr |
| 2nd | Rung 1: TP-8 MLP | 1 node-hr |
| 3rd | Rung 2: pipeline (all variants) | 1.5 node-hr |
| 4th | §4.2 (single-node portion): NVLink busbw chart | 0.5 node-hr |
| 5th | §4.5: checkpoint + elastic restart test | 0.5 node-hr |
| **Total** | | **~4.5 node-hrs ≈ ~$105** |

---

## Phase 5 — A6 Multi-Node + A7 Capstone

**Gate:** A6 single-node DoD boxes checked.

**Multi-node (2-node, 16× H100 over InfiniBand NDR):**
- Rung 3: EP all-to-all (NCCL AllToAll → DeepEP); DeviceMesh DP×TP×PP×EP
- §4.1: EP overlap under compute (DualPipe decomposition); nsys zero-SM proof
- §4.2: Bandwidth cliff (IB segment): annotate the full NVLink→IB ratio
- §4.3: MFU at 16/32/64 GPUs; six-killer decomposition
- §4.4: DeviceMesh single-config re-mapping

**A7 Capstone (Track B — Kernel Suite):**
- Integrate A2§4 WGMMA GEMM + A4 FA3 attention + A5 NVFP4
- Benchmark all three vs their ceilings; write the design doc
- PR to FlashInfer / CUTLASS / vLLM (strongly encouraged)

| Phase | Est. cost |
|---|---|
| Multi-node A6 | ~$50–100 (cluster hrs) |
| A7 capstone (H100/B200 kernel polish) | ~$30–50 |
| **Total Phase 5** | **~$80–150** |

---

## Total Cost and Timeline Estimate

| Phase | Duration | Compute cost (est.) |
|---|---|---|
| 1a–1e (sm_120) | 6–8 weeks part-time | $0 |
| 2 (H100 batch) | 1 session (~12 hrs) | ~$24 |
| 3 (B200 batch) | 1 session (~5 hrs) | ~$25 |
| 4 (8× H100 single-node) | 1–2 sessions | ~$105 |
| 5 (multi-node + capstone) | 1–2 sessions | ~$80–150 |
| **Total** | **~8–12 weeks** | **~$250–$310** |

Budget-conscious path: Phases 1 and 2 alone deliver A1–A5 with H100 frontier kernels for ~$25 in
rental spend and most of the principal-level competency. Phase 3 (B200) and Phase 4+ (multi-GPU)
are the frontier edge; they can be deferred without blocking the capstone kernel work.

---

## Rung-to-Rung Workflow (session template)

Every session working a rung:

```
1. /standup  →  orient: read "Current node" in this file; read the rung's spec in PERF_ENGINEERING_SPEC.md
2. Write prediction in bench/RESULTS.md BEFORE running  (predicted | — | — | — | next hypothesis)
3. Invoke bench-writer: "Write the failing correctness test + bench harness for [rung], oracle = [X]"
4. Human implements the kernel (Mode-3 boundary)
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
| A1 design note | `performance/notes/A1_design_note.md` | ⬜ |
| A2 design note | `performance/notes/A2_design_note.md` | ⬜ |
| A3 design note | `performance/notes/A3_design_note.md` | ⬜ |
| A4 design note | `performance/notes/A4_design_note.md` | ⬜ |
| A5 design note | `performance/notes/A5_design_note.md` | ⬜ |
| A6 design note | `performance/notes/A6_design_note.md` | ⬜ |
| A7 capstone design doc | `performance/notes/A7_capstone_design.md` | ⬜ |

Notes live in `performance/notes/` (create the dir when writing the first note).
