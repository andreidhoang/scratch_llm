# A7 Capstone Design — Track B: the Blackwell/Hopper kernel suite

> A7 integrates A2–A5 into one principal-grade artifact. Track chosen: **B (kernel suite)** — it needs
> H100/B200 hours, not a cluster (the cheapest capstone), and it composes the exact primitives this
> curriculum already built + prepared. Tracks A (vLLM-lite engine) and C (distributed at scale) are the
> alternatives; the DELTA GDN-2 kernel (private `../DELTA.md`) is the *separate* portfolio spike on top.

## 1. What A7 Track B integrates (and its state after this sprint)

Three kernels, benchmarked against their ceilings on locked clocks with roofline + Nsight evidence:

| kernel | curriculum source | state (2026-07-04) | ceiling target |
|---|---|---|---|
| warp-spec WGMMA+TMA persistent GEMM (Hopper) | A2 §4.2–4.3, A3 R3–§4.1 | **compile-verified `sm_90a`** (`performance/rental/kernels/wgmma_gemm_sm90a.cu`) + PTX artifact | ≥80% of cuBLAS on the target shape |
| FP8 FA3-style attention | A4 R4 | **compile-verified skeleton `sm_90a`** (`fa3_attention_sm90a.cu`) | in sight of FA3 ~75% util / ~740 TF/s |
| NVFP4 block-scaled GEMM vs BF16/FP8 | A5 R3, A3 §4.3 | **numerics done + verified on sm120** (NVFP4 MSE 1.48× < MXFP4); native MMA compile-verified `sm_100a` (`tcgen05_gemm_sm100a.cu`) | <1% accuracy of FP8, beats MXFP4 for the documented reason |

So the capstone is no longer a blank page: the numerics are measured, the WGMMA descriptor is hand-decoded
(`performance/artifacts/wgmma_descriptor_manual.md`), and the three kernels compile for their target ISAs.
The rental day is **integration + tuning + measurement**, not first-draft authoring.

## 2. The A7 deliverables (per the rubric)

1. **System reproducible from a clean instance via script** — the `H100_day_runbook.md` + `B200_day_runbook.md`
   are that script's spine (boot → clock-lock → ncu-verify → build → measure). A7 adds the integration
   harness that runs all three kernels + logs the table.
2. **Benchmark report vs named ceilings** on locked clocks, trace/shape/dtype/hardware pinned, roofline +
   Nsight evidence per kernel — discharging the **ncu-debt** registered across A2/A3/A4 (the sm120 box
   could not measure the SoL/bank-conflict/tensor-pipe counters; the H100/B200 days do).
3. **3–5 page design doc**: hypothesis → architecture → the number → the honest gap with per-component
   attribution (e.g. "WGMMA GEMM 84% of cuBLAS: 8% unhidden TMA, 5% epilogue, 3% wave quantization —
   here are the three Nsight sections") → next-with-2×-time. The A1–A5 design notes are the input.
4. **OSS PR (strongly encouraged):** the FA3 kernel → FlashInfer, or the NVFP4 GEMM → CUTLASS, or the
   paged/chunked serving work → vLLM. The R4.2b piggyback finding and the R4.4 CUDA-graph-over-paged-pool
   are the most PR-ready serving contributions.

## 3. The gate (§9) and why this curriculum passes it

A7's gate: *handed an unfamiliar model + GPU + SLO, within a day produce a roofline-grounded prediction,
a profiler-backed diagnosis, and a prioritized 3–5-change plan with maintenance-cost judgment.* This
sprint IS that skill exercised end to end: every rung predicted the bound before running (roofline-first),
diagnosed divergences (R4.2's honest negative, R1's overhead→memory arc), and named the next constraint
(ncu-debt, the WGMMA jump, the piggyback fix). The capstone is the same loop on real Hopper/Blackwell
silicon — the one thing this sm120 box could not provide, and the only thing the three rental days add.

## 4. Sequencing

A7 Track B runs on the **H100 day** (WGMMA GEMM + FA3) and the **B200 day** (NVFP4 native MMA), folded
into those rentals — no separate cluster. Gate to start: A1–A6 at "Strong" (A1–A5 sm120 ✅; A6 primitives
gloo-verified ✅; the ISA rungs compile-verified ✅). The DELTA GDN-2 spike (private workspace) then sits
on this base — same infrastructure, one deep differentiating artifact.
