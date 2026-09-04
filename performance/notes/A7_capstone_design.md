# A7 Capstone Design — Track B: the Blackwell/Hopper kernel suite

> A7 integrates A2–A5 into one principal-grade artifact. Track chosen: **B (kernel suite)** — it needs
> H100/B200 hours, not a cluster (the cheapest capstone), and it composes the exact primitives this
> curriculum already built + prepared. Tracks A (vLLM-lite engine) and C (distributed at scale) are the
> ⚠ 31/08: `../DELTA.md` and `docs/PERFORMANCE_TRACK.md` cited below are DELETED, and the DELTA lane
> is not in `PLAN.md` — read this note as a dated design record, not as a live target.
>
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

## 5. Re-verification 2026-07-09 — elevate DELTA to the headline; add the DSL layer

A perf/hiring deep-research pass (`wf_f3af3987-949`, primary-source-verified — findings in
`docs/PERFORMANCE_TRACK.md` §2.1 + §6) re-checked this capstone against the mid-2026 frontier. Two
structural updates that maximize the highest-paying-role signal:

**(a) DELTA is the #1 artifact — but re-scoped to the ONE thing no library ships: the fused
LOW-PRECISION decode kernel.** `[FACT]` GDN-2 (Gated DeltaNet-2, NVIDIA, arXiv **2605.22791**, May 2026 —
channel-wise erase+write gates, generalizing GDN and Kimi-KDA) beats Mamba-2/GDN/KDA/Mamba-3, biggest
gains on long-context RULER. At batch-1 GDN decode is **memory-bound** — the fixed-size recurrent state
round-trips HBM every token (arXiv 2603.05931). **⚠ Correction (second research pass):** GDN-2 is **already
in flash-linear-attention (FLA)** at the layer (`fla/layers/gdn2.py`) *and* the kernel level — **including a
standard-precision `fused_recurrent` DECODE kernel** (`fla/ops/gdn2/fused_recurrent.py`). Qwen's FlashQLA
is chunked-**prefill only**. So DELTA is **not** greenfield at the arch level *or* the plain-decode level;
its **only defensible scarcity is the FP8/NVFP4 recurrent-*state* materialization decode path** — low
precision directly cuts the HBM round-trip that bounds batch-1 decode (the "HBM-bounded, precision can't
help" framing was refuted 0-3). → **Re-scope DELTA to exactly the fp8/nvfp4 state-read/write decode kernel
on the GDN-2 two-gate recurrence, benchmarked against FLA's `fused_recurrent` as the baseline oracle.**
So-scoped, it stays the payload — the three A2–A5 kernels are the warm-up + reusable primitives (a fast
low-precision GEMM = the GDN state projection) — built *married to F10* (model-side Gated-DeltaNet in
`linear_attn.py`). Architecture↔kernel co-design is the scarcest RE signal `[INFERENCE — RQ6 comp evidence
did not survive verification]`; whether this niche out-signals a broadly-useful OSS PR is an open question.

**(b) Author in a Python DSL (CuTe DSL / TileLang), not only hand-PTX.** `[FACT]` **FlashAttention-4**
(arXiv 2603.05451, Mar 2026, MLSys oral) is **written entirely in CuTe-DSL (Python), zero CUDA C++** —
1605 TF/s B200 BF16 (71%), Blackwell-native (TMEM, 2-CTA MMA), 20–30× faster compile. **CUTLASS 4.0**
(Jun 2025) made Python the kernel-authoring surface; FlashQLA uses **TileLang**. The hand-PTX WGMMA/
tcgen05 work stays (it is the SASS you must be able to read — the DSL lowers to PTX), but the capstone's
authored kernels — above all DELTA — should be **written in CuTe DSL or TileLang** to match how 2026
frontier kernels actually ship. Deliverable framing: "here is the CuTe-DSL kernel, here is the PTX it
lowers to, here is why it hits X% of the roofline" — that pairing is the 2026 kernel-engineer signal.

**(c) Generation refresh.** FA3-on-H100 = *understand the Hopper warp-spec generation*; the **headline
attention kernel is FA4-class on Blackwell** (the standing sm120 + B200 rental are Blackwell). The
tcgen05/NVFP4 GEMM is now **vendor-served by CUTLASS 4.0** → its value is understanding + beating MXFP4
for the documented reason + using the primitive well, not reimplementing it. The OSS-PR target that
carries the most signal is a **linear-attention decode contribution** (FlashInfer / vLLM / the FLA
ecosystem), i.e. a productized slice of DELTA — the gap FlashQLA leaves open.
