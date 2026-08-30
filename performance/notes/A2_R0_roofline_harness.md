# A2 Rung 0 — Profiler + roofline harness · spec + result

> Parent: `performance/A2_kernel_optimization.md` §3 R0. Deliverable: the reusable measurement spine
> every A2 kernel (R1–R6) is scored against. Gate = reproduce the `bench/RESULTS.md` R0 baseline
> (~0.55 TB/s HBM, ~72 TF/s bf16). No perf target (it IS the harness).

## What was built + the ncu re-base

`bench/kernel_roofline.py` on top of the existing `bench/_harness.py` primitives (CUDA-event timing
+ per-rep L2 flush via `triton.testing.do_bench`; peaks measured on THIS GPU; the `Roofs.attainable`
roofline arithmetic, already CPU-tested in `src/scratch_llm/bench/roofline.py`):
- `profile(name, fn, flops, bytes) -> KernelProfile` — measures, places on the roofline (% of the
  binding roof, mem/cmp), records median + p20–p80 spread (>5% ⇒ distrust).
- `write_csv` (sweep-to-CSV) + `plot_roofline` (matplotlib log-log roofline PNG with the kernels
  placed and the ridge marked).

**ncu is BLOCKED on this box** (`ERR_NVGPUCTRPERM`, unprivileged container) — the A2 spec's
`ncu --set full` Speed-of-Light / MemoryWorkloadAnalysis / bank-conflict / WarpState counters are
unavailable. Re-base (documented in the module as `A2_ncu_debt`, 5 registered metrics): a kernel's
bound is established by **achieved-vs-measured-peak %** (this harness) + **nsys** for launch/timeline;
each A2 kernel names the ncu metric it WOULD inspect (sectors/request, bank conflicts, warp stalls,
tensor-pipe util).

**Routing corrected 2026-08-30.** The list was filed against "the H100 rental day" and sat 57 days.
**4 of the 5 are sm_120-bound and Hopper cannot discharge them even in principle:** the CUDA rungs are
`-arch=sm_120` (compute_120 does not load or JIT on sm_90 — PTX compat is forward-only), and the
Triton rungs would only run on Hopper by recompiling to different SASS + autotune configs, i.e. a
different kernel instance. Only tensor-pipe util (WGMMA) is genuinely Hopper. **Route ncu-debt by the
kernel's target arch, never by the largest card available** — the discharging card is a `vms_enabled`
KVM 5090 (~$0.33/hr), available now. Same compute capability ≠ same card: these rows are 70 SMs /
0.551 TB/s, a 5090 is ~170 SMs / ~1.8 TB/s, so re-run R0 first to re-anchor peaks; the counter claims
and the %-of-peak method transfer, the absolute numbers do not.

## Measured (2026-07-04, `bench/kernel_roofline.py`)

- **R0 gate PASS `[FACT]`:** measured peaks **0.551 TB/s HBM · 72.1 TF/s bf16 · ridge 131 FLOP/byte**
  — reproduces the 2026-06-29 baseline (0.55 TB/s / 72 TF/s / ridge ≈130) within noise.
- Two anchors validating the harness: `copy(256 MB)` at **551 GB/s = 100.1% of the mem roof** (AI 0.12,
  memory-bound); `gemm(8192³)` at **72.1 TF/s = 100.0% of the compute roof** (AI 2731, compute-bound).
  Spread <1.2% (stable enough to trust despite unlocked clocks).
- Artifacts: `bench/a2_roofline.{csv,png}`.

**Verdict — SHIPPED.** The A2 measurement spine reproduces the baseline and places kernels on the
roofline with mem/cmp classification + CSV/plot; the ncu gap is documented as debt discharged on
sm_120 + counters (4 of 5) and Hopper (tensor-pipe only). Ready for R1 (GEMV ladder). The R1–R6
CUDA-core kernels are sm_120-runnable; the §4.1–4.6 WGMMA/TMA/FP8 rungs are H100-code-only (rental
runbook).
