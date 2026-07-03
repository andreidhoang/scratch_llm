# Design spec — Roofline + predict-the-number harness

> **Status:** ✅ built same day (2026-06-29) and in daily use — landed as `src/scratch_llm/bench/`
> (`gpu_specs` · `roofline` · `harness` · `ledger`); every perf rung measures through it
> (`bench/RESULTS.md`). Kept as the design record.
> **Layer:** Perf spine (the seed every later kernel/system rung reuses). See
> [`../PERFORMANCE_TRACK.md`](../PERFORMANCE_TRACK.md) §3 Tier-1 and [`../GPU_FROM_ZERO.md`](../GPU_FROM_ZERO.md) Rung 1.
> **The deliverable is the discipline, not the script:** a reusable way to *predict the bound and the
> number before the run*, then overlay the measured profile (FOP-3, FOP-4).
> **Files:** `src/scratch_llm/bench/roofline.py` (the model — YOU write the FLOP/byte counting),
> `bench/roofline_plot.py` (the plot — AI may scaffold), `tests/test_roofline.py` (CPU), this spec.
> **Mode:** the *arithmetic-intensity counting logic* is the learning rep (yours); the plotting + the
> GPU-spec table + the `ncu`-parsing glue are agent-scaffoldable.

## 1. Why (the problem this solves)

Every kernel/system DoD in this repo is "land near a roofline you predicted." Today that exists once
(the FA2 roofline, hand-rolled in `bench/flash_roofline.py`). This generalizes it into the reusable
**predict → measure → place-on-roof** loop that all five 2026 research streams ranked the single
highest-EV artifact. It is the cheapest thing that makes "you think in numbers" visible.

## 2. The mechanics (what it computes)

Given `(op_descriptor, dtype, batch, seqlen, gpu)`:
```
AI            = useful_FLOPs / bytes_moved                 # arithmetic intensity
ridge         = peak_compute(gpu, dtype) / peak_bw(gpu)    # FLOP/byte; the ridge point
bound         = "memory" if AI < ridge else "compute"
t_predicted   = max(FLOPs / peak_compute, bytes / peak_bw) # the roofline time
pct_of_roof   = t_predicted / t_measured                   # filled in after the run
```
A **GPU-spec table** (sourced constants) supplies `peak_compute(gpu, dtype)` and `peak_bw(gpu)` for
H100/H200/B200 (+ the 4090 already used). A small **op library** counts FLOPs+bytes for: dense GEMM,
RMSNorm/softmax (elementwise), attention prefill, and **decode-step** (weights + KV traffic).

## 3. Correctness invariant (the CPU test, the floor)

> The analytic model must reproduce, on paper-checkable cases, the numbers you can derive by hand.

- `tests/test_roofline.py` (CPU): a 4096² BF16 GEMM is classified **compute-bound** (AI ≫ ridge);
  RMSNorm is **memory-bound**; batch-1 decode attention has **AI ≈ 1** and is memory-bound.
- The H100 dense ridge computes to **≈ 295 FLOP/byte** (989e12 / 3.35e12) — assert it, and assert the
  table does **not** silently use the sparse ~590.
- Decode tok/s lower bound for a stated (params, dtype, BW) matches the hand formula `bytes / BW` to ~1%.

## 4. Predict-before-run (write these numbers FIRST)

Before wiring any measurement, commit the falsifiable predictions (they are the debugging anchor):
> 1. Batch-1 decode of a 70B BF16 model on H100 sits **~300× below** the ridge (memory-bound), tok/s
>    ≈ `(140 GB + KV) / 3.35 TB/s` ≈ **~24 tok/s**.
> 2. A 4096² GEMM is **compute-bound** and lands at some stated % of the 989 TFLOP/s roof.
> 3. RMSNorm is **memory-bound**; fusing it ≈ 2× from halved byte traffic (Rung 3).

## 5. Honesty constants (assert them in the table)

Dense ridge ~295 (not sparse ~590) · `FP4 = 2× FP8` · B200 ≈ 9 PF dense FP4 / GB200 ≈ 10 dense, 20
sparse · always record which `(gpu, dtype, peak-kind)` a number used. Cite the datasheet per row.

## 6. Scope / deferred

- **In:** the analytic model + the CPU test + the plot + the GPU-spec table; an `ncu`/`nsys`-overlay
  hook (parse `dram__throughput` / `sm__throughput` / achieved-occupancy from a report).
- **Deferred:** live profiling integration (runs on the rented box, batched with the decode-roofline
  measurement, [`PERF_decode_roofline_SPEC.md`](PERF_decode_roofline_SPEC.md)).

## 7. Build sequence

1. *(CPU, you)* the op-library FLOP/byte counters + the GPU-spec table → the §3 CPU tests pass.
2. *(CPU, AI-scaffoldable)* the roofline-PNG plot (operating points vs the ridge line).
3. *(CPU)* write the §4 predictions into this spec as the pre-registered anchor.
4. *(rented box)* the `ncu`/`nsys` overlay; apply to the FA2 kernel + the decode timing.
5. Update [`STATUS.md`](../STATUS.md) + [`PERFORMANCE_TRACK.md`](../PERFORMANCE_TRACK.md); commit (after Tier-0 green-CI).
