# Design spec — Decode-step roofline (time the KV-cache)

> **⚠ HARDWARE CORRECTION (2026-08-31).** Lines below that say "the standing sm120 GPU / no rental
> needed" were TRUE when written and are FALSE now: this host has **no GPU and no CUDA toolchain**
> (`nvidia-smi`/`nvcc`/`ncu`/`nsys` absent, `triton` not importable, arm64 — measured 29–30/08).
> Every rung here is **rental-gated**; measured numbers already in the ledger stay valid as records.
> Law: `CLAUDE.md` § Hardware reality · `PLAN.md` § Hardware law.

> **Status:** ✅ built & measured 2026-07-01 as perf-curriculum **A1 R1** — on the standing sm120
> GPU, no rental needed (`bench/decode_roofline.py` + `bench/decode_overhead_strip.py`; 51 → 173
> tok/s eager→compiled, 15%→53% HBM; `bench/RESULTS.md` §A1 R1). Kept as the design record.
> **Layer:** Perf spine, Tier-1 #2. See [`../PERFORMANCE_TRACK.md`](../PERFORMANCE_TRACK.md) §1 ("the
> wedge") and [`../GPU_FROM_ZERO.md`](../GPU_FROM_ZERO.md) Rung 6.
> **The deliverable is the profile:** the first *measured* decode result — tok/s vs the bandwidth bound
> + an `nsys`/`ncu` profile proving memory-bound. Depends on [`PERF_roofline_harness_SPEC.md`](PERF_roofline_harness_SPEC.md).
> **Files:** `bench/decode_roofline.py` (the bench — AI-scaffoldable), the existing `KVCache` in
> `model.py` (the thing under test), this spec.
> **Mode:** reading/interpreting the profile is the rep (yours); the timing harness is agent-scaffoldable.

## 1. Why (the problem this solves)

The repo's KV-cache is **correctness-tested but never *timed*** — the single highest-value first result
available. Decode is the canonical memory-bandwidth-bound workload; timing it turns an untimed asset
into a measured artifact and makes the central physical fact of 2026 inference (decode ≈ 1 FLOP/byte,
~300× below the H100 ridge) concrete on *your own* stack. It is also the on-ramp to the DELTA decode
kernel — you cannot claim a decode-kernel speedup without first measuring the baseline decode roofline.

## 2. The mechanics (what it measures)

For the A1 `TransformerLM` with `KVCache`, sweep `(batch, context_len)` and measure steady-state
per-token decode latency → tok/s. Compare to the analytic bound from the roofline harness:
```
bytes_per_token = model_weight_bytes + 2·n_layers·n_kv_heads·head_dim·context_len·batch·dtype_bytes
tok_s_bound     = HBM_BW / bytes_per_token        # the memory-bandwidth ceiling
AI_decode       ≈ FLOPs_per_token / bytes_per_token  # expect ≈ 1, batch-1
```
Always `torch.cuda.synchronize()` around timed regions (CUDA is async — untimed sync = wrong numbers).

## 3. Correctness invariant (the floor — already green)

The KV-cache correctness is the existing "cached == recompute" test (`tests/test_kv_cache.py`,
[`L2_kv_cache_SPEC.md`](L2_kv_cache_SPEC.md)). This spec adds **timing**, not correctness — do not
re-derive correctness; assert the decode path still matches recompute, then measure.

## 4. Predict-before-run (write these FIRST — the kill/confirm anchor)

> 1. Decode is **memory-bound**: measured tok/s reaches **70–80% of `tok_s_bound`**, and `ncu` shows
>    DRAM throughput near peak while SM/tensor-core throughput is low (tensor cores idle).
> 2. **AI_decode ≈ 1** at batch-1, ~300× below the H100 dense ridge (~295).
> 3. tok/s **scales with HBM bandwidth, not FLOPs** — and increasing batch raises AI (weights amortize)
>    while KV traffic grows per-sequence (KV does *not* amortize).

**Confirm/falsify:** if measured tok/s is far from the bound, the gap is the finding (kernel-launch
overhead per decode step? un-fused norm? CUDA-graph-able?). Name the constraint; don't hand-wave.

## 5. Scope / deferred

- **In:** the decode-step timing bench, the tok/s-vs-bound table, the `nsys`/`ncu` profile, the
  batch-scaling curve (AI rising with batch).
- **Deferred:** the split-K / paged-KV decode *kernel* (Rung 6 / DELTA — a separate Mode-3 build that
  this baseline measures the speedup against); CUDA-graph capture of the decode loop (Rung 6 stretch).

## 6. Build sequence

1. *(rented box, AI-scaffoldable bench)* steady-state decode timing + `cuda.synchronize` discipline.
2. *(you)* write the §4 predictions first; compute `tok_s_bound` via the roofline harness.
3. *(rented box)* `nsys` timeline + `ncu` Speed-of-Light → confirm memory-bound; record % of bound.
4. *(you)* the batch-scaling curve showing AI rise; name the dominant non-bandwidth cost if any.
5. Update [`STATUS.md`](../STATUS.md) + [`PERFORMANCE_TRACK.md`](../PERFORMANCE_TRACK.md); commit.
