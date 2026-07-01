# Measurement ledger — predicted vs measured (the "DoD is a profile" record)

> The durable record of every **measured** performance number. Git + [`../docs/STATUS.md`](../docs/STATUS.md)
> track *what is built*; this file tracks *what is measured* — because a rented GPU is released and the
> number must persist. The discipline (FOP-3 / FOP-4, and every rung's *Profile (DoD)* in
> [`../docs/GPU_FROM_ZERO.md`](../docs/GPU_FROM_ZERO.md)): **predict the number and the bound first, then
> measure, then log the gap and the root cause.** The spine that says *which* numbers matter is
> [`../docs/PERFORMANCE_TRACK.md`](../docs/PERFORMANCE_TRACK.md).
>
> A row is a result only if it is `[FACT]` — measured under `cuda.synchronize`, fixed seed, warm-ups.
> An unmeasured expectation is `[INFERENCE]` and does not belong here until measured.

## Format

`| date | rung / artifact | hardware | metric | predicted | measured | bound | root cause (1 line) | next experiment |`

- **bound** = the roofline verdict: `compute` / `memory` / `comms` / `overhead` / `latency`.
- Use the honesty constants from `PERFORMANCE_TRACK.md §5` (dense ~295 FLOP/byte H100 ridge, FP4 = 2× FP8, …).

## Ledger

| date | rung / artifact | hardware | metric | predicted | measured | bound | root cause | next experiment |
|---|---|---|---|---|---|---|---|---|
| 2026-06 (prior) | Rung 5 · FA2 fwd roofline | RTX 4090 | % of SDPA @ seq 4k | ~65% | **53%** | memory | Triton fwd leaves HBM traffic on the table vs SDPA's fused schedule (below the 60% kill line — shipped as a documented negative) | re-measure post-bwd; larger tiles / fewer reloads; compare vs `torch.compile` fused attn |
| 2026-06-29 | hardware baseline | RTX PRO 4000 Blackwell (sm120) | bf16 GEMM 8192³ · HBM copy · ridge | — | **72 TF/s · 0.55 TB/s · ridge≈130 FLOP/B** | — | the standing card; bf16 modest (Blackwell headroom is FP4/FP8) — peaks now in `bench.py _PEAKS` | re-measure FA2 % of SDPA on *this* card; measure FP8/FP4 GEMM |
| 2026-06-29 | Triton stack check | RTX PRO 4000 Blackwell | gpu-test pass/fail | all pass | **FA2-Triton + gemv ✅ ; CUDA rmsnorm.cu = NaN** | — | Triton-primary (ADR-0011) validated on sm120; CUDA-C++ rmsnorm regresses (arch/build) — 2nd-tier, superseded by the R3 Triton norm rebuild | triage `rmsnorm.cu` sm120 build flags or quarantine `@pytest.mark.gpu` |
| 2026-07-01 | A1 R1 · decode tok/s (0.84B bf16) | RTX PRO 4000 Blackwell (sm120) | decode tok/s @B=1 | 315 (mem ceiling) | **51** (16% of roof) | memory | eager launch overhead (hundreds of tiny kernels per step) + host-to-device sync copy of token id | CUDA graphs (R4.4) / torch.compile to eliminate launch overhead; keep token id on GPU |

<!-- append new measurements below; never edit a logged row (it is a dated record) -->
