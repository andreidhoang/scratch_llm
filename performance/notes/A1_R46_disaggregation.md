# A1 Rung 4.6 — Prefill/decode disaggregation (demonstrate) · spec + result

> Parent: `performance/A1_transformer_inference.md` §4.6. Oracle = R4.1. Gate = **goodput vs a
> co-located baseline** + characterize the KV-transfer overhead. Toy scale (single GPU → the two
> measurable halves, not two real workers).

## Mechanism

Prefill is compute-bound (one big GEMM over the whole prompt); decode is memory-bound (GEMV, one
token). Co-locating them lets a prefill burst steal the GPU from in-flight decodes — the measured
ITL admission spike (R4.1/R4.2). Disaggregation puts prefill on a separate worker so the decode
worker's stream is never interrupted; the price is a one-time **KV-cache transfer** (prefill worker →
decode worker). The engineering question is whether the recurring spike removed is worth the transfer
added.

## Measured (2026-07-04, `bench/disagg.py`, sm120 0.84B bf16, N_SLOTS=32)

**KV-transfer tax** (512-tok prompt, 32 blocks/layer × 16 layers = **16.8 MB** KV):
D2D pool copy **0.180 ms** (187 GB/s measured, analytic 0.061 ms at 0.55 TB/s) — a **one-time**
per-request cost. `[FACT]` The 187 GB/s < HBM peak because the transfer is 32 small per-layer copies
(launch-bound at this size); a fused copy hits nearer peak, and a real 2-GPU disagg pays the NVLink
(~900 GB/s) or cross-node RDMA (~50 GB/s) bandwidth instead — the cliff the Phase-4 node day measures.

**ITL contrast — what the decode worker sees** (the disagg benefit):

| arm | ITL p50 / p95 / p99 (ms) | agg tok/s |
|---|---|---|
| disagg decode-only (clean stream) | 18.6 / 19.0 / **20.2** | **1707** |
| co-located (+interleaved prefill) | 18.8 / 20.8 / **61.5** | 1038 |

Disaggregation drops the decode worker's **ITL p99 from 61.5 → 20.2 ms (3×)** and raises throughput
1038 → 1707 tok/s, by removing prefill bursts from the decode stream — at the cost of the 0.18 ms
one-time transfer above. **Goodput under an ITL SLO (p99 ≤ 30 ms):** disagg MEETS it (20.2), co-located
VIOLATES it (61.5) → disagg goodput strictly higher. `[FACT]`

**Verdict (A1 R4.6 — SHIPPED, demonstrated).** The disaggregation tradeoff is measured on this
substrate: a large recurring ITL p99 spike (3×) traded for a negligible one-time KV transfer (0.18 ms
≪ the ~40 ms spike it removes), with higher decode throughput. Honest scope: single-GPU simulation —
the "disagg" arm is a clean decode-only stream (what a dedicated decode worker sees); a real 2-GPU
disagg adds the cross-device KV-transfer bandwidth (D2D here as the proxy; NVLink/RDMA at node scale,
Phase-4 P6). Closes the A1 serving rungs (R0–R4.6). Extension: a true two-process demo with the KV
handed over IPC/NVLink, and the DistServe/Splitwise resource-ratio tuning.
