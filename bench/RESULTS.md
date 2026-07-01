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
| 2026-07-01 | A1 R1 · decode tok/s (0.84B bf16) | RTX PRO 4000 Blackwell (sm120) | decode tok/s @B=1 | 315 (mem ceiling) | **51** (16% of roof) | overhead | memory-bound *workload* (AI≈0.96, 136× below ridge) but the eager run is launch-overhead-bound, NOT memory-saturated: achieves only 89 GB/s = 16% of the card's 550 GB/s; nsys = ~955 kernel launches/token (dominant kernel = GEMV, the decode shape) + a `.item()` host-sync per token; memory traffic is 16% of the 19.6 ms/token wall, 84% is overhead | CUDA graphs (R4.4) collapse the 955 launches → predict climb toward 315 tok/s & achieved-BW→550 (only then memory-bound); keep token-id on GPU to kill the per-token sync |

<!-- append new measurements below; never edit a logged row (it is a dated record) -->

### Pre-registration — A1 R2 GQA/MQA reduction (predict-before-run, D5)

Registered 2026-07-01 BEFORE running `bench/kv_memory.py`. Spec: `performance/notes/A1_R2_gqa.md`.
KV stored/token = `2·L·H_kv·d_head·dtype` = `4096·H_kv` bytes (Rung-1 config, bf16). Honest frame:
decode is overhead-bound at B=1, so GQA moves tok/s **only** at long ctx under `torch.compile` (pred 5).

| # | experiment | predicted | measured | bound | note |
|---|---|---|---|---|---|
| R2.1 | KV/token vs `4096·H_kv`, H_kv∈{32,8,1} | 128 / 32 / 4 KB (32:8:1) | — (pending) | — | footprint fact |
| R2.3 | crossover ctx (KV read = 1.68 GB weights) | MHA ~13 K · GQA-4 ~51 K · MQA ~410 K | — (pending) | — | GQA pushes KV-bound point out |
| R2.5 | decode tok/s @ ctx=16 K, compiled — MQA vs MHA | **MQA >20% faster** (MHA KV≈2.1 GB > weights) | — (pending) | memory | the only tok/s-moving falsifier |

**Kill line:** grouped math ≠ explicit-repeat reference ⇒ group-map bug (fix first). MQA no edge over
MHA at ctx=16 K under compile ⇒ not at the wall (re-check compiled BW ≈53%) or KV-traffic model wrong.

### Pre-registration — A1 R1 overhead-strip (predict-before-run, D5)

Registered 2026-07-01 BEFORE running `bench/decode_overhead_strip.py`. Baseline is the row above
(51 tok/s = 16% of the 315 ceiling; 89 GB/s = 16% of 550; ~955 launches/token + a `.item()`/token
host-sync). Hypothesis: the eager decode is **overhead-bound, not memory-bound** — strip the overhead
and achieved BW climbs toward the 550 GB/s wall (only then is the memory-bound thesis *demonstrated*).

| # | strip (one variable) | predicted tok/s | predicted BW | predicted bound after strip |
|---|---|---|---|---|
| A | kill per-token `.item()` host-sync (argmax stays on GPU) | 51 → **~60–70** | ~89 → ~110 GB/s | still overhead (launches dominate) |
| B | + `torch.compile` (fusion / reduce-overhead) collapse the ~955 launches | → **~150–300** | ~89 → **250–500 GB/s** | **memory** (if the wall appears) |

**Kill line (falsifier):** if achieved BW stays **< 200 GB/s** after both strips, launch overhead is
NOT the dominant confound — re-profile for the real one (unfused RMSNorm/RoPE eager ops? the growing
`torch.cat` KV realloc? Python-side dispatch?). A surprise up is as suspicious as a surprise down (D5).
**Known risk:** the KV cache grows via `torch.cat` (dynamic shape) — CUDA-graphs need static addresses,
so `reduce-overhead` may not capture the step; if so, that scopes R4.4 (static pre-allocated KV buffer).

### Measured — A1 R1 overhead-strip (2026-07-01, `bench/decode_overhead_strip.py`)

| date | rung / artifact | hardware | metric | predicted | measured | bound | root cause | next experiment |
|---|---|---|---|---|---|---|---|---|
| 2026-07-01 | A1 R1 · strip A: nosync (argmax on GPU) | RTX PRO 4000 Blackwell (sm120) | decode tok/s @B=1 | ~60–70 (pred A) | **50** (15% roof · 83 GB/s · 15% HBM) | overhead | **Pred A FALSIFIED.** Killing the per-token `.item()` host-sync is ~free (48→50): against ~955 sequential kernel launches/token the single sync is negligible — the confound is launch *dispatch*, not the sync | fuse the launches (`torch.compile`) — that's the real lever, not the sync |
| 2026-07-01 | A1 R1 · strip B: compiled[default] (inductor fusion) | RTX PRO 4000 Blackwell (sm120) | decode tok/s @B=1 | 150–300 · 250–500 GB/s (pred B) | **173** (53% roof · **291 GB/s · 53% HBM**) | overhead→memory | **Pred B CONFIRMED.** Inductor fuses the ~955 pointwise launches → achieved BW climbs 81→291 GB/s (15%→53% of the wall). **Memory-bound thesis demonstrated in trend** (3 pts on a line: strip launches → BW rises). Residual gap = the ~180 GEMV/matmul launches inductor still issues/token | **R4.4 CUDA-graph decode** to collapse the residual launches → predict climb toward the 327 tok/s ceiling / 550 GB/s |
| 2026-07-01 | A1 R1 · compiled[reduce-overhead] (cudagraphs) | RTX PRO 4000 Blackwell (sm120) | graph capture | expected to help | **capture FAILED** | — | `RuntimeError: accessing tensor output of CUDAGraphs overwritten by a subsequent run` — the `torch.cat`-grown KV cache (dynamic addresses) violates cudagraph static-memory capture. **Scopes R4.4 precisely:** need a static pre-allocated KV buffer with in-place writes (the same primitive PagedAttention R4.1 needs) | R4.4: rewrite `KVCache` as a fixed `[B, kv_heads, max_ctx, head_dim]` buffer, write at `length`, slice `[:length]` — then re-attempt cudagraph capture |

**Verdict (A1 R1 close-out):** the founding thesis — *decode is memory-bound; eager launch overhead hides
the wall* — is now **demonstrated in trend** on the standing GPU: stripping launches moves achieved HBM
15% → 53%, on a straight line. It is **not yet closed to the DoD bar** ("within 15% of ceiling / Nsight
SoL Memory%≫Compute%") — the last ~47% to the wall is the residual per-token launch overhead that only
static-graph capture removes, and that is blocked by the `cat`-cache. Next node **R4.4** (static KV
buffer → CUDA graphs) is now the pre-registered continuation, with the cudagraph failure as its spec.
