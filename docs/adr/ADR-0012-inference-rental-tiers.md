# ADR-0012 — Inference rental strategy: capability tiers, not GPU counts

- **Status:** Accepted (2026-07-03)
- **Layer:** performance curriculum resourcing (`performance/PERF_PLAN.md` Phases 2–5, `PERF_ENGINEERING_SPEC.md` §2) + Capstone DELTA hardware gates
- **Decides:** exactly which GPUs are rented, how many, in how many sessions, and what each is *for* — so hardware money maps to inference mastery and the question is never re-litigated per phase

## Context

The Phase 2–5 rental plan was written training-leaning (TP/PP scaling ladders, multi-node EP,
elastic checkpointing). The program goal has sharpened: **frontier inference** — the engineering of
actually serving 2026 frontier models. For that goal, GPU count is the wrong unit of planning.
What a rental buys is access to **architecture-gated behaviors**, and there are exactly three hard
gates; nothing below a gate substitutes for it at any GPU count:

1. **ISA generation.** `wgmma.mma_async` + TMA (`cp.async.bulk.tensor`) + FP8-WGMMA are
   **sm_90a-only** (Hopper). `tcgen05.mma` + TMEM + native NVFP4 block-scaled MMA are
   **sm_100a-only** (datacenter Blackwell). The standing sm_120 card (consumer Blackwell) has
   neither — these instructions do not compile for it (`PERF_ENGINEERING_SPEC.md` §1). `[FACT]`
2. **HBM bandwidth & capacity.** Decode is memory-bound (A1, measured); the wall itself is the
   spec: 0.55 TB/s / 24 GB (standing) → 3.35 TB/s / 80 GB (H100) → 4.8 TB/s / 141 GB (H200) →
   ~8 TB/s / ~192 GB (B200). A 70 GB FP8 model's B=1 decode ceiling is `BW / weight-bytes`:
   **47.9 tok/s on H100, 68.6 on H200, ~114 on B200** — the same law we measured on sm_120
   (0.84B bf16 → 327 tok/s ceiling). `[FACT: derivation; specs from vendor docs]`
3. **Interconnect domain.** One NVLink4 domain = 900 GB/s/GPU all-to-all; crossing nodes drops to
   ~50 GB/s/direction (IB NDR400) — the ~18× cliff. TP/EP/collectives/disaggregation are only real
   inside an NVLink domain; multi-node adds RDMA plumbing, not new primitives. `[FACT]`

**The fit-math that forces the node choice** (per-model FP8 weights vs single-node HBM):

| Model (2026 open frontier) | Total / active params | FP8 weights | 8×H100 (640 GB) | 8×H200 (1,128 GB) |
|---|---|---|---|---|
| DeepSeek-V3 / R1 (MLA + 256-expert MoE) | 671 B / 37 B | ~671 GB | **NO** (671 > ~600 usable; TP-8 shard = 83.9 GB/GPU > 80) | **YES** (~450 GB left for KV) |
| Kimi-K2 | ~1.03 T / 32 B | ~1 TB | NO | W4 (~550 GB) only |
| Llama-4-Maverick | 400 B / 17 B | ~400 GB | YES | YES |
| Qwen3-235B-A22B | 235 B / 22 B | ~235 GB | **YES** (~365 GB headroom) | YES |
| gpt-oss-120b (MXFP4-native) | 117 B / 5.1 B | ~60–65 GB | **single GPU** | single GPU |

KV economics at node scale (why MLA is the 2026 serving story): R1's MLA caches
`(c_kv 512 + k_pe 64) × 61 layers × 2 B` = **70.3 KB/token** (BF16) vs Llama-3.3-70B GQA-8's
`2 × 80 L × 8 kv × 128 d × 2 B` = **327.7 KB/token** — a 671B model with a **4.7× smaller**
per-token cache than a 70B. On 8×H200, ~450 GB of KV ≈ **6.4 M cached tokens** (e.g. 128
concurrent × 50 K ctx). `[FACT: arithmetic from published configs]`

## Decision

Own Tier 0; rent **exactly three sessions**; peak concurrency 8 GPUs; skip Tier 4 for the
inference track.

| Tier | Hardware | What it alone unlocks | Sessions × hrs | Est. cost |
|---|---|---|---|---|
| **0 (own)** | RTX PRO 4000 Blackwell, sm_120, 24 GB, 0.55 TB/s | ~80% of the work: every kernel ladder rung that isn't ISA-gated, every serving algorithm (KV/paged/continuous/spec-decode/CUDA-graphs — A1 R0–R4.6), all oracles + traces | standing | $0 |
| **1 (rent)** | 1× **H100 SXM** (sm_90a) | WGMMA/TMA/FP8-WGMMA, FA3, the CUDA-core→tensor-core ~10× jump; single-GPU frontier serving (70B-FP8 dense, gpt-oss-120b MoE) | 1 × 12–15 h | ~$25–45 |
| **2 (rent)** | 1× **8× H200 SXM NVLink node** — *the crown* | Serving a real frontier MoE: DeepSeek-R1 FP8 TP×EP, NVLink collectives at line rate, MLA KV at scale, PD-disaggregation, our batching stack at frontier batch | 1 × 6–10 h | ~$150–320 |
| **3 (rent)** | 1× **B200** (sm_100a) | `tcgen05`/TMEM, native NVFP4 MMA (2026 deployment precision) + the DELTA NVFP4-state probe | 1 × 4–6 h | ~$25–45 |
| ~~4~~ (skip) | 2-node 16× over IB | only the ~18× NVLink→IB cliff + RDMA ops — primitives already owned after Tier 2 | — | $0 |

Total: **~$200–410, three sessions.** H100-vs-H200 for Tier 1: same die, same ISA — take the
cheaper hour; H200's bandwidth/capacity lesson arrives with the node. Tier-2 fallback if 8×H200 is
unavailable or mispriced: 8×H100 serving **Qwen3-235B-A22B FP8** (fits with 365 GB headroom) —
loses the R1 flag, keeps every primitive. Tier 4 stays documented as an optional A6 completion
(multi-node EP/DeepEP, elastic ckpt) — production frontier serving *is* multi-node, but the solo
lesson-per-dollar there is poor: you learn its *shape* from single-node numbers + the cliff.

**Cut order under budget pressure:** B200 → fold Tier 1 into the node day (it's sm_90a too) →
the node is irreducible. A single GPU cannot teach frontier serving; frontier serving is a
multi-GPU problem. (For *this* repo B200 is deferred-not-dropped: DELTA's NVFP4 headline is
sm_100a-gated.)

## Consequences

- `PERF_PLAN.md` Phase 4 is re-aimed: 8×H100 training-parallelism day → **8×H200 frontier-serving
  day** (spec, predictions P1–P7, hour plan, kill criteria in the plan). Phase 5 (multi-node)
  becomes optional. Phases 2–3 keep their kernel content; Phase 2 gains a single-GPU serving block.
- Budget: ~$700 provisional → **~$230–410 committed** across three sessions.
- Rental discipline gains two node-day rules: **on-demand only** (a preempted TP/EP bring-up burns
  the day) and **verify NVLink topology + net bandwidth before the clock matters** (`nvidia-smi
  topo -m` must show NV*, not PIX/PHB; ≥5 Gbps down + ≥1.5 TB disk for the 671 GB weight pull).
- Market rates are `[UNCERTAIN]` (Vast-class 2026: H100 ~$2–2.5/h, H200 ~$2.5–3.5/h, 8×H200
  ~$20–32/node-h, B200 ~$4–7/h) — re-check at session time; the *tiers* are the decision, prices
  only move totals.
