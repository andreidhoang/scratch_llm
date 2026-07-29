# Book Chapter Map — interleaving 3 books into the 89-Bài curriculum

> **What this is.** The single source of truth for which book chapters to read, **when**, and
> **why** — mapped against the existing curriculum. Every chapter is scored: **READ** (genuine
> gap), **SKIM** (small delta), or **SKIP** (already owned deeper in code). This file is loaded
> by the AM/PM briefings and the job-sprint drill scheduler.
>
> **Rule (FOP-1/FOP-7):** Never read a chapter whose content you've already built and measured.
> Reading owned material is recognition theater. Take the gap, move on.
>
> **Engineering-first protocol (v2, 2026-07-25):** For every "READ + BUILD" entry below, the order
> is **BUILD → FAIL → READ the specific section → FIX → TEACH BACK** — NOT "read chapter then build."
> You try to solve the engineering problem from what you know. When you hit a wall, you read ONLY
> the specific section that fills the gap. This is the same protocol as the 🔒 answer-key: you
> build first (Phase 1), compare against the book second (Phase 2). Books are ammunition for
> fixing specific failures, not prerequisites for starting.
>
> **Date:** 2026-07-25 · **Status:** analyzed against full codebase (43,730 LOC)

---

## The three books

| Book | Author | Role | Redundancy | Priority |
|---|---|---|---|---|
| **CUDA for Deep Learning** (MEAP v5) | E. Arledge | Lane-K kernel curriculum backbone (ch5–10 → A1–A6) | ~40% — ledger overtaken ch5/6/8/9/10; ch7.3+ still ahead | **DONE** (integrated as 🔒 answer keys) |
| **5D Parallelism for Large Model Training** | — | Distributed training supplement (M5+S6) | ~85% — one genuine gap (Ch04) | **LOW** (one chapter) |
| **PPPM** (Programming Massively Parallel Processors, 4th ed 2026) | Kirk/Hwu/El Hajj | Raw CUDA C++ fundamentals that Triton abstracted away | ~60% overall, **0% on k_live patterns** | **🔥 HIGH** (closes the k_live gap) |

---

## CUDA for Deep Learning (Arledge) — already integrated

PDF at `~/Desktop/JOB_SPRINT/CUDA_for_Deep_Learning_v5_MEAP.pdf`. Chapter map: A1←ch5 · A2←ch6 · A3←ch7 · A4←ch8 · A5←ch9 · A6←ch10.

| Chapter | Maps to | Status | Access rule |
|---|---|---|---|
| ch5 (Inference) | A1 (S1+S2) | ✅ Ledger overtaken | 🔒 Answer key — Phase 2 of `/rebuild` only |
| ch6 (Kernels) | A2 (S3) | ✅ Ledger overtaken | 🔒 Answer key |
| ch7.3–7.5 (WGMMA/TMA) | A3 (S4) | ✅ **Only chapter still ahead** — pre-readable | ✅ Free reading (novel Hopper mechanism) |
| ch8 (Flash Attn) | A4 (S4) | ✅ Ledger overtaken (your FA2 ~50% SDPA vs book 5%) | 🔒 Answer key |
| ch9 (Quantization) | A5 (S5) | ✅ Ledger overtaken | 🔒 Answer key |
| ch10 (Distributed) | A6 (S6/M5) | ✅ Ledger overtaken | 🔒 Answer key |
| App A.9.2 | P4.2 | ✅ Owned | ✅ Free reading |

**Hard ceiling:** Hopper only. NVFP4/MXFP4 = 0 mentions. PagedAttention/speculative = 0. Triton = 2 mentions. ch11/ch12 unwritten in v5.

---

## 5D Parallelism — distributed training supplement

~85% redundant with M5+S6 (which are built + gloo-verified). **One genuine gap.**

| Chapter | Redundancy | Verdict | Time | When to read |
|---|---|---|---|---|
| Ch00 GPU Foundations | 95% | **SKIP** | — | — |
| Ch01 Data Parallelism | 90% | **SKIP** | — | (M5 5.1 owns this deeper) |
| Ch02 ZeRO | 95% | **SKIP** | — | (M5 5.2/5.3 owns this) |
| Ch03 Tensor Parallelism | 85% | **SKIM 1 video** | 13 min | After S6 6.1 |
| **🔥 Ch04** Sequence & Context Parallelism | **0%** | **READ ALL (5 videos)** | 70 min | **Before S6 6.3, after S6 6.2** |
| Ch05 Pipeline Parallelism | 85% | **SKIM 1 video** | 14 min | After S6 6.2 |
| Ch06 Expert Parallelism | 90% | **SKIP** | — | (S6 6.3 owns this) |
| Ch07 Putting It All Together | 60% | **SKIM** | 10 min | After M5 5.5 |

### Ch03 skim — "TP vs ZeRO for Inference" (13 min)
**Why:** The bridge between M5 (train-side ZeRO) and S6 (serve-side TP). When does inference
prefer TP vs ZeRO? k-serving interview probe.
**When:** After S6 6.1 (Megatron TP), as a 13-min bridge to S6 6.2.
**Insert as:** S6 Bài 6.1b — "TP vs ZeRO for inference" (skim reading).

### 🔥 Ch04 — Sequence & Context Parallelism + Ring Attention (70 min) — THE GAP
**Why:** The 5th axis your curriculum mentions but never derives. Ring Attention = FA2's online
softmax distributed across GPUs. Every frontier model ships 128K–1M context; "how do you
train/serve long-context?" is a live probe at both NVIDIA (k-arch, k-serving) and Anthropic
(a-systems).
**What it covers:**
- Sequence Parallelism: split the activation's seq dim across ranks → cuts the s² activation term
- Context Parallelism + Ring Attention: ranks form a ring, each holds a KV chunk, passes it
  ring-wise while computing partial attention with online-softmax rescale — same
  `alpha = exp((m_old−m_new)·log2e)` traced in FA3 (S6 6.6), now distributed
- Paper: arXiv:2310.01889 (referenced in `references.md` §6 but never derived/built)
**When:** Between S6 6.2 (PP) and S6 6.3 (EP-MoE).
**Insert as:** S6 Bài 6.2b — "Sequence & Context Parallelism + Ring Attention" (new derivation node).
**PRR protocol:**
1. Predict COLD: "Ring Attention = FA2 online softmax across GPUs. Each rank holds KV[seq_chunk],
   passes ring-wise, computes partial attention with running max/rescale. Ring has N-1 steps.
   Bandwidth = (N-1)/N · seq_chunk · KV_bytes per rank."
2. Watch 5 videos (70 min).
3. Reconcile: ring passes K and V separately; computation happens *while* passing (overlap).
4. Re-derive on paper: draw ring topology for N=4, label what each rank holds/computes/passes.
5. Teach-back gate: "How would you train a model with 1M context?" → GQA shrinks KV (model),
   Ring Attention/CP shards the computation (systems).

### Ch05 skim — "Megatron 3D Parallelism" (14 min)
**Why:** You have the 2D overlapped formula (`fsdp_tp_max_world` = (3/2)·B·D_ff·(W/C)²) but not
the 3D device-mesh composition (DP×TP×PP on a physical grid).
**When:** After S6 6.2.
**Insert as:** S6 Bài 6.2c — "3D device-mesh" (skim reading).

### Ch07 skim — "Composing 5D Parallelism" (10 min)
**Why:** Integration view — ties all axes into a device-mesh mental model.
**When:** After M5 5.5 (comms algebra).
**Insert as:** M5 Bài 5.5b — "5D composition" (skim reading).

---

## PPPM (Kirk/Hwu/El Hajj, 4th ed 2026) — raw CUDA C++ fundamentals

~60% redundant overall, **but 0% on the specific k_live interview patterns.** This book fills
the gap between your Triton/ISA-tier skills and the fundamental CUDA C++ patterns NVIDIA's
k_live tests. **Your Triton kernels abstracted away exactly the skills k_live demands.**

| Chapter | Redundancy | Verdict | Time | When to read |
|---|---|---|---|---|
| Ch1-2 Intro + Heterogeneous | 95% | **SKIP** | — | — |
| Ch3 Multidimensional Grids | 50% | **SKIM** | 14 min | Before Ch10 |
| **🔥 Ch4** Compute Architecture & Scheduling | 30% | **READ** | 14 min | **Before S3 3.1** |
| **🔥 Ch5** Memory Architecture & Data Locality | 40% | **READ** | 14 min | **Before S3 3.1** |
| **🔥 Ch6** Performance Considerations | 35% | **READ** | 14 min | **Before S3 3.1** |
| Ch7 Convolution | 0% relevant | **SKIP** | — | — |
| Ch8 Stencil | 0% relevant | **SKIP** | — | — |
| Ch9 Histogram (Atomics) | 40% | **SKIM** | 14 min | After Ch10 |
| **🔥 Ch10** Reduction | **0%** | **READ + BUILD** | 14 min read + 45 min build | **URGENT — k_live Level 2** |
| **🔥 Ch11** Prefix Sum (Scan) | **0%** | **READ + BUILD** | 14 min read + 45 min build | **After Ch10** |
| Ch12 Merge | — | **SKIP** | — | — |
| Ch13 Sorting | 10% | **LOW** | 14 min | If time (MoE routing) |
| Ch14 Filtering | 0% | **LOW** | 14 min | If time (sparse attn) |
| Ch15 Sparse Matrix | 20% | **SKIP** | — | — |
| Ch16-17 Wavefront/Graph | — | **SKIP** | — | — |
| Ch18 Deep Learning | 95% | **SKIP** | — | (you're 10× deeper) |
| Ch19 Multi-GPU | 90% | **SKIP** | — | (M5/S6 own this) |
| Ch20 Electrostatic | — | **SKIP** | — | — |
| Ch21 Computational Thinking | — | **SKIP** | — | — |
| Ch22 Cluster/MPI | 85% | **SKIP** | — | — |
| Ch23 Advanced GEMM | 95% | **SKIP** | — | (you're 3 gen ahead) |
| Ch24-25 Future/Conclusion | — | **SKIP** | — | — |

### The k_live gap (the Triton Trap)

The k_live interview (spec at `JOB_SPRINT/challenges/nvidia/k_live/spec.md`) asks for raw CUDA C++:
```
Level 1: vector add (grid-stride loop)     ← never written in raw CUDA
Level 2: reduction (warp shuffle)           ← NOT IN CODEBASE
Level 3: tiled transpose (bank conflicts)   ← NOT IN CODEBASE
Level 4: tiled GEMM (coalescing + occupancy)← csrc does tensor-core level, not CUDA-core
```

**Your codebase search confirms:**
- `__shfl_down_sync` / warp reduction / block reduction: **0 matches** in src/
- Parallel scan / prefix sum / Kogge-Stone / Blelloch: **0 matches** (only `torch.cumsum`)
- Shared memory tiling for CUDA-core (not tensor-core): minimal in csrc/ (`smem_tiled.cu` = 75 LOC, naive)
- Memory coalescing patterns: implicit in Triton, never hand-derived

### Ch4 — Compute Architecture & Scheduling (14 min read) — READ
**Why:** SM architecture from the **kernel writer's** perspective (you know it from the roofline/
profiler side). Warp divergence, occupancy, latency hiding via warp scheduling.
**Interview gate:** k-arch round — "explain warps, divergence, occupancy, latency hiding."
**When:** Before S3 3.1 (Triton kernels) — as the raw CUDA foundation.
**Insert as:** S3 Bài 3.0a — "SM architecture from the kernel writer's side" (read Ch4).

### Ch5 — Memory Architecture & Data Locality (14 min read) — READ
**Why:** Shared memory tiling from first principles, register→SMEM→L2→HBM hierarchy as the kernel
writer reasons about it. You know this from Triton (auto-tiled) but not in raw CUDA C++.
**Interview gate:** k_live Level 3 (tiled transpose) — shared memory tiling + `__syncthreads()` + bank conflicts.
**When:** Before S3 3.1.
**Insert as:** S3 Bài 3.0b — "Memory hierarchy & SMEM tiling" (read Ch5).

### Ch6 — Performance Considerations (14 min read) — READ
**Why:** Global memory coalescing patterns + thread coarsening. k_live grading explicitly weights
coalescing "High."
**Interview gate:** k_live grading axes (coalescing, occupancy).
**When:** Before S3 3.1.
**Insert as:** S3 Bài 3.0c — "Coalescing & performance" (read Ch6).

### 🔥 Ch10 — Reduction (14 min read + 45 min build) — URGENT
**Why:** k_live Level 2 verbatim. `__shfl_down_sync` warp reduction + shared memory block
reduction + atomic final. **Zero raw CUDA C++ reduction code in your codebase.** This is the
single highest-risk gap for the NVIDIA interview.
**Build target:** `csrc/fundamentals/reduction_warp.cu` — warp shuffle + block reduction + atomic.
**PRR:**
1. Predict COLD: "5 steps for 32 threads (log₂32), no SMEM for intra-warp, block-level needs
   `__syncthreads` between stages, final result via `atomicAdd`."
2. Read Ch10 (14 min).
3. Reconcile divergence in the halving loop (the `if (tid < 32)` problem).
4. Build from blank in `csrc/fundamentals/reduction_warp.cu`.
5. Teach-back: "explain your memory access pattern, your occupancy, your divergence."
**Insert as:** S3 Bài 3.0d — "Warp reduction" (read Ch10 + build `reduction_warp.cu`).

### 🔥 Ch11 — Prefix Sum / Scan (14 min read + 45 min build)
**Why:** Genuinely missing skill. Scan is fundamental: MoE routing uses scatter/gather, attention
masking, stream compaction. Kogge-Stone scan is an interview classic.
**Build target:** `csrc/fundamentals/prefix_scan.cu` — Kogge-Stone intra-warp scan + block-level.
**PRR:**
1. Predict COLD: "Kogge-Stone: each thread adds offset doubling (1,2,4,8,16) — 5 steps for 32
   threads. Work-inefficient (O(n log n)) but step-optimal (O(log n)). Blelloch is work-efficient
   (O(n)) but 2× the steps."
2. Read Ch11 (14 min).
3. Reconcile the work-depth tradeoff.
4. Build from blank in `csrc/fundamentals/prefix_scan.cu`.
5. Teach-back: "when would you choose Kogge-Stone vs Blelloch?" (compute-bound vs memory-bound).
**Insert as:** S3 Bài 3.0e — "Parallel prefix sum" (read Ch11 + build `prefix_scan.cu`).

---

## Summary: the daily system reminder

When the AM/PM briefing or drill scheduler loads, it should check this map and surface:

1. **Before any k_live mock:** Have you built `csrc/fundamentals/reduction_warp.cu` and
   `prefix_scan.cu`? If not → Ch10/Ch11 are the priority drill.
2. **Before any k-arch mock:** Have you read PPPM Ch4 (SM architecture from kernel writer's side)?
3. **Before any k-serving mock:** Have you read 5D Ch03.5 "TP vs ZeRO for Inference" (13 min)?
4. **When studying S6 6.2 (PP):** Read 5D Ch04 (Sequence/Context Parallelism + Ring Attention) —
   the one genuine distributed-training gap.
5. **When studying S6 6.1 (TP):** Skim 5D Ch03 "TP vs ZeRO for Inference" + Ch05 "3D Parallelism."

### Build targets (raw CUDA C++, in priority order)

| File | Chapter | k_live Level | Priority |
|---|---|---|---|
| `csrc/fundamentals/reduction_warp.cu` | PPPM Ch10 | Level 2 | **1 (URGENT)** |
| `csrc/fundamentals/tiled_transpose.cu` | PPPM Ch5/6 | Level 3 | **2** |
| `csrc/fundamentals/prefix_scan.cu` | PPPM Ch11 | (scan classic) | **3** |
| `csrc/fundamentals/tiled_gemm.cu` | PPPM Ch6/18 | Level 4 | **4** (you have tensor-core version, need CUDA-core version) |

---

*Maintenance: this map is derived from the codebase as of 2026-07-25. If a build target is
completed, mark it ✅ and move it to MASTERY_DEBT.md as a cleared row.*
