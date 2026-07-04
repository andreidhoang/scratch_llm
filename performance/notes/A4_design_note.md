# A4 Design Note — The marquee fused kernel

> A4 flash attention, sm120 (0.551 TB/s, 72 TF/s bf16). Reuses the CS336-A2 FA2 (`flash_attention.py`
> oracle + `FlashAttentionPyTorch` autograd; `flash_attention_triton.py` forward), adds the pedagogical
> R0/R1 rungs + the sm120 measurement. All `[FACT]` in `bench/RESULTS.md`; 41 tests.

## 1. Why flash attention exists (R0, measured)

Naive attention materializes the full `N×N` score matrix `S = QKᵀ` in HBM: at N=16K that is **1.0 GiB**,
**65536×** the on-chip working set a fused kernel needs. The memory is O(N²) and it OOMs. The honest,
reproducible statement (the absolute peak-MB is allocator-state-dependent) is that the naive kernel's
peak-memory **grows** cleanly as N² (measured increment ratios ×3.96/×3.98 per doubling), while a fused
kernel's grows ~143× less. That growth curve is the entire motivation: attention's *compute* is O(N²·d)
either way, but its *memory* can be O(N) if you never write S.

## 2. The mechanism (R1, measured to fp64)

Flash attention fuses softmax into the QKᵀ→PV tile loop using the **online-softmax recurrence**
(Milakov–Gimelshein): stream the score tiles, keep a running max `m` and denominator `l`, and rescale
the partial output by `exp(m_old − m_new)` when a new tile raises the max. Built in isolation
(`online_softmax.py`), it matches a 3-pass reference to **0–7e-18** (machine fp64) — including a +50
outlier arriving late in the stream (the case that tests the running-max rescale) and a ×1000
overflow-bait. This is the numerical core that lets FA never materialize S: it computes the exact
softmax-weighted sum in one pass over constant SRAM.

## 3. The kernel on sm120 (R2/R3, measured)

The fused FA2 Triton forward runs at **~50% of `F.scaled_dot_product_attention`** at seq 4096 (causal
48.3%) — pinning the repo's prior 53%-on-a-4090 number to this consumer-Blackwell card. It uses
**constant SMEM**, so it is **44× leaner in peak memory than the naive kernel at 8K and does not OOM**;
causal masking (skip the upper-triangle tiles) gives a **1.11→1.74×** speedup as N grows. The
recomputation backward (`FlashAttentionPyTorch`: save Q,K,V,O,L; recompute S,P in the backward;
softmax-Jacobian via the D-vector) passes a **gradcheck 5/5 vs SDPA autograd**. The GQA variant
(`n_kv_heads` 4 vs 32) shrinks the KV cache **8× (32→4 MB)** with flat runtime — the A1 serving
connection (paged R4.1 and MLA R4.5 are the other two variants).

## 4. Honest gap to the ceiling

FA2-on-consumer-Blackwell at ~50% of SDPA is the honest sm120 number — SDPA itself is a tuned
cuDNN/flash kernel, so 50% means our hand-Triton FA2 leaves ~2× on the table (fewer non-matmul FLOPs,
better tile scheduling, warp specialization). The real ceiling is the **FA3-class Hopper kernel**
(warp-spec producer/consumer + TMA + ping-pong + FP8, ~75% util / ~740 TF/s) — the marquee A4 §4 rung,
H100-gated (`H100_day_runbook.md` §4). The lesson: attention is where A2 (GEMM tiling) and A3 (tensor
cores) fuse into one kernel; the FA win is **memory (never write S), not FLOPs** — the same
memory-first theme as the A1 serving track.
