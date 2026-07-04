# A1 Design Note — Where does inference time and memory go, and what did I do about it?

> Assignment A1 (Transformer Inference as a System), standing GPU RTX PRO 4000 Blackwell (sm120,
> 24 GB, 0.55 TB/s HBM, 72 TF/s bf16, ridge ≈130 FLOP/byte). Model = `RUNG1_CONFIG` (~0.84 B bf16,
> GQA-4). Every number is `[FACT]`-ledgered in `bench/RESULTS.md`; written for a frontier-lab peer.

## 1. The roofline argument (the spine everything hangs off)

Autoregressive **decode** reads the entire weight matrix to produce **one** token: arithmetic
intensity **AI ≈ 1 FLOP/byte**, ~130× below the sm120 ridge — decode is **memory-bandwidth-bound**,
full stop. The ceiling is `HBM_BW / weight_bytes`: `0.55 TB/s ÷ 1.68 GB ≈ 327 tok/s` for this model.
Everything in A1 is a lever to move measured decode toward that wall, or to raise AI back up so the
idle tensor cores get used. **Prefill** is the opposite — a GEMM over the whole prompt, compute-bound —
which is why mixing the two (§4.2/§4.6) is a scheduling problem, not a kernel one.

Proven three ways (R1): hand roofline (AI≈1), and the measured climb as launch overhead is stripped.

## 2. Where the time actually went (R0–R2)

- **R1 — the memory wall is hidden by launch overhead, not absent.** Eager B=1 decode measured
  **51 tok/s = 16% of the 327 wall** (89 GB/s of 550), bound = **overhead** (~955 kernel launches +
  a `.item()` sync per token), NOT memory. `torch.compile` fuses the pointwise ops → **173 tok/s =
  53% of the wall** (291 GB/s). The residual 47% is per-token launch dispatch — the R4.4 target.
- **R2 — GQA/MQA shrink the KV read.** KV/token 128/32/4 KB (MHA/GQA-4/MQA); compiled MQA vs MHA
  **1.92× at 16 K context** — the KV cache, not the weights, dominates the byte bill at long context.

## 3. What I did about it (R3–R4.6, each with the before/after)

| Rung | Lever | Result `[FACT]` |
|---|---|---|
| **R3a static batch** | share one weight read across B requests (GEMV→GEMM, AI≈B) | agg **66× B=1** @ B=256; memory→compute crossover at B≈128 |
| **R3b continuous batch** | refill freed slots every step (Orca), not per-wave | **2.30× wall / 2.93× steps** vs static-wave on a heavy-tail trace; TTFT p95 **4.9×** better. Cost: ITL 5.4→9.6 ms; a **1.27× dense-padding tax** (the R4.1 target) |
| **R4.1 PagedAttention** | KV as paged virtual memory + a fused Triton decode kernel | frag **5.0%**, capacity **×9.3**; kernel **5.90 ms/step = ×3.52 vs wave**, the whole padding tax reclaimed |
| **R4.2 chunked prefill** | slice a long prefill so it doesn't head-of-line-block decode | mechanism token-exact; **but the sequential-interleave scheduler REGRESSED** (ITL p50 ×6.6–14.4, agg 547→145) — diagnosed: serialized admission + un-piggybacked chunk cost. The win needs a **fused ragged prefill+decode kernel** (Sarathi), scoped R4.2b. *The honest negative.* |
| **R4.3 speculative decode** | verify K drafts in one target forward (raise AI from the other side) | lossless (token-exact); **×1.2–1.4 wall / 1.3–1.5 tok/forward** (zero-cost n-gram). Acceptance tracks the *model's* output entropy, not the prompt |
| **R4.4 CUDA-graph decode** | one `cudaGraphLaunch` instead of ~54–68 kernel launches | B=1 **−74.3% step-time**, **253 tok/s = 77% of the wall** — the R1 gap CLOSED (eager 20% → compiled 53% → **graph 77%**) |
| **R4.5 MLA (toy)** | cache a low-rank latent instead of per-head K,V | weight-absorption identity to **machine eps (1.4e-15)**; KV **3.56× < GQA-8** per layer |
| **R4.6 PD-disagg (demo)** | run prefill off the decode worker's stream | decode-worker ITL p99 **3× better** (20.2 vs 61.5 ms) for a **0.18 ms** one-time KV transfer; goodput@SLO wins |

**The headline arc:** B=1 decode went **16% → 53% → 77%** of the memory wall as I removed overhead
(fusion, then CUDA graphs); batching + paging turned the idle tensor cores on (66× aggregate);
speculation, MLA, and disaggregation attack the remaining axes (AI, KV bytes, scheduling).

## 4. The honest gap to a production engine (vLLM/SGLang)

What this stack does NOT yet have, and roughly what each costs:
- **Chunked-prefill piggyback (R4.2b):** the biggest hole — my chunked prefill regresses because it
  runs prefill as a *separate sequential forward*; vLLM-V1 fuses the chunk into the decode batch
  (stall-free batching). Needs a mixed-query-length kernel (extend the R4.1 paged kernel).
- **CUDA-graph + continuous batching together:** R4.4 captures a *fixed* batch; production keeps a
  graph pool per batch size and re-captures on admission churn. Mechanical given the fixed `(B,H)`
  grid, not built.
- **Prefix caching / RadixAttention:** the paged pool supports block sharing (R4.1 CoW) but there is
  no prefix-match scheduler on top.
- **FP8/INT4 KV (A5):** halve the KV byte bill; MLA latent + FP8 compounds (bridged, not wired).
- **A real 2-GPU disaggregation** with NVLink KV transfer (Phase-4 P6).

Estimated standing: on B=1 latency this stack is now **within ~1.3× of a tuned engine** (77% of the
wall vs their ~85–90%); on *throughput/goodput* the gap is larger (~1.5–2×) and lives entirely in
scheduling (chunked-prefill piggyback + prefix caching), exactly as the roofline predicts — the win
in inference is in **memory and scheduling, not the matmul**.

## 5. Reusable artifacts

`serving/` (metrics · baseline · batched · continuous · speculative · cudagraph) · `model.py`
KV-cache family (KVCache/SlotKVCache/BatchedKVCache/PagedKVCache/PrefillView/ChunkPrefillView) ·
`mla.py` · `kernels/paged_decode_triton.py` · `bench/` (roofline · continuous · chunked_prefill ·
speculative · cudagraph_decode · disagg) — the substrate A2–A7 measure against.
