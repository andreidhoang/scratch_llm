# A1 Rung 4.5 — MLA latent cache (toy) · spec + result

> Parent: `performance/A1_transformer_inference.md` §4.5. Oracle = full K/V reconstruction. Gate =
> **weight-absorption identity** (latent-space scores == reconstructed scores, numerically) +
> KV-size reduction vs MHA/GQA. Toy scale (no training). The 2026 KV story (ADR-0012): R1's MLA
> caches `(512+64) × 61 L × 2 B = 70.3 KB/token` vs a Llama-70B GQA-8's 327.7 KB — **4.7× smaller**,
> so a 671B model has a smaller per-token cache than a 70B.

## Mechanism + the identity

Cache a low-rank **latent** `c_KV` (dim `d_latent`) per token plus a **decoupled-RoPE** key `k_R`
(dim `d_rope`, shared across heads) — not K,V per head. Content score
`q_c · K^C = q_c · (W_UK c_KV) = (W_UK^T q_c) · c_KV`, so fold `W_UK` into the query and attend in
`d_latent` space; output `Σ a_j V_j = W_UV (Σ a_j c_KV_j)` folds `W_UV` into the output. Decode reads
only `c_KV` (+ `k_R`) — never reconstructs per-head K,V. This works ONLY because RoPE is decoupled
into the separate `d_rope` pathway: a position-dependent rotation on content-K could not be folded
into the static `W_UK` (the reason DeepSeek split it out).

## Measured (2026-07-04, `tests/test_mla.py` 5/5, `src/scratch_llm/mla.py`)

- **Weight-absorption identity `[FACT]`:** naive reconstruct-then-attend == absorbed latent-space
  attention, **max |Δ| = 1.4e-15** (float64 = machine epsilon); float32 within 1e-4. Causal (no
  future leak) in both paths.
- **KV accounting `[FACT]`** (R1-like d_latent=512, d_rope=64, 128 heads × d_head=128): MLA
  **1152 B/token** vs MHA **65536 B** (1.8% of MHA) vs GQA-8 **4096 B** (MLA is **3.56× smaller**
  per layer; the published cross-model 4.7× vs Llama-70B reflects the differing layer counts/dims).

**Verdict — SHIPPED.** The identity that underwrites MLA is proven to machine precision: an engine can
cache the latent instead of K,V at **zero quality change**, and the cache shrinks 3.6× vs GQA-8 /
~55× vs MHA-128 per layer. Bridges to A5 (FP8 on the latent) and the Phase-4 8×H200 serving day
(P5: engine-reported MLA KV/token vs the 70.3 KB analytic). Toy scope: the module demonstrates the
math on a full-sequence forward; wiring the latent into an incremental decode cache (like the R4.1
paged pool, but storing `c_KV`+`k_R`) is the serving-integration extension.
