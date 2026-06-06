# Design spec — L2 FlashAttention-2 (forward kernel + roofline)

> **Status:** oracle built (CPU, green); Triton kernel + roofline next (GPU box, rented 4090).
> **Layer:** L2 Systems (A2.1) — the headline systems artifact. **The deliverable is the roofline
> number, not the kernel** (A2 guide §6/§8): a slow-but-correct kernel + an honest "% of SDPA" is
> complete.
> **Files:** `src/reasoning_llm/kernels/flash_attention.py` (pure-PyTorch oracle ✅),
> `kernels/flash_attention_triton.py` (Triton fwd, next), `tests/test_flash_attention.py` (✅) +
> `tests/test_flash_attention_triton.py` (gpu), `bench/flash_roofline.py`, this spec.

## 1. Why (the problem this solves)

Vanilla attention materializes the `N×N` score matrix → `Θ(N²)` HBM traffic and peak memory.
FlashAttention-2 tiles Q and K and runs an **online softmax**, so it never materializes `N×N`:
memory drops to `Θ(Nd)` and the kernel becomes bandwidth-efficient. Two payoffs for this repo:

1. **Cheap rollouts = a bigger ablation grid** — every attention speedup buys more VERA runs.
2. **It is the mechanistic root of `kl_train_infer`.** The train engine (our kernels/precision) and
   the serve engine (SGLang's fused attention, possibly low-bit KV) compute *different logits*.
   Building the kernel is how we learn *why* that gap exists — the discipline pillar A2 feeds.

This is the **A2 systems benchmark, not the dense v0.1.0 policy path** — `model.py` keeps the plain
`scaled_dot_product_attention`. Do not block v0.1.0 on kernel speed.

## 2. The mechanics (Algorithm 1, forward only)

Per query tile `Q_i`, loop over key tiles `K_j,V_j` maintaining running `m` (row-max), `ℓ` (denom),
`acc` (unnormalized output):
```
S = Q_i Kⱼᵀ / √d            (causal: mask key_pos > query_pos → −∞)
m_new = max(m, rowmax(S));  P = exp(S − m_new);  corr = exp(m − m_new)
ℓ = corr·ℓ + rowsum(P);     acc = corr·acc + P Vⱼ;   m = m_new
O_i = acc / ℓ;              L_i = m + log(ℓ)
```
The oracle (`flash_attention.py`) implements exactly this in PyTorch; the Triton kernel implements
the same recurrence with a query-tile launch grid, a single key loop, fp32 accumulators, and an
`is_causal: tl.constexpr` flag (−1e6 additive mask). **Backward is out of scope** — use the
`torch.compile` recomputation path if needed; the hand-rolled Triton backward is a SKIP (A2 §4.2.3).

## 3. Correctness invariant

> The Triton kernel must equal BOTH the pure-PyTorch oracle AND `F.scaled_dot_product_attention`,
> token-for-token to fp tolerance, causal and non-causal, including ragged tiles.

Oracle invariants (CPU, `tests/test_flash_attention.py`, ✅): equals SDPA in fp64; `L` equals
`logsumexp` of the masked scaled scores; output is invariant to tile size. The Triton test
(`tests/test_flash_attention_triton.py`, gpu) does `pytest.importorskip("triton")` +
`pytestmark = pytest.mark.gpu` so it skips on the Mac and runs on the box; it asserts against the
oracle on CUDA (bf16 + fp32).

## 4. Predict-before-run — the roofline (A2.1)

**Falsifiable prediction (write the number first; it is the debugging anchor):**

> On the rented **RTX 4090** (Ada, 24 GB, ~1 TB/s HBM, ~165 fp16 TFLOP/s dense), the from-scratch
> **Triton FA2 forward reaches ≈ 65 % of `F.scaled_dot_product_attention` throughput** at
> **seq = 4096, d = 64, causal, bf16** (B·H ≈ 16). Rationale: SDPA on Ada dispatches a hand-tuned
> fused (mem-efficient/flash) kernel; a clean first-pass Triton kernel with fp32 accumulators
> typically lands in the 55–75 % band before block-size / num-warps tuning.

**Kill criterion (A2 §6):** if FA2-Triton < **60 %** of SDPA at seq 4k, do **not** rabbit-hole on
tuning — ship the roofline plot + the honest gap (a well-analyzed negative is a valued artifact).

**Method:** `bench/flash_roofline.py` uses `triton.testing.do_bench`, sweeps seq ∈ {512…16384},
records TFLOP/s and % of SDPA, and emits the roofline (arithmetic intensity vs % peak, BW- vs
compute-bound boundary). Report the GPU honestly (4090, not B200).

## 5. Scope / deferred

- **In:** pure-PyTorch oracle (✅), Triton FA2 **forward** + causal flag, the roofline doc.
- **Deferred / SKIP:** Triton backward (use torch.compile recompute), the 8B leaderboard, FA3/FA4
  (Hopper/Blackwell-only — 4090 honestly targets FA2).

## 6. Build sequence

1. Pure-PyTorch oracle + CPU tests ✅ (done, green).
2. *(on the box)* Triton fwd kernel (Algorithm 1) → pass vs oracle, non-causal.
3. `is_causal` flag (−1e6 mask) → pass vs oracle, causal.
4. `bench/flash_roofline.py` + `do_bench` → roofline; compare to the §4 prediction.
5. Update `docs/STATUS.md`; commit; write the roofline artifact into this spec.
