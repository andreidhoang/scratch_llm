# Design spec — L2 FlashAttention-2 (forward kernel + roofline)

> **⚠ HARDWARE CORRECTION (2026-08-31).** Lines below that say "the standing sm120 GPU / no rental
> needed" were TRUE when written and are FALSE now: this host has **no GPU and no CUDA toolchain**
> (`nvidia-smi`/`nvcc`/`ncu`/`nsys` absent, `triton` not importable, arm64 — measured 29–30/08).
> Every rung here is **rental-gated**; measured numbers already in the ledger stay valid as records.
> Law: `CLAUDE.md` § Hardware reality · `PLAN.md` § Hardware law.

> **Status:** ✅ built & measured — oracle + Triton forward + the honest roofline (**53% of SDPA @
> seq 4k**, logged in `bench/RESULTS.md`) + the recomputation backward (D-vector, grads == SDPA
> autograd). The 4090 era ended — the standing GPU is sm120 Blackwell; re-measuring % of SDPA on
> this card is an open item in the ledger.
> **Layer:** L2 Systems (A2.1) — the headline systems artifact. **The deliverable is the roofline
> number, not the kernel** (A2 guide §6/§8): a slow-but-correct kernel + an honest "% of SDPA" is
> complete.
> **Files:** `src/scratch_llm/kernels/flash_attention.py` (pure-PyTorch oracle ✅),
> `kernels/flash_attention_triton.py` (Triton fwd, next), `tests/test_flash_attention.py` (✅) +
> `tests/test_flash_attention_triton.py` (gpu), `bench/flash_roofline.py`, this spec.

## 1. Why (the problem this solves)

Vanilla attention materializes the `N×N` score matrix → `Θ(N²)` HBM traffic and peak memory.
FlashAttention-2 tiles Q and K and runs an **online softmax**, so it never materializes `N×N`:
memory drops to `Θ(Nd)` and the kernel becomes bandwidth-efficient. Two reasons to build it here:

1. **It is the headline A2 systems artifact** — writing a fused attention kernel from scratch and
   measuring it against a hand-tuned baseline is the core A2.1 deliverable.
2. **It is the mechanistic root of train↔inference logit drift.** The training engine (our kernels
   and precision) and a serving engine (e.g. SGLang's fused attention, possibly low-bit KV) compute
   *different logits*. Building the kernel by hand is how we learn *why* that gap exists — the
   `kl_train_infer` measurement of A2.3 quantifies it.

This is the **A2 systems benchmark, not the main model's attention path** — `model.py` keeps the
plain `scaled_dot_product_attention`. Do not block the model on kernel speed.

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

### Result — prediction FALSIFIED, honest gap shipped (RTX 4090, d=64, causal, bf16, B·H=16)

| seq | Triton TFLOP/s | SDPA TFLOP/s | % of SDPA (single-config → **autotuned**) |
|---|---|---|---|
| 512  | 18.8 | 25.5  | 73.6 → **73.7 %** |
| 2048 | 44.5 | 66.9  | 58.6 → **66.6 %** |
| **4096** | **55.3** | **104.2** | 44.6 → **53.0 %** |
| 8192 | 63.4 | 130.4 | 38.9 → **48.6 %** |
| 16384| 68.5 | 148.4 | 36.1 → **46.1 %** |

**Predicted ≈65 % at seq 4k; measured 53 % (autotuned), below the 60 % kill line.** Autotune
(`triton.autotune` over BLOCK_Q/BLOCK_K ∈ {64,128}² × num_warps ∈ {4,8} × num_stages ∈ {2,3})
moved 4k from 44.6 → 53.0 %, but the gap is real and **widens with sequence length** — the Triton
kernel plateaus at ~68 TFLOP/s while SDPA's flash backend scales to ~148.

**Analysis (the value of the negative):** the bottleneck is the kernel, not the algorithm
(correctness is exact vs the oracle). PyTorch SDPA on Ada dispatches a hand-tuned flash/cuDNN
kernel with deeper software pipelining and scheduling than this single-key-loop Triton kernel.
Documented next levers (not pursued — kill-criterion + "don't let tuning eat the session"):
(1) **exp2 softmax** — scale by log₂e and use `tl.exp2` (faster hardware path) for ~10–15 %;
(2) wider autotune (num_stages ≥ 4, num_warps 16); (3) split-K / persistent-kernel scheduling.
**Shipped as-is**: a correct, autotuned, from-scratch FA2 forward at ~53 % of SDPA — a fair,
honest roofline (a well-analyzed negative is a valued artifact, A2 §6).

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
