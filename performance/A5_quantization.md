# A5 — Quantization: Precision as a Design Variable

> **Book chapter:** 9 (Quantization) · **Frontier thesis:** Precision is not a fixed property of a model — it is a *per-tensor, per-region design variable* you choose under an accuracy budget, and the only honest way to choose it is to **measure the error, never hope it's fine**. The book builds INT8/INT4 quant/dequant kernels at rising granularity; the frontier moved to **sub-byte float with block scaling** (MXFP4, NVFP4) and **outlier-aware PTQ** (SmoothQuant, AWQ, GPTQ, QuaRot). You will build the ladder INT8→INT4→NVFP4→FP8-KV, validating every rung with **SQNR/MSE**, and prove the one result that defines 2026 inference: NVFP4's two-level scaling beats MXFP4 for a *specific, derivable* reason.
>
> **Primary hardware:** RTX 4090/5090 for the INT8/INT4 rungs (R0–R2) and all the numerics work. NVFP4/MXFP4 tensor-core MMA (R3 frontier path) and FP8 WGMMA (R4, DeepGEMM) want **datacenter Blackwell (B200, sm_100)** and **Hopper (H100, sm_90)** respectively — see the hard gates in `00_foundations.md`. The *math and pack/unpack* of every format is checkable on a 4090; only the native MMA throughput needs the frontier part. · **Est. time:** 1.5–2 weeks · **Prereqs:** book ch1–4 and ch9; the benchmarking discipline in `00_foundations.md`; **A3** (tensor-core block scaling — NVFP4/MXFP4 GEMM *is* a block-scaled MMA) and the FP8 context from **A1** (FP8 KV cache).

---

## §0 Why this matters

A frontier model is deployed at a precision someone *chose*, and that choice is worth more money than almost any kernel rewrite. Going FP16→FP8 halves the weight bytes and roughly doubles tensor-core throughput; FP8→NVFP4 halves them again (3.5× smaller than FP16, 1.8× than FP8) and, on Blackwell, runs at ~2× the FP8 element rate. The DeepSeek-R1 FP8→NVFP4 PTQ result — **≈0.1% MMLU drop, within ~1% across seven benchmarks** — means a lab can serve a frontier model on *half the FP8 memory and ~2.3× the throughput* with accuracy loss in the noise. That is not an optimization; it is a change in unit economics.

The book's chapter 9 teaches the mechanism honestly and stops, by design, at INT8/INT4: build the quant/dequant kernels, push granularity finer (tensor → channel → group → block), calibrate, and *validate against a reference*. That last verb is the whole discipline. The chapter's most important sentence is the uncomfortable one on calibration: **"bad calibration can destroy your model worse than aggressive quantization ever could"** — 8-bit with a bad scale loses 20%, 4-bit with a good scale loses 2%. The error is not in the bit-width; it is in the *scale you chose and never measured*.

This assignment carries that discipline into the formats that actually run in 2026. The spine is one sentence: **precision is a design variable; measure, don't hope.** The universal instrument is **SQNR/MSE** — you compute it on every rung, and it is the difference between "I quantized it" and "I quantized it and can prove the error is where I claimed."

---

## §1 Learning objectives

You can:

1. **Lay out from memory** the bit fields of every format you'll touch — FP8 E4M3/E5M2, FP6 E3M2/E2M3, FP4 E2M1, the E8M0 and E4M3 *scale* types — and explain *why* E4M3 reclaims its top exponent for finite values while E5M2 keeps Inf, and why the FP4 code set is unusable without a block scale.
2. **Derive** the SQNR floor `≈ 6.02·b + c` dB and use it as the pass/fail oracle for every quant kernel: a clean symmetric round-trip *must* hit the floor, and a deficit is a bug (bad scale, missing clamp, wrong rounding).
3. **Build** quant/dequant kernels at rising granularity (per-tensor → per-channel/per-token → group-wise INT4 → block-wise), pack INT4 two-per-byte and **NVFP4 4-bit with a 16-element E4M3 block scale**, and verify each pack/unpack is *bit-exact* on round-trip.
4. **Implement and justify the NVFP4 two-level scaling math** — per-tensor FP32 global scalar + per-16-block FP8 block scale — and **measure** that its block MSE is lower than an MXFP4 (k=32, E8M0 power-of-two scale) variant on the same data, attributing the win to the two named mechanisms (E4M3 non-power-of-two scale + 16-element blocks).
5. **Reason about granularity vs bit-width** as a rule, not a vibe: *lower bit-width demands finer granularity*. FP8 tolerates per-tensor/per-token; FP4 **demands** block-wise. State which scheme each tensor class needs and why.
6. **Choose a scaling-recipe** — static (calibrated) vs dynamic (current-tensor amax) vs delayed (amax history) — and name the failure mode of each (delayed scaling's stale-scale overflow under distribution shift is the one that bit TransformerEngine).
7. **Apply a real PTQ method** (AWQ, GPTQ, or QuaRot) to an actual linear layer and report the measured accuracy recovery vs naive round-to-nearest, explaining the outlier mechanism each attacks.
8. **Build an FP8 E4M3 KV cache** (per-channel-K / per-token-V) and show it's near-lossless at half the bytes — connecting back to A1 — then stress sub-4-bit KV to see it break and (stretch) recover it with a Hadamard rotation.

---

## §2 First-principles theory

This section is the sourced reference you'll keep open while building. The book covers §2.1–§2.3 (data types, the quant/dequant operation, granularity, calibration, static/dynamic, sym/asym, AWQ); §2.4–§2.8 are the frontier additions.

### 2.1 The operation, and the noise model that justifies it

Quantization is three operations: **divide, round, clamp**; dequantization is one: **multiply** (book fig 9.5).
```
quantize:   q = clamp( round(x / s) [+ z], qmin, qmax )      # affine if z present
dequantize: x̂ = (q [- z]) · s
```
Everything is in the scale `s`. Too small → values saturate at `qmax`, you lose the tails. Too large → values bunch into a narrow band, you waste codes and effectively quantize to fewer bits. The book's diagnostic is the one to internalize: **clipping at the rail means s is too small; values bunched in the middle means s is too large.**

The reason any of this works is the **quantization-noise model**: rounding to a grid of step `Δ = s` injects error uniform on `[−Δ/2, +Δ/2]`, variance `Δ²/12`. Halving the step quarters the noise power (the book's "double the step → 4× the error"). Hold a *b*-bit signal across a range `R`: `Δ = R/2^b`, noise power `∝ 2^{−2b}`, so signal-to-quantization-noise ratio in dB is

> **SQNR ≈ 6.02·b + c (dB)** — each extra bit buys ~6 dB.

The constant `c` depends on the signal's distribution relative to the range (loading factor); for a well-scaled tensor it's a small positive offset. This is your **universal oracle**: an ideal INT8 symmetric round-trip on a clean tensor lands near ~44 dB (≈ 6.02·7 + c, since symmetric INT8 uses 7 bits of magnitude). If your kernel is materially below the floor, you have a bug — not "acceptable loss." [HONESTY FLAG: the exact `c` and thus the exact dB target depends on the tensor and on whether you count sign/magnitude bits; treat "~44 dB for clean INT8 sym" as a *calibrated expectation to reproduce*, not a spec constant. The *deficit-means-bug* logic is what's load-bearing.]

Why integer quantization is so flexible (book §9.1.1, fig 9.4): an affine integer map can **slide its zero-point and scale** to match the data, while a float format's exponent/significand split is **fixed** — FP8 cannot move its dynamic range to fit your tensor. That is exactly why floats need a *separate* scale (block scaling) to be competitive at low bit-widths, which is the whole frontier story below.

### 2.2 Granularity — the central design axis (coarse → fine)

One scale for how much data? The book's figures 9.6–9.9 walk tensor → channel → group → block; the frontier ranks them by what they cost and what they're for:

| Granularity | Scale shared over | Cost | Where it's used |
|---|---|---|---|
| **Per-tensor** | the whole tensor | cheapest (1 scalar) | FP8 weights/acts; one outlier wrecks all |
| **Per-channel** | each output feature (reduction axis) | 1 scale/col | **weights** — factors *out* of the GEMM (scale is constant along the contraction) |
| **Per-token** | each row/token | 1 scale/row, dynamic | **activations** — computed on the fly |
| **Group-wise** | g consecutive weights (g=64/128) | 1 scale/group | **the INT4 workhorse**; GPTQ/AWQ default **g=128** |
| **Block/tile-wise** | a 2-D tile or a k-element block | 1 scale/block | DeepSeek **128×128 weights / 1×128 acts**; **NVFP4 k=16**, **MXFP4 k=32** |

**The rule that ties bit-width to granularity:** *the fewer the bits, the finer the granularity you must use.* FP8 (E4M3, ±448 dynamic range) tolerates per-tensor or per-token. FP4 (8 codes of magnitude) has so little range that a single shared scale cannot serve a tensor with any outlier — **FP4 demands block-wise scaling**, and that is precisely what MX/NVFP4 are. Per-channel-on-weights has a structural bonus the book underplays: because the weight scale is constant along the *contraction* axis, it **factors out of the matmul** — you dequant the INT accumulator once at the end, not per-element. Per-token-on-activations is its natural partner.

### 2.3 Scaling recipes: when do you compute the scale?

- **Static (calibrated):** run a few hundred–thousand *representative* examples offline, bake fixed scales into the checkpoint (book listing 9.11 `StaticQuantParams`). Fast at inference (no reductions), but it's a *bet* that production data matches calibration data. Use percentile (95–99.9th) not min-max to be outlier-robust (book §9.2.2: one outlier of 100 among values in [0,5] forces a scale that crushes the real signal). **Always validate on held-out data — calibrating and measuring error on the same set is the cardinal sin** (book repeats this three times; so do I).
- **Dynamic / current scaling:** compute `amax` of *this* tensor *this* pass, then quantize (book listing 9.12 — a shared-memory reduction then a quantize pass). Most accurate, adapts to every input; costs an extra reduction/latency. The book's honest note: it is *never free*.
- **Delayed scaling (TransformerEngine):** keep an `amax` **history** and set this step's scale from *past* amaxes, so you skip the extra reduction pass and fuse scale into the cast. **Failure mode:** distribution shift / a sudden outlier → the history-derived scale is stale → under- or over-flow. This bit real training runs; TE 2.x re-added **current scaling** as the safer default. The lesson is general: a scale derived from the past silently breaks when the present changes.

### 2.4 Float formats — the bit layouts (sourced)

Notation `ExMy` = 1 sign + x exponent + y mantissa bits; stored value (normal) `= (−1)^s · 2^{E−bias} · (1 + m/2^y)`; subnormals `= (−1)^s · 2^{1−bias} · (m/2^y)`.

| Format | s-x-y | bias | Max finite | Inf/NaN | Role |
|---|---|---|---|---|---|
| **FP8 E4M3** | 1-4-3 | 7 | **±448** | **NaN only (no Inf)** | forward weights/activations |
| **FP8 E5M2** | 1-5-2 | 15 | **±57344** | Inf + NaN | backward **gradients** |
| FP6 E3M2 | 1-3-2 | 3 | ±28 | none | MX FP6 option |
| **FP6 E2M3** | 1-2-3 | 1 | ±7.5 | none | **preferred for MXFP6** (extra mantissa bit) |
| **FP4 E2M1** | 1-2-1 | 1 | ±6 | none | the FP4 element; needs block scale |

Two facts to carry:
- **E4M3 reclaims its top exponent for finite values** (it represents NaN with a single mantissa-bit pattern and has *no Inf*), buying max ±448 — the right call for forward tensors that are **bounded and want mantissa precision**. **E5M2 keeps Inf/NaN** and trades a mantissa bit for an exponent bit → max ±57344, the right call for **gradients, which are heavy-tailed and want dynamic range**. Mnemonic: *forward bounded → mantissa (E4M3); gradients heavy-tailed → exponent (E5M2).* (OCP FP8, arXiv:2209.05433.)
- **FP4 E2M1's entire representable magnitude set is `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`** (with sign: 16 codes total: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6). Eight positive magnitudes. A tensor quantized to these *without a scale* is garbage — there is no way to represent 12.7 or 0.03. The scale is not an optimization; it is what makes the format usable.

[HONESTY FLAG: exact subnormal boundaries and the smallest normal/subnormal of each format follow the formulas above, but **verify them against the OCP spec PDF before quoting a specific smallest-subnormal number** — they are easy to get off-by-a-power-of-two. The `~` in any subnormal boundary I state is unchecked unless I say "checked vs spec."]

### 2.5 Microscaling (MX) block formats — OCP MX v1.0

An **MX tensor** = a vector of low-precision elements + **one shared scale per block of `k` elements** (OCP Microscaling Formats v1.0, arXiv:2310.10537). All MX v1.0 formats fix:
- **block size k = 32**;
- **scale type E8M0** — an **8-bit unsigned, biased-127 exponent, NO mantissa**, representing powers of two `2^{−127 … 127}` (plus a NaN code). Because the scale has no mantissa, you must **round the block's amax to the nearest power of two** — which can be off by up to ~2× from the true amax. That rounding is the structural weakness of MX.

`MXFP4 = (k=32, element=E2M1, scale=E8M0)` → storage `= 4 + 8/32 = 4.25` bits/value. MXFP6/MXFP8 analogous with E2M3/E4M3 elements.

### 2.6 NVFP4 — two-level scaling (the math, this is the frontier core)

NVFP4 (NVIDIA) is more accurate than MXFP4 and differs in exactly **three** ways:
1. **Block size 16, not 32** — finer blocks, fewer elements share a scale, so a single outlier contaminates less.
2. **Block scale is FP8 E4M3, not E8M0** — *3 mantissa bits* means the block scale is **not constrained to a power of two**, so it can sit much closer to the true block amax (no up-to-2× rounding error).
3. **A second, per-tensor FP32 global scalar** on top of the per-block scale — "two-level."

**The two-level math** (the thing to implement and to be able to derive at a whiteboard):

```
Given a tensor X with amax  A = max_i |x_i|.
Global (per-tensor) scale:        s_global ≈ A / (448 × 6)
    # 448 = max FP8-E4M3 (the block-scale's own range); 6 = max |E2M1| code.
    # This maps the largest possible (block scale × element) product back to A.

For each 16-element block b with amax  a_b = max_{i∈b} |x_i|:
    block scale (stored as FP8 E4M3):
        s_b = quantize_e4m3( a_b / 6 / s_global )
    quantized element (stored as E2M1 code, q_i ∈ {0, ±0.5, …, ±6}):
        q_i = round_to_E2M1( x_i / (s_global · s_b) )

Dequantize:
        x̂_i ≈ s_global · s_b · q_i
```

Read the structure: the **per-tensor FP32 `s_global`** absorbs the *global* dynamic range so the FP8 block scales operate in a normalized band (where E4M3's 3 mantissa bits resolve them finely); the **per-block FP8 `s_b`** then tracks *local* amax with near-continuous resolution; the **E2M1 element** is the final 4-bit code. Storage `= 4 bits (element) + 8/16 (block scale) = 4.5 bits/value`, plus **one FP32 scalar per tensor**.

[HONESTY FLAG: **the exact placement of the constants — the `/6`, the `/448`, whether `s_global` is folded before or after the block-scale quantization, and whether the global scalar is computed from the raw amax or from the post-block-scale reconstruction — varies by implementation** (NVIDIA TensorRT-Model-Optimizer vs TransformerEngine differ in details, and reference kernels evolve). The *invariant* you must preserve is `x̂_i = s_global · s_b · q_i` and that `s_global` normalizes the FP8 block scales into their high-resolution band. **Pin your formula to one reference implementation and cite it**; do not assume two libraries place the constant identically.]

**Why it wins (the derivable reason — this is §4's required result):** MXFP4's E8M0 scale must round each k=32 block's amax to a power of two (≤2× error) over a *coarser* block; NVFP4's E4M3 scale lands near-exactly on a *finer* k=16 block's amax. So NVFP4's quantization grid hugs the data far more tightly → lower per-block MSE → **<1% accuracy degradation vs FP8 where MXFP4 shows visibly larger gaps**. (NVIDIA "Introducing NVFP4" blog; arXiv:2509.23202.)

[HONESTY FLAG on the headline number: NVIDIA's marquee figure is the **hardware ratio — ~5× dense PFLOPS NVFP4 vs FP8 on a Blackwell part** (e.g. ~10 PF NVFP4 vs ~2 PF FP8 on an H200-class comparison; FP4 element throughput ≈2× FP8 on Blackwell). The often-quoted **"~2.3× end-to-end throughput vs FP8" is a secondary, workload-dependent measurement**, and *real e2e speedups are lower than the raw hardware ratio* because of memory traffic, scale handling, and non-MMA work. Quote the 5× as a *hardware* number and the 2.3× as a *measured-e2e* number, and never conflate them.]

### 2.7 Outlier-aware PTQ — the four mechanisms

Low-bit quantization fails on **outliers** (a few large-magnitude activation channels in LLMs). Four ways to handle them:

- **SmoothQuant** (arXiv:2211.10438): activations have the outliers, weights are smooth — so **migrate** difficulty from activations into weights. With `Y = (X · diag(s)^{−1}) · (diag(s) · W)` (mathematically identical), choose per-channel `s_j = max|X_j|^α / max|W_j|^{1−α}`, `α≈0.5`. Result: a clean **W8A8** with both sides easy to quantize per-tensor/token.
- **AWQ** (arXiv:2306.00978) — *the book's §9.3 method*: weight-only **W4A16**. Not all weights matter equally; the **~1% salient channels** (identified by **activation** magnitude, not weight magnitude) dominate output error. **Scale those channels up before rounding** (grid-searched scale) so they land on a finer effective grid, and fold the inverse scale into the next op (book listing 9.15: `w' = w/√|a|`, `a' = a·√|a|`, product invariant). No backprop, no Hessian → fast. The book's intuition (§9.3.2): a weight ×25.0 activation matters; the same weight ×0.1 doesn't — protect the former.
- **GPTQ** (arXiv:2210.17323): layer-wise, rooted in **OBQ/OBS**. Quantize **column by column**; after fixing column `q`, **update the remaining columns to compensate** for the error using the Hessian `H = 2XX^T`: `δ = −(w_q − \hat w_q)/[H^{−1}]_{qq} · H^{−1}_{:,q}`. Cholesky for stability. **Best low-bit accuracy, slowest to produce** (book §9.5.1: error compensation, "balancing a seesaw").
- **QuaRot** (arXiv:2404.00456): multiply activations by an orthogonal **randomized Hadamard** `Q`, weights by `Q^T`; since `XQ · Q^T W = XW`, the layer is **invariant** but the rotated activations are **near-Gaussian and outlier-free** → enables end-to-end **W4A4 + KV4**. **Computational invariance** lets you *fuse most rotations into the weights* offline, leaving only ~½ a Hadamard online (Hadamard is `O(n log n)`). Reported up to **2.16× prefill, 3.39× KV savings**. **SpinQuant** (arXiv:2405.16406) replaces the random Hadamard with a **learned** rotation (optimized on the Stiefel manifold via Cayley), beating random by up to ~16 points where Hadamard alone underperforms.

### 2.8 FP8 KV cache and FP8/NVFP4 training (frontier context)

- **FP8 (E4M3) KV cache** is the production default — near-lossless, **2× capacity** (connects to A1 §2.2). The **asymmetry** to know: the **Key cache wants per-channel quant, the Value cache per-token** — per-channel-K gives ~2.5× smaller error than per-token-K because key outliers are channel-structured (KVTuner, arXiv:2502.04420). Sub-4-bit KV needs explicit outlier handling (KVQuant) or rotation (QuaRot KV4).
- **FP8 training** is real: **FP8-LM** (arXiv:2310.18313) — FP8 gradients → optimizer → comms with *precision decoupling* + auto scaling; GPT-175B on H100: **−39% memory, +75% throughput** vs BF16 Megatron. **DeepSeek-V3** (arXiv:2412.19437) is the reference recipe: fine-grained **1×128 tile (acts) + 128×128 block (weights)** dynamic scaling, and **two-level accumulation** — because Hopper FP8 WGMMA accumulates at only ~**14-bit ("FP22")** mantissa precision, partials are **promoted to FP32 on CUDA cores every 4th WGMMA (every 128 K-elements)** to stop error accumulation; master weights/optimizer/embeddings/norm/softmax/router stay BF16/FP32; **<0.25% loss vs BF16**; open-sourced as **DeepGEMM** (the kernel A2 studies). **NVFP4 *training*** is now demonstrated (NVIDIA, arXiv:2509.25149): a 12B Mamba-Transformer trained on **10T tokens** — the longest public 4-bit run — reaching **MMLU-pro 62.58 (NVFP4) vs 62.62 (FP8)**, the last gap closed by switching to BF16 for the final ~1.8T tokens. **QAT** uses **fake-quantize** nodes (quant→dequant in float so the forward sees the error) with the **straight-through estimator** (`round()` ≈ identity in backward), delaying fake-quant the first N steps; LLM-QAT does it data-free via logit distillation. By 2026, QAT/QAD is the standard **NVFP4 accuracy-recovery** step (arXiv:2601.20088).

### 2.9 The kernel idea: dequant fuses into the GEMM mainloop

The performance unlock for all weight-quantized formats: **keep weights low-bit in DRAM/L2, and dequantize in registers inside the GEMM mainloop, just before the MMA** — so the upconvert is hidden under tensor-core compute and you never pay the DRAM bandwidth of FP16 weights. This is why W4A16 is *faster*, not just smaller, at low batch. The reference kernels:
- **Marlin** (arXiv:2408.11743): FP16×INT4 **W4A16** on Ampere/Ada, ~**4× up to batch 16–32**, async-pipelined, GPTQ weight layout.
- **Machete**: the Hopper successor on **CUTLASS 3.x** (TMA + warpgroup), INT4/FP4.
- **DeepGEMM**: FP8 fine-grained, Hopper, the **two-level accumulation** above, up to **~1358 TFLOPS**, 1.1–2.7× over a CUTLASS baseline.

This is the bridge to **A2 (DeepGEMM)** and **A3 (block-scaled tensor-core MMA)**: an NVFP4/MXFP4 GEMM *is* a block-scaled MMA whose mainloop reads E2M1 codes + block scales and applies the scale at accumulate time. You already built the scaffolding in A3; here you feed it quantized data and prove the numerics.

---

## §3 The from-scratch build ladder

Validate **every rung with SQNR and MSE** against the FP32/BF16 reference *before* promoting — that is the non-negotiable discipline (§5). Keep the engineering journal per `00_foundations.md`. The book's INT8/INT4 kernels (listings 9.1–9.14) are your starting code; you extend them to the frontier formats.

**R0 — INT8 symmetric, per-tensor.** `s = max|x| / 127`; `q = round(x/s).clamp(−127, 127)`; `x̂ = q·s` (book listings 9.1–9.2). *Mechanism:* the simplest possible quantizer — divide, round, clamp, multiply. *Correctness:* round-trip **MSE** small and **SQNR ≈ the ~44 dB floor** (`6.02·7 + c`) on a clean (well-scaled, outlier-free) tensor; **no saturation** (nothing pinned at ±127 except the true amax). If you're below the floor on clean data, your scale or rounding is wrong. *Target:* reproduce the book's worked example errors `[1.2,−0.8,2.5,−1.7] → x̂` with two values exact and two off by 0.01.

**R1 — INT8 asymmetric (affine) + per-channel/per-token.** Add the zero-point: `s = (max−min)/255`, `z = round(−min/s)`, `q = round(x/s)+z` clamped to `[0,255]` (book listings 9.13–9.14). Then make it **per-channel for weights / per-token for activations** (book listings 9.5–9.8). *Mechanism:* asymmetric range placement recovers the half-range that symmetric wastes on one-sided (post-ReLU/GELU) data; per-channel/token adapts to heterogeneous feature scales. *Correctness:* on a **skewed post-GELU tensor**, asymmetric SQNR **beats** symmetric (show the gap); per-channel SQNR beats per-tensor by the book's 2–3×; a full **W8A8 linear** matches FP32 within ~1e-2 relative. *Target:* state the sym-vs-asym and per-tensor-vs-per-channel error deltas with numbers, and explain *why* per-channel weight scale factors out of the GEMM.

**R2 — Group-wise INT4 (g=128), packed.** Pack two nibbles per byte (book listings 9.3–9.4); one scale per **128** consecutive weights. *Mechanism:* the INT4 LLM workhorse — group granularity captures local statistics at <1% scale overhead; packing is bit manipulation (`(q1&0xF)<<4 | (q2&0xF)`, sign-extend on unpack). *Correctness:* **unpack round-trip is bit-exact** (pack then unpack returns the original nibbles — test this first, it's the most common bug); a **W4A16** linear's perplexity is within **~0.3–0.5 of FP16** (the GPTQ/Marlin bar). *Optional:* add an **AWQ or GPTQ scale-search step** before rounding and measure the perplexity improvement vs naive group-wise. *Target:* the bit-exact pack/unpack test passes on adversarial inputs (odd length, all-`0xF`, alternating signs).

**R3 — NVFP4 pack/unpack + block-scaled tensor-core GEMM (frontier path; the point of the assignment).** Implement the **two-level** quantizer of §2.6: per-tensor FP32 `s_global`, per-16-element FP8-E4M3 block scale `s_b`, E2M1 element LUT. Then feed it to a **block-scaled MMA** (start in **Triton or CUTLASS**, reusing the A3 block-scaling scaffold) that reads E2M1 + block scales and applies the scale in the mainloop. *Mechanism:* §2.6 — the global scalar normalizes block scales into E4M3's high-resolution band; the FP8 block scale hugs the local amax; the element is the 4-bit code; dequant fuses into accumulate. *Correctness:* **pack/unpack bit-exact** (`x̂ = s_global·s_b·q` reproduces the stored codes); **per-block MSE is strictly LOWER than an MXFP4 variant** (k=32, E8M0 power-of-two scale) you build on the *same data* — this is the required, derivable result (§2.6), and it's the whole point: the E4M3-scale-plus-16-block win is visible in the MSE; a **quantized linear lands <1% from BF16**. *Target:* a table — NVFP4 vs MXFP4 vs FP8 block MSE/SQNR on the same tensor — with NVFP4 closest to FP8 and beating MXFP4 by the documented mechanism. **[Native NVFP4/MXFP4 MMA throughput needs a B200; the *numerics* of this rung are fully checkable on a 4090 with the scales applied in software around an FP16 MMA.]**

**R4 — FP8 (E4M3) KV cache (connects to A1).** Write/read the KV cache in FP8 E4M3 with **per-channel-K / per-token-V** scales (§2.8). *Mechanism:* the production-default KV format — near-lossless at half the bytes, doubling capacity; the K/V asymmetry follows the outlier structure. *Correctness:* end-to-end generation quality (perplexity, or token-exact greedy on short prompts) **within noise of BF16 KV**, at **~half the KV memory** (reuse A1's metrics harness). Then **stress INT4 KV** to *see it degrade* (this is the point — show the failure), and *optionally* recover it with a **Hadamard rotation** on the KV (→ QuaRot). *Target:* FP8-KV quality delta in the noise; INT4-KV degradation visible; (stretch) rotation recovers most of it.

---

## §4 Frontier-2026 core (required)

The book stops at INT8/INT4 round-trip kernels. A principal-grade quantization engineer must demonstrate the three things that define 2026 low-precision inference. **All three are required.**

**4.1 NVFP4 two-level pack/unpack + block-scaled GEMM, proven more accurate than MXFP4 — for the documented reason.**
This is R3 taken to completion and *defended*. Implement the §2.6 two-level scheme against a **named reference** (TensorRT-Model-Optimizer or TransformerEngine — pin one and cite it, per the §2.6 honesty flag). Build the **MXFP4 (k=32, E8M0)** variant alongside it on the *same* tensors. *Deliver:* (a) bit-exact pack/unpack for both; (b) a measured table showing **NVFP4 block MSE < MXFP4 block MSE**, with the gap **attributed to the two named mechanisms** — E4M3 (non-power-of-two) block scale and the finer k=16 block; (c) a quantized linear (or a small attention/MLP block) **<1% from BF16** in NVFP4; (d) the block-scaled MMA applying scales in the mainloop (Triton/CUTLASS, on the A3 scaffold). The result you must be able to defend at a whiteboard: *"NVFP4 beats MXFP4 here by X% MSE because its FP8 scale lands within Y% of each k=16 block's amax while E8M0 rounds the k=32 amax to a power of two, up to 2× off — and here is the per-block error histogram that shows it."*

**4.2 FP8 E4M3 KV cache, near-lossless at 2× capacity (connect to A1).**
This is R4 taken to completion. Implement the E4M3 KV write/read with **per-channel-K / per-token-V** scaling and wire it into the A1 decoder. *Deliver:* end-to-end quality within noise of BF16 KV at half the KV bytes; the **doubled batch/sequence capacity** the freed memory enables (the throughput payoff, measured with A1's harness); and the **K/V asymmetry justified with measured error** (per-channel-K vs per-token-K, showing the ~2.5× error gap). State explicitly how this composes with A1's PagedAttention (FP8 blocks, scales stored per block).

**4.3 A PTQ method applied to a real layer, with measured accuracy.**
Pick **AWQ, GPTQ, or QuaRot** and apply it to an **actual linear layer of a real model** (a Llama/Qwen-class MLP or attention projection), not a synthetic tensor. *Deliver:* the **measured accuracy recovery vs naive round-to-nearest** at the same bit-width (perplexity on a held-out set, or the layer's output MSE against FP16), and a one-paragraph mechanism explanation of *which outlier problem it attacks* (AWQ: salient-channel protection signaled by activation magnitude; GPTQ: column-wise Hessian error compensation; QuaRot: Hadamard rotation to a near-Gaussian, outlier-free distribution). If you choose QuaRot, demonstrate the **computational-invariance fusion** (most rotations folded into weights, ~½ Hadamard online) and push to **W4A4**; if AWQ/GPTQ, show the W4A16 perplexity beating naive group-wise INT4 from R2.

**The spine, restated:** every claim in §4 is a *measured* SQNR/MSE/perplexity number against a reference. "I implemented NVFP4" is worth nothing; "NVFP4 block MSE is X, MXFP4 is 1.7X, FP8 is 0.6X on this tensor, and here's the histogram explaining why" is the deliverable. **Measure, don't hope.**

---

## §5 Correctness & numerics

- **SQNR/MSE on every rung, against FP32/BF16, on held-out data.** The dB floor `≈ 6.02·b + c` is the oracle: a clean symmetric round-trip *must* approach it; a deficit is a bug. Report **both** MSE (penalizes outliers — a single butchered value dominates) **and** MAE (typical error); the book's diagnostic — **MSE ≫ MAE means a few large errors dominate, usually clipping or a bad scale** — is your first triage step.
- **Pack/unpack must be bit-exact** for INT4 (R2) and NVFP4 (R3): pack then unpack returns the original codes, with adversarial inputs (odd length, all-`0xF`/all-max-code, alternating signs, a block that is all-zero, a block with one dominating outlier). This is the most common silent bug.
- **The NVFP4 > MXFP4 result must be demonstrated, not asserted** — same data, both pipelines, a per-block error histogram, and the gap attributed to the E4M3-scale + k=16 mechanism. If MXFP4 ties or beats NVFP4 on your data, you have a bug in the two-level scale placement (re-check against your pinned reference — the constants matter, per §2.6).
- **Calibrate and measure error on *different* data.** Computing the scale on a set and reporting SQNR on that same set is the book's cardinal sin — it measures fit, not generalization. Held-out validation, always.
- **FP8 KV must match BF16 KV within noise** on end-to-end generation, and the **K/V asymmetry must be justified by measured error** (per-channel-K beats per-token-K). INT4 KV *must visibly degrade* — if it doesn't, your stress test isn't stressing.
- **Adversarial / honesty cases:** a tensor whose amax is a single huge outlier (per-tensor INT8 should *fail* visibly — show it, then fix with per-channel/block); a post-ReLU all-positive tensor (symmetric should waste half the range — show it, then fix with asymmetric); a tensor that's all one value (scale=0 guard); the smallest-subnormal of each float format (verify the boundary **against the OCP spec**, per the §2.4 flag, before quoting it).

---

## §6 Profiling & performance

- **Memory-bound by nature.** Quant kernels move 1 byte/param (or 0.5 for INT4/FP4) + scales; they are usually **DRAM-bandwidth-bound** (book §9.5.4). Measure HBM utilization with Nsight Compute — if you're not saturating bandwidth, you're wasting time in dequant arithmetic or scale lookups. The whole point of weight quantization is *bandwidth*: prove the W4A16/FP8 path reads fewer bytes and runs faster at low batch *because* of it.
- **Dequant-in-mainloop is the win — show it's hidden.** For R3/§4.1 and the DeepGEMM/Marlin context, profile that the upconvert is **overlapped with the MMA** (the tensor cores never stall waiting on dequant). An `nsys`/`ncu` timeline should show compute-bound MMA with dequant tucked underneath, not a serial dequant-then-MMA.
- **The accuracy/throughput trade-off curve is the deliverable plot.** For each format (FP16, FP8, NVFP4, MXFP4, INT4), plot **accuracy (or SQNR) vs measured throughput/memory** on the same axes. This curve *is* the design-variable argument made visible — it shows NVFP4 sitting near FP8 accuracy at ~2× the element rate, and MXFP4 trading more accuracy for the same speed.
- **Quote hardware vs e2e numbers honestly** (per the §2.6 flag): the **~5× dense-PFLOPS NVFP4-vs-FP8 hardware ratio** is the ceiling; your **measured e2e** will be lower (memory traffic, scale handling) — report *both* and explain the gap. Don't present the 5× as if it were your throughput.
- **Benchmarking hygiene** per `00_foundations.md`: locked clocks, warmup discarded, CUDA-event timing, and on a clean GPU.

---

## §7 Stretch goals (competition-grade)

- **NVFP4 on a real B200 MMA:** take R3's software-scaled numerics to a native `tcgen05`/NVFP4 block-scaled MMA (sm_100) and measure the real element-rate win vs FP8 — closing the loop with A3's Blackwell rung.
- **DeepGEMM-style FP8 with two-level accumulation:** implement the **FP32-promotion every 4th WGMMA** (every 128 K-elements) on Hopper and *measure the accuracy difference* against naive ~14-bit ("FP22") FP8 accumulation — prove DeepSeek-V3's recipe matters numerically (bridges to A2).
- **SmoothQuant W8A8** end-to-end on a real model: migrate activation outliers into weights (`α=0.5`), show clean per-tensor/token quantization on both sides, and beat naive W8A8.
- **SpinQuant vs QuaRot:** replace the random Hadamard with a *learned* rotation and measure the W4A4 accuracy gain (up to ~16 points where Hadamard underperforms).
- **NF4 / QLoRA path** (book §9.5.2): implement NF4 (quantiles of a normal, block absmax, double-quantized scales) and compare its weight-only error to group-wise INT4 and NVFP4 on real weights.
- **FP8 KV → QuaRot KV4:** push the KV cache to 4-bit with a Hadamard rotation and recover the degradation you exposed in R4.

---

## §8 Deliverables & definition of done

1. **Code:** the full ladder — INT8 sym/asym (per-tensor → per-channel/token), group-wise INT4 (packed), **NVFP4 two-level pack/unpack + block-scaled GEMM**, the **MXFP4 comparison** variant, the **FP8 KV cache**, and **one PTQ method** (AWQ/GPTQ/QuaRot) on a real layer. Each with its SQNR/MSE oracle test and bit-exact pack/unpack tests.
2. **The numerics evidence:** a results table (FP16/FP8/NVFP4/MXFP4/INT4 — block MSE, SQNR, accuracy/perplexity), the **per-block error histogram proving NVFP4 < MXFP4**, the FP8-KV quality-vs-memory result, and the PTQ accuracy-recovery number. The accuracy-vs-throughput/memory trade-off plot.
3. **Design note (2–3 pages): "Precision as a design variable: what I chose for each tensor, and the error I measured."** For each tensor class (weights, activations, KV), state the format + granularity + scaling recipe you'd ship and *why*, backed by your SQNR/MSE numbers. Include the NVFP4-vs-MXFP4 derivation and the histogram. End with the honest gaps (where your e2e fell short of the hardware ratio and why; which format you would *not* ship and why). Write it for a frontier-lab peer.

**Done when:** every rung hits its SQNR/MSE oracle on held-out data; pack/unpack is bit-exact on adversarial inputs; **NVFP4 is demonstrably (not assertedly) lower-MSE than MXFP4 with the mechanism attributed**; the FP8 KV cache is near-lossless at half the bytes and composes with A1; a real PTQ method shows measured accuracy recovery on a real layer; and the design note would survive review by someone who ships low-precision inference.

---

## §9 Principal-level rubric

- **Pass:** the book's ladder works — INT8 sym/asym and group-wise INT4 quant/dequant kernels, bit-exact pack/unpack, validated by round-trip MSE.
- **Strong:** + per-channel/token granularity with the sym-vs-asym and per-tensor-vs-per-channel error deltas measured; SQNR floor reproduced and used as a bug detector; a working calibration with held-out validation.
- **Principal-grade:** all of §4 demonstrated — **NVFP4 two-level pack/unpack + block-scaled GEMM proven lower-MSE than MXFP4 with the E4M3-scale + k=16 mechanism attributed from a per-block histogram**; FP8 KV near-lossless at 2× capacity, composing with A1, K/V asymmetry justified by measured error; a real PTQ method (AWQ/GPTQ/QuaRot) with measured accuracy recovery on a real layer. You can **lay out every format's bit fields from memory**, **derive the NVFP4 two-level math and the SQNR floor at a whiteboard**, state the granularity↔bit-width rule, and **quote the 5× hardware ratio and ~2.3× e2e number without conflating them**. The differentiator is **numerical honesty**: you never say "it's quantized, it's fine" — you say "the per-block MSE is X, the floor is Y, here's the histogram, and here's the one tensor where per-tensor INT8 fails and why I'd use block-wise instead." *Precision is a design variable; you measured it, you didn't hope.*

---

## §10 References (see `references.md` for full list)

- OCP, **Microscaling (MX) Formats v1.0** + Rouhani et al., arXiv:2310.10537 (MXFP4/6/8, k=32, E8M0).
- OCP, **FP8 Formats for Deep Learning**, Micikevicius et al., arXiv:2209.05433 (E4M3/E5M2 bit layouts, no-Inf E4M3).
- NVIDIA, **"Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"** (blog) + arXiv:2509.23202 (two-level scaling, k=16, E4M3 scale).
- NVIDIA, **NVFP4 pretraining**, arXiv:2509.25149 (12B / 10T-token 4-bit run; MMLU-pro parity).
- Xiao et al., **SmoothQuant**, arXiv:2211.10438.
- Lin et al., **AWQ**, arXiv:2306.00978 (the book's §9.3 method).
- Frantar et al., **GPTQ**, arXiv:2210.17323 (OBQ/OBS, Hessian column compensation).
- Ashkboos et al., **QuaRot**, arXiv:2404.00456; Liu et al., **SpinQuant**, arXiv:2405.16406.
- Wang et al., **FP8-LM**, arXiv:2310.18313; DeepSeek-AI, **DeepSeek-V3**, arXiv:2412.19437 (1×128/128×128 scaling, two-level FP22 accumulation).
- Li et al., **KVTuner**, arXiv:2502.04420 (per-channel-K / per-token-V asymmetry).
- Frantar et al., **Marlin**, arXiv:2408.11743; **Machete** (vLLM/CUTLASS 3.x); **DeepGEMM**, github.com/deepseek-ai/DeepGEMM.
- OCP MX scale type, FP8 spec PDFs — **consult before quoting any subnormal boundary** (§2.4 honesty flag).
