# A5 Design Note — Precision as a design variable

> A5 quantization, CPU-buildable fake-quant numerics (`src/scratch_llm/quant/`). Oracle = SQNR/MSE and
> bit-exact pack/unpack, **never `allclose`** for FP4/FP8. 50 tests. All `[FACT]` in `bench/RESULTS.md`.
> The rule: precision is chosen **per tensor class**; a deficit below the SQNR floor is a bug, never
> "acceptable loss."

## 1. The floor and the law (INT8, R0/R1)

Symmetric per-tensor INT8 hits **40.5 dB** SQNR on Gaussian data and obeys the **~6 dB/bit law**
(measured 6.75 dB/bit; doubling the scale drops SQNR ~6 dB, verified — the sanity check that the scale
is right). Two granularity moves, each measured on the data class that needs it:
- **Asymmetric (affine) beats symmetric on skewed data:** +6.07 dB on an all-positive post-GELU-like
  tensor (a zero-point recentres the range that symmetric wastes).
- **Per-channel beats per-tensor:** +9.09 dB (each output row gets its own scale). Load-bearing insight:
  the per-channel *weight* scale is constant along the GEMM's contraction axis, so it **factors out of
  the matmul** — dequantize the INT32 accumulator once at the end, keeping the inner loop integer. That
  is why W8A8 is free and per-tensor-activation × per-channel-weight is the standard recipe (W8A8 int-GEMM
  matched fp32 to 1e-2 relative).

## 2. Sub-8-bit needs finer granularity (INT4 → FP4, R2/R3)

Fewer bits demand smaller scaling blocks. **Group-INT4 (g=128)** with 2-nibble packing: pack/unpack
**bit-exact** under a 500-case fuzz + adversarial (0x78/0xFF, odd length), SQNR 18.64 dB on its analytic
floor, group > per-tensor. At 4 bits the format itself matters:
- **NVFP4 (block-16, two-level: fp32 global × E4M3 per-block) beats MXFP4 (block-32, E8M0 per-block) by
  MSE 1.48×** (20.43 vs 18.74 dB) — for two named mechanisms: (a) the E4M3 block scale is a
  *non-power-of-two* fp8 number that tracks the ideal `block_amax/6` to ~1 part in 16, whereas E8M0
  rounds it to the nearest power of two (up to 2× off); (b) k=16 vs k=32 blocks localize the scale
  better. The adversarial verifier confirmed the MXFP4 baseline was the *stronger* ceil-E8M0 variant, so
  the NVFP4 win is real, not a rigged weak baseline. A software block-scaled GEMM lands 0.28% from BF16
  (same weights).

## 3. Precision in the serving loop (FP8-KV, R4) + PTQ (§4.3)

- **FP8 E4M3 KV cache** (real `torch.float8_e4m3fn`, clamp-before-cast since E4M3 is NaN>448):
  near-lossless end-to-end (24.45 dB vs BF16-KV) at **0.552× bytes**, with **per-channel-K 2.49× better
  than per-token-K** — and that gap drops to 0.60× on outlier-free data, proving it tracks the *data
  structure* (K has channel outliers), not the axis. INT4-KV **visibly degrades** (19.86 dB) — the stress
  test bites, as it must. This is the A5→A1 bridge: FP8-KV halves the KV byte bill the whole serving
  track fights.
- **AWQ PTQ (§4.3)** on a real linear layer: **1.71× MSE recovery** over naive INT4 by scaling up the
  ~1.5% activation-salient weight channels before rounding (fold 1/s into activations). Holds on held-out
  tokens (1.69×, not calibration-overfit); α=0 reproduces naive bit-exactly; the α grid has a genuine
  interior optimum. Salient-column error drops 2.87×.

## 4. The recipe table (what I'd ship, honestly)

| tensor class | format + granularity + recipe | why |
|---|---|---|
| dense weights (W8) | INT8 per-channel, static amax calibration | per-channel factors out of the GEMM; +9 dB |
| activations (post-GELU) | INT8 asymmetric per-token, dynamic | skewed range needs a zero-point; +6 dB |
| 4-bit weights | NVFP4 (block-16 two-level) + AWQ salient-scaling | non-pow2 block scale beats MXFP4 1.48×; AWQ +1.71× |
| KV cache | FP8 E4M3, per-channel-K / per-token-V | near-lossless at ½ bytes; per-ch-K tracks the K outliers |

**Honest gaps:** these are numerics demonstrations (layer-level MSE, not full-model perplexity — the
rental-gated SKIP) and *software* fake-quant; the native NVFP4/FP8 tensor-core throughput is B200/H100
(`B200_day_runbook.md` §7, `H100_day_runbook.md` §2.5). But the numerics — which format, which
granularity, which recipe, and the measured accuracy of each — are the design decisions, and they are
made here on measured SQNR/MSE, not hope.
