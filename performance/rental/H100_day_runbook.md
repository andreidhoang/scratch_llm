# H100 Day Runbook (Tier 1) — the ISA-gated kernel rungs + single-GPU serving blocks

> **Rental discipline (ADR-0012 §2):** do NOT rent until every sm_120-runnable rung of A2/A3/A4/A5 is
> oracle-correct + `[FACT]`-ledgered (✅ as of 2026-07-04: A2 R0–R6, A3 R0–R2, A4 R0–R3, A5 R0–R4 all
> shipped). Arrive with compiled, tested code and the exact commands below. Target: 1× H100 SXM
> (sm_90a, 80 GB, 3.35 TB/s), ~12–15 h, ~$25–45. **The `a` in `-arch=sm_90a` is mandatory** (WGMMA/TMA
> are `sm_90a`-only, not plain `sm_90`).

## 0. Boot + verify (0.5 h) — copy-paste

```bash
# on the rented box
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm --format=csv
sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits),\
                     $(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits)  # LOCK CLOCKS
ncu --version && ncu --query-metrics | head   # VERIFY ncu COUNTERS ARE ENABLED (they were blocked on
                                              # the sm120 dev box: ERR_NVGPUCTRPERM). If blocked here too,
                                              # the box is misconfigured — fix or pick another listing.
nvcc --version   # confirm CUDA >= 12.3 for full WGMMA/TMA + FP8 support
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"  # expect (9,0)
git clone <this repo> && cd scratch_llm && bash scripts/bootstrap-pod.sh   # venv + deps + hooks
# smoke: the sm120 suite must still pass (portable) before touching sm_90a code
.venv/bin/python -m pytest -m "not gpu" -q
```

## 1. The ncu-debt to discharge FIRST (the whole reason for the counters)

Every sm_120 kernel shipped a **ncu-debt** metric it could not measure (counters blocked). Run
`ncu --set full --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy` on each and
record the number in `bench/RESULTS.md` (this VALIDATES the sm120 %-of-peak bound calls):

| kernel (sm120) | ncu metric to capture | expected |
|---|---|---|
| A2 GEMV `gemv_triton` | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / requests` | ≈4 sectors/req (coalesced) |
| A2 softmax / norms | `sm__throughput` Memory% (Speed-of-Light) | ≫ Compute% (mem-bound) |
| A2 GEMM `gemm_triton` | Memory% → Compute% flip | Compute-bound |
| A2 TopK | `launch__waves_per_multiprocessor` + `No Eligible` warp stall | occupancy/stall-bound |
| A3 `a3_mma_sync` | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | ≈0 (XOR swizzle works) |
| A4 FA2 `flash_attention_triton` | `sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained` | tensor-pipe util |

## 2. A2 §4.1–4.5 — WGMMA / TMA / warp-spec / FP8 GEMM (the ~10× jump)

Prereq: A2 R0–R6 ✅ (the CUDA-core ceiling). Pre-registered targets (PERF_PLAN Phase 2, copy to
RESULTS.md BEFORE running — H100 SXM dense FP16 = 989 TF/s, FP8 = 1,979; state the SKU):

| # | build | target `[INFERENCE, pre-reg]` | ncu gate |
|---|---|---|---|
| §4.1 | `cp.async` multistage → TMA (`cp.async.bulk.tensor`, `cuTensorMapEncodeTiled`, 128B swizzle) | bit-identical vs sync loader; WarpStateStats load-stall ↓ | — |
| §4.2 | WGMMA mainloop (`wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16`, 64-bit SMEM descriptors — see `performance/artifacts/wgmma_descriptor_manual.md`) | **~318 TF/s** (the ~10× over CUDA-core) | `sm__pipe_tensor_op` ↑ |
| §4.3 | warp-specialized + persistent + Stream-K (`setmaxnreg` 24/240/240, grid=#SMs) | **~531 TF/s (~71% PCIe)** → CUTLASS-class ~630 (~84%); NOT the 21-TF spill | tensor-pipe ≥85% |
| §4.4 | epilogue fusion (scale/bias/act/cast in registers) | fused cast bit-exact; one kernel in nsys, not two | — |
| §4.5 | FP8 fine-grained GEMM (DeepGEMM two-level FP32 promotion every 128 K-elem) | **~1350–1550 TF/s** (H800 [UNCERTAIN]); promotion-OFF fails SQNR, ON passes | — |

Build first, compile-check on the sm120 box NOW with `nvcc -arch=sm_90a -ptx` (compiles, won't run);
the PTX artifact (`performance/artifacts/wgmma_descriptor_manual.md`) is the decode reference to diff
`nvcc -ptx` against. On the H100: run, oracle (`torch.matmul` rtol 1e-2, K-remainder/outlier adversarial),
measure %-of-cuBLAS, ncu the tensor-pipe util.

## 3. A3 R3–R4 + §4.1 — WGMMA + TMA multistage + FP8 + warp-spec persistent

Prereq: A3 R0–R2 ✅ (WMMA + mma.sync on sm120). Four sub-rungs (book H100 4096³ FP16 ladder):
basic WGMMA **~318** → larger tiles **~433** → +TMA 3-stage **~504** → **618 TF/s = 87% of cuBLAS**; §4.1
warp-specialized persistent **≥85% of 989 dense**; R4 FP8 WGMMA (`wait_group<N>` overlap, K=32) **≥75%
of 1,979 FP8**. Gate: swizzle atom matches TMA swizzle (else transposed garbage); validity bit set (else
silent zeros). This shares the §2 code — do A2 §4.2 first, A3 R3 reuses the WGMMA mainloop.

## 4. A4 R4 — FA3-class Hopper attention

Start from CUTLASS/CuTe. Warp-spec + TMA + ping-pong + FP8. Target: **~75% util / ~740 TF/s** (the
published FA3 number); benchmark vs FlashAttention-3 + cuDNN on the same shape. Oracle: matches SDPA
<1e-2 bf16; FP8 within its error band. Reuses the A2 §4.1 TMA + §4.5 FP8 primitives.

## 5. Serving blocks S1–S3 (single-GPU frontier serving) — pre-registered (PERF_PLAN)

Copy these rows to RESULTS.md before running:

| # | experiment | predicted | bound |
|---|---|---|---|
| S1 | Llama-3.3-70B FP8, B=1 decode tok/s | **35–45** (roofline ceiling 47.9 = 3.35 TB/s ÷ 70 GB) | memory |
| S2 | 70B-FP8 KV headroom on 80 GB | **~3–5 GB free ⇒ ~10–15 K cached tokens** (GQA-8 327.7 KB/tok) — the dense-70B-barely-batches lesson | capacity |
| S3 | gpt-oss-120b (5.1B active MXFP4), B=1 decode | **100–250 tok/s, latency-floor-bound** (not the naive ~1,100) | latency |

Engine: pin the vLLM/SGLang image + smoke the launch line beforehand. Drive with the two
`bench/continuous.py`-derived traces (heavy-tail saturated + shallow) — one trace can't measure both
throughput and TTFT (R3b finding).

## 6. Before releasing the box (non-negotiable)

- [ ] Every §2–§4 kernel: %-of-cuBLAS + ncu tensor-pipe util → RESULTS.md, SKU + dense/sparse stated.
- [ ] The §1 ncu-debt table fully discharged (the sm120 bound calls validated or corrected).
- [ ] S1–S3 six serving numbers logged.
- [ ] `git push` — the box is ephemeral; nothing survives release except what's committed.
