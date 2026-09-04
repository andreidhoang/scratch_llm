# Kernel benches — the ladder index

One roofline bench per kernel, mirroring `src/scratch_llm/kernels/<family>/<backend>/`. Every bench
targets a **specific backend** (not the dispatch layer) — by policy benches measure one implementation,
they do not dispatch (see `src/scratch_llm/kernels/CLAUDE.md` §"dispatch contract"). All share one
measurement methodology (`../_harness.py`: CUDA-event timing, per-rep L2 flush, median + p20–p80
spread, peaks measured on *this* GPU). The runner (`run.py`) is the single entry point.

## Run

```bash
# from the repo root, GPU venv active
python -m bench.kernels.run                       # list every bench + GPU availability
python -m bench.kernels.run --gpu-check           # print arch (common.arch) + measured peaks
python -m bench.kernels.run gemm                  # run every GEMM bench (family filter)
python -m bench.kernels.run gemm/triton_tiled     # run ONE bench
python -m bench.kernels.run --all                 # run the whole ladder
python -m bench.kernels.run gemm/triton_tiled -- --n 8192   # forward args to the bench
```

GPU benches auto-skip on a CPU box with a clear message (mirrors the `gpu` pytest marker). The runner
launches each bench as a subprocess so a Triton/nvcc failure in one bench cannot poison another.

## The ladder

| runner name | rung | script | kernel exercised | dtype |
|---|---|---|---|---|
| `gemm/triton_tiled` | A2 R5+6 | `gemm/triton_tiled.py` | `kernels.gemm.triton.tiled` (`gemm_naive` → `gemm_tiled` → `gemm_autotuned`) | bf16 |
| `gemm/triton_gemv` | A2 R1 | `gemm/triton_gemv.py` | `kernels.gemm.triton.gemv` (`gemv_naive` → `gemv_blockrow` → `gemv_split`) | bf16 |
| `gemm/cuda_smem` | A3 R0 | `gemm/cuda_smem.py` | `kernels.gemm.cuda.smem_tiled` (CUDA cores, **no tensor cores** — the floor) | bf16 |
| `gemm/cuda_mma_sync` | A3 R2 | `gemm/cuda_mma_sync.py` | `kernels.gemm.cuda.mma_sync` (PTX `mma.sync` m16n8k16 + `ldmatrix` + XOR-swizzled SMEM) | fp16 |
| `gemm/wmma` | A3 R1 | `gemm/wmma.py` | `kernels.gemm.wmma.gemm` (`nvcuda::wmma` fragments, fp32-acc; the fp16-acc anti-example is in the test) | fp16 |
| `attention/fa2_fwd_roofline` | A2.1 fwd | `attention/fa2_fwd_roofline.py` | `kernels.attention.prefill.fa2` (`flash_attention_triton_forward`) vs SDPA-flash | bf16 |
| `attention/fa2_bwd_roofline` | A2.1 bwd | `attention/fa2_bwd_roofline.py` | `kernels.attention.prefill.fa2` (`TritonFlashAttention` fwd+bwd) vs SDPA | bf16 |
| `attention/fa2_sm120` | A4 R2+3 | `attention/fa2_sm120.py` | `kernels.attention.prefill.fa2` on sm120: correctness + roofline + OOM-sweep + causal/GQA | bf16 |
| `norm/normalize` | A2 R3 | `norm/normalize.py` | `kernels.norm.normalize` (`rmsnorm_triton`, `layernorm_triton`) vs unfused + vendor | bf16 |
| `reduce/softmax` | A2 R2 | `reduce/softmax.py` | `kernels.reduce.softmax` (`twopass` → `online` → `fused`) | bf16 |
| `reduce/topk` | A2 R4 | `reduce/topk.py` | `kernels.reduce.topk` (`topk_last_dim` + `fused_softmax_topk`) | bf16 |
| `gemm/cuda_wgmma` | A3 R3.1 | `gemm/cuda_wgmma.py` | `kernels.gemm.cuda.wgmma` (Hopper WGMMA, **sm_90a**) — compile-verified, never run | fp16 |
| `gemm/cuda_tcgen05` | A3 R4.1 | `gemm/cuda_tcgen05.py` | `kernels.gemm.cuda.tcgen05` (Blackwell tcgen05, **sm_100a**) — compile-verified, never run | fp16 |
| `attention/fa3_hopper` | A3 R3.5 | `attention/fa3_hopper.py` | `kernels.attention.prefill.fa3` (Hopper FA3, **sm_90a**) — compile-verified, never run | bf16 |
| `gemm/cuda_fp8` | A3 R3.2 | `gemm/cuda_fp8.py` | `kernels.gemm.cuda.fp8` — **STUB, raises**; learning rep, no ledger row by design | fp8 |
| `gemm/cuda_stream_k` | A3 R3.3 | `gemm/cuda_stream_k.py` | `kernels.gemm.cuda.stream_k` — **STUB, raises** | fp16 |
| `gemm/cuda_persistent` | A3 R3.4 | `gemm/cuda_persistent.py` | `kernels.gemm.cuda.persistent` — **STUB, raises** | fp16 |

Each row's measured numbers live in [`../RESULTS.md`](../RESULTS.md) (the durable ledger — predict-
before-run, then the measured gap + root cause).

## Layout convention

```
bench/
  _harness.py                 # shared measurement spine (FLOPs, bytes, CTAs, roofline arithmetic)
  kernel_roofline.py          # the A2 R0 profiler + KernelProfile (imported by gemm/softmax benches)
  RESULTS.md                  # the measurement ledger (every bench's headline number, dated)
  kernels/                    ← THIS folder
    README.md                 ← this index
    run.py                    # the ladder runner (python -m bench.kernels.run)
    gemm/{triton_tiled,triton_gemv,cuda_smem,cuda_mma_sync,wmma}.py
    attention/{fa2_fwd_roofline,fa2_bwd_roofline,fa2_sm120}.py
    norm/normalize.py
    reduce/{softmax,topk}.py
```

The tree mirrors `src/scratch_llm/kernels/<family>/<backend>/` so a kernel and its bench are one
`grep` apart. Tests mirror it too: `tests/kernels/test_<kernel>.py`.

## How to add a kernel bench

1. **Oracle-first test** — write `tests/kernels/test_<your_kernel>.py` vs a pure-PyTorch reference,
   GPU-gated (`pytest.importorskip("triton")` + `pytestmark = pytest.mark.gpu`). See
   `tests/kernels/test_flash_attention_triton.py` for the pattern.
2. **Bench script** — drop `bench/kernels/<family>/<slug>.py`. Import `Roofs`, `bench_ms`,
   `provenance_line`, `spread_pct` from `bench/_harness.py` via the move-proof `_BENCH_ROOT` lookup
   (copy the header block from any existing bench here). Gate on correctness before timing.
3. **Register** — add a `Bench(...)` row to `_BENCHES` in `bench/kernels/run.py` (name, script, rung,
   kernel-exercised).
4. **Index** — add the matching row to the table above.
5. **Ledger** — once measured, append a dated row to `../RESULTS.md` with predicted vs measured + the
   bound + root cause. The DoD is the profiled number, not a green test (FOP-3).

## Standing artifacts (committed)

- `../a2_roofline.csv`, `../a2_roofline.png` — written by `../kernel_roofline.py` (the A2 R0 harness).
- `../a3_isoflop.png` — written by `../../scripts/a3_isoflop.py` (the A3 scaling fit; referenced from
  RESULTS.md / STATUS.md — not an orphan, do not delete).
- `../a5_nvfp4_mxfp4.png` — written by `../a5_nvfp4_mxfp4.py` (the A5 quantization bench, top-level).

## What is NOT here (and why)

- **TMA / FP4** — not implemented. (Corrected 31/08: WGMMA, tcgen05, FA3, FP8, persistent and
  stream-K *are* present — six rows added to the table above — and this bullet claiming otherwise was
  falsified by files landed in its own commit. Three are compile-verified with ledger rows
  `../RESULTS.md:966-968`; three are stubs that raise and deliberately carry **no** ledger row, because
  "implemented ≠ measured" — FOP-4.) The CUDA JIT loaders read the target ISA from
  `kernels/common/arch.arch_name()` at runtime, so a new arch rung is a branch + a `require_cc(...)`
  guard, not a loader rewrite. Roadmap: `docs/GPU_FROM_ZERO.md`.
- **Nothing here has run on datacenter silicon.** Compile-verified ≠ measured; every sm_90a/sm_100a
  row is rental-gated (`PLAN.md` § Hardware law).
- **System / serving benches** (continuous batching, chunked prefill, cudagraph decode, speculative,
  disagg, KV-memory, decode-roofline) live at `bench/` top level — they exercise `scratch_llm.serving`
  and `scratch_llm.model`, not a kernel backend, so they don't belong under `bench/kernels/`.
