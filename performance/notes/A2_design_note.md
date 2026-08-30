# A2 Design Note — Climbing from coalescing to the Hopper ceiling

> A2 kernel optimization, standing GPU sm120 (RTX PRO 4000 Blackwell, 0.551 TB/s HBM, 72 TF/s bf16,
> ridge ≈131 FLOP/byte). Triton-primary (ADR-0011). Every number `[FACT]` in `bench/RESULTS.md`,
> workflow-built + adversarially verified + main-thread gpu-tested (79 gpu tests). ncu blocked on this
> box → bounds established by achieved-vs-measured-peak % + nsys; per-kernel ncu-debt discharged on
> **sm_120 + counters** (KVM 5090 ~$0.33/hr), NOT the H100 day it was misfiled to until 2026-08-30 —
> these kernels are sm_120 and a Hopper recompile is a different kernel instance. Only the WGMMA
> tensor-pipe metric is genuinely Hopper. (The "standing GPU" above is also gone — see PLAN.md.)

## 1. The one idea: four of five kernels are memory-bound forever

GEMV, softmax, RMSNorm/LayerNorm all have arithmetic intensity ≈1 — they touch each byte O(1) times, so
their ceiling is **HBM bandwidth**, not FLOPs, no matter how clever the kernel. The only lever is to
stream memory at peak: coalesced 128-byte transactions, no redundant passes. GEMM is the one kernel that
crosses the ridge (AI ≈ 1365 at 4096³) and becomes **compute-bound** — the only place tensor-core
throughput is the story. The whole A2 ladder is: for the memory-bound four, get to the wall; for GEMM,
climb toward cuBLAS.

## 2. The measured ladder

| kernel | naive | optimized | ceiling | bound |
|---|---|---|---|---|
| **GEMV** | 431 GB/s (78% HBM) one-warp/row | **528.8 GB/s = 95.9% HBM** coalesced block-per-row (107% of torch.mv) | 0.551 TB/s | memory ✓ at wall |
| **softmax** | two-pass 4N bytes | **fused 1-pass ~100% HBM**, 3N bytes → 1.33× fewer, 1.17× faster >L2 | 0.551 TB/s | memory ✓ at wall |
| **RMSNorm / LN** | — | **both ~100–101% HBM @N≥4096**; RMS one reduction vs LN two (within ±1%, the small-N edge is noise) | 0.551 TB/s | memory ✓ at wall |
| **TopK** | torch.topk 15.7% | **iter-max 46.9% peak** — the honest poor-GPU-fit; **fused softmax+topk 3.4× faster / 3.0× less traffic** | occupancy | the "measure the failure" rung |
| **GEMM** | **0.2% of cuBLAS** (scalar, no tiling) | tiled tl.dot **128.5%** → autotuned **134.3% of cuBLAS-proxy** (101.9 TF/s) | compute | ridge crossed, AI 1365 |

The memory-bound four all reach **96–100% of the measured HBM peak** — the wall IS the bound and they hit
it. GEMM reproduces the siboehm ladder SHAPE (0.2% → 134%): the naive kernel is L1/issue-bound at ~0
reuse; SMEM tiling with `tl.dot` gives every output-tile 2× register reuse and lands on the tensor cores;
autotuning the block/warp/stage config beats torch.matmul's default cuBLAS heuristic at this shape
(disclosed honestly — %roof > 100% means the roof is a torch.matmul proxy Triton edges out, not a
super-peak claim). TopK is the deliberate negative: a selection op is serial dependent reductions with ~0
arithmetic intensity, so it is occupancy/latency-bound at 46.9% — the shippable win is *fusing* it with
softmax to kill a HBM round-trip (3.4×), not a faster standalone kernel.

## 3. The honesty flag: what Triton abstracts (and the raw-CUDA ladder would add)

The A2 spec's CUDA ladder (naive → coalescing → float4 vectorization → XOR swizzle → register blocking)
is pedagogically CUDA-idiomatic; in Triton the compiler manages most of it:
- **Coalescing + 128-bit vectorization**: Triton emits vectorized, coalesced loads for a contiguous
  `tl.load` over a wide `tl.arange` on its own — the GEMV `float4` rung and GEMM's `LDG.E.128` are free.
- **Bank-conflict avoidance**: Triton swizzles SMEM for `tl.dot`; the A2 XOR-swizzle rung is the
  compiler's job (and unmeasurable here anyway — ncu bank-conflict counter blocked → ncu-debt).
- **What a raw-CUDA kernel would add**: explicit `float4`/`__nv_bfloat162` packed loads, warp-shuffle
  tree reductions (vs `tl.sum`), `__ldg` read-only caching, hand register-blocking. None change the
  *bound* (we are HBM- or tensor-core-limited, not issue-limited); they only close the last few % a hand
  kernel reaches. This is stated per kernel in the module docstrings — not fabricated as a CUDA ladder we
  didn't write.

## 4. Floor / ceiling verdict

Floor = torch.compile/`torch.matmul` (cuBLAS proxy); ceiling = cuBLAS/CUTLASS. **Verdict:** the
memory-bound four beat any library by construction (they hit the wall — nothing can do better); the
autotuned GEMM *beats* the torch.matmul default at 4096³ (134%) but the real cuBLAS/CUTLASS ceiling on
Hopper is where the ~10× tensor-core jump lives — unreachable on sm120 (WGMMA is sm_90a-only). That jump
(CUDA-core 32 → WGMMA 318 → warp-spec 531 → CUTLASS 630 TF/s) is the H100 day (§4, `H100_day_runbook.md`),
where the ncu-debt registered per kernel here is also discharged. **The A2 lesson: on memory-bound
kernels the craft is getting to the wall; the wall itself is the ceiling. On GEMM the craft is tiling +
tensor cores, and the last 10% to cuBLAS costs more than it's worth unless you're the one writing cuBLAS.**
