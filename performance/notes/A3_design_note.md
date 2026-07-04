# A3 Design Note — The register→SMEM→async arc, made explicit

> A3 tensor cores, sm120 (72–84 TF/s cuBLAS proxy). Built in **CUDA C++** (torch cpp_extension,
> `-arch=sm_120`) on purpose: Triton's `tl.dot` uses the tensor cores but *hides* the fragment
> machinery, and A3's whole point is that machinery. 4096³ f16→fp32, 34 gpu tests, all `[FACT]`.

## 1. The ladder (measured)

| rung | kernel | TF/s | % of cuBLAS |
|---|---|---|---|
| R0 | naive 32×32 SMEM-tiled GEMM (CUDA cores, **no tensor cores**) | 3.5 | **4.1%** — the floor |
| R1 | WMMA GEMM (`nvcuda::wmma` 16×16×16 fragments) | 28.3 | **38.9%** — ~8× jump |
| R2 | `mma.sync` + `ldmatrix` + XOR-swizzled SMEM | 59.0 | **81.9%** |

The story is one number moving: **4.1% → 38.9% → 81.9%**. R0 establishes the CUDA-core floor (every
thread owns one C element, fp32 inner-product accumulate over SMEM tiles — the A2 GEMM lesson without
tensor cores). R1 is the **~8× tensor-core jump**: a warpgroup of `wmma::fragment`s does a 16×16×16 MMA
per instruction on the tensor cores instead of scalar FMAs. R2 drops to the **warp level** with the raw
`mma.sync.aligned.m16n8k16` PTX, loads the operand fragments with `ldmatrix` (a warp-collective load
straight into the tensor-core register layout), and lays out SMEM with an **XOR swizzle** so consecutive
rows hit distinct banks — reaching **82% of cuBLAS**.

## 2. The two load-bearing correctness facts

- **FP32 accumulation matters:** the WMMA kernel accumulates in `float` even with `half` inputs. An
  FP16-accumulate variant's error **grows with K** (tested) — a 4096-deep reduction in FP16 loses the
  low bits of each partial sum. This is why every tensor-core GEMM (and the A5 FP8 two-level promotion)
  accumulates in FP32.
- **Element-exactness, not "close enough":** each rung is element-exact vs `torch.matmul` at fp32-accum
  tolerance (R2: rel err **6.6e-6**), on adversarial K-remainders and outliers. A tensor-core kernel that
  is 1% wrong is worthless no matter how fast — the fragment layout / swizzle / drain is easy to get
  subtly wrong (a mis-set validity bit silently yields zeros), so the oracle gate is the whole game.

## 3. The arc, and the honest gap

A3 is the **register → SMEM → async** arc: R0 keeps operands in registers/SMEM with synchronous loads;
R1/R2 move to tensor-core fragment registers via `ldmatrix`. The next step is **`cp.async`
double-buffering** — overlap the next tile's GMEM→SMEM copy behind the current MMA — which our
synchronous R1/R2 kernels do NOT yet do (that's the climb into the higher tuned band). Beyond that is
**WGMMA + TMA** (Hopper, `sm_90a`-only): the warpgroup-async MMA + hardware tensor-memory-accelerator
copy that reaches ~318→618 TF/s. Those are **H100-gated** (`H100_day_runbook.md` §3), and the hand-decode
of the WGMMA descriptor is already written (`performance/artifacts/wgmma_descriptor_manual.md`) so the
rental day is pure execution. On Blackwell datacenter (B200) the arc ends at `tcgen05`/TMEM
(`B200_day_runbook.md`). **The A3 lesson: the tensor core is A2's register-tiling frozen into silicon;
each generation moves the operands one level further from the ALU (register → SMEM → TMEM) and the issue
from sync to async — and getting to 82% of cuBLAS on sm120 is `ldmatrix` + a conflict-free SMEM swizzle,
the same craft WGMMA automates in hardware.**
