# A2 — Kernel Optimization: From Coalescing to the Hopper Ceiling

> **Book chapter:** 6 (Optimizing our transformer kernels) · **Frontier thesis:** Hand-writing a kernel is only justified when you beat the `torch.compile`/Triton floor, and the path to the cuBLAS/CUTLASS ceiling is a *disciplined climb up arithmetic intensity* — coalesce, tile, register-block, vectorize on CUDA cores, then async-copy + tensor-core + warp-specialize + persistent-schedule on Hopper. The ladder is the lesson; the profiler is the spine.
>
> **Primary hardware:** RTX 4090/A6000/A100 for rungs 0–6 (CUDA-core ladder); **1× H100 (sm_90a)** is *required* for §4 (WGMMA/TMA/warp-spec/FP8). · **Est. time:** 2–2.5 weeks · **Prereqs:** book ch1–5; A1 (you reuse its GEMV/softmax intuition and its roofline reflex); the discipline in `00_foundations.md` (roofline-first, lock clocks, oracle-first, one-variable, journal).

---

## §0 Why this matters

A1 taught you that inference speed is a *systems* problem — scheduling, caching, batching. A2 is the other half of the principal skill set: when the system *is* right and you are still leaving performance on the table, you descend into the kernel. The book's chapter 6 is the honest, end-to-end version of this descent for the five kernels that a transformer actually spends its time in — GEMV (decode), softmax, LayerNorm/RMSNorm, Top-K, and GEMM (prefill/training) — and it climbs each one from naive to "approaching cuBLAS" while *profiling every rung in Nsight Compute*. That discipline — profile, find the bound, attack the bound, re-measure — is the entire job.

But the book deliberately stops at the CUDA-core ceiling (~78% of cuBLAS for GEMM on a small FP16 problem, p.237) and says explicitly that "cuBLAS uses tensor cores that provide FP16/BF16 operations at 10× FP32 throughput" — and then leaves the tensor-core chapter for later. **A2's frontier core is exactly that 10× gap.** On a 2026 datacenter GPU, a kernel that doesn't use tensor cores is leaving an order of magnitude on the floor, and a tensor-core kernel that doesn't use TMA, warp specialization, and a persistent scheduler is leaving the *next* 2–3× on the floor. The Colfax Hopper ladder makes the size of this concrete: introducing tensor cores + TMA is a single ~10× jump (32 → 317 TFLOP/s), and the full expert kernel reaches ~84% of the H100 PCIe FP16 roof.

What real systems depend on this: every GEMM in every training step and every prefill, every norm and softmax in every forward pass. cuBLAS/cuDNN/CUTLASS exist so you usually *don't* write these — so the principal question is never "can I write a GEMM" but "**should I**, and if so, can I beat the library floor and close enough of the gap to the ceiling to justify the code." This assignment makes you earn the right to answer that with numbers.

---

## §1 Learning objectives

You can:

1. **Derive** the arithmetic intensity of GEMV, softmax, LayerNorm/RMSNorm, Top-K, and GEMM from first principles, place each on the roofline, and predict its bound (memory vs compute) *before* profiling.
2. **Implement** the book's GEMV ladder (naive → coalesced row-per-block + warp-reduce → two-stage block reduction → vectorized `float4`) and reach the bandwidth roof; explain why GEMV is decode's workhorse (A1 §2.1).
3. **Implement** online softmax (Milakov & Gimelshein), a warp-shuffle reduction with correct masks, and a fused/vectorized variant — and prove the single-pass running-max/denom identity is numerically equivalent to two-pass.
4. **Implement** RMSNorm and LayerNorm with warp-shuffle mean/variance (Welford for LN), and quantify RMSNorm's one-reduction-vs-two win.
5. **Implement** the book's GEMM ladder on CUDA cores (naive → coalesce → SMEM tile → 1D blocktile → 2D blocktile 8×8/thread → vectorized 128-bit) and reproduce the siboehm-style %-of-cuBLAS progression, with the benchmark-hygiene caveats pinned.
6. **Implement** the Hopper frontier GEMM: `cp.async`/TMA async copy → WGMMA tensor-core mainloop → warp-specialized producer/consumer → persistent + Stream-K scheduler → epilogue fusion, climbing toward the H100 PCIe FP16 roof.
7. **Implement** a DeepGEMM-style FP8 fine-grained-scaled GEMM with two-level FP32 promotion, and explain why Hopper's FP8 WGMMA accumulation forces it.
8. **Read CUTLASS/CuTe** well enough to map its abstractions (Layout algebra, CollectiveMainloop, Epilogue Visitor Trees) onto the primitives you built by hand, and **beat the `torch.compile`/Triton floor** or state precisely why you can't.

---

## §2 First-principles theory

The governing principle, stated by siboehm and echoed in the book: **optimize arithmetic intensity as long as you are memory-bound.** Every rung either reduces bytes moved (coalescing, vectorization, fusion) or increases FLOPs per byte loaded (tiling, register blocking) — until the kernel crosses the roofline ridge and becomes compute-bound, after which only the math units (tensor cores, ILP) matter. Re-derive the ridge each time: H100 SXM FP16 ≈ `989e12 / 3.35e12 ≈ 295 FLOP/byte`; FP8 ≈ `≈ 590` (`00_foundations.md` §3).

### 2.1 The five kernels and where they sit on the roofline

| Kernel | Arithmetic intensity | Bound | Optimal strategy |
|---|---|---|---|
| GEMV (decode) | ~0.5 FLOP/byte | memory | coalescing + vectorization → aim ~peak HBM |
| Softmax | ~1 FLOP/byte | memory | single-pass online + warp-shuffle fusion |
| LayerNorm / RMSNorm | ~1 FLOP/byte | memory | warp-shuffle mean/var, fused load-to-SMEM |
| Top-K | low + serial | latency/serial | parallel block reduction; hard to parallelize |
| GEMM (prefill/train) | ~hundreds FLOP/byte (≈512 at 4096³ FP16, book p.222) | compute | tiling + register blocking + **tensor cores** |

The book's own summary table (p.240) matches this. The lesson is structural: **four of the five are memory-bound and stay that way no matter how you optimize — only GEMM has the arithmetic intensity to be worth pushing into the compute-bound, tensor-core regime.** That is why §4 (the frontier core) is almost entirely about GEMM, and the norm/softmax/Top-K kernels top out at "saturate HBM bandwidth."

### 2.2 GEMV — the decode workhorse (book 6.2)

GEMV computes `y = A·x` for `A` an `M×K` matrix, `x` a `K`-vector. It reads `A` (the weights) exactly once and does `2MK` FLOPs over `~MK·dtype` bytes → AI ≈ `1/dtype_bytes` ≈ 0.5 for FP32 → hopelessly memory-bound, ceiling = HBM bandwidth (this is the decode-step roofline from A1 §2.1). The ladder is therefore purely a *memory-traffic* ladder:

- **Naive:** one thread per output row strides `x` and `A[row,:]` — uncoalesced, abysmal.
- **Coalesced, block-per-row:** assign one block per output row; the block's threads stride across the row so consecutive threads touch consecutive `A` elements (coalesced 128-byte transactions), then a **`warpReduceSum`** combines partials. siboehm's analogous coalescing fix took global throughput from ~15 GB/s to ~110 GB/s.
- **Block-level reduction:** two-stage — warp-shuffle within each warp → 32 partials to shared memory → first warp reduces — so the block can be >32 threads and hide more latency.
- **Vectorized `float4`:** load 16 bytes/instruction (`LDG.E.128`), quartering instruction count and saturating the bus. *Target: approach peak HBM bandwidth, compare vs cuBLAS GEMV (cublasSgemv).*

### 2.3 Reductions: online softmax + warp shuffles (book 6.3/6.4)

**Online softmax** (Milakov & Gimelshein, arXiv:1805.02867) computes `softmax(x)` in a single pass by maintaining a running max `m` and running denominator `d`; when a new element raises the max from `m_old` to `m_new`, you rescale the accumulated denom by `e^(m_old − m_new)`:
```
m_new = max(m_old, x_i)
d_new = d_old · e^(m_old − m_new) + e^(x_i − m_new)
```
This is algebraically identical to the textbook two-pass (max, then sum of `e^(x−max)`) but moves the data once instead of three times → up to ~1.3× on softmax, ~5× on softmax+Top-K, purely a memory-traffic win. It is the *same recurrence* that FlashAttention uses to avoid materializing the score matrix (A4) — internalize it here.

**Warp-shuffle reductions** (`__shfl_down_sync`, CUDA 9+): a warp reduces 32 values in `log2(32) = 5` register-to-register steps with **no shared memory and no `__syncthreads`**. Masks are mandatory — derive the active mask with `__ballot_sync` and never assume a full warp on the remainder. A **block reduction** is two-stage: warp-shuffle → `shared[32]` → first warp shuffles again. The classic alternative is Mark Harris's 7-step shared-memory reduction (sequential addressing to kill bank conflicts) or `cub::BlockReduce` — know all three and when each wins. Softmax and norms are bandwidth-bound, so the win is fusing passes and vectorizing loads, not faster arithmetic.

### 2.4 LayerNorm vs RMSNorm (book 6.4)

LayerNorm needs mean *and* variance → two reductions (or one Welford single-pass that tracks mean and M2 together, numerically stable). **RMSNorm** (Zhang & Sennrich, arXiv:1910.07467) drops the mean-subtraction entirely — it normalizes by `sqrt(mean(x²) + ε)` only — so it needs **one** reduction instead of two, cutting 7–64% of runtime depending on shape. This is why modern LLMs (Llama, etc.) use RMSNorm: it is strictly less memory traffic for negligible quality cost. Build both; the warp-shuffle reduction is the shared core; the difference is one pass vs two.

### 2.5 Top-K (book 6.5)

The book's honest result (p.221): Top-K is *fundamentally hard to parallelize* — the naive find-next-max outer loop is sequential, and even an "optimized" parallel version shows ~0.07% memory throughput and "No Eligible: 70%" (warps stalled on data dependencies in the K-loop). The ladder (naive find-next-max → parallel min-heap load → parallel block reduction) teaches reduction patterns and the limits of parallelism, *not* a path to peak. Online softmax fused with Top-K (the ~5× case above) is the real systems win because it cuts memory passes.

### 2.6 GEMM — the compute-bound climb (book 6.6, frontier §4)

GEMM computes `C = A·B`, `A` is `M×K`, `B` is `K×N`, `C` is `M×N` → `2MNK` FLOPs over `(MK + KN + MN)·dtype` bytes → AI ≈ `2MNK / (4(MK+KN+MN))` ≈ **512 FLOP/byte at 4096³ FP16** (book p.222). That is far right of the ridge → compute-bound → the whole game is feeding the math units. The book switches to **FP16 for GEMM** (5 exp / 10 mantissa bits) for two stated reasons (p.221): halved bytes per element, and tensor cores provide 2–8× higher FP16 throughput — i.e., GEMM is explicitly the on-ramp to tensor cores.

**The CUDA-core ladder** (book 6.6, modeled on siboehm's SGEMM_CUDA) raises AI rung by rung:

| Rung | Mechanism | siboehm %-of-cuBLAS (Ampere FP32, 4092³) | Book FP16 1024³ (RTX 3090) |
|---|---|---|---|
| Naive | thread-per-output, uncoalesced B | ~1.3% | ~1.3% (300 GFLOP/s) |
| Global coalesce | thread remap so warp shares a row | ~8.5% | ~8.5% (2 TFLOP/s) |
| SMEM tiling | block-tile cached in shared memory | ~12.8% | ~13% (3 TFLOP/s) |
| 1D blocktiling | each thread computes TM=8 outputs (register reuse) | ~36.5% | ~36% (8.5 TFLOP/s) |
| 2D blocktiling | each thread computes 8×8=64 outputs | ~68.7% | ~68% (16 TFLOP/s) |
| Vectorized 128-bit | `LDG.E.128`/`STS.128`, `reinterpret_cast<float4>` | ~78.4% | ~78% (18 TFLOP/s) |
| Warptiling | warp-level tile hierarchy | ~93.7% | — (book stops at vectorized) |

**HONESTY FLAG — never cross-compare these setups.** siboehm's column is **RTX A6000 (Ampere), FP32, 4092³, cuBLAS = 23,250 GFLOP/s**. The book's column is **RTX 3090, FP16, 1024³** — and at 1024³ the whole problem fits in L1, so the book's naive and coalesced kernels are *identical* (1.23 ms, both cache-bound), which is itself the lesson "always profile at realistic sizes" (book p.239). The two ladders agree on *shape* (the %-progression) because the mechanism is the same; they do **not** share absolute numbers. Use siboehm's percentages as your CUDA-core target ladder and reproduce them on *your* GPU at a *large* size (≥4096³). cuBLAS itself runs at ~245 FLOP/byte effective and ~93–95% is the hand-written FP32 ceiling.

**The frontier extension (Hopper, §4):** the book stops at ~78–94% of an FP32/FP16 *CUDA-core* baseline. The frontier is the tensor-core regime, where the entire performance scale is ~10× higher and a new set of mechanisms governs:

- **Async copy.** Ampere `cp.async` (`LDGSTS`) copies GMEM→SMEM *bypassing registers*, enabling software-pipelined multistage prefetch (CUTLASS runs ~5–7 stages). Hopper's **TMA** (`cp.async.bulk.tensor`, sm_90) is a dedicated async bulk-copy engine: the host builds a `cuTensorMap`, a single elected thread issues the copy, completion is signaled by an **mbarrier transaction-byte count**, out-of-bounds is auto-predicated, SMEM writes are swizzled (bank-conflict-free), and the copy can multicast to a thread-block cluster. TMA is what frees the threads to do nothing but issue tensor-core instructions.
- **WGMMA** (`wgmma.mma_async`): a **warpgroup** (4 warps = 128 threads) issues one async tensor-core matmul. Operand `B` lives in SMEM addressed by 64-bit matrix descriptors; `A` in SMEM or registers; the accumulator in registers. Async handshake is `wgmma.fence` → `commit_group` → `wait_group<N>`. Introducing TC + TMA together is the ~10× jump (Colfax ladder: **32 → 317 TFLOP/s**).
- **Warp specialization.** Split warpgroups into **producers** (issue TMA loads) and **consumers** (issue WGMMA), and use `setmaxnreg` to reallocate registers asymmetrically (e.g., 24 for producers, 240/240 for consumers). **HONESTY FLAG:** this is fragile — an FP32-accumulate warp-specialized GEMM *without* the tuning spills to ~21 TFLOP/s; the *tuned* version recovers ~460. Register reallocation is load-bearing, not optional polish.
- **Persistent kernels + Stream-K.** Launch `grid = #SMs`, keep blocks resident, and use a **tile scheduler** to amortize launch cost and fight **wave quantization** (the partial final wave that wastes SMs). **Stream-K** (arXiv:2301.03598) splits work *fractionally along K* (vs split-K's fixed pieces) so every SM gets equal work regardless of problem geometry — the original paper reports up to **14× peak speedup** in the worst pre-Stream-K geometries and far more consistent performance across shapes; CUTLASS picks it via heuristic. **Ping-pong** persistent kernels overlap one tile's epilogue with the next tile's mainloop prologue.
- **Epilogue fusion.** Apply scale/bias/activation/cast **while the accumulator is still in registers** (CUTLASS Epilogue Visitor Trees), avoiding a separate memory-bound elementwise kernel — the same "fuse the pass" principle as online softmax, applied to GEMM's output.

**Measured ceilings (H100 PCIe, 8192³, vs ~750 TFLOP/s FP16 peak):** hand-written CuTe reaches ~531 TFLOP/s (~71%); best CUTLASS FP16 reaches ~630 TFLOP/s (~84% util) via a warp-specialized persistent cooperative kernel; CUTLASS is ≈ parity with cuBLAS (within ~1.8%). **HONESTY FLAG:** these are **H100 PCIe (114 SMs, ~750 TFLOP/s FP16 dense roof)** — *not* the H100 SXM (132 SMs, ~989 dense / 1,979 sparse). H100 SXM headline: ~3.35 TB/s HBM3, FP16/BF16 ≈ **989 dense / 1,979 sparse** TFLOP/s, FP8 ≈ **1,979 dense / 3,958 sparse**. **NVIDIA "marketing" TFLOPS usually include 2:4 sparsity = exactly 2× the dense number** — always state dense, always state which H100 SKU.

### 2.7 FP8 GEMM and the FP22 accumulation trap (DeepGEMM)

Hopper's FP8 WGMMA accumulates in a reduced-precision register format — effectively **~14-bit "FP22"** mantissa accumulation, not true FP32 — so a long-K FP8 GEMM accumulates error fast. **DeepGEMM** (DeepSeek, github.com/deepseek-ai/DeepGEMM, ~300 lines, JIT-compiled) fixes this with **two-level FP32 promotion**: accumulate a short chunk in the WGMMA FP22 path, then periodically promote and add into a true-FP32 register accumulator, bounding the error. It pairs this with **fine-grained scaling** — 1×128 per-token-block activation scales and 128×128 weight-block scales — so dynamic range is handled per tile rather than per tensor. **HONESTY FLAG:** reported **~1350–1550 FP8 TFLOP/s on H800 best-case, 1.4–2.7× a tuned CUTLASS baseline** — "best-case" and on H800 (the export H100 variant); verify on your hardware and your shapes before quoting.

---

## §3 The from-scratch build ladder

Rungs 0–6 are the **book's CUDA-core ladder** for all five kernels. Hardware: 4090/A6000/A100 is fine. Validate every rung against the oracle (§5) and profile every rung in Nsight (§6) *before* promoting. Keep the journal: *what I tested, the number, the next hypothesis* (`00_foundations.md` §7). The percentage targets are **targets to verify on your hardware** — reproduce the *shape* of the siboehm ladder, not the absolute numbers, and pin shape+dtype+GPU on every measurement.

**Rung 0 — Profiler + roofline harness.**
Before any kernel: set up `ncu` and a hand-roofline sheet. Reproduce the book's profiling command (book 6.7): `ncu --set full --section SpeedOfLight_HierarchicalDoubleRooflineChart --section MemoryWorkloadAnalysis --section Occupancy --print-summary per-kernel ./bench`. Build a `matplotlib` roofline plot and a sweep-to-CSV rig you reuse every rung. Lock clocks (`nvidia-smi -lgc`), warm up, CUDA-event timing. *Outcome:* the measurement spine. *Target:* none.

**Rung 1 — GEMV ladder to the bandwidth roof.**
naive → coalesced block-per-row + `warpReduceSum` → two-stage block reduction → vectorized `float4`. *Correctness:* match `torch.mv` / cublasSgemv, `rtol≈1e-5` FP32. *Profiling target:* SoL Memory% near 100%, sectors/request → ideal 4 for 32×4B, measured GB/s approaching peak HBM; compare vs cuBLAS GEMV. This is the decode kernel from A1 — close the loop on why decode is bandwidth-bound.

**Rung 2 — Softmax: two-pass → online → warp-shuffle → fused/vectorized.**
Implement the online running-max/denom recurrence (§2.3); then the warp-shuffle single-pass with correct `__ballot_sync` masks; then fuse with shared memory and vectorize loads. *Correctness:* match `F.softmax`, `rtol≈1e-3` FP16; **adversarial:** a single `+1e4` outlier (proves the running-max rescale), a row of identical values, an all-`-inf` masked row. *Profiling target:* naive-vs-warp comparison in Nsight; bandwidth-bound → aim ~peak HBM; reproduce the ~1.3× online-vs-two-pass traffic win.

**Rung 3 — RMSNorm + LayerNorm with warp reductions.**
Warp-shuffle mean (LN) and mean-of-squares (both); reciprocal-sqrt; fused load-to-shared. Implement LN with **Welford single-pass** for the variance. *Correctness:* match `F.layer_norm` / `F.rms_norm`, `rtol≈1e-3`. *Target:* quantify RMSNorm's one-reduction-vs-LN's-two win (book/§2.4: 7–64% depending on shape); both bandwidth-bound, aim ~peak HBM.

**Rung 4 — Top-K: naive → min-heap → parallel block reduction.**
Build all three; *measure the failure*. *Correctness:* exact top-k indices/values vs `torch.topk` on no-tie inputs; define tie-breaking and test it. *Target/finding:* reproduce the book's honest result — the sequential K-loop caps throughput (~0.07% memory, high "No Eligible" stall); document *why* Top-K resists parallelism. Then fuse online-softmax + Top-K and show the ~5× memory-traffic win over separate passes.

**Rung 5 — GEMM CUDA-core ladder, part 1: naive → coalesce → SMEM tile.**
naive (uncoalesced B) → thread-remap coalescing (`cRow = blockIdx.x*BS + threadIdx.x/BS`) → shared-memory block tiling. *Correctness:* match `torch.matmul` FP16, `rtol≈1e-2..1e-3` on a magnitude range. *Targets (verify on your GPU, large size ≥4096³):* ~1.3% → ~8.5% → ~12.8% of cuBLAS (siboehm Ampere FP32 ladder). Reproduce the book's **bank-conflict exercise** (which `As`/`Bs` accesses conflict; fix with XOR swizzle `(x^y)`, no padding) and the "Stall MIO Throttle" diagnosis (too many SMEM instructions vs FMAs).

**Rung 6 — GEMM CUDA-core ladder, part 2: 1D → 2D blocktiling → vectorized.**
1D blocktiling (each thread computes `TM=8` outputs, register reuse) → 2D blocktiling (`8×8=64` outputs/thread, 64 register accumulators) → vectorized 128-bit tile loads (`reinterpret_cast<float4>`, `LDG.E.128`/`STS.128`). *Targets:* ~36.5% → ~68.7% → ~78.4% of cuBLAS (siboehm), and as a stretch, **warptiling → ~93.7%** (the hand-written FP32 ceiling). *Profiling target:* watch SoL flip from Memory-bound to Compute-bound across the rungs (book p.239: memory throughput *drops* 86%→31% and that is **good** — you are now compute-bound, more FMAs per byte). **This is the inflection where GEMM becomes compute-bound and the only remaining lever is tensor cores → §4.**

> The book's own GEMM profiling table (listing 6.33, RTX 3090, 1024³ FP16) is your template for presenting the ladder: Duration / Memory% / L1 hit / Compute% / Occupancy side-by-side per variant, with a critical-analysis paragraph per row. Note the book's deliberate trap — at 1024³ naive==coalesced because it fits L1. Run *your* ladder at ≥4096³ so the optimizations actually separate.

---

## §4 Frontier-2026 core (required)

This is the point of the assignment: the ~10× tensor-core gap the book leaves open, then the FP8 frontier. **Hardware: 1× H100 (sm_90a) is required** — WGMMA and TMA do not exist below Hopper (`00_foundations.md` §1). Build at least **4.1 → 4.4 to completion** (async-copy → WGMMA → warp-specialized persistent → epilogue fusion); **4.5 (FP8/DeepGEMM)** and **4.6 (CUTLASS parity / Triton floor)** are required to be *demonstrated and benchmarked*. Compile with `-arch=sm_90a` (the trailing `a` is mandatory for WGMMA). Pin every number with shape+dtype+SKU and the dense-vs-sparse, PCIe-vs-SXM caveats.

**4.1 Async copy → multistage pipeline.**
Replace synchronous GMEM→SMEM tile loads with `cp.async` (`LDGSTS`, bypasses registers), then build a multistage software pipeline (prefetch stage `k+1` while computing stage `k`; CUTLASS runs ~5–7 stages). On Hopper, do it again with **TMA**: build the `cuTensorMap` on the host, elect one thread to issue `cp.async.bulk.tensor`, signal completion via an **mbarrier transaction-byte** wait, and write to swizzled SMEM. *Correctness:* bit-identical tile contents vs the synchronous loader; test an out-of-bounds tile (TMA auto-predicates). *Target:* the mainloop is no longer stalled on loads — Nsight WarpStateStats shows load-stall cycles drop; this is the *enabler* for the WGMMA jump, not yet the jump.

**4.2 WGMMA tensor-core mainloop.**
Replace the FMA inner loop with `wgmma.mma_async` issued by a warpgroup (128 threads): operand `B` in SMEM via 64-bit matrix descriptors, accumulator in registers, async handshake `wgmma.fence`/`commit_group`/`wait_group<N>`. Combine with 4.1's TMA. *Correctness:* match `torch.matmul` FP16, `rtol≈1e-2`; **adversarial:** K not a multiple of the WGMMA-K, a single large outlier (checks accumulation), the masked remainder tile. *Target:* reproduce the Colfax ~10× jump (**~32 → ~317 TFLOP/s** on a comparable problem) — this is the single biggest rung in the assignment. State your H100 SKU and the dense FP16 roof you are measuring against.

**4.3 Warp-specialized + persistent + Stream-K.**
Split warpgroups into TMA **producers** and WGMMA **consumers**; use `setmaxnreg` for asymmetric register allocation (e.g., 24/240/240). Make the kernel **persistent** (`grid = #SMs`) with a tile scheduler, and add **Stream-K** K-splitting (arXiv:2301.03598) to kill wave quantization. Add **ping-pong** to overlap epilogue/prologue across tiles. *Correctness:* same oracle; **adversarial geometries** — a skinny `M`, a `K` that wave-quantizes a naive grid, a problem size that leaves a partial final wave (Stream-K should flatten the cliff). *Target:* climb from ~317 toward the expert band; **HONESTY FLAG** — verify you did *not* land in the un-tuned ~21 TFLOP/s spill (the warning sign that `setmaxnreg`/staging is wrong); the tuned target is the hand-CuTe ~531 (~71% PCIe) → CUTLASS-class ~630 (~84% PCIe) band. Show the Stream-K speedup is *largest on the worst geometry* (the paper's point), not on the square case.

**4.4 Epilogue fusion.**
Fuse the output path — scale, bias, activation (GELU/SiLU), and cast-to-output-dtype — **while the accumulator is in registers**, instead of a separate kernel. Model it on CUTLASS Epilogue Visitor Trees. *Correctness:* match a reference `matmul`-then-elementwise, same tolerance; verify the fused cast matches the two-kernel cast bit-for-bit where dtypes allow. *Target:* eliminate the separate memory-bound epilogue kernel — show the saved HBM round-trip in `nsys` (one kernel, not two) and the end-to-end time drop.

**4.5 FP8 fine-grained-scaled GEMM (DeepGEMM-style).**
Implement an FP8 (E4M3) GEMM with **fine-grained scaling** (1×128 per-token-block activation scales, 128×128 weight-block scales) and **two-level FP32 promotion** to defeat Hopper's ~14-bit FP22 WGMMA accumulation (§2.7). *Correctness:* this is the hardest numerics rung — compare **SQNR** (not element-wise; `≈6.02·bits + const` dB floor, `00_foundations.md` §5) and end-to-end task metric; explicitly show that *without* two-level promotion, long-K error blows past tolerance (the whole reason DeepGEMM exists). *Target:* beat a tuned CUTLASS FP8 baseline; DeepGEMM reports **1.4–2.7× and ~1350–1550 TFLOP/s on H800 best-case** — treat as a target to verify, state your SKU, and remember FP8 SXM dense roof is ~1,979 (sparse ~3,958 — do not quote the sparse number for a dense kernel).

**4.6 Library floor & ceiling (required benchmark).**
Benchmark your best hand kernel against the **floor** (`torch.compile` / Triton `tl.dot`, which is ~parity with cuBLAS on Ampere FP16 and ~95% of CUTLASS on Hopper FP8 with TMA) and the **ceiling** (cuBLAS + CUTLASS 3.x). Read the matching CUTLASS/CuTe source: map its **Layout = (Shape, Stride)** algebra ("layouts are functions from integers to integers"), `CollectiveMainloop`, `CollectiveBuilder`, and Epilogue Visitor Trees onto the primitives you built by hand. *Deliverable judgment:* state plainly whether hand-writing beat the floor and by how much it trails the ceiling — and whether that gap is worth the code (`00_foundations.md` §8, axis 5).

> **Cross-reference, don't duplicate:** A1 already covers inference/KV-cache/PagedAttention/continuous-batching/speculative-decoding/MLA. A2 supplies the *kernels* those systems call (GEMV for decode, GEMM for prefill, fused norms/softmax). A4 takes attention/FlashAttention; A3 takes the WMMA→WGMMA tensor-core deep dive at full length; A5 takes quantization numerics. Build the GEMM here; do not re-derive FlashAttention.

---

## §5 Correctness & numerics

Oracle-first, always, per `00_foundations.md` §5. The oracle is the slow-but-trusted reference; the comparison uses an explicit, dtype-appropriate tolerance; adversarial inputs come *before* performance numbers.

- **GEMV / GEMM (FP32):** vs `torch.mv` / `torch.matmul`, `rtol≈1e-5`. **(FP16/BF16):** `rtol≈1e-2..1e-3`, check *relative* error across magnitudes (a small abs-error tolerance hides large relative error on small outputs).
- **Softmax:** vs `F.softmax`; adversarial — a `+1e4` outlier (the running-max rescale is the test), all-equal row, fully-masked `-inf` row (must produce uniform / not NaN).
- **RMSNorm/LayerNorm:** vs `F.rms_norm` / `F.layer_norm`; adversarial — zero row (ε path), single outlier, the row whose length is not a multiple of the warp/tile.
- **Top-K:** exact indices+values on no-tie inputs; *define and test tie-breaking*; this is where off-by-one in the heap hides.
- **GEMM remainder tiles:** the classic kernel bug lives in the masked remainder and the `K` that isn't a multiple of the tile/WGMMA-K. Test `M,N,K` each non-multiples; test a skinny `M=1` (degenerates to GEMV) and `M` smaller than a tile.
- **FP8 GEMM:** compare **SQNR** + task metric, *not* `allclose`; the load-bearing adversarial test is **long-K with two-level promotion OFF vs ON** — show error exceeds tolerance without promotion (proves you understand the FP22 trap), within tolerance with it.
- **The dependency test** (`00_foundations.md` §5): perturb one input element, confirm only the outputs that should depend on it move — catches batch-dim and masking bugs that `allclose` misses.

Rule (foundations): **if the kernel can't match the oracle on one small adversarial input, no performance number from it is real.**

---

## §6 Profiling & performance

Nsight is the spine of this assignment. The full discipline is in `00_foundations.md` §3/§4/§6 — do not re-derive it; apply it. The loop every rung: **predict on paper → profile → read Speed-of-Light first → drill into the bound → change one variable → record the delta.**

- **Speed-of-Light first:** whichever of Compute% / Memory% is nearer 100% is your bound. The book's whole-ladder check (listing 6.33): present Duration / Memory% / L1 hit / Compute% / Occupancy per variant side-by-side, with a one-paragraph critical analysis per row. The *signature of progress* on the GEMM ladder is Memory% **dropping** (86%→31%) as you go compute-bound — that is the win, not a regression (book p.239).
- **Roofline placement:** plot each kernel against the H100 FP16 ridge (≈295 FLOP/byte) / FP8 ridge (≈590). Memory-bound kernels (GEMV/softmax/norm/Top-K) live left of the ridge → their ceiling is `AI × peak_BW`; GEMM crosses to the right → ceiling `peak_FLOPs`.
- **The metrics that name the bound:** sectors/request (coalescing — ideal 4 for 32×4B; siboehm naive 15 GB/s → coalesced ~110 GB/s), `l1tex__data_bank_conflicts_*` (SMEM conflicts → XOR swizzle), the WGMMA pipe utilization (is the tensor-core pipe actually busy, or starved by shallow TMA staging?), WarpStateStats stall reasons ("Stall MIO Throttle" = too many SMEM ops; "No Eligible" = data-dependency serialization, the Top-K killer).
- **Occupancy is NOT the goal.** Volkov, "Better Performance at Lower Occupancy": ~100% of peak at ~8% occupancy with ~3-way ILP. The book's data agrees — 2D blocktiling runs at ~17% occupancy and is the *fast* rung. Report occupancy, but optimize the *bound*, not occupancy.
- **Hygiene (foundations §4):** locked clocks (`nvidia-smi -lgc`), warmup discarded, median of ≥100 CUDA-event-timed iterations, **CUDA graphs for the short kernels** (GEMV/softmax/norm launch overhead can dominate at small sizes), and **pin shape+dtype+SKU on every number** ("630 TFLOP/s" is meaningless without "FP16-in/FP32-acc, 8192³, H100 **PCIe**, dense").

---

## §7 Stretch goals (competition-grade)

- **Warptiling GEMM → ~93.7% of cuBLAS** (siboehm's top FP32 rung) on CUDA cores — close the hand-written ceiling before touching tensor cores.
- **ThunderKittens-style GEMM** ("tiles not threads", 16×16 base tile): a <100-line GEMM reaching ~855 TFLOP/s (~86% peak) on H100 (ICLR'25, arXiv:2410.20399). Build it, compare to your hand kernel, and judge the abstraction.
- **Triton autotuned GEMM** (`@triton.autotune` over `BLOCK_M/N/K`, `num_warps`, `num_stages`; `tl.dot`→WGMMA on Hopper) — reproduce the "~parity on Ampere FP16, leaving Hopper perf on the table in 2024, narrowing 2025–26 with TMA + auto warp-specialization" story with your own measurements.
- **Full DeepGEMM reproduction** — JIT, fine-grained scaling, two-level promotion — and push toward the ~1350–1550 H800 best-case band; document where you fall short and why.
- **Stream-K ablation:** construct the worst-case geometry for a fixed grid and show Stream-K's largest win there (toward the paper's 14× tail), vs negligible win on the square case — the judgment is *when* Stream-K matters.
- **Fused norm-into-GEMM-epilogue** (or GEMV-into-softmax) end-to-end micro-fusion, measured against the unfused two-kernel baseline.
- **MoE grouped GEMM** (the production kernel every 2026 MoE layer runs twice): ragged per-expert shapes — a different M per expert, one launch, no padding waste — via CUTLASS grouped GEMM or DeepGEMM's m-grouped path, on your §4 WGMMA+TMA+persistent scaffold. *Judgment to demonstrate:* per-expert token counts are load-imbalanced, so the tile scheduler, not the MMA, is the hard part. This is the kernel A6's all-to-all feeds.

---

## §8 Deliverables & definition of done

1. **Code:** the five book kernels at every rung (GEMV, softmax, RMSNorm+LayerNorm, Top-K, GEMM through vectorized), each with its oracle test; the Hopper GEMM (async-copy → WGMMA → warp-specialized persistent + Stream-K → fused epilogue); the FP8 fine-grained-scaled GEMM with two-level promotion. Compiled `-arch=sm_90a` for the Hopper rungs.
2. **The profiler/roofline harness** (reusable) and the experiment plots: the per-kernel roofline placement; the GEMM %-of-cuBLAS ladder (CUDA-core rungs 0–6 *and* the Hopper rungs) as a bar chart with shape+dtype+SKU pinned; the Speed-of-Light Memory%→Compute% flip across the GEMM ladder; the FP8 SQNR-with/without-two-level-promotion plot; the Stream-K-vs-geometry curve.
3. **Design note (2–3 pages): "Climbing from coalescing to the Hopper ceiling — where every rung's bottleneck was, how I found it, and whether hand-writing beat the library."** Lead with the roofline and the AI-raising principle; walk each rung with its before/after number and the Nsight evidence for the bound; carry the honesty flags (dense-vs-sparse, PCIe-vs-SXM, best-case); end with the explicit floor/ceiling verdict (did you beat `torch.compile`/Triton; how far from cuBLAS/CUTLASS; is the gap worth the code). Write it for a frontier-lab peer (`00_foundations.md` §7).

**Done when:** every rung matches its oracle on random *and* adversarial inputs at a stated tolerance; the GEMM ladder reproduces the siboehm %-progression's *shape* on your GPU at ≥4096³ with hygiene intact; the Hopper WGMMA+TMA kernel shows the ~10× jump and the warp-specialized persistent kernel reaches the expert band (and provably did *not* land in the un-tuned spill); the FP8 kernel demonstrates the two-level-promotion numerics win; and the design note correctly states the floor/ceiling verdict with evidence and the honesty flags intact.

---

## §9 Principal-level rubric

- **Pass:** the five CUDA-core kernels work and are token/tolerance-correct vs the oracle; the GEMM ladder runs through 2D blocktiling and you can place each kernel on the roofline.
- **Strong:** + the GEMM ladder reproduces the siboehm %-shape (to vectorized ~78%, ideally warptiling ~93%) at a realistic size with hygiene; the Hopper WGMMA+TMA rung lands the ~10× jump; you predict each kernel's bound before profiling and the prediction holds.
- **Principal-grade:** all of §4 demonstrated — warp-specialized persistent GEMM in the expert band *with the Nsight evidence that it isn't the un-tuned spill*, Stream-K's win shown largest on the worst geometry, fused epilogue saving the HBM round-trip, and the FP8 kernel's two-level-promotion numerics proven; you **beat the `torch.compile`/Triton floor or state precisely why you can't**, and you can name the remaining gap to cuBLAS/CUTLASS at the level of the specific instruction/tile (e.g., "we trail CUTLASS by 11% because our TMA staging is 4-deep not 6 and the WGMMA pipe shows 71% util — here's the SoL section"). The differentiator is **judgment**: knowing that four of these five kernels can only ever reach the bandwidth roof, that the GEMM win is tensor cores + scheduling not cleverer FMAs, and that a 95%-of-cuBLAS kernel that took three weeks and works on one shape may be the *wrong* answer.

---

## §10 References (see `references.md` for full list)

- Boehm, **"How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"**, siboehm.com/articles/22/CUDA-MMM + github.com/siboehm/SGEMM_CUDA (the CUDA-core ladder %s).
- **Colfax CUTLASS tutorials** — WGMMA, TMA, GEMM pipelining, persistent kernels + Stream-K (the Hopper ladder, 32→317 TFLOP/s).
- **NVIDIA CUTLASS 3.x + CuTe** docs (Layout algebra, CollectiveMainloop/Epilogue, CollectiveBuilder); **PyTorch CUTLASS ping-pong** blog (warp-specialized persistent, H100 PCIe ~531/630 TFLOP/s).
- Osama et al., **Stream-K**, arXiv:2301.03598 (fractional-K load balancing, up to 14× peak).
- Milakov & Gimelshein, **"Online normalizer calculation for softmax"**, arXiv:1805.02867.
- Zhang & Sennrich, **"Root Mean Square Layer Normalization" (RMSNorm)**, arXiv:1910.07467.
- Spector et al., **ThunderKittens**, arXiv:2410.20399 (ICLR'25; "tiles not threads", <100-line GEMM ~855 TFLOP/s).
- **DeepGEMM** (DeepSeek FP8, two-level FP32 promotion, fine-grained scaling), github.com/deepseek-ai/DeepGEMM.
- **Triton matmul tutorial** (`tl.dot`, `@triton.autotune`); **NVIDIA Nsight Compute Profiling Guide**; Volkov, **"Better Performance at Lower Occupancy"** (GTC 2010).
