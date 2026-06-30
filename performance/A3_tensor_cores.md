# A3 — Tensor Cores: From WMMA Fragments to Blackwell Tensor Memory

> **Book chapter:** 7 (Tensor Cores) · **Frontier thesis:** A tensor core is the *manual 2D-register-tiling pattern of A2 frozen into silicon* — it amortizes instruction-issue energy across thousands of FMAs. Every generation roughly doubled throughput/SM by forcing operands *out of registers* and making the MMA *asynchronous*: the issue thread-count collapsed quadpair(8)→warp(32)→warpgroup(128)→single-thread(1), and the accumulator's home moved registers→registers→**Tensor Memory**. Master that one arc and the whole instruction zoo (`mma.sync`, `wgmma`, `tcgen05`) is legible.
>
> **Primary hardware:** rungs 0–2 any GPU with tensor cores (RTX 4090/5090, A100); **rungs 3–4 require 1× H100 (sm_90a)** — WGMMA + TMA exist *only* on Hopper; **§4 B200 track requires 1× B200 (sm_100a)** — `tcgen05`/TMEM/NVFP4-MMA exist *only* on datacenter Blackwell. · **Est. time:** 2–2.5 weeks · **Prereqs:** A2 (you have a hand-tiled GEMM at ~85% of CUDA-core peak); book ch6–7; the `sm_XX`-gates table and benchmarking discipline in `00_foundations.md`.

---

## §0 Why this matters

The book's chapter 7 ends on a number that should reorganize how you think: on an **H100** at 4096³ FP16, the best hand-tuned *CUDA-core* GEMM from chapter 6 reaches ~43 TFLOPS; a basic **WMMA** kernel hits 71 (1.7×); and a fully-pipelined **WGMMA** kernel reaches **618 TFLOPS — 87% of cuBLAS** (~713), an **8.7× jump** over WMMA (book §7.2–7.3 and Table 7.2; all H100, 4096³ FP16). None of that 8.7× is a better *algorithm*. It is the same `D = A·B + C` inner product, run on hardware redesigned four times to stop wasting energy on instruction issue and register traffic. If you cannot reach the tensor cores you forfeit the bulk of a modern GPU's FLOPs — and the portable `nvcuda::wmma` API physically *cannot* reach Hopper's WGMMA or Blackwell's `tcgen05`: the book's WMMA kernel tops out at 71 TFLOPS where optimized WGMMA reaches 618 and the silicon offers ~989 *dense* FP16, so staying on WMMA leaves most of an H100's tensor throughput on the floor.

This is the assignment where you stop treating the tensor core as a cuBLAS implementation detail and start treating it as an instruction you *issue, feed, and synchronize by hand*. Every frontier kernel that matters in 2026 — FlashAttention-3, the GEMMs under vLLM/SGLang/TensorRT-LLM, NVFP4 inference — is built on the WGMMA/`tcgen05` machinery you build here. By the end you will look at a GEMM that hits 60% of peak and say, with a profile, "your accumulator is spilling, your TMA isn't deep enough to hide HBM latency, and you're not warp-specialized — here's the 25 points you're leaving on the floor."

The hard part is **not the matmul**. It is the asynchronous, descriptor-driven, barrier-synchronized *machinery* around it. That machinery is the actual content.

---

## §1 Learning objectives

You can:

1. **Derive** why a tensor MMA exists from an energy/instruction-issue argument, and explain the throughput-doubling mechanism of each generation (operands leave registers; issue goes async; issue-thread-count shrinks 8→32→128→1).
2. **Implement** a WMMA GEMM with `wmma::fragment` + `cp.async` double-buffering and hit **~40–60% of the dtype's dense peak** on Ampere/Ada; explain why the fragment layout is opaque and why WMMA caps out well below silicon peak.
3. **Implement** an Ampere `mma.sync` + `ldmatrix` + swizzled-SMEM GEMM and reach **~60–75% of dense peak**, demonstrating that you can hand-place the tensor-core register layout the WMMA API hides.
4. **Read and write PTX** for a tensor-core instruction: compile with `nvcc -ptx`, decode every dot-separated qualifier of `wgmma.mma_async`, and hand-encode a 64-bit SMEM **matrix descriptor** (address/leading-byte-offset/stride-byte-offset/swizzle).
5. **Implement** (H100) a WGMMA + TMA multistage GEMM — `m64nNk16` atoms, 64-bit descriptors, `mbarrier` producer/consumer, a GMMA SMEM swizzle atom, a 3-stage circular buffer — and reach **~70–85% of H100 989 TF**; with warp-specialization + ping-pong, push toward cuBLAS.
6. **Articulate and demonstrate** the `register → SMEM → TMEM` and `sync → async` arc: explain why WGMMA puts the accumulator in *registers* but `tcgen05` puts it in *Tensor Memory*, and why that "does for compute what TMA did for copy."
7. **(B200)** Implement a `tcgen05`/UMMA GEMM with the accumulator in **TMEM** and `tcgen05.commit`-on-mbarrier completion, and an **NVFP4 block-scaled** MMA (block-16 + UE4M3 scale via `tcgen05.cp`); state precisely what a consumer RTX 5090 (sm_120) **cannot** do.
8. **Distinguish [documented] from [unknown]** across the Blackwell stack: cite the TMEM *contract* (256 KB/SM, 128 lanes × 512 cols, column-granular alloc, 32-lane-per-warp access rule) while flagging that TMEM physical latency/banking/datapath are **not** publicly documented.

---

## §2 First-principles theory

### 2.1 Why a tensor MMA exists at all (the energy argument)

A scalar FMA on a CUDA core does ~2 FLOPs but pays the full overhead of *issuing an instruction*: fetch, decode, register-file read/write, scheduling. At the 7nm-and-below node, **moving the operands and issuing the instruction costs far more energy than the multiply-add itself.** The fix is amortization: issue *one* instruction that performs thousands of FMAs against operands the hardware reads in a tight, fixed pattern. A WMMA 16×16×16 op is **4,096 FMAs per instruction** (book §7.2.3); a WGMMA `m64n64k16` is **262,144 FMAs** (book §7.3); the largest Blackwell UMMA atom `128×256×16` is ~**512K FMAs** — 2× the largest WGMMA atom. Each generation pushed the FMAs-per-issue up by widening the tile, which *required* pulling operands out of the (finite) register file and into SMEM/TMEM, which *required* making the instruction asynchronous so the copy and the compute could overlap. That is the whole story; everything below is mechanism.

The book's own ladder makes the payoff concrete (4096³ FP16, **H100**; §7.2–7.3 and Table 7.2): the chapter-6 hand-tuned CUDA-core kernel tops out at ~**43 TFLOPS** (this is where A2 ends) → **WMMA 71** (1.7× — only the hardware premium over an already-good CUDA-core kernel) → WGMMA-basic **318** → +larger-tiles **433** → +TMA-async **504** → +max-tile 3-stage **618** = **87% of cuBLAS (~713)**. The 8.7× from WMMA→WGMMA-max is *all* async/descriptor machinery, not arithmetic — and note the *basic* WMMA kernel (71) sits far below H100's peak precisely because it is synchronous and unpipelined; the climb to 618 is the whole point of this assignment.

### 2.2 The instruction-evolution throughline (memorize this table)

| Instr | Arch (`sm_`) | Issue threads | Sync? | A operand | B operand | Accumulator | New machinery |
|---|---|---|---|---|---|---|---|
| `mma.sync` | Volta/Turing/Ampere (70/75/80) | **warp (32)** | **sync** | registers | registers | registers | `ldmatrix`, `cp.async` (Ampere) |
| `wgmma.mma_async` | Hopper (**90a**) | **warpgroup (128)** | **async** | SMEM *or* regs | **SMEM only** | **registers** | 64-bit SMEM descriptors, TMA, mbarrier, GMMA swizzle atoms |
| `tcgen05.mma` (CUTLASS: **UMMA**) | Blackwell-DC (**100a**) | **single thread (1)** | **async** | SMEM (*or* TMEM) | SMEM | **Tensor Memory (TMEM)** | `tcgen05.alloc/ld/st/cp`, `tcgen05.commit`, block-scaling |

The two spines to carry through the entire assignment:

- **`sync → async`.** `mma.sync` blocks all 32 threads until the MMA retires (book §7.2.4). `wgmma.mma_async` *issues and returns* — you fence with `wgmma.fence` / `commit_group` / `wait_group<N>`; choosing `N>0` leaves N groups in flight, which is exactly the knob FlashAttention-3 uses to overlap the softmax with the next MMA. `tcgen05.mma` is *fully* async and completes by *arriving on an mbarrier* (`tcgen05.commit`), so the MMA is decoupled from CTA execution entirely.
- **`register → SMEM → TMEM`.** Volta-era operands live in thread registers (book §7.2). Hopper forces operand **B into SMEM** and passes it as a **64-bit descriptor** (start addr, leading/stride byte offsets, swizzle mode) — threads never address the data, the tensor core walks it. Blackwell goes further: the *accumulator* leaves registers for **on-chip Tensor Memory**. Colfax's framing is the one to internalize — *"UMMA requires no registers for data… single-thread launch… further decouples MMA from CTA execution"*; **TMEM does for compute what TMA did for copy.**

### 2.3 Tensor Memory (TMEM) — the contract, and what's unknown

[documented] (NVIDIA PTX ISA, Tensor Memory section; CUTLASS Blackwell docs; Colfax "GEMM with Tensor Memory"):

- **256 KB per SM**, organized as **128 lanes × 512 columns** of 32-bit cells — *the same size as the SM register file*. The accumulator lands in TMEM in a **transparent row-major layout** (no thread-value "fragment" gymnastics), but **must be copied to registers before any post-processing** (epilogue, activation, requant).
- **Dynamic, column-granular allocation** via `tcgen05.alloc` / `tcgen05.dealloc` (power-of-two ≥ 32 columns, issued from a single warp).
- **HARD RULE:** each warp can access only **32 of the 128 lanes**. Draining a full accumulator therefore requires a **full warpgroup (128 threads)** in the epilogue. Get this wrong and you read garbage from 3/4 of the tile.
- **Movement instructions:** `tcgen05.ld` (TMEM→regs), `tcgen05.st` (regs→TMEM), `tcgen05.cp` (SMEM→TMEM, used to stage scale-factors); ordering via `tcgen05.fence`, completion via `tcgen05.wait::ld`.

[unknown] (NVIDIA documents the *contract*, not the silicon — flag this every time you reason about it): **TMEM physical latency, bandwidth, internal banking, and the cycle-level datapath are not public.** The microbenchmarking paper "Microbenchmarking NVIDIA Blackwell" (arXiv:2512.02189) is the place to look for empirical estimates, but treat them as measured-not-specified.

### 2.4 Shapes, datatypes, and the honesty flags

**WGMMA shapes [documented]:** M is **always 64**; N is a multiple of 8 in **[8, 256]**; K=16 for FP16/BF16 (32 bytes), K=32 for FP8/INT8. SMEM operands must use one of **8 canonical GMMA layout atoms** (swizzle 16/32/64/128 B). **WGMMA is deprecated on Blackwell** (runs for back-compat; won't reach `tcgen05` peak).

**`tcgen05` shapes [documented]:** `64×N×16` or `128×N×16`, N ≤ 256; largest atom **128×256×16** = 2× the largest WGMMA atom. The PTX skeleton: `tcgen05.mma.cta_group.kind [d-tmem], a-desc, b-desc, idesc, enable-input-d`, where `kind ∈ {f16, tf32, f8f6f4}` and the **32-bit `idesc`** packs dtypes/sparsity/transpose/scale-IDs. **1-SM vs 2-SM:** a *CTA pair* (2 adjacent CTAs = 1 TPC = 2 SMs); `cta_group::2` doubles `MMA_M` to 256, each CTA holds half of A and half the output, B is split, and only the *leader* issues.

**Datatypes by generation [documented]:** Volta FP16 → Turing +INT8/INT4 → Ampere +BF16/TF32/2:4-sparsity → Hopper +FP8(E4M3/E5M2) → Blackwell +FP6(E3M2/E2M3)/FP4(E2M1), MX-formats + NVFP4, **hardware block-scaling**. Block-scaling formats (Colfax Part 4): `mxf8/mxf6/mxf4` = block **32** + **UE8M0** (power-of-two) scale; **`nvf4` = block 16 + UE4M3 scale** (finer block, *mantissa-bearing* scale → more accurate). Scale factors are staged into TMEM via `tcgen05.cp`, duplicated across 4 lane-quadrants.

**HONESTY FLAGS — carry these verbatim in your writeup:**

- **Dense vs sparse TFLOPS.** NVIDIA datasheets, *dense / sparse(2:4)*: A100 **312 / 624** FP16 (no FP8). H100 **989 / 1,979** FP16, **1,979 / 3,958** FP8. B200 **2,250 / 4,500** FP16, **4,500 / 9,000** FP8, **9,000 / 18,000** FP4. Keynote "PFLOPS" almost always quotes the **sparse** number. A real, tuned B200 BF16 GEMM lands around **~1,476 TFLOPS ≈ 65% of the 2,250 dense spec** (≈98% of that machine's cuBLAS). Always say *dense* and always say *% of which peak*.
- **FP6 ≠ FP4 rate.** On Blackwell, **FP6 runs at the FP8 rate, not the FP4 rate** — FP6 and FP8 share datapath circuits; **only FP4 is the 2× tier**.
- **Consumer Blackwell is not datacenter Blackwell.** The **RTX 5090 is `sm_120`, B200 is `sm_100`.** The 5090 has **FP4 *inference*** but **NO `tcgen05`, NO TMEM, NO NVFP4-MMA, NO real NVLink** (see `00_foundations.md` gates). The 5090 *cannot* run any §4 B200-track kernel. Do not let "Blackwell" on the box fool you.

### 2.5 The supporting machinery (one paragraph each)

- **TMA** (Tensor Memory Accelerator): a per-SM DMA engine. You build a **`CUtensorMap`** descriptor *on the host* (tensor base, per-dim shape/stride, tile box, swizzle) and issue `cp.async.bulk.tensor` from a **single thread**; it computes addresses, clamps OOB, applies the swizzle on write, and can multicast. Threads become *supervisors*, not data movers (book §7.5.2).
- **mbarrier**: a 64-bit shared async-transaction barrier. Unlike `__syncthreads()` (threads only), it tracks a **byte-transaction count** *and* a thread-arrival count and a **phase bit** — so it can release only when *both* "128 threads arrived" *and* "the TMA delivered N bytes" are true (book §7.3.4, the producer/consumer glue). On Blackwell, `tcgen05.commit` **arrives on an mbarrier** to signal MMA completion.
- **Async pipeline** = a **circular SMEM buffer** (2–N stages) with a producer (TMA) and consumer (WGMMA/UMMA), optionally **warp-specialized** (dedicated producer warpgroup) and/or **ping-pong** (two consumer warpgroups alternating). The book's 3-stage buffer is the minimal version (504→618 TFLOPS from buffering alone).
- **Swizzling** = an **XOR permutation** of SMEM addresses (CuTe `Swizzle<B,M,S>`, e.g. 128-byte) that makes the tensor-core's fixed access pattern bank-conflict-free. You either pick a canonical GMMA atom (Hopper) or let the TMA swizzle on write.

### 2.6 The abstraction ladder (what to use when)

- **WMMA** (`nvcuda::wmma`): portable to any tensor-core GPU, but **cannot emit WGMMA or `tcgen05`** → caps ~63% of H100 peak. Right for *learning fragments* and for Ampere/Ada production where you don't need the ceiling.
- **CUTLASS 3.x / CuTe**: an **MMA Atom** = a PTX instruction + traits; `make_tiled_mma` + `cute::gemm`. Blackwell adds `UMMA::*`, `TMEM::Allocator`, block-scaled tiled MMA. **GOTCHA:** calling `cute::gemm` from a *single thread* **deadlocks** on Blackwell — CUTLASS internally *elects* the issuing thread, so you must call it from a full warp.
- **CuTe DSL / CUTLASS 4.x** (Python) — same atoms, Python authoring.
- **ThunderKittens**: a 16×16-tile DSL covering WGMMA *and* `tcgen05`, with an escape hatch down to PTX.
- **Drop to raw PTX** only for instruction variants the atoms don't expose. (This assignment makes you do it *once* on purpose, so the abstractions stop being magic.)

---

## §3 The from-scratch build ladder

Cumulative. Each rung's PERFORMANCE TARGET is **% of the *dense* peak for that dtype on that GPU** (the only honest unit). Validate against the §5 oracle and lock clocks (`00_foundations.md`) before recording any number. Keep the journal.

> **Hardware reality:** rungs 0–2 run on any tensor-core GPU (4090/5090/A100). **Rungs 3–4 REQUIRE an H100** — there is no WGMMA/TMA emulation. Don't rent the H100 until rung 2 is correct and fast on cheap hardware.

**Rung 0 — Naive SMEM GEMM baseline (re-anchor from A2).**
Bring forward your best A2 CUDA-core GEMM (2D-register-tiled + vectorized) and a plain SMEM-blocked GEMM. *Mechanism:* none new — this is the *floor* the tensor cores must beat, and the apples-to-apples problem size (4096³, fixed dtype). *Correctness:* token-/element-exact vs a cuBLAS/`torch.matmul` reference at FP32-accumulate tolerance. *Target:* reproduce the book's regime — the chapter-6 CUDA-core ceiling of ~**43 TFLOPS on an H100** (only ~4–5% of H100 dense FP16; the point is that even a *good* CUDA-core kernel is an order of magnitude from the tensor-core regime). *Caveat:* if your A2 kernel isn't already near its CUDA-core roofline, fix that first — you cannot interpret the tensor-core jump otherwise.

**Rung 1 — WMMA GEMM + `cp.async` double-buffer (Ampere/Ada).**
Build the book's §7.2 kernel: a 128×128 block tile, 8 warps each owning a 2×4 grid of 16×16×16 ops, `wmma::fragment<matrix_a/ b/ accumulator>`, vectorized `int4` tile loads to SMEM, `wmma::load_matrix_sync` → `wmma::mma_sync` → `wmma::store_matrix_sync`. Then add **`cp.async` (LDGSTS)** double-buffering so the next K-tile's GMEM→SMEM copy overlaps the current MMA, *bypassing registers*. *Mechanism:* the fragment is the hardware handing each of 32 threads an opaque "assignment slip" (book §7.2.1) — you program tiles, not elements; inputs are FP16/BF16, **accumulator is FP32** (mixed precision is what keeps it numerically stable). *Correctness:* element-exact vs Rung 0 within FP accumulation tolerance; verify the FP32 accumulator actually matters by trying FP16 accumulate and watching error grow. *Target:* the book's basic WMMA kernel hits **71 TFLOPS on an H100** (1.7× over its 43-TFLOPS chapter-6 CUDA-core kernel) — a *basic* kernel, far below peak; a well-tuned WMMA kernel reaches **~40–60% of the dtype's dense peak** (e.g. ≳130–185 TF of the A100's 312 dense, or ~70–100 of the 4090's 165 FP16). Aim for the tuned band; treat 71 as the book's first cut, not a ceiling. *Caveat:* WMMA's fragment layout is opaque *by design* — you cannot reach WGMMA from here, and this is the wall (§2.6).

**Rung 2 — `mma.sync` + `ldmatrix` + swizzle (Ampere, the layout-control rung).**
Drop below WMMA to the PTX `mma.sync.aligned.m16n8k16` and load operands with **`ldmatrix`** (a warp-collective load that deposits data *directly* in the tensor-core register layout), with an **XOR-swizzled** SMEM layout to kill bank conflicts. *Mechanism:* this is WMMA with the lid off — you now place the exact thread→value mapping the WMMA API hid, which is the prerequisite skill for *reading* WGMMA's descriptor world. *Correctness:* element-exact vs Rung 1; a bank-conflict count of ~0 in `ncu` (`shared_ld/st_bank_conflict`). *Target:* **~60–75% of dense peak** — the swizzle + direct register layout recovers most of the gap WMMA left. *Caveat:* still synchronous; the accumulator still lives in registers; you have *not* yet crossed into async.

**Rung 3 — WGMMA + TMA multistage pipeline (Hopper — THE conceptual jump). [requires H100]**
This is the heart of the assignment. Build the book's §7.3 progression on one H100:

1. **Basic WGMMA** (`book §7.3.1`): `wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16`, operands passed as **64-bit SMEM descriptors** you hand-encode (addr[17:0], leading-dim, stride, validity bit — book Listing 7.9). First compile it with `nvcc -ptx` and **decode every qualifier** (`wgmma`/`mma_async`/`sync`/`aligned`/`m64n64k16`/`f32`/`f16.f16`) — do the book's "handwrite PTX" exercise so inline asm stops being hieroglyphics. *Target:* book **318 TFLOPS** (4.5× over WMMA).
2. **Larger tiles** → `m64n128/256k16`; more FMAs per descriptor load. *Target:* book **433**.
3. **TMA async loads** (`book §7.3.4`): create the **`CUtensorMap`** on the host (`cuTensorMapEncodeTiled`, `CU_TENSOR_MAP_SWIZZLE_128B`), issue `cp.async.bulk.tensor` from thread 0, and synchronize with an **mbarrier transaction count** (`barrier_arrive_tx` for the producer, plain `arrive` for consumers, `barrier.wait` for all). *Target:* book **504**.
4. **3-stage circular buffer** (`book §7.3.5`): a producer warpgroup streams TMA loads into 3 rotating SMEM regions while 1–2 consumer warpgroups run WGMMA on whichever region is ready; `wgmma.fence` / `commit_group` / `wait_group<N>` order it. *Target:* book **618 TFLOPS = 87% of cuBLAS** (4096³ FP16, **H100** — these are H100 numbers throughout §7.3); aim **≥70% of H100's 989 dense FP16**, climbing toward the book's 87%-of-cuBLAS once you add warp-specialization in §4.

*Mechanism (state it explicitly in your journal):* this rung is where `sync→async` and `register→SMEM-descriptor` *both* land. The producer/consumer mbarrier is the glue binding the **software (thread) timeline** to the **hardware (TMA) timeline** — `__syncthreads()` cannot do this because it only knows about threads (book §7.3.4). *Correctness:* element-exact vs Rung 0; the FP32 accumulator path identical. *Caveats:* you must pick a **GMMA canonical swizzle atom** that matches your TMA swizzle or the tensor core reads transposed garbage; row→column-major conversion can cost ~30 ms for 4096² (more than the GEMM) so store native column-major or use the descriptor transpose bit; a single mis-set descriptor validity bit silently produces zeros.

**Rung 4 — WGMMA FP8 + FlashAttention-style overlap (Hopper). [requires H100]**
Switch inputs to **FP8 (E4M3)** with K=32, FP32 accumulate, and use **`wait_group<N>` with N>0** to keep the previous MMA group in flight while you start the next — the overlap pattern FA3 uses to hide the softmax. *Mechanism:* FP8 doubles the math-tier (1,979 dense vs 989), so memory/issue overlap matters even more. *Correctness:* compare FP8-accumulate-FP32 against a BF16 reference; report the relative error and confirm it's within the FP8-GEMM expectation (this previews A5 quantization). *Target:* **~75%+ of 1,979 TF.** *Caveat:* FP8 GEMM error is real — you are trading bits for throughput; this is a numerics decision, not a free win.

> Rungs 0–2 are reproducible on a rented 4090/5090/A100; **rungs 3–4 require an H100**; the §4 B200 track is the frontier rung. Budget your H100/B200 hours per `00_foundations.md` — get correctness on cheap silicon first.

---

## §4 Frontier-2026 core (required)

This is the point of the assignment. Implement **§4.1 to completion on an H100**; implement **§4.2 and §4.3 on a B200** (toy-scale problem sizes are acceptable, but they must *run* on real `sm_100a` silicon and be measured). If you have no B200 budget, you may *substitute a written design* for §4.2/§4.3 that (a) gives the exact PTX/CUTLASS call sequence, (b) states every TMEM constraint, and (c) explains why a 5090 can't run it — but the bar for "Principal-grade" (§9) requires running at least one B200 kernel.

**4.1 Warp-specialized, persistent WGMMA + TMA GEMM (toward the H100 ceiling). [H100, required]**
Take Rung 3's 3-stage pipeline and add the two techniques that close the gap to cuBLAS: **warp-specialization** (a *dedicated producer warpgroup* does nothing but issue TMA loads and `mbarrier` arrives; consumer warpgroups do nothing but WGMMA + epilogue) and a **persistent kernel** (launch exactly `#SM` CTAs, each looping over a queue of output tiles, so you pay launch/prologue cost once and keep the tensor cores saturated across tiles). Optionally add **ping-pong** (two consumer warpgroups alternate MMA and epilogue so the tensor core never idles during the store). *Demonstrate:* you cross from ~80% toward **cuBLAS (the book's 618/713 = 87% on an H100; target ≳85% of 989 TF dense)**, and an `ncu` profile shows tensor-pipe utilization (`sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained`) ≳85% with the issue-stall and long-scoreboard stalls gone. *This is the kernel that proves you can feed the tensor core, not just call it.*

**4.2 `tcgen05` / UMMA GEMM with the accumulator in Tensor Memory (B200). [sm_100a]**
Build a Blackwell GEMM where the MMA is **single-thread-issued** and the accumulator lives in **TMEM**, not registers:

- Allocate TMEM with `tcgen05.alloc` (≥32 cols, power-of-two, from one warp). Stage A/B tiles to SMEM via TMA; issue `tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, enable-input-d` from a single thread (CUTLASS path: `cute::gemm` from a **full warp** — single-thread `cute::gemm` **deadlocks**, §2.6).
- Completion arrives on an **mbarrier** via **`tcgen05.commit`** (not `wait_group`). Drain the accumulator with a **full warpgroup** (the 32-lane-per-warp rule, §2.3) using `tcgen05.ld` → registers → epilogue. Use the largest atom `128×256×16`.
- *Demonstrate:* a working BF16 GEMM at **~60% of B200's 2,250 TF dense early**, climbing to **~85–95% (~1,300–1,476 TF observed)** with warp-specialization + persistent tiles — the gau-nernst.github.io/tcgen05 kernel reaches ~98% of B200 cuBLAS, use it as your reference target.
- *Then* flip to a **2-SM (`cta_group::2`) CTA-pair** kernel: `MMA_M` doubles to 256, each CTA holds half of A and half the output, B is split, only the leader issues. *Demonstrate and report the honest result:* the measured B200 BF16 4096³ win from 1-SM-warp-spec → 2-SM is **only ~8% (1209 → 1302 TFLOPS)** — the gain is **SMEM-bandwidth relief, not raw math**. Say that explicitly; it is a classic "the big architectural feature buys less than the marketing implies" lesson.

*Mechanism (the spine, one more time):* this is `register → TMEM` made real. The accumulator never touches the register file until the epilogue; the MMA is fully decoupled from CTA execution (Colfax). *Correctness:* element-exact vs a BF16 reference; verify you drained all 128 lanes (a 32-lane bug shows as 3/4 of the output being stale/zero). *Caveat — say it loudly:* **none of this runs on a 5090.** `sm_120` has no `tcgen05`, no TMEM, no `cta_group` pairing. This kernel is B200-only.

**4.3 NVFP4 block-scaled MMA (B200, the frontier rung). [sm_100a]**
Implement a **block-scaled FP4** GEMM — the format under 2025–2026 frontier inference:

- Inputs in **NVFP4 (`nvf4`)**: block size **16**, each block carrying a **UE4M3** (mantissa-bearing) scale — finer and more accurate than the MX path's block-32 + UE8M0 power-of-two scale (§2.4, Colfax Part 4). Stage the **scale factors into TMEM via `tcgen05.cp`**, duplicated across the 4 lane-quadrants; set the scale-IDs in the 32-bit `idesc`.
- *Demonstrate:* the kernel runs on `f8f6f4`-kind `tcgen05.mma` and you can place it on the **9,000 TF dense FP4 tier** (vs 2,250 BF16). Report accuracy vs a BF16 reference on a real weight matrix — FP4 needs per-block scaling *and* outlier handling to be usable (this is the bridge to A5). *Caveat:* **FP6 would run here at FP8 rate, not FP4 rate** (§2.4) — only FP4 unlocks the 2× tier. *Frontier note:* optimizing NVFP4 GEMM/attention is **open research** (a 2026 GPU-mode competition topic); a kernel that is both fast *and* accurate is not a solved problem. Treat "85% of FP4 peak *with* acceptable error" as a genuine research result, not a homework checkbox.

---

## §5 Correctness & numerics

- **Every GEMM is element-exact against a trusted reference** (cuBLAS / `torch.matmul`) at the appropriate FP-accumulate tolerance. A tensor-core kernel that's "close" is usually wrong: the classic bugs are a **transposed operand** (descriptor/swizzle mismatch), a **wrong leading-dimension** in the descriptor, or an **off-by-one K-tile** — all of which produce *plausible-looking* garbage.
- **Mixed-precision accumulate is load-bearing.** Inputs FP16/BF16/FP8, **accumulator FP32**. Prove it: run an FP16-accumulate variant and watch the relative error grow with K (book §7.2.3 — this is *why* tensor cores stay numerically stable across long reductions). Report the error curve.
- **FP8 / FP4 are numerics decisions, not free wins.** For Rung 4 / §4.3, report relative error vs a BF16 reference and state whether it's within the format's expected band. FP4 *requires* per-block scaling and outlier handling to be usable — an unscaled FP4 GEMM will be visibly wrong. Carry this into A5.
- **The TMEM 32-lane drain rule (B200).** A `tcgen05` kernel that drains the accumulator from a single warp reads only 32 of 128 lanes → **3/4 of the output tile is stale**. Test specifically: compare every lane-quadrant against the reference, not just the top-left tile.
- **Descriptor / swizzle adversarial cases:** a non-128-aligned SMEM base (must fail loudly, not silently), a K dimension that isn't a multiple of the atom's K, a column-major vs row-major operand fed to a descriptor expecting the other (the transpose-bit test), and a request whose N isn't a multiple of 8 (WGMMA) — verify each is caught or correctly handled.
- **2-SM correctness (B200):** with `cta_group::2`, verify the two CTAs' half-outputs stitch into the full tile and that only the leader's issue path runs (a both-issue bug double-counts).

---

## §6 Profiling & performance

- **The headline measurement** for every rung: **achieved TFLOPS and % of the *dense* peak**, computed as `2·M·N·K / time`, with locked clocks, warmup discarded, CUDA-event timing (`00_foundations.md`). Plot the full ladder (Rung 0→4) on one chart — it should reproduce the *shape* of the book's Table 7.2 progression.
- **Tensor-pipe utilization** in `ncu`: `sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active` (Hopper HMMA) / the `tcgen05` equivalent (Blackwell). Below ~85% on a tuned kernel means you are **feed-starved**, not math-bound — chase the stall.
- **Stall attribution** is the whole game. In `ncu`, read the warp-stall breakdown: **long-scoreboard** (waiting on memory → your TMA pipeline isn't deep enough, add stages), **MMA/short-scoreboard** (waiting on the tensor pipe → you're actually math-bound, good), **barrier** (mbarrier imbalance → producer/consumer mistuned). The optimization is always "convert the dominant non-MMA stall into MMA stall."
- **The async-overlap proof:** an `nsys`/`ncu` timeline showing TMA copies overlapping WGMMA/UMMA compute (the book's Figure 7.5 producer/consumer staircase). If copy and compute are serialized, your `wait_group`/mbarrier logic is wrong and you'll see the staircase collapse into a sequence.
- **Bank conflicts:** `shared_ld_bank_conflict` / `shared_st_bank_conflict` ≈ 0 after swizzling (Rung 2+). A non-zero count means your swizzle atom doesn't match the access pattern.
- **The honesty table** in your writeup: for each kernel, *achieved TFLOPS / dense spec / sparse spec / cuBLAS-on-this-machine*, so the reader sees you're quoting dense and quoting the right peak. State the 5090-vs-B200 gate wherever a Blackwell number appears.

---

## §7 Stretch goals (competition-grade)

- **CUTLASS 3.x/CuTe reimplementation:** rebuild your best WGMMA kernel as a CuTe tiled-MMA (`make_tiled_mma` + `cute::gemm`) and compare against your hand-PTX version — you should *lose a little* peak for *much* more readability and portability; quantify it.
- **ThunderKittens port:** express the same GEMM in TK's 16×16-tile DSL (WGMMA *and* `tcgen05`), dropping to PTX only where needed; compare lines-of-code and TFLOPS (arXiv:2410.20399).
- **Beat cuBLAS on a skinny/odd shape:** library kernels are tuned for square/large; find a shape (e.g. a decode-time GEMV-ish `M=8` GEMM, or an attention-shaped tile) where your custom kernel wins, and explain *why* from the roofline.
- **Full NVFP4 attention tile (B200):** combine §4.3's block-scaled MMA with an online-softmax tile — the bridge to A4's FlashAttention and the actual 2026 frontier.
- **Microbenchmark TMEM (B200):** reproduce a latency/bandwidth estimate for `tcgen05.ld`/`st` and compare to arXiv:2512.02189 — turning an [unknown] into a measured number is principal-grade work.

---

## §8 Deliverables & definition of done

1. **Code:** Rungs 0–4 (naive baseline; WMMA + `cp.async`; `mma.sync`+`ldmatrix`+swizzle; WGMMA+TMA 3-stage; WGMMA-FP8), the **§4.1 warp-specialized persistent WGMMA GEMM**, and — on B200 — the **§4.2 `tcgen05`/TMEM GEMM (1-SM and 2-SM)** and **§4.3 NVFP4 block-scaled MMA**. Each with its oracle test.
2. **The PTX artifact:** your hand-decoded `wgmma.mma_async` (every qualifier annotated) and your hand-encoded 64-bit SMEM descriptor, plus the `nvcc -ptx` output you compared against — the proof you can read and write tensor-core PTX.
3. **The performance ladder:** one plot, Rung 0→§4, achieved TFLOPS and % of dense peak, reproducing the *shape* of book Table 7.2; the async-overlap timeline; the stall-attribution table per kernel; the honesty table (dense/sparse/cuBLAS columns).
4. **Design note (2–3 pages): "How a tensor core encodes the manual tiling pattern, and what it costs to feed it."** Lead with the `register→SMEM→TMEM` / `sync→async` arc; walk each rung with its before/after TFLOPS and its dominant-stall evidence; end with the honest gap to cuBLAS (and, for B200, the honest ~8% 2-SM result and the 5090-can't-do-this gate). Write it for a frontier-lab kernel peer.

**Done when:** every rung is element-exact and at or above its % -of-dense-peak target; the WGMMA+TMA pipeline demonstrably overlaps copy and compute; §4.1 reaches ≳85% of H100 dense (or the book's 87%-of-cuBLAS shape); at least one B200 `tcgen05` kernel runs correctly with the accumulator in TMEM and you've reported the honest 2-SM and NVFP4 results; and the design note correctly explains *where the FLOPs went and which stall you converted to get them*.

---

## §9 Principal-level rubric

- **Pass:** WMMA GEMM (Rung 1) works, element-exact, ≥40% of dense peak, FP32-accumulate justified; you can explain the fragment abstraction and why WMMA can't reach WGMMA.
- **Strong:** + `mma.sync`+`ldmatrix`+swizzle (Rung 2) and a **WGMMA+TMA 3-stage pipeline on H100** (Rung 3) hitting ≳70% of 989 TF; you can read/write the PTX and hand-encode a descriptor; your profile proves async copy/compute overlap; you state dense-vs-sparse and the right peak every time.
- **Principal-grade:** all of §4 demonstrated — the **warp-specialized persistent WGMMA GEMM** at ≳85% of H100 dense, **and at least one B200 `tcgen05`/TMEM kernel running correctly** with the honest 2-SM (~8%) and NVFP4 results reported. You can derive the instruction evolution from the energy/issue argument; you reason in the `register→SMEM→TMEM` / `sync→async` frame unprompted; you mark `[documented]` vs `[unknown]` for TMEM correctly; you never confuse `sm_120` with `sm_100`; and your stall-attribution shows you optimize by *converting non-MMA stalls into MMA stalls*, not by guessing. The differentiator is **mechanism judgment**: knowing the matmul was never the hard part — feeding it asynchronously, by descriptor, through the right memory, was.

---

## §10 References (see `references.md` for full list)

- **NVIDIA PTX ISA** — `wgmma.mma_async`; `tcgen05.mma` + **Tensor Memory** sections. https://docs.nvidia.com/cuda/parallel-thread-execution/
- **Colfax CUTLASS tutorials** — *WGMMA on Hopper*, *Mastering the TMA*, *Writing GEMM with Tensor Memory for Blackwell (Part 1)*, *Hardware-supported Block-scaling on Blackwell (Part 4)*. https://research.colfax-intl.com/
- **gau-nernst**, *tcgen05 GEMM to ~98% of B200 cuBLAS* (working Blackwell kernel). https://gau-nernst.github.io/tcgen05/
- **SemiAnalysis**, *NVIDIA Tensor Core Evolution: Volta to Blackwell*. https://semianalysis.com/
- **NVIDIA CUTLASS** — Blackwell functionality + **CuTe MMA-atom** docs. https://github.com/NVIDIA/cutlass
- **ThunderKittens**, Spector et al., arXiv:2410.20399 + the TK Blackwell blog. https://arxiv.org/abs/2410.20399
- **NVIDIA**, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference* (block-16 + UE4M3). https://developer.nvidia.com/blog/
- **Luo et al.**, *Microbenchmarking NVIDIA Blackwell* (TMEM microarchitecture [unknown]s), arXiv:2512.02189. https://arxiv.org/abs/2512.02189
- **NVIDIA datasheets** — H100 SXM, DGX B200 (dense/sparse TFLOPS of record). · Book ch7 (WMMA→WGMMA→TCGen05 progression, Table 7.2).
