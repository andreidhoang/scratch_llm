# 00 — Foundations: Hardware, Toolchain, and the Engineering Discipline

Read this once, fully, before A1. It defines the hardware you'll rent, the tools you'll install, and the **discipline that is graded in every assignment**. The discipline is not optional polish — it is the actual content. Anyone can copy a kernel; the principal-level skill is the workflow that makes silent failure visible.

---

## 1. Hardware: what gates what

The compute capability (`sm_XX`) of the GPU determines which instructions exist. This is not a performance question — it is a *will-it-compile* question. Get this wrong and you waste rental money.

| GPU | Arch | `sm_` | HBM / BW | BF16 dense | FP8 dense | FP4 dense | NVLink/GPU | What it unlocks |
|---|---|---|---|---|---|---|---|---|
| RTX 4090 | Ada | **89** | 24 GB / 1.0 TB/s | ~165 TF | ~330 TF | — | none | CUDA, WMMA, reductions, INT8/INT4, FA1/2 |
| RTX 5090 | Blackwell-consumer | **120** | 32 GB / 1.79 TB/s | ~210 TF* | ~420 TF* | ~840 TF* | none | + FP4 *inference*; **NOT** tcgen05/TMEM |
| A100 80GB | Ampere | **80** | 80 GB / 2.0 TB/s | 312 TF | — | — | 600 GB/s | CUDA, WMMA, `mma.sync`, NVLink multi-GPU |
| **H100 SXM** | Hopper | **90/90a** | 80 GB / 3.35 TB/s | **989 TF** | 1,979 TF | — | 900 GB/s | **WGMMA, TMA, FP8, FA3, NVLink** |
| H200 SXM | Hopper | **90/90a** | 141 GB / 4.8 TB/s | 989 TF | 1,979 TF | — | 900 GB/s | same compute as H100 + more memory |
| **B200** | Blackwell-DC | **100/100a** | 180 GB / ~8 TB/s | **2,250 TF** | 4,500 TF | **9,000 TF** | 1.8 TB/s | **tcgen05, TMEM, NVFP4, FA4** |
| GB200 | 2×B200+Grace | 100 | 360 GB / ~16 TB/s | 4,500 TF | 9,000 TF | 18,000 TF | 1.8 TB/s | NVL72 rack-scale (overkill for kernels) |
| **B300** (Blackwell Ultra) | Blackwell-DC | **103/103a** | 288 GB / ~8 TB/s | ~2,250 TF | ~4,500 TF | **~15,000 TF** | 1.8 TB/s | +50% dense FP4 vs B200; 160 SMs; FP64 ~removed (~1.2 TF); same tcgen05/TMEM contract (`compute_100f` family runs on both) |
| GB300 NVL72 | 72×B300+36×Grace | 103 | 20.7 TB HBM3e / rack | — | — | ~1.1 EF dense FP4 / rack | 1.8 TB/s | rack-scale, shipping 2026 (rental beyond curriculum) |
| **Rubin VR200** | Rubin | not public (2026-07) | 288 GB HBM4 / ~22 TB/s | — | — | **~50,000 TF NVFP4 [vendor figure]** | NVLink-6 3.6 TB/s | Vera Rubin NVL144 rack (72 packages/144 dies, 8 EF NVFP4); volume H2-2026; successor ISA not yet public |

\* Consumer FP8/FP4 TFLOPS are NVIDIA-uneven across published sources — treat as estimates; the firm facts are sm_120 ≠ sm_100 and **no NVLink**. All datacenter numbers are **dense**; NVIDIA keynote "PFLOPS" figures are usually the **2× sparse** number. B300 HGX-B300 boards carry 270 GB (the 288 GB figure is GB300/B300-SXM). Rubin figures are NVIDIA.com/OCP vendor announcements (2026-07): the dense-vs-sparse basis of "~50 PF NVFP4" and the tcgen05-successor ISA are **not yet public** — orientation, not rental targets. Sources: NVIDIA datasheets, SemiAnalysis Tensor-Core Evolution — see `references.md`.

**The three hard gates to memorize:**
- **WGMMA + TMA → Hopper (sm_90) only.** A 4090 cannot run `wgmma.mma_async` or `cp.async.bulk.tensor`. A3's Hopper rungs and A4's FA3 rung *require* an H100.
- **`tcgen05` + TMEM + NVFP4 MMA → datacenter Blackwell (sm_100) only.** Not the 5090. A3/A4/A5's Blackwell rungs *require* a B200.
- **Real NVLink → SXM datacenter parts only.** Consumer cards have NVLink physically removed; multi-GPU falls back to PCIe. Fine for learning NCCL *semantics* (A6 single-node), useless for realistic interconnect numbers.

### Rental plan (mid-2026, approximate, reverify)

Individual-friendly, cheapest-first: **Vast.ai** (marketplace, rock-bottom, variable) and **RunPod** (per-second, instant, clusters) are the best solo entry points; **Nebius** and **Lambda** are clean fixed-price; hyperscalers are worst value for one learner.

- **A2 rungs 0–6, A3 WMMA, A4 rungs 0–3, A5 INT:** RTX 4090/5090 spot, ~$0.35–0.70/hr.
- **A1, and FP8/WGMMA/TMA/FA3 rungs:** 1× H100, ~$1.5–2.9/hr (Vast/Nebius cheapest).
- **tcgen05/NVFP4/FA4 rungs:** 1× B200, ~$3.4–6/hr.
- **A6 single-node:** 8× H100 SXM node, ~$22–24/node-hr (RunPod/Lambda).
- **A6 multi-node:** RunPod Instant Clusters (2–8 nodes, up to 64 H100, 800 Gb–3.2 Tb/s IB) or Together clusters.

Budget discipline: A1–A5 required cores fit in the low hundreds of dollars of single-GPU time if you checkpoint and use spot. Batch your B200 and multi-node hours — don't rent them until your kernel already runs correctly on cheaper hardware and you're only chasing the frontier rung.

---

## 2. Toolchain

Install once per instance (script it; bake an image if your provider allows):

- **CUDA Toolkit 13.x** (13.1+ — the CuTe DSL requires it on Blackwell; 12.8 remains the floor for Hopper-only work). `nvcc`, `cuobjdump`, `nvdisasm`, `ptxas`. Confirm `nvcc --version` and that your `-arch=sm_90a`/`sm_100a`/`sm_103a` (B300) matches the GPU. The trailing `a` ("architecture-specific") matters for WGMMA/tcgen05 — use `sm_90a`, not `sm_90`.
- **Nsight Compute (`ncu`) and Nsight Systems (`nsys`).** Kernel-level and timeline profilers. Non-negotiable for this curriculum.
- **CUTLASS 4.x + CuTe / CuTe DSL** (4.6 as of 2026-07; C++ headers, plus `pip install nvidia-cutlass-dsl` for the Python DSL that FA4 is written in). You'll read it constantly and use it for the harder tensor-core rungs.
- **PyTorch (current) + Triton.** Your correctness oracles, your `torch.compile` floor, and Triton for the kernels where it's the right tool.
- **Python stack:** `numpy`, `matplotlib` (roofline + ladder plots), `pandas` (sweep results), `nvidia-ml-py`/`pynvml` (telemetry).
- **For A6:** NCCL (bundled), `nccl-tests` (clone + build), `mpirun`/`torchrun`, and SLURM if your cluster uses it.
- **Optional but recommended:** ThunderKittens and FlashInfer repos (read their kernels), `flash-attn-4` (pip, JIT — the FA4 reference implementation, CuTeDSL, Hopper+Blackwell), `nvtx` for annotating timelines.

Sanity check every instance the moment it boots: `nvidia-smi` (right GPU? right count?), `nvidia-smi -q -d CLOCK` (can you lock clocks?), a trivial WGMMA/tcgen05 compile if you're on Hopper/Blackwell (fail fast if the toolchain/arch is wrong, before you've spent an hour).

---

## 3. The roofline — your first reflex

Before writing or optimizing any kernel, place it on the roofline **on paper**.

- **Arithmetic intensity** `AI = FLOPs / bytes_moved_to_from_HBM` (FLOP/byte).
- **Ridge point** `= peak_FLOPs / peak_BW`. For H100 SXM FP16: `989e12 / 3.35e12 ≈ 295 FLOP/byte`. FP8: `≈ 590`. (These are the crossover intensities; below them you are bandwidth-bound, above them compute-bound.)
- **Decision:** if your kernel's AI < ridge, you are **bandwidth-bound** — optimize memory traffic (coalescing, fewer passes, vectorized loads, fusion), and your ceiling is `AI × peak_BW`. If AI > ridge, you are **compute-bound** — optimize the math units (tensor cores, ILP, occupancy), ceiling `peak_FLOPs`. If you're far below *both* roofs, you are **latency-bound** — more in-flight work (occupancy or ILP).

Worked examples you'll re-derive in the assignments: a **GEMV** (decode projection, batch 1) has AI ≈ 1 → hopelessly bandwidth-bound → its ceiling is HBM bandwidth, full stop (A1, A2). A large **GEMM** has AI in the hundreds → compute-bound → tensor cores mandatory (A2, A3). **Attention** materializing the N×N score matrix has low average AI because of the softmax traffic → the whole point of FlashAttention is to raise it by never writing the scores to HBM (A4).

Roofline first means you never waste a day adding tensor cores to a kernel whose ceiling is bandwidth, or chasing memory tricks on a kernel that's compute-bound.

---

## 4. Benchmarking hygiene — no number counts without it

A performance claim that violates any of these is inadmissible:

1. **Lock the clocks.** `sudo nvidia-smi -lgc <freq>,<freq>` (and `-lmc` where supported). Boosting/thermal clocks silently move ±15%; you cannot compare kernels under them. `ncu` locks to base clock by default — good, but for your own wall-clock timing you must lock manually.
2. **Warm up, then measure.** Discard the first ~10 iterations (JIT, cache fill, clock ramp). Report median of ≥100 timed iterations, not the min and not the mean.
3. **Amortize or measure launch overhead.** A kernel launch is ~5–10 µs CPU-side; for short kernels that dominates. Either use **CUDA graphs** to replay the launch sequence, or report the overhead explicitly. At batch 1, launch overhead can be 20–40% of decode time — this is itself a finding (A1).
4. **Flush caches between cold-cache runs** when measuring HBM-bound kernels (`ncu` does this per pass; for hand-timing, touch a large scratch buffer).
5. **Time the right thing.** Use CUDA events around the kernel, not Python wall-clock around the launch, unless launch *is* what you're measuring. Know the difference (this is the book's "wall clock vs GPU time" lesson, and it's load-bearing).
6. **Pin the shape and dtype in the report.** "630 TFLOP/s" is meaningless without "FP16-in/FP32-acc, 8192³, H100 SXM." A2's GEMM ladder lives or dies on this.

---

## 5. The correctness-oracle discipline

Every kernel has a reference implementation you trust, and a test that compares against it before you optimize.

- **The oracle** is the simplest correct thing: PyTorch eager (`torch.matmul`, `F.softmax`, `F.scaled_dot_product_attention`), NumPy, or a triple-loop in C. It is allowed to be slow. It is not allowed to be wrong.
- **The comparison** uses an explicit tolerance appropriate to the dtype. FP32: `rtol≈1e-5`. FP16/BF16 accumulation: `rtol≈1e-2..1e-3` and check *relative* error on a range of magnitudes. FP8/FP4: compare **SQNR** (signal-to-quantization-noise ratio, ≈ `6.02·bits + const` dB as a sanity floor) and end-to-end task metrics, not element-wise equality.
- **Adversarial inputs**, not just random ones: zeros, a single large outlier, NaN/Inf in an unused position, the empty/degenerate shape, the non-contiguous tensor, the sequence length that isn't a multiple of the tile. Most kernel bugs hide in the remainder tile and the masked position.
- **The gradient/dependency test** (for anything you'll backprop through, A4): perturb input element `i`, confirm only the outputs that *should* depend on it change. This catches silent batch-dimension and masking bugs that element-wise checks miss.

Rule: **if the kernel can't match the oracle on one small adversarial input, no performance number from it is real.** This is the systems analog of the frontier "overfit a single batch before anything else" rule.

---

## 6. Profiling workflow (Nsight Compute)

The loop for every optimization rung:

1. **Predict** (roofline, on paper): bound type, expected % of peak, where the bottleneck is.
2. **Profile** `ncu --set full -k <kernel> -o report ./bench` (or targeted `--section SpeedOfLight --section MemoryWorkloadAnalysis` while iterating — faster).
3. **Read the Speed-of-Light section first**: the Compute% and Memory% bars. Whichever is closer to 100% is your bound — confirm it matches your prediction.
4. **Drill in** based on the bound: MemoryWorkloadAnalysis (sectors/request → coalescing; L1/L2 hit rates), ComputeWorkloadAnalysis (pipe utilization — is the tensor-core pipe actually busy?), Occupancy (theoretical vs achieved + the limiter), WarpStateStats (stall reasons), `l1tex__data_bank_conflicts_*` (shared-memory conflicts).
5. **Change one thing, re-profile, write down the delta.** Promote the rung only if the number moved the way you predicted.

You are looking for specific, nameable causes: "uncoalesced loads — 32 sectors/request instead of 4," "the WGMMA pipe is 38% utilized because the TMA loads aren't deep enough to hide the latency," "84% occupancy but the bottleneck is bank conflicts on the shared-memory store, not occupancy." Vague "it's slow" is not a diagnosis.

---

## 7. The research journal & the written artifact

- **Journal**, every session, three sentences: *what I tested, the number I got, the next hypothesis.* This is where the compounding happens. Future-you debugging a regression will live or die by it.
- **Each assignment's definition of done includes a written artifact** — a 1–2 page design note (what you built, the ladder of numbers, the roofline analysis, the bottleneck at each rung and how you found it) or a postmortem (for anything that broke: the symptom, the forensic trace to root cause, the fix, the guardrail you added so it can't recur). 
- Write the artifact as if a peer at a frontier lab will read it. "I made it faster" fails. "Rung 7 (WGMMA) jumped 32→317 TF/s when I introduced TMA + tensor cores; the remaining gap to the 630 TF/s CUTLASS ceiling is the epilogue, which Nsight shows as a 9-way bank conflict on the `stmatrix` store, fixed by 128-byte swizzling — here's the before/after SoL section" passes.

---

## 8. The principal-level rubric (applies to all assignments)

Each assignment has its own targets, but the *grade* is the same five-axis bar. "Pass" is getting the kernel working. **Principal-grade** is all five:

1. **Correctness, adversarially.** Matches the oracle on random *and* adversarial inputs, at a stated tolerance, with the numerical regime understood (not "it passed `allclose`" but "FP8 SQNR is 41 dB, here's why that's the expected floor").
2. **Performance, with evidence.** Hits the rung's target *and* you can show the Nsight evidence for the bound at each rung, *and* you beat the relevant library floor (`torch.compile`/Triton) or explain precisely why the gap to the ceiling (cuBLAS/cuDNN/CUTLASS/NCCL) isn't worth closing.
3. **Roofline reasoning.** You predicted the bound before measuring and the prediction held — or you found out why it didn't.
4. **Forensics.** When something broke, the writeup names the root cause at the level of the specific instruction/tile/batch, not "restarted with a different seed."
5. **Judgment.** You killed dead directions fast, you didn't gold-plate a rung past its target, and you can articulate the *engineering trade-off* (this is 8% faster but 3× the code and only on this shape — ship it or not, and why).

The first four are craft. The fifth is what makes you principal. A kernel that's 95% of cuBLAS but took three weeks and only works on one shape may be the *wrong* answer, and knowing that is the job.

---

Now start **A1 — Transformer Inference**. Bring the discipline. Bring the journal.
