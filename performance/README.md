# Frontier GPU Performance & Inference Engineering — A From-Scratch Curriculum (2026 Edition)

> Six build-it-yourself assignments that take you from "I can write a CUDA kernel" to **principal-level AI performance, inference, and optimization engineer** — the person a frontier lab trusts to make their models train and serve faster, cheaper, and at larger scale than the defaults allow.

This curriculum is derived from chapters 5–10 of *CUDA for Deep Learning* (Manning MEAP v5, Arledge) — Transformer Inference, Kernel Optimization, Tensor Cores, Flash Attention, Quantization, and Distributed Computing — but it is **not a book summary**. Every assignment has been re-grounded in mid-2026 frontier reality: Hopper WGMMA/TMA and Blackwell `tcgen05`/TMEM, FlashAttention-3/4, NVFP4/MXFP4, MLA and disaggregated serving, DeepEP/DualPipe, and the engineering discipline that separates a fast kernel from a shipped system. The book gives the *spine*; the frontier research (see `references.md`) is the *required core build*.

---

## 1. Who this is for, and the goal

You are comfortable with C++/CUDA and PyTorch (the book's chapters 1–4 — naive kernels, raw-CUDA training, cuBLAS — are assumed). You want to operate at the level of the engineers who write the kernels and serving systems the rest of the field depends on.

**The target competency** — what a *principal* in this niche can do, that a strong senior cannot:

1. **Reason from the hardware up.** Given a workload and a GPU, predict on a napkin whether it is compute-bound or bandwidth-bound, what fraction of peak is achievable, and where the bottleneck will move when you fix the first one. Roofline is a reflex, not a tool you reach for.
2. **Build the primitive, not just call it.** Implement GEMM, attention, the collectives, and the quant kernels from scratch to within a known fraction of cuBLAS/cuDNN/NCCL — and know exactly which last 10% you are leaving on the table and why it costs more to claim than it is worth.
3. **Make precision a design variable.** Move a model through BF16 → FP8 → NVFP4 deliberately, with the numerics (scaling, accumulation, outlier handling) under control and the accuracy loss measured, not hoped for.
4. **Serve at the system level.** Turn a forward pass into a serving engine: KV-cache management, continuous batching, speculative decoding, prefill/decode disaggregation — optimizing for *goodput under an SLO*, which is the metric that pays the bills.
5. **Scale across the interconnect.** Map a model onto TP/PP/EP/DP/CP so the chattiest collective rides the fattest link, overlap communication with compute, and hold MFU as the GPU count grows by 100×.

Underneath all five is the meta-skill (Section 4): **making silent failure visible**. Neural systems fail without crashing — a kernel that is 4× too slow, a quantization that quietly drops 2 points of accuracy, a parallelism mapping that wastes 40% of the cluster, all run to completion and produce plausible numbers. The discipline of this curriculum is built to surface those.

---

## 2. The six assignments

Each assignment is one chapter's topic, rebuilt with frontier-2026 as the required deliverable. Full spec in the linked file; engineering discipline and hardware in `00_foundations.md`.

| # | Assignment | Book ch. | From-scratch core | Frontier-2026 required build | Primary HW |
|---|---|---|---|---|---|
| **[A1](A1_transformer_inference.md)** | **Transformer Inference** | 5 | KV-cache autoregressive decoder; prove decode is memory-bound | GQA→**MLA** cache, **PagedAttention**, continuous batching + chunked prefill, **speculative decoding**, CUDA graphs, **prefill/decode disaggregation** | 1× H100 (A100/4090 ok early) |
| **[A2](A2_kernel_optimization.md)** | **Kernel Optimization** | 6 | GEMV, Softmax, RMSNorm, Top-K, **GEMM** optimization ladders, Nsight-driven | Online-softmax fused kernels, warp-specialized **WGMMA+TMA** GEMM toward cuBLAS, **DeepGEMM**-style FP8 fine-grained scaling | 1× H100 (4090 for rungs 0–6) |
| **[A3](A3_tensor_cores.md)** | **Tensor Cores** | 7 | WMMA fragment GEMM; the register→SMEM→async arc | **WGMMA**+TMA pipelined GEMM (Hopper) → **`tcgen05`/UMMA + TMEM** GEMM and **NVFP4** block-scaled MMA (Blackwell) | 1× H100 → 1× B200 |
| **[A4](A4_flash_attention.md)** | **Flash Attention** | 8 | Naive 3-kernel attention → fused **online-softmax** tile kernel | FA2 work-partitioning → **FA3** (warp-spec, TMA, ping-pong, **FP8**) → FA4 ideas; **MLA**, attention sinks, paged attention | 1× H100 → 1× B200 |
| **[A5](A5_quantization.md)** | **Low-Precision & Quantization** | 9 | INT8 sym/asym, group-wise INT4 pack/unpack; SQNR discipline | **NVFP4** two-level pack + block-scaled GEMM, **FP8 KV cache**, **AWQ/GPTQ/QuaRot**, FP8/NVFP4 training recipe (DeepSeek-V3) | 1× H100 → 1× B200 |
| **[A6](A6_distributed.md)** | **Distributed Computing** | 10 | NCCL AllReduce, **tensor parallel** GEMM, **1F1B pipeline** | Multi-node **expert-parallel all-to-all** (DeepEP), comms/compute **overlap** (DualPipe), MFU at scale, fault tolerance | 8× H100 node → 2-node cluster |
| **[A7](A7_capstone.md)** | **Capstone** | 5–10 | — | Integrate into a **mini frontier inference engine** or a **Blackwell kernel suite** — your choice, principal-grade | per track |

**The spine.** A1 tells you *where the time goes* in inference (and hands you the GEMV/attention/KV workloads). A2 makes you *fast on CUDA cores* and gives you the profiling reflexes. A3 unlocks the *tensor cores* that every modern matmul rides. A4 fuses A2+A3 into *the* kernel that dominates LLM runtime. A5 makes it all *low-precision*. A6 makes it *multi-GPU*. The capstone makes it a *system*. Do them in order — each assumes the last.

---

## 3. What makes this "frontier" and not a tutorial

Three commitments, enforced in every assignment:

- **The target is the real artifact, at real precision, on real hardware.** Not a toy. A4 ends at an FP8 Hopper attention kernel benchmarked against FlashAttention-3 and cuDNN; A5 ends at an NVFP4 GEMM you prove is more accurate than MXFP4 *for the documented reason* (16-element blocks + a mantissa-bearing FP8 scale); A6 ends at a multi-node run where you measure the NVLink→InfiniBand bandwidth cliff and show your mapping survives it.
- **Every number is sourced or measured.** The performance targets in each assignment come from primary sources (NVIDIA datasheets, the FA/CUTLASS/DeepSeek papers, Colfax/SemiAnalysis writeups) cited in `references.md`, with the dense-vs-sparse and PCIe-vs-SXM caveats intact. When a target is "≈75% of H100 FP16 peak," that is the *published FA3 number*, and your job is to get within sight of it and explain the gap.
- **You compete with the libraries, honestly.** `torch.compile`/Triton is the floor you must beat to justify a hand-written kernel; cuBLAS/cuDNN/CUTLASS/NCCL are the ceilings you measure against. "I wrote a custom kernel" is worthless without "and it is 2.1× faster than the `torch.compile` baseline on this shape, here is the Nsight evidence, and here is why."

---

## 4. The operating discipline (read before A1, re-read monthly)

Frontier engineering is a *workflow*, not a bag of tricks. These rules are distilled from how top labs actually operate; they apply to **every** assignment and are graded.

1. **Correctness oracle before performance — always.** Before you optimize anything, you have a dead-simple reference (PyTorch eager, or a triple-loop in C) and a test that compares against it to a stated tolerance. The single most expensive mistake in this field is optimizing a kernel that is subtly wrong. (Analog of the frontier "overfit one batch first" rule: if your kernel can't match the oracle on one small input, nothing else matters.)
2. **Profile before optimizing; roofline before profiling.** Compute the arithmetic intensity and place the workload on the roofline *on paper first*, predict the bottleneck, then confirm with Nsight Compute's Speed-of-Light section. If your prediction and the profiler disagree, you have learned something — stop and find out what.
3. **Lock the clocks, warm up, amortize launch.** No performance number counts unless GPU clocks are pinned (`nvidia-smi -lgc`), you discarded warmup iterations, and launch overhead is either measured or removed (CUDA graphs). Wall-clock with boosting clocks is noise.
4. **One variable per experiment.** Change one thing, measure, write it down. If you change two and it gets faster, you have learned nothing. This rule feels obvious and is violated constantly under time pressure.
5. **Write predictions down before the run.** "I expect coalescing to take this kernel from 15 to ~110 GB/s and ~6× the throughput." When reality diverges, investigate *before* assuming the surprise is good news — broken and working runs look identical from the outside.
6. **Kill fast.** Each assignment has a stated compute/time budget. If a direction is dead after three honest attempts, document why and move on. Attachment to your own approach is the enemy.
7. **Keep a research journal.** Three sentences per session: what you tested, the result (with the number), the next hypothesis. The compound interest of this field lives in the journal and the postmortems, not in any single kernel.

Each assignment's **definition of done** includes a short written artifact — a design note or a postmortem — because the ability to write down *why* a thing is fast (or why it broke at step 8432 on a 50k-token all-code batch) is the actual principal-level skill. Tooling everyone has; taste and forensics are the differentiators.

---

## 5. Hardware & cost (summary — full plan in `00_foundations.md`)

You will rent, not buy. The architecture *gates* the work: WMMA runs anywhere (sm_70+), **WGMMA/TMA need Hopper (sm_90)**, **`tcgen05`/TMEM/NVFP4 need datacenter Blackwell (sm_100)** — and note the consumer RTX 5090 is **sm_120, not sm_100**, so it does *not* substitute for a B200 on kernel coursework. Indicative mid-2026 spot/on-demand rates (reverify before spending — prices move weekly):

| You need… | Use | ~$/GPU-hr | Assignments |
|---|---|---|---|
| CUDA fundamentals, WMMA, reductions, INT8/INT4, FA1/2 | RTX 4090 / 5090 (Vast/RunPod) | $0.35–0.70 | A2 (0–6), A3 (WMMA), A4 (0–3), A5 (INT) |
| FP8, WGMMA, TMA, FA3, near-cuBLAS GEMM, MLA serving | **1× H100** (Vast/Nebius) | $1.5–2.9 | A1, A2, A3, A4, A5 |
| `tcgen05`, TMEM, NVFP4, FA4 | **1× B200** (Vast/Nebius) | $3.4–6 | A3, A4, A5 (frontier rungs) |
| Real NVLink TP/PP/EP, NCCL tuning | **8× H100 node** | ~$22–24/node-hr | A6 |
| Multi-node TP/PP/EP over InfiniBand | **2 nodes / 16 GPU** (RunPod Instant Clusters) | cluster rate | A6 (multi-node) |

A disciplined learner can complete A1–A5's required core on **single H100/B200 hours measured in the low hundreds of dollars**, reserving the expensive multi-node hours for A6 and the capstone. Use spot/community instances and checkpoint often — all of this work is interruption-tolerant.

---

## 6. Suggested sequencing

This is a serious program — plan months, not weekends, if you are doing it part-time and to depth. A workable cadence:

- **A1 Transformer Inference** — 1.5–2 weeks. Sets the *why* (where inference time goes) and the metrics harness you reuse everywhere.
- **A2 Kernel Optimization** — 2–3 weeks. The longest grind; the GEMM ladder alone is where most of the CUDA-core craft lives.
- **A3 Tensor Cores** — 2–3 weeks. The hardest conceptual jump (async, descriptors, TMEM). Budget B200 hours late.
- **A4 Flash Attention** — 2 weeks. Fuses A2+A3; this is the marquee kernel.
- **A5 Quantization** — 1.5–2 weeks. Numerics-heavy; lighter on new kernel machinery once A3 is done.
- **A6 Distributed** — 2 weeks. Gated by multi-GPU access; do the single-node parts first, batch the cluster hours.
- **A7 Capstone** — 2–4 weeks. Integrative; pick the track that matches the job you want.

Do not rush A2 and A3. Everything downstream is GEMM and tensor cores wearing different hats.

---

## 7. How to use each assignment

Every assignment file follows the same structure so you can navigate it fast:

- **§0 Why this matters** — the principal-engineer framing: what real systems depend on this and what it's worth.
- **§1 Learning objectives** — measurable; "you can derive/implement/profile X."
- **§2 First-principles theory** — the actual math and mechanisms, sourced.
- **§3 The build ladder** — staged from-scratch milestones, each with its mechanism, correctness check, and **performance target**.
- **§4 Frontier-2026 core** — the required modern build (this is the point of the assignment).
- **§5 Correctness & numerics** — the oracle and the exact tests/tolerances.
- **§6 Profiling & performance** — the Nsight workflow, roofline placement, the metrics that matter.
- **§7 Stretch goals** — the competition-grade edge.
- **§8 Deliverables & definition of done** — code, plots, and the written artifact.
- **§9 Principal-level rubric** — what separates a pass from principal-grade.
- **§10 References** — the specific papers/repos for this assignment.

Work the ladder rung by rung. Validate against the oracle at every rung. Profile, predict, write it down. Promote a rung only when the number is real.

---

## 8. Files in this curriculum

```
README.md                      ← you are here: competency model, the arc, the discipline
00_foundations.md              ← hardware/rental detail, toolchain, the universal engineering discipline & rubric
A1_transformer_inference.md    ← Ch5: inference as a system
A2_kernel_optimization.md      ← Ch6: CUDA-core craft + profiling
A3_tensor_cores.md             ← Ch7: WMMA → WGMMA → tcgen05
A4_flash_attention.md          ← Ch8: the marquee fused kernel
A5_quantization.md             ← Ch9: FP8/FP4 numerics
A6_distributed.md              ← Ch10: scaling across the interconnect
A7_capstone.md                 ← integrative, principal-grade
references.md                   ← consolidated, primary-sourced bibliography (arXiv IDs + links)
```

---

## 9. An honesty note

This curriculum is aimed at a moving frontier. The **mechanisms** (online softmax, the roofline, WGMMA descriptors, two-level FP4 scaling, the parallelism taxonomy) are stable and load-bearing. Some **numbers and product details** evolve fast — FA4's low-precision benchmarks, exact NVFP4 accuracy on the newest models, this week's rental prices, the latest CUTLASS/Triton Blackwell support. Where the research flagged something as best-case, secondary-sourced, or not-yet-public, the assignments carry that flag forward; treat those as *targets to verify on your hardware*, not gospel. The skill you are building is precisely the ability to go measure it yourself. `references.md` is your starting set of primary sources — read them, don't just cite them.

Now go to `00_foundations.md`, set up your tools and your discipline, then start **A1**.
