# A7 — Capstone: Build Something a Frontier Lab Would Use

> **Integrates:** A1–A6 · **Thesis:** A principal engineer is not certified by finishing assignments — they are recognized by *artifacts that move a real number on a real system*. The capstone is where the six threads become one system, benchmarked honestly against the best the field has, with a writeup that would survive review at a frontier lab.
>
> **Hardware:** track-dependent (1× H100/B200 for kernel tracks; 8×H100 / 2-node for the distributed track) · **Est. time:** 2–4 weeks · **Prereqs:** A1–A6 to a "Strong" bar or better.

---

## §0 Why a capstone

Everything before this taught a primitive in isolation. Real impact comes from *composition under a constraint*: a serving engine is paged KV + continuous batching + a fused FP8 attention kernel + quantized weights, all fighting for the same SMs and HBM. The capstone forces the integration, which is where the hard, principal-level trade-offs live — the ones you cannot see when each kernel is benchmarked alone. It is also your portfolio: pick a track, build it to a number, write it up, and you have concrete evidence of the competency this curriculum targets.

Pick **one** track and take it to a genuinely competitive number. Depth over breadth — a single track done to principal-grade is worth more than three done to "it runs."

---

## Track A — A mini frontier inference engine ("vLLM-lite")

**Integrate:** A1 (paged KV, continuous batching, chunked prefill, speculative decoding, CUDA graphs) + A4 (FP8 FlashAttention-3 kernel, or paged FA) + A5 (NVFP4 or INT4 weights, FP8 KV cache).

**Goal:** serve a real open model (Llama-3.1-8B, Qwen3-class, or a small MoE) end-to-end, optimizing **goodput under an SLO** (e.g., TTFT < 200 ms, ITL < 40 ms), and benchmark it head-to-head against vLLM on the *same* request trace and hardware.

**The bar:** you will not beat vLLM outright — it is years of engineering. Principal-grade is (1) getting within a stated factor (say, 1.5–2×) on goodput, and (2) a writeup that *correctly diagnoses every part of the gap* with profiler evidence: "we're 1.7× behind — 0.3× from our paged-attention kernel not fusing dequant, 0.2× from lacking prefix caching, 0.2× from scheduler overhead our CUDA graphs don't cover; here are the four flame graphs." Diagnosing the gap *is* the deliverable.

**Stretch:** prefill/decode disaggregation across two GPUs with measured KV-transfer cost; RadixAttention prefix cache; integrate FlashInfer as the attention backend and compare to your hand-rolled kernel; MLA model with weight-absorbed decode.

---

## Track B — A Blackwell/Hopper kernel suite

**Integrate:** A2 (the optimization ladder, profiling discipline) + A3 (WGMMA/TMA, tcgen05/TMEM) + A4 (fused attention) + A5 (NVFP4 block scaling).

**Goal:** ship three production-quality kernels, each benchmarked against the right ceiling on locked clocks: (1) a **warp-specialized WGMMA+TMA persistent GEMM** (Hopper) or **tcgen05/UMMA GEMM** (Blackwell) vs cuBLAS/CUTLASS; (2) a **FlashAttention-3-style FP8 attention** kernel vs FlashAttention-3 and cuDNN; (3) an **NVFP4 block-scaled GEMM** vs the BF16/FP8 baseline, proving both the speedup *and* the accuracy (SQNR + a downstream task), and that it beats an MXFP4 power-of-two-scale variant for the documented reason.

**The bar:** each kernel hits its assignment's frontier target (e.g., GEMM ≥80% of cuBLAS on the target shape; FP8 attention in sight of FA3's ~75% util / ~740 TF/s; NVFP4 GEMM within <1% accuracy of FP8). Every claim carries a locked-clock benchmark, an Nsight Speed-of-Light + roofline placement, and the dense-vs-sparse / PCIe-vs-SXM caveats. Bonus principal move: pick one kernel and **open a PR to CUTLASS, FlashInfer, vLLM, or SGLang** — the real-world version of "ship it."

**Stretch:** the tcgen05 2-SM / NVFP4 path (open-research territory; a 2026 GPU-mode competition topic); a CuTe-DSL implementation; an autotuner over tile/stage configs.

---

## Track C — Distributed at scale

**Integrate:** A6 (TP/PP/EP, NCCL, overlap, MFU, fault tolerance) + A1 (inference) or A5 (FP8).

**Goal — pick one:**
- **MoE serving at scale:** multi-node expert-parallel inference with a DeepEP-style all-to-all + prefill/decode disaggregation; report goodput and the all-to-all busbw vs RDMA line rate as you scale 8→32+ GPUs, and show your parallelism mapping survives the NVLink→InfiniBand cliff.
- **FP8 distributed training step:** a TP+PP+DP training step in FP8 (DeepSeek-V3-style fine-grained scaling + two-level FP32 promotion) with compute/comm overlap (DualPipe idea), reporting **MFU** at 8→32 GPUs and how much overlap + the FP8 GEMM each bought.

**The bar:** MFU is the scoreboard. Principal-grade is holding MFU in the realistic 40–55% band as you scale, *with the loss budget accounted for* (overlap, bubbles, comm, straggler tax — each as a measured line item), plus async checkpointing + elastic restart actually exercised by killing a process mid-run and recovering.

**Stretch:** 5D parallelism composition; the veRL-style trainer↔inference weight-resharding bridge; SHARP/NVLS tuning measured via nccl-tests.

---

## §8 Deliverables (all tracks)

1. **The system**, reproducible from a clean instance via a script (this is the "releasable checkpoint" discipline — if a peer can't reproduce your number from your artifacts, it didn't happen).
2. **A benchmark report**: the headline number vs the named ceiling, on locked clocks, with the trace/shape/dtype/hardware pinned, and the roofline/Nsight evidence.
3. **A design doc (3–5 pages)** in the frontier-lab style: the hypothesis, the architecture, the number you hit, the honest gap to the best-in-class with per-component attribution, and what you'd do next with 2× the time. Plus the **research journal** that got you there.
4. **(Strongly encouraged)** an OSS contribution or a public writeup. The field is unusually open — the FA/CUTLASS/vLLM/SGLang/DeepGEMM teams ship in public. A merged PR or a blog post that a practitioner cites is the most credible possible artifact.

---

## §9 What "principal" looks like at the end

You can be handed an unfamiliar model + GPU + SLO and, within a day, produce: a roofline-grounded prediction of the achievable number, a profiler-backed diagnosis of where the current system loses it, and a prioritized plan of the 3–5 changes that recover the most — with the judgment to know which ones are worth the maintenance cost. You measure before you optimize, you kill dead directions fast, you write down why things are fast, and your numbers are always honest about dense-vs-sparse, best-case-vs-typical, and the gap to the library. That judgment — not any single kernel — is the deliverable of this entire curriculum.

Go build the thing. Then write down what you learned, and ship it where someone can use it.
