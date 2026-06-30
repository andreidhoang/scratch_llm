# References — Primary Sources for the Frontier GPU Curriculum

Curated and deduplicated from the research that grounds this curriculum (mid-2026). Organized by domain; each assignment's §10 points here. **Read these, don't just cite them** — the skill this curriculum builds is the ability to go to the primary source and verify the number on your own hardware. Honesty flags from the research are consolidated at the end.

Citation discipline: arXiv IDs are given where they exist. Blog/doc links are dated where the content is fast-moving. Numbers from these sources carry their original caveats (dense-vs-sparse TFLOPS, PCIe-vs-SXM peak, best-case shapes, not-yet-public) — see §End.

---

## 0. Foundations: discipline, roofline, profiling

- **The "CUDA for Deep Learning" book** (Manning MEAP v5, E. Arledge) — the chapters 5–10 this curriculum is built from.
- A. Karpathy, **"A Recipe for Training Neural Networks"** — the eval-first / overfit-one-batch / make-silent-failure-visible discipline. karpathy.github.io/2019/04/25/recipe/
- **Roofline model** — Williams, Waterman, Patterson (CACM 2009); NVIDIA Nsight Compute roofline blog: developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/
- **NVIDIA Nsight Compute Profiling Guide** — docs.nvidia.com/nsight-compute/ProfilingGuide/
- V. Volkov, **"Better Performance at Lower Occupancy"** (GTC 2010) — occupancy is not the goal; ILP hides latency.

---

## 1. Inference serving systems (A1)

- Kwon et al., **PagedAttention / vLLM**, SOSP 2023 — arXiv:2309.06180; + vLLM V1 architecture blog (2025-01-27, blog.vllm.ai/2025/01/27/v1-alpha-release.html); "Inside vLLM" (vllm.ai, 2025-09-05).
- Zheng et al., **SGLang / RadixAttention** — arXiv:2312.07104; LMSYS blog (2024-01-17).
- Ye et al., **FlashInfer** (MLSys 2025 best paper) — arXiv:2501.01005; github.com/flashinfer-ai/flashinfer.
- Zhong et al., **DistServe** (disaggregation for goodput) — arXiv:2401.09670.
- Patel et al., **Splitwise** (phase splitting), ISCA 2024 — microsoft.com research PDF.
- Qin et al., **Mooncake** (KVCache-centric, Kimi), FAST 2025 — arXiv:2407.00079; github.com/kvcache-ai/Mooncake.
- Li et al., **EAGLE-2** arXiv:2406.16858 / **EAGLE-3** arXiv:2503.01840; **Medusa** (Cai et al.). Speculative decoding.
- Yuan et al., **LLM Inference Unveiled: Survey & Roofline** — arXiv:2402.16363.

## 2. GPU kernel optimization & GEMM (A2)

- S. Böhm, **"How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"** — siboehm.com/articles/22/CUDA-MMM (the canonical from-scratch SGEMM ladder).
- Colfax Research CUTLASS tutorials — **WGMMA on Hopper**, **Mastering the TMA**, **GEMM Kernel Design / Pipelining**, **Persistent Kernels & Stream-K** — research.colfax-intl.com.
- PyTorch, **"Deep Dive on the CUTLASS Ping-Pong GEMM Kernel"** — pytorch.org/blog/cutlass-ping-pong-gemm-kernel/.
- NVIDIA, **CUTLASS 3.x: Orthogonal, Reusable, Composable Abstractions** + **CuTe layout docs** — developer.nvidia.com/blog + docs.nvidia.com/cutlass.
- **Triton** matmul tutorial + Hopper TMA blog — triton-lang.org; docs.pytorch.org/blog/hopper-tma-unit/.
- Spector et al., **ThunderKittens** (ICLR 2025) — arXiv:2410.20399; "GPUs Go Brrr" (hazyresearch.stanford.edu, 2024-05-12).
- **DeepGEMM** (DeepSeek FP8, fine-grained scaling, two-level FP32 promotion) — github.com/deepseek-ai/DeepGEMM.
- Osama et al., **Stream-K** — arXiv:2301.03598.
- Milakov & Gimelshein, **Online Normalizer Calculation for Softmax** — arXiv:1805.02867.
- Zhang & Sennrich, **RMSNorm** — arXiv:1910.07467.

## 3. Tensor cores: Hopper → Blackwell (A3)

- **NVIDIA PTX ISA** — `wgmma` (Asynchronous Warpgroup MMA) and `tcgen05` (5th-gen Tensor Core family) + Tensor Memory sections — docs.nvidia.com/cuda/parallel-thread-execution/.
- Colfax Research — **WGMMA on Hopper**; **Writing GEMM Kernels Using Tensor Memory for Blackwell (Part 1)**; **Hardware-Supported Block-Scaling on Blackwell (Part 4)**; **Mastering the TMA** — research.colfax-intl.com.
- SemiAnalysis, **"NVIDIA Tensor Core Evolution: From Volta to Blackwell"** — the register→SMEM→TMEM, sync→async narrative.
- gau-nernst, **"tcgen05 for dummies"** — a working B200 kernel to ~98% of cuBLAS (BF16); gau-nernst.github.io/tcgen05/.
- NVIDIA CUTLASS — **Blackwell SM100 functionality** + **CuTe MMA-atom docs** — docs.nvidia.com/cutlass.
- NVIDIA, **"Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"** — developer.nvidia.com/blog.
- **Microbenchmarking NVIDIA's Blackwell Architecture** — arXiv:2512.02189 (for the TMEM-microarchitecture unknowns).
- NVIDIA **H100** and **DGX/HGX B200** datasheets (peak TFLOPS, dense vs sparse).

## 4. Flash Attention & attention variants (A4)

- Dao et al., **FlashAttention** (NeurIPS 2022) — arXiv:2205.14135.
- Dao, **FlashAttention-2** — arXiv:2307.08691.
- Shah et al., **FlashAttention-3** (Hopper async + FP8) — arXiv:2407.08608.
- Dao et al., **FlashAttention-4** (Blackwell; blog + paper, 2026-03-05) — tridao.me/blog/2026/flash4/; + PyTorch "FlexAttention + FlashAttention-4" blog.
- Milakov & Gimelshein, **online softmax** — arXiv:1805.02867.
- Ainslie et al., **GQA** — arXiv:2305.13245; Shazeer, **MQA** — arXiv:1911.02150.
- DeepSeek-AI, **DeepSeek-V2 (MLA)** — arXiv:2405.04434.
- Xiao et al., **StreamingLLM / attention sinks** (ICLR 2024) — arXiv:2309.17453.
- Beltagy et al., **Longformer** (sliding window) — arXiv:2004.05150; **Mistral 7B** — arXiv:2310.06825.
- **PagedAttention** — arXiv:2309.06180; **FlashInfer** — arXiv:2501.01005 (cross-ref §1).

## 5. Low-precision & quantization (A5)

- **OCP Microscaling (MX) Formats v1.0** spec + **"Microscaling Data Formats for Deep Learning"** — arXiv:2310.10537.
- **OCP FP8 Formats for Deep Learning** (E4M3/E5M2) — arXiv:2209.05433.
- NVIDIA, **"Introducing NVFP4..."** — developer.nvidia.com/blog (cross-ref §3).
- NVIDIA, **Pretraining LLMs with NVFP4** — arXiv:2509.25149.
- Peng et al., **FP8-LM** — arXiv:2310.18313.
- DeepSeek-AI, **DeepSeek-V3 Technical Report** (FP8 recipe, two-level accumulation, MLA, MTP, DualPipe, DeepEP) — arXiv:2412.19437.
- Xiao et al., **SmoothQuant** — arXiv:2211.10438.
- Lin et al., **AWQ** — arXiv:2306.00978.
- Frantar et al., **GPTQ** — arXiv:2210.17323.
- Ashkboos et al., **QuaRot** — arXiv:2404.00456; Liu et al., **SpinQuant** — arXiv:2405.16406.
- Frantar/Castro et al., **Marlin** — arXiv:2408.11743; **Machete** (Neural Magic / vLLM); **DeepGEMM** (cross-ref §2).
- **KVTuner** (KV-cache quant granularity) — arXiv:2502.04420.

## 6. Distributed training & inference infra (A6)

- Rajbhandari et al., **ZeRO** — arXiv:1910.02054.
- Shoeybi et al., **Megatron-LM** (tensor parallel) — arXiv:1909.08053; Narayanan et al., **Megatron 3D / interleaved 1F1B** — arXiv:2104.04473.
- Korthikanti et al., **Reducing Activation Recomputation** (SP, MFU vs HFU) — arXiv:2205.05198.
- Huang et al., **GPipe** — arXiv:1811.06965.
- Chowdhery et al., **PaLM** (MFU definition) — arXiv:2204.02311.
- **Llama 3 Herd** (4D parallelism, MFU, failure stats) — arXiv:2407.21783.
- DeepSeek-AI, **DeepSeek-V3** — arXiv:2412.19437; **DualPipe** — github.com/deepseek-ai/DualPipe; **DeepEP** — github.com/deepseek-ai/DeepEP.
- HuggingFace, **The Ultra-Scale Playbook** — huggingface.co/spaces/nanotron/ultrascale-playbook.
- **NVIDIA NCCL User Guide** (collectives, ring/tree/NVLS, tuning) — docs.nvidia.com/deeplearning/nccl/.
- Liu et al., **Ring Attention** — arXiv:2310.01889; **DeepSpeed-Ulysses** — arXiv:2309.14509.
- Sheng et al., **veRL / HybridFlow** (RL trainer↔inference bridge) — arXiv:2409.19256; **NeMo-Aligner** — arXiv:2405.01481.
- PyTorch **FSDP2 / DTensor / Distributed Checkpoint** docs; **NVLink/NVSwitch/NVL72** + **SHARP** material — nvidia.com/data-center.

## 7. Hardware & rental (00_foundations)

- NVIDIA datasheets — **A100, H100, H200, B200, GB200 NVL72** product/spec pages; **CUDA GPU Compute Capability** list (developer.nvidia.com/cuda/gpus, for the sm_80/89/90/100/120 mapping).
- SemiAnalysis Tensor-Core Evolution (cross-ref §3); jianyuh.github.io Blackwell SM100 / TMEM writeup; gau-nernst tcgen05 (cross-ref §3).
- Cloud pricing (volatile — reverify): Vast.ai, RunPod, Lambda, Nebius, CoreWeave, Modal, Together pricing pages; AWS Capacity Blocks; Spheron GPU pricing comparison 2026.

---

## End — Consolidated honesty / uncertainty flags

The research that built this curriculum was explicit about what is solid vs evolving. Carry these into every assignment:

- **Dense vs sparse TFLOPS.** NVIDIA keynote "PFLOPS" numbers usually include 2:4 (or 4:8) structured sparsity = 2× the *dense* figure. Real kernels rarely approach the sparse number. Always benchmark and report dense unless you are actually exploiting sparsity. (E.g., H100 FP16 ≈ 989 dense / 1,979 sparse; B200 FP4 ≈ 9,000 dense / 18,000 sparse.)
- **PCIe vs SXM peak.** Many published Hopper GEMM numbers (Colfax) are on H100 **PCIe** (114 SMs, ~750 TF/s FP16 peak), not the SXM 132-SM ~990 part. Don't cross-compare; recalibrate % targets to your card.
- **Best-case shapes.** DeepGEMM's ~1,350–1,550 FP8 TFLOP/s, NVFP4's throughput multipliers, and library "% of peak" figures are on large, favorable shapes/specific GPUs. Your shape will differ.
- **FA4 low-precision is not yet public.** FlashAttention-4's published benchmarks are BF16 on B200 (~1605 TF/s, ~71% util). FP8/FP4 FA4 accuracy/throughput, the exact rescale threshold τ, and the MUFU/FMA exp split are not published as of the research date — treat as targets to measure.
- **Blackwell tcgen05 microarchitecture is a contract, not a datasheet.** TMEM size/addressing/alloc and the `tcgen05.*` programming model are documented; physical TMEM latency/bandwidth, banking, and the cycle-level datapath are not. The arXiv:2512.02189 microbenchmark paper targets these gaps.
- **Consumer ≠ datacenter Blackwell.** RTX 5090 is **sm_120**, not the B200's **sm_100**; it has FP4 inference but **no tcgen05/TMEM** and no NVLink. It cannot substitute for a B200 on the tensor-memory rungs.
- **NVFP4 two-level constant placement varies.** Where the per-tensor FP32 scale vs the per-block FP8 scale absorbs the 448 (E4M3 max) and 6 (FP4 max) differs between TensorRT-Model-Optimizer and TransformerEngine. Verify against the kernel you target.
- **Distributed numbers predate optimizations.** Some DeepEP bandwidths predate a reported +30% improvement; MoE capacity-factor ranges and Llama-3 per-category failure integers are secondary or version-specific. Reverify before quoting.
- **Prices move weekly.** Every rental rate is approximate and dated to mid-2026. Re-price before you spend.

When in doubt, the entire point of this curriculum is that you can **go measure it yourself**, on locked clocks, against the right oracle and the right ceiling. Do that.
