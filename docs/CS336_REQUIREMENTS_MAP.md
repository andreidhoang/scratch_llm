# CS336 Requirements Mapping & Mastery Curriculum Alignment

This document provides a deep, first-principles mapping of the CS336 assignment requirements (Assignments 1–5, plus the A5 RLHF supplement) onto our `mastery` curriculum. 

The goal of our `mastery` track is to **build the entire stack from scratch to production**. While CS336 provides the foundational algorithms, our curriculum intertwines **Model** and **Systems/Performance** optimizations (the "make it work -> make it fast" loops) and pushes further into the **2026 AI Research Frontier** (e.g., MoE, MLA, Serving Engines).

This map ensures that as you progress through the `mastery` PRR (Predict-Run-Reconcile) loops, you cover 100% of the CS336 requirements without duplicating effort. The structural files (skeletons and tests) are already mirrored in the `mastery_llm` codebase.

---

## Assignment 1: Basics -> Curriculum Chặng 1 (M1–M4)

**Core Goal:** Build a Transformer Language Model from scratch and train it on a single GPU.

| CS336 Requirement | Curriculum Node | File Location in `mastery_llm/` | First-Principles Concept |
| :--- | :--- | :--- | :--- |
| **BPE Tokenizer** (train, encode, decode, special tokens, multiprocessing) | **M1: Tokenizer** (1.1–1.4) | `tokenizer.py` | Byte -> subword compression. Why BPE merges greedily. How regex pre-tokenization prevents cross-word merging. The round-trip invariant. |
| **Transformer Block Components** (RMSNorm, SwiGLU, RoPE) | **M2: Transformer forward** (2.3–2.5) | `model.py` | RMSNorm vs LayerNorm (cost of 2nd reduction). SwiGLU as a gated MLP. RoPE for relative positional encoding without absolute embeddings. |
| **Attention** (Scaled Dot-Product, Causal Masking, Multi-Head) | **M2: Transformer forward** (2.1, 2.2, 2.6) | `model.py` | Attention as differentiable KV-retrieval. Masking to prevent future leakage. GQA/MQA efficiency. |
| **Training Objective & Optimizers** (Cross-Entropy, SGD, AdamW) | **M3: Objective + optimization** (3.1–3.2) | `optim.py`, `model.py` | MLE -> Cross-Entropy. Logsumexp for stability. AdamW decoupled weight decay. |
| **Training Loop** (Cosine schedule, Gradient clipping, Checkpointing, Dataloader) | **M3 (3.3), M4: Training loop** (4.1, 4.5) | `train.py`, `utils/` | Memory-mapped data loading to avoid OOM. Checkpointing optimizer states. |
| **Generation** (Softmax, Top-p/Nucleus, Temperature) | **M4 (4.1)** & **S2 (Serving engines)** | `sampling.py` | Autoregressive decoding. How temperature scales logits. Top-p truncation. |

---

## Assignment 2: Systems & Parallelism -> Chặng 2 (S3–S4) & Chặng 3 (M5, S6)

**Core Goal:** Optimize compute and memory to the hardware limits, then scale across multiple GPUs.

| CS336 Requirement | Curriculum Node | File Location in `mastery_llm/` | First-Principles Concept |
| :--- | :--- | :--- | :--- |
| **Benchmarking & Profiling** (Nsight Systems, PyTorch profiler, memory tracing) | **S3: CUDA-core kernels** (3.0) | `bench/` | Roofline model. Identifying memory-bound vs. compute-bound operations. |
| **Mixed Precision** (Autocast, FP16/BF16 accumulations) | **M4: Training loop** (4.2) | `train.py` | Why accumulations must stay in FP32/BF16 to prevent gradient underflow. |
| **Memory Optimizations** (Operator fusion, Activation Checkpointing) | **M4: Training loop** (4.3–4.4) | `train.py`, `model.py` | Trading compute for memory. Using `torch.compile` for fusion. Saving autograd residuals. |
| **Custom GPU Kernels** (Triton Weighted Sum, FlashAttention-2 Fwd/Bwd) | **S4: Tensor cores + Flash** (4.1–4.6) | `kernels/*_triton.py` | SRAM vs HBM memory hierarchy. Tiling. Online softmax. Recomputation in the backward pass to avoid O(N^2) memory. |
| **Single-Node Comms** (Gloo/NCCL, All-reduce, Reduce-scatter, All-gather) | **M5: Distributed training** (5.1) | `utils/comms_calc.py` | Ring communication primitives and their bandwidth costs. |
| **Distributed Data Parallel (DDP)** (Naive, Flat Gradients, Overlapped Comms) | **M5: Distributed training** (5.1) | `utils/ddp.py` | Hiding communication latency behind backward pass computation using hooks. |
| **Optimizer State Sharding** (ZeRO-1/2) | **M5: Distributed training** (5.2) | `utils/zero1.py` | Sharding AdamW's moment estimates to break the memory wall. |
| **Fully Sharded Data Parallel (FSDP)** (ZeRO-3) | **M5: Distributed training** (5.3) | `utils/fsdp.py` | Sharding parameters and all-gathering them just-in-time for forward/backward passes. |
| **Analyzing Parallelism** (TP, PP, 2D Parallelism) | **S6: Distributed + ISA** (6.1–6.3) | `performance/rental/` | Megatron-LM Tensor Parallelism (column vs row parallel). Pipeline bubbles. |

---

## Assignment 3: Scaling Laws -> Curriculum Chặng 4 (M6)

**Core Goal:** Predict the compute-optimal model size and hyperparameters for a fixed FLOPs budget.

| CS336 Requirement | Curriculum Node | File Location in `mastery_llm/` | First-Principles Concept |
| :--- | :--- | :--- | :--- |
| **IsoFLOPs Profiles** (Chinchilla method) | **M6: Scaling laws** (6.1–6.2) | `scaling/isoflop.py` | Why loss scales as a power law. Fitting L(N, D). |
| **Predicting Optimal Configurations** | **M6: Scaling laws** (6.3–6.4) | `scaling/planner.py` | Allocating tokens vs parameters (~20 tokens per parameter). Predicting validation loss for unseen scales. |

---

## Assignment 4: Filtering Language Modeling Data -> Curriculum Chặng 4 (M7)

**Core Goal:** Process raw Common Crawl web data into high-quality pretraining data.

| CS336 Requirement | Curriculum Node | File Location in `mastery_llm/` | First-Principles Concept |
| :--- | :--- | :--- | :--- |
| **HTML Extraction & Language ID** | **M7: Data pipeline** (7.1) | `data/` | Parsing WARC/WET files. Using fastText for language filtering. |
| **PII & Harmful Content Masking** | **M7: Data pipeline** (7.2) | `data/` | Regex for emails/IPs/phones. Toxic speech and NSFW classification using pre-trained fastText models. |
| **Quality Filtering** (Gopher rules, Quality Classifier) | **M7: Data pipeline** (7.2–7.3) | `data/` | Heuristic vs learned quality signals. Using Wikipedia references as positive distributions. |
| **Deduplication** (Exact Line, MinHash + LSH) | **M7: Data pipeline** (7.4) | `data/` | Locality Sensitive Hashing. Jaccard similarity approximation using minhashes to find fuzzy duplicates efficiently. |

---

## Assignment 5: Alignment and Reasoning RL -> Curriculum Chặng 5 (M8)

**Core Goal:** Align a base model to solve complex math problems (MATH dataset) and follow human instructions.

| CS336 Requirement | Curriculum Node | File Location in `mastery_llm/` | First-Principles Concept |
| :--- | :--- | :--- | :--- |
| **Zero-Shot Prompting & Evaluation** | **M8: Post-training / RL** (8.1) | `algos/`, `prompts/` | Prompting for Chain-of-Thought (`<think>...</think>`). Parsing symbolic math answers. |
| **Supervised Fine-Tuning (SFT)** | **M8: Post-training / RL** (8.2) | `algos/sft.py` | Masking out prompt tokens in the cross-entropy loss. Microbatching for gradient accumulation. |
| **Expert Iteration (STaR)** | **M8: Post-training / RL** (8.5) | `algos/expert_iteration.py` | Bootstrapping reasoning by filtering self-generated rollouts for correctness and training on the successful traces. |
| **Policy Gradients & GRPO** | **M8: Post-training / RL** (8.3) | `algos/grpo.py` | REINFORCE vs PPO vs GRPO. Group-normalized advantages. GRPO-Clip objective to prevent policy collapse. |
| **Reward Functions & Logging** | **M8: Post-training / RL** (8.4, 8.7) | `rewards/` | Verified rewards (format + correctness). Logging per-token entropies and KL divergence. |
| **RLHF & DPO** *(A5 Supplement)* | **M8: Post-training / RL** (8.6) | `algos/dpo.py` | Aligning to human preferences. Direct Preference Optimization as a reward-model-free alternative to RLHF. |

---

## Beyond CS336: The 2026 Frontier (Curriculum Chặng 6, 7, 8)

Our mastery curriculum incorporates advanced architectures and serving systems that are explicitly required for modern frontier lab engineering, but not covered in the base CS336 assignments:

- **Chặng 6 (M9): Frontier Architectures**
  - **MoE** (Mixture of Experts), **MLA** (Multi-Head Latent Attention), **MTP** (Multi-Token Prediction).
- **Chặng 7 (S1, S2, S5): Serving & Inference**
  - **PagedAttention**, Continuous Batching, Speculative Decoding, KV Cache management.
  - **Quantization** (INT8, INT4, FP8, AWQ).
- **Chặng 8 (M10): Capstone Ablations**
  - Closing the loop. Proving architectural choices through rigorous ablation studies.

## How to use this Map

When following the `mastery` track and executing the **PRR Loop (Predict -> Run -> Reconcile)**, use this map to understand *why* a particular node exists. If you are working on `M1: Tokenizer`, you are fulfilling the requirements of CS336 Assignment 1's BPE section. 

You do **not** need to re-create the file structures; the `mastery/src/mastery_llm/` skeletons already define the APIs matching the test suite. Focus purely on deriving the logic from first principles.
