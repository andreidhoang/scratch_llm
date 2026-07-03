# Book → Build → Harness map (read the right chapter at the right stage)

> **How to use this.** At each build stage: (1) read the mapped chapter/sections, (2) **reconstruct the
> kernel / derivation from blank** (the meat — yours), (3) master the *load-bearing 20%* cold, (4) invoke
> the listed harness command. The "When" column is your trigger — don't read ahead of the build.
> Section numbers are from the MEAP TOCs (CUDA v05 · RLHF v04).

**The two spines you climb (everything else hangs off these):**
- **CUDA:** the **matmul → cuBLAS climb** — Ch 4 (naive CUDA matmul → cuBLAS) → Ch 6.6 (the GEMM ladder:
  coalescing → shared-mem tiling → 1D/2D blocktiling → vectorize). Then FA (Ch 8) + quant (Ch 9).
- **RLHF:** **SFT (Ch 4) → Reward Modeling (Ch 5) → RL/GRPO (Ch 6) → DPO (Ch 8)**, with Over-optimization
  (Ch 14) = your RQ2 home turf and Regularization/KL (Ch 15) = your `monitors.py`.

**The meat boundary applies to both** (in `learn` mode — [ADR-0013](adr/ADR-0013-execution-mode-full-delegation.md)):
CUDA **kernel bodies** and RLHF **math derivations** you reconstruct from blank; agents
scaffold/tutor/profile/review — never write the rep. (See `src/scratch_llm/kernels/CLAUDE.md`.)
In `delegate` mode (current since 2026-07-03) the reps are agent-built and this map is the
reading order for the post-hoc mastery pass.

---

## Part 1 — CUDA for Deep Learning → A2 kernels + the kernel sprint

| Chapter · § | When (stage · sprint day) | Build (cs336) | Master COLD — load-bearing 20% | Harness |
|---|---|---|---|---|
| **1 · When PyTorch isn't enough** (1.4 memory hierarchy, 1.6 optimization stack) | read-once, before D8 | — (orientation) | the memory hierarchy HBM→L2→shared→regs; the optimization stack (naive→coalesce→tile→tensor-core); roofline intuition | `/tutor "roofline & memory hierarchy"` — no build |
| **2 · Your first CUDA program** (2.5 grids/blocks/boundary, 2.7 nD→1D) | **D8** · setup | warm-up kernel via `bench.load_cuda` | thread/block/grid indexing + boundary checks **from blank**; the nD→1D index pattern | `/kernel-day` → bench-writer scaffolds → reconstruct → `/profile` |
| **3 · Naive kernels** (3.1.2 transpose, 3.2.1 GEMM, 3.2.2 softmax, 3.3 conv/pool) | **D8–9** | `matmul_naive` (given baseline) + reconstruct transpose / softmax | the naive GEMM kernel + naive sequential softmax from blank | `/kernel-day`; `/tutor` for the GEMM index math |
| **4 · Training in raw CUDA** ★ (4.2 backprop derivations, **4.5 naive-CUDA matmul**, **4.6 cuBLAS + cublasSgemm + "a necessary lie"**, 4.5.5 wall-clock vs GPU-time) | **D8** · the % anchor | THE matmul→cuBLAS baseline → wire to `bench.matmul_roofline` | the naive CUDA matmul kernel; the `cublasSgemm` call (your % target); profiling wall-clock **vs** GPU-time | `/kernel-day` + `/profile` (roofline-analyst sets % of cuBLAS) |
| **5 · Transformer inference in CUDA** (5.2.1 KV-cache/autoregressive, **5.3.2 GEMV**, 5.3.1 memory layout) | **D13** | ties to your `KVCache` (built) + the decode path | why decode is **memory-bound** (GEMV, AI≈1); the KV-cache decode step | `/profile` → roofline-analyst (BOUND = memory) |
| **6 · Optimizing transformer kernels** ★★ **THE climb** (6.1 profiling/ncu · 6.2 GEMV coalesce→block-reduce→vectorize · 6.3 **online softmax**→warp-shuffle→fused · 6.4 fused LayerNorm · 6.5 Top-K · **6.6 GEMM: coalescing→thread-remap→shared-mem tiling→1D→2D blocktiling→vectorize→bottleneck**) | **D9–D13, D15** · most of the sprint | `matmul_tiled` — the reconstruct rungs **R1–R6** | **§6.6 the full GEMM ladder** (each rung + *why*); **§6.1 reading an ncu profile**; **§6.3 online softmax** (feeds FA) | **THE loop:** `/kernel-day` → reconstruct a rung → `/profile` (roofline-analyst: BOUND·WHY·NEXT) → `/kreview` |
| **7 · Tensor cores** (7.2 WMMA, 7.3 WGMMA+PTX, 7.5.4 what you still optimize, 7.6 TCGen05/Blackwell) | **D15** | R5 + the tensor-core matmul path (`tl.dot` already uses TCs) | when to use TCs; WMMA fragment/tiling; manual-tiling → hardware-tiling mapping | `/kernel-day`; `/profile` (target **≥80% of cuBLAS @4096³**) |
| **8 · Flash attention** ★ (8.1 memory-bound · 8.2 **tiling + online softmax** · 8.3 fused kernel / 3-level loop · 8.4 FA1–4 evolution) | **D16** | re-measure your FA2-Triton roofline; **beat 53% of SDPA** | online softmax + tiling (the FA invariant); why attention is memory-bound; the fused-kernel structure | `/kernel-day` on FA; `/profile` (%SDPA vs 53%); `/kreview` (numerics) |
| **9 · Quantization** (9.1 dtypes/quant-dequant · 9.2 granularity/calibration/static-dynamic/symmetric · 9.3 AWQ · 9.5 GPTQ/NF4/QAT) | **D17** | FP8/INT4 fake-quant + a **fused dequant-GEMM** (the C2 rung) | per-channel **vs** per-tensor scaling; the dequant-GEMM; the AWQ idea; NVFP4 framing (→ DELTA) | `/kernel-day`; `/profile` + accuracy-delta; feeds the **NVFP4-on-state** probe |
| **10 · Distributed computing** (10.1 NCCL/ring-vs-tree · 10.2 TP + AllReduce · 10.3 PP via CUDA streams · 10.4 multi-node) | **A2 distributed finish** (parallel track, not the kernel loop) | DDP ✅ → ZeRO-1 / FSDP → TP / PP | the comms algebra (AllReduce cost); TP = local matmul + AllReduce; ring vs tree | `/next` (the A2 driver) + `/ship` |
| **11 · CUTLASS** (code `9_cutlass/`) | **D18** · capstone | the DELTA **C2.5 real-kernel** / GDN reference | CuTe / CUTLASS GEMM literacy | `/kernel-day`; scout FlashInfer/CUTLASS refs |
| **Appendix A** (setup · dev workflow · profiling tools · **A.9 python bindings: pybind11 / CUDAExtension**) | at setup / when wiring CUDA→Python | mirrors `bench.load_cuda` | the kernel dev + profiling workflow; CUDAExtension | reference; `bench.load_cuda` |

---

## Part 2 — RLHF book → A5 alignment + reasoningLLM

| Chapter · § | When (stage · sprint day) | Build (cs336 / reasoningLLM) | Master COLD — load-bearing 20% | Harness |
|---|---|---|---|---|
| **1–2 · Intro + history** | read-once | — | orientation | `/tutor` |
| **3 · Training Overview** (3.1 problem formulation: thermostat/cartpole, fine-tuning+reg · 3.2 recipes: InstructGPT, Tülu 3, **DeepSeek R1**) | before Week 1 | — | the post-training recipe shape; the **R1-Zero recipe** (your Countdown "aha" target) | `/master "the RLHF recipe"` |
| **4 · Instruction Fine-tuning (SFT)** (4.1 chat templates · 4.3 implementation) | Week 1 prereq | A5 `algos/` SFT (masked-CE, `response_mask`) | **SFT masked cross-entropy + the `response_mask`** (everything downstream depends on it) | `/next` build; `/audit-rl` logging; **meat boundary** (derive) |
| **5 · Reward Modeling** (5.1 Bradley-Terry · 5.5 ORM · 5.6 PRM · 5.8 generative RM/LLM-judge) | **D6** | `rewards/` (the r1-zero grader, BT loss) | the Bradley-Terry RM loss; ORM vs PRM; the verifiable-reward grader | `/master "Bradley-Terry"` |
| **6 · Reinforcement Learning** ★★★ **R5 — the room the offer dies in** (6.1 derive PG · VPG · REINFORCE · **RLOO** · **PPO** · **GRPO** · GSPO · CISPO · 6.2 loss-aggregation · async RL · **TIS** · 6.3 **GAE**) | **Week 1 · D1–D4** | reasoningLLM `credit_assigners` + `off_policy` + A5 `algos/grpo` | **from blank:** ∇J=E[∇logπ·R]; REINFORCE→**GAE**→**PPO-clip**→**GRPO/k3-KL**; RLOO; TIS | **meat boundary — derive from blank**; `/master` each algo + teach-back; `/audit-rl` |
| **7 · Reasoning & Inference-Time Scaling** (7.1 why RL works now, **RLVR** · 7.2 reasoning training methods) | **D19** | RLVR notes → `reward.py` | RLVR; pass@k; the reasoning-model recipe | `/master "RLVR"` |
| **8 · Direct Alignment / DPO** ★ (8.1 how DPO works + **derivation** · 8.2 numerical concerns · 8.3 implementation · 8.5 online vs offline) | **Week 1 · D5** | A5 `algos/dpo` | **derive DPO from the RLHF-KL identity** (R5); the DPO loss; the numerical-stability concern | meat boundary (derive); `/master "DPO derivation"` + teach-back |
| **9 · Rejection Sampling** (9.1 generate→score→fine-tune · 9.3 best-of-N) | A5 | Expert Iteration (STaR) | the EI loop; best-of-N | `/next` |
| **13 · Tool Use & Function Calling** (13.1 interweave · 13.2 multi-step · 13.3 MCP) | forward edge (post-aha) | agentic RL on `envs/protocol` | multi-turn / agentic RL (the 2026 post-training frontier) | `/master`; the forward edge |
| **14 · Over-Optimization** ★ **YOUR HOME TURF — RQ2** (14.1 proxy objectives / over-refusal · 14.2 quantitative · 14.3 misalignment) | **D11** (the RQ2 day) | the RQ2 exploitability work (curve shipped) + reward-hack detectors | the over-opt framing → your **L0–L4 exploitability curve**; the reward-hacking taxonomy | `/master`; this **is** the research-taste pitch |
| **15 · Regularization** (15.1 KL divergence in RL + impl · 15.2 **SFT memorizes, RL generalizes**) | **D6** | `monitors.py` (the KLs, `kl_train_infer` HALT@0.10) | the **k3 KL estimator**; why the KL penalty; SFT-vs-RL generalization | `/audit-rl` (the KL-logging gate) |
| **16 · Evaluation** (16.1 prompting/CoT · 16.4 contamination · 16.5 tooling) | with eval discipline | eval notes | pass@k; contamination; eval calibration | `/master "eval"` |
| **10 · 11 · 12 · 17** (Nature of Preferences · Preference Data · Synthetic Data/CAI · Model Character) | breadth + interview awareness | — | the concepts for the interview answer (distillation, CAI, persona) | `/tutor` / scout |
| **Appendix A · C** (definitions · C.1 compute costs · **C.4 identifying bad training jobs**) | reference | — | "is this run interpretable / is it a bad job?" | `rl-run-auditor` / `/audit-rl` |

---

## The 20-day sprint index (which chapter, which day)

| Day | Book · section | Rep (from blank) |
|---|---|---|
| D1 Mon | RLHF 6.1 (REINFORCE / PG) | ∇J, `masked_mean` + naive PG loss → PG subset green |
| D2 | RLHF 6.3.1 (GAE) | derive GAE(λ): δ_t, λ=0→TD / λ=1→MC |
| D3 | RLHF 6.1.5 (PPO clip) | clipped surrogate → `test_grpo.py` green (CP#1 done) |
| D4 | RLHF 6.1.6 (GRPO + k3 KL) | GRPO objective + k3 KL |
| D5 | RLHF 8 (DPO) | derive DPO from the KL identity → `test_dpo.py` green |
| D6 | RLHF 5 + 15 (RM + KL) | Bradley-Terry loss; tie KL to `monitors.py` |
| D7 | RLHF 6/8 consolidation | cold re-test REINFORCE→GAE→PPO→GRPO→DPO |
| D8 Mon | CUDA 2–4 (first kernel → matmul baseline) | run `matmul_naive`; predict % of cuBLAS |
| D9 | CUDA 6.6 (GEMM rung 1) | `matmul_tiled` L2-reuse ordering |
| D10 | CUDA 6.6 (shared-mem tiling) | autotune + shared-mem blocking; ≥2× naive % |
| D11 | RLHF 14 (over-opt / RQ2) | RQ2 blog increment |
| D12 | CUDA 6.6 (block-tiling) | 1D/2D blocktiling; remove bank conflicts |
| D13 | CUDA 5 + 6 (inference / KV) | profile KV-cache decode → memory-bound |
| D14 | CUDA consolidation | cold re-test the matmul ladder |
| D15 Mon | CUDA 7 (tensor cores) | R5 vectorized + TC path; ≥80% of cuBLAS |
| D16 | CUDA 8 (flash attention) | beat 53% of SDPA |
| D17 | CUDA 9 (quantization) | fused dequant-GEMM + perplexity vs bf16 |
| D18 | CUDA 11 (CUTLASS) + capstone | GDN-decode + NVFP4-on-state probe → open the public artifact |
| D19 | RLHF 7 (RLVR + eval) | pass@k estimator; RLVR notes |
| D20 | retro | cold re-test EVERYTHING + 1-page "what I can do cold" |

---

## Harness quick-reference (which command at which stage)

| Command / agent | Use it for | Meat boundary |
|---|---|---|
| `/kernel-day` | the daily CUDA kernel loop (Ch 2–9, 11) | scaffolds; you reconstruct |
| `/tutor <concept>` | any concept, Socratic, no code (both books) | hints, never code |
| `/profile` → roofline-analyst | after running a kernel — BOUND·WHY·NEXT | diagnoses, never fixes |
| `/kreview` → kernel-ship-reviewer | before a kernel commit — correctness + speedup-real | gates, never rewrites |
| `/master <concept>` | RLHF derivations + concepts (teach-back gate) | you derive |
| `/next` | the A5 / distributed build driver (RLHF 4/9, CUDA 10) | test-first |
| `/audit-rl` → rl-run-auditor | RL run logging gate (RLHF 6/15, App C) | gates run interpretability |

**One rule across both books:** read the chapter, then **close it and reconstruct from blank** — the kernel
body (CUDA) or the derivation (RLHF). That's the rep, the interview bar, and the only thing that builds the
skill. The profile / teach-back is the DoD, not a green test.
