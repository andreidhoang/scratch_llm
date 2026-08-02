# Mastery-debt ledger — read-and-teach-back backlog (ADR-0013 + ADR-0014)

> Under delegate mode (ADR-0013) and the main-track sprint (ADR-0014) agents ship first; the Navigator masters from the shipped code.
> Every shipped main-track module adds one row here. A row is **cleared** when its Vietnamese
> Feynman lesson exists in `docs/learning/` and the Navigator has taught it back. Ordering is the
> suggested study order (dependencies first).

| # | Concept | Files (code + tests) | The interview question it answers | Lesson | Cleared |
|---|---------|----------------------|-----------------------------------|--------|---------|
| 1 | ZeRO-1 optimizer-state sharding (owner-broadcast) | `utils/zero1.py` · `tests/test_zero1.py` | How does ZeRO-1 cut memory, and what does it cost in comms vs DDP? | — | ☐ |
| 2 | Training-memory accounting (16–20 B/param → 100B ⇒ TBs) | `utils/memory_math.py` · `docs/design/A2_100B_MEMORY_ONEPAGER.md` | How would you train a 100B-parameter model? | — | ☐ |
| 3 | FSDP / ZeRO-3 mechanics (gather-fwd · reduce-scatter grads · fp32 master shards) | `utils/fsdp.py` · `tests/test_fsdp.py` | What moves on the wire in FSDP, and why ~1.5× DDP bytes? | — | ☐ |
| 4 | Parallelism comms algebra + ring all-reduce | `utils/comms_calc.py` · `docs/design/A2_COMMS_ALGEBRA.md` | When does scaling become comms-bound? | — | ☐ |
| 5 | IsoFLOP / Chinchilla fit (log-log, C=6ND, a+b≈1) | `scaling/isoflop.py` · `bench/a3_isoflop.png` | Derive and defend a scaling law — and when does it break? | — | ☐ |
| 6 | Budget-constrained query planning (reserve/refund semantics) | `scaling/planner.py` | Spend a fixed compute budget to fit a loss surface | — | ☐ |

| 7 | Exact + MinHash/LSH dedup (P[match]=Jaccard, LSH S-curve, cluster-and-drop) | `data/dedup.py` · `tests/test_data_dedup.py` | Explain MinHash+LSH and how you'd set the band count | — | ☐ |
| 8 | Data filter family + quality-classifier signal design | `data/filters.py` · `data/quality.py` · `data/pipeline.py` | How does data quality change a scaling outcome? Pipeline order? | — | ☐ |
| 9 | SFT masked cross-entropy + response-mask primitives | `algos/sft.py` · `tests/test_sft_algos.py` | Why does correct response-masking gate every downstream RL number? | — | ☐ |
| 10 | Expert Iteration (STaR) + verifiable-reward grader | `algos/expert_iteration.py` · `rewards/r1_zero.py` · `envs/countdown.py` | Why does filter-then-SFT already improve reasoning, and where does it plateau? | — | ☐ |
| 11 | GRPO / Dr.GRPO (group advantage, clip trust region, length-norm de-bias) | `algos/grpo.py` · `tests/test_grpo_algos.py` · `docs/adr/ADR-0017` | Derive GRPO's group-relative advantage; what bias does Dr.GRPO remove? | — | ☐ |
| 12 | DPO loss + Bradley-Terry reward modeling | `algos/dpo.py` · `tests/test_dpo_algos.py` | RLHF (PPO+RM) vs DPO — what's the closed-form reduction? | — | ☐ |
| 13 | Triton FlashAttention-2 backward (recomputation, D-vector, atomic dQ, model wiring) | `kernels/flash_attention_triton.py` · `model.py` | Derive FlashAttention-2's backward pass, the D-vector optimization, and explain why dQ needs atomic operations. | — | ☐ |

*(rows appended as modules ship — see `docs/EXECUTION_SPEC_CS336_FINISH.md` for the build DAG)*

> **Upcoming (2026-08-02) — K3 core/ hand-build modules** (`situ · kda · gated_mla · latent_moe ·
> attn_res`, per `docs/k3/ROADMAP.md:56–61`) become mastery-debt rows here as each lands; unlike the
> delegate-mode rows above these are hand-built from day one, so the debt is the teach-back gate, not
> the implementation.

---

## Raw CUDA C++ fundamentals (PPPM chapters — the Triton Trap gap)

> **Vì sao mục này tồn tại.** Codebase kernel work là ~95% Triton (`.py` files). Triton tự động
> hoá coalescing, tiling, bank-conflict avoidance, reduction — đúng là các skill mà NVIDIA k_live
> yêu cầu viết **bằng tay trong CUDA C++**. Khoảng trống này là rủi ro cao nhất cho Lane-K.
> Tham chiếu: [`BOOK_CHAPTER_MAP.md`](BOOK_CHAPTER_MAP.md) · k_live spec:
> `JOB_SPRINT/challenges/nvidia/k_live/spec.md`.

| # | Concept | Files (code + tests) | The interview question it answers | Source chapter | Lesson | Cleared |
|---|---------|----------------------|-----------------------------------|---------------|--------|---------|
| P1 | Warp shuffle reduction + block reduction + atomic (`__shfl_down_sync`) | `csrc/fundamentals/reduction_warp.cu` (TO BUILD) | k_live Level 2: write a reduction kernel using warp shuffle + shared memory + atomic | PPPM Ch10 | — | ☐ |
| P2 | Parallel prefix sum / scan (Kogge-Stone intra-warp + block-level) | `csrc/fundamentals/prefix_scan.cu` (TO BUILD) | Write a parallel scan; when Kogge-Stone vs Blelloch? (MoE routing, stream compaction) | PPPM Ch11 | — | ☐ |
| P3 | Tiled matrix transpose with bank-conflict-free SMEM | `csrc/fundamentals/tiled_transpose.cu` (TO BUILD) | k_live Level 3: tiled transpose, shared memory banking, coalescing | PPPM Ch5/6 | — | ☐ |
| P4 | Tiled GEMM in CUDA-core (NOT tensor-core) | `csrc/fundamentals/tiled_gemm.cu` (TO BUILD) | k_live Level 4: SMEM double-buffer, register accumulation, occupancy tuning | PPPM Ch6/18 | — | ☐ |
| P5 | SM architecture: warps, divergence, occupancy, latency hiding | (reading — no code) | k-arch: "explain warp execution model, divergence, occupancy" | PPPM Ch4 | — | ☐ |
| P6 | Memory coalescing patterns + thread coarsening | (reading — no code) | k_live grading: "explain your memory access pattern" | PPPM Ch6 | — | ☐ |
