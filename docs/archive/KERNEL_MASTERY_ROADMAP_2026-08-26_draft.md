> **ARCHIVED 2026-08-26 (same day as written).** This untracked root-level draft violated
> PLAN.md rule 6 ("no new plan documents"). Its ten load-bearing contributions (operating
> principles incl. clock-locking + correctness ladder, sm_120 hardware plan, GEMV decode
> twin, epilogue fusion, FA backward, MoE kernels, torch.library integration layer,
> agent+profiler loop, DELTA capstone spec, job-mapping) were harvested into
> `docs/KERNEL_MASTERY_SPEC.md` §6 (v2, 26/08 evening). Its week grid and capstone choice
> are superseded by the 26/08 PLAN.md reorder (K2-first; PLAN.md wins). Kept verbatim
> below for provenance.

# KERNEL MASTERY ROADMAP — Dang Huy × Claude
## From verified foundations to frontier-grade kernels, with DELTA as capstone
### Refactored from the Vizuara Kernel Workshop (Oct 2026 cohort) · calibrated to 2026 frontier-lab job requirements · v1.0, Aug 26 2026

---

## 0. Operating principles (non-negotiable, from frontier-lab practice)

1. **Profile before optimizing.** Nsight Systems for the timeline, Nsight Compute for the kernel. Never touch code before you know which roofline wall you're against.
2. **Lock clocks before benchmarking.** `nvidia-smi --lock-gpu-clocks=<base>,<base>` (or `-lgc tdp,tdp` where supported). Unlocked clocks make every A/B comparison a lie.
3. **Correctness before speed.** Every kernel ships with: numerical test vs reference (fp64 reference, measured ULP/rel-error), gradient test if it has a backward, and adversarial tests (NaN/Inf inputs, empty tensors, seq_len=1, non-contiguous strides, odd shapes).
4. **`torch.compile` is your floor.** If your custom kernel isn't clearly beating the compiled PyTorch path, it doesn't earn its maintenance cost.
5. **One variable at a time, written prediction before every run.** "I predict coalescing the transpose takes it from X GB/s to Y GB/s because Z" — write it, run it, reconcile. The reconciliation is where the learning is.
6. **Worklog everything.** Every kernel gets a `WORKLOG.md`: version table (v1, v2, …), % of peak, the one change, the NCU counter that explains the gain. This *is* your interview portfolio.
7. **Kill criteria.** Each optimization idea gets a time box. Past it, log the negative result and move on.

## 0.5 Hardware plan (you rent; you don't own — plan around it)

| Phase | Hardware | Why | Approx budget |
|---|---|---|---|
| Phases 1–3, 5 (most work) | RTX 5090 (vast.ai) | sm_120: full CUDA model, tensor cores w/ FP8+FP4 `mma`, TMA bulk-copy, Nsight works | ~$0.4–0.7/hr |
| Phase 4a (Hopper deep-dive) | H100 SXM (vast.ai) | `wgmma` + full TMA + warp-specialization only exist on sm_90a | ~$2–3/hr, ~30–40 hrs total |
| Phase 4b (FA4/Blackwell DC, optional) | B200 (short rental) | `tcgen05` + TMEM are sm_100-only | few hours, reproduce-numbers only |

**Critical fact:** RTX 5090 = sm_120 (consumer Blackwell). NO `wgmma`, NO `tcgen05`/TMEM. It DOES have: 5th-gen tensor cores via `mma` (incl. FP8, and block-scaled FP4/NVFP4 paths used by CUTLASS sm120 kernels), TMA (`cp.async.bulk.tensor`), large L2, GDDR7 bandwidth (~1.8 TB/s). This makes it a superb daily driver for everything except Hopper/DC-Blackwell-specific instruction work.

**Discipline for rentals:** prepare everything locally first (code compiles under `nvcc --arch=sm_120` via CI or CPU-side checks), write the exact experiment list, then rent, execute, capture `.ncu-rep` files + CSVs, kill the instance. GPU-hours are for *measurement*, not for *thinking*.

---

## Phase 1 — FOUNDATIONS, VERIFIED BY BUILDING (Weeks 1–2, compressed)
*Maps to Vizuara Part I (their weeks 1–3). You've studied this; now you prove it with builds-from-memory, timed.*

### 1.1 The roofline as the governing law
- Derive machine balance for: your CPU, RTX 5090, H100 SXM, B200 (peak FLOP/s ÷ peak bytes/s for each precision).
- **Lab R1:** benchmark harness (provided: `lab1_roofline.py`) — measure achieved bandwidth (memcpy/triad), achieved TFLOP/s (big GEMM), then classify 10 real ops (elementwise add, softmax, layernorm, GEMV, GEMM at several shapes, embedding lookup, attention prefill vs decode shapes) as compute- vs memory-bound. Predict first, measure second, reconcile in the worklog.
- Exit: you can compute, from first principles, the *maximum possible speedup* of any fusion before writing it.

### 1.2 CUDA model + SASS literacy
- Rebuild from memory: vector add, then a reduction (grid-stride, warp-shuffle `__shfl_down_sync`, then cooperative-groups version). No references open.
- `cuobjdump -sass` / Godbolt every kernel. You must be able to point at: the load instructions (LDG.E.128?), the FMA density, predication from divergence.
- GPU Puzzles (Sasha Rush) as a 1-evening speedrun — it's a calibration test, not a course.
- Exit: reduction hits >85% of measured peak bandwidth; you can narrate its SASS.

### 1.3 Memory hierarchy in anger — the transpose ladder
- **Lab T1 (the ladder):** naive transpose → coalesced via shared-memory tile → +1 padding to kill bank conflicts → vectorized 128-bit → occupancy tuning. Five NCU reports; a table of `gld_efficiency`, `shared_ld_bank_conflict`, achieved GB/s at each rung.
- Exit: ≥90% of your measured copy bandwidth; you can explain every rung's gain from the counters, not vibes.

## Phase 2 — THE GEMM WORKLOG (Weeks 3–5)
*Maps to Vizuara Part II. This is THE interview screen at every lab. You've read Boehm and CUTLASS-the-hard-way; now you produce your own worklog with your own numbers.*

### 2.1 SIMT GEMM ladder (on 5090)
K1 naive → K2 coalesced → K3 shared-mem tiled → K4 1D thread-tile → K5 2D thread-tile (register tiling) → K6 vectorized float4 → K7 autotuned (block/tile sweep) → K8 warptiling. Target: >90% of cuBLAS SGEMM on your card. Every version: % of peak, NCU counters, one-paragraph explanation.

### 2.2 Tensor-core GEMM
- K9 WMMA (portable API) → K10 raw `mma.sync` PTX with `ldmatrix`, shared-memory swizzling to kill bank conflicts on the fragment loads → K11 `cp.async` double-buffered pipeline (Ampere-style async), then TMA bulk-copy variant (sm_120 supports it).
- Precision menu measured, not recited: FP16/BF16/TF32 accumulate behavior, error vs fp64 reference at K=4096.
- **Add what Vizuara skips:** an epilogue — fused bias+GELU epilogue in the same kernel; this is the CUTLASS "epilogue fusion" concept every JD lists.
- Exit: tensor-core GEMM ≥85–90% of cuBLAS FP16 on 5090; a written worklog you could defend line-by-line in an interview.

### 2.3 The decode-side twin: GEMV / batched GEMV
*(Missing from Vizuara; essential for inference roles.)* Decode is GEMV-shaped and bandwidth-bound. Build a batched GEMV hitting >90% of memory bandwidth; derive tokens/sec ceiling for a 7B model on your card from bandwidth alone.

## Phase 3 — PROFILING MASTERY + ATTENTION (Weeks 6–8)
*Maps to Vizuara Part III, extended.*

### 3.1 Forensic profiling
- NCU sections fluency: SOL, memory workload analysis, scheduler stats, warp-state sampling (what does "stall: long scoreboard" vs "stall: MIO throttle" tell you?).
- **Sabotage drill (self-inflicted):** take your K8/K11 GEMM, introduce one of {broken coalescing, bank conflicts, register spill via `-maxrregcount`, low occupancy, divergent branch}, hand the binary to yourself the next morning, diagnose from NCU alone in <30 min each.

### 3.2 FlashAttention from scratch
- Derive online softmax on paper (the running-max + rescale recurrence; prove it equals the exact softmax). Feynman test: explain to a rubber duck in Vietnamese without notes.
- **FA-v1 in CUDA:** forward, causal, one head — correctness vs `torch.nn.functional.scaled_dot_product_attention` at fp32 tolerance.
- **FA-v2 restructure:** move the outer loop to Q-blocks, parallelize across seq for better SM occupancy, non-matmul-FLOP reduction. Measure vs SDPA/FlashAttention library on the 5090.
- **Backward pass** (Vizuara barely touches it; training roles demand it): recompute-based dQ/dK/dV, the logsumexp trick, atomics vs split reductions.
- **Decode kernel:** flash-decoding style split-K over KV, single-query. This connects directly to DELTA.

## Phase 4 — THE FRONTIER (Weeks 9–13)
*Maps to Vizuara Parts IV–V, re-ordered around hardware access.*

### 4a. Hopper block (H100 rental weeks)
- TMA: `cp.async.bulk.tensor` + mbarriers; producer/consumer warp specialization; `wgmma` warpgroup MMA; persistent kernels + cluster launch.
- Project: Hopper GEMM in the DeepGEMM style — study DeepGEMM's ~300-line scheduling core first, then implement your reduced version; target ≥ cuBLAS on at least a band of shapes.
- Read FA3 paper against your FA2 kernel: pingpong scheduling, softmax-MMA overlap, FP8 attention w/ incoherent processing.

### 4b. The authoring layer: Triton → CUTLASS → CuTe-DSL
- Triton: re-implement fused softmax, layernorm, FA2 fwd in Triton; compare your CUDA vs Triton perf and *iteration speed* honestly.
- CuTe: layouts as the algebra (shape:stride, composition, swizzles) — do the layout exercises until `Layout<Shape<_2,_2>,Stride<_1,_2>>` composition is mechanical.
- CuTe-DSL (Python): port your GEMM; this is the FA4-native authoring layer and the 2026 shift (FA4 & DeepGEMM-class kernels are Python-DSL now).
- Study FA4 source (Dao-AILab flash_attn/cute): async-MMA pipeline, software-emulated exp (polynomial on FMA units), conditional/adaptive softmax rescaling, tile scheduler. On 5090 you read + partially port concepts; numbers-reproduction on B200 is optional.

### 4c. Inference-serving kernels (5090)
- PagedAttention: implement a paged KV decode kernel (block tables, gather from non-contiguous pages).
- Speculative decoding: verification kernel (batched accept/reject logic) + measure end-to-end.
- Quantization kernels: FP8 GEMM w/ per-tensor + per-block scaling; W4A16 dequant-fused GEMV (marlin-style ideas); NVFP4 block-scaled GEMM via CUTLASS sm120 path — the 5090 actually supports this, which is a rare edge you have.
- **MoE kernels (missing from Vizuara, demanded by DeepSeek-era JDs):** grouped GEMM, token permute/unpermute (scatter/gather), fused expert epilogues.

### 4d. What Vizuara omits that JDs require — the integration layer
- Custom ops done right: `torch.library` + `torch.compile` interop (fake tensors, meta functions), CUDA graphs capture, stream semantics.
- Communication basics: NCCL all-reduce/all-gather mechanics, overlap of TP collectives with compute; read one NVSHMEM kernel-initiated-communication example. (Awareness-level, not mastery — but you must speak it.)
- vLLM/SGLang code reading: trace one forward pass from Python to kernel launch; know where your kernel would plug in.

## Phase 5 — AI-WRITTEN KERNELS (Week 14, interleaved earlier)
*Maps to Vizuara Part V — and to your Claude Code expertise, which makes this YOUR unfair advantage.*
- KernelBench: run the harness, get baseline numbers for an agent (Claude) on 10 problems.
- Build the agent+profiler loop: Claude Code writes kernel → compile+correctness gate → NCU metrics fed back → iterate. Log where it fails (correctness of async pipelines, occupancy reasoning) — that failure analysis is a publishable/portfolio artifact and matches where the field is (AlphaEvolve-style search + human supervision).
- Meta-skill: you supervising AI kernel-writers is precisely the 2026-2027 role description.

## Phase 6 — CAPSTONE: DELTA (Weeks 12–18, overlapping)
*Replaces Vizuara's Crusoe capstone with something better: fused hybrid-attention/SSM inference kernels.*

**Spec (frontier-lab format):**
- **Hypothesis:** a fused kernel for hybrid blocks (Mamba-2/Gated-DeltaNet scan + sliding/cross attention) can beat the composed baseline (separate scan kernel + FA kernel + glue) by ≥1.5× at decode batch ≤ 32 on sm_120, by eliminating intermediate HBM round-trips.
- **Baselines:** mamba-ssm reference kernels, FLA (flash-linear-attention) Triton kernels, composed PyTorch+FA path, `torch.compile` of the composed path.
- **Milestones:** M1 chunked selective-scan kernel (correctness vs reference, fp32); M2 decode-step fused scan+attention for one hybrid layer; M3 seeded-scan + mixed-mask cross-attention fusion (your DELTA design); M4 integration as a torch custom op into a runnable hybrid model; M5 worklog + writeup + (stretch) upstream a piece to FLA or vLLM.
- **Evals locked FIRST:** exactness (rel-err vs fp64 chunky reference over 1k random shapes incl. adversarial), tokens/sec at B∈{1,8,32}, memory traffic (NCU dram__bytes), vs all four baselines, clocks locked, 5 runs, report median±spread.
- **Kill criteria:** if M2 fusion shows <1.15× over composed baseline with profiler confirming no remaining HBM round-trip to eliminate, write the negative-result postmortem and pivot capstone to the MoE grouped-GEMM track.

## Job-mapping table (2026, verified against live postings)

| You build | JD line it answers | Source |
|---|---|---|
| GEMM worklog to ~90% cuBLAS + tensor cores + epilogue fusion | "CUDA kernel optimization", "tensor core optimization", "kernel fusion" | Anthropic Perf Eng (GPU), OpenAI Inference |
| FA1→FA4 lineage + backward + decode + paged KV | "Flash Attention", "attention mechanisms", "vLLM/SGLang stack" | Anthropic, RadixArk/SGLang |
| Triton + CUTLASS/CuTe + CuTe-DSL ports | "Triton or CUTLASS at scale", "ML kernel ecosystem" | OpenAI, NVIDIA-adjacent |
| FP8/NVFP4 quantized GEMMs on sm_120 | "low-precision: INT8/FP8", "emerging quantization formats" | Anthropic |
| torch.library/torch.compile integration | "PyTorch/JAX internals, torch.compile, custom operators" | Anthropic |
| NCCL awareness + overlap experiments | "NCCL, NVLink, collective communication" | Anthropic |
| Agent+profiler loop + KernelBench study | the AI-kernels loop frontier startups productionize | Wafer-class startups |
| DELTA capstone | "co-design attention mechanisms for next-gen hardware"; a defensible portfolio piece | Anthropic representative projects |

## Cadence
- Mon/Wed/Fri deep blocks (mirror the cohort rhythm): 1 concept session (with me, first-principles + math) + 1 build session (rented GPU, measured).
- End of every session: 3-sentence worklog entry. Weekly: promote findings, archive dead ends.
- Every kernel's WORKLOG.md is written as if a staff engineer will review it — because in interviews, one will.
