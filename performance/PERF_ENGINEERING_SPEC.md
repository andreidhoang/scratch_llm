# Performance Curriculum Engineering Spec — 2026

> **Pre-implementation registration.** FOP-2 (spec-with-falsifiers) + FOP-3 (predict-before-run)
> require predictions to be written *before* experiments. This doc is that registration. It encodes
> hardware gates, per-assignment scope, falsifiable predictions, definitions of done (DoD), and
> kill criteria for A1–A7. Do not start an assignment without reading its section here; do not
> log a result to `bench/RESULTS.md` without a corresponding prediction row here first.
>
> **Ordering mandate.** Complete A1 → A7 in order before resuming the CS336 A5 RL / DELTA thread.
> Rationale: A2–A5 kernel skills are direct prerequisites for DELTA (GDN-2 decode kernel); A7
> capstone is the natural bridge. Building DELTA before owning the tools is premature.
>
> **Mode-3 boundary.** The kernel implementations (the GEMM inner loop, the online softmax tile,
> the WGMMA mainloop) are Mode-3 territory: human implements from first principles. Claude agents
> write the failing correctness tests (bench-writer), explain the mechanism (kernel-tutor), run
> profiler diagnosis (roofline-analyst), and gate before commit (kernel-ship-reviewer). Never ask
> Claude to write the kernel body from scratch.

---

## 1. Standing Hardware — RTX PRO 4000 Blackwell (sm_120)

**Ground truth** (`bench/RESULTS.md` 2026-06-29, `[FACT]`):

| Metric | Measured | Used for |
|---|---|---|
| BF16 GEMM 8192³ | **~73 TF/s** | compute ceiling on standing GPU |
| HBM bandwidth | **~0.55 TB/s** | memory-bound ceilings |
| BF16 ridge point | **~130 FLOP/byte** | 73 TF / 0.55 TB; divides memory-bound from compute-bound |
| Memory capacity | **25 GB** | KV-cache sizing, model size limits |

**What sm_120 CAN run [documented]:**

| Instruction / feature | Arch gate | A#s benefiting |
|---|---|---|
| CUDA C++ kernels | all | A1–A7 |
| `nvcuda::wmma` / `mma.sync` fragment API | sm_70+ (Volta) | A3 rungs 0–2, A4 rungs 0–3 |
| `cp.async` LDGSTS (async SMEM copies) | sm_80+ (Ampere) | A2 SMEM pipelining |
| FP8 E4M3 / E5M2 math (compute ops) | sm_89+ (Ada+) | A5 FP8 numerics |
| FP4 E2M1 math (compute ops) | sm_120 (Blackwell consumer) | A5 FP4 numerics |
| Triton kernels compiled for sm_120 | sm_80+ | A2–A5 |

**What sm_120 CANNOT run [hard gates — will not compile or silently produce garbage]:**

| Instruction | Reason | Required for |
|---|---|---|
| `wgmma.mma_async` | Hopper (sm_90a) ONLY | A2 §4.1–4.5; A3 rungs 3–4 + §4.1; A4 rung 4 |
| `cp.async.bulk.tensor` (TMA) | sm_90+ | same |
| `tcgen05.mma` + TMEM | datacenter Blackwell (sm_100a) ONLY | A3 §4.2–4.3; A4 §7 |
| `cta_group::2` 2-SM MMA | sm_100 | A3 §4.2 |
| Real NVLink multi-GPU | SXM datacenter parts only | A6 |

**The single most important implication:** the ~10× jump from CUDA-core ceiling to tensor-core
ceiling (A2/A3 Colfax ladder: 32 → 317 TFLOP/s on H100) is **not reachable on sm_120**. Every
rung that requires WGMMA/TMA needs a rented H100. Plan accordingly.

---

## 2. Rental Plan — capability tiers, not GPU counts (ADR-0012, 2026-07-03)

> **First principle: a rental buys architecture-gated behaviors, not FLOPs.** Three hard gates
> decide what a tier can teach, and no GPU count below a gate substitutes for it:
> **(1) ISA generation** — `wgmma`/TMA/FP8-WGMMA are sm_90a-only; `tcgen05`/TMEM/native-NVFP4 are
> sm_100a-only; the standing sm_120 has neither (§1). **(2) HBM bandwidth & capacity** — decode is
> memory-bound (A1, measured), so the wall itself is the spec; B=1 ceiling = `BW / weight-bytes`.
> **(3) Interconnect domain** — TP/EP/collectives/disaggregation are only real inside one NVLink
> domain (900 GB/s/GPU); crossing nodes is the ~18× IB cliff, plumbing not primitives.

| Tier | Hardware | Gate it opens | B=1 ceiling, 70 GB-FP8 dense | Enables | Sessions × hrs · est. cost |
|---|---|---|---|---|---|
| 0 own | RTX PRO 4000 Blackwell (sm_120, 24 GB, 0.55 TB/s) | — (~80% of all work) | n/a (24 GB; 0.84B bf16 → 327 tok/s) | every non-ISA-gated rung of A1–A5; all serving algorithms + oracles + traces | standing · $0 |
| 1 rent | 1× H100 SXM (80 GB, 3.35 TB/s, sm_90a) | ISA: WGMMA/TMA/FP8 | 47.9 tok/s | A2 §4.1–4.5; A3 R3–4 + §4.1; A4 R4 (FA3); A5 FP8-WGMMA; **+ single-GPU frontier serving block** (Phase 2) | 1 × 12–15 h · ~$25–45 |
| 2 rent | **8× H200 SXM NVLink node** (1,128 GB, 4.8 TB/s/GPU) — *the crown* | interconnect: one NVLink domain **+ capacity: R1-FP8 fits** | 68.6 tok/s | A6 R0–R1 + **the frontier-MoE serving day**: DeepSeek-R1 FP8 TP×EP, MLA KV at scale, PD-disagg (Phase 4) | 1 × 6–10 h · ~$150–320 |
| 3 rent | 1× B200 (~192 GB, ~8 TB/s, sm_100a) | ISA: tcgen05/TMEM/NVFP4 | ~114 tok/s | A3 §4.2–4.3; A5 §7; DELTA NVFP4-state probe | 1 × 4–6 h · ~$25–45 |
| 4 skip | 2-node 16× over IB | the ~18× cliff itself | — | A6 R3 + §4.1/4.3/4.4 (optional; training-leaning) | optional |

**The capacity gate, worked (why Tier 2 is H200, not H100):** DeepSeek-R1 FP8 weights ≈ **671 GB**
vs 8×H100 = 640 GB raw (~600 usable) — *does not fit*, and the TP-8 shard (83.9 GB/GPU) exceeds one
H100's 80 GB anyway. 8×H200 = 1,128 GB → fits with ~450 GB for KV. MLA makes that KV budget huge:
R1 caches `(512+64) × 61 L × 2 B = 70.3 KB/token` (vs Llama-70B GQA-8: 327.7 KB — 4.7×), so
~450 GB ≈ **6.4 M cached tokens** (128 concurrent × 50 K ctx). Fallback if H200 unavailable:
8×H100 serving **Qwen3-235B-A22B FP8** (235 GB — fits with 365 GB headroom); every primitive
survives, only the R1 flag is lost. Budget-cut order: B200 → fold Tier 1 into the node day →
the node itself is irreducible (frontier serving is a multi-GPU problem, definitionally).

**Rental discipline (non-negotiable):**

1. Do NOT rent H100 until EVERY sm_120-runnable rung for that assignment is oracle-correct AND
   has a logged measurement in `bench/RESULTS.md`.
2. When you rent, arrive with a compiled, tested binary ready to run. Batch all H100 work into as
   few sessions as possible. Rental hours spent debugging compilation are wasted.
3. Checkpoint every rung's results to `bench/RESULTS.md` before the instance terminates.
4. On B200: read `tcgen05` PTX + Colfax Part 1–4 BEFORE renting. Compile locally first (cross-
   compile or check sm_120 ptx for sm_100 difference) to catch syntax bugs cheaply.
5. **Node day is on-demand, never interruptible** — a preempted TP/EP bring-up burns the day. A
   single-GPU kernel session may use interruptible pricing (work is checkpointed per rung).
6. **Verify the node before the clock matters:** `nvidia-smi topo -m` must show NV-links (NV8/NV18)
   between all pairs — reject PIX/PHB (PCIe) listings; ≥5 Gbps down + ≥1.5 TB disk (the R1 weight
   pull is 671 GB: ~20 min at 5 Gbps, kill the listing if ETA > 90 min); engine flags (SGLang/vLLM
   TP/EP/disagg) change fast — pin the engine image + smoke the exact launch command on Tier 0/1
   *before* the node session.

---

## 3. Universal Engineering Disciplines (applied to all A#s)

These are non-optional. A rung that violates any one of these produces no valid result.

**D1 — Oracle first.** Before any optimization, have a dead-simple reference and a test that
compares against it at a stated tolerance. The oracle can be slow; it cannot be wrong. If the
kernel cannot match the oracle on one small adversarial input, no performance number from it is
real. `[CITE: 00_foundations.md §5, CLAUDE.md engineering discipline §1]`

**D2 — Roofline on paper first.** Compute arithmetic intensity (`AI = FLOPs / HBM_bytes`).
Compare to the ridge (`peak_TFLOPS / peak_BW`). For sm_120: ridge ≈ 130 FLOP/byte (BF16). For
H100 SXM FP16: ridge ≈ 295 FLOP/byte. Memory-bound: optimize traffic. Compute-bound: optimize
math units. Roofline before profiler, profiler before code change.

**D3 — Lock clocks, warm up, use CUDA events.** `nvidia-smi -lgc <max>,<max>` before every
bench session. Discard ≥10 warmup iterations. Report median of ≥100 CUDA-event-timed runs. For
short kernels: use CUDA graphs to amortize launch overhead, or report the launch overhead
separately. Wall-clock with boosting clocks is inadmissible.

**D4 — One variable per experiment.** Change one thing, measure, write it down. Violating this
makes every measurement ambiguous.

**D5 — Pre-register predictions.** Write the expected number (+ bound type) in `bench/RESULTS.md`
before running. When reality diverges, investigate FIRST — a surprising improvement is as
suspicious as a regression.

**D6 — Adversarial inputs in correctness suite.** Random inputs are insufficient. Every kernel
needs: a large outlier, a zero row, a size non-multiple of tile, the degenerate shape (M=1), a
masked remainder. Production bugs hide in remainder tiles and boundary conditions.

**D7 — Write the artifact.** Every assignment produces a 2–3 page design note (what was built,
the roofline ladder, the Nsight evidence per rung, the gap to the library ceiling and why). Written
for a frontier-lab peer. "It got faster" is not an artifact; the specific named bound and the
evidence that moving it cost what was paid are the artifact.

---

## 4. Per-Assignment Engineering Spec

### A1 — Transformer Inference as a System

**Hardware:** sm_120 standing GPU for all rungs (0–3 + frontier core 4.1–4.6).
MLA demonstration and disaggregation can be at toy scale. FP8 KV bridges to A5.

**What to build (scope):**

| Rung | Build | Oracle | Correctness gate |
|---|---|---|---|
| 0 | Metrics harness (TTFT/ITL/throughput/goodput at p50/p95/p99) + PyTorch eager baseline | — | Metrics reproducible: re-run = same numbers (fixed seed, locked clocks) |
| 1 | Contiguous KV-cache decoder | Rung-0 greedy | Token-exact vs oracle |
| 2 | GQA/MQA reduction | Rung-0 GQA reference | Grouped math correct vs full-MHA on fixed seed |
| 3 | Continuous batching (iteration-level, Orca-style) | Rung-1 per-request greedy | Each request token-exact; ≥2× aggregate throughput vs static batch |
| 4.1 | PagedAttention (Triton paged-KV kernel; 16-token blocks; block table; CoW) | Rung-1 | Paged attn matches contiguous attn bit-for-bit on non-contiguous allocations |
| 4.2 | Chunked prefill (interleave decode steps between prefill chunks) | Rung-3 greedy | TTFT/ITL curve vs chunk size measurable; no output divergence |
| 4.3 | Lossless speculative decoding (rejection sampling; prove distribution-identical) | Rung-1 greedy | With greedy target: spec output = non-spec output token-exactly |
| 4.4 | CUDA graph decode | Rung-4.1 | Outputs identical; nsys shows CPU-gap elimination |
| 4.5 | MLA latent cache (toy scale) | Full K/V reconstruction | Weight-absorption identity: latent-space scores = reconstructed scores numerically |
| 4.6 | Prefill/decode disaggregation (2 processes, IPC KV transfer) | Rung-4.1 | Goodput vs co-located baseline on SLO-constrained trace |

**Falsifiable predictions (pre-registered, sm_120, 0.55 TB/s HBM):**

| Experiment | Predicted | Predicted bound |
|---|---|---|
| Decode AI (any BF16 model) | **1.00 FLOP/byte** | memory (`[FACT]` bench/RESULTS.md) |
| AI vs sm_120 ridge (130 FLOP/byte) | **130× below ridge** | — |
| Decode ceiling, 1B BF16 | **0.55e12 / 2e9 ≈ 275 tok/s** | memory |
| Decode ceiling, 7B BF16 | **0.55e12 / 14e9 ≈ 39 tok/s** | memory |
| PagedAttention memory waste (16-token blocks) | **<4%** vs >60% naive | overhead |
| Continuous batching throughput gain (B=32, mixed 50–500 tok) | **>2×** vs static | — |
| CUDA graph step-time reduction (B=1) | **~20–28%** (vLLM-V1 reported 28%) | launch overhead |
| Speculative decoding acceptance rate — code domain | **65–80%** | — |
| Speculative decoding acceptance rate — open-ended | **40–55%** | — |

**DoD (definition of done — all must be [FACT]-logged in bench/RESULTS.md):**
- [ ] Decode tok/s approaches `0.55 TB/s / 2P` ceiling (within 15%) for a BF16 model — proven memory-bound three ways (hand roofline, Nsight SoL Memory%≫Compute%, measured tok/s)
- [ ] PagedAttention waste <4%, contiguous-match test passing (non-contiguous blocks)
- [ ] Speculative decoding provably lossless: greedy output token-exact vs non-spec
- [ ] CUDA graph step-time reduction measured via nsys (CPU-gap before/after)
- [ ] Metrics harness produces TTFT/ITL/throughput/goodput at p50/p95/p99
- [ ] Design note written (2–3 pages, "where does inference time and memory go")

**Kill criteria:**
- If Rung-1 KV-cache decoder is not token-exact after 2 debug sessions → bisect to the attention slice; never proceed to Rung-2 with a broken oracle
- If PagedAttention overhead vs contiguous is >20% on a warm path → profile the gather first; do not add complexity (CoW, chunked prefill) on a slow baseline
- If speculative decoding output diverges from greedy target even once → the rejection rule is wrong; stop and fix the math before measuring speedup

---

### A2 — Kernel Optimization: From Coalescing to the Hopper Ceiling

**Hardware:**
- Rungs 0–6 (CUDA-core ladder): sm_120 standing GPU
- §4.1–4.5 (WGMMA/TMA/FP8 frontier): requires 1× H100 (sm_90a) → batch with A3/A4

**What to build (scope):**

| Rung | Build | Oracle | Correctness gate |
|---|---|---|---|
| 0 | Roofline harness: ncu automation, roofline plot, sweep-to-CSV | — | Reproduces existing bench/RESULTS.md baseline |
| 1 | GEMV ladder (naive → coalesced block-per-row → two-stage reduction → vectorized float4) | `torch.mv`, cublasSgemv, rtol≈1e-5 | Exact on random + adversarial (outlier, non-mult-of-warp) |
| 2 | Online softmax (running max + denom recurrence, warp-shuffle, fused) | `F.softmax`, rtol≈1e-3 FP16 | Adversarial: +1e4 outlier, all-equal, all-`-inf` masked row |
| 3 | RMSNorm + LayerNorm (Welford LN, warp shuffle) | `F.rms_norm`, `F.layer_norm`, rtol≈1e-3 | Zero row (ε path), single outlier, non-mult-of-warp length |
| 4 | Top-K ladder (naive → min-heap → parallel); fused online-softmax+TopK | `torch.topk` | Exact indices+values; tie-breaking defined and tested |
| 5 | GEMM CUDA-core part 1: naive → coalesced → SMEM tile | `torch.matmul`, rtol≈1e-2 | Must test ≥4096³ to separate L1 from real memory effects |
| 6 | GEMM CUDA-core part 2: 1D blocktiling → 2D → vectorized → (stretch: warptiling) | same | GEMM remainder tiles: M/N/K each non-multiples |
| 4.1 | Async copy → multistage TMA pipeline (H100) | Rung-6 | Bit-identical vs sync loader; out-of-bounds tile auto-predicated |
| 4.2 | WGMMA tensor-core mainloop (H100) | `torch.matmul`, rtol≈1e-2 | K non-mult of WGMMA-K; single outlier; masked remainder |
| 4.3 | Warp-specialized + persistent + Stream-K (H100) | same | Skinny-M, wave-quantized K, partial final wave |
| 4.4 | Epilogue fusion (scale/bias/activation/cast while accumulator in registers) | reference matmul+elementwise | Fused cast = two-kernel cast bit-for-bit |
| 4.5 | FP8 fine-grained-scaled GEMM (DeepGEMM style, two-level FP32 promotion) (H100) | BF16 reference; compare SQNR | Long-K with promotion OFF vs ON shows error exceeds tolerance without promotion |
| 4.6 | Benchmark: torch.compile/Triton floor + cuBLAS/CUTLASS ceiling | — | Statement: did hand kernel beat torch.compile? By how much vs CUTLASS? |

**Falsifiable predictions — sm_120 rungs:**

| Experiment | Predicted | Predicted bound |
|---|---|---|
| GEMV naive (FP32, 4096×4096) | **~12–20 GB/s** | memory (uncoalesced) |
| GEMV coalesced (block-per-row) | **~100–140 GB/s** (~5–8× naive) | memory |
| GEMV vectorized float4 | **~380–520 GB/s** (approaching 0.55 TB/s) | memory |
| Softmax online vs 2-pass: traffic ratio | **~1.3× fewer bytes** | memory |
| RMSNorm vs LayerNorm: time ratio | **~0.6–0.8× LN time** (1 reduction vs 2) | memory |
| Top-K "No Eligible" stall (ncu) | **>50%** warp stall on serial K-loop | latency/serial |
| GEMM naive (BF16, 4096³) | **~1.0–1.5 TF/s** (~1–2% of cuBLAS) | memory (uncoalesced) |
| GEMM 2D blocktiled (BF16) | **~46–54 TF/s** (~63–74% of ~73 TF/s cuBLAS) | compute |
| GEMM vectorized (BF16) | **~52–60 TF/s** (~71–82% of cuBLAS) | compute |
| GEMM Nsight: Memory% → Compute% inflection | between SMEM-tile and 2D-blocktile rungs | — |

**Falsifiable predictions — H100 (sm_90a) rungs (pre-registered before renting):**

| Experiment | Predicted | Source |
|---|---|---|
| WGMMA basic (4096³ FP16, H100 PCIe) | **~318 TF/s** | Colfax ladder |
| WGMMA larger tiles (H100) | **~433 TF/s** | Colfax |
| WGMMA + TMA 3-stage (H100) | **~504 TF/s** | Colfax |
| WGMMA + TMA + max-tile (H100) | **~618 TF/s = ~87% of cuBLAS** | book Table 7.2 |
| Warp-specialized persistent (H100 PCIe) | **~531 TF/s** (~71%) | PyTorch CUTLASS blog |
| Best CUTLASS FP16 (H100 PCIe) | **~630 TF/s** (~84%) | PyTorch CUTLASS blog |
| FP8 DeepGEMM two-level (H800 best-case) | **~1350–1550 TF/s** | `[INFERENCE]` DeepGEMM paper |

> **Honesty flag:** H100 PCIe ≠ H100 SXM. PCIe: 114 SMs, ~750 TF/s FP16 dense roof. SXM: 132 SMs,
> ~989 TF/s FP16 dense. If renting SXM (the common Vast/RunPod default), adjust all targets
> proportionally. Always state which SKU and dense vs sparse in every reported number.

**DoD:**
- [ ] GEMV ladder reaches bandwidth roof (>80% of 0.55 TB/s on sm_120); Nsight SoL Memory%≈100%
- [ ] Online-softmax proof: matches 3-pass on +1e4 adversarial row to <1e-3
- [ ] GEMM %-of-cuBLAS ladder reproduces siboehm shape (1%→8%→13%→37%→68%→78%) at ≥4096³ BF16
- [ ] Nsight shows Memory%→Compute% flip between SMEM-tile and 2D-blocktile rungs
- [ ] (H100) WGMMA+TMA hits ~10× jump vs CUDA-core ceiling; warp-specialized reaches ≥80% of PCIe FP16 peak
- [ ] FP8 kernel shows error exceeds tolerance without two-level promotion; within tolerance with it
- [ ] Design note: "Climbing from coalescing to the Hopper ceiling" (2–3 pages, per A2 §8)

**Kill criteria:**
- If GEMM rung N doesn't improve by >5% over rung N-1 after 2 attempts → document the failure mode, promote (it's likely L1 saturation or a profiler mis-read) — do NOT gold-plate
- If WGMMA kernel after warp-specialization lands at ~21 TF/s (the un-tuned spill) → this means `setmaxnreg` is wrong; stop, read the Colfax tutorial on register reallocation, fix, re-try
- If FP8 without two-level promotion performs BETTER than with → the promotion logic is wrong; two-level is always necessary for long-K FP8

---

### A3 — Tensor Cores: From WMMA Fragments to Blackwell Tensor Memory

**Hardware:**
- Rungs 0–2 (WMMA, mma.sync): sm_120 standing GPU
- Rungs 3–4 + §4.1 (WGMMA/TMA/warp-specialized): 1× H100 (sm_90a) — batch with A2
- §4.2–4.3 (tcgen05/TMEM/NVFP4-MMA): 1× B200 (sm_100a) — separate rental

**What to build (scope):**

| Rung | Build | Oracle | Correctness gate |
|---|---|---|---|
| 0 | Naive SMEM GEMM baseline (re-anchor from A2); 4096³ FLOP count | `torch.matmul` | Token-exact to FP-accumulate tolerance |
| 1 | WMMA GEMM + cp.async double-buffer (128×128 block, 8 warps, FP32 accumulate) | same | Element-exact vs Rung 0; FP16 accumulate variant shows error grow with K |
| 2 | mma.sync + ldmatrix + XOR-swizzled SMEM (explicit thread→value layout) | same | Bank-conflict count ≈0 in ncu (shared_ld/st_bank_conflict ≈0) |
| 3 | WGMMA + TMA 3-stage circular buffer (H100): basic → larger tiles → TMA → 3-stage | same | Swizzle atom matches TMA swizzle; descriptor validity bit checked; column-major path tested |
| 4 | WGMMA FP8 + wait_group<N>≥1 overlap (H100) | BF16 reference | FP8 relative error within FP8-GEMM expectation; report the error curve |
| §4.1 | Warp-specialized + persistent WGMMA GEMM (H100) | same | Tensor-pipe util ≥85% in ncu; not in the 21-TF un-tuned spill trap |
| §4.2 | tcgen05/UMMA GEMM with accumulator in TMEM; 1-SM then 2-SM (B200) | same | Full 128-lane drain verified (3/4 of tile not stale); 2-SM correctness: no double-count |
| §4.3 | NVFP4 block-scaled MMA: E2M1 elements + UE4M3 block scale via tcgen05.cp (B200) | BF16 reference | Scales staged correctly; FP4 needs per-block scaling to be usable |

**Falsifiable predictions — sm_120:**

| Experiment | Predicted | Note |
|---|---|---|
| WMMA GEMM (BF16, 4096³, sm_120) | **~30–44 TF/s** (40–60% of ~73 TF/s peak) | book basic WMMA = 1.7× CUDA-core baseline |
| mma.sync + ldmatrix + swizzle (sm_120) | **~44–55 TF/s** (60–75% of peak) | swizzle recovers bank-conflict tax |
| Book result: WMMA on H100 vs CUDA-core | **71 vs 43 TF/s** (1.7×) | `[FACT]` book Table 7.2 |

**Falsifiable predictions — H100 (sm_90a) (pre-registered):**

| Experiment | Predicted |
|---|---|
| WGMMA basic (4096³ FP16, H100) | **~318 TF/s** (from book) |
| WGMMA + larger tiles | **~433 TF/s** |
| WGMMA + TMA (3-stage) | **~618 TF/s = 87% of cuBLAS** |
| Warp-specialized persistent §4.1 | **≥85% of H100 989 TF/s dense FP16** |
| FP8 WGMMA rung 4 | **≥75% of 1,979 TF/s FP8 dense** |

**Falsifiable predictions — B200 (sm_100a) (pre-registered):**

| Experiment | Predicted |
|---|---|
| tcgen05/TMEM BF16 (4096³, B200, 1-SM) | **~1,209 TF/s** (~54% of 2,250 dense) early; **~1,300 TF/s** (~58%) with warp-spec |
| 1-SM → 2-SM gain (B200) | **~8%** (1,209 → 1,302 TF/s) — SMEM-bandwidth relief, not raw math |
| NVFP4 block-scaled element rate vs BF16 | **2× throughput** on the FP4 MMA tier |

> **Honesty flag carried verbatim:** A real tuned B200 BF16 GEMM lands **~1,476 TF/s ≈ 65% of the
> 2,250 dense spec ≈ 98% of cuBLAS** (gau-nernst tcgen05 reference). B200 dense BF16 spec =
> 2,250 TF/s; FP8 = 4,500; FP4 = 9,000. Keynote figures are usually **2× sparse** — always quote
> dense. sm_120 ≠ sm_100: the 5090 and PRO 4000 cannot run §4.2–4.3.

**DoD:**
- [ ] WMMA GEMM (sm_120): element-exact, ≥40% of sm_120 BF16 peak, FP32-accumulate justified
- [ ] mma.sync+swizzle (sm_120): bank-conflict count ≈0; ≥60% of peak
- [ ] PTX artifact: hand-decoded `wgmma.mma_async` (every qualifier annotated) + hand-encoded 64-bit SMEM descriptor
- [ ] (H100) WGMMA+TMA pipeline overlaps copy+compute on nsys timeline (not serialized)
- [ ] (H100) §4.1 warp-specialized persistent: ≥85% of H100 dense FP16 peak; tensor-pipe util ≥85%
- [ ] (B200, if rented) §4.2 tcgen05: accumulator in TMEM, full 128-lane drain correct, 2-SM result ~8% gain documented
- [ ] Design note: "register→SMEM→TMEM / sync→async arc" (2–3 pages, per A3 §8)

**Kill criteria:**
- If mma.sync swizzle does not reduce bank conflicts below 10% → profile the specific access pattern and re-derive the XOR stride; do not move to WGMMA until bank conflicts are solved
- If tcgen05 kernel produces garbage on 3/4 of the output tile → you are violating the 32-lane-per-warp drain rule; stop, re-read §2.3, fix the epilogue
- If the 2-SM tcgen05 gain is >25% → suspect a double-issue bug; re-check the leader-only issue path

---

### A4 — Flash Attention: The Kernel That Never Writes S to HBM

**Hardware:**
- Rungs 0–3 (WMMA-based FA1/FA2): sm_120 standing GPU
- Rung 4 (FA3-class Hopper, WGMMA+TMA+ping-pong): 1× H100 (sm_90a) — batch with A2/A3
- Rung 5 (FA4 stretch, tcgen05/TMEM): 1× B200 — separate rental

**What to build (scope):**

| Rung | Build | Oracle | Correctness gate |
|---|---|---|---|
| 0 | Naive 3-kernel attention (QKᵀ kernel, row-softmax, PV, host logic; explicit S in HBM) + metrics harness (FLOP/s, OOM at large N) | PyTorch SDPA no-flash backend | max|Δ| < 1e-3 FP16 |
| 1 | Single-row online softmax in Numba/1-thread CUDA (running m, d, o; no S stored) | 3-pass safe softmax | max|Δ| < 1e-6; adversarial: +50 outlier row |
| 2 | Fused tiled FA1: outer Q-tile loop, inner K/V-tile loop, WMMA QKᵀ, online-softmax update in SRAM, deferred ÷d, causal mask | Rung-0 | max|Δ| < 1e-3; constant SMEM regardless of N; no OOM at N=64K |
| 3 | FA2 work-partitioning: fewer non-matmul FLOPs, seq-dim parallelism, split-Q warp partitioning | Rung-0 | same; occupancy rose (nsys) |
| 4 | FA3-class (H100): producer/consumer warp-spec, TMA tile loads, WGMMA async matmul, ping-pong scheduling (SFU ∥ TC), FP8+incoherent processing | Rung-0 + PyTorch SDPA | FP16: < 1e-3; FP8: documented error budget vs FP16 |
| §4.3 | One variant: GQA (easiest), paged/sink attn, or MLA (hardest — proves weight-absorption) | variant reference | Variant output matches reference; KV-memory consequence measured |
| Backward | Flash-attention backward (store O+logsumexp, recompute S/P); gradient-check dQ/dK/dV | `torch.autograd` | < 1e-2 FP16 vs finite differences |

**Falsifiable predictions:**

| Experiment | Predicted | Predicted bound |
|---|---|---|
| Naive attention AI (FP16, N=4K, d=128) | **~d ≈ 128 FLOP/byte** (far left of ridge 130) | memory (heavily) |
| Naive attention: OOM threshold (25 GB, FP16) | **N ≈ 51K** (N² × 4heads × 2B = 25 GB) | — |
| Fused FA1/FA2 (WMMA, sm_120): % of peak | **~40–60% of sm_120 BF16 peak** (~29–44 TF/s) | compute (fused) |
| FA2 on A100 (book result) | **~50–73% of A100 BF16 peak** (~156–228 TF/s) | compute |
| FA2 on H100 (book result) | **~35% of H100 FP16 peak** (~346 TF/s) — motivates FA3 | compute |
| FA3 (H100 FP16, warp-spec + TMA + WGMMA + ping-pong) | **~700–740 TF/s = ~70–75% of 989 TF/s** | compute |
| FA3 FP8 | **~1.2 PF/s** | compute |
| Ping-pong proof: SFU and TC concurrent on nsys | **SFU pipe and tensor pipe busy simultaneously** | — |
| Online-softmax exactness | **Bit-exact vs 3-pass safe softmax** on any input (no approximation) | — |

> **Honesty flags:**
> - **FA3 numbers are from the paper** (arXiv:2407.08608): FP16 ~740 TF/s, FP8 ~1.2 PF/s. Your
>   implementation will trail FA3 by some gap — diagnosing that gap is the deliverable.
> - **FA4 BF16 B200 public numbers**: 1605 TF/s = 71% of B200 BF16 peak, 1.3× cuDNN 9.13, 2.7× Triton.
>   **FP8/FP4 FA4 numbers are NOT yet public** — do not invent them.

**DoD:**
- [ ] Fused FA (Rung 2) bit-exact vs R0; constant SMEM verified (constant regardless of N); no OOM at N=64K
- [ ] Online-softmax exactness proven at Rung 1: +50-outlier row to <1e-6 vs 3-pass
- [ ] Memory-bound → compute-bound shift demonstrated three ways (hand roofline, Nsight SoL, achieved TF/s)
- [ ] (H100) FA3-class: ≥70% of H100 FP16 peak; ping-pong concurrent SFU+TC on nsys; FP8 block-quant + incoherent processing within documented error budget; head-to-head vs FA3 + cuDNN with gap diagnosed
- [ ] One variant integrated and oracle-verified; KV-memory consequence measured
- [ ] Backward gradient-checks to <1e-2 FP16 vs finite differences
- [ ] Design note: "Why attention is a memory problem" (3–4 pages, per A4 §8)

**Kill criteria:**
- If Rung-2 fused FA matches R0 but SMEM footprint grows with N → the score tile is being written to HBM; bisect to the tile and find the store
- If ping-pong FA3 shows SFU and TC serialized on nsys → check the warpgroup assignment; re-schedule so exp-warpgroup ≠ WGMMA-warpgroup
- If FP8 attention without incoherent processing has max|Δ| > 0.1 vs FP16 → per-block quantization is insufficient; add the orthogonal-M rotation and verify the identity first

---

### A5 — Quantization: Precision as a Design Variable

**Hardware:** sm_120 for all numerics (R0–R4) and block-scaled GEMM (Triton/CUTLASS software path).
NVFP4 native MMA throughput on B200 — batch with A3 rental.

**What to build (scope):**

| Rung | Build | Oracle | Correctness gate |
|---|---|---|---|
| R0 | INT8 symmetric per-tensor quant/dequant | FP32 tensor | Round-trip MSE near zero; SQNR ≈ 44 dB clean tensor |
| R1 | INT8 asymmetric + per-channel (weights) / per-token (activations) | same | Asymmetric SQNR > symmetric on post-GELU skewed tensor; per-channel beats per-tensor |
| R2 | Group-wise INT4 (g=128), packed two-per-byte; unpack round-trip bit-exact | FP16 tensor | Pack then unpack = original codes; adversarial: odd length, all-0xF, all-zero block |
| R3 | NVFP4 two-level pack/unpack (per-tensor FP32 s_global + per-16-block FP8 E4M3 s_b + E2M1 element) + MXFP4 comparison variant (k=32, E8M0 scale) + block-scaled GEMM (Triton/CUTLASS) | BF16 tensor | Per-block MSE: NVFP4 < MXFP4 (the required result); round-trip bit-exact |
| R4 | FP8 E4M3 KV cache (per-channel-K / per-token-V) wired into A1 decoder | BF16 KV reference | End-to-end quality within noise of BF16 KV; K/V asymmetry measured |
| §4.3 | AWQ or GPTQ or QuaRot on a real linear layer | FP16 layer output | Measured accuracy recovery vs naive group-INT4 at same bit-width |

**Falsifiable predictions:**

| Experiment | Predicted |
|---|---|
| INT8 sym SQNR (clean tensor, no outlier) | **≈44 dB** (6.02×7+1.8; `[INFERENCE]`) |
| INT8 asym vs sym SQNR on post-GELU tensor | **asymmetric wins by ≥3 dB** |
| INT4 group-128 SQNR vs INT8 | **≈26 dB** (6.02×4+2; `[INFERENCE]`) |
| NVFP4 per-block MSE vs MXFP4 (same tensor) | **NVFP4 < MXFP4 by ≥20%** (finer block + mantissa-bearing scale) |
| FP8 E4M3 KV quality vs BF16 KV (perplexity) | **<0.1 perplexity point delta** |
| INT4 KV stress quality (no quant-aware finetuning) | **visible degradation: >1 perplexity point** |
| DeepGEMM two-level FP32 promotion: error reduction | **long-K error with promotion OFF exceeds tolerance; ON is within it** |
| NVFP4 dense element rate vs FP8 on B200 | **~2× element throughput** (hardware ratio, not e2e) |

> **Honesty flags:**
> - SQNR floors `≈ 6.02·b + c` are `[INFERENCE]` — the exact `c` depends on tensor and sign/magnitude
>   convention. Treat the floor as a calibrated expectation, not a spec constant. Deficit-means-bug is
>   load-bearing; the exact dB target is a sanity check.
> - NVFP4 two-level math constants (the `/6`, `/448`, placement order) **vary by implementation**.
>   Pin to one reference (TensorRT-Model-Optimizer or TransformerEngine), cite it, and document.
> - Quote "~5× hardware PFLOPS NVFP4 vs FP8" as a hardware ratio and "~2.3× e2e throughput" as a
>   workload-dependent measurement. Never conflate them.

**DoD:**
- [ ] INT8 sym/asym + per-channel/token: SQNR oracle on held-out data (calibration ≠ validation set); sym-vs-asym and per-tensor-vs-per-channel error deltas reported with numbers
- [ ] INT4 group-128: pack/unpack bit-exact on adversarial inputs (odd length, all-0xF, alternating signs, all-zero block)
- [ ] NVFP4: per-block MSE < MXFP4 per-block MSE **demonstrated by per-block error histogram**, gap attributed to E4M3 scale + k=16 mechanism
- [ ] FP8 KV: end-to-end quality within noise of BF16 KV; doubled capacity demonstrated; K/V asymmetry justified with measured error (per-channel-K vs per-token-K)
- [ ] PTQ method: measured accuracy recovery on real layer, not synthetic tensor
- [ ] Design note: "Precision as a design variable" (2–3 pages, per A5 §8)

**Kill criteria:**
- If INT4 pack/unpack fails the bit-exact test on any adversarial input → stop here; do NOT proceed to NVFP4 pack with a broken nibble implementation
- If NVFP4 per-block MSE ≥ MXFP4 on the same tensor → bug in the two-level scale placement; re-check against pinned reference before assuming the spec prediction is wrong
- If FP8 KV shows quality parity with BF16 even at INT4 KV stress intensity → the stress test is wrong; raise quantization aggressiveness until degradation is visible

---

### A6 — Distributed Training & Inference as One Communication Problem

> **Inference re-aim (ADR-0012, 2026-07-03).** A6's *primitives* stand, but their execution vehicle
> is the **Phase-4 serving day on 8×H200** (`PERF_PLAN.md` Phase 4): Rung 0 (topology + busbw) and
> Rung 1 (TP MLP micro) run inside it; the serving-native distributed surface — TP×EP on a real
> frontier MoE, NVLink collectives at decode message sizes, PD-disaggregation — replaces the
> training-leaning depth. Rung 2 (1F1B pipeline: PP is a cross-node *serving* tool but a
> single-node *training* exercise) and Rung 3 + §4.1/§4.3/§4.4 (multi-node) move to the optional
> Phase 5 — the one fact they add for inference (the ~18× NVLink→IB cliff) is §4.2, learnable from
> single-node numbers + nccl-tests docs.

**Hardware:**
- Rung 0 (NCCL baseline + topology map) + Rung 1 (TP micro): inside the Phase-4 **8× H200** day
- Rung 2 (1F1B pipeline) + Rung 3 + §4 multi-node EP: **optional Phase 5** (2-node, 16× H100)

**What to build (scope):**

| Rung | Build | Oracle | Correctness gate |
|---|---|---|---|
| 0 | `nvidia-smi topo -m` read; nccl-tests busbw sweep; Ring vs Tree vs NVLS comparison | — | busbw >80% of NVLink line rate for large messages |
| 1 | Megatron TP MLP (GEMM-1 col-parallel, GEMM-2 row-parallel, one AllReduce; f/g backward conjugate) | Single-GPU MLP | Numerically identical to single-GPU; exactly 2 AllReduce in forward |
| 2 | 1F1B pipeline (naive blocking → streams+events → 1F1B → interleaved 1F1B, v virtual stages) | Single-GPU forward | Pipeline output matches single-GPU; bubble fraction tracks `(p-1)/m` |
| 3 | Multi-node EP all-to-all (hand-rolled NCCL AllToAll → DeepEP integration); DeviceMesh DP×TP×PP×EP | Single-GPU reference | MoE routing tolerance-validated vs CPU reference; busbw vs RDMA line rate |
| §4.1 | EP all-to-all with compute/comm overlap (DualPipe decomposition) | same | nsys timeline: all-to-all not on critical path; ~0-SM comm overhead (DeepEP) |
| §4.2 | Bandwidth-cliff plot: NVLink vs IB busbw across message sizes | nccl-tests | Ratio ≈ 18× (900 vs 50 GB/s); mis-mapped TP/EP collapses MFU |
| §4.3 | MFU/HFU at 16/32/64 GPUs; six-killer decomposition | C≈6ND formula | MFU in 35–55% band; six killers itemized with numbers |
| §4.4 | DeviceMesh single-config re-mapping | — | Config change: one parallelism layout → another without correctness loss |
| §4.5 | Async DCP checkpoint + elastic torchrun --max-restarts + rank-kill recovery | — | Kill rank mid-run; recover from last async checkpoint with re-sharding on load |

**Falsifiable predictions:**

| Experiment | Predicted | Note |
|---|---|---|
| AllReduce busbw (NVLink, 8× H100, large tensor) | **>80% of 900 GB/s = >720 GB/s** | `[INFERENCE]` from book |
| TP-8 scaling efficiency (single-node, 8× H100) | **~98–100%** | book result |
| Pipeline naive efficiency (4 GPUs, 1 microbatch) | **~27%** (1/(4−1+1) = 25%, book gets 27%) | book result |
| Pipeline streamed (4 GPUs, 8 microbatches) | **~100%+ overlap** | book's 4.17× |
| MFU at 8 GPUs (dense, well-tuned) | **~85–90% scaling efficiency** | `[INFERENCE]` |
| MFU at 16 GPUs (2-node) | **~70–80% scaling efficiency** | book band |
| MFU at 32+ GPUs | **~50–70% scaling efficiency** | book band |
| NVLink→IB bandwidth cliff | **~18× drop** (900 → 50 GB/s/GPU) | |
| Mis-mapped TP across node boundary: MFU drop | **>20% MFU collapse** (the book's "28% per-GPU tax") | |

**DoD:**
- [ ] nccl-tests busbw >80% of NVLink line rate; Ring/Tree/NVLS crossover measured and plotted
- [ ] TP-8 numerically identical to single-GPU; exactly 2 AllReduce in forward; comm <20% of step time
- [ ] Pipeline bubble tracks `(p-1)/m` formula; interleaved 1F1B shows factor-v bubble shrink
- [ ] Multi-node EP all-to-all working and measured; DeviceMesh composes DP×TP×PP×EP
- [ ] EP all-to-all provably hidden under compute (zero SM cost on nsys timeline)
- [ ] Bandwidth cliff plot: NVLink vs IB annotated with line rates and ~18× ratio
- [ ] MFU at 16/32/64 GPUs in the realistic band, with six-killer decomposition
- [ ] Async checkpoint + elastic restart exercised (kill-rank → recover → re-shard)
- [ ] Design note: "Which tensor did I split, which wire paid for it, and what's my MFU?" (3–4 pages)

**Kill criteria:**
- If TP AllReduce takes >30% of step time → TP degree is too high for the GEMM size; reduce TP or increase batch; never add PP/EP on a broken TP baseline
- If MFU stays <30% after 2 optimization attempts → unhidden collective is the culprit; use nsys to find it before proceeding
- If elastic restart cannot re-shard the checkpoint → the DTensor descriptors are wrong; do not accept "it works on same layout" as passing this gate

---

### A7 — Capstone: Build Something a Frontier Lab Would Use

**Pick one track. Depth over breadth.**

**Track recommendation for this repo:** **Track B (Kernel Suite)** — directly bridges to DELTA
(GDN-2 decode kernel). The three kernels are exactly what DELTA needs: a fast GEMM (the linear
recurrence projection), a fused attention variant (the hybrid attention component), and low-precision
inference (NVFP4 weights + FP8 KV). Track B is the natural pre-DELTA warmup.

**Track B scope:**
1. Warp-specialized WGMMA+TMA persistent GEMM (H100) OR tcgen05/UMMA GEMM (B200) vs cuBLAS
2. FA3-class FP8 attention kernel vs FA3 + cuDNN (uses A4 work)
3. NVFP4 block-scaled GEMM (proves speedup + accuracy vs MXFP4 for documented reason)

**Capstone DoD:**
- [ ] All three kernels: oracle-correct, locked-clock benchmarks, Nsight SoL + roofline placement, dense-vs-sparse + PCIe-vs-SXM caveats
- [ ] Each hits its A# frontier target (GEMM ≥80% of cuBLAS on target shape; FA3 ≥70% H100 peak; NVFP4 within <1% accuracy of FP8)
- [ ] Reproducible from clean instance via a script
- [ ] Design doc: hypothesis → architecture → number → gap attribution → next experiments (3–5 pages)
- [ ] (Strongly encouraged): a PR to FlashInfer / CUTLASS / vLLM or a public writeup

---

## 5. Measurement Protocol (universal, every rung)

```bash
# Step 0: lock clocks (do this before EVERY benchmark session)
sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)

# Step 1: warm-up (≥10 iters, discarded)
# Step 2: CUDA-event-timed iterations (≥100)
# Step 3: report: median, p5, p95 (not min, not mean)

# Step 4: compute achieved TFLOP/s = 2×M×N×K / median_time_seconds  (for GEMM)
# Step 5: compute % of peak = achieved / (peak_dense_TFLOPS_for_this_GPU)
# Step 6: log to bench/RESULTS.md immediately (before the session ends)
```

**Every number in bench/RESULTS.md must pin:** date · rung · GPU model + sm_XX · shape + dtype ·
achieved metric · bound type · root cause (1 line) · next experiment. A number without these
qualifiers is inadmissible.

---

## 6. Context Engineering Integration

**Where to look to orient:**
- This file (`performance/PERF_ENGINEERING_SPEC.md`): what exactly to build and the pre-registered predictions
- `performance/PERF_PLAN.md`: phased sequencing, which rung to start next, rental batching
- `docs/STATUS.md` §"Performance Curriculum": current pass/fail status per rung
- `bench/RESULTS.md`: the actual measured numbers (append-only, dated)
- `performance/A#_*.md`: the full curriculum spec for each assignment

**What to update on completing a rung:**
1. Log the measurement to `bench/RESULTS.md` (with the prediction from this doc for comparison)
2. Update `docs/STATUS.md` §"Performance Curriculum" to mark the rung complete
3. Update `performance/PERF_PLAN.md` "Current phase" to the next rung
4. If a prediction was wrong: add a `[FACT]` note here explaining the discrepancy

**Agent types for each phase of work:**
- Writing the failing test + bench scaffold before kernel work: `bench-writer`
- Understanding a mechanism before implementing: `kernel-tutor`
- Post-implementation profiler diagnosis: `roofline-analyst`
- Pre-commit gate: `kernel-ship-reviewer`

**The Mode-3 boundary in every rung:**
- Claude WRITES: the failing correctness test (oracle comparison), the benchmark harness, the roofline prediction
- Human IMPLEMENTS: the kernel body (GEMV, softmax, GEMM loop, WGMMA mainloop, online softmax tile)
- Claude REVIEWS: after implementation, before commit

---

## 7. Integration with CS336 A5 RL / DELTA

Complete A1 → A7 before resuming:
- **A5 RL (GRPO/Dr.GRPO)**: unblocked immediately after A7
- **DELTA (GDN-2 decode kernel)**: requires A1 (serving primitives), A2 (kernel optimization), A3 (tensor cores), A4 (attention fusion), A5 (quantization) — completing this curriculum IS the prerequisite
- **A2 distributed (DDP/ZeRO-1/FSDP)**: can be done in parallel with A6, both are distributed work

**Bridge point:** A7 Capstone Track B kernel suite → DELTA is a direct continuation. The GDN-2
decode kernel is essentially a specialized fused GEMM+attention kernel at low precision for
recurrent models — exactly what A2+A3+A4+A5 build toward.
