# A2 — Systems · BUILD GUIDE (CS336 → reasoningLLM L2 Systems)

> **One-liner.** A2 makes a single GPU fast (Triton FlashAttention-2, mixed precision, activation checkpointing) and many GPUs coherent (DDP→overlap, ZeRO-1, FSDP) — the **L2 Systems** layer. It feeds two source files: `rollout/sglang_client.py` (the rollout/inference engine) and `utils/monitors.py` (the three-KL monitor incl. `kl_train_infer` HALT@0.10).
> **Through-line.** Cheap rollouts = an affordable ablation grid (every kernel/comms win buys more VERA runs), AND the same train-vs-infer engine machinery is exactly what `kl_train_infer` measures. The systems layer is not a detour from the RLVR thesis — it is the instrument that makes the thesis measurable and cheap.

---

## 1. What CS336 actually requires (every deliverable)

CS336 A2 (v26.1.3) has six implementation themes: (1) benchmarking/profiling harness, (2) activation checkpointing, (3) FlashAttention-2 Triton kernel, (4) DDP naive→overlapped, (5) optimizer-state sharding (ZeRO-1), (6) FSDP. Plus an analytical parallelism section (DP/FSDP/TP comms math) and an 8B leaderboard. Test adapters live in `tests/adapters.py`; tests are run with `uv run pytest`.

| # | Deliverable (exact PDF problem name) | PDF §  | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|---|
| 1 | `benchmarking_script` — init model, random batch, w warm-ups + n timed steps (fwd / fwd+bwd / +optim), `cuda.synchronize()` | 2.1.3 | (script; no adapter) | 0.5 d | **LOAD-BEARING** (the *predict-before-run* + sync discipline) |
| 2 | `nsys_profile` — Nsight Systems profiling, 5 written Q&A (top kernel, non-matmul kernels, softmax vs matmul runtime-vs-FLOPs) | 2.1.4 | (analysis; no adapter) | 0.5 d | COURSE-ROTE (the *tables*); LOAD-BEARING (the *softmax-vs-matmul* intuition) |
| 3 | `mixed_precision_accumulation` — fp32 vs fp16 accumulation drift | 2.1.5 | (written) | 0.1 d | LOAD-BEARING (1 pt; the numerics lesson behind `kl_train_infer`) |
| 4 | `benchmarking_mixed_precision` — autocast dtypes of params/activations/LN/logits/loss/grads; BF16 vs FP32 timings | 2.1.5 | (written + script) | 0.3 d | LOAD-BEARING (autocast dtype rules; LN-in-fp32) |
| 5 | `memory_profiling` — `torch.cuda.memory._record_memory_history`, memory_viz timelines; residual-activation accounting | 2.1.6 | (script + written) | 0.5 d | LOAD-BEARING (activation memory math = A2.2 one-pager) |
| 6 | `gradient_checkpointing` — memory-optimal recursive checkpointing strategy + O(√N) tradeoff | 3.2 | (written + script) | 0.5 d | **LOAD-BEARING** (O(√N) peak-activation memory; the recompute lever) |
| 7 | `pytorch_attention` — benchmark vanilla SDPA across d∈{16,32,64,128} × seq∈{256…16384}; find OOM point; memory accounting | 4.1.1 | (script) | 0.3 d | LOAD-BEARING (the *why FA2 exists* roofline seed) |
| 8 | `torch_compile` — compiled vs eager attention + whole-model fwd/bwd/optim | 4.2 | (script) | 0.2 d | COURSE-ROTE (but anchors the "beat-the-compiler" framing) |
| 9 | `flash_forward` (a) **pure-PyTorch** FA2 fwd autograd.Function | 4.2.2 | `get_flashattention_autograd_function_pytorch` · `test_flash_forward_pass_pytorch` | 0.5 d | **LOAD-BEARING** (the tiled/online-softmax reference) |
| 10 | `flash_forward` (b) **Triton** FA2 fwd kernel + autograd.Function (Algorithm 1) | 4.2.2 | `get_flash_autograd_function_triton` · `test_flash_forward_pass_triton` | 1.5 d | **LOAD-BEARING** (the headline kernel; A2.1) |
| 11 | `flash_forward` (c) causal masking flag (`is_causal: tl.constexpr`, −1e6 mask) | 4.2.2 | (same triton fn, `is_causal=True`) | 0.3 d | LOAD-BEARING (required for the leaderboard + real rollouts) |
| 12 | `flash_backward` — FA2 backward via recomputation, PyTorch + `torch.compile` (Eqs 13–19) | 4.2.2 | `test_flash_backward` | 0.5 d | LOAD-BEARING (the recomputation backward; D-vector trick) |
| 13 | `flash_benchmarking` — `triton.testing.do_bench`, FA2 vs PyTorch fwd/bwd/e2e, seq 128…65536, bf16+fp32, B200 | 4.2.2 | (script) | 0.3 d | LOAD-BEARING (this IS A2.1's roofline data) |
| — | OPTIONAL: Triton FA2 backward (Algorithm 2; P twice to skip atomics) | 4.2.3 | (leaderboard) | (skip) | **SKIP** unless chasing leaderboard |
| 14 | `distributed_communication_single_node` — benchmark all-reduce, 1MB…1GB, 2/4/6 GPUs, gloo vs nccl | 5.1.1 | (script) | 0.3 d | COURSE-ROTE (but the comms-cost intuition is LOAD-BEARING) |
| 15 | `naive_ddp` — all-reduce each param grad after backward | 5.2 | `get_ddp` (opt: `ddp_on_after_backward`) · `tests/test_ddp.py` | 0.3 d | LOAD-BEARING (the DP baseline; A2.2) |
| 16 | `naive_ddp_benchmarking` — time/iter + % in comms, xl, 1×2 GPU | 5.2 | (script) | 0.2 d | COURSE-ROTE |
| 17 | `minimal_ddp_flat_benchmarking` — flatten grads into one all-reduce | 5.3.1 | (script) | 0.2 d | LOAD-BEARING (gradient bucketing — the real DDP trick) |
| 18 | `ddp_overlap_individual_parameters` — `register_post_accumulate_grad_hook`; overlap comm with bwd | 5.3.2 | `get_ddp` container · `tests/test_ddp.py` | 0.5 d | **LOAD-BEARING** (overlap = the actual production DDP design) |
| 19 | `ddp_overlap_individual_parameters_benchmarking` — time/iter + Nsight overlap proof | 5.3.2 | (script + screenshots) | 0.2 d | COURSE-ROTE |
| 20 | `optimizer_state_sharding` — wrap an Optimizer; each rank owns ~1/world_size of params; broadcast after step | 6 | `get_sharded_optimizer` · `tests/test_sharded_optimizer.py` | 1.0 d | **LOAD-BEARING** (ZeRO-1; A2.2 memory math) |
| 21 | `optimizer_state_sharding_accounting` — peak-mem breakdown (params/grads/Adam) + ZeRO-1 ($P_{os}$) compare | 6 | (written) | 0.2 d | **LOAD-BEARING** (the 100B memory one-pager) |
| 22 | `fsdp` — wrap module; all-gather weights for fwd/bwd, reduce-scatter grads, fp32 master / bf16 compute | 7 | `get_fsdp` · `tests/test_fsdp.py` | 1.5 d | LOAD-BEARING (full ZeRO-3; but **time-box** — see §6) |
| 23 | `fsdp_accounting` — expected peak-mem savings vs §6; Nsight all-gather-in-time check | 7 | (written + screenshots) | 0.2 d | COURSE-ROTE |
| 24 | `alternate_ring_all_reduce` — time of an alt ring all-reduce | 8.1 | (written) | 0.1 d | COURSE-ROTE |
| 25 | `data_parallel_calcs`, `fsdp_calcs`, `tp_calcs`, `fsdp_tp_calcs` — comms-vs-compute bottleneck algebra | 8.2–8.5 | (written) | 0.5 d | LOAD-BEARING (the roofline-for-comms; *when does scaling stop*) |
| 26 | Leaderboard — fastest full training step, 8B, BF16, causal, 2×B200, B=2/seq=32768 | 9 | (passes basics tests) | (skip) | **SKIP** (capped, GPU-dollar sink) |

> Honest note on the GPU target: §4.2 benchmarks specify **B200** and the leaderboard **2×B200**. On a rented A100/4090 you run the *same code* at smaller seq-lengths and report your GPU honestly. FA2 (not FA3/FA4) is the correct target on pre-Hopper hardware — see §5.

---

## 2. The equations/algorithms that matter (senior extraction)

1. **FlashAttention-2 tiling + online softmax (Algorithm 1).** Outer loop over query tiles $Q_i$ ($B_q\times d$), inner loop over key tiles. Maintain running max $m_i^{(j)}=\max(m_i^{(j-1)},\,\mathrm{rowmax}(S_i^{(j)}))$, unnormalized numerators $\tilde P_i^{(j)}=\exp(S_i^{(j)}-m_i^{(j)})$, running denominator $l_i^{(j)}=\exp(m_i^{(j-1)}-m_i^{(j)})\,l_i^{(j-1)}+\mathrm{rowsum}(\tilde P_i^{(j)})$, and rescale the accumulator $O_i^{(j)}=\mathrm{diag}(\exp(m_i^{(j-1)}-m_i^{(j)}))O_i^{(j-1)}+\tilde P_i^{(j)}V^{(j)}$. Normalize once at the end ($O_i=\mathrm{diag}(l_i)^{-1}O_i$) and store $L_i=m_i+\log l_i$. *Why it matters:* memory IO and peak no longer scale as seq_len² — this is the single change that unblocks long-context rollouts and is the canonical frontier-kernel interview question.
2. **FA2 backward via recomputation (Eqs 13–19).** Precompute $D=\mathrm{rowsum}(O\circ dO)$. Recompute $S=QK^\top/\sqrt d$, $P=\exp(S-L)$ from saved $Q,K,V,O,L$ (never store $P$). Then $dV=P^\top dO$, $dP=dO\,V^\top$, $dS_{ij}=P_{ij}(dP_{ij}-D_i)$, $dQ=dS\,K/\sqrt d$, $dK=dS^\top Q/\sqrt d$. *Why it matters:* the $D$-vector + $L$-vector trick is what lets you skip storing the seq_len² probability matrix in the backward — the whole point of FlashAttention's memory win, and the cleanest demonstration of recompute-vs-store.
3. **The roofline (arithmetic intensity, % of peak).** Attention does $\Theta(N^2 d)$ FLOPs against $\Theta(N^2)$ (vanilla) or $\Theta(Nd)$ (flash) HBM traffic; arithmetic intensity = FLOPs/bytes, compared against the GPU's ridge point (peak FLOP/s ÷ peak HBM BW). Below the ridge you are **memory-bound**, above it **compute-bound**. *Why it matters:* it tells you *which* number to optimize and lets you state "FA2-Triton hit X% of SDPA peak at seq 4k, here's why" — a senior signal, and a well-analyzed negative is itself a valued artifact.
4. **AdamW / optimizer-state memory accounting (~16–20 bytes/param).** Per param: 4 B fp32 master weight + 4 B grad + 4 B Adam $m$ + 4 B Adam $v$ = **16 B** (mixed precision adds a 2 B bf16 weight copy and possibly 2 B bf16 grad → ~18–20 B). *Why it matters:* this is the "train a 100B model" math — 100B × 16 B = **1.6 TB** of state, far past one GPU, which is *why* DP+TP+PP and ZeRO exist. The crisp one-pager is A2.2.
5. **DDP gradient bucketing + overlap.** Naive DDP all-reduces each param tensor separately (one collective per tensor, all after backward). Improvements: (a) **flatten** all grads into one buffer → one all-reduce (amortizes per-call latency); (b) **overlap** — fire each grad's all-reduce from a `register_post_accumulate_grad_hook` the instant it is ready, so communication hides under the still-running backward. *Why it matters:* overlap is the difference between toy DDP and production DDP; comms hidden under compute ≈ free.
6. **ZeRO-1 optimizer-state sharding (and ring all-reduce cost).** Each rank holds optimizer state for only ~1/world_size of params, steps its shard, then broadcasts updated params back. Ring all-reduce (= reduce-scatter + all-gather) costs $2\frac{N-1}{N}\frac{S}{W}$ — communication volume is independent of N to first order. *Why it matters:* ZeRO-1 cuts the 16 B/param optimizer state by world_size at almost no extra comms vs DDP; ZeRO-2/3 (grads, then params) is FSDP. This is the memory-vs-comms tradeoff every distributed-training role probes.

> **Hardware caveat (state explicitly — it's a senior signal).** FlashAttention-**3** is **Hopper-only** (warp-specialization, TMA, FP8); FlashAttention-**4** moved to **CuTeDSL on Blackwell**. On A100/4090 you honestly target **FA2**. And `torch.compile` already emits fused Triton attention — so the premium is on *beating the compiler*, which you must say out loud.

---

## 3. Map to reasoningLLM_scratch source files

The clean-room `src/` tree is stubbed-but-empty; the targets below are the planned L2 files named in `CLAUDE.md` and `README.md`. A2 builds the systems substrate; only a thin slice lands in the repo (v0.1.0 ships the *engine + smoke run*, not a kernel competition).

| CS336 deliverable | → reasoningLLM target | keep / adapt vs course-only |
|---|---|---|
| `flash_forward` Triton kernel + `flash_backward` (FA2) | (kernel lives in the served policy / `cs336-basics`; **not** a v0.1.0 ship file) | **course + add-on A2.1**; the *artifact* is the roofline, not a repo file. Keep as a benchmark; do not block v0.1.0 on it. |
| `pytorch_attention` / `flash_benchmarking` roofline | A2.1 roofline plot + Nsight trace (docs artifact) | **adapt** → the "% of SDPA" number; feeds the *cheap-rollout* argument (faster attention = more ablation runs). |
| `benchmarking_script` + `cuda.synchronize` discipline | `rollout/sglang_client.py` (latency/throughput timing of rollouts) | **adapt** → reuse the warm-up + sync + predict-before-run harness to benchmark rollout throughput. |
| `torch_compile`, `memory_profiling`, KV/activation memory math | `rollout/sglang_client.py` (KV-cache sizing) + A2.2 one-pager | **adapt** → activation/KV memory accounting is the same algebra that sizes the rollout engine's KV cache. |
| mixed precision (autocast dtypes; LN-in-fp32; fp16 accumulation drift) | `utils/monitors.py` (`kl_train_infer`) | **the bridge.** The train engine (bf16 autocast, fp32 master) and the serving engine (often fp8/int8 KV, different kernels) compute *different logits*. That divergence **is** `kl_train_infer`. The mixed-precision lesson is the mechanistic *why*. |
| DDP overlap / ZeRO-1 / FSDP collectives | `rollout/sglang_client.py` (multi-GPU serving) + A2.2 | **adapt** the comms/overlap mental model; **course-only** for the full FSDP class (v0.1.0 doesn't need it, but the *vocabulary* does). |
| `optimizer_state_sharding_accounting` (100B memory math) | A2.2 one-pager (docs artifact, interview answer) | **course + add-on**; a standalone senior-signal artifact, not a repo file. |

> **The single bridge to internalize.** `kl_train_infer = KL(train‖infer)` (HALT @ 0.10, `utils/monitors.py`) measures the gap between **training-engine logits** and **serving-engine logits**. A2 is where you *learn why that gap exists*: different kernels (your Triton FA2 vs the engine's fused attention), different precision (bf16 train vs fp8/int8-KV serve), different reduction order (the fp16-accumulation problem in `mixed_precision_accumulation`). Without A2 you cannot explain *why* `kl_train_infer` is ever nonzero; with A2 it becomes the discipline pillar of the whole repo (R3 measured ~94% of tokens differ in ≥1 layer under naive MoE-RL).

---

## 4. Map to core context docs

- **`UNIFIED_FRONTIER_PROJECT_SPEC.md §3 — "L2 · A2 — Systems: the money layer".** Core = Triton FA2 (fwd+bwd) + DDP + optimizer-state sharding + Nsight. Three add-ons:
  - **A2.1 kernel roofline** — FA2-Triton vs `F.scaled_dot_product_attention`, roofline plot (arithmetic intensity, % peak, BW- vs compute-bound). *Hypothesis:* FA2-Triton ≥ X% of SDPA at seq 4k. *Kill:* if you can't beat 60% of SDPA, ship the roofline + honest gap (a well-analyzed negative is a valued artifact, per Neel Nanda).
  - **A2.2 100B memory-math + ZeRO** — DDP + ZeRO-1 (+ a TP toy). Write params+grads+Adam ≈ 16–20 B/param → why 100B needs DP+TP+PP and hundreds of GB of state.
  - **A2.3 rollout engine + `kl_train_infer`** — minimal KV-cache + SGLang client; measure train-logits vs serve-logits divergence. *Falsifiable:* `kl_train_infer` > HALT@0.10 on a MoE policy under naive GRPO. *Feeds:* `rollout/sglang_client.py`, `utils/monitors.py` — *the* reasoningLLM discipline pillar.
  - §5 / §3 framing: the L2 layer maps to the **Inference / Systems RE** column; verbatim interview leverage = "what is `kl_train_infer` and why does it matter for RL?", "train a 100B model (DP+TP+PP + memory math)", "GPU kernels … low-precision numerics" (xAI).
- **`CAPSTONE_AND_STUDY_PLAN.md §3 — row A2** (line 127). "Efficient kernels + the inference engine the `kl_train_infer` pillar monitors → `rollout/sglang_client.py`, `utils/monitors.py`; cheap rollouts = affordable ablation grid; backed by CS336 L10 (inference)." §1.4 (lines 33): FA3 Hopper-only → FA2 is itself a falsifiable finding. **Threading order** (line 134): A1 → **A2** → A5; A2 runs partly in parallel — every speedup pays for more ablation runs (line 169).
- **`STUDY_PLAN_2026.md §0.8 — A2-first block, Days 1–4** (lines 254–258):
  - Day 1: A2 profiling harness + `cuda.synchronize` + mixed precision → repo audit / CI.
  - Day 2: A2 activation checkpointing (O(√N)) + fusion + **L4 MoE** → `envs/exploitability.py`.
  - Day 3: A2 FlashAttention-2 Triton kernel (fwd+bwd) → `envs/true_quality.py`.
  - Day 4: A2 **DDP + ZeRO-1 + FSDP; derive `kl_train_infer`** → `rewards/hack_detector.py`.
  - Companion table (lines 274–277): Day 4 = "distributed #3 · inference #5 → train-infer drift (R3/GSPO)"; Day 3 = "xAI Inference RE · GPU kernels low-precision numerics". Note **G1 (EOD 3)**: cannot Feynman-pass L2/L4 → Day 4 re-foundation.
- **repo `CLAUDE.md` — L2 row** (line 34: "A2 · Triton FA2 · DDP/ZeRO · KV-cache · SGLang client → `rollout/sglang_client.py`, `utils/monitors.py`") + **discipline #4** (lines 46–49): *mandatory RL logging* = the **three KLs separately** — `KL(current‖ref)`, `KL(current‖old)`, **`kl_train_infer = KL(train‖infer)`** + IS-ratio histograms + reward stats + length stats; `kl_train_infer` **HALT@0.10**. This is the non-negotiable that A2 makes implementable.

---

## 5. The frontier 2026 lens

**Commoditized (do not over-invest):**
- `torch.compile` emits fused Triton attention automatically — a *baseline* fused kernel is free. Writing one to match the compiler proves competence but isn't a moat.
- Naive all-reduce DDP and "wrap-it-in-DDP" are table stakes; every framework ships them.
- Exhaustive Nsight tables and per-kernel timing dumps are mechanical reporting, not insight.

**Scarce (where the value and the comp are):**
- **Beating the compiler / hand-tuned kernels** — the premium is the delta over `torch.compile`, stated with a roofline.
- **EP all-to-all overlap** and MoE-aware comms — every frontier flagship is sparse; hiding expert-routing comms under compute is hard and hot.
- **Inference as the frontier P&L** — serving is where the money and the wall-clock live; SGLang/vLLM internals, speculative decoding (EAGLE/Medusa), and **KV-cache as the #1 cost lever** (60–85% of wall-clock past ~1M context) are the scarce skills.
- **The train-vs-infer engine drift** — the *measurement* of `kl_train_infer` is itself frontier: R3 (`2510.11370`) and GSPO (`2507.18071`) exist precisely because MoE-RL collapses when train and infer engines disagree. Owning this measurement is the differentiator.

**2026 signals to cite:** FA3 (Hopper) → FA4 (CuTeDSL/Blackwell); DeepSeek-V3 at ~250 TFLOP/s/GPU (and MLA's KV-cache cut); SGLang/vLLM + spec-decode + paged KV-cache + low-bit-KV (KIVI 2-bit); and the train↔infer drift (R3/GSPO) that `kl_train_infer` operationalizes.

---

## 6. Prioritization verdict — what matters / what to skip

**The ~20% that is load-bearing (do these to Feynman depth):**
1. **FA2 Triton kernel (fwd `flash_forward`, bwd `flash_backward`) + the roofline (`flash_benchmarking`, A2.1).** The headline systems artifact and the canonical interview kernel. The roofline + honest "% of SDPA" is shippable even if the kernel is slow.
2. **DDP overlap + ZeRO-1 + the 100B memory math (A2.2).** `ddp_overlap_individual_parameters`, `optimizer_state_sharding`, and the `optimizer_state_sharding_accounting` one-pager — the "train a 100B model" answer.
3. **The `kl_train_infer` bridge (A2.3).** Mixed-precision + KV-cache + multi-engine logit divergence → `utils/monitors.py` HALT@0.10. **This is the single most load-bearing thing in A2** — it is the repo's discipline pillar and the reason the systems layer exists in this project at all.

**Course-rote (do them, but don't gold-plate — minimum to pass + the one insight):**
- `nsys_profile` written tables, `naive_ddp_benchmarking`, `ddp_overlap…_benchmarking`, `fsdp_accounting`, `distributed_communication_single_node`, `alternate_ring_all_reduce`. Extract the *one* intuition from each (softmax-vs-matmul runtime; comms-cost scaling) and move on. Don't manufacture exhaustive Nsight comparison tables.

**Explicit SKIP (time-boxed sprint; GPU-dollar sinks):**
- The **8B leaderboard** (§9) — capped reward, large GPU spend.
- The **OPTIONAL Triton FA2 backward** (§4.2.3, Algorithm 2) — do the `torch.compile` PyTorch backward instead.
- **Exotic parallelism** beyond the analytical `tp_calcs`/`fsdp_tp_calcs` algebra — do the *math* (when does scaling become comms-bound), skip implementing TP/PP/2D mesh. A **toy TP** only if Day-4 has slack.
- **FSDP (`fsdp`)** implementation — LOAD-BEARING conceptually but **time-box hard**: if Day 4 is tight, do ZeRO-1 fully + the FSDP *memory accounting* (written), defer the FSDP *class*. v0.1.0 ships the engine + smoke run, not a sharded trainer.

---

## 7. Build checklist (ordered, with discipline gates)

> Discipline gates from repo `CLAUDE.md`: **predict-before-you-run** (write the falsifiable number first), **fixed-seed reproducibility**, and **mandatory three-KL logging**.

**Day 1 — harness + numerics (predict first):**
1. [ ] `benchmarking_script`: init model, random batch, w warm-ups, n timed steps; **always `torch.cuda.synchronize()` before/after timing** (CUDA is async). Toggle fwd / fwd+bwd / +optim via CLI flags.
2. [ ] **Predict-before-run:** write down expected fwd-pass ms for `small` at seq 512 *before* running; the gap is your debugging anchor.
3. [ ] `mixed_precision_accumulation` + `benchmarking_mixed_precision`: nail the autocast dtype table (matmuls→bf16, LN/reductions→fp32, master weights→fp32). This is the *why* of `kl_train_infer`.

**Day 2 — memory levers:**
4. [ ] `memory_profiling`: dump `memory_snapshot.pickle`, read the active-memory timeline; compute residual-activation MiB per `TransformerBlock` from first principles and check it matches.
5. [ ] `gradient_checkpointing`: implement recursive checkpointing; state the **O(√N) peak vs O(N) compute** tradeoff and validate measured peak against the next-smaller/larger block size.

**Day 3 — the kernel (the headline):**
6. [ ] `flash_forward` (a) pure-PyTorch tiled FA2 (online softmax, save $L,Q,K,V,O$) → pass `test_flash_forward_pass_pytorch`. This is your *correctness oracle* for the Triton port.
7. [ ] `flash_forward` (b) Triton kernel per **Algorithm 1**, launch grid $(T_q, \text{batch})$, single key-loop, advance block pointers at loop end, fp32 accumulators → pass `test_flash_forward_pass_triton`.
8. [ ] `flash_forward` (c) `is_causal` flag (default False so prior tests pass; −1e6 additive mask).
9. [ ] `flash_backward` via `torch.compile` recomputation (Eqs 13–19, compute $D$) → pass `test_flash_backward`.
10. [ ] **Predict-before-run + roofline (A2.1):** predict "% of SDPA at seq 4k" *first*; then `flash_benchmarking` (`triton.testing.do_bench`) and **plot the roofline** (arithmetic intensity vs % peak, mark BW- vs compute-bound). Kill-criterion: <60% of SDPA → ship the roofline + honest gap.

**Day 4 — distributed + the bridge:**
11. [ ] `naive_ddp` → `minimal_ddp_flat_benchmarking` (flatten) → `ddp_overlap_individual_parameters` (post-accumulate-grad hook) → pass `tests/test_ddp.py` (run ×5).
12. [ ] `optimizer_state_sharding` (ZeRO-1) → pass `tests/test_sharded_optimizer.py` (×5); write the **A2.2 100B memory one-pager** (16–20 B/param → 1.6 TB).
13. [ ] **Derive `kl_train_infer` and wire the three-KL logging** (`utils/monitors.py`): log `KL(current‖ref)`, `KL(current‖old)`, **`kl_train_infer`** separately + IS-ratio histogram + reward/length stats; assert **HALT @ 0.10**. This is the day's bridge artifact and clears **G1** (Feynman-pass L2).
14. [ ] (slack only) FSDP class or toy TP; otherwise FSDP *accounting* written.

---

## 8. Open questions / ADR triggers

1. **FA2-only on rented GPU.** FA3 is Hopper-gated, FA4 is Blackwell/CuTeDSL. Decision: target FA2 on A100/4090 and state the hardware honestly. **ADR if** the rented GPU is actually Hopper (then FA3 is fair game and the roofline target shifts).
2. **SGLang vs vLLM for `rollout/sglang_client.py`.** The repo names SGLang; vLLM has wider adoption and paged-attention maturity. **ADR trigger:** if multi-turn tool-call rollouts or a specific KV-cache feature force the choice — log the train↔infer logit-parity behavior of each, since that directly determines baseline `kl_train_infer`.
3. **Where does the FA2 kernel actually live in the repo?** It is not a v0.1.0 ship file (the engine + smoke run is). **ADR:** keep FA2 as a benchmark/`cs336-basics` artifact and the roofline as a docs artifact, OR vendor a minimal kernel into the served policy path. Don't block the sprint on it.
4. **FSDP scope.** Full FSDP class vs ZeRO-1 + FSDP-accounting-only. **ADR trigger:** if Day 4 runs over, defer the FSDP *implementation* to post-sprint and ship the *memory math*; v0.1.0 does not require sharded training.
5. **What precision does the serving engine actually use?** If SGLang/vLLM silently quantizes the KV-cache (fp8/int8), baseline `kl_train_infer` may be nonzero *before any RL*. **ADR:** measure and record the serving engine's precision contract as a fixed input to the HALT@0.10 threshold.
