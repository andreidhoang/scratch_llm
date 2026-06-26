# A2 — Systems · BUILD GUIDE (CS336 → scratch_llm Systems layer)

> **📖 Read first (slides → this build):** Lectures **5 → 6 → 7 → 8** — GPUs (pdf) · kernels (py) ·
> multi-GPU parallelism (py) · parallelism basics (pdf); then **10** (inference/KV-cache, py) and
> revisit **2** for the 100B memory math. Full map + read-order: [`../LECTURE_MAP.md`](../LECTURE_MAP.md).

> **STATUS: 🟡 partial.** Built and green: FlashAttention-2 (pure-PyTorch oracle + autotuned Triton
> forward + the roofline) in `kernels/`, KV-cache incremental decode, `utils/monitors.py`, the
> rollout seam + `LocalBackend`. **To build next (all CPU/gloo-buildable):** DDP (naive → flat-bucket
> → overlap), ZeRO-1 optimizer-state sharding, **FSDP** (a full graded deliverable), and the
> 100B-model memory one-pager. Real SGLang serving needs a Hopper box (ADR-0008). See
> [`../STATUS.md`](../STATUS.md).

> **One-liner.** CS336 A2 ("Systems") makes a **single GPU fast** (a Triton FlashAttention-2 kernel,
> mixed precision, activation/gradient checkpointing) and **many GPUs coherent** (DDP → comm/compute
> overlap, ZeRO-1 optimizer-state sharding, FSDP). This is the **Systems layer** of the scratch_llm
> stack — the engineering that turns the A1 substrate into something you can train at scale and serve
> cheaply. It is also the densest interview surface in the whole course: GPU kernels + the roofline,
> the KV-cache, and "how would you train a 100B-parameter model?" (DP + TP + PP + the memory math) are
> the questions a frontier lab asks an inference/systems engineer cold. A1 proved you can build the
> model; A2 proves you can make it run.

---

## 1. What CS336 actually requires (every deliverable)

CS336 A2 has six implementation themes: (1) a benchmarking/profiling harness, (2) activation
(gradient) checkpointing, (3) a FlashAttention-2 Triton kernel, (4) DDP naive → overlapped,
(5) optimizer-state sharding (ZeRO-1), (6) FSDP. Plus an analytical parallelism section
(DP / FSDP / TP communication-vs-compute algebra) and an 8B leaderboard. Test adapters live in the
official scaffold's `tests/adapters.py`; tests run with `uv run pytest`.

Priority key: **LOAD-BEARING** = master it cold (it carries the assignment *and* the interviews) ·
**COURSE-ROTE** = implement correctly to pass tests, extract the one insight, don't over-invest ·
**SKIP** = a capped leaderboard / GPU-dollar sink with no mastery carry.

| # | Deliverable (PDF problem name) | PDF § | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|---|
| 1 | `benchmarking_script` — init model, random batch, w warm-ups + n timed steps (fwd / fwd+bwd / +optim), `cuda.synchronize()` | 2.1.3 | (script; no adapter) | 0.5 d | COURSE-ROTE (the *script*); LOAD-BEARING (the `cuda.synchronize` + *predict-before-run* discipline) |
| 2 | `nsys_profile` — Nsight Systems profiling, 5 written Q&A (top kernel, non-matmul kernels, softmax-vs-matmul runtime-vs-FLOPs) | 2.1.4 | (analysis; no adapter) | 0.5 d | COURSE-ROTE (the *tables*); LOAD-BEARING (the *softmax-vs-matmul* intuition) |
| 3 | `mixed_precision_accumulation` — fp32 vs fp16 accumulation drift | 2.1.5 | (written) | 0.1 d | **LOAD-BEARING** (1 pt; the numerics lesson — fp16 accumulation loses small addends) |
| 4 | `benchmarking_mixed_precision` — autocast dtypes of params/activations/LN/logits/loss/grads; BF16 vs FP32 timings | 2.1.5 | (written + script) | 0.3 d | **LOAD-BEARING** (autocast dtype rules; LayerNorm-in-fp32) |
| 5 | `memory_profiling` — `torch.cuda.memory._record_memory_history`, memory_viz timelines; residual-activation accounting | 2.1.6 | (script + written) | 0.5 d | **LOAD-BEARING** (activation-memory math = the 100B one-pager input) |
| 6 | `gradient_checkpointing` — memory-optimal recursive checkpointing strategy + O(√N) tradeoff | 3.2 | (written + script) | 0.5 d | **LOAD-BEARING** (O(√N) peak-activation memory; the recompute lever) |
| 7 | `pytorch_attention` — benchmark vanilla SDPA across d∈{16,32,64,128} × seq∈{256…16384}; find OOM point; memory accounting | 4.1.1 | (script) | 0.3 d | LOAD-BEARING (the *why FA2 exists* roofline seed) |
| 8 | `torch_compile` — compiled vs eager attention + whole-model fwd/bwd/optim | 4.2 | (script) | 0.2 d | COURSE-ROTE (but anchors the "beat-the-compiler" framing) |
| 9 | `flash_forward` (a) **pure-PyTorch** FA2 fwd autograd.Function | 4.2.2 | `get_flashattention_autograd_function_pytorch` · `test_flash_forward_pass_pytorch` | 0.5 d | **LOAD-BEARING** (the tiled / online-softmax reference oracle) |
| 10 | `flash_forward` (b) **Triton** FA2 fwd kernel + autograd.Function (Algorithm 1), incl. causal masking flag (`is_causal: tl.constexpr`, −1e6 mask) | 4.2.2 | `get_flashattention_autograd_function_triton` · `test_flash_forward_pass_triton` (parametrized `is_causal`) | 1.5 d | **LOAD-BEARING** (the headline kernel) |
| 11 | `flash_backward` — FA2 backward via recomputation, PyTorch + `torch.compile` (Eqs 13–19) | 4.2.2 | `test_flash_backward_pytorch` | 0.5 d | **LOAD-BEARING** (the recomputation backward; the D-vector trick) |
| 12 | `flash_benchmarking` — `triton.testing.do_bench`, FA2 vs PyTorch fwd/bwd/e2e, seq 128…65536, bf16 + fp32 | 4.2.2 | (script) | 0.3 d | **LOAD-BEARING** (this *is* the roofline data) |
| — | OPTIONAL: Triton FA2 **backward** (Algorithm 2; pass P twice to skip atomics) | 4.2.3 | `test_flash_backward_triton` | (skip) | **SKIP** — do the `torch.compile` recomputation backward instead |
| 13 | `distributed_communication_single_node` — benchmark all-reduce, 1 MB…1 GB, 2/4/6 GPUs, gloo vs nccl | 5.1.1 | (script) | 0.3 d | COURSE-ROTE (but the comms-cost intuition is LOAD-BEARING) |
| 14 | `naive_ddp` — all-reduce each param grad after backward | 5.2 | `get_ddp` (opt: `ddp_on_after_backward`) · `test_DistributedDataParallel` | 0.3 d | LOAD-BEARING (the DP baseline) |
| 15 | `naive_ddp_benchmarking` — time/iter + % in comms, xl, 1×2 GPU | 5.2 | (script) | 0.2 d | COURSE-ROTE |
| 16 | `minimal_ddp_flat_benchmarking` — flatten grads into one all-reduce | 5.3.1 | (script) | 0.2 d | LOAD-BEARING (gradient bucketing — the real DDP trick) |
| 17 | `ddp_overlap_individual_parameters` — `register_post_accumulate_grad_hook`; overlap comm with backward | 5.3.2 | `get_ddp` container · `test_DistributedDataParallel` | 0.5 d | **LOAD-BEARING** (overlap = the actual production DDP design) |
| 18 | `ddp_overlap_individual_parameters_benchmarking` — time/iter + Nsight overlap proof | 5.3.2 | (script + screenshots) | 0.2 d | COURSE-ROTE |
| 19 | `optimizer_state_sharding` — wrap an Optimizer; each rank owns ~1/world_size of params; broadcast after step | 6 | `get_sharded_optimizer` · `test_sharded_optimizer` | 1.0 d | **LOAD-BEARING** (ZeRO-1; the memory math) |
| 20 | `optimizer_state_sharding_accounting` — peak-mem breakdown (params/grads/Adam) + ZeRO-1 ($P_{os}$) compare | 6 | (written) | 0.2 d | **LOAD-BEARING** (the 100B memory one-pager) |
| 21 | `fsdp` — wrap module; all-gather weights for fwd/bwd, reduce-scatter grads, fp32 master / bf16 compute | 7 | `get_fsdp`, `fsdp_on_after_backward`, `fsdp_gather_full_params` · `test_fsdp_correctness`, `test_fsdp_gradient_sync` | 1.5 d | **LOAD-BEARING** (full ZeRO-3 — a graded deliverable, *not* skip; see §5) |
| 22 | `fsdp_accounting` — expected peak-mem savings vs §6; Nsight all-gather-in-time check | 7 | (written + screenshots) | 0.2 d | COURSE-ROTE |
| 23 | `alternate_ring_all_reduce` — time of an alternate ring all-reduce | 8.1 | (written) | 0.1 d | COURSE-ROTE |
| 24 | `data_parallel_calcs`, `fsdp_calcs`, `tp_calcs`, `fsdp_tp_calcs` — comms-vs-compute bottleneck algebra | 8.2–8.5 | (written) | 0.5 d | **LOAD-BEARING** (the roofline-for-comms; *when does scaling stop*) |
| 25 | Leaderboard — fastest full training step, 8B, BF16, causal, B=2/seq=32768 | 9 | (passes basics tests) | (skip) | **SKIP** (capped, GPU-dollar sink) |

> **GPU-target honesty (state it explicitly — it is a senior signal).** §4.2 benchmarks and the §9
> leaderboard target large recent GPUs. On a rented **A100 / 4090** you run the *same code* at smaller
> sequence lengths and report your hardware honestly. **FA2 (not FA3/FA4) is the correct target on
> pre-Hopper hardware** — see §4. The roofline result already shipped on this repo is **53% of SDPA at
> seq 4k on a 4090** (an honest gap, kept as the artifact — see §6).

---

## 2. The equations/algorithms that matter (senior extraction)

1. **FlashAttention-2 tiling + online softmax (Algorithm 1).** Outer loop over query tiles $Q_i$
   ($B_q\times d$), inner loop over key tiles. Maintain a running max
   $m_i^{(j)}=\max(m_i^{(j-1)},\,\mathrm{rowmax}(S_i^{(j)}))$, unnormalized numerators
   $\tilde P_i^{(j)}=\exp(S_i^{(j)}-m_i^{(j)})$, a running denominator
   $l_i^{(j)}=\exp(m_i^{(j-1)}-m_i^{(j)})\,l_i^{(j-1)}+\mathrm{rowsum}(\tilde P_i^{(j)})$, and rescale
   the accumulator $O_i^{(j)}=\mathrm{diag}(\exp(m_i^{(j-1)}-m_i^{(j)}))\,O_i^{(j-1)}+\tilde P_i^{(j)}V^{(j)}$.
   Normalize **once** at the end ($O_i=\mathrm{diag}(l_i)^{-1}O_i$) and store $L_i=m_i+\log l_i$.
   *Why it matters:* HBM IO and peak memory no longer scale as seq_len² — the single change that
   unblocks long-context training and serving, and the canonical frontier-kernel interview question.
2. **FA2 backward via recomputation (Eqs 13–19).** Precompute $D=\mathrm{rowsum}(O\circ dO)$.
   Recompute $S=QK^\top/\sqrt d$ and $P=\exp(S-L)$ from the saved $Q,K,V,O,L$ (never store $P$). Then
   $dV=P^\top dO$, $dP=dO\,V^\top$, $dS_{ij}=P_{ij}(dP_{ij}-D_i)$, $dQ=dS\,K/\sqrt d$,
   $dK=dS^\top Q/\sqrt d$. *Why it matters:* the $D$-vector + $L$-vector trick is what lets you skip
   storing the seq_len² probability matrix in the backward — the whole point of FlashAttention's
   memory win, and the cleanest demonstration of recompute-vs-store.
3. **The roofline (arithmetic intensity, % of peak).** Attention does $\Theta(N^2 d)$ FLOPs against
   $\Theta(N^2)$ (vanilla) or $\Theta(Nd)$ (flash) HBM traffic; arithmetic intensity = FLOPs/bytes,
   compared against the GPU's ridge point (peak FLOP/s ÷ peak HBM bandwidth). Below the ridge you are
   **memory-bound**, above it **compute-bound**. *Why it matters:* it tells you *which* number to
   optimize and lets you state "FA2-Triton hit X% of SDPA peak at seq 4k, here's why" — a senior
   signal, and a well-analyzed negative (the shipped 53%) is itself a valued artifact.
4. **AdamW / optimizer-state memory accounting (~16–20 bytes/param).** Per param: 4 B fp32 master
   weight + 4 B grad + 4 B Adam $m$ + 4 B Adam $v$ = **16 B** (mixed precision adds a 2 B bf16 weight
   copy and possibly a 2 B bf16 grad → ~18–20 B). *Why it matters:* this is the "train a 100B model"
   math — 100B × 16 B = **1.6 TB** of state, far past one GPU, which is *why* DP + TP + PP and ZeRO
   exist. The crisp one-pager (deliverable #20) is the artifact.
5. **DDP gradient bucketing + overlap.** Naive DDP all-reduces each param tensor separately (one
   collective per tensor, all fired after backward completes). Improvements: (a) **flatten** all grads
   into one buffer → one all-reduce (amortizes per-call latency); (b) **overlap** — fire each grad's
   all-reduce from a `register_post_accumulate_grad_hook` the instant that grad is ready, so
   communication hides under the still-running backward. *Why it matters:* overlap is the difference
   between toy DDP and production DDP; comms hidden under compute ≈ free.
6. **ZeRO-1 optimizer-state sharding (and ring all-reduce cost).** Each rank holds optimizer state for
   only ~1/world_size of params, steps its own shard, then broadcasts the updated params back. Ring
   all-reduce (= reduce-scatter + all-gather) costs $2\frac{N-1}{N}\frac{S}{W}$ — communication volume
   is independent of N to first order. *Why it matters:* ZeRO-1 cuts the 16 B/param optimizer state by
   world_size at almost no extra comms vs DDP; ZeRO-2/3 (shard grads, then params) is FSDP. This is the
   memory-vs-comms tradeoff every distributed-training role probes.

> **Hardware caveat (state it explicitly — it is a senior signal).** FlashAttention-**3** is
> **Hopper-only** (warp specialization, TMA, FP8); FlashAttention-**4** moved to **CuTeDSL on
> Blackwell**. On A100 / 4090 you honestly target **FA2**. And `torch.compile` already emits a fused
> Triton attention kernel — so the premium is on *beating the compiler*, which you must say out loud
> when you report the roofline.

---

## 3. Map to `src/scratch_llm/`

The Systems layer lives in `src/scratch_llm/kernels/`, `src/scratch_llm/utils/`, and
`src/scratch_llm/rollout/`. The forward/kernel pieces are built and green; the distributed pieces are
the next build (all CPU-constructable via the gloo backend, then validated on rented GPUs for the
real benchmarks). Implement against the official scaffold's `tests/adapters.py` — do not copy
solutions (see the repo `CLAUDE.md` "own every line" rule).

| CS336 deliverable | → `src/scratch_llm/...` | Status / note |
|---|---|---|
| `flash_forward` (a) pure-PyTorch FA2 oracle | `kernels/flash_attention.py` (`flash_attention_forward`) | ✅ built — the online-softmax recurrence in plain PyTorch; CPU-testable; the **correctness oracle** the Triton kernel is checked against. Spec: [`../design/L2_flash_attention_SPEC.md`](../design/L2_flash_attention_SPEC.md). |
| `flash_forward` (b) Triton FA2 fwd + causal | `kernels/flash_attention_triton.py` | ✅ built — autotuned Triton forward (causal flag), validated on a 4090 vs the oracle and SDPA. GPU-only; not imported on CPU/CI. |
| `flash_backward` (recomputation) | `kernels/flash_attention.py` returns `(O, L)`; backward via `torch.compile` recomputation | ✅ forward + L done; the recomputation backward (Eqs 13–19, the D-vector) is the remaining kernel piece — the `torch.compile` path, **not** a hand-rolled Triton backward (that is the SKIP, deliverable “OPTIONAL Alg. 2”). |
| `pytorch_attention` / `flash_benchmarking` (the roofline) | the roofline benchmark + plot (a `bench/` artifact + the spec) | ✅ shipped — **53% of SDPA at seq 4k on a 4090**, the honest gap kept as the artifact. This is the "% of SDPA" interview number. |
| `benchmarking_script` + the `cuda.synchronize` discipline | the timing harness (reused to benchmark KV-cache decode + rollout throughput) | the warm-up + sync + **predict-before-run** harness is the reusable piece; the KV-cache decode path it times lives in `model.py` / `rollout/`. |
| `memory_profiling`, KV/activation memory math | KV-cache (`model.py` `KVCache` + cache-aware attention) + the 100B one-pager | ✅ KV-cache built (cached == recompute for MHA/GQA/batch). Spec: [`../design/L2_kv_cache_SPEC.md`](../design/L2_kv_cache_SPEC.md). The activation/KV memory algebra is the same math that sizes a serving engine's KV cache. |
| `mixed_precision_accumulation` + `benchmarking_mixed_precision` (autocast dtypes; LayerNorm-in-fp32; fp16 accumulation drift) | the numerics lesson; observable in `utils/monitors.py` | the train engine (bf16 autocast, fp32 master) and a serving engine (fused kernels, possibly fp8/int8 KV) compute **different logits**. This **train-engine vs inference-engine logit drift** is a real, measurable systems phenomenon — `utils/monitors.py` exposes it as a neutral diagnostic (`kl_train_infer`, with a HALT threshold); the mixed-precision lesson is the mechanistic *why* (see note below). Spec: [`../design/L2_kl_train_infer_SPEC.md`](../design/L2_kl_train_infer_SPEC.md). |
| `naive_ddp` → `minimal_ddp_flat` → `ddp_overlap_individual_parameters` | `utils/` (DDP container — **to build**) | ⬜ CPU/gloo-buildable now; the post-accumulate-grad-hook overlap is the production design. |
| `optimizer_state_sharding` (ZeRO-1) + accounting | `utils/` (sharded optimizer — **to build**) + the memory one-pager | ⬜ CPU/gloo-buildable now; the accounting is the 100B answer. |
| `fsdp` + `fsdp_accounting` | `utils/` (FSDP container — **to build**) | ⬜ CPU/gloo-buildable now; a **full graded deliverable** (all-gather weights / reduce-scatter grads / fp32-master-bf16-compute). |
| `data_parallel_calcs` / `fsdp_calcs` / `tp_calcs` / `fsdp_tp_calcs` | written analysis (the comms-vs-compute algebra) | the "when does scaling become comms-bound" math; pairs with the memory one-pager as the distributed-training interview answer. |

> **On the train-vs-infer logit-drift lesson (neutral framing).** `utils/monitors.py` measures
> `kl_train_infer = KL(train‖infer)`, the KL between the training engine's next-token distribution and
> the serving engine's. A2 is where you learn **why that gap exists**: different kernels (your Triton
> FA2 vs the engine's fused attention), different precision (bf16 train vs fp8/int8-KV serve), and
> different reduction order (the fp16-accumulation lesson in `mixed_precision_accumulation`). The
> measured result on this repo is instructive: an HF engine pair on Qwen2.5-0.5B showed
> **eager/fp32-train is ~5.7× closer to sdpa/bf16-serve than eager/bf16-train is** — because
> `F.scaled_dot_product_attention` accumulates the softmax/PV in fp32 even with bf16 tensors, so the
> drift driver is *accumulation precision + kernel choice, not the storage dtype* (a prediction that
> was falsified, and the falsification is the finding — see the spec). Treat this as a concrete
> systems lesson about mixed precision and kernel parity, not as a project thesis.

> **Where to look next.** The build order, definition-of-done, and CPU-vs-GPU resourcing for the
> distributed pieces are in [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) (A2 section). The
> spec + test oracle is the official scaffold at `../../../lectures/assignment2-systems/` (the PDF
> defines each deliverable; `tests/adapters.py` + `tests/` verify correctness).

---

## 4. The frontier 2026 lens

**Commoditized (do not over-invest):**
- `torch.compile` emits a fused Triton attention kernel automatically — a *baseline* fused kernel is
  free. Writing one to match the compiler proves competence but isn't a moat.
- Naive all-reduce DDP and "wrap-it-in-DDP" are table stakes; every framework ships them.
- Exhaustive Nsight tables and per-kernel timing dumps are mechanical reporting, not insight.

**Scarce (where the value and the comp are):**
- **Beating the compiler / hand-tuned kernels** — the premium is the *delta over `torch.compile`*,
  stated with a roofline (arithmetic intensity, % of peak, BW- vs compute-bound).
- **EP all-to-all overlap** and MoE-aware comms — every frontier flagship is sparse; hiding
  expert-routing communication under compute is hard and hot.
- **Inference as the frontier P&L** — serving is where the money and the wall-clock live; SGLang /
  vLLM internals, speculative decoding (EAGLE / Medusa), and **the KV-cache as the #1 cost lever**
  (the dominant share of wall-clock past ~1M context) are the scarce skills. This repo's KV-cache and
  rollout seam are the on-ramp.
- **Train-vs-infer kernel/precision parity** — knowing *why* two engines running the same weights
  produce different logits (kernel choice, accumulation precision, quantized KV) and being able to
  measure it is a real systems skill the `kl_train_infer` diagnostic exercises.

**2026 signals to cite:** FA3 (Hopper, warp-specialized, FP8) → FA4 (CuTeDSL / Blackwell);
DeepSeek-V3 sustaining high TFLOP/s/GPU and MLA's KV-cache reduction; SGLang / vLLM + speculative
decoding + paged KV-cache + low-bit KV (e.g. KIVI 2-bit). The takeaway: past a point, **systems
work — not architecture — is what moves the frontier's cost and latency.**

---

## 5. Prioritization verdict — what matters / what to skip

**The ~20% that is load-bearing (do these to whiteboard depth):**
1. **FA2 Triton kernel (fwd `flash_forward`, bwd `flash_backward`) + the roofline (`flash_benchmarking`).**
   The headline systems artifact and the canonical interview kernel. The roofline + an honest "% of
   SDPA" is shippable even when the kernel is slower than SDPA (the 53% result is the model here).
2. **DDP overlap + ZeRO-1 + FSDP + the 100B memory math.** `ddp_overlap_individual_parameters` (via
   `register_post_accumulate_grad_hook`), `optimizer_state_sharding` (ZeRO-1) +
   `optimizer_state_sharding_accounting`, and **`fsdp`** (full ZeRO-3 — all-gather weights /
   reduce-scatter grads / fp32-master-bf16-compute). FSDP is a **graded deliverable, not a skip and
   not time-boxed.** The accounting + the comms algebra together are the "how would you train a 100B
   model?" answer.
3. **Gradient checkpointing (O(√N)) + mixed precision.** The recompute-vs-store memory lever and the
   autocast dtype rules (matmuls → bf16, LayerNorm/reductions → fp32, master weights → fp32), plus the
   fp16-accumulation lesson — the numerics that make large-batch training fit and stay stable.
4. **The parallelism comms algebra** (`data_parallel_calcs` / `fsdp_calcs` / `tp_calcs` /
   `fsdp_tp_calcs`): the analytical "when does scaling become comms-bound."

**Course-rote (do them, but don't gold-plate — minimum to pass + the one insight):**
- `nsys_profile` written tables, `naive_ddp_benchmarking`, `ddp_overlap…_benchmarking`,
  `fsdp_accounting`, `distributed_communication_single_node`, `alternate_ring_all_reduce`,
  `torch_compile`. Extract the *one* intuition from each (softmax-vs-matmul runtime; comms-cost
  scaling; the compiler already fuses) and move on. Keep the `benchmarking_script`
  `cuda.synchronize` + predict-before-run discipline as **load-bearing** — that habit is the point,
  not the tables.

**Explicit SKIP (GPU-dollar sinks with no mastery carry):**
- The **8B leaderboard** (§9) — capped reward, large GPU spend.
- The **OPTIONAL Triton FA2 backward** (§4.2.3, Algorithm 2 — passing P twice to avoid atomics,
  `test_flash_backward_triton`) — do the **`torch.compile` recomputation backward** instead; same
  learning (recompute-vs-store, the D-vector), far less kernel-debugging time.
- **Implementing exotic parallelism** (TP / PP / 2D mesh) beyond the analytical `tp_calcs` /
  `fsdp_tp_calcs` algebra — do the *math* (when scaling becomes comms-bound), skip the
  implementation. A **toy TP** only if there is slack after FSDP is green.

---

## 6. Build checklist (ordered, with discipline gates)

> Discipline gates from the repo `CLAUDE.md`: **predict-before-you-run** (write the falsifiable number
> first — it is your debugging anchor), **fixed-seed reproducibility**, **always
> `torch.cuda.synchronize()` around timed regions** (CUDA is async), and validate each distributed
> piece with a **2-rank gloo equivalence test run ×5** (single-process vs distributed must match).

**Day 1 — harness + numerics (predict first):**
1. [ ] `benchmarking_script`: init model, random batch, w warm-ups, n timed steps;
   **`torch.cuda.synchronize()` before/after the timed region**. Toggle fwd / fwd+bwd / +optim via CLI
   flags.
2. [ ] **Predict-before-run:** write down the expected fwd-pass ms for `small` at seq 512 *before*
   running; the gap is your debugging anchor.
3. [ ] `mixed_precision_accumulation` + `benchmarking_mixed_precision`: nail the autocast dtype table
   (matmuls → bf16, LayerNorm/reductions → fp32, master weights → fp32) and the fp16-accumulation
   drift result.

**Day 2 — memory levers:**
4. [ ] `memory_profiling`: dump `memory_snapshot.pickle`, read the active-memory timeline; compute the
   residual-activation MiB per `TransformerBlock` from first principles and check it matches.
5. [ ] `gradient_checkpointing`: implement the recursive checkpointing strategy; state the **O(√N)
   peak vs O(N) recompute** tradeoff and validate the measured peak against the next-smaller/larger
   block size.

**Day 3 — the kernel (the headline; largely ✅ on this repo):**
6. [ ] `flash_forward` (a) pure-PyTorch tiled FA2 (online softmax, save $L,Q,K,V,O$) → pass
   `test_flash_forward_pass_pytorch`. This is the **correctness oracle** for the Triton port. *(built)*
7. [ ] `flash_forward` (b) Triton kernel per **Algorithm 1**, launch grid $(T_q,\ \text{batch·head})$,
   single key-loop, advance block pointers at loop end, fp32 accumulators, `is_causal` flag (−1e6
   additive mask) → pass `test_flash_forward_pass_triton`. *(built; autotuned)*
8. [ ] `flash_backward` via `torch.compile` recomputation (Eqs 13–19, compute $D$) → pass
   `test_flash_backward_pytorch`. *(remaining kernel piece)*
9. [ ] **Predict-before-run + roofline:** predict "% of SDPA at seq 4k" *first*; then
   `flash_benchmarking` (`triton.testing.do_bench`) and **plot the roofline** (arithmetic intensity vs
   % peak, mark BW- vs compute-bound). *(shipped: 53% of SDPA @ seq 4k on a 4090 — the honest gap is
   the artifact, per Neel Nanda's "a well-analyzed negative is valuable.")*

**Day 4 — distributed (the next build; CPU/gloo first):**
10. [ ] `naive_ddp` → `minimal_ddp_flat_benchmarking` (flatten grads) →
    `ddp_overlap_individual_parameters` (post-accumulate-grad hook) → pass `test_DistributedDataParallel`
    (run ×5).
11. [ ] `optimizer_state_sharding` (ZeRO-1) → pass `test_sharded_optimizer` (×5); write the
    **100B memory one-pager** (`optimizer_state_sharding_accounting`: 16–20 B/param → 1.6 TB → why
    DP + TP + PP).
12. [ ] `fsdp` (all-gather weights for fwd/bwd, reduce-scatter grads, fp32 master / bf16 compute) →
    pass `test_fsdp_correctness` and `test_fsdp_gradient_sync` (×5); `fsdp_accounting` written +
    Nsight all-gather-in-time check. **This is a full graded deliverable — do not defer it.**
13. [ ] The comms algebra: `data_parallel_calcs` / `fsdp_calcs` / `tp_calcs` / `fsdp_tp_calcs`
    (written) + `alternate_ring_all_reduce`.

> The DDP / ZeRO-1 / FSDP containers are CPU-constructable via the gloo backend (build + correctness
> there), then re-run the *benchmarks* on rented multi-GPU hardware (use the `vastai` skill — a step
> that needs a GPU is resourced, never dropped).

---

## 7. Open questions / ADR triggers

1. **FA2-only on the rented GPU.** FA3 is Hopper-gated, FA4 is Blackwell/CuTeDSL. Decision: target FA2
   on A100/4090 and state the hardware honestly. **ADR if** the rented GPU is actually Hopper (then
   FA3 is fair game and the roofline target shifts).
2. **SGLang vs vLLM for serving** (`rollout/sglang_client.py`). The repo names SGLang; vLLM has wider
   adoption and paged-attention maturity. **ADR trigger:** when a specific KV-cache feature or
   multi-turn tool-call rollout forces the choice — log each engine's logprob/precision contract,
   since that determines the baseline train-vs-infer logit drift. (SGLang needs sm90+ — see
   [`../adr/ADR-0008-sglang-hopper-only-on-ada.md`](../adr/ADR-0008-sglang-hopper-only-on-ada.md).)
3. **The Triton FA2 backward.** Decision: ship the `torch.compile` recomputation backward; the
   hand-rolled Triton backward (Algorithm 2) is SKIP. **ADR if** a real serving-path latency need
   makes the fused backward worth the kernel-debugging time.
4. **FSDP wrap granularity.** Per-parameter vs per-block sharding units, and where the all-gather /
   reduce-scatter boundaries fall. **ADR trigger:** if the gloo correctness test passes but the
   GPU benchmark shows all-gather *not* overlapping compute, the wrap granularity is the lever to log.
5. **Serving-engine precision contract.** A real serving engine may silently quantize the KV-cache
   (fp8/int8), so the train-vs-infer logit drift can be nonzero before any change to the model.
   **ADR:** record the serving engine's precision contract as a fixed input whenever you report the
   drift number.
