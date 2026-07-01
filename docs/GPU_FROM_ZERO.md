# GPU & Kernel Engineering — from zero → frontier decode kernels

> **What this is.** A laddered curriculum for someone with **no GPU-programming or kernel-optimization
> background** who is leveling to senior/principal frontier-lab performance engineering. It is the
> *on-ramp* the rest of the repo assumes you already have. Rungs 0 → 9 take you from "what is a warp"
> to the repo's frontier targets (FlashAttention, decode-attention, FP8/FP4 numerics, distributed
> comms, RL-systems), each rung mapped to the A2/A5/DELTA build and to the 2026 frontier practice.
>
> **The organizing law you will internalize:** *decode is memory-bandwidth-bound; every 2026 inference
> technique is one lever to raise decode arithmetic intensity back toward the bandwidth ceiling.* The
> spine that ties these rungs to the assignments is [`PERFORMANCE_TRACK.md`](PERFORMANCE_TRACK.md).

---

## How to climb a rung (the forced-mastery protocol)

This is the repo's learning mode (`CLAUDE.md` "How we build"), bound to GPU work. **Every rung runs
the same six steps — and the division of labor is fixed by the Mode boundary:**

| Step | What happens | Who does it |
|---|---|---|
| 1. **First principles** | Derive the mechanism — the problem, the math, why this design. | **You derive.** AI asks Socratic questions, never hands you the derivation. |
| 2. **Visualize (3 lenses)** | (a) tensor shapes through the op + exact `src/…` lines; (b) the system/data-flow (ASCII); (c) a tiny hand-traced numeric example. | **AI visualizes + explains.** This is the agent's primary job here. |
| 3. **Predict-before-run** | Write the falsifiable number/shape **first** (the bound, the % of peak, the bytes). | **You predict.** It is your debugging + learning anchor. |
| 4. **Build test-first** | Write the invariant as a test, then make it pass. | **AI scaffolds the test + bench harness; YOU write the kernel/derivation body.** |
| 5. **Teach-back — the gate** | Explain it back in your own words + modify-and-predict one variation. **No advancing until you can.** | **You teach. AI quizzes.** |
| 6. **Connect to frontier** | Tie it to a dated 2026 paper / the interview question. | AI provides the citation tree. |

> ### The Mode-3 boundary (read this once, it governs the whole ladder)
> The **kernel bodies and the RL-loss/math derivations are YOURS to write** — that is the reps the
> interview tests, and an agent writing them for you defeats the entire point. The repo enforces this
> with a hard hook (`.claude/hooks/kernel-write-guard.sh`) that **blocks** an agent's Edit/Write to
> kernel files. **The AI's job is everything *around* the meat:** explain + visualize the mechanism,
> scaffold the failing test + the bench, profile/diagnose the result, review it after (`/kreview`),
> and tutor Socratically (`/tutor`, `/master`). When you are stuck, ask for a *failing test*, a
> *visualization*, or a *Socratic hint* — never "write the kernel."

**The Definition of Done is a profile, not a green test.** A green correctness test is the *floor*.
The *result* is a measured profile that lands near a roofline you predicted **before** running (FOP-3).
"Implemented" ≠ "measured" (FOP-4) — only a profiled run is a result.

**Resourcing — we develop on a standing GPU** (RTX PRO 4000 Blackwell, sm120, 25 GB). Rungs 0–1, 5
(oracle), 7 (CPU codec) need no GPU at all (mental-model / pure-torch); the kernel rungs (2–6) run on
the box **now** — write, run, **measure + profile here** (`CLAUDE.md` "Develop on the GPU"). Get the
oracle green first (the floor), then the roofline (the result). Rails: **25 GB cap** + **"% of *this*
Blackwell," not datacenter**. Rent a bigger / multi-GPU box (`vastai`) only for what this card can't do
(full-scale throughput, multi-GPU NCCL); never silently drop a step. **Empirical (sm120): Triton ✅
(FA2-fwd).** *(The Jun-29 perf CUDA suite incl. `rmsnorm.cu` was reset 2026-07-01 for the from-scratch
A1–A7 rebuild; preserved at tag `pre-perf-kernel-reset` — see ADR-0011 + `bench/RESULTS.md`.)*

**Framework policy ([ADR-0011](adr/ADR-0011-kernel-framework-policy.md)).** Write every ladder rung +
the DELTA kernel in **Triton** (primary); do R4's GEMM in **CUDA C++** too (the compute-bound feel) and
treat CUDA/CUTLASS/CuTe as the read-and-contribute tier for FlashInfer/FA. The rule for any new kernel:
**memory-bound → Triton; compute-bound chasing tensor-core peak → CUDA/CUTLASS.** CuTeDSL/Pallas = know-it.

---

## The ladder at a glance

```
 RUNG                              YOU DERIVE / BUILD          THE PROFILE (DoD)            MAPS TO
 ─────────────────────────────────────────────────────────────────────────────────────────────────
 0  GPU execution & memory model   (mental model, no code)     bytes/FLOP of H100           foundation
 1  The roofline model             bench/roofline.py counters  a roofline PNG               PERF spine
 2  First kernel: vector-add        a Triton elementwise kernel % of HBM bandwidth           A2.1 warmup
 3  Reductions + a fused kernel     fused RMSNorm / softmax     near-HBM-roof, ~2× unfused   A2 / decode
 4  Tiled matmul                    a tiled GEMM                % of peak TFLOP/s            A2.1
 5  Online softmax → FlashAttn-2    the FA2 fwd recurrence      % of SDPA (the 53% negative) A2.1 ✅partial
 6  The DECODE-attention kernel     split-K decode + paged KV   % of HBM BW (not FLOPs)      DELTA axis
 7  Quantization numerics           quant-error harness + FP8   error-vs-bits; accum gap     A2 / DELTA
 8  Distributed comms               DDP→ZeRO-1→FSDP + comms-RL  comms-overlap timeline       A2 finish
 9  RL-systems instrumentation      GRPO + train↔infer drift    Δlogp histogram; collapse    A5
```

Climb **in order** — each rung's mental model is load-bearing for the next. You may *ship* out of order
(the repo's SHIP order is EV-ranked), but you *learn* the ladder bottom-up.

---

## Rung 0 — The GPU execution & memory model *(mental model; no code)*

**Derive (you):** Why is a GPU a *throughput* machine, not a latency machine? Work out the hierarchy
from first principles: thousands of threads grouped into **warps of 32** (SIMT — one instruction, 32
lanes), warps grouped into **blocks**, blocks scheduled onto **SMs** (streaming multiprocessors, ~132
on an H100). Then the memory pyramid and its two numbers that decide everything — **capacity** and
**bandwidth**:

```
        registers   (~256 KB/SM, ~1-cycle)        ◄ fastest, tiniest
        SMEM/L1     (~228 KB/SM, ~20-cycle)        ◄ "shared memory" — you manage it by hand
        L2 cache    (~50 MB,    ~200-cycle)
        HBM/global  (80 GB, 3.35 TB/s, ~400-cycle) ◄ the cliff: huge but "far"
```

**Predict (you):** H100 does ~989 BF16 TFLOP/s of matmul but HBM moves only 3.35 TB/s. Compute the
**FLOP-per-byte the hardware can sustain** = 989e12 / 3.35e12 ≈ **295**. Write that number down — it is
the *ridge point* and it governs Rung 1.

**Visualize (AI):** the pyramid with sizes/bandwidths; a warp executing one `FMA` across 32 lanes; an
SM with its register file + SMEM.

**Teach-back gate:** *Why can a kernel that moves a lot of HBM bytes per FLOP never be sped up by a
faster matmul unit?* (Answer in terms of the 295 number.) *What is a warp, and what is "occupancy"?*

**Frontier connection:** This is the substrate for the roofline. The constraints coding agents miss —
**launch overhead, occupancy, bank conflicts, latch/mbarrier** — all live in this model (FOP-3).

---

## Rung 1 — The roofline model *(the organizing law; build the harness)*

**Derive (you):** Arithmetic Intensity `AI = useful_FLOPs / bytes_moved`. Attainable performance =
`min(peak_compute, AI × peak_bandwidth)`. The **ridge point** = `peak_compute / peak_bandwidth` (the
295 from Rung 0) is the minimum AI to escape the memory roof. Three regimes (Horace He): **compute-bound**
(big GEMMs → need a better MMA schedule), **memory-bandwidth-bound** (elementwise/norm/decode → *fuse*),
**overhead-bound** (tiny kernels / eager Python → CUDA graphs / compile).

**Build test-first (you write the FLOP/byte counting; AI scaffolds the plot + GPU table):**
`bench/roofline.py` — given `(dims, dtype, batch, seqlen, GPU)` → AI, predicted bound, predicted
time; overlay measured `ncu` numbers. **Spec:** [`design/PERF_roofline_harness_SPEC.md`](design/PERF_roofline_harness_SPEC.md).
This harness is the **seed every later rung reuses.**

**Predict (you):** Place three ops on the roofline *before* running: a 4096×4096 GEMM (compute-bound),
RMSNorm (memory-bound), batch-1 decode attention (`AI ≈ 1`, ~300× below ridge).

**Honesty constants to bake in** (the research flagged these — getting them wrong is a silent
credibility tell):
- Use the **dense ~295 FLOP/byte** H100 ridge, **not** the sparse ~590. Always state which peak.
- `FP4 = 2× FP8` throughput is the durable invariant; B200 ≈ 9 PF dense FP4, GB200 ≈ 10 dense / 20 sparse.
- Decode tok/s lower bound ≈ `(weights + KV bytes) / HBM_BW`.

**Teach-back gate:** *Given an op's FLOPs and bytes, predict its bound and its runtime, then name the
fix for that regime.* **This is the single highest-value skill in the ladder** — all five research
streams ranked it #1.

**Frontier connection:** every kernel DoD below is "land near the roof you predicted." This *is* FOP-3.

---

## Rung 2 — First kernel: vector-add / SAXPY *(the launch model)*

**Derive (you):** the launch grid — `grid`, `block`, thread indexing, bounds masking. Why is `z = x + y`
memory-bound? Count it: 3 arrays × 4 bytes moved per 1 add → `AI ≈ 1/12` → deep in the memory roof.

**Build test-first:** correctness vs `torch` (the floor), then bench vs the HBM bound. **You write the
Triton kernel body; AI scaffolds `test_` + the `do_bench` harness.**

**Predict (you):** the kernel should hit ~80–90% of peak HBM bandwidth (it's a pure copy-ish op).

**Profile (DoD):** `ncu` Speed-of-Light showing **memory-bound**, % of HBM peak. Not % FLOPs.

**Teach-back gate:** *Why does adding more compute (e.g. `z = x*y + y`) not change the runtime here?*

**Frontier connection:** this "hello world" teaches launch/occupancy — the cheapest place to first see
the launch-overhead and grid-sizing constraints (Rung 0) bite.

---

## Rung 3 — Reductions + the fused memory-bound kernel *(the fusion lever)*

**Derive (you):** a parallel reduction (sum/max) within a block — the tree, **shared memory**, warp
shuffles, and **bank conflicts** (32 banks × 4 bytes; the XOR-swizzle fix). Then RMSNorm or softmax: count
the HBM traffic of the **unfused** path (read x, write mean, read x, write y ≈ 4 passes) vs the **fused**
path (read x, write y ≈ 2 passes).

**Build test-first:** a fused RMSNorm (or softmax) kernel. **You write the body.**

**Predict (you):** fusion ≈ **2×** from halving byte traffic; the result lands near the HBM roof.

**Profile (DoD):** before/after `ncu` — eager = overhead/bandwidth-bound, fused = near HBM roof; report
the byte-traffic reduction.

**Teach-back gate:** *Why is fusion the canonical fix for the memory-bound regime, and what is its ceiling?*

**Frontier connection:** fusion is THE memory-bound lever; this is the mechanism behind every "fused
LayerNorm/softmax" in production. Pairs with `utils/mixed_precision.py` (the fp32-accumulation rule).

---

## Rung 4 — Tiled matmul *(the compute-bound archetype + tensor cores)*

**Derive (you):** shared-memory tiling + register blocking; why a tile of size `T` raises AI ∝ `T`
(each loaded element is reused `T` times) so a big-enough tile crosses the ridge into compute-bound. The
**occupancy ↔ register-pressure** tradeoff (more registers/thread → fewer resident warps).

**Build test-first:** a tiled GEMM vs `torch.matmul`. **You write the body.**

**Predict (you):** % of peak TFLOP/s your tiles achieve (a clean first pass lands well under cuBLAS —
that gap is the lesson, like the FA2 negative).

**Profile (DoD):** `ncu` SM-throughput; identify whether you're tensor-core-bound or memory-bound.

**Teach-back gate:** *Why does a bigger tile move you up the roofline — and what caps the tile size?*

**Frontier connection:** the next step is **WGMMA + TMA** (Hopper async tensor cores) — *awareness now*,
a Rung-7+ stretch. Skipping WGMMA caps an H100 GEMM at ~63% of peak; with it, ≥94%. This is where
"tensor cores" stop being a word and become a schedule.

---

## Rung 5 — Online softmax → FlashAttention-2 forward *(partially built ✅)*

**Derive (you):** the **online-softmax recurrence** — running `(m, ℓ, acc)` updated per key-tile so the
`N×N` score matrix is never materialized (memory `Θ(N²) → Θ(Nd)`). The repo already has the
CPU oracle (`kernels/flash_attention.py`) and the Triton fwd — but for a from-zero learner, **re-derive
the recurrence by hand** (the exact update is in [`design/L2_flash_attention_SPEC.md`](design/L2_flash_attention_SPEC.md) §2).

**Build test-first:** the Triton FA2 fwd (against the oracle + SDPA). **You write the body.**

**Predict (you):** % of SDPA. The repo's honest result: **predicted ≈65%, measured 53%** at seq 4k on
a 4090 — *below the 60% kill line, shipped as a documented negative.* Study that negative: it's a
model of FOP-4 claims-honesty.

**Profile (DoD):** the roofline (AI vs % peak), the BW- vs compute-bound boundary.

**Teach-back gate:** *Walk the (m, ℓ, acc) update for two key-tiles by hand. Why is the rescale `exp(m−m_new)`
needed?*

**Frontier connection:** FA3 (Hopper async/warp-specialization/FP8 → 740 TFLOP/s ≈ 75% peak) → FA4
(Blackwell, CuTeDSL). You target FA2; you can *discuss* FA3/FA4.

---

## Rung 6 — The decode-attention kernel *(the memory-wall climax — the DELTA axis)*

**Derive (you):** why decode ≠ prefill. In autoregressive decode the **query length is 1**, so the
batch×query parallelism FA relies on collapses → batch-1 uses **<1% of the GPU**, `AI ≈ 1`, hard
memory-bound. The fixes: **split-K / Flash-Decoding** (chunk the KV sequence, run partial attention per
split with an extra LSE scalar, recombine via log-sum-exp), **GQA** (raises AI ∝ group size at equal
FLOPs), **paged-KV gather** (block table → non-contiguous KV).

**Build test-first:** ① **first, *time* the existing KV-cache decode** — it's correctness-tested but
never timed (spec: [`design/PERF_decode_roofline_SPEC.md`](design/PERF_decode_roofline_SPEC.md)); ② then
a split-K decode kernel + a paged-KV block table. **You write the bodies.**

**Predict (you):** tok/s ≈ `(weights + KV)/HBM_BW`; for the kernel, **% of HBM bandwidth (target
>80–90%, cite the ~93% GLA bar)** — *not* % FLOPs.

**Profile (DoD):** `ncu` showing memory-bound, tensor cores idle, HBM pipe saturated; decode latency
flat vs context length to 16–32k.

**Teach-back gate:** *Why is decode graded on % HBM bandwidth, not % FLOPs? How does batching raise the
decode AI, and why doesn't KV amortize across the batch?*

**Frontier connection:** FlashInfer (−29–69% ITL), PagedAttention (waste 60–80%→<4%), and **linear-attention
state** (Gated DeltaNet / Mamba-2: a *constant-size* recurrent state replaces the growing KV cache →
decode flat in context). **This is exactly the DELTA capstone** (GDN-2 decode kernel + NVFP4-on-state).

---

## Rung 7 — Quantization numerics *(the precision lever)*

**Derive (you):** FP formats (E4M3/E5M2 for FP8, E2M1 for FP4); **block/microscaling** (MXFP4 block-32
+ E8M0 scale vs NVFP4 block-16 + fractional E4M3 inner + per-tensor FP32 outer); the **accumulation-precision
trap** — Hopper FP8 tensor cores accumulate in ~13–14 mantissa bits, so DeepSeek-V3 **promotes partials
to FP32 every 128 accumulations**. The Key/Value KV asymmetry (Keys per-channel, Values per-token).

**Build test-first:** ① a **quant-error-vs-bits harness** (CPU, pure numpy — sweep granularity × format,
reproduce "NVFP4 < MXFP4" error direction); ② a blockwise **FP8 GEMM** with FP32-promotion (the *body*
is yours/Mode-3; AI scaffolds the test that exposes the naive-vs-promoted accumulation gap).

**Predict (you):** the error ordering across {per-tensor, per-channel, g128, MX-32, NVFP4-16}; the FP8
kernel's ~2× peak vs the realized ~1.3× (and explain the gap = quant/transpose overhead).

**Profile (DoD):** an error-vs-effective-bits plot; the accumulation-gap test; (GPU) the FP8 roofline.

**Teach-back gate:** *Why does weight-only INT4 speed decode but do nothing for compute-bound prefill?
Why is per-tensor activation quant wrong for LLMs?*

**Frontier connection:** DeepSeek-V3 FP8 training (loss-rel-error <0.25%), NVFP4 pretraining (12B/10T,
<1% gap), gpt-oss MXFP4. **The DELTA spike's open seam = NVFP4 on the GDN recurrent state vs context length.**

---

## Rung 8 — Distributed comms *(the systems layer — the A2 finish)*

**Derive (you):** the collectives (all-reduce / all-gather / reduce-scatter / all-to-all) and their
costs; the **comms roofline** (NVLink ~900 GB/s vs IB ~50 GB/s/NIC — an ~18:1 raw gap); DDP comm volume
= `2N`; the **16-bytes/param** Adam memory math (→ ZeRO sharding); compute/comm **overlap**.

**Build test-first (all CPU/gloo-buildable):** naive→flat-bucket→**overlapped DDP** (✅ in repo as `utils/ddp.py`),
then **ZeRO-1** (`utils/zero.py`, the stated D2/next), then **FSDP2** per-param, then a **comms-roofline
harness** (reuses Rung 1). The 2-rank gloo equivalence test is the correctness floor. (These are systems,
not Mode-3 kernels — AI can pair more freely; the *understanding* is still yours via teach-back.)

**Predict (you):** the overlap speedup from `2N / NVLink_BW`; the ZeRO-1 4× optimizer-memory reduction.

**Profile (DoD):** a **comms-overlap timeline** (torch profiler / Chrome trace) showing all-reduce hidden
behind backward; a per-GPU memory curve hitting the 4× cut.

**Teach-back gate:** *Order TP / EP / PP / DP by comms cost and explain why frontier configs nest them
that way. Write the 671B memory math.*

**Frontier connection:** FSDP2 (≈ZeRO-3), **EP all-to-all** for MoE (DeepEP), the 100B/671B one-pager
(the verbatim "how would you train a 100B model?" answer). Stretch differentiator: an MoE all-to-all
dispatch/combine + a load-imbalance plot.

---

## Rung 9 — RL-systems instrumentation *(A5 — the scarcest 2026 cluster)*

**Derive (you):** the two-engine split (a fast inference engine generates rollouts; the trainer computes
gradients) and why **rollout generation is ~66–91% of step time**. Then the hot 2025–26 result: the
**train↔infer logprob mismatch** — the inference and training engines return *different logprobs for the
same tokens at identical weights* (batch-invariance / reduction-order, not just FP non-associativity),
so "on-policy" GRPO is **secretly off-policy with bias** → silent reward collapse. The fix: **truncated
importance sampling** (ρ = π_θ/π_rollout, truncated) — and the honest nuance that this is **contested**
(IS-essential camp vs clipping-matters-more camp).

**Build test-first:** ① the GRPO/Dr.GRPO loop (**the loss body is yours/Mode-3**; AI scaffolds tests +
the dashboard); ② the **train↔infer drift measurement** harness (`utils/monitors.py` already has the
three KLs + IS/ESS) — Δlogp histogram, argmax-flip rate, KL(rollout‖train); ③ TIS + the on/off ablation.

**Predict (you):** the drift magnitude (mean/P99 |Δlogp|); that reward collapses without TIS and holds
with it (your miniature of the 0.574→0.255 collapse).

**Profile (DoD):** the Δlogp histogram + heavy tail; the reward-collapse-without-TIS curve; the mandatory
dashboard (entropy · KL(cur‖ref) **and** KL(cur‖old) separately · IS-ratio · reward · **length**).

**Teach-back gate:** *Why does a small average logprob gap silently break GRPO? What does ESS→0 tell you?
Argue both sides of the IS-vs-clipping debate.*

**Frontier connection:** DeepSeek-R1, Dr.GRPO (drop the length/std norm → stops verbosity hacking),
DAPO clip-higher (entropy control), the `audit-rl` skill, RLVR graders. Use `/audit-rl` to gate any run.

---

## What you will be able to defend cold (the interview surface)

By the top of the ladder you can whiteboard, from first principles: the roofline + ridge point; why
decode is memory-bound and every serving technique is an AI-raising lever; the online-softmax recurrence;
split-K decode + paged KV; FP8 block-scaling + the accumulation trap; the comms-cost ordering + the 671B
memory math; and why train↔infer drift breaks GRPO. Each was *built and profiled by you*, not read.

**Pointers:** the spine + the 2026 findings → [`PERFORMANCE_TRACK.md`](PERFORMANCE_TRACK.md); per-rung
specs → `docs/design/PERF_*` and `docs/design/L2_*`; frontier defaults per pillar →
[`FRONTIER_PRACTICE_2026.md`](FRONTIER_PRACTICE_2026.md); use `/master <concept>` to go deep on any
rung and `/tutor` / `/kernel-day` for the kernel reps.
