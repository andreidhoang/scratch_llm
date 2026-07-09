# Build status — scratch_llm (CS336 from-scratch)

Single source of truth for what is built, tested, and green. Updated as modules land.
Green-CI baseline: `ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`.

> 🆕 **Third front opened (2026-07-04 · [ADR-0018](adr/ADR-0018-close-the-loop-nanochat-front.md)).**
> **Close the loop:** adopt nanochat's end-to-end `speedrun.sh` spine + a report card to train an
> actual *talking* model from our own code (the loop has never closed — everything real is
> rental-gated; serving numbers were measured on random-weight toys, `bench/RESULTS.md:383`), then
> run an EV-ranked, pre-registered, iso-FLOP **frontier ablation study** (F1 MuonAdamW · F2 MTP
> draft head · F3 de-confound serving · F4 bf16+compile · F5 MLA-real · F6 MoE-balancing · F7 GRPO
> "aha" · F8 DSA · F9 logit-guard). Runs **in parallel** with perf + DELTA. Spec + DAG:
> [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md); ledger `bench/RESULTS.md` §Frontier
> ablations. Headline artifact = **nanochat d20** (~561M, ~$100, 8×H100), target CORE ≈ GPT-2.
>
> **✅ THE LOOP CLOSES (2026-07-04):** F1 Muon · train-wiring/F4 (bf16/compile + NaN guard) · eval
> report card (`val_bpb`/MC/generative/CORE-style) · `speedrun.py` + `scripts/speedrun.sh` spine all
> SHIPPED + pushed — GPU-verified end-to-end talking sample (`RESULTS.md` §Phase 0). **A1 real-corpus
> shards ✅ 2026-07-09** (`data/shards.py` + shard-backed `speedrun --data-dir`, FineWeb-EDU slice
> measured). **▶ NEXT NODE → A2** (checkpoint chaining) → F1-run. **The buildable next-phase DAG (23
> rungs, code-grounded, EV-ranked, with a START-HERE block) is
> [`FRONTIER_2026_TASKSPEC.md`](FRONTIER_2026_TASKSPEC.md)** — a
> fresh session reads that + the SessionStart `frontier node →` line and builds immediately.

**As of 2026-07-03: A1 substrate complete; perf-curriculum A1 serving rungs R0–R4.1 SHIPPED &
MEASURED on the standing sm120 GPU** — metrics harness → decode roofline (15%→53% HBM) → GQA/MQA
(MQA 1.92× @16K) → **continuous batching (2.30× wall / 2.93× by steps vs static-wave)** →
**PagedAttention (frag 5.0%, capacity ×9.3, fused Triton decode kernel 5.90 ms/step = ×3.52 vs
wave, +55% vs dense-continuous)** → **R4.2 chunked prefill (2026-07-04: mechanism ✓ token-exact,
44 tests; spike-reduction FALSIFIED — sequential-interleave regresses, R4.2b piggyback deferred)** →
**R4.3 speculative decoding (2026-07-04: SHIPPED lossless, 27 tests; ×1.2–1.4 wall / 1.3–1.5
tok/forward via n-gram drafting)** → **R4.4 CUDA-graph decode (2026-07-04: SHIPPED, B=1 −74% step /
253 tok/s = 77% of the memory wall; eager 20%→compiled 53%→graph 77%, R1 gap closed)**. Perf curriculum:
**ALL sm120-runnable rungs of A1–A5 COMPLETE 2026-07-04** — A1 R0–R4.6 (serving) · A2 R0–R6 (kernels:
GEMV/softmax/norm 96–100% HBM, TopK 46.9% + 3.4× fusion, GEMM 134% cuBLAS-proxy) · A3 R0–R2 (tensor
cores 4.1%→38.9%→81.9% cuBLAS, CUDA WMMA/mma.sync) · A4 R0–R3 (flash attn ~50% SDPA, 44× leaner) · A5
R0–R4 + §4.3 (INT8/INT4/NVFP4 1.48×<MXFP4/FP8-KV/AWQ 1.71×) · **A6 code-only** (TP MLP/1F1B/EP-MoE/MFU
gloo-verified, 112 tests) · **ISA kernels compile-verified** (WGMMA sm_90a, FA3 sm_90a, tcgen05 sm_100a
— PTX-checked on-box, runtime deferred). **Design notes A1–A7 ✅**; H100/B200/8×H200 runbooks + WGMMA
PTX artifact + `performance/rental/kernels/` compiled source ✅. **Every buildable-without-a-rental piece
is DONE + green**; the only remainder is the three rental DAYS themselves (run the compiled ISA kernels
on H100/B200 + the 8×H200 serving day — needs the hardware, not more code). (`performance/PERF_PLAN.md`). **CS336-A2 distributed half ✅ SHIPPED 2026-07-03** (ZeRO-1 ·
FSDP · 100B one-pager · comms algebra, W1–W4) **+ A3 scaling ✅** (fitter a=0.469/b=0.531 +
query planner, W5–W6); **A4 data pipeline ✅ + A5 alignment stack ✅ code-complete 2026-07-04**
(dedup/filters/quality/pipeline · SFT/EI/GRPO/Dr.GRPO/DPO + grader + envs, W7–W8; graded GPU runs
rental-gated per `deploy/runbooks/`) under the Delivery-Mode sprint (ADR-0014). **456 CPU + 14
GPU-marked tests green, ruff clean, pyright 0 errors.** Rentals decided: ADR-0012 (3
capability-tier sessions: H100 · 8×H200 serving day · B200). Mastery lessons (VI):
[`docs/learning/`](learning/INDEX.md).

> ⚡ **Delivery-Mode sprint opened (2026-07-03 · ADR-0013 `delegate` + ADR-0014).** Two agent
> fronts now run concurrently: the perf curriculum continues at its own node, and the **CS336 main
> track (A2 distributed finish → A3 → A4 → A5) is being finished autonomously** per
> [`EXECUTION_SPEC_CS336_FINISH.md`](EXECUTION_SPEC_CS336_FINISH.md) (task DAG W1–W11; >24 GB work
> ships code-complete + `deploy/runbooks/`). The 2026-06-30 "perf first" ordering mandate is
> dissolved into the two-front split. Official scaffolds re-cloned to `/workspace/lectures/`
> (adapter tests = acceptance oracle). Mastery is post-hoc: `learning/MASTERY_DEBT.md`.

> ✅ **Green-CI restored (2026-06-29).** The checkout had lost several files (no git to restore from):
> rebuilt `rollout/` (types/local/__init__ per `design/L2_rollout_seam_SPEC.md`), `envs/protocol.py`
> + the `algos/rewards/envs/scaling/data` package stubs, and `importorskip`'d the missing Mode-3
> `kernels/matmul.py` in its gpu-marked test. The real "blocked every Bash" deadlock was that **deps
> (torch/regex) weren't on the system Python the gate ran** — the uv venv (`.venv`, py3.11) has them;
> the gate now (a) fires only on `git commit` (its documented intent) and (b) uses the venv tools.
> **Suite green: 106 passed · 1 skipped · 33 deselected; ruff + pyright clean.** Box: **NVIDIA RTX PRO
> 4000 Blackwell, 24 GB, CUDA available** — real roofline/kernel runs are possible here, not just CPU.

> 🆕 **Performance & inference track (added 2026-06-29).** A six-stream deep-research pass (serving ·
> quant · MoE/parallelism · kernels/roofline · RL-systems · internal-doc audit) was synthesized into the
> **decode-memory-wall spine** + a **GPU-from-zero curriculum** for someone with no GPU background:
> [`PERFORMANCE_TRACK.md`](PERFORMANCE_TRACK.md) (the spine + 2026 findings + EV-ranked build list),
> [`GPU_FROM_ZERO.md`](GPU_FROM_ZERO.md) (rung 0→9, forced-mastery, AI-explains/human-implements), and
> the first specs `design/PERF_roofline_harness_SPEC.md` + `design/PERF_decode_roofline_SPEC.md`. The
> highest-value first result: the KV-cache is correctness-tested **but never *timed*** — time it.
>
> **Apparatus built (2026-06-29):** `src/scratch_llm/bench/` — `gpu_specs` (sourced dense roofline
> table + `measure_hbm_bandwidth`), `roofline` (the engine + op-counters; the decode-step counter
> computes to **AI = 1.00 FLOP/byte, ~297× below the H100 ridge** — the thesis, asserted in a test),
> `harness` (CUDA-event timing + p-quantiles), `ledger` (predict-vs-measure JSONL + regression guard).
> 9 CPU tests green (**115 total**, ruff/pyright clean). First live numbers on the RTX PRO 4000
> Blackwell: ~0.55 TB/s HBM, 4096³ bf16 matmul ~73 TFLOP/s. The ledger awaits your first
> *predict-then-measure* cycle — the prediction is your Mode-3 rep.

## End-to-end map — the whole stack at a glance

Two orderings, both true: **BUILD** is numeric (each layer rests on the last); **SHIP** is EV-ranked
(scarcest-skill-first, per [`../STRATEGY.md`](../STRATEGY.md) §8).

```
BUILD ▸ A1 ─► A2 ─► A3 ─► A4 ─► A5 ─► DELTA      SHIP ▸ perf A1–A7 (mandate 06-30) → A5 "aha" → DELTA
        byte ───────────── own every layer of a language model ───────────── RL update

 A1 Basics      ████████████ 100%  ✅  BPE · Transformer · AdamW · train · sample
 A2 Systems     ███████████░  ~95% 🟢  distributed half ✅ 2026-07-03 (W1–W4); only rental-tier serving left
 A3 Scaling     ██████████░░  ~85% 🟢  fitter + planner ✅ (W5–W6); Stanford-API leaderboard BLOCKED-EXTERNAL
 A4 Data        ███████████░  ~90% 🟢  extract→filter→quality→MinHash dedup ✅ (W7); full 5000-WET run = SKIP
 A5 Alignment   ███████████░  ~90% 🟢  SFT·EI·GRPO/Dr.GRPO·DPO + grader + envs ✅ (W8); graded runs rental-gated
 DELTA capstone  design ✅      0%  ⬜  GDN-2 decode kernel — base-first, gated on A5 ship + Step-0

 ►► TWO ACTIVE FRONTS: perf curriculum (node R4.2) · main-track sprint (W1–W8 ✅ → W9 acceptance/W10 runbooks/W11).
```

**A2 breakdown — CS336 systems (single-GPU half ✅; distributed half ✅ shipped 2026-07-03):**

```
 make ONE GPU fast  (single-GPU half ✅)        make MANY GPUs coherent  (distributed half, gloo)
 ─────────────────────────────────────          ─────────────────────────────────────────────
 FA2 forward (oracle + Triton)   ✅              DDP  naive → flat → overlap        ✅  D1
 FA2 backward (D-vector)         ✅  K           ZeRO-1 optimizer-state sharding    ✅  D2  (W1 d8142ef)
 activation/selective ckpt       ✅  M1          + 100B memory one-pager (tested)   ✅  D2  (W2 7cfb114)
 mixed-precision numerics        ✅  M2          FSDP per-param ZeRO-3 (graded)     ✅  D3  (W3 4c806bb)
 KV-cache · monitors · rollout   ✅              comms algebra (DP/FSDP/TP calcs)   ✅  D4  (W4 06127b5)
 roofline (53% SDPA @4k, 4090)   ✅              ── then Phase C frontier labs (opt-in) ──
 paged-KV + continuous batching  ✅  (shipped    speculative · ring-CP · TP toy · FP8 sim
   via perf-curriculum R3b/R4.1)
```

**Resourcing — we develop on a standing GPU now.** Hardware (2026-06-29): **NVIDIA RTX PRO 4000
Blackwell, sm120, 25 GB**, torch 2.12.1+cu130, triton 3.7.1. Measured peaks: **72 TFLOP/s bf16 ·
0.55 TB/s HBM · ridge ≈ 130 FLOP/byte** (in `scratch_llm.bench.gpu_specs`). **Empirically: Triton ✅
(FA2-fwd on sm120).** *(The Jun-29 perf CUDA suite — gemv/gemm/`rmsnorm.cu` — was reset 2026-07-01 to
rebuild `performance/` A1–A7 from scratch; preserved at tag `pre-perf-kernel-reset`.)* No build step is
deferred — **write the kernel, measure + profile it here, now.**
"CPU-buildable" = *doesn't require* a GPU (oracles, gloo-correctness, fake-quant); build it on the box
too. Rails: **25 GB cap** (size models) and **report "% of *this* Blackwell," not datacenter numbers**.
A bigger / multi-GPU box is rented (`vastai` skill) only for what this card can't do — full-scale
throughput vs H100/B200, real multi-GPU NCCL ([ADR-0008](adr/ADR-0008-sglang-hopper-only-on-ada.md) is
an open re-check on sm120). The `pytest -m "not gpu"` gate stays as the HW-agnostic commit/CI floor.

**Main-track sprint COMPLETE (2026-07-04, ADR-0014).** The A5 `algos/` stack is built, green, and
official-suite-accepted; the RL "aha" is now a *rental-execution* step, not a build step — code +
runbook ready (`deploy/runbooks/A5_countdown_aha.md`). Historical framing below (pre-sprint):
`algos/` SFT masked-CE → GRPO/Dr.GRPO wired to `utils/monitors.py` → reproduce the R1-Zero "aha" on
Countdown (Qwen2.5-1.5B; CPU-scaffolded + one ~$30–100 burst). The A2 distributed finish + OSS Rung-1
run in parallel; **DELTA is base-first** — its kernel is gated behind the A5 ship + the Step-0 gate.

**Capstone — DELTA (GDN-2 decode kernel):** design doc + 4-week barbell plan written & fact-checked
against primary sources (2026-06-14); **no kernel code yet** — build gated on the Step-0 dependency
gate. Tracked in the Capstone section below; the build spine is in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7.

---

## Performance Curriculum — Frontier GPU Engineering (A1→A7)

> **Ordering mandate (2026-06-30):** Complete the `performance/` curriculum (A1–A7) before resuming
> CS336 A5 RL / DELTA. A2–A5 kernel skills are direct DELTA prerequisites. Spec + plan:
> [`performance/PERF_ENGINEERING_SPEC.md`](../performance/PERF_ENGINEERING_SPEC.md) ·
> [`performance/PERF_PLAN.md`](../performance/PERF_PLAN.md).
> Hardware: standing GPU (sm_120, 0.55 TB/s, ridge≈130 FLOP/byte) for Phase 1; then **three
> capability-tier rentals** ([ADR-0012](adr/ADR-0012-inference-rental-tiers.md)): 1× H100 SXM
> (Phase 2, Hopper ISA + single-GPU serving), **1× 8×H200 NVLink node (Phase 4 — the frontier-MoE
> serving day: DeepSeek-R1 FP8, TP×EP, MLA KV, PD-disagg)**, 1× B200 (Phase 3, tcgen05/NVFP4);
> multi-node is optional. Estimated total: ~$235–460 in GPU spend, peak 8 GPUs concurrent.

**Current node: Phase 1a — A1 Rung 4.2 (chunked prefill: kill the measured ITL p99 admission spikes)**
_R0–R4.1 shipped 2026-07-01→03. R3b: continuous batching **2.30× wall / 2.93× by steps** vs
static-wave (R3.4 PASS), TTFT p95 **4.9×** (R3.6 PASS) on the `BatchedKVCache` slot buffer +
`serving/continuous.py` engine. R4.1: `PagedKVCache` (block table, CoW, admission guard) + the
fused Triton paged decode kernel — **frag 5.0%, capacity ×9.3, kernel 5.90 ms/step = ×3.52 vs
wave-dense, +55% over dense-continuous** at identical scheduling; the entire R3b padding tax
reclaimed (decomposition: ~3.2 ms padded-SDPA compute, ~0.5 ms bytes). 157 CPU + 14 GPU-marked
tests green. Details: `bench/RESULTS.md` §2026-07-03, node pointer `performance/PERF_PLAN.md`._

> **Clean slate (2026-07-01).** The exploratory Jun-29 perf kernels (GEMV/GEMM/RMSNorm CUDA +
> `kernels/bench.py`) were removed to build A1–A7 from scratch against `PERF_ENGINEERING_SPEC.md`.
> Kept: CS336 A2 FlashAttention-2, the `scratch_llm.bench` measurement apparatus, and `bench/RESULTS.md`.
> Prior work is at git tag `pre-perf-kernel-reset`. All rungs below are genuinely `⬜ not started`.

```
Assignment  Phase       Rungs                                         Status
──────────  ──────────  ────────────────────────────────────────────  ──────
A1 Infer    1a (sm120)  R0-R3 + 4.1 PagedAttn + 4.2-4.6 frontier     🔵  R0-R4.1 done → R4.2
A2 Kernel   1b (sm120)  R0-R6 CUDA-core ladder                        ⬜  not started
            Phase 2     §4.1-4.5 WGMMA+TMA+FP8 (H100)                ⬜  gate: Phase 1b done
A3 TC       1c (sm120)  R0-R2 WMMA + mma.sync                         ⬜  not started
            Phase 2     R3-R4 + §4.1 warp-spec (H100)                 ⬜  gate: Phase 2
            Phase 3     §4.2-4.3 tcgen05/NVFP4 (B200)                 ⬜  gate: Phase 3
A4 FlashA   1d (sm120)  R0-R3 + §4.3 variant + backward               ⬜  not started
            Phase 2     R4 FA3-class (H100)                            ⬜  gate: Phase 2
A5 Quant    1e (sm120)  R0-R4 + §4.3 PTQ                              ⬜  not started
            Phase 3     §7 NVFP4 native MMA (B200)                    ⬜  gate: Phase 3
A6 Dist     Phase 4     R0-R1 inside the 8×H200 SERVING DAY (R1-FP8   ⬜  gate: Phase 4
                        TP×EP · MLA KV · PD-disagg; ADR-0012)
            Phase 5     R2 pipeline + R3/§4.1 multi-node — OPTIONAL    ⬜  opt-in only
A7 Cap      Phase 5     Track B kernel suite                           ⬜  gate: A1-A6 done
```

Measurement ledger: all numbers log to `bench/RESULTS.md` (append-only, dated).
Design notes: `performance/notes/A#_design_note.md` (create dir on first note).

**Frontier CORE-defaults to adopt** (the cheap, high-value 2026 upgrades — see
[`FRONTIER_PRACTICE_2026.md`](FRONTIER_PRACTICE_2026.md)): A1 — output z-loss · WSD
schedule · depth-scaled init (QK-norm ✅ landed, opt-in); A2 — selective recompute ✅ landed · FSDP2 (D3 ahead); A3 — inference-aware allocation ·
robust Huber+bootstrap fit; A4 — decontamination gate · DCLM-style classifier positives; A5 — KL k3
estimator · RLOO · off-policy epochs>1. *Adoption is tracked as green commits in the build state above
— no separate ledger (build state + git is the source of truth).*

## A1 — Basics (substrate) ✅ complete

| Module | What it owns | Tests |
|---|---|---|
| `tokenizer.py` | byte-level BPE train + encode/decode/`from_files`, special-token boundaries | `bpe_example` exact repro, round-trip, specials, lexicographic tie-break |
| `model.py` | decoder LM (RMSNorm · RoPE · SwiGLU · GQA-ready MHA · opt-in QK-norm) + `cross_entropy` | **loss-at-init ≈ log V**, causal-no-leak, RoPE-relative, shapes, config, **QK-norm off=identity / bounds logits** |
| `moe.py` (opt-in) | DeepSeek-style MoE FFN (sigmoid gate · aux-loss-free balancing · shared+routed experts · z-loss) | dense-equiv, decode/cache parity, loss-at-init, **balancer-overcomes-preference**, ckpt round-trip |
| `optim.py` | AdamW (decoupled wd, β₂=0.95) + global-ℓ₂ clip + cosine schedule | quadratic, decoupled-wd, clip, schedule, **overfit-one-batch** |
| `train.py` | `np.memmap` batches + checkpoint + loop | next-token align, ckpt round-trip, **seed-repro**, learns |
| `sampling.py` | temperature / top-p (nucleus) decode | greedy=argmax, nucleus, stop, budget, seed-repro |
| `utils/seeding.py` | `seed_everything` (py/numpy/torch) | — |
| `tests/test_integration_l1.py` | end-to-end: BPE → tokenize → train → sample | composes |

## A2 — Systems — ✅ complete (single-GPU + distributed halves; real-serving rental-gated)

| Module | Status | Notes |
|---|---|---|
| `kernels/` FA2 fwd (oracle + Triton) + roofline | ✅ | pure-PyTorch tiled oracle (CPU tests) + autotuned Triton fwd+causal, validated on a 4090 vs oracle/SDPA. **Roofline: 53 % of SDPA @ seq 4k** (honest gap shipped). Spec: [`design/L2_flash_attention_SPEC.md`](design/L2_flash_attention_SPEC.md) |
| `kernels/` FA2 **backward** (recomputation) | ✅ | `FlashAttentionPyTorch(autograd.Function)`: save `(Q,K,V,O,L)`, recompute `S,P` in bwd, softmax-Jacobian via the D-vector `D=(O∘dO).sum(-1)`; grads == SDPA autograd to 1e-10/1e-8 (causal+non-causal, ragged). torch.compile recomputation path, not hand-rolled Triton bwd (SKIP) |
| activation checkpointing (none/full/**selective**) | ✅ | `utils/checkpointing.py`: recompute-vs-store on two CPU-measured axes — grads transparent; boundary memory full,selective≪none; matmul recompute full>selective==none (selective = the 2026 SAC default) |
| mixed-precision numerics | ✅ | `utils/mixed_precision.py`: fp16 sequential accumulation stalls (under-counts small addends) vs fp32; autocast rule matmul→bf16, LayerNorm/softmax→fp32 |
| KV-cache (incremental decode) | ✅ | `KVCache` + cache-aware attention/forward; cached == recompute (MHA/GQA/batch). Spec: [`design/L2_kv_cache_SPEC.md`](design/L2_kv_cache_SPEC.md) |
| `utils/monitors.py` | ✅ | entropy, the KL divergences, IS ratios + ESS, reward/length stats |
| `rollout/` client seam + `LocalBackend` | ✅ | `Rollout`/`RolloutClient` contract + CPU `LocalBackend`; per-token log-probs feed `monitors`. Spec: [`design/L2_rollout_seam_SPEC.md`](design/L2_rollout_seam_SPEC.md) |
| DDP (naive → flat-bucket → overlap) | ✅ | `utils/ddp.py` — naive/flat/overlap containers + 2-rank gloo equivalence (665b6e6) |
| ZeRO-1 optimizer-state sharding | ✅ | `utils/zero1.py` — owner-broadcast sharded optimizer; 2-rank gloo equivalence ×3 seeds (W1, d8142ef) |
| FSDP | ✅ | `utils/fsdp.py` — per-param ZeRO-3: ≥2-D matrices sharded (reduce-scatter), ≤1-D norm/bias replicated (all-reduced) per the official gradient-sync contract; gloo trajectory equivalence (W3 4c806bb, policy 47f49a1). **Official `test_fsdp` 4/4 PASS.** |
| ZeRO-1 accounting / comms algebra | ✅ | `utils/memory_math.py` + `docs/design/A2_100B_MEMORY_ONEPAGER.md` (W2 7cfb114); `utils/comms_calc.py` + `docs/design/A2_COMMS_ALGEBRA.md` — ring/DP/FSDP/TP byte model, verified vs the official PDF §8 (W4 06127b5) |
| real serving (SGLang, fp8/INT4-KV) | ⬜ Hopper box | `rollout/sglang_client.py` ready; SGLang needs sm90+ ([ADR-0008](adr/ADR-0008-sglang-hopper-only-on-ada.md)). Multi-GPU NCCL benchmarks: runbook `deploy/runbooks/A2_multigpu_nccl_bench.md` |

**Official acceptance (W9):** A2 **10 PASS / 0 FAIL** (+4 Triton GPU-blocked) — `deploy/runbooks/OFFICIAL_SUITES.md`.

## A3 Scaling — ✅ complete · A4 Data — ✅ complete · A5 Alignment — ✅ complete (graded runs rental-gated)

- **A3** — ✅ 2026-07-03: `scaling/isoflop.py` (a=0.469 · b=0.531 · a+b gate PASS · N_opt ≈70B @1e23 / ≈206B @1e24 · `bench/a3_isoflop.png`) + `scaling/planner.py` (budget reserve/refund semantics). Stanford-API leaderboard BLOCKED-EXTERNAL — runbook `deploy/runbooks/A3_stanford_api_leaderboard.md`.
- **A4** — ✅ 2026-07-04 (W7): `data/dedup.py` (exact + MinHash/LSH, P[match]=Jaccard, cluster-and-drop), `data/filters.py` (extract·langid·PII·NSFW/toxic·Gopher, one (label,score) interface), `data/quality.py` (trained-fastText signal design, ADR-0016), `data/pipeline.py` (canonical order + discard accounting). **Official acceptance 20 PASS / 0 FAIL** (+1 quality-model-artifact blocked). Full 5000-WET run = SKIP; slice runbook `deploy/runbooks/A4_full_slice.md`.
- **A5** — ✅ 2026-07-04 (W8): `algos/sft.py` (masking primitives + SFT step), `algos/expert_iteration.py` (STaR), `algos/grpo.py` (GRPO/Dr.GRPO + clip + dispatcher + train loop, ADR-0017), `algos/dpo.py` (+ Bradley-Terry), `rewards/r1_zero.py` (verifiable grader), `envs/{countdown,gsm_math}.py`. RL logging wired through `utils/monitors.py`; toy GRPO end-to-end on Countdown (reward-rises, CPU). **Official acceptance 20 PASS / 0 FAIL** (+6 out-of-scope blocked). Graded Qwen2.5-Math-1.5B runs + the R1-Zero "aha" repro are rental-gated — runbooks `deploy/runbooks/A5_qwen_math_rl.md` · `A5_countdown_aha.md`. Frontier lab (GDM-aligned): `algos/distill.py` — knowledge distillation, the serve-cheap student track (see `FRONTIER_PRACTICE_2026.md` A5 🔵), still open.

## Capstone — DELTA (GDN-2 decode kernel · the barbell *spike*) — designed, build pending

The portfolio **spike** that sits on the A2/A5 **base** (the barbell: one deep differentiating artifact +
the broad CS336 base — see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7). A fused **decode-step**
kernel for **GatedDeltaNet-2** (NVIDIA, arXiv 2605.22791), targeting ≥85% of the measured H100 memory
roofline and the "erase/write decoupling is free at decode" thesis. Design + plan live in the **workspace
root** (one dir up from this repo).

> ⚠️ **Re-aim (2026-06-27).** The *plain* GDN-decode gap is now **closed** — FlashInfer ships `gated_delta_rule_decode` and MLSys-2026 has a GDN contest track. Re-point the spike to **NVFP4 on the GDN recurrent state** (production NVFP4 keeps the recurrent state in BF16; FP4-state error-accumulation vs context length is unmeasured) and land it as **the public artifact** (goal G3 of the private workspace `ROADMAP.md` — not in this repo). The Phase-1 harness (= the A2 finish) is unchanged.
>
> **Artifact target (ready to scope).** The MLSys-2026 FlashInfer GDN track has *concluded* (winners public) — so its problem + harness are now a gold **reference to reproduce-and-beat**, not a live submission. Ship: (1) a fused **GDN-decode** kernel matched to that benchmark; (2) the **NVFP4-on-recurrent-state** numerics probe — error vs context length, the genuinely open seam. → as a **FlashInfer/SGLang PR + a GPU MODE leaderboard entry** (permanent) + a short writeup. **Kill gate:** if the plain GDN-decode kernel can't get within ~10% of FlashInfer's, drop to "NVFP4-state numerics writeup only." Gated on the A5 ship + a rented H100 (Step-0).

| Artifact | State | Where |
|---|---|---|
| Design doc (RFC) — hypothesis · P1–P7 · scope · kill criteria | ✅ written, fact-checked 2026-06-14 | `../../DELTA.md` (Part I) |
| Execution plan — 4-week barbell schedule · rental-compute discipline | ✅ written | `../../DELTA.md` (Part II) |
| Step-0 dependency gate (rental H100 + checkpoint story) | ⬜ **next action** | RFC §10 / plan §2 |
| Phase-1 harness (`fla` baseline + Nsight roofline + correctness oracle) | ⬜ **= the A2 inference-systems finish** | `src/scratch_llm/kernels/` |
| Decode kernel (Triton-first) + ablations + forensic postmortem | ⬜ build | new |
| A5 RL "aha" (GRPO/Dr.GRPO) — the barbell's *base* track, interleaved | ⬜ (see A5 above) | `algos/`, `rewards/`, `envs/` |

**Scope-shaping facts (verified):** NVlabs released **training code only — no checkpoint** (so
correctness/roofline run architecture-only; e2e is *projected* via Amdahl); config 16 heads,
`d_k=d_v=128` → **32 KB/head** state; non-linear hybrid layers are **2K sliding-window attention**;
**NVIDIA Source-Code-NC** (non-commercial) license. The Phase-1 harness **is** the A2 deliverable — spike
and base share infrastructure, they do not compete.

## Decisions locked (`docs/adr/`)

ADR-0001 tokenizer of record · ADR-0002 GQA in the substrate · ADR-0003 sampler parity · ADR-0004
dense default / MoE opt-in · ADR-0006 rollout log-prob convention · ADR-0007 MoE FFN (opt-in A1
extension) · ADR-0008 SGLang is Hopper-only on Ada · ADR-0011 kernel-framework policy (Triton-primary, CUDA/CUTLASS second-tier) · ADR-0012 inference rental tiers (3 sessions: H100 · 8×H200 serving day · B200; multi-node optional).
