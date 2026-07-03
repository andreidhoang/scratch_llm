# Build status — scratch_llm (CS336 from-scratch)

Single source of truth for what is built, tested, and green. Updated as modules land.
Green-CI baseline: `ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`.

**As of 2026-06-27: A1 substrate complete; A2 systems — the single-GPU half is done (FA2 fwd+bwd,
activation/selective checkpointing, mixed-precision numerics, KV-cache, monitors, rollout seam); the
distributed half (DDP ✅ → ZeRO-1 → FSDP + the 100B memory one-pager) is in progress, all
CPU/gloo-buildable. A3/A4/A5 are clean stubs. 106 tests green, ruff clean, pyright 0 errors.**

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
BUILD ▸ A1 ─► A2 ─► A3 ─► A4 ─► A5 ─► DELTA      SHIP ▸ A5 "aha" · (A2 finish ∥ OSS) · DELTA
        byte ───────────── own every layer of a language model ───────────── RL update

 A1 Basics      ████████████ 100%  ✅  BPE · Transformer · AdamW · train · sample
 A2 Systems     ███████░░░░░  ~60% 🟡  ACTIVE — see breakdown below
 A3 Scaling     ░░░░░░░░░░░░    0%  ⬜  IsoFLOP / Chinchilla fitter (clean stub)
 A4 Data        ░░░░░░░░░░░░    0%  ⬜  extract → filter → classify → MinHash dedup (clean stub)
 A5 Alignment   ░░░░░░░░░░░░    0%  ⬜  SFT → Expert-Iter → GRPO/Dr.GRPO  ◄ highest-EV SHIP target
 DELTA capstone  design ✅      0%  ⬜  GDN-2 decode kernel — base-first, gated on A5 ship + Step-0
```

**A2 breakdown — the active front (the densest interview surface):**

```
 make ONE GPU fast  (single-GPU half ✅)        make MANY GPUs coherent  (distributed half, gloo)
 ─────────────────────────────────────          ─────────────────────────────────────────────
 FA2 forward (oracle + Triton)   ✅              DDP  naive → flat → overlap        ✅  D1
 FA2 backward (D-vector)         ✅  K           ZeRO-1 optimizer-state sharding    ⬜  D2  ◄ NEXT
 activation/selective ckpt       ✅  M1          + 100B memory one-pager (written)  ⬜  D2
 mixed-precision numerics        ✅  M2          FSDP2 per-param (graded)           ⬜  D3
 KV-cache · monitors · rollout   ✅              comms algebra (DP/FSDP/TP calcs)   ⬜  D4
 roofline (53% SDPA @4k, 4090)   ✅              ── then Phase C frontier labs (opt-in) ──
                                                 paged-KV · speculative · ring-CP · TP toy · FP8 sim
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

**Next ship — EV-ranked, not numeric (per [`../STRATEGY.md`](../STRATEGY.md) §8): the A5 RL "aha".**
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
> Hardware: standing GPU (sm_120, 0.55 TB/s, ridge≈130 FLOP/byte) for Phase 1; rent H100 (Phase 2),
> B200 (Phase 3), 8×H100 (Phase 4), 2-node (Phase 5). Estimated total: ~$250–$310 in GPU spend.

**Current node: Phase 1a — A1 Rung 4.1 (PagedAttention: reclaim the measured 1.27× padding tax)**
_R0–R3 shipped 2026-07-01→03. R3b (2026-07-03): continuous batching measured **2.30× wall / 2.93×
by steps** vs static-wave on the heavy-tail trace (R3.4 PASS), TTFT p95 **4.9×** (R3.6 PASS), on the
new `BatchedKVCache` static slot buffer + `serving/continuous.py` engine (147 CPU tests green). The
wall-vs-steps gap is the dense buffer's padding traffic — measured at ~1.27× — which is R4.1's
motivation, quantified. Details: `bench/RESULTS.md` §2026-07-03, node pointer `performance/PERF_PLAN.md`._

> **Clean slate (2026-07-01).** The exploratory Jun-29 perf kernels (GEMV/GEMM/RMSNorm CUDA +
> `kernels/bench.py`) were removed to build A1–A7 from scratch against `PERF_ENGINEERING_SPEC.md`.
> Kept: CS336 A2 FlashAttention-2, the `scratch_llm.bench` measurement apparatus, and `bench/RESULTS.md`.
> Prior work is at git tag `pre-perf-kernel-reset`. All rungs below are genuinely `⬜ not started`.

```
Assignment  Phase       Rungs                                         Status
──────────  ──────────  ────────────────────────────────────────────  ──────
A1 Infer    1a (sm120)  R0-R3 + 4.1 PagedAttn + 4.2-4.6 frontier     🔵  R0-R3 done → R4.1
A2 Kernel   1b (sm120)  R0-R6 CUDA-core ladder                        ⬜  not started
            Phase 2     §4.1-4.5 WGMMA+TMA+FP8 (H100)                ⬜  gate: Phase 1b done
A3 TC       1c (sm120)  R0-R2 WMMA + mma.sync                         ⬜  not started
            Phase 2     R3-R4 + §4.1 warp-spec (H100)                 ⬜  gate: Phase 2
            Phase 3     §4.2-4.3 tcgen05/NVFP4 (B200)                 ⬜  gate: Phase 3
A4 FlashA   1d (sm120)  R0-R3 + §4.3 variant + backward               ⬜  not started
            Phase 2     R4 FA3-class (H100)                            ⬜  gate: Phase 2
A5 Quant    1e (sm120)  R0-R4 + §4.3 PTQ                              ⬜  not started
            Phase 3     §7 NVFP4 native MMA (B200)                    ⬜  gate: Phase 3
A6 Dist     Phase 4     R0-R2 + §4.2-4.5 single-node (8×H100)        ⬜  gate: Phase 4
            Phase 5     R3 + §4.1 multi-node (2-node, 16×H100)        ⬜  gate: Phase 5
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

## A2 — Systems — partial

| Module | Status | Notes |
|---|---|---|
| `kernels/` FA2 fwd (oracle + Triton) + roofline | ✅ | pure-PyTorch tiled oracle (CPU tests) + autotuned Triton fwd+causal, validated on a 4090 vs oracle/SDPA. **Roofline: 53 % of SDPA @ seq 4k** (honest gap shipped). Spec: [`design/L2_flash_attention_SPEC.md`](design/L2_flash_attention_SPEC.md) |
| `kernels/` FA2 **backward** (recomputation) | ✅ | `FlashAttentionPyTorch(autograd.Function)`: save `(Q,K,V,O,L)`, recompute `S,P` in bwd, softmax-Jacobian via the D-vector `D=(O∘dO).sum(-1)`; grads == SDPA autograd to 1e-10/1e-8 (causal+non-causal, ragged). torch.compile recomputation path, not hand-rolled Triton bwd (SKIP) |
| activation checkpointing (none/full/**selective**) | ✅ | `utils/checkpointing.py`: recompute-vs-store on two CPU-measured axes — grads transparent; boundary memory full,selective≪none; matmul recompute full>selective==none (selective = the 2026 SAC default) |
| mixed-precision numerics | ✅ | `utils/mixed_precision.py`: fp16 sequential accumulation stalls (under-counts small addends) vs fp32; autocast rule matmul→bf16, LayerNorm/softmax→fp32 |
| KV-cache (incremental decode) | ✅ | `KVCache` + cache-aware attention/forward; cached == recompute (MHA/GQA/batch). Spec: [`design/L2_kv_cache_SPEC.md`](design/L2_kv_cache_SPEC.md) |
| `utils/monitors.py` | ✅ | entropy, the KL divergences, IS ratios + ESS, reward/length stats |
| `rollout/` client seam + `LocalBackend` | ✅ | `Rollout`/`RolloutClient` contract + CPU `LocalBackend`; per-token log-probs feed `monitors`. Spec: [`design/L2_rollout_seam_SPEC.md`](design/L2_rollout_seam_SPEC.md) |
| DDP (naive → flat-bucket → overlap) | ⬜ **next** | CPU-buildable via the gloo backend |
| ZeRO-1 optimizer-state sharding | ⬜ | CPU/gloo |
| FSDP | ⬜ | CPU/gloo; a full graded A2 deliverable |
| 100B memory-math one-pager | ⬜ | the verbatim "train a 100B model" answer (params+grads+Adam ≈ 16–20 B/param) |
| real serving (SGLang, fp8/INT4-KV) | ⬜ Hopper box | `rollout/sglang_client.py` ready; SGLang needs sm90+ ([ADR-0008](adr/ADR-0008-sglang-hopper-only-on-ada.md)) |

## A3 Scaling · A4 Data · A5 Alignment — to build (clean stubs)

- **A3** — `scaling/`: the IsoFLOP / Chinchilla fitter (CPU/numpy: min-picking, `N_opt ∝ C^a`, `C=6ND`, `a+b≈1`, extrapolation). The Stanford training-API leaderboard runs in `../lectures/assignment3-scaling`.
- **A4** — `data/`: extract → filter → quality-classify → exact + MinHash/LSH dedup; pipeline order + discard accounting.
- **A5** — `algos/`, `rewards/`, `envs/`: SFT → Expert Iteration → GRPO/Dr.GRPO + DPO; the env/grader protocol (`envs/protocol.py`) is already in place. Frontier lab (GDM-aligned): `algos/distill.py` — knowledge distillation (logit / on-policy reverse-KL / sequence-level), the serve-cheap student track (see `FRONTIER_PRACTICE_2026.md` A5 🔵).

## Capstone — DELTA (GDN-2 decode kernel · the barbell *spike*) — designed, build pending

The portfolio **spike** that sits on the A2/A5 **base** (the barbell: one deep differentiating artifact +
the broad CS336 base — see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7). A fused **decode-step**
kernel for **GatedDeltaNet-2** (NVIDIA, arXiv 2605.22791), targeting ≥85% of the measured H100 memory
roofline and the "erase/write decoupling is free at decode" thesis. Design + plan live in the **workspace
root** (one dir up from this repo).

> ⚠️ **Re-aim (2026-06-27).** The *plain* GDN-decode gap is now **closed** — FlashInfer ships `gated_delta_rule_decode` and MLSys-2026 has a GDN contest track. Re-point the spike to **NVFP4 on the GDN recurrent state** (production NVFP4 keeps the recurrent state in BF16; FP4-state error-accumulation vs context length is unmeasured) and land it as **the public artifact** (`ROADMAP.md` G3). The Phase-1 harness (= the A2 finish) is unchanged.
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
extension) · ADR-0008 SGLang is Hopper-only on Ada · ADR-0011 kernel-framework policy (Triton-primary, CUDA/CUTLASS second-tier).
