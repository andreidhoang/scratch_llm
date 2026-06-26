# Build status — scratch_llm (CS336 from-scratch)

Single source of truth for what is built, tested, and green. Updated as modules land.
Green-CI baseline: `ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`.

**As of 2026-06-20: A1 substrate complete; A2 systems partial (FlashAttention-2 + KV-cache +
monitors + rollout seam done; DDP/ZeRO-1/FSDP + the memory one-pager remain). A3/A4/A5 are clean
stubs. 92 tests green, ruff clean, pyright 0 errors, CPU suite ≈ 16 s; last code commit Jun 8.**

**Next ship — EV-ranked, not numeric (per [`../STRATEGY.md`](../STRATEGY.md) §8): the A5 RL "aha".**
`algos/` SFT masked-CE → GRPO/Dr.GRPO wired to `utils/monitors.py` → reproduce the R1-Zero "aha" on
Countdown (Qwen2.5-1.5B; CPU-scaffolded + one ~$30–100 burst). The A2 distributed finish + OSS Rung-1
run in parallel; **DELTA is base-first** — its kernel is gated behind the A5 ship + the Step-0 gate.

**Capstone — DELTA (GDN-2 decode kernel):** design doc + 4-week barbell plan written & fact-checked
against primary sources (2026-06-14); **no kernel code yet** — build gated on the Step-0 dependency
gate. Tracked in the Capstone section below; the build spine is in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §7.

**Frontier CORE-defaults to adopt** (the cheap, high-value 2026 upgrades — see
[`FRONTIER_PRACTICE_2026.md`](FRONTIER_PRACTICE_2026.md)): A1 — output z-loss · WSD
schedule · depth-scaled init (QK-norm ✅ landed, opt-in); A2 — selective recompute · FSDP2; A3 — inference-aware allocation ·
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
| `kernels/` FA2 (oracle + Triton) + roofline | ✅ | pure-PyTorch tiled oracle (CPU tests) + autotuned Triton fwd+causal, validated on a 4090 vs oracle/SDPA. **Roofline: 53 % of SDPA @ seq 4k** (honest gap shipped). Spec: [`design/L2_flash_attention_SPEC.md`](design/L2_flash_attention_SPEC.md) |
| KV-cache (incremental decode) | ✅ | `KVCache` + cache-aware attention/forward; cached == recompute (MHA/GQA/batch). Spec: [`design/L2_kv_cache_SPEC.md`](design/L2_kv_cache_SPEC.md) |
| `utils/monitors.py` | ✅ | entropy, the KL divergences, IS ratios + ESS, reward/length stats |
| `rollout/` client seam + `LocalBackend` | ✅ | `Rollout`/`RolloutClient` contract + CPU `LocalBackend`; per-token log-probs feed `monitors`. Spec: [`design/L2_rollout_seam_SPEC.md`](design/L2_rollout_seam_SPEC.md) |
| DDP (naive → flat-bucket → overlap) | ⬜ | CPU-buildable via the gloo backend |
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
extension) · ADR-0008 SGLang is Hopper-only on Ada.
