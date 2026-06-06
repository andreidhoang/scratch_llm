# Build status — reasoningLLM v0.1.0

Single source of truth for what is built, tested, and green. Updated as modules land.
Green-CI baseline: `ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`.

**As of last update: L1 substrate complete (+ A1.1 MoE pulled forward, opt-in); L2 =
`utils/monitors.py` + KV-cache + rollout seam complete. Repo now under git (green-CI pre-commit
hook wired). 76 tests green, ruff clean, pyright 0 errors, suite ≈ 12s on CPU.**

## L1 — Substrate (A1) ✅ complete

| Module | What it owns | Tests |
|---|---|---|
| `tokenizer.py` | byte-level BPE train + encode/decode/`from_files` | bpe_example exact repro, round-trip, specials, tie-break (8) |
| `model.py` | GQA-ready decoder LM (RMSNorm·RoPE·SwiGLU·MHA) + `cross_entropy` | loss-at-init≈logV, causal-no-leak, RoPE-relative, shapes, config (10) |
| `moe.py` (A1.1, **opt-in**) | DeepSeek-V3 MoE FFN: sigmoid gate · aux-loss-free bias · shared+routed experts · z-loss · seq-aux · entropy diag ([ADR-0007](adr/ADR-0007-moe-pulled-forward.md); traced end-to-end in [`design/L1_moe_WALKTHROUGH.md`](design/L1_moe_WALKTHROUGH.md)). Dense default unchanged | **dense-equiv**, **decode/cache parity**, loss-at-init≈logV, overfit-one-batch, bias-update dir, entropy>0.9·logNᵣ, **balancer-overcomes-preference**, train-loop, ckpt-bias-roundtrip (16) |
| `optim.py` | AdamW (decoupled wd, β₂=0.95) + global-ℓ₂ clip + cosine LR | quadratic, decoupled-wd, clip, schedule, **overfit-one-batch** (5) |
| `train.py` | `np.memmap` batches + checkpoint + loop | next-token align, ckpt round-trip, **seed-repro**, learns (6) |
| `sampling.py` | temperature/top-p decode (shared `SamplingParams`, ADR-0003) | greedy=argmax, nucleus, stop, budget, seed-repro (7) |
| `utils/seeding.py` | `seed_everything` (py/numpy/torch) | — |
| `tests/test_integration_l1.py` | end-to-end: BPE→tokenize→train→sample | composes (1) |

## L2 — Systems (A2) — in progress

| Module | Status | Notes |
|---|---|---|
| `utils/monitors.py` | ✅ complete | three KLs, **`kl_train_infer` HALT@0.10**, IS ratios + ESS, reward/length stats (12 tests) |
| KV-cache (incremental decode) | ✅ complete | `KVCache` + cache-aware attention/forward + `generate(use_cache=True)`; cached==recompute (MHA/GQA/batch, 6 tests). Spec: [`design/L2_kv_cache_SPEC.md`](design/L2_kv_cache_SPEC.md) |
| `rollout/` client seam + `LocalBackend` | ✅ complete | `Rollout`/`RolloutClient` contract + CPU `LocalBackend` over `generate`; per-token policy log π (temp 1), `score`/`distribution_logprobs` feed `monitors` → the train↔infer (`kl_train_infer`) harness. Contract/seed/batch/stop/KL-scaffold (6 tests). Spec: [`design/L2_rollout_seam_SPEC.md`](design/L2_rollout_seam_SPEC.md) |
| `kernels/` FA2 forward (oracle + Triton) + roofline (A2.1) | ✅ complete | pure-PyTorch tiled oracle (8 CPU tests) + autotuned Triton fwd+causal, validated on a 4090 vs oracle/SDPA (10 gpu tests). **Roofline: 53 % of SDPA @ seq 4k (predicted 65 %) — honest gap shipped per kill-criterion.** Spec: [`design/L2_flash_attention_SPEC.md`](design/L2_flash_attention_SPEC.md) |
| `kl_train_infer` bridge (A2.3) — serve vs train | ✅ measured | **exact full-vocab `KL(train‖infer)` on a 4090** (`monitors.mean_kl`): serve=sdpa/bf16 vs train=eager/{bf16,fp32} on Qwen2.5-0.5B → 0.0098 / 0.0017 (HALT ok); falsified prediction → **drift = accumulation-precision + kernel, not storage dtype**. `SGLangBackend` written but **sgl_kernel is sm90/sm100-only, no sm89** → SGLang can't run on Ada ([ADR-0008](adr/ADR-0008-sglang-hopper-only-on-ada.md)); HF engine-pair stand-in used. Spec: [`design/L2_kl_train_infer_SPEC.md`](design/L2_kl_train_infer_SPEC.md) |
| Real SGLang serving (fp8/INT4-KV drift) | ⬜ Hopper box | `sglang_client.py` ready; needs sm90+ (rent H100) to exercise SGLang's fused kernels / quantized KV — the toward-HALT demonstration |
| DDP-overlap/ZeRO-1 + 100B memory one-pager | ⬜ GPU/CPU-gloo | remaining A2 "money layer" — DDP/ZeRO are CPU-buildable via the gloo backend; **not skipped** (CLAUDE.md "Follow the plan") |

## L5 — RLVR engine (A5) — after L2

| Module | Status |
|---|---|
| `algos/` scoring/masking primitives + SFT step (CPU) | ⬜ after L2 inference |
| `algos/advantage.py` (GRPO/Dr.GRPO/RLOO) · `algos/off_policy.py` (clip, truncated-IS + ESS) | ⬜ after L2 inference |
| `rewards/*`, `envs/exploitability.py`, `envs/true_quality.py` | ⬜ after the RL spine |

## L3 Scaling (A3) · L4 Data (A4) — not started

## Decisions locked (`docs/adr/`)
- ADR-0001 tokenizer of record · ADR-0002 GQA in substrate · ADR-0003 sampler parity · ADR-0004 dense v0.1.0 · ADR-0006 rollout policy log-prob convention · ADR-0007 MoE (A1.1) pulled forward, opt-in (supersedes ADR-0004 sequencing)
- *(L5 GRPO decisions — Dr.GRPO default, DPO scope, IS truncation — will be logged as ADRs when the RL spine lands.)*
