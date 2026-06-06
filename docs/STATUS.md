# Build status — reasoningLLM v0.1.0

Single source of truth for what is built, tested, and green. Updated as modules land.
Green-CI baseline: `ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`.

**As of last update: L1 substrate complete; L2 = `utils/monitors.py` + KV-cache complete.
53 tests green, ruff clean, pyright 0 errors, suite ≈ 16s on CPU.**

## L1 — Substrate (A1) ✅ complete

| Module | What it owns | Tests |
|---|---|---|
| `tokenizer.py` | byte-level BPE train + encode/decode/`from_files` | bpe_example exact repro, round-trip, specials, tie-break (8) |
| `model.py` | GQA-ready decoder LM (RMSNorm·RoPE·SwiGLU·MHA) + `cross_entropy` | loss-at-init≈logV, causal-no-leak, RoPE-relative, shapes, config (10) |
| `optim.py` | AdamW (decoupled wd, β₂=0.95) + global-ℓ₂ clip + cosine LR | quadratic, decoupled-wd, clip, schedule, **overfit-one-batch** (5) |
| `train.py` | `np.memmap` batches + checkpoint + loop | next-token align, ckpt round-trip, **seed-repro**, learns (6) |
| `sampling.py` | temperature/top-p decode (shared `SamplingParams`, ADR-0003) | greedy=argmax, nucleus, stop, budget, seed-repro (7) |
| `utils/seeding.py` | `seed_everything` (py/numpy/torch) | — |
| `tests/test_integration_l1.py` | end-to-end: BPE→tokenize→train→sample | composes (1) |

## L2 — Systems (A2) — in progress (NEXT)

| Module | Status | Notes |
|---|---|---|
| `utils/monitors.py` | ✅ complete | three KLs, **`kl_train_infer` HALT@0.10**, IS ratios + ESS, reward/length stats (12 tests) |
| KV-cache (incremental decode) | ✅ complete | `KVCache` + cache-aware attention/forward + `generate(use_cache=True)`; cached==recompute (MHA/GQA/batch, 6 tests). Spec: [`design/L2_kv_cache_SPEC.md`](design/L2_kv_cache_SPEC.md) |
| `rollout/` client seam + local backend | ⬜ **next (CPU)** | defines the rollout contract over `generate`; SGLang backend slots in on a GPU box |
| Triton FA2 kernel · DDP/ZeRO · SGLang serving | ⬜ GPU (rent vast.ai) | scheduled on a rented GPU — **not skipped** (no CUDA/Triton on this Mac); see CLAUDE.md "Follow the plan" |

## L5 — RLVR engine (A5) — after L2

| Module | Status |
|---|---|
| `algos/` scoring/masking primitives + SFT step (CPU) | ⬜ after L2 inference |
| `algos/advantage.py` (GRPO/Dr.GRPO/RLOO) · `algos/off_policy.py` (clip, truncated-IS + ESS) | ⬜ after L2 inference |
| `rewards/*`, `envs/exploitability.py`, `envs/true_quality.py` | ⬜ after the RL spine |

## L3 Scaling (A3) · L4 Data (A4) — not started

## Decisions locked (`docs/adr/`)
- ADR-0001 tokenizer of record · ADR-0002 GQA in substrate · ADR-0003 sampler parity · ADR-0004 dense v0.1.0
- *(L5 GRPO decisions — Dr.GRPO default, DPO scope, IS truncation — will be logged as ADRs when the RL spine lands.)*
