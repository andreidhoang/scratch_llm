# scratch_llm

> A from-scratch, production-grade implementation of the **CS336** stack
> ("Language Modeling from Scratch", Stanford) — every layer owned end to end,
> from the byte to the RL update.

## Measured results

All numbers below are `[FACT]` rows from [`bench/RESULTS.md`](bench/RESULTS.md) — measured under
`cuda.synchronize`, fixed seed, with warm-ups, on one consumer card. The discipline is
**predict the number and the roofline bound first, then measure, then log the gap and the root cause.**
Unmeasured expectations are marked `[INFERENCE]` and are kept out of the ledger.

**Hardware baseline —** RTX PRO 4000 Blackwell (sm120): **72 TF/s** bf16 · **0.55 TB/s** HBM ·
ridge ≈ **130 FLOP/byte** (measured, not from a spec sheet).

| Artifact | Measured | Bound |
|---|---|---|
| CUDA GEMM ladder (naive → WMMA → mma.sync + XOR swizzle) | **4.1% → 38.9% → 81.9% of cuBLAS**, rel err 6.6e-6 | compute |
| Triton tiled GEMM, autotuned @4096³ | **134.3% of cuBLAS-proxy** (101.9 TF/s) | compute |
| FlashAttention-2 (Triton) fwd @ seq 4k | **50.0% of SDPA**; **44× leaner peak memory @ 8K** | memory |
| Decode @B=1: eager → `torch.compile` → CUDA graphs | **51 → 173 → 253 tok/s** = **16% → 53% → 77%** of the memory roofline | overhead → memory |
| CUDA-graph decode step time | **15.38 → 3.96 ms (−74.3%)**; ~955 kernel launches/token collapsed | overhead |
| Fused Triton paged decode | **5.90 ms/step · ×3.52 vs wave-dense · 3,528 tok/s agg** | memory |
| Continuous batching vs static wave (heavy-tail trace) | **2.30× wall · 2.93× by step count** | memory |
| PagedKVCache | fragmentation **5.0%**, capacity **×9.3** | memory |
| Batched decode scaling | agg(B=32)/agg(B=1) = **24.9×**; peak **12,220 tok/s @B=256**; roofline crosses memory→compute at **B≈128** | memory→compute |
| Speculative decode (n-gram draft) | **×1.21–1.39 wall**, token-exact | latency |

The decode row is the one to read closely: the workload is memory-bound in *theory* (arithmetic
intensity ≈ 1, ~130× below the ridge), but the eager run sat at 16% of the wall because it was
**launch-overhead-bound, not memory-saturated** — ~955 kernel launches per token. Killing the launch
tax (compile → CUDA graphs) is what actually walked it to 77%. Diagnosing *which* wall you are
against, rather than assuming, is the point of the whole ledger.

### Honest limitations

Stated up front rather than buried:

- **No kernel here has run on datacenter silicon.** Every number above is sm120 (consumer Blackwell).
- **`csrc/` is partly aspirational:** of 6 `.cu` files, 3 (`wgmma_sm90`, `tcgen05_sm100`, `fa3_hopper`)
  are **compile-verified only** — runtime correctness is deferred to a rental that has not happened —
  and 3 (`fp8_gemm_sm90`, `stream_k_sm90`, `persistent_gemv_sm90`) are **stubs** that raise. The
  working kernel ladder is Triton + the CUDA GEMM ladder.
- **The d20 speedrun has never been run.** Its runbook was retired 31/08 with the rest of the
  frontier-front scope; it documented its own blockers and lives in `git log`.
- **The S3 scaling sweep failed its own gate** — R² 0.77/0.83 against a pre-registered ≥0.98 — so the
  fit was **rejected** and no extrapolation is quoted. A failed pre-registered gate is logged, not hidden.

This is the **mastery vehicle**: built by hand to the engineering standard a frontier lab
screens for, not glued together from libraries. The official course — lecture code plus the
five assignment scaffolds with their `tests/adapters.py` — lives in [`../lectures/`](../lectures/)
and is the **spec + test oracle**: the PDFs define each deliverable, the adapter tests verify
your implementation is correct.

## The five assignments → this repo

| CS336 assignment | What you build (load-bearing core) | Source | Status |
|---|---|---|---|
| **A1** Basics | BPE · Transformer (RMSNorm·RoPE·SwiGLU·GQA) · AdamW · training loop · sampling | `tokenizer/model/moe/optim/train/sampling.py` | ✅ |
| **A2** Systems | FlashAttention-2 (Triton) + roofline · KV-cache · DDP/ZeRO-1/FSDP · monitors | `kernels/`, `rollout/`, `utils/` | ✅ (distributed half shipped 2026-07-03; real SGLang serving rental-gated) |
| **A3** Scaling | IsoFLOP / Chinchilla fits (compute-optimal N, D) | `scaling/` | ✅ (Stanford-API leaderboard blocked-external) |
| **A4** Data | filter → quality-classify → exact + MinHash/LSH dedup | `data/` | ✅ |
| **A5** Alignment | SFT · Expert Iteration · GRPO/Dr.GRPO · DPO | `algos/`, `rewards/`, `envs/` | ✅ (graded GPU runs rental-gated) |

See [`docs/assignment_guides/`](docs/assignment_guides/) for the per-assignment guides — every
deliverable tagged **LOAD-BEARING / COURSE-ROTE / SKIP** and mapped to a source file. **The plan and
the live state are [`PLAN.md`](PLAN.md); measured numbers are [`bench/RESULTS.md`](bench/RESULTS.md).**
The 2026 planning generations that used to be linked here (`FRONTIER_2026_*`, `IMPLEMENTATION_PLAN`,
`STATUS`) were collapsed into PLAN.md and deleted on 2026-08-31 — `git log` holds them.
The K3 track (chartered 2026-07-31): **K3** — build & host **Kimi K3** from scratch, reusing this
repo's substrate (GDN→KDA, MLA→Gated MLA-NoPE, MoE→Stable LatentMoE). Roadmap + verified-facts ledger:
[`docs/k3/ROADMAP.md`](docs/k3/ROADMAP.md) + [`docs/k3/FACTS.md`](docs/k3/FACTS.md) (K3 tech report
arXiv:2607.24653 is the spec of record).

**Capstone — DELTA** (the barbell *spike*, sitting on the A2/A5 base): a fused **GatedDeltaNet-2
decode-step** kernel (target ≥85% of the H100 memory roofline; the "erase/write decoupling is free at
decode" thesis). The design RFC was retired 31/08 — the field closed the original gap (FlashInfer
shipped production GDN decode) and its own banner said Part II must be re-derived. What survives is
the load-bearing summary, kept inline as *extension material* in `docs/KERNEL_MASTERY_SPEC.md` §6:
hypothesis ≥1.5× at decode B≤32 vs a composed baseline, four named baselines, evals locked first,
kill at <1.15×.
(K3 roadmap **K10.2** re-aims the co-designed kernel at **KDA** — per-channel decay — building on
DELTA's GDN-2 base; see `docs/k3/ROADMAP.md`.)

## Engineering disciplines (baked into the tests)

loss-at-init ≈ `log(vocab)` · overfit-one-batch · fixed-seed reproducibility · mandatory RL logging
(entropy + KL divergences + reward/length stats) · predict-before-you-run. These *are* the hiring
signal — clean, reproducible code that you can defend from first principles.

## Develop

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # CPU core: numpy/pydantic/pyyaml/regex
# uv pip install -e ".[gpu]"    # on a rented GPU: torch/transformers
ruff check src tests && ruff format --check src tests
pyright
pytest -m "not gpu"             # the CPU gate (mirrors CI)
```

## Layout

```
src/scratch_llm/
├── tokenizer.py model.py moe.py optim.py train.py sampling.py   A1  substrate
├── kernels/     A2  FlashAttention-2 (oracle + Triton) + roofline
├── rollout/     A2  serving seam (LocalBackend now; SGLang client)
├── utils/       A2  training monitors (entropy, KLs, IS/ESS), seeding
├── scaling/     A3  IsoFLOP / Chinchilla fitter
├── data/        A4  filtering, dedup, quality classification
├── algos/       A5  SFT, Expert Iteration, GRPO/Dr.GRPO, DPO
├── rewards/     A5  verifiable-reward grading (r1-zero: format + answer)
├── envs/        A5  verifiable task environments + the env/grader protocol
├── k3/          K3  build & host Kimi K3 from scratch (KDA · Gated MLA-NoPE · LatentMoE)
├── serving/     serving engines (continuous batching · paged KV · CUDA-graph decode)
├── bench/       roofline/measurement apparatus (gpu_specs · harness · ledger)
├── eval/        eval report card + optimizer-race harness
├── quant/       quantization (INT8/INT4 · NVFP4/MXFP4 · FP8-KV)
├── linear_attn.py  F10 Gated-DeltaNet (KDA base for the K3 track)
├── mtp.py       F2 multi-token-prediction draft head
├── chat_cli.py  A6 chat REPL over the serving path
├── speedrun.py  nanochat-style end-to-end spine (train → SFT → chat preview)
├── scaling/s3_sweep.py  S3 scaling-sweep driver (plan/run/fit)
└── data/shards.py       real-corpus shard loading (FineWeb-EDU slice)
```
