# scratch_llm

> A from-scratch, production-grade implementation of the **CS336** stack
> ("Language Modeling from Scratch", Stanford) — every layer owned end to end,
> from the byte to the RL update.

This is the **mastery vehicle**: built by hand to the engineering standard a frontier lab
screens for, not glued together from libraries. The official course — lecture code plus the
five assignment scaffolds with their `tests/adapters.py` — lives in [`../lectures/`](../lectures/)
and is the **spec + test oracle**: the PDFs define each deliverable, the adapter tests verify
your implementation is correct.

## The five assignments → this repo

| CS336 assignment | What you build (load-bearing core) | Source | Status |
|---|---|---|---|
| **A1** Basics | BPE · Transformer (RMSNorm·RoPE·SwiGLU·GQA) · AdamW · training loop · sampling | `tokenizer/model/moe/optim/train/sampling.py` | ✅ |
| **A2** Systems | FlashAttention-2 (Triton) + roofline · KV-cache · DDP/ZeRO-1/FSDP · monitors | `kernels/`, `rollout/`, `utils/` | 🟡 FA2 + KV-cache + monitors + rollout done; DDP/ZeRO/FSDP next |
| **A3** Scaling | IsoFLOP / Chinchilla fits (compute-optimal N, D) | `scaling/` | ⬜ |
| **A4** Data | filter → quality-classify → exact + MinHash/LSH dedup | `data/` | ⬜ |
| **A5** Alignment | SFT · Expert Iteration · GRPO/Dr.GRPO · DPO | `algos/`, `rewards/`, `envs/` | ⬜ |

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) for the A1→A5 build spine and
[`docs/assignment_guides/`](docs/assignment_guides/) for the per-assignment guides — every
deliverable tagged **LOAD-BEARING / COURSE-ROTE / SKIP** and mapped to a source file. Live status:
[`docs/STATUS.md`](docs/STATUS.md). The 2026 **frontier-practice layer** — per-pillar modern-default
upgrades, opt-in build labs, and interview-awareness items (fact-checked) — is in
[`docs/FRONTIER_PRACTICE_2026.md`](docs/FRONTIER_PRACTICE_2026.md). The pipeline-level **end-to-end
training plan** (2026-07-30, 9-angle 2026 research pass — S0→S8, data → tokenizer → pretrain → d20 →
RL → serve/eval) is [`docs/FRONTIER_2026_END_TO_END_PLAN.md`](docs/FRONTIER_2026_END_TO_END_PLAN.md) —
read it first; rung-level specs stay in `docs/FRONTIER_2026_TASKSPEC.md` / `docs/FRONTIER_2026_ABLATIONS.md`.

**Capstone — DELTA** (the barbell *spike*, sitting on the A2/A5 base): a fused **GatedDeltaNet-2
decode-step** kernel (target ≥85% of the H100 memory roofline; the "erase/write decoupling is free at
decode" thesis). Design + dated 4-week plan live in the workspace root —
[`../DELTA.md`](../DELTA.md) (merged design RFC + 4-week plan) — and are tracked in
[`docs/STATUS.md`](docs/STATUS.md) and [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) §7.

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
└── envs/        A5  verifiable task environments + the env/grader protocol
```
