# reasoningLLM — Operating Constitution

> **What this repo is.** A clean, independent implementation of the reasoningLLM stack,
> built to production engineering standards. It is **separate from the ship repo** at
> `../../reasoningLLM` (mature, the public v0.1.0 artifact); do not copy code between them —
> this is its own implementation, not a transcription.
>
> **Harness manual:** [`docs/CONTEXT_ENGINEERING.md`](docs/CONTEXT_ENGINEERING.md) explains
> *why every file under `.claude/` exists and on which context lever*. Read it once.

## The one number (the through-line)

```
true_quality_gap = reward − true_quality        (+ hack_rate, kl_train_infer)
```
Owned end-to-end, from the token to the reward. Every subsystem exists to measure or
protect this quantity. If a change does not move or defend the gap, question it.

## Frozen v0.1.0 scope (do not relitigate — log an ADR in `docs/adr/` instead)

- **Ship:** the engine + a tiny smoke run (≤100 problems, 3 of 5 `HardeningLevel`s, 1 seed —
  proves the plumbing). Day-14 test: *does it run, is it public, is the smoke run logged with
  `reward / hack_rate / true_quality_gap / kl_train_infer` across 3 `HardeningLevel`s?*
- **VERA** (`v0.1.x`, post-sprint): the first scientific finding on the engine (math/1.5B).
- **v0.2.0** (additive): MoE×RL collapse, multimodal. ADR stubs only — **never smuggle into v0.1.0.**

## The five layers → CS336 assignment → source file

| Layer | A# | Subsystem | Source it feeds |
|---|---|---|---|
| **L5** RLVR engine | A5 | GRPO/Dr.GRPO · reward · verifier-exploitability | `algos/`, `rewards/`, `envs/exploitability.py`, `envs/true_quality.py` |
| **L4** Data | A4 | reward/verifier-data curation · dedup · contamination | `data/curation.py`, `data/contamination.py` |
| **L3** Scaling | A3 | IsoFLOP machinery repurposed to fit `hack_rate` vs compute | `scaling/hack_rate_fit.py` |
| **L2** Systems | A2 | Triton FA2 · DDP/ZeRO · KV-cache · SGLang client | `rollout/sglang_client.py`, `utils/monitors.py` |
| **L1** Substrate | A1 | BPE · Transformer · GQA/RoPE/SwiGLU · (MoE) | the policy served in rollouts |

Module layout: `src/reasoning_llm/{algos,rewards,envs,rollout,scaling,data,utils}/` (plus
flat `tokenizer.py`, `model.py`, `optim.py`, `train.py`, `sampling.py` for the L1 substrate).

**Build status:** L1 substrate ✅ · `utils/monitors.py` ✅ · KV-cache ✅ · **rollout-client seam next**
(CPU; FA2/SGLang/DDP scheduled on a rented vast.ai GPU — not skipped). See [`docs/STATUS.md`](docs/STATUS.md).

## Engineering disciplines (these are how labs silently screen)

1. **Loss-at-init check** — a fresh LM's cross-entropy must be ≈ `log(vocab_size)`. If not, the
   head/embedding/masking is wrong. Assert it in the model's first test.
2. **Overfit-one-batch** — before any real training, drive train loss to ~0 on a single batch.
   If it can't, the optimizer/data/loss wiring is broken, not the data.
3. **Fixed-seed reproducibility** — seed python/numpy/torch; a re-run reproduces the metric.
4. **Mandatory RL logging** (any RL run; absence = the run is uninterpretable):
   log the **three KL divergences separately** — `KL(current‖ref)`, `KL(current‖old)`,
   **`kl_train_infer = KL(train‖infer)`** — plus IS-ratio histograms, reward distribution stats,
   and **length stats** (catches verbosity reward-hacking). `kl_train_infer` HALT threshold = **0.10**.
5. **Predict before you run** — write the falsifiable number first; it is the debugging anchor.

## Follow the plan — never skip a step

Build the plan's steps **in order and in full**. A step that needs hardware is **not** a step to
skip or silently drop — it is a step to *resource*. If a step requires a GPU (Triton/FA2 kernels,
DDP/ZeRO/FSDP, real SGLang serving, the TinyZero/VERA RL runs), **rent one on vast.ai** (use the
`vastai` skill) and complete it. "GPU-deferred" means *scheduled on rented hardware*, never
*abandoned*. Order CPU-buildable steps first when that is cheaper, but CPU-vs-GPU is never an
excuse to drop scope. The only legitimate non-builds are items the plan itself tags **SKIP** (e.g.
leaderboards) or **v0.2.0** (MoE/multimodal, per ADR-0004/0005). Everything else gets built.

## Green-CI rule (enforced, not requested)

A commit is shipped only if green: `ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`.
`.claude/hooks/green-ci-gate.sh` enforces this as a Claude Code `PreToolUse` hook — a red `git commit`
issued **through Claude Code** is blocked (exit 2), not merely discouraged. For enforcement on *every*
commit path (external terminal included), symlink the same script as a git-native hook at `git init`:
`ln -s ../../.claude/hooks/green-ci-gate.sh .git/hooks/pre-commit`. Commit messages are conventional and
scoped: `<area>: <imperative>` (e.g. `tokenizer: add byte-level BPE trainer`).

## Build / test

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"          # CPU core: numpy/pydantic
ruff check src tests && ruff format --check src tests
pyright
pytest -m "not gpu"                  # the CPU smoke gate (mirrors CI)
```

## Where things live (read on-demand, not auto-loaded)

| Need | Path |
|---|---|
| Sprint cockpit (tracker, `/morning`, `/ship`, gates) | `../../daily/` |
| **Master plan** — §2 stack · §3 per-layer briefs (add-ons + kill criteria) · §5 role map | `../UNIFIED_FRONTIER_PROJECT_SPEC.md` |
| **Per-layer build guides** — the load-bearing 20% per assignment → `src/reasoning_llm/` (start at `INDEX.md`) | `docs/assignment_guides/` |
| Capstone scope + VERA study | `../CAPSTONE_AND_STUDY_PLAN.md` |
| The public ship repo (do not copy from) | `../../reasoningLLM/` |
| This repo's harness, explained | `docs/CONTEXT_ENGINEERING.md` |
| **Build status** — what's built / tested / green (single source of truth) | `docs/STATUS.md` |
| **Next-implementation spec** — L2 KV-cache design | `docs/design/L2_kv_cache_SPEC.md` |

## Implementation rules

- This is its own clean implementation — **do not copy code from `../../reasoningLLM`** (the ship repo).
- Every module's docstring names its §3 brief, falsifiable prediction, and kill criterion — the engineering
  intent is documented in the code, not just in the guides.
- Land changes test-first where practical: write the invariant (loss-at-init, decode round-trip,
  causal-no-leak, overfit-one-batch) as a test, then make it pass.
