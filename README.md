# reasoningLLM

> Open, trustworthy verifier-and-environment harness for reasoning-model RLVR —
> **the first open implementation of verifier exploitability as a controlled
> independent variable.**

One vertically-integrated reasoning-model stack, built from the kernel up
(BPE → kernels → scaling → data → RLVR), shipping **one honest finding**: the
gap between what a verifier *rewards* and what a model is *truly worth*, measured
as a controlled experiment.

The through-line metric owned end-to-end, from the token to the reward:

```
true_quality_gap = reward − true_quality      (+ hack_rate, kl_train_infer)
```

See [`../UNIFIED_FRONTIER_PROJECT_SPEC.md`](../UNIFIED_FRONTIER_PROJECT_SPEC.md)
(positioning / job-market map) and
[`../CAPSTONE_AND_STUDY_PLAN.md`](../CAPSTONE_AND_STUDY_PLAN.md) (capstone scope +
VERA study). This repo implements the `src/reasoning_llm/` files those docs name.

---

## The five layers → this repo

The five CS336 assignments are the five **subsystems** of one stack. Data flows
top-to-bottom; each layer feeds a real source file here.

| Layer | CS336 | Subsystem | Source it feeds | Status |
|---|---|---|---|---|
| **L5** RLVR engine | A5 | GRPO/Dr.GRPO · reward · verifier-exploitability | `algos/`, `rewards/`, `envs/exploitability.py`, `envs/true_quality.py` | ⬜ |
| **L4** Data | A4 | reward/verifier-data curation · dedup · contamination | `data/curation.py`, `data/contamination.py` | ⬜ |
| **L3** Scaling | A3 | IsoFLOP machinery repurposed to fit `hack_rate` vs compute | `scaling/hack_rate_fit.py` | ⬜ |
| **L2** Systems | A2 | Triton FA2 · DDP/ZeRO · KV-cache · SGLang client | `rollout/sglang_client.py`, `utils/monitors.py` | 🟡 `monitors.py` ✅ · KV-cache ✅ · rollout seam next |
| **L1** Substrate | A1 | BPE · Transformer · GQA/RoPE/SwiGLU · (MoE) | `tokenizer/model/optim/train/sampling.py` | ✅ complete |

> **Build status** is tracked in [`docs/STATUS.md`](docs/STATUS.md): L1 substrate + `utils/monitors.py`
> + KV-cache done, 53 tests green. **Next: the rollout-client seam** (CPU); Triton FA2 / SGLang
> serving / DDP wait for a GPU box.

## Scope discipline

- **v0.1.0** (frozen 14-day sprint): the engine + a tiny R2E smoke run
  (≤100 problems, 3 of 5 `HardeningLevel`s, 1 seed — proves the plumbing).
- **VERA** (`v0.1.x`, post-sprint): the first *scientific finding* on the engine
  — `true_quality_gap` across model families × hardening levels on a cheap
  math/1.5B env (`envs/math_verifiable.py`).
- **v0.2.0** (additive): MoE×RL collapse, multimodal (`envs/vlm_verifiable.py`).

MoE/VLM axes are tracked as ADR stubs in [`docs/adr/`](docs/adr/), **not** smuggled
into v0.1.0.

## The Day-14 ship test (mechanical, unchanged)

> Does it run, is it public, is the smoke run logged with
> `reward / hack_rate / true_quality_gap / kl_train_infer` across 3
> `HardeningLevel`s?

## Develop

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # CPU core: numpy/pydantic only
# uv pip install -e ".[gpu]"    # on the rented A100/4090: torch/transformers

ruff check src tests && ruff format --check src tests
pyright
pytest -m "not gpu"             # the CPU smoke gate (mirrors CI)
```

## Layout

```
src/reasoning_llm/
├── algos/        L5  advantage estimation (RLOO + sign-robust clip), off-policy (IS+ESS)
├── rewards/      L5  reward composition under the 5 levels, anti-hack detector catalog
├── envs/         L5  the exploitability dial, the true-quality oracle, env interface, tasks
├── rollout/      L2  SGLang rollout client (cheap rollouts = affordable ablation grid)
├── scaling/      L3  hack_rate-vs-compute fitter (IsoFLOP machinery, repurposed)
├── data/         L4  reward-data curation + train↔eval contamination check
└── utils/        L2  the three-KL monitor (incl. kl_train_infer HALT@0.10), metrics, manifests
```

Every stub names its §3 brief, falsifiable prediction, and kill criterion in its
docstring. Fill them in as the corresponding CS336 assignment lands.
