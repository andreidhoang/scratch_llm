# scratch_llm

Substrate for ladders **K1 GEMM · K2 attention · K3 KDA/GDN · S1 serving · T1 training** of the sixty-day plan at
`~/Desktop/ladders` (its `CLAUDE.md` governs every session; this file is repo facts only).
Rewritten 2026-09-03 from a 21 KB constitution; the old text is in `git log`. Nothing binding was lost — the invariants
(predict → measure → mechanism; floors at matched dtype; correctness before perf; one variable) now live in the workspace.

## What lives here
| path | what | ladder |
|---|---|---|
| `src/scratch_llm/kernels/gemm/cuda/mma_sync.cu` (+ `smem_tiled.cu`) | CUDA GEMM ladder — 81.9% of cuBLAS on sm120, element-exact | K1 start |
| `csrc/wgmma_sm90.cu · tcgen05_sm100.cu · fa3_hopper.cu` | Hopper/Blackwell kernels, **never executed** (compile-only); 3 stubs beside them | L0 → K1/K2 |
| `src/scratch_llm/kernels/attention/prefill/fa2.py` | FA2 in Triton — 50% of SDPA on sm120, backward smem overflow | K2 A-R1 |
| `src/scratch_llm/kernels/`, `bench/kernels/` | norms, reduce, roofline benches | K1/K2 |
| `src/scratch_llm/serving/` + `bench/{continuous,cudagraph_decode,speculative,kv_memory}.py` | continuous batching, paged KV, CUDA-graph decode, spec decode | S1 |
| `src/scratch_llm/train.py`, `utils/` (monitors), `data/`, `scaling/` | FSDP/ZeRO substrate, entropy/KL monitors, shards, IsoFLOP fits | T1 |
| `src/scratch_llm/linear_attn.py`, `mastery/{reference,paths,divergence}.py`, `experiments/e001_gate_sweep.py`, `tests/test_e001_regression.py` | GDN reference (explicit-matrix) + oracle (rank-1 apply) + chunked-WY divergence harness | K3 |
| `bench/RESULTS.md`, `src/scratch_llm/bench/{gpu_specs,harness,roofline}.py` | ledger of measured `[FACT]` rows; measurement apparatus | all |

## Commands
```
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"     # CPU
uv pip install -e ".[gpu]"                                               # on a GPU box (torch/transformers)
ruff check src tests && ruff format --check src tests && pyright         # the commit gate (hook runs it)
pytest -m "not gpu"                                                      # CPU suite; mirrors CI
python -m experiments.e001_gate_sweep --self-test | --smoke | --run      # K3 numerics
PYTHONPATH=src python -m scratch_llm.bench.gpu_specs                     # measured device table
```
Measurements go through `~/Desktop/ladders/infra/bench.sh` (clock lock, ncu, provenance, prediction gate), then a
`[FACT]` row in `bench/RESULTS.md` with torch · sm · seed · commit.

## Conventions
- Correctness before perf: oracle + stated tolerance in a test before a benchmark runs. Numbers without a floor at matched dtype/shape do not enter `RESULTS.md`.
- The first version of a rung's kernel core is Huy's; agents write harness, oracle adapters, tests, maps, and fixes after he names the diagnosis (workspace rule 4). No write-guard hooks enforce this — say it instead.
- Known ledger debts to clear in K1/K2: the "134.3% of cuBLAS-proxy" row (`RESULTS.md:531`, mis-baselined) and the FA2 50%/backward rows.
- Hooks: `lint-on-edit` (ruff on save), `green-ci-gate` (lint+types on commit, +tests on push). Nothing blocks writes.

## Frozen — history, not law (read-only until 2026-11-02)
`PLAN.md` · `MASTERY_LEDGER.md` · `AGENTS.md` · `docs/KERNEL_MASTERY_SPEC.md` · `docs/k3/*` · `docs/assignment_guides/*` ·
`performance/*.md` · `deploy/runbooks/*` · `docs/CONTEXT_ENGINEERING.md`. Predictions live in the workspace ledger and in
`tests/test_e001_regression.py::PREDICTED`, not in Markdown. If any frozen file contradicts the workspace plan, the workspace wins.
