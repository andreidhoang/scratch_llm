# scratch_llm — GPU and numerical substrate

Repository context reviewed 2026-09-14. The active plan is
`../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md` **v5**; `../ladders/CLAUDE.md` governs campaign work.
Read those and `AGENTS.md` before selecting a unit. One spine: KDA (Kimi Linear / K3 linear
attention) from the equation to upstream PRs (A1), a Kimi-Linear-class miniature trained on the
existing spine with one honest ablation (A2), then the trusted grader and RL pilot (A3).
K1/K2/S1/T1 material is archive (plan §7), not execution order.

## What lives here

| Path | Substrate and evidence boundary | Ladder (v5: K3 live · K1/K2/S1/T1/L0 archive, plan §7) |
|---|---|---|
| `src/scratch_llm/kernels/gemm/cuda/` | CUDA GEMM learning path; historical sm120 results require matched-baseline audit before reuse | K1 |
| `csrc/gemm/`, `csrc/attention/`, `csrc/persistent/` | Hopper/Blackwell sources; compile-only reports and remaining stubs do not prove runtime correctness | L0/K1/K2 |
| `src/scratch_llm/kernels/attention/prefill/fa2.py` | Triton FA2; historical sm120 forward and backward limitations are recorded in `bench/RESULTS.md` | K2 |
| `src/scratch_llm/kernels/`, `bench/kernels/` | Norms, reductions, dispatch and kernel benchmarks | K1/K2 |
| `src/scratch_llm/serving/`, `bench/` | Serving/cache/graphs/speculation substrate; S1 is killed under v5 — the serving evidence is A1c, the vLLM memory-profiling PR | S1 · archive |
| `src/scratch_llm/train.py`, `src/scratch_llm/training/`, `src/scratch_llm/utils/`, `src/scratch_llm/data/`, `src/scratch_llm/scaling/` | Training, monitoring, data and scaling-study substrate; fixtures are distinct from real distributed runs | T1 |
| `src/scratch_llm/linear_attn.py`, `src/scratch_llm/mastery/`, `experiments/e001_gate_sweep.py`, `tests/test_e001_regression.py` | GDN reference, independent paths and divergence studies | L0/K3 |
| `src/scratch_llm/k3/` | **The v5 spine.** `config.py` + `param_count.py` are done and exact; `core/` is Huy's hand-written lane (agents read-only; templates there are deleted before the v0, not filled); `tests/test_kda_parity_fla.py` is A1a's instrument (FLA naive loaded by path from `../ladders/oss/fla`, needs `einops` in the venv) | K3 · A1/A2 |
| `bench/RESULTS.md`, `src/scratch_llm/bench/` | Historical result index and measurement utilities; retain raw provenance | All |

The active unit under v5 is `K3/kda-v0` (E001's sweep is its step 3; E001's oracle passed
`--self-test` on this Mac on 2026-09-14). The campaign ledger has no measurements; this does not
erase earlier work in this repository. R1/R-R5 remains a planning contract, not an implemented
kernel-RL system. Audit the actual measured code, inputs,
baseline and raw artifacts before reusing local historical percentages. The `134.3% of
cuBLAS-proxy` result has a known baseline debt and is not evidence of beating matched cuBLAS.

## Verification commands and actual hooks

```bash
uv pip install -e ".[dev,scaling]"   # CI also installs CPU torch separately
ruff check src tests bench/kernels experiments/plot_e001.py
ruff format --check src tests bench/kernels experiments/plot_e001.py
pyright
pytest -m "not gpu and not slow and not hole and not drydock" --cov=scratch_llm
```

These are the observed main CI commands, not a claim they were rerun by this documentation edit.
Use the pinned GPU environment from the active rung for silicon work. The broader local test
sets, GPU tests and drydock have distinct prerequisites.

Inspected 2026-09-12: `.claude/settings.json` registers `session-start`, `lint-on-edit` and
`green-ci-gate`. `.git/hooks/pre-commit` and `pre-push` point to `green-ci-gate.sh`, with
`core.hooksPath=.git/hooks`. The edit hook lints/formats Python; the CI hook attempts lint/types
on commits and adds `pytest -m "not gpu and not slow"` on pushes. It skips tools that are absent;
its test markers are not identical to CI's exclusion of `hole` and `drydock`. These are actual
mechanisms with limitations, not proof that every command path or missing dependency is gated.
No `oracle-guard.sh`, `kernel-write-guard.sh` or `.claude/execution-mode` exists here. No hook,
setting, symlink or implementation was changed by this context alignment.

## Ownership and measurement

Huy owns the first kernel core / loss math / memory model, derivations, predictions, tolerance
design and causal diagnosis. Agents write harnesses, independent references, adapters, tests,
maps and reproductions, and fixes after Huy names the diagnosis. `src/scratch_llm/k3/core/`
retains the stricter read-only agent boundary in `src/scratch_llm/k3/HANDCRAFTED.md`; `AGENTS.md` explains it.

Campaign measurements use `../ladders/infra/bench.sh` and the active rung's result/ledger path.
The wrapper attempts clocks, profiling and provenance; runners must implement the declared
correctness, cache, warm-up, timing and sample protocol. The wrapper's commit-path/provenance
defect was fixed on 2026-09-14 (commit + dirty flag recorded for ladders, scratch_llm and
reasoningLLM); T-R2 is killed under v5, so its unevaluated quality gate is moot. Do not turn a
wrapper's successful exit or a raw speedup into a validated quality/performance claim.

## Historical references and remote context

`PLAN.md`, `MASTERY_LEDGER.md`, `docs/KERNEL_MASTERY_SPEC.md`, `docs/k3/*`,
`docs/assignment_guides/*`, `performance/*.md`, `deploy/runbooks/*` and
`docs/CONTEXT_ENGINEERING.md` stay historical. `AGENTS.md`, this file, nested kernel instructions,
README files and live `.claude/agents|commands` route current work to v5. Old FOP numbers,
execution-mode switches, deadlines and automatic push rules do not override the workspace.

`deploy/`, `docs/VASTAI_BOOTSTRAP.md` and `scripts/bootstrap-pod.sh` are legacy helpers/references:
they still point to archived PLAN context and the scripts do not provide the ladders plan/checkouts.
Their blanket container-profiling claims must be replaced by an actual counter-permission probe
on the chosen host, and their historical dependency recipes do not replace active rung pins.
Stage the workspace context and active experiment before use. This documentation edit leaves
those deployment scripts and historical runbooks unchanged.
