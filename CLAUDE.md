# scratch_llm — GPU and numerical substrate

Repository context reviewed 2026-09-21. The active plan is
`../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md` **v6**; `../ladders/CLAUDE.md` governs campaign work and
`../ladders/experiments/KW/map.md` binds the workshop reps to call sites here. KW is the ordered
mechanism-building lane; KDA/A1–A3 is the production spine. Old K1/K2/S1/T1 schedules remain archive,
but their mechanisms are live where v6 routes them through KW into real callers.

**VI:** `scratch_llm` là nơi code được model thật gọi. Bài thử cô lập nằm ở `ladders/experiments/KW`;
chỉ khi có dispatch/call site, parity test và benchmark toàn bước ở đây thì bài học mới thành bằng chứng production.

Mentoring cadence is plan §2.3: an integrated first-principles explanation should lead into an actual source/test/measurement boundary. Do not gate progress on repeated toy arithmetic or copy quizzes; retain those correctness checks inside the harness and preserve Huy's core and prediction ownership.

**VI:** Giải thích đủ để ra quyết định, rồi làm và đo trên code thật. Ví dụ nhỏ chỉ dùng khi tháo gỡ một hiểu lầm cụ thể, không trở thành một vòng phỏng vấn liên tục.

Interview context: `../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md` §2.4 owns the no-AI cold/AI-assisted practice protocol; `../ladders/worklogs/2026-09-21_interview_engineering.md` §§3, 6 gives the original mocks and detailed Anthropic MLE Deep-ML prompt → mechanism → local code → evidence mapping. Deep-ML is a third-party exercise index, not Anthropic's official question bank or a new execution order. Its state/concurrency prompts route to `kv_cache.py` and `serving/` (R10); ML primitives to `model.py`, KDA parity and A2; systems/evals to profiling, the bounded distributed extension and A3. A correct toy challenge is practice evidence only; the production claim still requires the active rung's oracle, caller and matched measurement.

**VI:** Khi gặp câu hỏi phỏng vấn, hãy truy ngược từ invariant tới test và consumer trong repo. Ví dụ KV cache phải đúng cả prefix, phân trang và request scheduling; chỉ giải được bài dictionary TTL chưa chứng minh hệ serving này. Deep-ML không thể thăng cấp một template K3 thành model đã train.

## What lives here

Anthropic lifecycle extension: `../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md` §4.1 and the interview worklog §8 define the same-miniature pretrain→SFT→verifiable-reward update→evaluation→hybrid serving contract. Reuse the existing data, loss, recovery and serving substrates; model-family reconstruction, true held-out splitting and hybrid-state adapters are pending. `faults/capsule.py` provides a dense fp32 exact-recovery instrument, not validated hybrid/AMP recovery. The current `stage_eval` tail comes from the same token array used for pretraining.

**VI:** Gate mới kiểm tra đường nối thật: dataset/tokenizer nào, checkpoint cha nào, token nào được tính loss và state nào được phục hồi. Đây là kế hoạch tích hợp, chưa có claim đã train hay deploy mini-K3.

| Path | Substrate and evidence boundary | v6 lane |
|---|---|---|
| `src/scratch_llm/kernels/gemm/cuda/` | CUDA GEMM learning path; historical sm120 results require matched-baseline audit before reuse | KW/R04–R05 |
| `csrc/gemm/`, `csrc/attention/`, `csrc/persistent/` | Hopper/Blackwell sources; compile-only reports and remaining stubs do not prove runtime correctness | KW/R08–R11 |
| `src/scratch_llm/kernels/attention/prefill/fa2.py` | Triton FA2; historical sm120 forward and backward limitations are recorded in `bench/RESULTS.md` | KW/R07 |
| `src/scratch_llm/kernels/`, `bench/kernels/` | Norms, reductions, dispatch and reusable kernel benchmarks | KW/R03–R13 |
| `src/scratch_llm/serving/`, `bench/` | Serving/cache/graphs/speculation substrate; the old S1 schedule is archived — live serving work is KW/R10 and A1c | KW/R10 · A1c |
| `src/scratch_llm/train.py`, `src/scratch_llm/training/`, `src/scratch_llm/utils/`, `src/scratch_llm/data/`, `src/scratch_llm/scaling/` | Training, monitoring, data and scaling-study substrate; fixtures are distinct from real distributed runs | T1 |
| `src/scratch_llm/linear_attn.py`, `src/scratch_llm/mastery/`, `experiments/e001_gate_sweep.py`, `tests/test_e001_regression.py` | GDN reference, independent paths and divergence studies | L0/K3 |
| `src/scratch_llm/k3/` | **The product spine.** Live state ledger = the `k3/__init__.py` docstring (STATUS, NEXT, Huy-owned decisions, doc debt). As of 2026-09-22 the whole text model is built and gated against Moonshot's HF reference (`../ladders/oss/kimi_k3_hf/`): `core/` (situ, kda, gated_mla, latent_moe + Quantile Balancing, attn_res, norm), `model.py` (whole model ≡ HF transcription fp64), `muon.py` (Per-Head Muon + MuonClip), `checkpoint.py` (HF names + MXFP4), and `train()`/speedrun pretrain wiring. `run_speedrun(model_family="k3_mini")` runs every stage end to end (decode through `HybridState`). Open: `HUY_V0_REL_TOL` (KDA 5/4), the GPU A2 run, vision (reading-only until Huy opens it). No file-ownership boundary; the gate is `tests/test_k3_*.py` + `tests/test_kda_parity_fla.py` | K3 · A1/A2 · KW/R14 |
| `bench/RESULTS.md`, `src/scratch_llm/bench/` | Historical result index and measurement utilities; retain raw provenance | All |

mini-K3 is now a runnable, CPU-gated model (forward, decode, `train()`, checkpoints, the full speedrun chain), but no run of any size has produced a quality claim: that needs the A2 comparison against a matched dense control on a GPU host. Read the `k3/__init__.py` ledger before starting K3 work. The A2 integration/baseline and A3 grader remain gates to build, not claims earned by interview-prep exercises.

**VI:** Đếm đúng tham số mới kiểm tra cấu hình. Muốn nói đã rebuild, cần forward/train/decode chạy qua core thật, so sánh dense control cùng ngân sách, rồi giữ raw evidence; một mock interview không thay các phép thử này.

The planned next unit under v6 is CPU `K3/kda-v0`, followed by GPU `KW/R01-roofline` on an accepted NVIDIA host; the owner-controlled
`../ladders/experiments/CURRENT` still points to `L0/E001` until Huy changes it. Run `make -C ../ladders context`
to see the pointer, plan, folders and provider gate together. This Mac supports KDA derivation/parity, not NVIDIA roofs or counter claims. E001's oracle passed `--self-test` on this Mac
on 2026-09-14. The campaign ledger has no measurements; this does not
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

Huy owns derivations, predictions, tolerance design and causal diagnosis. Agents write harnesses,
independent references, adapters, tests, maps and reproductions, and fixes after Huy names the
diagnosis. The `src/scratch_llm/k3/core/` file-ownership boundary (formerly
`src/scratch_llm/k3/HANDCRAFTED.md`) was removed 2026-09-22 by explicit request: anyone — Huy or
an agent — may implement any K3 module now. The evidence bar is unchanged and is the actual gate:
`tests/test_kda_parity_fla.py` at 9/9, the three-path equivalence (chunkwise ≡ recurrent ≡
float64 reference), and a matched dense control at equal parameter budget before any A2 claim.
`AGENTS.md` carries the same update.

Campaign measurements use `../ladders/infra/bench.sh` and land under
`../ladders/experiments/<L>/<R>/results/<timestamp>/`; the ledger is only the cross-run claim index.
The wrapper attempts clocks, profiling and provenance; runners must implement the declared
correctness, cache, warm-up, timing and sample protocol. The wrapper's commit-path/provenance
defect was fixed on 2026-09-14 (commit + dirty flag recorded for ladders, scratch_llm and
reasoningLLM); the old T-R2 rung is not live, so its unevaluated quality gate is moot. Do not turn a
wrapper's successful exit or a raw speedup into a validated quality/performance claim.

## Historical references and remote context

`PLAN.md`, `MASTERY_LEDGER.md`, `docs/KERNEL_MASTERY_SPEC.md`, `docs/k3/*`,
`performance/*.md`, `deploy/runbooks/*` and
`docs/CONTEXT_ENGINEERING.md` stay historical. `AGENTS.md`, this file, nested kernel instructions,
README files and live `.claude/agents|commands` route current work to v6. Old FOP numbers,
execution-mode switches, deadlines and automatic push rules do not override the workspace.

`deploy/`, `docs/VASTAI_BOOTSTRAP.md` and `scripts/bootstrap-pod.sh` are legacy helpers/references:
they still point to archived PLAN context and the scripts do not provide the ladders plan/checkouts.
Their blanket container-profiling claims must be replaced by an actual counter-permission probe
on the chosen host, and their historical dependency recipes do not replace active rung pins.
Stage the workspace context and active experiment before use. This documentation edit leaves
those deployment scripts and historical runbooks unchanged.
