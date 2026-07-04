# scratch_llm — CS336 From-Scratch · Operating Constitution

> **What this repo is.** A from-scratch, production-grade implementation of the **CS336
> (Stanford, "Language Modeling from Scratch")** stack — every layer owned end to end, from the
> byte to the RL update. It is the *mastery vehicle*: you build it yourself, to engineering
> standards a frontier lab screens for. The **official course materials live in `../lectures/`**
> (lecture code + the five assignment scaffolds with their `tests/adapters.py`) — that is the
> **spec and the test oracle**: the PDFs define each deliverable, the adapter tests verify your
> implementation is correct. This repo is *your* implementation, not a transcription.
>
> **Harness manual:** [`docs/CONTEXT_ENGINEERING.md`](docs/CONTEXT_ENGINEERING.md) explains why
> every file under `.claude/` exists. Read it once.

<!-- FOP:start (generated from cs336/FEINBERG_INTERVIEW_MAP.md — edit the 7 below, keep the markers) -->
## Frontier Operating Principles (FOP)

> Distilled from `../FEINBERG_INTERVIEW_MAP.md` (Vlad Feinberg / GDM). Load-bearing for every Claude agent and human session in this repo. Throughline: frontier labs hire on shipped, defensible artifacts — taste is necessary, **execution is the gate**.

1. **Execution > analysis.** Ship beats plan. No new doc without a same-day commit hash. "Close a node, then delete the urge to write the next doc." When the doc-to-code ratio climbs, stop writing and resolve a stochastic node.
2. **Spec-with-falsifiers.** Before non-trivial code, state pre-registered, falsifiable predictions + pre-committed kill/abandon thresholds (the DELTA P1–P7 shape).
3. **Roofline-first / predict-the-number.** Predict the bound (comm vs flop vs memory) and the number before the run. Kernel & perf DoD is a *profile*, not a green test. Name the next unmodeled constraint a coding agent misses (launch / latch / occupancy / bank-conflict).
4. **Claims honesty.** Label `[FACT]` / `[INFERENCE]` / `[UNCERTAIN]`. "Implemented" ≠ "measured" — only a measured/profiled run is a result. Verify against primary sources; never overclaim.
5. **Research-as-MDP / taste.** One active experiment at a time. Pull the high-variance signal node (A5 convergence, DELTA roofline), not deterministic scaffolding. Ruthless kill criteria. Subtract-before-add.
6. **AI-mode boundary — governed by the execution-mode switch ([ADR-0013](docs/adr/ADR-0013-execution-mode-full-delegation.md)).** The taxonomy stands: Mode 1 delegate (plumbing) · Mode 2 human-leads-AI-assists (research-critical) · Mode 3 AI-OFF (the RL-math + kernel reps the interview tests). *Which* contract applies is set by `.claude/execution-mode`: **`delegate`** (current, 2026-07-03) — agents implement everything end-to-end, kernel bodies included; mastery is post-hoc via the `docs/learning/INDEX.md` study queue; the teach-back gate never blocks the build. **`learn`** — the historical contract: agents MUST refuse to write Mode-3 targets from scratch — offer failing tests, a Socratic critique, or a post-hoc review instead. All other FOPs (pre-registration, roofline-first, claims honesty, green-CI, adversarial review) apply identically in both modes.
7. **Citation-tree mastery.** Traverse to the non-redundant gap; reuse before re-deriving; don't rebuild owned work.
<!-- FOP:end -->

## ⚡ Three active fronts (2026-07-04) — pick your lane before building

> Execution mode is **`delegate`** (`.claude/execution-mode`,
> [ADR-0013](docs/adr/ADR-0013-execution-mode-full-delegation.md)): agents implement everything
> end-to-end; mastery is post-hoc via the study queue / mastery-debt ledger. Two agent fronts run
> concurrently on this checkout ([ADR-0014](docs/adr/ADR-0014-cs336-main-track-delivery-sprint.md)):
>
> - **Perf front** — the `performance/` curriculum. **✅ ALL sm120-runnable rungs A1–A6 COMPLETE
>   (2026-07-04):** A1 R0–R4.6 serving · A2 R0–R6 kernels · A3 R0–R2 tensor cores (CUDA) · A4 R0–R3+bwd
>   flash attn · A5 R0–R4+§4.3 quant · A6 TP/1F1B/EP-MoE/MFU (gloo). Design notes A1–A7, H100/B200/8×H200
>   runbooks + compile-verified ISA kernels (`performance/rental/kernels/`) + WGMMA PTX artifact done.
>   Mastery roadmap: [`docs/learning/roadmap/`](docs/learning/roadmap/README.md). Only the 3 rental DAYS
>   remain (hardware-gated). Node pointer `performance/PERF_PLAN.md`. Zone: `performance/`,
>   `src/scratch_llm/{serving,kernels,quant}/`, `mla.py`, `utils/{tp_mlp,pipeline_schedule,ep_moe,mfu}.py`,
>   perf sections of `bench/RESULTS.md`, `docs/learning/roadmap/`.
> - **Main-track front** — the CS336 A2→A5 finish, plan =
>   [`docs/EXECUTION_SPEC_CS336_FINISH.md`](docs/EXECUTION_SPEC_CS336_FINISH.md) (task DAG W1–W11,
>   per-node DoD, rental runbooks for >24 GB work). Zone: `utils/` (distributed), `scaling/`,
>   `data/`, `algos/`, `rewards/`, `envs/` + main-track docs/tests. The 2026-06-30 "perf first"
>   ordering mandate is dissolved — the main track no longer queues behind perf.
> - **Close-the-loop / frontier-ablation front** (2026-07-04, [ADR-0018](docs/adr/ADR-0018-close-the-loop-nanochat-front.md)) —
>   adopt nanochat's `speedrun.sh` spine + report card to train a real *talking* model, then run the
>   EV-ranked iso-FLOP ablation study (F1 MuonAdamW → F9). Spec + DAG:
>   [`docs/FRONTIER_2026_ABLATIONS.md`](docs/FRONTIER_2026_ABLATIONS.md); node pointer **F1**. Zone:
>   `src/scratch_llm/{eval,}`, `scripts/`, `optim.py`, `train.py`, additive `model.py`/`mla.py`/`moe.py`,
>   the F-rung sections of `bench/RESULTS.md`.
>
> Shared files (`CLAUDE.md`, `docs/STATUS.md`, `pyproject.toml`, `bench/RESULTS.md`): pull-rebase
> before every commit, additive edits only, never `git add -A`.

## Orient before you build — the standing protocol (every agent, every session)

> Load-bearing for **every** Claude agent and human session. You operate to the standard of a **lead
> senior AI-performance engineer & researcher at a frontier lab**: the bar is not "finish the task" but
> "advance the research program *correctly*, with the context and rigor a frontier RE brings." That bar
> is procedural — **orient first, then build.** This is the FOP applied *as a workflow* (it is why
> `/standup` and `/next` exist); skipping it is how an agent silently re-does closed work, trusts a stale
> doc over a fresh commit, or pulls a low-signal node.

Before any engineering work — a new task, a resumed thread, or a fresh `/clear` — an agent MUST, in order:

1. **Read the state, don't assume it.** `git log --oneline -15` (what just shipped) + the current-node
   pointer (`performance/PERF_PLAN.md` for the perf curriculum, else `docs/STATUS.md`) + the live ledger
   (`bench/RESULTS.md`) + the one spec/guide governing the active node. The SessionStart hook surfaces
   the latest commits + node as the *starting* context — then read deeper; never trust a doc line that a
   later commit has moved.
2. **Reconstruct the thread.** State, in your own words, what the last commits established, what is
   **measured vs merely implemented** (FOP-4), and the active node's DoD / kill criteria — *before*
   writing code. Never restart work a recent commit already closed.
3. **Reason the next task from that context** (don't pattern-match a default): pick the ONE highest-EV
   node (FOP-5), pre-register its falsifiable prediction + kill criterion (FOP-2/3), build test-first
   (green-CI). If plan-sequence and highest-signal disagree, name the fork — the ordering call is the
   human's (Mode-2).
4. **Carry context forward, durably.** Every result updates `bench/RESULTS.md`, the node pointer, and —
   when it changes cross-session truth — the auto-memory. The next session must orient from what you
   *left*, not from what anyone remembers.

**Analyze → reconstruct → reason → build.** Depth of orientation scales with the task: a one-line fix
needs a glance at `git log`; resuming a curriculum thread needs the full sweep above.

## The organizing principle

> **Own every layer of a language model — byte → BPE → Transformer → systems → scaling → data →
> RL post-training — to production engineering standard, and be able to whiteboard and defend each
> piece cold in a frontier-lab interview.**

The engineering disciplines below *are* the hiring signal: weak engineering is the most common
silent rejection of otherwise-strong candidates. Mastery is the means; a clean, public, green-CI
repo that you can explain from first principles is the end.

## How we build — forced first-principles mastery × relentless execution

Two mandates, every session. The Navigator (the user) is leveling to senior frontier-RE; treat
every load-bearing concept as something they must own, not just ship.

> **Execution-mode note (ADR-0013, 2026-07-03).** In `delegate` mode the "master understanding"
> protocol below is **pull-based, not gating**: agents build and ship without waiting for
> teach-backs; every shipped rung is appended to the `docs/learning/INDEX.md` study queue and the
> protocol runs later, on demand (`/master`). In `learn` mode it gates, as written.

**Master understanding (forced).** For each load-bearing concept, before/while we build it:
1. **First principles** — derive the mechanism (don't assert it): the problem, the math, why this design.
2. **Visualize, three lenses** — the **tensor shapes** through the op (+ the exact `src/scratch_llm/…` lines), the **system/data-flow** (ASCII), and a **tiny worked numeric example** (hand-traced small numbers).
3. **Predict-before-run** — the user writes the falsifiable number/shape first (debugging *and* learning anchor).
4. **Build test-first** — the invariant as a test, then make it pass, green-CI.
5. **Teach-back — the gate** — the user explains it back in their own words + modifies-and-predicts one variation. **We do not advance to the next concept until they can teach the current one back.** This is the *force*: a concept gate, **not** a commit gate — no F-IDs, no ceremony (that was retired; understanding is the working *mode*, not paperwork).
6. **Connect to frontier** — tie it to `docs/FRONTIER_PRACTICE_2026.md` (what 2026 labs do / the interview question).

The full generic learning protocol (pair-programming, Socratic, question-everything) lives in the
global `~/.claude/CLAUDE.md` and loads every turn — the above is only its **repo binding** to the
A1→A5 build, not a duplicate.

**Execute relentlessly.** Always know the current pillar + the next load-bearing step (`docs/STATUS.md`
+ `docs/IMPLEMENTATION_PLAN.md`); never leave the build idle; resource GPU steps (vastai), don't drop
them. Use **`/master <concept>`** to go deep on one concept and **`/next`** to orient + start the next
step test-first. Treat each *optional* path (a `🔵` build-lab, an ablation) as a **research-taste** call —
an a-priori EV(success ÷ time) bet on a stochastic DAG whose nodes fail, gated by a predict-before-run
invariant and a kill-criterion (`docs/CONTEXT_ENGINEERING.md` §3.9).

## Scope — master CS336, from first principles, scratch → production

Build the five assignments A1 → A5 to the *load-bearing 20%* (the per-assignment
guides tag every deliverable LOAD-BEARING / COURSE-ROTE / SKIP). **Build-order is numeric (each layer
builds on the last); ship-order is EV-ranked.** With A1 ✅ done, the next artifact to *ship* is the
**A5 RL "aha"** (scarcest 2026 cluster, highest-EV), with the A2 systems finish + OSS Rung-1 in parallel,
then DELTA on that base — the one canonical sequence is **`../STRATEGY.md` §8** (it wins over any other
doc's ordering). "Production" means: green CI,
tests as executable spec, design docs and ADRs for non-obvious decisions, reproducible runs.
GPU steps are **developed on the standing GPU** (rented out only for what this card can't do), never silently dropped (see "Develop on the GPU").

## The five assignments → layer → source files

| CS336 assignment | Layer | What you build (the load-bearing core) | Lives in |
|---|---|---|---|
| **A1** Basics | Substrate | byte-level BPE · Transformer (RMSNorm·RoPE·SwiGLU·MHA) · cross-entropy · AdamW · cosine schedule · grad clip · data loading · checkpoint · decoding | `tokenizer.py`, `model.py`, `moe.py`, `optim.py`, `train.py`, `sampling.py` |
| **A2** Systems | Systems | Triton FlashAttention-2 (fwd+bwd) + roofline · DDP (naive→overlap) · ZeRO-1 · FSDP · gradient checkpointing · mixed precision · the 100B memory math | `kernels/`, `rollout/`, `utils/monitors.py`, `utils/` (DDP/ZeRO to build) |
| **A3** Scaling | Scaling | IsoFLOP / Chinchilla fit (compute-optimal N, D) + the budget-constrained training-API leaderboard | `scaling/` (IsoFLOP fitter); the Stanford-API leaderboard runs in `../lectures/assignment3-scaling` |
| **A4** Data | Data | CommonCrawl pipeline: extract → filter → **quality classifier** → exact + **MinHash/LSH dedup**; pipeline order + discard accounting | `data/` |
| **A5** Alignment | Post-training | SFT → Expert Iteration → **GRPO / Dr.GRPO** + the verifiable-reward grader; supplement: **DPO**, reward modeling, safety | `algos/`, `rewards/`, `envs/` |

Module layout: `src/scratch_llm/{algos,rewards,envs,rollout,scaling,data,utils,kernels}/` plus the
flat A1 substrate (`tokenizer.py`, `model.py`, `moe.py`, `optim.py`, `train.py`, `sampling.py`).

**Build status (2026-07-03):** ⚡ two-front Delivery sprint (ADR-0013/0014): **CS336 main track A2→A5 COMPLETE 2026-07-04**
(code + tests + official-scaffold acceptance 50P/0F; graded GPU runs rental-gated per `deploy/runbooks/`) — see `docs/EXECUTION_SPEC_CS336_FINISH.md` · A1 substrate ✅ · **perf-curriculum A1–A6 ALL sm120-runnable
rungs ✅ COMPLETE 2026-07-04** (A1 serving R0–R4.6 = CUDA-graph decode 77% of the memory wall + 4 more;
A2 R0–R6 kernels 96–100% HBM / GEMM 134% cuBLAS-proxy; A3 R0–R2 tensor cores in CUDA 4.1%→38.9%→81.9%
cuBLAS; A4 R0–R3+bwd flash attn ~50% SDPA; A5 R0–R4+§4.3 quant NVFP4 1.48%<MXFP4 / AWQ 1.71×; A6 TP/1F1B/
EP-MoE/MFU gloo-verified) + design notes A1–A7 + H100/B200/8×H200 runbooks + compile-verified ISA kernels
+ **mastery roadmap** `docs/learning/roadmap/`. **Remaining = only the 3 rental DAYS** (hardware-gated). ·
CS336-A2 distributed half ✅ shipped 2026-07-03 (ZeRO-1 · FSDP · one-pager · comms algebra, W1–W4) + A3 ✅ (W5–W6).
Repo green (ruff/pyright 0 · CPU + GPU suites). Rentals: 3 capability-tier sessions
([ADR-0012](docs/adr/ADR-0012-inference-rental-tiers.md) — H100 · 8×H200 serving day · B200).
The A5 RL "aha" + **Capstone DELTA** stay gated behind the perf curriculum (ordering mandate
2026-06-30). See [`docs/STATUS.md`](docs/STATUS.md) and `../STRATEGY.md` §8.

## Engineering disciplines (how labs silently screen — bake these into tests)

1. **Loss-at-init ≈ log(vocab_size).** A fresh LM's cross-entropy on random data must be ≈ uniform
   (`log V`). Off ⇒ head/embedding/masking bug. The cheapest correctness oracle in the stack.
2. **Overfit-one-batch.** Before any real run, drive train loss → ~0 on a single batch. If it
   can't, the optimizer/data/loss wiring is broken — not the data.
3. **Fixed-seed reproducibility.** Seed python/numpy/torch; a re-run reproduces the metric.
4. **Mandatory RL logging** (any RL run; absence = an uninterpretable run): log **entropy**, the
   **KL divergences separately** — `KL(current‖ref)`, `KL(current‖old)` (and, when train and
   inference engines differ, the train↔infer drift) — plus IS-ratio histograms, reward-distribution
   stats, and **length stats** (catches verbosity reward-hacking).
5. **Predict before you run.** Write the falsifiable number first; it is the debugging anchor.

## Develop on the GPU — never skip a step

**We develop on a standing GPU** (RTX PRO 4000 Blackwell, **sm120**, 25 GB — check `nvidia-smi` /
`vast-capabilities`). No build step is deferred for lack of hardware: write the kernel, run it, and
**measure + profile it here, now**. Build the plan's steps **in order and in full**. "CPU-buildable"
means a step *doesn't require* a GPU (pure-torch oracles, gloo distributed-correctness, CPU fake-quant)
— build those on the GPU box too; it is never an excuse to drop scope. Two honesty rails: the **25 GB
cap** (size models to fit) and **report "% of *this* Blackwell," never imply datacenter numbers**. A
bigger / multi-GPU box is **rented (`vastai` skill) only for what this card can't do** — full-scale
throughput vs H100/B200, real multi-GPU NCCL — never abandoned. The only legitimate non-builds are
items the guides tag **SKIP** (e.g. the perplexity/8B/Paloma leaderboards — capped GPU-dollar sinks
with no mastery carry).

## Green-CI rule (enforced, not requested)

A commit ships only if green: `ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`.
`.claude/hooks/green-ci-gate.sh` enforces this as a Claude Code `PreToolUse` hook — a red `git commit`
through Claude Code is blocked (exit 2). For every commit path (external terminal included), symlink
it as a git hook: `ln -s ../../.claude/hooks/green-ci-gate.sh .git/hooks/pre-commit`. Commit messages
are conventional and scoped: `<area>: <imperative>` (e.g. `tokenizer: add byte-level BPE trainer`).

## Build / test

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"          # CPU core: numpy/pydantic/pyyaml/regex
ruff check src tests && ruff format --check src tests
pyright
pytest -m "not gpu"                  # the HW-agnostic gate (mirrors CI + the commit hook); stays
pytest -m gpu                        # the kernels/measurements — run on THIS GPU box (sm120 Blackwell)
```

The `-m "not gpu"` gate is the **commit/CI floor** and stays HW-agnostic; the `-m gpu` suite (Triton/CUDA
kernels, KV-cache decode, real-precision) is exercised on the standing GPU as you develop — not deferred.

## Where things live (read on-demand, not auto-loaded)

| Need | Path |
|---|---|
| Workspace map + CS336→interview-readiness | `../README.md` |
| **The build plan** — A1→A5 spine, load-bearing 20%, discipline gates, build order | `docs/IMPLEMENTATION_PLAN.md` |
| **Performance & inference track** — the decode-memory-wall spine · 2026 frontier findings · EV-ranked perf build list | `docs/PERFORMANCE_TRACK.md` |
| **GPU & kernels from zero** — laddered rung 0→9 curriculum (no GPU background assumed; AI explains/visualizes, you derive/implement) | `docs/GPU_FROM_ZERO.md` |
| **Perf curriculum — engineering spec** — hardware gates, per-assignment falsifiable predictions, DoD, kill criteria (A1–A7); read before any perf-curriculum session | `performance/PERF_ENGINEERING_SPEC.md` |
| **Perf curriculum — implementation plan** — phased sequencing, current node, rental batching strategy, rung-by-rung status | `performance/PERF_PLAN.md` |
| **Daily operating rhythm** — the frontier-engineer day on this harness (`/standup` → deep-work → `/eod`); one active node, predict-before-run, measured>implemented | `docs/OPERATING_RHYTHM.md` |
| **Measurement ledger** — durable predicted-vs-measured record (what git can't track; the "DoD is a profile" gate) | `bench/RESULTS.md` |
| **Vision & plan (Vietnamese)** — tổng hợp tầm nhìn + specs + kế hoạch (synthesis, not source-of-truth) | `docs/VISION_VI.md` |
| **Learning track (Vietnamese)** — chuỗi bài mastery Feynman/teach-back, trace code + số đo thật từng component (viết từng bài khi dạy, theo giao thức "one concept at a time") | `docs/learning/` (start at `INDEX.md`) |
| **MODEL mastery ROADMAP (Vietnamese)** — first-principles DERIVATION walkthrough of the WHOLE LLM-from-scratch side: 10 série / ~50 Bài (M1 tokenizer → M2 transformer → M3 optimization → M4 training → M5 distributed → M6 scaling → M7 data → M8 RL → M9 MoE·MLA·MTP → M10 close-the-loop + F1–F9), derivation-first + heavily cross-referenced to DeepSeek/GLM/Kimi/Qwen/nanochat (vendored impls). The twin of the perf roadmap; read this first | `docs/learning/roadmap_model/` (start at `README.md`) |
| **Perf mastery ROADMAP (Vietnamese)** — first-principles walkthrough of the WHOLE perf build: 41 Bài / 6 series (S1 serving substrate → S6 distributed+ISA), each tracing a core file to `file·func·line` (pinned commit) with the measured "aha", teach-back gate, frontier link. The map for going through every implementation cold | `docs/learning/roadmap/` (start at `README.md`) |
| **Fresh-pod continuity** — rebuild everything on a newly rented Vast.ai GPU (Claude Code install · torch cu130/sm120 · hook re-link · **auto-memory restore**); the one-command `scripts/bootstrap-pod.sh` + what survives destroy vs what you rebuild | `docs/VASTAI_BOOTSTRAP.md` · `.claude/memory-snapshot/` |
| **Per-assignment build guides** — every deliverable tagged + mapped to `src/scratch_llm/` (start at `INDEX.md`) | `docs/assignment_guides/` |
| **Build status** — what's built / tested / green (single source of truth) | `docs/STATUS.md` |
| Design specs — KV-cache · rollout seam · FA2 roofline · MoE walkthrough | `docs/design/` |
| **Frontier practice (2026)** — per-pillar modern-default upgrades · build labs · know-it items (fact-checked; + tagged GDM-aligned additions) | `docs/FRONTIER_PRACTICE_2026.md` |
| Architecture decisions | `docs/adr/` |
| **The official course (spec + test oracle)** — lectures + the 5 assignment scaffolds | `../lectures/` |
| **Capstone — DELTA** (GDN-2 decode-kernel *spike*; merged RFC + 4-week barbell plan) | `../DELTA.md` |

## Implementation rules

- **Own every line.** This is your from-scratch implementation. The official `../lectures/assignment*`
  scaffolds are the **spec + test oracle** — implement against their `tests/adapters.py`; do not copy
  solutions.
- **Reference-as-oracle (re-implement to truly own).** Treat *all existing code* — this repo's
  `src/scratch_llm/` and any reference repo — as **oracle, not your work**: the goal is mastery you can
  rebuild blind. The teach-back gate decides what to re-own; don't rebuild what you can already defend
  cold (FOP-7). To genuinely re-own a module, **blank-slate it one at a time**: git/tag holds the
  reference → reduce the body to its signature + `raise NotImplementedError` → confirm the tests go
  **red** (proves they have teeth) → re-derive from first principles → green **+ measured**. Never blank
  the whole repo; green-CI still gates every commit. Curriculum: `docs/GPU_FROM_ZERO.md` (rung protocol)
  + `docs/PERFORMANCE_TRACK.md` (what/why). **Measured > implemented — log numbers to `bench/RESULTS.md`.**
- Every module's docstring states its intent, the key invariant it must satisfy, and (where relevant)
  the interview question it answers — the engineering rationale lives in the code, not only the guides.
- Land changes **test-first** where practical: write the invariant (loss-at-init, decode round-trip,
  causal-no-leak, overfit-one-batch) as a test, then make it pass.
