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
6. **AI-mode boundary.** Mode 1 delegate (plumbing) · Mode 2 human-leads-AI-assists (research-critical) · Mode 3 AI-OFF (the RL-math + kernel reps the interview tests). Agents MUST refuse to write Mode-3 targets (the loss bodies, the kernel the human is learning) from scratch — offer failing tests, a Socratic critique, or a post-hoc review instead.
7. **Citation-tree mastery.** Traverse to the non-redundant gap; reuse before re-deriving; don't rebuild owned work.
<!-- FOP:end -->

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
GPU-bound steps are **resourced** (rented), never silently dropped (see "Follow the plan").

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

**Build status (2026-06-20):** A1 substrate ✅ (BPE · model · MoE opt-in · AdamW · train · sampling) ·
A2 partial (FlashAttention-2 oracle+Triton+roofline ✅ · KV-cache ✅ · `utils/monitors.py` ✅ · rollout
seam ✅; **DDP/ZeRO-1/FSDP + the memory one-pager remain**) · A3/A4/A5 to build. 92 tests green,
ruff/pyright clean; last code commit Jun 8. **Next ship (EV-ranked): the A5 RL "aha"** (GRPO/Dr.GRPO) ·
**Capstone DELTA** (GDN-2 decode kernel): design + plan written, **base-first** — gated behind the A5
ship + the Step-0 gate. See [`docs/STATUS.md`](docs/STATUS.md) and `../STRATEGY.md` §8.

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

## Follow the plan — never skip a step

Build the plan's steps **in order and in full**. A step that needs hardware is **not** a step to
skip — it is a step to *resource*. If a step requires a GPU (Triton/FA2 kernels, DDP/ZeRO/FSDP,
real serving, RL runs), **rent one on vast.ai** (use the `vastai` skill) and complete it.
"GPU-deferred" means *scheduled on rented hardware*, never *abandoned*. Order CPU-buildable steps
first when cheaper, but CPU-vs-GPU is never an excuse to drop scope. The only legitimate non-builds
are items the guides tag **SKIP** (e.g. the perplexity/8B/Paloma leaderboards — capped, GPU-dollar
sinks with no mastery carry).

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
pytest -m "not gpu"                  # the CPU gate (mirrors CI); src/ layout via pytest pythonpath
```

## Where things live (read on-demand, not auto-loaded)

| Need | Path |
|---|---|
| Workspace map + CS336→interview-readiness | `../README.md` |
| **The build plan** — A1→A5 spine, load-bearing 20%, discipline gates, build order | `docs/IMPLEMENTATION_PLAN.md` |
| **Performance & inference track** — the decode-memory-wall spine · 2026 frontier findings · EV-ranked perf build list | `docs/PERFORMANCE_TRACK.md` |
| **GPU & kernels from zero** — laddered rung 0→9 curriculum (no GPU background assumed; AI explains/visualizes, you derive/implement) | `docs/GPU_FROM_ZERO.md` |
| **Daily operating rhythm** — the frontier-engineer day on this harness (`/standup` → deep-work → `/eod`); one active node, predict-before-run, measured>implemented | `docs/OPERATING_RHYTHM.md` |
| **Measurement ledger** — durable predicted-vs-measured record (what git can't track; the "DoD is a profile" gate) | `bench/RESULTS.md` |
| **Vision & plan (Vietnamese)** — tổng hợp tầm nhìn + specs + kế hoạch (synthesis, not source-of-truth) | `docs/VISION_VI.md` |
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
