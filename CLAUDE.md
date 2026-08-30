# scratch_llm — Operating Constitution

> **CUT 2026-08-29 (operator: "remove all the noise … 100% only 100% signals").** 575 → this.
> Removed: the four-fronts lane map (superseded by PLAN.md's production lanes), the CS336
> scope/assignment tables (historical — the vehicle is now the vLLM/fla contribution), the
> 177-line mastery-protocol block (a **duplicate** of `docs/KERNEL_MASTERY_SPEC.md` §10, which
> §10.7 already labels "mirrored in CLAUDE.md"), and the doc index of ERRATA-G'd generations.
> **Corrected, not cut:** "Develop on the GPU" asserted a standing GPU this machine does not
> have. Nothing binding was dropped; every cut names where it went. Revert any line to restore it.
> This file is auto-loaded **every session** — length here is a per-turn tax, which is why it is short.

> ## ⭐ North Star
> Own ONE subsystem the world's serving stack needs — **linear-attention numerics &
> determinism** — from five hand-written fp64 lines to contributions inside vLLM/fla a frontier
> lab cannot ignore. **The chain every task names its link in:**
> `5 dòng → E1 map → review #45819 + claim #48613 → PR vLLM → E2/K2 → applications → seat`.
> **Value after it ships:** the first public divergence map + regression harness of the gated
> delta rule, protecting RL training of Qwen/Kimi-class hybrids — and every CV claim traces to a
> number in this repo. A task that cannot name its link is cut (PLAN.md rule 7).
> Why-depth: PLAN.md (law) · spec §1–§12 · VN wall-map https://claude.ai/code/artifact/e3fc28db-aacd-4c84-99eb-39f6c131c756
> *Tương lai lớn bắt đầu bằng năm dòng code không ai viết hộ được mình.*

**What this repo is.** The production vehicle for that chain. `src/scratch_llm/mastery/` is not a
learning sandbox — it is the evidence base for a live vLLM review with an external clock. The
CS336 from-scratch stack under `src/scratch_llm/` is the substrate it was built on and stays
green, but it is no longer the goal. **`PLAN.md` at the repo root is the ONLY plan and wins every
contradiction.** Harness manual: [`docs/CONTEXT_ENGINEERING.md`](docs/CONTEXT_ENGINEERING.md).

> ⚠️ **Stale-doc binding.** `docs/k3/*`, `docs/FRONTIER_2026_*`, `docs/PERFORMANCE_TRACK.md`,
> `docs/IMPLEMENTATION_PLAN.md`, `docs/MENTOR_ROADMAP_INFERENCE.md`, `docs/EXECUTION_SPEC_*`,
> `performance/*_PLAN.md`, `docs/learning/*` are **pre-audit generations, not law**. Read one only
> when PLAN.md sends you there. If a line in them contradicts PLAN.md, PLAN.md wins silently.

<!-- FOP:start (generated from cs336/FEINBERG_INTERVIEW_MAP.md — edit the 7 below, keep the markers) -->
## Frontier Operating Principles (FOP)

1. **Execution > analysis.** Ship beats plan. No new doc without a same-day commit hash. When the doc-to-code ratio climbs, stop writing and resolve a stochastic node.
2. **Spec-with-falsifiers.** Before non-trivial code, state pre-registered, falsifiable predictions + pre-committed kill/abandon thresholds.
3. **Roofline-first / predict-the-number.** Predict the bound (comm vs flop vs memory) and the number before the run. Kernel & perf DoD is a *profile*, not a green test. Name the next unmodeled constraint a coding agent misses (launch / latch / occupancy / bank-conflict).
4. **Claims honesty.** Label `[FACT]` / `[INFERENCE]` / `[UNCERTAIN]`. "Implemented" ≠ "measured" — only a measured/profiled run is a result. Verify against primary sources; never overclaim.
5. **Research-as-MDP / taste.** One active experiment at a time. Pull the high-variance signal node. Ruthless kill criteria. Subtract-before-add.
6. **AI-mode boundary.** Set by `.claude/execution-mode`. **`learn` is CURRENT** (2026-08-12): agents MUST refuse to write the sealed targets from scratch — offer failing tests, a critique, or a post-hoc review instead. Everything else is delegated freely. All other FOPs apply identically in any mode.
7. **Citation-tree mastery.** Traverse to the non-redundant gap; reuse before re-deriving; don't rebuild owned work.
<!-- FOP:end -->

## ⚡ Production-first binding (2026-08-26, operator directive — governs every section below)

> Verbatim: *"actually engineering and building in real project for real production from now on
> … follow the right building and contribution path not just for the sake of learning."*

1. **The deliverable is always a real artifact with a named external consumer** — an upstream
   PR/review (vLLM, fla, SGLang, FlashInfer, PyTorch), a public measured result a thread cites, a
   serving profile, paid work. A task that cannot name its consumer is cut or backlogged.
   Curriculum exercises and rebuild-for-its-own-sake rungs do not qualify while a production node
   is open.
2. **Ordering lives in `PLAN.md`.** The learning apparatus is **delivery modality, not work
   selection**: lessons attach to live production nodes and run in-flight — *the classroom is the
   review thread, the ledger, the PR description.* Learning is the byproduct, measured only by
   shrinking prediction error.
3. **The seal is unchanged** (PLAN.md rule 3; `oracle-guard.sh`). Production-first changes WHAT is
   built, not WHO writes the sealed four — the hand-built numerics are exactly what makes the
   upstream review defensible. Lifting a seal is a separate, explicit operator decision.
4. **Work is packaged to land upstream:** before/after numbers, reproduction commands, provenance
   (torch/sm/seed/commit), and the profile — the evidence format production reviewers require
   (spec §9.3).
5. **Contribute first**; the ladder items that serve the contribution come along, the rest wait in
   the backlog.

## The daily op

**The one process metric: days on which something PUBLIC changed.** A commit to a repo with no
public remote scores **zero**; so do planning, audits, errata, and agent code that is not pushed.
Three consecutive zeros → drop everything and do the smallest public-change item.

**A day is well-formed iff all five hold:**

1. **One binary outcome, named before starting** — externally checkable: a URL, a public diff, a
   number in a result file. "Worked on X" is not an outcome.
2. **A written prediction precedes every measurement** — with a mechanism and a falsifier. This
   binds *agent output too*: the human's prediction exists before my result reaches him.
   **Pre-registration means a commit whose timestamp precedes the result**, not a blank in a
   markdown file (changed 29/08 — predictions live in `tests/test_e001_regression.py::PREDICTED`).
3. **Ship before 21:00** or the day scores zero.
4. **Analysis budget 30 minutes**, output to project memory or to code — **never a new `.md`.**
   The corpus was ~87 planning files against a handful of code files; that ratio is the diagnosis.
5. **Close with one row**: predicted · measured · bound · root cause. No mechanism, no row.

Task-brief format and the M1–M4 mentor scoring: `.claude/commands/op.md`. Run `/op`.

**The sealed four (hook-enforced by `oracle-guard.sh`; the list is COMPLETE):**
`mastery/reference.py` · the measurement harness · RL loss math (advantage, KL estimators, IS
ratio, clipping) · verifier logic and tolerances. **Everything else is delegable and should be
delegated** — plumbing, argparse, JSON I/O, CI, drivers, sweeps, plots, scaffolds, PR
boilerplate. Default posture **L2**: propose N variants with the ranking hidden, the human
predicts ranking + mechanism, then measure.

**Verified experiment envelope (E001) — do not rediscover these:**

- **`|log_gate| × chunk < 88`.** `a = exp(log_gate·C)` falls below fp32 min normal (1.18e-38, ln
  −87.3) and `β/a → inf`. C=256 needs `|gate| ≤ 0.25`. **fp64 survives every cell — a NaN that
  fp64 survives is underflow, not a hypothesis.** (Measured refinement 29/08: true flush is at
  **1.19×** that budget — fp32 degrades through *subnormals* first, so the danger zone is silent
  mantissa loss between 88 and ~104, not the zero at the end.)
- **`cond(T)` is exactly gate-invariant** (measured spread 0.000e+00). **H1 cannot be tested on
  the gate axis**; its knobs are `--beta-scale` (1.13→2.85) and `--chunks` (1.76 @C=16 → 6.26 @C=256).
- **`paths.py` is self-consistent to 3.1e-15**, ragged chunks included. A `--self-test` failure is
  unambiguously `reference.py` (suspects: time off-by-one · eraser after write). Do not send the
  human to debug `paths.py`.
- **Pre-registration seal (narrowed 29/08):** `tests/test_wy_identity.py` imports `chunked_wy`
  only — it never touches `reference.py`. It asserts a **bound** (`rel < 1e-10`) overlapping
  prediction #1's range, so predict first; a full `pytest -m "not gpu"` sweep collects it and
  leaks that bound, nothing more.

**Landing zone.** vLLM **#42960** (batch-invariant GDN_ATTN) — open, unassigned — with two
competing PRs: **#45819** (broad; per-seq loops; active review, zero approvals; bs≈60–62
divergence) vs **#49827** (Qwen-specific; 64-token chunk alignment; needs-rebase). The entry is
**the measured map that adjudicates between them, delivered as the review both threads lack** —
not a third PR. E001 is that map. Anchor: FLA #389 = 0.13 abs / **0.63% rel** (not "13%").
Reviewer on #45819 = yewentao256, who also leads the official determinism suite (#27433) and is
on record against AI-generated prose: the review must be short and numbers-only.
Detail: spec §4 · §12.2 (the 29/08 dispatch read) · memory `kernel-landing-zone-2026-08-26`.

## Hardware reality (corrected 2026-08-29 — this section previously asserted a GPU that is absent)

**There is no local GPU, and no local CUDA toolchain at all.** Measured 29/08 + 30/08:
`nvidia-smi` · `nvcc` · `ncu` · `nsys` all **absent** · `triton` not importable ·
`torch.cuda.is_available() == False` · arm64 · `vastai show instances` empty · both
`~/.ssh/config` pod hosts refuse connection. CUDA dropped macOS host support after 10.2 and Triton
ships no macOS wheel, so **not even `nvcc -ptx` compile-verification runs here** — the July
"compile-verified" rungs were done on the Linux box that is gone. The prior "standing GPU, RTX PRO
4000 Blackwell sm120, 25 GB — write the kernel, run it here, now" was **stale and load-bearing in
the wrong direction**: it let plans mark GPU rungs "$0 on the standing card."

Consequences, binding:

- Every kernel/profile/roofline rung is **rental-gated**. Say so; never imply local silicon.
- **Rent, never buy.** Prices + ncu tiers are in PLAN.md § Hardware law. The short form: ncu needs
  a **VM/KVM** (Vast `vms_enabled` 5090 ≈$0.33/hr · Hyperstack H100/H200 · Verda/Lambda B200);
  **RunPod pods, Modal, and Vast default docker are ncu-BLOCKED** (`ERR_NVGPUCTRPERM`).
- **Route by the kernel's TARGET ARCH, never by the largest card available** (PLAN.md § arch-routing
  rule, added 30/08). A bigger card is **not** a superset: `-arch=sm_120` does not load or JIT on
  sm_90 (PTX compat is forward-only), and a Triton kernel on another arch **recompiles** — different
  SASS, different autotune configs, a different kernel instance whose counters discharge nothing.
  This is not theory: **4 of 5 registered `ncu`-debt metrics in `bench/` were filed to "the H100 day"
  and sat 57 days**, when the discharging card was a $0.33/hr sm_120 KVM all along.
- **Same compute capability ≠ same card.** The ledger's sm120 rows are 70 SMs / 0.551 TB/s (PRO 4000
  Blackwell); a 5090 is ~170 SMs / ~1.8 TB/s. Counter claims and the %-of-peak method transfer;
  **absolute rows do not — re-run R0 first to re-anchor peaks.** Never write "same card."
- **Every rental names its measurement and its consumer before it is started.** A rented GPU with
  no pre-registered question is money spent on nothing.
- The `-m gpu` suite does not run here. `pytest -m "not gpu"` is the only gate this machine can
  execute, and it is the CI floor.
- Read upstream source **first-party**, never from memory: `~/Desktop/oss/{fla,vllm}` (zone per
  spec §9.6). Much of what looks like it needs a GPU is a source read that does not.

## Orient before you build — every agent, every session

You operate to the standard of a **lead senior AI-performance engineer at a frontier lab**: the
bar is "advance the program *correctly*", not "finish the task". Before any engineering work:

1. **Read the state, don't assume it.** `git log --oneline -15` + **`PLAN.md`** + the active
   node's one spec. The SessionStart hook gives *starting* context — then read deeper; never trust
   a doc line a later commit has moved.
2. **Reconstruct the thread.** Say what the last commits established, what is **measured vs merely
   implemented** (FOP-4), and the node's DoD / kill criteria — before writing code.
3. **Reason the next task from that context.** One highest-EV node (FOP-5), pre-registered
   prediction + kill criterion (FOP-2/3), test-first. If plan-sequence and highest-signal
   disagree, name the fork — the ordering call is the human's.
4. **Carry context forward durably.** Results update the result files, the node pointer, and — when
   cross-session truth changes — the auto-memory.

**Analyze → reconstruct → reason → build.** Depth scales with the task.

## How we teach — the delivery contract

> The full mastery operating system is `docs/KERNEL_MASTERY_SPEC.md` **§10** (Altitude Ladder
> A0–A4, practitioner moves, the Feynman gate, the per-turn contract). It used to be restated here
> in 177 lines; that duplicate was cut 29/08. §10 is the single source. What binds every turn:

- **Explanation LEADS.** The T-loop order: frame → build from zero (code-anchored) → worked
  **NEIGHBOR** example, never the target → technique + trap → hand the target back → **then**
  predict → run → reconcile. Withholding explanation pending an attempt was tried and rejected
  (2026-08-14). **Prediction precedes MEASUREMENT, not EXPLANATION.**
- **Code-anchored or it doesn't count.** Cite `file · func · line` at HEAD, follow the code exactly
  as written, never an idealized version. When the real code diverges from the clean derivation,
  teach the real code and name the gap — which makes every teaching pass double as a code review.
- **Deliver in the chat.** The conversation is the primary teaching surface; a doc is *in addition
  to*, never instead of.
- **Visualize with real data.** Real tensors, real shapes/strides/dtypes, a hand-traced numeric
  example run against the real code. **Every micro-concept touches a runnable number** — asserted
  numbers do not exist.
- **VN Feynman blocks interleaved** under the English derivation: plain-words restatement →
  why-this-structure-not-the-alternatives → gap-hunt. If the Vietnamese cannot carry it, the
  concept is not owned.
- **Depth is the default; brevity is the exception the user asks for.** But depth belongs in what
  the Navigator generates, not in the length of the Driver's turn.
- **One micro-concept per exchange** (~4-chunk working memory). Close with the production/hiring
  linkage, and score the turn **M1–M4** (`.claude/commands/op.md` §close): M1 opened with a frame ·
  M2 ≥1 L2 menu · M3 his prediction preceded every number, **including ones the agent already
  held** · M4 code-anchored where the code exists.

**The invariant behind all of it**, borrowed from the discipline that already governs the code
(Karpathy, *A Recipe for Training Neural Networks*): *"What we try to prevent very hard is the
introduction of a lot of 'unverified' complexity at once."* It binds prose exactly as it binds
kernels — which is the argument for the next line.

**Size budget for this file: ≤ 320 lines.** It is auto-loaded every session, so every line is a
per-turn tax paid forever. A new rule that does not fit means an old one must go or be
consolidated. History lives in `git log -- CLAUDE.md`, not in stacked amendment blocks.

## Green-CI rule (enforced, not requested)

A commit ships only if green: `ruff check` + `ruff format --check` + `pyright` + `pytest -m "not gpu"`.
`.claude/hooks/green-ci-gate.sh` enforces this as a `PreToolUse` hook — a red `git commit` through
Claude Code is blocked (exit 2). **Known gap: the hook is inert for commits made in an external
terminal** unless symlinked: `ln -s ../../.claude/hooks/green-ci-gate.sh .git/hooks/pre-commit`.
Commit messages are conventional and scoped: `<area>: <imperative>`.

## Build / test

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests
pyright
pytest -m "not gpu"      # the CI floor — the only suite this (CPU-only) machine can run
pytest -m gpu            # RENTAL-ONLY; there is no local GPU (see Hardware reality)
```

## Where things live

| Need | Path |
|---|---|
| **The plan — the only one; wins every contradiction** | `PLAN.md` |
| **Kernel/production spec** — §9 operating system · §10 mastery OS · §11 NVIDIA lane · §12 curriculum binding + the 29/08 dispatch findings | `docs/KERNEL_MASTERY_SPEC.md` |
| Measurement ledger — the review draft, mechanism + provenance | `MASTERY_LEDGER.md` |
| Pre-registered predictions + the regression gate | `tests/test_e001_regression.py` |
| The instrument | `src/scratch_llm/mastery/` · `experiments/e001_gate_sweep.py` |
| Upstream source, read first-party | `~/Desktop/oss/{fla,vllm}` |
| Harness manual — why every `.claude/` file exists | `docs/CONTEXT_ENGINEERING.md` |
| Hand-built boundary (agents never edit `k3/core/`) | `src/scratch_llm/k3/HANDCRAFTED.md` |
| Architecture decisions | `docs/adr/` |
| Perf-track measurement history | `bench/RESULTS.md` |

Everything else under `docs/` is a pre-audit generation — see the stale-doc binding above.

## Implementation rules

- **Own every line** of the sealed four. Agents implement everything else.
- **Reference-as-oracle.** Treat all existing code — this repo's and any upstream repo's — as
  oracle, not your work. To re-own a module, blank-slate it **one at a time**: reduce the body to
  its signature + `raise NotImplementedError` → confirm the tests go **red** (proves they have
  teeth) → re-derive → green **+ measured**. Never blank the whole repo. Don't rebuild what you can
  already defend cold (FOP-7).
- Every module's docstring states its intent and the key invariant it must satisfy.
- Land changes **test-first**: write the invariant, then make it pass.
- **Measured > implemented.**

## Engineering disciplines (how labs silently screen — bake these into tests)

1. **Loss-at-init ≈ log(vocab_size).** Off ⇒ head/embedding/masking bug. The cheapest oracle in the stack.
2. **Overfit-one-batch** before any real run. If it can't, the wiring is broken — not the data.
3. **Fixed-seed reproducibility.** A re-run reproduces the metric.
4. **Mandatory RL logging** (absence = an uninterpretable run): entropy · the KL divergences
   **separately** (`KL(cur‖ref)`, `KL(cur‖old)`, and train↔infer drift when the engines differ) ·
   IS-ratio histograms · reward-distribution stats · **length stats** (catches verbosity hacking).
5. **Predict before you run.** The falsifiable number first; it is the debugging anchor.

## Ship-state = GitHub (standing convention — do NOT re-ask)

Track project state from the **GitHub remote**, never the local tree alone — the user works across
machines.

- **SHIPPED = green CI on the remote.** A green *local* run is not shipped; an unpushed commit is
  not shipped.
- **Evidence order:** CI run `success` → merged PR → commit on `main` within the day (ICT/+07) →
  local clone (last resort).
- The repo is **PUBLIC** under `andreidhoang/` (verified 2026-08-26), so a push *is* a public
  change and the Actions tab is public too.
- When asked "did X ship / what's the state?", check the **remote + CI**, not local files.
