# ADR-0013 — Execution-mode switch: full delegation (deliver first, master later)

- **Status:** Accepted (2026-07-03, Navigator mandate — verbatim: "bypass all rules of me building
  by hand, I change strategy to save time and focus on delivering value first while mastering
  after seeing the code")
- **Layer:** the whole operating constitution (CLAUDE.md FOP-6, the Mode-3 boundary, the teach-back
  gate) + the agent harness that enforces it
- **Decides:** who implements, what gates a rung, and what happens to the mastery track — for the
  entire `performance/` curriculum and repo-wide, until the mode is flipped back

## Context

The repo was built around a *learning* contract: kernel bodies and RL-math are Mode-3 (AI-OFF)
human reps, enforced by prose (CLAUDE.md FOP-6, `PERF_ENGINEERING_SPEC.md` §6), a physical hook
(`.claude/hooks/kernel-write-guard.sh`), four refusal-encoded agents, and a teach-back concept gate.
The Navigator has changed strategy: **ship the entire performance curriculum end-to-end by agent
execution now; build mastery afterwards by studying the shipped, measured code** (the Vietnamese
Feynman lessons remain available on demand — they just no longer gate the build).

A single prose instruction cannot override a layered enforcement stack — the hook would still
`exit 2`, the agents would still refuse. The mode must be switched at every layer, durably and
reversibly.

## Decision

1. **`.claude/execution-mode` is the single mode switch** (git-tracked, one word):
   - **`delegate`** (set 2026-07-03): Claude implements *everything* end-to-end — serving systems,
     Triton/CUDA kernel bodies, RL math — as a senior frontier performance engineer.
     `kernel-write-guard.sh` allows kernel-file writes. The teach-back gate does not block
     advancement. Mastery is post-hoc: every shipped rung is appended to the
     `docs/learning/INDEX.md` **study queue** (rung → commit → key files → the one number),
     consumed later via `/master`.
   - **`learn`**: the historical contract, unchanged — Mode-3 refusal, human reconstructs kernels
     from blank, teach-back gates concepts. Flip back by writing `learn` to the file.
2. **What survives in BOTH modes (correctness rails, not mastery gates):**
   - Pre-registration before running (SPEC D5 / FOP-2/3): a prediction + kill line in
     `bench/RESULTS.md` **before** every measurement.
   - Oracle-first, test-first, adversarial inputs (D1/D6); green-CI gates every commit
     (`green-ci-gate.sh` untouched).
   - Measured > implemented (FOP-4): a rung closes only with `[FACT]` ledger rows; append-only ledger.
   - Rental discipline (SPEC §2, ADR-0012): nothing is rented until every sm_120-runnable rung of
     the relevant assignment is oracle-correct and ledgered; H100/B200/8×H200 work is prepared as
     compile-ready code + runbooks now, executed on a rented box later.
   - Adversarial review before commit: `kernel-ship-reviewer` / `ship-reviewer` still gate; in
     delegate mode the *implementer* they gate is Claude, and independent review is MORE important,
     not less.
3. **Run policies (Navigator, 2026-07-03):** push to `origin main` after every green rung commit;
   Vietnamese lessons deferred entirely during the run (study queue only).
4. **Supersession clause:** any doc/docstring line in this repo asserting the Mode-3 refusal, the
   "human implements" slot, or teach-back gating **without an execution-mode qualifier** is
   superseded by this ADR while the mode is `delegate`. (High-traffic sites are amended in-place;
   long-tail mentions — old notes, lesson files, docstrings — are covered by this clause.)

## Consequences

- `CLAUDE.md` FOP-6 and "How we build", `PERF_ENGINEERING_SPEC.md` (preamble + §6),
  `PERF_PLAN.md` (rung workflow), `src/scratch_llm/kernels/CLAUDE.md`, the agent definitions
  (bench-writer, roofline-analyst, kernel-ship-reviewer), the commands (`/standup`, `/ship`,
  `/kernel-day`, `/profile`, `/kreview`), `docs/OPERATING_RHYTHM.md`, `docs/GPU_FROM_ZERO.md`,
  `docs/GEMINI_CONTEXT.md`, `.agents/AGENTS.md` are amended to read the switch.
- `bench-writer` keeps writing tests/benches *only* — re-justified as **separation of duties**
  (the spec writer must be independent of the implementer so tests can't be tuned to the code),
  no longer as a learning boundary.
- The interview-readiness risk is real and accepted: shipped-but-not-yet-internalized code. The
  mitigation is the study queue + `/master` on demand — mastery debt is tracked, not silently
  dropped.
- The `docs/learning/` track, `/master`, and `kernel-tutor` remain fully functional — they are
  pull-based now instead of gating.
