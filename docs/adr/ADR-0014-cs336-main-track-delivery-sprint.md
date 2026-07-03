# ADR-0014 — CS336 main-track delivery sprint (autonomous A2→A5 finish, two-front protocol)

**Status:** accepted · 2026-07-03
**Depends on:** [ADR-0013](ADR-0013-execution-mode-full-delegation.md) (`delegate` execution mode —
the switch that lets agents implement everything end-to-end; this ADR does not restate it)

## Context

With `.claude/execution-mode = delegate` (ADR-0013), two agent fronts now run concurrently on this
repo under the same Navigator mandate ("deliver first, master after seeing the code"):

- **Perf front (other agents):** the `performance/` curriculum (current node A1 R4.2), governed by
  `performance/PERF_PLAN.md`.
- **Main-track front (this ADR):** the remaining CS336 assignment deliverables — A2 distributed
  finish (ZeRO-1 · FSDP · 100B one-pager · comms algebra), A3 scaling, A4 data, A5 alignment —
  previously parked behind the 2026-06-30 perf ordering mandate.

The Navigator's 2026-07-03 instruction un-parks the main track: finish **all** of it autonomously,
without Navigator involvement, and for anything needing hardware beyond the standing sm120 24 GB
box, ship the code + a rental runbook now and run it when the box is rented.

## Decision

1. **The one canonical plan** for the main-track finish is
   [`docs/EXECUTION_SPEC_CS336_FINISH.md`](../EXECUTION_SPEC_CS336_FINISH.md) — task DAG (W1→W11),
   per-node files/tests/DoD, GPU-tier tags, loop protocol. It supersedes the ship-order of
   `../STRATEGY.md` §8 (absent from this pod) for the sprint's duration. The 2026-06-30 perf
   ordering mandate is **dissolved into the two-front split**: perf continues on its own plan; the
   main track no longer queues behind it.
2. **Mastery is deferred, not deleted.** Every shipped main-track module adds a row to
   [`docs/learning/MASTERY_DEBT.md`](../learning/MASTERY_DEBT.md) (concept · files · interview
   question); Vietnamese Feynman lessons are written post-hoc from shipped code (same contract as
   ADR-0013's study queue for perf rungs).
3. **Scope and quality law unchanged:** guides' LOAD-BEARING/COURSE-ROTE/SKIP tags govern scope
   (leaderboards stay SKIP); green-CI, test-first, predict-before-run, ADRs, docstring
   intent+invariant all still gate.
4. **GPU-tier honesty:** nodes needing >1×24 GB (multi-GPU NCCL benchmarks, H100-class A5 runs)
   ship **code-complete + CPU/gloo-tested** with a self-contained runbook in `deploy/runbooks/`
   (exact `vastai` commands, launch lines, predicted numbers, cost). Never silently dropped.
5. **Two-front collision protocol:**
   - Perf zone (main-track agents never write): `performance/`, `src/scratch_llm/serving/`,
     `src/scratch_llm/kernels/paged_decode_triton.py`, perf sections of `bench/RESULTS.md`.
   - Main-track zone (perf agents never write): `src/scratch_llm/{utils (distributed),scaling,
     data,algos,rewards,envs}/`, `docs/EXECUTION_SPEC_CS336_FINISH.md`, main-track docs/tests.
   - Shared files (`CLAUDE.md`, `docs/STATUS.md`, `pyproject.toml`, `bench/RESULTS.md`,
     `docs/learning/INDEX.md`): `git pull --rebase` before every commit; small atomic commits;
     additive edits only; never `git add -A` (the other front may have work in flight in the same
     checkout).

## Consequences

- The full CS336 stack (A1–A5) reaches code-complete + tested on agent wall-clock; the Navigator's
  calendar stops being the critical path.
- Rental sessions (ADR-0012 tiers + the new runbooks) become pure execution days: rent → run →
  paste numbers into `bench/RESULTS.md`.
- Risk: two fronts in one checkout can race on shared files — mitigated by the zone split +
  pull-rebase + additive-only rule above.
