---
description: Orient and start the next load-bearing step — where we are, what's next, then build it test-first in mentor mode
argument-hint: "(optional) a specific pillar/step to jump to"
---
Relentless-execution orientation. $ARGUMENTS

1. **Where we are.** Read `docs/STATUS.md` (build state) + `docs/IMPLEMENTATION_PLAN.md` (the A1→A5
   spine). State the current pillar and what is green.
2. **The next load-bearing step.** Name the single next step for the current pillar from the plan +
   the relevant `docs/assignment_guides/A*` — the **load-bearing 20%**, not a SKIP / COURSE-ROTE
   detour. If a frontier **CORE-default** applies (`docs/FRONTIER_PRACTICE_2026.md`), fold it in.
3. **Build it — mentor mode, test-first.** Run the `CLAUDE.md` "How we build" loop: first-principles +
   visualize → predict-before-run → write the invariant as a test → implement → green
   (`ruff` / `pyright` / `pytest -m "not gpu"`) → quiz me to teach it back. If the step needs a GPU,
   resource one (vastai skill) — never drop scope.
4. **One step at a time.** Finish it green before proposing the next. Then stop and let me drive.
