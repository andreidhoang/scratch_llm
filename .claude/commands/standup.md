---
description: Morning open — orient, pick the ONE EV-ranked node for today, write its falsifiable prediction + kill criterion, then hand off to the build
argument-hint: "(optional) a node you already want to commit to today"
---
Daily open (the `docs/OPERATING_RHYTHM.md` Open). $ARGUMENTS

1. **Orient (terse).** Branch + uncommitted count (from the session-start line) + what is green now
   (`docs/STATUS.md`). One or two lines — do not re-dump the plan.
2. **Pick ONE node.** From `docs/IMPLEMENTATION_PLAN.md` + `docs/STATUS.md`, name the single
   highest-EV *open* node for today (one experiment / one load-bearing step) — EV-ranked per
   `STRATEGY.md` §8, not numeric, not a SKIP/COURSE-ROTE detour. If two compete, state the EV call in
   one line and choose one. **Active-node count today = 1.**
3. **Write the contract (PAUSE for my prediction).** For that node, draft and show me:
   - the **falsifiable prediction** — the number / shape / roofline bound I expect (ask me to write or
     confirm it; predict-before-run is a hard gate);
   - the **kill criterion** — the threshold at which I abandon this node today.
4. **Mode check.** Read `.claude/execution-mode` (ADR-0013). If `delegate` (current): Claude
   implements the node end-to-end — bench-writer still authors the tests independently, and
   kernel-ship-reviewer still gates the commit; note the rung in the `docs/learning/INDEX.md` study
   queue. If `learn` and the node is Mode-3 (RL-math loss body / a kernel being learned): the human
   writes it; agents supply failing tests / critique / review — never the body.
5. **Hand off.** Once the contract is set, run `/next` to build it test-first in mentor mode. Stop
   after the contract if I want to drive the build myself.
