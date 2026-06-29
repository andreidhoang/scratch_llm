---
description: End-of-day close — make today's state durable, name tomorrow's single node, then reset the window for a clean start
argument-hint: "(optional) a note on what landed / what got falsified today"
---
Daily close (the `docs/OPERATING_RHYTHM.md` Close). $ARGUMENTS

1. **What landed / what was falsified.** One honest line. Label `[FACT]` (measured/green) vs
   `[INFERENCE]` (implemented but unmeasured). A cleanly falsified hypothesis is a valid day's output.
2. **Make state durable.** If code landed, update `docs/STATUS.md` (what is green *now*) and route the
   diff through `/ship` (review → green-CI → you run the commit). Do not leave green work uncommitted.
3. **Capture the non-obvious.** If today produced a hard-won, non-obvious fact (a numeric prediction
   that held/broke, a gotcha not encoded in the repo), write it to project memory — *not* what the
   code/git already records.
4. **The four-metric glance (terse).** active-node = 1? · doc-to-code ratio sane (no new doc without a
   commit)? · every run predicted-before-run? · perf/RL claims measured, not just implemented?
   Flag any that drifted in one line each.
5. **Name tomorrow's single node** (so `/standup` starts warm) — the next EV-ranked open node.
6. **Reset.** Remind me to `/clear` — external memory (`STATUS.md`, memory, the repo) carries state;
   a clean window tomorrow beats a rotted one. Do not `/clear` for me.
