---
description: Drive today's kernel rep from BOOK_SPRINT — name the rung to reconstruct, scaffold its test/bench, then PAUSE for you to implement; profile + review after. Never implements the kernel.
argument-hint: "(optional) the kernel/rung to work, e.g. matmul_tiled"
---
Kernel day. $ARGUMENTS

Read `src/scratch_llm/kernels/CLAUDE.md` (the meat boundary) first, then:

1. **Orient.** Get today's kernel target — from `BOOK_SPRINT_2026.md` (the Week 2/3 grid in the private `interview_synthesis/` folder) if it's present; **on vast.ai it isn't cloned, so I'll ask you for today's rung + the % target** — plus `docs/STATUS.md` (in this repo). Name the ONE kernel/rung to reconstruct today and the predict-the-% target. Keep it to one.
2. **Scaffold.** Route to the **bench-writer** subagent → ensure the failing test + the `bench.py` roofline call exist for that kernel. Hand me the target number + the one bench command.
3. **PAUSE — I reconstruct the kernel from blank in my editor.** Do NOT write it. If I ask you to, switch to **kernel-tutor** (`/tutor`) and ask what I tried first.
4. When I say done: `/profile` (→ **roofline-analyst**), then `/kreview` (→ **kernel-ship-reviewer**).
5. **Close.** Confirm the roofline line is logged (the DoD) and the day-card `Kernel roofline` field is filled. One line: "naive → today's **% of cuBLAS/SDPA**".
