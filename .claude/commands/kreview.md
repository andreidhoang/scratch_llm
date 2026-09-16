---
description: Review a kernel's numerical and measurement evidence against its v5 contract; does not rewrite the kernel.
argument-hint: "(optional) note on what the kernel is"
---
Kernel review. $ARGUMENTS

1. Read repository AGENTS.md/CLAUDE.md, nested kernel instructions and the active ladders v5
   contract. Inspect git diff / git diff --staged.
2. Route to kernel-ship-reviewer for oracle independence, numerical contract, actual dispatch,
   matched baseline, raw timings/profile and the supported scope of each claim.
3. Run relevant checks from CLAUDE.md and the active rung when available. Report skipped or
   unexecuted GPU gates explicitly; a correct null result does not become a failed scientific
   observation merely because it misses a performance target.
4. Return the verdict, concrete fixes and proposed commit message. Huy owns first core
   implementations and diagnosis; agents implement a named diagnosis within an authorized
   change task. No historical execution-mode switch overrides that boundary.
