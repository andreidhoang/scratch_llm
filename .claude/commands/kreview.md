---
description: Gate the kernel I just wrote before commit — correctness vs oracle, numerics, is-the-speedup-real, profile-produced. Returns ACCEPT/REJECT. Does not rewrite the kernel.
argument-hint: "(optional) note on what the kernel is"
---
Kernel ship review. $ARGUMENTS

1. **Diff.** Show `git diff` / `git diff --staged` (read-only).
2. **Gate.** Route to the **kernel-ship-reviewer** subagent: correctness vs oracle, numerics (fp32 accumulate / bf16 edges), speedup-is-real (the Sakana check), profile-produced (the DoD).
3. If **ACCEPT**: run green-CI (`ruff check src tests`, `ruff format --check src tests`, `pyright`, `pytest -m "not gpu"`) and propose the commit message (`kernels: <imperative>`) — you run `git commit`, not me. If **REJECT**: give me the specific failing item — **I** fix it.
