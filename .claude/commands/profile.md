---
description: Run the kernel's bench/roofline (or import an ncu report) and get a structured bottleneck diagnosis — bound, why, the single next experiment. No code fixes.
argument-hint: "(optional) kernel name or path to an ncu/nsys report"
---
Profile + diagnose. $ARGUMENTS

1. **Measure.** Prefer `kernels/bench.py` (`matmul_roofline` / `roofline`); print the DoD line. Use `ncu --set full -o profile/ncu_$(date +%F).ncu-rep <cmd>` only when a full trace is wanted.
2. **Diagnose.** Route the result to the **roofline-analyst** subagent → BOUND / WHY / NEXT EXPERIMENT / PREDICT. (It returns a short summary, not the raw report.)
3. Hand over the next experiment — per `.claude/execution-mode` (ADR-0013): `delegate` (current) →
   Claude implements it; `learn` → the human implements it, do not write the fix.
