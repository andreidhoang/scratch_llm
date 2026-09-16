---
description: Inspect a kernel benchmark or profiler report and propose one controlled experiment; diagnosis only.
argument-hint: "(optional) kernel name or path to an ncu/nsys report"
---
Profile and diagnose. $ARGUMENTS

1. Read repository AGENTS.md/CLAUDE.md and the active ladders v5 contract. Identify the existing
   prediction, numerical contract, matched baseline and measurement budget.
2. Inspect an existing report or use the active rung benchmark when its prerequisites and
   authorization are satisfied. Reuse bench/kernels/run.py for the selected backend as appropriate.
   Record actual cache/timing/profiling behavior; a full ncu trace is not always needed.
3. Route to roofline-analyst for the suspected bound, supporting evidence, uncertainty and one
   discriminating experiment. Huy owns the prediction and causal diagnosis.
4. A profile request does not authorize a core rewrite. Agents may implement a fix after Huy names
   it within an implementation task. The archived execution-mode switch is not active.
