---
description: Audit an RL run against its v5 objective, policy alignment, verifier, controls and evidence contract.
argument-hint: "[path/to/run-logs] (defaults to the actual latest run)"
---
Read repository AGENTS.md/CLAUDE.md and the active ladders v5 contract. Route the run at
"$ARGUMENTS" to rl-run-auditor; if empty, identify the actual latest run under runs/, logs/
or checkpoints/.

Verify real updates, masks/reductions, behavior/old/current/reference policy versions, aligned
behavior logprobs, precisely named sampled estimators, reward/entropy/length distributions and
IS tails/ESS where applicable. Include verifier controls, held-out isolation, task-level
uncertainty and costs. A universal KL≈0.10 threshold or a merged "KL" label is not sufficient.

Return INTERPRETABLE, LIMITED or UNINTERPRETABLE FOR THE CLAIM, with supporting evidence and
specific missing or violated gates. Preserve measured facts and distinguish their limitations
from claims the run cannot support. An audit does not change the loss or tune thresholds.
