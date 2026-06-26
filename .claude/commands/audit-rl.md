---
description: Audit the latest RL run's logs for mandatory guardrails before trusting the numbers
argument-hint: "[path/to/run-logs]  (defaults to latest run)"
---
Route to the **rl-run-auditor** subagent: audit the RL run at "$ARGUMENTS"
(if empty, find the latest run under `runs/`, `logs/`, or `checkpoints/`).

Verify the mandatory guardrail logging: per-token entropy, the KLs logged SEPARATELY
(`KL(current‖ref)`, `KL(current‖old)`, and `kl_train_infer` when a separate serving engine is used),
IS-ratio histograms + ESS, reward distribution stats, and completion-length stats. Report
**INTERPRETABLE** or **UNINTERPRETABLE** with the exact missing or violated items, in priority order.
A run missing guardrails is no result — say so plainly.
