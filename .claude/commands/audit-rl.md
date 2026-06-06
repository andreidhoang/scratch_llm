---
description: Audit the latest RL run's logs for mandatory guardrails before trusting the numbers
argument-hint: "[path/to/run-logs]  (defaults to latest run)"
---
Route to the **rl-run-auditor** subagent: audit the RL run at "$ARGUMENTS"
(if empty, find the latest run under `runs/`, `logs/`, or `checkpoints/`).

Verify the mandatory guardrail logging: the three KLs logged SEPARATELY
(`KL(current‖ref)`, `KL(current‖old)`, `kl_train_infer`), IS-ratio histograms + ESS, reward
distribution stats, completion-length stats, and the four ship metrics
(`reward / hack_rate / true_quality_gap / kl_train_infer`). Check `kl_train_infer ≤ 0.10` (HALT
threshold). Report **INTERPRETABLE** or **UNINTERPRETABLE / HALT** with the exact missing or violated
items, in priority order. A run missing guardrails is no result — say so plainly.
