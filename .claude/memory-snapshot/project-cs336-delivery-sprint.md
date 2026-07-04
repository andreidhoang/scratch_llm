---
name: project-cs336-delivery-sprint
description: CS336 main track A2→A5 SHIPPED 2026-07-04 (delegate mode, autonomous); graded GPU runs rental-gated; two-front zone split with perf agents
metadata: 
  node_type: memory
  type: project
  originSessionId: 11074331-4b6f-4b06-9b7b-20bc6c37a2a3
---

**STATUS 2026-07-04: SPRINT COMPLETE.** All exec-spec nodes W1–W11 shipped. A2 (ZeRO-1/FSDP/100B
one-pager/comms-algebra) · A3 (IsoFLOP fitter + query planner) · A4 (dedup/filters/quality/pipeline)
· A5 (SFT/EI/GRPO/Dr.GRPO/DPO + grader + envs) are code-complete, green (461 CPU tests), and
official-scaffold-accepted (50 pass / 0 fail / 11 explained-blocked — `deploy/runbooks/OFFICIAL_SUITES.md`).
What remains is **rental execution** (graded Qwen2.5-Math-1.5B RL runs, R1-Zero "aha", multi-GPU NCCL
benchmarks) — code + runbooks in `deploy/runbooks/`, run when a box is rented — and **post-hoc
mastery** (`docs/learning/MASTERY_DEBT.md`, 12 rows). Next real build frontier: DELTA capstone (still
gated) and the A5 distillation lab (`algos/distill.py`, open).

---

On 2026-07-03 the user switched the repo `/workspace/scratch_llm` to **delegate execution mode**
(`.claude/execution-mode`, ADR-0013): agents implement everything end-to-end (loss bodies, kernels
included); the old teach-back/Mode-3 refusal rules no longer block builds. Mastery is post-hoc via
`docs/learning/MASTERY_DEBT.md` + Vietnamese Feynman lessons written from shipped code.

The **CS336 main-track finish** (A2 ZeRO-1/FSDP/one-pager/comms-algebra → A3 → A4 → A5) runs as an
autonomous sprint (ADR-0014) with the canonical plan in
`docs/EXECUTION_SPEC_CS336_FINISH.md` (task DAG W1–W11, per-node DoD). Work the checklist without
asking the user; tick nodes + push after each.

**Why:** user changed strategy to "deliver value first, master after seeing the code" — save
wall-clock, stop gating builds on his hand-building.

**How to apply:**
- Never write in the perf agents' zone: `performance/`, `src/scratch_llm/serving/`,
  `kernels/paged_decode_triton.py`, perf sections of `bench/RESULTS.md` (other agents own it,
  possibly concurrently in the same checkout).
- Shared files (CLAUDE.md, STATUS.md, pyproject.toml, bench/RESULTS.md): pull-rebase before
  commit, additive edits, never `git add -A`.
- Tasks needing >24 GB / multi-GPU (H100, B200, 8×GPU node): write code + tests (CPU/gloo) now,
  plus a self-contained runbook in `deploy/runbooks/`; the user rents later and runs it.
- Official CS336 scaffolds (spec + adapter-test oracle) live at `/workspace/lectures/` — recloned
  2026-07-03; reclone from github.com/stanford-cs336 if a fresh pod loses them.

Related: [[user-execution-and-teaching-style]]
