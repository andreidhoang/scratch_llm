---
name: project-perf-curriculum-sprint
description: 2026-07-03 mandate — perf front autonomously finishes the entire performance/ curriculum A1 R4.2→A7; run policies (push every rung, lessons deferred); ncu blocked on the box
metadata:
  node_type: memory
  type: project
  originSessionId: e559c1e8-3a03-40db-810d-881a4279b964
---

The **perf front** of the 2026-07-03 two-front split (ADR-0013 delegate mode + ADR-0014 zones —
see [[project-cs336-delivery-sprint]]): autonomously execute the ENTIRE `performance/` curriculum
without the user — A1 R4.2→R4.6 + design note, A2 R0–R6, A3 R0–R2 + PTX artifact, A4 R0–R3+bwd,
A5 R0–R4+PTQ, all design notes; every sm_120-runnable rung MEASURED on the standing RTX PRO 4000
(sm_120, 24 GB, 0.55 TB/s). Anything needing H100/B200/8×H200 (A2§4, A3 R3+, A4 R4, A5§7, A6
Phase-4 serving day, A7) ships **compile-ready code + runbooks under `performance/rental/`** —
the user rents later and executes the runbook.

**Run policies (user-confirmed 2026-07-03):**
- Push to `origin main` after EVERY green rung commit (not just at milestones).
- Vietnamese lessons deferred entirely — each shipped rung appends a row to
  `docs/learning/INDEX.md` §Study queue (rung → commit → files → headline number); no teach-back
  waits.
- Per-rung protocol unchanged: pre-register in `bench/RESULTS.md` (D5) → test-first → measure →
  ledger `[FACT]` → adversarial review → commit+push → advance the PERF_PLAN node pointer.

**Box constraint `[FACT]` (verified 2026-07-03):** `ncu` hardware counters are BLOCKED in this
container (`ERR_NVGPUCTRPERM`, host-driver setting). `nsys` kernel traces WORK; `nvcc` 13.0 works.
All "Nsight SoL %" DoD gates re-based to CUDA-event timing + analytic bytes/FLOPs vs measured
peaks (0.55 TB/s, 72 TF/s); per-kernel "ncu debt" is listed in the rental runbooks (H100 day
discharges it — rental checklist includes verifying counters are enabled).

**Zone discipline:** perf agents write only `performance/`, `src/scratch_llm/serving/`,
`src/scratch_llm/kernels/`, `tests/` + `bench/` perf files; shared files (CLAUDE.md, STATUS.md,
pyproject.toml, bench/RESULTS.md, docs/learning/INDEX.md) get pull-rebase + additive edits +
precise `git add` (never `-A`) — the CS336 front works the same checkout concurrently.
