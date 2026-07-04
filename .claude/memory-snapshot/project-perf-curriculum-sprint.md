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

**DONE (2026-07-04, this session — ALL committed + pushed to origin main): the entire sm120-runnable
A1–A5 curriculum is COMPLETE, measured, ledgered, tested, green.** Built A2/A3/A4/A5 kernel rungs via
parallel Workflow orchestration (build → oracle-test gate → adversarial verifier → main-thread gpu
re-test → commit). Headlines: A1 R0–R4.6 serving (CUDA-graph decode 77% of the memory wall = R1 gap
closed; R4.2 chunked-prefill an honest NEGATIVE — sequential-interleave regresses, R4.2b piggyback
deferred); A2 R0–R6 kernels (GEMV/softmax/norms 96–100% HBM, TopK 46.9% + 3.4× fusion, GEMM 134%
cuBLAS-proxy); A3 R0–R2 tensor cores in CUDA (4.1%→38.9% WMMA→81.9% mma.sync of cuBLAS, element-exact);
A4 R0–R3+bwd flash attn (FA2 ~50% SDPA, 44× leaner, gradcheck); A5 R0–R4+§4.3 quant (NVFP4 1.48×<MXFP4,
FP8-KV E2E 24.45 dB, AWQ 1.71× recovery). Design notes A1–A5 written. Verification: ruff + pyright (0
errors) + CPU suite + full gpu suite all green. **Remaining = ONLY the ISA-gated rental days**
(WGMMA/TMA/FP8→H100; tcgen05/NVFP4→B200; 8×H200 serving day), code-only + runbook'd
(`performance/rental/{H100,B200,serving_day_8xH200}_day_runbook.md`, WGMMA PTX artifact) — rental
discipline satisfied (every sm120 prereq ledgered). Pyright note: new GPU-only kernels are in the
pyproject.toml pyright exclude (Triton launch syntax + JIT CUDA can't be statically typed).

**Mastery roadmap + context updated (2026-07-04):** `docs/learning/roadmap/` — 41 Bài / 6 series (VI,
first-principles, traced to file·func·line pinned to a commit) walking through every core perf file; the
delegate-mode "ship first, master after" study map (INDEX.md leads with it). CLAUDE.md (two-fronts +
build-status + "Where things live") + CONTEXT_ENGINEERING.md updated: perf front marked COMPLETE, roadmap
in the Lever-2 knowledge base, and §3.3 documents the delegate-mode BUILD fan-out pattern (Workflow
one-agent-per-rung + three-gate pipeline: oracle test → independent adversarial verifier → main-thread
re-test → commit; main thread is the sole committer). A fresh session orients to "sm120 done, only rental
days remain" from the SessionStart hook (PERF_PLAN Current Node) + STATUS + CLAUDE.md.

**Zone discipline:** perf agents write only `performance/`, `src/scratch_llm/serving/`,
`src/scratch_llm/kernels/`, `tests/` + `bench/` perf files; shared files (CLAUDE.md, STATUS.md,
pyproject.toml, bench/RESULTS.md, docs/learning/INDEX.md) get pull-rebase + additive edits +
precise `git add` (never `-A`) — the CS336 front works the same checkout concurrently.
