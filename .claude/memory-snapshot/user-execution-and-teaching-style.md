---
name: user-execution-and-teaching-style
description: "How the user wants me to operate in scratch_llm — full senior-engineer execution, Vietnamese Feynman lessons one concept at a time"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3d47580f-f0bf-4bca-88a2-5a3c041ab8dd
---

The user (Huy Hoang Dang, danghuy19990804@gmail.com, GitHub andreidhoang) directs me to "execute as
senior AI research and performance engineer at frontier lab" — full end-to-end execution of perf-
curriculum systems rungs (spec → pre-register → build test-first → measure → ledger → commit),
including model-layer changes and Triton kernels that are reuse-of-owned-work (FOP-7).

**Superseded 2026-07-03:** the Mode-3 human-implements boundary is now fully suspended — the user
switched the repo to `delegate` execution mode ("bypass all rules of me building by hand … deliver
value first, master after seeing the code"). Agents implement EVERYTHING end-to-end, fresh kernel
reps and RL math included; mastery is post-hoc (study queue + [[project-cs336-delivery-sprint]]).
Do not refuse implementation work on Mode-3 grounds unless `.claude/execution-mode` is flipped
back to `learn`. Front plans: [[project-perf-curriculum-sprint]] (perf) ·
[[project-cs336-delivery-sprint]] (CS336 main track).

**Why:** they twice confirmed full delegation ("yes let execute...") and accepted R3b/R4.1 built
this way; the repo constitution's Mode-3 clause is for the reps they are personally learning.

**How to apply:** on a new rung, pre-register in bench/RESULTS.md first, build without asking
permission per step, commit with the green gate, and **push to origin main after every green rung
commit** (standing policy since 2026-07-03; gh CLI is authenticated on the box; repo-local
credential.helper is set to `!gh auth git-credential`).
For teaching: Vietnamese, Feynman + teach-back gate, ONE component per lesson, wait for "move on";
each lesson becomes a durable doc in [[docs/learning]] (`docs/learning/INDEX.md`, line anchors
pinned to a commit hash). Ad-hoc analyses/reports/explanations are also delivered in Vietnamese
(technical terms stay English) — requested explicitly again 2026-07-03.
