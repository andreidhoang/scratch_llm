---
description: Daily P0 kernel-quarry rep (KERNEL_ROADMAP_2026 §6) — run ONE of Q1–Q6 on an already-built kernel; AI scaffolds, sabotages, quizzes, grades; NEVER implements. 30–45 min, opens the 14:00–18:30 kernel block (mornings belong to the RL lane — JOB_SPRINT §8.1 dual-block pairing).
argument-hint: "(optional) Q1..Q6 and/or a kernel, e.g. 'Q4 paged_decode'"
---
Quarry rep. $ARGUMENTS

> **Learn-mode rep, always.** Regardless of `.claude/execution-mode`, in this command the human types
> every kernel body and every derivation (the interview is AI-prohibited; an agent doing the rep fails
> the task even if the code is perfect — JOB_SPRINT prime directive). Complements `/kernel-day` (full
> rung reconstruction): quarry reps are the daily warm-up on **already-built, already-ledgered** kernels.
> Zero new build — P0 of `performance/KERNEL_ROADMAP_2026.md` (§6 has the rep table + gates).

1. **Orient (2 min).** Read `performance/KERNEL_ROADMAP_2026.md` §6 + the tail of
   `docs/learning/MASTERY_DEBT.md`. Pick with the human: ONE rep × ONE kernel. Rotation default if
   unspecified: Q1 Mon/Thu · Q2 Tue · Q3 Wed · Q4 Fri · Q5 Sat · Q6 Sun. Honor the spaced queue
   (+48h/+1wk re-enqueues) before the rotation.

2. **Set up (per rep):**
   - **Q1 blank-page ladder** — name the rung + timer (softmax 20' · GEMM 30' · FA2 inner loop 45' ·
     paged-decode step 45' · GEMV ladder 30'); confirm its oracle test exists; **PAUSE — human rewrites
     from memory in their editor.** If asked for help: switch to kernel-tutor behavior, Socratic only.
   - **Q2 predict-the-number** — pick a random ledgered kernel row from `bench/RESULTS.md`; do NOT
     reveal the number; demand bytes/FLOPs, bound classification, and predicted %-of-peak; then reveal
     and compare. Gate: ±20% + correct bound.
   - **Q3 teach-back** — oldest unpaid `MASTERY_DEBT.md` row; 10-min spoken/written explanation; then
     ONE modify-and-predict variation; grade 1–5 on the JOB_SPRINT rubric (claim-first · no unexplained
     jumps · correct magnitudes · recovery under probing · honesty). ≥4 avg clears the row. For the
     full inverted loop (human teaches, Claude plays confused student) run `/feynman <concept>`.
   - **Q4 sabotage drill** — COPY the target kernel to `performance/drills/<YYYYMMDD>_<kernel>_sabotaged.py`
     (**never touch real kernel files** — `kernel-write-guard.sh` enforces this anyway); plant 1–3
     realistic bugs (race / missing `__syncthreads` / bank conflict / wrong mask / silent dtype cast /
     removed fence / off-by-one tile); tell the human only the SYMPTOM (wrong output or slow profile).
     25' timer. Gate: root cause **named before** the fix; one postmortem sentence ledgered.
   - **Q5 PTX/SASS rep** — human generates PTX/SASS for one owned kernel and annotates 10 lines; you
     cross-examine exactly one annotation. Gate: one concrete compiler finding ledgered.
   - **Q6 frontier read** — serve this week's primary source from roadmap §8; human writes a 5-line
     summary tying it to a rung; cross-examine it against the source. Interrogation pass (§6.5): the
     human must name one flaw or missing caveat in any Claude-provided summary before accepting it.

3. **Gate + ledger (never skip).** Grade against the rep's gate (roadmap §6). Append ONE line to
   `docs/learning/MASTERY_DEBT.md` (or the MASTERY_LEDGER if present): date · rep · target · result ·
   grade · next-due. A failed gate re-enqueues at +48h — it never silently passes. If the human asks to
   waive a gate: push back once (interviews are AI-off), then require an explicit override and log it.

4. **Close (1 line).** "Q<N> <kernel> — <grade>; kernel teach-backs at X/89." Then close out — the
   21:00 debrief grades this rep alongside the morning RL rep (JOB_SPRINT loop).
