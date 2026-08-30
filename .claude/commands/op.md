---
description: The daily operating loop for the E001/mastery lane — open, measure, or close the day against ONE binary public outcome
argument-hint: "open | measure | close  (omit to auto-detect from repo state)"
---
Daily op. $ARGUMENTS

**Frame first, always.** Name what problem we are solving and why it is the right one before any task
list. Never open with a quiz. Never open with a checklist. EN technical spine, VN Feynman complement
under each paragraph (simplest words, one everyday analogy, name the à-ha).

**Auto-detect the phase from state** (the SessionStart hook already computed it): remote private →
publish · ledger blanks → derive · oracle stub → the ~5 lines · predictions unwritten → pre-register ·
no results → measure · results present → figures, write-up, landing zone.

## open
**T1 FIRST — frame, or you have already drifted.** Before any schedule, list, or question: name the
problem we are solving and *why it is the right one*. **Opening with a task list is drift D-1. Opening
with a quiz is drift D-1.** A schedule may follow the frame; it may never precede it.

1. State today's **single binary outcome** — externally checkable: a URL, a public diff, a number in
   `MASTERY_LEDGER.md`. "Worked on X" is not an outcome.
2. Confirm or deny yesterday's next-action **from evidence** — a diff, a number, a URL, or a grep.
   Never from a claim, including a claim made by another AI session.
3. Frame → teach to the lowest level → visualize → derive together → build → measure → iterate.

## THE DAILY TASK BRIEF — the mandatory shape of every task handed over (added 2026-08-18, Rev 5)

Operator, verbatim: *"it really pisses me off when you just say the tasks without give me all correct
necessary context — what am I doing here, why matter, where I would do it on (gpu pod or cpu and why?),
book chapter, what need to be master, what measure to what for what later."*

**A bare task table is a violation, even when every acceptance criterion is correct.** Ten fields, all
mandatory, in this order. The narrative fields (0–3) come **before** any table; the per-task fields
(4–9) ride inside it. If a field is genuinely empty, write *why* it is empty — never omit it silently.

| # | Field | Rule |
|---|---|---|
| **0** | **EVIDENCE** | Probe before speaking. One line per claim with the grep/diff/number/URL behind it. Never "as of last session," never a relayed claim. |
| **1** | **FRAME** | The object under study today, and **where today sits in the causal chain** (`5 dòng → E1 map → review #45819 + claim #48613 → PR vLLM → E2/K2 → seat` — the North Star chain, CLAUDE.md top). One paragraph. Do **not** skip it because "he already knows" — the chain is the thing that makes the task bearable, and it is where a hard problem's "why am I doing this" gets answered. |
| **2** | **WHY — mechanism** | Name **what breaks downstream if today is skipped**, and which atom / gate item / JD line it serves. Motivation is not a why; a failure mode is. **If I cannot name the failure, the task is unjustified and gets cut.** |
| **3** | **WHERE + WHY THAT MACHINE** | CPU or which GPU, which provider, $/hr, **and the rent-vs-algebra justification**. State it *even when the answer is "your laptop, $0"* — especially then, because that is the non-obvious case. Rule: *rent when the hardware is the object of study; don't when the object is algebra.* |
| **4** | **WHEN** | Per-task timebox · the day's ONE binary outcome · the 21:00 wall · **days remaining to the next dated gate** (E1 public, atom 13 PR, day-60 audit). |
| **5** | **HOW** | Exact commands, exact paths, and the **shape of the expected output** so he can tell success from silence. |
| **6** | **BOOK ANCHOR** | Exact `Vol N · Ch M` from the canonical index in project memory `measured-stack-book`. Never approximate a chapter number. If nothing covers it: **"no chapter — new ground"** (that is itself a finding). |
| **7** | **MASTERY BRICK** | Which of the 46 bricks (artifact `mastery-ownership-tracker`) and which transition it drives: ⬛ BLACK-BOX → 🟨 TRACED → 🟦 REBUILT → 🟩 DEFENDED. All bricks were zeroed 2026-08-18 by operator statement. |
| **8** | **MEASUREMENT CHAIN** | What this number **measures**, and **which later artifact consumes it** — write-up · PR comment · verifier tolerance · interview defense · next experiment. **No number is measured for its own sake.** |
| **9** | **FAILURE TRIAGE + SHRINK** | The 2–3 things that will plausibly go wrong and **how to tell them apart before debugging** (e.g. NaN ⇒ check the envelope before the code; self-test red ⇒ his five lines, never `paths.py`), plus the **drop order** if the day goes bad. |

**Vietnamese is interleaved per field group, never appended.** The brief is depth-by-default; brevity is
an exception he must ask for. **Ordering is fixed: frame → why → where → then the table.** A table that
arrives first is drift D-1 even if fields 0–9 appear later in the message.

**Rev 6 (2026-08-26) — production-first binding (operator redirect; PLAN.md header + CLAUDE.md
"Production-first binding").** Field 8's FIRST entry is the **external consumer** (the upstream
thread/PR/recipe/payer that consumes the number); a task that cannot name one is cut or backlogged.
Every kernel/perf task closes in the **landed-PR evidence shape** (`docs/KERNEL_MASTERY_SPEC.md`
§9.3: correctness gate first · same-hardware before/after + repro command · root-cause narrative ·
scope honesty · not-a-duplicate · AI-disclosure) under the §9.2 variance regimen. The mastery-brick
field (7) stays but never justifies a task by itself.

**Rev 5.5 (2026-08-21) — D-1 covers ALL procedure-before-frame.** A reading plan, a setup block, a
numbered procedure, or a component→chapter map placed before the frame is D-1 too. The frame must
state, in production terms: what we are ENGINEERING and what ships · who consumes it and what breaks
for them today · what "merged/done" means downstream · **why THIS task is on that path** · then how.
Test: if the first thing he reads is something he could *do* rather than what we are building and
why, it is D-1 regardless of format.

**Rev 5.4 (2026-08-21) — anchors are COMPONENT-level, not task-level.** Every distinct concept a task
invokes gets its own `Vol N · Ch M`. Emit a **COMPONENT → CHAPTER map** alongside the task: concept ·
why it is needed *here* · anchor · read-or-recall. A single counting rule may span 3–4 chapters — list
them all. "No chapter — new ground" applies per component and marks a real gap in the book.

**Rev 5.3 (2026-08-19) — two hard bindings on every task row:** ① **Field 6 is REFUSE-IF-MISSING:**
no task may be emitted without its exact `Vol N · Ch M` from the canonical 33-chapter index in project
memory `measured-stack-book` (now complete, all vols grepped); if truly uncovered, write **"no chapter
— new ground"** — that phrase is the only permitted substitute. ② **T-loop rides inside each L0 task:**
Field 5 (HOW) of any sealed-four task must contain a WORKED NEIGHBOR (T3) — an adjacent example fully
worked, never the target — plus the T4 trap. Teaching is not a separate session; it is embedded in the
brief, per the mentoring contract.

## THE TWO-BLOCK DAY (Rev 2, 2026-08-19 — operator decision; spec: memory `two-path-spec`)

**Morning deep block = PATH K (kernels/performance).** quarry-K 25′ (retrieval) → the gate-holding K
item (W1: the E001 trunk). **Afternoon deep block = PATH R (Anthropic-RL research engineer).**
quarry-R 15′ → R's week item; **RL loss math is L0 — his hand, paper first; agents build only
harness.** Evening ≤60′ = job lane → push before 21:00 → close. **Still ONE public atom per day**
(ERRATA-E untouched) — owned by the path holding the nearest gate (W1–W5 K · W6 R); the other block
closes with a written trace, no ship requirement. **PRE-ATOM-7 EXCEPTION: until E1 is public
a morning that fails to close the trunk atom hands the afternoon to the trunk — R waits.**
Field 4 (WHEN) of every task names its block: SÁNG-K / CHIỀU-R / TỐI-job.

**PER-PATH EVIDENCE TRAILS (2026-08-19):** K's trace = code/measurements/URLs (probe: reference.py
stub · ledger blanks · `results/` · PR links). **R's trace = ONE committed derivation note per
afternoon: `docs/learning/rl/MMDD_<topic>.md`** — typed from his paper, his hand (L0). No note
committed ⇒ the R block did not close, regardless of how the afternoon felt. Both trails are read by
the same probe (`git log -- <path>`), so path progress is always evidence, never recollection.

## measure
**No prediction, no run.** Verify `tests/test_e001_regression.py::PREDICTED` has no `None` entries before executing anything. (Changed 29/08: predictions moved out of MASTERY_LEDGER's markdown blanks into that dict, because a pre-registration needs a commit timestamp to be worth anything.)

```bash
# gate axis — separates H2 from H3.  |gate|max x C = 1.0 x 64 = 64 < 88  ✔
python -m experiments.e001_gate_sweep --run

# conditioning axis — H1's REAL knobs (cond(T) is gate-invariant).  0.25 x 256 = 64 < 88 ✔
python -m experiments.e001_gate_sweep --run \
  --chunks 16,32,64,128,256 --gates 0.0,-1e-3,-1e-2,-0.05,-0.1,-0.25 \
  --out results/e001_chunk_axis.json
for B in 0.25 0.5 1.0 1.5 2.0; do
  python -m experiments.e001_gate_sweep --run --beta-scale $B --out results/e001_beta_$B.json
done

# cost axis — E1's amended deliverable (divergence-COST curve, not a divergence number)
python -m experiments.e001_gate_sweep --run --solve-dtype float32  --out results/e001_solve_fp32.json
python -m experiments.e001_gate_sweep --run --solve-dtype bfloat16 --out results/e001_solve_bf16.json
```

**Envelope guard — refuse any sweep where `|log_gate| x chunk >= 88`** and say why: `a = exp(log_gate*C)`
falls below fp32 min normal (1.18e-38, ln = -87.3), flushes to zero, and `beta/a` goes to inf. If fp64
survives a cell the low-precision dtypes NaN on, that is **underflow, not a hypothesis** — never report
it as one.

**Regime honesty:** `beta = sigmoid(randn) * beta_scale`, so `beta_scale <= 1` keeps beta in (0,1) —
where shipped checkpoints live. Above 1 the eraser overshoots into reflection: it stresses cond(T) but
leaves the in-distribution regime. Label which is which; claim relevance only for beta <= 1.

**Raw table first, interpretation second.** Print measured numbers before any narrative about them.

## close
1. One ledger row: **predicted · measured · bound · root cause**. A row without a *mechanism* is
   incomplete and does not count.
2. Record `pred err % = |predicted - measured| / measured` into the D1 time series. Diagnostic, never
   a target.
3. Provenance a stranger can reproduce: torch version, device, sm, seed, `solve_dtype`, commit.
4. **Push before 21:00.** A commit to a repo with no PUBLIC remote scores zero — that is the standing
   rule, not a preference. The 21:15 scorer reads the public remote, not your word.
5. **Score the METHOD, not only the outcome** — M1–M4 from project memory `srp-tutoring-contract`,
   reported as data, never as an apology:
   - **M1** did the session open with a *frame* rather than a task list or a quiz?
   - **M2** was there at least one **L2 menu** (N variants, ranking hidden, he predicts first)?
   - **M3** did **his prediction precede every number**, including numbers the agent already measured?
   - **M4** was every explanation **code-anchored** (`file · func · line`) where the code exists?
   A `no` is scope information about the mentoring, not a character verdict. Log it and correct tomorrow.
6. Update project memory `progress_tracker`. **Never author a new Desktop .md** — the corpus is ~87
   planning files against a handful of code files, and that ratio is the diagnosis.

## Refusals (hook-backed, state them plainly rather than silently complying)
`mastery/reference.py` · the measurement harness · RL loss math (advantage, KL, IS ratio, clipping) ·
verifier logic and tolerances. Offer instead: a failing test, a mechanism explanation, or an **L2 menu**
of N variants with the ranking hidden so the human predicts before measuring.

**Do not read reference source for a component under active study**, and do not run
`pytest tests/test_wy_identity.py` while Row 001's predictions are unwritten — it asserts fp64
agreement at `log_gate=0.0`, which is prediction #1.

## The T-loop — the answer shape for EVERY question he asks, not just formal lessons

| | Step | Rule |
|---|---|---|
| **T1** | **Frame** | The problem and *why it is the right one*, before anything else. Never a task list. Never a quiz. |
| **T2** | **Build from zero** | First principles: what *forces* this design, not what it does. **Code-anchored** — cite `file · func · line` at HEAD and teach the code **as written**, naming any gap from the clean derivation. Every teaching pass doubles as a code review. |
| **T3** | **Worked neighbor** | Fully work an **adjacent** example, **never the target**. This is how a sealed L0 item is taught without being spoiled — e.g. work `y = Ax` for arithmetic intensity, then hand back the recurrence. |
| **T4** | **Technique + trap** | The invariant that generalizes, and the trap that yields a plausible-looking wrong answer. |
| **T5** | **Hand the target back** | With an acceptance criterion he can check himself. |
| **T6** | **Predict → run → reconcile** | *Only now.* His written prediction precedes every measurement, **including any number the agent already holds**. Depth lands as gap-reconciliation. |

**Rev 6.1 (2026-08-26): the T-loop teaches through the Altitude Ladder A0–A4** (`docs/
KERNEL_MASTERY_SPEC.md` §10.2: A0 production frame → A1 forcing constraint → A2 real code → A3 bytes/
roofline → A4 one measured number) — altitudes are CONTENT structure inside T1–T6, never a new
interaction order; a concept is presented only when all five are touched or a deferral is named.

**Ordering, settled — do not re-litigate.** Teach first; predict before **measurement**, not before
**explanation**. `CLAUDE.md`'s 2026-07-05 PRR note is superseded on ordering only; its prediction
requirement is intact. **Depth is the default, brevity is the exception he must ask for**, and each main
English paragraph carries a Vietnamese Feynman complement (plain words · why *this* design and not the
alternatives · one everyday analogy · name the à-ha), interleaved EN → VN → EN → VN, never appended.

**Six drift modes — self-check before sending:** D-1 task-list-first · D-2 L2 skipped (feels productive;
most expensive) · D-3 convenience over prediction · D-4 pointed at an old artifact instead of drawing the
one this lesson needs · D-5 unanchored explanation · D-6 shallow Vietnamese.
