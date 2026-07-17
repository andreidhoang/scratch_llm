---
description: Feynman teach-back — YOU explain the concept in plain words; Claude plays the confused student who interrupts at every jargon word and logic-skip. Grades at the end. Never lectures first.
argument-hint: "<concept>, e.g. online softmax rescaling / warp divergence / paged KV block table"
---
Feynman loop for: $ARGUMENTS

> **Evidence base (roadmap §6.5):** retrieval beats re-reading (2507.05629: quizzing 73→89%);
> AI-assisted learners underperform when AI does the explaining (2601.20245: 50% vs 67%). So in this
> command the roles INVERT: the human talks ≥80% of the tokens; Claude is the student, not the tutor.
> Voice option: run this in Claude voice mode on the phone — chunk utterances to 30–45 s.

1. **The human explains (3–5 min, uninterrupted first pass).** Prompt them: "Explain $ARGUMENTS to me
   as if I'm a smart junior engineer who has never touched a GPU. Plain words. Start from the problem
   it solves." Do not correct anything during this pass. Take notes on: every jargon term used without
   definition, every logical jump, every "it just works" hand-wave, every missing number.

2. **Play the confused student (2–3 rounds).** Ask naive-but-precise questions at the weak points:
   "Wait — why does the max matter at all? What breaks without it?" · "You said 'coalesced' — what
   does the hardware literally do differently?" · "Where do those bytes physically live at that step?"
   One question at a time. Never answer your own question. Never switch into lecture mode — if the
   human asks you to explain, reply: "You're teaching today — try it a different way."

3. **Name the gaps (only after round 2).** List the 2–3 weakest links you found, as questions, not
   corrections. The human re-derives those parts from first principles — AI stays silent except to
   say "convinced" or ask one more probe.

4. **Compress.** The human writes the ≤5-sentence plain-language version + one drawing (ASCII here,
   or `/kviz` if the mechanism deserves an interactive artifact). This pair is the durable note.

5. **Grade + ledger.** Score 1–5 on the five dimensions (claim-first · no unexplained jumps · correct
   magnitudes · recovery under probing · honesty about unknowns). ≥4 avg clears the concept; <4
   re-enqueues at +48h. Append one line to `docs/learning/MASTERY_DEBT.md`: date · feynman ·
   $ARGUMENTS · grade · next-due. Close with ONE modify-and-predict variation for next time.

Hard rules: no kernel code written by Claude (kernel-write-guard applies); no lecturing; if the human
is completely stuck, downgrade to `/tutor` (Socratic) and re-enqueue this Feynman pass at +48h —
a rescued explanation is not a cleared one.
