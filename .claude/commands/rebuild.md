---
description: The Rebuild Ladder — re-derive and re-type one agent-built kernel from blank in mastery/, against the existing oracle tests and ledgered numbers. Learn-mode ALWAYS. The agent never writes into mastery/src.
argument-hint: "(optional) ladder rung L0..L8 or a kernel name; default = current rung"
---
Rebuild rep. $ARGUMENTS

> **Why this command exists.** The production kernels were agent-built under delegate mode (ADR-0013);
> the ledger numbers are real but the human's skill is not yet (MASTERY_DEBT: 4/89). Evidence
> (roadmap §6.5): memory forms from what YOU generate (Anthropic RCT 2601.20245: 50% vs 67%, d=0.74);
> so here the human types EVERYTHING inside `mastery/src/mastery_llm/`. Claude's only jobs: Socratic
> derivation, timed worked-example interrogation, running tests/benches, diff cross-examination,
> grading, ledger. **HARD RULE: never Write/Edit under `mastery/src/**` — that directory is
> human-only, same status as JOB_SPRINT `attempts/`.**

## The ladder (dependency-ordered; ≈10.5 blocks ≈ Jul 15 → Aug 1)

| Rung | Target (blank rebuild) | Oracle / grader | Blocks |
|---|---|---|---|
| L0 | GPU execution + memory model — PAPER: draw SM/warp/hierarchy from memory; derive divergence cost; derive the ridge point for H100 (~295 FLOP/B) and this box (72 TF/s ÷ 0.55 TB/s ≈ 131) | Claude cross-examines the drawing | 0.5 |
| L1 | Roofline re-derivation: pick 5 rows from `bench/RESULTS.md`, hand-compute bytes/FLOPs/bound/% of peak, then reveal | ±20% + correct bound on ≥4/5 | 0.5 |
| L2 | GEMV ladder (naive → coalesced → vectorized) | `mastery/tests` + >80% of 0.55 TB/s | 1 |
| L3 | Fused reduction → softmax | `test_softmax_triton.py` | 1 |
| L4 | RMSNorm fused | mastery test + HBM% | 0.5 |
| L5 | Tiled smem GEMM (+ register tiling) | vs cuBLAS-proxy ledger row | 2 |
| L6 | Online softmax (paper+numpy) → FA2 fwd inner loop → FA2 **bwd derivation on paper** (D-vector, atomic dQ — MASTERY_DEBT row 13) | `test_flash_attention_triton.py` | 2.5 |
| L7 | Decode-attention / paged step (the DELTA axis) | paged oracle + step-time vs ledger | 1.5 |
| L8 | Quant numerics by hand: FP8-E4M3 grid, NVFP4(16,E4M3) vs MXFP4(32,E8M0) block-scale error; verify in torch | derivation matches measured A5 row | 1 |

## The 6-phase loop (per rung — this is the whole method)

1. **DERIVE (paper, 15–30', AI Socratic-only).** From the physical problem: what bytes move, what's
   the bound, what invariant makes it correct. No code visible. If stuck, `/tutor`-style hints only.
2. **WORKED-EXAMPLE STUDY (timed 20–30', then CLOSED).** Open the agent's implementation in
   `src/scratch_llm/kernels/` as a worked example. The human annotates aloud: why each line, one
   interrogation finding (something suboptimal, surprising, or unexplained — §6.5 evaluation-gap
   rule). Then the file CLOSES and stays closed.
3. **BLANK REBUILD (the rep).** Human types into `mastery/src/mastery_llm/…` from an empty buffer.
   Timer per ladder table. Claude may only: run `PYTHONPATH=mastery/src pytest mastery/tests/<test>`,
   report failures VERBATIM (no fixes, no hints unless explicitly downgraded to Socratic), and keep
   time. Compile/test errors are the human's to read first.
4. **PREDICT → BENCH.** Before running the bench: write the predicted number (% of peak, ms, or
   bytes). Run it. Compare against the LEDGERED number from the agent's version. Gap >20% → the human
   names the cause before looking at anything.
5. **DIFF-DEFEND (the unique asset).** `diff` the human's rebuild vs the agent's production version.
   For every material divergence the human must either (a) defend theirs as equivalent/better, or
   (b) explain precisely why the agent's choice wins (coalescing, bank conflicts, masking, numerics).
   This turns the agent code from crutch into grader — an asset no from-scratch beginner has.
6. **FEYNMAN + LEDGER.** `/feynman <rung>` (grade ≥4/5 avg). Append to
   `docs/learning/MASTERY_DEBT.md`: date · rebuild · rung · test result · bench Δ · grade · next-due
   (+48h paper re-derivation of the invariant, +1wk blank re-type of the inner loop). Mark the
   ladder rung done in roadmap §4 P0.5. A failed gate re-enqueues — it never silently passes.

## Session shape

Orient (which rung, from the last MASTERY_DEBT entry) → run the 6 phases → close with one line:
"L<n> <kernel> — test <green/red>, bench <yours> vs <ledger>, feynman <grade>; ladder x/9."
If the human asks Claude to write or fix anything under `mastery/src/`: refuse, offer a failing
test, a question, or a post-hoc review. That refusal is the product working.
