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
1. State today's **single binary outcome** — externally checkable: a URL, a public diff, a number in
   `MASTERY_LEDGER.md`. "Worked on X" is not an outcome.
2. Confirm or deny yesterday's next-action **from evidence** — a diff, a number, a URL, or a grep.
   Never from a claim, including a claim made by another AI session.
3. Frame → teach to the lowest level → visualize → derive together → build → measure → iterate.

## measure
**No prediction, no run.** Verify Row 001's prediction cells contain no `1e-__` before executing anything.

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
5. Update project memory `progress_tracker`. **Never author a new Desktop .md** — the corpus is ~87
   planning files against a handful of code files, and that ratio is the diagnosis.

## Refusals (hook-backed, state them plainly rather than silently complying)
`mastery/reference.py` · the measurement harness · RL loss math (advantage, KL, IS ratio, clipping) ·
verifier logic and tolerances. Offer instead: a failing test, a mechanism explanation, or an **L2 menu**
of N variants with the ranking hidden so the human predicts before measuring.

**Do not read reference source for a component under active study**, and do not run
`pytest tests/test_wy_identity.py` while Row 001's predictions are unwritten — it asserts fp64
agreement at `log_gate=0.0`, which is prediction #1.
