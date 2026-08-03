# Frontier 2026 — The Scaling-Law Program of Record (fit engineering + the K3 decision path)

> **Doc role.** `SCALING_LADDER.md` owns the *sequencing* methodology (calibrate → ablate →
> family fit → validate → freeze → re-fit → flagship); `ARCH_SCALING.md` owns the *one-backbone
> doctrine* (offsets vs exponents, anchors vs re-sweeps). This doc owns the missing third layer:
> **the metrology of the fit itself** — estimator, grid geometry, horizon discipline,
> uncertainty quantification, extrapolation honesty — redesigned from first principles after
> the S3 R²-gate failure, and sequenced so the program terminates in the K3 architecture
> decision. Authored 2026-08-03. Companion: `FRONTIER_2026_D20_CERTAINTY_PLAN.md` (P5/d20),
> `docs/k3/ABLATIONS.md` (R0–R2 arms).

---

## §0 Why this doc exists

S3 bought 12.09 GPU-h of measurements and **zero law**: the pre-registered R² gate fired, the
fit was rejected, and the D:N decision correctly fell back to HOLD-by-economics. That was the
right call for the $100 d20 (bounded risk, shallow basin). But it leaves the lab with no
working scaling law of its own — and the K3 front has a hard dependency on one: the GQA-vs-KDA
**family fit is the attention-architecture decision** (SCALING_LADDER Rung 2). If we rerun the
same measurement design on the K3 families, we will buy the same failure twice at K3 prices.

The S3 failure was not bad luck and not a recipe bug. It was a **measurement-design failure**:
wrong grid geometry for the estimator, too-narrow span, winner's-curse point selection, and —
quietly — an unswept LR recipe underneath every point. Each is fixable, cheaply. This doc is
the fix, specified to rung level.

---

## §1 What a frontier-grade fit actually requires (the measurement spec)

Everything below is either literature-verified (cited from the repo's own reference set) or
labeled *our-judgment*.

### 1.1 The estimator: joint parametric fit, with the replication fixes baked in

- Fit `L(N,D) = E + A·N^(−α) + B·D^(−β)` jointly on **all** points (Hoffmann Approach 3),
  with the Besiroglu-et-al. corrections that made Chinchilla's Method 3 reproducible:
  **Huber loss (δ = 1e-3) on log-loss residuals, summed not averaged; L-BFGS-B from a grid of
  initializations**, not a single start (arXiv:2404.10102; `ARCH_SCALING.md` §1.1).
- **Why not Approach 2 (isoFLOP profiling):** it requires several (N, D) splits *at the same
  C* per budget. Our grid has exactly one same-C pair (s6 vs s7). Building split series at
  4–5 budgets costs multiples of the joint-fit grid below. (*our-judgment*, cost-driven.)
- **Fit axis: val_bpb on the pinned val set, never raw CE, never CORE** (tokenizer-invariant;
  CORE's ±0.008–0.016 noise floor is fit-poison — nanochat discussion #481).
- **Log-space residuals, never raw** (heteroscedasticity; `isoflop.py` docstring lesson).
- **D is the recorded token count, never the C/(6N) bridge** — the bridge makes a+b ≡ 1
  identically and mutes the driver bugs the a+b diagnostic exists to catch
  (`s3_sweep.py` docstring). a+b is *reported* as a diagnostic, never enforced.

### 1.2 Grid geometry: the two design rules S3 violated

1. **Match the grid to the estimator.** Per-budget min-picking (Approach-2-style selection on
   a grid that wasn't built for it) is what produced the D-zigzag (0.16→0.40→0.80→0.47→1.19→
   2.37→1.08B): ratios cycled 8→20→40 within each depth ray, so the min-picked D crossed depth
   boundaries and no power law could fit. The old design rule ("vary ratio within one depth,
   or depth within one ratio, never both in a cycle") binds **only for min-pick estimators**.
   The joint fit *relaxes* it — more ratio levels per depth = better D-coverage — but demands
   instead: ≥1.5× more points than free parameters, no duplicated (N, D) cells, and leverage
   spread across the span. (*Precise statement; the old rule stays for any Approach-2 work.*)
2. **Span ≥ 2 orders of magnitude in C, with the top end leaning toward the target.** S3
   spanned 1.92e16 → 8.79e17 = **45×** — at that span, exponent estimation is leverage-
   dominated (dropping s7 swung ratio@d20 from 26.9 to 78.2). Frontier fits run 100–1000×
   spans (DeepSeek: 8 budgets; MiniMax: 70M–7B; Llama 3: 6e18–1e22). We cannot buy 1000×, but
   we can buy ~235× (§3) and we must *say* what our span can and cannot certify.

### 1.3 Horizon and schedule discipline (the silent confound)

- **Every grid point is trained to its own registered budget with the full production schedule
  shape** (same warmup fraction, decay to a fixed floor fraction). Intermediate checkpoints of
  a longer run are NOT iso-budget points — a partially-decayed run's mid-loss is schedule-
  confounded. (Standard practice; Gadre et al. overtraining work trains each point at its
  horizon.) Single-run/multi-horizon tricks (WSD-style) exist in the literature; **rejected
  for the record** — our points are cheap enough to do it right. (*our-judgment*.)
- **LR is part of the recipe backbone, not a nuisance knob.** A scaling law is only defined on
  a frozen recipe = {corpus, tokenizer, optimizer **+ LR/wd**, schedule shape, precision, ctx}
  (`ARCH_SCALING.md` §0). S3's points were run at a *borrowed, never-swept* LR — they are
  **recipe-v0** measurements. P5 (the LR sweep) is therefore not just a d20 rehearsal; it is
  the **precondition for every fit in this program**, and it creates the recipe-v0 → recipe-v1
  reconciliation problem handled in §3 stage 1.

### 1.4 Uncertainty quantification is part of the fit, not an afterthought

- **Nonparametric bootstrap over runs** (resample points with replacement, refit, ≥1000 reps)
  ⇒ CIs on α, β, and on the *prediction at the d20's C* — the number that matters.
- **Hold-one-out stability** (leave each point out, refit, record the swing at target C) — the
  generalized form of today's 26.9↔78.2 sensitivity alarm.
- **Residual structure check:** residuals vs depth must show no monotone/zigzag trend (the
  signature S3 exhibited). A high R² with structured residuals is still a rejected fit.
- **Acceptance gates, pre-registered (replacing the bare R² gate):**
  (i) R² ≥ 0.98 retained for continuity — necessary, not sufficient;
  (ii) bootstrap 90% prediction-interval half-width at C = 2.77e19 ≤ **0.02 bpb**;
  (iii) leave-one-out swing of ratio@(2.77e19) ≤ **2×** (today: 2.9×);
  (iv) residual-structure test passes.
  A fit passing all four may be quoted *within its span*; extrapolation to the d20 carries its
  interval or it is not quoted.

### 1.5 Extrapolation honesty (the multiplier protocol)

- Fits interpolate in shape class; they do not prophesy. Multiplier claims (e.g. "KDA = 1.2×
  compute at equal loss") are **anchored at the largest measured point** and read mid-curve —
  the AttnRes-paper protocol exactly (SCALING_LADDER §1; the 1.25× @ 5.6 PF-days belongs to
  arXiv:2603.15031's own sweep, not Kimi Linear).
- Advantage must **hold or grow across the fitted range** — a single-scale win is not
  adoptable (kill/adopt asymmetry, SCALING_LADDER rule 1).
- Every conclusion is labeled *directional at our scale*; transfer to 2.8T is what Moonshot's
  papers establish, not ours.

### 1.6 Compute accounting

- Bookkeeping axis `C = 6ND` is fine **within** one family at fixed ctx (ratio decisions
  cancel most per-depth distortion) — with the recorded caveat that the attention quadratic is
  21–57% of true FLOPs across our shapes at ctx 2048 (`ARCH_SCALING.md` §1.3). Our `C` axis is
  not metrology-grade; measured FLOPs (step-time × achieved throughput) are logged alongside
  for every point.
- **KDA-hybrid:** at ctx 2048 the linear-attention terms are small corrections to 6ND — the
  dense bookkeeping axis stays usable for the family fit, but every point carries the
  retrieval probes because the family's claim lives at long ctx (M2 anti-parity guard).
- **LatentMoE / any MoE:** `C = 6·N_active·D` while capacity ∝ N_total — the dense law does
  not apply; the joint (N_active, D, sparsity/granularity) mini-grid of `ARCH_SCALING.md`
  §3.4 governs. Never compare MoE and dense points on the dense curve.

---

## §2 The S3 postmortem, stated as design constraints

| S3 defect | Consequence | Constraint it imposes (where handled) |
|---|---|---|
| Ratio cycle 8→20→40 within rays + min-pick selection | D-zigzag ⇒ R² gate fired | §1.2 rule 1; joint-fit grid in §3 |
| Span 45×, top-heavy leverage | drop-s7 swing 26.9→78.2 | §1.2 rule 2; §3 adds d14 points |
| Winner's-curse min-pick, no UQ | sensitivity discovered by accident | §1.4 bootstrap + hold-one-out + gates |
| Borrowed, unswept LR under every point | recipe-v0 measurements | §1.3; P5 precondition; §3 stage 1 |
| 7 points vs 5 free params | over-flexible fit | §3: 9–10 points, structural coverage |

None of this invalidates the *measurements* — s1–s7 remain valid points on recipe-v0 and
enter the re-fit as data, with their v0 flag. The recipe health certification (monotone rays,
s7 below d8-ray extrapolation) stands.

---

## §3 The redesigned program (stages, points, gates, costs)

Sequencing rule: **free box first, rental gated, every gate pre-registered before launch.**

### Stage 0 — P5: recipe-v1 closure (rental, $10–15 / $20 cap) — *precondition for everything below*
Per `FRONTIER_2026_D20_CERTAINTY_PLAN.md` §6 (revised 2026-08-03): LR sweep at d12@ratio-4,
tie-break lower-LR within 0.003 bpb, **confirmation run at d12@ratio-8**, width probe at d8,
compile sm90, kill/resume drill, step-time. Products: locked (LR*, wd*) = recipe-v1; the
confirmation run *is* the first recipe-v1 calibration point (d12@r8 with locked LR).

### Stage 1 — recipe-v0 → v1 reconciliation (free box, ~2.7 GPU-h, conditional)
- **Trigger:** if P5's locked LR* differs from the S3 recipe LR by more than the transfer
  band ×[0.7, 1.4] — re-run the two cheapest S3 points (**s2 + s5**) at recipe-v1.
- **Verdict rule:** |Δbpb| ≤ 0.005 on both ⇒ recipe offset negligible; v0 points enter the
  re-fit unflagged. Otherwise ⇒ v0 points are down-weighted/flagged and the calibration fit
  runs on v1 points only (s2′, s5′, the P5 confirmation point, + stage-2 points).
- **Batch-confound resolution (already pre-registered in `docs/RESULTS.md` §S3, amendment
  2026-08-03):** the matched-batch d12@r8 rerun (batch 8, ~4 GPU-h) runs in this stage — it
  de-confounds the s6-vs-s7 comparison before any re-fit leans on it.
- **Recipe-surface parity check (free, before anything trains):** verify the P5/d20 config
  surface matches the S3 ladder's — tokenizer/vocab, untied embeddings, qk_norm, schedule
  shape, and **mtp_depth** — **resolved 2026-08-03: the d20 runs mtp_depth=0** (decision of
  record, `docs/RESULTS.md` §S3.5); the ladder and the production recipe share the same
  recipe surface by construction.
- Cost is trivial; skipping it risks fitting a law to an LR basin we no longer use.

### Stage 2 — S3.5 calibration re-fit (free box, 2–3 nights)
New points (all recipe-v1, full schedule shape, registered before launch):

| point | N | D | C | wall (est.) | role |
|---|---|---|---|---|---|
| d12-r20 | 135.29M | 2.71B | 2.20e18 | ~8–10 h | completes d12 ray; pairs with P5's d12-r8 |
| d14-r8 | ~193.6M | 1.55B | 1.80e18 | ~8 h | new depth; leverage toward d20 |
| d14-r20 (optional) | ~193.6M | 3.87B | 4.50e18 | ~20 h | span extends to 235×; top-end leverage |

Plus: s1–s7 (v0, per stage-1 verdict) + P5's d12-r8 confirmation point (v1).
**Batch rule:** the S3 batch-consistency precedent applies unchanged (largest batch that
fits, LR held fixed, deviation logged up front). **Reconciliation:** the d12-r20 point above
IS the ladder's s8 — re-homed here from the P5 budget (the $10–15/$20-cap sizing never had
room for 2.7B tokens; the `FRONTIER_STATUS.md` s8 row now points here).
**Fit:** §1.1 estimator on 9–10 points (span 1.92e16 → 2.2–4.5e18 ≈ 115–235×).
**Gates:** the four §1.4 acceptance gates, pre-registered in `docs/RESULTS.md §S3.5` before
the first new point launches. Pass ⇒ the lab has its first certified law: D:N at the d20's C
quoted *with its interval*, replacing HOLD-by-economics with a measured answer (or confirming
the HOLD inside the interval — both are wins). Fail ⇒ report, and the D decision stays
HOLD-by-economics *forever* — we do not buy a third attempt.

### Stage 3 — d20-GQA ($100, gated) — control-family anchor, runs regardless
Decision of record 2026-08-02 (E2E §5 item 6): d20-GQA runs independent of the KDA outcome.
It is also the calibration law's **holdout test**: the d20's measured bpb must land inside the
stage-2 prediction interval — the first true out-of-sample check of our own law.

### Stage 4 — K3 component ablations R1/R2 (HP inheritance, probes mandatory)
Per `docs/k3/ABLATIONS.md`, unchanged: challenger arms inherit the control family's HPs
verbatim (a win under inherited HPs is conservative; self-tuned challengers are
uninterpretable); needle/induction @ 2K/8K/32K on every axis-A arm; 2 seeds on factorial
corners. Negatives are trusted; positives are provisional.

### Stage 5 — THE FAMILY FIT: GQA vs KDA-hybrid (Rung 2, the attention decision)
- **Design:** 4 scale points per family — d8, d12, d14, d16 at ratio 8 — on recipe-v1,
  full schedule, same corpus/val/seeds. KDA-hybrid points reuse R1's proven modules; GQA
  points ride on the stage-2 calibration where overlapping (d12-r8 exists twice by then:
  once as the P5 confirmation = v1 control point).
- **Analysis:** shared-slope test first (offset estimated with slope tied across families;
  default per `ARCH_SCALING.md` §1.2 — Kimi Linear's own fits are offset-dominant:
  −0.0536 vs −0.0527). Escalate to per-family slopes **only** if the shared-slope test is
  rejected (MiniMax-01 precedent: radical mixers CAN move slopes) or the multiplier is
  decision-critical — and here it is, so per-family CIs are reported regardless.
- **Multiplier claim protocol:** §1.5 — anchored at d16, read mid-curve, hold-or-grow across
  the span required, retrieval probes at every point (loss alone cannot see the claim).
- **Adoption rule:** KDA-hybrid multiplier ≥ 1 holding across the range ⇒ adopt; else kill,
  report measured negatives, d20 lineage stays GQA. Single-point wins adopt nothing.
- **Cost:** gated; the d14/d16 points double as the T3 confirmations `ARCH_SCALING.md` §4
  already mandates — no double spend.

### Stage 6 — freeze → Rung 4b family HP re-fit → flagship (Rung 3/5)
K3 §3.2 rule (verbatim in SCALING_LADDER): architecture changes alter the optimal training
regime ⇒ **dedicated re-fit of LR/batch/TPP/shape for the frozen winning family** — P5-style
sweep protocol per family, per-schedule independent searches if schedules are compared.
Then mini-K3 at d20 scale with the production recipe = the mid-scale validation rung and the
portfolio flagship (it absorbs the second d20 slot only if the family fit adopted KDA).

---

## §4 What this program deliberately refuses (anti-goals)

1. **No per-architecture re-sweeps for offset variants** (MLA/DSA/GDN at frozen ctx) —
   anchors, not sweeps (`ARCH_SCALING.md` §3).
2. **No HP laws on an unfrozen architecture** (SCALING_LADDER rule 2).
3. **No quoting fits past their span**; no un-flagged v0/v1 mixing; no CORE in any fit.
4. **No intermediate-checkpoint fake budgets**; no single-seed adoption claims.
5. **No third calibration attempt** if stage 2 fails — the D decision degrades to economics
   permanently, honestly labeled.

---

## §5 The DAG in one block

```
P5 (recipe-v1: LR*, wd*, sm90, step-time)         rental, $10–15 / $20 cap   [PRECONDITION]
  └─ stage 1: v0→v1 check (s2′+s5′ if triggered)  free, ~2.7 GPU-h
  └─ stage 2: S3.5 re-fit (+d12-r20, +d14-r8[20]) free, 2–3 nights           [GATES §1.4]
  └─ stage 3: d20-GQA ($100) — control anchor + law holdout test             [runs regardless]
  └─ stage 4: R1/R2 ablations (after core/kda.py PROVEN)
  └─ stage 5: FAMILY FIT GQA vs KDA — THE attention decision                 [Rung 2]
  └─ stage 6: freeze → family HP re-fit → flagship mini-K3 @ d20             [Rung 4b/3/5]
```

Every arrow is gated; every gate is registered before the spend. That — not any single fit —
is what makes this a scaling-law *program* rather than a scaling-law *attempt*.

---

*Grounded in: `docs/k3/SCALING_LADDER.md` (sequencing + lab evidence table),
`FRONTIER_2026_ARCH_SCALING.md` §1–5 (doctrine + refs 1–17), `docs/RESULTS.md` §S3 (the failed
fit of record), `FRONTIER_2026_D20_CERTAINTY_PLAN.md` §6 (P5 protocol, revised 2026-08-03),
`docs/k3/ABLATIONS.md` (R0–R2), K3 tech report §3.2 (frozen-family re-fit rule).*
