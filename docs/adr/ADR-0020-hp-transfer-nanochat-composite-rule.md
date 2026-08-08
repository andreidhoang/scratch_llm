# ADR-0020 — HP transfer: adopt nanochat's composite rule (not µP), add a target-scale d20 probe, re-measure D:N on the free path

**Status:** accepted · 2026-08-07 (operator-delegated; amends `../FRONTIER_2026_D20_CERTAINTY_PLAN.md` §5/§6/§8)
**Scope:** the nanochat front's d20 certainty plan only. K3 track unchanged (Moonshot's own
methodology — per-size LR grid search, no µP — already governs there; this ADR makes the two
fronts consistent rather than divergent).

**Status note (2026-08-07, ~03:00 UTC — overtaken-by-events reconciliation):** while this ADR
was being written, the S3.5 pod run (vast.ai, human-authorized, ~$41–43 projected vs $48
budget) was already executing the substance of items 2–3: the P5 LR sweep completed on the pod
(**η\* = 0.0021**, i.e. ×0.7 of the 3e-3 recipe; d8 width probes at {0.7, 1.0, 1.4}× confirmed
the basin transfers), and the **d14 rung is in flight right now** (d14_r8 lands ~06:00 UTC;
d14_r20 finale ~Aug 8 evening) — not on the standing box as item 3 proposed, but as S3.5
stage-2 grid points on the paid pod. The A′ **re-registration hook stands unchanged**: when the
13-point joint fit runs, its D:N implication is checked against ratio-20 **before** the d20
launches. Item 2 (P5.5 target-scale probe) is unaffected — the joint fit predicts *loss*, it
does not validate *LR at d20 scale*. Two consequences to verify when the pod syncs: (a) whether
the pod P5 also covered the engineering items (sm90 compile re-validation, kill/resume drill,
step-time) — the status summary covers only the HP products; (b) the quarantined old s1–s7
(different tokenizer, 2.4× bpb mismatch) means the original S3 R²-gate failure was at least
partly a data-provenance artifact — the new clean, monotonic 11-point grid is the evidence base
now, and `FRONTIER_STATUS.md` / `docs/RESULTS.md` should say so when the fit lands.

**Status note 2 (2026-08-07, ~03:45 UTC — handoff-audit corrections):** the executing agent's
handoff document labels the S3.5 grid "AdamW (NOT Muon) — deliberate, to match Chinchilla's
setup." **Code-verified FALSE:** `scripts/s3_scaling_sweep.py run` builds `SpeedrunConfig`
without an `optimizer` argument (`src/scratch_llm/scaling/s3_sweep.py:395-409`), so the
`speedrun.py:62` default `optimizer="muon_adamw"` governs every grid point, and the schedule is
cosine with warmup = steps/20 (`speedrun.py:232`). The handoff's "Standard Parametrization
(NOT µP)" half is accurate. Consequences: (a) **good news** — η\* = 0.0021 was tuned under the
same muon_adamw family the d20 runs, so the P5→d20 transfer chain is intact as designed; the
"matched Chinchilla's AdamW setup" framing was a documentation error, not a run error — all
measured points stand. (b) Item 4's schedule verdict ("adopt nanochat's constant+warmdown at
P5") is **overtaken**: P5 and the entire 13-point grid already ran cosine, so the law, the
gates, and the CORE band are all calibrated to recipe-v1 = muon_adamw + cosine. The d20
**keeps cosine** to preserve the fit's validity; schedule shape (constant+warmdown-0.65) is
re-registered as a post-d20 F-rung ablation candidate. (c) The d20's batch (524,288 tok) is
32× the grid's (16,384 tok) — under the composite rule the probe grid centers on
η\*·√32 ≈ 0.0119, so **P5.5 doubles as the first measurement of the √B assumption** at target
batch (nanochat's own "not studied carefully, assumption!" label applies). (d) Closeout docs
(`docs/RESULTS.md` §S3.5, `FRONTIER_STATUS.md`) must carry the corrected recipe label
"muon_adamw, standard parametrization, cosine" when the fit lands.

## Context

A code-verified research pass on `karpathy/nanochat` master (`base_train.py`, `gpt.py`,
`optim.py`, `dev/LOG.md`, discussions #420/#481) established how the anchor stack actually
transfers hyperparameters. The findings that change our plan:

1. **nanochat does not use µP.** Its transfer stack ("Scaling laws and muP extrapolations"
   section of `base_train.py`) is a hand-rolled composite anchored at d12 (768-wide):
   - AdamW-group LRs × `(d_model/768)^-0.5` (comment: "tuned for 768 dim model");
   - Muon `matrix_lr = 0.02` **width-constant**, only a per-shape `max(1, rows/cols)^0.5`
     correction;
   - all LRs × `√(B/B_ref)`, B_ref = 2¹⁹ = 524,288 tok, with the honest comment *"Muon: same
     scaling as AdamW (not studied carefully, assumption!)"*;
   - weight decay via the T_epoch framework (arXiv:2405.13698): `λ = 0.28·√(B/B_ref)·(D_ref/D)`,
     comment *"blindly following AdamW theory… hoping it ~works for Muon too"*;
   - batch itself scales `B ∝ D^0.383` (Power Lines, arXiv:2505.13738).
2. **Karpathy's load-bearing lesson (320-experiment sweep, d12→d16→d20):** *"small-scale tuning
   doesn't transfer. Validate at target scale."* d12-optimal settings (emb_lr ~0.4, wd ~0.14,
   matrix_lr ~0.026) **actively hurt at d20**; only `x0_beta1=0.96` survived. Stability at
   d24–d32 was achieved by validating at/near target scale, not by any parametrization theory.
3. **D:N was measured, not borrowed — and it is not 20.** Launch-day 20:1 was taken from
   Chinchilla; nanochat's own IsoFLOP fits gave ≈8 compute-optimal, settling at ~10.5, current
   default 12, speedrun deliberately 8. *"It's important to do the actual experiment on your own
   network."* Our ratio-20 HOLD therefore rests on a literature argument (Sardana 2401.00448)
   that the **closest reference implementation has already falsified for this stack**.
4. Our certainty plan §5 currently frames the LR bridge as "µP transfer theory" with wd
   "1/width-style from the µP literature." That framing is both inaccurate (the anchor doesn't
   do it) and weaker than what the anchor actually ships.

Honesty labels: items 1–3 are **reported** (read from nanochat master + Karpathy's own posts),
**not yet measured on our stack**. Karpathy's answer to the explicit "did you consider µP?"
question in #420 comments was not retrievable — recorded as unknown, not inferred.

## Decision

1. **Retire the µP framing; adopt nanochat's composite transfer rule verbatim as the d20
   default machinery.** The shipped LR-transfer machinery (TASKSPEC P1–P6) implements exactly:
   Adam groups × `(d_model/768)^-0.5`; Muon LR width-constant with the per-shape correction;
   both × `√(B/2¹⁹)`; wd via T_epoch. All four rules carry the ledger label *reported, not
   verified on our stack* until P5 + the probe (item 2) measure them. No µP lab, no
   transfer-slope fitting program — the width/depth axes are confounded by design
   (`d_model = 64·depth`), and nanochat's answer to that confound is item 2, not a fit.
2. **Add a target-scale probe gate ("P5.5") at the start of the d20 rental.** Before committing
   the full budget: 2–3 LR candidates (the P5 winner ×{0.7, 1.0, 1.4}) at the real d20 config,
   ~0.5B tokens each (≈5% of budget, ≈$5–8 on the 8×H100), ranked by val_bpb on the pinned val
   set; commit the remaining ~95% at the probe winner. This is Karpathy's
   "validate at target scale" applied to a one-shot run, and it directly retires the two
   residual risks the plan currently accepts in writing (depth transfer 12→20; horizon drift
   ratio-4→20). The $90 abort criterion stays armed throughout.
3. **Amend the T1 decision (§8) from A to A′: P5 proceeds as decided AND the d14 rung runs on
   the standing box in parallel.** The 2026-08-03 reasoning still holds — P5 is the gating
   unknown — but new evidence (finding 3) upgrades d14 from "modest marginal value" to "the only
   measurement of the single biggest budget lever," and the free path costs **zero dollars and
   zero rental days**: the standing box is otherwise idle between F-rungs, and P5 is a rental.
   Rules per §7 (vary ratio within d14, identical protocol, kill criterion pre-registered before
   launch). **Re-registration hook:** if the d14-extended fit supports an optimal ratio ≲12, D
   for the d20 is re-registered (with the CORE band re-anchored in the same commit) *before* the
   d20 launches; if the fit is again gate-rejected, ratio-20 stands with the ledger note
   "anchor stack measures 8–12; our overtrain is deliberate and bounded (≈0.017 nats)."
4. **Recipe-divergence inventory (anchor-validity hygiene).** Recorded, each marked
   adopt-or-deviate-with-reason, in the certainty plan §5: schedule (our cosine +
   warmup steps/20 vs nanochat constant + linear warmdown 0.65 → 5% + 40-step warmup) —
   **adopt nanochat's at P5** (the sweep already pins "d20's schedule shape"; make that shape
   the anchor's); wd (our 0.1 vs 0.28 T_epoch-scaled) — **adopt T_epoch rule** per item 1;
   peak-LR parameterization (our single 3e-3 vs per-group matrix/embedding/unembedding) —
   **deviate with reason** (our muon_adamw groups differ; the P5 sweep covers it); batch 2¹⁹ —
   already aligned. Every remaining divergence weakens the CORE-band anchor and must stay
   visible in this table.
5. **Agent sweep budget (autoresearch precedent).** Karpathy's March-2026 agent-run ~700
   experiments (−11% GPT-2 speedrun time) are the template for this repo's native mode: agents
   may launch any **standing-box, d12-class-or-smaller** probe that is pre-registered in
   `FRONTIER_2026_ABLATIONS.md` with prediction + kill criterion, without per-run human
   approval; results land in `bench/RESULTS.md` with hardware + method. Paid/rental work stays
   human-gated, unchanged.

## Justification

- **EV ordering is unchanged, the costs just got truer.** The plan's own ranking (HP closure ≫
  D:N certification) survives; what changes is that D:N certification turned out to be free
  (parallel, standing box) and the anchor's D:N measurement turned out to contradict our HOLD.
  Ignoring finding 3 would be exactly the "literature over own measurement" failure the repo's
  predict-before-you-run discipline exists to prevent.
- **The probe is the cheapest insurance that exists.** $5–8 to retire the two residual risks
  currently accepted in writing, on a $58–79 run whose worst failure mode is a *silent* 0.01–0.02
  bpb regression from a mis-set LR. No theory (µP or otherwise) offers that at any price.
- **Verbatim-adoption beats theory-shopping at our scale.** Our target is $100-class; the
  anchor's composite rule is measured on the same architecture family at the same scales. The
  rules come with their author's own honesty labels ("assumption!", "blindly following") — we
  inherit the labels with the numbers and clear them at P5/P5.5.
- **Ruthlessness (what we deliberately do NOT do):** no µP implementation, no 2-axis
  width/depth transfer-law fit (confounded ladder by design), no re-opening of the F12 corpus
  decision, no change to the K3 methodology, no re-run of S3. Each was considered and cut as
  low-EV against items 1–3.

## Consequences

- `FRONTIER_2026_D20_CERTAINTY_PLAN.md` amended: §5 (transfer-rule framing + divergence table),
  §6 (P5.5 probe in go/no-go; width probe reinterpreted as a check of the composite rule, not a
  slope fit), §8 (decision A → A′ with the d14 re-registration hook).
- `docs/k3/FACTS.md` / `bench/RESULTS.md`: nanochat transfer constants and the D:N ≈ 8–12
  measurement logged as **reported** (source: nanochat master + discussions #420/#481), pending
  our own P5/d14 measurements.
- The d20 pre-registration (E2E §S4) gains one line: the P5.5 probe gate with its LR grid and
  selection rule, so the full-budget commit is never an unmeasured step.
- No code changes are authorized by this ADR; the LR-transfer machinery's *defaults* change per
  item 1 as a normal TASKSPEC-tracked task. `src/scratch_llm/k3/core/` untouched, as always.
