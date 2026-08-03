# Frontier 2026 — d20 Certainty Plan: how size, data, and hyperparameters are actually defined

> **Audience:** the AI research engineer who owns the next two nodes (P5, then the d20).
> **Status:** living document, written 2026-08-02 after S3 s1–s7 completed and the CORE band was
> re-anchored. **Revised 2026-08-03 (senior-review pass):** P5 sweep protocol pinned (horizon,
> tie-break, confirmation run, width probe — §6 item 1), the oracle gate reconciled with the E2E
> T2 amendment and de-confounded (§6 item 3), P5 given a $20 hard cap (§6/§10), and the §3
> ratio-20 evidence chain stripped of its compute-confounded clause. It consolidates, in one
> place, *what is certified vs borrowed vs still undefined*
> about the d20's three defining quantities — model size N, token count D, and the optimizer
> hyperparameters — and the exact engineering plan that converts the undefined parts into measured
> ones before $100 is exposed.
> **Owner docs:** this file cross-references but does not replace `FRONTIER_2026_END_TO_END_PLAN.md`
> §S3/§S4, `FRONTIER_2026_TASKSPEC.md` (Next-node marker), `docs/RESULTS.md` §F12/§S3/§S4-pre,
> `bench/RESULTS.md` §Frontier ablations + §$100 d20 run.

---

## §0 The risk ledger (read this first)

| Quantity | Value | Epistemic status | Basis | Where it gets certified |
|---|---|---|---|---|
| Corpus | ClimbMix | **Borrowed** | nanochat's larger-scale result; our F12 measured it *worse* at 35M (+0.110 bpb, single-seed) | d20 itself is the final arbiter (`docs/RESULTS.md` §F12 consequence 3) |
| N | 480.4M (d20, d_model 1280, vocab 32,768 untied) | **Anchored** | 86% of nanochat d20's N at 73% of its C (anchor CORE 0.2219 @ 3.77e19) | P5 step-time + d20 report card vs band |
| D | 9.6B tokens (ratio 20) | **Deliberate overtrain, uncertified by our fit** | nanochat's published optimum 8–10.5 + s6-vs-s7 isoFLOP (training-optimal ≤8 at ~9e17 ⇒ 20 is an overtrain by design, §3) + Sardana 2401.00448 serving economics | **d14 (T1 option)** — the only rung that can repair the fit |
| Muon LR / wd | **UNDEFINED** | **Not yet measured on our stack at any scale** | µP transfer theory + shipped LR-transfer machinery | **P5 LR sweep at d12 on H100** |
| Expected CORE | 0.23–0.25 (central 0.24) | Re-anchored 2026-08-02 | published ClimbMix curve (0.257–0.269 @ ~4e19) − recipe discount | d20 run vs band; KILL <0.15 |

One sentence: **size and data are chosen and honestly labeled as anchor/economics-driven;
the hyperparameters are the genuinely open variable, and P5 exists to close them for $10–15
($20 hard cap).**

---

## §1 The epistemic frame: four grades of "known"

Every number the d20 depends on carries one of four grades. Mixing them up is how labs lie to
themselves; the repo's honesty ledger exists to keep them separate.

1. **Certified** — measured on our stack, on our corpus, pre-registered, reproducible.
   Example: the S3 ladder bpb curve (1.2553 → 0.9402 over C = 1.92e16 → 8.79e17).
2. **Anchored** — chosen relative to a published external reference point, with the arithmetic
   written down. Example: N = 480.4M (86% of nanochat d20's N at 73% of its C).
3. **Borrowed** — taken from external results whose transfer to our stack is *assumed, not
   verified*. Example: ClimbMix as the d20 corpus (operator override of our own F12 kill).
4. **Undefined** — not yet measured anywhere on our stack. Example: the Muon learning rate
   and weight decay at d20 scale. **Undefined is acceptable only when a funded, pre-registered
   rung exists whose sole job is to define it** — that rung is P5.

The plan's discipline: no undefined quantity may reach the $100 run; every borrowed quantity
must carry its transfer assumption in writing.

---

## §2 Model size N = 480.4M — anchored adequacy, not optimality

**Derivation (of record, 2026-07-16 refactor):** depth 20 / d_model 1280 / vocab 32,768 untied
⇒ exactly 480.4M instantiated params. nanochat's original d20 anchor: CORE 0.2219 @ C = 3.77e19.
We buy **73% of the anchor's compute at 86% of its N** — the artifact is a CORE-vs-FLOPs point on
nanochat's published curve, *never* a depth-matched headline (framing locked in E2E §S4(e)).

**Why not derive N from our own scaling law?** Three honest reasons:

1. **The budget caps C first.** At ~$100 / 8×H100, C ≈ 2.77e19 is the constraint; N and D then
   trade inside the isoFLOP basin. The question is never "what N is optimal in absolute terms"
   but "what (N, D) split of this fixed C serves the artifact best."
2. **The basin is shallow near the optimum.** The repo's own earlier fit work (bench/RESULTS.md)
   puts ratio 8↔20 within ≈0.017 nats of each other at matched C — thousandths of CORE. Errors in
   N choice inside the basin cost far less than errors in LR choice (§5).
3. **Our law cannot extrapolate anyway** (§4) — and the plan said so *before* running: borrowed
   exponents are tolerable for shape, but constants are stack-specific (tokenizer, vocab, untied
   embeddings, qk_norm, precision). Anchor calibration is the correct tool at this budget.

**What could still invalidate N:** a P5 step-time measurement so far off the 0.48–0.58 s
prediction that the $90 abort line trips before 18,311 steps complete — handled by the
pre-registered abort, not by re-deriving N.

---

## §3 Data D = 9.6B tokens (ratio 20) — a deliberate economic choice, labeled uncertified

**The evidence chain, in order:**

1. **Borrowed prior:** nanochat's own fit says compute-optimal D:N ≈ **8–10.5**
   ([discussion #420](https://github.com/karpathy/nanochat/discussions/420), VERIFIED in-repo).
   Ratio 20 is deliberately ≈2× above training-optimal.
2. **Why deliberately above:** the d20 is an *inference artifact*. Overtraining (smaller N, more
   D at fixed C) buys a permanently cheaper-to-serve model at a small training-time loss penalty —
   the inference-aware allocation argument (Sardana [2401.00448](https://arxiv.org/abs/2401.00448)),
   re-registered 2026-07-30 ("decide, don't inherit").
3. **What our S3 was supposed to do:** certify or re-register D via our own N\*(C)/D\*(C) fit.
   **It could not** — the R² gate raised (§4). What the data *does* support, per the
   pre-registered fitted-interval rule: the within-ray comparisons (ratio-20 beats ratio-8 at
   equal N) are **compute-confounded** (at fixed N, ratio 20 is 2.5× the C, so it *must* win) and
   are non-evidence for allocation; the one true isoFLOP comparison (s6 vs s7, 1.043× C) favors
   bigger-N/ratio-8 by −0.0096 bpb — i.e. the
   training-optimal ratio at ~9e17 is ≤8 and drifting down, consistent with nanochat's 8–10.5
   and consistent with 20 being an *overtrain*, as intended.
4. **Verdict of record (2026-08-02):** HOLD ratio 20 / 9.6B — with the written caveat that the
   ladder cannot certify it. The corpus side is similarly borrowed (F12 measured FineWeb-EDU
   better at 35M; operator override chose ClimbMix on nanochat's larger-scale result; the d20
   run itself is the final corpus arbiter).

**What would actually reduce this uncertainty: d14** (§7). Nothing else on the free box moves it.

---

## §4 What the S3 fit can and cannot say — and the grid-design lesson

**Measured (certified):**

| rung | depth | N | ratio | C | val_bpb | wall |
|---|---|---|---|---|---|---|
| s1–s3 | 4 | 19.99M | 8/20/40 | 1.92e16 → 9.59e16 | 1.2553 / 1.1772 / 1.1408 | 0.14–0.68 h |
| s4–s6 | 8 | 59.26M | 8/20/40 | 1.69e17 → 8.43e17 | 1.0124 / 0.9676 / 0.9498 | 0.81–4.06 h |
| s7 | 12 | 135.29M | 8 | 8.79e17 | 0.9402 | 4.01 h (batch 4) |

**Fit outcome:** a = 0.4515, b = 0.5485, a+b = 1.0000 — but the pre-registered R² gate
**raised** (R²_N = 0.7709, R²_D = 0.8324 < 0.98), so **no extrapolation to the d20's C is
quoted** (T1 semantics). Two structural facts, both written into `docs/RESULTS.md` §S3:

- **a+b = 1 is near-vacuous here.** Recorded D = ratio×N = C/6N *exactly*, so the exponent sum
  is an identity of the C=6ND wiring, not evidence of a clean law. The gate that has real teeth
  is R².
- **R² fails by grid geometry, not by recipe.** Ratios cycle 8→20→40 within each depth ray, so
  the min-picked D zigzags across depth boundaries (0.16→0.40→0.80→0.47→1.19→2.37→1.08B). No
  power law fits a zigzag — this was predictable in hindsight and is now a **design rule for
  the next grid**: vary ratio *within* one depth, or vary depth *within* one ratio, never both
  in a cycle. Sensitivity arm confirms fragility: dropping s7 swings ratio@d20 from 26.9 to 78.2.

**Recipe health (also certified):** bpb descends monotonically on every ray; the pre-registered
kill (s5/s6 vs toy-corpus floor) did not trigger; s7 lands *below* the d8-ray extrapolation
(0.9402 vs ≈0.9487) — no batch-attributable excess from the pre-registered batch-4 deviation.
The training pipeline is sound at every scale tested. That, plus one real allocation comparison,
is what 12.09 GPU-h bought.

---

## §5 Hyperparameters — the undefined third, and the transfer theory that bounds it

**Inventory of what d20 needs and does not yet have (measured on our stack):**

| HP | Status | Note |
|---|---|---|
| Muon LR (matrix params) | **undefined** | the single most consequential number in the run |
| AdamW LR (embeddings/scalars/head) | partially borrowed | follows nanochat ratios; must be co-validated in the sweep |
| weight decay | borrowed | 1/width-style scaling assumptions from the µP literature |
| warmup / schedule shape | borrowed | nanochat-consistent; validated indirectly by S3 stability |
| global batch | **defined** | 524,288 tok (P-locked in the d20 pre-registration) |
| precision | **defined with an open validation** | bf16; compile NaN observed on sm120 — sm90 re-validation is a P5 item |

**Why LR is the dangerous unknown (first principles):** loss at fixed (N, D) is far more
sensitive to LR mis-setting than to any in-basin (N, D) reallocation — a wrong LR doesn't crash
the run, it silently yields a worse model for the same $100. And Muon's transfer behavior under
scale is exactly what the cited HP-transfer study ([2512.05620](https://arxiv.org/abs/2512.05620))
identifies as the deciding factor in whether Muon's small-scale edge survives. We adopted Muon on
external evidence (F1 descoped mid-flight, optimizer ADOPTED, LR sweep *folded into P5* — not
dropped).

**What µP buys us (the bridge):** under maximal-update parametrization, the optimal LR is
width-transferable *if* the parametrization rules hold; the d12 sweep then pins the constants
and the d20 width extrapolation is theory-guided rather than blind. **What is shipped:**
the LR-transfer machinery + fused optimizer are a completed d20-gate item (TASKSPEC P1–P6).
**What has never run:** an actual LR sweep on our stack at any scale. That gap is P5's first
work item, and it is why "final hyperparameters" do not exist as numbers today — they exist as
a *procedure with a price tag of $10–15 ($20 hard cap)*.

---

## §6 P5 — the HP-definition rung (full protocol)

**One rung, four products:** (1) the locked Muon LR + wd for d20; (2) sm90 re-validation of
compile + precision; (3) the s8 ladder point + oracle-divergence check; (4) a rehearsed,
measured path into the $100 run (kill/resume drill, step-time). $10–15 planned / **$20 hard
cap**, 1×H100. Budget math: 5-point sweep at the ratio-4 horizon ≈ 1.5–2 h + winner
confirmation at ratio 8 ≈ 0.75 h + width probe ≈ 0.25 h + gates/drills ≈ 1 h ⇒ 3.5–4 h at
$2.5–3.5/GPU-h. (The earlier "1–2B tokens total" sizing in TASKSPEC forces a sweep horizon too
short to transfer — the protocol below spends the tokens where they resolve the LR.)

**Protocol (each item maps to a gate in `bench/RESULTS.md` §$100 d20 run / E2E §S4(e)):**

1. **Muon LR sweep at d12.** Grid: the transferred point estimate ×{0.5, 0.7, 1.0, 1.4, 2.0}
   (coarse, single seed — the goal is locating the basin and checking the transfer prediction,
   not a publication-quality response surface). **Horizon pinned: ratio 4 at d12 (0.55B tokens,
   C ≈ 4.4e17, ~25 min/point on H100) run with the d20's schedule shape** (same warmup fraction,
   same decay form) — the horizon must be stated because LR optima drift downward as horizon
   lengthens; ratio 4 is the cheapest horizon that still exercises the full schedule.
   Selection metric: val_bpb on the pinned val set (never raw CE). **Tie-break: grid points
   within 0.003 bpb (single-seed val_bpb noise) are ties ⇒ take the *lower* LR** — cheap
   insurance against the horizon shift between the sweep and the d20's 18,311 steps.
   **Confirmation run:** the winner once at ratio 8 (1.08B tokens, C ≈ 8.8e17) — this sits on
   the measured s7 ray and doubles as the T2 anchor check (item 3). **Width probe:** a 3-point mini-sweep
   at d8 (width 512, ratio-4 horizon) at {×0.7, 1.0, 1.4} of the d12 winner (~30 min total) —
   a single LR point per width cannot measure a transfer slope; three locate the d8 basin. If
   the d8 optimum sits more than one grid step from the µP-consistent position, treat the
   768→1280 extrapolation as empirical-only and widen the d12 sweep once.
   Depth transfer (12→20) stays a residual risk, accepted in writing: nanochat's own recipe
   holds LR fixed across depths. **Transfer check:** if the empirical optimum lands outside
   ×[0.7, 1.4] of the µP prediction, transfer is weaker than assumed ⇒ widen the sweep once;
   if it lands flat/unstable, **stop — do not proceed to d20 on a guessed LR.**
2. **Compile-on-sm90 re-validation.** bf16+compile NaN is a measured sm120-inductor fact;
   re-run the minimal repro on sm90. Outcome decides whether d20 trains compiled (planned
   step-time) or eager (re-price the run before starting it).
3. **Anchor divergence vs the measured d12 ray** (the T2 gate, reconciled with the E2E
   amendment of 2026-08-03; this item's earlier wording mixed units — CORE compared against a
   bpb threshold — and is corrected here). **Primary gate (bpb):** the item-1 confirmation run
   (d12@r8) must land inside the re-anchored band **0.89–0.92 bpb** (s7 d12@r8 0.9402 with the
   batch-4 caveat, extended within-ray to ratio 20 — `docs/RESULTS.md` §S3); **> 0.01 bpb
   outside the band ⇒ trigger T2 (d16)**, stop and diagnose before the d20. **Secondary
   (CORE):** our d12 vs nanochat's published d12-class point (P3 suite, byte-identical to
   nanochat core.yaml), interpreted only against the measured CORE noise floor
   (±0.008–0.016, Appendix): divergence **≥ 0.02** (≈2× floor) is signal ⇒ diagnose; below
   that it is inconclusive-by-noise, not a stack bug.
4. **Checkpoint kill/resume drill.** Kill mid-run, resume, verify continuation (optimizer-state
   snapshots via `--checkpoint-every` → `work_dir/pretrain_ckpt.pt`, consolidated path under
   torch.distributed — all shipped and CPU-verified; this is the *live-fire* proof).
5. **Step-time + MFU measurement at d12 on H100.** Re-prices the d20: the 2.4–3.3 h / $58–79
   window and the $90 abort are only as good as this number.

**Go/no-go into d20 (all five required):** LR locked inside the predicted basin and confirmed
at the ratio-8 horizon · compile clean
(or eager re-priced and approved) · bpb inside 0.89–0.92 (±0.01) with CORE divergence < 0.02 ·
resume verified · step-time
inside 0.48–0.58 s-equivalent scaling. Any failure ⇒ fix and re-rehearse; **no paid d20 on a
failed rehearsal.**

---

## §7 d14 — the T1 option (what it buys, what it costs, when it's worth it)

**What it is:** the escalation rung fired by the S3 R²-gate failure (E2E §S3(g), T1) — a
larger-model ladder point (d14-class) that extends the sweep toward the d20's regime.

**What it buys:** a repaired fit. With a big-model anchor, the N\*(C)/D\*(C) fit gains the
leverage it currently lacks (today's extrapolation swings 26.9 ↔ 78.2 on one point). It would
convert the D decision from "HOLD on interval evidence" into "measured on our stack."

**Design rules (from the §4 zigzag lesson), if d14 runs:**
- vary ratio **within** d14 (e.g. ratios {8, 20} at fixed depth) — do not introduce a new
  depth *and* new ratios in one point;
- keep every protocol element identical (tokenizer, val set, ctx 2048, LR recipe);
- batch as large as fits, LR held fixed, deviation logged up front (the batch-consistency
  precedent from S3);
- pre-register the kill/interpretation *before* launching.

**Cost:** free-but-slow on the standing box (multi-day at d14 scale; blocks P5 prep) or paid
spot (guardrail: paid work needs explicit human go-ahead).

**Decision framing:** d14 is worth it iff the D:N uncertainty is worth days of wall-clock
(free path) or dollars (paid path). Given ratio-20 is a *deliberate* overtrain whose error cost is bounded
by the shallow basin (≈0.017 nats), the marginal value of certifying it is modest — but that is
the human's call, and it is on the table precisely because the gate fired honestly.

---

## §8 Decision matrix and current recommendation

| Option | Cost | Buys | Risk left |
|---|---|---|---|
| A. Accept HOLD, go straight to P5 | $10–15 (cap $20) | HP closure (the real gap) | D:N stays borrowed+interval — bounded by shallow basin |
| B. d14 free on standing box, then P5 | days of wall + $10–15 (cap $20) | D:N certified on our stack | LR still only P5-closed (fine) |
| C. Paid d14, then P5 | $ + $10–15 (cap $20) | same as B, faster | same as B |

**Decision of record (2026-08-03, operator-delegated): A — accept HOLD, straight to P5.**
(Supersedes the 2026-08-02 recommendation of A with B as the conservative alternative.) Rationale:
the genuinely *undefined* quantity is hyperparameters, and only P5 closes it; the D:N question
is *bounded* (borrowed optimum + interval evidence + shallow-basin economics), so d14's marginal
information is real but not gating. If the human weights corpus/D certainty above schedule,
B is the defensible pick. C requires explicit paid-work approval per guardrails.

---

## §9 Execution checklist

**Now (free, standing box — no GPU conflict):**
- [x] F12 measured + logged; corpus override FINAL; license note (CC BY-NC 4.0) queued for A9
- [x] S3 s1–s7 + fit attempt + gate outcome logged; figures committed
- [x] CORE band re-anchored 0.23–0.25 (`docs/RESULTS.md` §S4-pre)
- [x] This plan doc
- [ ] Confirm staged ClimbMix train shards ≥ 9.6B tokens (+ pinned val set present) — free
      check, must pass before P5 day
- [x] Human decision: T1 — **A** (accept HOLD → P5), decided 2026-08-03 (§8 decision of record)

**P5 day (rental, $10–15 planned / $20 hard cap — after human go-ahead):**
- [ ] Provision 1×H100 per `deploy/runbooks/d20_speedrun_8xH100.md` (reuse its env steps)
- [ ] Run §6 items 1–5 in order; log each gate outcome into `bench/RESULTS.md` §$100 d20 run
- [ ] Write the P5 verdict (go/no-go per §6) into `docs/RESULTS.md` and the STATUS board

**Post-P5 (only on go):**
- [ ] Lock the measured Muon LR/wd + step-time into the d20 config; update cost window
- [ ] Human authorization for the $100 run (guardrail)
- [ ] Execute d20 per runbook; enforce every pre-registered kill line mechanically
- [ ] Report card: CORE vs 0.23–0.25 band, bpb vs ≈0.74 cross-check, CORE-vs-FLOPs point on
      nanochat's curve — honest framing locked (never a depth-matched headline). **Noise rule,
      pre-committed:** the band half-width (0.01) is below the CORE noise floor
      (±0.008–0.016) — a single measurement within ±0.016 of a band edge is *inconclusive*,
      not a pass/fail; repeat the eval (or mean over k reruns) before declaring either.

---

## §10 Consolidated kill / abort lines (mechanical, no mid-run debate)

- P5: LR basin not found (flat/unstable) ⇒ stop · confirmation run (d12@r8) outside
  0.89–0.92 bpb by > 0.01 ⇒ T2/d16, stop · CORE-vs-nanochat divergence ≥ 0.02 (noise-aware)
  ⇒ diagnose before d20 · compile NaN on sm90 ⇒ re-price eager before any d20 approval ·
  **$20 cumulative ⇒ abort, re-scope the sweep.**
- d20: MFU < 28% sustained ⇒ kill · step > 0.70 s ⇒ kill · comm > 3% ⇒ kill · scaling < 93%
  ⇒ kill · loss-at-init deviates from ln 32768 = 10.40 ⇒ kill · CORE < 0.15 ⇒ stack bug ·
  **$90 cumulative ⇒ abort → downsize d16.**

---

## Appendix — constants inventory (with provenance)

- d20: 480.4M params · 9.6B tok · C = 2.77e19 · 18,311 steps @ 524,288 tok global batch
- Anchor: nanochat original d20 CORE 0.2219 @ 3.77e19; we buy 73% C / 86% N
- nanochat CORE-fit (FWE-era): 1 − 3.7555·C^(−0.0344); at 2.77e19 → 0.195
- nanochat ClimbMix curve (current): 0.257–0.269 @ ~4e19 (leaderboard d24 ratio-8 FP8:
  0.2626, val_bpb 0.718); GPT-2 XL 0.256525
- nanochat compute-optimal D:N ≈ 8–10.5 (discussion #420)
- CORE noise floor: ±0.008–0.016 (7× identical rerun: 0.2512–0.2677, discussion #481)
- Our S3: bpb 1.2553 → 0.9402 over 1.92e16 → 8.79e17; fit R²_N 0.7709 / R²_D 0.8324 (gate
  raised); s6-vs-s7 isoFLOP Δ = −0.0096 bpb for bigger-N
- Re-anchored d20 band: **0.23–0.25 (central 0.24)**; predicted d20 val_bpb ≈ 0.74 (indicative)
- P5 protocol (pinned 2026-08-03): sweep d12@ratio-4 (0.55B tok/point, C ≈ 4.4e17, ~25 min) ×
  grid {0.5, 0.7, 1.0, 1.4, 2.0}, d20 schedule shape; tie-break = lower LR within 0.003 bpb;
  confirmation d12@ratio-8 (1.08B tok) vs band 0.89–0.92 bpb; width probe: 3-point mini-sweep
  d8@ratio-4 {×0.7, 1.0, 1.4}; budget
  $10–15 planned / $20 hard cap
- F12: FWE 1.19197 vs ClimbMix 1.30205 bpb (Δ +0.1101); CORE 0.0510 vs 0.0551 (single-seed);
  decontam overlap 1.92% / 0.07%
