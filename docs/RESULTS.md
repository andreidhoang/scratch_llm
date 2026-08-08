# Frontier ablation results — predicted vs measured

Pre-registered measurement ledger for the F-front ablations defined in
`docs/FRONTIER_2026_TASKSPEC.md`, now also hosting the §S3 scaling-sweep and §K3
anatomy/interaction pre-registrations.  A row is `[FACT]` only once measured; until
then it records the falsifiable prediction and the kill criterion.

## F6 · MoE balancing ablation

Registered **before** running. Spec: `docs/FRONTIER_2026_TASKSPEC.md` §F6.
Thesis: aux-loss-free bias balancing (`BIAS_FREE`) achieves final validation CE
≤ sequence-level auxiliary loss (`SEQ_AUX`) while keeping router entropy above
`0.9·log(N_r)`.  The granularity axis (`COARSE` vs `FINE`) holds total routed
parameters and active routed FLOPs constant via the mapping in
`src/scratch_llm/eval/moe_ablation.py`.

| arm | granularity | predicted final val CE | measured final val CE | entropy > 0.9·log(N_r) | status |
|---|---|---|---|---|---|
| BIAS_FREE | COARSE | ≤ SEQ_AUX | — (pending) | yes | pending |
| BIAS_FREE | FINE | ≤ SEQ_AUX | — (pending) | yes | pending |
| SEQ_AUX | COARSE | baseline | — (pending) | yes | pending |
| SEQ_AUX | FINE | baseline | — (pending) | yes | pending |
| NONE | COARSE | worse by ≤ 0.02 nats or collapsed | — (pending) | no | pending |
| NONE | FINE | worse by ≤ 0.02 nats or collapsed | — (pending) | no | pending |

**Expected falsifier:** `BIAS_FREE` final val CE ≤ `SEQ_AUX` with entropy
> `0.9·log(N_r)` after training.

**Kill line:** `BIAS_FREE` worse than `SEQ_AUX` by > 0.02 nats, or entropy
< `0.9·log(N_r)` after training (balancer fails to prevent collapse).

**Artifacts:** `scripts/f6_moe_ablation.py`, `tests/test_moe_ablation.py`,
`artifacts/f6_moe_ablation/`.

**Smoke run logged (2026-07-30, commit `98f62e3`) — NOT a result.** All six arms ran end-to-end
(`artifacts/f6_moe_ablation/{json,md}`) at toy scale: 30 steps, d=32, synthetic data. Numbers
(coarse val_ce ≈ 1.926–1.928 vs fine ≈ 2.296–2.300; balance score ≥ 0.97 everywhere) validate
harness plumbing only; at this scale the granularity/balancer comparisons are vacuous and no
"coarse arms equivalent / fine degrades" conclusion may be cited from them. Table rows stay
`pending` until re-run at ≥1B real tokens per the taskspec.

## F2a · MTP train head (λ = 0.3)

Spec: `docs/FRONTIER_2026_TASKSPEC.md` §F2a. Shipped in commit `3b89119`
(`src/scratch_llm/mtp.py`, `model.py` `mtp_depth`/`_trunk`/`forward_train`,
`train.py` `mtp_loss_weight`; tests `tests/test_mtp.py`).

| quantity | prediction (pre-registered) | measured | status |
|---|---|---|---|
| MTP-head loss at init | ≈ log V | passes (`test_mtp_head_loss_at_init_is_log_vocab`) | [FACT] |
| `forward_train` main logits | == `forward()` | passes | [FACT] |
| `mtp_depth ∈ {0,1}` base forward | bit-identical, shared seed | passes (params + forward) | [FACT] |
| Δ = val_CE(λ=0.3) − val_CE(no-MTP) | ∈ [−0.02, +0.01] nats | **+0.0027 nats** (toy scale: tiny Markov corpus, 300 steps, CPU) | [FACT @ toy scale] |

**Kill line:** MTP aux degrades base val loss by > +0.01 nats — NOT hit at toy
scale. **Caveat:** the toy-scale Δ is not the d20 measurement; the bake-into-d20
decision stays falsifier-gated (nanochat's own MTP A/B failed, nanochat #481 —
see `docs/FRONTIER_2026_END_TO_END_PLAN.md` §2 S1).

## F12 · FineWeb-EDU vs ClimbMix-400B corpus ablation

Registered **before** running. Spec: `docs/FRONTIER_2026_TASKSPEC.md` §F12 and
`docs/FRONTIER_2026_END_TO_END_PLAN.md` §S0.
Thesis: at **35M params / 700M tokens / iso-FLOP** (`C ≈ 1.5×10¹⁷`), with the tokenizer retrained
**per corpus** on the same byte budget and identical decontamination/packing, **NVIDIA ClimbMix-400B**
([VERIFIED], Nemotron-CLIMB [2504.13161](https://arxiv.org/abs/2504.13161); HF card
[nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix); nanochat switch
[324e69c](https://github.com/karpathy/nanochat/commit/324e69c.patch), 2026-03-04:
2h46m → 2h01m (−27%), val_bpb 0.7465 → 0.7185) yields lower val_bpb and CORE than FineWeb-EDU-100B
on a shared, decontaminated held-out set.

| corpus | predicted final val_bpb | measured final val_bpb | predicted CORE | measured CORE | license / provenance | status |
|---|---|---|---|---|---|---|
| FineWeb-EDU-100B | baseline | **1.19197** | baseline | **0.0510** | FineWeb-EDU license (ODC-BY 1.0) | **[FACT]** |
| ClimbMix-400B | < FineWeb-EDU bpb (predict Δ ≈ −0.010 to −0.030; nanochat saw −0.028, [324e69c](https://github.com/karpathy/nanochat/commit/324e69c.patch)) | **1.30205** (Δ = **+0.1101**) | ≤ FineWeb-EDU CORE | **0.0551** | **CC BY-NC 4.0** — would have been stated in A9 model card ([nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)) | **[FACT]** |

**License note.** ClimbMix-400B is released under **CC BY-NC 4.0**
([HF card](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)). If F12 flips the d20 corpus
to ClimbMix, the A9 model card must state the license and the dataset source.

**Expected falsifier:** ClimbMix final val_bpb < FineWeb-EDU final val_bpb at iso-FLOP
(35M/700M, tokenizer retrained per corpus, shared decontaminated held-out set, bpb + CORE).

**Harness shipped 2026-07-31** (CPU-green): `eval/corpus_ablation.py` (`CorpusAblationArm`,
`split_held_out` + A0 13-gram decontam with logged overlap, `run_corpus_ablation`),
`scripts/tok_train.py` (per-corpus BPE with `--byte-budget`), ClimbMix staging in
`data/shards.py` (additive `--repo-id` generalization + stdlib GPT-2 detokenizer for
ClimbMix's tokenized-parquet layout — no raw-text column, [FACT hub-verified 2026-07-31]);
tests `tests/test_corpus_ablation.py` + `tests/test_tok_train.py`. Run CLI:
`scripts/f12_corpus_ablation.py` (toy-mode smoke ✅ CPU: per-arm BPE → shards → bpb →
incremental per-arm JSON → verdict JSON with falsifier evaluation).

**Measured 2026-07-31 on standing RTX 5090.** Both arms at iso-FLOP: 35,789,184 params,
699,990,016 tokens, Muon+AdamW, seed 0, vocab 32,768 retrained per corpus, shared 2,048-doc / 9,124,101-byte
held-out set, 13-gram A0 decontamination. Batch size reduced from default 32 to 8 due to OOM on
RTX 5090 32 GB; ClimbMix arm additionally emitted CUDA OOM warnings during final eval/CORE but
completed and wrote its result.

| quantity | FineWeb-EDU | ClimbMix | note |
|---|---|---|---|
| val_bpb | **1.19197** | **1.30205** | shared bytes; lower is better |
| CORE | **0.0510** | **0.0551** | secondary, single-seed noise floor |
| decontam overlap | 1.92% | 0.07% | docs dropped by A0 13-gram gate |
| wall time | ~58 min | ~59 min | standing box, batch 8 |

**bpb Δ = +0.1101** (ClimbMix minus FineWeb-EDU). The pre-registered falsifier
(ClimbMix bpb < FineWeb-EDU bpb) is **NOT confirmed**; the kill criterion **IS triggered**.

**Kill line:** ClimbMix bpb ≥ FineWeb-EDU bpb at iso-FLOP ⇒ keep the banked FineWeb-EDU corpus;
do **not** stage ClimbMix for the d20.

**Verdict: KEEP FineWeb-EDU for d20 (measurement-only).** ClimbMix does not beat the banked corpus at iso-FLOP on
this recipe/scale. *(Measurement-only verdict of record — superseded as the operating decision by
the operator override below: the corpus flip DID happen by override, so the CC BY-NC 4.0 license
note is NOT moot; the A9 model card must state it.)*

**User override (2026-07-31).** Despite the F12 kill criterion being triggered, the operator elected to
stage **ClimbMix** for the S3 scaling-law sweep and the d20 run, following the nanochat/Karpathy corpus
choice. This deviates from the pre-registered protocol. The override is logged here and in
`bench/RESULTS.md`; the A9 model card must still state the **CC BY-NC 4.0** license and source for
NVIDIA Nemotron-ClimbMix.

**Decision FINAL (2026-08-02).** The operator confirmed ClimbMix for S3/d20 and declined the proposed
second-scale (d8/ratio-20) confirmation arm. Consequences of record: (1) the F12 measurement above
stands unchanged — ClimbMix measured +0.110 bpb worse at 35M/700M with our recipe; the d20 corpus
choice rests on nanochat's larger-scale result, not on a scale-stable verdict of ours; (2) the d20
CORE band must be re-anchored against the *current* published ClimbMix curve (d24-class 0.257–0.269
at ~4e19 FLOPs) before P5, per E2E §S4(e) — **DONE 2026-08-02: re-anchored band 0.23–0.25 (central
≈0.24), see §S4-pre below**; (3) ~~the d20 run itself becomes the corpus arbiter at our
largest scale~~ — **struck (Amendment 2026-08-03):** the d20 trains on ClimbMix *only* (the
confirmation arm was declined), so it CANNOT arbitrate the corpus choice — there is no FWE arm to
compare against. The d20 validates the *recipe* against the re-anchored CORE band; the corpus
choice rests on nanochat's larger-scale result, full stop.

**Amendment 2026-08-03 (epistemic status of the +0.110 delta).** The +0.1101 bpb delta is DECISIVE
for the *sign* of the effect at 35M/700M single-seed — the kill criterion fired correctly. It is
NOT evidence about the effect at d20 scale: magnitude large, single-seed, scale-limited. No bpb
seed-noise floor exists anywhere in this program, so any "below noise" characterization (previously
in §S4-pre) is struck. **Pre-registered here — the pending S3-noise measurement:** 3 seeds at s1
scale (~0.5 GPU-h) to establish the bpb seed-noise floor, before any future small-scale
corpus/architecture verdict quotes a delta against noise.

**Artifacts:** `data/shards.py` (FineWeb-EDU arm), new ClimbMix staging path,
`scripts/tok_train.py` per-corpus BPE, `eval/corpus_ablation.py` (or reuse `eval/optimizer_race.py`),
`docs/RESULTS.md` §F12, `bench/RESULTS.md` §Frontier ablations.

## S3 · Scaling-law calibration sweep (pre-registration)

Registered **before** running (2026-07-31). Spec: `docs/FRONTIER_2026_END_TO_END_PLAN.md` §S3.
Thesis: our own recipe (Muon+AdamW, chosen corpus — **ClimbMix, by operator override**; the F12
measurement winner was FineWeb-EDU — vocab 32,768, untied) admits a clean
power-law fit at sweep scale, and the compute-optimal D:N it implies either confirms or
re-registers the d20's 9.6B-token budget *before* any paid run.

**Operator deviation (2026-07-31).** The F12 measurement said KEEP FineWeb-EDU, but the operator
overrode the kill criterion and selected **ClimbMix** for S3/d20 (following nanochat/Karpathy). The
scaling-law fit therefore runs on ClimbMix. Batch sizes may be reduced from the pre-registered
default for OOM safety on RTX 5090 32 GB (s1-s6 batch=8, s7 batch=4 with fallback to 2).
*(Superseded for s7/s8 by the 2026-08-02 batch-consistency decision below: s7/s8 batch 4,
fallback 2, LR held fixed.)*

**Harness fix (2026-07-31).** The S3 grid planner (`scaling/s3_sweep.py::_instantiated_params`) was
missing the `qk_norm=True` parameters that the real GPU training path adds; this caused a param-count
mismatch on the first point. Fixed so the planned `N` matches the instantiated model.

**Batch-consistency decision (2026-08-02, pre-registered before s7 launches).** The pretraining path
(`train.py`/`speedrun.py`) has **no gradient accumulation** (verified at HEAD — only `algos/sft.py`
supports it), so s7/s8 cannot hold batch 8 at d12 within the 5090's 32 GB. Decision: s7/s8 run at
the largest batch that fits (batch 4, fallback 2) with **LR held fixed — no mid-grid LR rescale**.
Rationale: a systematic batch-regime break is detectable and attributable post-hoc, whereas a
mid-grid LR change would confound the fit irrecoverably. Pre-registered validity checks: (1) the
d12 points must land on the d4/d8 envelope's log-log extrapolation — if they land HIGH, the excess
is attributed to batch, not to the law; (2) the a+b fit is reported **with and without s7/s8** as a
sensitivity arm; (3) no point estimate of the D:N optimum is quoted — the decision rule (optimum
< 15 ⇒ re-register d20's D) is evaluated over the fitted interval only. Adding grad accumulation
mid-sweep was considered and rejected: code churn on a live checkout is a larger risk to the grid
than a documented, systematic batch break on two points.

Grid: depths {4, 8, 12} × D:N {8, 20, 40} minus (12, 40) = points s1–s8 (largest: d12 @ ratio-20,
C ≈ 2.2e18, ~28 h standing box; s6/s8 droppable if budget trips). Fit `N_opt ∝ C^a`,
`D_opt ∝ C^b` in **bpb** on the pinned val set via `scaling/isoflop.py` (log-log, per-budget
min-pick).

| quantity | prediction (pre-registered) | measured | status |
|---|---|---|---|
| exponent sum a+b | ∈ [0.95, 1.05] (forced by C = 6ND) | **1.0000** (a=0.4515, b=0.5485) | pass *(near-vacuous — recorded D = ratio×N = C/6N exactly, so the sum is identity; see measured note)* |
| log-log fit R² (bpb) | ≥ 0.98 | **R²_N = 0.7709, R²_D = 0.8324** | **FAIL — gate tripped (< 0.98 ⇒ fit rejected), no extrapolation reported (trigger T1)** |
| compute-optimal D:N | ≥ 15 ⇒ d20 stays ratio-20 (9.6B, deliberate inference-aware overtrain); < 15 ⇒ re-register D before P5 | 26.94 on the failed fit (78.24 in the w/o-s7 sensitivity arm) — **not a usable point estimate**; decision taken on the pre-registered fitted-interval rule, see below | **HOLD ratio-20 (9.6B) — by default rule + s6-vs-s7 evidence, not by fit** |
| nanochat CORE-fit at d20's C = 2.77e19 | ≈ 0.195 — inside the pre-registered 0.19–0.22 band | **0.1949** — arithmetic evaluation of the *published* fit, not a measurement (oracle overlay, independent of our fit) | n/a (oracle self-check) |

**Kill:** s5–s6 (59M, ≤2.4B tokens) cannot beat the toy-corpus loss floor by a clear margin ⇒
data/recipe bug; stop, do not scale. **Not triggered** — s5 0.9676 / s6 0.9498 bpb, far clear of
any toy-corpus floor; recipe healthy.

### Measured — S3 sweep s1–s7 (2026-08-02, ClimbMix, standing RTX 5090, 12.09 GPU-h total)

| rung | depth | params | ratio | tokens | C | val_bpb | val_loss | wall |
|---|---|---|---|---|---|---|---|---|
| s1 | 4 | 19.99M | 8 | 159.9M | 1.92e16 | 1.2553 | 3.5523 | 0.14 h |
| s2 | 4 | 19.99M | 20 | 399.8M | 4.80e16 | 1.1772 | 3.3314 | 0.34 h |
| s3 | 4 | 19.99M | 40 | 799.7M | 9.59e16 | 1.1408 | 3.2283 | 0.68 h |
| s4 | 8 | 59.26M | 8 | 474.0M | 1.69e17 | 1.0124 | 2.8652 | 0.81 h |
| s5 | 8 | 59.26M | 20 | 1.185B | 4.21e17 | 0.9676 | 2.7382 | 2.04 h |
| s6 | 8 | 59.26M | 40 | 2.370B | 8.43e17 | 0.9498 | 2.6879 | 4.06 h |
| s7 | 12 | 135.29M | 8 | 1.082B | 8.79e17 | 0.9402 | 2.6608 | 4.01 h (batch 4) |

Batch per the pre-registered consistency decision: s1–s6 batch 8, s7 batch 4 (LR fixed, no
grad accumulation). Figures: `artifacts/s3_scaling_sweep/figs/fig1–fig4`.

**R² gate failure — cause and verdict.** The gate raised `ValueError: log-log R²=0.7709 < 0.98`
(T1 semantics), so no N\*(C)/D\*(C) extrapolation is quoted. The cause is **geometric, in the
grid itself**: because recorded D = ratio×N and ratios cycle 8→20→40 within each depth ray,
the min-picked D zigzags (0.16→0.40→0.80→0.47→1.19→2.37→1.08B) as budgets cross depth
boundaries — no power law can fit that, by construction. The a+b=1.0000 pass is likewise
near-vacuous (identity of the C=6ND wiring), not evidence of a clean law. This is a **grid
design limitation, not a data/recipe bug**: bpb descends cleanly and monotonically along
every ray, and s7 passes pre-registered validity check (1) — it lands *below* the d8-ray
log-log extrapolation at its C (0.9402 vs ≈0.9487), i.e. no batch-attributable excess.
Sensitivity arm (2): without s7, a=0.3655/b=0.6345, R² 0.7310/0.8912, ratio@d20 = 78.24 —
wildly unstable, confirming the ladder cannot support extrapolation to C = 2.77e19 (~33×).

**What the data does support (fitted-interval evidence only):** the one genuine iso-FLOP
allocation comparison is **s6 vs s7** (C 8.43e17 vs 8.79e17, 1.043×): the larger model at
ratio 8 **beats** the smaller model at ratio 40 by −0.0096 bpb. Combined with within-ray
monotone improvement, the compute-optimal allocation at ~9e17 sits at **ratio ≤ 8**
~~and drifting downward with scale~~ — *(Amendment 2026-08-03: "≤8 at ~9e17" is INDICATIVE,
possibly batch-confounded — s7 ran batch 4 vs s6's batch 8 at fixed LR, and the pre-registered
validity check was one-sided (only HIGH landings were attributed to batch). "Drifting downward
with scale" is struck: a single pair cannot support a trend. Matched-batch rerun pre-registered:
d12@r8 at batch 8, ~4 GPU-h, before P5.)* Qualitatively Chinchilla-consistent (optimal N grows
with C), but at far lower D:N than the registered 20.

**D:N decision for d20 (HOLD, per rule 3 of the batch-consistency pre-registration).** The
decision rule is evaluated over the fitted interval, not the broken extrapolation: nothing in
[s1, s7] shows ratio-20 *hurting* at any budget (s2/s5 beat their ratio-8 siblings at equal N),
and the registered 9.6B-token budget was already a **deliberate inference-aware overtrain**
(Sardana 2401.00448), knowingly above the training-optimal ratio. **Decision: HOLD at
ratio 20 / 9.6B tokens.** The honest caveat: our ladder cannot certify 20 as optimal, and the
s6-vs-s7 crossing suggests the training-optimal ratio at d20's C is likely single-digit
(indicative only — batch-confounded, see the Amendment 2026-08-03 above) —
acceptable for an inference-serving artifact, recorded here so P5 can re-anchor.

**Trigger status (E2E §S3(g)).** **T1 has formally fired** (fit failure ⇒ d14 rung). d14 is a
free-but-slow standing-box run (multi-day at d12-scale+) or a paid spot run; per the mission
guardrails no paid work without explicit human go-ahead, and the free d14 blocks P5 prep by
days. **Surfaced for human decision: run d14 (free, slow), rent for d14 (paid), or accept the
HOLD-on-interval-evidence rationale above and proceed to P5.** T2/T3 not evaluated (P5 and
F8/F10 respectively, both downstream).

**Amendment 2026-08-03 (T2 re-anchored).** T2 as written in E2E §S3(g) references "the S3-fit
prediction", which does not exist post-T1 (the fit was rejected). T2's anchor is re-based on the
measured d12 ray — s7: d12@r8 bpb 0.9402 (batch-4 caveat noted) — extended within-ray to ratio 20:
**P5 expected band 0.89–0.92 bpb** (indicative, fitted-interval estimate, not an extrapolation).
T2 fires if |P5 measured − band| > 0.01 bpb (or CORE oracle-divergence > 0.02) ⇒ d16.

## S3.5 · Calibration re-fit — pre-registration (2026-08-03, per `FRONTIER_2026_SCALING_PROGRAM.md`)

**Why.** T1 fired: the S3 grid cannot produce a law (§S3 above). The K3 front has a hard
dependency on a certified calibration law (the family fit is the attention-architecture
decision). This re-fit is the redesigned second — and final — attempt; if its gates fail,
the d20 D:N decision stays HOLD-by-economics permanently.

**Preconditions (stage 0–1 of the program doc).** P5 locks recipe-v1 (LR*/wd*) first. If LR*
moves outside ×[0.7, 1.4] of the S3 recipe LR, s2+s5 are re-run at recipe-v1 before the fit.
The matched-batch d12@r8 rerun (batch 8, ~4 GPU-h — registered above) runs in the same stage.
**Recipe-surface parity check before anything trains** (tokenizer, untied, qk_norm, schedule,
mtp_depth — see the open decision below).

**Points (recipe-v1, full production schedule shape per point, pinned val set, logged batch
per the S3 batch-consistency precedent — largest that fits, LR held fixed):**

| point | N | D | C | role |
|---|---|---|---|---|
| d12-r20 (= ladder s8, re-homed from P5) | 135.29M | 2.706B | 2.20e18 | completes the d12 ray |
| d14-r8 | ~193.6M | 1.549B | 1.80e18 | new depth, leverage toward d20 |
| d14-r20 (optional, if box time allows) | ~193.6M | 3.872B | 4.50e18 | span extends to ~235× |

Plus s1–s7 (recipe-v0, flagged per the stage-1 verdict) and the P5 d12@r8 confirmation point
(recipe-v1) ⇒ 9–10 points vs 5 fit parameters.

**Estimator (fixed in advance).** Joint parametric `L(N,D) = E + A·N^(−α) + B·D^(−β)`;
Huber loss (δ = 1e-3) on **log-space residuals, summed not averaged**; L-BFGS-B from a grid
of initializations (Besiroglu corrections). D is the recorded token count, never the C/(6N)
bridge; a+b reported as a diagnostic, not enforced.

**Acceptance gates (ALL four required; they replace the bare R² gate):**
1. R² ≥ 0.98 (necessary, not sufficient);
2. bootstrap (≥1000 resamples over runs) 90% prediction-interval half-width at C = 2.77e19
   **≤ 0.02 bpb**;
3. leave-one-out swing of ratio@(2.77e19) **≤ 2×** (S3 today: 2.9×);
4. no structured residual trend vs depth (the S3 zigzag signature).

**Outcomes.** Pass ⇒ D:N at the d20's C quoted *with its interval* (measured answer replacing
HOLD-by-economics, or HOLD confirmed inside the interval — both are wins) and the d20 run
becomes the law's pre-registered holdout test (measured bpb must land inside the PI).
Fail ⇒ report; D:N stays HOLD-by-economics; **no third attempt**.

**OUTCOME (2026-08-08, measured on the vast.ai RTX 6000 Ada pod, ~81 GPU-h / ~$46–48 total
program cost).** All 13 points completed on recipe-v1 — **muon_adamw, standard parametrization,
cosine schedule, warmup = steps/20, η\* = 0.0021 fixed** (corrects an earlier handoff document
that mislabeled the grid "AdamW"; code-verified via `s3_sweep.py:395` → `speedrun.py:62`
default — ADR-0020 status note 2). Zero bpb inversions across the grid. Final joint fit (13
points): `L(N,D) = 0.2061 + 6401·N^(−0.589) + 2380·D^(−0.502)`, log-L R² = 0.9983, 46/48
starts in the best basin. **Gates: R² PASS (0.9983) · bootstrap PI half-width FAIL (0.0207 vs
≤ 0.02 — marginal) · LOO D:N swing FAIL (3.15× vs ≤ 2×; worsened from 1.86× when the
highest-leverage point d14-r20 was added) · residual trend PASS (p=0.62). Overall: REJECTED.**
Per the pre-registered clause above: D:N stays HOLD-by-economics permanently; **no third
attempt**. Interpretation (labeled directional, not decision-grade): the loss prediction is
stable (12→13-point point estimate moved 0.2818 → 0.2782 bpb) but the N-vs-D *allocation* is
lever-arm-unstable at the 2.5× extrapolation from 195M to 480M params — the gates correctly
refused to certify it. Both rejected fits put the compute-optimal ratio at ≈3.4–3.8 at the
d20's C, consistent with nanochat's measured 8–12 (reported, not ours) — ratio-20 therefore
stands as a *deliberate, bounded* inference-economics overtrain, and the ADR-0020
re-registration hook does not fire (it required a gate-passing fit). **d20 tripwire restated on
the pinned-tokenizer scale: predicted val_bpb = 0.2782, 90% PI [0.2538, 0.2952]** — this
replaces the old-tokenizer ≈0.74 cross-check, which is void. The d20 run remains the law's
pre-registered holdout test (measured bpb should land inside the PI). Side products: P5 LR bowl
well-formed (×0.7 winner, tie-break rule validated); d8 width probe confirms LR transfer at
fixed depth; s7 anomaly fully explained as the batch-4 confound (batch8 − batch4 = −0.0187 bpb).
Artifacts: `artifacts/s35/`, `artifacts/s35_analysis/` (fit JSON + figures), pod commit
`4101806`.

**Decision of record (2026-08-03, operator-delegated): option (a) — the d20 runs
mtp_depth = 0.** The d20 spec said "MTP head baked into pretrain (F2a)", but the speedrun
spine never sets `mtp_depth` (default 0 — S3 ran without it), and every certified constant
(480.4M / 2.77e19 / 18,311 steps / the 0.89–0.92 T2 band) assumes mtp_depth=0. Enabling it
would add ≈ +14·d² params (d20: ≈ +22.9M ⇒ N ≈ 503M), shifting C, step-time, cost, and the
anchor band days before a paid run; F2a's measured falsifier was neutral (+0.0027 nats), so
no training gain is forgone. Option (b) (enable + re-derive N/C/cost/band before P5) was
rejected: the d20's value is comparability with the anchor chain, and (a) is the reversible
choice. F2a/F2b retarget to a post-d20 run.

## S4-pre · d20 CORE band re-anchor (2026-08-02 — pre-P5, per E2E §S4(e) + §F12 consequence 2)

**Why.** The pre-registered band **0.19–0.22** (anchor: nanochat's original d20 CORE 0.2219 @
3.77e19; cross-check: nanochat CORE-fit `1 − 3.7555·C^−0.0344` = 0.195 @ our 2.77e19) was
calibrated to the **Oct-2025 FineWeb-EDU recipe**. The corpus moved to **ClimbMix** (F12
operator override, FINAL 2026-08-02), and nanochat's *current* published ClimbMix curve sits
materially above that fit. E2E §S4(e) therefore requires re-anchoring before P5.

**Facts used (all published/verified in-repo).**
- nanochat CORE-fit (FWE-era): `CORE = 1 − 3.7555·C^(−0.0344)` → 0.169 @ miniseries C=1.09e19
  (actual 0.1708 ✓ — the fit is sound *for its era*), 0.195 @ our d20 C=2.77e19, **0.205 @ 4e19**.
- nanochat current ClimbMix curve: **0.257–0.269 @ ~4e19** (leaderboard d24 ratio-8 FP8:
  0.2626, val_bpb 0.718). Recipe-progress delta vs the stale fit at matched C:
  **+0.052..+0.064** — 3–8× the CORE noise floor (±0.008–0.016, nanochat 7× rerun
  0.2512–0.2677), so it is *real recipe progress* (ClimbMix data + FP8 + tuning), not noise.
- GPT-2 XL anchor: 0.256525.

**Re-anchor arithmetic (two methods, identical result).**
- (A) *Progress shift:* stale fit @ 2.77e19 (0.195) + the matched-C progress delta
  (+0.052..+0.064) = **0.247–0.259**.
- (B) *Curve step:* take the current ClimbMix curve at 4e19 (0.257–0.269) and step down to
  2.77e19 using the fit's own slope (Δ = −0.008 over 0.69× C — the curve is flat here) =
  **0.247–0.259**.
- *(Amendment 2026-08-03: the "two methods, identical result" agreement is overstated — (A) and
  (B) are algebraically the same computation, so this is NOT independent corroboration. Arithmetic
  slip noted in (B): the fit's own slope gives fit(4e19) ≈ 0.204 and fit(2.77e19) ≈ 0.194, i.e.
  Δ ≈ −0.010, not −0.008 — which would give 0.249–0.261. The headline band 0.23–0.25 stands on
  method (A) plus the −0.01..−0.02 recipe discount — itself unverified until P5 measures it.)*
- **Recipe discount (ours vs their leaderboard stack):** we train ratio-20 (deliberate
  overtrain vs their optimal ≈8–10.5), bf16 not FP8, first-run recipe maturity; and our own
  F12 measured ClimbMix *worse* than FWE at 35M (single-seed, scale-limited — decisive for the
  sign at that scale, silent about d20 scale; no bpb seed-noise floor exists in the program — the
  pending S3-noise measurement is pre-registered in §F12; and no evidence we
  yet reproduce nanochat's ClimbMix edge). Discount **−0.01..−0.02**.

**Re-anchored band (supersedes 0.19–0.22): CORE 0.23–0.25, central ≈ 0.24.**
- The **KILL <0.15** tripwire is unchanged — it detects stack bugs, far below any
  recipe-difference scenario.
- **GPT-2 XL parity (0.2565) is the stretch case, not the base case** — central estimate sits
  0.017 below; the band's top edge (0.25) just touches parity territory. This is the honest
  framing for the report card.
- Cross-check from our own ladder: chaining our measured s7 bpb 0.9402 @ 8.79e17 to nanochat's
  d24 0.718 @ 4e19 (log-log slope −0.071) predicts **d20 val_bpb ≈ 0.74** — between the two,
  closer to d24; sane (indicative only: cross-recipe chaining, and §S3 showed our ladder
  cannot certify extrapolations on its own).
- Width note: the band (±0.01) is ≈ the CORE noise floor; a single d20 CORE number cannot
  resolve finer. One seed, reported with the noise floor attached.

**Consequence for P5/d20 gates:** the P5 CORE-vs-public-checkpoint comparison and the d20
report card now score against **0.23–0.25 (central 0.24)** on the ClimbMix curve; a d20 CORE
in 0.19–0.22 would be a *miss* vs the re-anchored expectation (recipe gap to investigate),
not a confirmation. Updates applied: `FRONTIER_STATUS.md` d20 spec table + ledger, E2E §S4(e),
TASKSPEC §F12 + Next-node marker, `bench/RESULTS.md` §$100 d20 run (amendment — the original
0.19–0.22 row is preserved as the pre-registration of record).


**d14/d16 escalation:** trigger-gated per E2E §S3(g) (T1 fit failure ⇒ d14; T2 P5-anchor
divergence > 0.01 bpb ⇒ d16; T3 F8/F10 sweep-scale win + RULER pass ⇒ d14 confirm; none ⇒ skip,
d16 stays the $90-abort fallback).

**Artifacts:** `src/scratch_llm/scaling/s3_sweep.py`, `scripts/s3_scaling_sweep.py`,
`artifacts/s3_scaling_sweep/`, `tests/test_s3_scaling_sweep.py`.

## K3 · Anatomy + interaction of the K3 stack (pre-registration)

Registered **before** running (2026-07-31). Spec: `docs/k3/ABLATIONS.md` (arm configs,
iso-FLOP rule, eval protocol). R0 is CPU/weight-space and its census is **COMPLETE** (all four
rows below done, 2026-07-31); R1/R2 launch only after
`core/kda.py` is PROVEN (HANDCRAFTED.md) and mini-K3 is wired into the speedrun spine (K6).

| arm | prediction (pre-registered) | measured | status |
|---|---|---|---|
| R0 closure | census param count == config-derived == HF total | **EXACT 0-residual closure** (2,779,931,837,184; 2026-07-31); A_log mismatch found ([128] ckpt vs [96] ref code) | **done** |
| R0 `attnres` census | layers ≥ 1 res_proj weights non-zero; layer 0 ≈ 0 | ✓ all 186 non-zero; layer-0 attn-side absmax 1.7e-05 (no-gradient + weight decay); query L2 grows with depth (0→2→3→6) | **done** |
| R0 `qb` census | per-layer bias mean ≈ 0 (construction), no collapse; structure over depth | ✓ mean ≈ 0 all 92 layers; std drifts up with depth (0.020→0.098) — balancing harder deep | **done** |
| R0 `decay` census | dt_bias ≪ 0 across KDA layers (default-long-retention); A_log spreads per layer | ✓ dt_bias −4.63 ± 0.05 all 69 layers (no flips); A_log means −0.17→+0.29 with depth | **done** |
| R1-A (KDA-hybrid vs GQA) | ≥ 1% val_bpb at iso-FLOP; gap grows with context | — | pending |
| R1-A×B (interaction) | AttnRes marginal gain larger on GQA corner than KDA corner (substitution) | — | pending |
| R1-C (LatentMoE+QB, 64×4) | ≈ standard MoE + sign-step within ±0.5% bpb (null) | — | pending |
| R2 g_min sweep | flat ∈ [−3,−7]; degrades ≤ −10; bf16 chunkwise unstable at ≤ −10 (e^160 > BF16 max) | — | pending |
| R2 KDA vs GDN | KDA ≥ 0.5% val_bpb at iso-FLOP | — | pending |

**Kill lines:** R1-A: KDA-hybrid loses to GQA at iso-FLOP ⇒ the architecture claim does not
transfer to our scale — stop and report, do not re-tune to force it. R2: no measurable
degradation at g_min = −10 in bf16 ⇒ the paper's hardware rationale is weaker than claimed
(report it, then re-examine our chunkwise precision split). R1-A×B: |Δgain| < 0.25% bpb ⇒
additivity holds at 0.4B (record as null, still a result).

**Artifacts:** `scripts/k3_fetch_tensors.py`, `artifacts/k3_anatomy/`,
`tests/test_k3_param_count.py`, `docs/k3/ABLATIONS.md`.
