# Scaling from first principles — re-plan after two rejected fits (2026-08-11)

Status: **analysis + decision recommendation**, research-grounded. Supersedes nothing; amends
`FRONTIER_2026_SCALING_PROGRAM.md` (whose S3.5 "no third attempt" clause stands) and feeds the
P5/d20 gate. Honesty labels per claim: **[measured]** = measured by us · **[reported]** = published
by others · **[derived]** = arithmetic from labeled inputs.

---

## 0. The decision this document exists to serve

The d20 run needs exactly three numbers decided: **N = 480.4M** (fixed by the anchor chain),
**D** (currently ratio-20 = 9.6B tokens, HOLD-by-economics), and **LR/batch at 524,288-tok batch**
(currently the √B-transferred 0.011879, P5.5-probe-gated). The question this re-plan answers:
*does any further scaling-law work change those numbers, and if not, what scaling work (if any)
is worth doing at all?*

**Answer (decision of record, proposed):** no further *global-law* work — the two rejected fits
(S3, S3.5) already told us the ladder can't certify allocation at the required lever arm, and the
first-principles analysis below shows the d20 decision is **not law-limited**. What de-risks d20 is
a **hyperparameter-transfer law** (the thing DeepSeek/MiniCPM/Step-Law fit *before* any
architecture law), which the P5.5 probe already operationalizes. What has future-science value
(K3 family fits, d24+) is a **WSD-based IsoFLOP grid** (Hägele et al. 2024 cost structure), which
we defer until it has a customer.

---

## 1. What frontier labs actually published and rely on

Ordered by evidential weight for our situation.

### 1.1 Chinchilla's three approaches are not interchangeable — and Approach 2 is the workhorse
Hoffmann et al. (2022) fit the same data three ways (fixed-N varying-D frontiers; **IsoFLOP
slices**; joint parametric fit) and reported Approach-1/2/3 in that order of trust. Besiroglu et
al. 2024 (*Chinchilla Scaling: A replication attempt*, arXiv 2404.10102) then showed the published
Approach-3 numbers were **mis-optimized**: Huber loss **averaged instead of summed** (optimizer
stopped early), CIs implausibly narrow (would need >600k runs; they ran <500), and rounded TeX
parameters that bias predictions. Their corrections — Huber δ=1e-3 on log-residuals **summed**,
L-BFGS-B from an initialization grid — are exactly what S3.5 implemented, which is why our fit was
numerically clean (R² 0.9983) yet still uncertifiable. **Lesson: fitting mechanics are solved;
the failure mode is leverage and noise, not numerics.** [reported]

### 1.2 Parameter counting is a first-order confound at our scale — Pearce & Song 2024
*Reconciling Kaplan and Chinchilla Scaling Laws* (arXiv 2406.12907) attributes the Kaplan-vs-
Chinchilla exponent war to (a) **non-embedding vs total parameter counting** and (b) fitting at
small scale, where embeddings are a large fraction of N. Our ladder is the extreme case of (b)
**[derived, from `scaling/s3_sweep.py` config `d_model = 64·depth`, vocab 32768 untied]**:

| point | N total | embed + unembed | fraction |
|---|---|---|---|
| s1–s3 (d4, d_model 256) | 19.99M | 16.78M | **84%** |
| s4–s6 (d8, d_model 512) | 59.26M | 33.55M | 57% |
| s7 (d12, d_model 768) | 135.29M | 50.33M | 37% |

The parametric form `L = E + A·N^(−α) + B·D^(−β)` assumes N is a homogeneous compute-relevant
quantity; at our scale N is mostly a **lookup table** (embedding gather ≈ 0 FLOPs; only the
unembedding head costs 6·V·d·D). Our `C = 6ND` bookkeeping therefore both (i) overstates compute
per token at small N and (ii) makes the fitted `α` describe embedding dilution more than
transformer capacity. This alone can produce the lever-arm LOO instability the gates caught.
**Any future fit must use non-embedding N and measured FLOPs (incl. attention at ctx 2048), never
the C=6ND bridge** — the bridge is also what made our a+b=1.0000 "pass" vacuous. [reported +
derived]

### 1.3 Hyperparameter laws come first — DeepSeek (2401.02954), Step Law, MiniCPM
DeepSeek's scaling paper fits **LR\* and batch\* as power laws of C from IsoFLOP grids *before***
fitting the architecture law; "Predictable Scale / Step Law" (arXiv 2409.04777 line of work) and
MiniCPM's muP-flavored transfer do the same. The ordering is the point: **an architecture law fit
on mistuned runs measures your hyperparameter error, not the architecture.** Our program did this
backwards until P5 (η\* = 0.0021 swept at d12/batch-16K **[measured]**, then √B-transferred to
batch-524K — a rule nanochat itself labels *"not studied carefully, assumption!"* **[reported]**).
The P5.5 probe (3 arms ×{0.7,1.0,1.4} at the true d20 config) is precisely the DeepSeek-style
hyperparameter measurement, one point at the target C. It is the single highest-EV scaling
measurement available to us and it is already pre-registered. [reported + in-repo]

**Muon transfer note.** Moonlight (arXiv 2502.16982) reports Muon ≈ 2× compute efficiency over
AdamW *at compute-optimal* [reported, their sub-2B-dense fits, extrapolated] and — more relevant
to us — that **RMS-matched updates + decoupled WD make one LR/WD schedule serve both optimizers
and transfer across width**. Audit result: our `optim.py` Muon already implements exactly this
(`rms_scale·√max(A,B)`, default 0.2, into AdamW's 0.2–0.4 update band; decoupled wd; `optim.py:221–234`).
So our recipe is *transfer-ready by construction* in width; the untested residual is **batch
scaling** (the √B assumption) — which is what P5.5 measures. [measured in-repo]

### 1.4 Cosine is the cost multiplier; WSD + cooldown collapses it — Hägele et al. 2024
*Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (arXiv 2405.18392):
a **constant LR with a short cooldown** matches cosine at matched horizon, and because the trunk
never commits to a duration, **one trunk run yields many (N, D) points** — branch cooldowns at
several D per size, plus SWA giving free mid-run estimates. Chinchilla-style grids are expensive
*precisely because* cosine must be re-horizoned per point; our s1–s7 grid paid full price per
point. If we ever run another sweep, it runs WSD. [reported]

### 1.5 Overtraining is reliable and the optimum is flat — Gadre 2024 + Sardana & Frankle 2024
Gadre et al. (arXiv 2403.08540) measure scaling with D/N up to ~1500 and find it **reliably
predictable** with an overtraining-corrected law [reported]; Sardana & Frankle (arXiv 2401.00448)
formalize inference-aware allocation: for a serving artifact, optimal N is several× below
training-optimal and D correspondingly above [reported]. Combined with the known flatness of the
IsoFLOP minimum (loss is locally quadratic in log N; a 2–3× allocation miss costs a small
single-digit % of loss), this means: **ratio-20 at d20 is a deliberate, bounded, literature-
endorsed overtrain** — S3.5's rejected fits put the training-optimal ratio at ≈3.4–3.8 (directional
only), so ratio-20 is a ~5× overtrain, deep inside Gadre's reliably-modeled regime and justified
by Sardana-Frankle economics for a model whose purpose includes being served. [reported + derived]

### 1.6 Seed noise is the floor no gate can beat — measured by us
Our 3-seed floor at s1: **max−min 0.024 bpb, σ = 0.012** [measured]. S3.5's bootstrap PI gate
(≤0.02 bpb half-width at 2.5× extrapolation from 13 noisy points, 5 free params) was set *at* the
noise floor — it was almost guaranteed to fail. Frontier practice is not to bootstrap-certify
33× (or even 2.5×) extrapolations from noisy small runs; it is to **extrapolate one octave and
verify at the target** (every lab's "predict the flagship, then train it" protocol). Any future
gate must be sized against the measured noise floor, e.g. PI threshold ≥ 2·σ_seed·√(leverage).
[measured + reported]

---

## 2. Autopsy — why our two fits failed, mapped to first principles

| failure | root cause | the published fix |
|---|---|---|
| S3 R² gate (0.77/0.83) | grid confounded N↔D by fixed ratios; min-picked D zigzags across depth boundaries | IsoFLOP design (Approach 2): vary N at fixed C |
| S3 a+b "pass" vacuous | recorded D = C/6N by construction | measured D + measured FLOPs, never the bridge |
| S3.5 LOO swing 3.15× | lever arm 2.5× on N that is 37–84% embeddings (§1.2) | non-embedding N; larger model-to-embedding ratio |
| S3.5 bootstrap PI 0.0207 vs ≤0.02 | gate set below the seed-noise floor (§1.6) | noise-aware gate sizing; replicate leverage points |
| both fits: allocation unstable while **loss prediction stable** (0.2818→0.2782) | flat optimum + quadratic local geometry — the signal we actually need for d20 (loss at fixed N, D chosen) is the *stable* one | stop asking the ladder for allocation; ask it only for loss prediction, with PI |

The unifying statement: **we asked a 13-point, embedding-dominated, single-seed ladder a
question (allocation at 2.5× lever) that the information content of the data cannot answer, and
our gates correctly refused.** The loss-prediction question it *can* answer is the one d20
actually needs (tripwire 0.2782, PI [0.2538, 0.2952] — already registered).

---

## 3. The re-plan

### T0 — d20 allocation: CLOSE the question (no new measurement)
Decision of record: **N = 480.4M, D = 9.6B (ratio-20), mtp_depth = 0** — deliberate
inference-aware overtrain, per §1.5. Kill-reopen only if P5.5's LR probe or the d20 holdout
tripwire fires. This converts "HOLD-by-economics" from a default into a cited decision.

### T1 — Hyperparameter transfer: the only scaling measurement that de-risks d20
1. **P5.5 probe as pre-registered** (3 arms, 916 steps, 8×H100 first hour, $9–12) — unchanged.
2. Add to the probe's logged outputs: per-arm loss-at-step-{100,300,916} so a **η\*(C) two-point
   check** (d12-P5 point vs d20-probe point) can distinguish "√B right" from "√B lucky at 916
   steps". Zero extra cost — logging only.
3. If lr1.4 wins outright (pre-registered falsifier): do **not** silently take the win — the
   pre-registered action is re-deriving the transfer rule first.

### T2 — Defer the certified-allocation law until it has a customer (K3 family fits / d24)
When (and only when) a future run needs a defensible D:N, build the grid the way the literature
says, at an estimated **15–25 GPU-h** on a rented 5090/6000 (~$8–15), not the ~81 GPU-h S3.5 paid:
- **WSD trunks**: 4–5 sizes (embedding fraction ≤ 35% — i.e. d12 and up, or tied embeddings if the
  recipe allows), constant LR at the P5-derived η\*, cooldown branches at 3 token counts each ⇒
  12–15 (N, D) points from 4–5 trunk runs (Hägele cost structure).
- **Accounting**: non-embedding N; measured FLOPs (matmul counter incl. attention and unembed,
  excluding the embedding gather); recorded D; seeds ×2 at the two highest-leverage points sized
  against the 0.024 bpb floor.
- **Fit**: Approach-2 quadratics per budget first (the robust one), parametric fit second
  (Besiroglu corrections — already in our `scaling/isoflop.py` lineage), and **extrapolate ≤ one
  octave (≤4× C) with a target-scale holdout**, never 33×.
- **Gates**: PI threshold sized to the measured noise floor (§1.6); LOO on log-ratio ≤ 2×; the
  target run remains the law's pre-registered holdout test.

### T3 — Standing metrology rules (zero cost, binding from now)
1. No bpb comparisons across stagings (val = staging tail, `speedrun.py:344`) — documented trap.
2. No allocation claim (D:N "optimum") quoted without a gate-passing fit *and* a target-scale
   holdout; directional labels mandatory otherwise.
3. Any small-scale verdict quotes its delta against the measured floor (0.024 bpb at s1) or a
   floor re-measured at the relevant scale.
4. Corpus/architecture decisions at new scales require either iso-FLOP measurement at that scale
   or an explicit "rests on [reported] external evidence" label (the F12 precedent).

---

## 4. What we will NOT do

- **No third global L(N,D) fit** on the existing ladder (S3.5 clause, reaffirmed — the data
  cannot answer the question; re-fitting the same points with a different estimator is not new
  information).
- No R²-only acceptance of any fit, ever (R² 0.9983 coexisted with 3.15× LOO swing).
- No use of published exponents (Chinchilla 20:1, nanochat 8–12, our directional 3.4–3.8) as a
  *certification* of d20's ratio — they are context, not evidence, for our recipe at our scale.
- No mid-grid LR or batch changes on any future sweep (the S3 batch-4 confound cost us the
  cleanest comparison we had).

## 5. References (labeled)

- Hoffmann et al. 2022 (Chinchilla) — three approaches [reported]
- Besiroglu et al. 2024, arXiv 2404.10102 — replication; Huber-summed, init grid, CI critique [reported]
- Pearce & Song 2024, arXiv 2406.12907 — embedding-counting + small-scale bias [reported]
- DeepSeek-AI 2024, arXiv 2401.02954 — hyperparameter laws first; IsoFLOP method [reported]
- Hägele et al. 2024, arXiv 2405.18392 — constant-LR + cooldown + SWA; scaling-experiment cost [reported]
- Gadre et al. 2024, arXiv 2403.08540 — reliable overtraining to D/N ~1500 [reported]
- Sardana & Frankle 2024, arXiv 2401.00448 — inference-aware allocation [reported]
- Liu et al. 2025 (Moonlight), arXiv 2502.16982 — Muon scaling; RMS-match transfer [reported]
- Muennighoff et al. 2023, arXiv 2305.16264 — data-constrained scaling (context for ratio ≥ 40) [reported]
- In-repo: `docs/RESULTS.md` §S3/§S3.5/§F12/§S4-pre; `bench/RESULTS.md` noise floor + P5.5
  pre-registration; `src/scratch_llm/optim.py:221–234` (Muon RMS-match audit) [measured]
