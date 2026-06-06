# A3 — Scaling · BUILD GUIDE (CS336 → reasoningLLM L3 Scaling)

> **One-liner.** CS336 A3 teaches you to *fit a curve and extrapolate it under a fixed FLOP budget* — IsoFLOP profiles + a compute-budgeted training-API "leaderboard." **Layer L3 (Scaling).** **Source file fed:** `src/reasoning_llm/scaling/hack_rate_fit.py`. **Through-line tie:** the deliverable of A3 is not the Chinchilla numbers — it is the *fitter machinery*. Repurpose it to fit a SAFETY/RL curve (`hack_rate` / `true_quality_gap` vs model size N and inference compute C_infer), **not just loss.** "Measure the thing you're scaling." (`true_quality_gap = reward − true_quality`.)

---

## 1. What CS336 actually requires (every deliverable)

A3 has **two graded problems** of very unequal weight: a 5-point synthetic IsoFLOP fit, and a **50-point** real-training-API leaderboard. Everything in §3 of the PDF (the training API, budget accounting, endpoints) is scaffolding for that 50-pointer.

| # | Deliverable (PDF) | PDF section | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|---|
| 1 | **`chinchilla_isoflops` — IsoFLOPs scaling laws** (5 pts): load `data/isoflops_curves.json`; for each compute budget C_i take the run with **lowest `final_loss`** as N_opt(C_i) (PDF explicitly says: skip Hoffmann's per-profile quadratic fit, just take the min). | §2.1 | fit `N_opt ∝ C^a` on the `(C_i, N_opt)` pairs | 1.5–2 h | **LOAD-BEARING** (this *is* the fitter) |
| 2 | **(a) Extrapolated N_opt** + plot of `(C_i, N_opt(C_i))` with the power-law line, extrapolated to ≥10^24 FLOPs; predict N_opt at **10^23 and 10^24** FLOPs; one-sentence answer. | §2.1(a) | power-law fit + extrapolation + plot | 0.5 h | **LOAD-BEARING** |
| 3 | **(b) Extrapolated D_opt** + plot of `(C_i, D_opt(C_i))`; predict D_opt at 10^23 and 10^24; one-sentence answer. D_opt derived via **C ≈ 6ND ⇒ D_opt = C / (6·N_opt)**. | §2.1(b) | second power-law `D_opt ∝ C^b` | 0.25 h | **LOAD-BEARING** |
| 4 | **`scaling_laws` — leaderboard** (50 pts): query the training API under a **12 B200-hour** fitting budget (= 25% of the 48 B200-hr "big run"); fit a scaling law over the queried `(config → val_loss)` points; **choose the model size + hyperparameters** predicted to minimize val loss at 48 B200-hr; predict that loss. | §3, §3.3 | budget-aware query planner + multi-point loss fitter | 4–6 h | **LOAD-BEARING** (the budget-query discipline) |
| 5 | **`POST /final_submission`** — submit predicted optimal `training_config` + `predicted_final_loss` to the API (graded partly on real model performance). | §3.2, §3.3 | API client call | 0.25 h | **COURSE-ROTE** (leaderboard plumbing) |
| 6 | **Training-API client** — `POST /submit`, `GET /budget`, `GET /experiments`, `GET /experiment/{id}`; poll until `status_type=="completed"`; read final `val_losses[-1]`; handle 409 (dup config) / 400 (over-budget); respect that **queued/running reserve full `max_runtime_seconds`** and only refund actual runtime on completion. | §3.1–3.2 | thin `requests` wrapper + poller | 1.5 h | **COURSE-ROTE** (Stanford-network-only API; reusable *pattern* not the code) |
| 7 | **Hyperparameter-effect analysis** — small-scale sweeps to set the *non-N* hyperparameters (depth/width ratio, LR, batch, etc.) for the predicted optimal config; PDF asks you to comment on each factor. Param count for a config est. via **non-embedding params ≈ 12·n_layer·d_model²**. | §3.3 | sweep harness (uses #6) | 1.5–2 h | **COURSE-ROTE** (good practice; not the reusable instrument) |
| 8 | **`writeup.pdf`** — methodology: which runs you queried & why, fit method, fit quality, predicted N/loss for 48 B200-hr, hyperparameter reasoning. **`code.zip`.** | §1 "How to submit", §3.3 | — | 1–2 h | **COURSE-ROTE** (Gradescope artifact) |

**Accurate-to-PDF notes that catch people out:**
- The IsoFLOP intuition the PDF gives: at fixed C_i, `final_loss` is **quadratic-ish in N** — too-small N can't absorb the compute (loss high), too-large N can't take enough gradient steps within C_i (loss high), minimum in between. You want the N at that minimum per budget.
- `C = 6ND` and `D = C/(6N)` are the only bridge between the two power laws; D_opt is *derived*, not separately fit from scratch.
- The API **fixes** `seq_len=512`, `n_validation_tokens = 2^18 ≈ 262k`, `vocab=32k`; enforces `hidden_size == num_attention_heads * head_dim`, `num_attention_heads % num_key_value_heads == 0`, `total_train_tokens % (512 * train_batch_size) == 0`. Config-consistency validation is part of "spend the budget without wasting it."
- **Budget is wall-clock, not FLOPs, and it's adversarial:** a timed-out run is charged the *full* `max_runtime_seconds`. The skill is choosing `max_runtime_seconds` tight enough to not over-reserve but loose enough to not time out. 12 B200-hr is a hard API-enforced cap (`remaining_seconds` hits 0 → 400).

---

## 2. The equations / algorithms that matter

1. **Parametric loss law (Chinchilla / Hoffmann 2022)** — `L(N,D) = E + A/N^α + B/D^β`. The irreducible-loss term `E` plus two power-law terms in params and data. *Why it matters:* this is the object whose minimum (subject to `C=6ND`) gives compute-optimal allocation. CS336 A3 does **not** make you fit all 5 parameters (E,A,B,α,β) — it uses the cheaper IsoFLOP shortcut — but you must know this is the underlying law to defend the method.
2. **Compute identity** — `C ≈ 6ND` (6 FLOPs/param/token: ~2 fwd + 4 bwd). *Why it matters:* the single equation that turns one power law (`N_opt(C)`) into the other (`D_opt(C)`), and converts a wall-clock/FLOP budget into a feasible `(N, D)` grid. Get the constant wrong and every extrapolation is off by a multiplicative factor.
3. **IsoFLOP profile construction** — for each fixed budget `C_i`, sweep N, record `final_loss`; `N_opt(C_i) = argmin_N L`. *Why it matters:* it's the *measurement primitive*. The senior reframe replaces "L" with "hack_rate" — you sweep N at fixed C and record where the *safety* metric is extremal. Same primitive, different y-axis.
4. **Optimal-allocation power laws** — `N_opt ∝ C^a`, `D_opt ∝ C^b`, with `a + b = 1` (since `C = 6ND` ⇒ exponents must sum to one). *Why it matters:* Chinchilla found `a ≈ b ≈ 0.5` (scale params and data ~equally). The `a+b=1` constraint is a built-in **sanity check** on your fit — if your two independently-fit exponents don't sum near 1, your min-picking or your `C=6ND` is wrong.
5. **Power-law fitting in log space** — `log N_opt = a·log C + const`; fit by **linear regression on log-log points** (or `scipy.optimize.curve_fit` on the power form). *Why it matters:* this is the entire mechanical core of `hack_rate_fit.py`. Cheap, robust, two parameters, gives you slope `a` (the headline) + intercept. Fit on log-log, *predict by exponentiating*, never fit raw values (heteroscedastic).
6. **Extrapolation + non-embedding param estimate** — evaluate the fitted law at a *target budget larger than any queried point* (10^23, 10^24, or 48 B200-hr); map predicted N back to an architecture via **non-embedding params ≈ 12·n_layer·d_model²**. *Why it matters:* extrapolation beyond the data is the *entire point* of a scaling law and its biggest risk — the fit is only as good as the regime stability. State the extrapolation factor (how far past your max data point) explicitly; that honesty is the senior signal.

---

## 3. Map to reasoningLLM_scratch source files

The repo's L3 file is **`src/reasoning_llm/scaling/hack_rate_fit.py`** — README/CLAUDE both describe it as *"IsoFLOP machinery repurposed to fit `hack_rate` vs compute."* The map is: **keep the FITTER, drop the Chinchilla numbers.**

| CS336 deliverable | → `scaling/hack_rate_fit.py` component | Keep / adapt (the machinery) vs course-only |
|---|---|---|
| #1 IsoFLOP min-picking (`argmin_N L` per C_i) | `isoflop_min(points, budget_key, metric_key) -> (C_i, x_opt)` — generic over the y-metric | **KEEP** the per-budget argmin primitive. **Course-only:** that the metric is `final_loss` and the data is `isoflops_curves.json`. Generalize `metric_key` so it accepts `hack_rate` / `true_quality_gap`. |
| #2/#3 power-law fit + extrapolate (`N_opt ∝ C^a`, `D_opt ∝ C^b`) | `fit_powerlaw(xs, ys) -> PowerLaw(a, const)` + `PowerLaw.predict(x)` (log-log linfit) | **KEEP** verbatim — this is *the* reusable instrument. **Course-only:** the specific exponents/predicted N at 10^23–10^24 FLOPs (Chinchilla numbers). |
| #4 budget-aware query planner | `QueryPlanner` over a `(N, C_infer)` grid with a cost budget | **KEEP the discipline** (predeclare grid + budget, never exceed). **Course-only:** the 12 B200-hr cap and the `hyperturing.stanford.edu` endpoint. |
| #6 training-API client | — (not in scope for `hack_rate_fit.py`) | **COURSE-ONLY / DROP.** The VERA analog queries our **own** rollout engine (`rollout/sglang_client.py`) for `hack_rate` at each `(N, C_infer)` cell, not a Stanford training API. |
| #2/#3 `C = 6ND` bridge | `compute_from_params_tokens(N, D)` helper | **KEEP** as a util; reused to convert inference compute (best-of-n × CoT length) into a comparable C_infer axis. |

**The reframe (the whole point of L3).** The fitter's y-axis is swapped from `loss` to a **non-loss safety/RL metric**, and the x-axes become **(model size N, inference compute C_infer)** instead of (training) C:

```
CS336:   loss        = f( N_opt , C_train )           # compute-optimal model
VERA L3: hack_rate   = f( N , C_infer )               # does the gap scale?
         true_quality_gap = reward − true_quality      (per HardeningLevel)
```

`hack_rate_fit.py` therefore exposes the **same** `isoflop_min` + `fit_powerlaw` + `predict` API, fed by the VERA grid (model family × N × C_infer × HardeningLevel) instead of `isoflops_curves.json`. The stub's docstring should name VERA **axis-3** (`CAPSTONE §2.4 H3`) as its §3 brief, the falsifiable prediction (H3 slope sign), and the kill criterion (no monotone trend ⇒ axis-3 is null, report it).

---

## 4. Map to core context docs

- **`UNIFIED_FRONTIER_PROJECT_SPEC.md §3 · L3·A3.** "Scaling: *measurement discipline as a weapon*." Core = IsoFLOP fits. **Add-on A3.1** = fit `hack_rate` (or `true_quality_gap`) vs (model size N, inference compute C_infer); *hypothesis (tests "LLMs Gaming Verifiers"):* under a weak verifier, `hack_rate` **grows** with inference compute (best-of-n / CoT length); *falsifiable prediction:* slope **positive at HardeningLevel 1–2, flattens/negates at level 5**; *artifact:* a scaling-law fitter applied to a verifier-exploitability metric — "a genuinely novel reframing and a clean plot for the paper." **Add-on A3.2** = a **data-constrained scaling mini-fit** (Muennighoff-style, `2305.16264`): repeated-token value decay, one small sweep, feeds the L4 argument that token *quality* changes the exponent. **Feeds:** VERA axis-3 (does the gap scale?). Interview leverage: *"how would you measure whether an eval is real at scale,"* IsoFLOP methodology.
- **`UNIFIED_FRONTIER_PROJECT_SPEC.md §5** (evidence ledger). The A3 artifact is item-3 material: a **"well-analyzed negative, stated as a win"** — e.g. *"hack_rate did not grow with inference compute at HardeningLevel 5; here's the fit and why."* Clean plot + honest extrapolation caveat = the engineering-quality signal labs screen for.
- **`CAPSTONE_AND_STUDY_PLAN.md §3** (row A3): *"Fitting laws from small sweeps → VERA axis 3: does `true_quality_gap` / `hack_rate` scale with model size & inference compute? Reuse IsoFLOP to fit the hack-rate-vs-compute curve."* Lecture backing **CS336 L9**.
- **`CAPSTONE §2.4** (VERA pre-registered axes). **Axis-3 = inference compute** (short vs long CoT / best-of-n). **H3 (tests `2604.15149`): under a weak verifier, `hack_rate` grows with inference compute.** This is the exact hypothesis `hack_rate_fit.py` is the instrument for. (Axis-1 = model family / H1; axis-2 = HardeningLevel / H2.)
- **`daily/STUDY_PLAN_2026.md §0.8`** — **A3 lands Day 12** ("A3 IsoFLOP (VERA scaling axis) → ship target = the smoke run / engine validation"). Companion row (§0.8 table, Day 12): Master = *A3 IsoFLOP → L3 · measurement #2*; requirement answered = *"measure the signal you're scaling" (IsoFLOP)*; target role = **Evals/Research RE**. Note A3 lands late (Day 12) and light — the spine is A1→A2→A5; A3 is an analytical axis, not the engine.
- **Repo `CLAUDE.md` (and README) L3 row:** *"L3 Scaling | A3 | IsoFLOP machinery repurposed to fit `hack_rate` vs compute | `scaling/hack_rate_fit.py`."* The stub must name its §3 brief, falsifiable prediction, and kill criterion in its docstring (clean-room rule).

---

## 5. The frontier 2026 lens

**Commoditized.** Re-deriving Chinchilla — fitting `L(N,D)`, recovering `a ≈ b ≈ 0.5`, predicting a compute-optimal model. Every pretraining candidate can do this; `scipy.optimize.curve_fit` on log-log points is a solved, teachable skill. The leaderboard-budget-optimization game (squeezing the most signal out of 12 B200-hr of queries) is *good craft* but it's the **course's** artificial constraint, not a frontier scarcity.

**Scarce (where the value is).**
1. **Fitting a *non-loss* curve.** The 2026 consensus (`UNIFIED §1`, skill #2) is that the bottleneck moved off "can you train a transformer" onto "**can you make your reward signal real and measure when it isn't.**" Almost nobody points scaling-law machinery at a *safety/RL* metric. A `hack_rate`-vs-(N, C_infer) IsoFLOP plot is a research instrument, not a homework.
2. **Data-constrained & quality-aware scaling.** "Data, not compute, is the bottleneck" (`UNIFIED §1` skill #8; `2305.16264`, DataComp-LM `2406.11794`). The A3.2 repeated-token mini-fit is the cheap demonstration that token *quality* changes the exponent — directly feeding the L4 reward-data-cleanliness argument.

**The senior move:** take the IsoFLOP machinery — a textbook pretraining tool — and turn it into a **safety/RL instrument.** Same `isoflop_min` + `fit_powerlaw` code; the y-axis is `hack_rate`/`true_quality_gap`, the x-axes are (N, C_infer), the result is a clean, falsifiable plot for the VERA paper (H3). That reframing — *"I measured whether verifier exploitability scales the way loss does"* — is the line that lands "measure the thing you're scaling" in an evals/research-RE interview. The deliverable is a *measured gap and its scaling exponent*, not a SOTA number — which is exactly why it's affordable for a solo budget.

---

## 6. Prioritization verdict — what matters / what to skip

**The ~20% load-bearing (build this, own it cold):**
1. **The reusable fitter** — `isoflop_min(points, metric_key)` + `fit_powerlaw(xs, ys)` + `PowerLaw.predict(x)`, log-log linear fit, `C=6ND` bridge, the `a+b≈1` sanity check. This is `scaling/hack_rate_fit.py`. It transfers verbatim from CS336 #1–#3.
2. **The hack-rate-vs-compute reframe** — swap y=`loss`→`hack_rate`/`true_quality_gap`, x=`C_train`→`(N, C_infer)`. The single insight that makes A3 a research instrument (A3.1).
3. **H3 + its falsifiable prediction** — slope **positive at HardeningLevel 1–2, flat/negative at 5**. Write the predicted sign *before* fitting (predict-before-you-run).
4. (Stretch) **A3.2 data-constrained mini-fit** — one small sweep; feeds the L4 quality-changes-the-exponent argument.

**Course-rote (do it for the grade / the skill, don't over-invest):** the training-API client + budget-query planner (#5, #6, #7), the leaderboard budget-optimization, the hyperparameter sweep to set non-N config, the `writeup.pdf`/`code.zip` Gradescope artifacts. Good craft, transferable *pattern* — but the specific Stanford API and the 12-B200-hr cap are course scaffolding.

**SKIP (for the reasoningLLM through-line / v0.1.0):**
- Chasing the exact 48-B200-hr leaderboard score / squeezing the last point out of the query budget — it's a graded game with zero carry into `hack_rate_fit.py`.
- Fitting the full 5-parameter `L(N,D)=E+A/N^α+B/D^β` — the PDF itself tells you to use the cheaper IsoFLOP min-picking instead.
- Reproducing Chinchilla's `a≈b≈0.5` numbers as a deliverable — know them as a sanity check (`a+b≈1`), don't enshrine them.
- Any of A3 *during* the frozen v0.1.0 sprint beyond the Day-12 IsoFLOP build — the scaling axis is **VERA (`v0.1.x`)**, not v0.1.0. Folding a full VERA scaling sweep into the 14 days is a scope/G6 violation; the v0.1.0 deliverable on Day 12 is the *smoke run*, with `hack_rate_fit.py` as a stub + its first IsoFLOP fit on synthetic data.

---

## 7. Build checklist (ordered, with discipline gates)

1. **PRE-READ** PDF §2.1 + the three equations (`L(N,D)`, `C=6ND`, `N_opt∝C^a`). Know them cold before code.
2. **PREDICT-BEFORE-YOU-RUN (gate).** Write the falsifiable numbers first: (i) from `isoflops_curves.json` you expect `a ≈ b ≈ 0.5` and `a+b≈1`; (ii) for VERA, **H3: hack-rate-vs-C_infer slope is positive at HardeningLevel 1–2 and flat/negative at 5.** Commit these to paper before fitting. This is the debugging anchor.
3. **Build #1** — `isoflop_min`: load synthetic JSON, group by `compute_budget`, take min-`final_loss` N per budget. Test: returns one `(C_i, N_opt)` per distinct budget.
4. **Build #2/#3** — `fit_powerlaw` on log-log `(C_i, N_opt)`; derive `D_opt = C/(6·N_opt)`; fit second law. **Gate: assert `a + b ≈ 1` (within ~0.05).** If not, the min-picking or `C=6ND` is wrong — fix before extrapolating.
5. **Extrapolate + plot** — predict N_opt, D_opt at 10^23, 10^24 FLOPs; plot data points + power-law line out to ≥10^24; **state the extrapolation factor** (how far past max data). Two one-sentence answers. (CS336 #2/#3 done.)
6. **(Course-rote) Build the API client + budget planner** — `POST /submit`, poll `GET /experiment/{id}`, refund-on-complete budget math; predeclare your query grid so you never blow the 12-B200-hr cap; small hyperparameter sweeps; submit `final_submission`; write `writeup.pdf`. Do this for the grade, time-boxed.
7. **REFRAME → `hack_rate_fit.py`** — generalize `isoflop_min(metric_key)` and `fit_powerlaw` so y = `hack_rate`/`true_quality_gap`, x = `(N, C_infer)`. Wire the VERA grid (model family × N × C_infer × HardeningLevel). Docstring names §3 brief A3.1, prediction H3, kill criterion.
8. **VERA axis-3 fit (post-sprint, `v0.1.x`)** — fit `hack_rate ∝ C_infer^γ` at each HardeningLevel; **check predicted sign vs H3.** Report the slope per level — *including a null result honestly* if H3 fails.
9. **(Stretch) A3.2 data-constrained mini-fit** — one repeated-token sweep; show the exponent shifts as tokens are re-used; one plot feeding the L4 quality argument.
10. **FEYNMAN** the IsoFLOP primitive + the reframe on paper; **CONNECT** sentence: *"A3's `fit_powerlaw` is now `hack_rate_fit.py`; it tests VERA H3."*; green tests → atomic commit referencing the F-ID.

---

## 8. Open questions / ADR triggers

- **What `(N, C_infer)` grid is affordable for the VERA hack-rate fit?** Needs ≥3–4 N points × ≥3–4 C_infer points × HardeningLevels to fit a power law per level with the `a+b≈1`-style sanity check — but VERA's whole budget is ~$260–390 (`CAPSTONE §2.6`). Likely: hold model family fixed (Qwen-1.5B), use a coarse N grid (or a single N with a C_infer sweep first), reuse rollouts across HardeningLevels. **ADR trigger** when the grid is fixed (`ADR: VERA axis-3 (N, C_infer) grid`).
- **Is `C_infer` best parameterized as best-of-n, CoT length, or total inference FLOPs?** They aren't interchangeable (best-of-n changes selection pressure; CoT length changes per-sample compute). H3's "inference compute" needs one concrete operationalization before fitting — pick the one the verifier is most gameable under. **ADR trigger.**
- **How few IsoFLOP points can support an honest extrapolation?** The fit is only as good as regime stability; with a tiny N grid the extrapolation factor to a target budget may be too large to defend. Decide the max defensible extrapolation factor up front (and state it in the plot caption).
- **Does the `a+b=1` constraint hold for the *hack-rate* fit?** It's a loss/`C=6ND` identity — there's no a-priori reason a safety metric obeys it. Decide whether to *impose* it (fewer free params, more bias) or *fit freely* (and use the deviation as a diagnostic). **ADR/design-note trigger.**
- **A3.2 scope:** is the data-constrained mini-fit in v0.1.x or deferred to v0.2.0? It feeds L4, not the core VERA finding — likely a stretch, gate behind its own note so it isn't smuggled into the critical path.
