# A3 — Scaling · BUILD GUIDE (CS336 mastery → production → interview)

> **📖 Read first (slides → this build):** Lectures **9 → 11** — scaling-law basics (pdf) · scaling
> case study & `C=6ND` details (pdf); then **12** (evaluation, py) for the loss surface. Full map +
> read-order: [`../LECTURE_MAP.md`](../LECTURE_MAP.md).

> **One-liner.** CS336 A3 teaches you to *fit a power law and extrapolate it under a fixed compute budget* — the core skill of compute-optimal pretraining. Two problems: a 5-point synthetic **IsoFLOP** fit (`chinchilla_isoflops`) and a **50-point** budget-constrained **training-API leaderboard** (`scaling_laws`) — the latter *is* the assignment. **The transferable instrument is a clean log-log power-law fitter**: per-budget min-picking → `N_opt ∝ C^a`, `D_opt` via `C = 6ND`, the `a+b≈1` sanity check, honest extrapolation. You build it once in `src/scratch_llm/scaling/` and reuse it for both problems. The leaderboard adds a *query planner* that spends the budget to fit a loss surface, then predicts and submits the compute-optimal config.

---

## 1. What CS336 actually requires (every deliverable)

A3 (v26.0.0) has **two graded problems** of very unequal weight: a 5-point synthetic IsoFLOP fit, and a **50-point** real-training-API leaderboard. Everything in §3 of the PDF (the training API, budget accounting, endpoints) is scaffolding for that 50-pointer. Tests run with `uv run pytest`; the leaderboard is graded on Gradescope + live API performance, so it has no local adapter.

| # | Deliverable (exact PDF problem name) | PDF § | Adapter / test fn | Est. effort | Priority |
|---|---|---|---|---|---|
| 1 | **`chinchilla_isoflops` — IsoFLOPs scaling laws** (5 pts): load `data/isoflops_curves.json`; for each compute budget `C_i` take the run with **lowest `final_loss`** as `N_opt(C_i)` (PDF explicitly says: skip Hoffmann's per-profile quadratic fit, just take the min). | §2.1 | fit `N_opt ∝ C^a` on the `(C_i, N_opt)` pairs | 1.5–2 h | **LOAD-BEARING** (this *is* the fitter) |
| 2 | **(a) Extrapolated N_opt** + plot of `(C_i, N_opt(C_i))` with the power-law line, extrapolated to ≥10²⁴ FLOPs; predict N_opt at **10²³ and 10²⁴** FLOPs; one-sentence answer. | §2.1(a) | power-law fit + extrapolation + plot | 0.5 h | **LOAD-BEARING** |
| 3 | **(b) Extrapolated D_opt** + plot of `(C_i, D_opt(C_i))`; predict D_opt at 10²³ and 10²⁴; one-sentence answer. D_opt derived via **C ≈ 6ND ⇒ D_opt = C / (6·N_opt)**. | §2.1(b) | second power-law `D_opt ∝ C^b` | 0.25 h | **LOAD-BEARING** |
| 4 | **`scaling_laws` — leaderboard** (50 pts): query the training API under a **12 B200-hour** fitting budget (= 25% of the 48 B200-hr "big run" budget); fit a scaling law over the queried `(config → val_loss)` points; **choose the model size + hyperparameters** predicted to minimize val loss at 48 B200-hr; predict that loss. | §3, §3.3 | budget-aware query planner + multi-point loss-surface fitter | 4–6 h | **LOAD-BEARING** (the heart of the assignment) |
| 5 | **`POST /final_submission`** — submit predicted optimal `training_config` + `predicted_final_loss` to the API (graded partly on real model performance; resubmittable, latest replaces previous). | §3.2, §3.3 | API client call | 0.25 h | **COURSE-ROTE** (leaderboard plumbing) |
| 6 | **Training-API client** — `POST /submit`, `GET /budget`, `GET /experiments`, `GET /experiment/{id}`; poll until `status_type=="completed"`; read final `val_losses[-1]`; handle 409 (dup config) / 400 (over-budget); respect that **queued/running reserve full `max_runtime_seconds`** and only refund actual runtime on completion. | §3.1–3.2 | thin `requests` wrapper + poller | 1.5 h | **COURSE-ROTE** (Stanford-network-only API; reusable *pattern*, not the code) |
| 7 | **Hyperparameter-effect analysis** — small-scale sweeps to set the *non-N* hyperparameters (depth/width ratio, LR, batch, etc.) for the predicted optimal config; PDF asks you to comment on each factor. Param count for a config est. via **non-embedding params ≈ 12·n_layer·d_model²**. | §3.3 | sweep harness (uses #6) | 1.5–2 h | **COURSE-ROTE** (good practice; not the reusable instrument) |
| 8 | **`writeup.pdf`** — methodology: which runs you queried & why, fit method, fit quality, predicted N/loss for 48 B200-hr, hyperparameter reasoning. **`code.zip`.** | §1 "How to submit", §3.3 | — | 1–2 h | **COURSE-ROTE** (Gradescope artifact) |

**Accurate-to-PDF notes that catch people out:**
- `isoflops_curves.json` is a flat array of `{parameters, compute_budget, final_loss}` objects — **72 runs across 9 distinct compute budgets** (`6e18 … 3e21`). For each budget, the IsoFLOP profile is the subset of runs at that `compute_budget`; `N_opt` is the `parameters` of the lowest-`final_loss` run in that subset.
- The IsoFLOP intuition the PDF gives: at fixed `C_i`, `final_loss` is **quadratic-ish in N** — too-small N can't absorb the compute (loss high), too-large N can't take enough gradient steps within `C_i` (loss high), minimum in between. You want the N at that minimum per budget.
- `C = 6ND` and `D = C/(6N)` are the only bridge between the two power laws; `D_opt` is *derived*, not separately fit from scratch.
- The API **fixes** `seq_len=512`, `n_validation_tokens = 2^18 ≈ 262k`, `vocab=32k`; enforces `hidden_size == num_attention_heads * head_dim`, `num_attention_heads % num_key_value_heads == 0`, `total_train_tokens % (512 * train_batch_size) == 0`. Config-consistency validation is part of "spend the budget without wasting it."
- **Budget is wall-clock, not FLOPs, and it's adversarial:** a timed-out run is charged the *full* `max_runtime_seconds`; a completed run is refunded down to its actual runtime (≥1 s). The skill is choosing `max_runtime_seconds` tight enough to not over-reserve but loose enough to not time out. **12 B200-hr = 43 200 s** is a hard API-enforced cap (`max_runtime_seconds` > `remaining_seconds` → 400).
- The data order is fixed (no epoching); `model_seed` controls init only, not data order. Architecture ≈ A1 (RMSNorm/RoPE/SwiGLU), trained on tokenized DCLM with a 32K vocab.

---

## 2. The equations / algorithms that matter (senior extraction)

1. **Parametric loss law (Chinchilla / Hoffmann 2022)** — `L(N,D) = E + A/N^α + B/D^β`. The irreducible-loss term `E` plus two power-law terms in params and data. *Why it matters:* this is the object whose minimum (subject to `C=6ND`) gives compute-optimal allocation. CS336 A3 does **not** make you fit all five parameters (E,A,B,α,β) — it uses the cheaper IsoFLOP shortcut — but you must know this is the underlying law to defend the method in a writeup or an interview.
2. **Compute identity** — `C ≈ 6ND` (6 FLOPs/param/token: ~2 fwd + 4 bwd). *Why it matters:* the single equation that turns one power law (`N_opt(C)`) into the other (`D_opt(C)`), and converts a wall-clock/FLOP budget into a feasible `(N, D)` grid. Get the constant wrong and every extrapolation is off by a multiplicative factor.
3. **IsoFLOP profile construction** — for each fixed budget `C_i`, sweep N, record `final_loss`; `N_opt(C_i) = argmin_N L`. *Why it matters:* it's the *measurement primitive* of the whole method. The PDF explicitly tells you to take the per-budget argmin rather than fit Hoffmann's quadratic to each profile — cheaper, and good enough for these data. The same primitive generalizes the moment you can compute a per-budget minimum of any metric you measure.
4. **Optimal-allocation power laws** — `N_opt ∝ C^a`, `D_opt ∝ C^b`, with `a + b = 1` (since `C = 6ND` ⇒ the exponents must sum to one). *Why it matters:* Chinchilla found `a ≈ b ≈ 0.5` (scale params and data ~equally). The `a+b=1` constraint is a built-in **sanity check** on your fit — if your two independently-fit exponents don't sum near 1, your min-picking or your `C=6ND` is wrong. Fail this check loudly *before* extrapolating.
5. **Power-law fitting in log space** — `log N_opt = a·log C + const`; fit by **linear regression on log-log points** (or `scipy.optimize.curve_fit` on the power form — the PDF suggests `scipy`). *Why it matters:* this is the entire mechanical core of the fitter you build in `src/scratch_llm/scaling/` (`isoflop_min`, `fit_powerlaw`, `PowerLaw.predict`). Cheap, robust, two parameters; gives you slope `a` (the headline) + intercept. Fit on log-log, *predict by exponentiating*, never fit raw values (heteroscedastic — the largest losses would dominate the residual).
6. **Extrapolation + non-embedding param estimate** — evaluate the fitted law at a *target budget larger than any queried point* (10²³, 10²⁴, or the 48 B200-hr leaderboard budget); map predicted N back to an architecture via **non-embedding params ≈ 12·n_layer·d_model²**. *Why it matters:* extrapolation beyond the data is the *entire point* of a scaling law and its biggest risk — the fit is only as good as the regime's stability. State the extrapolation factor (how far past your max data point you are reaching) explicitly; that honesty is the senior signal, and stating *when the law would break* is the interview answer.

---

## 3. Map to `src/scratch_llm/scaling/`

The repo's scaling module is **`src/scratch_llm/scaling/`** — a clean, CPU/numpy IsoFLOP/Chinchilla fitter (today a stub: `__init__.py` only). Build the **reusable log-log fitter once** and call it from both problems. The map below is the build target.

| CS336 deliverable | → `scaling/` component | Notes |
|---|---|---|
| #1 IsoFLOP min-picking (`argmin_N L` per `C_i`) | `isoflop_min(runs) -> list[(C_i, N_opt)]` — group by `compute_budget`, take min-`final_loss` N per budget | Pure function over the parsed `isoflops_curves.json` rows. One `(C_i, N_opt)` pair per distinct budget. |
| #2/#3 power-law fit + extrapolate | `fit_powerlaw(xs, ys) -> PowerLaw(a, const)` + `PowerLaw.predict(x)` (log-log linfit; exponentiate to predict) | **The reusable instrument.** Used for `N_opt ∝ C^a` and again for `D_opt ∝ C^b`. |
| #2/#3 `C = 6ND` bridge | `compute_from_params_tokens(N, D)` and `tokens_from_compute_params(C, N)` helpers | `D_opt = C / (6·N_opt)`; derive the second law's points from the first plus the compute identity. |
| #1–#3 sanity check | assert `a + b ≈ 1` (within ~0.05) in the fit path | Fail loudly before extrapolating — it catches min-picking and `6ND` bugs. |
| #6 param → architecture | `nonembed_params(n_layer, d_model) ≈ 12·n_layer·d_model²` | Inverts a predicted N back to a concrete `(n_layer, d_model)` shape for the writeup / config. |
| #4 budget-aware query planner | `QueryPlanner` — predeclare a `(config)` grid + a `max_runtime_seconds` per run; track spent vs the 43 200 s cap; never exceed it | Spends the 12 B200-hr fitting budget to collect `(config → val_loss)` points, then feeds `fit_powerlaw`. |
| #4 loss-surface fit | reuse `fit_powerlaw` over `(C, val_loss)` (and the IsoFLOP min-pick over varied-N runs at matched compute) | Same instrument as #1–#3; the leaderboard just feeds it *real* API points instead of synthetic JSON. |

**The leaderboard client is network-locked, not in-repo.** The Stanford training API (`http://hyperturing.stanford.edu:8000`) is reachable only on the Stanford network (VPN) and is graded live, so the **HTTP client + poller live in the official scaffold** (`../../../lectures/assignment3-scaling/cs336_scaling/client.py` + `examples/client_example.ipynb`), not in `src/scratch_llm/scaling/`. What you own in this repo is the **method**: the query planner's budget discipline and the loss-surface fitter. The client is a reusable *pattern* (submit → poll → read `val_losses[-1]` → refund-on-complete budget math), not code to reproduce here.

**Pointers:**
- Build plan: [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) §A3 — A3 sits after A2's distributed pieces in the linear build order; `scaling/` is a CPU/numpy module buildable now.
- Official scaffold (spec + the live API client): `../../../lectures/assignment3-scaling/` — the PDF (`cs336_assignment3_scaling.pdf`), `cs336_scaling/` (client, schemas, training reference), `data/isoflops_curves.json`, and `examples/client_example.ipynb`.
- The module's docstring should state its intent (compute-optimal `N, D` from small sweeps via IsoFLOP), the invariant it must satisfy (`a + b ≈ 1`), and the interview question it answers ("derive and defend a scaling law").

---

## 4. The frontier 2026 lens

**Commoditized.** The *mechanical* scaling-law fit — `scipy.optimize.curve_fit` (or a log-log `polyfit`) on a handful of `(C, N_opt)` points, recovering `a ≈ b ≈ 0.5`, predicting a compute-optimal model — is a solved, teachable skill. Every pretraining candidate can do it. The leaderboard's budget-optimization game (squeezing the most signal out of 12 B200-hr of queries) is *good craft*, but it is the **course's** artificial constraint, not a frontier scarcity.

**Scarce (where the value is).**
1. **Compute-optimal vs. *inference-optimal*.** Chinchilla minimizes *training* loss at a fixed *training* budget. But if you will *serve* a model to billions of tokens, the Llama-style move is to **over-train a smaller model far past its Chinchilla-optimal token count** — you pay more at train time to pay less, forever, at inference. Knowing *which budget you are optimizing* (one-time train vs. amortized serving) is the senior distinction the bare fit hides.
2. **Data-constrained scaling.** "Data, not compute, is the bottleneck" past a point: repeated-token returns decay (Muennighoff et al. 2023) — re-using tokens beyond a few epochs buys far less than fresh tokens, and eventually nothing. A Chinchilla law fit on unique-token assumptions over-promises once you hit the data wall. This is *why* A4 (data curation) matters and where the exponent stops being constant.
3. **Upstream loss vs. downstream capability.** A clean `L(N,D)` fit predicts validation *loss*, but the thing you ship is benchmark/task performance — and the loss→capability map is non-linear and can break (emergence, saturation). Honest extrapolation means stating that your law predicts *loss*, and that loss is a proxy.

**The senior move:** treat the power-law fitter as a textbook instrument, then be the person who knows **when the law breaks** — regime shifts, the data wall, train-vs-inference budget, loss-vs-capability. State the extrapolation factor (how far past your data you are reaching) in every plot caption, and name the regime where the fit would stop holding. That honesty — *"here is the fit, here is how far I trust it, and here is what would invalidate it"* — is the line that lands "derive and defend a scaling law" and "compute-optimal allocation" in an evals/pretraining-research interview. The deliverable is a *defensible exponent with stated limits*, not a SOTA number.

---

## 5. Prioritization verdict — what matters / what to skip

**The ~20% load-bearing (build this, own it cold):**
1. **The reusable log-log power-law fitter** — `isoflop_min(runs)` + `fit_powerlaw(xs, ys)` + `PowerLaw.predict(x)`, log-log linear fit, the `C=6ND` bridge, the `a+b≈1` sanity check. This is the core of `src/scratch_llm/scaling/`. It serves `chinchilla_isoflops` (#1–#3) and is the same instrument the leaderboard fit reuses.
2. **`chinchilla_isoflops` end to end** — per-budget min-picking → fit `N_opt ∝ C^a` and `D_opt ∝ C^b` → check `a+b≈1` → extrapolate + plot to ≥10²⁴ FLOPs → predict N/D at 10²³ and 10²⁴. The full IsoFLOP methodology on synthetic data.
3. **`scaling_laws` — the 50-point leaderboard method** (THIS IS THE ASSIGNMENT): a **query planner** that spends the fixed 12 B200-hr budget to collect `(config → val_loss)` points, **fits the loss surface** with the same log-log fitter, then **predicts + submits** the compute-optimal config for the 48 B200-hr run and its expected loss. The budget-spending discipline (predeclare grid, never exceed the cap) and the loss-surface fit are the transferable instrument.

**Course-rote (do it for the grade / the skill, don't over-invest):** the training-API HTTP client + poller (#6), `POST /final_submission` (#5), the hyperparameter-effect sweep to set non-N config (#7), the `writeup.pdf` / `code.zip` Gradescope artifacts (#8). Good craft, transferable *pattern* — but the specific Stanford API and the 12-B200-hr cap are course scaffolding (network-locked; reproduce the pattern, not the code).

**SKIP (no mastery carry):**
- Chasing the exact 48-B200-hr leaderboard *score* / grinding the last point out of the query budget — build the method correctly; don't grind the number. It's a graded game with zero transfer.
- Fitting the full 5-parameter `L(N,D)=E+A/N^α+B/D^β` — the PDF itself tells you to use the cheaper IsoFLOP min-picking instead.
- Enshrining Chinchilla's `a≈b≈0.5` numbers as a deliverable — know them as a *sanity check* (`a+b≈1`), don't reproduce them as the goal.

**Interview leverage:** "derive and explain scaling laws," IsoFLOP methodology, compute-optimal vs. inference-optimal allocation, honest extrapolation (state the factor + when the law breaks).

---

## 6. Build checklist (ordered, with discipline gates)

1. **PRE-READ** PDF §2.1 + the three equations (`L(N,D)`, `C=6ND`, `N_opt∝C^a`). Know them cold before code.
2. **PREDICT-BEFORE-YOU-RUN (gate).** Write the falsifiable numbers first: from `isoflops_curves.json` you expect `a ≈ b ≈ 0.5` and `a+b≈1`. Commit them to paper before fitting — this is the debugging anchor when the fit comes back wrong.
3. **Build #1** — `isoflop_min`: parse the JSON, group by `compute_budget`, take the min-`final_loss` N per budget. Test: returns exactly one `(C_i, N_opt)` per distinct budget (9 pairs for this data).
4. **Build #2/#3** — `fit_powerlaw` on log-log `(C_i, N_opt)`; derive `D_opt = C/(6·N_opt)`; fit the second law. **Gate: assert `a + b ≈ 1` (within ~0.05).** If it fails, the min-picking or `C=6ND` is wrong — fix before extrapolating.
5. **Extrapolate + plot** — predict `N_opt`, `D_opt` at 10²³ and 10²⁴ FLOPs; plot data points + power-law line out to ≥10²⁴; **state the extrapolation factor** (how far past max data) in the caption. Two one-sentence answers. (`chinchilla_isoflops` #1–#3 done; green tests → atomic commit.)
6. **(Course-rote) Build the API client + budget planner** — `POST /submit`, poll `GET /experiment/{id}` until `completed`, refund-on-complete budget math; predeclare your query grid + `max_runtime_seconds` so you never blow the 43 200 s cap; handle 409/400. Run on the Stanford network (VPN). Time-boxed.
7. **Leaderboard fit + predict-before-submit (gate)** — feed the queried `(config → val_loss)` points to `fit_powerlaw`; **predict the 48-B200-hr optimal N + val loss from the fitted law before you `POST /final_submission`** (an out-of-sample check spent within the 12-hr budget). Map predicted N → `(n_layer, d_model)` via `12·n_layer·d_model²`; run the small hyperparameter sweeps to set non-N config; submit.
8. **Writeup** — which runs you queried & why, the fit method + fit quality, predicted N/loss for 48 B200-hr, the hyperparameter reasoning, and the extrapolation caveat. `code.zip`. (`scaling_laws` + `writeup.pdf` done.)

---

## 7. Open questions / ADR triggers

- **How to spend the 12 B200-hr fitting budget: a few large runs or many small ones?** Many small runs give more `(C, loss)` points to fit but each is noisier and farther (in extrapolation factor) from the 48-hr target; a few larger runs sit closer to the target but give a sparser fit. A fixed predeclared grid is safest against the adversarial timeout-charges-full math; an adaptive plan (refine where the surface is steep) extracts more signal but risks over-reserving. **ADR trigger** when the query plan is fixed (`ADR: A3 scaling-law query budget allocation`).
- **Impose `a+b=1` or fit the two exponents freely?** The constraint is a `C=6ND` identity, so imposing it (fit `a`, set `b=1−a`) cuts a free parameter and reduces variance; fitting freely lets the *deviation* from 1 serve as a diagnostic of fit/data quality. **Design-note / ADR trigger** — decide and state which, and report the free-fit `a+b` either way.
- **How few IsoFLOP points support an honest extrapolation?** The fit is only as good as regime stability; with a sparse N grid the extrapolation factor to a target budget may be too large to defend. Decide the **max defensible extrapolation factor** up front and put it in the plot caption.
- **How do you fit the loss surface when configs vary beyond N?** The leaderboard lets you vary depth/width ratio, LR, batch, etc. — not just N. Decide whether to (a) hold non-N hyperparameters at sensible defaults and fit a 1-D `N_opt(C)` law (clean, the IsoFLOP path), or (b) fit a higher-dimensional surface (more general, far hungrier on the 12-hr budget). **ADR trigger.**
- **`max_runtime_seconds` per run — how tight?** Too loose over-reserves the cap (queued/running hold the full reservation); too tight risks a timeout charged at full cost with no usable loss. Calibrate from a couple of cheap completed runs before committing the grid.
