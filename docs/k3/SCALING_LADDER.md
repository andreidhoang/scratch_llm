# THE SCALING LADDER — how frontier labs sequence ablations vs scaling laws, and our order

> ⚠️ **ERRATA-G binding (2026-08-26).** Pre-audit planning generation — NOT current law. The sealed
> measure-first corpus governs (Desktop `plan/` + memory tracker; precedence G > F > E > D > C > B >
> body). K3-from-scratch = post-gate season W9–W16. Full binding: ERRATA-G blocks in
> `docs/k3/ROADMAP.md` and `docs/k3/MERGED_KERNELS_K3_ROADMAP.md`.

> Compiled 2026-07-31 from a 3-angle primary-source pass (DeepSeek + NVIDIA · Moonshot ·
> Meta/Qwen/MiniMax/open-labs). Question answered: **architecture ablations first, or scaling
> laws first?** Answer the evidence gives, with near-zero dissent: **calibration fit on a proven
> baseline → architecture ablations → family scaling-law fit (the decision) → freeze →
> hyperparameter re-fit for the frozen family → flagship.** Scaling laws appear TWICE —
> bracketing the architecture work — never once, and never on an unfrozen architecture.

## 1. What the labs actually do (primary sources, exact scales)

| Lab | Architecture ablated at | Family fit | Hyperparameter fit | Order / notes |
|---|---|---|---|---|
| **DeepSeek** (LLM'24→V4) | two-scale fixed-budget: ~16B/1.33T ablate → confirm at ~250B/420B tok (MLA, V2 App.D.2) · 228.7B/540B (MTP) · 228.7B/578B (aux-loss-free) · ~230B/~0.9T (FP8); HPs inherited V2-Lite→V2 | **none published after 2024**; V3.2 = full-scale continued-training parity; V4 (Jun 2026) publishes **zero** ablations | once, dense era: η=0.3118C^−0.125, B=0.2920C^0.3271; IsoFLOP N/D 8 budgets × ~10 alloc, 1000× extrapolation | systems target → ablate → mid-scale confirm → flagship; V3's $5.576M *excludes* ablation costs (verbatim) |
| **Moonshot** (Kimi Linear→K3) | smallest fit scale: **653M-active/38.8B tokens** (ratio/gate/conv arms vs MLA + GDN-H comparators — no GQA arm; 48B-A3B/1.4T was *validation*, not ablation) | **5 points** (653M→1.7B active), per-family L=A·C^−α (MLA −0.0536 vs KDA −0.0527 — near-equal but fit independently). KDA multiplier 1.16× read mid-curve (anchor unstated, loss ≈2.1–2.15); the **1.25× @5.6 PFLOP/s-days anchored at the largest point is from the separate AttnRes paper** (arXiv:2603.15031, own 5-pt sweep at 194M–528M active) — do not attribute it to Kimi Linear. **Challenger inherits incumbent HPs verbatim** ("we adhered strictly to the MLA training configuration without any modifications"; AttnRes: "this setup intentionally favors the baseline"); **no µP anywhere in Moonshot papers** — per-size LR grid search | **K3 §3.2, verbatim:** "Since these changes also alter the optimal training regime, we conduct dedicated scaling-law studies to retune key hyperparameters, including the batch size, learning rate, tokens-per-parameter ratio (TPP) and the model shape" — held-out OOD val, **independent search per LR schedule** (cosine > WSD only when each tuned separately) | ablate → 5-pt fit → 48B/1.4T validate → assemble → re-fit → 2.8T. AMA: "if the model shows any instability, scaling stops immediately" |
| **NVIDIA** (LatentMoE/Nemotron-3) | 16B-total/2B-active (**on DeepSeek-V2-Lite's recipe** — borrowed proven baseline), iso-FLOP+iso-param (BOTH total and active matched), val loss | — | — | design principles from roofline first → ablate → 95B/8A scaling test (300B tok transformer; 1T hybrid) → flagship adoption (25T tokens) |
| **Meta** (Llama 3) | architecture by **fiat** (dense, stability); GQA by inference constraint | IsoFLOP 6e18–1e22 FLOPs, 40M–16B models, N\*=0.29·C^0.53 (**token** law) → predicted 402B/16.55T; actual 405B/15.6T (flat-minimum robustness); fit range ends ~4 orders below flagship | — | fit only for sizing; data-mix via small-model scaling + cheap anneals (8B model / 40B tokens @30% new data) |
| **MiniMax** (-01) | per-attention-type Chinchilla fits, 6 sizes 70M–7B/≤300B: softmax L=3.7087C^−0.0798 vs lightning 3.5391C^−0.0768 vs hybrid 3.4797C^−0.0763 — **different exponents per family**; module ablations at 28B-total/5B-active/1T | per-family (required!) | custom MoE law after Clark/Hoffmann fits broke at 9.3B-active | strongest direct evidence: the fit comes AFTER the architecture is chosen, per family |
| **MiniMax M2** (caution) | hybrid attention showed small-scale **parity** — "this superficial parity concealed deep capability deficits"; at scale, "clear shortcomings in complex, multi-hop reasoning" → **reverted to full attention** | — | — | small-scale parity can hide capability deficits; hard probes required, loss alone insufficient; their production-scale hybrid ablation degraded retrieval/multi-hop/ICL while ≤32K benchmarks stayed mixed |
| **OLMo 2 / SmolLM3 / OLMo 3 / DCLM** (open) | OLMo2: stability ablations; LR crossover "well past 200B tokens" — short experiments mislead. SmolLM3 (**Smol Training Playbook**, not a paper): two tiers — full 3B×100B arms + 1B-proxy×45B, **HPs fixed across arms**, **>100 ablations ≈ 42% of flagship GPU-hours** (+debug ≈ 58%). OLMo 3: recipe dev ≤7B → 32B; development = **82% of total compute** (6.85M H100-h, arXiv:2605.01158) | DCLM ladder 412M→7B (5 rungs, rank r = 0.885/0.919 vs 7B) | per-stage | transfer rule (HF playbook, verbatim): "if something hurts performance at small scale, you can confidently rule it out for large scale. But if something works at small scale, you should still make sure you've trained on a reasonable number of tokens" |

## 2. The ladder (what a frontier research-engineering team runs)

```
Rung 0  COMPASS + CALIBRATION FIT (on the proven baseline)
        Pick a proven family close to target; fit hyperparameter + N/D laws on it.
        Job: make every later measurement trustworthy (budgets, LR, val protocol).
        NOT an architecture decision. (DeepSeek-2024 on LLaMA-like; our S3 on d-series.)
Rung 1  COMPONENT ABLATIONS at the smallest trustworthy scale
        One variable per arm · iso-FLOP + iso-param · baseline-favored HPs — challenger
        inherits the incumbent's config VERBATIM (Moonshot: "we adhered strictly to the
        MLA training configuration without any modifications"; AttnRes: "this setup
        intentionally favors the baseline"; SmolLM3: HPs fixed across arms)
        · train loss + held-out OOD val + EARLY-SIGNAL + HARD probes (retrieval/reasoning
        — M2 + OlmPool lesson: ≤32K standard benchmarks stay silent on retrieval/multi-hop
        deficits) · 2 seeds on corner arms (kills single-seed, adoptions never).
        Gate: negatives are trusted (kill cheap); positives are PROVISIONAL.
Rung 2  FAMILY SCALING-LAW FIT (the architecture decision)
        5 scale points per candidate family; L=A·C^−α; compute multiplier at equal loss,
        anchored at the largest point. Advantage must HOLD OR GROW across the range —
        a single-scale win is not adoptable. (MiniMax: exponents differ per family.)
Rung 3  MID-SCALE VALIDATION with the production recipe (~1–2 orders up)
        Confirms ranking + stability + downstream/hard-task behavior. (Kimi 48B/1.4T;
        DeepSeek 230B/540B; NVIDIA 95B/1T.) Moonshot rule: instability ⇒ stop scaling.
Rung 4  FREEZE → RE-FIT for the frozen family
        N/D (IsoFLOP) + LR/batch/TPP/shape, per-schedule independent searches
        (K3 §3.2 rule: architecture changes alter the optimal regime).
Rung 5  FLAGSHIP + cheap endgame (anneal/mid-training decisions from intermediate
        checkpoints: Llama3 8B/40B; OLMo2 19 microanneals/130B).
Every rung: systems constraints can veto (heads, sparsity, g_min=−5, NoPE-for-ops).
```

**The three decision rules (encode everywhere):**
1. **Kill vs adopt asymmetry** — a single-scale ablation can kill a mechanism but cannot adopt
   one. Adoption requires the family curve (Rung 2) + hard probes (M2 lesson).
2. **Never fit hyperparameter laws on an unfrozen architecture** — exponents are per-family
   (MiniMax-01); the regime changes with the architecture (K3 §3.2).
3. **Stability gate between rungs** — gradient/logit monitors; any instability stops the ladder
   (Moonshot AMA). Our `utils/` monitors + NaN guard already implement this.

## 3. Our order of operations (mapped, with status)

```
S3 calibration (d-series control)          RUNNING  = Rung 0. Sets budgets/LR/D:N for ALL
                                                       arms; it is the measuring stick, NOT
                                                       the K3-family fit. (DeepSeek-2024 analog)
R1/R2 component ablations @ d12            NEXT     = Rung 1. After core/kda.py PROVEN.
                                                       Pre-registered kill lines; retrieval
                                                       probes mandatory (needle/induction
                                                       @8K–32K) — M2 anti-parity guard.
FAMILY FIT: GQA vs KDA-hybrid, 5 points    Rung 2   = THE ATTENTION DECISION.
   (d8…d20 span), L=A·C^−α, multiplier                     Compute multiplier at equal loss,
   anchored at largest point                               AttnRes-1.25× protocol exactly.
   ➜ if KDA multiplier ≥ 1 and holds across range: adopt; else: kill, report, d20 stays GQA.
mini-K3 @ d20 scale, production recipe     Rung 3   = validation + absorbs the d20 slot.
                                                       (If KDA family wins, THIS run is the
                                                       portfolio flagship; if not, d20 runs
                                                       GQA-family and K3 arms are reported
                                                       as measured negatives.)
FREEZE mini-K3 config                      Rung 4a  = which mechanisms survive (R1 verdicts).
Frozen-family re-fit (LR/batch/TPP/shape)  Rung 4b  = K3 §3.2 rule; per-schedule independent
                                                       search if we compare schedules.
Flagship + annealed endgame (8K→64K)       Rung 5   = anneal-from-checkpoint decisions,
                                                       Llama3/OLMo pattern.
```

Budget logic (evidence-anchored): SmolLM3's ablations ≈ 42–58% of flagship spend and OLMo 3's
development = 82% of total compute (arXiv:2605.01158) bracket the honest ratio; our ~$150–400
ablation budget vs ~$100 flagship is consistent with it.
Scale honesty: our family-fit span d8→d20 is ~8× in params — *as a ratio* comparable to Kimi's
653M→1.7B (2.6×) and MiniMax's 70M–7B (100×) — but our largest point is ~100× smaller in
absolute compute, and our fit must never be extrapolated past its span. Conclusions are
registered as *directional at our scale*, never as 2.8T-transfer claims (that transfer is what
K9/their papers establish).

**Immediate consequence (no schedule change, one addition):** the R1 factorial already IS
Rung 1. The addition this methodology mandates: **Rung 2's 5-point family fit becomes an
explicit deliverable between R1 and the d20 decision** — the attention architecture is chosen
by the curve, not by the single d12 arm. Everything else stays as chartered.

## 4. Reconciliation with `FRONTIER_2026_ARCH_SCALING.md` (same repo, F12/S3 front)

The sister doc (landed 2026-07-31, commit 7803bbc) answers "must S3 be re-run per
architecture?" with: **one law per recipe backbone; mild architecture variants move the law's
OFFSET, not its exponents; attention variants are compared by iso-FLOP anchor pairs against
the S3 reference points; MoE is the exception (compute model changes → small joint grid);
every small-scale win is trigger-gated at d14 (E2E §S3(g) T3).** This ladder prescribes
per-family fits — apparent tension, resolved as follows (the operating rule):

1. **Default = shared-slope prior.** The dense-transformer literature and Kimi Linear's own
   fits (MLA −0.0536 vs KDA −0.0527 — offset-dominant) support estimating each family's
   OFFSET from S3's shared reference law + 2 anchor points (d12, d20), with slope-equality
   explicitly tested. This is the cheap path and is statistically better at our narrow span.
2. **Escalation = full per-family fit (Rung 2), in two cases only:** (a) the shared-slope
   test is rejected — MiniMax-01's softmax-vs-lightning exponents (−0.0798 vs −0.0763) prove
   radical mixers CAN move slopes; (b) the multiplier is decision-critical — and the GQA-vs-KDA
   attention decision IS decision-critical, so it keeps the 4-point per-family fit regardless.
3. **LatentMoE/MoE arms follow ARCH_SCALING's MoE exception:** C = 6·N_active·D while
   capacity ∝ N_total — the dense law does not apply; axis-C conclusions use the small joint
   grid, not the dense reference curve.
4. **The d14 trigger gate binds here too:** no small-scale K3-arm win touches the frozen d20
   recipe without the d14/d20 confirmation run — that run IS Rung 3 in §3 above.
