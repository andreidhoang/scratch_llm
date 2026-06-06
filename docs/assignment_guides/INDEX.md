# CS336 → reasoningLLM · Assignment Build Guides — INDEX

> **What this is.** Five senior-engineer reviews of the CS336 assignment PDFs (`../*.pdf`), each mapped onto this repo's source files and the core context docs, so the 14-day build has a single guide and direction. Each guide is *comprehensive + prioritized*: every required deliverable is enumerated, then tagged **LOAD-BEARING** (it advances the engine), **COURSE-ROTE** (do it to pass tests, don't over-invest), or **SKIP**.
>
> **How to read these.** Don't build five homeworks. Build one vertically-integrated reasoning-model stack and let each assignment supply one layer. The question every guide answers is: *which 20% of this assignment is load-bearing for the through-line, and where does it live in `src/reasoning_llm/`?*
>
> **STATUS (actual build order).** Execution went **L1-first** (strict dependency DAG), not the A2-first §0.8 schedule below: **L1 substrate ✅** and **`utils/monitors.py` (kl_train_infer) ✅**. **Next is L2 inference** — the CPU-buildable pieces (KV-cache, then the rollout-client seam); Triton FA2 / SGLang serving / DDP wait for a GPU box. Live progress: [`../STATUS.md`](../STATUS.md).

## The one number (every layer serves this)

```
true_quality_gap = reward − true_quality        (+ hack_rate, kl_train_infer)
```

Owned end-to-end, from the token (L1) to the reward (L5). If a deliverable does not help *measure* or *protect* this quantity, it is COURSE-ROTE or SKIP.

## Master map — five layers, five guides

| Layer | Assignment | Guide | Source files it feeds | The single most load-bearing thing | Schedule (§0.8) |
|---|---|---|---|---|---|
| **L1** Substrate | A1 Basics | [`A1_basics_BUILD_GUIDE.md`](A1_basics_BUILD_GUIDE.md) | the policy served in rollouts (tokenizer · logits · sampler · optimizer) | **Owning the policy end-to-end** — the gap is defined over *your* vocab + *your* sampler. Cheapest proof: loss-at-init ≈ `log(vocab)` | Days 9, 11 |
| **L2** Systems | A2 Systems | [`A2_systems_BUILD_GUIDE.md`](A2_systems_BUILD_GUIDE.md) | `rollout/sglang_client.py`, `utils/monitors.py` | **The `kl_train_infer` bridge** — `KL(train‖infer)`, HALT@0.10; A2 is the mechanistic *why* it's nonzero | Days 1–4 |
| **L3** Scaling | A3 Scaling | [`A3_scaling_BUILD_GUIDE.md`](A3_scaling_BUILD_GUIDE.md) | `scaling/hack_rate_fit.py` | **The reusable IsoFLOP fitter + the reframe** — fit `hack_rate` vs `(N, C_infer)`, not loss (VERA axis-3 / H3) | Day 12 |
| **L4** Data | A4 Data | [`A4_data_BUILD_GUIDE.md`](A4_data_BUILD_GUIDE.md) | `data/curation.py`, `data/contamination.py` | **Contamination check (Phase-3 gate) + reward-data curation** — no VERA number on an uncontaminated-unchecked eval | Day 10 |
| **L5** RLVR engine | A5 Alignment (+ supplement) | [`A5_alignment_BUILD_GUIDE.md`](A5_alignment_BUILD_GUIDE.md) | `algos/{advantage,off_policy}.py`, `rewards/{reward,hack_detector}.py`, `envs/{exploitability,true_quality}.py` | **The verifier-exploitability dial + true-quality oracle** — net-new, *no CS336 equivalent*; CS336 teaches you to optimize a reward, never to audit it | Days 5–8 |

## The build order (A2-first — per STUDY_PLAN §0.8)

The schedule deliberately **ships L5 reasoningLLM files while the morning work is A2/A5** — "what you master" and "what you ship" diverge by design.

```
Days 1–4   L2 (A2)  systems spine → ships envs/exploitability.py, true_quality.py, rewards/hack_detector.py
                    Day 4 derives kl_train_infer — the bridge that makes the rest interpretable
Days 5–8   L5 (A5)  the RLVR engine → ships rewards/reward.py, algos/advantage.py, algos/off_policy.py, rollout/async_orchestrator.py
Day 9      L1 (A1)  Transformer substrate + sampler → ships rollout/sglang_client.py
Day 10     L4 (A4)  data/curation → ships envs/r2e_wrapper.py
Day 11     L1 (A1)  BPE + AdamW + tiny LM → wires utils/monitors.py (kl_train_infer HALT@0.10)
Day 12     L3 (A3)  IsoFLOP → smoke run
Days 13–14 VERA pre-register H1–H3 + Countdown-1.5B → plots + paper → tag v0.1.0
```

De-risk early (Day ~3–4 of A5 mastery): get **GRPO on Countdown-1.5B** reproducing the TinyZero "aha" (~$30) — it validates the whole capstone.

## The load-bearing spine (the ~20% across all five)

Build exactly these and the engine exists; everything else is rote or skip:

1. **L1** — a transformer LM you own (tokenizer + logits + sampler + AdamW), proven by loss-at-init ≈ `log(vocab)` and overfit-one-batch.
2. **L2** — Triton FA2 + roofline (cheap rollouts) and the **three-KL monitor** including `kl_train_infer` (HALT@0.10).
3. **L5** — **GRPO + Dr.GRPO** (group-relative advantage, IS-clip, length/difficulty de-bias) and the **exploitability dial (5 HardeningLevels) + true-quality oracle**.
4. **L4** — dedup + **train↔eval contamination check** (protects every gap number) and the **reward-data quality → `hack_rate`** ablation.
5. **L3** — the IsoFLOP **fitter**, repurposed to fit `hack_rate`/`true_quality_gap` vs `(N, C_infer)` (H3).

## Recurring engineering disciplines (these are how labs silently screen)

Apply where each guide marks them: **loss-at-init ≈ log(vocab)** (A1) · **overfit-one-batch** (A1) · **fixed-seed reproducibility** (all) · **three-KL logging** `KL(cur‖ref) / KL(cur‖old) / kl_train_infer` + IS-ratio histograms + reward/length stats (A2, A5) · **train↔eval contamination check before any eval number** (A4) · **predict-before-you-run** the falsifiable number (all) · **pre-register H1–H3** before any VERA ablation (A3, A5).

## Aggregated SKIP list (do NOT over-invest)

- **A1:** OpenWebText perplexity-leaderboard chase; 32K-OWT BPE; the C++/Rust BPE speedup; GQA/MoE (→ v0.2.0 ADR).
- **A2:** the 8B leaderboard (§9); the *optional* Triton FA2 **backward** (Alg. 2); exotic parallelism beyond the analytical `tp_calcs`/`fsdp_tp_calcs` (toy TP on slack only); exhaustive Nsight tables.
- **A3:** chasing the exact 48-B200-hr leaderboard score; the full 5-param `L(N,D)=E+A/Nᵅ+B/Dᵝ` fit (use IsoFLOP min-picking); enshrining Chinchilla `a≈b≈0.5`; any full VERA scaling sweep inside the frozen v0.1.0 (it's `v0.1.x`).
- **A4:** the full 5000-WET / 375 GB `filter_data` run; the Paloma C4-100 200K-iter training leaderboard; Slurm/`submitit` orchestration. Run only a token-budget CC slice to exercise pipeline order.
- **A5:** `look_at_sft`, `look_at_hh`, all `*_sft` eval-delta re-runs, `dpo_training` on Anthropic-HH, AlpacaEval/SST annotator runs, the red-teaming writeup — all generalist-chat, off the verifiable-reward thesis. Keep only the misuse *catalog* (seeds `hack_detector.py`) and the `dpo_loss` primitive for mastery.

## ADR triggers (log in `../adr/`, do not relitigate inline)

- **A1:** dense vs MoE substrate for v0.1.0; tokenizer choice.
- **A2:** FA2-only on the rented GPU (FA3 is Hopper-gated); SGLang vs vLLM for the rollout client.
- **A3:** which `(N, C_infer)` grid is affordable for the VERA hack-rate fit.
- **A4:** the reward-data noise model (A4.1); the trusted-positive source for the reward-data quality classifier; contamination n-gram size for short/templated MATH/AIME items.
- **A5:** keep DPO in v0.1.0 or GRPO-only; single-family fallback (Qwen-1.5B) if cross-family RL is too finicky; supplement-safety scope.

## Source-of-truth pointers

- Repo constitution + the L1–L5 → source map + disciplines: [`../../CLAUDE.md`](../../CLAUDE.md)
- Positioning / job-market map (the §3 per-layer briefs, §5 evidence ledger): [`../../../UNIFIED_FRONTIER_PROJECT_SPEC.md`](../../../UNIFIED_FRONTIER_PROJECT_SPEC.md)
- Capstone scope + VERA study (H1–H3, the 5 HardeningLevels, budget): [`../../../CAPSTONE_AND_STUDY_PLAN.md`](../../../CAPSTONE_AND_STUDY_PLAN.md)
- The 14-day schedule + per-day skill/role companion: `../../../../daily/STUDY_PLAN_2026.md §0.8`
- Assignment PDFs (the primary sources these guides review): `../*.pdf`

> **Scope discipline.** Everything here builds the **frozen v0.1.0** (engine + smoke run). VERA = `v0.1.x`; MoE/VLM = `v0.2.0` — ADR stubs in `../adr/`, never smuggled into v0.1.0.
