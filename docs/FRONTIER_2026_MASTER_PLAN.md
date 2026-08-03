# Frontier 2026 — Master Plan (curated entry point)

> **Start here.** This file is the curated entry point to the Frontier 2026 program. It explains how
> the four plan documents fit together and gives a senior-engineer overview of the whole front. It
> intentionally does **not** duplicate detailed rung cards or build specs — those live in the linked
> owner docs.
>
> **Document map**
> | File | Role | Read when you need... |
> |---|---|---|
> | [`FRONTIER_STATUS.md`](FRONTIER_STATUS.md) | One-page live status | "What is done and what is next?" |
> | [`FRONTIER_2026_END_TO_END_PLAN.md`](FRONTIER_2026_END_TO_END_PLAN.md) | Integrated pipeline S0→S8 | "How does data flow to a talking model?" |
> | [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) | Ablation strategy + rung cards | "Why and what do we measure?" |
> | [`FRONTIER_2026_TASKSPEC.md`](FRONTIER_2026_TASKSPEC.md) | Buildable file→test spec | "Which file do I edit and what test do I write?" |
> | `bench/RESULTS.md` + `docs/RESULTS.md` | Measurement ledger | "What did we actually measure?" |
> | `deploy/runbooks/d20_speedrun_8xH100.md` | Paid-run script | "How do I launch the $100 d20?" |
| [`FRONTIER_2026_ARCH_SCALING.md`](FRONTIER_2026_ARCH_SCALING.md) | Architecture × scaling doctrine | "Does a variant need a re-sweep or an anchor?" |
| [`FRONTIER_2026_D20_CERTAINTY_PLAN.md`](FRONTIER_2026_D20_CERTAINTY_PLAN.md) | d20 certainty + P5 protocol of record | "What is certified vs borrowed vs undefined?" |
| [`FRONTIER_2026_SCALING_PROGRAM.md`](FRONTIER_2026_SCALING_PROGRAM.md) | Fit engineering + S3.5 → K3 family fit | "How do we produce a law that passes its own gates?" |

---

## 1. North star

Own every layer of a language model, close the loop end-to-end into a talking d20 with an honest
public report card, and use that working baseline to run **pre-registered, iso-FLOP ablations** that
test the 2026 frontier's *contested* claims.

The artifact is not any single technique. The artifact is the **discipline**: pre-registration,
independently tuned baselines, adversarial controls, kill criteria, and published negative results.

---

## 2. Decision principles

We decide like a frontier research engineering team, not like a hype-driven repo:

1. **Evidence hierarchy first.** No money is spent below `[VERIFIED]` evidence. See the label
   convention in [`END_TO_END_PLAN.md`](FRONTIER_2026_END_TO_END_PLAN.md) §1.
2. **EV ranking.** `EV = (leverage × frontier scarcity) / (effort × risk)`. Attention efficiency and
   RL honesty/agentic are Tier 1; table stakes are Tier 2; hygiene is Tier 3.
3. **Iso-constraints.** Every ablation changes exactly one variable while holding `C = 6ND` (iso-FLOP),
   total params (iso-param), or wall-time constant.
4. **Pre-register + kill.** Every rung has a falsifiable prediction and a kill criterion written
   *before* the run.
5. **Scaling law is architecture-family dependent.** Switching attention family requires a new small
   ladder; we never borrow a dense curve for hybrid or sparse models.
6. **d20 is the engineering reference, not the science probe.** Frontier attention variants enter d20
   only if they pass val-loss + recall-probe falsifiers **before P5**.

---

## 3. Pipeline overview (S0→S8)

See [`END_TO_END_PLAN.md`](FRONTIER_2026_END_TO_END_PLAN.md) for the full S0→S8 treatment.

| Stage | What | Key file | Status |
|---|---|---|---|
| **S0 Data** | FineWeb-EDU / ClimbMix shards, tokenizer 32k, decontamination | `data/shards.py`, `data/decontaminate.py` | A1 ✅, A0 ✅; F12 DONE — kill fired, operator OVERRIDE → **ClimbMix FINAL 2026-08-02** |
| **S1 Architecture** | Dense d20: 480.4M, 20L/1280d/10h, full GQA/MHA, untied embeddings | `model.py` | Locked at P5 |
| **S2 Pretraining** | Muon+AdamW, bf16, `C = 6ND`; **no MTP head** (decision of record 2026-08-03: d20 runs `mtp_depth=0`; F2a/F2b retarget post-d20) | `train.py`, `optim.py`, `mtp.py` | Muon ADOPTED; F2a ✅ (post-d20) |
| **S3 Scaling law** | IsoFLOP ladder s1–s8, fit `N*(C)`, `D*(C)`, `a+b≈1` gate | `scaling/isoflop.py`, `scripts/s3_scaling_sweep.py` | ✅ DONE 2026-08-02 (s1–s7, 12.09 GPU-h) — R² gate tripped ⇒ fit rejected, T1 fired, no extrapolation; D:N HOLD ratio-20 (9.6B); d14 escalation = human decision |
| **S4 Distributed pretrain / d20** | 8×H100 ZeRO-2, ~9.6B tokens, report card | `utils/dist_train.py`, runbook | A7 ✅, A8 ✅; gated on P5 |
| **S5 Midtrain** | Chat-shaped continued pretrain (A4, optional) | `data/chat_adapters.py` | Spec done, unbuilt |
| **S6 SFT** | Assistant-only masked CE on chat template | `algos/chat_sft.py` | ✅ CPU-green |
| **S7 RL** | GRPO aha debunk + agentic tool-use RL | `algos/grpo.py`, `envs/tool_env.py` | Harness ✅; real run rental-gated |
| **S8 Serving/eval** | Report card + chat UI + speculative decode payoff | `eval/`, `serving/`, `chat_cli.py` | ✅ CPU-green |

---

## 4. Ablation program overview

See [`ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) for full rung cards and citations.

### Tier 1 — scarce, differentiating

| Rung | Technique | Question | Falsifier (summary) |
|---|---|---|---|
| **F8** | DSA sparse attention | Can sparse indexing recover dense mass? | Recall ≥0.95; +0.03 nats @8k |
| **F10** | Hybrid linear attention (GDN) | Does 3:1 linear:full match quality with KV cut? | +0.03 val loss; state ≥2× smaller; recall parity |
| **F7** | RL "aha" debunk | Is the aha real or GRPO artifact? | Random-reward control recovers most gain |
| **F11** | Agentic / tool-use RL | Can multi-turn RL learn verifiable tool use? | Success-rate↑; format-only control hacks |

### Tier 2 — table stakes

| Rung | Technique | Question | Falsifier (summary) |
|---|---|---|---|
| **F1** | Muon vs tuned AdamW | Does Muon win at our scale? | 1.1–1.4× band vs LR-tuned AdamW |
| **F2** | MTP train head + drafter | Free draft head, neutral bpb? | Δbpb ∈ [−0.02, +0.01]; a2-acceptance ≥0.30 |
| **F4** | bf16 + compile (+ NVFP4 stretch) | Cheap MFU without quality loss? | +10–20% MFU; KL tolerance |

### Tier 3 — necessary hygiene

| Rung | Technique | Question | Falsifier (summary) |
|---|---|---|---|
| **F3** | De-confound serving | Is speculative acceptance real on trained model? | Prose <10%, code/JSON 40–60% |
| **F5** | MLA for real | Substrate for F8/F10; KV cut without loss? | +0.02 val loss vs GQA-8; KV ≥3× |
| **F6** | MoE balancing granularity | Batch-wise vs seq-wise vs bias-free? | BIAS_FREE ≤ others; entropy >0.9·log N |
| **F9** | QK-clip guard | Do logits explode sub-1B? | Max logit <~30 with qk_norm |
| **F12** | ClimbMix vs FineWeb-EDU | Which corpus wins with our recipe? | ClimbMix bpb < FWE at iso-FLOP |

---

## 5. Current next node

See [`FRONTIER_STATUS.md`](FRONTIER_STATUS.md) for the live version.

> ~~**F12 corpus ablation**~~ ✅ DONE (kill fired; operator OVERRIDE → **ClimbMix FINAL 2026-08-02**) → ~~**S3 scaling sweep**~~ ✅ DONE 2026-08-02 (R² gate tripped ⇒ fit rejected, T1 fired; D:N HOLD ratio-20/9.6B) → **human decision: T1/d14 escalation (free-slow vs paid vs proceed-to-P5)** → **P5 d12 dress rehearsal ($10–15)** → **8×H100 d20 (~$100, gated on P5 + user go-ahead)**

F12 was the highest-EV free experiment because data quality often beats optimizer changes at fixed
compute; S3 calibrated our own scaling ladder so the d20 D:N ratio rests on our fitted-interval
evidence, not Chinchilla folklore (the power-law fit itself was rejected — R² gate tripped).

---

## 6. Cost gates

| Gate | Spend | What must be true before triggering |
|---|---|---|
| Phase 0 nano | $0–5 | None — always run to protect downstream spend |
| F12 / S3 / F8 / F10 / F6 | $0 | Standing box (24GB Blackwell) |
| P5 d12 dress rehearsal | ~$10–15 | F12 + S3 entrypoints green; compile re-validated on sm90 |
| 8×H100 d20 | ~$48–100 | P5 pass + user go-ahead + node busbw ≥350 GB/s |
| d32 stretch | ~$800 | A measured ablation result justifies scale |

Abort rule: `$90 cumulative → downsize to d16`.

---

## 7. Key first-principles anchors

These are non-negotiable engineering facts that drive every decision:

- **`C = 6ND`** — the compute identity. Forward ~2N FLOP/token, backward ~4N.
- **Scaling laws are fit in log-log** — raw-space least-squares is dominated by the largest-budget
  point.
- **`a + b = 1` structural gate** — if `N* ∝ C^a` and `D* ∝ C^b`, then `a+b` must be 1 under
  `C=6ND`.
- **Architecture family matters** — dense, MoE, hybrid, sparse each need their own `L(N,D)`.
- **bpb is the only vocab-invariant loss** — never compare per-token CE across different tokenizers.

---

## 8. Honesty ledger — what we have revised

| Finding | Old belief | Updated belief | Source |
|---|---|---|---|
| Muon | 1.4–2× win over AdamW | 1.1–1.4× vs *tuned* AdamW; scale-dependent | Wen 2509.02046 |
| RL "aha" | Reward↑ + length growth = success | GRPO clipping-bias artifact; random rewards recover gain | Dr.GRPO / GSPO |
| Attention frontier | MLA is converged | MLA is legacy; hybrid/DSA/sparse are contested | V4, Kimi Linear, MiniMax |
| d20 ratio | Chinchilla ~20 tok/param | Ratio-20 deliberate overtrain for served artifact | nanochat fit, Sardana 2401.00448 |
| Data | FineWeb-EDU fixed | F12 decided (kill fired — ClimbMix +0.110 bpb worse at 35M/700M); operator OVERRIDE → **ClimbMix FINAL 2026-08-02** | nanochat 324e69c; `docs/RESULTS.md` §F12 |

---

## 9. How to use these docs as a team

- **New to the front?** Read this file, then [`STATUS.md`](FRONTIER_STATUS.md), then the stage you
  care about in [`END_TO_END_PLAN.md`](FRONTIER_2026_END_TO_END_PLAN.md).
- **Picking what to build next?** Read [`STATUS.md`](FRONTIER_STATUS.md) next node + [`ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) EV ranking.
- **Implementing a rung?** Read [`TASKSPEC.md`](FRONTIER_2026_TASKSPEC.md) for exact files/tests/DoD.
- **Reviewing a claim?** Read [`ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) falsifier + `bench/RESULTS.md` measured column.
- **Updating the plan?** Update the **owner doc** for that information, then sync [`STATUS.md`](FRONTIER_STATUS.md)
  and this file's summary. Never let the three source docs diverge on status.

---

## 10. Zone discipline (three fronts)

This front coexists with:

- **Perf/kernel front:** `performance/`, `csrc/`, `bench/kernels/` — do not edit; satisfy their
  Protocols (`Drafter`, `RewardFn`, `VerifiableEnv`).
- **DELTA capstone:** `deploy/`, distributed runbooks — coordinate on shared runbook files.

Rules: additive edits, precise `git add` (never `-A`), pull-rebase shared files, green-CI before
commit.

---

*This file is a curated integration, not a replacement. Detailed strategy lives in
`FRONTIER_2026_ABLATIONS.md`; detailed pipeline lives in `FRONTIER_2026_END_TO_END_PLAN.md`; detailed
build instructions live in `FRONTIER_2026_TASKSPEC.md`.*

*Last updated: 2026-07-31*
