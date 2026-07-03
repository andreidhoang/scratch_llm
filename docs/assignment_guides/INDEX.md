# CS336 Assignment Build Guides — INDEX

> **What this is.** Five senior-engineer reviews of the CS336 assignments, each mapping every
> deliverable onto this repo's `src/scratch_llm/` and tagging it **LOAD-BEARING** (master it cold —
> it's the 20% that carries the assignment and the interviews), **COURSE-ROTE** (implement correctly
> to pass the tests, don't over-invest), or **SKIP** (a capped leaderboard / dollar-sink with no
> mastery carry).
>
> **How to read these.** For each assignment: learn the load-bearing equations first, build the
> modules from scratch in this repo, and verify against the official `../../../lectures/assignment*`
> `tests/adapters.py`. The question every guide answers: *which 20% of this assignment masters it,
> and where does it live in `src/scratch_llm/`?*

## The organizing principle (every assignment serves this)

> **Own every layer — byte → BPE → Transformer → systems → scaling → data → RL post-training — to
> production standard, and be able to whiteboard and defend each piece cold.**

## Master map — five assignments, five guides

| Assignment | Guide | Source files it builds | The single most load-bearing thing |
|---|---|---|---|
| **A1** Basics | [`A1_basics_BUILD_GUIDE.md`](A1_basics_BUILD_GUIDE.md) | `tokenizer.py`, `model.py`, `moe.py`, `optim.py`, `train.py`, `sampling.py` | **A Transformer + BPE you own end-to-end** — proven by loss-at-init ≈ `log(vocab)` and overfit-one-batch. The "implement a Transformer in ~45 min" gate. |
| **A2** Systems | [`A2_systems_BUILD_GUIDE.md`](A2_systems_BUILD_GUIDE.md) | `kernels/`, `rollout/`, `utils/`, DDP/ZeRO/FSDP | **FlashAttention-2 (Triton) + the roofline**, and **DDP-overlap / ZeRO-1 / FSDP + the 100B memory math**. |
| **A3** Scaling | [`A3_scaling_BUILD_GUIDE.md`](A3_scaling_BUILD_GUIDE.md) | `scaling/` | **The IsoFLOP / Chinchilla fitter + the budget-constrained training-API leaderboard** (predict the compute-optimal config). |
| **A4** Data | [`A4_data_BUILD_GUIDE.md`](A4_data_BUILD_GUIDE.md) | `data/` | **The dedup machinery (MinHash + LSH) + the trained quality classifier** + the filtering pipeline. |
| **A5** Alignment | [`A5_alignment_BUILD_GUIDE.md`](A5_alignment_BUILD_GUIDE.md) | `algos/`, `rewards/`, `envs/` | **GRPO + Dr.GRPO** (and the SFT masking primitives) + **DPO** — the post-training engine, the scarcest 2026 skill cluster. |

## Build order (linear A1 → A5)

```
A1 Basics ✅      substrate: BPE · Transformer · AdamW · training · sampling          (the policy)
A2 Systems 🟡     FA2 ✅ · KV-cache ✅ · monitors ✅ · rollout ✅ · DDP ✅ → ZeRO-1/FSDP + memory math (queued behind the perf mandate)
A3 Scaling ⬜     IsoFLOP / Chinchilla fitter (+ the Stanford-API leaderboard in the official scaffold)
A4 Data ⬜        extract → filter → quality-classify → exact + MinHash/LSH dedup
A5 Alignment ⬜   SFT → Expert Iteration → GRPO/Dr.GRPO → DPO + safety        (the RL crown)
```

A1 and A2 are the foundation (the model A5 fine-tunes; the systems that make A5's rollouts cheap),
so although the build is linear, they carry the most weight. **Optional A5 capstone:** reproduce the
R1-Zero "aha" on Countdown-1.5B (~$30 GPU) — the cheapest end-to-end RL validation.

## The load-bearing spine (the ~20% across all five)

1. **A1** — a Transformer LM you own (tokenizer + logits + sampler + AdamW), proven by loss-at-init ≈ `log(vocab)` and overfit-one-batch.
2. **A2** — Triton FA2 + the roofline; DDP-overlap + ZeRO-1 + FSDP; the ~16–20 B/param memory math (the "train a 100B model" answer).
3. **A3** — the log-log IsoFLOP power-law fitter + the `C=6ND` bridge + the `a+b≈1` check; the budget-constrained query/leaderboard discipline.
4. **A4** — exact + MinHash/LSH dedup (`P[match]=Jaccard`, the LSH S-curve); the quality-classifier signal design; pipeline order + discard accounting.
5. **A5** — SFT masked-CE + correct `response_mask`; GRPO group-relative advantage; the GRPO-clip trust region; the Dr.GRPO de-biasing toggle; the DPO objective.

## Recurring engineering disciplines (how labs silently screen)

Apply where each guide marks them: **loss-at-init ≈ log(vocab)** (A1) · **overfit-one-batch** (A1, A5) ·
**fixed-seed reproducibility** (all) · **mandatory RL logging** — entropy + the KL divergences +
IS-ratio/reward/length stats (A5) · **predict-before-you-run** the falsifiable number (A2, A3, A5).

## Aggregated SKIP list (do NOT over-invest — capped leaderboards / dollar-sinks)

- **A1:** the OpenWebText perplexity leaderboard / hyperparameter chase; the 32K-vocab OWT BPE; the C++/Rust BPE speedup (`multiprocessing` is enough). *(Keep GQA — it's in the model — and the opt-in MoE; do the `*_accounting` memory math.)*
- **A2:** the 8B leaderboard; the *optional* Triton FA2 **backward** (Alg. 2 — do the `torch.compile` recomputation backward); implementing exotic TP/PP/2D-mesh parallelism beyond the analytical comms algebra (a toy TP only on slack). *(FSDP is LOAD-BEARING, not skip.)*
- **A3:** the full 5-parameter `L(N,D)=E+A/Nᵅ+B/Dᵝ` fit (the PDF says use IsoFLOP min-picking); chasing the exact 48-B200-hr leaderboard score. *(The leaderboard itself is LOAD-BEARING.)*
- **A4:** the full 5000-WET / 375 GB `filter_data` run; the Paloma C4-100 200K-iter training leaderboard; Slurm/`submitit` orchestration. Run a token-budget slice to exercise pipeline order, then stop. *(The quality classifier + dedup are LOAD-BEARING.)*
- **A5:** the generalist-chat eval re-runs (AlpacaEval / SST / MMLU-delta); the full Anthropic-HH **DPO training run** on Llama-8B (build the DPO *loss* and understand it — that's core — but the heavy run is optional). *(SFT, Expert Iteration, GRPO/Dr.GRPO, the DPO loss, and the safety concepts are LOAD-BEARING.)*

## ADR triggers (log in `../adr/`, don't relitigate inline)

- **A1:** tokenizer of record; dense-vs-MoE default (dense default, MoE opt-in).
- **A2:** FA2-only on the rented GPU (FA3 is Hopper-gated); FSDP scope if time is tight.
- **A3:** which `(N, D)` grid is affordable for the leaderboard query budget.
- **A4:** the trusted-positive source for the quality classifier; the dedup Jaccard threshold (near-dup removal vs diversity).
- **A5:** GRPO vs Dr.GRPO default (lean Dr.GRPO, toggle exposed); `masked_mean` vs `masked_normalize` aggregation; whether DPO ships in `algos/` or stays a course exercise.

## Source-of-truth pointers

- Repo constitution + the assignment→source map + disciplines: [`../../CLAUDE.md`](../../CLAUDE.md)
- The build spine (build order, briefs, definition of done): [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)
- Which lectures feed which assignment (read-first order, slides → source modules): [`../LECTURE_MAP.md`](../LECTURE_MAP.md)
- The 2026 frontier-practice layer (per-pillar modern defaults · build labs · interview-awareness, fact-checked): [`../FRONTIER_PRACTICE_2026.md`](../FRONTIER_PRACTICE_2026.md)
- Build status: [`../STATUS.md`](../STATUS.md)
- The official course (spec + test oracle): `../../../lectures/` (lecture code + the 5 assignment scaffolds + PDFs)
