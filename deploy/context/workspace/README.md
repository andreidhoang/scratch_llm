# CS336 — Language Modeling from Scratch

> Master Stanford **CS336** from first principles — build the whole language-model stack by hand,
> A1 → A5, to production engineering standard — and come out able to whiteboard and defend every
> layer cold in a frontier-lab interview.

This workspace has two parts:

| Directory | What it is |
|---|---|
| [`lectures/`](lectures/) | **The official course** — lecture code (`lecture_01.py` … `lecture_17.py`), the slide PDFs, and the five assignment scaffolds (`assignment1-basics` … `assignment5-alignment`) with their PDFs and `tests/adapters.py`. This is the **spec and the test oracle**: the PDFs define every deliverable, the adapter tests verify your implementation. *Read-only reference — don't modify it.* |
| [`scratch_llm/`](scratch_llm/) | **Your from-scratch implementation** (Python package `scratch_llm`) — production-grade: green CI (ruff/pyright/pytest), tests-as-spec, ADRs, design docs. This is where you build, and the artifact you show. |
| [`reference/`](reference/) | **Inference-frontier reference** (external research, dated 2026-06-22) — a serving-stack / kernel curriculum + a cited 2026 landscape brief. *Reference only; subordinate to [`STRATEGY.md`](STRATEGY.md) §6 + [`DELTA.md`](DELTA.md) (the inference plan of record) and [`scratch_llm/docs/FRONTIER_PRACTICE_2026.md`](scratch_llm/docs/FRONTIER_PRACTICE_2026.md) (the 2026 frontier layer). Not repo canon.* |

> **Capstone — DELTA** (the barbell *spike*, sitting on the A2/A5 base): a fused **GatedDeltaNet-2
> decode-step** kernel. Merged design RFC + dated 4-week plan live in this root —
> [`DELTA.md`](DELTA.md) — tracked in
> `scratch_llm/docs/STATUS.md` and `IMPLEMENTATION_PLAN.md` §7.

The workflow for each assignment: learn the load-bearing equations → build the module from scratch
in `scratch_llm` → verify against the official `lectures/assignment*/tests/adapters.py`.

## The curriculum — five assignments, one stack

| # | Assignment | What you build (load-bearing core) | The single most load-bearing thing |
|---|---|---|---|
| **A1** | Basics | byte-level BPE · Transformer (RMSNorm·RoPE·SwiGLU·GQA) · cross-entropy · AdamW · cosine schedule · training loop · sampling | a Transformer + tokenizer you own, proven by **loss-at-init ≈ log(vocab)** and overfit-one-batch |
| **A2** | Systems | Triton **FlashAttention-2** (fwd+bwd) + roofline · DDP (naive→overlap) · ZeRO-1 · FSDP · gradient checkpointing · mixed precision | FA2 + the roofline, and the **100B-model memory math** (DP+TP+PP) |
| **A3** | Scaling | **IsoFLOP / Chinchilla** fit (compute-optimal N, D) + the budget-constrained training-API leaderboard | the log-log power-law fitter + `C=6ND` + honest extrapolation |
| **A4** | Data | CommonCrawl → filter → **quality classifier** → exact + **MinHash/LSH dedup** | the dedup machinery (`P[match]=Jaccard`, the LSH S-curve) + the quality-signal design |
| **A5** | Alignment | SFT → Expert Iteration → **GRPO / Dr.GRPO** + the verifiable-reward grader; supplement: **DPO**, reward modeling, safety | GRPO + Dr.GRPO (and correct SFT masking) — the scarcest 2026 skill cluster |

You **build** the stack in order A1 → A5 (A1/A2 are the foundation everything stands on); but with
A1 ✅ done, the **ship order is EV-ranked, not numeric**: **the A5 RL "aha" ships first** (highest-EV,
the scarcest 2026 cluster), the A2 systems finish runs in parallel (it *is* the DELTA Phase-1 harness),
and DELTA is the spike on that base. The one canonical sequence is **[`STRATEGY.md`](STRATEGY.md) §8** —
if any other doc implies a different order, §8 wins. Full per-assignment guides — every deliverable tagged
**LOAD-BEARING / COURSE-ROTE / SKIP** — are in
[`scratch_llm/docs/assignment_guides/`](scratch_llm/docs/assignment_guides/INDEX.md);
the build spine is [`docs/IMPLEMENTATION_PLAN.md`](scratch_llm/docs/IMPLEMENTATION_PLAN.md).

## The engineering disciplines (the hiring signal)

These are baked into the tests, not bolted on — weak engineering is the most common silent rejection
of otherwise-strong candidates:

- **loss-at-init ≈ log(vocab_size)** — the cheapest correctness oracle.
- **overfit-one-batch** — before any real run, drive train loss → 0 on one batch.
- **fixed-seed reproducibility** — a re-run reproduces the metric.
- **mandatory RL logging** — entropy + the KL divergences + reward/length stats (any RL run).
- **predict-before-you-run** — write the falsifiable number first.
- **green CI** — `ruff check` + `ruff format --check` + `pyright` + `pytest` on every commit.

## CS336 → frontier-lab interview readiness

The 2026 bottleneck moved off "can you train a transformer" (commoditized) and onto **post-training
(stable RL × a real reward signal)** and **inference/systems (serve it economically)**. This
curriculum is built to hit exactly those, from first principles. The mapping:

| Scarce 2026 skill | Where you build it |
|---|---|
| RL post-training (GRPO/Dr.GRPO, reward modeling, RLHF vs DPO) | **A5** |
| Distributed training (DDP/ZeRO/FSDP, comms overlap) | **A2** |
| GPU kernels (Triton, FlashAttention-class) | **A2 + DELTA** |
| Inference & serving (KV-cache, quantization, train/inference drift) | **A2** (+ DELTA decode kernel) |
| MoE (routing, aux-loss-free load balancing) | **A1** (opt-in `moe.py`) |
| Data curation (filtering, dedup, quality) | **A4** |
| Scaling laws (IsoFLOP, compute-optimal allocation) | **A3** |

**The universal interview gates this prepares you for** (and where each is earned):
- *Implement Multi-Head Attention / a full Transformer from scratch in ~45 min* → **A1**.
- *"How would you train a 100B+ parameter model?"* (DP+TP+PP + the ~16–20 B/param memory math) → **A2**.
- *Derive / explain scaling laws; compute-optimal N and D* → **A3**.
- *Explain RLHF vs DPO; what makes RL stable; GRPO group-relative advantage* → **A5**.
- *Why is batch-1 decode memory-bound? Fuse a linear-attention decode recurrence; what would falsify your result?* → **DELTA capstone** (the inference/kernel axis).

**Role → the assignments that carry it:**
- RL / Post-Training Research Engineer → **A5** (+ A1, A4)
- Inference / Systems Research Engineer → **A2 + DELTA** (+ A1) · **+ the upstream serving-stack OSS lane** → [`STRATEGY.md`](STRATEGY.md) §6 (vLLM/SGLang/TRT-LLM/Dynamo · the PR ladder · the A2/DELTA→PR crosswalk · role targets)
- Pretraining / Core Modeling Research Engineer → **A1 + A2 + A3**
- Data Research Engineer → **A4** (+ A3)

The portfolio signal that gets you hired (per solo-engineer lab-hiring guidance): a **public GitHub
repo with clean engineering** — green CI, tests, design docs — that you can explain from first
principles. That repo is `scratch_llm/`.

**Career & hiring strategy** — the verdict (planning vs execution), the GDM/Feinberg signals (the Scaling-Book
+ transformer-from-scratch video; the internal-transfer route; the FUD rebuttal), the serving-stack OSS PR
ladder, role targets, and the one prioritized ship-next — all live in [`STRATEGY.md`](STRATEGY.md).

## Start here

1. Read [`scratch_llm/CLAUDE.md`](scratch_llm/CLAUDE.md) (the operating constitution + disciplines).
2. Read [`scratch_llm/docs/IMPLEMENTATION_PLAN.md`](scratch_llm/docs/IMPLEMENTATION_PLAN.md) (the A1→A5 build spine) and [`docs/STATUS.md`](scratch_llm/docs/STATUS.md) (what's built).
3. Pick the next assignment, open its guide in [`docs/assignment_guides/`](scratch_llm/docs/assignment_guides/INDEX.md), build the load-bearing 20% in `scratch_llm`, and verify against the official `lectures/assignment*` adapter tests.
4. For each pillar, the **2026 frontier-practice layer** — modern-default upgrades to adopt, opt-in build labs, and interview-awareness items, all fact-checked against primary sources — is in [`scratch_llm/docs/FRONTIER_PRACTICE_2026.md`](scratch_llm/docs/FRONTIER_PRACTICE_2026.md).

**Status (2026-06-20):** A1 substrate ✅ · A2 partial (FA2 + KV-cache + monitors + rollout ✅; DDP/ZeRO-1/FSDP ⬜) · A3/A4/A5 ⬜. 92 tests green; last code commit Jun 8. **Next ship — EV-ranked, not numeric: the A5 RL "aha"** (GRPO/Dr.GRPO on Countdown, Qwen2.5-1.5B; CPU-scaffolded + one ~$30–100 burst) — the canonical sequence is [`STRATEGY.md`](STRATEGY.md) §8. **Capstone DELTA** (GDN-2 decode kernel): design + 4-week barbell plan written (fact-checked 2026-06-14); **base-first** — the kernel is gated behind the A5 ship + the Step-0 dependency gate.
