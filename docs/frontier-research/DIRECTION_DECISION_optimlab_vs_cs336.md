# Decision — `optim_lab` vs `cs336`: which inference direction, and what to retire

**This is a decision record, not another strategy doc.** Both repos' own audits say the bottleneck is execution, not planning (cs336 is at a 5.3:1 doc:code ratio with code frozen since Jun 8; optim_lab is at Phase −1 having spent zero GPU-hours). So: the call, the evidence, the cut list, then code. **Date:** 2026-06-21. **Method:** 3 parallel research agents (NVIDIA/startup JDs; frontier-lab JDs; live novelty scan), primary sources in §6.

---

## 0. TL;DR — the one decision

**Make `cs336` the primary direction. Archive `optim_lab` as a standalone plan and harvest its one still-open idea into `cs336`'s spike.** They are not two strategies — they are the *same GatedDeltaNet-decode/numerics competency* at two settings. `cs336` is the better vehicle on every axis that matters for hiring: it has **working code** (92 tests green, real FA2, the RL monitors), an explicit **vLLM/SGLang PR ladder** (the single hiring signal every source converges on), and it already contains the **RL base** (A5 GRPO/Dr.GRPO) that optim_lab's most novel experiment depends on. `optim_lab`'s broad numerics map is ~80% pre-empted by its own admission, its 35B/company framing points at the layer your own `COMPANY_THESIS` rejects, and its **one genuinely-open seam** (FP4 numerics on the GDN recurrent state) is better executed inside cs336's cheaper, working, RL-equipped substrate.

**But re-aim the cs336 spike.** Research finding that flips DELTA: **a decode-optimized GDN kernel already ships** (FlashInfer `gated_delta_rule_decode_pretranspose`, in SGLang prod), a faster CuTe-DSL rewrite is mid-flight, and "GDN decode" is an explicit **MLSys-2026 FlashInfer contest track**. DELTA's "decode is the open gap" premise is substantially false for plain GDN; the only open sliver (GDN-2's dual-gate decode) has **no checkpoint → zero production pull**. Replace the spike's novelty claim with the open seam: **low-precision (NVFP4) GDN recurrent-state decode** — the exact tensor NVIDIA's own NVFP4 checkpoints refuse to quantize.

---

## 1. The finding that decides it

| Axis | `optim_lab` (kernel numerics / verifier-suite) | `cs336` + DELTA (curriculum + decode kernel) | Edge |
|---|---|---|---|
| **Maturity** | Phase −1, scaffolding, 0 GPU-hrs, 18 uncommitted files | A1 done, A2 half, **92 tests green, real FA2 + RL monitors shipped** | **cs336** |
| **Hiring legibility** | verifier-suite/numerics map reads as "performance analyst" | Triton kernels + roofline + **vLLM/SGLang PR ladder** = "kernel engineer" (the JD vocabulary verbatim) | **cs336** |
| **Novelty / defensibility** | FP4-on-recurrent-state + hack-rate-vs-precision = **OPEN** (3–6mo window) | GDN decode = **effectively CLOSED** (ships in FlashInfer; MLSys contest track); GDN-2 sliver = thin, no checkpoint | **optim_lab** |
| **Cost** | ~$6.6K, 35B target, B200 | ~$300 (DELTA 20–40 H100-hr) + ~$30–100 (A5) on 1.3B/1.5B | **cs336** |
| **Keeps doors open** | inference + (its real home) the safety/company track | inference **and** post-training (the barbell) | **cs336** |
| **Contains the other's dependency** | needs an RL spine it doesn't have | **has the RL spine** (A5) optim_lab's headline needs | **cs336** |

The table is lopsided for a reason: **cs336 is the executable, legible, cheaper version of the same bet, and optim_lab's surviving advantage (the open seam) can be moved into it.** That is the textbook "merge into the stronger vehicle, retire the weaker plan" situation.

---

## 2. Talent-demand evidence (the user's actual question)

**Inference / kernel market (NVIDIA + startups) — favors the cs336 competency, decisively.**

- NVIDIA JDs name it in plain words. *AI Inference Performance Engineer*: "implement … across **TensorRT-LLM, SGLang, and vLLM**"; "**Apply roofline analysis** and systematic profiling." *Principal SWE – AI Inference*: "**author and land PRs** in vLLM/SGLang … build durable maintainer relationships." Kernel track = CUDA/CUTLASS/Triton/CuTeDSL. Comp (levels.fyi): senior ~$305K, IC5 ~$540K, principal ~$1M+.
- Startups (Baseten $180–360K, Fireworks, Together, Perplexity, Groq) ask for CUDA/Triton/CUTLASS + Nsight before/after + FP8/FP4 + OSS contributions.
- **The merged upstream PR is the universal signal** — NVIDIA states it; Neural Magic (top-vLLM-contributor shop) was acquired by Red Hat; SGLang core spun out as RadixArk ($400M); GPU MODE's pitch is literally "hire the cracked engineers who produced the fast code." A benchmark/verifier artifact does **not** trigger this; a kernel PR does.
- Caveat: NVIDIA Vietnam exists but hires model/platform/speech, not the US inference-kernel title — a no-relocation path is weak today.

**Frontier-lab market — bigger demand is the *other* track, not optim_lab's kernel framing.**

- Anthropic's largest, highest-paid cluster (~14+ reqs, **$500–850K**) is **RL/environments/evals/reward** ("catch reward hacking, reward design") — i.e., your *main `reasoningLLM`* track, not optim_lab. Its newest inference JD even softens the kernel bar ("reasoning about kernels matters more than having written them").
- OpenAI's prominent new family is **Frontier Evals & Environments**; inference sits under crisp SWE titles.
- Epoch AI's 18-interview study: "**robustness against reward hacking** [is] the most important quality criterion." The Information: Anthropic discussed **>$1B** on RL environments.
- GDM: TPU/XLA/JAX-weighted; CUDA unmentioned. The **Feinberg route is real** (verified verbatim, his blog 2026-05-10 + the podcast) — Scaling-Book exercises + a transformer-from-scratch video + his own problems (derive Chinchilla MoE; a Pallas kernel beating `ragged_dot`) → email him in NYC. But it's a personal, headcount-capped, TPU-flavored invite.

**Reconciliation:** the inference/kernel doors (NVIDIA, Baseten, Fireworks — most accessible, visa-friendlier) reward **cs336**. The highest-volume frontier-lab doors reward the **reasoningLLM safety/verifier track**. optim_lab as currently framed (kernel verifier-benchmark, 35B) is the weakest fit for *both* — too analyst-flavored for kernel jobs, too kernel-flavored for the evals orgs.

---

## 3. Novelty / landscape evidence (June 2026)

- **Wedge 1 — GDN-2 decode kernel (cs336/DELTA headline): effectively CLOSED.** FlashInfer ships `gated_delta_rule_decode_pretranspose` (decode-optimized, in SGLang prod); a CuTe-DSL GDN decode kernel is being tuned (FlashInfer #2493); FlashInfer-Bench has a GDN decode benchmark; **MLSys-2026 FlashInfer contest has a GDN-decode track.** The GDN-2-specific sliver is open but derivative and has **no checkpoint → zero production pull**. Scoop risk **HIGH/imminent**.
- **Wedge 2 — FP4 on the GDN recurrent state + hack-rate-vs-precision (optim_lab headline): OPEN.** Shipped NVFP4 checkpoints (Qwen3.5-NVFP4, Nemotron-QAD) **deliberately exclude the recurrent/linear-attn state, keeping it BF16** ("FP4 collapses the recurrence") — so FP4-on-the-state is an unstudied seam the big players are *avoiding*. No quantized-rollout-RL paper (QuRL/AIS/QaRL/FP8-RL/Jet-RL) measures hack-rate as the outcome. Scoop risk **MODERATE (3–6mo)**.

**Implication:** the open novelty lives in optim_lab's seam, not DELTA's. So the winning spike **merges the two**: a *kernel* (DELTA's legible competency + the universal PR signal) on the *open seam* (optim_lab's FP4 recurrent-state numerics). That is simultaneously the most hireable form (a kernel/quant PR on the exact layer production is bottlenecked on) and the most defensible (a seam NVIDIA is routing around).

---

## 4. The verdict, by goal

| If the target is… | Primary direction | Why |
|---|---|---|
| **NVIDIA / Baseten / Fireworks / Together** (most accessible) | **cs336**, spike re-aimed to FP4-GDN-state decode | legible kernel signal + the merged vLLM/SGLang PR every JD names |
| **Anthropic Horizons / OpenAI Frontier Evals** (largest demand) | **main `reasoningLLM`** (RQ2 verifier/reward-hacking) — *not* optim_lab | the $500–850K cluster; matches the JDs verbatim |
| **GDM pretraining (Feinberg)** | **cs336** A1–A3 + the Scaling-Book video, ported to TPU/JAX/Pallas | the route he literally named; TPU-flavored |

In **all three**, standalone `optim_lab` is dominated. Its value survives only as (a) the FP4-recurrent-state seam folded into cs336's spike, and (b) the hack-rate-vs-precision question folded into the reasoningLLM RL track (where it always belonged — it needs the RL spine).

---

## 5. Keep / merge / retire — the concrete cut list

**KEEP & make primary — `cs336`:**
- The A1–A5 substrate + the PR ladder + the roofline discipline + the working `scratch_llm` code.
- **Ship order unchanged from cs336's own audit:** A5 RL "aha" first (highest floor, scarcest cluster), one vLLM/SGLang Rung-1 PR in parallel from week 1.

**MERGE in (from optim_lab → cs336), one idea only:**
- Re-aim the DELTA spike to **NVFP4 GDN recurrent-state decode**: the fused decode kernel *at low precision on the state* (the open seam), delivered as a FlashInfer/vLLM/SGLang PR + a measured numerics writeup (error-accumulation-vs-context + roofline). Optionally tie to **hack-rate-vs-rollout-precision** using cs336's A5 RL spine — that bridges cleanly to the main reasoningLLM safety work.
- Drop DELTA's "novel GDN-2 decode gap" framing; keep its correctness-before-speed discipline and the contest/PR angle (entering the MLSys-2026 FlashInfer GDN track is itself a hiring signal).

**RETIRE / ARCHIVE — `optim_lab` as a standalone plan (don't delete history):**
1. Commit the 18 uncommitted files; push.
2. Add a `SUPERSEDED.md` pointing to this decision + cs336; tag the repo `v-archived-2026-06`.
3. Migrate the one seam (above) into a cs336 spec; leave the GitHub repo for provenance.
- Rationale: ~80% novelty pre-empted (its own triage), heavier/costlier, less legible for hiring, and its surviving idea now lives in cs336.

**HYGIENE — kill the three-`reasoningLLM` confusion:**
- `cs336/reasoningLLM_scratch` is actually the CS336 build (`scratch_llm`), unrelated to the main reasoningLLM safety repo. **Rename it `cs336/scratch_llm`** (or document the distinction in its README) so there aren't three "reasoningLLM" things.

---

## 6. The meta-point (both repos' own audits agree)

Whichever way you cut it, **the next action is code, not a doc.** cs336's FEINBERG_MAP says it outright: "stop analyzing and resolve one stochastic node." Your taste is already frontier-grade and unfalsified; one converged A5 run + one merged PR is worth more than this memo. So treat this as the last planning artifact on this fork: **archive optim_lab this week, re-aim the spike, then ship the A5 node and open one vLLM/SGLang PR.** Don't write the next strategy doc.

---

## Sources

**Talent:** NVIDIA AI Inference Performance Engineer — jobs.nvidia.com/careers/job/893393953033 · NVIDIA Principal SWE AI Inference (JR2013753) · Baseten GPU Kernels — jobs.ashbyhq.com/baseten · Anthropic Performance Engineer GPU / Inference Systems — job-boards.greenhouse.io/anthropic/jobs/4926227008, /5224564008 · Anthropic RE (RL) /4613568008 · OpenAI Frontier Evals & Environments — openai.com/careers · Epoch AI RL-envs FAQ — epoch.ai/gradient-updates/state-of-rl-envs · Feinberg — vladfeinberg.com/2026/05/10/how-to-land-a-job-at-a-frontier-lab.html · Scaling Book — jax-ml.github.io/scaling-book · levels.fyi/companies/nvidia
**Novelty:** GDN-2 — arxiv.org/abs/2605.22791 · NVlabs/GatedDeltaNet-2 (training-only, no checkpoint) · FlashInfer GDN decode — sglang #20791, flashinfer #2493 · FlashInfer-Bench gdn_decode · MLSys-2026 FlashInfer contest — mlsys26.flashinfer.ai · USC FPGA decode — arxiv.org/abs/2603.05931 · CHON NVFP4 — arxiv.org/abs/2602.02047 · Nemotron NVFP4-QAD — arxiv.org/pdf/2601.20088 · nvidia/Qwen3.5-397B-A17B-NVFP4 (excludes recurrent state) · QuRL 2602.13953 · AIS 2605.13907 · QaRL 2604.07853
**Internal:** cs336/{README,STRATEGY,DELTA,FEINBERG_INTERVIEW_MAP}.md · optim_lab/{PLAN,decisions}.md · reasoningLLM/docs/_private/{NORTH_STAR,COMPANY_THESIS}.md
