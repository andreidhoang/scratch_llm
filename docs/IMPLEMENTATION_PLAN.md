# CS336 from scratch → production — Master Implementation Plan

> **What this doc is.** The **build spine**: the cross-assignment view the per-assignment guides
> (`docs/assignment_guides/A*.md`) cannot give — the build order, the dependency flow, the
> engineering disciplines applied at each step, and the definition of done. It does **not** restate
> the guides (which enumerate every deliverable and tag it LOAD-BEARING / COURSE-ROTE / SKIP) — it
> cites them. Per-deliverable derivations live there; the *spine* lives here.
>
> **The goal:** master CS336 by building all five assignments yourself, A1 → A5, to production
> engineering standard — the artifact + understanding that gets you hired at a frontier lab. The
> official course (`../lectures/`) is the spec and the test oracle; this repo is your implementation.

---

## 0. The through-line (replaces any single metric)

> **Own every layer — byte → BPE → Transformer → systems → scaling → data → RL post-training — to
> production standard, and be able to whiteboard and defend each piece cold.**

The engineering disciplines are the spine that runs through all five assignments — they are how
labs silently screen, and they are baked into the tests, not bolted on:

1. **loss-at-init ≈ log(vocab_size)** — the cheapest correctness oracle (A1 `model.py`).
2. **overfit-one-batch** — drive train loss → 0 on one batch before any real run (A1 `optim`/`train`, A5 SFT/GRPO).
3. **fixed-seed reproducibility** — every stochastic module (`utils/seeding.py`).
4. **mandatory RL logging** — entropy + the KL divergences + IS-ratio/reward/length stats (A5; the absence of this makes an RL run uninterpretable).
5. **predict-before-you-run** — write the falsifiable number first; it is the debugging anchor.

If a change doesn't build, test, or defend one of the five assignments to this standard, question it.

---

## 1. Current state (what is green, what is greenfield)

From `docs/STATUS.md` (456 CPU + 14 GPU-marked tests green, ruff/pyright clean — STATUS carries the
live numbers; this table tracks the CS336 track only, the perf curriculum lives in
`performance/PERF_PLAN.md`):

| Assignment | Built & green | To build |
|---|---|---|
| **A1** Basics | ✅ tokenizer · model (RMSNorm/RoPE/SwiGLU/GQA-ready MHA + opt-in QK-norm) · MoE (opt-in) · AdamW+clip+cosine · train · sampling | — (complete) |
| **A2** Systems | ✅ FlashAttention-2 (oracle+Triton+roofline) · KV-cache · `utils/monitors.py` · rollout seam (`LocalBackend`) · ✅ DDP (naive→flat→overlap, gloo) · ✅ ZeRO-1 · ✅ FSDP · ✅ 100B memory one-pager + comms algebra (shipped 2026-07-03, W1–W4) | ⬜ real SGLang serving (Hopper) — rental-gated |
| **A3** Scaling | ✅ `scaling/` IsoFLOP/Chinchilla fitter (a=0.469 · b=0.531) + budget query planner (2026-07-03, W5–W6) | ⬜ Stanford-API leaderboard (blocked-external, runbook in `deploy/runbooks/`) |
| **A4** Data | ✅ `data/` extract·filter·quality-classify·dedup (exact + MinHash/LSH; 2026-07-04, W7) | ⬜ full 5000-WET run (SKIP; slice runbook) |
| **A5** Alignment | ✅ env/grader protocol (`envs/protocol.py`) · ✅ `algos/` (SFT·EI·GRPO/Dr.GRPO·DPO) · `rewards/` · `envs/` (code-complete 2026-07-04, W8) | ⬜ graded Qwen2.5-Math GPU runs + R1-Zero "aha" repro (rental-gated) |
| **Capstone** DELTA | ✅ design doc + 4-week barbell plan (fact-checked 2026-06-14) | ⬜ Step-0 gate → Phase-1 harness (= A2 finish) → Triton decode kernel → ablations → postmortem (see §7) |

All of `algos/`, `rewards/`, `envs/`, `scaling/`, `data/` are built and green (main-track sprint,
ADR-0014); graded GPU runs are the only rental-gated remainder.

> **Ordering mandate (2026-06-30) — DISCHARGED (perf curriculum complete 2026-07-04).** The
> `performance/` curriculum (A1–A7) shipped first as mandated; all sm120-runnable rungs are done
> (only the rental DAYS remain — `performance/PERF_PLAN.md`). The mandate was dissolved into the
> two-front split (ADR-0013/0014). The paragraph below is the canonical post-mandate sequence.

**Build-order is numeric
(each layer builds on the last); ship-order is EV-ranked — they are not the same thing.** A1's substrate
is the policy A5 fine-tunes and A2's systems make A5's rollouts cheap, so A1/A2 are the foundation; but
with **A1 ✅ done**, the **next artifact to *ship* is the A5 RL "aha"** (the scarcest 2026 cluster,
highest-EV, CPU-scaffolded + one ~$30–100 burst), with the **A2 distributed finish + OSS Rung-1 in
parallel** (the A2 harness *is* DELTA's Phase-1 — §7), then **DELTA base-first**, and **A3/A4 as thin
slices only** (least scarce). *(Historical, 2026-06: this ship-order has since played out — A2
distributed ✅, A3/A4 ✅; and DELTA's kernel payload is re-aimed at **KDA** per the K3 roadmap K10.2,
2026-07-31. The live critical path is K3 K0–K10 + S3 → P5 → d20 — see `docs/STATUS.md`.)* The canonical sequence is **`../STRATEGY.md` §8** + **§7 below** — they win
over any "linear A1→A5" phrasing. (You still *master* each assignment in numeric order; you *prioritize
shipping* by EV.)

**The capstone (DELTA) sits on top.** The five assignments are still *built* in numeric order A1→A5
(mastery order); **DELTA — the barbell *spike*** — is sequenced by its own execution plan so that its
Phase-1 harness *is* the A2 inference-systems finish, and it runs alongside A5 (the RL *base*). It does
not reorder the build;
it gives the A2/A5 work a deep, differentiating endpoint. See **§7**.

---

## 2. The repo ↔ official-scaffold split

| You build from scratch in this repo | You run in the official `../lectures/assignment*` scaffold |
|---|---|
| A1 substrate · A2 kernels/distributed · A3 IsoFLOP fitter · A4 curation+dedup · A5 SFT/GRPO/DPO | the network-locked / leaderboard pieces that can't be "owned": A3's Stanford **training-API leaderboard**, A2's **8B leaderboard**, A4's full **Paloma** training run |

The scaffolds' `tests/adapters.py` are the **acceptance tests** for your from-scratch modules: wire
your implementation behind the adapter, run the assignment's test suite, and you have an objective
correctness oracle. Leaderboards are SKIP for mastery (capped, GPU-dollar sinks) — run them only if
compute is free.

---

## 3. Per-assignment build briefs

Each brief: the **core** (what CS336 makes you build), the **load-bearing 20%** (master cold), the
**discipline gates**, the **production polish**, and the **interview leverage**. Full deliverable
tables + equations + checklists are in the cited guide.

### A1 · Basics — the substrate ✅ (guide `A1_basics_BUILD_GUIDE.md`)
- **Core:** byte-level BPE (train + `Tokenizer`), the full Transformer chain (Linear → Embedding → RMSNorm → RoPE → SwiGLU → softmax → SDPA → MHA → block → LM), `cross_entropy`, AdamW, cosine schedule + warmup, gradient clipping, `np.memmap` data loading, checkpointing, the training loop, and temperature/top-p decoding.
- **Load-bearing 20%:** the BPE merge rule (deterministic tie-break), RoPE, SDPA + causal masking, SwiGLU, the decoupled-AdamW update, and the loss-at-init identity. These are the "implement a Transformer from scratch in ~45 min" gate.
- **Gates:** loss-at-init ≈ log V · overfit-one-batch · seed-repro.
- **Production polish:** every primitive has a unit test; the `bpe_example` reproduces exactly; end-to-end integration test (BPE→train→sample). ✅ done.
- **SKIP:** the OpenWebText perplexity leaderboard, 32K-vocab OWT BPE, C++/Rust BPE speedups. Do `transformer_accounting` + `adamw_accounting` (the memory math) — they recur as the A2 "100B model" interview answer.
- **Interview leverage:** "implement MHA / a Transformer layer," tensor-shape & masking fluency, "what's the loss at init?"

### A2 · Systems — make one GPU fast and many GPUs coherent (guide `A2_systems_BUILD_GUIDE.md`)
- **Core:** Triton FlashAttention-2 (forward + backward) + the roofline; DDP (naive → flat-bucket → overlap via `register_post_accumulate_grad_hook`); ZeRO-1 optimizer-state sharding; **FSDP**; gradient (activation) checkpointing; mixed precision; the parallelism comms algebra.
- **Load-bearing 20%:** FA2 tiling + online softmax (and the recomputation backward with the D-vector); the roofline (arithmetic intensity, % of peak, BW- vs compute-bound); the ~16–20 B/param optimizer-state memory math → why 100B needs DP+TP+PP; DDP overlap; ZeRO-1.
- **Gates:** predict-before-you-run (write expected ms first) · `cuda.synchronize()` around all timing · fixed-seed.
- **Production polish:** FA2 validated against the pure-PyTorch oracle and SDPA; DDP/ZeRO/FSDP each with a 2-rank gloo equivalence test (×5); the 100B memory one-pager.
- **Status:** ✅ complete (both halves) — single-GPU ✅ (FA2 fwd+bwd · selective checkpointing · mixed-precision numerics · KV-cache · monitors · rollout seam · roofline) and distributed ✅ shipped 2026-07-03 (DDP naive/flat/overlap · ZeRO-1 + 100B one-pager · FSDP per-param ZeRO-3, graded · comms algebra — W1–W4, gloo-verified). Remainder: real SGLang on sm120 is an open **ADR-0008** re-check (was Hopper-gated on Ada); multi-GPU NCCL throughput rents a multi-GPU box.
- **SKIP:** the 8B leaderboard; the optional Triton FA2 backward (Alg. 2) — do the `torch.compile` recomputation backward.
- **Interview leverage:** the xAI inference loop (kernels, KV-cache, quant), "how would you train a 100B model?" (DP+TP+PP + the memory math), **MFU / inference co-design** (why 100% MFU is an anti-goal; pick matrix topologies that saturate the units), and the **MoE serving** tradeoff (expert-parallel all-to-all vs pipeline-prefill).

### A3 · Scaling — fit a curve and extrapolate under a FLOP budget (guide `A3_scaling_BUILD_GUIDE.md`)
- **Core:** the IsoFLOP scaling-law fit (`chinchilla_isoflops`: per-budget min-picking → `N_opt ∝ C^a`, `D_opt` via `C = 6ND`, the `a+b≈1` sanity check, extrapolation to ≥10²⁴ FLOPs) **and** the budget-constrained `scaling_laws` training-API leaderboard — the 50-point heart of the assignment: a query planner that spends a fixed compute budget to fit a loss surface, then predicts the optimal config.
- **Load-bearing 20%:** the reusable log-log power-law fitter + the `C=6ND` bridge + the `a+b≈1` check + honest extrapolation; the budget-aware query discipline (predeclare the grid, never over-reserve).
- **Repo:** `scaling/` is a clean CPU/numpy IsoFLOP/Chinchilla fitter (`isoflop_min`, `fit_powerlaw`, `PowerLaw.predict`). The Stanford-API leaderboard runs in the official scaffold (network-locked).
- **Gates:** predict-before-you-run (write `a ≈ b ≈ 0.5`, `a+b≈1` first).
- **SKIP:** the full 5-param `L(N,D)=E+A/Nᵅ+B/Dᵝ` fit (the PDF itself says use IsoFLOP min-picking); chasing the exact leaderboard score.
- **Interview leverage:** "derive/explain scaling laws," IsoFLOP methodology, compute-optimal allocation.

### A4 · Data — the CommonCrawl → filter → dedup → train pipeline (guide `A4_data_BUILD_GUIDE.md`)
- **Core:** HTML→text extraction, language ID, Gopher quality heuristics, PII masking, NSFW/toxic classifiers, the **trained quality classifier** (15 pts: trusted-positive vs random-CC fastText), `exact_deduplication`, **`minhash_deduplication`** (MinHash signatures + LSH banding + Jaccard), and the filtering pipeline (`filter_data`).
- **Load-bearing 20%:** the dedup machinery (MinHash `P[match]=Jaccard`, the LSH S-curve), the quality-classifier *signal design* (quality is defined by the label source), and the pipeline order (cheap/destructive-early; dedup last) + per-filter discard accounting.
- **Repo:** `data/` — `ngram`/dedup primitives, `curate` (filters + quality), against the official `tests/adapters.py`.
- **Gates:** verify `P[minhash match] ≈ Jaccard` on a hand-built pair before trusting LSH; "become one with the data" (inspect kept/removed examples).
- **SKIP at scale:** the full 5000-WET / 375 GB run and the Paloma 200K-iter training leaderboard — run a small slice to exercise pipeline order, then stop. Slurm/`submitit` orchestration.
- **Interview leverage:** data-pipeline design, "how do you prevent eval contamination," "how does data quality change a scaling outcome."

### A5 · Alignment — the RL post-training crown (guide `A5_alignment_BUILD_GUIDE.md`)
- **Core (main):** the SFT primitives (`tokenize_prompt_and_output` + `response_mask`, `compute_entropy`, `get_response_log_probs`, `masked_normalize`, `masked_mean`, `sft_microbatch_train_step`); **Expert Iteration** (STaR); **GRPO** (group-normalized advantage Eq. 28) and **Dr.GRPO** (Eq. 31, drop the std-normalization), the GRPO-clip loss (Eq. 33), the policy-gradient-loss dispatcher (the 3-variant ablation), `grpo_microbatch_train_step`, the `grpo_train_loop`; and the verifiable-reward grader (format + answer).
- **Core (supplement — part of mastery):** **DPO** loss (Eq. 3), reward modeling / Bradley-Terry, and the safety / red-teaming concepts. (The heavy Llama-3-8B SFT/DPO *runs* are optional/GPU; the loss + concepts are core for "explain RLHF vs DPO".)
- **Load-bearing 20%:** SFT masked cross-entropy (correct `response_mask` — everything downstream depends on it); the GRPO group-relative advantage; the GRPO-clip trust region + importance-sampling ratio; the Dr.GRPO de-biasing toggle (std-norm + length-norm); the DPO objective.
- **Repo:** `algos/` (SFT/EI/GRPO/Dr.GRPO/DPO + advantage + off-policy), `rewards/` (the r1-zero grader), `envs/` (a verifiable math/Countdown env behind `envs/protocol.py`, already in place).
- **Gates:** loss-at-init on the SFT head · overfit-one-batch on the SFT step · mandatory RL logging (entropy + KLs + reward/length) wired *before* the first GRPO run · predict-before-you-run.
- **Optional capstone:** reproduce the R1-Zero "aha" on Countdown with Qwen2.5-1.5B — runs on the standing **Blackwell (25 GB fits 1.5B)**; the cheapest end-to-end validation that your RL engine works on a real model (emerges at 1.5B, fails at 0.5B).
- **Frontier lab (GDM-aligned, 🔵):** knowledge distillation — `algos/distill.py` (logit KD · on-policy reverse-KL · sequence-level), the *serve-cheap student* lever a GDM pre-training lead weights most heavily; reuses `cross_entropy` + the rollout seam + the SFT step, and pairs with the A3 distillation-scaling-law note. Detail in `FRONTIER_PRACTICE_2026.md` (A5 🔵 + A3 ⚪).
- **Forward edge (post-aha, 🔵):** multi-turn / agentic tool-use RL on the existing `envs/protocol` — the live 2026 post-training frontier (agentic coding · computer-use · deep-research); single-turn GRPO is the degenerate case. See `FRONTIER_PRACTICE_2026.md` A5 🔵. *(PEFT/LoRA stays awareness-only — frontier flagships full-fine-tune; QLoRA/NF4 and fine-tune-vs-RAG cut as non-frontier.)*
- **SKIP:** the generalist-chat eval re-runs (AlpacaEval/SST/MMLU-delta), the full Anthropic-HH DPO training run — skim for the interview answer, don't invest.
- **Interview leverage:** the whole post-training family — GRPO/Dr.GRPO/PPO, reward modeling, "RLHF vs DPO," RL stability; **distillation** (forward vs reverse KL, the serve-cheap student, distillation scaling laws — the Flash-class infra lever).

---

## 4. Engineering disciplines & green-CI (every commit)

Per `CLAUDE.md`. Apply the gates named in each brief. **Green-CI** (enforced by
`.claude/hooks/green-ci-gate.sh`): `ruff check` + `ruff format --check` + `pyright` +
`pytest -m "not gpu"`. GPU items are `@pytest.mark.gpu` (skipped in CI, exercised on rented
hardware). Commit messages: `<area>: <imperative>`.

---

## 5. Definition of done (per assignment)

An assignment is "mastered to production" when:
- the load-bearing modules are built from scratch in `src/scratch_llm/`, green under ruff/pyright/pytest;
- they pass the official `../lectures/assignment*/tests/adapters.py` where an adapter exists;
- the discipline gates for that assignment are asserted as tests (loss-at-init, overfit-one-batch, seed-repro, RL logging);
- non-obvious decisions are captured as ADRs and the module docstrings state intent + the key invariant;
- you can whiteboard the load-bearing 20% and answer the interview question cold.

---

## 6. Source-of-truth pointers

- Per-assignment how-to (the load-bearing 20%): `docs/assignment_guides/A{1..5}_*_BUILD_GUIDE.md` (start at `INDEX.md`).
- Lecture → assignment → source map (which slides to read first, in what order): `docs/LECTURE_MAP.md`.
- Frontier-practice layer (2026 modern defaults · build labs · interview-awareness, per pillar, fact-checked): `docs/FRONTIER_PRACTICE_2026.md`.
- **Performance curriculum (the active front):** `performance/PERF_PLAN.md` (phases, current node, rentals) + `performance/PERF_ENGINEERING_SPEC.md` (per-assignment falsifiable predictions, DoD, kill criteria). Reference layer: `docs/PERFORMANCE_TRACK.md` (thesis · 2026 findings · honesty constants); on-ramp: `docs/GPU_FROM_ZERO.md` (rung 0→9, no GPU background assumed).
- Build status (single source of truth): `docs/STATUS.md`.
- Constitution + disciplines + green-CI: `CLAUDE.md`.
- Workspace map + interview-readiness: `../README.md`.
- The official course (spec + test oracle): `../lectures/` (lectures + the 5 assignment scaffolds + PDFs).
- **The capstone (DELTA):** `../../DELTA.md` (merged RFC + 4-week barbell plan), in the workspace root. See §7.
- **The K3 track (newest, chartered 2026-07-31):** `docs/k3/ROADMAP.md` + `docs/k3/FACTS.md` — build & host Kimi K3 from scratch on this repo's substrate; the K3 tech report (arXiv:2607.24653) is the spec of record.

---

## 7. Capstone · DELTA — the barbell spike (the deep differentiator)

**The decision (recorded here so it is not re-litigated).** Given a **rental-GPU budget** and an
**undecided target role**, the portfolio is a **barbell**, not a single bet: one high-variance **deep
spike** that makes a specialized team lean forward, plus the low-variance **CS336 base** (A1→A5) that
survives any generalist screen and keeps every door open.

- **Spike = DELTA** — a fused **decode-step** kernel for **GatedDeltaNet-2** (NVIDIA, arXiv 2605.22791):
  correctness oracle + the *measured* H100 memory roofline + the "erase/write decoupling is free at
  decode" thesis + the `C(B)` batch-crossover. The **NVFP4-on-state numerics seam runs on the standing Blackwell now** (native FP4); the full GDN-decode throughput roofline vs FlashInfer wants a rented datacenter card (~20–40 H100-hrs). The
  differentiator — on the exact layer (GDN/linear-attention) the current production wave is decode-bound on.
- **Base = A5 (GRPO/Dr.GRPO "aha") + the A2 finish** — the *other* scarce-2026 cluster (stable RL with a
  verifiable reward) and the systems spine. The A2 finish (`fla` baseline + Nsight harness + the tightened
  kernel + the 100B memory one-pager) **is** DELTA's Phase-1 infrastructure — spike and base share it.

**Sequencing (4 weeks; full detail in the execution plan).** Step-0 dependency gate → Week 1 harness
(= A2 finish) → Week 2 Triton decode kernel (correctness-first) → Week 3 roofline + free-decoupling
ablation + B\* → Week 4 *projected* e2e + postmortem + bilingual writeup. A5 is interleaved CPU-side
throughout (near-zero rental cost) — the insurance against finishing with no post-training signal.

**Scope guards (verified, so the plan can't drift).** No NVlabs checkpoint exists → architecture-only
correctness/roofline, e2e **projected** via Amdahl; the 1.3B config (16 heads, `d_k=d_v=128`) is the whole
substrate (35B dropped); non-linear hybrid layers are 2K SWA; Triton-first; NVIDIA Source-Code-NC license
(portfolio-only, non-commercial).

**Source-of-truth doc (workspace root, one dir above this repo):**
`../../DELTA.md` — the merged RFC (Part I) + dated 4-week plan (Part II).
**K3 cross-link:** the K3 roadmap's **K10.2** re-aims this co-designed kernel at **KDA**
(per-channel decay), building on DELTA's GDN-2 base — see `k3/ROADMAP.md` (K10 stretch).

**Mastery & interview leverage (wire into `/master` + the daily teach-back).** DELTA fills the
**inference/kernel axis** the A1–A5 map is light on. Five `/master`-able concepts, each teach-back-gated
*as you build it* — mastery rides the build, not a separate study track:
1. **Roofline + arithmetic intensity** (Week 1) — *"what is the arithmetic intensity of a linear-attention decode step, and why is it <1 FLOP/byte?"*
2. **The GDN-2 / WY fast-weight recurrence** (Week 2) — *"derive the gated delta rule; what is `S` and how does it update per token?"*
3. **Erase/write decoupling + the free-decoupling thesis** (Week 2–3) — *"why decouple erase from write, and when does the extra cost vanish at decode?"*
4. **Kernel fusion / state residency** (Week 2) — *"how would you fuse this recurrence and keep `S` resident; which passes leak to HBM?"*
5. **Predict-before-you-run as research taste** (all weeks) — *"what would falsify your result?"* (P1–P7 + kill criteria).

`/master` each concept the week you implement it; the end-of-day teach-back covers it. Connect-to-frontier
layer: [`FRONTIER_PRACTICE_2026.md`](FRONTIER_PRACTICE_2026.md) (linear-attention decode); interview-gate
map: [`../README.md`](../README.md).

> **ADR note:** the "DELTA capstone as the barbell spike" decision would normally be an ADR. The
> active set is ADR-0001–0004 · 0006–0008 (0005/0009/0010 were retired — ADRs are immutable records, so
> the freed numbers stay as gaps rather than being renumbered). Promote this §7 to the next ADR number
> when it stabilizes.

---

## 8. Beyond CS336 — the close-the-loop / frontier-ablation front (ADR-0018)

CS336 A1→A5 gave us every *layer*; it never ran the *loop*. The next build spine adopts Karpathy's
**nanochat** integration harness (`speedrun.sh` + report card) over the components we already own to
train an actual **talking model** (headline: nanochat **d20**, measured **480.4M** @ vocab 32768,
~$100 on 8×H100, pre-registered CORE band **0.19–0.22** *(re-anchored 2026-08-02 → **0.23–0.25**,
central ≈0.24 — see `docs/RESULTS.md` §S4-pre; 0.19–0.22 preserved as the pre-registration of
record)* vs the original-d20 anchor **0.2219** — the
old "target CORE ≈ GPT-2" line was mis-anchored, corrected 2026-07-16), then runs an EV-ranked,
pre-registered, **iso-FLOP frontier ablation study** — F1 MuonAdamW,
F2 MTP draft head, F3 de-confound serving, F4 bf16+compile, F5 MLA-for-real, F6 MoE balancing,
F7 GRPO "aha", F8 DSA, F9 logit-guard, F10 hybrid linear attention, F11 agentic/tool-use RL —
plus **F12 (ClimbMix-400B vs FineWeb-EDU corpus ablation): DONE, decision FINAL 2026-08-02 =
ClimbMix by operator override** (measured +0.110 bpb worse, kill criterion fired, overridden
following nanochat's larger-scale result; details `docs/RESULTS.md` §F12) — each with a falsifiable
prediction + kill criterion. This
runs as a **third front** in parallel with the perf curriculum and DELTA; the trained model becomes
what those fronts finally measure against. Per the K3 roadmap, **F5/F6/F10 converge into K6
(mini-K3)** (`docs/k3/ROADMAP.md`). **The full engineering spec + execution DAG is the
source of truth:** [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) (decision:
[`adr/ADR-0018`](adr/ADR-0018-close-the-loop-nanochat-front.md); ledger `bench/RESULTS.md`
§Frontier ablations; plan of record `~/.claude/plans/misty-sniffing-cerf.md`). **Pipeline-level
end-to-end plan (2026-07-30, 9-angle 2026 research pass) — read first:**
[`FRONTIER_2026_END_TO_END_PLAN.md`](FRONTIER_2026_END_TO_END_PLAN.md) (S0→S8 pipeline · re-ranked EV
order · S3 scaling-law gate before the d20 run; rung-level specs stay in ABLATIONS/TASKSPEC).
