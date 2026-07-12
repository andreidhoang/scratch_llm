# Frontier Hiring Map — mastery ⇒ the hireable artifact (wired into the 89-Bài curriculum)

> **What this is.** The single curriculum-facing map that ties every Bài of mastery to the **frontier-RE
> hiring signal**: which universal interview gate it earns, which Frontier Operating Principle (FOP) trait
> it demonstrates, whether it is **build-from-scratch** or **know-it-discuss**, and which scarce-2026 skill
> bucket it fills. Mastering the codebase from first principles (derive → run → re-derive cold) **is** the
> act of building the portfolio that gets hired — this doc makes that linkage explicit and durable, so PRR
> step 6 ("Frontier") is never hand-wavy.
>
> **Source lineage (correct to the last bit).** Original `FEINBERG_INTERVIEW_MAP.md` (distilled from a
> **Vlad Feinberg / GDM** interview) → archived to `_archive/`; its FOP content extracted into
> [`../../CLAUDE.md`](../../CLAUDE.md) §Frontier Operating Principles. The career verdict + trait scorecard
> live in [`../frontier-research/STRATEGY.md`](../frontier-research/STRATEGY.md); the interview gates +
> role→artifact map + portfolio signal in [`../../../README.md`](../../../README.md) §"CS336 → interview
> readiness"; the per-pillar build-vs-know-it doctrine in
> [`FRONTIER_PRACTICE_2026.md`](../FRONTIER_PRACTICE_2026.md). This map is the curriculum join of those four.
>
> **The throughline (`CLAUDE.md §FOP`):** *frontier labs hire on shipped, defensible artifacts — **taste is
> necessary, execution is the gate.***

---

## 1. The Operation — the 7 Frontier Operating Principles (how a frontier RE works)

Verbatim-faithful to `CLAUDE.md §FOP`. These are the *character*, not a checklist:

1. **Execution > analysis.** Ship beats plan. **No new doc without a same-day commit hash.** "Close a
   node, then delete the urge to write the next doc." When the doc-to-code ratio climbs, stop writing and
   resolve a stochastic node.
2. **Spec-with-falsifiers.** Before non-trivial code, state **pre-registered, falsifiable predictions +
   pre-committed kill/abandon thresholds** (the DELTA P1–P7 shape).
3. **Roofline-first / predict-the-number.** Predict the bound (comm vs flop vs memory) **and the number
   before the run**. A kernel's DoD is a **profile, not a green test**. Name the next unmodeled constraint
   others miss (launch / latch / occupancy / bank-conflict).
4. **Claims honesty.** Label `[FACT]` / `[INFERENCE]` / `[UNCERTAIN]`. **"Implemented" ≠ "measured"** —
   only a measured/profiled run is a result. Verify against primary sources; never overclaim.
5. **Research-as-MDP / taste.** **One active experiment at a time.** Pull the high-variance signal node,
   not deterministic scaffolding. Ruthless kill criteria. **Subtract-before-add.**
6. **AI-mode boundary.** Know which work is delegate-plumbing (Mode 1) vs human-leads-AI-assists (Mode 2)
   vs AI-OFF reps the interview tests (Mode 3).
7. **Citation-tree mastery.** Traverse to the non-redundant gap; **reuse before re-deriving; don't rebuild
   owned work** (the ✅=OWNED rule; PROGRESS.md).

**PRR is FOP applied to learning:** predict-cold (FOP-2/3), run the real number (FOP-3/4), reconcile the
gap, subtract-before-add (FOP-5), never re-derive owned Bài (FOP-7).

---

## 2. The Character scorecard — traits a committee screens for (`STRATEGY.md §3`)

Each trait, its evidence, and the grade the repo self-assigns. **The two A+ traits are the hardest to
fake and the loudest positive signal.**

| Trait screened for | Evidence it wants | Grade | Earned in |
|---|---|---|---|
| **Spec before code** (hypothesis → falsifiable numeric predictions → kill criteria) | DELTA RFC P1–P7 w/ explicit falsifiers | **A+** ("most candidates never write this") | Stage 8 / every predict-before-run |
| **Epistemic honesty** | `[FACT]/[INFERENCE]/[UNCERTAIN]`, flag unverified, self-correct | **A+** ("*this is* the forensic-honesty signal") | every Bài's honesty label (FOP-4) |
| **Lock evals before architecture** | correctness oracle + roofline harness *before* kernel opt | A | S3 Bài 3.0, M10 Bài 10.2 |
| **Overfit/sanity gates** | loss-at-init ≈ log V; overfit-one-batch; 512-step drift | A | M1/M3 (log V), M4 (overfit) |
| **Mandatory RL logging** | 3 KLs separately + KL(train‖infer) w/ 0.10 HALT, IS/ESS, reward+length | **A** ("scarce skill shipped") | M8 Bài 8.7 |
| **Ablate one variable, kill fast, document negatives** | a failed prediction = a *publishable negative* | A | M10 Bài 10.4 (F1–F9) |
| **GPU perf discipline** | profile/roofline first, lock clocks, **measured not theoretical peak** | A | S3–S4, S1 Bài 1.0 |
| **Research-as-stochastic-MDP / taste** | EV bet on a stochastic DAG; subtract-before-add | A | M6, M10, every "build-vs-know-it" call |

The recurring **senior tell**: *owning the toggle* — e.g. knowing the KL k1/k2/k3 estimators cold **and
when to turn KL off** (GRPO/DAPO drop it for pure-RLVR; RLHF-with-reward-model keeps it).

---

## 3. The axis you are actually hired on (`STRATEGY.md §1` — the verdict)

> **"Planning, design and direction are frontier-grade; the gap to an offer is *execution*, not taste."**
> Planning axis ≈ **90%** aligned; **execution axis ≈ 25% — and that is the axis a committee hires on.**

The named failure mode is the **doc-to-code ratio** (a **5.3:1** md:py byte ratio at the time of the
verdict; "code froze, days produced only strategy documents"). One-liner: *"you are planning like a
frontier lab; you are not yet shipping like one."* This is the **scaffolding treadmill** — and it is why
FOP-1 is *Execution > analysis*, and why the PRR loop insists every micro-concept **touch a runnable
number**, not a paragraph.

**The portfolio signal that gets you hired (`README.md`):** *a public GitHub repo with clean engineering —
**green CI, tests, design docs** — that you can explain from first principles.* That repo is `scratch_llm/`.
A job is landed on **shipped, defensible, differentiating artifacts** — and the two *differentiators* (the
**A5 RL "aha"** and the **DELTA decode kernel**) are the scarce, hard, high-EV nodes.

**GDM entry route (`STRATEGY.md §5.3`, cheapest highest-signal):** Feinberg explicitly invited candidates
to do the **Scaling-Book handwritten exercises + a transformer-from-scratch exercise + a recorded video
walkthrough** — *"GDM routinely interviews and advances candidates who submit these"* — and it **bypasses
résumé screens**. `model.py` already *is* the transformer exercise; the mastery you build here is the
walkthrough content.

---

## 4. The universal interview gates → where each is earned (`README.md §gates`)

What you must do **cold**, mapped to the curriculum stage that earns it:

| Universal interview gate | Curriculum stage / Bài |
|---|---|
| **Implement MHA / a full Transformer from scratch in ~45 min** | Stage 1 · M1–M2 (A1 substrate) |
| **"How would you train a 100B+ model?"** (DP+TP+PP + the ~16–20 B/param memory math) | Stage 3 · M5 + S6 (A2) |
| **Derive / explain scaling laws; compute-optimal N and D** | Stage 4 · M6 (A3) |
| **RLHF vs DPO; what makes RL stable; GRPO group-relative advantage** | Stage 5 · M8 (A5) |
| **Why is batch-1 decode memory-bound? Fuse a linear-attn decode recurrence; what would falsify it?** | Stage 2 + Stage 7 · S1/S3/S4 + DELTA capstone |

---

## 5. Role → the artifacts that carry it (`README.md`)

| Target role | Carry artifacts |
|---|---|
| **RL / Post-Training RE** | A5 (Stage 5 · M8) + A1, A4 |
| **Inference / Systems RE** | A2 + DELTA (Stage 2 · S3/S4 + Stage 7 · S1/S2/S5) + the vLLM/SGLang/TRT-LLM/Dynamo OSS-PR lane |
| **Pretraining / Core Modeling RE** | A1 + A2 + A3 (Stages 1, 3, 4) |
| **Data RE** | A4 (Stage 4 · M7) + A3 |

---

## 6. Scarce-2026 skill buckets → curriculum location (`README.md §readiness`)

The 2026 bottleneck moved **off** "can you train a transformer" (commoditized) **onto ① post-training**
(stable RL × a real reward) and **② inference/systems** (serve it economically). Those two are the
differentiators; everything else is table-stakes you must still own cold.

| Scarce skill | Where built | Tier |
|---|---|---|
| RL post-training (GRPO/Dr.GRPO, reward modeling, RLHF vs DPO) | **A5 / M8** | ★ differentiator |
| Inference & serving (KV-cache, quant, train/inference drift) | **A2 + DELTA / S1·S2·S5** | ★ differentiator |
| GPU kernels (Triton, FlashAttention-class) | **A2 + DELTA / S3·S4** | ★ differentiator |
| Distributed training (DDP/ZeRO/FSDP, comms overlap) | **A2 / M5·S6** | table-stakes |
| MoE (routing, aux-loss-free balancing) | **A1 / M9** | table-stakes |
| Data curation (filtering, dedup, quality) | **A4 / M7** | table-stakes |
| Scaling laws (IsoFLOP, compute-optimal) | **A3 / M6** | table-stakes |

---

## 7. Build-vs-know-it doctrine (`FRONTIER_PRACTICE_2026.md`)

A senior tell is **knowing what NOT to build from scratch**. These are `⚪ know-it — discuss in interviews,
do NOT reimplement` (each has a real reason it's hardware/scale-gated). Own the *framing + tradeoff*, not
the code:

- **MLA** — DeepSeek(V2/V3)/Kimi KV-compression bet; **GQA is the broad default** (Llama4, Qwen3, Gemma3,
  gpt-oss, OLMo2). Tradeoff: extra up-proj compute + fiddly decoupled-RoPE vs a much smaller cache.
- **Long-context RoPE scaling** (YaRN/NTK) — fine-tune/serve-time concern on the RoPE you already own.
- **FA3 / FA4** — Hopper/Blackwell-gated; async warpgroup pipelining + TMA + low-precision. Target
  awareness for the roofline discussion; from-scratch deliverable correctly targets **FA2**.
- **FP8 training** — Transformer-Engine's job (amax bookkeeping, current-vs-delayed scaling, per-block
  outliers). DeepSeek-V3 trains FP8; Blackwell adds FP4. Discuss, don't reimplement.
- **PD disaggregation** — default in vLLM/SGLang/TRT-LLM/Dynamo by mid-2025; multi-node infra, not a
  learning module. The rollout seam (M8.7) is the on-ramp to *discuss* it.
- **Precision / distillation / MoE scaling laws** — need real GPU grids / many trainings; whiteboard the
  allocation tradeoff, point at where the ladder *would* attach.
- **Data-mixing (DoReMi), annealing/high-quality decay, synthetic-data-at-scale** — value appears only at
  real scale; build the small measurement-loop version, understand the automated one.
- **Adopt (design choice, same effort):** FSDP2 per-parameter DTensor sharding (`fully_shard`) — legacy
  FlatParameter is out; per-param sharding composes with TP/PP, ~7% less mem / ~1.5% more throughput.

---

## 8. How this wires into the PRR loop — every Bài's step-6 "Frontier"

When you close a Bài (PRR step 6), name **four things** out loud — this is the hiring linkage, not a vibe:

1. **Gate** — which universal interview gate (§4) this Bài earns a piece of.
2. **Trait** — which FOP character trait (§2) the way you did it demonstrates (e.g. predicted the number
   first = FOP-3; labelled the honesty status = FOP-4).
3. **Build-or-know-it** — is this a from-scratch deliverable or a §7 know-it? If know-it, state the
   *tradeoff framing* an interviewer wants, not code.
4. **Scarce bucket** — differentiator (RL / inference / kernels) or table-stakes (§6).

The `Neo đã defend` anchor you write into `PROGRESS.md` on a ✅ **should include the gate/trait it earns**,
so the ledger doubles as an interview-readiness map. Blank-slating ONE function to re-derive cold (PRR
faded-scaffold) **is** the "implement from scratch in 45 min" gate rehearsed.

---

## 9. Hiring signal per stage (the curriculum join — correct to the last bit)

| Stage · Series | Gate earned (§4) | FOP trait foregrounded (§2) | Tier (§6) | Build / know-it |
|---|---|---|---|---|
| **1 · M1–M4** tokenizer→transformer→optim→train | *MHA/Transformer from scratch in 45 min* | overfit/sanity gates (loss-at-init≈log V, overfit-one-batch); spec-with-falsifiers | table-stakes (pretrain/core) | **build** |
| **2 · S3–S4** CUDA-core + tensor/Flash | *why batch-1 decode memory-bound; roofline* | roofline-first / predict-the-number; GPU perf discipline (measured≠theoretical) | ★ kernels | **build** FA2; know-it FA3/FA4 |
| **3 · M5 + S6** distributed | *train a 100B model: DP+TP+PP + memory math* | comms discipline; lock evals before arch | table-stakes | **build** gloo-correct; adopt FSDP2 |
| **4 · M6 + M7** scaling + data | *derive scaling laws; compute-optimal N,D* | extrapolation honesty (a+b=1 lie-detector); data-judgment (dedup scope) | table-stakes | **build** core; know-it DoReMi/anneal |
| **5 · M8** post-training / RL | *RLHF vs DPO; RL stability; GRPO advantage* | **mandatory RL logging**; owning the KL toggle | **★ differentiator** | **build** |
| **6 · M9** MoE·MLA·MTP | *MLA vs GQA tradeoff; aux-loss-free balancing* | convergent-defaults reasoning | table-stakes | **build** MoE/MTP; **know-it MLA** |
| **7 · S1–S2 + S5** serving + quant | *KV-cache; quantization; train/inference drift* | inference co-design; TCO≈electricity | **★ differentiator** | **build** + know-it FP8/PD-disagg |
| **8 · M10** close-the-loop + F1–F9 | *what would falsify your result?* (capstone) | **spec-with-falsifiers (A+)**; ablate-one-variable; **epistemic honesty (A+)**; research-as-MDP | ★ the differentiator artifact | **build** |

---

*Maintenance: this map is derived, not source. If a figure here disagrees with `STRATEGY.md` / `README.md`
/ `CLAUDE.md` / `FRONTIER_PRACTICE_2026.md`, **those win** — fix this map (or it's a review finding). Re-verify
version-sensitive numbers before quoting live (per STRATEGY.md caveat).*
