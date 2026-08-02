# Frontier 2026 — End-to-End Training Plan (the integrated spine)

> **Doc role.** This file owns the **integrated pipeline view** (S0→S8): data, architecture,
> pretraining, scaling-law calibration, distributed pretrain, midtrain, SFT, RL, serving/eval. For
> ablation strategy and rung cards see [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md); for
> the buildable file→test spec see [`FRONTIER_2026_TASKSPEC.md`](FRONTIER_2026_TASKSPEC.md); for a
> one-page status board see [`FRONTIER_STATUS.md`](FRONTIER_STATUS.md); for the curated entry point
> see [`FRONTIER_2026_MASTER_PLAN.md`](FRONTIER_2026_MASTER_PLAN.md); for the K3 track (chartered
> 2026-07-31) see [`k3/ROADMAP.md`](k3/ROADMAP.md) + [`k3/FACTS.md`](k3/FACTS.md) (claim ledger).

> **What this is.** The refactored **end-to-end training plan** for `scratch_llm`: one document that
> walks the whole pipeline — data → architecture → pretraining → scaling-law calibration →
> distributed pretrain → midtrain → SFT → RL → serving/eval — and, at each stage, grounds (a) what
> we build *in this repo* (exact files), (b) the first-principles rationale a frontier RE must be
> able to derive, (c) the mid-2026 state of the art with citations, (d) what is settled vs
> contested, (e) the measurable gate/kill criterion, (f) status. It **supersedes nothing**:
> [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) remains the ablation strategy + rung
> cards, [`FRONTIER_2026_TASKSPEC.md`](FRONTIER_2026_TASKSPEC.md) remains the buildable rung DAG,
> `deploy/runbooks/d20_speedrun_8xH100.md` remains the paid-run script. This doc integrates them
> into one pipeline view and folds in the 2026-07-30 research pass.
>
> **Provenance.** Written 2026-07-30 from (i) a 9-angle adversarial research swarm (optimizers ·
> attention/KV · MoE · data · post-training · precision/stability · eval · MTP/speculative ·
> nanochat ground-truth; the infra angle failed on quota — infra here is sourced from the repo's
> GPU-verified runbook + taskspec instead), and (ii) repo ground truth at HEAD, incl. two commits
> landed today: **`3b89119`** (F2a MTP train head) and **`98f62e3`** (F6 MoE balancing harness).
>
> **Claims-honesty convention (FOP-4, extended with the swarm's labels).**
> - `[VERIFIED]` — primary-source verified in the research pass (citation given).
> - `[REPORTED]` — credible secondary source; not fetched primary. Treat with care.
> - `[UNCERTAIN]` — unresolved or conflicting evidence. Stays uncertain until measured.
> - `[MEASURED]` — a number off a box, logged in `bench/RESULTS.md` or `docs/RESULTS.md`.
> - `[PRE-REGISTERED PREDICTION]` — a falsifiable number committed *before* the run.
> - `[INFERENCE]` — reasoning from verified premises; not itself measured.
> A repo claim always carries a file path or commit hash. A 2026-research claim always carries a
> URL or arXiv ID. If it has neither, it is not a claim — it is decoration, and it does not ship.

> **2026-07-31 external review & verification pass.** An external "lead engineer brief" restating
> this plan was reviewed adversarially against (i) repo ground truth and (ii) re-fetched primary
> sources. Verdict: **plan upheld** — the brief restates this doc set (d20 spec, tiers, rungs,
> contingent-cost path are all already here); its genuine errors are corrected below and folded
> back into the ledger docs.
>
> *Verified / updated 2026 claims (primary sources re-fetched 2026-07-31):*
> - **DeepSeek V4 exists** (released 2026-04-24; V4-Pro 1.6T/49B-active, V4-Flash 284B/13B-active;
>   [model card](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf),
>   [2606.19348](https://arxiv.org/abs/2606.19348)): **CSA+HCA hybrid attention; MLA's KV-latent
>   compression dropped** (low-rank q/o projections survive). "Frontier attention is contested"
>   stands. `[VERIFIED]`
> - **GLM-5.2 IndexShare** = cross-layer reuse of DSA's lightning indexer (1 indexer / 4 sparse
>   layers, −2.9× per-token FLOPs @1M), not a new attention family
>   ([z.ai/blog/glm-5.2](https://z.ai/blog/glm-5.2)). `[VERIFIED]`
> - **Kimi Linear**: 3 KDA : 1 MLA hybrid; "outperforms full attention under fair comparison" is a
>   vendor preprint claim, no independent replication
>   ([2510.26692](https://arxiv.org/abs/2510.26692)). `[VERIFIED as claim]`
> - **Qwen3-Next**: exactly 3:1 Gated-DeltaNet : gated-full layout (HF model card). `[VERIFIED]`
> - **Muon deflation is scale-dependent**: 1.4× @0.1B → 1.1× @1.2B under equal tuning
>   ([2509.02046](https://arxiv.org/abs/2509.02046)); 10–15% fewer tokens @100M–4B
>   ([2505.02222](https://arxiv.org/abs/2505.02222)); ~20–30% recoverable at 1.2B via
>   Hyperball-style norm control ([2606.16899](https://arxiv.org/abs/2606.16899)). ⇒ Muon+AdamW
>   baseline decision upheld; expected edge at d20 scale ≈1.2–1.3×, not the headline 2×.
> - **Spurious Rewards is now ICML 2026** ([2506.10947](https://arxiv.org/abs/2506.10947)) — but
>   the naive reading is weakened by peer-reviewed counter-evidence: contamination (AAAI-26,
>   [paper 40687](https://ojs.aaai.org/index.php/AAAI/article/view/40687/44648)) and an
>   "Anchor-Adapter" memorization circuit ([2601.11061](https://arxiv.org/abs/2601.11061)). ⇒ F7
>   gains: contamination probe (partial-prompt completion) + ≥1 non-Qwen-family control +
>   budget-matched eval, per the emerging minimum RLVR standard
>   ([2509.21882](https://arxiv.org/abs/2509.21882)).
> - The "Wang et al." recall paper = **Dustin Wang et al., [2507.06457](https://arxiv.org/abs/2507.06457)**
>   (*A Systematic Analysis of Hybrid Linear Attention*; preprint): LM loss flat across linear:full
>   ratios while **RULER** recall collapses at high ratios; ~3:1 approaches Transformer recall.
>   Probes: RULER (primary) / MQAR (Arora [2312.04927](https://arxiv.org/abs/2312.04927)) / NIAH —
>   MQAR is *not* from that paper. ⇒ F8/F10 gates keep a mandatory recall probe, RULER-first; the
>   3:1 F10 design point is supported.
>
> *Corrections to the reviewed brief (do NOT propagate):*
> 1. **Its §"what F6 taught us" is void.** `artifacts/f6_moe_ablation/` is a 30-step, d=32 synthetic
>    smoke run (commit `98f62e3`) — harness plumbing only. The coarse-vs-fine "takeaways" are not
>    evidence; real F6 needs ≥1B real-token arms. Logged as smoke in `docs/RESULTS.md` §F6.
> 2. ClimbMix −27% is a **wall-clock speedrun** confounded with the d26→d24 model shrink (nanochat
>    LOG 2026-03-04), not iso-FLOP; the paper's ClimbMix-vs-FineWeb-EDU comparison is **1B @ equal
>    token budget**, not 35–500M. F12 at 35M/700M is a *novel* measurement — cite it that way.
> 3. NVFP4 residual gaps are documented at **8B/12B** (~1–1.5% rel. loss vs FP8/BF16,
>    [2509.25149](https://arxiv.org/abs/2509.25149)); small scale (1.2B) is *more* forgiving, not
>    less. B200 is **sm_100**; sm_120 is consumer Blackwell (RTX 5090 class).
> 4. "MiniMax retreated to full attention" is **stale**: true for M2 (Nov 2025), reversed in M2.5
>    (7:1 hybrid, Feb 2026) and M3 (**MSA** sparse attention,
>    [2606.13392](https://arxiv.org/abs/2606.13392), Jun 2026). The contested-attention thesis is
>    *stronger*, not weaker.
> 5. The brief's "AdamW baseline" is pre-pivot: the d20 optimizer is **Muon+AdamW** (commit
>    `f9e8f3b`; F1-run descoped 07-17).
>
> *Adopted from the review:* F7 contamination probe + non-Qwen control; F8/F10 recall probe fixed
> to RULER-primary. Ordering unchanged — **next: S3 fit-gate → P5 d12 ($10–15) → 8×H100
> d20 (gated)**.

---

## §1 — The north star

**Own every layer of a language model, close the loop end-to-end into a talking d20 with an honest
public report card, and use that working baseline to run pre-registered, iso-FLOP ablations that
test the 2026 frontier's *contested* claims.** The model is the artifact; the *discipline* —
pre-registration, independently tuned baselines, adversarial controls, kill criteria phrased as
falsifiable sentences, negative results published next to positive ones — is the hireable artifact.
The 2026 evidence keeps proving this is the right bet: the year's biggest optimizer and RL findings
were **methodology corrections** (Muon deflated by a tuned AdamW baseline,
[2509.02046](https://arxiv.org/abs/2509.02046); the RL "aha" deflated by a random-reward control,
[2506.10947](https://arxiv.org/abs/2506.10947)). Pre-registration itself is becoming a field norm
([2606.11217](https://arxiv.org/html/2606.11217v1)). The one-line thesis: *"I measured the frontier
claims honestly, and three of them did not survive a tuned baseline or a proper control."*

---

## §2 — The stage-by-stage pipeline

### S0 — Data

**(a) In this repo.** `data/shards.py` (A1 ✅: memmap uint16/uint32 shards, staged tokenizer,
FineWeb-EDU slice, parquet bulk path `--fineweb-parquet --target-tokens 1e10` proven at 0.7B
sub-hour `[MEASURED 2026-07-17]`); `data/decontaminate.py` (A0 ✅: 13-gram gate + shard
`doc_filter` seam); `tokenizer.py` (byte-level BPE, GPT-4-style, vocab 32,768); `data/pipeline.py`
(canonical filter order + discard accounting); shard-backed `speedrun --data-dir`. Corpus +
calibration banked on stopped vast box 43676999 (taskspec Next-node marker).

**(b) First principles.** Tokens are the D in `C = 6ND`; quality filtering moves the constant, not
the exponent — which is why data decisions are *bigger* than optimizer decisions at fixed compute
(see ClimbMix below). Decontamination is not hygiene, it is the difference between measuring
generalization and measuring memorization: any n-gram overlap between train and the CORE/GSM8K
eval sets inflates every downstream number the program pre-registers. bpb (bits-per-byte) is the
only honest cross-vocab loss: CE-per-token shrinks mechanically as vocab grows, so all corpus and
tokenizer comparisons must be in bpb on a pinned val set ([VERIFIED] — Karpathy's framing,
[nanochat discussion #1](https://github.com/karpathy/nanochat/discussions/1); formalized as
BPB = Σℓᵢ/(ln 2·Σbᵢ) in [2605.26797](https://arxiv.org/html/2605.26797v1)).

**(c) 2026 state of the art.**
- **Corpus:** the live result at our exact scale is nanochat's switch FineWeb-EDU-100B →
  **NVIDIA ClimbMix-400B** ([VERIFIED], commit
  [324e69c](https://github.com/karpathy/nanochat/commit/324e69c.patch), 2026-03-04): time-to-GPT-2
  **2h46m → 2h01m (−27%)**, val_bpb 0.7465 → 0.7185 — *"by far the single biggest improvement to
  nanochat's GPT-2 speedrun"*; five prior challengers (incl. DCLM/OLMo variants) all failed.
  ClimbMix = Nemotron-CLIMB ([2504.13161](https://arxiv.org/abs/2504.13161)): semantic clustering +
  mixture search; 1B model on 400B tokens beats Llama-3.2-1B by +2.0%
  ([HF card](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)). **License CC BY-NC 4.0** —
  must be stated in the A9 model card. The corpus wars are otherwise self-serving: DCLM
  ([2406.11794](https://arxiv.org/abs/2406.11794)), FineWeb ([2406.17557](https://arxiv.org/abs/2406.17557)),
  Nemotron-CC ([2412.02595](https://arxiv.org/abs/2412.02595), +5.6 MMLU over DCLM at 8B/1T) each
  win on their own eval.
- **Overtraining:** settled and quantified. Loss + downstream scale reliably into overtraining
  (Gadre et al., [2403.08540](https://arxiv.org/abs/2403.08540)); the inference-aware optimum is
  train-smaller-longer, quality improving to ~10,000 tok/param — but scaling laws fit at typical
  ratios *overestimate* token value at extreme ratios (Sardana & Frankle,
  [2401.00448](https://arxiv.org/abs/2401.00448)); ≤4 epochs of repetition ≈ fresh data
  ([2305.16264](https://arxiv.org/abs/2305.16264)). Frontier budgets: Llama-3 8B @ 15T (~1,875:1)
  [REPORTED]; Qwen3 @ 36T [REPORTED via [2505.09388](https://arxiv.org/abs/2505.09388)]; K2 @ 15.5T
  on 1T total = 15.5:1 [VERIFIED, [2507.20534](https://arxiv.org/abs/2507.20534)]; CLIMB 1B @ 400:1.
  **nanochat's own fit puts compute-optimal at ≈10.5:1 and its speedrun deliberately undertrains at
  8** ([VERIFIED], [runs/speedrun.sh](https://raw.githubusercontent.com/karpathy/nanochat/master/runs/speedrun.sh)).
  **Our d20 (480.4M @ 9.6B) is ratio-20 ≈ 2× overtrained by that fit** — a deliberate choice for a
  *served/demoed* artifact (Sardana's inference-aware argument), not Chinchilla folklore; it is
  re-registered as such here.
- **Tokenizer:** vocab **32,768 = 2¹⁵** confirmed at HEAD ([VERIFIED], commit
  [ccf4b7f9](https://github.com/karpathy/nanochat/commit/ccf4b7f9.patch)): rustbpe training +
  tiktoken inference, GPT-4 split modified `\p{N}{1,3}`→`\p{N}{1,2}`, 9 specials
  ([tokenizer.py](https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/tokenizer.py)).
  Vocab-scaling theory pushes *up* at scale ([2407.13623](https://arxiv.org/abs/2407.13623)); at
  ≤1B untied, the embedding share explodes (our d20: 84M/480M ≈ 17%; at 128k it would be ~52%), so
  32k is right for us. SuperBPE ([2503.13423](https://arxiv.org/abs/2503.13423)) is the strongest
  tokenizer-algorithm signal of 2025 — stretch spike, not core. **Retrain the BPE on the corpus you
  actually pretrain on** (changing corpus without retraining conflates the ablation).
- **Packing:** best-fit packing avoids truncation ([2404.10830](https://arxiv.org/abs/2404.10830));
  document-boundary-aware attention for packed sequences ([2407.09105](https://arxiv.org/html/2407.09105v1)).
- **Decontamination:** 10–13-gram is the standard gate; **the released FineWeb-EDU was not
  benchmark-decontaminated by its authors** [REPORTED]; **nanochat does no decontamination anywhere
  in speedrun.sh** [VERIFIED by absence] — our A0 gate is a *differentiator*. N-gram is provably
  insufficient vs paraphrase/translation ([2410.01560](https://arxiv.org/pdf/2410.01560v2));
  measured FineWeb 13-gram overlap ≈ **0.16% of docs** [REPORTED,
  [OpenReview](https://openreview.net/pdf?id=ufhH8YXxOB)]; "soft contamination" survives n-gram
  gates ([gleech.org](https://www.gleech.org/files/papers/soft-contamination)) — report the method
  + overlap rate, never claim purity.
- **Synthetic data:** collapse is a *replacement* phenomenon (Shumailov et al., Nature 2024);
  accumulation alongside real data is provably safe ([2404.01413](https://arxiv.org/abs/2404.01413)).

**(d) Settled vs contested.** *Settled:* model-based quality filtering > heuristics; dedup but not
aggressive global fuzzy dedup; decontaminate eval sets yourself; overtraining small models past
20:1 is standard and beneficial; bpb as the vocab-invariant metric. *Contested:* which filtered
corpus is best at ≤1B — now **self-measured**: our F12 at 35M/700M said the opposite of nanochat's
head-to-head (ClimbMix +0.110 bpb *worse* at iso-FLOP, `docs/RESULTS.md` §F12); the d20 corpus
(ClimbMix, operator override FINAL 2026-08-02) rests on nanochat's *larger-scale* result, and the d20
run itself becomes the corpus arbiter at our scale (RESULTS.md:120-127); optimal vocab
at ≤1B; staged-mixture curriculum vs one good static mix (ClimbMix wins *statically* at GPT-2
scale); reasoning-trace synthetic data at ≤1B (capacity floor unprobed).

**(e) Gate / kill.** A0 kill (pre-registered, runbook Step 1): >20% of docs flagged by
`--decontaminate` ⇒ an eval set leaked — investigate before training. Shard kill: any id ≥ 2¹⁶ ⇒
uint32 switch. Epochs ≤ 1.0 (9.6B / ~10B corpus ≈ 0.96). **NEW pre-registered decision (§3, F12):**
FineWeb-EDU vs ClimbMix at F1-harness scale decides the d20 corpus with *our* recipe — data was a
bigger nanochat win than Muon or FP8, and the program currently treats the corpus as fixed. That
is the cheapest large-win ablation we are not running.

**(f) Status.** A0/A1 shipped; P2 parquet path proven; **F12 DONE 2026-07-31** (`docs/RESULTS.md`
§F12): kill criterion triggered (ClimbMix bpb 1.30205 ≥ FineWeb-EDU 1.19197 at iso-FLOP, Δ=+0.1101),
measurement-only verdict = KEEP FineWeb-EDU; **operator OVERRIDE → ClimbMix anyway** (following
nanochat's larger-scale result), decision **FINAL 2026-08-02**. d20 corpus choice **settled =
ClimbMix**, already staged on the pod; the d20 run itself becomes the corpus arbiter at our scale.

**F12 corpus ablation.** Pre-registered head-to-head that decides the d20 corpus *with our recipe*:
**35M params / 700M tokens / iso-FLOP** (`C ≈ 1.5×10¹⁷`), tokenizer retrained **per corpus** on the
same byte budget, identical decontamination and packing. Primary metrics: **val_bpb + CORE** on a
shared held-out set. Arms: banked **FineWeb-EDU-100B** vs **NVIDIA ClimbMix-400B** ([VERIFIED],
[Nemotron-CLIMB 2504.13161](https://arxiv.org/abs/2504.13161); HF card
[nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix); nanochat
switch [324e69c](https://github.com/karpathy/nanochat/commit/324e69c.patch), 2026-03-04:
2h46m → 2h01m (−27%), val_bpb 0.7465 → 0.7185). **License CC BY-NC 4.0** — must be stated in the
A9 model card if ClimbMix wins. **Prediction:** ClimbMix bpb < FineWeb-EDU bpb, re-anchoring the
d20 CORE band if the corpus flips. **Kill:** ClimbMix ≤ FineWeb-EDU at iso-FLOP ⇒ keep the banked
FineWeb-EDU corpus.

---

### S1 — Architecture

**(a) In this repo.** `model.py`: RMSNorm (fp32) · RoPE (θ=1e4 default, `model.py:68`) · SwiGLU ·
GQA-ready MHA · opt-in QK-norm (`model.py:70,257-258`, before RoPE) · untied embeddings
(`model.py:69`, `tie_embeddings=False`; `speedrun.py:128`). d20 = `model_config_for_depth(20)`
(`speedrun.py:117-129`): d_model 1280, 20 layers, 10 heads, head_dim 128 → **480,431,360 params @
vocab 32768** `[FACT, instantiated at HEAD — runbook scorecard]`. `moe.py` (fine-grained + shared +
aux-loss-free bias, opt-in). `mla.py` (toy oracle). `dsa.py` (F8.1 ✅), `linear_attn.py` (F10.1
GatedDeltaNet ✅). `mtp.py` (F2a ✅ today, `3b89119`).

**(b) First principles.** Depth×64 aspect scaling keeps head_dim pinned at 128 (the size flash
kernels are tuned for) and buys capacity in the residual stream, where representation lives.
Untied embeddings matter because the input-embedding and output-head gradients want *different*
learning rates (see S2, Kalra). QK-norm bounds attention logits by construction: unit-RMS q,k over
head_dim ⇒ logit ≤ √d_head · (value scale) — with d_head=128 that caps logits at ~30-ish, which is
*why* QK-Clip is unnecessary below 1B (F9's pre-registered expectation). RoPE θ trades positional
resolution against extrapolation; at 2–4k context, θ∈[1e4,1e5] full-rotary and no scaling —
nothing to do, though nanochat uses 1e5 and we use 1e4 ([VERIFIED] attention-angle audit).

**(c) 2026 state of the art — the attention seam is *contested*, not settled.**
- **MLA is legacy.** DeepSeek itself abandoned it in V4 ([VERIFIED],
  [2606.19348](https://arxiv.org/abs/2606.19348), Apr-2026): CSA (4× sequence-compressed KV +
  DSA-style indexer) interleaved with HCA + 2 exact-window layers — 27% of per-token FLOPs, 10% of
  KV vs V3.2 @1M. Lineage: MLA (V2) → NSA (research-only) → DSA (V3.2,
  [2512.02556](https://arxiv.org/pdf/2512.02556)) → CSA/HCA (V4). GLM-5 ships DSA "lossless by
  construction" ([2602.15763](https://arxiv.org/html/2602.15763v2)); GLM-5.2 adds IndexShare
  ([2603.12201](https://arxiv.org/pdf/2603.12201)).
- **Linear-hybrid: real, parity-plus-efficiency, long-context-only.** Kimi Linear
  ([2510.26692](https://arxiv.org/abs/2510.26692)): KDA 3:1 : MLA, 75% KV cut, up to ~6× TPOT @1M (FACTS B9) — but
  the [repo](https://github.com/MoonshotAI/Kimi-Linear) shows the fair comparisons were 1.4T-token
  runs and **at 4k context the hybrid is the same speed as full attention**; speedups appear at
  128k+. The strongest negative result is official: MiniMax's M2 retreat
  ([minimax.io](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model)) — linear
  hybrid matched full attention on suites at dev scale, then showed *"clear deficits in complex,
  multi-hop reasoning at scale"*; linear state is precision-sensitive (hit in RL); and **their
  CPT-into-hybrid experiment failed — retrieval/induction heads form early, so hybrids must be
  trained from scratch**.
- **The 3:1 ratio has direct small-scale evidence.** Wang et al.
  ([2507.06457](https://arxiv.org/abs/2507.06457), 72 models at 340M/20B-tok and 1.3B/100B-tok):
  LM loss is *insensitive* to linear:full ratio, but **recall degrades sharply when full layers drop
  below ~1:3**; recommend GatedDeltaNet/HGRN-2 at 3:1–6:1. This is why val loss alone is blind to
  the hybrid failure mode — **a recall probe is mandatory in any F8/F10 falsifier**.
- **The block moved two generations past our GDN.** GDN-2 ([2605.22791](https://arxiv.org/abs/2605.22791),
  decoupled erase/write gates, beats GDN/KDA at 1.3B/100B — exactly our scale); EDA
  ([2606.26560](https://arxiv.org/abs/2606.26560), Qwen-team); Mamba-3
  ([2603.15569](https://arxiv.org/abs/2603.15569)). Our `linear_attn.py` ships GDN — the chunkwise
  machinery carries over (GDN-2 = GDN + one gate projection).
- **KV economics at our scale say attention efficiency buys ~nothing locally** [computed from
  verified configs]: the d20 (MHA) has KV = 20L×2×10×128×2B = **100 KiB/token** → ~205 MB/seq @2k
  vs ~0.96 GB bf16 weights; attention is ~10–15% of train FLOPs at 2k. MLA/GDN/DSA pay off only at
  batch serving or ≥16k. **Consequence: the d20 stays full attention; F8/F10 are science artifacts,
  and any "decode uplift ≤8k" is a predicted null.**
- **nanochat HEAD drifted far from the recipe we replicate** [VERIFIED],
  [gpt.py](https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/gpt.py) +
  [#481](https://github.com/karpathy/nanochat/discussions/481): ReLU² MLP, softcap 15, QK-norm
  after RoPE with ×1.2 sharpening, **sliding-window `SSSL`** (3 short : 1 full, final layer full),
  **value embeddings on alternating layers (44% of params at d24)**, per-layer residual scalars,
  smear gate + backout. Our `model.py` has no window pattern — an unowned divergence from the
  recipe whose published CORE curve we benchmark against. SSSL + value-embeds are the two cheap,
  additive candidates to close that gap before the paid run (value-embeds break param-matched
  comparisons — decide deliberately).
- **Stability stack converged:** QK-norm everywhere; sigmoid output gating post-SDPA
  ([2505.06708](https://arxiv.org/abs/2505.06708), NeurIPS'25 oral — Qwen3-Next/3.5, KAT, Step-3.7);
  partial RoPE (25–50%) is the 2026 majority at long context.

**(d) Settled vs contested.** *Settled:* QK-norm + output gating as table stakes; GQA as minimum
compression; hybrid > pure-linear with ~25% full layers the convergent knee; delta-rule the winning
linear family; hybrids trained from scratch; attention-efficiency wins only at long ctx/high batch.
*Contested:* whether linear hybrids beat full attention *at frontier scale* (Kimi yes-at-48B vs
MiniMax measured-retreat — both cannot be right at every scale); sparse/selection (DeepSeek, GLM,
MiniMax-M3) vs linear/state (Moonshot, Qwen, NVIDIA); optimal ratio and its scale-dependence;
**MLA's param-efficiency below ~1B — unpublished, likely poor** (its 512+64 latent was dimensioned
for d=7168).

**(e) Gate / kill.** Architecture freezes at P5. Falsifiable sentences: the d20 ships full-attention
unless an F8/F10 arm passes its quality-parity + recall-probe falsifier at iso-param on the
standing box; any attention variant that wins val loss but loses the MQAR/NIAH recall probe is a
KILL (Wang et al.'s decoupling). Value-embeds/SSSL enter the d20 only with a measured bpb win at
d12 scale in P5.

**(f) Status.** Dense d20 spec locked + instantiated. F2a MTP head **shipped today** (`3b89119`).
F8.1/F10.1 cores shipped; F8.2/F10.2 wiring pending (standing box). F5 MLA pending + downweighted
(substrate, pre-register both outcomes). SSSL/value-embeds adoption decision **open** — pre-P5.

---

### S2 — Pretraining

**(a) In this repo.** `optim.py` (AdamW β₂=0.95 + Muon NS5 + Moonlight RMS-match +
`CombinedOptimizer`, `optim.py:170,318`); `train.py` (memmap batches, bf16-autocast, NaN guard
`train.py:363-380`, `mtp_loss_weight=0.3` at `train.py:241`, aux term at `train.py:409`);
`scaling/isoflop.py` (iso-FLOP fitter, a=0.469/b=0.531, the a+b≈1 gate); `eval/optimizer_race.py`
+ `bench/optimizer_race.py` (F1 harness, shipped, descoped mid-flight — commit `f9e8f3b`).

**(b) First principles.** The pretraining compute identity is `C = 6ND`: 2 FLOPs/param/token for
the forward matmuls, 4 for backward (grad-acts + grad-weights each matching the forward). Power
laws are fit in log-log because loss vs {N, D, C} is scale-free over decades — a straight line in
log-log is the signature, and curvature is the signal something changed regime. AdamW's per-coord
adaptive LR is the right shape for embeddings/head/norms (sparse, heavy-tailed gradients); Muon's
claim is that hidden-matrix updates should be *orthogonalized* (Newton–Schulz of the momentum) so
every singular direction moves at unit rate — a full-rank orthogonalized update on an [A,B] matrix
has RMS **1/√max(A,B)**, so scaling by 0.2·√max(A,B) lands in AdamW's 0.2–0.4 band and lets you
reuse AdamW-like LRs (Moonlight 2502.16982, Lemma 1 — the repo's F1 card math, verifier-corrected
in `FRONTIER_2026_ABLATIONS.md` §8). µP matters because the optimal LR shifts with width; Kalra et
al. ([2605.21486](https://arxiv.org/abs/2605.21486)) [VERIFIED] show the AdamW-µP benefit comes
*overwhelmingly from maximizing the embedding-layer LR* — the concrete argument for per-tensor LR
groups with a large embedding LR, and a threat to any single-LR baseline.

**(c) 2026 state of the art — the optimizer fight is three-way live, not settled.**
- **Wen I** (*Fantastic Pretraining Optimizers I*,
  [2509.02046](https://arxiv.org/abs/2509.02046)) [VERIFIED]: the 1.4–2× Muon headlines came from
  under-tuned AdamW + mid-checkpoint eval; tuned per-optimizer LRs shrink the edge to **1.4×@0.1B →
  1.1×@1.2B**. **Qiu** ([2512.05620](https://arxiv.org/abs/2512.05620)) [VERIFIED]: with µP-scaled
  LR + independent wd ∝ 1/width, Muon/SOAP/Shampoo **sustain ~1.4× from 190M→1.4B** — the decay is
  an HP-transfer artifact. These two papers directly conflict; both are live.
- **Wen II / MuonH** ([2606.16899](https://arxiv.org/html/2606.16899v1)) [VERIFIED]: plain MuonW
  ≈10% @1.2B; MuonH sustains 20–30%, growing with horizon. **Xiao**
  ([2607.22444](https://arxiv.org/abs/2607.22444)) [VERIFIED]: MuonH's gain is *effective step-size
  evolution, not a better update direction* — careful scheduling captures most of it. Karpathy
  independently: Hyperball *"intriguing idea, didn't work out of the box"* ([VERIFIED],
  [#481](https://github.com/karpathy/nanochat/discussions/481)).
- **Production reality:** Muon is real at 1T scale (Kimi K2, MuonClip, 15.5T tokens, zero loss
  spike — [2507.20534](https://arxiv.org/abs/2507.20534)); Moonshot + Zhipu use Muon; **Qwen,
  DeepSeek, Llama, SmolLM3 still AdamW** [REPORTED]. Essential AI 2505.02222: Muon expands the
  compute-time Pareto to 4B and survives past critical batch size (supports our large global batch).
- **nanochat HEAD Muon is a different, evolved variant** [VERIFIED,
  [optim.py](https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/optim.py)]: Polar
  Express coefficients (arXiv 2505.16932) replacing Newton–Schulz + MuonEq (2603.28254) + Muon+
  (√min(m,n) renorm, 2602.21545) + NorMuon variance reduction (2510.05491) + cautious WD + aspect
  rule `×max(1,m/n)^0.5` — **not** Moonlight RMS-match. Our vanilla-NS Muon is a different variant
  family; their constants do not transfer 1:1 (`bench/RESULTS.md` §Decision). modded-nanogpt Track
  3 tuned per-group LRs ([leaderboard README](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/README.md)):
  embed Adam 0.7, head 0.004, 1D 0.015 (wd 0.001), Muon 0.025 (wd 0.05); *"the most sensitive
  hyperparameter is weight decay, then learning rate."*
- **Schedules:** linear decay-to-zero ≈ optimal under tuned peak LR (Bergsma et al., 2502.15938
  [VERIFIED via primary PDF]); WSD remains the multi-stage default; schedule-free underperforms WSD
  in LLM pretraining [REPORTED]; horizon-free WSqD ([2607.10959](https://arxiv.org/abs/2607.10959)).
- **Stability:** **softcap is dead, QK-norm won** — Gemma 3 replaced softcapping with QK-norm for
  accuracy *and* speed ([2503.19786](https://arxiv.org/html/2503.19786v1)); our `model.py` already
  has qk_norm and no softcap (grep-verified). MuonClip QK-Clip is a trillion-scale fix (K2's
  explosion was at 9B-active) — correctly gated OFF below 1B (F9). OLMo 2's stability recipe adds
  **z-loss 1e-4** ([2501.00656](https://arxiv.org/pdf/2501.00656)) — keep as the P5 contingency if
  spikes appear.
- **Precision:** FP8 accuracy-parity settled at frontier scale (V3 <0.25%), **but** late-emerging
  SwiGLU outlier spikes → FP8 divergence in long runs ([2409.12517](https://arxiv.org/pdf/2409.12517)),
  and **FP8 ROI at ≤1B is ~1.1–1.3×, not 1.3–1.5×** (torchao [2507.16099](https://arxiv.org/html/2507.16099v1);
  TE FP8 = 20–30% over bf16 at 561M *on Blackwell* — [nanochat #382](https://github.com/karpathy/nanochat/discussions/382)).
  **H100 has zero FP4 hardware** — the d20 gets nothing from FP4. NVFP4 stretch (sm120): the 12B/10T
  claim ([2509.25149](https://arxiv.org/abs/2509.25149)) is *mixed-precision* (first-2 + final-8
  layers + embeds stayed BF16, baseline was FP8); the only ≤2B replication shows a **0.6–0.9%
  residual gap at 1.3B** (CHON, [2602.02047](https://arxiv.org/abs/2602.02047)); NVFP4 > MXFP4 is
  now contested (Intel OAS, [2603.08713](https://arxiv.org/abs/2603.08713)).
- **What WE do at ≤1B:** Muon+AdamW **ADOPTED** (commit `f9e8f3b`, `bench/RESULTS.md` §Decision —
  FOP-7 reuse > re-measure), per-tensor LR groups, bf16+compile core recipe, qk_norm on, WSD-style
  decay; the Muon LR comes from the P5 sweep, not from nanochat's constants. MuonH: optional arm,
  pre-registered expectation ≈0 at our horizon (its evidence is 1.2B × 1–8× Chinchilla). Skip:
  schedule-free, SOAP/Shampoo standalone, Track-3 barnacles.

**(d) Settled vs contested.** *Settled:* independently LR-tuned baselines + end-of-training eval
are mandatory; the hybrid partition (matrices→Muon, embed/head/norms/scalars→Adam) is universal;
at ≤0.1B preconditioners beat tuned AdamW ~1.3–1.4×; QK-clip unnecessary sub-1B under qk_norm;
decay-to-low-LR is where the gain lives. *Contested:* whether Muon's edge decays with scale
intrinsically (Wen I) or is rescued by µP + 1/width-wd (Qiu); whether MuonH is mechanism or
schedule (Wen II vs Xiao); which Track-3 micro-techniques are load-bearing.

**(e) Gate / kill.** The F1 methodology debt (from the optimizer angle, load-bearing): the
registered race swept AdamW's LR then *reused the winner for Muon* via RMS-match — exactly the
cross-optimizer transfer Wen I falsifies. Before any future F1 number is quoted: independent Muon
LR sweep + wd sweep both arms, per-group Adam LRs (the tied embed/head blocks the Kalra fix — note
as confound or untie). Pretrain recipe gate (P5): bf16+compile re-validated on sm90; FP8 arm only
behind the train↔serve logit-KL gate with a recalibrated 1.1–1.3× band.

**(f) Status.** Optimizer adopted; harness shipped/descoped; F9 QK-clip guard shipped
(`apply_qk_clip`, observer) with its falsifier riding any instrumented run; bf16-eager ✅ sm120,
bf16+compile NaN is an sm120-inductor fact (retest on sm90 in P5); NVFP4 numerics study = stretch,
standing box.

---

### S3 — Scaling-law calibration (NEW — run before the $100)

> **Why this stage exists.** We currently anchor the d20's CORE band on nanochat's *published*
> curve and Chinchilla's 20:1 on folklore. Both are borrowed fits. nanochat's own fit says
> compute-optimal D:N ≈ 8–10.5 ([VERIFIED], [#420](https://github.com/karpathy/nanochat/discussions/420));
> our run is ratio-20. The optimizer angle adds that HP transfer (µP LR + 1/width-wd) is exactly
> what decides whether Muon's edge survives scale ([2512.05620](https://arxiv.org/abs/2512.05620)).
> A from-scratch lab that cannot fit its own scaling law on its own data/tokenizer is trusting
> someone else's constants with $100.

**(a) In this repo.** `scaling/isoflop.py` (the fitter + `check_exponent_sum` a+b≈1 gate —
CS336-A3, a=0.469/b=0.531 on the toy harness); `speedrun.py --depth` (one knob);
`eval/core_suite.py` + `bench/core_eval.py` (CORE at each point); `data/shards.py` (the real
corpus, not the toy); the standing sm120 box (free).

**(b) First principles.** Chinchilla parametric form `L(N,D) = E + A/N^α + B/D^β`; at iso-FLOP the
optima scale as `N_opt ∝ C^a`, `D_opt ∝ C^b`, and `C = 6ND` *forces* a+b = 1 — a fitted a+b off 1
means the grid was too narrow, the loss floor E leaked, or training was off-recipe. That is why
`scaling/isoflop.py:104-110` raises loudly. The bpb-vs-CORE divergence rule (from the eval angle):
bpb is the low-noise comparator; CORE's run-to-run spread at this scale is **±0.008–0.016**
(nanochat repeated one run 7× identically: CORE 0.2512–0.2677 — [VERIFIED],
[#481](https://github.com/karpathy/nanochat/discussions/481)) — so curve-fitting uses bpb, and any
CORE claim below the noise floor needs ≥3 seeds.

**(c) The concrete grid (standing box, ~2–3 GPU-days total, $0; drop s6/s8 to halve).**

| point | depth | N (approx) | D:N | D | C=6ND | est. wall @72TF/30% MFU |
|---|---|---|---|---|---|---|
| s1 | 4 | ~20M | 8 | 0.16B | 1.9e16 | ~15 min |
| s2 | 4 | ~20M | 20 | 0.4B | 4.8e16 | ~35 min |
| s3 | 4 | ~20M | 40 | 0.8B | 9.6e16 | ~75 min |
| s4 | 8 | ~59M | 8 | 0.47B | 1.7e17 | ~2.1 h |
| s5 | 8 | ~59M | 20 | 1.2B | 4.2e17 | ~5.5 h |
| s6 | 8 | ~59M | 40 | 2.4B | 8.5e17 | ~11 h |
| s7 | 12 | ~135M | 8 | 1.1B | 8.9e17 | ~11.5 h |
| s8 | 12 | ~135M | 20 | 2.7B | 2.2e18 | ~28 h — *drop if the budget trips; P5 gives a d12 anchor on H100 instead* |

Exact N from instantiation (`model_config_for_depth`); bf16, ctx 2048, Muon+AdamW, the adopted
recipe. Fit `L(N,D)` in bpb on the pinned val set; keep seeds fixed.

**(d) External oracle overlay.** Plot nanochat's published points on the same axes: miniseries d20
= 477M / 3.82B tok / CORE 0.1708 [VERIFIED, [#420](https://github.com/karpathy/nanochat/discussions/420)];
the leaderboard line (d24 ratio-8 FP8 ClimbMix, 1.65h, CORE 0.2626, val_bpb 0.718 —
[leaderboard](https://github.com/karpathy/nanochat#time-to-gpt-2-leaderboard)); GPT-2 XL anchor
CORE 0.256525; and nanochat's CORE-fit `CORE = 1 − 3.7555·FLOPs^−0.0344`. That fit at our d20's
C ≈ 2.77e19 predicts CORE ≈ **0.195** `[INFERENCE from the published fit]` — inside our
pre-registered 0.19–0.22 band; a useful independent cross-check that the band is sane.

**(e) Gates / kill (all falsifiable).**
- a+b ∈ [0.95, 1.05] (the `C=6ND` identity) — else the grid or recipe is broken, fix before spending.
- log-log fit R² ≥ 0.98 on bpb — else report "no clean power law at our scale" (itself a result).
- **D:N decision rule:** if the measured compute-optimal ratio is <15, re-register the d20's D away
  from 9.6B *before* P5, with the serving/overtraining argument written down (Sardana
  [2401.00448](https://arxiv.org/abs/2401.00448)) — decide, don't inherit.
- **Oracle-divergence gate:** our d12 bpb/CORE vs nanochat's published d12-class point at matched
  FLOPs: divergence > 0.02 CORE ⇒ a recipe gap (SSSL? value-embeds? Polar-Express Muon? tokenizer?)
  to close or explicitly accept before $100.
- **Kill:** if even s5–s6 (59M, ≤2.4B tokens) cannot beat the loss floor of the toy corpus by a
  clear margin, the data/recipe stack has a bug — stop, do not scale.

**(g) The d14/d16 mid-scale policy of record (2026-07-31, user-approved).** The largest free
calibration point is s8 (d12, C ≈ 2.2e18); the d20 is C ≈ 2.77e19 — a 12.6× extrapolation. A
**d14 (~193.6M, C ≈ 4.5e18)** or **d16 (~268.4M, C ≈ 8.6e18)** point cuts that to ~6×/~3.2×, but
they are a **trigger-gated verification rung, never a scheduled spend** (standing box ~57/109 h
free-but-slow; 1×H100 spot ~$8–12 / ~$15–25). Any one trigger fires ⇒ run; none ⇒ skip, $0:

| trigger | condition | response |
|---|---|---|
| **T1 fit failure** | S3 gates fail (a+b ∉ [0.95,1.05] or R² < 0.98) | run **d14** — cheapest grid extension before trusting any extrapolation |
| **T2 anchor divergence** | P5 d12 bpb diverges from the S3-fit prediction by > 0.01 bpb (or CORE > 0.02, the oracle-divergence gate) | run **d16** to localize where the curve breaks before the $100 |
| **T3 science promotion** | F8 (DSA) or F10.2 (GDN-2 3:1) beats full attention on bpb at sweep scale **and** passes the mandatory RULER recall probe | confirm at **d14** before the result enters the public report (small-scale rankings can flip) |
| none | — | d16 stays the $90-cumulative abort fallback only |

Mid-scale runs use the *frozen* d20 recipe (full attention, Muon+AdamW, F12-winning corpus) — they
are calibration points, not architecture bake-offs; the d20 architecture stays frozen regardless.

**(f) Status.** **Running** (RTX 5090 pod, ClimbMix corpus): s1–s4 banked, s5 in flight as of
2026-08-02. Blocks nothing that ships code; gates the D:N
re-registration and the P5 go/no-go.

---

### S4 — Distributed pretrain / the d20 gate

**(a) In this repo.** `utils/dist_train.py` (A7 ✅ 2026-07-18: optimizer-embedded ZeRO-2 —
`reduce_scatter` grads → owner-rank whole-matrix update → `all_gather` params; gloo-verified 12/12
incl. single-proc byte-identity + 3-rank oracle); `train.py` (per-rank data sharding `train.py:269`,
collective NaN guard `train.py:363-380`, grad-clip global norm by distributed reduction);
`DistMuonAdamW.consolidated_state_dict()` / `save_consolidated_checkpoint()` (A8 ✅ 07-19; launcher
wiring ✅ 07-28, commit `eec2b04`); `eval/core_suite.py` + `bench/core_eval.py` (P3 ✅ CORE 22-task
suite, byte-identical to nanochat `core.yaml`); `deploy/runbooks/d20_speedrun_8xH100.md` (345 lines,
the press-play script). Remaining entrypoint gaps G1/G2/G4 (runbook §0.5, `[FACT, verified at HEAD]`):
SDPA-off-in-training (`model.py:80` default False → ~107 GB retained scores at B=32 — the run
cannot physically execute), no torchrun launcher shim (`grep init_process_group src/` = 0 hits → 8
silent replicas), no memory-bounded 10B streamer (`data/shards.py:170` concatenates into RAM —
accept-with-verification at ≥64 GB free host RAM). All three land inside P5.

**(b) First principles.** ZeRO-2 with an optimizer-embedded shard: Muon's Newton–Schulz needs
*whole matrices*, so the sharding granularity is the matrix, never the element — classic
element-wise ZeRO-1 is structurally incompatible. Ring traffic for grads is `2(k−1)/k · payload`:
480.4M × 2 B bf16 = 0.96 GB → ≈4–7 ms vs a ~0.5 s step ⇒ comm <1.5% wall, 8-GPU scaling ≥97%
(pre-registered in the runbook). DP is chosen on the roofline: DP moves ~3.5 B/param/step vs
6·B_gpu FLOPs/param/step of compute — ~51× headroom over the 989 TF / 450 GB·s machine balance;
TP adds 4 unoverlapped all-reduces/layer while shrinking GEMMs 8×; 1F1B bubble ≥6.5%; FSDP
all-gathers params to solve a memory problem that doesn't exist (~6–9 GB state vs 80 GB). The
silent-failure discipline: 8 ranks drawing *identical* batches (a global-seed `get_batch`) is an 8×
data loss that "completes" with a plausible loss — hence the tested per-rank seed offset.

**(c) 2026 state of the art** (the swarm's infra angle failed on quota; this is repo-verified +
nanochat-HEAD [VERIFIED]): nanochat does *not* use DDP — same optimizer-embedded ZeRO-2 shape,
3-phase async-overlapped, whole-matrix stacking ([optim.py](https://raw.githubusercontent.com/karpathy/nanochat/master/nanochat/optim.py));
auto batch size `B ∝ D^0.383` (Power-Lines 2505.13738) with `√(B/B_ref)` LR scaling — the P4
machinery; the GPT-2-grade artifact at HEAD is **d24/1.38B-total @ ratio 8 + FP8, 1.65h, CORE
0.2626, ~$48 nominal / ~$15 spot** ([leaderboard](https://github.com/karpathy/nanochat#time-to-gpt-2-leaderboard)) —
the frontier artifact got bigger *and* cheaper than the Oct-2025 d20 we replicate.

**(d) Settled vs contested.** *Settled:* data-parallel-first at 480M on one node; node-quality gate
(nccl busbw ≥350 GB/s or destroy + re-rent); preemption-resumable consolidated checkpoints; a loud
collective NaN kill-switch. *Contested:* nothing material at this scale — the open questions are
recipe-side (S2/S3), not parallelization.

**(e) Gate / kill (all pre-registered in `bench/RESULTS.md` §*$100 d20 run*).** P5 dress rehearsal
(d12, 1×H100, **$10–15**: LR sweep for the single Muon LR + compile-on-sm90 re-validation + CORE
vs a public checkpoint + ckpt kill/resume drill + the G1/G2/G4 entrypoints) must pass first. Then
8×H100 d20 (~$100): MFU 33–40% (KILL <28% sustained), step 0.48–0.58 s (KILL >0.70 s), comm <1.5%
(KILL >3%), scaling ≥97% (KILL <93%), loss-at-init ≈ log 32768 = 10.40 (kill on deviation), CORE
0.19–0.22 (KILL <0.15 ⇒ stack bug), **$90 cumulative ⇒ abort → downsize d16**. Honest framing
(locked): a CORE-vs-FLOPs point on nanochat's published curve — we buy 73% of the anchor's compute
at 86% of its N — **never** a depth-matched headline, and bpb not CE for any loss comparison.
Re-anchor caveat from the data/eval angles: the 0.19–0.22 band is calibrated to the Oct-2025
FineWeb-EDU recipe; the corpus **moved to ClimbMix** (F12 operator override, FINAL 2026-08-02), so
the band **must be re-anchored** against the *current*
published ClimbMix curve (d24-class 0.257–0.269 at ~4e19) **before P5** (RESULTS.md §F12, lines 120-127).

**(f) Status.** All buildable gate rungs **done** (A7, A8, P1 ✅ SDPA 07-16/GPU-measured 07-17,
P2 ✅ parquet 0.7B, P3 ✅ CORE suite, launch calibration B=16/18.2 GiB/0.283 s-step `[MEASURED
07-17]`). **Next: P5 ($10–15, rental, needs user go-ahead) → d20 ($100, gated on P5 + user
authorization).** Midtrain descoped from the paid run (`speedrun.py` raises `NotImplementedError`).

---

### S5 — Midtrain (A4)

**(a) In this repo.** Spec complete in `FRONTIER_2026_TASKSPEC.md` §A4: NEW `data/chat_adapters.py`
(`adapt_smoltalk` / `adapt_mc` — assistant = exactly the gold letter / `adapt_tool_use` /
`build_midtrain_shard` — **ids only, NO mask**: midtrain is full-CE continued pretraining on
chat-shaped data; oversize convos dropped whole + counted; deterministic seeded
`DEFAULT_MIDTRAIN_MIX`); `speedrun.py stage_midtrain` under the A2 stage-transition policy (fresh
optimizer + LR re-warmup); `midtrain_steps=0` ⇒ byte-identical skip.

**(b) First principles.** Midtrain is distributional bridging: continued pretraining shifts the base
toward the chat/MC/tool distribution *before* the SFT stage has to teach both format and behavior
at once. The timing×weight interaction ([2510.14865](https://arxiv.org/abs/2510.14865) [VERIFIED])
says late introduction can't be compensated by higher mixture weight — bake the chat/MC/tool data
in once, early enough, at modest weight.

**(c) 2026 state of the art.** **nanochat deleted midtraining** — #481: *"BOS-aligned dataloader…
Made midtraining unnecessary (deleted it)"*; the HEAD speedrun goes tokenizer → pretrain →
base_eval → chat_sft ([VERIFIED quote + script listing]). One swarm agent fetched a
`mid_train.py` at master (SmolTalk 460K + MMLU auxiliary_train 100K + GSM8K 8K, full-CE, mask
discarded) — `[UNCERTAIN: likely a stale/historical file; the #481 quote + current speedrun.sh are
the stronger evidence]`. The real midtrain anchors are OLMo 2/3 (**Dolmino 100B mid-train**,
[allenai.org/blog/olmo3](https://allenai.org/blog/olmo3); OLMo 2
[2501.00656](https://arxiv.org/abs/2501.00656): the annealing-phase mix "significantly improves"
downstream) and SmolLM3 ([huggingface.co/blog/smollm3](https://huggingface.co/blog/smollm3):
11.2T @ ~3,700:1, three-stage mixture with math/code upsampled in decay, then 100B long-context +
140B reasoning midtrain). The convergent cheap upgrade: **reserve the last ~10% of LR decay for an
annealing mix** (math/code/MC upsample). Reasoning-trace midtrain (R1-style) is unproven at 0.5B —
exploratory. nanochat's midtrain design note validates our `adapt_mc`: MMLU `auxiliary_train` is
format teaching (*"the model doesn't understand how Multiple Choice works"*), not the test split.

**(d) Settled vs contested.** *Settled:* full-CE (no mask) at midtrain; math/code upsample in decay
is the one thing every 2025+ recipe agrees on. *Contested:* whether a distinct midtrain stage adds
anything at 0.5B over one good static mix + annealing (all positive evidence is ≥3B; ClimbMix wins
*statically* at GPT-2 scale — if staged upsampling is flat at 0.5B, A4's complexity collapses into
the pretrain decay phase).

**(e) Gate / kill.** Kill (pre-registered): >30% of the mix exceeds `context_length` (can't include
without cutting a special) — shrink sources or raise ctx, never truncate through a special. Value
gate: A4 ships into the pipeline only if a midtrained d12 beats the base d12 on ChatCORE at
iso-compute in the dress-rehearsal environment; otherwise fold the annealing mix into S2 and skip
the stage (the nanochat-HEAD shape).

**(f) Status.** Spec complete; **unbuilt**; descoped from the paid run. CPU/standing-box buildable
(free).

---

### S6 — SFT (A5, shipped)

**(a) In this repo.** `algos/chat_sft.py` (✅ 2026-07-13: `collate_chat_batch` + `chat_sft_epoch`,
assistant-only masked CE, imports `algos/sft.py` unedited); `chat.py` (A3 ✅: specials +
`render_conversation` / `render_for_completion`); `chat_cli.py` (A6 ✅: `ChatSession` /
`batch_reply` over the public serving path — the loop TALKS through the chat template, greedy reply
reproduces the trained answer and stops at `<|eot|>`).

**(b) First principles.** Assistant-only masking is credit assignment: the loss should only shape
tokens the model is responsible for at serve time; masking user/tool tokens out of CE prevents the
model from spending capacity modeling the prompter. Mask off-by-one after the shift is the classic
silent bug — hence the pre-registered kill (overfit NLL plateaus >0.1 ⇒ mask bug).

**(c) 2026 state of the art.** nanochat's SFT ([VERIFIED], #481): SmolTalk 460K + MMLU
auxiliary_train ×3 epochs + GSM8K train ×4 (math + Python-REPL tool use), masked CE on assistant
spans, bestfit-**pad** packing (conversations never cropped), SFT inherits pretrain HPs,
warm-starts the Muon/AdamW momentum buffers with LRs reset (`init_lr_frac 0.8`), WD 0. Our
per-conversation shard stream + collate-time mask sidesteps packed-cross-contamination entirely
([2407.09105](https://arxiv.org/html/2407.09105v1)) — a feature to state in the model card. Mild
2026 signal: same-optimizer-family finetuning forgets less (*Optimizer-Model Consistency*,
2605.06654 [REPORTED]) — consider Muon-hybrid SFT as a cheap arm. MTP policy through post-training
(from the MTP angle): keep the MTP aux active through SFT (Nemotron-3-Super does, λ=0.3) — SFT
distribution shift silently kills draft acceptance otherwise; freeze/drop during RL (MiMo freezes
MTP in RL, [2505.07608](https://arxiv.org/html/2505.07608v2)).

**(d) Settled vs contested.** *Settled:* assistant-only masking; document-boundary-aware packing;
MC-format teaching via auxiliary_train; never truncate through a special. *Contested:* distill-first
vs SFT-only at ≤1B (the working ≤1B reasoning recipes distill then RL —
DeepScaleR/Phi-4-Mini-Reasoning [2504.21233]).

**(e) Gate / kill (pre-registered, taskspec §A5).** response_mask True exactly at assistant
positions + eot; masked loss-at-init ≈ log V; ≤200 steps overfit one chat batch to NLL<0.1 (KILL:
plateaus >0.1 ⇒ mask off-by-one); greedy completion decodes the trained assistant string and stops
at eot (KILL: never emits `<|eot|>`).

**(f) Status.** **Shipped** (CPU-green); the real-data SFT of the d20 is post-pretrain (rental
tail).

---

### S7 — RL / agentic

**(a) In this repo.** `algos/grpo.py` (GRPO/Dr.GRPO + clip + train loop, ADR-0017), `envs/{countdown,
gsm_math}.py`, `rewards/r1_zero.py` (verifiable grader), `utils/monitors.py` (entropy, both KLs,
IS-ratio + ESS, reward/length stats — the mandatory RL logging), plus the pending harness rungs:
F7a `eval/aha.py`, F7b aha-curve detector, F7c `rewards/neural_rm_control.py`, F11 `envs/tool_env.py`
+ multi-turn rollout + `rewards/tool_format_control.py` (all specced in the taskspec; CPU-green
first).

**(b) First principles.** Policy-gradient with a group-relative baseline (GRPO: advantage = reward
minus group mean, normalized) trades the variance of a learned value function for the bias of a
group baseline. The 2025–26 deflations are mechanism stories: GRPO's length/std normalization
inflates response length especially for wrong answers (Dr.GRPO), and its clipping asymmetrically
amplifies *pretraining priors* — which is why random rewards can produce reward↑ + length-growth
curves that look exactly like the "aha": the optimizer is sharpening what the base already
believed, not learning to reason.

**(c) 2026 state of the art.**
- **The "aha" debunk, three layers deep** [all VERIFIED]: Spurious Rewards
  ([2506.10947](https://arxiv.org/abs/2506.10947)) — random rewards recover **+21.4 of +29.1 pts**
  MATH-500 on Qwen2.5-Math-7B (clipping-bias prior amplification; *fails on Llama3/OLMo2*); the
  pass@k ceiling ([2504.13837](https://arxiv.org/abs/2504.13837)) — RLVR raises pass@1 but the
  **base model beats the RL model at large k** (RL sharpens, doesn't expand); contamination caveat
  (Wu et al., AAAI 2026 [REPORTED]) — part of the random-reward gain is reinforcing memorized
  trajectories. DeepSeek itself now *penalizes* length (V3.2,
  [2512.02556](https://arxiv.org/html/2512.02556v1)).
- **The toolbox:** Dr.GRPO ([2503.20783](https://arxiv.org/abs/2503.20783), drop length/std
  normalization); DAPO ([2503.14476](https://arxiv.org/abs/2503.14476), clip-higher 0.28/0.2
  anti-entropy-collapse + dynamic sampling + token-level loss + overlong shaping, **KL removed**);
  GSPO ([2507.18071](https://arxiv.org/abs/2507.18071), sequence-level IS — its headline win is
  stabilizing *MoE* RL; less load-bearing for a dense synchronous single-node run); ScaleRL (Meta,
  [2510.13786](https://arxiv.org/abs/2510.13786)): RL compute→performance is a *sigmoid* — fit it
  early, cap RL compute; the winning recipe (truncated-IS REINFORCE, FP32 logits, prompt-level
  averaging, zero-variance filtering) is mostly free at our scale. Off-policy honesty: train-vs-rollout
  backend mismatch breaks on-policy-ness even in bf16 — log IS ratios and mask/correct divergent
  sequences ([MiniMax](https://fengyao.notion.site/off-policy-rl); V3.2 institutionalizes the same).
- **nanochat's actual RL is REINFORCE-lite** [VERIFIED,
  [chat_rl.py](https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/chat_rl.py)]: no
  clip, no KL-to-ref, mean-baseline (r−μ), DAPO-style token-level normalization, GSM8K only —
  *"we are on policy, so there's no need for PPO ratio+clip."* Our clipped GRPO is *stronger* than
  the reference; note the spurious-rewards clipping analysis applies to ours, not theirs.
- **≤1B reality:** SimpleRL-Zoo ([2503.18892](https://arxiv.org/abs/2503.18892)) — zero-RL works
  0.5B→32B *on Qwen bases* because Qwen bases already carry instruction/self-reflection priors;
  format-reward tuning + query-difficulty control are the load-bearing knobs. "Reasoning Under 1
  Billion" ([2504.02273](https://arxiv.org/pdf/2504.02273)): ≤1B models hit reward sparsity and
  **reward mode collapse** — they overfit the easy format reward and ignore correctness. **At our
  scale the format-only control is not optional; it is the expected failure mode.**
- **Agentic is the labs' stated #1 priority** [VERIFIED across DeepSeek/Moonshot/Qwen]: V3.2
  post-training = specialist distillation → one *mixed* GRPO stage (reasoning+agent+alignment
  together, avoiding multi-stage forgetting), 1,800 synthesized envs; K2 Thinking
  ([kimi.com/blog/kimi-k2-thinking](https://www.kimi.com/blog/kimi-k2-thinking)) — 200–300
  sequential tool calls coherently. 2026 eval shifted chatbot→agentic — but **SWE-bench Verified
  was withdrawn by its own creator** ([openai.com](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/),
  Feb 2026: flawed tests, verbatim gold-patch recall; successor = SWE-bench Pro) — the plan's F11
  citation is stale; the direction is unaffected. Agentic reliability is **pass^k, not single-run**
  (τ-bench [2406.12045](https://arxiv.org/abs/2406.12045): GPT-4o pass⁸ <25% vs >50% single-run).

**(d) Settled vs contested.** *Settled:* GRPO's length bias is an artifact (drop length-growth as
an oracle); zero-variance/difficulty filtering is free signal; clip-higher beats entropy bonus;
rule-based > learned RM at small scale; log IS-ratio/KL/entropy every step. *Contested:* whether
RLVR *expands* reasoning (ProRL [2505.24864] vs the pass@k ceiling) — genuinely open; whether the
random-reward effect is clipping-bias vs contamination (likely both); KL-to-ref (DAPO/nanochat drop
it, V3.2 keeps it weak with the unbiased K3 estimator — no consensus); distill-then-RL vs zero-RL
per-FLOP at ≤1B.

**(e) Gate / kill (pre-registered).** F7: the random-reward control *recovers most of the gain* ⇒
the aha is spurious (the honest 2026 artifact); KILL = the control *fails to* reward-hack (then it
was real learning — re-baseline). **Additions from this pass:** pre-register pass@1 **and** pass@8
(the ceiling result), and the sharper discriminator — on our decontaminated FineWeb-EDU base (no
Qwen-style math priors) the random-reward control *should fail*; if it "works," we've rediscovered
contamination/prior-amplification. F11: verifiable-reward success-rate ↑ with turns-used bounded
while the format-only control stays flat (KILL: flat success under the true reward, OR the control
not caught by the logging — then monitoring is uninterpretable); report **pass^k + turns-used**
(τ-bench norm).

**(f) Status.** F7a/F7b/F7c/F11 harnesses: pending, **CPU-green by construction** (standing box,
free). Real RL runs need a ~0.5–1.5B base ⇒ **rental-gated** (post-d20 or a P5-class burst);
Countdown (calibrated difficulty) is the primary env; GSM8K zero-RL on a from-scratch 560M base is
the honest null hypothesis.

---

### S8 — Serving & eval

**(a) In this repo.** `eval/core_suite.py` + `bench/core_eval.py` (P3 ✅: DCLM CORE 22-task suite,
fidelity-verified byte-identical to nanochat `core.yaml`); `eval/report_card.py` + `speedrun.py`
report stages (`val_bpb`/MC/generative); `eval/spec_acceptance.py` + `bench/` CLI (F3 harness ✅);
`serving/` (perf-front: paged KV, continuous batching, speculative decode — measured but against
random-weight toys, the `bench/RESULTS.md:383` confound F3 exists to fix); `serving/speculative.py`
`Drafter` Protocol (the F2b seam, structural satisfaction from `mtp.py`).

**(b) First principles.** bpb normalizes per-token loss by bytes, making it tokenizer-invariant —
but it is still *corpus-dependent*: only comparable on a pinned val set. CORE's centered accuracy
(linearly rescaled per task so 0 = random, 1 = perfect) was explicitly designed for **low variance
at small scale** ([VERIFIED], [DCLM 2406.11794](https://arxiv.org/html/2406.11794v4)) — at ≤561M
raw MMLU/GSM8K sit at/near chance (nanochat d20: MMLU 31% vs 25% chance; GSM8K 2.5%), so centering
is what makes small models rankable at all. Speculative decoding is exactness-for-free: the target
verifies k drafted tokens in one forward, so speedup = acceptance × draft depth, and acceptance is
domain-stratified (code/structured ≫ open prose).

**(c) 2026 state of the art.**
- **Report-card norm:** CORE-22 + val_bpb for the base; ChatCORE (ARC-E/C, MMLU, GSM8K, HumanEval,
  centered vs baselines) per stage BASE/MID/SFT/RL ([VERIFIED],
  [chat_eval.py](https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/chat_eval.py)).
  HumanEval → EvalPlus/LiveCodeBench; MMLU saturated at frontier → MMLU-Pro [REPORTED]. OLMo 3 sets
  the honest-release bar: publish *which tasks carry signal at your scale* (task-clustering + SNR
  analysis, [allenai.org/blog/olmo3](https://allenai.org/blog/olmo3)). Marin gates ablations on
  Paloma macro held-out loss ([Marin retro](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/)).
- **CORE noise:** ±0.008–0.016 run-to-run (7 identical repeats: 0.2512–0.2677) [VERIFIED, #481] —
  all kill bands are re-tuned to this; bpb is the tie-break comparator. A dataset swap moved
  per-task scores but not CORE ([discussion #469](https://github.com/karpathy/nanochat/discussions/469)).
- **MTP/spec-decode serving reality (harsher than "free 1.8×"):** MTP merged into llama.cpp
  (2026-05); 260 controlled runs
  ([The Frontier Lab](https://thefrontierlab.ai/mtp-defaults-are-a-trap/) [REPORTED-engineering]):
  dense 27B α≈0.63 aggregate, sweet spot γ=2–3 → +61% (code +129%, prose +56–59%); MoE only +10%;
  old default γ=16 → −39%/−75%; SWA/GDN hybrids collapse to ~0.35–0.37 acceptance (state
  invalidation). ≥11 production families ship MTP heads by mid-2026 (V3, GLM-5/5.2, Qwen3-Next/3.5,
  Nemotron-3, Step-3.5, MiMo, MiniMax-M2, LongCat, **Gemma 4 down to E2B/E4B edge models**
  ([blog.google](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)),
  ERNIE 4.5). Sequential > parallel, decisively (the parallel "token salad" flaw; Medusa's fixes
  all went sequential — Hydra, ReDrafter). EAGLE-3 ([2503.01840](https://arxiv.org/abs/2503.01840))
  is the post-hoc upgrade path if baked-in MTP disappoints.
- **Serving quantization at ≤1B:** weight quant of a 480M bf16 (~1 GB) artifact is pointless; the
  levers are MTP draft + KV precision. FP8 KV is the safe lever; **4-bit PTQ is the worst case at
  ≤1B** ([2409.11055](https://arxiv.org/abs/2409.11055): small models suffer severe 4-bit drops;
  [2505.20276](https://arxiv.org/abs/2505.20276): up to 59% drop on long-context); NVFP4 serving
  works via QAD-distillation, not PTQ ([NVIDIA QAD](https://research.nvidia.com/labs/nemotron/nemotron-qad/));
  NVFP4 KV on sm120 is research-only (vLLM doesn't route to it as of 2026-07-18,
  [vllm#49011](https://github.com/vllm-project/vllm/issues/49011)).

**(d) Settled vs contested.** *Settled:* CORE-centered-22 as the small-scale base gate; bpb as the
cross-vocab metric; per-task raw + centered reporting; single-run agentic scores are meaningless;
4-bit PTQ dangerous at ≤1B; FP8 KV safe. *Contested:* CORE cross-harness comparability (nanochat
core.yaml ≠ llm-foundry ≠ lm-eval numbers — our byte-identical pin is the right anchor); bpb-vs-CORE
tie-break rule (no literature consensus — pre-register ours: bpb gates loss claims, CORE gates
capability claims, a divergence is reported not resolved).

**(e) Gate / kill.** F3 falsifier: on a trained ckpt (val_bpb ≪ log2 V), n-gram acceptance prose
<10% (predict 5–8%) while code/JSON stays 40–60%; KILL: prose ≥ code/JSON (still confounded).
F2b falsifier (recalibrated by this pass): MTPDrafter a2-acceptance ≥0.30 on open prose and
strictly > n-gram (≈0); γ pre-registered 1–2; **speedup prediction moved to code/JSON (+60–130%
realistic); open-prose speedup measured-not-predicted** (2026 field data: α≈0.6 at 27B dense,
lower at ≤1B). F10×F2b interaction: do not assume the MTP draft survives a hybrid backbone
(acceptance collapse + state-rewind bugs are the 2026 failure mode) — scope F2b to the dense model
or pre-register an acceptance measurement on the hybrid.

**(f) Status.** P3 + F3 harness shipped (measured falsifier awaits a trained ckpt); F2b pending
(post-d20, needs F2a-trained weights); the report card's per-stage BASE/MID/SFT/RL table ships with
the d20.

---

## §3 — The updated ablation program

> **Compact status summary.** This table gives the 2026-07-30 verdict, status, and falsifier for
> each rung. Full rung cards (hypothesis, prediction, kill, DoD, files, interview questions,
> citations) live in [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) §3 and §10.
> [`FRONTIER_STATUS.md`](FRONTIER_STATUS.md) provides a one-page view of the same information.

Every rung: pre-registered falsifier in `bench/RESULTS.md` / `docs/RESULTS.md` *before* running,
one variable at iso-FLOP, kill criterion checked. Status: ✅ shipped · 🆕 shipped today (2026-07-30)
· ⬜ pending · 💰 GPU/rental-gated.

| Rung | Purpose (one line) | 2026 verdict | Status | Pre-registered falsifier | Kill criterion |
|---|---|---|---|---|---|
| **A0** | 13-gram decontamination gate | Settled + differentiator (nanochat skips it) | ✅ `data/decontaminate.py` | planted eval string dropped; overlap rate logged | >20% of a slice flagged ⇒ leak, investigate |
| **A1** | Real-corpus memmap shards | Settled | ✅ `data/shards.py` | memmap round-trip; next-token-aligned | id ≥ 2¹⁶ ⇒ uint32 |
| **A2** | Config-carrying checkpoint chaining | Settled | ✅ | save→rebuild `torch.equal`; resumed loss ≪ log V | config shape-mismatch |
| **A3** | Chat specials + template | Settled | ✅ `chat.py` | specials = 1 id; mask covers assistant+eot only | `len(encode(special))>1` |
| **A4** | Midtrain stage (full-CE chat data) | Contested at 0.5B (nanochat deleted it; OLMo/SmolLM3 keep it) | ⬜ spec done, unbuilt | midtrained d12 > base d12 ChatCORE at iso-compute | >30% of mix exceeds ctx |
| **A5** | Assistant-masked SFT | Settled | ✅ `algos/chat_sft.py` | overfit-one-batch NLL<0.1; stops at eot | NLL plateau >0.1 (mask off-by-one) |
| **A6** | Chat REPL over serving | Settled | ✅ `chat_cli.py` | reply grows history; no leaked specials | never emits `<\|eot\|>` |
| **A7** | Optimizer-embedded ZeRO-2 | Settled (nanochat's shape) | ✅ `utils/dist_train.py` gloo 12/12 | ranks draw different batches; single-proc byte-identity | comm >3% wall; scaling <93% |
| **A8** | d20 runbook + guardrails | Settled | ✅ runbook + consolidated ckpt | bitwise ckpt round-trip cross-topology | $90 cumulative ⇒ abort → d16 |
| **A9** | Public release surface | Settled (OLMo-3 bar) | ⬜ post-d20 | model card + repro bundle + decontam statement | — |
| **F1** | Muon vs *tuned* AdamW, iso-FLOP | **Contested — three-way fight live** (Wen I 2509.02046 / Qiu 2512.05620 / Wen II 2606.16899 + Xiao 2607.22444) | ✅ harness; 💰 run descoped (Muon ADOPTED, `f9e8f3b`) | 1.1–1.4× band vs independently-LR-tuned AdamW; NS overhead <1% | saving <5% vs tuned AdamW; divergence; **LR-reuse protocol hole must be fixed before any number is quoted** |
| **F2a** | MTP train head (D=1 sequential) | Convergent default (≥11 families); **nanochat's own A/B failed** (#481); BabyLM caveat at ≤1B | 🆕 `3b89119` (`mtp.py`, `model.py` mtp_depth/_trunk/forward_train, `train.py:241,409`) | Δval-CE ∈ [−0.02, +0.01] nats — **measured +0.0027 today, in band** (session-reported; to be logged in `docs/RESULTS.md`) | MTP aux degrades base val loss >+0.01 nats |
| **F2b** | MTPDrafter self-spec decode | Settled direction; acceptance domain-stratified; hybrid-backbone collapse risk | ⬜ post-d20 💰 | prose a2-acceptance ≥0.30, > n-gram; γ=1–2; speedup claim on code/JSON | a2 ≤ n-gram |
| **F3** | De-confound serving on a real model | Hygiene, necessary | ✅ harness; 💰 falsifier awaits trained ckpt | prose acceptance <10%, code/JSON 40–60% | prose ≥ code/JSON |
| **F4** | bf16+compile; stretch NVFP4 numerics | bf16 settled; FP8 ROI ≤1B recalibrated 1.1–1.3×; NVFP4 = mixed-precision + 0.6–0.9% gap @1.3B (CHON 2602.02047) | ✅ bf16/compile + NaN guard; ⬜ NVFP4 stretch (sm120) | compile+bf16 +10–20% MFU; NVFP4 gap band 0.6–0.9% pre-registered (not "val-loss = FP8") | compile NaN on sm90 ⇒ eager fallback |
| **F5** | MLA-for-real (substrate) | MLA is legacy incumbent (V4 moved to CSA/HCA); sub-1B param-efficiency unpublished — pre-register both outcomes | ⬜ first-slice pending | within +0.02 val loss of GQA-8 iso-param; KV ≥3× smaller | Δ>+0.05 nats |
| **F6** | MoE balancing: granularity axis | Reframed: bias-free vs **global-batch LBL** vs seq-aux (seq = known-bad control; Qiu 2501.11873); "none collapses" is settled — smoke arm only | 🆕 harness `98f62e3` (`eval/moe_ablation.py`, discriminating test green; artifact numbers vacuous — synthetic 60-step smoke, redo ≥1B real tokens) | BIAS_FREE ≤ global-batch/SEQ_AUX val CE, entropy >0.9·log N_r | worse by >0.02 nats or entropy <0.9·log N_r |
| **F7** | Debunk the aha (random-reward control) | **Reframed + strengthened** (2506.10947; pass@k ceiling 2504.13837; length-growth dropped as oracle) | ⬜ F7a/b/c CPU-green harness pending; real run 💰 | random-reward recovers most of the gain ⇒ aha spurious; **add pass@8 + "control fails on our base" pre-registration** | control fails to reward-hack ⇒ real learning, re-baseline |
| **F8** | DSA sparse attention | "Freshest" is stale (V4 CSA/HCA Apr-2026; GLM-5.2 IndexShare) — now the *best-documented* sparse recipe; DSA-from-scratch genuinely unexplored | F8.1 ✅ `dsa.py`; F8.2 ⬜ | indexer recall ≥0.95; +0.03 nats @8k; **FLOP crossover pre-registered NULL at 8k** | gap >0.1 nats; recall <0.95 |
| **F9** | QK-clip guard (gated >1B) | Settled (softcap dead — Gemma 3 2503.19786; K2 explosion was 9B-active); qk_norm suffices sub-1B | ✅ observer + `apply_qk_clip` | with qk_norm: max logit <30 ⇒ clip γ≡1 | sustained S_max>30 |
| **F10** | Hybrid linear attention | The live attention frontier; **block of record is now GDN-2** (2605.22791) not GDN; recall probe mandatory (Wang 2507.06457) | F10.1 ✅ `linear_attn.py`; F10.2 ⬜ | 3:1 hybrid within +0.03 val loss of full-attn iso-param; state ≥2× smaller; **MQAR/NIAH recall parity**; decode uplift ≤8k predicted null | val gap >0.1 nats; no state win; recall deficit |
| **F11** | Agentic / tool-use RL | #1 stated lab priority; SWE-bench Verified citation stale (withdrawn) — direction unaffected; reward mode collapse ≤1B documented (2504.02273) | ⬜ CPU-green harness pending | verifiable-reward success ↑, turns bounded; format-only control flat; pass^k + turns-used reported | flat success; control not caught by logging |
| **F12** 🆕 | **Data ablation: FineWeb-EDU vs ClimbMix** | The missing rung — data was nanochat's biggest win (−27%, [324e69c](https://github.com/karpathy/nanochat/commit/324e69c.patch)); decides the d20 corpus with *our* recipe | ✅ **done 2026-07-31**: kill fired (ClimbMix bpb 1.30205 ≥ FWE 1.19197, Δ=+0.1101); **operator OVERRIDE → ClimbMix** (nanochat's larger-scale result), FINAL 2026-08-02; CORE band re-anchor now required (`docs/RESULTS.md` §F12) | ClimbMix bpb < FWE bpb at iso-FLOP (35M/700M, tokenizer retrained per corpus, CORE+bpb) | ClimbMix ≤ FWE ⇒ keep banked FWE corpus; CC BY-NC noted in A9 |
| **P1–P6** | d20 gate prerequisites | Settled engineering | P1 ✅ SDPA · P2 ✅ parquet · P3 ✅ CORE suite · P4 LR machinery folds into P5 · P5 💰 $10–15 · P6 busbw ≥350 GB/s gate | per-runbook scorecard | per-runbook scorecard |

**Re-ranked EV order (justified by the 2026-07-30 pass):**

1. **S3 scaling sweep + F12 data ablation (free, immediate)** — they decide the D:N re-registration
   and the corpus *before* $100 is spent; data > optimizer in measured effect size.
2. **F7-reframed + F11 harness (free)** — RL honesty + the #1 lab priority; CPU-green now, rental
   payoff post-d20; the random-reward and format-only controls are the scarce 2026 artifacts.
3. **F10.2 (GDN-2 upgrade + recall probe) / F8.2 (DSA-from-scratch)** — the contested attention
   frontier, with the corrected expectations (null decode uplift ≤8k, recall-gated).
4. **P5 → d20 (paid)** — the loop-closure artifact; gates on the free work above.
5. **F2b / F3 (post-d20)** — the train↔serve payoff on a real checkpoint.
6. **F1 methodology fix (only if quoted again)** — independent LR/wd sweeps; the tuned-baseline
   methodology is the artifact, not the win.
7. **F5 first-slice · F6 real-data run · F9 (rides any instrumented run) · F4-NVFP4 stretch** —
   necessary but minimized.

---

## §4 — What a frontier RE must be able to derive/explain

Interview checklist, per stage. Each answer must be derivable at a whiteboard, not recalled.

**S0 Data.** Why `C = 6ND` (2 fwd + 4 bwd FLOPs/param/token). Why bpb is the only honest
cross-vocab metric (CE/token is vocab-size-confounded; bpb normalizes by bytes). Why overtraining a
served model past Chinchilla is rational (inference-aware optimum; amortize train FLOPs over many
served tokens) and where it breaks (scaling laws overestimate token value at extreme ratios;
>4 epochs of repetition decays to zero). Why n-gram decontamination is necessary but insufficient
(paraphrase/semantic contamination survives).

**S1 Architecture.** Why head_dim is pinned at 128 (kernel tiling). Why untied embeddings want
different LRs (embedding-LR bottleneck — Kalra). Why QK-norm bounds logits at ~√d_head scale and
why that makes QK-clip dead code below 1B. Why a 3:1 linear:full hybrid keeps *any* full-attention
layers (retrieval/induction heads are loss-invisible but recall-visible — Wang; multi-hop deficits
at scale — MiniMax; hybrids must be trained from scratch because retrieval heads form early). Why
KV/token bytes = 2·n_layers·n_kv_heads·head_dim·(bytes/el) and why that makes attention-efficiency
irrelevant at 2k for a 480M model. Why MLA's latent sizing is param-unfair below ~1B.

**S2 Pretraining.** Derive the Muon update (orthogonalized momentum via Newton–Schulz); why a
full-rank orthogonalized [A,B] update has RMS 1/√max(A,B) and why 0.2·√max(A,B) reuses AdamW's LR
band. Why µP LR transfer matters and which layer dominates it (embedding). Why WSD/linear-to-zero
(decay phase is where the gain lives). Why per-tensor LR groups (matrices/embed/head/norms want
different scales — Track-3 numbers). Why an under-tuned baseline produces a fake win (Wen I) and
why cross-optimizer LR reuse is exactly that sin.

**S3 Scaling.** Why power laws are fit in log-log (scale-free over decades; curvature = regime
change). Why a+b = 1 (the C=6ND identity forces it) and what a violated gate means. Why bpb for
curve-fitting but CORE for capability (noise floors: CORE ±0.008–0.016 run-to-run). Why you cannot
compare per-token loss across tokenizers or val sets.

**S4 Distributed.** Ring all-reduce traffic 2(k−1)/k·payload; why DP > TP/PP/FSDP at 480M/8×H100
(51× headroom over machine balance; TP's unoverlapped all-reduces; 1F1B bubble; FSDP solves a
non-problem). Why Muon's Newton–Schulz forces whole-matrix sharding (element-wise ZeRO-1
incompatible). Why identical batches across ranks is a silent 8× data loss. Why loss-at-init must
equal log V (uniform logits at init).

**S5/S6 Midtrain/SFT.** Why full-CE at midtrain but assistant-only mask at SFT (statistics of the
format vs credit assignment for behavior). Why mask off-by-one is the classic silent bug and how
the log-V-at-init + overfit-one-batch oracles catch it. Why timing×weight interact (bridging
paper).

**S7 RL.** Derive the GRPO group baseline; why its length/std normalization inflates wrong-answer
length (Dr.GRPO); why random rewards produce aha-like curves (clipping asymmetry amplifies
pretraining priors — and why it fails on bases without those priors); why pass@k(large) is the
honest ceiling metric (RL sharpens, doesn't expand); why the format-only control is the expected
failure mode at ≤1B (reward mode collapse); why log IS-ratio even single-node (train/rollout
backend mismatch).

**S8 Serving/eval.** Why speculative decoding is lossless (target-model verification). Why
acceptance is domain-stratified and why draft depth >4 with one chained head is counterproductive
(geometric decay). Why centered accuracy is what makes ≤561M models rankable (raw scores at
chance). Why MTP is sequential (parallel heads can't model the joint — the token-salad flaw) and
why its bpb effect is neutral at small scale (payoff is the free draft head; gains are downstream).
Why weight quantization is pointless for a 1 GB artifact and why 4-bit PTQ is the worst case at
≤1B. Why aux-loss-free balancing works (bias shifts routing *probabilities* without a gradient
fighting the task loss; constraint at corpus level, not per-sequence — per-sequence constraints
kill specialization).

---

## §5 — The immediate next-actions DAG

Ordered; effort `[S/M/L]`, deps, cost. Standing box = sm120 (free). Every rung: pre-register →
build test-first → green-CI → measure → fill the ledger.

1. **S3 scaling-law mini-sweep** `[L, free, no deps]` — the §3 grid (8 points, ~2–3 GPU-days
   standing box). Fit L(N,D) in bpb; a+b gate; D:N decision rule; nanochat-oracle overlay. *Output:
   the d20's D re-registered or confirmed, and a bpb-vs-FLOPs curve of our own.* Gates P5.
2. ~~**F12 data ablation**~~ — **DONE 2026-07-31** (`docs/RESULTS.md` §F12): kill criterion fired
   (ClimbMix bpb 1.30205 ≥ FineWeb-EDU 1.19197 at iso-FLOP, Δ=+0.1101; CORE 0.0551 vs 0.0510) —
   measurement-only verdict was KEEP FineWeb-EDU; **operator OVERRIDE chose ClimbMix anyway**
   (following nanochat's larger-scale result), decision **FINAL 2026-08-02**. Corpus is settled and
   ClimbMix is already staged on the pod — no longer gates P5 corpus staging; the CORE-band
   re-anchor (§S4(e)) rides (5).
3. **Free harness batch (standing box / CPU)** `[M each, parallel]` — **F8.2** (DSA wired +
   measured, recall-gated), **F10.2** (`attn_schedule` 3:1 + GDN-2 block upgrade + MQAR/NIAH probe
   + iso-param quality/state ablation), **F7a/F7b/F7c** (aha harness + detector + CPU-green
   reward-hack proof), **F11 harness** (safe AST evaluator + multi-turn masked rollout +
   format-only control). *Output: the scarce-2026 science artifacts, ready for a real base.*
4. **A4 midtrain build** `[M, free, deps: A1/A3 ✅]` — `data/chat_adapters.py` + `stage_midtrain`;
   value-gated on a d12 A/B before entering any pipeline default.
5. **P5 d12 dress rehearsal** `[M, 💰 ~$10–15, 1×H100, deps: G1/G2/G4 entrypoints + S3 D:N
   decision]` — Muon LR sweep, compile-on-sm90 re-validation, CORE vs public checkpoint, ckpt
   kill/resume drill, step-time measurement (the d20 go/no-go).
6. **The 8×H100 d20** `[💰 ~$100, deps: P5 pass + user authorization]` — 480.4M, 9.6B tokens,
   18,311 steps, MTP head baked (F2a falsifier passed today: +0.0027 in band), CORE 0.19–0.22
   pre-registered, $90 abort → d16. **Runs unconditionally** (decision of record, 2026-08-02):
   the d20-GQA is simultaneously (i) the loop-closure artifact with its pre-registered CORE band,
   (ii) the Rung-0/1 control-family anchor the KDA multiplier is measured against, and (iii) the
   insurance run if KDA slips. The Rung-2 family fit decides the flagship *science claim*
   (whether a d20-scale mini-K3 follows as flagship), not whether the loop closes. This resolves
   the §5-vs-`k3/SCALING_LADDER.md` §3 ordering question in favor of running both.
7. **Post-d20 rungs** `[M each, 💰-free once the ckpt exists]` — **F3** (de-confound acceptance on
   the real ckpt) → **F2b** (MTPDrafter on the trained head) → **F5 first-slice** (MLA substrate)
   → **F7 real run** (Countdown zero-RL + random-reward control + pass@8, on the d20 base) →
   **F6 real-data re-run** (≥1B tokens).
8. **A9 release** `[S, dep: 6]` — model card (config, data provenance + decontamination statement +
   overlap rate + ClimbMix license if F12 flipped the corpus, CORE-vs-FLOPs table, per-task SNR
   statement, kill outcomes incl. negatives) + reproducible bundle (weights + tokenizer + config +
   manifest + transcript).

---

## §6 — Honesty ledger

**What this research pass changed vs the previous plan** (each with its citation):

- **F1's "Muon superseded by MuonH" was one-sided.** The fight is three-way live: Wen I deflation
  ([2509.02046](https://arxiv.org/abs/2509.02046)) vs Qiu's µP-transfer counter
  ([2512.05620](https://arxiv.org/abs/2512.05620)) vs Wen II MuonH
  ([2606.16899](https://arxiv.org/html/2606.16899v1)) vs Xiao's schedule-shaping counter-counter
  ([2607.22444](https://arxiv.org/abs/2607.22444)). And our registered race protocol commits the
  exact sin F1 exists to debunk (AdamW-LR reuse across optimizers) — fixed as a methodology gate
  in §3.
- **F8's "freshest, scarcest technique" is stale.** DeepSeek V4 (CSA/HCA,
  [2606.19348](https://arxiv.org/abs/2606.19348)) moved past DSA in April; GLM-5.2 commoditized it
  in June ([2603.12201](https://arxiv.org/pdf/2603.12201)). DSA is now the *best-documented*
  sparse recipe; DSA-from-scratch is the open angle.
- **F10's named block is two generations old.** GDN → GDN-2 ([2605.22791](https://arxiv.org/abs/2605.22791))
  / EDA ([2606.26560](https://arxiv.org/abs/2606.26560)); and the rung lacked the recall probe that
  Wang et al. ([2507.06457](https://arxiv.org/abs/2507.06457)) prove is mandatory (LM loss is blind
  to the hybrid failure mode). MiniMax's official retreat
  ([minimax.io](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model)) is the
  strongest negative result in attention this year.
- **The program treated the corpus as fixed while nanochat got its biggest win from data.**
  ClimbMix-400B: −27% wall-clock, −0.028 bpb ([324e69c](https://github.com/karpathy/nanochat/commit/324e69c.patch));
  license CC BY-NC 4.0. F12 added; the banked FineWeb-EDU corpus + the 0.19–0.22 CORE band are
  both gated on it.
- **The d20's ratio-20 was Chinchilla folklore.** nanochat's own fit: compute-optimal ≈10.5,
  speedrun 8 ([VERIFIED], [speedrun.sh](https://raw.githubusercontent.com/karpathy/nanochat/master/runs/speedrun.sh));
  our 9.6B is a deliberate inference-aware overtrain ([2401.00448](https://arxiv.org/abs/2401.00448))
  — re-registered as a *decision*, gated on the S3 sweep.
- **MTP "bake by default" weakened.** nanochat's own MTP A/B failed (+13 GB, no improvement —
  [#481](https://github.com/karpathy/nanochat/discussions/481)); the plan's flagship citation was
  mis-scaled (V3's ablation is at 15.7B/228.7B, not ≤561M — [2412.19437](https://arxiv.org/html/2412.19437v2));
  BabyLM: vanilla MTP *underperforms* at 130M without a curriculum
  ([aclanthology.org/2025.babylm-main.41](https://aclanthology.org/2025.babylm-main.41/)). Our F2a
  stands (shipped today, falsifier +0.0027 in band) but the d20 bake stays falsifier-gated, and
  the F2b speedup claim moved to code/JSON (prose α≈0.6 at 27B dense —
  [thefrontierlab.ai](https://thefrontierlab.ai/mtp-defaults-are-a-trap/) [REPORTED]).
- **nanochat deleted midtraining** (#481) — A4 re-anchored to OLMo 2/3 Dolmino + SmolLM3 and
  value-gated. Conflicting fetch (a `mid_train.py` at master) kept `[UNCERTAIN]`.
- **F7 strengthened, not just reframed.** The 21.4/29.1 numbers re-verified
  ([2506.10947](https://arxiv.org/abs/2506.10947)); added the pass@k ceiling
  ([2504.13837](https://arxiv.org/abs/2504.13837)), the "control should fail on our base"
  discriminator, DAPO ([2503.14476](https://arxiv.org/abs/2503.14476)) and ScaleRL
  ([2510.13786](https://arxiv.org/abs/2510.13786)) as primary sources; GSPO demoted (MoE-motivated);
  nanochat's RL is REINFORCE-lite, our GRPO is stronger than the reference.
- **F11's agentic citation was stale.** SWE-bench Verified was withdrawn by OpenAI
  ([openai.com](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)); the
  direction is unaffected; pass^k ([2406.12045](https://arxiv.org/abs/2406.12045)) replaces
  single-run success in the falsifier; reward mode collapse at ≤1B is documented
  ([2504.02273](https://arxiv.org/pdf/2504.02273)) — the format-only control is the expected
  failure mode.
- **F4's bands recalibrated.** FP8 at ≤1B ≈ 1.1–1.3×, not 1.3–1.5×
  ([2507.16099](https://arxiv.org/html/2507.16099v1)); NVFP4's "val-loss = FP8" is a 12B/10T
  mixed-precision claim with 10 BF16 layers retained ([2509.25149](https://arxiv.org/abs/2509.25149));
  the ≤2B replication shows a 0.6–0.9% residual gap ([2602.02047](https://arxiv.org/abs/2602.02047));
  H100 has zero FP4 hardware.
- **F6's citation was misattributed.** `2408.15664` has no batch-wise arm — the granularity result
  is Qiu et al. global-batch LBL ([2501.11873](https://arxiv.org/abs/2501.11873)), and its finding
  is "global-batch LBL is *good*", not "≈ bias-free". Qwen3's arXiv ID was wrong (2505.09388, not
  2501.15383). Today's shipped smoke numbers (`artifacts/f6_moe_ablation/`) are vacuous (d=32,
  60 steps, synthetic) — the harness is real, the table is not evidence.
- **Attention-efficiency decode claims at ≤8k are predicted null** (KV math §S1; Kimi's own 4k
  parity; MiniMax's crossover at "a few thousand tokens") — F8/F10 falsifiers re-pointed at quality
  parity + recall + state bytes.
- **Eval kill bands re-tuned to measured CORE noise** (±0.008–0.016, 7 identical repeats — #481);
  bpb is the curve-fit comparator; A0 reframed from hygiene to differentiator (nanochat skips
  decontamination entirely).
- **nanochat factual drift (resolved):** HEAD = d24/ratio-8/FP8/ClimbMix, 1.65h, CORE 0.2626; the
  Oct-2025 d20 (561M/65536/midtrain/0.2219) is archival; ReLU² (not SwiGLU), softcap 15, value
  embeds (44% of d24 params), SSSL windows, Polar-Express Muon stack — our replication target is
  the Oct-2025 recipe, and that is now an explicit, documented choice.

**What remains [UNCERTAIN]:**
- Whether Muon's edge decays intrinsically with scale or is rescued by µP + 1/width-wd (Wen I vs
  Qiu — direct conflict, both verified).
- Whether MuonH is a mechanism or schedule shaping (Wen II vs Xiao); whether any of it matters at
  ≤100M × ~1× Chinchilla (its evidence is ≥1.2B).
- Whether linear hybrids beat full attention *at frontier scale* (Kimi yes at 48B vs MiniMax's
  measured retreat) — and whether the ~25%-full-layer knee holds at 30–300M from scratch (all
  convergent evidence is ≥3B or CPT-based).
- Whether a nanochat `mid_train.py` exists at master (one agent fetched it; the #481 deletion quote
  + current speedrun.sh say deleted) — resolved against deletion, kept uncertain.
- Whether ClimbMix's win survives *our* recipe (Muon-hybrid, our BPE, MTP head, ratio-20) —
  **answered by F12 (2026-07-31): it does not** at 35M/700M (+0.110 bpb worse); ClimbMix is the d20
  corpus by operator override resting on nanochat's larger-scale result (RESULTS.md §F12).
- Whether zero-RL does anything measurable at all on a from-scratch 560M FineWeb base (every
  success story uses Qwen/DeepSeek priors or distills first).
- K3's Attention Residuals: **[VERIFIED] + measured by us** — learned pseudo-queries, block size 12
  (FACTS A9); R0 anatomy census confirms which pseudo-queries receive gradient (FACTS A19,
  `artifacts/k3_anatomy/`). Gemma-4 MTP acceptance numbers, Track-3 barnacle load-bearingness:
  [REPORTED]/unmeasured.

**Claims we still must NOT make until measured:**
1. Any CORE number with precision below ±0.008 (single-run noise floor) — multi-seed or bpb-backed.
2. "MTP improves pretraining loss" at our scale — our pre-registration is *neutral* bpb; the V3
   ablation is at ≥15.7B; BabyLM measured a small-scale regression.
3. Any DSA/linear-hybrid decode or FLOP-crossover win at ≤8k — predicted null; crossovers in the
   literature are ≥32k.
4. "Linear-hybrid beats full attention" at ≤300M — unmeasured anywhere; and a val-loss win without
   a recall probe is not a win.
5. Any MoE wall-clock claim — the dense↔MoE wall-clock crossover at our scale is unpublished;
   iso-FLOP ≠ iso-time (Marin: 6.7× theoretical vs 3.6× realized).
6. Any FP8 speedup on H100 at 480M — P5 must measure it behind the KL gate (band 1.1–1.3×, can be
   config-dependent slower).
7. "NVFP4 = FP8 parity" — the parity claim is 12B/10T with 10 BF16 layers; the ≤2B gap is 0.6–0.9%.
8. "Decontaminated" without the method + overlap-rate qualifier — n-gram gates miss paraphrase;
   soft contamination is expected, and nanochat skips the gate entirely (we are *better*, not
   *pure*).
9. Any RLVR "reasoning emergence" claim at ≤1B — pass@1↑ with pass@8 flat/↓ is sharpening; and a
   random-reward control that "works" on our base means contamination, not reasoning.
10. Any single-run agentic success number — pass^k or it didn't happen.
11. Any "nanochat-parity" recipe claim — HEAD nanochat (SSSL, value-embeds, ReLU², softcap,
    Polar-Express Muon) is a different recipe from the Oct-2025 d20 we replicate; comparability is
    to the published CORE-vs-FLOPs curve, not to HEAD.

---

## Source-of-truth pointers

- Ablation strategy + rung cards: [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) (§10 =
  the 2026-07-09 re-verification; this doc's §6 is the 2026-07-30 delta on top).
- Buildable rung DAG + Next-node marker: [`FRONTIER_2026_TASKSPEC.md`](FRONTIER_2026_TASKSPEC.md).
- The paid-run script + scorecard: `deploy/runbooks/d20_speedrun_8xH100.md`; pre-registrations in
  `bench/RESULTS.md` (§*$100 d20 run*, §Decision — Muon+AdamW ADOPTED) and `docs/RESULTS.md`.
- Research swarm output (the 9 angle reports behind every citation here):
  `~/.kimi-code/sessions/wd_scratch_llm_1c6bbaa8f639/session_2986d58f-1e7e-4343-a6ec-74bf7a32c080/agents/main/tool-results/AgentSwarm-tool_iCUaGZw2psVDBRQBiPmxx3m5-4b823469-a2be-4940-8e53-f81706cf0454.txt`
  (infra angle failed on quota; infra content here is repo-verified instead).
- Build status: `docs/STATUS.md`. Decision record: `docs/adr/ADR-0018-close-the-loop-nanochat-front.md`.
- K3 track (build & host Kimi K3 from scratch; chartered 2026-07-31): `docs/k3/ROADMAP.md` +
  `docs/k3/FACTS.md` (claim ledger — tech report wins over secondary sources).
- Reference oracle (re-own, do not copy): karpathy/nanochat at HEAD.
