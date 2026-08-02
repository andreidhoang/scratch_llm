# K3 TRACK — Build & Host Kimi K3 from Scratch

> Chartered 2026-07-31. Spine: the two Vizuara books (*Build Kimi K3 from Scratch*, 35 capsules;
> *How to Host Kimi K3*, 40 capsules — both open 2026-08-03, $49/mo). Spec of record: Moonshot's
> primary sources (tech report arXiv:2607.24653, config.json, reference code). Every load-bearing
> claim is ledgered in [`FACTS.md`](FACTS.md) — where a book and the tech report disagree, **the
> tech report wins** and the book capsule is flagged.
>
> Role frame: this is run as if we were the Moonshot team rebuilding our own model — which is
> literally how K3 was built: Kimi Linear 48B-A3B was the intermediate open artifact, K3 is the
> scaled successor. Our roadmap mirrors that lineage: **GDN → KDA → Kimi-Linear-class miniature →
> K3-ify (gates, AttnRes, LatentMoE, SiTU, Per-Head Muon, MXFP4 QAT) → serve the real checkpoint.**

---

## 0. Decision — build in THIS repo (monorepo), no new repo

**Verdict: `scratch_llm` monorepo, new `k3` subpackage. Do not create a new repo.**

Reasoning, as a principal researcher would weigh it in 2026:

1. **We already own ~70% of the substrate.** KDA is a delta-rule linear-attention layer with
   per-channel decay — our `linear_attn.py` (Gated DeltaNet, F10.1, with the
   chunkwise==recurrent==float64 contract) is its direct ancestor, and F10.2 (short conv +
   gated output norm) is literally the missing KDA piece. `mla.py` ships MLA with decoupled RoPE
   and the weight-absorption identity. `moe.py` + the F6 balancing harness, `optim.py` (Muon,
   F1 shipped), `quant/nvfp4_mxfp4.py` (A5 measured), `mtp.py` (F2a), `serving/` (paged,
   continuous, speculative, cudagraph), the `speedrun.py` training spine, the eval report card,
   and the ADR-0012 rental discipline are all here, tested, and green. A new repo orphans all of it.
2. **This is how frontier labs actually organize.** Moonshot did not start a new codebase for K3 —
   K3 inherits Kimi Linear's kernels (fla/ops/kda), K2's MLA and weight-clipping, DeepSeek's
   aux-loss-free routing. One monorepo, models as composed modules, each generation a diff on the
   last. Rebuilding that pattern *is* the lesson.
3. **CI and the honesty ledger stay single-sourced.** 456+ CPU tests, ruff/pyright gate, and
   `bench/RESULTS.md` measured-claims discipline already exist. A second repo splits the ledger.
4. **When to carve out (and only then):** if mini-K3 becomes a public artifact worth releasing,
   `git subtree split` the `k3/` paths into a standalone repo at that point — history preserved,
   zero cost until then. Trigger: external users, not aesthetics.

**Two classes of citizenship (the hand/delegate split, encoded 2026-07-31):** `k3/core/` is
hand-built by the human — agents are read-only there (adversarial tests + retyped proposals
only); the boundary is binding via root `AGENTS.md`, rules and per-module mastery bars in
`src/scratch_llm/k3/HANDCRAFTED.md`. Criterion: *silent-bug surfaces are human-owned,
loud-bug surfaces are delegated.* Paired modules (`muon.py`, `qat.py`): human writes the
math, agents write plumbing + tests. Everything else is delegated (agents own, human reviews).

Layout (mirrors existing conventions — flat `tests/test_*.py`, docs under `docs/`):

```
docs/k3/                    ← this roadmap + FACTS.md (claim ledger)
AGENTS.md                   ← the boundary, binding on every agent session
src/scratch_llm/k3/
├── HANDCRAFTED.md          # two-citizenship rules + mastery bars + status table
├── config.py               # ✅ K3Config: full-fidelity (from config.json) + mini presets
├── param_count.py          # ✅ accounting; EXACT closure to HF total, residual 0 (FACTS A18)
├── core/                   # ⬜ HAND-BUILT ONLY (agents read-only) — serial order:
│   ├── situ.py             #   1. SiTU-GLU (β1=4 / β2=25, |f| ≤ 100) — warmup
│   ├── kda.py              #   2. KDA: per-channel Diag(α), scaled-sigmoid g_min=−5, conv+Swish+L2Norm, full-rank gate  [UPGRADE linear_attn.py]
│   ├── gated_mla.py        #   3. MLA + full-rank output gate, NoPE (RoPE kept as heritage flag)                      [UPGRADE mla.py]
│   ├── latent_moe.py       #   4. 0.5× latent → sigmoid router → Quantile Balancing → RMSNorm → up; 2 shared        [UPGRADE moe.py]
│   └── attn_res.py         #   5. Block AttnRes: learned pseudo-queries, online-softmax merge
├── muon.py                 # PAIRED — human: per-head NS + weight clipping; agents: plumbing  [UPGRADE optim.py]
├── qat.py                  # PAIRED — human: MXFP4 fake-quant + STE math; agents: training wiring [UPGRADE quant/]
├── model.py                # DELEGATED — mini-K3 assembly (3:1 pattern, dense L1, terminal MLA)
└── serve.py                # DELEGATED — hybrid-state cache: KDA state + MLA latent KV
tests/test_k3_param_count.py  ✅ gate green (2.78T closure + 104.2B active convention)
tests/test_k3_{situ,kda,gated_mla,attn_res,latent_moe,muon,qat,model}.py  ⬜ (core: agent red-team only)
deploy/runbooks/k3_8xb300_modal.md          ✅ K9 runbook, written before the rental
bench/RESULTS.md §K3                        # measured claims, repo convention
```

---

## 1. What the two books are, honestly assessed

**Build book (35 cap, ~5h)** — component-wise rebuild: tokenizer → baseline block → KDA (7 cap,
deepest section) → Gated MLA → AttnRes → LatentMoE → assembly + param accounting + mini training
→ MXFP4 QAT → 1M-context analysis → serving/vision/outro. Strengths: correct architecture numbers
throughout (all blurb-level numbers verified against config.json), runnable-laptop aspiration,
capsule 25 "Counting to 2.8 trillion" is exactly the param-accounting gate we want. Weaknesses:
**capsule 15 "Decoupled RoPE in the MLA layers" contradicts K3** (the model is fully NoPE — FACTS
A11; learn decoupled RoPE as K2 heritage, build NoPE); **no optimizer capsule** (Per-Head Muon —
K3's actual optimizer — is absent from the TOC); **no data capsule** (corpus, dedup, quality);
**no MTP/DSpark build** (K3's EAGLE-3 draft is how the real model decodes fast); post-training
(SFT/RL) and vision are overview-only.

**Hosting book (40 cap, ~9h)** — a measured deployment field guide: memory arithmetic → the one
8×B300 Modal vLLM run (flags, cold boot, measured numbers) → GPU menu → the A100/GGUF trap →
reported-only techniques (DSpark, prefix caching, P/D disagg, EPLB) → cost → ops. Strengths: the
measured/reported/not-verified label discipline (same as our ledger), reproducible-runbook
ambition, cost-first framing ($1,363/day forces scale-to-zero thinking). Weaknesses: every
headline measured number (0.93 s TTFT, 92.1 tok/s, $190.13/M, 27-min cold boot, A100 ~9 tok/s) is
the author's own and **unverifiable until Aug 3** (FACTS S5, S7); the "fp8 KV cache trap" framing
is contradicted by vLLM's official recipe which *uses* fp8 KV (FACTS S6) — read that chapter as
"one team's footgun", not gospel; Modal $56.79/hr is real but not the cheapest B300 tier.

**How they fit together:** the build book ends where the hosting book begins (capsule 32 "Serving
a 2.8T MoE" ≈ hosting capsules 04–18). Sequence: **build book 01–31 first** (with our K-phases
below), **hosting book 01–18 in parallel with K9** (the rental), hosting 19–40 as reference during
K10. Neither book is a prerequisite for starting — every component spec is public (papers, config,
FLA kernels, HF reference code) as of 2026-07-27. The books are the guided-reading layer; the
primary sources are the spec. Books open Aug 3 — phases K0–K5 need nothing from them.

---

## 2. What we already own → what K3 turns it into

| Existing asset (repo) | K3 component it becomes | Gap to close |
|---|---|---|
| `linear_attn.py` (GDN, F10.1; chunk/recur/f64 contract) | `k3/kda.py` | per-channel Diag(α) decay; scaled-sigmoid g_min=−5; short conv k=4 + Swish (F10.2); full-rank output gate |
| `mla.py` (MLA + decoupled RoPE + absorption identity) | `k3/gated_mla.py` | full-rank sigmoid output gate; NoPE mode (RoPE stays as heritage flag); FP32 attention-output option |
| `moe.py` + F6 balancing harness | `k3/latent_moe.py` | latent down/up 0.5×; sigmoid router + bias; **Quantile Balancing** (replaces sign-step); RMSNorm pre-up; 2 shared |
| `optim.py` (Muon, F1) | `k3/muon.py` | per-head NS partitioning on Q/K/V momentum blocks; K2-style weight clipping |
| `quant/nvfp4_mxfp4.py` (A5) | `k3/qat.py` | STE/fake-quant training path; experts-only scope; MXFP8 activation path |
| `mtp.py` (F2a) + `serving/speculative.py` (R4.3) | DSpark-style draft (K10) | EAGLE-3-style features + 7-step unroll; acceptance measurement harness exists (`eval/spec_acceptance.py`) |
| `serving/` (paged/continuous/cudagraph) | `k3/serve.py` | hybrid state: constant-size KDA state + MLA latent KV; prefix-cache semantics over recurrent state |
| `speedrun.py` + report card + data shards | mini-K3 training (K6) | wire k3.model into the spine; iso-FLOP arms vs d20 baseline |
| `utils/ep_moe.py` (A6, gloo-verified) | EP for LatentMoE (K10) | MoonEP-style redundant experts is reading-only |
| DELTA capstone (GDN-2 decode kernel) | KDA decode kernel (K10 stretch) | per-channel decay variant; DELTA's correctness contract inherits from kda.py's three-path equivalence |

**Net new code (no existing base):** `situ.py`, `attn_res.py`, `param_count.py`, `config.py`. Everything else is a scoped upgrade.

---

## 3. The phases — K0 → K10

Gates are the repo's usual discipline: loss-at-init ≈ log V, overfit-one-batch, fixed seeds,
predict-before-you-run, and every measured number lands in `bench/RESULTS.md §K3` with hardware +
method. Rung order is dependency order; K2 is the critical path.

### K0 · Orientation & the config of record  *(build book 01–03 · hosting 01–03)*
- Read: tech report §1–2, model card, config.json; book ch1 (free, sign-in).
- Build: `k3/config.py` (full K3 table from FACTS A13 + mini presets); `param_count.py`.
- **Gate:** `test_k3_param_count.py` reproduces **2,779,931,837,184 total / 104.2B active** from
  config fields alone, with the accounting convention documented (routed experts:
  92 MoE layers × 896 × 3 × 3584 × 3072 ≈ 2.72T; non-expert ≈ 57.2B BF16 — closes to the HF count).
- ~2 sessions. No GPU.

### K1 · Baseline: tokenizer, embeddings, standard block, the O(N²) wall  *(build book 04–06)*
- Load the **real** K3 tiktoken tokenizer (160K) for parity probes; train-tokenizer stays our BPE-32K.
- Standard block already exists (A1). New: measure attention cost vs context to 128K on the standing
  GPU (roofline harness exists) — the number that motivates KDA.
- **Gate:** loss-at-init ≈ log(32768); attention time/memory curve matches N² prediction.
- ~1 session (mostly wiring + a measurement).

### K2 · KDA — the workhorse  *(build book 07–12; Kimi Linear paper; fla/ops/kda)*
The deepest rung. Incremental from `linear_attn.py`:
1. per-channel decay: scalar α → Diag(α), low-rank decay projection (rank = head dim);
2. decay map: negative-softplus → **scaled sigmoid, g_min = −5** (predict: changes α floor to e⁻⁵);
3. F10.2 pieces land here: short causal conv (k=4) + Swish on q/k/v, L2Norm on q/k;
4. full-rank sigmoid output gate + RMSNorm before W_o;
5. chunkwise training form (WY/UT, chunk 64) upgraded from GDN's.
- **Gates:** (a) `test_k3_kda.py`: chunkwise == recurrent == float64-ref, now with per-channel α;
(b) GPU day: **numerical parity vs `fla.ops.kda`** (fla-core ≥ 0.4.0) on random tensors, fwd+bwd;
(c) micro-benchmark: decode step constant-memory, prefill scaling linear.
- ~2–3 weeks part-time. **This is the rung where the mastery happens — do not rush it.**

### K3 · Gated MLA, NoPE  *(build book 13–16 — with the capsule-15 correction)*
- From `mla.py`: add full-rank sigmoid output gate (from layer input x_t, pre-W_o);
  add `use_nope=True` mode (positions carried by the KDA layers — Kimi Linear §6.1:
  gated delta rule is a data-dependent multiplicative positional encoding).
- Keep decoupled-RoPE + absorption identity as the K2-heritage path; the book's capsule 15 teaches
  that heritage — learn it, then note K3 deletes it (FACTS A11, B8).
- **Gates:** absorption identity still exact under NoPE+gate (float64); KV-bytes/token accounting
  test: 24 MLA layers × (512+0) × 2 B vs GQA baseline.
- ~1 week.

### K4 · AttnRes  *(build book 17–18; arXiv:2603.15031)*
- `attn_res.py`: learned pseudo-query per sublayer; softmax over RMSNorm-ed block reps;
  Block AttnRes (block size 12 full-scale; size 4 in mini); online-softmax merge of inter/intra-block.
- **Gates:** equivalence of block vs full AttnRes on tiny L; parameter overhead = L·d; ablation arm
  wired (AttnRes on/off) for K6.
- ~1 week.

### K5 · Stable LatentMoE  *(build book 19–23; arXiv:2601.18089)*
- `situ.py` first (tiny): softcap identities, bound |f| ≤ β1β2 = 100, fp32 tanh-saturation test.
- `latent_moe.py`: down 0.5× → sigmoid router + learned bias → top-k (weights renormalized, bias
  excluded) → experts at latent width → RMSNorm → up; 2 full-width shared experts;
  **Quantile Balancing** (histogram over margins, one-step delay, mean removal) vs our F6
  sign-step bias — run both arms.
- **Gates:** balancing converges to target load q = mk/n without aux loss; activation-bound test
  (no |x| > 100 post-SiTU); router-frozen-at-inference path.
- ~1–2 weeks.

### K6 · Assemble & train mini-K3  *(build book 24–26)*
- `model.py`: the 93-layer pattern in miniature (config below), dense MLP on layer 1, terminal MLA,
  AttnRes wired, Per-Head Muon (`muon.py`) + weight clipping.
- Train via the `speedrun.py` spine on real shards (ClimbMix — F12 decision FINAL 2026-08-02,
  operator override),
  iso-FLOP arms: **mini-K3 vs our d-series dense baseline vs (optional) GDN-hybrid** — this
  *absorbs* ablations F5/F6/F10 into one artifact.
- **Gates:** overfit-one-batch < 1e-2; loss-at-init ≈ log V; val_bpb parity-vs-baseline at
  iso-FLOP pre-registered in `docs/RESULTS.md §K3` **before** the run; report card generated.
- ~2 weeks + 1–2 rental days ($10–100 tier, reuses P5/d20 runbooks).

**Mini-K3 reference config (d12, ~0.3–0.4B total / ~0.15B active — exact numbers fall out of K0's
`param_count.py`):**

| Field | mini-K3 | K3 (record) | Ratio kept |
|---|---|---|---|
| layers / pattern | 12 = 9 KDA + 3 MLA (2 blocks of 4 + terminal) | 93 = 69 + 24 | 3:1 + terminal |
| hidden | 1024 | 7168 | — |
| KDA heads × dim | 16 × 64 | 96 × 128 | — |
| MLA heads; kv_lora; q_lora; nope | 16; 128; 256; 64 | 96; 512; 1536; 128 | kv_lora/hidden ≈ 1/8…1/14 |
| MoE | 64 experts, top-4, latent 512 (0.5×), inter 192; 2 shared | 896, top-16, 3584, 3072; 2 shared | sparsity 16 vs 56; 0.5× latent |
| AttnRes block | 4 | 12 | — |
| vocab | 32768 (our BPE) | 163840 | training-only substitution |
| decay / gate / act | scaled-sigmoid g_min=−5; full-rank gates; SiTU 4/25 | same | exact |
| ctx | 8K → 64K (curriculum) | 8K→64K→256K→1M | staged |

### K7 · MXFP4 QAT  *(build book 27–29; OCP MX spec; Jacob et al. 2018)*
- `qat.py`: E2M1/block-32/E8M0 fake-quant + STE on routed-expert weights only; MXFP8 (E4M3)
  activation path; everything else bf16 — exactly K3's scope (FACTS A6).
- **Gates:** quant error accounting reproduces 4.25 bit/param ⇒ 1.561 TB arithmetic from the config
  (book capsule 29's number, derived not cited); QAT vs post-training-quant quality arm on mini-K3
  (pre-registered); STE gradient test.
- ~1 week.

### K8 · The 1M-context claim, examined  *(build book 30–31)*
- Memory math (`utils/memory_math.py`): KDA state (96 × 128 × 128 × 2 B/layer, constant) + MLA
  latent KV (24 × 512 × 2 B/token) vs full-attention baseline at 1M — reproduce the book's table
  from the config, not from the book.
- Eval: needle/RULER-style harness on mini-K3 at its context tier; report honestly (a 0.3B model
  will not "do" 1M — the lesson is the *memory/scaling structure*, plus how to evaluate).
- **Gate:** the 1M memory table reproduces; eval harness runs end-to-end on one long document.
- ~1 week.

### K9 · Serve the real checkpoint — the hosting book's main path, re-measured  *(hosting 04–18; build 32)*
**Rental-gated (ADR-0012 discipline; write the runbook before spending a dollar).**
- Pre-flight (free, CPU): download plan (96 shards / 1.561 TB; Xet caveats — FACTS S10), inspect
  tensors on one shard (safetensors headers), verify config parity with `k3/config.py`.
- Rent **8×B300 on Modal** ($56.79/hr — budget a 2–3 h session ≈ $120–170, NOT the $1,363/day
  leave-it-running failure): vLLM Docker (`kimi-k3` branch images), the book's flags + the official
  recipe; measure **TTFT, steady-state tok/s, cold-boot minutes, $/M tokens** ourselves.
- **Gate — the honesty exercise:** our measured numbers vs the book's (0.93 s / 92.1 / $190.13) vs
  vLLM's (111 tok/s bs1, 370 DSpark) — three-way ledger entry in `bench/RESULTS.md §K3`. Also test
  the fp8-KV question directly (quality spot-check with and without `--kv-cache-dtype fp8`).
- ~1 weekend + the rental. **This converts the book's unverifiable claims into our measured ones.**

### K10 · Frontier extensions (opt-in, EV-ranked)  *(hosting 19–40; build 33–35)*
1. **DSpark-style draft for mini-K3** — EAGLE-3 features + 7-step unroll on `mtp.py` +
   `serving/speculative.py`; acceptance harness exists. (The books only *report* DSpark; we build it.)
2. **KDA decode kernel** — port DELTA (GDN-2) to per-channel decay; the #1 co-designed artifact,
   now against the FLA oracle.
3. **GGUF/A100-tier path** — quantify the quant-of-quant quality cost on mini-K3 (QAT-native vs
   GGUF after the fact) instead of renting 8×A100.
4. **EP + EPLB, P/D disagg, prefix caching over recurrent state** — reading + design notes against
   vLLM/SGLang sources (measured only if K9 budget doubles).
5. **Vision (MoonViT-V2) + always-on reasoning** — reading-only capsules; our A5 GRPO stack covers
   the reasoning-training substance the books skip.

---

## 4. Gaps the books don't cover (frontier-lab additions, already in this repo)

| Gap in both books | Where it lives here |
|---|---|
| Optimizer: Per-Head Muon + weight clipping (K3's actual optimizer) | `k3/muon.py` (K6), from `optim.py` F1 |
| Data: curation, dedup, decontamination, corpus ablation | A4 pipeline + F12 (DONE — decision FINAL 2026-08-02: ClimbMix, operator override) |
| Distributed training: EP (MoonEP), KDA context parallelism, PP/ZeRO | A2/A6 (`fsdp.py`, `ep_moe.py`) + reading |
| Post-training: SFT → RL with QAT active throughout | A5 (`algos/`, GRPO) + K7 QAT scope |
| MTP/EAGLE-3 draft as a buildable artifact | K10.1 (`mtp.py` F2a base) |
| Scaling-law gate before any real training spend | S3 sweep (running, s1–s4 banked) + Chinchilla fitter |

## 5. Budget & risk

| Item | Cost | Note |
|---|---|---|
| Books | $49 (1 mo) or $399/yr | opens Aug 3; one month likely enough — primary sources are the spec |
| K2 GPU parity + micro-bench | ~$0–20 | standing box / small rental |
| K6 mini-K3 training | ~$10–100 | P5/d20 tier runbooks; S3 scaling gate first |
| K9 8×B300 session | ~$120–170 | 2–3 h at $56.79/h; **scale-to-zero or it's $1,363/day** |
| K10 | opt-in | kernel work is free (standing GPU); second B300 day only if budgeted |

Total cash exposure to finish K0–K9: roughly **$200–350** — the books are the minority of it.
The real cost is K2's calendar time.

## 6. START-HERE (next session)

1. `pytest -m "not gpu"` green; K0/K1 banked — `config.py` + `param_count.py` shipped,
   `test_k3_param_count.py` green (EXACT 2.78T closure, residual 0 — FACTS A18).
2. K2 (KDA) is the critical path — start there. Proposal of record:
   [`K2_PROPOSAL_KDA.md`](K2_PROPOSAL_KDA.md); re-read `linear_attn.py` docstring (the
   three-path contract), Kimi Linear §3–4, and `fla/ops/kda` — then hand-build
   `core/kda.py` per the per-channel-decay diff. Reading guide (Vietnamese, section-by-section
   with teach-back checkpoints): [`READING_GUIDE_K3_REPORT.md`](READING_GUIDE_K3_REPORT.md).
3. Aug 3: read build book capsules 01–12 against what we built; log any disagreements in FACTS.md
   (expected: capsule 15 RoPE framing).
4. Keep the d20 speedrun and S3 on schedule — K3 is the integration front, not a replacement;
   mini-K3 (K6) is where F5/F6/F10 converge. **Decision of record (2026-08-02):** the d20-GQA run
   happens *regardless* of the KDA family-fit outcome — it is the control-family anchor and the
   loop-closure artifact; the family fit decides whether a d20-scale mini-K3 follows as the
   flagship science claim, not whether d20-GQA runs (E2E §5 item 6, SCALING_LADDER §3).
5. Ablation program (R0 anatomy census COMPLETE 2026-07-31 — 417 small tensors, all 96 shards,
   `scripts/k3_fetch_tensors.py`, FACTS A19; R1 interaction factorial
   + R2 decay sweep launch after KDA is PROVEN): [`ABLATIONS.md`](ABLATIONS.md), pre-registered
   in `docs/RESULTS.md §K3`.
