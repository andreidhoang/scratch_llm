# Frontier ablation results — predicted vs measured

Pre-registered measurement ledger for the F-front ablations defined in
`docs/FRONTIER_2026_TASKSPEC.md`.  A row is `[FACT]` only once measured; until
then it records the falsifiable prediction and the kill criterion.

## F6 · MoE balancing ablation

Registered **before** running. Spec: `docs/FRONTIER_2026_TASKSPEC.md` §F6.
Thesis: aux-loss-free bias balancing (`BIAS_FREE`) achieves final validation CE
≤ sequence-level auxiliary loss (`SEQ_AUX`) while keeping router entropy above
`0.9·log(N_r)`.  The granularity axis (`COARSE` vs `FINE`) holds total routed
parameters and active routed FLOPs constant via the mapping in
`src/scratch_llm/eval/moe_ablation.py`.

| arm | granularity | predicted final val CE | measured final val CE | entropy > 0.9·log(N_r) | status |
|---|---|---|---|---|---|
| BIAS_FREE | COARSE | ≤ SEQ_AUX | — (pending) | yes | pending |
| BIAS_FREE | FINE | ≤ SEQ_AUX | — (pending) | yes | pending |
| SEQ_AUX | COARSE | baseline | — (pending) | yes | pending |
| SEQ_AUX | FINE | baseline | — (pending) | yes | pending |
| NONE | COARSE | worse by ≤ 0.02 nats or collapsed | — (pending) | no | pending |
| NONE | FINE | worse by ≤ 0.02 nats or collapsed | — (pending) | no | pending |

**Expected falsifier:** `BIAS_FREE` final val CE ≤ `SEQ_AUX` with entropy
> `0.9·log(N_r)` after training.

**Kill line:** `BIAS_FREE` worse than `SEQ_AUX` by > 0.02 nats, or entropy
< `0.9·log(N_r)` after training (balancer fails to prevent collapse).

**Artifacts:** `scripts/f6_moe_ablation.py`, `tests/test_moe_ablation.py`,
`artifacts/f6_moe_ablation/`.

**Smoke run logged (2026-07-30, commit `98f62e3`) — NOT a result.** All six arms ran end-to-end
(`artifacts/f6_moe_ablation/{json,md}`) at toy scale: 30 steps, d=32, synthetic data. Numbers
(coarse val_ce ≈ 1.926–1.928 vs fine ≈ 2.296–2.300; balance score ≥ 0.97 everywhere) validate
harness plumbing only; at this scale the granularity/balancer comparisons are vacuous and no
"coarse arms equivalent / fine degrades" conclusion may be cited from them. Table rows stay
`pending` until re-run at ≥1B real tokens per the taskspec.

## F2a · MTP train head (λ = 0.3)

Spec: `docs/FRONTIER_2026_TASKSPEC.md` §F2a. Shipped in commit `3b89119`
(`src/scratch_llm/mtp.py`, `model.py` `mtp_depth`/`_trunk`/`forward_train`,
`train.py` `mtp_loss_weight`; tests `tests/test_mtp.py`).

| quantity | prediction (pre-registered) | measured | status |
|---|---|---|---|
| MTP-head loss at init | ≈ log V | passes (`test_mtp_head_loss_at_init_is_log_vocab`) | [FACT] |
| `forward_train` main logits | == `forward()` | passes | [FACT] |
| `mtp_depth ∈ {0,1}` base forward | bit-identical, shared seed | passes (params + forward) | [FACT] |
| Δ = val_CE(λ=0.3) − val_CE(no-MTP) | ∈ [−0.02, +0.01] nats | **+0.0027 nats** (toy scale: tiny Markov corpus, 300 steps, CPU) | [FACT @ toy scale] |

**Kill line:** MTP aux degrades base val loss by > +0.01 nats — NOT hit at toy
scale. **Caveat:** the toy-scale Δ is not the d20 measurement; the bake-into-d20
decision stays falsifier-gated (nanochat's own MTP A/B failed, nanochat #481 —
see `docs/FRONTIER_2026_END_TO_END_PLAN.md` §2 S1).

## F12 · FineWeb-EDU vs ClimbMix-400B corpus ablation

Registered **before** running. Spec: `docs/FRONTIER_2026_TASKSPEC.md` §F12 and
`docs/FRONTIER_2026_END_TO_END_PLAN.md` §S0.
Thesis: at **35M params / 700M tokens / iso-FLOP** (`C ≈ 1.5×10¹⁷`), with the tokenizer retrained
**per corpus** on the same byte budget and identical decontamination/packing, **NVIDIA ClimbMix-400B**
([VERIFIED], Nemotron-CLIMB [2504.13161](https://arxiv.org/abs/2504.13161); HF card
[nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix); nanochat switch
[324e69c](https://github.com/karpathy/nanochat/commit/324e69c.patch), 2026-03-04:
2h46m → 2h01m (−27%), val_bpb 0.7465 → 0.7185) yields lower val_bpb and CORE than FineWeb-EDU-100B
on a shared, decontaminated held-out set.

| corpus | predicted final val_bpb | measured final val_bpb | predicted CORE | measured CORE | license / provenance | status |
|---|---|---|---|---|---|---|
| FineWeb-EDU-100B | baseline | — (pending) | baseline | — (pending) | FineWeb-EDU license (ODC-BY 1.0) | pending |
| ClimbMix-400B | < FineWeb-EDU bpb (predict Δ ≈ −0.010 to −0.030; nanochat saw −0.028, [324e69c](https://github.com/karpathy/nanochat/commit/324e69c.patch)) | — (pending) | ≤ FineWeb-EDU CORE | — (pending) | **CC BY-NC 4.0** — must be stated in A9 model card ([nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)) | pending |

**License note.** ClimbMix-400B is released under **CC BY-NC 4.0**
([HF card](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)). If F12 flips the d20 corpus
to ClimbMix, the A9 model card must state the license and the dataset source.

**Expected falsifier:** ClimbMix final val_bpb < FineWeb-EDU final val_bpb at iso-FLOP
(35M/700M, tokenizer retrained per corpus, shared decontaminated held-out set, bpb + CORE).

**Harness shipped 2026-07-31** (CPU-green): `eval/corpus_ablation.py` (`CorpusAblationArm`,
`split_held_out` + A0 13-gram decontam with logged overlap, `run_corpus_ablation`),
`scripts/tok_train.py` (per-corpus BPE with `--byte-budget`), ClimbMix staging in
`data/shards.py` (additive `--repo-id` generalization + stdlib GPT-2 detokenizer for
ClimbMix's tokenized-parquet layout — no raw-text column, [FACT hub-verified 2026-07-31]);
tests `tests/test_corpus_ablation.py` + `tests/test_tok_train.py`. Run CLI:
`scripts/f12_corpus_ablation.py` (toy-mode smoke ✅ CPU: per-arm BPE → shards → bpb →
incremental per-arm JSON → verdict JSON with falsifier evaluation). The run itself stays
**pending** on the standing box; table rows fill on completion.

**Kill line:** ClimbMix bpb ≥ FineWeb-EDU bpb at iso-FLOP ⇒ keep the banked FineWeb-EDU corpus;
do **not** stage ClimbMix for the d20.

**Artifacts:** `data/shards.py` (FineWeb-EDU arm), new ClimbMix staging path,
`scripts/tok_train.py` per-corpus BPE, `eval/corpus_ablation.py` (or reuse `eval/optimizer_race.py`),
`docs/RESULTS.md` §F12, `bench/RESULTS.md` §Frontier ablations.

## S3 · Scaling-law calibration sweep (pre-registration)

Registered **before** running (2026-07-31). Spec: `docs/FRONTIER_2026_END_TO_END_PLAN.md` §S3.
Thesis: our own recipe (Muon+AdamW, F12-winning corpus, vocab 32,768, untied) admits a clean
power-law fit at sweep scale, and the compute-optimal D:N it implies either confirms or
re-registers the d20's 9.6B-token budget *before* any paid run.

Grid: depths {4, 8, 12} × D:N {8, 20, 40} minus (12, 40) = points s1–s8 (largest: d12 @ ratio-20,
C ≈ 2.2e18, ~28 h standing box; s6/s8 droppable if budget trips). Fit `N_opt ∝ C^a`,
`D_opt ∝ C^b` in **bpb** on the pinned val set via `scaling/isoflop.py` (log-log, per-budget
min-pick).

| quantity | prediction (pre-registered) | measured | status |
|---|---|---|---|
| exponent sum a+b | ∈ [0.95, 1.05] (forced by C = 6ND) | — | pending |
| log-log fit R² (bpb) | ≥ 0.98 | — | pending |
| compute-optimal D:N | ≥ 15 ⇒ d20 stays ratio-20 (9.6B, deliberate inference-aware overtrain); < 15 ⇒ re-register D before P5 | — | pending |
| nanochat CORE-fit at d20's C = 2.77e19 | ≈ 0.195 — inside the pre-registered 0.19–0.22 band | — | pending |

**Kill:** s5–s6 (59M, ≤2.4B tokens) cannot beat the toy-corpus loss floor by a clear margin ⇒
data/recipe bug; stop, do not scale.

**d14/d16 escalation:** trigger-gated per E2E §S3(g) (T1 fit failure ⇒ d14; T2 P5-anchor
divergence > 0.01 bpb ⇒ d16; T3 F8/F10 sweep-scale win + RULER pass ⇒ d14 confirm; none ⇒ skip,
d16 stays the $90-abort fallback).

**Artifacts:** `src/scratch_llm/scaling/s3_sweep.py`, `scripts/s3_scaling_sweep.py`,
`artifacts/s3_scaling_sweep/`, `tests/test_s3_scaling_sweep.py`.
