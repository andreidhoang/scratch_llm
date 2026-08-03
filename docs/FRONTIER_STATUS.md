# Frontier 2026 — Live Status Board

> **Single-page aggregator.** This file is the fastest way to see what is done, what is next, and which
> detailed doc owns each rung. It is updated whenever any of the three plan docs change.
> 
> **Source-of-truth docs:**
> - Strategy + rung cards + kill criteria → [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md)
> - End-to-end pipeline (S0→S8) + scaling law + d20 gate → [`FRONTIER_2026_END_TO_END_PLAN.md`](FRONTIER_2026_END_TO_END_PLAN.md)
> - Buildable spec (file→change, tests, DoD) → [`FRONTIER_2026_TASKSPEC.md`](FRONTIER_2026_TASKSPEC.md)
> - Integrated entry point → [`FRONTIER_2026_MASTER_PLAN.md`](FRONTIER_2026_MASTER_PLAN.md)
> - Architecture × scaling-law program (when a re-sweep is/isn't needed) → [`FRONTIER_2026_ARCH_SCALING.md`](FRONTIER_2026_ARCH_SCALING.md)
> - d20 certainty plan (what's certified vs borrowed vs undefined; P5/d14 protocol) → [`FRONTIER_2026_D20_CERTAINTY_PLAN.md`](FRONTIER_2026_D20_CERTAINTY_PLAN.md)
> - Scaling-law program of record (post-S3 fit-engineering redesign; S3.5 gates → K3 family fit) → [`FRONTIER_2026_SCALING_PROGRAM.md`](FRONTIER_2026_SCALING_PROGRAM.md)

---

## North star

Own every layer of a language model, close the loop end-to-end into a talking d20 with an honest
public report card, and use that working baseline to run pre-registered, iso-FLOP ablations that test
the 2026 frontier's *contested* claims.

---

## Current next node

> **S3 scaling-law sweep s1–s7 ✅ DONE (2026-08-02, 12.09 GPU-h, ClimbMix corpus by operator override)** → **decisions CLOSED 2026-08-03: T1 = A (accept HOLD → P5) + d20 runs mtp_depth=0 (`docs/RESULTS.md` §S3.5)** → matched-batch d12@r8 rerun (free, ~4 GPU-h — §S3 amendment) → **P5 d12 dress rehearsal ($10–15, cap $20, gated on paid go-ahead)** → **8×H100 d20 (~$100, gated on P5 + user go-ahead)**

See `FRONTIER_2026_TASKSPEC.md` Next-node marker for exact commands and prerequisites.

F12 completed 2026-07-31: measured **KEEP FineWeb-EDU** (ClimbMix bpb 1.30205 ≥ FWE bpb 1.19197 at
iso-FLOP; kill criterion triggered). **Operator override:** S3/d20 will use **ClimbMix** anyway,
following the nanochat/Karpathy corpus choice; deviation logged in `docs/RESULTS.md` §F12 and
`bench/RESULTS.md` §Frontier ablations.

S3 completed 2026-08-02: all seven rungs landed (bpb 1.2553 → 0.9402). The pre-registered R² gate
**tripped** (R² 0.7709/0.8324 < 0.98 ⇒ fit rejected — grid-geometry zigzag of recorded D across
depth boundaries, not a recipe bug), so no scaling-law extrapolation is quoted; a+b = 1.0000 is
near-vacuous by C=6ND construction. d20 D:N **HOLDS at ratio-20 (9.6B tokens)** on fitted-interval
evidence (s6-vs-s7 iso-FLOP: bigger-N/ratio-8 wins by −0.0096 bpb) plus the deliberate
inference-aware overtrain registration. **Trigger T1 (d14 escalation) formally fired** — surfaced
for human decision; no paid GPU work without explicit go-ahead.

---

## Track A: close the loop into a chat model

| # | Rung | Status | Size / Cost | Owner doc | Blocker / next action |
|---|---|---|---|---|---|
| A1 | Real-corpus shards | ✅ Shipped | — | TASKSPEC §A1 | — |
| A0 | Decontamination gate | ✅ Shipped | — | TASKSPEC §A0 | — |
| A2 | Checkpoint chaining | ✅ Shipped | — | TASKSPEC §A2 | — |
| A3 | Chat template + specials | ✅ Shipped | — | TASKSPEC §A3 | — |
| A4 | Midtrain stage | ⬜ Spec done, unbuilt | — | TASKSPEC §A4 | Build midtrain shard writer + stage |
| A5 | SFT stage | ✅ CPU-green | — | TASKSPEC §A5 | Run on real d20 ckpt post-pretrain |
| A6 | Chat REPL / serve | ✅ CPU-green | — | TASKSPEC §A6 | Point at SFT'd d20 ckpt |
| A7 | Distributed d20 pretrain | ✅ Shipped | 480.4M @ 8×H100 | TASKSPEC §A7 | — |
| A8 | d20 rental runbook + guardrails | ✅ Shipped | ~$100 cap | TASKSPEC §A8 | Execute when P5 passes |
| A9 | Public release surface | ⬜ Post-d20 | — | TASKSPEC §A9 | Model card + repro bundle |

### d20 gate prerequisites (P1–P6)

| # | Gate | Status | Owner doc |
|---|---|---|---|
| P1 | SDPA/flash attention in training path | ✅ | TASKSPEC §P1 |
| P2 | ~10B-token shard streamer | ✅ | TASKSPEC §P2 |
| P3 | CORE 22-task suite | ✅ | TASKSPEC §P3 |
| P4 | Fused/compiled optimizer + LR transfer | Folds into P5 | TASKSPEC §P4 |
| P5 | d12 dress rehearsal on 1×H100 | 💰 Pending ~$10–15 | END_TO_END §S4 |
| P6 | Node-quality + abort guardrails ($90 abort → d16) | ✅ In runbook | TASKSPEC §P6 |

### d20 target spec

| Field | Value |
|---|---|
| Params | 480.4M |
| Vocab | 32,768 |
| Layers / d_model / heads | 20 / 1280 / 10 |
| Tokens | ~9.6B (ratio-20, deliberate overtrain) |
| Compute | ~2.77e19 FLOP |
| Cost | ~$48–100 (8×H100, 2–4h) |
| Target CORE | **0.23–0.25 (central ≈0.24)** — re-anchored 2026-08-02 vs the published ClimbMix curve (supersedes 0.19–0.22; KILL <0.15 unchanged) — `docs/RESULTS.md` §S4-pre |
| Attention | Full GQA/MHA (frozen at P5) |
| Optimizer | Muon+AdamW (ADOPTED, commit `f9e8f3b`) |
| Precision | bf16 (compile NaN on sm120 → validated on sm90/H100 in P5) |
| MTP head | **Not in d20** — decision of record 2026-08-03 (option a): `mtp_depth=0`, all certified constants stand; F2a/F2b retarget post-d20 (`docs/RESULTS.md` §S3.5) |

---

## Track B: frontier ablation study

### Tier 1 — scarce, differentiating (do first / do best)

| # | Rung | Status | Scale / Cost | Falsifier | Owner doc |
|---|---|---|---|---|---|
| F8 | DSA sparse attention | F8.1 ✅ mechanism shipped; F8.2 ⬜ wiring pending | ≤300M, standing box | Recall ≥0.95; +0.03 nats @8k; kill >0.1 | ABLATIONS §F8 |
| F10 | Hybrid linear attention (GDN) | F10.1 ✅ block shipped; F10.2 ⬜ wiring pending | ≤300M, standing box | 3:1 within +0.03 val loss; state ≥2× smaller; recall parity | ABLATIONS §F10 |
| F7 | RL "aha" reframed | F7a ✅ harness; F7b ✅ detector; F7c ✅ CPU-green control | Real run needs ~0.5–1.5B base | Random-reward control recovers most gain | ABLATIONS §F7 |
| F11 | Agentic / tool-use RL | ⬜ Pending | ≤300M harness; real base rental | Success-rate↑ with turns bounded; format-only control hacks | ABLATIONS §F11 |

### Tier 2 — table stakes done well

| # | Rung | Status | Scale / Cost | Falsifier | Owner doc |
|---|---|---|---|---|---|
| F1 | Muon vs tuned AdamW | Harness ✅; race run descoped; **Muon+AdamW ADOPTED** | 35M/700M, standing box | 1.1–1.4× band vs independently LR-tuned AdamW | ABLATIONS §F1 |
| F2a | MTP train head | ✅ Shipped | Baked into d20 | Δbpb ∈ [−0.02, +0.01]; kill >+0.01 | ABLATIONS §F2 |
| F2b | MTPDrafter | ⬜ Post-d20 | d20 ckpt | a2-acceptance ≥0.30 on prose, > n-gram | ABLATIONS §F2 |
| F4 | bf16 + torch.compile (+ NVFP4 stretch) | ✅ Wired; NaN guard on sm120 | d20 H100 re-validation in P5 | +10–20% MFU; train↔serve KL tolerance | ABLATIONS §F4 |

### Tier 3 — necessary but minimize

| # | Rung | Status | Scale / Cost | Falsifier | Owner doc |
|---|---|---|---|---|---|
| F3 | De-confound serving | ✅ Harness shipped; awaits trained ckpt | d20 ckpt | Prose acceptance <10%; code/JSON 40–60% | ABLATIONS §F3 |
| F5 | MLA for real | ⬜ First-slice pending | 0.2–0.5B, standing box | +0.02 val loss vs GQA-8; KV ≥3×; kill >0.05 | ABLATIONS §F5 |
| F6 | MoE balancing | Harness ✅ (`98f62e3`); smoke 30-step (vacuous); real run ≥1B pending | 30–300M, standing box | BIAS_FREE ≤ SEQ_AUX val CE; entropy >0.9·log N | ABLATIONS §F6 |
| F9 | QK-clip / logit guard | ⚠️ Instrument shipped; **S_max falsifier PENDING** — no instrumented run has reported S_max (F1 died step 4,880; the S3 sweep did not carry `track_attn_logits`). Must ride the next instrumented run (P5 or later) | Sub-1B | Max logit <~30 with qk_norm | ABLATIONS §F9 |
| F12 | ClimbMix vs FineWeb-EDU corpus | ✅ **DONE** — kill fired (+0.110 bpb, FWE better) but **OPERATOR OVERRIDE: ClimbMix FINAL 2026-08-02** for S3/d20 | 35M/700M, standing box | ClimbMix bpb 1.30205 ≥ FWE bpb 1.19197; kill triggered, overridden | ABLATIONS §F12 |

---

## S3 scaling law ladder

| Rung | Depth | Params | Tokens | Compute | Status |
|---|---|---|---|---|---|
| s1 | 4 | 19.99M | 159.9M | 1.92e16 | ✅ bpb 1.2553 (batch 8) |
| s2 | 4 | 19.99M | 399.8M | 4.80e16 | ✅ bpb 1.1772 (batch 8) |
| s3 | 4 | 19.99M | 799.7M | 9.59e16 | ✅ bpb 1.1408 (batch 8) |
| s4 | 8 | 59.26M | 474.0M | 1.69e17 | ✅ bpb 1.0124 (batch 8) |
| s5 | 8 | 59.26M | 1.18B | 4.21e17 | ✅ bpb 0.9676 (batch 8) |
| s6 | 8 | 59.26M | 2.37B | 8.43e17 | ✅ bpb 0.9498 (batch 8) |
| s7 | 12 | 135.29M | 1.08B | 8.79e17 | ✅ bpb 0.9402 (batch 4) |
| s8 | 12 | 135.29M | 2.71B | 2.20e18 | ⬜ Re-homed to S3.5 stage 2 (free box, post-P5 recipe-v1) — `FRONTIER_2026_SCALING_PROGRAM.md` §3 |

**DONE 2026-08-02 (12.09 GPU-h):** fit `N*(C)`/`D*(C)` raised the pre-registered R² gate
(R²_N 0.7709 / R²_D 0.8324 < 0.98 — grid-geometry zigzag, not a recipe bug; a+b = 1.0000
near-vacuous). No extrapolation quoted. d20 D:N **HOLD at ratio-20 (9.6B)** on fitted-interval
evidence (s6-vs-s7 iso-FLOP favors bigger-N by −0.0096 bpb) + the inference-aware overtrain
registration. **T1 (d14) formally fired — surfaced for human decision.** Full writeup:
`docs/RESULTS.md` §S3, `bench/RESULTS.md` §Frontier ablations.

Owner: `FRONTIER_2026_END_TO_END_PLAN.md` §S3 + `scaling/isoflop.py`.

---

## EV-ranked priority (2026 scarce signal)

1. **F8 + F10** — attention efficiency (contested frontier)
2. **F7 reframed + F11** — RL honesty + agentic (#1 lab priority)
3. **F12** — data > optimizer (gates d20 corpus)
4. **F1** — optimizer methodology (ADOPTED, harness kept)
5. **F2 + F4** — table stakes done well
6. **F3 / F5 / F6 / F9** — necessary hygiene, minimize investment

---

## Honesty ledger — recent updates

| Date | Change | Why |
|---|---|---|
| 2026-07-09 | Muon deflated: 1.4×→1.1× vs tuned AdamW | Wen 2509.02046 verified |
| 2026-07-09 | RL "aha" reframed to debunk-with-control | GRPO clipping artifact |
| 2026-07-09 | F8 DSA promoted to core; F10 linear-hybrid added | Attention frontier contested |
| 2026-07-09 | F11 agentic RL added | #1 stated 2026 lab priority |
| 2026-07-30 | F12 ClimbMix data ablation added | Data > optimizer; gates d20 corpus |
| 2026-07-30 | d20 ratio-20 overtrain re-registered | Inference-aware allocation, not Chinchilla folklore |
| 2026-07-31 | F12 measured: KEEP FineWeb-EDU | ClimbMix bpb 1.30205 ≥ FWE 1.19197 at iso-FLOP; falsifier not confirmed |
| 2026-07-31 | S3 grid fixed to include qk_norm params | Planned N now matches the instantiated GPU training model; batch-size deviations logged for OOM safety |
| 2026-08-02 | S3 s1–s2 landed: bpb 1.2553 / 1.1772 | Grid fix verified on GPU; sweep continuing s3→s7 |
| 2026-08-02 | `FRONTIER_2026_ARCH_SCALING.md` added | One law per recipe backbone; per-architecture anchors, not re-sweeps; MoE the sole mini-fit exception |
| 2026-08-02 | `stage_pretrain` device policy pinned (qk_norm on; SDPA on cuda) + stale checkpoint-chain test updated | Green CI restored after the S3 grid fix |
| 2026-08-02 | F12 decision FINAL: operator OVERRIDE → **ClimbMix** for S3/d20 | Measurement-only verdict was KEEP FineWeb-EDU (+0.110 bpb at 35M/700M); override follows nanochat's larger-scale result; confirmation arm declined; A9 model card must state ClimbMix's CC BY-NC 4.0 license |
| 2026-08-02 | S3 sweep s1–s7 DONE (12.09 GPU-h); R² gate tripped (0.7709/0.8324 < 0.98 ⇒ fit rejected), T1 fired; d20 D:N HOLD ratio-20 on interval evidence | Grid-geometry D zigzag breaks the power-law fit by construction; s6-vs-s7 iso-FLOP favors bigger-N; no extrapolation quoted; d14 decision escalated to human |
| 2026-08-02 | d20 CORE band re-anchored 0.19–0.22 → **0.23–0.25 (central 0.24)** | Band was calibrated to the FWE-era nanochat fit; current ClimbMix curve sits +0.052..+0.064 above it at matched C (real recipe progress, 3–8× noise floor); GPT-2 XL parity is the stretch case, not base case |
| 2026-08-02 | `FRONTIER_2026_D20_CERTAINTY_PLAN.md` added | One-page answer to "how are size/data/HPs defined": N anchored, D deliberate-overtrain (uncertified, d14 is the repair option), HPs undefined until the P5 sweep; decision matrix recommends accept-HOLD → P5 |
| 2026-08-03 | `FRONTIER_2026_SCALING_PROGRAM.md` added; §S3.5 gates pre-registered in `docs/RESULTS.md`; P5 sweep protocol pinned (horizon/tie-break/confirmation/3-point width probe; $20 cap) | Senior-review pass: post-S3 fit-engineering redesign (joint Huber estimator, bootstrap+LOO gates, recipe-v0→v1 reconciliation, s8 re-homed to free-box stage 2) and the path to the K3 family-fit attention decision |
| 2026-08-03 | ⚠ MTP recipe-surface drift flagged — **human decision required before P5** | d20 spec says "MTP baked into pretrain (F2a)" but `speedrun.py` never sets `mtp_depth` (default 0; S3 ran without it); certified constants (480.4M / 2.77e19 / T2 band) assume mtp_depth=0 — options registered in `docs/RESULTS.md` §S3.5 |
| 2026-08-03 | **Decisions CLOSED: T1 = A (accept HOLD → P5) and d20 runs mtp_depth=0** (F2a/F2b retarget post-d20) | Operator-delegated senior-review calls: hyperparameters are the only unbounded unknown and only P5 closes them; the d20's value is anchor comparability and every certified constant assumes mtp_depth=0 — records in `docs/RESULTS.md` §S3.5 + `FRONTIER_2026_D20_CERTAINTY_PLAN.md` §8 |

---

## How to update this file

When a rung ships or a doc changes:

1. Update the `Status` column here.
2. Update the `Next node` section if the critical path moves.
3. Do **not** duplicate rung-card detail here — link to the owner doc.
4. Keep this file <300 lines so it loads fast.

*Last updated: 2026-08-03*
