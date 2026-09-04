# K3 ABLATIONS — pre-registered research program

> Chartered 2026-07-31. **Methodology (read first): [`SCALING_LADDER.md`](SCALING_LADDER.md) —
> where this program sits in the frontier ablate→fit→freeze→re-fit sequence (3-angle primary-source
> pass: DeepSeek, Moonshot, Meta/MiniMax/open labs).** Frame: the K3 release opened two gaps in
> the public record, and this program attacks exactly those — nothing else.
> **Gap 1 (science):** Moonshot measured the interaction structure of their stack but published
> none of it. Whether KDA, AttnRes, and LatentMoE/QB are a *menu* (independent levers) or a
> *package* (only the composition works) is knowable at 0.4B scale and currently known by no
> one public.
> **Gap 2 (artifacts):** 1.56 TB of open frontier hybrid-linear weights has never existed
> before. The checkpoint's small tensors answer real questions with zero GPU.
> Anti-goals (refused at charter): reproducing K3 headline numbers · benchmark-maxing the
> d-series · Per-Head Muon quality arms at our scale (paper itself claims large-scale benefit
> only) · any arm without iso-FLOP matching and a pre-registered kill line.
>
> Rules (repo discipline): every arm has a falsifiable prediction + kill line registered in
> `docs/PRE_REGISTRATIONS.md §K3` **before** launch · iso-FLOP matching (active params ±2% via
> `k3/param_count.py`, tokens equalized on 6·N_active·D, measured FLOPs logged) · fixed seeds ·
> val_bpb primary, CORE-style MC secondary, retrieval probes (induction + needle @ 2K/8K/32K)
> for mechanism claims · ledger labels: measured-by-us / reported / not-verified.

## Relationship to the F-front — ONE program, two arm families

This is not a second ablation program; it is a second *axis set* in the existing one
(`docs/FRONTIER_2026_ABLATIONS.md`). **Shared:** the speedrun spine + eval report card +
iso-FLOP discipline + seeds + the F12-chosen corpus (**ClimbMix** — measurement winner was
FineWeb-EDU; operator override, decision FINAL 2026-08-02 — one harness); `docs/PRE_REGISTRATIONS.md` (one
pre-registration ledger, same format — F-sections and §K3 side by side); the d-series (one
control family — the all-minus corner of the shared factorial space); S3/P5 (one budget gate).

**Converged arms (run once, not twice):**

| F-arm | becomes / feeds | how |
|---|---|---|
| F10 hybrid linear attn | R1 axis A **+** R2 GDN point | GDN arm already built = the per-head-decay comparison; KDA arm is F10's upgrade, not a duplicate |
| F6 MoE balancing | R1 axis C level − | sign-step bias runs on F6's harness + metrics (entropy, load variance); QB is the level + extension |
| F5 MLA-real | `core/gated_mla.py` heritage | K3 deltas are NoPE + full-rank gate on top of the F5 absorption work |
| F1 Muon | Per-Head Muon follow-on | correctness-only at our scale (anti-goal: quality claims) |
| F2 MTP | K10.1 DSpark/EAGLE-3 draft | acceptance harness (F3) reused |

**Untouched F-arms (different questions, continue as planned):** F3 serving de-confound ·
F4 bf16/compile · F7 GRPO "aha" · F8 DSA · F11 agentic RL. (F12 corpus is DONE — decision
FINAL 2026-08-02: ClimbMix by operator override; S3 runs on it.)

**Never merges:** (1) the architectures — the factorial exists to keep single-variable
separation; no Frankenstein d20-with-KDA-parts; (2) the `k3/core/` hand-built boundary —
F-series code stays regular delegated/paired track.

---

## R0 — Anatomy of Kimi K3 (zero-cost, CPU, census COMPLETE 2026-07-31 — 417 small tensors, all 96 shards)

Method: HTTP range reads against safetensors layout (`scripts/k3_fetch_tensors.py` — index →
shard headers → exact byte ranges; BF16 decoded by bit-shift to FP32). No shard downloads.
Outputs: `artifacts/k3_anatomy/` (census.json, tensors/*.npy, anatomy_stats.json).
Side effect: census param count closes FACTS A18 to tensor level.

| Set | Tensors (count) | Question it answers |
|---|---|---|
| `qb` | `gate.e_score_correction_bias` (92 × 896) | What load-balancing equilibrium did QB actually reach at 2.8T? (Ground truth for R1-axis-3 and the Tier-2 QB arm.) |
| `attnres` | `{self_attention,mlp,output_attn}_res_proj.weight` (187 × 7168) | Depth-reuse map: which depths does a frontier model retrieve from, per sublayer? First public map at this scale. |
| `decay` | `self_attn.A_log` (69 × 128 — per-dim, census-measured; this table said 69 × 96 pre-correction 2026-08-03 — the per-head framing does not explain the [128] shapes, semantics OPEN, FACTS A18), `self_attn.dt_bias` (69 × 12288) | What decay timescales did 2.8T training choose, per layer/head/channel? Constrains the g_min question (R2). |

**First observations (measured by us, 2026-07-31, smoke = layers 0–1; full census COMPLETE — FACTS A19):**

- `layers.0.self_attention_res_proj.weight` is **exactly all-zero** — std 0.0000. With only
  the embedding available as a source, softmax over a single entry is constant 1 for any query,
  so the first layer's pseudo-query receives no gradient and stays at init. The checkpoint
  itself confirms the Eq. 8–10 semantics. (Prediction: layers ≥ 1 are non-zero; verify at census.)
- QB bias (layer 1): mean ≈ 0 (by the mean-removal construction), std 0.020, range
  [−0.084, +0.029] — near-uniform equilibrium with a slight negative skew, NOT collapsed.
- `dt_bias` mean ≈ −4.6 at layers 0–1 (ranges down to −9): under `g = g_min·σ(e^A·z)`, strongly
  negative decay logits push α → 1 — hypothesis (soft): **the default learned behavior is long
  retention, with input-dependent modulation doing the forgetting**. Test across all 69 KDA
  layers at census; look for layers where this flips.
- `A_log` means diverge already at layers 0 vs 1 (−0.17 vs +0.05) → per-layer timescales are
  learned apart early.

## R1 — The interaction factorial (FLAGSHIP; launches after `core/kda.py` PROVEN)

Question: do the three information-flow mechanisms compose **additively** (Moonshot's
"complementary dimensions" framing) or do they **substitute**? Design: 2×2×2 factorial on the
mini-K3 d12 skeleton, d-series as control. mini_k3_d12 IS the (KDA, AttnRes, LatentMoE) corner;
the (GQA, plain, standard-MoE) corner ≈ the repo's current d-family config → 6 new runs +
2 shared corners + control = 9.

| Axis | Level − | Level + |
|---|---|---|
| A · attention | GQA + RoPE (control family) | KDA-hybrid 3:1 (NoPE) |
| B · depth | plain residuals | Block AttnRes (block 4) |
| C · width | standard full-width MoE + sign-step bias | LatentMoE (0.5×) + Quantile Balancing |

Param-matching rule for axis C: solve standard-MoE expert width so total and active params
match the LatentMoE corner within ±2% (`param_count.py` computes; record the solved width in
the run card). All arms: 64 experts top-4, 2 shared, same vocab/data/order/seeds.

**Arm protocol (verified frontier practice, SCALING_LADDER.md §1–2):**

- **HP inheritance, verbatim.** Every challenger arm runs the control family's hyperparameters
  unmodified (Moonshot: KDA "adhered strictly to the MLA training configuration without any
  modifications"; AttnRes: "this setup intentionally favors the baseline"; SmolLM3: HPs fixed
  across arms). A challenger win under inherited HPs is conservative; a challenger win under
  self-tuned HPs is uninterpretable (Wen-I Muon lesson). Per-arm HP tuning is FORBIDDEN at R1 —
  it happens once, for the frozen family, at Rung 4b (K3 §3.2 rule).
- **Seeds.** Kill lines stay single-seed (negatives are trustworthy — HF playbook rule). The
  two factorial CORNERS ((A−B−C−) control and (A+B+C+) mini-K3) run **2 seeds**; any adoption
  claim requires both seeds plus the Rung-2 family curve. Single-seed results are registered
  as directional only.
- **Decay telemetry.** Every KDA arm logs its learned `dt_bias` / `A_log` distributions
  post-training, compared against the R0 census (dt_bias ≈ −4.63 ± 0.05 across all 69 KDA
  layers at 2.8T ⇒ long-retention default). This is the free R0→R2 loop-closer: it tests
  whether the g_min floor ever *binds* in practice, per arm, at our scale.
- **Probes are mandatory, not secondary.** needle + induction @ 2K/8K/32K on every axis-A arm
  (M2 lesson: standard losses stay silent on retrieval/multi-hop deficits; OlmPool 2026: arch
  choices cost up to 47% long-context while standard benchmarks stay flat).

Pre-registered predictions (kill lines in RESULTS.md §K3):

- **P1 (axis A):** KDA-hybrid ≥ GQA at iso-FLOP by ≥ 1% val_bpb, gap growing with context
  length. Confidence: HIGH (replicates Kimi Linear @ ~1/100 scale — if it fails at our scale
  that is itself the headline).
- **P2 (A×B interaction — the open question):** AttnRes's marginal gain is LARGER on the GQA
  corner than on the KDA corner (substitution: KDA's recurrent state already enriches each
  layer's representation, shrinking the value of depth-retrieval). The reverse story (AttnRes
  re-exposes early, less-decayed KDA states → complement) is equally coherent a priori — this
  is precisely why the arm matters. Confidence: MEDIUM; a null result is also publishable
  (additivity holds at 0.4B, consistent with Moonshot's framing).
- **P3 (axis C at 64×4):** LatentMoE+QB ≈ standard MoE + sign-step within ±0.5% bpb.
  Confidence: HIGH (QB's benefit is regime-dependent; 64 experts is not the regime).

## R2 — Decay-floor sweep + the KDA/GDN comparison (cheapest novel science)

Base config: the (KDA, plain, standard-MoE) corner from R1 — its run doubles as the g_min=−5
point. Arms: g_min ∈ {−3, −7, −10, −20} (4 new runs) + GDN arm (per-head scalar decay,
negative-softplus map — the repo's tested `linear_attn.py` block swapped into the identical
skeleton; 1 run).

- **P4:** val_bpb flat for g_min ∈ [−3, −7]; degradation at ≤ −10. Confidence: MEDIUM.
  **Numerical sub-prediction (directly tests the paper's §2.1.1 motivation):** at g_min ≤ −10
  the 16-token-tile cumulative log-decay crosses the BF16 reciprocal range (e^160 > BF16 max
  ≈ e^88) — we expect measurable instability/overflow in a bf16 chunkwise path that an fp32
  path survives. If −10 trains *cleanly* in bf16, the paper's hardware rationale is weaker than
  claimed; if it blows up, we've reproduced the constraint that forced the design. Either way
  a measured result.
  **Floor-binding sub-prediction (from R0, free):** R0 found dt_bias ≈ −4.63 ± 0.05 across
  all 69 KDA layers at 2.8T — the learned default is long retention, far from any floor. If
  our g_min ∈ {−3, −7} arms also converge to dt_bias ≪ 0 with the floor never active, the
  floor's practical role at our scale is nil and P4's flat region is explained by
  non-bindingness, not robustness — a different (and more interesting) claim. Per-arm decay
  telemetry (R1 protocol) makes this check zero-cost.
- **P5:** KDA (per-channel) > GDN (per-head) by ≥ 0.5% val_bpb at iso-FLOP. Confidence:
  MEDIUM-HIGH (Kimi Linear's central claim, third-party scale check).

## Tier 2 (fund only after R1/R2 land)

- **Ratio sweep:** 1:1 and 7:1 (3:1 shared with R1) + retrieval probes. **P6:** 3:1 ≥ 1:1 on
  loss; 7:1 degrades needle@32K disproportionately. Confidence: MEDIUM.
- **SiTU × low-precision:** SiTU vs SwiGLU × {bf16, fake-quant MXFP8 activations} — short
  continuation runs, not full pretraining. **P7:** null in bf16; SiTU wins under fake-quant
  (the designed-in interaction; SwiGLU's unbounded outliers quantize worse). Confidence: MEDIUM.
- **QB regime push:** 256 experts × top-4 at fixed width. **P8:** null at 64×4 (replicates P3),
  QB > sign-step at 256×4 on load-variance + bpb. Confidence: MEDIUM.

## Budget & sequencing

~15 training runs + short continuations: **$150–300** on the P5/d20 rental tier; S3
scaling-law calibration gates the exact token budgets (existing discipline). **Arm horizon
(pinned 2026-08-03, post-S3):** with the S3 fit rejected, budgets are set by registration,
not by a fitted law — every R1/R2 arm runs **d12 @ ratio-20 (≈2.71B tokens, C ≈ 2.2e18)**
with the control family's full production schedule shape (same warmup fraction, same decay
form); ~9–11 GPU-h per arm on the standing box, free-box-first before any rental. Sequence:
**R0 now** (script shipped + verified live 2026-07-31; census COMPLETE — 417 tensors, all 96
shards, FACTS A19) → `core/kda.py`
hand-built + PROVEN → K6 wiring (mini-K3 on the speedrun spine) → **R1 → R2** → Tier 2.
d-series (S3/d20) stays on schedule as the control family — do not bend it.

Contribution artifact at the end: "Anatomy and interaction of the K3 stack: controlled
evidence at 0.4B" — pre-registered predictions vs measured outcomes, the R0 weight maps
(QB equilibrium, depth-reuse, learned timescales), and open code/data. This is the study
frontier labs hold internally and have not published.
