# Frontier 2026 — Architecture × Scaling-Law Program

> **Doc role.** The question that came up while S3 was running: *if we change the architecture
> (MLA / MoE / GDN-hybrid / DSA), do we have to re-run the whole S3 sweep for each architecture?*
> This is the senior-research-team answer, specified to rung level: the first-principles
> reasoning, the literature it stands on, the per-architecture decision matrix, and the concrete
> post-S3 rung DAG. Companion docs: [`FRONTIER_STATUS.md`](FRONTIER_STATUS.md) (board),
> [`FRONTIER_2026_ABLATIONS.md`](FRONTIER_2026_ABLATIONS.md) (rung cards + falsifiers),
> [`FRONTIER_2026_END_TO_END_PLAN.md`](FRONTIER_2026_END_TO_END_PLAN.md) §S3 (the sweep itself),
> [`docs/RESULTS.md`](RESULTS.md) §S3 (pre-registration).

---

## §0 — The verdict in one paragraph

**No — one full law per recipe backbone; per-variant anchor points, not re-sweeps; a small
joint fit only for MoE, the one variant that changes the compute model.** A scaling law is a
property of a *recipe* = {corpus, tokenizer, optimizer + HP-transfer rule, architecture
family, precision, context length, LR schedule}. The literature's load-bearing finding is that
architecture changes move the **offset** of the loss-vs-compute power law (the irreducible
loss `E` and the multipliers `A`, `B`), not the **exponents** — the slope is a property of the
data domain, not the model [1][2][3]. What *does* move exponents is **data quality** [4] —
which is exactly why F12 (the corpus decision) gated S3, and why the operator's ClimbMix
override means the fit now running *is* the ClimbMix law. Attention variants (MLA, DSA,
GDN-hybrid) keep the dense compute model `C = 6ND` at our fixed ctx 2048 and are therefore
compared by **iso-FLOP anchor pairs** against the S3 reference points. **MoE is the single
exception**: `C = 6·N_active·D` while capacity scales with `N_total`, so the dense law does
not apply and the joint forms from the MoE scaling literature [5][6] need a small dedicated
grid. Every small-scale win is trigger-gated (E2E §S3(g) T3) at d14 before it can touch the
frozen d20 recipe.

---

## §1 — First principles: what the S3 fit measures (and what it cannot)

### 1.1 The object being fit

Chinchilla's parametric form: `L(N,D) = E + A/N^α + B/D^β`. At fixed compute `C = 6ND`, one
of `(N, D)` is redundant — `D = C/(6N)` — so the compute-optimal allocations obey
`N_opt ∝ C^a`, `D_opt ∝ C^b` with **`a + b = 1` forced by the identity**. A fitted sum off 1
means the grid was too narrow, the loss floor `E` leaked into the reducible term, or training
was off-recipe. That is why `scaling/isoflop.py::check_exponent_sum` raises loudly instead of
warning.

Our harness (`scaling/s3_sweep.py`) implements Hoffmann's **Approach 2 (isoFLOP profiling)**
with deliberate simplifications:

- **Per-budget argmin on val_bpb** — one `(C, N_opt)` point per compute budget, then a log-log
  linear fit. We do *not* use Approach 3 (joint parametric L-BFGS on `E, A, B, α, β`): the
  Besiroglu et al. replication [7] showed Chinchilla's Method 3 was fragile (Huber losses
  averaged instead of summed → premature optimizer termination; reported CIs implausibly
  narrow). The replication recovered `α ≈ 0.35` and the central conclusion survived — but the
  lesson for a small lab is: **use the robust estimator, and let structural gates (a+b, R²)
  carry the falsification load.**
- **Log-log space, never raw** — raw least squares on a power law is heteroscedastic; the
  largest budget dominates the residual and drags the exponent (`isoflop.py` module docstring,
  with a planted counterexample in the tests).
- **Fit axis is bpb, never raw CE** — bpb is tokenizer-invariant (required for any comparison
  involving the per-corpus tokenizers of F12), and low-noise: CORE's run-to-run spread at this
  scale is ±0.008–0.016 (nanochat repeated one run 7× identically: CORE 0.2512–0.2677,
  discussion #481), so CORE never enters the fit.
- **`D_opt` is the min-picked run's *recorded* token budget, not the `C/(6N)` bridge** — the
  bridge makes `a + b ≡ 1` identically and would mute exactly the driver bugs the gate exists
  to catch (`s3_sweep.py` module docstring). The bridge is still reported as a consistency
  diagnostic (`d_opts_bridge`).

### 1.2 The exponents-vs-offsets finding (the load-bearing fact)

- **Hestness et al. 2017** [1] (Baidu, pre-LLM scaling survey across NMT, image, speech, LM):
  model improvements *shift the error curve* but do not change the power-law exponent —
  *"architecture changes the offset (`E`) of the fit, not the exponent (`α`); the slope is a
  property of the problem domain."*
- **Rosenfeld et al. 2020** [2]: the joint form `L(D,N) = A/N^α + B/D^β + E` transfers across
  ResNet/WRN/LSTM/Transformer and SGD/Adam — the *functional form* is architecture-free; the
  *constants* absorb the architecture and optimizer.
- **DeepSeek LLM 2024** [4]: exponents **do** move with data quality (their fits across
  corpora differ materially), and optimal batch/LR have a wide stable plateau.
- Weng's 2026 survey [3] consolidates the same picture.

Consequence for us: **the backbone that needs exactly one law is (data × tokenizer ×
optimizer) at a fixed architecture family.** F12 chose the corpus (operator: ClimbMix); the
tokenizer is retrained per corpus at fixed byte budget; the optimizer is ADOPTED
(Muon+AdamW, commit `f9e8f3b`). The S3 sweep running now is therefore *the* law for the d20
recipe. Architecture variants ride on top as anchors (§3) — re-fitting per variant would
spend 3× the compute to re-measure the quantity the literature says barely moves.

### 1.3 The `C = 6ND` approximation at our scale — an honest caveat

`6ND` counts parameter matmuls only (2 FLOPs/param/token forward + 4 backward). The attention
quadratic term adds ≈ `12·T·d` FLOPs/layer/token (fwd+bwd) at context `T`, i.e. a fraction
`T/(6d)` of the parameter term. With our `d_model = 64·depth` shapes at ctx 2048:

| depth | d_model | T/(6d) | attn-quadratic share of true FLOPs |
|---|---|---|---|
| 4 | 256 | 1.33 | ~57% |
| 8 | 512 | 0.67 | ~40% |
| 12 | 768 | 0.44 | ~31% |
| 20 | 1280 | 0.27 | ~21% |

So the narrow sweep models spend up to ~half their *true* FLOPs on attention, and `6ND`
undercounts more at small depth. This does **not** break the fit: `C` is a bookkeeping axis,
ctx is constant across the grid, and the D:N decision rule is a *ratio* in which most of the
per-depth distortion cancels. It **does** matter for (a) any cross-architecture comparison
whose claim lives at different ctx (F10's long-context claim — §3), and (b) absolute-FLOPs
comparisons against nanochat's FP8 d24 leaderboard point. Recorded here so nobody treats our
`C` axis as metrology-grade.

### 1.4 Batch size (why the OOM deviation is second-order)

The pre-registered batch plan was reduced for VRAM safety (s1–s6 batch 8, s7 batch 4→2;
logged in `docs/RESULTS.md` §S3). Justification, first principles: the **critical batch
size** (gradient-noise-scale, McCandlish 2018 [8]; LLM-scale confirmations in [9][10]) at
`C ≈ 1e16–1e18` FLOPs is orders of magnitude above our 4k–16k tokens/step, and it *grows* as
loss falls. Below the CBS, at fixed `D` and fixed recipe, batch size changes only optimization
noise per step (smaller batch = more steps, same tokens); DeepSeek LLM [4] found a wide
near-optimal HP plateau around it. The iso-FLOP comparisons hold `C` constant, so the
deviation is second-order and the fit tolerates it. What would **not** be tolerable
mid-sweep: changing the LR schedule, ctx, corpus, or seed policy — those move the backbone.

### 1.5 The overtraining regime

Our grid spans D:N = 8–40; the d20 is registered at ratio 20 as a *deliberate* inference-aware
overtrain (re-registered 2026-07-30). This is on the predictable manifold: Gadre et al. 2024
[11] trained 0.011–0.411B models at token multipliers 20–640 and found **near-constant
exponent** in the over-trained regime ("parallel lines" in reducible loss) — scaling stays
predictable far past compute-optimal; Sardana & Frankle [12] formalize *why* a model that
will serve many requests should be trained smaller-and-longer than Chinchilla-optimal
(Sardana v2 explores to ~10⁴ tokens/param). The S3 decision rule (re-register D only if the
measured compute-optimal ratio < 15) is the honest version of "decide, don't inherit."

---

## §2 — The one-backbone doctrine (what frontier labs actually do)

- **One law per recipe generation.** DeepSeek fit its law once [4] and carried it into V2/V3
  with architecture adjustments (MLA, fine-grained MoE) rather than re-fitting from zero;
  Qwen3-Next and Kimi Linear validated hybrid-vs-full attention at a small number of
  matched-FLOP scales (Kimi Linear: matched 1.4T-token runs [13] — one anchor *at scale*,
  backed by small-scale sweeps — not a per-architecture Chinchilla redo); nanochat re-anchored
  its curve when the corpus switched (FineWeb-EDU → ClimbMix, commit `324e69c`) and reports
  its own compute-optimal D:N ≈ 8–10.5 (discussion #420) — a corpus-side re-anchor, not an
  architecture-side one.
- **Cost math on our box.** Full 7-point sweep ≈ 2–3 GPU-days. Re-sweeping per attention
  variant (F5, F8, F10) = 3 × 2–3 GPU-days to re-measure the exponent the literature says
  barely moves. An anchor pair (d4-r20 + d8-r20, matched N/D/seed against s2/s5) ≈ 0.5–1
  GPU-day per variant and measures the quantity that *does* move: the offset `Δbpb` at
  iso-FLOP.
- **The discipline.** Sweeps answer *"how much N vs D at budget C"*; anchors answer *"is
  variant X better at the same C"*. Confusing the two questions is how small labs burn weeks.

---

## §3 — Per-architecture decision matrix

| variant | rung | changes the compute model? | re-sweep? | protocol | gates (rung card) | escalation |
|---|---|---|---|---|---|---|
| MLA | F5 | **No** — FLOPs/param ≈ unchanged; the latent KV is a *memory/serving* variable, not training compute | No | iso-FLOP anchor pairs + KV-bytes measurement | within +0.02 val loss of GQA-8; KV ≥3× smaller; kill >+0.05 | none — its win is serving-side |
| DSA sparse attention | F8 | **≈ No** at training — the lightning indexer adds small per-token compute; top-k sparsity is a decode/long-ctx phenomenon | No | anchors at **8k ctx** + attention-mass recall probe | recall ≥ 0.95; within +0.03 nats @8k; kill >0.1 | T3: d14 confirm if it wins *and* passes RULER |
| GDN 3:1 hybrid | F10 | Changes the **ctx-dependence**, not the ctx-2048 point (§1.3) | No | iso-param 3:1 anchor A/B + state-bytes + decode crossover; long-ctx probe **mandatory** | within +0.03 val loss; state/token ≥2× smaller; kill >0.1 or no state win | T3: d14 confirm |
| MoE balancing | F6 | **Yes** — `C = 6·N_active·D`; capacity scales with `N_total` | **Mini joint fit** | 2 scales × {sparsity, granularity} grid (§3.4) | BIAS_FREE ≤ SEQ_AUX val CE; router entropy >0.9·log N | re-derive d20 compute accounting before any MoE d-class run |

### 3.1 F5 MLA — the easy case

MLA re-parametrizes the KV projection through a low-rank latent (DeepSeek-V2/V3); per-token
training FLOPs per parameter are essentially unchanged, and the KV-cache compression is a
serving-memory property. The dense `C = 6ND` law therefore applies as-is; the ablation is a
pure offset question: same N, same D, same seed → compare final val_bpb against the s2/s5
reference points, plus the KV-bytes check (≥3× smaller than GQA-8). Its falsifier (+0.02 nats)
and kill (>+0.05) are already pre-registered in the F5 rung card. **No scaling content.**

### 3.2 F8 DSA — a ctx-dependent correction, not a new law

DSA (DeepSeek-V3.2-style) keeps dense softmax attention but *selects* top-k keys via a cheap
lightning indexer trained by KL to the dense attention distribution. Training-time compute is
dense + indexer overhead (small); the FLOP win materializes at long ctx and decode. At our
sweep ctx 2048 the mechanism is nearly neutral on the `6ND` axis — which is why the rung card
runs the A/B at **8k** with the recall probe (≥0.95 held-out attention mass at k=2048).
Protocol: anchor pairs at 8k vs a dense-8k baseline (one extra baseline run — the s-points at
2048 are *not* the right comparator), then T3.

### 3.3 F10 GDN 3:1 — the win lives past our sweep's ctx, so the probe must go there

The 2026 attention frontier settled on *hybrids*: Kimi Linear (KDA:full = 3:1, 75% KV cut,
up to 6.3× decode at 1M ctx, beats full MLA at matched 1.4T tokens [13]), Qwen3-Next
(GDN:full = 3:1), Solar Open 2 (3:1, β ∈ (0,2) negative-eigenvalue extension), Nemotron-H
(Mamba-2:FA 9:1), MiniMax M1 (lightning:FA 7:1) [14]. The retained full-attention layers carry
exact recall; the linear layers carry local structure at fixed state size.

Two consequences for us. (a) At ctx 2048 the attention quadratic is 20–57% of true FLOPs
(§1.3), so a ctx-2048 iso-`6ND` A/B is *approximately* apples-to-apples for both arms — the
quality offset is measurable on the standing box. (b) But the architecture's actual claim
(fixed state vs linearly-growing KV; decode throughput) is invisible at 2048 — the ablation
is vacuous without the long-ctx probe: state-bytes/token vs GQA-8 KV (rung card: ≥2×
smaller), a decode-throughput crossover measurement, and the T3-mandated RULER recall probe
before any d14 confirmation. The 3:1 ratio itself is an empirical inheritance from
[13][14] — testing *the ratio* is a separate, larger question we explicitly do not open.

### 3.4 F6 MoE — the one genuine exception

MoE breaks the `C = 6ND` law's premise: compute scales with **active** params
(`C = 6·N_active·D`) while capacity scales with **total** params. The dense `N_opt(C)`,
`D_opt(C)` fit does not transfer — not because the exponents differ, but because the *compute
model* has an extra degree of freedom (sparsity = N_total/N_active, and granularity in the
fine-grained regime). The literature's joint forms: Clark et al. 2022's biquadratic
`log L(N,E) ≈ a·log N + b·log E + c·log N·log E + d` over expert count `E` [5]; Ludziejewski
et al. 2024's fine-grained extension (granularity enters; compute-optimal granularity rises
with budget) [6]; optimal sparsity itself rises with compute. HP transfer has its own μP
extension for MoE [15].

Protocol if F6 promotes beyond the harness stage: **2 scales × {sparsity, granularity}** on
the frozen recipe — e.g. d4-r20 and d8-r20 dense-equivalent-active, sparsity ∈ {4, 8} ×
granularity ∈ {coarse, fine} = 8 runs ≈ 1–1.5 GPU-days — fitting the joint form, with the
rung card's discriminating tests (induced-bias recovery; entropy >0.9·log N) as the
mechanism gate. This is a *mini* fit, not a re-run of s1–s7: the question is the sparsity
manifold, not the base exponents.

---

## §4 — The post-S3 rung DAG

- **Rung 0 — S3 fit (in progress).** s1 done (val_bpb 1.2553, C = 1.92e16); s2–s7 running
  sequentially on the standing box (tmux `s3`). On completion:
  `uv run python scripts/s3_scaling_sweep.py fit --out-dir artifacts/s3_scaling_sweep` →
  `fit.json` / `fit.md`; gates a+b ∈ [0.95, 1.05] and R² ≥ 0.98 fire loudly; the D:N rule
  decides whether the d20's 9.6B-token registration stands; the nanochat CORE-fit oracle at
  the d20's C (≈0.195) cross-checks the 0.19–0.22 band. Results → `docs/RESULTS.md` §S3 +
  `bench/RESULTS.md`.
- **Rung 1 — Anchor protocol (spec for every architecture variant).**
  1. Matched N (same depth knob), matched D (ratio 20), matched seed, same
     corpus/tokenizer/optimizer/precision as the S3 reference points.
  2. Comparator: final val_bpb on the pinned held-out slice. **Never** CORE for offsets
     (noise floor, §1.1).
  3. Two anchor scales minimum (d4-r20 vs s2; d8-r20 vs s5). A flat `Δbpb` across the two ⇒
     report the offset; a *diverging* `Δbpb` ⇒ exponent suspicion → add the d12-r8 anchor
     before any claim (this is the only path that can escalate an anchor study into a fit).
  4. Single-seed `|Δbpb| < 0.005` is noise; sub-floor claims need ≥3 seeds (mirrors the CORE
     ±0.008–0.016 discipline).
  5. Long-ctx variants (F8, F10) add the recall/RULER probe — quality-at-2048 cannot see
     their claim (§3.2/§3.3).
- **Rung 2 — T3 confirmations.** A variant that wins at sweep scale *and* passes the recall
  probe confirms at d14 (frozen recipe, ~57 h standing box or ~$8–12 spot — trigger-gated
  per E2E §S3(g), never a scheduled spend). Small-scale rankings can flip; the d14 is the
  flip detector.
- **Rung 3 — MoE joint mini-fit (conditional).** Only if F6's mechanism gates pass (§3.4).
- **Rung 4 — P5 → d20.** P5 d12 dress rehearsal ($10–15, rental, human go-ahead) validates
  the recipe on sm90 + the single Muon LR + CORE-vs-public-checkpoint + kill/resume drill.
  Then the 8×H100 d20 ($100 cap). **Architecture freeze = before P5** (status board:
  "Attention: Full GQA/MHA (frozen at P5)"). The freeze re-opens only for a T3-confirmed d14
  win whose serving-cost argument survives the re-anchored CORE band. Nothing else re-opens
  it.

---

## §5 — What WOULD force a full re-sweep (the exhaustive list)

1. **Corpus change** — exponents move with data quality [4]. F12 closed this axis: the law
   is on ClimbMix by operator override.
2. **Optimizer change** (e.g. Muon → MuonH [16], or a MuonClip-style variant) — moves the
   offset and possibly the HP-transfer rule; 2512.05620 [17] argues HP transfer is exactly
   what decides whether Muon's edge survives scale.
3. **Precision change** (bf16 → FP8/FP4) — nanochat's leaderboard point is FP8; precision
   has its own scaling laws.
4. **Context-length regime change** — the sweep is at 2048; training/serving far past it
   re-enters the attention quadratic (§1.3) and the long-ctx data distribution.
5. **MoE adoption** — compute model change (§3.4).

Everything else — MLA, DSA, GDN-hybrid at the frozen ctx — is an **anchor, not a sweep**.

---

## §6 — Risks and honesty notes

- **12.6× extrapolation** from the largest free point (s8-class, 2.2e18) to the d20 (2.77e19).
  Mitigations: P5 d12 anchor on H100; T1/T2 trigger-gated d14/d16; the honest CORE-vs-FLOPs
  framing (never a depth-matched headline).
- **Ranking flips at scale.** Small-scale winners can lose at 10×. Mitigation is T3 at d14 —
  never promotion by assumption.
- **F12 ↔ S3 cross-check (run when s4 lands).** F12's ClimbMix arm: 35M / 700M tok /
  C ≈ 1.47e17 / bpb 1.30205. The ladder's nearest-C point is s4 (59M / 474M / C = 1.69e17,
  ratio 8). Similar C, different N:D split and a different tokenizer (F12 trains a per-corpus
  BPE; S3 uses the staged ClimbMix tokenizer) — s4 landing *near but not at* 1.30 is
  consistent; a large gap flags a recipe difference to investigate before trusting the fit.
- **Batch-size deviations** (s7 at batch 4→2) are sub-critical (§1.4) and logged in
  `docs/RESULTS.md` §S3.
- **License.** ClimbMix is CC BY-NC 4.0 — the A9 model card must carry the license note and
  the HF source (`nvidia/Nemotron-ClimbMix`); the corpus override makes this a hard release
  obligation, not a footnote.

---

## §7 — References

1. Hestness et al. 2017, *Deep Learning Scaling is Predictable, Empirically* — via [3].
2. Rosenfeld et al. 2020, *A Constructive Prediction of the Generalization Error Across
   Scales* — via [3].
3. Weng 2026, *Scaling Laws, Carefully* — https://lilianweng.github.io/posts/2026-06-24-scaling-laws/
4. DeepSeek-AI 2024, *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism* —
   https://arxiv.org/abs/2401.02954
5. Clark et al. 2022, *Unified Scaling Laws for Routed Language Models* — via [3].
6. Ludziejewski et al. 2024, *Scaling Laws for Fine-Grained Mixture of Experts* —
   https://arxiv.org/abs/2402.07871
7. Besiroglu et al. 2024, *Chinchilla Scaling: A Replication Attempt* —
   https://arxiv.org/abs/2404.10102
8. McCandlish et al. 2018, *An Empirical Model of Large-Batch Training* — via [9].
9. *Predictable Scale: Optimal Hyperparameter Scaling Law in LLM Pretraining* —
   https://arxiv.org/abs/2503.04715
10. *Scaling Law for Language Models Training Considering Batch Size* —
    https://arxiv.org/abs/2412.01505
11. Gadre et al. 2024, *Language Models Scale Reliably With Over-Training and on Downstream
    Tasks* — https://openreview.net/pdf?id=PijcXNntQ9
12. Sardana & Frankle 2024, *Beyond Chinchilla-Optimal: Accounting for Inference in Language
    Model Scaling Laws* — https://arxiv.org/abs/2401.00448
13. Kimi Team 2025, *Kimi Linear: An Expressive, Efficient Attention Architecture* —
    https://arxiv.org/abs/2510.26692
14. *Super Apriel* 2026, hybrid-attention related-work survey (Nemotron-H 9:1, Falcon-H1,
    MiniMax M1 7:1, Qwen3-Next 3:1) — https://arxiv.org/html/2604.19877v1 ; Solar Open 2 —
    https://arxiv.org/html/2607.20062v1
15. *μ-Parameterization for Mixture of Experts* — https://arxiv.org/abs/2508.09752
16. MuonH — arXiv 2606.16899 (cited in `FRONTIER_2026_ABLATIONS.md` §10).
17. HP-transfer/Muon-at-scale — arXiv 2512.05620 (cited in `FRONTIER_2026_END_TO_END_PLAN.md`
    §S3).

*Authored 2026-08-02 during the S3 sweep run; grounded in the repo harness
(`scaling/isoflop.py`, `scaling/s3_sweep.py`) and the pre-registered gates in
`docs/RESULTS.md` §S3.*
