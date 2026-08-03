# K2 PROPOSAL — `core/kda.py` (GDN → KDA delta), for hand-retyping

> **Boundary notice (HANDCRAFTED.md Class 1):** this is an agent-written *proposal*, not a
> diff. The human retypes what they keep — retyping is the mechanism. Every equation here is
> traced to a primary source: Kimi Linear arXiv:2510.26692 (§3, §4, App. C), K3 arXiv:2607.24653
> (§2.1.1), `fla-org/flash-linear-attention` (`fla/ops/kda`, `fla/layers/kda.py`), and the
> released HF configs/code for Kimi-Linear-48B-A3B and Kimi-K3. Extracted 2026-08-02; anything
> not found in a primary source is marked **[NOT-FOUND — code-derived]**.
>
> Base: `src/scratch_llm/linear_attn.py` (GDN, F10.1 — chunkwise ≡ recurrent ≡ f64 contract
> green). KDA is an *incremental* change over it, per ROADMAP §3 K2.

---

## 0. What changes, one glance

| # | Piece | GDN (ours, green) | KDA (target) | Source |
|---|---|---|---|---|
| 1 | decay granularity | per-head scalar α_t | **per-channel** α_t ∈ (0,1)^{d_k}, Diag(α_t) | KL §3 Eq. 1 |
| 2 | decay logits | `a_proj: d_model → n_heads` | **low-rank**: `f_a_proj: d_model → head_dim` then `f_b_proj: head_dim → H·head_dim`, + per-channel `dt_bias [H·d_k]` | K3 §2.1.1 Eq. 2; ref code |
| 3 | decay map | `−e^{A_log}·softplus(z)` (keep as Kimi-Linear mode) | K3 mode: **`g = g_min · σ(e^{A_log}·z)`, g_min = −5**, α = e^g ∈ (e^{−5}, 1) | K3 §2.1.1 Eq. 5, Fig. 3 |
| 4 | short conv | none (F10.1 scope cut) | **depthwise causal conv1d k=4 + SiLU** on q, k, v (each its own conv, no bias) | KL §4/§5.2; fla `ShortConvolution` |
| 5 | output gate | none (F10.1 scope cut) | **full-rank sigmoid gate** from x_t: `W_o[ σ(W_g x_t) ⊙ RMSNorm(õ_t) ]`, RMSNorm per-head over d_v, gate before W_o | K3 §2.1.1 Eq. 6 (Kimi Linear's was low-rank, Eq. 10 — heritage note) |
| 6 | value expansion | `expand_v = 2` (GDN convention) | **none**: d_v = d_k = head_dim (Kimi Linear: key_dim = value_dim) | KL §4; HF configs |
| 7 | β write strength | `sigmoid(b_proj x)`, `b_proj: d_model → n_heads` | **unchanged** (same form, both papers) | KL §4; K3 Eq. 2 |
| 8 | q/k L2Norm + q scale | `_l2norm` eps=1e-6, q × d_k^{−1/2} | **unchanged** — placement: after conv+SiLU, before recurrence | KL §4; fla default |
| 9 | recurrence order | decay first, then rank-1 delta update | **unchanged** — verified identical in fla `naive_recurrent_kda` | fla source |
| 10 | chunk body | per-token loop inside chunks (taskspec concession) | WY/UT matmul form with per-channel Γ (log-cumsum, 16-token secondary tiles) — **K2.5 perf upgrade, NOT needed for the correctness contract** | KL §3.1 Eqs. 2–9, App. C |

The three-path contract is preserved verbatim: **chunkwise ≡ recurrent-scan ≡ float64
reference**, now with per-channel α. The step path (`gated_delta_rule_step`) changes by exactly
one line of math: `alpha_t[..., None, None]` (scalar per head) → `alpha_t[..., :, None]`
(per-key-channel, broadcasting over d_v).

---

## 1. The recurrence (both papers, identical form)

Per head, state `S_t ∈ R^{d_k × d_v}`:

```
S_t = (I − β_t k_t k_tᵀ) · Diag(α_t) · S_{t−1}  +  β_t k_t v_tᵀ
õ_t = S_tᵀ q_t
```

- α_t ∈ (0,1)^{d_k} — the whole point: **one decay per key channel**, not per head.
- β_t ∈ (0,1) scalar per head: `β_t = sigmoid(W_β x_t)`, `W_β: d_model → n_heads` (unchanged from GDN).
- Ordering confirmed against fla `naive_recurrent_kda`: decay applies to the state FIRST
  (`S = S * g.exp()`), then the delta-rule read/erase/write. Our `gated_delta_rule_step`
  already implements this order — the contract survives.
- q scaled by d_k^{−1/2} (fla `scale` default `1/sqrt(K)`; our `project()` already does this).

## 2. The decay path (the module's heart — get this exactly right)

**Logits (K3 §2.1.1 Eq. 2):** `z_t^h = W_α↑(W_α↓ x_t) + b_α^h ∈ R^{d_k}`

- `f_a_proj = Linear(d_model, head_dim, bias=False)` — **rank = head_dim** (64 mini, 128 K3).
- `f_b_proj = Linear(head_dim, n_heads · head_dim, bias=False)`.
- `dt_bias ∈ R^{n_heads · head_dim}` — per-channel bias, added to the logits pre-activation.
  Init Mamba-2-style: `dt = exp(U(log 0.001, log 0.1)).clamp(min=1e-4)`;
  `dt_bias = dt + log(−expm1(−dt))` (inverse softplus; fla `layers/kda.py:180–184`)
  **[NOT-FOUND in papers — code-derived]**. R0 measured the *learned* dt_bias ≈ −4.63 ± 0.05
  at 2.8T (long-retention default) — our R1/R2 arms log this distribution (ABLATIONS.md protocol).

**Map — two modes behind one flag:**

- `kimi_linear` mode (heritage, GDN-like but per-channel): `g = −e^{A_log} · softplus(z)`,
  α = e^g ∈ (0, 1). Unbounded below — this is what forced the 16-token-tile position-pair
  special-casing in Kimi Linear's kernel (K3 Fig. 3 narrative).
- `k3` mode (default for us): `g = g_min · σ(e^{A_log} · z)` with **g_min = −5 fixed**,
  α = e^g ∈ (e^{−5}, 1). Bounds cumulative log-decay over a 16-token tile to (−80, 0) →
  reciprocal rescale < e^80, inside BF16 range (K3 §2.1.1). This is the number R2's P4
  sub-prediction attacks (e^160 > BF16 max ≈ e^88 at g_min ≤ −10).

**A_log:** checkpoint of record: **per-dim `[128]` (= head_dim) in all 69 KDA layers** (R0 census;
FACTS A18 — `param_count.py` closes EXACT, residual 0, only with `a_log_size=128`). The
2026-08-02 "resolution" stated here ("the per-channel tensor is `dt_bias [H·d_k]`; `A_log` is
per-head `[n_heads]` in the released code") is **RETRACTED 2026-08-03**: per-head-96 is exactly
what the census falsified — it leaves the pre-census residual 69×(128−96) = 2,208 unclosed. The
released code's per-head framing does not explain the [128] shapes; how A_log maps to the
per-channel decay Diag(α) (broadcast / interpolation per channel group?) is **OPEN — pre-flight
for the hand-build**: resolve by direct read of `modeling_kimi_linear.py` checkpoint loading
before `core/kda.py`.
Init: K3 paper §2.1.1 says A^h = 0; K3 released code defaults `log(U(1,16))` (Mamba lineage);
fla uses 0 under `safe_gate=True`. **Proposal: init 0 in k3 mode, `log(U(1,16))` in kimi_linear
mode** — and note R0's measured A_log means drift only −0.17 → +0.29 with depth, so the choice
is second-order; the R1 decay telemetry will show where training takes it.

## 3. The new block extras (F10.2 pieces, landing here per ROADMAP)

**Short conv (KL §4):** on q, k, v after their projections, before L2Norm/recurrence:

- each its own `nn.Conv1d(channels, channels, kernel_size=4, groups=channels, bias=False)` —
  depthwise; causal (pad k−1 left, trim) **[causality NOT-FOUND in paper — fla/HF-code-derived]**;
- then SiLU; then for q/k: L2Norm (eps 1e-6); v: no norm.
- Kimi Linear Table 1 ablation: removing conv degrades val PPL 5.65 → 5.70 at 653M — small but
  real; it's also the induction-head mechanism at our scale, so it stays.
- **Causality is a silent-bug surface**: a non-causal conv leaks future tokens and still
  trains — losses look *better*. Red-team will include a shift/future-leak probe (§6).

**Output gate (K3 §2.1.1 Eq. 6):** `y_t = W_o[ σ(W_g x_t) ⊙ RMSNorm(õ_t) ]`

- `W_g = Linear(d_model, n_heads · head_dim, bias=False)` — **full-rank** (K3 upgrade; Kimi
  Linear's was low-rank bottleneck-128 "for fair param comparison" — heritage note only).
- RMSNorm is **per-head** over the d_v vector (fla `FusedRMSNormGated(head_dim)`), eps 1e-5
  **[eps code-derived]**, applied to the recurrence output BEFORE the gate multiply.
- Gate computed from the layer input x_t (same source as everything else).
- `W_o = Linear(n_heads · head_dim, d_model, bias=False)`.

## 4. Chunkwise form — what to build now vs later

The correctness contract does NOT require the WY/UT matmuls: our chunk body may stay a
per-token loop (F10.1 concession) as long as chunk ≡ recurrent ≡ f64 holds bitwise-in-f64.
**Proposal: K2 ships the per-channel loop body; the WY/UT chunkwise (KL §3.1 Eqs. 2–9) is the
K2.5/F10.2 perf rung**, where it lands with log-space cumulative Γ, forward-substitution
inverse, and secondary 16-token chunking. For the retype, the load-bearing facts are:

- chunk size C = 64 everywhere (KL App. C, fla default).
- The intra-chunk matrices contain `1/Γ` terms (Γ = cumulative per-channel decay product);
  in log space: `g.cumsum(-2)`, differences of cumsums, never materialized ratios in fp32.
- All chunkwise math in fp32; activations bf16 at the boundaries (matches our `project()`
  gate-dtype promotion already).

## 5. fla parity plan (K2 gate b — GPU day, on the pod)

- `pip install "fla-core>=0.4.0"` (KDA ops since v0.4.0, 2025-10-27). **Not installed in the
  laptop venv as of 2026-08-02 — install on the pod at parity time; CPU-only parity is not the
  gate, fwd+bwd CUDA parity is.**
- API: `from fla.ops.kda import chunk_kda, fused_recurrent_kda`.
- Layout trap: fla expects **[B, T, H, D]**; ours is [B, H, T, D]. Transpose at the boundary.
- For parity we precompute the log-decay ourselves and pass it as `g [B, T, H, d_k]` (fp32),
  β post-sigmoid; `use_qk_l2norm_in_kernel=False` (we normalize outside), `scale=None`,
  `initial_state=None`, `output_final_state=True`. Do NOT use `use_gate_in_kernel` for the
  first parity pass (it fuses the −e^A·softplus map — that's the kimi_linear-mode check;
  for k3-mode use `safe_gate=True, lower_bound=−5` with raw logits).
- Parity targets: fwd outputs and final state allclose (bf16 tolerance), and grads wrt
  q/k/v/g/β via autograd on both sides (fused_recurrent for the step path).
- Doc inconsistency to know: `fused_recurrent_kda` docstring says `dt_bias [H]`; the chunk
  path and reference use per-channel `[H·d_k]`. Per-channel is correct.
- Serve-time note (K10, not K2): K3's released code stores the recurrent state `[V, K]`
  (`state_v_first=True`); ours is `[d_k, d_v]`. Transpose at the cache boundary later.

## 6. Gates & the red-team menu (what I run after you author it)

Mastery bar (HANDCRAFTED.md): delete-test + chunkwise≡recurrent≡f64 + FLA parity explained.

1. **Three-path equivalence with per-channel α** (the existing F10.1 test pattern, α now a
   [B, H, T, d_k] tensor): chunk-size independence (bitwise), ragged tails, initial_state.
2. **Extreme-decay probes:** z → −∞ gives g → 0⁻ (α → 1, perfect retention); z → +∞ gives
   g → g_min (floor binds, α → e^{−5}). Drive the floor deliberately: huge positive logits
   must NOT NaN in bf16 (that's g_min's job); same input in kimi_linear mode must show
   α → 0 (contrast arm).
3. **Conv causality probe:** permuting future tokens must not change past outputs (shift test
   on the conv path specifically — catches a non-causal conv that training would *reward*).
4. **Gate ablation identity:** W_g = 0 ⇒ output scaled by σ(0) = 0.5 exactly; RMSNorm-off
   delta matches hand-computed norm.
5. **Train/decode drift:** full-sequence chunkwise vs token-by-token step from a trained-ish
   (few-step) module, not just random init — drift often only shows up off-init.
6. **Loss-at-init** through a mini block stack: ≈ log V with the gate active.

## 7. mini-K3 instantiation (from ROADMAP K6 table — the shapes your module must accept)

hidden 1024 · 16 heads × 64 (d_k = d_v = 64) · conv k=4 · g_min = −5 · full-rank gate ·
β per head · A_log per head init 0 (k3 mode — our mini-scale choice; the real checkpoint carries
[128] per-dim and its mapping is OPEN, §2 — resolve before loading real weights). New parameter
tensors per layer vs GDN:
`f_a_proj 1024×64`, `f_b_proj 64×1024`, `dt_bias [1024]`, 3 depthwise convs `[1024, 1, 4]`,
`g_proj 1024×1024`; dropped: none (q/k/v/out same widths at expand_v=1). `param_count.py`
owns the exact accounting; K0's closure discipline applies.

---

## Session-1 reading order for the human (ROADMAP §6)

1. `src/scratch_llm/linear_attn.py` docstring (Intent / Invariant / Conventions) — 10 min.
2. Kimi Linear §3 (recurrence + WY form) and §4 (the layer: conv, L2Norm, gate, β) — with
   `docs/k3/READING_GUIDE_K3_REPORT.md` as the section-by-section companion.
3. K3 report §2.1.1 + Fig. 3 (the g_min = −5 motivation — this is what R2 P4 attacks).
4. `fla/ops/kda/naive.py` (`naive_recurrent_kda`, `naive_kda_lowerbound_gate`) — the 20-line
   oracle; read it against §1–2 above.
5. Then hand-write, in serial order: `core/situ.py` (warmup — spec verified and proposal
   ready: [`K5_PROPOSAL_SITU.md`](K5_PROPOSAL_SITU.md)) → `core/kda.py`.
