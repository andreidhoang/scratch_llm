# KDA `A_log` / `dt_bias` MAPPING — PROPOSAL (pre-flight for `core/kda.py`)

> **PROPOSAL, not verified math.** Boundary notice (HANDCRAFTED.md Class 1): agent-written
> analysis for the human to retype from; nothing here touches `core/`. Compiled 2026-08-10.
> Verdict labels per FACTS.md conventions: **VERIFIED** (primary source read directly) ·
> **MEASURED BY US** (our own reads of the released checkpoints) · **NOT VERIFIED** (flagged,
> not guessed). Sources: `fla-org/flash-linear-attention` @ `7843b32` (2026-08-09),
> `vllm-project/vllm` @ main (fetched 2026-08-10), `huggingface.co/moonshotai/Kimi-K3`
> (raw/main, fetched 2026-08-10), Kimi Linear arXiv:2510.26692 (v2 PDF), our census
> (`artifacts/k3_anatomy/`).

## 0. TL;DR — the A18 OPEN question is resolved

**`A_log` is per-HEAD `[96]`, not per-dim. The checkpoint's `[128]` storage is a 96-entry live
tensor padded with 32 dead zeros.** Every consumer in the released stack reads only
`A_log[0:96]`, indexed by head; entries 96–127 are exactly `0.0` in all 69 KDA layers, are
never read by any kernel, and (being at their zero init with no gradient path and
`_no_weight_decay`) can never have moved. The "96 ≠ 128" conflict dissolves: 128 is a storage
artifact, 96 is the semantics. This closes A18's residual (69×32 = 2,208 dead parameters) with
a mechanism, not just arithmetic.

## 1. Evidence chain

| # | Fact | Verdict | Evidence |
|---|---|---|---|
| E1 | Checkpoint stores `A_log` F32 `[128]`, `dt_bias` F32 `[12288]` in all 69 KDA layers | MEASURED BY US | `artifacts/k3_anatomy/census.json` (2026-07-31 census, all 96 shards) |
| E2 | `A_log[96:128] ≡ 0.0` exactly, all 69 layers; `A_log[0:96]` trained (25% of entries outside init range `log U(1,16)=[0, 2.77]`, min −2.47) | MEASURED BY US (2026-08-10) | `artifacts/k3_anatomy/tensors/*A_log*.npy`; tail min=max=0.0 across all layers |
| E3 | vLLM (official serving path, produced the published K3 benchmarks) loads `A_log` by **narrowing the 1-D checkpoint tensor to the first `num_heads` entries** (per TP rank); the tail is discarded at load | VERIFIED | `vllm/models/kimi_k3/nvidia/kda.py:55-77` (`a_log_weight_loader`: `loaded_weight.narrow(shard_axis, tp_rank*shard_size, shard_size)` with `shard_size = local_num_heads = 96/TP`); param allocated `[local_num_heads]` at lines 414-417 |
| E4 | Every FLA KDA kernel indexes `A_log` **by head**: `b_A = tl.load(A_log + i_h)` (fwd/bwd gate, chunk-cumsum) and `tl.load(A_log + i_hv)` (fused recurrent); torch reference: `A_log.view(H, 1)` | VERIFIED | `fla/ops/kda/gate.py:108, 171, 417` and `gate.py:54, 69`; `fla/ops/kda/fused_recurrent.py:162` |
| E5 | `dt_bias` is per-(head, channel) `[96·128]`, viewed `[H, K]`: `dt_bias.view(H, -1)` / `tl.load(dt_bias + i_h*K + o_k)` | VERIFIED | `fla/ops/kda/gate.py:52, 414`; `fused_recurrent.py:165` |
| E6 | Under `safe_gate` (K3's mode), FLA initializes `A_log = torch.zeros(num_v_heads)` and sets `A_log._no_weight_decay = True` — so an over-allocated buffer's unread tail stays exactly 0 forever | VERIFIED | `fla/layers/kda.py:175-179` |
| E7 | Kimi Linear 48B-A3B checkpoint (the KDA reference artifact) stores `A_log` as `[1, 1, 32, 1]` — per-head (32 heads), 4-D — and `dt_bias [4096]` = 32×128. vLLM's loader explicitly handles both layouts ("old 4D or current 1D") | MEASURED BY US (2026-08-10) + VERIFIED | shard-header range reads on `moonshotai/Kimi-Linear-48B-A3B-Instruct` (same method as `scripts/k3_fetch_tensors.py`); `vllm/.../kda.py:58, 65-72` |
| E8 | K3 config: `linear_attn_config = {num_heads: 96, head_dim: 128, gate_lower_bound: -5.0, short_conv_kernel_size: 4, use_full_rank_gate: true}`; HF reference layer allocates `A_log [num_heads]` and `dt_bias [projection_size=12288]` and calls `chunk_kda(..., safe_gate=True, lower_bound=-5.0, use_gate_in_kernel=True)` | VERIFIED | `moonshotai/Kimi-K3/raw/main/config.json`; `modeling_kimi_linear.py` (`KimiDeltaAttention.__init__`, `forward`) |

Mechanism reading E2+E4+E6 together: the training-side buffer was allocated 128 wide
(consistent with a `zeros(head_dim)` slip, `head_dim` vs `num_heads` being the two 128/96
constants of the layer), the kernels only ever touch the first 96 entries, and the tail sits at
its zero init with no gradient and no weight decay. **Why 128 was allocated is NOT DISCLOSED —
intent is not verifiable from primary sources; the dead-zero tail is.**

## 2. The decay computation chain (K3 mode — what `core/kda.py` must implement)

All line citations: `fla/ops/kda/gate.py` and `fla/ops/kda/naive.py` @ `7843b32`.

Per token `x_t`, per head `h ∈ [0,96)`, per key-channel `k ∈ [0,128)`:

```
z_t[h,k]      = (f_b_proj(f_a_proj(x_t))).view(96, 128)[h,k]        # low-rank logits, rank = head_dim
z'_t[h,k]     = z_t[h,k] + dt_bias.view(96, 128)[h,k]               # gate.py:52, 68
log α_t[h,k]  = g_min · σ( exp(A_log[h]) · z'_t[h,k] ),  g_min = -5 # gate.py:69, 124, 422 (USE_LOWER_BOUND)
α_t[h,k]      = exp(log α_t[h,k]) ∈ (e^-5, 1] ≈ (0.0067, 1]         # naive.py:61 (S = S * g.exp())
```

- `A_log[h]` enters as a **per-head positive pre-scale** `exp(A_log[h])` inside the sigmoid
  argument. It is NOT a Mamba-style decay rate in K3 mode (that is the heritage map, below).
  At its zeros init the scale is exactly 1 (neutral); the census's trained values give
  `exp(A_log)` layer-means ∈ [0.84, 1.34] — a mild per-head sharpening of the gate response.
- Recurrence (order verified against `naive.py:61-63`): decay first, then delta rule —
  `S ← Diag(α_t)·S;  S ← S + β_t k_t (v_t − k_tᵀS)ᵀ-outer;  o_t = Sᵀ q_t`, i.e.
  `S_t = (I − β_t k kᵀ) Diag(α_t) S_{t−1} + β_t k vᵀ`. `β_t[h] = σ(b_proj(x_t))`
  (`use_beta_sigmoid_in_kernel=True`), q/k L2-normed post-conv (`use_qk_l2norm_in_kernel=True`).
- Heritage mode (Kimi Linear 48B, `safe_gate=False`, `lower_bound=None`):
  `log α_t[h,k] = −exp(A_log[h]) · softplus(z'_t[h,k])` (gate.py:54, 122) — the Mamba/GDN-style
  map, unbounded below. Paper anchor: decay `f(·)` "similar to those used in GDN and Mamba"
  (arXiv:2510.26692 §4, text after Eq. 9). K3's `g_min = −5` scaled sigmoid replaces it to bound
  the per-tile cumulative log-decay for all-Tensor-Core tiles (FACTS A12; K3 report §2.1.1;
  `chunk_kda` docstring, `fla/ops/kda/chunk.py:249-260`).
- Output gate (K3, full-rank, `use_full_rank_gate: true`):
  `o = o_proj( RMSNormGated_per-head(o_rec, σ(g_proj(x_t))) )`, RMSNorm over `head_dim`
  (`FusedRMSNormGated(128, activation='sigmoid')`).

## 3. Loading the real checkpoint — contract for `core/kda.py`

To load `moonshotai/Kimi-K3` KDA weights correctly, the hand-built module must:

1. Allocate `A_log` as **`[num_heads]` (96)** and load **only `checkpoint_A_log[0:96]`**
   (drop the [96:128] tail). Assert the tail is all-zero before dropping — that assertion is
   the load-time proof the semantics were understood (see §4, check F2).
2. Allocate `dt_bias` as **`[num_heads · head_dim]` (12288)**, viewed `[96, 128]` —
   per-channel within each head, **not** shared across heads.
3. Compute the decay in **K3 mode**: `log α = -5 · σ(exp(A_log[h]) · (z + dt_bias))` per
   `[h, k]`; `α = exp(log α)`. Do NOT use the softplus map for K3 weights (that map is for
   Kimi-Linear-48B weights only).
4. Keep everything in the decay path fp32 (kernels do `.to(tl.float32)` before the gate).
5. β: `sigmoid(b_proj(x))` per head; q,k: L2-norm after conv+SiLU.

Note what this means for the A18 accounting: `param_count.py` counts **storage** (128) to match
the HF safetensors total; the **live** parameter count is 96. Both statements are now
reconciled — no change needed to the (green, exact) gate.

## 4. Falsifiable checks (for the red team and the R1 telemetry)

- **F1 (range):** every per-step decay must satisfy `α ∈ (e^-5, 1] = (0.00674, 1]`. Any α
  outside this interval ⇒ wrong map (e.g. softplus heritage map used by mistake).
- **F2 (dead tail):** `A_log[96:128] ≡ 0.0` bit-exactly in all 69 layers of the real
  checkpoint (MEASURED true 2026-08-10). If a future checkpoint revision breaks this, the
  per-head reading must be re-opened.
- **F3 (bias-only default):** at `z = 0`, `log α = -5·σ(dt_bias)`. With the census's
  `dt_bias` layer means −4.72…−4.54: `log α ∈ [−0.053, −0.044]`, `α ∈ [0.949, 0.957]`,
  retention half-life **≈ 13–16 tokens**. Per-channel spread (dt_bias min −9.0, max +0.18)
  spans `log α ∈ [−2.73, −0.0006]` — the full range from near-total forgetting to near-perfect
  retention. **Corollary: FACTS A19(d)'s "strongly long-retention" gloss was computed under
  the heritage softplus map (α ≈ 0.990, half-life ≈ 71 tokens); under the actual K3 map the
  bias-only default is α ≈ 0.953.** The qualitative conclusion (retention-biased default,
  forgetting input-modulated) survives; the number changes.
- **F4 (A_log neutrality probe):** at init (`A_log = 0`) the per-head scale is exactly 1, so
  zeroing a *trained* `A_log` changes outputs only through `exp(A_log[h]) ∈ [0.84, 1.34]`
  (layer-mean) — a bounded perturbation. A hand-built module that accidentally broadcasts
  `A_log` per-channel instead of per-head is caught by F5:
- **F5 (parity probe):** against `fla.ops.kda.fused_recurrent_kda` with
  `use_gate_in_kernel=True, lower_bound=-5`, feeding raw logits `z`: our
  `-5·σ(exp(A_log[h])·(z+dt_bias.view(H,K)))` must match bitwise-in-f64 (CPU reference path).
  FLA parity plan already in K2_PROPOSAL_KDA.md §5 — note it passes `A_log [H]`.

## 5. What is NOT verified (explicit)

- **N1 — Training-side intent of the 128 allocation.** Not disclosed anywhere. Our mechanism
  reading (over-allocated zero-init buffer, dead tail) is an inference from E2/E4/E6, not a
  documented fact.
- **N2 — HF `from_pretrained` behavior on the raw repo.** The HF reference
  `modeling_kimi_linear.py` allocates `A_log [96]` while the checkpoint carries `[128]`; a
  strict load would mismatch. We did not run it (no 1.5 TB download); the official and
  benchmarked path is vLLM, whose custom loader (E3) handles the 1-D layout. Treat the plain
  transformers path as untested.
- **N3 — K3 tech report §2.1.1 equation text.** arXiv:2607.24653 has no HTML full-text as of
  2026-08-10; the scaled-sigmoid / g_min = −5 framing is taken from FACTS A12 (verified there
  against the report) and is independently confirmed by config (`gate_lower_bound: -5.0`) and
  the FLA kernel path K3 selects. No conflict found.
- **N4 — Whether any other runtime (SGLang, llama.cpp port, TensorRT) reads `A_log`
  differently.** Only vLLM (E3) and FLA (E4) were read line-by-line. llama.cpp's
  `ggml_gated_delta_net` (FACTS S8) was not audited for this tensor.

## 6. Relation to existing docs

- Resolves the **OPEN** question in `K2_PROPOSAL_KDA.md` §2 ("A_log", lines 75-87) and §7:
  the proposal's mini-K3 choice "A_log per head, init 0 (k3 mode)" is now confirmed as the
  *checkpoint-compatible* choice, and §7's caveat "the real checkpoint carries [128] per-dim
  and its mapping is OPEN — resolve before loading real weights" is answered by §3 above
  (load first 96, assert dead tail).
- Sharpens FACTS A18 (ledger edit in the same commit) and corrects A19(d)'s interpretation
  (F3 corollary).
