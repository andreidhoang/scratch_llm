# Design spec — L2 kl_train_infer (training-engine vs inference-engine KL)

> **Status:** building on the rented 4090 (SGLang serving).
> **Layer:** L2 Systems (A2.3) — an A2 systems measurement of the gap between the training and
> serving engines. See [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) and
> [`../STATUS.md`](../STATUS.md) for where it sits in the build.
> **Files:** `src/scratch_llm/rollout/sglang_client.py` (serve), `rollout/hf_reference.py` (train),
> `bench/kl_train_infer.py` (the measurement), reuses `utils/monitors.py`.

## 1. Why

`kl_train_infer` is the KL divergence between the **training-engine** and the
**inference/serving-engine** next-token distributions over the same vocabulary, evaluated on the
same realized contexts. It is nonzero even for the *same weights* because the two engines differ in
two ways that matter numerically: **mixed precision** (bf16 storage vs fp32 accumulation) and
**different attention kernels** (a hand-tuned fused/flash kernel vs an eager reference). This is a
legitimate A2 systems measurement in its own right, and it becomes practically important when doing
RL with a separate serving engine: if the two engines disagree, every gradient is computed against a
policy that isn't the one actually being served. The CPU rollout seam built the *scaffold*
(LocalBackend vs LocalBackend ⇒ KL≈0); this spec measures the **real** drift between two genuinely
different engines on the same model.

## 2. The two engines (same model: Qwen2.5-0.5B-Instruct)

- **Serve = SGLang** (`SGLangBackend`, a `RolloutClient`): generates rollouts; `return_logprob`
  gives the per-token logprob of each *sampled* token under the serving engine (fused kernels,
  bf16/quantized KV, continuous batching).
- **Train = HF transformers eager** (`HFReferenceBackend`): teacher-forces the same (prompt,
  response) to the per-token policy log π under a plain eager forward — the "training" engine.

## 3. Exact full-vocab KL (HF pair) vs sampled estimator (real SGLang)

Because the Ada stand-in uses **two HF engines, both of which expose full logits**, `kl_train_infer`
is the **EXACT full-vocab KL** via `monitors.mean_kl(train_rows, serve_rows)` — note the order:
p = train first, so it is `KL(train‖infer)`, the metric's defined direction (not the reverse). No
sampled estimator, no truncation bias, evaluated at every realized context. IS ratios + ESS
(`monitors.importance_ratios`, `normalized_ess`) and the direction-independent `mean|Δlogp|`
corroborate. HALT@0.10 applies to the exact KL.

> **Why this matters / the deferred case:** a *real* serving engine (SGLang) returns only the
> **taken-token** logprobs, not the full vocab — the caveat `monitors.py` flags. There `mean_kl` is
> unavailable and `kl_train_infer` must be the sampled **k3** estimator `mean(r − 1 − log r)`,
> `r = p_train/p_serve`, with the direction set by *which engine drew the tokens*. That path is
> deferred to a Hopper box (ADR-0008); on the HF pair we get the exact number.

## 4. The falsifiable demonstration

> Train in **bf16** (matches serve storage) vs **fp32** — predicted: fp32 drifts *more* (precision
> mismatch). **This prediction is wrong** (see §5): the driver is *accumulation* precision + kernel,
> not storage dtype. Same model / tokenizer / sampled tokens, so kernel and precision are the only
> variables — exactly what the metric must isolate.

## 5. Result (measured, RTX 4090)

> **SGLang could not run on the 4090** (sgl_kernel sm90/sm100 only, no sm89 — ADR-0008). Measured
> via the HF engine pair: serve = sdpa/bf16, on Qwen2.5-0.5B-Instruct, 4 prompts / 185 response
> tokens, temp 0.7. `sglang_client.py` stands ready for a Hopper box.

| train engine | kl_train_infer = KL(train‖infer), exact | IS mean | ESS | mean \|Δlogp\| | HALT@0.10 |
|---|---|---|---|---|---|
| eager / **bf16** (kernel only) | **0.00982** | 1.025 | 0.954 | 0.057 | ok |
| eager / **fp32** (kernel + precision) | **0.00173** | 1.004 | 0.996 | 0.030 | ok |

**Prediction FALSIFIED — and the falsification is the finding.** I predicted fp32-train would drift
*more* from bf16-serve (precision mismatch). The opposite held: **eager/fp32 is ~5.7× closer to
sdpa/bf16 than eager/bf16 is.** Mechanism: **`F.scaled_dot_product_attention` accumulates the
softmax/PV in fp32 even when the tensors are bf16**, so sdpa/bf16 is effectively fp32-precision
attention — and *eager/bf16* (true bf16 accumulation) is the real outlier, while *eager/fp32* nearly
matches the serve engine. **The drift driver is accumulation precision + kernel choice, not the
storage dtype.** That is precisely the mechanistic "why" of `kl_train_infer` (A2.3's teaching point).
The direction-independent `mean|Δlogp|` (0.057 vs 0.030) corroborates the ordering, so it is not an
artifact of the KL direction or estimator.

Both values sit well under HALT@0.10 — expected for the same fp32-master weights with no quantized
KV — yet are clearly nonzero, so the metric is live and sensitive. A real serving engine with fp8 /
INT4-KV (SGLang on a Hopper box) would drift *more* than this HF floor, toward HALT — the deferred
demonstration (ADR-0008).

## 6. Scope / deferred

- **In:** `SGLangBackend` (generate + score behind the Protocol), `HFReferenceBackend`, the
  measurement + k3 estimator, HALT check, the bf16-vs-fp32 demonstration.
- **Deferred:** full-vocab KL from the serving engine (needs serve-side full logits — not exposed);
  a fp8/INT4-KV serve config to push drift further; wiring the snapshot into a live RL loop (L5).
- **ADR trigger (A2 §8 #2):** SGLang vs vLLM — log each engine's logprob/precision contract when the
  choice is forced.
