# Design spec — L2 kl_train_infer bridge (SGLang serve vs HF train)

> **Status:** building on the rented 4090 (SGLang serving).
> **Layer:** L2 Systems (A2.3) — **the single most load-bearing thing in A2** (A2 guide §6): the
> repo's discipline pillar made concrete. Realizes the through-line `kl_train_infer` HALT@0.10.
> **Files:** `src/reasoning_llm/rollout/sglang_client.py` (serve), `rollout/hf_reference.py` (train),
> `bench/kl_train_infer.py` (the measurement), reuses `utils/monitors.py`.

## 1. Why

`kl_train_infer = KL(train‖infer)` measures the divergence between the **training-engine** logits
and the **serving-engine** logits. If they disagree, every gradient is computed against a policy that
isn't the one being served — the failure R3/GSPO document for MoE-RL. The CPU rollout seam built the
*scaffold* (LocalBackend vs LocalBackend ⇒ KL≈0); this spec measures the **real** drift between two
genuinely different engines on the same model.

## 2. The two engines (same model: Qwen2.5-0.5B-Instruct)

- **Serve = SGLang** (`SGLangBackend`, a `RolloutClient`): generates rollouts; `return_logprob`
  gives the per-token logprob of each *sampled* token under the serving engine (fused kernels,
  bf16/quantized KV, continuous batching).
- **Train = HF transformers eager** (`HFReferenceBackend`): teacher-forces the same (prompt,
  response) to the per-token policy log π under a plain eager forward — the "training" engine.

## 3. The honest limitation (and why the estimator)

A serving engine returns only **taken-token** logprobs, not the full-vocab distribution — the caveat
`utils/monitors.py` documents. So the full-vocab `monitors.mean_kl` (used by the CPU scaffold) is
unavailable here. Instead `kl_train_infer` is the **sampled k3 estimator** over the rollout tokens:
`r = exp(logp_train − logp_serve)`, `KL ≈ mean(r − 1 − log r) ≥ 0` (Schulman). The exact, rigorous
per-token drift is the **IS ratio** `exp(logp_train − logp_serve)` + ESS (`monitors.importance_ratios`,
`normalized_ess`). HALT@0.10 applies to the estimator.

## 4. The falsifiable demonstration

> Train in **bf16** (matches serve) ⇒ *kernel-only* drift ⇒ small `kl_train_infer` (HALT ok).
> Train in **fp32** (≠ serve bf16) ⇒ *precision* drift ⇒ larger `kl_train_infer` (toward HALT).

Same model, same tokenizer, same sampled tokens — so the only variables are kernels and precision,
which is exactly what the metric must isolate.

## 5. Result (measured, RTX 4090)

> **SGLang could not run on the 4090** (sgl_kernel sm90/sm100 only, no sm89 — ADR-0008). Measured
> via the HF engine pair: serve = sdpa/bf16, on Qwen2.5-0.5B-Instruct, 4 prompts / 185 response
> tokens, temp 0.7. `sglang_client.py` stands ready for a Hopper box.

| train engine | kl_train_infer (k3) | IS mean | ESS | mean \|Δlogp\| | HALT@0.10 |
|---|---|---|---|---|---|
| eager / **bf16** (kernel only) | **0.0141** | 1.025 | 0.954 | 0.057 | ok |
| eager / **fp32** (kernel + precision) | **0.0022** | 1.004 | 0.996 | 0.030 | ok |

**Prediction FALSIFIED — and the falsification is the finding.** I predicted fp32-train would drift
*more* from bf16-serve (precision mismatch). The opposite held: **eager/fp32 is ~6× closer to
sdpa/bf16 than eager/bf16 is.** Mechanism: **`F.scaled_dot_product_attention` accumulates the
softmax/PV in fp32 even when the tensors are bf16**, so sdpa/bf16 is effectively fp32-precision
attention — and *eager/bf16* (true bf16 accumulation) is the real outlier, while *eager/fp32* nearly
matches the serve engine. **The drift driver is accumulation precision + kernel choice, not the
storage dtype.** That is precisely the mechanistic "why" of `kl_train_infer` (A2.3's teaching point).

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
