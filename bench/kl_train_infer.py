"""A2.3 — measure REAL kl_train_infer between distinct train/serve engines on the same model.

The repo's discipline pillar made concrete (A2 guide §6: "the single most load-bearing thing in
A2"). The intended serve engine was SGLang (`SGLangBackend`), but current sgl_kernel builds ship only
sm90/sm100 kernels — not sm89 — so SGLang cannot run on this Ada RTX 4090 (ADR-0008). Reliable
stand-in: two HF engines on the same model that differ in the two axes that actually cause
train↔infer drift — **attention kernel** and **precision**:

  - serve = sdpa + bf16  (the fast inference path)
  - train = eager + {bf16, fp32}  (the reference forward; fp32 adds precision drift)

For rollouts sampled by serve, each engine's teacher-forced per-token log π is compared with
`utils/monitors` (IS ratios + ESS) and a sampled k3 KL estimator = `kl_train_infer` vs HALT@0.10.
A serving engine exposes only taken-token logprobs, so this is the sampled estimator, not the
full-vocab `monitors.mean_kl` — the honest limitation `monitors.py` flags.

Run on the GPU box:  python bench/kl_train_infer.py
"""

from __future__ import annotations

import numpy as np

from reasoning_llm.rollout.hf_reference import HFReferenceBackend
from reasoning_llm.sampling import SamplingParams
from reasoning_llm.utils.monitors import (
    KL_TRAIN_INFER_HALT,
    importance_ratios,
    normalized_ess,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "The capital of France is",
    "In one sentence, why is the sky blue?",
    "List three prime numbers:",
    "Explain gradient descent to a child:",
]


def k3_kl(logp_train: list[float], logp_serve: list[float]) -> float:
    """k3 estimator of KL(train‖serve) over tokens sampled from serve (Schulman):
    r = exp(logp_train − logp_serve);  KL ≈ mean(r − 1 − log r) ≥ 0."""
    d = np.asarray(logp_train, dtype=np.float64) - np.asarray(logp_serve, dtype=np.float64)
    r = np.exp(d)
    return float(np.mean(r - 1.0 - d))


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = [tok(t, add_special_tokens=True)["input_ids"] for t in PROMPTS]
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=48, seed=0)

    serve = HFReferenceBackend(MODEL, dtype="bfloat16", attn_implementation="sdpa")
    rollouts = [serve.generate(p, params) for p in prompt_ids]
    serve_lp = [lp for r in rollouts for lp in r.logprobs]
    print(
        f"# serve=sdpa/bf16 | {len(rollouts)} rollouts, lengths {[len(r.response_ids) for r in rollouts]}"
    )
    print(f"# total scored tokens: {len(serve_lp)}")

    import torch

    for label, dtype, attn in [
        ("eager/bf16 (kernel only)", "bfloat16", "eager"),
        ("eager/fp32 (kernel+precision)", "float32", "eager"),
    ]:
        train = HFReferenceBackend(MODEL, dtype=dtype, attn_implementation=attn)
        train_lp = [lp for r in rollouts for lp in train.score(r.prompt_ids, r.response_ids)]
        is_r = importance_ratios(train_lp, serve_lp)
        kl = k3_kl(train_lp, serve_lp)
        halt = "TRIP" if kl > KL_TRAIN_INFER_HALT else "ok"
        mad = float(np.mean(np.abs(np.asarray(train_lp) - np.asarray(serve_lp))))
        print(
            f"train={label:<32} | kl_train_infer(k3)={kl:.5f} "
            f"[HALT@{KL_TRAIN_INFER_HALT} -> {halt}] | "
            f"IS_mean={is_r.mean():.4f} ESS={normalized_ess(is_r):.3f} mean|Δlogp|={mad:.5f}"
        )
        del train
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
