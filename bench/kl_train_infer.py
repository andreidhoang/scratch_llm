"""A2.3 — measure REAL kl_train_infer between distinct train/serve engines on the same model.

The repo's discipline pillar made concrete (A2 guide §6: "the single most load-bearing thing in
A2"). The intended serve engine was SGLang (`SGLangBackend`), but current sgl_kernel builds ship only
sm90/sm100 kernels — not sm89 — so SGLang cannot run on this Ada RTX 4090 (ADR-0008). Reliable
stand-in: two HF engines on the same model that differ in the two axes that actually cause
train↔infer drift — **attention kernel** and **precision**:

  - serve = sdpa + bf16  (the fast inference path)
  - train = eager + {bf16, fp32}  (the reference forward; fp32 adds precision drift)

Both HF engines expose full logits, so `kl_train_infer = KL(train‖infer)` is the EXACT full-vocab
KL via `monitors.mean_kl` (p=train first → correct direction), not a sampled estimator. IS ratios +
ESS (`monitors`) and direction-independent mean|Δlogp| corroborate. HALT@0.10.

(A real SGLang serve engine returns only taken-token logprobs — there the k3 sampled estimator and
the truncation caveat in `monitors.py` would apply; that path is deferred to a Hopper box.)

Run on the GPU box:  python bench/kl_train_infer.py
"""

from __future__ import annotations

import numpy as np

from reasoning_llm.rollout.hf_reference import HFReferenceBackend
from reasoning_llm.sampling import SamplingParams
from reasoning_llm.utils.monitors import (
    KL_TRAIN_INFER_HALT,
    importance_ratios,
    mean_kl,
    normalized_ess,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "The capital of France is",
    "In one sentence, why is the sky blue?",
    "List three prime numbers:",
    "Explain gradient descent to a child:",
]


def _taken(rows: np.ndarray, ids: tuple[int, ...]) -> list[float]:
    return [float(rows[i, t]) for i, t in enumerate(ids)]


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = [tok(t, add_special_tokens=True)["input_ids"] for t in PROMPTS]
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=48, seed=0)

    serve = HFReferenceBackend(MODEL, dtype="bfloat16", attn_implementation="sdpa")
    rollouts = [serve.generate(p, params) for p in prompt_ids]
    serve_rows = [
        serve.distribution_logprobs(r.prompt_ids, r.response_ids).numpy() for r in rollouts
    ]
    serve_lp = [
        lp for r, sr in zip(rollouts, serve_rows, strict=True) for lp in _taken(sr, r.response_ids)
    ]
    n_tok = len(serve_lp)
    print(
        f"# serve=sdpa/bf16 | {len(rollouts)} rollouts, {n_tok} response tokens (exact full-vocab KL)"
    )

    for label, dtype, attn in [
        ("eager/bf16 (kernel only)", "bfloat16", "eager"),
        ("eager/fp32 (kernel+precision)", "float32", "eager"),
    ]:
        train = HFReferenceBackend(MODEL, dtype=dtype, attn_implementation=attn)
        kl_weighted, train_lp = 0.0, []
        for r, s_rows in zip(rollouts, serve_rows, strict=True):
            t_rows = train.distribution_logprobs(r.prompt_ids, r.response_ids).numpy()
            kl_weighted += mean_kl(t_rows, s_rows) * len(r.response_ids)  # KL(train‖infer)
            train_lp += _taken(t_rows, r.response_ids)
        kl = kl_weighted / n_tok
        is_r = importance_ratios(train_lp, serve_lp)
        mad = float(np.mean(np.abs(np.asarray(train_lp) - np.asarray(serve_lp))))
        halt = "TRIP" if kl > KL_TRAIN_INFER_HALT else "ok"
        print(
            f"train={label:<32} | kl_train_infer=KL(train‖infer)={kl:.5f} "
            f"[HALT@{KL_TRAIN_INFER_HALT} -> {halt}] | "
            f"IS_mean={is_r.mean():.4f} ESS={normalized_ess(is_r):.3f} mean|Δlogp|={mad:.5f}"
        )
        del train
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
