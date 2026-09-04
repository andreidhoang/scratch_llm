"""HF transformers engine — a configurable train/serve engine for the kl_train_infer bridge.

L2 Systems (A2.3). `kl_train_infer` compares a *serving* engine against a *training* engine on the
same model. The intended serve engine is SGLang (`sglang_client.py`), but current SGLang/sgl_kernel
builds ship only sm90/sm100 kernels — no sm89 — so they cannot run on an Ada RTX 4090 (see
docs/adr/ADR-0008). As a reliable stand-in on Ada, this class instantiates an HF engine configurable
by **precision** (dtype) and **attention kernel** (`attn_implementation` ∈ eager/sdpa/flash_attention_2):
an *eager + fp32* instance is the "train" engine, an *sdpa + bf16* instance is the "serve" engine —
a genuine kernel+precision-distinct pair, which is exactly the drift `kl_train_infer` must capture.

GPU/model-only (imports `transformers`, downloads weights). Not imported by `rollout/__init__.py`,
so the package still imports on a CPU/CI box; the measurement gates on the gpu marker.
See docs/design/L2_kl_train_infer_SPEC.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from scratch_llm.rollout.types import Rollout
from scratch_llm.sampling import SamplingParams


class HFReferenceBackend:
    """An HF causal-LM engine; precision + attention kernel pick its train-vs-serve identity."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        device: str = "cuda",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, dtype),
                attn_implementation=attn_implementation,
            )
            .to(device)
            .eval()
        )
        self.device = device

    @torch.no_grad()
    def generate(self, prompt_ids: Sequence[int], params: SamplingParams) -> Rollout:
        """Sample a continuation via `model.generate` (the cached-decode serve path). Returns a
        :class:`Rollout`; logprobs are filled by re-scoring (so they are the policy log π, not the
        temperature-warped sampling scores)."""
        if params.seed is not None:
            torch.manual_seed(params.seed)
        x = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        out = self.model.generate(
            x,
            do_sample=params.temperature > 0.0,
            temperature=params.temperature if params.temperature > 0.0 else None,
            top_p=params.top_p,
            max_new_tokens=params.max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = tuple(int(t) for t in out[0, len(prompt_ids) :].tolist())
        logprobs = tuple(self.score(prompt_ids, response_ids))
        stop_reason: Literal["stop", "length"] = (
            "stop" if response_ids and response_ids[-1] in params.stop_ids else "length"
        )
        return Rollout(
            prompt_ids=tuple(prompt_ids),
            response_ids=response_ids,
            logprobs=logprobs,
            stop_reason=stop_reason,
        )

    @torch.no_grad()
    def _response_logprob_rows(
        self, prompt_ids: Sequence[int], response_ids: Sequence[int]
    ) -> torch.Tensor:
        """Full per-position ``log_softmax`` rows that predict each response token: (T_resp, vocab),
        on CPU. Response token at sequence index p+i is predicted by the logit row at p+i-1."""
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids must be non-empty")
        p = len(prompt_ids)
        seq = list(prompt_ids) + list(response_ids)
        x = torch.tensor([seq], dtype=torch.long, device=self.device)
        log_probs = torch.log_softmax(self.model(x).logits[0].float(), dim=-1)
        return log_probs[p - 1 : p - 1 + len(response_ids)].cpu()

    def score(self, prompt_ids: Sequence[int], response_ids: Sequence[int]) -> list[float]:
        """Teacher-forced per-token policy log π of the taken tokens (IS-ratio input). No grad."""
        if not response_ids:
            return []
        rows = self._response_logprob_rows(prompt_ids, response_ids)
        return [float(rows[i, tok]) for i, tok in enumerate(response_ids)]

    def distribution_logprobs(
        self, prompt_ids: Sequence[int], response_ids: Sequence[int]
    ) -> torch.Tensor:
        """Full ``log_softmax`` rows for the response positions — the input to the EXACT full-vocab
        ``monitors.mean_kl`` (both HF engines expose full logits, so no sampled estimator needed)."""
        return self._response_logprob_rows(prompt_ids, response_ids)
