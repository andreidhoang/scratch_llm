"""In-process rollout backend — the CPU reference engine and the train↔infer harness.

L2 Systems (A2.3). ``LocalBackend`` wraps the from-scratch model + sampler to satisfy
:class:`RolloutClient`. It plays a double role: it is **both** the train-engine logprob source
**and**, until the GPU ``SGLangBackend`` lands, the stand-in infer engine — so the
``kl_train_infer`` plumbing (utils/monitors.py) can be exercised end-to-end on CPU. Re-scoring one
rollout with two identical ``LocalBackend``s gives KL ≈ 0; the number becomes meaningful when the
serving engine's kernels/precision actually differ. See docs/design/L2_rollout_seam_SPEC.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from reasoning_llm.model import TransformerLM
from reasoning_llm.rollout.types import Rollout, StopReason
from reasoning_llm.sampling import SamplingParams, generate_with_logprobs


class LocalBackend:
    """A :class:`RolloutClient` over an in-process :class:`TransformerLM`."""

    def __init__(self, model: TransformerLM, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    def generate(self, prompt_ids: Sequence[int], params: SamplingParams) -> Rollout:
        ids, logprobs = generate_with_logprobs(self.model, prompt_ids, params, self.device)
        stop_reason: StopReason = "stop" if ids and ids[-1] in params.stop_ids else "length"
        return Rollout(
            prompt_ids=tuple(prompt_ids),
            response_ids=tuple(ids),
            logprobs=tuple(logprobs),
            stop_reason=stop_reason,
        )

    def generate_batch(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[Rollout]:
        return [self.generate(p, params) for p in prompts]

    @torch.no_grad()
    def _response_logprob_rows(
        self, prompt_ids: Sequence[int], response_ids: Sequence[int]
    ) -> Tensor:
        """Full ``log_softmax`` rows that predict each response token: shape ``(T_resp, vocab)``.

        Teacher-forced over ``prompt + response``. Row ``i`` is the distribution at sequence
        position ``p+i-1`` (which predicts ``response_ids[i]``), so rows span ``[p-1 : p-1+T]``.
        Requires a non-empty prompt (``p >= 1``), as :func:`generate` does."""
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids must be non-empty")
        self.model.eval()
        p = len(prompt_ids)
        seq = list(prompt_ids) + list(response_ids)
        x = torch.tensor([seq], dtype=torch.long, device=self.device)
        log_probs = torch.log_softmax(self.model(x)[0], dim=-1)  # (seq_len, vocab)
        return log_probs[p - 1 : p - 1 + len(response_ids)]

    def score(self, prompt_ids: Sequence[int], response_ids: Sequence[int]) -> list[float]:
        """Teacher-forced per-token policy log π of the *taken* tokens (the IS-ratio input and
        the contract-test oracle — independent of the decode path, so equality is a real check)."""
        rows = self._response_logprob_rows(prompt_ids, response_ids)
        return [float(rows[i, tok]) for i, tok in enumerate(response_ids)]

    def distribution_logprobs(
        self, prompt_ids: Sequence[int], response_ids: Sequence[int]
    ) -> Tensor:
        """Full per-position ``log_softmax`` rows for the response — the ``kl_train_infer`` hook:
        a second engine re-scores the same rollout, then ``monitors.mean_kl`` compares the rows."""
        return self._response_logprob_rows(prompt_ids, response_ids)
