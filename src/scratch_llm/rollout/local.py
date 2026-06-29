"""``LocalBackend`` — the CPU rollout engine over the from-scratch model + ``sampling``.

It plays a double role until SGLang lands: the train-engine logprob source *and* the stand-in
infer engine. Re-scoring one rollout with two identical ``LocalBackend``s gives
``kl_train_infer ≈ 0`` — the A2.3 comparison harness proven on CPU before any GPU spend; the number
becomes meaningful the moment a real serving engine's kernels/precision differ. See
``docs/design/L2_rollout_seam_SPEC.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from scratch_llm.model import TransformerLM
from scratch_llm.rollout.types import Rollout
from scratch_llm.sampling import SamplingParams, generate_with_logprobs


class LocalBackend:
    """A :class:`~scratch_llm.rollout.types.RolloutClient` backed by the local model."""

    def __init__(self, model: TransformerLM, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    def generate(self, prompt_ids: Sequence[int], params: SamplingParams) -> Rollout:
        """Decode one continuation, capturing the inline per-token policy log π (temperature 1)."""
        ids, logprobs = generate_with_logprobs(self.model, prompt_ids, params, self.device)
        stopped = bool(ids) and ids[-1] in params.stop_ids
        return Rollout(
            prompt_ids=tuple(prompt_ids),
            response_ids=tuple(ids),
            logprobs=tuple(logprobs),
            stop_reason="stop" if stopped else "length",
        )

    def generate_batch(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[Rollout]:
        # Per-prompt loop → the batch output is identical to calling generate() on each (the parity
        # invariant). A fused batched decode is a GPU-backend concern, not the CPU seam.
        return [self.generate(p, params) for p in prompts]

    @torch.no_grad()
    def score(self, prompt_ids: Sequence[int], response_ids: Sequence[int]) -> list[float]:
        """Teacher-forced per-token policy log π of the *taken* response tokens.

        Independent of the decode path (a real cross-check, not a tautology): one full forward over
        prompt+response, then ``log_softmax`` read at each predicting position and gathered at the
        taken token. Equals the inline ``generate`` logprobs because cached decode is logit-identical
        to recompute (the KV-cache invariant).
        """
        rows = self._response_logprob_rows(prompt_ids, response_ids)
        taken = torch.tensor(list(response_ids), dtype=torch.long, device=self.device)
        return rows.gather(-1, taken.unsqueeze(-1)).squeeze(-1).tolist()

    @torch.no_grad()
    def distribution_logprobs(
        self, prompt_ids: Sequence[int], response_ids: Sequence[int]
    ) -> Tensor:
        """Full per-position ``log_softmax`` rows ``(T_resp, vocab)`` at the response positions —
        the input ``utils.monitors.mean_kl`` consumes to compute the three KLs."""
        return self._response_logprob_rows(prompt_ids, response_ids)

    def _response_logprob_rows(
        self, prompt_ids: Sequence[int], response_ids: Sequence[int]
    ) -> Tensor:
        """``log_softmax`` rows that predict each response token. The response token at sequence
        index ``p+i`` is predicted by the logit row at ``p+i-1`` → rows ``[p-1 : p-1+T]`` (a
        non-empty prompt, already enforced by ``generate``, guarantees ``p-1 >= 0``)."""
        self.model.eval()
        p = len(prompt_ids)
        full = list(prompt_ids) + list(response_ids)
        x = torch.tensor([full], dtype=torch.long, device=self.device)
        logits: Tensor = self.model(x)[0]  # (L, vocab)
        rows = logits[p - 1 : p - 1 + len(response_ids)]
        return torch.log_softmax(rows, dim=-1)
