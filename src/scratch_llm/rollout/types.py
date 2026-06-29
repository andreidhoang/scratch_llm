"""The rollout seam — a backend-agnostic contract for generating rollouts and re-scoring them.

L2 Systems (A2.3). One ``RolloutClient`` Protocol, a CPU ``LocalBackend`` now, the GPU
``SGLangBackend`` later behind the *same* interface — so swapping engines is a one-line change and
``kl_train_infer`` (``utils.monitors``) measures real engine drift, not an API mismatch. See
``docs/design/L2_rollout_seam_SPEC.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from scratch_llm.sampling import SamplingParams


@dataclass(frozen=True)
class Rollout:
    """One generated trajectory + the per-token policy log π(a_t|s_t) at temperature 1.

    Frozen with tuple fields so two rollouts compare *by value* (the seed-reproducibility and
    batch-parity invariants rely on ``==``). ``stop_reason`` is ``"stop"`` when a ``stop_ids`` token
    was emitted, else ``"length"`` (the ``max_tokens`` budget was exhausted).
    """

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    stop_reason: Literal["stop", "length"]


class RolloutClient(Protocol):
    """The backend-agnostic seam: generate one rollout, or a batch, from a prompt + params."""

    def generate(self, prompt_ids: Sequence[int], params: SamplingParams) -> Rollout: ...

    def generate_batch(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[Rollout]: ...
