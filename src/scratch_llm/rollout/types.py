"""The rollout contract — backend-agnostic so train and serve engines are interchangeable.

L2 Systems (A2.3). One :class:`Rollout` shape and one :class:`RolloutClient` Protocol; the CPU
``LocalBackend`` implements it now and a GPU ``SGLangBackend`` slots in behind the *same* interface
later. Keeping the contract fixed is what lets ``kl_train_infer`` (utils/monitors.py) measure real
train↔infer engine drift rather than an API mismatch. See docs/design/L2_rollout_seam_SPEC.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from scratch_llm.sampling import SamplingParams

StopReason = Literal["stop", "length"]


@dataclass(frozen=True)
class Rollout:
    """One sampled continuation and the policy log-probs the RL spine scores it with.

    ``logprobs[t]`` is the policy log π(response_ids[t] | prompt + response_ids[:t]) at
    temperature 1 (raw model distribution, not temperature/top-p scaled) — the convention
    ``monitors.importance_ratios`` expects. ``stop_reason`` is ``"stop"`` iff a ``stop_ids``
    token was emitted, else ``"length"`` (the ``max_tokens`` budget was exhausted)."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    stop_reason: StopReason

    def __post_init__(self) -> None:
        if len(self.response_ids) != len(self.logprobs):
            raise ValueError(
                f"response_ids ({len(self.response_ids)}) and logprobs "
                f"({len(self.logprobs)}) must be the same length"
            )


class RolloutClient(Protocol):
    """The seam L5 develops rollouts against; both LocalBackend and the future SGLang backend
    satisfy it structurally, so the policy engine is swappable without touching callers."""

    def generate(self, prompt_ids: Sequence[int], params: SamplingParams) -> Rollout: ...

    def generate_batch(
        self, prompts: Sequence[Sequence[int]], params: SamplingParams
    ) -> list[Rollout]: ...
