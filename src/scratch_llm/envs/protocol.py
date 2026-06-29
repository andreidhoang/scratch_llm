"""The A5 verifiable-environment contract — the seam every RL env plugs into.

A ``VerifiableEnv`` yields tokenized ``Task``s, a ``DecodeFn`` (ids → text), and a ``grade`` that
turns a rollout into a ``Graded`` reward. All structural Protocols, so a toy CPU env and a real HF
env satisfy the same interface without inheritance. Pure-Python (no torch): ``Rollout`` is a
type-checking-only annotation, so the grader contract stays importable on any box.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from scratch_llm.rollout.types import Rollout


class RewardDict(TypedDict):
    """The grader's output: the scalar ``reward`` + its ``format``/``answer`` components (the
    r1-zero decomposition that lets a run log *why* a response scored what it did)."""

    reward: float
    format_reward: float
    answer_reward: float


@dataclass(frozen=True)
class Task:
    """One verifiable problem: a tokenized prompt + the ground truth its answer is graded against."""

    task_id: str
    prompt_ids: tuple[int, ...]
    ground_truth: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Graded:
    """The result of grading one rollout: the reward decomposition + the decoded response text."""

    reward: float
    format_reward: float
    answer_reward: float
    response_text: str


@runtime_checkable
class RewardFn(Protocol):
    """A text-in grader: ``response_text`` + ``ground_truth`` → ``RewardDict``. Keyword-only so a
    call site can never silently transpose the two strings."""

    def __call__(self, *, response_text: str, ground_truth: str) -> RewardDict: ...


@runtime_checkable
class DecodeFn(Protocol):
    """Token ids → response text (the env owns detokenization, so the engine stays tokenizer-free)."""

    def __call__(self, ids: Sequence[int]) -> str: ...


@runtime_checkable
class VerifiableEnv(Protocol):
    """The structural env contract: expose a decoder, the task set, and a grader."""

    def decode(self) -> DecodeFn: ...

    def tasks(self) -> list[Task]: ...

    def grade(self, task: Task, rollout: Rollout) -> Graded: ...
