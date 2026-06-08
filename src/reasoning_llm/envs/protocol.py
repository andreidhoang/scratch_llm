"""The L5 contract leaf — RewardFn / DecodeFn / Task / Graded / VerifiableEnv.

L5 envs · the keystone leaf (``docs/IMPLEMENTATION_PLAN.md`` §3.1; ADR-0010). Owns the contracts
every reward/env/algo module builds into, so nothing imports ``rewards.reward`` merely for a type —
together with ``levels.py`` this is what makes the L5 dependency graph acyclic.

Pinned contracts (ADR-0010, the four-metric ADR):
  * ``RewardFn`` is **text-in / dict-out**: the env decodes ``Rollout.response_ids`` to text once
    (via its ``DecodeFn``) and grades on text — a grader never sees token ids, because the
    from-scratch BPE and an HF tokenizer split the same string differently, so a token-id grader is
    not backend-agnostic.
  * advantage stays **array-in** (elsewhere): the env grades, the trainer passes a raw-reward array
    down; this module carries no scoring logic, only the shapes.

Falsifiable prediction: a grader fed token ids instead of text scores differently under the
from-scratch BPE vs an HF tokenizer on the same string. Kill criterion: if the env cannot own decode
at grade time, the text-in contract fails — fix the seam before building ``reward.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from reasoning_llm.rollout.types import Rollout

#: The grader's return shape: ``reward`` (the scalar the RL spine optimizes) plus the
#: ``format_reward`` / ``answer_reward`` decomposition the hack detector reads.
RewardDict = dict[str, float]


@dataclass(frozen=True)
class Task:
    """One verifiable problem. ``prompt_ids`` are tokenizer ids (the env owns the tokenizer);
    ``ground_truth`` is the reference answer the grader matches against, as text."""

    task_id: str
    prompt_ids: tuple[int, ...]
    ground_truth: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Graded:
    """The outcome of grading one rollout against one task — the per-sample row the trainer
    aggregates into ``reward_mean`` and the detector reads for ``hack_rate``."""

    reward: float
    format_reward: float
    answer_reward: float
    response_text: str


@runtime_checkable
class DecodeFn(Protocol):
    """Token ids → text. The env's single decode seam; called once per rollout before grading."""

    def __call__(self, ids: Sequence[int]) -> str: ...


@runtime_checkable
class RewardFn(Protocol):
    """Text-in / dict-out grader. Keyword-only so call sites are unambiguous and a token-id grader
    cannot be passed by accident (ADR-0010)."""

    def __call__(self, *, response_text: str, ground_truth: str) -> RewardDict: ...


@runtime_checkable
class VerifiableEnv(Protocol):
    """A pool of verifiable tasks that owns its tokenizer (``decode``) and grades rollouts under its
    configured ``HardeningLevel``. ``LocalBackend`` (CPU, CI) and a GPU HF/SGLang backend feed the
    *same* env, so the engine is backend-agnostic."""

    def decode(self) -> DecodeFn: ...

    def tasks(self) -> Sequence[Task]: ...

    def grade(self, task: Task, rollout: Rollout) -> Graded: ...
