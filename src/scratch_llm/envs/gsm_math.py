"""GSM-style math env — the minimal numeric-answer ``VerifiableEnv`` over a (question, answer)
fixture list.

Intent: the thin bridge between a math QA dataset (GSM8K/MATH rows, or the built-in hermetic
fixture) and the RL loop: render each question through the r1-zero template, grade responses with
the :mod:`scratch_llm.rewards.r1_zero` grader (format gate × normalized-numeric answer match).
Where :mod:`~scratch_llm.envs.countdown` grades by *evaluating* an expression, this env grades by
*matching* a final numeric answer — the two together exercise both halves of verifiable rewarding.

Invariant: ``grade`` is exactly ``r1_zero_reward`` applied to the decoded response against
``Task.ground_truth`` — this env adds no grading logic of its own, so official-grader semantics
(malformed → all-zeros, formatted-but-wrong → reward 0) hold here by construction.

Interview question this module answers: "You have a list of (question, numeric answer) pairs —
what is the *minimum* machinery to turn it into an RL environment with verifiable rewards?"
"""

from __future__ import annotations

from collections.abc import Sequence

from scratch_llm.envs.countdown import ByteTextCodec, TextCodec
from scratch_llm.envs.protocol import DecodeFn, Graded, Task
from scratch_llm.rewards.r1_zero import r1_zero_reward, render_r1_zero_prompt
from scratch_llm.rollout.types import Rollout

# Hermetic built-in fixture (hand-written GSM-style word problems; answers are exact integers so
# the numeric-tolerance path is the only leniency exercised). Tests run without any download.
DEFAULT_FIXTURE: tuple[tuple[str, str], ...] = (
    (
        "Natalia sold 48 clips in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?",
        "72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes of babysitting. "
        "How much did she earn, in dollars?",
        "10",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
        "3",
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. "
        "How many pages does he write in a year?",
        "624",
    ),
    (
        "Mark has 5 boxes with 6 apples each. He gives away 8 apples. How many apples remain?",
        "22",
    ),
)


class GSMMathEnv:
    """Structural ``VerifiableEnv`` over ``(question, answer)`` pairs (defaults to the hermetic
    built-in fixture); r1-zero prompts, r1-zero grading, byte-level codec unless one is given."""

    def __init__(
        self,
        items: Sequence[tuple[str, str]] | None = None,
        *,
        codec: TextCodec | None = None,
    ) -> None:
        rows = tuple(items) if items is not None else DEFAULT_FIXTURE
        if not rows:
            raise ValueError("GSMMathEnv needs at least one (question, answer) pair")
        self._codec: TextCodec = codec if codec is not None else ByteTextCodec()
        self._tasks = [
            Task(
                task_id=f"gsm-{i}",
                prompt_ids=tuple(self._codec.encode(render_r1_zero_prompt(question))),
                ground_truth=answer,
                metadata={"question": question},
            )
            for i, (question, answer) in enumerate(rows)
        ]

    def decode(self) -> DecodeFn:
        return self._codec.decode

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def grade(self, task: Task, rollout: Rollout) -> Graded:
        text = self._codec.decode(rollout.response_ids)
        rd = r1_zero_reward(response_text=text, ground_truth=task.ground_truth)
        return Graded(
            reward=rd["reward"],
            format_reward=rd["format_reward"],
            answer_reward=rd["answer_reward"],
            response_text=text,
        )
