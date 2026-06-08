"""Keystone leaf — the L5 contracts: text-in RewardFn, frozen Task/Graded, structural
VerifiableEnv (ADR-0010). Imports no torch (Rollout is a TYPE_CHECKING-only annotation)."""

import dataclasses

import pytest

from reasoning_llm.envs.protocol import (
    DecodeFn,
    Graded,
    RewardDict,
    RewardFn,
    Task,
    VerifiableEnv,
)


def test_reward_dict_keys():
    rd: RewardDict = {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}
    assert set(rd) == {"reward", "format_reward", "answer_reward"}


def test_task_and_graded_are_frozen():
    t = Task(task_id="q1", prompt_ids=(1, 2, 3), ground_truth="5")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.ground_truth = "6"  # type: ignore[misc]
    g = Graded(reward=1.0, format_reward=1.0, answer_reward=1.0, response_text="<answer>5</answer>")
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.reward = 0.0  # type: ignore[misc]


def test_task_metadata_defaults_independently():
    a = Task(task_id="a", prompt_ids=(1,), ground_truth="x")
    b = Task(task_id="b", prompt_ids=(2,), ground_truth="y")
    assert a.metadata == {} and b.metadata == {}
    assert a.metadata is not b.metadata  # default_factory, not a shared mutable default


def test_text_in_reward_fn_satisfies_protocol():
    def grader(*, response_text: str, ground_truth: str) -> RewardDict:
        ok = float(response_text.strip() == ground_truth.strip())
        return {"reward": ok, "format_reward": 1.0, "answer_reward": ok}

    assert isinstance(grader, RewardFn)
    assert grader(response_text="5", ground_truth="5")["reward"] == 1.0
    assert grader(response_text="7", ground_truth="5")["reward"] == 0.0


def test_verifiable_env_is_structural():
    # A minimal structural implementer satisfies the Protocol without inheriting it — this is what
    # lets the toy CPU env and a real HF env both feed the same engine.
    class _Env:
        def decode(self):
            return lambda ids: "".join(map(str, ids))

        def tasks(self):
            return [Task(task_id="q", prompt_ids=(1,), ground_truth="1")]

        def grade(self, task, rollout):
            return Graded(1.0, 1.0, 1.0, "1")

    assert isinstance(_Env(), VerifiableEnv)
    assert isinstance(_Env().decode(), DecodeFn)
