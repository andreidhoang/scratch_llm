"""W8b — Countdown + GSM math envs (`scratch_llm.envs.countdown`, `scratch_llm.envs.gsm_math`).

Pins: pool solvability-by-construction (every task's witness solution grades correct), seeded
determinism, safe expression grading (number-reuse cheating rejected, division-by-zero and
arbitrary code never raise), and structural `VerifiableEnv` conformance. Hermetic, CPU-only.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from scratch_llm.envs.countdown import (
    ByteTextCodec,
    CountdownEnv,
    evaluate_countdown_expression,
    generate_countdown_tasks,
    grade_countdown_answer,
    grade_countdown_response,
    render_countdown_question,
)
from scratch_llm.envs.gsm_math import DEFAULT_FIXTURE, GSMMathEnv
from scratch_llm.envs.protocol import DecodeFn, VerifiableEnv
from scratch_llm.rollout.types import Rollout


def _rollout_for(env: CountdownEnv | GSMMathEnv, prompt_ids: tuple[int, ...], text: str) -> Rollout:
    codec = ByteTextCodec()
    ids = tuple(codec.encode(text))
    return Rollout(
        prompt_ids=prompt_ids, response_ids=ids, logprobs=(0.0,) * len(ids), stop_reason="stop"
    )


# ------------------------------------------------------------------------------------------
# Countdown: safe expression evaluation (the grading edge cases)
# ------------------------------------------------------------------------------------------


def test_evaluate_basic_precedence():
    assert evaluate_countdown_expression("3 + 5 * 2", [3, 5, 2]) == Fraction(13)
    assert evaluate_countdown_expression("(3 + 5) * 2", [3, 5, 2]) == Fraction(16)


def test_subset_of_numbers_is_allowed():
    # Standard Countdown: each number used AT MOST once — using fewer is legal.
    assert evaluate_countdown_expression("3 + 5", [3, 5, 7]) == Fraction(8)


def test_reuse_number_cheating_rejected():
    assert evaluate_countdown_expression("3 + 3", [3, 5]) is None  # 3 provided once, used twice
    assert evaluate_countdown_expression("3 + 3", [3, 3, 5]) == Fraction(6)  # multiplicity honored


def test_unlisted_number_rejected():
    assert evaluate_countdown_expression("4 + 1", [3, 5]) is None


def test_division_by_zero_is_safe():
    assert evaluate_countdown_expression("5 / (3 - 3)", [5, 3, 3]) is None  # no exception


def test_exact_rational_arithmetic():
    # Float evaluation would give 7.999...; Fraction keeps it exact.
    assert evaluate_countdown_expression("8 / 3 * 3", [8, 3, 3]) == Fraction(8)
    assert grade_countdown_answer("8 / 3 * 3", [8, 3, 3], 8)


@pytest.mark.parametrize(
    "expression",
    [
        "2 ** 5",  # power not in the op allowlist
        "7 // 2",  # floor-div not allowed
        "-3 + 5",  # unary minus rejected (numbers are positive)
        "3.0 + 5",  # float literals are not Countdown numbers
        "__import__('os').system('true')",  # names/calls rejected — never executed
        "print(3)",
        "[3][0] + 5",
        "3; 5",  # not an expression
        "",  # empty
        "(((" + "3" + ")" * 500,  # unbalanced garbage
        "(" * 300 + "3" + ")" * 300,  # deep nesting hits the char cap, not the recursion limit
    ],
)
def test_hostile_expressions_return_none(expression: str):
    assert evaluate_countdown_expression(expression, [3, 5, 7]) is None


# ------------------------------------------------------------------------------------------
# Countdown: pool generation — solvable by construction, seeded
# ------------------------------------------------------------------------------------------


def test_pool_every_task_solvable_by_its_witness():
    tasks = generate_countdown_tasks(25, seed=7, n_numbers=4, max_value=10)
    assert len(tasks) == 25
    for task in tasks:
        numbers = [int(n) for n in task.metadata["numbers"].split(",")]
        target = int(task.ground_truth)
        assert task.metadata["target"] == task.ground_truth
        assert target >= 1  # construction keeps the running value a positive integer
        # The stored witness solution must grade correct — solvability is guaranteed, not hoped.
        assert grade_countdown_answer(task.metadata["solution"], numbers, target)
        # And through the full r1-zero-shaped response grader:
        response = f"trace </think> <answer>{task.metadata['solution']}</answer>"
        rd = grade_countdown_response(response, numbers=numbers, target=target)
        assert rd == {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}


def test_pool_seeded_determinism():
    a = generate_countdown_tasks(10, seed=3)
    b = generate_countdown_tasks(10, seed=3)
    c = generate_countdown_tasks(10, seed=4)
    assert a == b
    assert a != c


def test_prompt_is_r1_zero_rendered():
    (task,) = generate_countdown_tasks(1, seed=0)
    prompt = ByteTextCodec().decode(task.prompt_ids)
    assert prompt.endswith("Assistant: <think>")
    assert task.metadata["question"] in prompt
    assert f"exactly {task.ground_truth}" in task.metadata["question"]
    assert render_countdown_question([1, 2], 3).startswith("Using the numbers [1, 2]")


def test_countdown_response_grading_composition():
    rd_malformed = grade_countdown_response("<answer>3 + 5</answer>", numbers=[3, 5], target=8)
    assert rd_malformed == {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
    rd_wrong = grade_countdown_response(
        "x </think> <answer>3 + 5</answer>", numbers=[3, 5], target=9
    )
    assert rd_wrong == {"reward": 0.0, "format_reward": 1.0, "answer_reward": 0.0}


# ------------------------------------------------------------------------------------------
# CountdownEnv / GSMMathEnv: the VerifiableEnv seam
# ------------------------------------------------------------------------------------------


def test_countdown_env_satisfies_protocol_and_grades():
    env = CountdownEnv(4, seed=11)
    assert isinstance(env, VerifiableEnv)
    assert isinstance(env.decode(), DecodeFn)
    tasks = env.tasks()
    assert len(tasks) == 4
    task = tasks[0]
    good = _rollout_for(
        env, task.prompt_ids, f"t </think> <answer>{task.metadata['solution']}</answer>"
    )
    graded = env.grade(task, good)
    assert (graded.reward, graded.format_reward, graded.answer_reward) == (1.0, 1.0, 1.0)
    assert task.metadata["solution"] in graded.response_text
    bad = _rollout_for(env, task.prompt_ids, "no tags at all")
    assert env.grade(task, bad).reward == 0.0


def test_byte_codec_round_trip():
    codec = ByteTextCodec()
    text = "<think> 1+1 </think> <answer>2</answer>"
    assert codec.decode(codec.encode(text)) == text
    assert codec.vocab_size == 256


def test_gsm_env_default_fixture_and_grading():
    env = GSMMathEnv()
    assert isinstance(env, VerifiableEnv)
    tasks = env.tasks()
    assert len(tasks) == len(DEFAULT_FIXTURE) >= 5
    natalia = tasks[0]
    assert natalia.ground_truth == "72"
    correct = _rollout_for(env, natalia.prompt_ids, "24 + 48 </think> <answer> 72.0 </answer>")
    graded = env.grade(natalia, correct)
    assert (graded.reward, graded.format_reward, graded.answer_reward) == (1.0, 1.0, 1.0)
    wrong = _rollout_for(env, natalia.prompt_ids, "x </think> <answer>96</answer>")
    g_wrong = env.grade(natalia, wrong)
    assert (g_wrong.reward, g_wrong.format_reward, g_wrong.answer_reward) == (0.0, 1.0, 0.0)
    malformed = _rollout_for(env, natalia.prompt_ids, "72")
    assert env.grade(natalia, malformed).format_reward == 0.0


def test_gsm_env_custom_items_and_prompt_render():
    env = GSMMathEnv([("What is 2+2?", "4")])
    (task,) = env.tasks()
    prompt = env.decode()(task.prompt_ids)
    assert "What is 2+2?" in prompt
    assert prompt.endswith("Assistant: <think>")
    with pytest.raises(ValueError):
        GSMMathEnv([])
