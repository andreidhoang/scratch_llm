"""W8b — r1-zero verifiable grader (`scratch_llm.rewards.r1_zero`).

The truth table pins the official `drgrpo_grader.r1_zero_reward_fn` composition: reward = 1 only
when format AND answer both pass; formatted-but-wrong → format_reward 1 but reward 0; malformed →
all zeros with the answer never graded. Hermetic, CPU-only.
"""

from __future__ import annotations

import pytest

from scratch_llm.envs.protocol import RewardFn
from scratch_llm.rewards.r1_zero import (
    R1_ZERO_PROMPT_TEMPLATE,
    answers_match,
    extract_answer_span,
    r1_zero_reward,
    r1_zero_reward_fn,
    render_r1_zero_prompt,
    response_format_reward,
)

WELL_FORMED_CORRECT = "I add the halves. </think> <answer>42</answer>"
WELL_FORMED_WRONG = "I add the halves. </think> <answer>41</answer>"
FORMAT_ONLY = "no idea, guessing </think> <answer>banana</answer>"
MALFORMED_NO_TAGS = "The answer is 42"
MALFORMED_NO_SPACE = "reasoning</think><answer>42</answer>"  # official grader is space-strict
MALFORMED_UNCLOSED = "reasoning </think> <answer>42"


# ------------------------------------------------------------------------------------------
# The grader truth table (spec: well-formed correct / well-formed wrong / malformed / format-only)
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "gt", "expected"),
    [
        (WELL_FORMED_CORRECT, "42", {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}),
        (WELL_FORMED_WRONG, "42", {"reward": 0.0, "format_reward": 1.0, "answer_reward": 0.0}),
        (FORMAT_ONLY, "42", {"reward": 0.0, "format_reward": 1.0, "answer_reward": 0.0}),
        (MALFORMED_NO_TAGS, "42", {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}),
        (MALFORMED_NO_SPACE, "42", {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}),
        (MALFORMED_UNCLOSED, "42", {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}),
    ],
    ids=["correct", "wrong", "format-only", "no-tags", "no-space", "unclosed"],
)
def test_truth_table(response: str, gt: str, expected: dict[str, float]):
    assert r1_zero_reward(response_text=response, ground_truth=gt) == expected


def test_reward_is_format_times_answer_composition():
    # The official-grader convention, verified over the whole table: no row where a formatted-but-
    # wrong answer earns partial total reward (that would be a reward-hacking surface).
    for response in [
        WELL_FORMED_CORRECT,
        WELL_FORMED_WRONG,
        FORMAT_ONLY,
        MALFORMED_NO_TAGS,
        MALFORMED_NO_SPACE,
        MALFORMED_UNCLOSED,
    ]:
        rd = r1_zero_reward(response_text=response, ground_truth="42")
        assert rd["reward"] == rd["format_reward"] * rd["answer_reward"]
        assert set(rd) == {"reward", "format_reward", "answer_reward"}
        assert all(v in (0.0, 1.0) for v in rd.values())


def test_malformed_never_grades_answer_even_when_correct_text_present():
    # "42" appears verbatim, but the format gate failed → answer must NOT be inspected (official
    # semantics: the unformatted branch returns zeros unconditionally).
    rd = r1_zero_reward(response_text="42 </think><answer>42</answer>", ground_truth="42")
    assert rd == {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}


def test_last_answer_tag_wins():
    # Official extraction: split("<answer>")[-1] — a response with several tags is graded on the
    # final one.
    response = "x </think> <answer>1</answer> <answer>2</answer>"
    assert extract_answer_span(response) is not None
    assert r1_zero_reward(response_text=response, ground_truth="2")["reward"] == 1.0
    assert r1_zero_reward(response_text=response, ground_truth="1")["reward"] == 0.0


# ------------------------------------------------------------------------------------------
# Answer normalization: strip / casefold / numeric tolerance
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        (" 42 ", "42", True),  # strip
        ("42.000", "42", True),  # float-equal strings
        ("-.5", "-0.5", True),  # leading-dot float
        ("1,000", "1000", True),  # digit-grouping commas
        ("2e3", "2000", True),  # scientific notation
        ("Paris", "paris", True),  # casefold on the string path
        ("3.1416", "3.14159", False),  # 1e-5 relative difference — outside tolerance
        ("43", "42", False),
        ("banana", "42", False),  # non-numeric vs numeric falls to string compare
    ],
)
def test_answers_match_normalization(a: str, b: str, match: bool):
    assert answers_match(a, b) is match
    assert answers_match(b, a) is match  # symmetric


def test_numeric_tolerance_via_full_grader():
    rd = r1_zero_reward(
        response_text="half of 144 </think> <answer> 72.0 </answer>", ground_truth="72"
    )
    assert rd == {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}


# ------------------------------------------------------------------------------------------
# Protocol conformance + prompt template
# ------------------------------------------------------------------------------------------


def test_grader_satisfies_reward_fn_protocol():
    assert isinstance(r1_zero_reward, RewardFn)


def test_positional_alias_matches_official_call_shape():
    # The W9 adapter calls r1_zero_reward_fn(response, ground_truth) positionally.
    assert r1_zero_reward_fn(WELL_FORMED_CORRECT, "42")["reward"] == 1.0


def test_render_prompt_ends_with_open_think_tag():
    prompt = render_r1_zero_prompt("What is 2+2?")
    assert prompt.endswith("Assistant: <think>")
    assert "What is 2+2?" in prompt
    assert "{question}" in R1_ZERO_PROMPT_TEMPLATE


def test_format_reward_helper_matches_gate():
    assert response_format_reward(WELL_FORMED_CORRECT) == 1.0
    assert response_format_reward(MALFORMED_NO_SPACE) == 0.0
    assert extract_answer_span(MALFORMED_NO_TAGS) is None
