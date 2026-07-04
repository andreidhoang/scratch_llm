"""The r1-zero verifiable grader — rule-based format + answer reward, no human, no learned RM.

Intent: the binary grader that makes reasoning RL *verifiable* (A5 §3.2 / guide §2(8)). The rollout
is graded on two axes and the scalar the update optimizes composes them:

- **format_reward** — the response continues the r1-zero prompt (which already ends with
  ``"Assistant: <think>"``), so a well-formed response must contain the exact separator
  ``"</think> <answer>"`` (single space, official-grader strict) *and* a closing ``"</answer>"``.
- **answer_reward** — normalized exact match of the extracted ``<answer>…</answer>`` span against
  the ground truth: whitespace-stripped, numeric-with-tolerance when both sides parse as floats
  (``"42.000" == "42"``, digit-grouping commas dropped), casefolded string equality otherwise.
- **reward = format_reward × answer_reward** — mirrored from the official
  ``cs336_alignment/drgrpo_grader.r1_zero_reward_fn`` convention: reward is 1.0 **only when both**
  format and answer pass; a *formatted-but-wrong* response gets ``format_reward=1.0`` recorded but
  ``reward=0.0`` (no partial credit — partial format credit is a reward-hacking surface); a
  malformed response is all-zeros and its answer is **never graded**, even if the correct string
  appears somewhere in the text.

Invariant: for every input, ``reward == format_reward * answer_reward`` and each component is
exactly 0.0 or 1.0. Answer extraction mirrors the official grader verbatim:
``response.split("<answer>")[-1].replace("</answer>", "")`` (last ``<answer>`` wins).

Interview question this module answers: "Why does verifiable-reward RL grade *format* separately
from *answer*, and why must a formatted-but-wrong response score 0 total reward rather than
partial credit?"
"""

from __future__ import annotations

import math
import re

from scratch_llm.envs.protocol import RewardDict

# The official A5 r1-zero prompt (cs336_alignment/prompts/r1_zero.prompt). The prompt itself ends
# with "<think>", so the *response* carries "…</think> <answer>…</answer>".
R1_ZERO_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant "
    "solves it. The Assistant first thinks about the reasoning process in the mind and then "
    "provides the User with the answer. The reasoning process is enclosed within <think> "
    "</think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)

# Strict single-space separator — the official grader's exact format gate.
_FORMAT_SEPARATOR = "</think> <answer>"
_ANSWER_CLOSE = "</answer>"

# Numeric-looking string with optional digit-grouping commas ("1,000.5", "-3", "2e-4").
_NUMERIC_RE = re.compile(r"[+-]?[\d,]*\.?\d+(?:[eE][+-]?\d+)?")


def render_r1_zero_prompt(question: str) -> str:
    """Render ``question`` into the official r1-zero prompt (ends with the open ``<think>`` tag)."""
    return R1_ZERO_PROMPT_TEMPLATE.format(question=question)


def response_format_reward(response_text: str) -> float:
    """1.0 iff the response has the exact r1-zero shape: ``…</think> <answer>…</answer>``.

    Strict on the single space between the tags (official-grader semantics): a model that drops
    the separator earns no format reward, which is what teaches the policy the output contract.
    """
    ok = _FORMAT_SEPARATOR in response_text and _ANSWER_CLOSE in response_text
    return 1.0 if ok else 0.0


def extract_answer_span(response_text: str) -> str | None:
    """The raw text inside the answer tags, or None when the format gate fails.

    Mirrors the official grader verbatim: split on ``<answer>`` and keep the *last* chunk with all
    ``</answer>`` occurrences removed — a response emitting several answer tags is graded on the
    final one.
    """
    if response_format_reward(response_text) == 0.0:
        return None
    return response_text.split("<answer>")[-1].replace(_ANSWER_CLOSE, "")


def _parse_number(text: str) -> float | None:
    """Parse ``text`` as a float, tolerating digit-grouping commas; None when not numeric."""
    candidate = text.strip()
    if not _NUMERIC_RE.fullmatch(candidate):
        return None
    try:
        return float(candidate.replace(",", ""))
    except ValueError:  # pathological comma placement, e.g. ",,1"
        return None


def answers_match(model_answer: str, ground_truth: str) -> bool:
    """Normalized exact match: numeric-with-tolerance when both parse, else casefolded strings.

    Numeric tolerance (``rel_tol=1e-6``) accepts representation drift ("42.000", "1,000" vs
    "1000") but rejects genuinely different values ("3.1416" vs "3.14159" differ at 1e-5 — no
    match). Casefolding is safe on the string path because *both* sides are folded.
    """
    a, b = model_answer.strip(), ground_truth.strip()
    num_a, num_b = _parse_number(a), _parse_number(b)
    if num_a is not None and num_b is not None:
        return math.isclose(num_a, num_b, rel_tol=1e-6, abs_tol=1e-9)
    return a.casefold() == b.casefold()


def r1_zero_reward(*, response_text: str, ground_truth: str) -> RewardDict:
    """Grade one response: the ``RewardFn``-conforming (keyword-only) r1-zero grader.

    Composition per the official grader: malformed → all zeros (answer never inspected);
    well-formed → ``answer_reward`` from :func:`answers_match`, ``reward = format × answer``.
    """
    fmt = response_format_reward(response_text)
    if fmt == 0.0:
        return {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
    span = extract_answer_span(response_text)
    assert span is not None  # format gate passed above
    ans = 1.0 if answers_match(span, ground_truth) else 0.0
    return {"reward": fmt * ans, "format_reward": fmt, "answer_reward": ans}


def r1_zero_reward_fn(response: str, ground_truth: str) -> RewardDict:
    """Positional-argument alias matching the official ``drgrpo_grader.r1_zero_reward_fn`` shape
    (the W9 adapter one-liner); delegates to :func:`r1_zero_reward`."""
    return r1_zero_reward(response_text=response, ground_truth=ground_truth)
