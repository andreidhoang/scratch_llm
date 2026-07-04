"""Countdown — a guaranteed-solvable arithmetic-puzzle env behind ``envs/protocol.py``.

Intent: the CPU-cheap verifiable environment for the RL loop (guide §5's "R1-Zero aha on
Countdown" capstone runs on exactly this task family). Each task gives a list of numbers and a
target; the policy must emit, in r1-zero format, an arithmetic expression over ``+ - * /`` that
evaluates to the target using **each provided number at most once** (subsets allowed, standard
Countdown rules).

Three pieces:

- **Task pool generator** (:func:`generate_countdown_tasks`) — solvable **by construction**: a
  seeded RNG samples numbers, then builds a random left-associated expression over them, choosing
  only ops that keep the running value a positive integer (``/`` only when it divides exactly);
  the final value *is* the target, so a witness solution always exists (stored in
  ``Task.metadata["solution"]`` and verified in tests).
- **Expression grader** (:func:`evaluate_countdown_expression`) — a *safe* evaluator: parses with
  ``ast`` and walks an allowlist (int literals + binary ``+ - * /`` only — no names, calls,
  attributes, powers, or unary minus), enforces the numbers-multiset constraint (reusing a number
  more times than provided is cheating and rejects), evaluates in exact ``Fraction`` arithmetic
  (so ``(8/3)*3 == 8`` exactly), and returns ``None`` — never raises — on any invalid input
  including division by zero.
- **Env** (:class:`CountdownEnv`) — structural :class:`~scratch_llm.envs.protocol.VerifiableEnv`:
  prompts rendered via the r1-zero template, a byte-level codec for tokenize/decode (the env owns
  detokenization; a real tokenizer can be swapped in), and ``grade`` composing the r1-zero format
  gate with the expression grader (``reward = format_reward × answer_reward``, same convention as
  :mod:`scratch_llm.rewards.r1_zero`).

Invariants: every generated task's ``metadata["solution"]`` grades to ``answer_reward == 1.0``;
the same seed reproduces the same pool bit-for-bit; ``evaluate_countdown_expression`` never
raises on adversarial input.

Interview question this module answers: "Your RL env must grade model-written code/expressions —
how do you evaluate untrusted text safely, and how do you guarantee every task in the pool is
actually solvable?"
"""

from __future__ import annotations

import ast
import random
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from typing import Protocol, runtime_checkable

from scratch_llm.envs.protocol import DecodeFn, Graded, RewardDict, Task
from scratch_llm.rewards.r1_zero import render_r1_zero_prompt, response_format_reward
from scratch_llm.rollout.types import Rollout

# Adversarial-input guard: a model can emit arbitrarily long/deep text; cap before ast.parse so
# neither the parser nor the recursive evaluator can be driven to RecursionError/MemoryError.
MAX_EXPRESSION_CHARS = 256


def evaluate_countdown_expression(
    expression: str, allowed_numbers: Sequence[int]
) -> Fraction | None:
    """Safely evaluate an arithmetic expression; ``None`` on *any* invalid input.

    Valid means: parseable, only int literals and binary ``+ - * /`` (parentheses are transparent
    in the AST), every literal drawn from ``allowed_numbers`` with multiset multiplicity (each
    provided number used at most once), and no division by zero. Exact ``Fraction`` arithmetic —
    no float round-trip, so ``8/3*3`` equals 8 exactly.
    """
    if len(expression) > MAX_EXPRESSION_CHARS:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None

    used: Counter[int] = Counter()

    def ev(node: ast.expr) -> Fraction | None:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, int):
                return None  # floats/strings/True are not Countdown numbers
            used[value] += 1
            return Fraction(value)
        if isinstance(node, ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return None if right == 0 else left / right
            return None  # **, //, %, etc.
        return None  # names, calls, attributes, unary ops, …

    try:
        result = ev(tree.body)
    except RecursionError:  # belt-and-braces; the char cap should already prevent this
        return None
    if result is None:
        return None
    budget = Counter(allowed_numbers)
    if any(count > budget[number] for number, count in used.items()):
        return None  # reused a number beyond its multiplicity, or used an unlisted one
    return result


def grade_countdown_answer(expression: str, numbers: Sequence[int], target: int) -> bool:
    """True iff ``expression`` is valid under ``numbers`` and evaluates exactly to ``target``."""
    value = evaluate_countdown_expression(expression, numbers)
    return value is not None and value == target


def grade_countdown_response(
    response_text: str, *, numbers: Sequence[int], target: int
) -> RewardDict:
    """r1-zero-shaped grading of a full response: format gate × expression correctness.

    Same composition convention as the official grader: malformed → all zeros (the expression is
    never evaluated); well-formed but wrong/invalid expression → ``format_reward=1, reward=0``.
    """
    fmt = response_format_reward(response_text)
    if fmt == 0.0:
        return {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
    span = response_text.split("<answer>")[-1].replace("</answer>", "")
    ans = 1.0 if grade_countdown_answer(span.strip(), numbers, target) else 0.0
    return {"reward": fmt * ans, "format_reward": fmt, "answer_reward": ans}


def render_countdown_question(numbers: Sequence[int], target: int) -> str:
    """The user-turn question text for one Countdown task (wrapped by the r1-zero template)."""
    numbers_str = ", ".join(str(n) for n in numbers)
    return (
        f"Using the numbers [{numbers_str}], write an arithmetic expression that evaluates to "
        f"exactly {target}. You may use each number at most once, the operations + - * / and "
        "parentheses. Give only the expression inside the answer tags."
    )


def _build_solution(numbers: list[int], rng: random.Random) -> tuple[str, int]:
    """Fold the numbers into a random left-associated expression whose value stays a positive
    integer at every step; returns ``(expression, value)`` — the value becomes the target."""
    order = list(numbers)
    rng.shuffle(order)
    expr = str(order[0])
    acc = order[0]
    for n in order[1:]:
        ops = ["+", "*"]
        if acc - n >= 1:
            ops.append("-")
        if n != 0 and acc % n == 0 and acc // n >= 1:
            ops.append("/")
        op = rng.choice(ops)
        expr = f"({expr} {op} {n})"
        if op == "+":
            acc += n
        elif op == "-":
            acc -= n
        elif op == "*":
            acc *= n
        else:
            acc //= n
    return expr, acc


@runtime_checkable
class TextCodec(Protocol):
    """Structural encode/decode contract — any tokenizer with these two methods plugs in."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class ByteTextCodec:
    """UTF-8 byte-level codec: ids are raw bytes (0..255) — dependency-free tokenize/decode for
    CPU envs. Swap in a real tokenizer for GPU rollouts; the env interface does not change."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")


def generate_countdown_tasks(
    n_tasks: int,
    *,
    seed: int,
    n_numbers: int = 3,
    max_value: int = 10,
    codec: TextCodec | None = None,
) -> list[Task]:
    """A seeded pool of guaranteed-solvable Countdown :class:`Task` s.

    ``metadata`` carries ``numbers`` (comma-joined), ``target``, the witness ``solution``
    expression, and the raw ``question`` text. ``ground_truth`` is the target as a string
    (protocol contract); the numbers live in metadata because grading needs both.
    """
    if n_tasks < 1 or n_numbers < 2:
        raise ValueError(f"need n_tasks ≥ 1 and n_numbers ≥ 2, got {n_tasks}, {n_numbers}")
    rng = random.Random(seed)
    codec = codec or ByteTextCodec()
    tasks: list[Task] = []
    for i in range(n_tasks):
        numbers = [rng.randint(1, max_value) for _ in range(n_numbers)]
        solution, target = _build_solution(numbers, rng)
        question = render_countdown_question(numbers, target)
        prompt = render_r1_zero_prompt(question)
        tasks.append(
            Task(
                task_id=f"countdown-{seed}-{i}",
                prompt_ids=tuple(codec.encode(prompt)),
                ground_truth=str(target),
                metadata={
                    "numbers": ",".join(str(n) for n in numbers),
                    "target": str(target),
                    "solution": solution,
                    "question": question,
                },
            )
        )
    return tasks


class CountdownEnv:
    """Structural ``VerifiableEnv`` over a seeded Countdown pool (CPU-only, hermetic)."""

    def __init__(
        self,
        n_tasks: int = 16,
        *,
        seed: int = 0,
        n_numbers: int = 3,
        max_value: int = 10,
        codec: TextCodec | None = None,
    ) -> None:
        self._codec: TextCodec = codec if codec is not None else ByteTextCodec()
        self._tasks = generate_countdown_tasks(
            n_tasks, seed=seed, n_numbers=n_numbers, max_value=max_value, codec=self._codec
        )

    def decode(self) -> DecodeFn:
        return self._codec.decode

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def grade(self, task: Task, rollout: Rollout) -> Graded:
        text = self._codec.decode(rollout.response_ids)
        numbers = [int(n) for n in task.metadata["numbers"].split(",")]
        rd = grade_countdown_response(text, numbers=numbers, target=int(task.ground_truth))
        return Graded(
            reward=rd["reward"],
            format_reward=rd["format_reward"],
            answer_reward=rd["answer_reward"],
            response_text=text,
        )
