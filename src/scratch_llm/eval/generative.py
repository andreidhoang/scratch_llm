"""Generative report-card family — GSM8K / HumanEval-style tasks scored by *generation + grading*.

Unlike multiple-choice (scored by likelihood), a generative task samples a completion and grades it
with a task-specific verifier. This reuses the from-scratch decoder (`sampling.generate`) and any
`grade_fn(completion_text, reference) -> bool` — e.g. the existing verifiable-reward graders in
`rewards/` for math (exact-answer) or a unit-test runner for code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from scratch_llm.eval.protocols import TextTokenizer
from scratch_llm.model import TransformerLM
from scratch_llm.sampling import SamplingParams, generate


@dataclass(frozen=True)
class GenResult:
    accuracy: float
    n: int


@torch.no_grad()
def evaluate_generative(
    model: TransformerLM,
    tokenizer: TextTokenizer,
    examples: Sequence[tuple[str, str]],
    grade_fn: Callable[[str, str], bool],
    *,
    params: SamplingParams | None = None,
    device: str = "cpu",
) -> GenResult:
    """Accuracy over ``(prompt, reference)`` examples: generate a completion for each prompt,
    decode it to text, and count it correct iff ``grade_fn(completion, reference)`` is True.

    Defaults to greedy decoding (temperature 0) so the score is deterministic.
    """
    if not examples:
        raise ValueError("examples must be non-empty")
    sp = params or SamplingParams(temperature=0.0, max_tokens=64)
    correct = 0
    for prompt, reference in examples:
        prompt_ids = tokenizer.encode(prompt)
        gen_ids = generate(model, prompt_ids, sp, device=device)
        completion = tokenizer.decode(gen_ids)
        if grade_fn(completion, reference):
            correct += 1
    return GenResult(correct / len(examples), len(examples))
