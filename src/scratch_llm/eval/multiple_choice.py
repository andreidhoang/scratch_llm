"""Multiple-choice scoring — the ARC / MMLU report-card family.

A multiple-choice task is scored *by likelihood*, not by generation: for each option, compute the
model's log-probability of the option tokens **conditioned on the prompt** (teacher-forced), and
pick the argmax. Length-normalization (average log-prob per option token) is on by default — it
removes the bias toward shorter options that raw summed log-prob introduces (the standard
ARC/MMLU convention).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from scratch_llm.eval.protocols import TextTokenizer


@dataclass(frozen=True)
class MCResult:
    accuracy: float
    n: int


@torch.no_grad()
def option_logprob(
    model: nn.Module,
    prompt_ids: Sequence[int],
    option_ids: Sequence[int],
    *,
    device: str = "cpu",
    length_normalize: bool = True,
) -> float:
    """log p(option | prompt): the summed (or length-averaged) per-token log-prob of the option
    tokens, teacher-forced after the prompt."""
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty (option token 0 is scored from prompt[-1])")
    if len(option_ids) == 0:
        raise ValueError("option_ids must be non-empty")
    seq = torch.tensor([list(prompt_ids) + list(option_ids)], dtype=torch.long, device=device)
    logits = model(seq)[0]  # (T, V)
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = len(prompt_ids)
    total = 0.0
    for j, token in enumerate(option_ids):
        total += float(logp[p + j - 1, token])  # logits at pos (p+j-1) predict option token j
    return total / len(option_ids) if length_normalize else total


@torch.no_grad()
def predict_choice(
    model: nn.Module,
    prompt_ids: Sequence[int],
    options_ids: Sequence[Sequence[int]],
    *,
    device: str = "cpu",
    length_normalize: bool = True,
) -> int:
    """Index of the highest-log-prob option."""
    scores = [
        option_logprob(model, prompt_ids, opt, device=device, length_normalize=length_normalize)
        for opt in options_ids
    ]
    return max(range(len(scores)), key=scores.__getitem__)


@torch.no_grad()
def evaluate_multiple_choice(
    model: nn.Module,
    tokenizer: TextTokenizer,
    examples: Sequence[tuple[str, Sequence[str], int]],
    *,
    device: str = "cpu",
    length_normalize: bool = True,
) -> MCResult:
    """Accuracy over ``(prompt, options, correct_index)`` examples."""
    if not examples:
        raise ValueError("examples must be non-empty")
    correct = 0
    for prompt, options, gold in examples:
        pids = tokenizer.encode(prompt)
        oids = [tokenizer.encode(o) for o in options]
        if (
            predict_choice(model, pids, oids, device=device, length_normalize=length_normalize)
            == gold
        ):
            correct += 1
    return MCResult(correct / len(examples), len(examples))
