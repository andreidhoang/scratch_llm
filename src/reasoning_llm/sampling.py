"""Decoding — temperature / top-p sampling, the exploration knob of every rollout.

L1 substrate (A1). ``SamplingParams`` is the **single source of truth** for decode config
(ADR-0003): the same object configures this from-scratch sampler and, later,
``rollout/sglang_client.py``. If the training-time sampler and the serving sampler ever
disagree on temperature/top-p, ``kl_train_infer`` measures a *config* mismatch instead of
real engine drift — so this contract is shared, not duplicated.

Correctness invariants (tested in tests/test_sampling.py):
- **Greedy = argmax:** ``temperature == 0`` picks the argmax token, deterministically.
- **Nucleus:** top-p keeps the smallest high-probability set and renormalizes.
- **Stop:** generation halts the moment a ``stop_ids`` token is emitted.
- **Budget:** without a stop token, exactly ``max_tokens`` tokens are produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from reasoning_llm.model import KVCache, TransformerLM, softmax


@dataclass(frozen=True)
class SamplingParams:
    """Decode configuration. Frozen so it is a stable contract shared across engines."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    stop_ids: tuple[int, ...] = ()
    seed: int | None = None


def _top_p_filter(probs: Tensor, top_p: float) -> Tensor:
    """Zero out the tail outside the nucleus and renormalize. ``probs`` is 1-D over vocab.

    Keeps the smallest set of highest-probability tokens whose cumulative mass reaches
    ``top_p`` (the boundary-crossing token is kept), matching the standard HF semantics.
    """
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()  # shift right so the crossing token is kept
    remove[0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum()
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out


def _sample_next(logits: Tensor, params: SamplingParams) -> int:
    """Pick the next token id from final-position logits (1-D over vocab)."""
    if params.temperature == 0.0:
        return int(logits.argmax().item())
    probs = softmax(logits / params.temperature, dim=-1)
    probs = _top_p_filter(probs, params.top_p)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: Sequence[int],
    params: SamplingParams,
    device: str = "cpu",
    use_cache: bool = True,
) -> list[int]:
    """Autoregressively decode a continuation for a single prompt.

    Returns the generated token ids only (not the prompt). Stops at ``max_tokens`` or as
    soon as a ``stop_ids`` token is emitted (the stop token is included in the output).

    ``use_cache=True`` uses a :class:`KVCache` (prefill the prompt once, then attend each new
    token against the cache); ``use_cache=False`` recomputes the full prefix each step and is
    the correctness oracle. The two paths must produce identical output. Both assume the whole
    sequence stays within ``model.cfg.context_length`` (no sliding-window eviction yet).
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")
    if params.seed is not None:
        torch.manual_seed(params.seed)

    model.eval()
    context_length = model.cfg.context_length
    ids = list(prompt_ids)
    generated: list[int] = []

    if use_cache:
        cache = KVCache(len(model.blocks))
        x = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)
        logits = model(x, cache)[0, -1]  # prefill the prompt
        for _ in range(params.max_tokens):
            next_id = _sample_next(logits, params)
            generated.append(next_id)
            if next_id in params.stop_ids:
                break
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = model(x, cache)[0, -1]  # decode one token against the cache
        return generated

    for _ in range(params.max_tokens):
        window = ids[-context_length:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        next_id = _sample_next(logits, params)
        ids.append(next_id)
        generated.append(next_id)
        if next_id in params.stop_ids:
            break
    return generated
