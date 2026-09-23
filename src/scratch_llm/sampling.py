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

Two model families decode through the same loop (:data:`CausalLM`). ``use_cache=True`` prefills
the prompt once and then feeds one token per forward; ``use_cache=False`` recomputes the whole
window every step and is the oracle the cached path must match.

- **TransformerLM:** the cache is a :class:`~scratch_llm.model.KVCache`; the window is
  ``cfg.context_length`` (RoPE's table ends there).
- **K3Model:** the cache is a :class:`~scratch_llm.k3.model.HybridState` (one constant-size KDA
  state per KDA layer, a growing latent KV per MLA layer), allocated in the embedding's dtype,
  which is the dtype every activation starts in. K3 has no positional encoding, so nothing in
  the recurrence caps the length; ``cfg.max_position_embeddings`` plays the window's role. The
  uncached path crops to it, like the dense one. The cached path cannot: a KDA state is a sum
  over every token it has read, and no operation removes the oldest one. So a cached K3 decode
  whose prompt plus budget exceeds the window is refused before the first forward, instead of
  silently computing something the oracle does not. Past the window, the uncached K3 logprobs
  score the cropped context, so they no longer equal a teacher-forced re-score of the whole
  sequence (``LocalBackend.score`` does not crop; on a TransformerLM it raises there instead).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from scratch_llm.k3.model import HybridState, K3Model
from scratch_llm.model import KVCache, TransformerLM, softmax

#: The models this module decodes. Each carries its own cache type and window; see
#: :func:`_context_length` and :func:`_decode_cache`.
CausalLM = TransformerLM | K3Model


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


def _logprob_of(logits: Tensor, token_id: int) -> float:
    """Policy log π(token | context) at temperature 1 = log_softmax(logits)[token].

    The taken token's log-prob under the *raw* model distribution (not temperature/top-p
    scaled) — the PPO/GRPO convention that feeds advantages and IS-ratios (ADR / see
    docs/design/L2_rollout_seam_SPEC.md §3)."""
    return float(torch.log_softmax(logits, dim=-1)[token_id])


def _context_length(model: CausalLM) -> int:
    """The most trailing tokens one forward may see: ``context_length`` for a TransformerLM,
    ``max_position_embeddings`` for a K3Model (module docstring).

    This and :func:`_decode_cache` both test for K3Model and decode every other model as dense,
    so they agree on every type. A wrapper around a TransformerLM, such as ``torch.compile``'s
    OptimizedModule, is not an instance of it but forwards ``cfg``, ``blocks`` and the cache
    argument, and decodes on the dense path as it did before K3 existed."""
    if isinstance(model, K3Model):
        return model.cfg.max_position_embeddings
    return model.cfg.context_length


def _decode_cache(
    model: CausalLM, n_prompt: int, max_tokens: int, device: str
) -> KVCache | HybridState:
    """An empty cache for one stream. For K3, refuse up front when ``n_prompt + max_tokens``
    exceeds the window: the cached loop feeds every generated token back (the last one too), so
    the state would grow past the window the uncached oracle crops to, and a recurrent state
    cannot drop its oldest tokens. It is the bound the dense cache hits too, as an IndexError on
    RoPE's table. The check reads the budget, not the tokens produced, so the refusal never
    depends on what was sampled (a stop token that would have ended the decode early)."""
    if isinstance(model, K3Model):
        limit = model.cfg.max_position_embeddings
        if n_prompt + max_tokens > limit:
            raise ValueError(
                f"a cached K3 decode needs prompt + max_tokens <= max_position_embeddings, got "
                f"{n_prompt} + {max_tokens} > {limit}: the recurrent state cannot drop its "
                "oldest tokens the way the uncached window does. Shorten the prompt or the "
                "budget, or pass use_cache=False."
            )
        dtype = model.embed_tokens.weight.dtype
        return HybridState.empty(model.cfg, 1, torch.device(device), dtype)
    return KVCache(len(model.blocks))


@torch.no_grad()
def _decode(
    model: CausalLM,
    prompt_ids: Sequence[int],
    params: SamplingParams,
    device: str,
    use_cache: bool,
) -> tuple[list[int], list[float]]:
    """Shared decode core: returns (generated_ids, per-token policy log π at temperature 1).

    The single source of decode truth for both :func:`generate` and
    :func:`generate_with_logprobs`. Each logprob is read from the *same* logits its token was
    sampled from, so it is exact (and matches an independent teacher-forced re-score)."""
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")
    if params.seed is not None:
        torch.manual_seed(params.seed)

    model.eval()
    context_length = _context_length(model)
    ids = list(prompt_ids)
    generated: list[int] = []
    logprobs: list[float] = []

    if use_cache:
        cache = _decode_cache(model, len(ids), params.max_tokens, device)
        x = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)
        logits = model(x, cache)[0, -1]  # prefill the prompt
        for _ in range(params.max_tokens):
            next_id = _sample_next(logits, params)
            generated.append(next_id)
            logprobs.append(_logprob_of(logits, next_id))
            if next_id in params.stop_ids:
                break
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits = model(x, cache)[0, -1]  # decode one token against the cache
        return generated, logprobs

    for _ in range(params.max_tokens):
        window = ids[-context_length:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        next_id = _sample_next(logits, params)
        ids.append(next_id)
        generated.append(next_id)
        logprobs.append(_logprob_of(logits, next_id))
        if next_id in params.stop_ids:
            break
    return generated, logprobs


def generate(
    model: CausalLM,
    prompt_ids: Sequence[int],
    params: SamplingParams,
    device: str = "cpu",
    use_cache: bool = True,
) -> list[int]:
    """Autoregressively decode a continuation for a single prompt.

    Returns the generated token ids only (not the prompt). Stops at ``max_tokens`` or as
    soon as a ``stop_ids`` token is emitted (the stop token is included in the output).

    ``use_cache=True`` uses the model's cache (prefill the prompt once, then feed each new
    token against it); ``use_cache=False`` recomputes the window each step and is the
    correctness oracle. The two paths must produce identical output while the whole sequence
    fits the window. Past it, the uncached path crops to the last window of tokens; the dense
    cached path has no sliding-window eviction yet, and the K3 cached path refuses the call
    (module docstring).
    """
    return _decode(model, prompt_ids, params, device, use_cache)[0]


def generate_with_logprobs(
    model: CausalLM,
    prompt_ids: Sequence[int],
    params: SamplingParams,
    device: str = "cpu",
    use_cache: bool = True,
) -> tuple[list[int], list[float]]:
    """Like :func:`generate`, but also returns the per-token policy log π(a_t|s_t) at
    temperature 1. The rollout seam and RL spine need these logprobs for advantages and
    IS-ratios; see ``rollout/local.py`` and ``utils/monitors.py``."""
    return _decode(model, prompt_ids, params, device, use_cache)
