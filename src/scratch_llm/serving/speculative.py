"""A1 Rung 4.3 — lossless speculative decoding.

Decode is memory-bound (one weight read per token, AI ≈ 1). Speculation raises arithmetic intensity
from the *other* side than batching: a cheap **drafter** proposes K tokens, the target verifies all
K in ONE forward, and we accept the longest prefix that matches the target's own greedy argmax —
taking the target's token at the first mismatch. The committed sequence is EXACTLY what the target
would have produced alone: **lossless for greedy by construction** (kill line: any divergence from
``sampling.generate`` greedy means the accept/rollback logic is wrong — fix the math, never measure
speedup on a broken oracle).

Drafter-agnostic (`Drafter` protocol). The measured default is :class:`NGramDrafter`
(prompt-lookup / Saxena 2023) — training-free, model-agnostic, real acceptance on structured/repeated
text; a smaller draft model (:class:`ModelDrafter`) slots into the same protocol.

Interview question this answers: how does speculative decoding stay *distribution-identical* to the
target while doing fewer target forwards, and where does its speedup come from (and vanish)?
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from scratch_llm.model import KVCache, TransformerLM


class Drafter(Protocol):
    """Proposes up to ``k`` continuation tokens for a context. Cheap and best-effort — wrong
    guesses cost nothing but a rejected verification slot (the target corrects them)."""

    def propose(self, context_ids: Sequence[int], k: int) -> list[int]: ...


@dataclass
class NGramDrafter:
    """Prompt-lookup speculation (Saxena 2023): find the most recent earlier occurrence of the last
    ``n`` context tokens and propose the ``k`` tokens that followed it. Training-free and lossless
    (the target verifies every guess); acceptance is high on repetitive/structured text (code, JSON,
    repeated phrases), ~0 on unstructured text — the honest regime bound."""

    n: int = 3

    def propose(self, context_ids: Sequence[int], k: int) -> list[int]:
        ids = list(context_ids)
        n = self.n
        if k <= 0 or len(ids) <= n:
            return []
        pattern = ids[-n:]
        # scan right-to-left for the most recent earlier match of the last-n-gram
        for start in range(len(ids) - n - 1, -1, -1):
            if ids[start : start + n] == pattern:
                return ids[start + n : start + n + k]
        return []


@dataclass
class ModelDrafter:
    """A smaller draft model proposes ``k`` tokens by greedy autoregression from the context. Same
    protocol as :class:`NGramDrafter`; losslessness holds for any draft quality (the target verifies),
    only the acceptance rate — hence speedup — depends on how well the draft tracks the target."""

    model: TransformerLM
    device: str = "cpu"

    @torch.no_grad()
    def propose(self, context_ids: Sequence[int], k: int) -> list[int]:
        if k <= 0:
            return []
        self.model.eval()
        ctx = self.model.cfg.context_length
        cache = KVCache(len(self.model.blocks))
        window = list(context_ids)[-ctx:]
        x = torch.tensor([window], dtype=torch.long, device=self.device)
        logits = self.model(x, cache)[0, -1]
        out: list[int] = []
        for _ in range(k):
            nxt = int(logits.argmax())
            out.append(nxt)
            x = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
            logits = self.model(x, cache)[0, -1]
        return out


@dataclass(frozen=True)
class SpecStats:
    """Speculative-decode accounting. ``mean_tokens_per_forward`` is the speedup proxy (plain decode
    = 1.0); ``acceptance_rate`` = accepted / drafted."""

    n_tokens: int
    n_target_forwards: int  # includes the 1 prefill forward
    n_drafted: int
    n_accepted: int

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / self.n_drafted if self.n_drafted else 0.0

    @property
    def mean_tokens_per_forward(self) -> float:
        # exclude the prefill forward: it exists in plain decode too
        decode_forwards = max(self.n_target_forwards - 1, 1)
        return self.n_tokens / decode_forwards


@torch.no_grad()
def speculative_generate(
    target: TransformerLM,
    drafter: Drafter,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    device: str = "cpu",
    k: int = 4,
    on_round: Callable[[int, int], None] | None = None,
) -> tuple[list[int], SpecStats]:
    """Greedy speculative decode — token-IDENTICAL to ``sampling.generate(temperature=0)``.

    ``on_round(n_drafted, n_accepted)`` is called once per verification round, if given. It exists
    so per-position acceptance can be measured without a second copy of the accept loop:
    ``SpecStats`` carries only run totals, and an average over positions is exactly the summary
    S1/S-R3 is built to show is insufficient (``serving/acceptance.PositionAcceptance``).

    Invariant (the ``pending`` token): at the top of every round ``pending`` is the next committed
    token whose K/V is NOT yet in the cache and which equals the target's greedy argmax (correct by
    construction). We draft K guesses for the tokens after it, run ONE target forward over
    ``[pending, *drafts]``, accept the longest draft prefix matching the target's argmax, take the
    target's argmax at the first mismatch as the next ``pending``, and roll the cache back to discard
    the rejected drafts' K/V. Each round commits ``1 + accepted`` tokens for ONE target forward — the
    speedup — while the output stays exactly the target's greedy sequence.
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be ≥ 1")
    if k < 0:
        raise ValueError("k must be ≥ 0")
    target.eval()
    ctx = target.cfg.context_length
    cache = KVCache(len(target.blocks))
    prompt = list(prompt_ids)[-ctx:]

    x = torch.tensor([prompt], dtype=torch.long, device=device)
    logits = target(x, cache)[0, -1]  # prefill; cache.length == len(prompt)
    pending = int(logits.argmax())

    generated: list[int] = []
    n_forwards = 1
    n_drafted = 0
    n_accepted = 0

    while len(generated) < max_new_tokens:
        context = prompt + generated + [pending]
        drafts = list(drafter.propose(context, k))[:k]
        n_drafted += len(drafts)

        base = cache.length
        inp = torch.tensor([[pending, *drafts]], dtype=torch.long, device=device)
        vlogits = target(inp, cache)[0]  # (1 + len(drafts), vocab); cache grew by that many
        n_forwards += 1
        greedy: list[int] = vlogits.argmax(dim=-1).tolist()  # greedy[i] = target token after inp[i]

        # accept the longest draft prefix matching the target's greedy prediction
        accepted = 0
        for i, d in enumerate(drafts):
            if d == greedy[i]:
                accepted += 1
            else:
                break
        n_accepted += accepted
        if on_round is not None:
            on_round(len(drafts), accepted)

        generated.append(pending)
        generated.extend(drafts[:accepted])
        pending = int(greedy[accepted])  # target's correction / free continuation = next pending
        cache.truncate(base + 1 + accepted)  # keep pending + accepted drafts; drop the rest

    tokens = generated[:max_new_tokens]
    return tokens, SpecStats(
        n_tokens=len(tokens),
        n_target_forwards=n_forwards,
        n_drafted=n_drafted,
        n_accepted=n_accepted,
    )
