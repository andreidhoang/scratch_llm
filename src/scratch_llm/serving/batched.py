"""A1 Rung 3a — static batched decode (the weight-amortization lever).

Continuous batching beats the decode memory wall by reading each weight **once** and applying it to
**B** sequences per step → arithmetic intensity ≈ B, weight traffic amortized (R1 proved B=1 is memory-
and overhead-bound; a faster kernel can't fix an AI≈1 workload — batching raises the AI). This module is
the R3a precursor: **equal-length** greedy batched decode in lockstep, which the existing
:class:`~scratch_llm.model.KVCache` already supports — the batch axis is independent and RoPE positions
are shared when all rows are the same length. Row ``b`` is token-identical to a single-stream greedy
decode of prompt ``b`` (batch independence).

Variable-length + join/leave scheduling (real Orca continuous batching, R3b) needs the static
``BatchedKVCache`` buffer + per-row length masks + per-row positions — see
``performance/notes/A1_R3_continuous_batching.md``. That static buffer is also the R4.1/R4.4 linchpin.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from scratch_llm.model import KVCache, TransformerLM


@torch.no_grad()
def batched_greedy_decode(
    model: TransformerLM,
    prompts: Sequence[Sequence[int]],
    max_tokens: int,
    device: str = "cpu",
) -> list[list[int]]:
    """Greedy-decode a batch of **equal-length** prompts in lockstep (R3a static batch).

    Weights are read once per step and applied to all B rows → AI≈B (the lever that beats the B=1 memory
    wall). Requires equal prompt lengths so RoPE positions align across the batch; variable lengths +
    iteration-level scheduling is R3b (continuous batching). Returns the generated ids per prompt — row
    ``b`` == single-stream greedy decode of prompt ``b``.
    """
    if len(prompts) == 0:
        raise ValueError("prompts must be non-empty")
    if len({len(p) for p in prompts}) != 1:
        raise ValueError(
            "batched_greedy_decode requires equal-length prompts (R3a static batch); "
            "variable lengths are R3b continuous batching"
        )
    model.eval()
    b = len(prompts)
    cache = KVCache(len(model.blocks))
    x = torch.tensor([list(p) for p in prompts], dtype=torch.long, device=device)  # (B, L)
    logits = model(x, cache)[:, -1]  # (B, vocab) — prefill, take last position
    out: list[list[int]] = [[] for _ in range(b)]
    for _ in range(max_tokens):
        nid = logits.argmax(dim=-1)  # (B,) greedy — one weight read serves all B rows
        for i, tok in enumerate(nid.tolist()):
            out[i].append(tok)
        x = nid.unsqueeze(1)  # (B, 1)
        logits = model(x, cache)[:, -1]
    return out
