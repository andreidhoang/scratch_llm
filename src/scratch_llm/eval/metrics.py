"""Intrinsic report-card metric — validation **bits per byte** (val_bpb).

The primary language-modeling metric on the close-the-loop report card (nanochat reports `val_bpb`
as the headline pretraining number; it is tokenizer-invariant, unlike per-token loss). bpb is the
total teacher-forced negative log-likelihood of a held-out token stream, in **bits**, divided by the
number of **UTF-8 bytes** that stream decodes to:

    bpb = ( Σ_t −ln p(x_t | x_<t) ) / ln 2 / n_bytes

Because the denominator is bytes (not tokens), a better tokenizer that packs more bytes per token is
not rewarded for free — the metric compares models, not tokenizers.

Correctness invariant (tested): a uniform model (all-equal logits over a V-vocab) scores exactly
log2(V) bits per token; if the stream is one byte per token, that is log2(V) bpb.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BpbResult:
    bits_per_byte: float
    nats_per_token: float
    n_tokens: int  # transitions scored
    n_bytes: int


@torch.no_grad()
def bits_per_byte(
    model: nn.Module,
    token_ids: Sequence[int] | Tensor | np.ndarray,
    num_bytes: int,
    *,
    context_length: int | None = None,
    device: str = "cpu",
) -> BpbResult:
    """Teacher-forced bits-per-byte of ``token_ids`` (a 1-D held-out stream).

    The stream is scored in non-overlapping windows of ``context_length`` (nanochat/GPT-2 chunked
    bpb): every transition ``x_<t → x_t`` for ``t ≥ 1`` is scored exactly once. The first token of
    each window predicts with truncated in-window context — the standard, slightly pessimistic
    chunked approximation (use a stride if you need the tighter estimate). ``num_bytes`` is the byte
    length of the text the stream decodes to (for byte-level BPE, the sum of token byte-lengths).
    """
    if num_bytes <= 0:
        raise ValueError("num_bytes must be positive")
    ids = torch.as_tensor(token_ids).to(dtype=torch.long).flatten()
    n = int(ids.numel())
    if n < 2:
        raise ValueError("need at least 2 tokens to score a transition")
    ctx = context_length
    if ctx is None:
        ctx = getattr(getattr(model, "cfg", None), "context_length", None)
    if ctx is None:
        raise ValueError("context_length is required (model has no .cfg.context_length)")
    ctx = int(ctx)

    if hasattr(model, "eval"):
        model.eval()
    total_nll = 0.0  # nats
    n_tokens = 0
    for start in range(0, n - 1, ctx):
        inp = ids[start : start + ctx]
        tgt = ids[start + 1 : start + 1 + int(inp.numel())]
        m = int(tgt.numel())
        if m == 0:
            break
        logits = model(inp[:m].unsqueeze(0).to(device))[0]  # (m, V)
        logp = torch.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, tgt[:m].to(device).unsqueeze(-1)).squeeze(-1).sum()
        total_nll += float(nll)
        n_tokens += m
    bpb = total_nll / math.log(2) / num_bytes
    return BpbResult(bpb, total_nll / n_tokens, n_tokens, num_bytes)
