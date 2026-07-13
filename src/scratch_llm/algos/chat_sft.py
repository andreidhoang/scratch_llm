"""A5 chat-SFT — assistant-masked supervised fine-tuning over the A3 chat template.

The one new idea beyond ``algos/sft.py`` is *where the mask comes from*. Plain SFT
(:func:`~scratch_llm.algos.sft.tokenize_prompt_and_output`) splits a row into a prompt and a
response and masks the prompt; chat-SFT instead renders a whole multi-turn conversation with
:func:`~scratch_llm.chat.render_conversation`, whose mask is already True on **every assistant
turn's content + that turn's closing ``<|eot|>``** and False on the bos, both role markers, and
all user tokens. So the model learns to *speak and to stop*, across arbitrarily many turns, never
to imitate the user or emit its own role header.

Everything downstream of the mask is reused verbatim from ``algos/sft.py`` (no edits to it):
:func:`~scratch_llm.algos.sft.get_response_log_probs` scores the labels and
:func:`~scratch_llm.algos.sft.sft_microbatch_train_step` computes the masked NLL and calls
``backward()``. The single load-bearing invariant is the **shift-after-pad** alignment, identical
to plain SFT: render → pad the token sequence → ``input_ids = padded[:, :-1]``,
``labels = padded[:, 1:]``, and the label-coordinate ``response_mask`` is the token-coordinate
render mask shifted left by one (``mask[:, 1:]``) — because label index ``j`` scores the token at
full-sequence position ``j+1``. Get this off by one and the loss silently trains on the wrong
tokens (the overfit-one-batch test in ``tests/test_chat_sft.py`` is the tripwire).

Interview question this answers: "how does SFT on a multi-turn chat template differ from SFT on
(prompt, response) pairs — which tokens are supervised, and why must the assistant's eot be inside
the mask but the ``<|assistant|>`` marker outside it?"
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from scratch_llm.algos.sft import get_response_log_probs, sft_microbatch_train_step
from scratch_llm.chat import Message, render_conversation
from scratch_llm.optim import CombinedOptimizer, gradient_clipping
from scratch_llm.tokenizer import Tokenizer

Conversation = list[Message]
_OptimizerLike = torch.optim.Optimizer | CombinedOptimizer


def collate_chat_batch(
    conversations: list[Conversation],
    tokenizer: Tokenizer,
    *,
    pad_token_id: int = 0,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Render, right-pad, and shift a batch of conversations into SFT tensors.

    Returns ``input_ids``/``labels``/``response_mask`` of shape ``(B, max_len − 1)``. The mask is
    in *label* coordinates and is True exactly where the label token is an assistant content token
    or an assistant turn's closing eot (from :func:`~scratch_llm.chat.render_conversation`), so
    padding is never scored. ``pad_token_id`` fills the ragged tail; those positions are masked off
    and their id is inert (default 0, the repo BPE has no reserved pad).
    """
    if not conversations:
        raise ValueError("empty batch")
    rendered = [render_conversation(conv, tokenizer) for conv in conversations]
    max_len = max(len(ids) for ids, _ in rendered)
    if max_len < 2:
        raise ValueError("every conversation must render to ≥2 tokens (bos + at least one turn)")

    batch = len(rendered)
    padded = np.full((batch, max_len), pad_token_id, dtype=np.int64)
    mask_full = np.zeros((batch, max_len), dtype=bool)
    for i, (ids, mask) in enumerate(rendered):
        padded[i, : len(ids)] = ids
        mask_full[i, : len(mask)] = mask

    ids_t = torch.from_numpy(padded).to(device)
    mask_t = torch.from_numpy(mask_full).to(device)
    # Shift AFTER padding (the sft.py-pinned alignment): label j scores full position j+1.
    return {
        "input_ids": ids_t[:, :-1],
        "labels": ids_t[:, 1:],
        "response_mask": mask_t[:, 1:],
    }


def chat_sft_step(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    optimizer: _OptimizerLike,
    *,
    normalize_constant: float = 1.0,
    grad_clip: float | None = 1.0,
) -> dict[str, Tensor]:
    """One gradient step on a pre-collated chat batch; returns ``sft_microbatch_train_step`` meta.

    Composes the ``algos/sft.py`` primitives unchanged: score labels → masked-NLL microbatch step
    (which calls ``backward()``) → optional global-ℓ₂ grad clip → ``optimizer.step()``. The caller
    reads ``meta["mean_token_nll"]`` — the ``log V``-comparable overfit/underfit signal.
    """
    optimizer.zero_grad()
    out = get_response_log_probs(model, batch["input_ids"], batch["labels"])
    _, meta = sft_microbatch_train_step(
        out["log_probs"],
        batch["response_mask"],
        gradient_accumulation_steps=1,
        normalize_constant=normalize_constant,
    )
    if grad_clip is not None:
        gradient_clipping(model.parameters(), grad_clip)
    optimizer.step()
    return meta


def chat_sft_epoch(
    model: torch.nn.Module,
    conversations: list[Conversation],
    tokenizer: Tokenizer,
    optimizer: _OptimizerLike,
    *,
    batch_size: int,
    pad_token_id: int = 0,
    normalize_constant: float = 1.0,
    grad_clip: float | None = 1.0,
    device: str = "cpu",
) -> list[dict[str, Tensor]]:
    """One pass over ``conversations`` in contiguous batches; returns per-batch metadata.

    Deterministic order (no shuffle) so an epoch is reproducible under a fixed seed — the caller
    (speedrun ``stage_sft``) loops epochs until ``sft_steps`` gradient steps are taken.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1, got {batch_size}")
    model.train()
    metas: list[dict[str, Tensor]] = []
    for start in range(0, len(conversations), batch_size):
        batch = collate_chat_batch(
            conversations[start : start + batch_size],
            tokenizer,
            pad_token_id=pad_token_id,
            device=device,
        )
        metas.append(
            chat_sft_step(
                model,
                batch,
                optimizer,
                normalize_constant=normalize_constant,
                grad_clip=grad_clip,
            )
        )
    return metas


__all__ = ["chat_sft_epoch", "chat_sft_step", "collate_chat_batch"]
