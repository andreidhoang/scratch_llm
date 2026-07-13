"""Chat template + special tokens — the seam between conversations and the token stream (A3).

A conversation renders to ids as ``<|bos|> <|user|> …content… <|eot|> <|assistant|>
…content… <|eot|> …``. Each special is exactly ONE vocab id — trained into the BPE by
threading :data:`CHAT_SPECIAL_TOKENS` through ``train_bpe`` AND the ``Tokenizer``
constructor (``SpeedrunConfig.chat=True``) — so turn structure is tokenizer-atomic: a role
boundary can never be split by the merge table or synthesized by ordinary text merges.

Key invariant (tested in tests/test_chat.py): the mask from :func:`render_conversation` is
True on assistant CONTENT tokens plus the assistant turn's closing ``<|eot|>``, and False
everywhere else (bos, both role markers, all user tokens, the user's eot) — downstream SFT
(A5) learns to *speak and stop*, never to imitate the user or emit scaffolding.
:func:`render_for_completion` ends with the ``<|assistant|>`` id and no trailing eot, so
generation starts exactly at the assistant's turn (A6 stops on the eot id).

Interview question this answers: "why must chat specials be single tokens, and what breaks
if the SFT mask includes the role marker or drops the eot?" — multi-token specials make
turn boundaries spoofable from plain text and stop-detection ambiguous; masking the role
marker teaches the model to emit its own header; dropping the eot means the model never
learns to terminate its turn (exactly A6's kill criterion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scratch_llm.tokenizer import Tokenizer

BOS = "<|bos|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
EOT = "<|eot|>"

# Order matters only for readability; ids come from the tokenizer, never from this list.
# Tool-boundary specials are A4's problem (taskspec §D note), not A3's.
CHAT_SPECIAL_TOKENS: list[str] = [BOS, USER, ASSISTANT, EOT]

Role = Literal["user", "assistant"]
_ROLE_TOKEN: dict[str, str] = {"user": USER, "assistant": ASSISTANT}


@dataclass(frozen=True)
class Message:
    """One chat turn. ``role`` is the speaker; ``content`` is the raw text of the turn."""

    role: Role
    content: str


def _special_id(tokenizer: Tokenizer, special: str) -> int:
    """Resolve one special to its id via the PUBLIC encode path — the kill-criterion probe.

    ``encode`` maps a threaded special to exactly one id; anything else means the tokenizer
    was trained/constructed without it and the special would shatter into byte merges.
    """
    ids = tokenizer.encode(special)
    if len(ids) != 1:
        raise ValueError(
            f"special token {special!r} encodes to {len(ids)} ids, expected exactly 1 — "
            f"this tokenizer was not trained with CHAT_SPECIAL_TOKENS (thread them through "
            f"both train_bpe and the Tokenizer constructor, e.g. SpeedrunConfig.chat=True)."
        )
    return ids[0]


def render_conversation(
    messages: list[Message] | tuple[Message, ...],
    tokenizer: Tokenizer,
) -> tuple[list[int], list[bool]]:
    """Render a conversation to (ids, mask): ``<|bos|>`` then per turn ``<role> content <|eot|>``.

    The same-length bool mask is True ONLY on assistant content tokens and the assistant
    turn's closing eot — the supervised positions for A5 SFT. Role markers themselves are
    never supervised (the template supplies them; the model must not learn to emit its own
    header), while the assistant's eot IS (emitting the terminator is learned behavior).

    Content is encoded via the tokenizer's normal path: a special-token string appearing
    literally inside ``content`` tokenizes as that special. Upstream data hygiene (the A4
    adapters) must sanitize untrusted text; this renderer does not rewrite content.
    """
    bos_id = _special_id(tokenizer, BOS)
    eot_id = _special_id(tokenizer, EOT)
    role_ids = {role: _special_id(tokenizer, tok) for role, tok in _ROLE_TOKEN.items()}

    ids: list[int] = [bos_id]
    mask: list[bool] = [False]
    for msg in messages:
        role_id = role_ids.get(msg.role)
        if role_id is None:
            raise ValueError(f"unknown role {msg.role!r}: expected 'user' or 'assistant'")
        is_assistant = msg.role == "assistant"

        ids.append(role_id)
        mask.append(False)  # the role marker is template scaffolding, never a target
        content_ids = tokenizer.encode(msg.content)
        ids.extend(content_ids)
        mask.extend([is_assistant] * len(content_ids))
        ids.append(eot_id)
        mask.append(is_assistant)  # the model must LEARN to stop its own turn
    return ids, mask


def render_for_completion(
    messages: list[Message] | tuple[Message, ...],
    tokenizer: Tokenizer,
) -> list[int]:
    """Render for generation: the full conversation, then the ``<|assistant|>`` id.

    No trailing eot after the marker — decoding starts exactly at the assistant's first
    content token, and A6 stops when the model emits ``<|eot|>`` itself.
    """
    ids, _ = render_conversation(messages, tokenizer)
    ids.append(_special_id(tokenizer, ASSISTANT))
    return ids
