"""Tests for the chat template + specials (F-front A3, git show 07f3de4:docs/archive/FRONTIER_2026_TASKSPEC.md §A).

DoD coverage, one test per bullet:
- each special encodes to exactly 1 id (the KILL test: >1 means specials weren't threaded
  into BOTH ``train_bpe`` and the ``Tokenizer`` constructor);
- a turn round-trips ``<|user|>…<|eot|><|assistant|>…<|eot|>`` in order (decode the ids back);
- the mask covers ONLY assistant content tokens + the assistant turn's closing eot
  (hand-derived on a tiny two-turn conversation);
- the completion prompt ends with the ``<|assistant|>`` id, no trailing eot;
- ValueError (naming the missing special) on a tokenizer trained WITHOUT the specials;
- ``SpeedrunConfig.chat=False`` leaves the tokenizer stage byte-identical (no specials in
  the vocab), ``chat=True`` trains them in as single ids inside the same vocab budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from scratch_llm.chat import (
    CHAT_SPECIAL_TOKENS,
    Message,
    Role,
    render_conversation,
    render_for_completion,
)
from scratch_llm.speedrun import SpeedrunConfig, stage_tokenizer
from scratch_llm.tokenizer import Tokenizer, train_bpe

_CORPUS = "the cat sat on the mat. the dog ran! hello world, ok?\n" * 20


def _chat_tokenizer(tmp_path: Path) -> Tokenizer:
    """A small BPE with CHAT_SPECIAL_TOKENS threaded through train_bpe AND the constructor."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS, encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=290, special_tokens=CHAT_SPECIAL_TOKENS)
    return Tokenizer(vocab, merges, special_tokens=CHAT_SPECIAL_TOKENS)


def _special_ids(tok: Tokenizer) -> tuple[int, int, int, int]:
    bos, user, assistant, eot = (tok.encode(s)[0] for s in CHAT_SPECIAL_TOKENS)
    return bos, user, assistant, eot


# -- DoD: single-id specials (the KILL test) ----------------------------------------------------


def test_each_special_encodes_to_exactly_one_id(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    for special in CHAT_SPECIAL_TOKENS:
        ids = tok.encode(special)
        assert len(ids) == 1, f"{special!r} split into {len(ids)} ids — kill criterion hit"


def test_specials_are_four_distinct_ids(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    ids = {tok.encode(s)[0] for s in CHAT_SPECIAL_TOKENS}
    assert len(ids) == len(CHAT_SPECIAL_TOKENS) == 4


# -- DoD: round-trip order + the exact hand-derived mask ----------------------------------------


def test_render_conversation_round_trips_turn_layout(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    msgs = [Message("user", "the cat sat"), Message("assistant", "on the mat.")]
    ids, mask = render_conversation(msgs, tok)
    assert len(ids) == len(mask)
    assert tok.decode(ids) == ("<|bos|><|user|>the cat sat<|eot|><|assistant|>on the mat.<|eot|>")


def test_mask_hand_derived_on_two_turn_conversation(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    bos, user, assistant, eot = _special_ids(tok)
    u = tok.encode("the cat sat")
    a = tok.encode("on the mat.")

    ids, mask = render_conversation(
        [Message("user", "the cat sat"), Message("assistant", "on the mat.")], tok
    )
    # Exact id layout: bos · user-marker · user content · eot · assistant-marker · content · eot.
    assert ids == [bos, user, *u, eot, assistant, *a, eot]
    # Mask True ONLY on assistant content + its closing eot; False on bos, both role
    # markers, all user tokens, and the user's eot.
    assert mask == [False, False, *[False] * len(u), False, False, *[True] * len(a), True]


def test_mask_multi_turn_covers_only_assistant_spans(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    bos, user, assistant, eot = _special_ids(tok)
    turns = [
        Message("user", "hello world"),
        Message("assistant", "ok?"),
        Message("user", "the dog ran!"),
        Message("assistant", "the cat sat on the mat."),
    ]
    ids, mask = render_conversation(turns, tok)

    expected_ids: list[int] = [bos]
    expected_mask: list[bool] = [False]
    for msg in turns:
        content = tok.encode(msg.content)
        is_asst = msg.role == "assistant"
        expected_ids += [assistant if is_asst else user, *content, eot]
        expected_mask += [False, *[is_asst] * len(content), is_asst]
    assert ids == expected_ids
    assert mask == expected_mask
    # Supervised-token count = assistant content tokens + one eot per assistant turn.
    n_asst = sum(len(tok.encode(m.content)) + 1 for m in turns if m.role == "assistant")
    assert sum(mask) == n_asst


# -- DoD: completion prompt ends with the <|assistant|> id, no trailing eot ---------------------


def test_render_for_completion_ends_with_assistant_id_no_trailing_eot(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    _, _, assistant, eot = _special_ids(tok)
    msgs = [Message("user", "hello world")]

    ids = render_for_completion(msgs, tok)
    conv_ids, _ = render_conversation(msgs, tok)
    assert ids == conv_ids + [assistant]
    assert ids[-1] == assistant and ids[-1] != eot


def test_render_for_completion_multi_turn_prefix(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    _, _, assistant, _ = _special_ids(tok)
    msgs = [Message("user", "hello"), Message("assistant", "ok?"), Message("user", "world")]
    ids = render_for_completion(msgs, tok)
    conv_ids, _ = render_conversation(msgs, tok)
    assert ids == conv_ids + [assistant]


# -- DoD: ValueError on a tokenizer trained without the specials --------------------------------


def test_missing_specials_raise_value_error_naming_the_special(tmp_path: Path) -> None:
    corpus = tmp_path / "plain.txt"
    corpus.write_text(_CORPUS, encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=280)  # NO chat specials
    tok = Tokenizer(vocab, merges)

    with pytest.raises(ValueError) as excinfo:
        render_conversation([Message("user", "hi")], tok)
    assert any(s in str(excinfo.value) for s in CHAT_SPECIAL_TOKENS)

    with pytest.raises(ValueError):
        render_for_completion([Message("user", "hi")], tok)


def test_unknown_role_raises_value_error(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    bad = Message(cast(Role, "system"), "x")
    with pytest.raises(ValueError):
        render_conversation([bad], tok)


# -- Config: SpeedrunConfig.chat wires specials through the tokenizer stage ---------------------


def _nano_cfg(**overrides: object) -> SpeedrunConfig:
    base: dict[str, object] = {"depth": 2, "vocab_size": 300, "context_length": 32}
    base.update(overrides)
    return SpeedrunConfig(**base)  # type: ignore[arg-type]


def test_speedrun_chat_defaults_false_and_stage_is_unchanged() -> None:
    cfg = _nano_cfg()
    assert cfg.chat is False
    tokenizer, tokens, prompt_ids, name = stage_tokenizer(cfg)
    assert name == "tokenizer"
    assert tokenizer.special_tokens == []  # no specials leak into the default path
    assert len(tokenizer.vocab) == cfg.vocab_size
    # Without threading, the chat specials shatter into many ids — the pre-A3 behavior.
    assert all(len(tokenizer.encode(s)) > 1 for s in CHAT_SPECIAL_TOKENS)
    assert tokens.size > 0 and len(prompt_ids) > 0


def test_speedrun_chat_true_trains_single_id_specials() -> None:
    cfg = _nano_cfg(chat=True)
    tokenizer, _, _, _ = stage_tokenizer(cfg)
    for special in CHAT_SPECIAL_TOKENS:
        assert len(tokenizer.encode(special)) == 1
    # Specials live INSIDE the vocab budget (train_bpe path), not appended past it by the
    # constructor fallback — the model's vocab axis stays cfg.vocab_size.
    assert len(tokenizer.vocab) == cfg.vocab_size
    assert max(tokenizer.encode(s)[0] for s in CHAT_SPECIAL_TOKENS) < cfg.vocab_size

    ids, mask = render_conversation([Message("user", "hi"), Message("assistant", "yo")], tokenizer)
    assert len(ids) == len(mask) and sum(mask) >= 2
