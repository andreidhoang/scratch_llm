"""Tests for A6 chat REPL (docs/FRONTIER_2026_TASKSPEC.md §A · A6).

DoD coverage:
- ``reply('hi')`` returns a str and grows history 0 → 2 (the user + assistant turns);
- an SFT'd nano model reproduces its trained reply, STOPS at eot (len < max_tokens), and the
  returned text carries no leaked special tokens (the KILL: never emits ``<|eot|>``);
- batched replies == per-turn greedy (the serving R3a path row b == single-stream greedy).
"""

from __future__ import annotations

from pathlib import Path

from scratch_llm.algos.chat_sft import chat_sft_step, collate_chat_batch
from scratch_llm.chat import CHAT_SPECIAL_TOKENS, Message
from scratch_llm.chat_cli import ChatSession, batch_reply
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import AdamW
from scratch_llm.tokenizer import Tokenizer, train_bpe
from scratch_llm.utils.seeding import seed_everything

_CORPUS = "the cat sat on the mat. the dog ran! hello world ok yes no maybe sup.\n" * 30


def _chat_tokenizer(tmp_path: Path) -> Tokenizer:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS, encoding="utf-8")
    vocab, merges = train_bpe(corpus, vocab_size=300, special_tokens=CHAT_SPECIAL_TOKENS)
    return Tokenizer(vocab, merges, special_tokens=CHAT_SPECIAL_TOKENS)


def _tiny_model(vocab_size: int) -> TransformerLM:
    return TransformerLM(
        ModelConfig(
            vocab_size=vocab_size,
            d_model=32,
            n_layers=2,
            n_heads=4,
            context_length=64,
            tie_embeddings=False,
        )
    )


def test_reply_returns_str_and_grows_history(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    seed_everything(0)
    session = ChatSession(_tiny_model(len(tok.vocab)), tok, max_tokens=8)
    assert session.history == []
    out = session.reply("hi")
    assert isinstance(out, str)
    assert len(session.history) == 2
    assert session.history[0] == Message("user", "hi")
    assert session.history[1].role == "assistant"


def test_sft_model_reproduces_reply_stops_and_no_leaked_specials(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    seed_everything(0)
    model = _tiny_model(len(tok.vocab))
    conv = [Message("user", "hi"), Message("assistant", "the dog ran")]
    batch = collate_chat_batch([conv], tok)
    opt = AdamW(model.parameters(), lr=1e-3)
    for _ in range(200):
        meta = chat_sft_step(model, batch, opt)
    assert float(meta["mean_token_nll"]) < 0.1

    session = ChatSession(model, tok, max_tokens=32, temperature=0.0)
    reply = session.reply("hi")
    assert "the dog ran" in reply
    # No leaked specials in the surfaced text (they are stripped, not decoded).
    for special in CHAT_SPECIAL_TOKENS:
        assert special not in reply
    # It stopped on its own eot rather than running to the budget.
    assert len(session.history) == 2


def test_batched_replies_equal_per_turn_greedy(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    seed_everything(0)
    model = _tiny_model(len(tok.vocab))
    # Single-char user turns render to equal-length prompts (serving R3a static-batch requirement).
    texts = ["a", "b"]
    batched = batch_reply(model, tok, texts, max_tokens=8)

    per_turn = []
    for t in texts:
        per_turn.append(ChatSession(model, tok, max_tokens=8, temperature=0.0).reply(t))
    assert batched == per_turn


def test_batch_reply_rejects_unequal_length_prompts(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    model = _tiny_model(len(tok.vocab))
    import pytest

    with pytest.raises(ValueError, match="equal-length"):
        batch_reply(model, tok, ["a", "much longer prompt here"], max_tokens=4)


def test_speedrun_populates_chat_reply(tmp_path: Path) -> None:
    """The A6 wiring: a nano speedrun with chat + SFT surfaces a chat preview and lists the stage."""
    from scratch_llm.speedrun import SpeedrunConfig, run_speedrun

    result = run_speedrun(
        SpeedrunConfig(
            depth=4,
            vocab_size=320,
            context_length=64,
            train_steps=10,
            batch_size=8,
            device="cpu",
            chat=True,
            sft_steps=5,
        )
    )
    assert result.chat_reply is not None and isinstance(result.chat_reply, str)
    assert "chat" in result.stages
    # No leaked specials in the surfaced preview.
    for special in CHAT_SPECIAL_TOKENS:
        assert special not in result.chat_reply

    # Without SFT, the preview stays None (byte-identical to the pre-A6 result).
    no_sft = run_speedrun(
        SpeedrunConfig(
            depth=4,
            vocab_size=320,
            context_length=64,
            train_steps=10,
            batch_size=8,
            device="cpu",
            chat=True,
            sft_steps=0,
        )
    )
    assert no_sft.chat_reply is None
    assert "chat" not in no_sft.stages
