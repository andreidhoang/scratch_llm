"""Tests for A5 chat-SFT (docs/FRONTIER_2026_TASKSPEC.md §A · A5).

DoD coverage, one test per bullet:
- ``response_mask`` is True EXACTLY at assistant content + the assistant turn's closing eot,
  in *label* coordinates (hand-derived on a tiny two-turn conversation — the off-by-one gate);
- padding never contributes (a short row's pad positions are all masked False);
- masked loss-at-init ≈ log V (discipline #1, through the chat mask);
- ≤200 steps overfit one chat batch to mean-token NLL < 0.1 (discipline #2);
- after overfit, greedy ``generate(render_for_completion([user]), stop=eot)`` decodes the
  trained assistant string AND stops at eot (the A6 handoff);
- ``chat_sft_epoch`` reduces loss over batches without editing ``algos/sft.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from scratch_llm.algos.chat_sft import (
    chat_sft_epoch,
    chat_sft_step,
    collate_chat_batch,
)
from scratch_llm.chat import EOT, Message, render_conversation, render_for_completion
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import AdamW
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.tokenizer import Tokenizer, train_bpe
from scratch_llm.utils.seeding import seed_everything

_CORPUS = "the cat sat on the mat. the dog ran! hello world, ok? yes no maybe.\n" * 30


def _chat_tokenizer(tmp_path: Path) -> Tokenizer:
    from scratch_llm.chat import CHAT_SPECIAL_TOKENS

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


def _conv(user: str, assistant: str) -> list[Message]:
    return [Message("user", user), Message("assistant", assistant)]


# ---------------------------------------------------------------------------------------------
# The mask — the whole point of A5. If this is off by one every downstream number is wrong.
# ---------------------------------------------------------------------------------------------


def test_collate_mask_is_label_coord_assistant_plus_eot(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    conv = _conv("hi", "yo")
    batch = collate_chat_batch([conv], tok)

    ids, mask_full = render_conversation(conv, tok)  # mask_full in TOKEN coords
    # collate shifts after (no) padding: label index j scores token at full position j+1.
    expected_labels = torch.tensor(ids[1:])
    expected_mask = torch.tensor(mask_full[1:])
    torch.testing.assert_close(batch["labels"][0], expected_labels)
    torch.testing.assert_close(batch["input_ids"][0], torch.tensor(ids[:-1]))
    assert batch["response_mask"][0].tolist() == expected_mask.tolist()

    # Every masked label is an assistant-side token (content or the closing eot), never the
    # <|assistant|> marker, never a user/bos token.
    eot_id = tok.encode(EOT)[0]
    labels = batch["labels"][0]
    resp = batch["response_mask"][0]
    assistant_ids = set(tok.encode("yo")) | {eot_id}
    for j in range(len(labels)):
        if resp[j]:
            assert int(labels[j]) in assistant_ids


def test_collate_padding_never_contributes(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    short, long = _conv("hi", "yo"), _conv("hello there", "general kenobi")
    batch = collate_chat_batch([short, long], tok)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["response_mask"].shape
    # The short row is padded on the right; those trailing label positions must be masked off.
    short_len = len(render_conversation(short, tok)[0])
    pad_region = batch["response_mask"][0][short_len - 1 :]
    assert not pad_region.any()


# ---------------------------------------------------------------------------------------------
# Loss-at-init + overfit — the two disciplines, through the chat mask.
# ---------------------------------------------------------------------------------------------


def test_masked_loss_at_init_is_log_vocab(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    seed_everything(0)
    model = _tiny_model(len(tok.vocab))
    batch = collate_chat_batch([_conv("hi", "yo"), _conv("hello", "world")], tok)
    from scratch_llm.algos.sft import get_response_log_probs, masked_mean

    with torch.no_grad():
        out = get_response_log_probs(model, batch["input_ids"], batch["labels"])
        nll = masked_mean(-out["log_probs"], batch["response_mask"])
    import math

    assert abs(float(nll) - math.log(len(tok.vocab))) < 0.3


def test_overfit_one_chat_batch_and_speak(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    seed_everything(0)
    model = _tiny_model(len(tok.vocab))
    conv = _conv("hi", "the dog ran")
    batch = collate_chat_batch([conv], tok)
    opt = AdamW(model.parameters(), lr=1e-3)

    last_nll = float("inf")
    for _ in range(200):
        meta = chat_sft_step(model, batch, opt)
        last_nll = float(meta["mean_token_nll"])
    assert last_nll < 0.1, f"overfit stalled at NLL {last_nll:.3f} — suspect a mask off-by-one"

    # The A6 handoff: greedy-continue the user turn, stop at eot, decode the trained answer.
    eot_id = tok.encode(EOT)[0]
    prompt = render_for_completion([Message("user", "hi")], tok)
    gen = generate(
        model, prompt, SamplingParams(temperature=0.0, max_tokens=32, stop_ids=(eot_id,))
    )
    assert gen and gen[-1] == eot_id, "model never emitted <|eot|> — it won't stop its turn"
    text = tok.decode([g for g in gen if g != eot_id])
    assert "the dog ran" in text


def test_speedrun_sft_stage_composes(tmp_path: Path) -> None:
    """The A5 wiring: a nano speedrun with chat=True + sft_steps>0 runs the SFT stage and lists it.

    Also asserts the guard: sft_steps>0 without chat=True fails loud (the specials must be trained).
    """
    from scratch_llm.speedrun import SpeedrunConfig, run_speedrun

    cfg = SpeedrunConfig(
        depth=4,
        vocab_size=320,
        context_length=64,
        train_steps=10,
        batch_size=8,
        device="cpu",
        chat=True,
        sft_steps=5,
    )
    result = run_speedrun(cfg)
    assert "sft" in result.stages
    assert result.stages.index("sft") > result.stages.index("pretrain")

    import pytest

    with pytest.raises(ValueError, match="chat=True"):
        run_speedrun(
            SpeedrunConfig(
                depth=4,
                vocab_size=320,
                context_length=64,
                train_steps=10,
                batch_size=8,
                device="cpu",
                chat=False,
                sft_steps=5,
            )
        )


def test_chat_sft_epoch_reduces_loss(tmp_path: Path) -> None:
    tok = _chat_tokenizer(tmp_path)
    seed_everything(0)
    model = _tiny_model(len(tok.vocab))
    convs = [_conv("hi", "yo"), _conv("hello", "world"), _conv("hey", "there"), _conv("yo", "sup")]
    opt = AdamW(model.parameters(), lr=1e-3)

    first = chat_sft_epoch(model, convs, tok, opt, batch_size=2)
    for _ in range(30):
        last = chat_sft_epoch(model, convs, tok, opt, batch_size=2)
    assert float(last[-1]["mean_token_nll"]) < float(first[0]["mean_token_nll"])
    assert len(first) == 2  # 4 convs / batch_size 2
