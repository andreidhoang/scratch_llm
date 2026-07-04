"""Tests for the end-to-end speedrun spine (F-front, ADR-0018 §5) — the whole loop composes."""

from __future__ import annotations

import math

from scratch_llm.speedrun import SpeedrunConfig, model_config_for_depth, run_speedrun


def test_model_config_for_depth_matches_d20_shape() -> None:
    d20 = model_config_for_depth(20, 65536, 1024)  # the nanochat d20 headline
    assert d20.n_layers == 20 and d20.d_model == 1280 and d20.n_heads == 10
    assert d20.head_dim == 128
    nano = model_config_for_depth(4, 384, 64)
    assert nano.d_model == 256 and nano.n_layers == 4 and nano.n_heads == 2


def test_speedrun_nano_composes_end_to_end() -> None:
    """tokenizer → pretrain → eval → sample runs at nano scale and yields a real report card."""
    cfg = SpeedrunConfig(
        depth=2,
        vocab_size=300,  # 256 bytes + 44 merges
        context_length=48,
        train_steps=40,
        batch_size=8,
        device="cpu",
        seed=0,
    )
    res = run_speedrun(cfg)
    assert res.stages == ["tokenizer", "pretrain", "eval", "sample"]
    assert res.report_card.val_bpb is not None and math.isfinite(res.report_card.val_bpb)
    assert isinstance(res.sample, str)
    assert res.n_params > 0 and res.n_tokens > cfg.context_length
    assert "val_bpb" in res.summary()
