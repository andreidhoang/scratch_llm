"""Tests for the end-to-end speedrun spine (F-front, ADR-0018 §5) — the whole loop composes."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

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


def test_speedrun_checkpoint_every_writes_intra_stage_snapshot(tmp_path: Path) -> None:
    """A8/d20 wiring: --checkpoint-every threads train()'s optimizer-state checkpoint path
    through stage_pretrain — pretrain_ckpt.pt carries moments; pretrain.pt stays the
    optimizer-free stage-boundary artifact."""
    cfg = SpeedrunConfig(
        depth=2,
        vocab_size=300,
        context_length=48,
        train_steps=30,
        batch_size=8,
        device="cpu",
        seed=0,
        work_dir=str(tmp_path),
        checkpoint_every=20,
    )
    res = run_speedrun(cfg)
    assert "pretrain" in res.stages

    intra = torch.load(tmp_path / "pretrain_ckpt.pt", weights_only=False)
    assert intra["step"] == 20  # last multiple of checkpoint_every ≤ train_steps
    assert intra["optim"] is not None  # optimizer state — the resume file

    boundary = torch.load(tmp_path / "pretrain.pt", weights_only=False)
    assert boundary["step"] == 30
    assert boundary["optim"] is None  # stage-boundary policy unchanged


def test_speedrun_checkpoint_every_requires_work_dir() -> None:
    cfg = SpeedrunConfig(
        depth=2,
        vocab_size=300,
        context_length=48,
        train_steps=4,
        batch_size=8,
        device="cpu",
        checkpoint_every=2,
    )
    with pytest.raises(ValueError, match="checkpoint_every>0 needs work_dir"):
        run_speedrun(cfg)
