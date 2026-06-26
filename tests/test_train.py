"""Tests for data loading, checkpointing, and the training loop."""

from pathlib import Path

import numpy as np
import torch

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import AdamW
from scratch_llm.train import (
    TrainConfig,
    get_batch,
    load_checkpoint,
    save_checkpoint,
    train,
)
from scratch_llm.utils.seeding import seed_everything


def test_get_batch_shapes_and_next_token_alignment() -> None:
    np.random.seed(0)
    data = np.arange(100, dtype=np.int64)  # targets must equal inputs + 1
    inputs, targets = get_batch(data, batch_size=8, context_length=12)
    assert inputs.shape == (8, 12) and targets.shape == (8, 12)
    torch.testing.assert_close(targets, inputs + 1)


def test_get_batch_rejects_too_short_corpus() -> None:
    data = np.arange(5, dtype=np.int64)
    try:
        get_batch(data, batch_size=2, context_length=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a corpus shorter than the context window")


def _tiny_model() -> TransformerLM:
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, context_length=16)
    return TransformerLM(cfg)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    opt = AdamW(model.parameters(), lr=1e-3)
    # take one step so optimizer state is non-trivial
    ids = torch.randint(0, 32, (2, 8))
    loss = model(ids).sum()
    loss.backward()
    opt.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, step=42, out=path)

    fresh = _tiny_model()
    fresh_opt = AdamW(fresh.parameters(), lr=1e-3)
    restored_step = load_checkpoint(path, fresh, fresh_opt)
    assert restored_step == 42
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh.named_parameters(), strict=True):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)


def _structured_corpus() -> np.ndarray:
    # A repeating cycle has learnable structure → loss should fall well below log(V).
    return np.tile(np.arange(16, dtype=np.int64), 400)


def test_train_reduces_loss_on_structured_data() -> None:
    data = _structured_corpus()
    cfg = TrainConfig(
        max_steps=300, batch_size=16, context_length=12, max_lr=3e-3, warmup_steps=20, seed=0
    )
    model = TransformerLM(
        ModelConfig(vocab_size=32, d_model=64, n_layers=2, n_heads=4, context_length=16)
    )
    history = train(cfg, data, model)
    first_loss = history[0][1]
    last_loss = history[-1][1]
    assert last_loss < 1.0, f"loss did not fall enough: {last_loss:.3f} (started {first_loss:.3f})"


def test_training_is_reproducible_with_same_seed() -> None:
    data = _structured_corpus()

    def run() -> list[tuple[int, float]]:
        # Seed before constructing the model: weight init must be identical across runs,
        # not just the data sampling (which train() reseeds via cfg.seed).
        seed_everything(7)
        cfg = TrainConfig(max_steps=30, batch_size=8, context_length=12, max_lr=3e-3, seed=7)
        model = TransformerLM(
            ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, context_length=16)
        )
        return train(cfg, data, model)

    assert run() == run()
