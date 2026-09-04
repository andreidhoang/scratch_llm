"""F2a — MTP (multi-token prediction) training head, depth D=1 (DeepSeek-V3 style).

DoD oracles (git show 07f3de4:docs/archive/FRONTIER_2026_TASKSPEC.md §F2a):
- MTP-head loss at init ≈ log(vocab_size) (uniform-prediction invariant on the aux head).
- ``forward_train``'s main logits are identical to ``forward()``.
- ``mtp_depth`` ∈ {0, 1} under a shared seed gives bit-identical base weights and forward.
- A few optimizer steps reduce the MTP loss; the train() loop runs with mtp_loss_weight.
"""

import math

import numpy as np
import torch

from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy
from scratch_llm.train import TrainConfig, train
from scratch_llm.utils.seeding import seed_everything


def _small_cfg(**overrides: object) -> ModelConfig:
    base: dict[str, object] = dict(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        context_length=64,
        mtp_depth=1,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def test_mtp_head_loss_at_init_is_log_vocab() -> None:
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (8, 32))
    targets = torch.randint(0, cfg.vocab_size, (8, 32))
    _, aux, mtp_logits = model.forward_train(ids, targets)
    assert aux is None  # dense model → no MoE aux
    assert mtp_logits.shape == ids.shape + (cfg.vocab_size,)
    # the MTP head predicts one token further ahead: position i ↔ targets[:, i+1]
    loss = cross_entropy(mtp_logits[:, :-1], targets[:, 1:]).item()
    expected = math.log(cfg.vocab_size)
    assert abs(loss - expected) < 0.3, f"MTP loss {loss:.3f} vs log V {expected:.3f}"


def test_forward_train_main_logits_match_forward() -> None:
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (3, 16))
    targets = torch.randint(0, cfg.vocab_size, (3, 16))
    logits, _, _ = model.forward_train(ids, targets)
    torch.testing.assert_close(logits, model(ids))  # bit-identical: same graph, same order


def test_mtp_depth_zero_and_one_share_base_weights() -> None:
    def build(depth: int) -> TransformerLM:
        torch.manual_seed(1234)
        return TransformerLM(_small_cfg(mtp_depth=depth))

    base, with_mtp = build(0), build(1)
    assert base.mtp_head is None and with_mtp.mtp_head is not None
    # every BASE parameter is bit-identical (the head is built last, RNG-order-neutral)
    base_params = dict(base.named_parameters())
    for name, p in with_mtp.named_parameters():
        if name.startswith("mtp_head."):
            continue
        assert name in base_params
        torch.testing.assert_close(base_params[name], p)
    # and the base forward is bit-identical under the same input
    ids = torch.arange(16).unsqueeze(0) % 512
    with torch.no_grad():
        torch.testing.assert_close(base(ids), with_mtp(ids))


def test_forward_train_requires_mtp_head() -> None:
    model = TransformerLM(_small_cfg(mtp_depth=0))
    ids = torch.randint(0, 512, (1, 8))
    try:
        model.forward_train(ids, ids)
    except ValueError:
        return
    raise AssertionError("expected ValueError on a model without an MTP head")


def test_optimizer_steps_reduce_mtp_loss() -> None:
    torch.manual_seed(0)
    cfg = _small_cfg(vocab_size=64)
    model = TransformerLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # a repeating cycle has learnable structure one AND two tokens ahead
    cycle = torch.arange(32) % cfg.vocab_size
    ids = cycle.repeat(4)[:32].unsqueeze(0)
    targets = torch.cat([ids[:, 1:], ids[:, :1]], dim=1)

    def total_loss() -> torch.Tensor:
        _, _, mtp_logits = model.forward_train(ids, targets)
        return cross_entropy(mtp_logits[:, :-1], targets[:, 1:]) + cross_entropy(
            model(ids), targets
        )

    first = 0.0
    loss = total_loss()
    for _step in range(60):
        loss = total_loss()
        if _step == 0:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first, f"MTP+main loss {loss.item():.3f} vs start {first:.3f}"


def test_train_loop_with_mtp_loss_weight() -> None:
    # Smoke: the train() MTP branch runs and learns on structured data.
    data = np.tile(np.arange(16, dtype=np.int64), 400)
    seed_everything(0)
    cfg = TrainConfig(
        max_steps=60,
        batch_size=8,
        context_length=12,
        max_lr=3e-3,
        warmup_steps=5,
        seed=0,
        mtp_loss_weight=0.3,
    )
    model = TransformerLM(
        ModelConfig(
            vocab_size=32, d_model=32, n_layers=2, n_heads=4, context_length=16, mtp_depth=1
        )
    )
    history = train(cfg, data, model)
    first_loss, last_loss = history[0][1], history[-1][1]
    assert last_loss < first_loss, f"loss did not fall: {first_loss:.3f} → {last_loss:.3f}"
