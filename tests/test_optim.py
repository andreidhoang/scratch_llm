"""Tests for AdamW, gradient clipping, the cosine schedule, and the overfit-one-batch
discipline (the single highest-leverage check that the training wiring is correct)."""

import math

import torch

from reasoning_llm.model import ModelConfig, TransformerLM, cross_entropy
from reasoning_llm.optim import AdamW, cosine_lr, gradient_clipping


def test_adamw_minimizes_quadratic() -> None:
    torch.manual_seed(0)
    x = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    target = torch.tensor([5.0, -3.0])
    opt = AdamW([x], lr=0.1, weight_decay=0.0)
    for _ in range(500):
        opt.zero_grad()
        loss = ((x - target) ** 2).sum()
        loss.backward()
        opt.step()
    torch.testing.assert_close(x.detach(), target, atol=1e-2, rtol=0)


def test_adamw_decoupled_weight_decay_shrinks_param() -> None:
    # With zero gradient, decoupled wd alone should shrink the parameter by (1 - lr*wd)/step.
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = AdamW([x], lr=0.1, weight_decay=0.5)
    x.grad = torch.zeros_like(x)
    opt.step()
    # m=v=0 so the adam term is 0; only wd applies: 1 - lr*wd = 1 - 0.05 = 0.95
    torch.testing.assert_close(x.detach(), torch.tensor([0.95]), atol=1e-6, rtol=0)


def test_gradient_clipping_scales_when_over_and_noop_when_under() -> None:
    p_big = torch.nn.Parameter(torch.zeros(3))
    p_big.grad = torch.tensor([3.0, 4.0, 0.0])  # norm = 5
    norm = gradient_clipping([p_big], max_l2_norm=1.0)
    assert math.isclose(norm.item(), 5.0, rel_tol=1e-5)
    torch.testing.assert_close(p_big.grad.norm(), torch.tensor(1.0), atol=1e-4, rtol=0)

    p_small = torch.nn.Parameter(torch.zeros(2))
    p_small.grad = torch.tensor([0.3, 0.4])  # norm = 0.5 < 1.0 → unchanged
    gradient_clipping([p_small], max_l2_norm=1.0)
    torch.testing.assert_close(p_small.grad, torch.tensor([0.3, 0.4]))


def test_cosine_schedule_phases() -> None:
    max_lr, min_lr, warmup, cosine = 1.0, 0.1, 10, 110
    assert cosine_lr(0, max_lr, min_lr, warmup, cosine) == 0.0
    assert math.isclose(cosine_lr(5, max_lr, min_lr, warmup, cosine), 0.5)  # mid-warmup
    assert math.isclose(cosine_lr(warmup, max_lr, min_lr, warmup, cosine), max_lr)
    # midpoint of the cosine phase → halfway between max and min
    mid = cosine_lr(warmup + (cosine - warmup) // 2, max_lr, min_lr, warmup, cosine)
    assert math.isclose(mid, (max_lr + min_lr) / 2, abs_tol=1e-6)
    assert math.isclose(cosine_lr(cosine, max_lr, min_lr, warmup, cosine), min_lr)
    assert cosine_lr(cosine + 50, max_lr, min_lr, warmup, cosine) == min_lr


def test_overfit_one_batch() -> None:
    """The wiring check: model + AdamW + cross_entropy must memorize a single batch."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, context_length=32)
    model = TransformerLM(cfg)
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    ids = torch.randint(0, cfg.vocab_size, (4, 17))
    inputs, targets = ids[:, :-1], ids[:, 1:]

    initial = cross_entropy(model(inputs), targets).item()
    final = initial
    for _ in range(300):
        opt.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        opt.step()
        final = loss.item()

    assert final < 0.05, f"failed to overfit: final loss {final:.4f} (started {initial:.3f})"
