"""Optimizer, gradient clipping, and LR schedule — the training-time machinery we own.

L1 substrate (A1). AdamW's state (the two moment buffers) is 2× the parameter count in
memory — the dominant term in the training memory budget, and the thing we checkpoint and
restore to resume an RL run. Owning the update means we can reason about it cold.

Correctness invariants (tested in tests/test_optim.py):
- **Overfit one batch:** model + AdamW + cross_entropy must drive the loss on a single
  fixed batch to ≈0. If it can't, the optimizer/data/loss *wiring* is broken — not the data.
- **AdamW** minimizes a simple quadratic; **clipping** rescales only when the global ℓ₂ norm
  exceeds the threshold; the **cosine schedule** is linear in warmup, cosine in the middle,
  flat at ``min_lr`` after.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn


class AdamW(torch.optim.Optimizer):
    """AdamW with **decoupled** weight decay (Loshchilov & Hutter 2019).

    Update per step t:
        m ← β₁ m + (1−β₁) g
        v ← β₂ v + (1−β₂) g²
        α_t ← lr · √(1−β₂ᵗ) / (1−β₁ᵗ)          (bias correction folded into the step size)
        θ ← θ − α_t · m / (√v + ε)
        θ ← θ − lr · λ · θ                       (decay decoupled from the gradient)

    LM default β₂=0.95 (vs Adam's 0.999): reasoning-LM gradients are noisier, so the second
    moment should adapt faster.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss


def gradient_clipping(
    parameters: Iterable[nn.Parameter], max_l2_norm: float, eps: float = 1e-6
) -> Tensor:
    """Clip gradients in place by their **global** ℓ₂ norm (across all parameters), and
    return that pre-clip norm (useful to log). Scales only when the norm exceeds the cap."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))  # type: ignore[arg-type]
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
    return total_norm


def cosine_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    """Cosine-with-linear-warmup LR. Three phases:
    - ``step < warmup_steps``: linear ramp 0 → max_lr.
    - ``warmup_steps ≤ step ≤ cosine_steps``: cosine anneal max_lr → min_lr.
    - ``step > cosine_steps``: flat at min_lr.
    """
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step <= cosine_steps:
        progress = (step - warmup_steps) / max(1, cosine_steps - warmup_steps)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
    return min_lr
