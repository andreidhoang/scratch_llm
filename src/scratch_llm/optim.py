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
import time
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


# ---------------------------------------------------------------------------------------------
# Muon (F1 — close-the-loop frontier ablation, ADR-0018 / docs/FRONTIER_2026_ABLATIONS.md)
# ---------------------------------------------------------------------------------------------


def _zeropower_via_newtonschulz5(g: Tensor, steps: int = 5) -> Tensor:
    """Orthogonalize a 2-D matrix ``g`` via a 5-step Newton–Schulz iteration (Keller Jordan's
    quintic, coeffs (3.4445, −4.7750, 2.0315)), run in bfloat16.

    Given ``g = U Σ Vᵀ`` the iteration drives every singular value toward ~1, so the returned
    matrix ≈ ``U Vᵀ`` (the semi-orthogonal factor) — the direction of ``g`` with a *uniform*
    spectrum. The quintic is deliberately tuned to be fast, not exact: after 5 steps the singular
    values sit in roughly ``[0.7, 1.3]`` rather than exactly 1 (F1 DoD; KILL if any σ ∉ [0.5,1.5]).

    The matrix is normalized by its Frobenius norm first (an upper bound on the spectral norm, so
    all σ ≤ 1 going in), and transposed to the wide orientation so the ``XXᵀ`` products are as
    small as possible. Bit-width note: bf16 is numerically sufficient — the iteration is
    self-correcting toward the fixed point (Jordan's writeup).
    """
    if g.ndim != 2:
        raise ValueError(
            f"Newton–Schulz orthogonalization needs a 2-D matrix, got shape {tuple(g.shape)}"
        )
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:  # work in the wide orientation (fewer FLOPs in X @ Xᵀ)
        x = x.T
    x = x / (x.norm() + 1e-7)  # Frobenius ≥ spectral ⇒ all σ ≤ 1 before iterating
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)  # the quintic: X ← a·X + (b·A + c·A²)·X, A = X Xᵀ
        x = a * x + bb @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton–schulz (Keller Jordan 2024) with Moonlight RMS-matching.

    For each **2-D** weight matrix: take the (Nesterov) SGD-momentum gradient, orthogonalize it via
    :func:`_zeropower_via_newtonschulz5` so the applied update has a near-uniform spectrum, then
    scale it to reuse AdamW's learning-rate band. Muon is used **only** on hidden 2-D block matrices
    (attention/MLP projections); embeddings, the LM head, and every 1-D parameter (RMSNorm gains,
    biases, scalars) stay on :class:`AdamW` — split with :func:`split_muon_adamw_params`.

    **Moonlight RMS-matching (arXiv 2502.16982, Lemma 1 — corrected).** A full-rank orthogonalized
    update on an ``[A, B]`` matrix has per-element RMS ``1/√max(A, B)`` (‖O‖_F² = min(A,B) spread
    over A·B entries). Scaling by ``rms_scale·√max(A, B)`` (default 0.2) lands its RMS at ``0.2`` —
    squarely in AdamW's usual 0.2–0.4 update band — so **one LR/WD schedule serves both optimizers**
    and no separate Muon LR sweep is needed. (This corrects the drafting error ``1/max(A,B)`` flagged
    by the F1 research verifier; the ``0.2·√max`` scale and ``wd=0.1`` were already right.)

    Update (decoupled weight decay, matching AdamW's convention):
        buf ← momentum·buf + g ;   g̃ ← g + momentum·buf   (Nesterov)
        O   ← NewtonSchulz₅(g̃)
        θ   ← (1 − lr·wd)·θ − lr·(rms_scale·√max(A,B))·O

    Interview question this answers: "derive the Muon update; why orthogonalize the momentum, and
    why does RMS-matching let you reuse AdamW's learning rate?"
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, object]],
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.1,
        rms_scale: float = 0.2,
        profile_ns: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be ≥ 1, got {ns_steps}")
        if weight_decay < 0 or rms_scale < 0:
            raise ValueError("weight_decay and rms_scale must be ≥ 0")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            rms_scale=rms_scale,
        )
        super().__init__(params, defaults)
        # F1-run NS wall-time instrument (off by default). Instance attributes, deliberately NOT
        # in ``defaults``/param_groups: an instrument is not a hyperparameter, and keeping it out
        # of ``state_dict`` leaves checkpoint round-trips byte-identical. When on, each
        # Newton–Schulz call is timed with a CUDA sync fence — that perturbs the run, which is
        # fine: the mode exists only to measure the NS-overhead falsifier (<1% predicted, >3% kill).
        self.profile_ns = profile_ns
        self.ns_seconds: float = 0.0
        self.ns_calls: int = 0

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            rms_scale = group["rms_scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only optimizes 2-D matrices; got shape {tuple(p.shape)}. Route "
                        "embeddings/head/1-D params to AdamW (see split_muon_adamw_params)."
                    )
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                g_eff = grad.add(buf, alpha=momentum) if nesterov else buf
                if self.profile_ns:
                    if p.is_cuda:
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    ortho = _zeropower_via_newtonschulz5(g_eff, ns_steps)
                    if p.is_cuda:
                        torch.cuda.synchronize()
                    self.ns_seconds += time.perf_counter() - t0
                    self.ns_calls += 1
                else:
                    ortho = _zeropower_via_newtonschulz5(g_eff, ns_steps)
                scale = rms_scale * math.sqrt(max(p.shape[0], p.shape[1]))
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)  # decoupled WD (uses current θ)
                p.add_(ortho, alpha=-lr * scale)

        return loss


def split_muon_adamw_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition ``model``'s parameters into ``(muon_params, adamw_params)`` — the nanochat/Moonlight
    hybrid split.

    Rule (exactly the F1 contract): the token embedding and the LM head go to **AdamW** even when
    2-D (the input/output layers Muon deliberately excludes — and, critically, when
    ``tie_embeddings`` shares them as **one** tensor, that 2-D tensor must NOT go to Muon,
    ``model.py:917``); every remaining 2-D matrix (attention/MLP block projections) goes to **Muon**;
    every 1-D parameter (RMSNorm gains, biases, scalars) and any ≠2-D tensor (e.g. stacked 3-D MoE
    experts — deferred to F6) goes to AdamW.

    ``model.parameters()`` already yields a shared (tied) tensor once, so the partition is a true
    disjoint cover: ``len(muon)+len(adamw)`` unique tensors == the model's unique parameter count,
    with no overlap.
    """
    special_ids: set[int] = set()
    for attr in ("token_emb", "lm_head"):
        module = getattr(model, attr, None)
        weight = getattr(module, "weight", None)
        if isinstance(weight, Tensor):
            special_ids.add(id(weight))

    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    seen: set[int] = set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in special_ids or p.ndim != 2:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    return muon_params, adamw_params


class CombinedOptimizer:
    """Steps a list of optimizers as one — the F1/F4 hybrid (Muon on block matrices + AdamW on
    embeddings/head/norms), presented through the single-optimizer interface ``train.py`` and
    checkpointing already expect.

    - ``param_groups`` returns the sub-optimizers' **live** group dicts concatenated, so an LR
      schedule that writes ``group["lr"]`` mutates the real groups (no copy).
    - ``state_dict`` / ``load_state_dict`` round-trip every sub-optimizer's state so a checkpoint
      resumes exactly (the same contract as a single optimizer).
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("CombinedOptimizer needs at least one optimizer")
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, object]]:
        return [group for opt in self.optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict[str, object]:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state: dict[str, object]) -> None:
        subs = state["optimizers"]
        if not isinstance(subs, list):
            raise ValueError("CombinedOptimizer state_dict must hold a list under 'optimizers'")
        for opt, sub in zip(self.optimizers, subs, strict=True):
            opt.load_state_dict(sub)


def build_optimizer(
    model: nn.Module,
    *,
    kind: str = "adamw",
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    muon_momentum: float = 0.95,
    muon_profile_ns: bool = False,
) -> torch.optim.Optimizer | CombinedOptimizer:
    """Construct the training optimizer.

    ``kind="adamw"`` — the A1 default (one AdamW over every parameter). ``kind="muon_adamw"`` — the
    F1 hybrid: :class:`Muon` on the 2-D block matrices, :class:`AdamW` on the embed/head/1-D params,
    stepped together by :class:`CombinedOptimizer`. Both optimizers share ``lr`` — the Moonlight
    RMS-match makes Muon's effective update land in AdamW's band, so ONE LR schedule serves both
    (per-group LR tuning à la nanochat's 0.02/0.2/0.004 split is a later refinement).
    """
    if kind == "adamw":
        return AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if kind == "muon_adamw":
        muon_params, adamw_params = split_muon_adamw_params(model)
        muon = Muon(
            muon_params,
            lr=lr,
            momentum=muon_momentum,
            weight_decay=weight_decay,
            profile_ns=muon_profile_ns,
        )
        adamw = AdamW(adamw_params, lr=lr, betas=betas, weight_decay=weight_decay)
        return CombinedOptimizer([muon, adamw])
    raise ValueError(f"unknown optimizer kind {kind!r} (expected 'adamw' or 'muon_adamw')")
