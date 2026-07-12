"""Iso-FLOP optimizer race — the F1-run harness (Muon vs LR-tuned AdamW at fixed C=6ND).

The pending F1 headline (`bench/RESULTS.md` §Frontier ablations) is a *tokens-to-match* claim:
at a fixed compute budget, how many fewer tokens does the MuonAdamW hybrid need to reach the
AdamW baseline's final validation loss? The 2026-07-09 recalibration made the **LR-tuned AdamW
baseline mandatory** — the deflation literature (arXiv 2509.02046) shows the 1.4–2× Muon headlines
came from under-tuned baselines, so `sweep_lr` is a first-class part of this harness, not an
afterthought. Predicted band: 1.1–1.4× (scale-dependent); KILL: <5% saving vs the *tuned* baseline.

Key invariants:
- **Iso-FLOP by construction:** both arms share N (same model config, same seeded init) and D
  (same step/batch/context budget), so C = 6ND is held constant without bookkeeping.
- **Comparable curves:** every arm evaluates on the same fixed sequential val windows
  (``train._val_loss`` — no RNG), so curve deltas are optimizer signal, not eval noise.
- **Pure metrics:** ``tokens_to_match`` / ``token_saving_fraction`` / ``nats_delta_at_budget``
  are curve-level functions with loud validation — unit-testable without any training.

Interview question this answers: "design an optimizer A/B that survives review — what do you hold
constant, what do you tune, and what number do you report?" (Hold C and the data stream; tune the
baseline's LR independently; report tokens-to-match against the *tuned* baseline + the optimizer's
wall-clock overhead.)
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import CombinedOptimizer, Muon
from scratch_llm.scaling.isoflop import compute_from_params_tokens
from scratch_llm.train import TrainConfig, train
from scratch_llm.utils.seeding import seed_everything

Curve = Sequence[tuple[float, float]]  # ((tokens_seen, val_loss), ...) — tokens strictly increasing


# ---------------------------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------------------------


def _validate_curve(curve: Curve, name: str) -> None:
    if len(curve) == 0:
        raise ValueError(f"{name} curve is empty")
    prev_x = -math.inf
    for x, y in curve:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{name} curve has a non-finite point ({x}, {y})")
        if x <= prev_x:
            raise ValueError(f"{name} curve token counts must be strictly increasing (at x={x})")
        prev_x = x


def tokens_to_match(baseline: Curve, challenger: Curve) -> float | None:
    """Tokens the challenger needs to first reach the baseline's FINAL val loss, or ``None``.

    The crossing is linearly interpolated between the challenger's bracketing eval points. If the
    challenger's very first measurement is already at/below target, its token count is returned
    as-is — we can honestly claim no earlier than the first measured point.
    """
    _validate_curve(baseline, "baseline")
    _validate_curve(challenger, "challenger")
    target = baseline[-1][1]
    for i, (x, y) in enumerate(challenger):
        if y <= target:
            if i == 0:
                return float(x)
            x_prev, y_prev = challenger[i - 1]
            # y_prev > target >= y here, so the denominator is strictly positive.
            return float(x_prev + (x - x_prev) * (y_prev - target) / (y_prev - y))
    return None


def token_saving_fraction(baseline: Curve, challenger: Curve) -> float | None:
    """``1 − tokens_to_match / D_baseline`` — the headline saving, or ``None`` if never matched."""
    matched = tokens_to_match(baseline, challenger)
    if matched is None:
        return None
    return 1.0 - matched / baseline[-1][0]


def nats_delta_at_budget(baseline: Curve, challenger: Curve) -> float:
    """``challenger − baseline`` final val loss at the SHARED token budget (negative = better).

    Raises if the two curves end at different token counts — that would mean the race was not
    iso-FLOP and the comparison is void.
    """
    _validate_curve(baseline, "baseline")
    _validate_curve(challenger, "challenger")
    bx, by = baseline[-1]
    cx, cy = challenger[-1]
    if abs(bx - cx) > 1e-6 * max(abs(bx), abs(cx), 1.0):
        raise ValueError(
            f"final token budgets differ (baseline {bx} vs challenger {cx}): not an iso-FLOP race"
        )
    return float(cy - by)


# ---------------------------------------------------------------------------------------------
# The A/B driver — runs the REAL training loop, one arm at a time
# ---------------------------------------------------------------------------------------------


@dataclass
class ArmResult:
    """One arm of the race: its config knobs, both loss histories, and wall-time instruments."""

    label: str
    optimizer: str
    lr: float
    train_history: list[tuple[int, float]]
    val_curve: list[tuple[float, float]]  # (tokens_seen, val_ce)
    wall_seconds: float
    ns_seconds: float
    ns_calls: int

    @property
    def ns_overhead(self) -> float:
        """Newton–Schulz wall time as a fraction of the arm's total wall time."""
        return self.ns_seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0


def run_arm(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    *,
    optimizer: str,
    max_lr: float,
    label: str,
    profile_ns: bool = False,
) -> ArmResult:
    """Train one arm from a fresh seeded init and return its curves.

    ``seed_everything(train_cfg.seed)`` runs BEFORE model construction, and ``train`` reseeds the
    same value for data sampling — so two arms with the same seed share both the initial weights
    and the exact training-batch stream; only the optimizer (and its LR) differs. ``min_lr`` is
    rescaled to keep the cosine schedule's max/min *ratio* fixed while sweeping ``max_lr``.
    """
    if train_cfg.eval_every <= 0:
        raise ValueError(
            "run_arm needs train_cfg.eval_every > 0 — the race is judged on val curves"
        )
    if train_cfg.max_lr <= 0:
        raise ValueError(f"train_cfg.max_lr must be positive, got {train_cfg.max_lr}")
    cfg = replace(
        train_cfg,
        optimizer=optimizer,
        max_lr=max_lr,
        min_lr=max_lr * (train_cfg.min_lr / train_cfg.max_lr),
        muon_profile_ns=profile_ns,
    )
    tokens_per_step = cfg.batch_size * cfg.context_length

    val_curve: list[tuple[float, float]] = []

    def _hook(step: int, val_ce: float) -> None:
        val_curve.append(((step + 1) * tokens_per_step, val_ce))

    seed_everything(cfg.seed)
    model = TransformerLM(model_cfg)
    optimizer_out: list[object] = []
    t0 = time.perf_counter()
    history = train(
        cfg,
        train_data,
        model,
        val_data=val_data,
        eval_hook=_hook,
        optimizer_out=optimizer_out,  # type: ignore[arg-type]
    )
    wall = time.perf_counter() - t0

    ns_seconds, ns_calls = 0.0, 0
    built = optimizer_out[0] if optimizer_out else None
    subs = built.optimizers if isinstance(built, CombinedOptimizer) else [built]
    for sub in subs:
        if isinstance(sub, Muon):
            ns_seconds += sub.ns_seconds
            ns_calls += sub.ns_calls

    return ArmResult(
        label=label,
        optimizer=optimizer,
        lr=max_lr,
        train_history=history,
        val_curve=val_curve,
        wall_seconds=wall,
        ns_seconds=ns_seconds,
        ns_calls=ns_calls,
    )


@dataclass
class SweepResult:
    """An LR sweep's arms plus the winner — the tuned-baseline prerequisite of the race."""

    arms: list[ArmResult]
    best_lr: float


def sweep_lr(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    lrs: Sequence[float],
    *,
    optimizer: str = "adamw",
) -> SweepResult:
    """Run one arm per LR and pick the argmin final-val-loss winner (ties → the smaller LR).

    The 2509.02046 lesson: a Muon "win" over an untuned AdamW is a fake win. Run this sweep at a
    shortened horizon (the caller shrinks ``max_steps``), then race at the winner's LR.
    """
    if len(lrs) == 0:
        raise ValueError("sweep_lr needs at least one learning rate")
    arms = [
        run_arm(
            model_cfg,
            train_cfg,
            train_data,
            val_data,
            optimizer=optimizer,
            max_lr=lr,
            label=f"{optimizer}@{lr:g}",
        )
        for lr in lrs
    ]
    best = min(arms, key=lambda arm: (arm.val_curve[-1][1], arm.lr))
    return SweepResult(arms=arms, best_lr=best.lr)


@dataclass
class RaceResult:
    """The full iso-FLOP verdict: both arms, the shared budget, and the pre-registered metrics."""

    baseline: ArmResult
    challenger: ArmResult
    n_params: int
    total_tokens: int
    compute_flops: float
    tokens_to_match: float | None
    token_saving: float | None
    nats_delta: float


def run_race(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    *,
    baseline_lr: float,
    challenger_lr: float,
    baseline_optimizer: str = "adamw",
    challenger_optimizer: str = "muon_adamw",
    profile_ns: bool = True,
) -> RaceResult:
    """Race challenger vs baseline at identical N, D, seed, and data stream — only the optimizer
    (and its independently chosen LR) differs. Returns every pre-registered F1 metric."""
    baseline = run_arm(
        model_cfg,
        train_cfg,
        train_data,
        val_data,
        optimizer=baseline_optimizer,
        max_lr=baseline_lr,
        label=f"{baseline_optimizer}-baseline",
    )
    challenger = run_arm(
        model_cfg,
        train_cfg,
        train_data,
        val_data,
        optimizer=challenger_optimizer,
        max_lr=challenger_lr,
        label=f"{challenger_optimizer}-challenger",
        profile_ns=profile_ns,
    )
    seed_everything(train_cfg.seed)  # param COUNT is init-independent; reseed only for hygiene
    n_params = sum(p.numel() for p in TransformerLM(model_cfg).parameters())
    total_tokens = train_cfg.max_steps * train_cfg.batch_size * train_cfg.context_length
    return RaceResult(
        baseline=baseline,
        challenger=challenger,
        n_params=n_params,
        total_tokens=total_tokens,
        compute_flops=compute_from_params_tokens(n_params, total_tokens),
        tokens_to_match=tokens_to_match(baseline.val_curve, challenger.val_curve),
        token_saving=token_saving_fraction(baseline.val_curve, challenger.val_curve),
        nats_delta=nats_delta_at_budget(baseline.val_curve, challenger.val_curve),
    )


__all__ = [
    "ArmResult",
    "RaceResult",
    "SweepResult",
    "nats_delta_at_budget",
    "run_arm",
    "run_race",
    "sweep_lr",
    "token_saving_fraction",
    "tokens_to_match",
]
