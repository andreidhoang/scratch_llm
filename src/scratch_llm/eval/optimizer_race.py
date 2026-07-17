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
- **Divergence is data, not a crash:** ``run_arm`` contains train()'s non-finite-loss
  RuntimeError as ``ArmResult.diverged=True`` (divergence at the reused LR is itself a
  pre-registered KILL outcome — it must be *recordable*, never lose the run). ``sweep_lr``
  excludes diverged arms from the argmin; ``run_race`` raises on a diverged baseline (the
  tuned LR came from the sweep — divergence means the sweep lied) and returns ``None`` metrics
  on a diverged challenger. The strict curve validation still guards every non-diverged path.

Interview question this answers: "design an optimizer A/B that survives review — what do you hold
constant, what do you tune, and what number do you report?" (Hold C and the data stream; tune the
baseline's LR independently; report tokens-to-match against the *tuned* baseline + the optimizer's
wall-clock overhead.)
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
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
    # F9 ride-along: max over layers/heads of the observer's running max pre-softmax attention
    # logit (None unless the model was built with ModelConfig.track_attn_logits=True). The F1 GPU
    # run reads its F9 falsifier (S_max < 30 under qk_norm ⇒ QK-Clip γ≡1 sub-1B) from this field.
    max_attn_logit: float | None = None
    # Divergence containment: True when train() raised its non-finite-loss RuntimeError. A
    # diverged arm's val_curve is the raw partial signal (it may hold NaN points recorded after
    # the weights went bad) and train_history is empty — never feed either to the pure metrics.
    diverged: bool = False

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

    Divergence containment: train()'s non-finite-loss RuntimeError (and ONLY that — anything
    else re-raises) returns as ``diverged=True`` with the partial ``val_curve`` collected up to
    the failing step and an empty ``train_history`` (train() owns its history; the raise
    discards it). The NS counters and the F9 max-logit observer are still read — the optimizer
    lands in ``optimizer_out`` before the loop starts and the model outlives the raise.
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
    history: list[tuple[int, float]] = []
    diverged = False
    t0 = time.perf_counter()
    try:
        history = train(
            cfg,
            train_data,
            model,
            val_data=val_data,
            eval_hook=_hook,
            optimizer_out=optimizer_out,  # type: ignore[arg-type]
        )
    except RuntimeError as err:
        # Narrow match on train()'s fail-loud divergence signal: a diverged arm is DATA (the
        # pre-registered KILL outcome must be recordable); any other RuntimeError is a real crash.
        if "non-finite loss" not in str(err):
            raise
        diverged = True
    wall = time.perf_counter() - t0

    ns_seconds, ns_calls = 0.0, 0
    built = optimizer_out[0] if optimizer_out else None
    subs = built.optimizers if isinstance(built, CombinedOptimizer) else [built]
    for sub in subs:
        if isinstance(sub, Muon):
            ns_seconds += sub.ns_seconds
            ns_calls += sub.ns_calls

    max_attn_logit: float | None = None
    running_maxima = [
        float(running.max().item())
        for block in getattr(model, "blocks", [])
        if (running := getattr(getattr(block, "attn", None), "max_logits_running", None))
        is not None
    ]
    if running_maxima:
        max_attn_logit = max(running_maxima)

    return ArmResult(
        label=label,
        optimizer=optimizer,
        lr=max_lr,
        train_history=history,
        val_curve=val_curve,
        wall_seconds=wall,
        ns_seconds=ns_seconds,
        ns_calls=ns_calls,
        max_attn_logit=max_attn_logit,
        diverged=diverged,
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

    Diverged arms stay in ``arms`` (a diverged LR is sweep signal worth ledgering) but are
    excluded from the argmin — as is any arm whose final val is non-finite (divergence the
    train-loss check missed between log steps). Every LR diverging raises: there is no tunable
    baseline, and racing against one would be the exact fake-win 2509.02046 warns about.
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
    finishers = [
        arm
        for arm in arms
        if not arm.diverged and arm.val_curve and math.isfinite(arm.val_curve[-1][1])
    ]
    if not finishers:
        raise RuntimeError(
            f"sweep_lr: every LR in {[f'{lr:g}' for lr in lrs]} diverged — no tunable baseline "
            "exists at this horizon; re-sweep on a lower LR grid before racing"
        )
    best = min(finishers, key=lambda arm: (arm.val_curve[-1][1], arm.lr))
    return SweepResult(arms=arms, best_lr=best.lr)


@dataclass
class RaceResult:
    """The full iso-FLOP verdict: both arms, the shared budget, and the pre-registered metrics.

    All three metric fields are ``None`` when the challenger diverged (``challenger.diverged``)
    — a diverged curve has no honest crossing or endpoint delta, and the caller records the
    pre-registered KILL (divergence at the reused LR) instead of a number. ``tokens_to_match``
    and ``token_saving`` are additionally ``None`` for a healthy challenger that never reaches
    the baseline's final loss; ``nats_delta`` is always a float on the non-diverged path.
    """

    baseline: ArmResult
    challenger: ArmResult
    n_params: int
    total_tokens: int
    compute_flops: float
    tokens_to_match: float | None
    token_saving: float | None
    nats_delta: float | None


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
    arm_hook: Callable[[str, ArmResult], None] | None = None,
) -> RaceResult:
    """Race challenger vs baseline at identical N, D, seed, and data stream — only the optimizer
    (and its independently chosen LR) differs. Returns every pre-registered F1 metric.

    ``arm_hook(role, arm)`` (role ∈ {"baseline", "challenger"}) fires the moment each arm
    finishes — the driver's incremental-persistence seam: a 5–7 h race must never hold its only
    copy of a completed arm in memory. The hook fires for a diverged baseline too (so the
    evidence hits disk) BEFORE this raises: a diverging baseline means the sweep that chose its
    LR lied (different horizon or data reshuffle) — the race is void and unrecoverable. A
    diverged *challenger* is the pre-registered KILL: the race returns with all three metric
    fields ``None`` so the caller can record it.
    """
    baseline = run_arm(
        model_cfg,
        train_cfg,
        train_data,
        val_data,
        optimizer=baseline_optimizer,
        max_lr=baseline_lr,
        label=f"{baseline_optimizer}-baseline",
    )
    if arm_hook is not None:
        arm_hook("baseline", baseline)
    if baseline.diverged:
        raise RuntimeError(
            f"baseline ({baseline.label}, lr={baseline.lr:g}) diverged — its LR was the sweep "
            "winner, so the sweep lied (horizon or data mismatch); the race is void, re-sweep "
            "before racing"
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
    if arm_hook is not None:
        arm_hook("challenger", challenger)
    seed_everything(train_cfg.seed)  # param COUNT is init-independent; reseed only for hygiene
    n_params = sum(p.numel() for p in TransformerLM(model_cfg).parameters())
    total_tokens = train_cfg.max_steps * train_cfg.batch_size * train_cfg.context_length
    if challenger.diverged:
        matched, saving, nats = None, None, None
    else:
        matched = tokens_to_match(baseline.val_curve, challenger.val_curve)
        saving = token_saving_fraction(baseline.val_curve, challenger.val_curve)
        nats = nats_delta_at_budget(baseline.val_curve, challenger.val_curve)
    return RaceResult(
        baseline=baseline,
        challenger=challenger,
        n_params=n_params,
        total_tokens=total_tokens,
        compute_flops=compute_from_params_tokens(n_params, total_tokens),
        tokens_to_match=matched,
        token_saving=saving,
        nats_delta=nats,
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
