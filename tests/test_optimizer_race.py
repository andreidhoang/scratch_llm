"""Tests for the F1-run iso-FLOP optimizer race (eval/optimizer_race.py + the train.py val hook).

Executable spec for the pending F1 headline (Muon vs LR-tuned AdamW at fixed C=6ND):
- **Pure metrics:** ``tokens_to_match`` finds the first (interpolated) token count where the
  challenger reaches the baseline's final val loss; ``token_saving_fraction`` and
  ``nats_delta_at_budget`` derive from it. All fail loud on malformed curves.
- **Byte-identical default:** turning the val-eval hook ON must not perturb the training batch
  stream (val eval uses fixed sequential windows — no RNG), and the OFF path is untouched.
- **NS instrument:** ``Muon(profile_ns=True)`` accumulates Newton–Schulz wall time; OFF costs zero.
- **Race mechanism:** both arms share seed/init/data; the driver returns comparable curves at the
  same token budget (the iso-FLOP contract) deterministically.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from scratch_llm.eval.optimizer_race import (
    nats_delta_at_budget,
    run_arm,
    run_race,
    sweep_lr,
    token_saving_fraction,
    tokens_to_match,
)
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import Muon
from scratch_llm.train import TrainConfig, train
from scratch_llm.utils.seeding import seed_everything

# ---------------------------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------------------------


def test_tokens_to_match_interpolates_known_crossing() -> None:
    # Baseline ends at loss 2.0 after 1000 tokens; challenger crosses 2.0 between 400 and 600:
    # linear interp: 400 + 200 * (2.5 - 2.0) / (2.5 - 1.5) = 500.
    baseline = [(500.0, 3.0), (1000.0, 2.0)]
    challenger = [(200.0, 3.5), (400.0, 2.5), (600.0, 1.5), (1000.0, 1.0)]
    assert tokens_to_match(baseline, challenger) == 500.0


def test_tokens_to_match_first_point_already_below() -> None:
    # Challenger's first measurement is already at/below target: we can honestly claim no
    # earlier than the first measured point.
    baseline = [(1000.0, 2.0)]
    challenger = [(250.0, 1.9), (1000.0, 1.0)]
    assert tokens_to_match(baseline, challenger) == 250.0


def test_tokens_to_match_none_when_never_reached() -> None:
    baseline = [(1000.0, 2.0)]
    challenger = [(500.0, 3.0), (1000.0, 2.5)]
    assert tokens_to_match(baseline, challenger) is None


def test_token_saving_fraction_identical_curves_is_zero() -> None:
    curve = [(250.0, 3.0), (500.0, 2.5), (1000.0, 2.0)]
    saving = token_saving_fraction(curve, list(curve))
    assert saving is not None
    assert abs(saving) < 1e-12  # matches exactly at the shared final point


def test_token_saving_fraction_better_challenger() -> None:
    baseline = [(500.0, 3.0), (1000.0, 2.0)]
    challenger = [(200.0, 3.5), (400.0, 2.5), (600.0, 1.5), (1000.0, 1.0)]
    saving = token_saving_fraction(baseline, challenger)
    assert saving is not None
    assert abs(saving - 0.5) < 1e-12  # crossed at 500 of 1000 baseline tokens


def test_nats_delta_at_budget_sign_convention() -> None:
    # Challenger 0.3 nats LOWER at the shared budget -> delta = -0.3.
    baseline = [(1000.0, 2.0)]
    challenger = [(1000.0, 1.7)]
    assert abs(nats_delta_at_budget(baseline, challenger) - (-0.3)) < 1e-12


def test_nats_delta_rejects_mismatched_budgets() -> None:
    baseline = [(1000.0, 2.0)]
    challenger = [(800.0, 1.7)]  # not iso-FLOP: final token counts differ
    try:
        nats_delta_at_budget(baseline, challenger)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched final token budgets")


def test_metrics_reject_malformed_curves() -> None:
    good = [(1000.0, 2.0)]
    for bad in (
        [],  # empty
        [(500.0, 2.0), (400.0, 1.9)],  # tokens not increasing
        [(500.0, float("nan"))],  # non-finite loss
    ):
        for args in ((bad, good), (good, bad)):
            try:
                tokens_to_match(*args)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for malformed curve {bad!r}")


# ---------------------------------------------------------------------------------------------
# train.py val-eval hook — the additive no-op contract
# ---------------------------------------------------------------------------------------------


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, context_length=16)


def _structured_corpus(n: int = 2048, vocab: int = 32) -> np.ndarray:
    return np.tile(np.arange(vocab, dtype=np.int64), n // vocab + 1)[:n]


def test_val_eval_hook_does_not_perturb_training() -> None:
    """The DoD contract: identical train-loss history with the hook OFF vs ON.

    Stronger than 'default path unchanged' — the ON path samples val loss from FIXED sequential
    windows (no RNG), so even an active hook must leave the batch stream byte-identical.
    """
    data = _structured_corpus()
    val = _structured_corpus(512)
    base = TrainConfig(max_steps=12, batch_size=4, context_length=16, log_every=1, seed=7)

    seed_everything(7)
    hist_off = train(base, data, TransformerLM(_tiny_cfg()))

    collected: list[tuple[int, float]] = []
    seed_everything(7)
    hist_on = train(
        replace(base, eval_every=4),
        data,
        TransformerLM(_tiny_cfg()),
        val_data=val,
        eval_hook=lambda step, loss: collected.append((step, loss)),
    )

    assert hist_on == hist_off  # bitwise-identical loss history
    steps = [s for s, _ in collected]
    assert steps and steps[-1] == 11  # final step always evaluated (the iso-FLOP endpoint)
    assert all(np.isfinite(loss) for _, loss in collected)


def test_val_eval_is_deterministic() -> None:
    data = _structured_corpus()
    val = _structured_corpus(512)
    curves: list[list[tuple[int, float]]] = []
    for _ in range(2):
        collected: list[tuple[int, float]] = []
        seed_everything(3)
        train(
            TrainConfig(max_steps=8, batch_size=4, context_length=16, seed=3, eval_every=4),
            data,
            TransformerLM(_tiny_cfg()),
            val_data=val,
            eval_hook=lambda step, loss, c=collected: c.append((step, loss)),
        )
        curves.append(collected)
    assert curves[0] == curves[1]


# ---------------------------------------------------------------------------------------------
# Muon NS wall-time instrument
# ---------------------------------------------------------------------------------------------


def test_muon_profile_ns_counters() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([p], lr=1e-3, profile_ns=True)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    assert opt.ns_calls == 3
    assert opt.ns_seconds > 0.0


def test_muon_profile_ns_off_by_default() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.ns_calls == 0
    assert opt.ns_seconds == 0.0
    # The instrument is an instance attribute, NOT optimizer state: checkpoints are unaffected.
    assert "profile_ns" not in opt.state_dict()["param_groups"][0]


# ---------------------------------------------------------------------------------------------
# The A/B driver — determinism + the iso-FLOP contract
# ---------------------------------------------------------------------------------------------


def test_run_arm_is_deterministic() -> None:
    data = _structured_corpus()
    val = _structured_corpus(512)
    cfg = TrainConfig(max_steps=12, batch_size=4, context_length=16, seed=11, eval_every=4)
    a = run_arm(_tiny_cfg(), cfg, data, val, optimizer="adamw", max_lr=3e-3, label="a")
    b = run_arm(_tiny_cfg(), cfg, data, val, optimizer="adamw", max_lr=3e-3, label="b")
    assert a.val_curve == b.val_curve
    assert a.train_history == b.train_history


def test_run_race_iso_flop_contract() -> None:
    data = _structured_corpus(4096)
    val = _structured_corpus(512)
    cfg = TrainConfig(max_steps=16, batch_size=4, context_length=16, seed=5, eval_every=4)
    race = run_race(_tiny_cfg(), cfg, data, val, baseline_lr=3e-3, challenger_lr=3e-3)

    # Both arms end at the same token budget — C=6ND held constant by construction.
    assert race.baseline.val_curve[-1][0] == race.challenger.val_curve[-1][0]
    assert race.total_tokens == 16 * 4 * 16
    assert race.compute_flops == 6.0 * race.n_params * race.total_tokens
    assert len(race.baseline.val_curve) >= 3
    # The metric fields are populated coherently.
    if race.tokens_to_match is not None:
        assert 0 < race.tokens_to_match <= race.total_tokens
        assert race.token_saving is not None
    assert np.isfinite(race.nats_delta)
    # Muon arm carries the NS instrument when asked.
    assert race.challenger.ns_seconds >= 0.0


def test_sweep_lr_picks_argmin_final_val_loss() -> None:
    data = _structured_corpus(4096)
    val = _structured_corpus(512)
    cfg = TrainConfig(max_steps=10, batch_size=4, context_length=16, seed=9, eval_every=5)
    sweep = sweep_lr(_tiny_cfg(), cfg, data, val, lrs=[3e-3, 1e-9])
    finals = {arm.lr: arm.val_curve[-1][1] for arm in sweep.arms}
    assert sweep.best_lr == min(finals, key=lambda lr: finals[lr])
    # An LR of 1e-9 cannot learn anything in 10 steps; 3e-3 must beat it on this corpus.
    assert sweep.best_lr == 3e-3
