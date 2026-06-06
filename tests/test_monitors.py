"""Tests for the RL run monitors — KL divergences, kl_train_infer HALT, IS ratios/ESS,
distribution stats, and the all-mandatory MonitorSnapshot."""

import math

import numpy as np
import pytest

from reasoning_llm.utils.monitors import (
    KL_TRAIN_INFER_HALT,
    MonitorSnapshot,
    build_snapshot,
    distribution_stats,
    effective_sample_size,
    importance_ratios,
    kl_divergence,
    log_softmax,
    mean_kl,
    normalized_ess,
)


def test_log_softmax_normalizes() -> None:
    lp = log_softmax(np.array([1.0, 2.0, 3.0]))
    assert math.isclose(np.exp(lp).sum(), 1.0, rel_tol=1e-9)


def test_kl_is_zero_for_identical_distributions() -> None:
    lp = log_softmax(np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 5.0]]))
    assert np.allclose(kl_divergence(lp, lp), 0.0, atol=1e-12)


def test_kl_is_nonnegative_and_asymmetric() -> None:
    # Non-symmetric distributions (not mirror images), so KL(p‖q) != KL(q‖p).
    p = np.log([0.7, 0.2, 0.1])
    q = np.log([0.1, 0.3, 0.6])
    kl_pq = mean_kl(p, q)
    kl_qp = mean_kl(q, p)
    assert kl_pq > 0 and kl_qp > 0
    assert not math.isclose(kl_pq, kl_qp)  # KL is asymmetric


def test_kl_matches_hand_computation() -> None:
    # p = [0.5, 0.5], q = [0.25, 0.75]; KL = 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75)
    log_p = np.log([0.5, 0.5])
    log_q = np.log([0.25, 0.75])
    expected = 0.5 * math.log(2.0) + 0.5 * math.log(0.5 / 0.75)
    assert math.isclose(float(kl_divergence(log_p, log_q)), expected, rel_tol=1e-9)


def test_kl_train_infer_detects_engine_drift() -> None:
    rng = np.random.default_rng(0)
    train_logits = rng.normal(size=(8, 50))
    train_lp = log_softmax(train_logits)
    # identical engines → ~0, no halt
    assert mean_kl(train_lp, train_lp) <= KL_TRAIN_INFER_HALT
    # a perturbed serving engine → positive divergence
    infer_lp = log_softmax(train_logits + rng.normal(scale=2.0, size=train_logits.shape))
    assert mean_kl(train_lp, infer_lp) > 0


def test_importance_ratios_are_one_when_policies_match() -> None:
    logp = np.log([0.2, 0.5, 0.3])
    assert np.allclose(importance_ratios(logp, logp), 1.0)


def test_ess_uniform_equals_n_and_concentrated_goes_to_one() -> None:
    assert math.isclose(effective_sample_size(np.ones(10)), 10.0, rel_tol=1e-9)
    spiked = np.array([1e6, 1.0, 1.0])
    assert effective_sample_size(spiked) < 1.01
    assert 0.0 <= normalized_ess(np.ones(10)) <= 1.0


def test_ess_rejects_empty() -> None:
    with pytest.raises(ValueError):
        effective_sample_size(np.array([]))


def test_distribution_stats_on_known_array() -> None:
    s = distribution_stats([0.0, 1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.0 and s["min"] == 0.0 and s["max"] == 4.0
    assert s["median"] == 2.0


def test_snapshot_halt_threshold_is_strict() -> None:
    def snap(kl_ti: float) -> MonitorSnapshot:
        return build_snapshot(
            kl_current_ref=0.01,
            kl_current_old=0.02,
            kl_train_infer=kl_ti,
            is_ratios=np.ones(4),
            rewards=np.array([0.0, 1.0]),
            lengths=np.array([10, 20]),
        )

    assert snap(0.05).halt is False
    assert snap(KL_TRAIN_INFER_HALT).halt is False  # boundary: strict >
    assert snap(0.20).halt is True


def test_build_snapshot_populates_all_fields() -> None:
    snap = build_snapshot(
        kl_current_ref=0.01,
        kl_current_old=0.02,
        kl_train_infer=0.03,
        is_ratios=np.array([1.0, 1.0, 1.0, 1.0]),
        rewards=np.array([0.0, 0.5, 1.0]),
        lengths=np.array([5, 10, 100]),
    )
    assert snap.is_ratio_mean == 1.0
    assert math.isclose(snap.is_ratio_ess, 1.0)  # uniform ratios → full ESS
    assert snap.reward_max == 1.0
    assert snap.length_max == 100.0
    assert snap.halt is False
