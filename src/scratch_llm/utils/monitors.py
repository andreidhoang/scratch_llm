"""RL run monitors — the guardrails that make a run *interpretable*.

The reasoningLLM discipline (CLAUDE.md #4): an RL run with these absent is uninterpretable.
Every run logs, separately:
- the **three KL divergences** — ``KL(current‖ref)``, ``KL(current‖old)``, and
  ``kl_train_infer = KL(train‖infer)`` (training-engine vs serving-engine logits);
- the **importance-sampling ratios** (histogram + effective sample size);
- the **reward** distribution; and the **completion-length** distribution (the verbosity
  reward-hacking tell).

``kl_train_infer`` is the load-bearing one: if the training engine and the serving engine
disagree on next-token distributions, every gradient is computed against a policy that
isn't the one being served. HALT threshold = **0.10**.

Pure numpy so it is engine-agnostic (train=torch, infer=SGLang both reduce to numpy
log-probs) and CPU-testable. Inputs are log-probabilities over the vocab; convert model
logits with :func:`log_softmax` first.

Caveat: serving engines often return only top-k log-probs. These KLs assume aligned
full-vocab (or consistently truncated) distributions; reconcile truncation before trusting
the number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# kl_train_infer HALT threshold — above this the run is not measuring the served policy.
KL_TRAIN_INFER_HALT = 0.10


def log_softmax(logits: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Numerically stable log-softmax (subtract the max before exp)."""
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - arr.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def kl_divergence(log_p: ArrayLike, log_q: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """KL(p‖q) per distribution, from log-probabilities. Collapses the vocab axis.

    KL(p‖q) = Σ_x p(x)·(log p(x) − log q(x)) ≥ 0, with 0·log0 ≡ 0. Asymmetric in p, q.
    """
    log_p = np.asarray(log_p, dtype=np.float64)
    log_q = np.asarray(log_q, dtype=np.float64)
    p = np.exp(log_p)
    terms = np.where(p > 0, p * (log_p - log_q), 0.0)  # enforce 0·log0 = 0
    return terms.sum(axis=axis)


def mean_kl(log_p: ArrayLike, log_q: ArrayLike) -> float:
    """Mean per-token KL(p‖q) over all positions — the scalar a run logs."""
    return float(kl_divergence(log_p, log_q).mean())


def importance_ratios(logp_current: ArrayLike, logp_old: ArrayLike) -> NDArray[np.float64]:
    """Per-token IS weights π_current(a|s)/π_old(a|s) = exp(logp_current − logp_old),
    where the log-probs are of the *taken* action under each policy."""
    logp_current = np.asarray(logp_current, dtype=np.float64)
    logp_old = np.asarray(logp_old, dtype=np.float64)
    return np.exp(logp_current - logp_old)


def effective_sample_size(weights: ArrayLike) -> float:
    """ESS = (Σw)² / Σ(w²). Equals N for uniform weights, → 1 when one weight dominates.
    A collapsing ESS means the off-policy correction has blown up."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        raise ValueError("effective_sample_size: empty weights (a run with no samples)")
    s2 = float((w**2).sum())
    if s2 == 0.0:
        return 0.0
    return float(w.sum() ** 2 / s2)


def normalized_ess(weights: ArrayLike) -> float:
    """ESS as a fraction of N, in [0, 1]. Below ~0.1 the gradient estimate is unreliable."""
    w = np.asarray(weights, dtype=np.float64)
    return effective_sample_size(w) / w.size


def histogram(values: ArrayLike, bins: int = 20) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """(counts, bin_edges) — for logging the IS-ratio (or any) distribution shape."""
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    return counts.astype(np.int64), edges


def distribution_stats(values: ArrayLike) -> dict[str, float]:
    """Descriptive stats (mean/std/min/max/median/p10/p90) for rewards, lengths, etc."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        raise ValueError("distribution_stats: empty input (a run with no samples)")
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


@dataclass(frozen=True)
class MonitorSnapshot:
    """The loggable summary of one RL step. Constructing it requires *every* mandatory
    field — absence of any one makes the run uninterpretable, so the type enforces it."""

    kl_current_ref: float
    kl_current_old: float
    kl_train_infer: float
    is_ratio_mean: float
    is_ratio_ess: float  # normalized ESS in [0, 1]
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    length_mean: float
    length_p90: float
    length_max: float

    @property
    def halt(self) -> bool:
        """True when kl_train_infer has crossed the HALT threshold."""
        return self.kl_train_infer > KL_TRAIN_INFER_HALT


def build_snapshot(
    *,
    kl_current_ref: float,
    kl_current_old: float,
    kl_train_infer: float,
    is_ratios: ArrayLike,
    rewards: ArrayLike,
    lengths: ArrayLike,
) -> MonitorSnapshot:
    """Bundle one step's metrics into a :class:`MonitorSnapshot`. Keyword-only and
    all-required, so no mandatory guardrail can be silently omitted."""
    reward = distribution_stats(rewards)
    length = distribution_stats(lengths)
    return MonitorSnapshot(
        kl_current_ref=kl_current_ref,
        kl_current_old=kl_current_old,
        kl_train_infer=kl_train_infer,
        is_ratio_mean=float(np.asarray(is_ratios, dtype=np.float64).mean()),
        is_ratio_ess=normalized_ess(is_ratios),
        reward_mean=reward["mean"],
        reward_std=reward["std"],
        reward_min=reward["min"],
        reward_max=reward["max"],
        length_mean=length["mean"],
        length_p90=length["p90"],
        length_max=length["max"],
    )
