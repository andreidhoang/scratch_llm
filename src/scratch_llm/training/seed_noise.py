"""Seed-to-seed sigma of bpb, its degrees of freedom, and the tolerance that is built on it.

T1/T-R2's quality gate is "Delta bpb <= 2 sigma". That sentence is only a gate if sigma is the
right sigma, and the right sigma here is a *measured* property of this setup: the standard
deviation of final val bpb across training seeds, at the same token budget, same data, same eval
stream, everything but the seed held fixed. It is not a constant from a paper and it is not
transferable between models, corpora, or token budgets — it is the noise floor of THIS experiment,
and the fp8 arm's delta means nothing except against it.

Two seeds per arm is what the plan budgets, so the estimate has n = 2 and

    sigma_hat = |x1 - x2| / sqrt(2)          (the sample standard deviation, ddof = 1)

with **one** degree of freedom. This module refuses to let that be invisible.
:meth:`SeedSigma.ci` turns the chi-square sampling distribution of s^2 into an interval for the
true sigma, and at dof = 1 that interval is [0.446 sigma_hat, 31.9 sigma_hat]: the truth is
somewhere between half the estimate and thirty-two times it. Everything downstream of a one-dof
sigma inherits that width, which is the fact the tolerance has to be argued against.

The estimator lives here. The tolerance does not — see the hole below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: Two-sided 95% chi-square quantiles (0.025, 0.975) by degrees of freedom, for the interval
#: [ (dof-1) s^2 / chi2_hi , (dof-1) s^2 / chi2_lo ] on the variance. Values from
#: ``scipy.stats.chi2.ppf(q, dof)`` (scipy 1.18.0); hardcoded because scipy is an optional extra
#: (``pyproject [scaling]``) and this module must import on a bare box.
#: ``tests/training/test_t1_t_r2.py`` re-derives them from scipy when it is installed.
CHI2_95: dict[int, tuple[float, float]] = {
    1: (0.0009820691171752555, 5.02388618731489),
    2: (0.05063561596857975, 7.377758908227872),
    3: (0.21579528262389794, 9.348403604496145),
    4: (0.4844185570879299, 11.143286781877796),
    5: (0.8312116134866626, 12.832501994030025),
    6: (1.2373442457912025, 14.449375335447918),
    7: (1.6898691806773551, 16.012764274629323),
    8: (2.1797307472526497, 17.534546139484647),
    9: (2.7003894999803584, 19.02276779864163),
    10: (3.246972780236841, 20.48317735080739),
    11: (3.8157482522360993, 21.9200492610212),
    12: (4.4037885069817015, 23.33666415864534),
}


def sigma_ci_factors(dof: int) -> tuple[float, float]:
    """Multiply ``sigma_hat`` by these to get a 95% interval for the TRUE sigma.

    ``(sqrt(dof / chi2_0.975), sqrt(dof / chi2_0.025))``. At dof = 1 that is (0.446, 31.9) — a
    two-order-of-magnitude window, which is what "estimated from two seeds" actually buys.
    """
    if dof not in CHI2_95:
        raise KeyError(
            f"no chi-square quantiles tabulated for dof={dof} (have {sorted(CHI2_95)}) — add the "
            "row from scipy.stats.chi2.ppf rather than interpolating"
        )
    lo, hi = CHI2_95[dof]
    return math.sqrt(dof / hi), math.sqrt(dof / lo)


@dataclass(frozen=True)
class SeedSigma:
    """The seed-noise estimate of one arm: its values, their spread, and how little that spread
    is known. ``values`` are final val bpb, one per seed, at a fixed token budget."""

    values: tuple[float, ...]
    mean: float
    sigma: float

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def dof(self) -> int:
        """n - 1. At two seeds this is 1, and every interval below is as wide as that implies."""
        return self.n - 1

    @property
    def stderr_of_mean(self) -> float:
        """sigma_hat / sqrt(n) — the noise on the ARM MEAN, which is not the noise on one run."""
        return self.sigma / math.sqrt(self.n)

    def ci(self) -> tuple[float, float]:
        """95% interval for the true sigma, given this many degrees of freedom."""
        lo, hi = sigma_ci_factors(self.dof)
        return self.sigma * lo, self.sigma * hi

    def as_dict(self) -> dict[str, object]:
        lo, hi = self.ci()
        return {
            "values": list(self.values),
            "mean": self.mean,
            "sigma": self.sigma,
            "n": self.n,
            "dof": self.dof,
            "stderr_of_mean": self.stderr_of_mean,
            "sigma_ci95": [lo, hi],
        }


def seed_sigma(values: Sequence[float]) -> SeedSigma:
    """Sample mean and sample standard deviation (ddof = 1) of one arm's per-seed bpb.

    ddof = 1, not 0: these seeds are a sample from the population of seeds, and dividing by n
    would understate the spread by sqrt(2) at n = 2 — a 30% tighter gate, for free, in the wrong
    direction. Fewer than two values raises: one seed carries no information about seed noise,
    and a nan or a 0.0 returned here would be read downstream as "no noise".
    """
    xs = [float(v) for v in values]
    if len(xs) < 2:
        raise ValueError(
            f"seed sigma needs at least 2 seeds, got {len(xs)} — with one run there is no "
            "estimate of seed noise, and no gate of the form 'Delta <= k sigma' exists"
        )
    if not all(math.isfinite(x) for x in xs):
        raise ValueError(f"non-finite bpb in {xs} — a diverged run is not a sample")
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return SeedSigma(values=tuple(xs), mean=mean, sigma=math.sqrt(var))


# HUY: the Delta-bpb tolerance for T-R2 — what sigma is here, how two seeds estimate it, and what "<= 2 sigma" therefore admits — spec: experiments/T1/T-R2/spec.md — fill before T-R2
def bpb_gate_tolerance(sigma: SeedSigma) -> float:
    """The largest ``|Delta bpb|`` the fp8 arm may show and still pass, given the bf16 arm's
    measured seed noise.

    Everything the decision needs is in ``sigma``: the per-seed values, ``sigma.sigma`` (ddof = 1),
    ``sigma.dof`` (1, at two seeds), ``sigma.stderr_of_mean``, and ``sigma.ci()`` — the 95% window
    for the true sigma, which at one degree of freedom spans 0.446x to 31.9x the estimate. Three
    questions the argument has to answer, and none of them has a default:

    1. **Which quantity is the gate on** — the difference of arm means (noise
       ``sigma sqrt(2/n)``), or the per-seed paired delta (the runs share a data order, so the
       pairing removes the data-order component of the noise and leaves a smaller one)?
       ``Comparison`` reports both; they are not the same number and they do not have the same
       noise.
    2. **Which sigma multiplies the k** — the point estimate, or a bound from ``ci()``? A gate at
       ``2 * sigma_hat`` with dof = 1 admits any true effect below roughly ``0.9 sigma_true``
       only if ``sigma_hat`` happened to land near ``sigma_true``, and the interval says it need
       not have.
    3. **What does passing then license**, and what would a failure at this width have had to be
       to be detected? A tolerance that cannot state its own detectable effect size is a
       rubber stamp with a number on it.

    Plan section 05 carries a prior for this sigma, quoted from a run on a different card, model,
    and corpus. It is not this experiment's sigma and no number from it belongs in this file: the
    bf16 floor run measures the real one and hands it in as ``sigma``.
    """
    raise NotImplementedError(
        "HUY: T-R2's Delta-bpb tolerance is unset. Decide it from the bf16 arm's measured seed "
        "noise (mean/paired, point estimate or CI bound, and what the resulting width detects), "
        "write the argument on the 'Correctness gate' line of experiments/T1/T-R2/spec.md, then "
        "return the number here as a function of `sigma`."
    )


__all__ = [
    "CHI2_95",
    "SeedSigma",
    "bpb_gate_tolerance",
    "seed_sigma",
    "sigma_ci_factors",
]
