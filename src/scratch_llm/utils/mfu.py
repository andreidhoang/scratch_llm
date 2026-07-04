"""MFU / HFU instrumentation — the scoreboard every A6 scaling measurement reports against.

A6 §"MFU/HFU instrumentation". Pure CPU arithmetic, no torch, no GPU: given a model's param
count N, the tokens processed in a step, the step wall-time, the device count, and each device's
*peak* FLOP/s, this turns a stopwatch reading into the one number a training run is judged by —
what fraction of the hardware you actually paid for did useful model work.

Two utilizations, one FLOP count apart (Chowdhery et al. 2022, PaLM, Appendix B — the paper that
defined the pair):

- **MFU (Model FLOPs Utilization)** = achieved *model* FLOP/s ÷ peak FLOP/s. Model FLOPs are the
  FLOPs the math *requires*: the canonical estimate is ``C ≈ 6·N·D`` — 2 FLOPs/param/token forward
  + 4 back (1 for the input grad, 1 for the weight grad, each a matmul of the same shape as the
  forward) = 6, times N params times D tokens. This is what you would report to compare two
  *models*: it does not reward wasted work.
- **HFU (Hardware FLOPs Utilization)** = achieved *hardware* FLOPs ÷ peak. Hardware FLOPs add the
  work the *implementation* does but the math did not need — chiefly activation **recomputation**
  (gradient checkpointing): a fully-recomputed layer runs its forward twice, so backward pays an
  extra +2N/token. HFU ≥ MFU always; the gap is exactly the rematerialization tax. HFU tells you
  how busy the chip is; MFU tells you how much of that busy-ness was necessary.

The 6ND identity is exact by construction here (``training_flops`` *is* 6·N·D); the PaLM anchor
(46.2% MFU / 57.8% HFU on 6144 TPU-v4 @ 275 TFLOP/s bf16) is reproduced to within the ~0.5pp the
6N approximation drops by omitting the attention term — see ``tests/test_mfu.py``.

**The six MFU killers** (why a real run lands at 40–55%, not 100%) and the log-additive attribution
that splits a measured gap among them live in ``MFU_KILLERS`` / ``mfu_gap_attribution`` below.

Interview question: "You clocked 6.2 s/step on 512 H100s for a 70B model at 4M tokens/step — is
that good, and where's the missing half?" Answer: ``mfu(...)`` gives the number; ``compose_mfu`` /
``mfu_gap_attribution`` name the half you lost (unoverlapped comm, bubble, memory-bound kernels,
tiny per-GPU batch, MoE token-drop, stragglers) as a decomposition that multiplies back to it.
"""

from __future__ import annotations

import math

# FLOP accounting constants (per param, per token). The 2/4 split is the load-bearing part:
# forward is one matmul (2 FLOPs — a multiply + an add per param), backward is two matmuls of the
# same shape (input-grad + weight-grad), so 4. Their sum, 6, is the "6" in 6ND.
FLOPS_FWD_PER_PARAM = 2.0
FLOPS_BWD_PER_PARAM = 4.0
FLOPS_PER_PARAM = FLOPS_FWD_PER_PARAM + FLOPS_BWD_PER_PARAM  # 6

TFLOP = 10**12  # 1 TFLOP/s in FLOP/s

# Vendor spec-sheet DENSE bf16/fp16 tensor-core peaks, FLOP/s. These are published peak numbers
# (data-sheet maxima at the rated clock, *not* measured on this box); the "with-sparsity" figures
# vendors headline are 2x these and are the wrong denominator for a dense-training MFU. Use as the
# `peak_flops_per_device` argument; a real run's achieved FLOP/s is always well under peak.
PEAK_FLOPS_BF16_DENSE: dict[str, float] = {
    "H100_SXM": 989.5 * TFLOP,  # Hopper, 989.5 TFLOP/s dense bf16 (1979 w/ 2:4 sparsity)
    "H200_SXM": 989.5 * TFLOP,  # same compute die as H100; the win is HBM3e capacity/bandwidth
    "A100_SXM": 312.0 * TFLOP,  # Ampere bf16 tensor, dense
    "TPU_v4": 275.0 * TFLOP,  # Google TPU v4 bf16 — the PaLM anchor's hardware
    "B200": 2250.0 * TFLOP,  # Blackwell dense bf16 (data-sheet); FP8/FP4 are higher
}


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _check_nonneg(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def training_flops(
    n_params: int, tokens: int, *, flops_per_param: float = FLOPS_PER_PARAM
) -> float:
    """Model FLOPs for a step: the ``C ≈ 6·N·D`` identity (``flops_per_param·n_params·tokens``).

    This is the numerator's core — the FLOPs the forward+backward *math* requires, independent of
    how efficiently the hardware runs them. ``flops_per_param`` defaults to 6 (=2 fwd + 4 bwd);
    override only to model a variant accounting (e.g. inference-only forward = 2).
    """
    if n_params < 0 or tokens < 0:
        raise ValueError("n_params and tokens must be >= 0")
    _check_positive("flops_per_param", flops_per_param)
    return flops_per_param * n_params * tokens


def hardware_flops(
    n_params: int,
    tokens: int,
    *,
    recompute_fraction: float = 1.0,
    flops_per_param: float = FLOPS_PER_PARAM,
) -> float:
    """Hardware FLOPs for a step: model FLOPs + activation-recompute FLOPs.

    Gradient checkpointing reruns the forward in the backward pass; a *fully* recomputed step
    (``recompute_fraction=1.0``) therefore executes one extra forward = ``+2·N·D`` FLOPs, so
    hardware = ``(6 + 2)·N·D = 8·N·D``. ``recompute_fraction`` in [0, 1] scales the extra forward:
    0 = no checkpointing (hardware == model), 1 = every layer recomputed, and selective/partial
    schemes sit between (PaLM's 46.2/57.8 split implies ≈0.75). HFU numerator, always ≥ the MFU
    numerator.
    """
    if not 0.0 <= recompute_fraction <= 1.0:
        raise ValueError(f"recompute_fraction must be in [0, 1], got {recompute_fraction}")
    model = training_flops(n_params, tokens, flops_per_param=flops_per_param)
    extra_forward = recompute_fraction * FLOPS_FWD_PER_PARAM * n_params * tokens
    return model + extra_forward


def peak_flops_aggregate(n_devices: int, peak_flops_per_device: float) -> float:
    """Cluster peak FLOP/s = ``n_devices · peak_flops_per_device`` — the MFU/HFU denominator."""
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    _check_positive("peak_flops_per_device", peak_flops_per_device)
    return n_devices * peak_flops_per_device


def mfu(
    n_params: int,
    tokens: int,
    step_time_s: float,
    n_devices: int,
    peak_flops_per_device: float,
    *,
    flops_per_param: float = FLOPS_PER_PARAM,
    extra_flops: float = 0.0,
) -> float:
    """Model FLOPs Utilization: achieved model FLOP/s ÷ cluster peak FLOP/s.

    ``(training_flops + extra_flops) / step_time_s / (n_devices · peak_flops_per_device)``. The
    default ``extra_flops=0`` gives the pure 6ND estimate (what almost everyone reports);
    ``extra_flops`` lets you fold in the attention term (``≈ 12·L·s·D`` per step, or the exact
    ``72·B·s·L·d²·s/(6d)`` correction) when you want to close the last point of the PaLM anchor.
    Dimensionless, in (0, 1]; > 1 means your peak or FLOP count is wrong.
    """
    _check_positive("step_time_s", step_time_s)
    _check_nonneg("extra_flops", extra_flops)
    achieved = (
        training_flops(n_params, tokens, flops_per_param=flops_per_param) + extra_flops
    ) / step_time_s
    return achieved / peak_flops_aggregate(n_devices, peak_flops_per_device)


def hfu(
    n_params: int,
    tokens: int,
    step_time_s: float,
    n_devices: int,
    peak_flops_per_device: float,
    *,
    recompute_fraction: float = 1.0,
    flops_per_param: float = FLOPS_PER_PARAM,
    extra_flops: float = 0.0,
) -> float:
    """Hardware FLOPs Utilization: achieved *hardware* FLOP/s ÷ cluster peak FLOP/s.

    Same denominator as ``mfu``; the numerator adds the recomputation FLOPs (see ``hardware_flops``)
    plus the same optional ``extra_flops``. HFU ≥ MFU for the same run, with equality iff
    ``recompute_fraction == 0``.
    """
    _check_positive("step_time_s", step_time_s)
    _check_nonneg("extra_flops", extra_flops)
    achieved = (
        hardware_flops(
            n_params, tokens, recompute_fraction=recompute_fraction, flops_per_param=flops_per_param
        )
        + extra_flops
    ) / step_time_s
    return achieved / peak_flops_aggregate(n_devices, peak_flops_per_device)


def hfu_over_mfu(
    recompute_fraction: float = 1.0, *, flops_per_param: float = FLOPS_PER_PARAM
) -> float:
    """Analytic HFU/MFU ratio at ``extra_flops=0``: ``(fpp + 2·recompute_fraction) / fpp``.

    With the default 6 FLOPs/param this is ``(6 + 2r)/6`` — 1.0 at r=0, 4/3 at full recompute.
    The clean closed form the numeric ``hfu``/``mfu`` must agree with; ``r`` is the *only* knob
    that separates the two utilizations.
    """
    if not 0.0 <= recompute_fraction <= 1.0:
        raise ValueError(f"recompute_fraction must be in [0, 1], got {recompute_fraction}")
    _check_positive("flops_per_param", flops_per_param)
    return (flops_per_param + FLOPS_FWD_PER_PARAM * recompute_fraction) / flops_per_param


# --- The six MFU killers -----------------------------------------------------------------------
#
# Why a real run lands at 40-55%, not 100%. Each is an independent multiplicative efficiency
# η ∈ (0, 1] (fraction of ideal FLOP throughput NOT lost to it); realized MFU = ideal · ∏ η.
# The tuple order is the canonical A6 checklist order.
MFU_KILLERS: tuple[str, ...] = (
    "unoverlapped_comm",  # all-reduce / all-gather on the critical path, not hidden behind compute
    "pipeline_bubble",  # PP warmup/drain idle — the (p-1)/(m+p-1) fraction of a 1F1B schedule
    "memory_bound_kernels",  # norms, softmax, elementwise, attention @ low arithmetic intensity
    "small_per_gpu_batch",  # micro-batch too small to fill the tensor cores (launch/occupancy limited)
    "moe_imbalance",  # expert load skew + capacity-factor token-drop wasting the padded slots
    "stragglers",  # the slowest rank sets the barrier: bad node, thermal throttle, network hotspot
)


def compose_mfu(ideal_mfu: float, efficiencies: dict[str, float]) -> float:
    """Model realized MFU from the ideal ceiling and per-killer efficiencies: ``ideal · ∏ η_i``.

    ``efficiencies`` maps a subset of ``MFU_KILLERS`` to η ∈ (0, 1] (a missing killer = 1.0, no
    loss). The killers compose multiplicatively because each independently scales the *achieved
    FLOP/s*: a 0.9-comm × 0.8-bubble run does 0.72 of the work. This is the forward model; invert
    it with ``mfu_gap_attribution``.
    """
    _check_positive("ideal_mfu", ideal_mfu)
    product = 1.0
    for name, eta in efficiencies.items():
        if name not in MFU_KILLERS:
            raise ValueError(f"unknown MFU killer {name!r}; expected one of {MFU_KILLERS}")
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"efficiency for {name!r} must be in (0, 1], got {eta}")
        product *= eta
    return ideal_mfu * product


def mfu_gap_attribution(efficiencies: dict[str, float]) -> dict[str, float]:
    """Split the total MFU loss among the killers, log-additively — shares sum to exactly 1.

    The multiplicative gap ``E = ∏ η_i`` becomes additive in log space: ``-ln E = Σ_i(-ln η_i)``,
    so killer *i*'s blame is ``(-ln η_i) / (-ln E)``. Exact (no Shapley approximation) *because*
    the model is a pure product; a lossless killer (η=1) contributes 0. Returns a share in [0, 1]
    per killer present in ``efficiencies`` (all 0 if every η is 1, i.e. no gap to attribute).
    """
    log_losses: dict[str, float] = {}
    for name, eta in efficiencies.items():
        if name not in MFU_KILLERS:
            raise ValueError(f"unknown MFU killer {name!r}; expected one of {MFU_KILLERS}")
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"efficiency for {name!r} must be in (0, 1], got {eta}")
        log_losses[name] = -math.log(eta)
    total = sum(log_losses.values())
    if total == 0.0:
        return {name: 0.0 for name in efficiencies}
    return {name: loss / total for name, loss in log_losses.items()}


def unexplained_factor(
    ideal_mfu: float, measured_mfu: float, efficiencies: dict[str, float]
) -> float:
    """Residual ``measured_mfu / compose_mfu(ideal, efficiencies)`` — 1.0 iff the six killers fully
    explain the gap.

    > 1 means the killers over-account (their η are too pessimistic / a killer is double-counted);
    < 1 means an *un-modeled* loss remains (there is a seventh thing eating FLOPs). The honest
    close-the-loop check: after attributing to the six named killers, how much of the measured gap
    is still unexplained.
    """
    _check_positive("measured_mfu", measured_mfu)
    return measured_mfu / compose_mfu(ideal_mfu, efficiencies)
