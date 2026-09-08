"""T1/T-R3 — induced failures and the detectors that catch them.

Three faults, three detections, and — the part that makes them worth anything — a proof for each
that it stays SILENT on a clean run. A detector with no false-negative rate because it always
fires detects nothing; every detector here is exercised in both directions.

| fault (``inject``)                          | detection                                   |
|---------------------------------------------|---------------------------------------------|
| :class:`~.inject.RankKill` — one rank ``os._exit``s mid-collective | launcher sees the death; :func:`~.capsule.verify_bit_exact` proves the resume is bitwise identical to the uninterrupted run |
| :class:`~.inject.NaNBatch` — a batch's first float tensor becomes NaN | :class:`~.spike.LossSpikeDetector` skips the step before the optimizer sees the NaN, and escalates rather than livelocking |
| :func:`~.inject.flip_bit_` — one bit of one gradient element on one rank | :class:`~.gradnorm.CrossRankGradCheck` fires and names the rank |

Nothing here has a tolerance. Every comparison is exact — ``torch.equal`` on a byte view, a
bitwise digest, ``math.isfinite`` — because a resume that is merely *close* passes a careless
test forever. The one free operating point that cannot be avoided (the spike detector's z) is a
REQUIRED constructor argument with no default, so no threshold is chosen inside this package.

:mod:`~.loop` is the instrument's training loop, not a replacement for
:func:`scratch_llm.train.train`: ``train()`` documents that it always begins at step 0 with a
fresh warmup, so it cannot express a resume at all. The loop here reuses ``train()``'s own
``get_batch`` / ``cross_entropy`` / ``gradient_clipping`` / ``cosine_lr`` and is pinned to it by
``tests/training/test_t1_t_r3.py::test_the_instrument_loop_reproduces_train_step_for_step``.
"""

from __future__ import annotations

from scratch_llm.faults.capsule import (
    ResumeReport,
    TrainSnapshot,
    as_train_py_checkpoint,
    bitwise_equal,
    capture,
    restore,
    verify_bit_exact,
    without_rng,
)
from scratch_llm.faults.gradnorm import CrossRankGradCheck, GradCheckReport
from scratch_llm.faults.inject import (
    BitFlip,
    NaNBatch,
    RankKill,
    describe_bit,
    flip_bit_,
    flip_grad_bit_,
)
from scratch_llm.faults.loop import LoopResult, StepHooks, StepRecord, pin_determinism, run_steps
from scratch_llm.faults.spike import LossSpikeDetector, SpikeAction, SpikeEvent

__all__ = [
    "BitFlip",
    "CrossRankGradCheck",
    "GradCheckReport",
    "LoopResult",
    "LossSpikeDetector",
    "NaNBatch",
    "RankKill",
    "ResumeReport",
    "SpikeAction",
    "SpikeEvent",
    "StepHooks",
    "StepRecord",
    "TrainSnapshot",
    "as_train_py_checkpoint",
    "bitwise_equal",
    "capture",
    "describe_bit",
    "flip_bit_",
    "flip_grad_bit_",
    "pin_determinism",
    "restore",
    "run_steps",
    "verify_bit_exact",
    "without_rng",
]
