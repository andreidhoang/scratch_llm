"""The loss-spike detector and its skip/restart policy.

Two channels, and they are deliberately not the same kind of thing:

* **non-finite** — ``math.isfinite`` on the (world-reduced) loss. Exact, no threshold, no false
  positives. This is the NaN-batch fault's detection, and it is the only one that matters for
  correctness: a NaN loss produces NaN gradients, the optimizer writes NaN into every parameter it
  touches, and every subsequent step is garbage. The step must be skipped BEFORE ``optimizer.step``.
* **magnitude** — a robust z against a rolling median/MAD baseline. This one has an operating
  point, and there is no honest default for it: too tight and a warm-up bump halts the run, too
  loose and a slow divergence is skipped past. ``spike_z`` is therefore a REQUIRED argument.

**The livelock is a failure mode of the fix, not of the fault.** A policy that skips any batch
above baseline, and only admits non-skipped losses into the baseline, will meet a legitimately
harder phase of the corpus — a domain switch, a longer-context shard — and skip every batch
forever: the run burns money, the step counter advances, and no gradient is ever applied. So:

* a run of CONSECUTIVE **finite** spikes longer than ``max_consecutive_skips`` is read as a regime
  shift, not a fault. The pending losses are admitted into the baseline (``_adapt``), one RESTART
  is emitted so the caller can rewind to the last good checkpoint with the new baseline, and
  training continues. Bounded, and it terminates.
* a run of consecutive **non-finite** losses never adapts. NaN is never a legitimate regime. It
  escalates SKIP -> RESTART -> HALT and stops the run within
  ``(max_restarts + 1) * (max_consecutive_skips + 1)`` steps.

Both directions are tested: it fires on the injected NaN batch, it is silent across a clean run,
it does not livelock on a hard phase, and it never adapts its way past an all-NaN stream.

The rolling window is trajectory state: a resume that drops it makes different skip decisions than
the run it claims to continue. :meth:`LossSpikeDetector.state_dict` exists so the window rides in
the resume capsule's ``extra`` alongside the optimizer moments.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SpikeAction(Enum):
    """What the loop should do with this step."""

    CONTINUE = "continue"  # normal: clip, step, advance
    SKIP = "skip"  # drop the gradients, do NOT step, advance to the next batch
    RESTART = "restart"  # rewind to the last good capsule and continue from there
    HALT = "halt"  # stop the run — the escalation ladder ran out


@dataclass(frozen=True)
class SpikeEvent:
    """One firing. ``reason`` is ``"nonfinite"`` or ``"magnitude"``."""

    step: int
    loss: float
    reason: str
    action: SpikeAction
    baseline: float
    scale: float

    def __str__(self) -> str:
        return (
            f"step {self.step}: {self.reason} loss={self.loss!r} "
            f"baseline={self.baseline:.4f} scale={self.scale:.4g} -> {self.action.value}"
        )


class LossSpikeDetector:
    """Rolling-median spike detector with a bounded skip/restart/halt ladder.

    Args:
        spike_z: how many robust sigmas above the rolling median counts as a spike. No default —
            the operating point is a design decision, not a library constant.
        window: rolling-baseline length in steps.
        min_observations: steps of history before the magnitude channel arms. Below this the
            baseline is noise and every step would look like a spike; the non-finite channel is
            live from step 0 regardless.
        max_consecutive_skips: how many spikes in a row before escalating.
        max_restarts: how many escalations before HALT.
    """

    def __init__(
        self,
        *,
        spike_z: float,
        window: int,
        min_observations: int,
        max_consecutive_skips: int,
        max_restarts: int,
    ) -> None:
        if window < 2 or min_observations < 2:
            raise ValueError("window and min_observations must be >= 2 for a median/MAD baseline")
        if max_consecutive_skips < 1 or max_restarts < 0:
            raise ValueError("max_consecutive_skips >= 1 and max_restarts >= 0")
        self.spike_z = float(spike_z)
        self.window = int(window)
        self.min_observations = int(min_observations)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.max_restarts = int(max_restarts)
        self._history: deque[float] = deque(maxlen=self.window)
        self._pending: list[float] = []  # finite losses skipped since the last accepted step
        self._consecutive = 0
        self._restarts = 0
        self.events: list[SpikeEvent] = []
        self.halted = False

    # -- baseline ------------------------------------------------------------------------------

    @property
    def baseline(self) -> float:
        """Rolling median, or NaN while the magnitude channel is disarmed."""
        if len(self._history) < self.min_observations:
            return float("nan")
        return statistics.median(self._history)

    @property
    def scale(self) -> float:
        """MAD scaled to a sigma equivalent (x1.4826). A constant stream gives 0, which makes any
        deviation a spike — correct, and exact: there is nothing to be uncertain about."""
        if len(self._history) < self.min_observations:
            return float("nan")
        med = statistics.median(self._history)
        return 1.4826 * statistics.median([abs(x - med) for x in self._history])

    def _adapt(self) -> None:
        """Admit the skipped-but-finite losses into the baseline. This is the anti-livelock move:
        a persistently harder phase becomes the new normal instead of being fought forever."""
        for value in self._pending:
            self._history.append(value)
        self._pending.clear()

    # -- the detector --------------------------------------------------------------------------

    def observe(self, step: int, loss: float) -> SpikeAction:
        """Classify one step's (world-reduced) loss and return what the loop should do.

        The loss MUST already be reduced across ranks: if rank 3 sees NaN and the others do not,
        an un-reduced decision skips on one rank and steps on the others, which desyncs the
        replicas — a worse failure than the one being detected.
        """
        if self.halted:
            return SpikeAction.HALT

        finite = math.isfinite(loss)
        base, scl = self.baseline, self.scale
        armed = len(self._history) >= self.min_observations
        magnitude_spike = armed and finite and loss > base + self.spike_z * scl

        if finite and not magnitude_spike:
            self._history.append(loss)
            self._pending.clear()
            self._consecutive = 0
            return SpikeAction.CONTINUE

        self._consecutive += 1
        if finite:
            self._pending.append(loss)

        if self._consecutive <= self.max_consecutive_skips:
            action = SpikeAction.SKIP
        elif finite:
            # A finite regime shift: adapt the baseline, rewind once, keep training. Bounded.
            self._adapt()
            self._consecutive = 0
            self._restarts += 1
            action = (
                SpikeAction.RESTART if self._restarts <= self.max_restarts else SpikeAction.HALT
            )
        else:
            # NaN never becomes the new normal.
            self._restarts += 1
            self._consecutive = 0
            action = (
                SpikeAction.RESTART if self._restarts <= self.max_restarts else SpikeAction.HALT
            )

        if action is SpikeAction.HALT:
            self.halted = True
        self.events.append(
            SpikeEvent(
                step=step,
                loss=loss,
                reason="nonfinite" if not finite else "magnitude",
                action=action,
                baseline=base,
                scale=scl,
            )
        )
        return action

    @property
    def fired(self) -> bool:
        """True iff the detector has flagged at least one step. Silence on a clean run is the
        other half of the proof, so this is what a clean-run assertion reads."""
        return bool(self.events)

    def skip_count(self) -> int:
        return sum(1 for e in self.events if e.action is SpikeAction.SKIP)

    # -- capsule interop -----------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Trajectory state for the resume capsule. Without it a resumed run has an empty baseline
        and re-arms from scratch, so it makes DIFFERENT skip decisions than the run it continues."""
        return {
            "history": list(self._history),
            "pending": list(self._pending),
            "consecutive": self._consecutive,
            "restarts": self._restarts,
            "halted": self.halted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._history = deque(state["history"], maxlen=self.window)
        self._pending = list(state["pending"])
        self._consecutive = int(state["consecutive"])
        self._restarts = int(state["restarts"])
        self.halted = bool(state["halted"])


__all__ = ["LossSpikeDetector", "SpikeAction", "SpikeEvent"]
