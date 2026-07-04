"""Pipeline-parallel schedules + bubble accounting — GPipe vs 1F1B, on paper (no GPUs).

A6 systems. Pipeline parallelism splits a model *depth-wise* across ``p`` stages (devices) and
feeds it ``m`` microbatches so the stages work concurrently — but a pipeline must **fill** and
**drain**, and during fill/drain some stages sit idle. That idle time is the *bubble*. This module
owns the two things you can compute before renting a multi-GPU node: the **schedule** (the ordered
``(stage, microbatch, F/B)`` op stream each device runs) and its **accounting** (bubble fraction +
peak activation memory) from a fake per-op timing model.

The one identity everything rests on — with equal forward/backward op times, the bubble fraction is

    bubble_fraction = (p - 1) / m                                      (GPipe *and* 1F1B)

Derivation: each device does ``m`` forwards + ``m`` backwards, so its ideal busy time is
``m·(t_f+t_b)``. The pipeline makespan is ``fill + steady + drain`` — the forward wave takes
``(m+p-1)`` op-times to clear all stages and the backward wave another ``(m+p-1)``, so
``makespan = 2·(m+p-1)·t`` at ``t_f=t_b=t``. Bubble ``= makespan − busy = 2(p-1)t``; divide by the
busy time ``2mt`` and the ``t`` and the factor 2 cancel: ``(p-1)/m``. More microbatches ⇒ a thinner
bubble; deeper pipelines ⇒ a fatter one.

**GPipe and 1F1B have the *identical* bubble** — 1F1B buys nothing on throughput. What 1F1B buys is
**memory**: GPipe runs all ``m`` forwards before any backward, so every stage must stash all ``m``
microbatches' activations at once (peak ``= m``); 1F1B interleaves one-forward-one-backward in
steady state so a stage frees an activation as fast as it makes new ones, capping live activations
at the pipeline depth (peak ``= min(p, m)``, independent of ``m``). That is why 1F1B is the default:
same bubble, activation memory bounded by ``p`` instead of ``m``.

Falsifiable invariant (``tests/test_pipeline_schedule.py``): (1) both emitted schedules are *valid* —
every dependency respected (microbatch ``i``'s backward on stage ``s`` needs its own forward on ``s``
and, for ``s<p-1``, its backward on ``s+1``; its forward on ``s`` needs its forward on ``s-1``), no
deadlock; (2) the bubble fraction measured by a DAG-longest-path simulation of a fake timing model
equals the analytic ``(p-1)/m`` to fp tolerance across many ``(p,m)``; (3) 1F1B peak activation
memory ``= min(p,m) ≤ m =`` GPipe peak, with equality only when ``m ≤ p``. Kill: any mismatch ⇒ a
wrong warmup count, a dropped dependency edge, or an off-by-one in the fill/drain.

Interview question: "You pipeline a model over ``p`` stages with ``m`` microbatches — what's your
bubble, and what does switching GPipe→1F1B change?" (Answer: bubble ``(p-1)/m`` for *both*; 1F1B
leaves throughput alone and drops peak activation memory from ``O(m)`` to ``O(p)`` — pick ``m ≫ p``
to make the bubble small, then 1F1B to afford the activations. Interleaving ``v`` chunks per device
divides the bubble by ``v`` at the cost of ``v×`` the pipeline communication.)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

FORWARD = "F"
BACKWARD = "B"


@dataclass(frozen=True)
class Op:
    """One scheduled operation: the ``kind`` (F/B) pass of ``microbatch`` on virtual ``stage``.

    ``stage`` is the *virtual* pipeline-stage index (``0..V-1``); for a non-interleaved schedule it
    equals the physical device index. Forward flows ``stage 0 → V-1``; backward flows ``V-1 → 0``.
    """

    stage: int
    microbatch: int
    kind: str  # FORWARD ("F") or BACKWARD ("B")

    def __str__(self) -> str:
        return f"{self.kind}{self.microbatch}@s{self.stage}"


@dataclass(frozen=True)
class Schedule:
    """A full pipeline schedule: per-device ordered op streams.

    ``device_ops[d]`` is the sequence of ops physical device ``d`` executes, in order. For the
    non-interleaved schedules here ``num_virtual_stages == num_stages`` and device ``d`` owns
    virtual stage ``d``.
    """

    num_stages: int  # p — physical pipeline stages / devices
    num_microbatches: int  # m
    num_virtual_stages: int  # V — == p unless interleaved
    device_ops: tuple[tuple[Op, ...], ...]  # device_ops[d] in execution order

    def all_ops(self) -> list[Op]:
        return [op for dev in self.device_ops for op in dev]


@dataclass(frozen=True)
class ScheduleStats:
    """Accounting for a simulated schedule under a fake per-op timing model."""

    makespan: float  # wall time of the whole pipeline (end of last op)
    busy_per_device: float  # time a device spends computing (never idle) = m·(t_f+t_b)
    bubble_time: float  # makespan − busy_per_device (idle time on the critical device)
    bubble_fraction: float  # bubble_time / busy_per_device  → (p-1)/m at t_f=t_b
    peak_activations: int  # max live activations on any single device (stashed fwd, not yet bwd)
    peak_activations_per_device: tuple[int, ...]


# --------------------------------------------------------------------------------------------------
# Schedule generators
# --------------------------------------------------------------------------------------------------


def _check(p: int, m: int) -> None:
    if p < 1:
        raise ValueError(f"num_stages p must be >= 1, got {p}")
    if m < 1:
        raise ValueError(f"num_microbatches m must be >= 1, got {m}")


def gpipe_schedule(p: int, m: int) -> Schedule:
    """GPipe: every stage runs **all m forwards, then all m backwards** (fill-drain once).

    Backwards are emitted last-in-first-out (``m-1 … 0``) — a stage backwards the microbatch it
    forwarded most recently first, the canonical GPipe order that frees activations as it drains.
    Because no backward starts until every forward is done, all ``m`` activations are live at the
    peak on stage 0 (see ``gpipe_peak_activations``).
    """
    _check(p, m)
    device_ops = tuple(
        tuple(
            [Op(s, i, FORWARD) for i in range(m)] + [Op(s, i, BACKWARD) for i in reversed(range(m))]
        )
        for s in range(p)
    )
    return Schedule(p, m, p, device_ops)


def one_f_one_b_schedule(p: int, m: int) -> Schedule:
    """1F1B: warmup forwards, then steady-state one-forward-one-backward, then cooldown backwards.

    Stage ``s`` runs ``w = min(p-1-s, m)`` warmup forwards (deeper stages warm up less — the last
    stage starts backpropagating immediately), then alternates ``F,B`` for the remaining ``m-w``
    microbatches, then drains its ``w`` outstanding backwards. Forwards and backwards each advance
    microbatch index ``0…m-1`` in order. Total ops per device: ``m`` F + ``m`` B, same as GPipe —
    the bubble is identical; only the *interleaving* differs, which is what bounds the memory.
    """
    _check(p, m)
    device_ops: list[tuple[Op, ...]] = []
    for s in range(p):
        w = max(0, min(p - 1 - s, m))
        ops: list[Op] = []
        f = b = 0
        for _ in range(w):  # warmup
            ops.append(Op(s, f, FORWARD))
            f += 1
        for _ in range(m - w):  # steady state 1F1B
            ops.append(Op(s, f, FORWARD))
            f += 1
            ops.append(Op(s, b, BACKWARD))
            b += 1
        while b < m:  # cooldown
            ops.append(Op(s, b, BACKWARD))
            b += 1
        device_ops.append(tuple(ops))
    return Schedule(p, m, p, tuple(device_ops))


# --------------------------------------------------------------------------------------------------
# Dependency model, validity, and the timing simulator
# --------------------------------------------------------------------------------------------------


def _predecessors(sched: Schedule) -> dict[Op, list[Op]]:
    """Build the dependency edges of the schedule DAG.

    Three edge families:
      * device order — each op depends on the previous op *on the same device* (a device runs one
        op at a time);
      * forward flow — ``F(i,s)`` needs ``F(i,s-1)`` (activations arrive from the earlier stage);
      * backward flow — ``B(i,s)`` needs its own ``F(i,s)`` (the stashed activation) and, for
        ``s < V-1``, ``B(i,s+1)`` (the incoming gradient from the later stage).
    """
    index: dict[tuple[int, int, str], Op] = {}
    for dev in sched.device_ops:
        for op in dev:
            key = (op.stage, op.microbatch, op.kind)
            if key in index:
                raise ValueError(f"duplicate op {op} in schedule")
            index[key] = op

    preds: dict[Op, list[Op]] = {op: [] for op in index.values()}
    for dev in sched.device_ops:
        for k in range(1, len(dev)):
            preds[dev[k]].append(dev[k - 1])
    for op in index.values():
        if op.kind == FORWARD:
            if op.stage > 0:
                dep = index.get((op.stage - 1, op.microbatch, FORWARD))
                if dep is None:
                    raise ValueError(
                        f"{op} is missing its forward predecessor on stage {op.stage - 1}"
                    )
                preds[op].append(dep)
        else:  # BACKWARD
            own_fwd = index.get((op.stage, op.microbatch, FORWARD))
            if own_fwd is None:
                raise ValueError(f"{op} is missing its own forward activation on stage {op.stage}")
            preds[op].append(own_fwd)
            nxt = index.get((op.stage + 1, op.microbatch, BACKWARD))
            if nxt is not None:
                preds[op].append(nxt)
    return preds


def simulate(sched: Schedule, t_f: float = 1.0, t_b: float = 1.0) -> ScheduleStats:
    """Time the schedule under a fake model: forward op = ``t_f`` s, backward op = ``t_b`` s.

    Computes each op's end time as the DAG longest path (topological / Kahn) — an op starts the
    instant all its predecessors (device-order + data dependencies) have finished. This is the
    optimal timing *for the emitted order*; a wrong order shows up as a bigger makespan (fatter
    bubble) or, if a cycle exists (a self-blocking order = deadlock), a ``ValueError``.

    Returns makespan, per-device busy time, and the bubble fraction ``(makespan−busy)/busy`` — the
    measured quantity the tests pin to the analytic ``(p-1)/m``. Also reports peak live activations
    per device (walk each device's op stream: ``+1`` per forward, ``−1`` per backward, track the max).
    """
    if t_f <= 0 or t_b <= 0:
        raise ValueError(f"op durations must be > 0, got t_f={t_f}, t_b={t_b}")
    dur = {FORWARD: t_f, BACKWARD: t_b}
    preds = _predecessors(sched)
    ops = list(preds)

    succ: dict[Op, list[Op]] = {op: [] for op in ops}
    indeg: dict[Op, int] = {op: 0 for op in ops}
    for op, plist in preds.items():
        for pr in plist:
            succ[pr].append(op)
            indeg[op] += 1

    end: dict[Op, float] = {
        op: 0.0 for op in ops
    }  # holds running max-of-predecessor-ends (= start)
    queue = deque(op for op in ops if indeg[op] == 0)
    processed = 0
    while queue:
        op = queue.popleft()
        processed += 1
        end[op] += dur[op.kind]  # start + duration = end
        for nb in succ[op]:
            if end[op] > end[nb]:
                end[nb] = end[op]
            indeg[nb] -= 1
            if indeg[nb] == 0:
                queue.append(nb)
    if processed != len(ops):
        raise ValueError("schedule has a cyclic dependency (deadlock): not a valid pipeline order")

    makespan = max(end.values()) if end else 0.0

    peaks: list[int] = []
    for dev in sched.device_ops:
        live = 0
        peak = 0
        for op in dev:
            live += 1 if op.kind == FORWARD else -1
            peak = max(peak, live)
        peaks.append(peak)

    busy = sched.num_microbatches * (t_f + t_b) * (sched.num_virtual_stages // sched.num_stages)
    bubble_time = makespan - busy
    bubble_fraction = bubble_time / busy if busy > 0 else 0.0
    return ScheduleStats(
        makespan=makespan,
        busy_per_device=busy,
        bubble_time=bubble_time,
        bubble_fraction=bubble_fraction,
        peak_activations=max(peaks) if peaks else 0,
        peak_activations_per_device=tuple(peaks),
    )


def validate_schedule(sched: Schedule) -> None:
    """Raise ``ValueError`` unless the schedule is complete and dependency-respecting.

    Complete = every ``(virtual stage, microbatch)`` pair has exactly one forward and one backward.
    Dependency-respecting = the DAG has no cycle (``simulate`` raises on a deadlock/self-block).
    The oracle: a schedule the tests accept is one this function does not reject.
    """
    v, m = sched.num_virtual_stages, sched.num_microbatches
    seen = {(op.stage, op.microbatch, op.kind) for op in sched.all_ops()}
    expected = {(s, i, k) for s in range(v) for i in range(m) for k in (FORWARD, BACKWARD)}
    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        raise ValueError(
            f"incomplete schedule; missing={sorted(missing)[:4]} extra={sorted(extra)[:4]}"
        )
    simulate(sched)  # raises on a cyclic (deadlocking) order


# --------------------------------------------------------------------------------------------------
# Closed-form accounting (the analytic the simulation is checked against)
# --------------------------------------------------------------------------------------------------


def gpipe_bubble_fraction(p: int, m: int) -> float:
    """GPipe bubble fraction (idle / busy) at equal op times: ``(p-1)/m``."""
    _check(p, m)
    return (p - 1) / m


def one_f_one_b_bubble_fraction(p: int, m: int) -> float:
    """1F1B bubble fraction: ``(p-1)/m`` — identical to GPipe. 1F1B saves memory, not bubble."""
    _check(p, m)
    return (p - 1) / m


def interleaved_1f1b_bubble_fraction(p: int, m: int, v: int) -> float:
    """Interleaved-1F1B (``v`` model chunks per device) bubble fraction: ``(p-1)/(v·m)``.

    Analytic accounting only — the exact per-op *ordering* that realizes this bound is Megatron's
    interleaved schedule, whose measured verification is gated on a real multi-GPU node (runbook),
    not simulated here. Derivation: slicing each device's layers into ``v`` interleaved chunks makes
    every pipeline op ``1/v`` as long, so the fill/drain bubble ``(p-1)·t`` shrinks by ``v`` against
    the unchanged ``m·(t_f+t_b)`` of useful work ⇒ bubble ``/v``. The cost is ``v×`` the number of
    stage-to-stage activation transfers (more pipeline communication), and peak activation memory
    rises back toward ``v·p`` — the standard bubble↔memory/comms trade.
    """
    _check(p, m)
    if v < 1:
        raise ValueError(f"virtual chunks v must be >= 1, got {v}")
    return (p - 1) / (v * m)


def gpipe_peak_activations(p: int, m: int) -> int:
    """GPipe peak live activations on a stage: ``m`` — all microbatches stashed before any backward."""
    _check(p, m)
    return m


def one_f_one_b_peak_activations(p: int, m: int) -> int:
    """1F1B peak live activations on a stage: ``min(p, m)`` — bounded by pipeline depth, not ``m``.

    This is the whole point of 1F1B: stage 0 warms up ``min(p-1, m)`` forwards then holds at most
    one more in flight, so its live-activation count never exceeds ``p`` however large ``m`` grows.
    """
    _check(p, m)
    return min(p, m)
