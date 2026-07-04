"""Pipeline schedule + bubble accounting: the emitted GPipe / 1F1B op streams must be valid
(every forward/backward dependency respected, no deadlock), their simulated bubble fraction must
equal the analytic ``(p-1)/m`` to fp tolerance across many ``(p,m)``, and 1F1B must cap peak
activation memory at ``min(p,m) <= m`` (its whole reason to exist). Pure CPU, deterministic."""

from __future__ import annotations

import pytest

from scratch_llm.utils.pipeline_schedule import (
    BACKWARD,
    FORWARD,
    Op,
    Schedule,
    gpipe_bubble_fraction,
    gpipe_peak_activations,
    gpipe_schedule,
    interleaved_1f1b_bubble_fraction,
    one_f_one_b_bubble_fraction,
    one_f_one_b_peak_activations,
    one_f_one_b_schedule,
    simulate,
    validate_schedule,
)

CASES = [(2, 4), (4, 8), (3, 6), (8, 16), (4, 4), (2, 1), (4, 2), (1, 5), (5, 20)]


# ---- op-count / structural sanity -------------------------------------------------------------


@pytest.mark.parametrize("p,m", CASES)
@pytest.mark.parametrize("gen", [gpipe_schedule, one_f_one_b_schedule])
def test_schedule_is_valid_and_complete(p, m, gen) -> None:
    sched = gen(p, m)
    validate_schedule(sched)  # raises on incomplete or deadlocking schedule
    # exactly m forwards + m backwards per device
    for dev in sched.device_ops:
        assert sum(o.kind == FORWARD for o in dev) == m
        assert sum(o.kind == BACKWARD for o in dev) == m


@pytest.mark.parametrize("p,m", CASES)
def test_one_f_one_b_forward_before_own_backward(p, m) -> None:
    # on every device, microbatch i's forward must be emitted before its backward (activation dep)
    for dev in one_f_one_b_schedule(p, m).device_ops:
        seen_f = set()
        for op in dev:
            if op.kind == FORWARD:
                seen_f.add(op.microbatch)
            else:
                assert op.microbatch in seen_f, f"B{op.microbatch}@s{op.stage} before its forward"


# ---- the bubble oracle: measured == analytic (p-1)/m ------------------------------------------


@pytest.mark.parametrize("p,m", CASES)
def test_gpipe_bubble_matches_analytic(p, m) -> None:
    stats = simulate(gpipe_schedule(p, m))
    assert stats.bubble_fraction == pytest.approx((p - 1) / m)
    assert stats.bubble_fraction == pytest.approx(gpipe_bubble_fraction(p, m))
    # makespan closed form at t_f=t_b=1: forward wave (m+p-1) + backward wave (m+p-1)
    assert stats.makespan == pytest.approx(2 * (m + p - 1))


@pytest.mark.parametrize("p,m", CASES)
def test_one_f_one_b_bubble_matches_analytic(p, m) -> None:
    stats = simulate(one_f_one_b_schedule(p, m))
    assert stats.bubble_fraction == pytest.approx((p - 1) / m)
    assert stats.bubble_fraction == pytest.approx(one_f_one_b_bubble_fraction(p, m))
    assert stats.makespan == pytest.approx(2 * (m + p - 1))


@pytest.mark.parametrize("p,m", CASES)
def test_gpipe_and_1f1b_have_identical_bubble(p, m) -> None:
    # the headline fact: 1F1B saves no throughput over GPipe — same makespan, same bubble
    g = simulate(gpipe_schedule(p, m))
    f = simulate(one_f_one_b_schedule(p, m))
    assert g.bubble_fraction == pytest.approx(f.bubble_fraction)
    assert g.makespan == pytest.approx(f.makespan)


def test_bubble_fraction_is_time_scale_invariant() -> None:
    # (p-1)/m derivation cancels the op duration: doubling both t_f and t_b leaves the fraction fixed
    p, m = 4, 8
    a = simulate(one_f_one_b_schedule(p, m), t_f=1.0, t_b=1.0).bubble_fraction
    b = simulate(one_f_one_b_schedule(p, m), t_f=3.5, t_b=3.5).bubble_fraction
    assert a == pytest.approx(b) == pytest.approx((p - 1) / m)


# ---- the memory advantage: 1F1B peak = min(p,m), bounded by depth not m -----------------------


@pytest.mark.parametrize("p,m", CASES)
def test_peak_activation_advantage(p, m) -> None:
    g = simulate(gpipe_schedule(p, m))
    f = simulate(one_f_one_b_schedule(p, m))
    assert g.peak_activations == m == gpipe_peak_activations(p, m)
    assert f.peak_activations == min(p, m) == one_f_one_b_peak_activations(p, m)
    assert f.peak_activations <= g.peak_activations  # 1F1B never worse
    # strictly better whenever there are more microbatches than pipeline stages
    if m > p:
        assert f.peak_activations < g.peak_activations


def test_one_f_one_b_peak_independent_of_m() -> None:
    # the load-bearing claim: grow m 10x at fixed p, 1F1B peak stays pinned at p
    p = 4
    peaks = {
        simulate(one_f_one_b_schedule(p, m)).peak_activations for m in (p, 2 * p, 10 * p, 50 * p)
    }
    assert peaks == {p}


# ---- interleaved analytic (bubble / v) --------------------------------------------------------


@pytest.mark.parametrize("p,m", [(4, 8), (8, 16), (2, 4)])
@pytest.mark.parametrize("v", [1, 2, 4])
def test_interleaved_divides_bubble_by_v(p, m, v) -> None:
    assert interleaved_1f1b_bubble_fraction(p, m, v) == pytest.approx(
        one_f_one_b_bubble_fraction(p, m) / v
    )


# ---- the validity oracle has teeth ------------------------------------------------------------


def test_deadlocking_schedule_is_rejected() -> None:
    # backward emitted before its own forward on the same device -> activation edge + device-order
    # edge form a cycle; simulate must detect the deadlock rather than return a number.
    p, m = 2, 2
    bad = Schedule(
        p,
        m,
        p,
        (
            (Op(0, 0, BACKWARD), Op(0, 0, FORWARD), Op(0, 1, FORWARD), Op(0, 1, BACKWARD)),
            (Op(1, 0, FORWARD), Op(1, 0, BACKWARD), Op(1, 1, FORWARD), Op(1, 1, BACKWARD)),
        ),
    )
    with pytest.raises(ValueError, match="cycl"):
        simulate(bad)


def test_incomplete_schedule_is_rejected() -> None:
    # drop microbatch 1's backward on stage 0 -> completeness check fails
    good = one_f_one_b_schedule(2, 2)
    trimmed = tuple(
        tuple(o for o in dev if not (o.stage == 0 and o.microbatch == 1 and o.kind == BACKWARD))
        for dev in good.device_ops
    )
    with pytest.raises(ValueError, match="incomplete"):
        validate_schedule(Schedule(2, 2, 2, trimmed))


def test_bad_durations_rejected() -> None:
    with pytest.raises(ValueError):
        simulate(gpipe_schedule(2, 2), t_f=0.0)
