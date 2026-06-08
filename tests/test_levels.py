"""Keystone leaf — HardeningLevel is one ordered enum with a 3-of-5 smoke span (ADR-0010)."""

from reasoning_llm.envs.levels import SMOKE_LEVELS, HardeningLevel


def test_five_levels_present():
    assert len(HardeningLevel) == 5


def test_levels_are_ordered_ints():
    values = [lvl.value for lvl in HardeningLevel]
    assert values == sorted(values)
    assert (
        HardeningLevel.L1_EXTENSIONAL
        < HardeningLevel.L3_NORMALIZED
        < HardeningLevel.L5_PERTURBATION
    )


def test_round_trip_by_value():
    assert HardeningLevel(3) is HardeningLevel.L3_NORMALIZED


def test_smoke_levels_are_three_of_five_spanning_min_to_max():
    assert len(SMOKE_LEVELS) == 3
    assert len(set(SMOKE_LEVELS)) == 3  # distinct
    assert min(SMOKE_LEVELS) is HardeningLevel.L1_EXTENSIONAL  # spans the floor …
    assert max(SMOKE_LEVELS) is HardeningLevel.L5_PERTURBATION  # … and the ceiling
    assert all(lvl in HardeningLevel for lvl in SMOKE_LEVELS)
