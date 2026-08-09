"""P5.5 d20 probe gate — arm math, the selection rule, idempotency, and a CPU end-to-end arm."""

from __future__ import annotations

import json
import math

import pytest

from scratch_llm.scaling import d20_probe
from scratch_llm.scaling.d20_probe import (
    D20_GLOBAL_BATCH_TOKENS,
    ETA_STAR,
    GRID_BATCH_TOKENS,
    PROBE_TOKENS,
    TIE_BAND_BPB,
    ProbeRecord,
    build_arms,
    center_lr,
    load_arm_record,
    run_arm,
    select_winner,
    steps_for_probe,
)


def test_center_lr_is_eta_star_times_sqrt32() -> None:
    """The composite rule's √B transfer: 0.0021 · √(524288/16384) = 0.0021·√32 ≈ 0.0119."""
    assert D20_GLOBAL_BATCH_TOKENS / GRID_BATCH_TOKENS == 32
    assert center_lr() == pytest.approx(ETA_STAR * math.sqrt(32))
    assert center_lr() == pytest.approx(0.011879, abs=1e-6)


def test_arms_are_the_registered_bracket() -> None:
    arms = build_arms()
    assert [a.mult for a in arms] == [0.7, 1.0, 1.4]
    assert [a.name for a in arms] == ["lr0.7", "lr1.0", "lr1.4"]
    for arm in arms:
        assert arm.lr == pytest.approx(center_lr() * arm.mult)
    # The bracket spans 2× around the prediction — wide enough to catch a √B failure.
    assert arms[0].lr < 0.009 < arms[1].lr < 0.015 < arms[2].lr


def test_probe_budget_is_five_percent_of_the_d20() -> None:
    assert PROBE_TOKENS == 480_000_000
    steps = steps_for_probe(PROBE_TOKENS, D20_GLOBAL_BATCH_TOKENS)
    assert steps == 916  # ceil(480,000,000 / 524,288) — never undershoots
    assert steps * D20_GLOBAL_BATCH_TOKENS >= PROBE_TOKENS


def test_steps_for_probe_ceils_and_validates() -> None:
    assert steps_for_probe(524_288, 524_288) == 1
    assert steps_for_probe(524_289, 524_288) == 2
    with pytest.raises(ValueError, match="positive"):
        steps_for_probe(1000, 0)


def _record(arm: str, lr: float, bpb: float) -> ProbeRecord:
    return ProbeRecord(
        arm=arm,
        mult=lr,
        lr=lr,
        depth=20,
        n_params=480_431_360,
        tokens=PROBE_TOKENS,
        steps=916,
        val_bpb=bpb,
        val_loss=bpb * 1.93 * math.log(2),
        wall_s=500.0,
        seed=0,
        world_size=8,
    )


def test_select_winner_clean_win() -> None:
    arms = build_arms()
    records = [
        _record("lr0.7", arms[0].lr, 0.301),
        _record("lr1.0", arms[1].lr, 0.290),
        _record("lr1.4", arms[2].lr, 0.299),
    ]
    sel = select_winner(arms, records)
    assert sel.winner.name == "lr1.0"
    assert sel.winner_bpb == 0.290
    assert sel.tied == []


def test_select_winner_tie_breaks_to_lower_lr() -> None:
    """Two arms inside TIE_BAND_BPB of the best ⇒ the LOWER LR commits (P5's rule, reused)."""
    arms = build_arms()
    records = [
        _record("lr0.7", arms[0].lr, 0.2900),
        _record("lr1.0", arms[1].lr, 0.2929),
        _record("lr1.4", arms[2].lr, 0.310),
    ]
    sel = select_winner(arms, records)
    assert sel.winner.name == "lr0.7"
    assert sel.tied == ["lr1.0"]


def test_select_winner_band_boundary_is_exclusive_of_outsiders() -> None:
    """An arm exactly at best + band is IN the tie; one epsilon beyond is out."""
    arms = build_arms()
    inside = _record("lr1.0", arms[1].lr, 0.290 + TIE_BAND_BPB)
    outside = _record("lr1.4", arms[2].lr, 0.290 + TIE_BAND_BPB + 1e-9)
    sel = select_winner(arms, [_record("lr0.7", arms[0].lr, 0.290), inside, outside])
    assert sel.winner.name == "lr0.7"
    assert sel.tied == ["lr1.0"]


def test_select_winner_refuses_a_partial_probe() -> None:
    arms = build_arms()
    with pytest.raises(ValueError, match="missing results"):
        select_winner(arms, [_record("lr1.0", arms[1].lr, 0.290)])


def test_run_arm_skips_a_finished_arm(tmp_path, monkeypatch) -> None:
    """Idempotency: an existing results.json short-circuits before ANY training."""
    arm = build_arms()[1]
    done = _record(arm.name, arm.lr, 0.290)
    out = tmp_path / arm.name
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps([done]))

    def _boom(_cfg):  # pragma: no cover - must never run
        raise AssertionError("run_speedrun called for a finished arm")

    monkeypatch.setattr(d20_probe, "run_speedrun", _boom)
    assert run_arm(arm, out_root=tmp_path) == done
    assert load_arm_record(tmp_path, arm) == done
    assert load_arm_record(tmp_path, build_arms()[0]) is None  # absent arm ⇒ None


def test_run_arm_end_to_end_nano_cpu(tmp_path) -> None:
    """The wiring composes at nano scale on CPU: 4 steps at depth 2, record written, skip on
    re-run — the same pre-flight philosophy as speedrun --nano, for the probe path."""
    arm = build_arms()[1]
    record = run_arm(
        arm,
        out_root=tmp_path,
        corpus_path=None,  # the built-in nano corpus
        device="cpu",
        batch_size=8,
        context_length=64,
        vocab_size=384,
        bf16=False,
        depth=2,
        tokens=8 * 64 * 4,  # exactly 4 steps at world=1
    )
    assert record["steps"] == 4
    assert record["arm"] == "lr1.0"
    assert record["world_size"] == 1
    assert record["n_params"] > 0
    assert record["val_bpb"] > 0
    # Rank-0 artifact landed, and a second invocation is a free skip.
    assert load_arm_record(tmp_path, arm) == record
    assert (
        run_arm(
            arm,
            out_root=tmp_path,
            device="cpu",
            batch_size=8,
            context_length=64,
            vocab_size=384,
            bf16=False,
            depth=2,
            tokens=8 * 64 * 4,
        )
        == record
    )
