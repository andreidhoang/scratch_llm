"""Tests for the F1-run iso-FLOP optimizer race (eval/optimizer_race.py + the train.py val hook).

Executable spec for the pending F1 headline (Muon vs LR-tuned AdamW at fixed C=6ND):
- **Pure metrics:** ``tokens_to_match`` finds the first (interpolated) token count where the
  challenger reaches the baseline's final val loss; ``token_saving_fraction`` and
  ``nats_delta_at_budget`` derive from it. All fail loud on malformed curves.
- **Byte-identical default:** turning the val-eval hook ON must not perturb the training batch
  stream (val eval uses fixed sequential windows — no RNG), and the OFF path is untouched.
- **NS instrument:** ``Muon(profile_ns=True)`` accumulates Newton–Schulz wall time; OFF costs zero.
- **Race mechanism:** both arms share seed/init/data; the driver returns comparable curves at the
  same token budget (the iso-FLOP contract) deterministically.
- **Divergence containment:** a non-finite-loss arm returns ``diverged=True`` instead of killing
  the run; ``sweep_lr`` excludes it from the argmin; ``run_race`` raises on a diverged baseline
  (the sweep lied) and Nones the metrics on a diverged challenger (the pre-registered KILL).
- **CLI driver (bench/optimizer_race.py):** incremental atomic persistence (every finished stage
  is on disk under ``status``), the epochs guard, the qk-norm/attention flags, and the full
  pre-registered PASS/KILL ternary (``verdict_string``) — all exercised on a tiny CPU corpus.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scratch_llm.data.shards import tokenize_to_shard
from scratch_llm.eval.optimizer_race import (
    nats_delta_at_budget,
    run_arm,
    run_race,
    sweep_lr,
    token_saving_fraction,
    tokens_to_match,
)
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import Muon
from scratch_llm.train import TrainConfig, train
from scratch_llm.utils.seeding import seed_everything

# bench/ is not a package — load the CLI driver by path (same interpreter, no subprocess).
_BENCH_DRIVER = Path(__file__).resolve().parents[1] / "bench" / "optimizer_race.py"
_spec = importlib.util.spec_from_file_location("bench_optimizer_race", _BENCH_DRIVER)
assert _spec is not None and _spec.loader is not None
bench_race = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_race)

# ---------------------------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------------------------


def test_tokens_to_match_interpolates_known_crossing() -> None:
    # Baseline ends at loss 2.0 after 1000 tokens; challenger crosses 2.0 between 400 and 600:
    # linear interp: 400 + 200 * (2.5 - 2.0) / (2.5 - 1.5) = 500.
    baseline = [(500.0, 3.0), (1000.0, 2.0)]
    challenger = [(200.0, 3.5), (400.0, 2.5), (600.0, 1.5), (1000.0, 1.0)]
    assert tokens_to_match(baseline, challenger) == 500.0


def test_tokens_to_match_first_point_already_below() -> None:
    # Challenger's first measurement is already at/below target: we can honestly claim no
    # earlier than the first measured point.
    baseline = [(1000.0, 2.0)]
    challenger = [(250.0, 1.9), (1000.0, 1.0)]
    assert tokens_to_match(baseline, challenger) == 250.0


def test_tokens_to_match_none_when_never_reached() -> None:
    baseline = [(1000.0, 2.0)]
    challenger = [(500.0, 3.0), (1000.0, 2.5)]
    assert tokens_to_match(baseline, challenger) is None


def test_token_saving_fraction_identical_curves_is_zero() -> None:
    curve = [(250.0, 3.0), (500.0, 2.5), (1000.0, 2.0)]
    saving = token_saving_fraction(curve, list(curve))
    assert saving is not None
    assert abs(saving) < 1e-12  # matches exactly at the shared final point


def test_token_saving_fraction_better_challenger() -> None:
    baseline = [(500.0, 3.0), (1000.0, 2.0)]
    challenger = [(200.0, 3.5), (400.0, 2.5), (600.0, 1.5), (1000.0, 1.0)]
    saving = token_saving_fraction(baseline, challenger)
    assert saving is not None
    assert abs(saving - 0.5) < 1e-12  # crossed at 500 of 1000 baseline tokens


def test_nats_delta_at_budget_sign_convention() -> None:
    # Challenger 0.3 nats LOWER at the shared budget -> delta = -0.3.
    baseline = [(1000.0, 2.0)]
    challenger = [(1000.0, 1.7)]
    assert abs(nats_delta_at_budget(baseline, challenger) - (-0.3)) < 1e-12


def test_nats_delta_rejects_mismatched_budgets() -> None:
    baseline = [(1000.0, 2.0)]
    challenger = [(800.0, 1.7)]  # not iso-FLOP: final token counts differ
    try:
        nats_delta_at_budget(baseline, challenger)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched final token budgets")


def test_metrics_reject_malformed_curves() -> None:
    good = [(1000.0, 2.0)]
    for bad in (
        [],  # empty
        [(500.0, 2.0), (400.0, 1.9)],  # tokens not increasing
        [(500.0, float("nan"))],  # non-finite loss
    ):
        for args in ((bad, good), (good, bad)):
            try:
                tokens_to_match(*args)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for malformed curve {bad!r}")


# ---------------------------------------------------------------------------------------------
# train.py val-eval hook — the additive no-op contract
# ---------------------------------------------------------------------------------------------


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, context_length=16)


def _structured_corpus(n: int = 2048, vocab: int = 32) -> np.ndarray:
    return np.tile(np.arange(vocab, dtype=np.int64), n // vocab + 1)[:n]


def test_val_eval_hook_does_not_perturb_training() -> None:
    """The DoD contract: identical train-loss history with the hook OFF vs ON.

    Stronger than 'default path unchanged' — the ON path samples val loss from FIXED sequential
    windows (no RNG), so even an active hook must leave the batch stream byte-identical.
    """
    data = _structured_corpus()
    val = _structured_corpus(512)
    base = TrainConfig(max_steps=12, batch_size=4, context_length=16, log_every=1, seed=7)

    seed_everything(7)
    hist_off = train(base, data, TransformerLM(_tiny_cfg()))

    collected: list[tuple[int, float]] = []
    seed_everything(7)
    hist_on = train(
        replace(base, eval_every=4),
        data,
        TransformerLM(_tiny_cfg()),
        val_data=val,
        eval_hook=lambda step, loss: collected.append((step, loss)),
    )

    assert hist_on == hist_off  # bitwise-identical loss history
    steps = [s for s, _ in collected]
    assert steps and steps[-1] == 11  # final step always evaluated (the iso-FLOP endpoint)
    assert all(np.isfinite(loss) for _, loss in collected)


def test_val_eval_is_deterministic() -> None:
    data = _structured_corpus()
    val = _structured_corpus(512)
    curves: list[list[tuple[int, float]]] = []
    for _ in range(2):
        collected: list[tuple[int, float]] = []
        seed_everything(3)
        train(
            TrainConfig(max_steps=8, batch_size=4, context_length=16, seed=3, eval_every=4),
            data,
            TransformerLM(_tiny_cfg()),
            val_data=val,
            eval_hook=lambda step, loss, c=collected: c.append((step, loss)),
        )
        curves.append(collected)
    assert curves[0] == curves[1]


# ---------------------------------------------------------------------------------------------
# Muon NS wall-time instrument
# ---------------------------------------------------------------------------------------------


def test_muon_profile_ns_counters() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([p], lr=1e-3, profile_ns=True)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()
    assert opt.ns_calls == 3
    assert opt.ns_seconds > 0.0


def test_muon_profile_ns_off_by_default() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8))
    opt = Muon([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    assert opt.ns_calls == 0
    assert opt.ns_seconds == 0.0
    # The instrument is an instance attribute, NOT optimizer state: checkpoints are unaffected.
    assert "profile_ns" not in opt.state_dict()["param_groups"][0]


# ---------------------------------------------------------------------------------------------
# The A/B driver — determinism + the iso-FLOP contract
# ---------------------------------------------------------------------------------------------


def test_run_arm_is_deterministic() -> None:
    data = _structured_corpus()
    val = _structured_corpus(512)
    cfg = TrainConfig(max_steps=12, batch_size=4, context_length=16, seed=11, eval_every=4)
    a = run_arm(_tiny_cfg(), cfg, data, val, optimizer="adamw", max_lr=3e-3, label="a")
    b = run_arm(_tiny_cfg(), cfg, data, val, optimizer="adamw", max_lr=3e-3, label="b")
    assert a.val_curve == b.val_curve
    assert a.train_history == b.train_history


def test_run_race_iso_flop_contract() -> None:
    data = _structured_corpus(4096)
    val = _structured_corpus(512)
    cfg = TrainConfig(max_steps=16, batch_size=4, context_length=16, seed=5, eval_every=4)
    race = run_race(_tiny_cfg(), cfg, data, val, baseline_lr=3e-3, challenger_lr=3e-3)

    # Both arms end at the same token budget — C=6ND held constant by construction.
    assert race.baseline.val_curve[-1][0] == race.challenger.val_curve[-1][0]
    assert race.total_tokens == 16 * 4 * 16
    assert race.compute_flops == 6.0 * race.n_params * race.total_tokens
    assert len(race.baseline.val_curve) >= 3
    # The metric fields are populated coherently.
    if race.tokens_to_match is not None:
        assert 0 < race.tokens_to_match <= race.total_tokens
        assert race.token_saving is not None
    assert race.nats_delta is not None  # None only on a diverged challenger — this one is healthy
    assert np.isfinite(race.nats_delta)
    # Muon arm carries the NS instrument when asked.
    assert race.challenger.ns_seconds >= 0.0


def test_run_arm_surfaces_f9_max_logit_when_tracked() -> None:
    """The F9 ride-along: an observer-built model yields a finite max_attn_logit; default None."""
    from dataclasses import replace as dc_replace

    data = _structured_corpus()
    val = _structured_corpus(512)
    cfg = TrainConfig(max_steps=4, batch_size=4, context_length=16, seed=2, eval_every=2)

    plain = run_arm(_tiny_cfg(), cfg, data, val, optimizer="adamw", max_lr=3e-3, label="plain")
    assert plain.max_attn_logit is None  # observer off by default — byte-identical base path

    tracked_cfg = dc_replace(_tiny_cfg(), track_attn_logits=True)
    tracked = run_arm(tracked_cfg, cfg, data, val, optimizer="adamw", max_lr=3e-3, label="obs")
    assert tracked.max_attn_logit is not None
    assert np.isfinite(tracked.max_attn_logit)


def test_sweep_lr_picks_argmin_final_val_loss() -> None:
    data = _structured_corpus(4096)
    val = _structured_corpus(512)
    cfg = TrainConfig(max_steps=10, batch_size=4, context_length=16, seed=9, eval_every=5)
    sweep = sweep_lr(_tiny_cfg(), cfg, data, val, lrs=[3e-3, 1e-9])
    finals = {arm.lr: arm.val_curve[-1][1] for arm in sweep.arms}
    assert sweep.best_lr == min(finals, key=lambda lr: finals[lr])
    # An LR of 1e-9 cannot learn anything in 10 steps; 3e-3 must beat it on this corpus.
    assert sweep.best_lr == 3e-3


# ---------------------------------------------------------------------------------------------
# Divergence containment — a non-finite arm is DATA (the pre-registered KILL), not a crash
# ---------------------------------------------------------------------------------------------


def test_run_arm_contains_divergence() -> None:
    """lr=1e9 explodes to a non-finite loss within steps; run_arm returns instead of raising."""
    data = _structured_corpus()
    val = _structured_corpus(512)
    cfg = TrainConfig(
        max_steps=8, batch_size=4, context_length=16, seed=9, eval_every=2, log_every=1
    )
    arm = run_arm(_tiny_cfg(), cfg, data, val, optimizer="adamw", max_lr=1e9, label="boom")
    assert arm.diverged is True
    assert arm.train_history == []  # train() owns its history; the raise discards it
    assert arm.wall_seconds > 0.0


def test_run_arm_reraises_foreign_runtime_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only train()'s non-finite-loss signal is contained — any other RuntimeError is a crash."""
    import scratch_llm.eval.optimizer_race as race_mod

    def _boom(*args: object, **kwargs: object) -> list[tuple[int, float]]:
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(race_mod, "train", _boom)
    cfg = TrainConfig(max_steps=4, batch_size=4, context_length=16, eval_every=2)
    with pytest.raises(RuntimeError, match="device-side assert"):
        run_arm(
            _tiny_cfg(),
            cfg,
            _structured_corpus(),
            _structured_corpus(512),
            optimizer="adamw",
            max_lr=3e-3,
            label="crash",
        )


def test_sweep_lr_excludes_diverged_arm() -> None:
    data = _structured_corpus(4096)
    val = _structured_corpus(512)
    cfg = TrainConfig(
        max_steps=10, batch_size=4, context_length=16, seed=9, eval_every=5, log_every=1
    )
    sweep = sweep_lr(_tiny_cfg(), cfg, data, val, lrs=[3e-3, 1e9])
    by_lr = {arm.lr: arm for arm in sweep.arms}
    assert by_lr[1e9].diverged is True
    assert by_lr[3e-3].diverged is False
    assert len(sweep.arms) == 2  # the diverged arm is ledgered, only the argmin excludes it
    assert sweep.best_lr == 3e-3


def test_sweep_lr_all_diverged_raises() -> None:
    cfg = TrainConfig(
        max_steps=6, batch_size=4, context_length=16, seed=9, eval_every=3, log_every=1
    )
    with pytest.raises(RuntimeError, match="every LR"):
        sweep_lr(
            _tiny_cfg(), cfg, _structured_corpus(4096), _structured_corpus(512), lrs=[1e9, 2e9]
        )


def test_run_race_baseline_divergence_raises() -> None:
    """A diverging tuned baseline means the sweep lied — unrecoverable, never a silent KILL."""
    cfg = TrainConfig(
        max_steps=8, batch_size=4, context_length=16, seed=5, eval_every=4, log_every=1
    )
    with pytest.raises(RuntimeError, match="sweep lied"):
        run_race(
            _tiny_cfg(),
            cfg,
            _structured_corpus(4096),
            _structured_corpus(512),
            baseline_lr=1e9,
            challenger_lr=3e-3,
        )


def test_run_race_diverged_challenger_nones_metrics_and_fires_hooks() -> None:
    roles: list[tuple[str, bool]] = []
    cfg = TrainConfig(
        max_steps=8, batch_size=4, context_length=16, seed=5, eval_every=4, log_every=1
    )
    race = run_race(
        _tiny_cfg(),
        cfg,
        _structured_corpus(4096),
        _structured_corpus(512),
        baseline_lr=3e-3,
        challenger_lr=1e9,
        arm_hook=lambda role, arm: roles.append((role, arm.diverged)),
    )
    assert roles == [("baseline", False), ("challenger", True)]  # per-arm persistence seam fired
    assert race.challenger.diverged is True
    assert race.tokens_to_match is None
    assert race.token_saving is None
    assert race.nats_delta is None  # the driver records the pre-registered KILL, not a number


# ---------------------------------------------------------------------------------------------
# The CLI driver (bench/optimizer_race.py) — persistence, guards, flags, verdict
# ---------------------------------------------------------------------------------------------


class _ByteEncoder:
    """Trivial TokenEncoder for shard fixtures: byte % 31 (eot_id=31 ⇒ detected vocab 32)."""

    def encode(self, text: str) -> list[int]:
        return [b % 31 for b in text.encode("ascii")]


def _tiny_shards(tmp_path: Path, n_chars: int = 6000) -> Path:
    data_dir = tmp_path / "shards"
    data_dir.mkdir()
    doc = ("abcdefghijklmnopqrstuvwxyz0123 " * (n_chars // 31 + 1))[:n_chars]
    tokenize_to_shard([doc], _ByteEncoder(), eot_id=31, out_path=data_dir / "shard_000.bin")
    return data_dir


def _driver_argv(data_dir: Path, out: Path, *extra: str) -> list[str]:
    # depth 1 ⇒ d_model 64, 1 head; 64 tokens/step ⇒ --tokens 1024 is a 16-step race arm,
    # long enough for the log_every=10 divergence check to trip at lr=1e9.
    return [
        "optimizer_race.py",
        "--data-dir",
        str(data_dir),
        "--depth",
        "1",
        "--batch-size",
        "4",
        "--context-length",
        "16",
        "--eval-batches",
        "4",
        "--amp",
        "none",
        "--device",
        "cpu",
        "--out",
        str(out),
        *extra,
    ]


def test_driver_sweep_excludes_diverged_arm_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _tiny_shards(tmp_path)
    out = tmp_path / "f1.json"
    monkeypatch.setattr(
        sys,
        "argv",
        _driver_argv(
            data_dir, out, "--tokens", "1024", "--sweep-lrs", "3e-3,1e9", "--sweep-frac", "1.0"
        ),
    )
    bench_race.main()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"  # the JSON advanced through every stage
    rows = {row["lr"]: row for row in payload["sweep"]}
    assert rows[1e9]["diverged"] is True
    assert rows[3e-3]["diverged"] is False
    assert payload["baseline_lr"] == 3e-3  # the diverged LR never wins the sweep


def test_driver_diverged_challenger_records_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One diverged challenger arm must yield a clean exit + a recorded KILL, never a crash."""
    data_dir = _tiny_shards(tmp_path)
    out = tmp_path / "no" / "such" / "dir" / "f1.json"  # parents must be created, not assumed
    monkeypatch.setattr(
        sys,
        "argv",
        _driver_argv(
            data_dir, out, "--tokens", "1024", "--baseline-lr", "3e-3", "--muon-lr", "1e9"
        ),
    )
    bench_race.main()  # returns cleanly — exit 0
    assert "KILL (divergence at reused LR)" in capsys.readouterr().out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    race = payload["race"]
    assert race["diverged"] is True
    assert race["tokens_to_match"] is None
    assert race["token_saving_fraction"] is None
    assert race["nats_delta"] is None
    assert race["muon_wall_perturbed_by_profiler"] is True
    assert race["epochs"] == payload["epochs"]
    assert payload["epochs"] < 1.5  # the pass side of the epochs guard, at the default cap
    assert payload["baseline_arm"]["diverged"] is False
    assert payload["challenger_arm"]["diverged"] is True


def test_driver_baseline_divergence_persists_stage_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crash contract: the newest completed stage is on disk when run_race raises."""
    data_dir = _tiny_shards(tmp_path)
    out = tmp_path / "f1.json"
    monkeypatch.setattr(
        sys, "argv", _driver_argv(data_dir, out, "--tokens", "1024", "--baseline-lr", "1e9")
    )
    with pytest.raises(RuntimeError, match="sweep lied"):
        bench_race.main()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "baseline_arm_done"
    assert payload["baseline_arm"]["diverged"] is True
    assert "race" not in payload


def test_driver_epochs_guard_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = _tiny_shards(tmp_path)
    out = tmp_path / "f1.json"
    # ~5.9k-token train split at --tokens 64000 ≈ 10.8 epochs ≫ the default 1.5 cap.
    monkeypatch.setattr(sys, "argv", _driver_argv(data_dir, out, "--tokens", "64000"))
    with pytest.raises(SystemExit, match="max-epochs"):
        bench_race.main()
    assert "epochs = D / corpus" in capsys.readouterr().out  # the number is ALWAYS printed
    assert not out.exists()  # aborted before any compute was spent or stage persisted


def test_driver_epochs_guard_passes_at_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # train split = 1001 − 65(val) = 936 tokens; --tokens 1404 = exactly 1.5 epochs ⇒ allowed
    # (the guard is strict >), and the race runs to completion.
    data_dir = _tiny_shards(tmp_path, n_chars=1000)
    out = tmp_path / "f1.json"
    monkeypatch.setattr(
        sys, "argv", _driver_argv(data_dir, out, "--tokens", "1404", "--baseline-lr", "3e-3")
    )
    bench_race.main()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["epochs"] == pytest.approx(1.5)


def test_verdict_string_covers_the_preregistered_ternary() -> None:
    v = bench_race.verdict_string
    # KILL: divergence at the reused LR (metrics are None by construction).
    assert v(diverged=True, saving=None, nats_delta=None, ns_overhead=0.0) == (
        "KILL (divergence at reused LR)"
    )
    # KILL: saving <5% (or never matched) without the ≥0.02-nats rescue.
    assert v(diverged=False, saving=0.04, nats_delta=-0.001, ns_overhead=0.0) == "KILL"
    assert v(diverged=False, saving=None, nats_delta=-0.001, ns_overhead=0.0) == "KILL"
    # The nats rescue: ≥0.02 nats lower at iso-FLOP saves a sub-5% token saving.
    assert v(diverged=False, saving=0.04, nats_delta=-0.05, ns_overhead=0.0) == "PASS"
    # KILL: NS overhead >3% — explicit, never a PASS-with-warn (the old ternary's bug).
    assert v(diverged=False, saving=0.2, nats_delta=-0.1, ns_overhead=0.031) == (
        "KILL (NS overhead >3%)"
    )
    # PASS band: warn tag from the 1% analytic bound up to the 3% KILL.
    assert v(diverged=False, saving=0.2, nats_delta=-0.1, ns_overhead=0.02) == (
        "PASS (NS overhead ⚠)"
    )
    assert v(diverged=False, saving=0.2, nats_delta=-0.1, ns_overhead=0.005) == "PASS"


def test_model_cfg_flags_reach_config() -> None:
    """--no-qk-norm / --attention must reach ModelConfig — F9 is conditioned on qk_norm=True."""
    parser = bench_race.build_parser()

    defaults = parser.parse_args(["--data-dir", "unused"])
    cfg = bench_race.model_cfg_from_args(defaults, vocab_size=32)
    assert cfg.qk_norm is True  # ON by default: the pre-registered F9 falsifier requires it
    assert cfg.use_sdpa is True
    assert cfg.track_attn_logits is True

    flipped = parser.parse_args(
        ["--data-dir", "unused", "--no-qk-norm", "--attention", "eager", "--no-track-logits"]
    )
    cfg = bench_race.model_cfg_from_args(flipped, vocab_size=32)
    assert cfg.qk_norm is False
    assert cfg.use_sdpa is False
    assert cfg.track_attn_logits is False
