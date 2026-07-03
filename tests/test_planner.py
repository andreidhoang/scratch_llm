"""A3 query-planner semantics: the training API's adversarial budget accounting, offline.

Every rule the Stanford API enforces over HTTP (reserve-full on submit, refund-on-complete
floored at 1 s, timeout charged full, 409 on duplicate config, 400 on over-budget, config
validation) is asserted here as pure logic, then the end-to-end test spends the 43 200 s cap
on a predeclared IsoFLOP grid against a planted loss surface L(N,D)=E+A/N^a+B/D^b and must
recover the planted compute-optimal config at the 48 B200-hour target via the W5 fitter."""

from __future__ import annotations

import importlib
import time
from types import ModuleType

import pytest

from scratch_llm.scaling.planner import (
    MIN_CHARGE_SECONDS,
    SEQ_LEN,
    TOTAL_BUDGET_SECONDS,
    DuplicateConfigError,
    InsufficientBudgetError,
    InvalidConfigError,
    PlannedRun,
    PlannerError,
    QueryPlanner,
    RunConfig,
    UndeclaredConfigError,
)
from scratch_llm.utils.seeding import seed_everything


def _config(
    layers: int = 2,
    heads: int = 4,
    head_dim: int = 64,
    kv_heads: int | None = None,
    hidden: int | None = None,
    batch: int = 32,
    tokens: int | None = None,
    lr: float = 3e-4,
) -> RunConfig:
    return RunConfig(
        num_hidden_layers=layers,
        num_attention_heads=heads,
        head_dim=head_dim,
        num_key_value_heads=kv_heads if kv_heads is not None else heads,
        hidden_size=hidden if hidden is not None else heads * head_dim,
        train_batch_size=batch,
        total_train_tokens=tokens if tokens is not None else SEQ_LEN * batch * 4,
        learning_rate=lr,
    )


def _assert_ledger_identity(planner: QueryPlanner) -> None:
    total = planner.spent_seconds + planner.reserved_seconds + planner.remaining_seconds
    assert total == pytest.approx(planner.total_budget_seconds)


def test_submit_reserves_full_max_runtime() -> None:
    planner = QueryPlanner()
    planner.submit(_config(), max_runtime_seconds=600.0)
    assert planner.reserved_seconds == 600.0
    assert planner.spent_seconds == 0.0
    assert planner.remaining_seconds == TOTAL_BUDGET_SECONDS - 600.0
    _assert_ledger_identity(planner)


def test_completion_refunds_down_to_actual_runtime() -> None:
    planner = QueryPlanner()
    rid = planner.submit(_config(), max_runtime_seconds=600.0)
    charge = planner.complete(rid, actual_runtime_seconds=42.5, final_loss=3.1)
    assert charge == 42.5
    assert planner.reserved_seconds == 0.0
    assert planner.spent_seconds == 42.5
    assert planner.remaining_seconds == TOTAL_BUDGET_SECONDS - 42.5
    assert planner.results() == [(_config(), 3.1)]
    _assert_ledger_identity(planner)


def test_completion_charge_floors_at_one_second() -> None:
    planner = QueryPlanner()
    rid = planner.submit(_config(), max_runtime_seconds=600.0)
    assert planner.complete(rid, actual_runtime_seconds=0.05, final_loss=3.0) == MIN_CHARGE_SECONDS


def test_completion_charge_never_exceeds_reservation() -> None:
    planner = QueryPlanner()
    rid = planner.submit(_config(), max_runtime_seconds=600.0)
    assert planner.complete(rid, actual_runtime_seconds=1e6, final_loss=3.0) == 600.0


def test_timeout_charged_full_reservation_and_yields_no_result() -> None:
    planner = QueryPlanner()
    rid = planner.submit(_config(), max_runtime_seconds=600.0)
    assert planner.fail_timeout(rid) == 600.0
    assert planner.spent_seconds == 600.0
    assert planner.reserved_seconds == 0.0
    assert planner.results() == []
    _assert_ledger_identity(planner)


def test_over_budget_submission_refused_with_400_semantics() -> None:
    planner = QueryPlanner(total_budget_seconds=1000.0)
    planner.submit(_config(layers=2), max_runtime_seconds=600.0)
    with pytest.raises(InsufficientBudgetError) as exc:
        planner.submit(_config(layers=3), max_runtime_seconds=600.0)
    assert exc.value.status_code == 400
    # the server rejects only strictly-greater reservations: exactly-remaining is accepted
    rid = planner.submit(_config(layers=4), max_runtime_seconds=400.0)
    assert planner.remaining_seconds == 0.0
    planner.complete(rid, actual_runtime_seconds=10.0, final_loss=3.0)
    assert planner.remaining_seconds == 390.0
    _assert_ledger_identity(planner)


def test_duplicate_config_rejected_with_409_semantics() -> None:
    planner = QueryPlanner()
    rid = planner.submit(_config(), max_runtime_seconds=100.0)
    with pytest.raises(DuplicateConfigError) as exc:
        planner.submit(_config(), max_runtime_seconds=50.0)
    assert exc.value.status_code == 409
    # still duplicate after the run finishes — the server checks all experiments ever
    planner.complete(rid, actual_runtime_seconds=5.0, final_loss=3.0)
    with pytest.raises(DuplicateConfigError):
        planner.submit(_config(), max_runtime_seconds=50.0)


@pytest.mark.parametrize(
    "bad_config",
    [
        _config(hidden=512),  # hidden_size != heads * head_dim
        _config(heads=4, kv_heads=3),  # heads % kv_heads != 0
        _config(tokens=SEQ_LEN * 32 * 4 + 1),  # not divisible by seq_len * batch
        _config(layers=-1),
        _config(lr=-1e-4),
    ],
)
def test_invalid_config_rejected_before_any_budget_is_risked(bad_config: RunConfig) -> None:
    planner = QueryPlanner()
    with pytest.raises(InvalidConfigError) as exc:
        planner.submit(bad_config, max_runtime_seconds=100.0)
    assert exc.value.status_code == 422
    assert planner.remaining_seconds == TOTAL_BUDGET_SECONDS


def test_max_runtime_below_one_second_rejected() -> None:
    planner = QueryPlanner()
    with pytest.raises(InvalidConfigError):
        planner.submit(_config(), max_runtime_seconds=0.5)


def test_declared_grid_worst_case_must_fit_the_cap() -> None:
    planner = QueryPlanner(total_budget_seconds=1000.0)
    over = [PlannedRun(_config(layers=n), 300.0) for n in range(2, 7)]  # worst case 1500
    with pytest.raises(InsufficientBudgetError):
        planner.declare_grid(over)
    fitting = [PlannedRun(_config(layers=n), 300.0) for n in range(2, 5)]
    assert planner.declare_grid(fitting) == 900.0
    with pytest.raises(PlannerError):
        planner.declare_grid(fitting)  # the predeclared grid is immutable


def test_grid_rejects_internal_duplicates() -> None:
    planner = QueryPlanner()
    with pytest.raises(DuplicateConfigError):
        planner.declare_grid([PlannedRun(_config(), 10.0), PlannedRun(_config(), 20.0)])


def test_submissions_outside_declared_grid_refused() -> None:
    planner = QueryPlanner()
    planner.declare_grid([PlannedRun(_config(layers=2), 10.0)])
    with pytest.raises(UndeclaredConfigError):
        planner.submit(_config(layers=9), max_runtime_seconds=10.0)
    planner.submit(_config(layers=2), max_runtime_seconds=10.0)


def test_finished_run_cannot_be_finished_twice() -> None:
    planner = QueryPlanner()
    rid = planner.submit(_config(), max_runtime_seconds=100.0)
    planner.complete(rid, actual_runtime_seconds=5.0, final_loss=3.0)
    with pytest.raises(PlannerError):
        planner.complete(rid, actual_runtime_seconds=5.0, final_loss=3.0)
    with pytest.raises(PlannerError):
        planner.fail_timeout(rid)


def test_ledger_identity_holds_through_mixed_lifecycle() -> None:
    planner = QueryPlanner()
    rids = [planner.submit(_config(layers=n), max_runtime_seconds=500.0) for n in range(2, 8)]
    _assert_ledger_identity(planner)
    planner.complete(rids[0], actual_runtime_seconds=30.0, final_loss=3.2)
    _assert_ledger_identity(planner)
    planner.fail_timeout(rids[1])
    _assert_ledger_identity(planner)
    planner.complete(rids[2], actual_runtime_seconds=0.1, final_loss=3.3)
    _assert_ledger_identity(planner)
    assert planner.spent_seconds == 30.0 + 500.0 + 1.0
    assert planner.reserved_seconds == 3 * 500.0
    assert len(planner.results()) == 2


# --- end-to-end: spend the budget on an IsoFLOP grid, recover the planted optimum ----------

# Planted Chinchilla-style surface L(N, D) = E + A/N^alpha + B/D^beta.
_E, _A, _B, _ALPHA, _BETA = 1.69, 406.4, 410.7, 0.34, 0.28
_BUDGETS = [1e17, 3e17, 1e18, 3e18, 1e19, 3e19]
# 48 B200-hours at ~2.25e15 dense bf16 FLOP/s and 0.4 MFU — the leaderboard's big-run target.
_C_TARGET = 48 * 3600 * 2.25e15 * 0.4
_SECONDS_PER_FLOP = 1e-18  # the simulated cluster speed the calibration runs must discover
_DIMS = [
    256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
    1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2304, 2560, 2816, 3072,
]  # fmt: skip
_ARCH_LADDER = [(max(2, d // 128), d) for d in _DIMS]  # (n_layer, d_model), head_dim 64


def _wait_for_isoflop() -> ModuleType:
    """Import the W5 fitter, polling up to 60 s — the two nodes are built in parallel."""
    deadline = time.monotonic() + 60.0
    while True:
        try:
            return importlib.import_module("scratch_llm.scaling.isoflop")
        except ModuleNotFoundError:
            if time.monotonic() > deadline:
                pytest.fail("scaling.isoflop (node W5) not present after 60 s")
            importlib.invalidate_caches()
            time.sleep(2.0)


def _oracle_loss(config: RunConfig) -> float:
    n, d = config.nonembed_params, config.total_train_tokens
    return _E + _A / n**_ALPHA + _B / d**_BETA


def _n_true(compute: float) -> float:
    """Analytic argmin_N of L(N, C/(6N)): N^(a+b) = (aA/(bB)) · (C/6)^b."""
    return ((_ALPHA * _A / (_BETA * _B)) * (compute / 6.0) ** _BETA) ** (1.0 / (_ALPHA + _BETA))


def _grid_config(n_layer: int, d_model: int, tokens: int) -> RunConfig:
    heads = d_model // 64
    return RunConfig(
        num_hidden_layers=n_layer,
        num_attention_heads=heads,
        head_dim=64,
        num_key_value_heads=heads,
        hidden_size=d_model,
        train_batch_size=32,
        total_train_tokens=tokens,
    )


def _build_grid(secs_per_flop: float) -> list[tuple[PlannedRun, float]]:
    """IsoFLOP grid: per budget, every ladder arch with tokens-per-param in [1, 1000];
    max_runtime calibrated to 2x the predicted runtime (floored at the 1 s minimum)."""
    step = SEQ_LEN * 32
    grid: list[tuple[PlannedRun, float]] = []
    for compute in _BUDGETS:
        for n_layer, d_model in _ARCH_LADDER:
            n = 12 * n_layer * d_model**2
            d_raw = compute / (6.0 * n)
            if not 1.0 <= d_raw / n <= 1000.0:
                continue
            tokens = max(1, round(d_raw / step)) * step
            config = _grid_config(n_layer, d_model, tokens)
            max_runtime = max(1.0, 2.0 * config.train_flops * secs_per_flop)
            grid.append((PlannedRun(config, max_runtime), compute))
    return grid


def test_end_to_end_planner_recovers_planted_optimum() -> None:
    seed_everything(0)
    isoflop = _wait_for_isoflop()
    planner = QueryPlanner()

    # 1. Calibrate seconds-per-FLOP from two cheap completed runs (the runbook discipline).
    calib_estimates = []
    for layers in (2, 3):
        config = _config(layers=layers, heads=4, batch=16, tokens=SEQ_LEN * 16 * 8)
        rid = planner.submit(config, max_runtime_seconds=60.0)
        actual = config.train_flops * _SECONDS_PER_FLOP
        planner.complete(rid, actual, _oracle_loss(config))
        calib_estimates.append(actual / config.train_flops)
    secs_per_flop = sum(calib_estimates) / len(calib_estimates)

    # 2. Predeclare the grid; its all-timeout worst case must fit the remaining budget.
    grid = _build_grid(secs_per_flop)
    worst_case = planner.declare_grid([run for run, _ in grid])
    assert worst_case <= planner.remaining_seconds

    # 3. Spend the budget: submit reserves full cap, completion refunds to actual runtime.
    rows = []
    for run, compute in grid:
        rid = planner.submit(run.config, run.max_runtime_seconds)
        actual = run.config.train_flops * secs_per_flop
        assert actual <= run.max_runtime_seconds, "calibrated cap must never time out"
        planner.complete(rid, actual, _oracle_loss(run.config))
        rows.append(
            {
                "compute_budget": compute,
                "parameters": run.config.nonembed_params,
                "final_loss": _oracle_loss(run.config),
            }
        )
    assert planner.reserved_seconds == 0.0
    assert planner.spent_seconds < TOTAL_BUDGET_SECONDS
    _assert_ledger_identity(planner)
    assert len(planner.results()) == len(grid) + 2

    # 4. Fit N_opt ∝ C^a through the W5 fitter and extrapolate to the 48 B200-hr target.
    pairs = isoflop.isoflop_min(rows)
    assert len(pairs) == len(_BUDGETS)
    n_fit = isoflop.fit_powerlaw([c for c, _ in pairs], [n for _, n in pairs])
    d_fit = isoflop.fit_powerlaw([c for c, _ in pairs], [c / (6.0 * n) for c, n in pairs])
    assert 0.35 < n_fit.exponent < 0.55
    assert n_fit.exponent + d_fit.exponent == pytest.approx(1.0, abs=1e-6)

    n_pred = n_fit.predict(_C_TARGET)
    n_star = _n_true(_C_TARGET)
    assert 1 / 1.35 < n_pred / n_star < 1.35

    # 5. Recover the planted optimum architecture (nearest ladder point in log space).
    def nearest_arch(n_target: float) -> tuple[int, int]:
        import math

        return min(_ARCH_LADDER, key=lambda a: abs(math.log(12 * a[0] * a[1] ** 2 / n_target)))

    assert nearest_arch(n_pred) == nearest_arch(n_star)
    n_layer, d_model = nearest_arch(n_pred)
    step = SEQ_LEN * 32
    tokens = max(1, round(_C_TARGET / (6.0 * 12 * n_layer * d_model**2) / step)) * step
    final = _grid_config(n_layer, d_model, tokens)
    final.validate()  # the submitted big-run config passes every API rule
