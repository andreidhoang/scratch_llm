"""Budget-constrained query planner for the A3 ``scaling_laws`` leaderboard (offline method).

The Stanford training API charges wall-clock seconds against a hard 12 B200-hour cap
(43_200 s) with adversarial accounting: a submitted run **reserves its full
``max_runtime_seconds``** while queued/running; a completed run is **refunded down to its
actual runtime** (floored at 1 s); a timed-out run is **charged the full reservation**; a
duplicate config is rejected (HTTP 409); a submission whose reservation exceeds the remaining
budget is refused (HTTP 400). This module is that accounting as pure logic — the HTTP client
lives in the network-locked official scaffold (``cs336_scaling/client.py``); what this repo
owns is the *method*: predeclare the grid, reserve pessimistically, refund optimistically,
never race the cap. (The A3 leaderboard runbook was deleted 31/08 — the API is Stanford-VPN-only,
so the rung was permanently BLOCKED-EXTERNAL; `git log` holds it.)

Invariants:
- ``spent + reserved + remaining == total_budget_seconds`` after every operation; money only
  moves from ``remaining`` → ``reserved`` (submit) → ``spent``/back (finish), never appears.
- A finished run is charged exactly ``clamp(actual, 1, max_runtime)``; an in-flight run holds
  exactly ``max_runtime`` — mirroring the server's ``budget.py`` case expression.
- A declared grid's **worst case** (every run times out at full reservation) fits the
  remaining budget, so no completion order can strand the plan half-run.
- Duplicate rejection here keys on the config alone (stricter than the server, which hashes
  config *plus* ``max_runtime_seconds``): re-buying a config under a different cap is wasted
  budget, so the planner forbids what the API would merely permit.

Interview question: "You get 12 B200-hours of training-API queries to place a 48-hour
compute-optimal bet — how do you spend them?" Predeclare an IsoFLOP grid whose full-timeout
cost fits the cap, calibrate ``max_runtime_seconds`` from two cheap completed runs, argmin
loss per budget, fit ``N_opt ∝ C^a`` in log-log, extrapolate once, submit once.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

SEQ_LEN = 512
"""Fixed by the training API; tokens per optimizer step = SEQ_LEN * train_batch_size."""

TOTAL_BUDGET_SECONDS = 43_200.0
"""12 B200-hours — the hard API-enforced cap on total reserved+spent wall-clock seconds."""

MIN_CHARGE_SECONDS = 1.0
"""A finished run is never refunded below 1 s (server: ``greatest(1.0, used_runtime)``)."""

RunStatus = Literal["reserved", "completed", "timed_out"]


class PlannerError(Exception):
    """Base planner failure; ``status_code`` mirrors the HTTP status the live API returns."""

    status_code: int = 400


class InvalidConfigError(PlannerError):
    """Config violates an API validation rule (server: pydantic 422)."""

    status_code = 422


class InsufficientBudgetError(PlannerError):
    """Reservation would exceed the remaining budget (server: 400 'insufficient budget')."""

    status_code = 400


class DuplicateConfigError(PlannerError):
    """Config was already submitted (server: 409 'experiment already exists')."""

    status_code = 409


class UndeclaredConfigError(PlannerError):
    """Config is outside the predeclared grid — the discipline this planner exists to enforce."""

    status_code = 400


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One training-API configuration — the knobs the leaderboard lets you vary.

    ``seq_len`` (512), ``n_val_tokens`` and vocab are fixed server-side and not represented.
    """

    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    num_key_value_heads: int
    hidden_size: int
    train_batch_size: int
    total_train_tokens: int
    learning_rate: float = 3e-4

    def validate(self) -> None:
        """Apply the API's config-consistency rules locally, before any budget is risked."""
        for f in dataclasses.fields(self):
            if getattr(self, f.name) <= 0:
                raise InvalidConfigError(f"{f.name} must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise InvalidConfigError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise InvalidConfigError("num_attention_heads must be divisible by num_key_value_heads")
        if self.total_train_tokens % (SEQ_LEN * self.train_batch_size) != 0:
            raise InvalidConfigError(
                "total_train_tokens must be divisible by seq_len * train_batch_size"
            )

    @property
    def nonembed_params(self) -> int:
        """N ≈ 12 · n_layer · d_model² — the scaling-law abscissa for this architecture."""
        return 12 * self.num_hidden_layers * self.hidden_size**2

    @property
    def train_flops(self) -> float:
        """C ≈ 6·N·D — the compute this run buys (the IsoFLOP budget it sits on)."""
        return 6.0 * self.nonembed_params * self.total_train_tokens


@dataclass(frozen=True, slots=True)
class PlannedRun:
    """A grid cell: the config plus the wall-clock reservation you are willing to risk on it."""

    config: RunConfig
    max_runtime_seconds: float


@dataclass(slots=True)
class LedgerEntry:
    """One submitted run's budget line: reservation, terminal charge, and observed loss."""

    run_id: int
    config: RunConfig
    max_runtime_seconds: float
    status: RunStatus = "reserved"
    charged_seconds: float = 0.0
    final_loss: float | None = None


@dataclass
class QueryPlanner:
    """Offline twin of the training API's budget ledger + submission gate.

    Every rule the server enforces with an HTTP error is enforced here first, so a plan that
    survives the planner cannot be rejected (or silently drained) by the live API.
    """

    total_budget_seconds: float = TOTAL_BUDGET_SECONDS
    _entries: list[LedgerEntry] = field(default_factory=list, repr=False)
    _seen: set[RunConfig] = field(default_factory=set, repr=False)
    _declared: set[RunConfig] | None = field(default=None, repr=False)

    @property
    def spent_seconds(self) -> float:
        """Terminal charges: completed runs at clamp(actual, 1, max), timeouts at full max."""
        return sum(e.charged_seconds for e in self._entries if e.status != "reserved")

    @property
    def reserved_seconds(self) -> float:
        """In-flight holds — each queued/running run pins its full ``max_runtime_seconds``."""
        return sum(e.max_runtime_seconds for e in self._entries if e.status == "reserved")

    @property
    def remaining_seconds(self) -> float:
        return self.total_budget_seconds - self.spent_seconds - self.reserved_seconds

    def ledger(self) -> list[LedgerEntry]:
        return list(self._entries)

    def results(self) -> list[tuple[RunConfig, float]]:
        """(config, final val loss) for every completed run — the fitter's input."""
        return [
            (e.config, e.final_loss)
            for e in self._entries
            if e.status == "completed" and e.final_loss is not None
        ]

    def declare_grid(self, grid: Sequence[PlannedRun]) -> float:
        """Predeclare the full query grid; returns its worst-case (all-timeout) cost.

        Rejects the whole grid if any config is invalid or duplicated, or if the worst case
        would not fit the remaining budget — the all-or-nothing check that makes the plan
        safe against the timeout-charges-full adversary. One grid per planner; submissions
        after this call must come from it.
        """
        if self._declared is not None:
            raise PlannerError("grid already declared — the predeclared grid is immutable")
        configs: set[RunConfig] = set()
        worst_case = 0.0
        for run in grid:
            run.config.validate()
            if run.max_runtime_seconds < MIN_CHARGE_SECONDS:
                raise InvalidConfigError("max_runtime_seconds must be >= 1")
            if run.config in configs or run.config in self._seen:
                raise DuplicateConfigError("duplicate config in declared grid")
            configs.add(run.config)
            worst_case += run.max_runtime_seconds
        if worst_case > self.remaining_seconds:
            raise InsufficientBudgetError(
                f"grid worst case {worst_case:.0f}s exceeds remaining "
                f"{self.remaining_seconds:.0f}s of the {self.total_budget_seconds:.0f}s cap"
            )
        self._declared = configs
        return worst_case

    def submit(self, config: RunConfig, max_runtime_seconds: float) -> int:
        """Reserve the FULL ``max_runtime_seconds`` and queue the run; returns its run id."""
        config.validate()
        if max_runtime_seconds < MIN_CHARGE_SECONDS:
            raise InvalidConfigError("max_runtime_seconds must be >= 1")
        if config in self._seen:
            raise DuplicateConfigError("experiment already exists for this training config")
        if self._declared is not None and config not in self._declared:
            raise UndeclaredConfigError("config is not in the predeclared grid")
        if max_runtime_seconds > self.remaining_seconds:
            raise InsufficientBudgetError(
                f"insufficient budget: reservation {max_runtime_seconds:.0f}s > "
                f"remaining {self.remaining_seconds:.0f}s"
            )
        run_id = len(self._entries)
        self._entries.append(LedgerEntry(run_id, config, float(max_runtime_seconds)))
        self._seen.add(config)
        return run_id

    def complete(self, run_id: int, actual_runtime_seconds: float, final_loss: float) -> float:
        """Finish a run: refund the reservation down to clamp(actual, 1, max); record the loss.

        Returns the terminal charge.
        """
        entry = self._reserved_entry(run_id)
        charge = min(max(actual_runtime_seconds, MIN_CHARGE_SECONDS), entry.max_runtime_seconds)
        entry.status = "completed"
        entry.charged_seconds = charge
        entry.final_loss = float(final_loss)
        return charge

    def fail_timeout(self, run_id: int) -> float:
        """A timed-out run yields no loss and is charged its FULL reservation (the adversary)."""
        entry = self._reserved_entry(run_id)
        entry.status = "timed_out"
        entry.charged_seconds = entry.max_runtime_seconds
        return entry.charged_seconds

    def _reserved_entry(self, run_id: int) -> LedgerEntry:
        if not 0 <= run_id < len(self._entries):
            raise PlannerError(f"unknown run_id {run_id}")
        entry = self._entries[run_id]
        if entry.status != "reserved":
            raise PlannerError(f"run {run_id} already finished ({entry.status})")
        return entry
