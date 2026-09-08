"""The per-checkpoint report card — the artifact that makes two checkpoints comparable (T1/T-R4).

:class:`~scratch_llm.eval.report_card.ReportCard` answers "how did this model score?". This module
answers the harder question a speedrun actually asks: **"is this checkpoint better than that one,
or was it evaluated differently?"** Every field here exists to make the second question answerable
without trusting anyone's memory of how last night's run was configured.

A :class:`CheckpointCard` carries three things:

* **What was measured** — ``metrics``: ``val_bpb`` (intrinsic), ``core`` (the DCLM CORE aggregate
  from :mod:`scratch_llm.eval.core_suite`), and the vibe summary. A metric required for the card's
  stage and absent is a :class:`MissingMetric` **error at construction**, never a blank cell: a
  report card with a hole in it is how a run gets declared finished on the metrics that happened
  to compute.
* **What it was measured on** — :class:`EvalProtocol`: the task list, the subsample limit, the
  eval-bundle version, the held-out split identity, the vibe prompt set and decode id. Fingerprinted
  in three independent *scopes* (``core`` / ``val`` / ``vibe``) because they travel differently:
  two checkpoints of ours share all three, while the production floor (a nanochat d20 scored by
  nanochat's own harness on the same bundle) shares only ``core`` — same tasks, same recipe,
  different tokenizer and different held-out split. One fingerprint over everything would refuse
  the floor comparison the rung exists to make; one fingerprint per scope refuses exactly the
  comparisons that are meaningless.
* **What produced it** — ``provenance``: checkpoint path, step, tokens seen, params, device,
  tokenizer id, commit. Model-side facts (BOS policy, max_seq_len, tokenizer) live here and *not*
  in the protocol, because each model legitimately brings its own — they are documented deltas,
  not comparability blockers.

:func:`d20_gate` is the rung's exit predicate: a conjunction of structural checks (which need no
numbers: same protocol, same architecture, monotone tokens, byte-identical vibe re-run) and
threshold checks against a :class:`D20Target`. **The thresholds are not defined here.** They are
loaded (``D20Target.from_mapping``) and every field is mandatory — a null is an error naming the
field. The sixty-day plan states T-R4's target as the word "checkpoint" and no number, so the
numbers are Huy's to write, once, in one place, before the run.

Comparisons are exact. ``>=`` passes at equality and fails one ulp below; there is no epsilon,
because a gate with a hidden tolerance is a gate you have to re-derive every time you read it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from scratch_llm.eval.vibe import VibeRun

Stage = Literal["pretrain", "midtrain", "sft", "reference"]
Scope = Literal["core", "val", "vibe"]

#: Metrics a card of each stage MUST carry. Absent ⇒ MissingMetric at construction.
#: ``reference`` is the production-floor card (a public checkpoint scored on the same suite): it
#: has a CORE number and nothing else, because nobody else's held-out split or chat template is ours.
REQUIRED_METRICS: dict[str, frozenset[str]] = {
    "pretrain": frozenset({"val_bpb", "core"}),
    "midtrain": frozenset({"val_bpb", "core"}),
    "sft": frozenset({"val_bpb", "core", "vibe_eot_rate", "vibe_deterministic"}),
    "reference": frozenset({"core"}),
}

_SCOPE_FIELDS: dict[str, tuple[str, ...]] = {
    "core": ("core_tasks", "core_limit", "core_bundle"),
    "val": ("val_split_id", "val_num_bytes"),
    "vibe": ("vibe_set_id", "vibe_prompts_hash", "vibe_decode_id"),
}

_SCOPE_METRICS: dict[str, tuple[str, ...]] = {
    "core": ("core",),
    "val": ("val_bpb",),
    "vibe": ("vibe_eot_rate", "vibe_mean_new_tokens", "vibe_truncated_rate"),
}


class MissingMetric(KeyError):
    """A metric the card must carry is absent. Raised instead of returning ``None``."""


class IncomparableCards(ValueError):
    """Two cards were evaluated differently in the requested scope, so they cannot be compared."""


@dataclass(frozen=True)
class EvalProtocol:
    """The identity of the *evaluation* — never of the model. Fingerprinted per scope.

    Unset scopes use the empty sentinel (``""`` / ``0``): a reference card that was never scored on
    our held-out split says so, and asking to compare it in the ``val`` scope raises rather than
    silently comparing two different splits.
    """

    core_tasks: tuple[str, ...] = ()
    core_limit: int | None = None
    core_bundle: str = ""
    val_split_id: str = ""
    val_num_bytes: int = 0
    vibe_set_id: str = ""
    vibe_prompts_hash: str = ""
    vibe_decode_id: str = ""

    def has(self, scope: Scope) -> bool:
        """True iff this protocol actually carries an evaluation in ``scope``."""
        if scope == "core":
            return bool(self.core_tasks)
        if scope == "val":
            return bool(self.val_split_id) and self.val_num_bytes > 0
        return bool(self.vibe_set_id) and bool(self.vibe_decode_id)

    def fingerprint(self, scope: Scope) -> str:
        payload = {name: getattr(self, name) for name in _SCOPE_FIELDS[scope]}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def differences(self, other: EvalProtocol, scope: Scope) -> list[str]:
        """Field names that differ in ``scope`` — the body of every refusal message."""
        return [
            name for name in _SCOPE_FIELDS[scope] if getattr(self, name) != getattr(other, name)
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "core_tasks": list(self.core_tasks),
            "core_limit": self.core_limit,
            "core_bundle": self.core_bundle,
            "val_split_id": self.val_split_id,
            "val_num_bytes": self.val_num_bytes,
            "vibe_set_id": self.vibe_set_id,
            "vibe_prompts_hash": self.vibe_prompts_hash,
            "vibe_decode_id": self.vibe_decode_id,
            "fingerprints": {s: self.fingerprint(s) for s in ("core", "val", "vibe")},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> EvalProtocol:
        limit = d.get("core_limit")
        return cls(
            core_tasks=tuple(str(t) for t in list(d.get("core_tasks", []))),  # type: ignore[arg-type]
            core_limit=None if limit is None else int(limit),  # type: ignore[arg-type]
            core_bundle=str(d.get("core_bundle", "")),
            val_split_id=str(d.get("val_split_id", "")),
            val_num_bytes=int(d.get("val_num_bytes", 0)),  # type: ignore[arg-type]
            vibe_set_id=str(d.get("vibe_set_id", "")),
            vibe_prompts_hash=str(d.get("vibe_prompts_hash", "")),
            vibe_decode_id=str(d.get("vibe_decode_id", "")),
        )


def split_id(token_bytes: bytes, name: str) -> str:
    """A stable id for a held-out split: its name plus a hash of the exact bytes scored.

    Two cards may only be compared on ``val_bpb`` if this matches. bpb is tokenizer-invariant by
    construction (bits over *bytes*), but it is not split-invariant, and "our bpb fell" across a
    silent change of validation shard is the single easiest self-deception in a speedrun.
    """
    return f"{name}:{hashlib.sha256(token_bytes).hexdigest()[:16]}"


@dataclass(frozen=True)
class CheckpointCard:
    """One checkpoint, one evaluation protocol, every required metric present."""

    checkpoint_id: str
    stage: Stage
    step: int
    n_params: int
    tokens_seen: int
    protocol: EvalProtocol
    metrics: Mapping[str, float]
    core_tasks: Mapping[str, float] = field(default_factory=dict)
    vibe: VibeRun | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in REQUIRED_METRICS:
            raise ValueError(
                f"unknown stage {self.stage!r}; expected one of {sorted(REQUIRED_METRICS)}"
            )
        if self.tokens_seen <= 0:
            raise ValueError(
                f"{self.checkpoint_id}: tokens_seen must be positive — it is the compute axis the "
                "d20 target is stated on, and it is NOT recoverable from a checkpoint file "
                "(train.save_checkpoint stores model+config+step only). Pass it from the run manifest."
            )
        missing = sorted(REQUIRED_METRICS[self.stage] - set(self.metrics))
        if missing:
            raise MissingMetric(
                f"{self.checkpoint_id} ({self.stage}): missing required metric(s) {missing}. "
                "A report card with a blank cell is a card that reports whatever happened to "
                "compute; re-run the evaluation that produces them."
            )
        for name, value in self.metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise MissingMetric(
                    f"{self.checkpoint_id}: metric {name!r} is {value!r}, not a finite number."
                )

    def metric(self, name: str) -> float:
        """The metric, or :class:`MissingMetric`. There is no ``.get(name, None)`` on purpose."""
        if name not in self.metrics:
            raise MissingMetric(
                f"{self.checkpoint_id} ({self.stage}) has no metric {name!r}; it carries "
                f"{sorted(self.metrics)}."
            )
        return float(self.metrics[name])

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "stage": self.stage,
            "step": self.step,
            "n_params": self.n_params,
            "tokens_seen": self.tokens_seen,
            "protocol": self.protocol.to_dict(),
            "metrics": dict(self.metrics),
            "core_tasks": dict(self.core_tasks),
            "vibe": None if self.vibe is None else self.vibe.to_dict(),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> CheckpointCard:
        vibe = d.get("vibe")
        stage = str(d["stage"])
        if stage not in REQUIRED_METRICS:
            raise ValueError(f"unknown stage {stage!r}")
        return cls(
            checkpoint_id=str(d["checkpoint_id"]),
            stage=stage,  # type: ignore[arg-type]
            step=int(d["step"]),  # type: ignore[arg-type]
            n_params=int(d["n_params"]),  # type: ignore[arg-type]
            tokens_seen=int(d["tokens_seen"]),  # type: ignore[arg-type]
            protocol=EvalProtocol.from_dict(dict(d["protocol"])),  # type: ignore[arg-type]
            metrics={str(k): float(v) for k, v in dict(d["metrics"]).items()},  # type: ignore[arg-type]
            core_tasks={str(k): float(v) for k, v in dict(d.get("core_tasks", {})).items()},  # type: ignore[arg-type]
            vibe=None if vibe is None else VibeRun.from_dict(dict(vibe)),  # type: ignore[arg-type]
            provenance={str(k): str(v) for k, v in dict(d.get("provenance", {})).items()},  # type: ignore[arg-type]
        )

    def to_table(self) -> str:
        """A plain-text block for stdout — the human-facing face of the card."""
        head = (
            f"{self.checkpoint_id} · stage={self.stage} · step={self.step} · "
            f"params={self.n_params:,} · tokens_seen={self.tokens_seen:,}"
        )
        rows = [f"  {name:<24} {self.metrics[name]:.6f}" for name in sorted(self.metrics)]
        fps = "  protocol " + " ".join(
            f"{s}={self.protocol.fingerprint(s) if self.protocol.has(s) else '-'}"
            for s in ("core", "val", "vibe")
        )
        return "\n".join([head, *rows, fps])


def save_cards(cards: Sequence[CheckpointCard], out: str | Path) -> None:
    """Write the cards as one JSON array — the run's durable artifact."""
    payload = [c.to_dict() for c in cards]
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_cards(src: str | Path) -> list[CheckpointCard]:
    raw = json.loads(Path(src).read_text(encoding="utf-8"))
    return [CheckpointCard.from_dict(d) for d in raw]


def assert_comparable(a: CheckpointCard, b: CheckpointCard, scope: Scope) -> None:
    """Raise :class:`IncomparableCards` unless ``a`` and ``b`` share the ``scope`` protocol."""
    for card in (a, b):
        if not card.protocol.has(scope):
            raise IncomparableCards(
                f"{card.checkpoint_id} carries no {scope} evaluation — nothing to compare in that "
                "scope. Evaluate it on the same protocol first."
            )
    diffs = a.protocol.differences(b.protocol, scope)
    if diffs:
        detail = ", ".join(
            f"{name}: {getattr(a.protocol, name)!r} vs {getattr(b.protocol, name)!r}"
            for name in diffs
        )
        raise IncomparableCards(
            f"{a.checkpoint_id} and {b.checkpoint_id} were evaluated on different {scope} "
            f"protocols ({detail}). Re-evaluate both under one protocol; a delta across two "
            "protocols measures the protocol."
        )


def compare(
    a: CheckpointCard,
    b: CheckpointCard,
    *,
    scope: Scope,
    metrics: Sequence[str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """``{metric: (a, b, b − a)}`` for the scope's metrics, after refusing incomparable cards."""
    assert_comparable(a, b, scope)
    names = tuple(metrics) if metrics is not None else _SCOPE_METRICS[scope]
    out: dict[str, tuple[float, float, float]] = {}
    for name in names:
        if name in a.metrics and name in b.metrics:
            av, bv = a.metric(name), b.metric(name)
            out[name] = (av, bv, bv - av)
    if not out:
        raise MissingMetric(
            f"neither card carries any of {list(names)} — nothing to compare in scope {scope!r}."
        )
    return out


# -----------------------------------------------------------------------------------------------
# The d20 gate. Structural clauses need no numbers; threshold clauses need D20Target, and every
# one of its fields is mandatory. The plan (§05, row T-R4) states the target as the word
# "checkpoint" — the numeric target is not written anywhere in it, so it is not written here either.
# -----------------------------------------------------------------------------------------------

_TARGET_FIELDS: tuple[str, ...] = (
    "min_core",
    "max_val_bpb",
    "min_tokens_seen",
    "min_vibe_eot_rate",
)


@dataclass(frozen=True)
class D20Target:
    """The four numbers "at the d20 target" resolves to. No defaults: every field must be given."""

    min_core: float
    max_val_bpb: float
    min_tokens_seen: int
    min_vibe_eot_rate: float

    @classmethod
    def from_mapping(cls, d: Mapping[str, object], *, source: str = "<mapping>") -> D20Target:
        """Build from JSON. A missing or null field is an error that names it — not a default.

        A silent default here would be the whole rung's failure mode in one line: the checkpoint
        would be declared "at the d20 target" against a threshold nobody chose.
        """
        unset = [k for k in _TARGET_FIELDS if d.get(k) is None]
        if unset:
            raise ValueError(
                f"{source}: d20 target field(s) {unset} are unset. These are Huy's numbers — the "
                "plan states T-R4's target as the word 'checkpoint' and no value; write them once "
                "in the target file, before the run, alongside the ledger PREDICTION."
            )
        return cls(
            min_core=float(d["min_core"]),  # type: ignore[arg-type]
            max_val_bpb=float(d["max_val_bpb"]),  # type: ignore[arg-type]
            min_tokens_seen=int(d["min_tokens_seen"]),  # type: ignore[arg-type]
            min_vibe_eot_rate=float(d["min_vibe_eot_rate"]),  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, src: str | Path) -> D20Target:
        return cls.from_mapping(json.loads(Path(src).read_text(encoding="utf-8")), source=str(src))


_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": lambda x, t: x >= t,
    "<=": lambda x, t: x <= t,
    "==": lambda x, t: x == t,
}


@dataclass(frozen=True)
class GateClause:
    """One yes/no term of the gate. ``measured``/``threshold`` are floats or fingerprint strings."""

    name: str
    measured: float | str
    op: str
    threshold: float | str
    passed: bool
    source: str

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name:<22} {self.measured} {self.op} {self.threshold}  ({self.source})"


@dataclass(frozen=True)
class GateResult:
    """The verdict. ``passed`` is the conjunction — one failing clause fails the rung's exit."""

    clauses: tuple[GateClause, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.clauses)

    def failures(self) -> tuple[GateClause, ...]:
        return tuple(c for c in self.clauses if not c.passed)

    def to_table(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return "\n".join([f"d20 gate: {verdict}", *(c.line() for c in self.clauses)])

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "clauses": [
                {
                    "name": c.name,
                    "measured": c.measured,
                    "op": c.op,
                    "threshold": c.threshold,
                    "passed": c.passed,
                    "source": c.source,
                }
                for c in self.clauses
            ],
        }


def _numeric(name: str, measured: float, op: str, threshold: float, source: str) -> GateClause:
    return GateClause(name, measured, op, threshold, _OPS[op](measured, threshold), source)


def _structural(name: str, measured: object, threshold: object, source: str) -> GateClause:
    return GateClause(name, str(measured), "==", str(threshold), measured == threshold, source)


def d20_gate(base: CheckpointCard, chat: CheckpointCard, target: D20Target) -> GateResult:
    """The rung's exit predicate over the base (pretrain) and chat (SFT) cards.

    Structural clauses first — they need no thresholds and they are the ones that catch a run
    that measured two different things: the two cards must share the core and val protocols, the
    SFT checkpoint must be the same architecture trained further, and its vibe re-run must have
    been byte-identical (``vibe_deterministic == 1``, produced by running the fixed set twice).

    Threshold clauses second. CORE and val_bpb are read off the **base** card: the published d20
    anchor is a base-model CORE, and instruction tuning is free to move CORE in either direction —
    gating the headline on the SFT card would score two different things at once. The SFT card's
    own clause is turn termination, which is exactly what the assistant mask teaches.

    Comparisons are exact: ``>=`` passes at equality, fails one ulp below. No epsilon.
    """
    clauses = [
        _structural(
            "protocol_core",
            base.protocol.fingerprint("core"),
            chat.protocol.fingerprint("core"),
            "base vs chat",
        ),
        _structural(
            "protocol_val",
            base.protocol.fingerprint("val"),
            chat.protocol.fingerprint("val"),
            "base vs chat",
        ),
        _structural("same_architecture", base.n_params, chat.n_params, "base vs chat"),
        GateClause(
            "tokens_monotone",
            chat.tokens_seen,
            ">=",
            base.tokens_seen,
            chat.tokens_seen >= base.tokens_seen,
            "base vs chat",
        ),
        _numeric(
            "vibe_deterministic",
            chat.metric("vibe_deterministic"),
            "==",
            1.0,
            chat.checkpoint_id,
        ),
        _numeric("core", base.metric("core"), ">=", target.min_core, base.checkpoint_id),
        _numeric("val_bpb", base.metric("val_bpb"), "<=", target.max_val_bpb, base.checkpoint_id),
        _numeric(
            "tokens_seen",
            float(base.tokens_seen),
            ">=",
            float(target.min_tokens_seen),
            base.checkpoint_id,
        ),
        _numeric(
            "vibe_eot_rate",
            chat.metric("vibe_eot_rate"),
            ">=",
            target.min_vibe_eot_rate,
            chat.checkpoint_id,
        ),
    ]
    return GateResult(tuple(clauses))


__all__ = [
    "REQUIRED_METRICS",
    "CheckpointCard",
    "D20Target",
    "EvalProtocol",
    "GateClause",
    "GateResult",
    "IncomparableCards",
    "MissingMetric",
    "Scope",
    "Stage",
    "assert_comparable",
    "compare",
    "d20_gate",
    "load_cards",
    "save_cards",
    "split_id",
]
