"""The predict-vs-measure ledger — make ``predict-before-run`` a data discipline, not a good intention.

Every measured artifact gets a record: the bound + number you *predicted* (before the run), the
*measured* result, the % of the roofline it hit, and a one-line root cause. Records append to a JSONL
file so the history is durable and diff-able, and :func:`check_regressions` turns "is the speedup real /
did it rot?" into a test the next run must pass. This is the artifact a principal review opens first.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Record:
    """One measured artifact. ``predicted_*`` are written BEFORE the run (the falsifiable anchor)."""

    artifact: str  # e.g. "decode-attention:split-k"
    gpu: str
    dtype: str
    predicted_bound: str  # "compute" | "memory"
    predicted_seconds: float
    measured_seconds: float
    pct_of_roof: float  # predicted/measured, in [0, 1+]
    date: str  # ISO date — passed in (no wall-clock here, for reproducibility)
    root_cause: str = ""  # one line: why the gap (launch / occupancy / bank-conflict / ...)
    notes: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def append(record: Record, path: str | Path) -> None:
    """Append one record as a JSON line, creating the file/dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load(path: str | Path) -> list[Record]:
    """Load all records from a JSONL ledger (empty list if the file does not exist)."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[Record] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(Record(**json.loads(line)))
    return out


@dataclass(frozen=True)
class Regression:
    """A measured slowdown of one artifact vs its best prior measurement."""

    artifact: str
    best_seconds: float
    latest_seconds: float
    slowdown: float  # latest/best (>1 = slower)


def check_regressions(records: Iterable[Record], *, tolerance: float = 0.10) -> list[Regression]:
    """Flag any artifact whose latest measured time is >``tolerance`` slower than its best prior one.

    "Best" = the minimum measured time seen for that artifact before the latest; a kernel that silently
    rots (an autotune regression, a dtype slip) trips this. Returns one :class:`Regression` per offender.
    """
    by_artifact: dict[str, list[Record]] = {}
    for r in records:
        by_artifact.setdefault(r.artifact, []).append(r)

    flagged: list[Regression] = []
    for artifact, recs in by_artifact.items():
        if len(recs) < 2:
            continue
        *prior, latest = recs  # insertion order = chronological (append-only ledger)
        best = min(p.measured_seconds for p in prior)
        if latest.measured_seconds > best * (1.0 + tolerance):
            flagged.append(
                Regression(
                    artifact=artifact,
                    best_seconds=best,
                    latest_seconds=latest.measured_seconds,
                    slowdown=latest.measured_seconds / best,
                )
            )
    return flagged
