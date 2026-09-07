"""The hole contract — how a rung's one un-delegated function is represented in the test suite.

A *hole* is the single block the sixty-day plan (§08) never delegates: a kernel core body, the
loss/KL math, a memory model, a tolerance constant. Everything around it — harness, oracle,
launch glue, build wiring, the test itself — is written ahead of time, so that the moment Huy
fills the hole the rung runs end to end. Marking it in the test suite has two requirements that
pull in opposite directions:

  1. Today the test MUST fail. An unfilled hole that reports green is a lie the ledger cannot see.
  2. After Huy fills it the test MUST pass — against the oracle, at the spec's tolerance.

A plain ``xfail(strict=True)`` satisfies (1) and breaks (2): once the hole is filled the test
XPASSes, and strict XPASS is a failure. So the xfail is applied *conditionally*, keyed on whether
the hole is still open in the source. ``@pytest.mark.hole("<L>/<R>", "<source path>")`` declares
which source carries the hole; this hook reads that file and looks for the sentinel the hole
convention writes:

    Python:  raise NotImplementedError("HUY: ...")
    CUDA:    #error "HUY: ..."

Hole open  -> strict xfail. The test runs, fails, and reports ``xfailed``. Strict, so a stub that
              silently satisfies it is caught as XPASS.
Hole filled -> no marker. The test is an ordinary test and must pass on its own merits.

``LADDERS_STUB_HOLES=1`` is the third state: a trivially-correct slow placeholder stands in for the
hole so the *surrounding* plumbing (distributed launch, serving loop, sweep runner) can be exercised
on CPU. Under that flag no xfail is applied — the stub is supposed to make the test pass. The flag
never reaches a measurement: ``infra/bench.sh`` refuses to run while it is set.

``make holes`` (infra/holes.py) lists every hole and the test that guards it, joined on the rung id.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The sentinels the hole convention writes into a source. Either one present => hole still open.
_OPEN_SENTINELS = ('NotImplementedError("HUY:', '#error "HUY:')


def hole_is_open(source: str) -> bool:
    """True iff ``source`` (repo-relative) still carries an unfilled hole sentinel.

    A missing file counts as open: the rung's source has not been written yet, so the test that
    guards its hole cannot pass. Silently treating it as filled would flip the suite green on a
    typo in the marker's path.
    """
    p = _REPO_ROOT / source
    if not p.is_file():
        return True
    return any(s in p.read_text(encoding="utf-8", errors="replace") for s in _OPEN_SENTINELS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply the conditional strict xfail to every ``@pytest.mark.hole`` test."""
    if os.environ.get("LADDERS_STUB_HOLES") == "1":
        return  # the stub is meant to satisfy the test; an xfail would invert the signal
    for item in items:
        marker = item.get_closest_marker("hole")
        if marker is None:
            continue
        rung = marker.args[0] if marker.args else "?"
        sources = [a for a in marker.args[1:]] or []
        if not sources or any(hole_is_open(s) for s in sources):
            item.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    reason=(
                        f"HUY hole open for {rung} ({', '.join(sources) or 'source not declared'}) — "
                        f"this test is the gate on filling it; see experiments/{rung}/spec.md"
                    ),
                )
            )
