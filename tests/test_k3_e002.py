"""K3/E002 — the fla divergence sweep, gated in three tiers, in ascending cost.

  CPU      the grid's shape, the row schema, resume, the verdict, the figure, and the arithmetic
           of the runner against E001's paths directly. Milliseconds, no GPU, no fla.
  upstream the file:line citations the whole experiment rests on, read out of the pinned
           ``oss/fla`` checkout. Skips if the checkout is absent.
  gpu      one real fla cell. Needs a CUDA box with triton and fla installed.

The CPU tier carries more weight here than in a kernel rung because E002's failure modes are not
"wrong answer", they are "right answer about the wrong thing": a floor measured in the wrong
precision arm, a resume set that skips a cell it never ran, a heatmap coloured so that every bf16
row is red for reasons that have nothing to do with fla. Each of those is decidable on a laptop
and none of them announces itself on the box.

Spec: experiments/K3/E002/spec.md
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from scratch_llm.kernels.gemm.cuda._k1_loader import workspace_root
from scratch_llm.mastery import e002_fla as e002
from scratch_llm.mastery.divergence import _rel, make_inputs
from scratch_llm.mastery.divergence import sweep as e001_sweep
from scratch_llm.mastery.paths import chunked_wy

_E002 = workspace_root() / "experiments" / "K3" / "E002"
FIXTURE = _E002 / "fixture_sweep.jsonl"


def _load(name: str):
    """Import ``experiments/K3/E002/<name>.py`` by path.

    These two files live in the ladders workspace, not in this repo, and there is no package to
    import them from — ``experiments/`` is a directory of runners, not a library. Loading by path
    is what lets the runner keep living next to its spec.md and run.sh (where anyone looking for
    it will look) while still being tested by this suite.
    """
    path = _E002 / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"{path} absent")
    spec = importlib.util.spec_from_file_location(f"_e002_{name}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================================================
# THE HOLE — exactly one guard (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K3/E002", "src/scratch_llm/mastery/e002_fla.py")
def test_tolerance_ratio_decides_the_map() -> None:
    """Fails while ``tol_ratio()`` is unwritten; passes when Huy has chosen the ratio.

    The tolerance is the only thing E002 produces that an agent must not produce. Everything else
    here — the grid, the kernels, the floor, the figure — is machinery for putting one number in
    front of a person: the multiple of the dtype's own unit roundoff above which a divergence
    stops being arithmetic and starts being a bug. That number becomes the assertion in the
    regression test proposed to fla, so it is also the number that has to be defended in the PR
    thread, which is precisely why it is not delegated (CLAUDE.md, Division of labor).
    """
    ratio = e002.tol_ratio()
    rows = e002.read_rows(FIXTURE)
    assert rows, f"no fixture at {FIXTURE} — regenerate with sweep.py --fixture"
    counts = e002.verdict_counts(rows, ratio)

    if not math.isfinite(ratio):
        # LADDERS_STUB_HOLES=1: the placeholder exists so the plumbing can run, and it judges
        # NOTHING. A stub that produced EXPECTED would paint a green map from no evidence.
        assert counts["UNJUDGED"] == len(rows), counts
        return

    assert ratio >= 1.0, (
        f"tol_ratio() = {ratio}: a threshold below one unit roundoff cannot be met by any dtype, "
        "so every cell would be a BUG. That is arithmetic, not a tolerance choice."
    )
    assert counts["UNJUDGED"] < len(rows), (
        "the ratio judges nothing on the fixture — check err_over_eps is populated"
    )


# =============================================================================================
# CPU — the grid. A union, not a cross product, and the difference is load-bearing.
# =============================================================================================


def test_grid_is_the_union_the_docstring_claims() -> None:
    """9 gates × 3 dtypes × (2 precisions at C=64 + 1 each at C=16, C=32) + 9 × 3 recurrent."""
    cells = e002.grid()
    chunk = [c for c in cells if c.path == "chunk"]
    rec = [c for c in cells if c.path == "fused_recurrent"]
    n_g, n_d = len(e002.GATES), len(e002.SWEEP_DTYPES)
    assert len(chunk) == n_g * n_d * 4
    assert len(rec) == n_g * n_d
    assert len(cells) == len(chunk) + len(rec)


def test_fused_recurrent_carries_neither_a_chunk_size_nor_a_solve_precision() -> None:
    """It has no WY representation, so it has no triangular solve to run in either precision.

    Giving it those fields anyway would produce four identical rows per (gate, dtype) — a heatmap
    with four identical rows reads as a reproducible measurement and is a duplicated one.
    """
    for c in e002.grid():
        if c.path == "fused_recurrent":
            assert c.chunk_size is None
            assert c.solve_precision == e002.NO_SOLVE


def test_chunk_16_and_32_are_never_labelled_tf32() -> None:
    """Upstream fixes the unfused solve at IEEE (``fla/ops/utils/solve_tril.py:19-22``); the fused
    kernel that reads ``SOLVE_TRIL_DOT_PRECISION`` is only reached at BT == 64
    (``chunk_fwd.py:381``). Asking for tf32 at 16 or 32 therefore cannot produce a tf32 row — only
    a mislabelled IEEE one, which is the single worst kind of row a divergence map can contain.
    """
    for c in e002.grid(solve_precisions=("tf32", "ieee")):
        if c.path == "chunk" and c.chunk_size != e002.FUSED_CHUNK_SIZE:
            assert c.solve_precision == "ieee", c
    only_tf32 = e002.grid(solve_precisions=("tf32",), chunk_sizes=(16, 32), paths=("chunk",))
    assert only_tf32 == [], "a tf32-only request at C=16/32 must yield no cells, not fake ones"


def test_gate_column_is_e001s_own_and_includes_the_reported_case() -> None:
    """Read out of ``divergence.sweep``'s signature, not retyped.

    E002 exists to check whether E001's pure-PyTorch result survives contact with fla's kernels.
    That comparison is only a comparison if both experiments are evaluated at the same gates, and
    two hand-maintained tuples drift the first time either is edited.
    """
    e001_gates = tuple(inspect.signature(e001_sweep).parameters["log_gates"].default)
    assert e001_gates == e002.GATES
    assert e002.GATE_AT_ONE in e002.GATES, "log_gate = 0 is the fla #104/#389 case"


def test_floor_is_the_most_accurate_configuration_fla_has() -> None:
    """fp32 operands, IEEE solve, C=64 — and it is a cell of the grid, not a separate code path."""
    assert (e002.FLOOR_PATH, e002.FLOOR_DTYPE, e002.FLOOR_SOLVE_PRECISION) == (
        "chunk",
        "float32",
        "ieee",
    )
    assert any(c.is_floor() for c in e002.grid())
    most_accurate = min(e002.SWEEP_DTYPES, key=e002.unit_eps)
    assert most_accurate == e002.FLOOR_DTYPE, "the floor must be the dtype with the smallest eps"


# =============================================================================================
# CPU — the metric and the verdict
# =============================================================================================


def test_err_over_eps_puts_every_dtype_on_one_ruler() -> None:
    """The reason the map is coloured by err/eps and not by raw relative error.

    1e-3 is a catastrophe in fp32 and beneath notice in bf16. Normalised, both configurations
    "doing what their dtype does" land at the same place, and only a departure from that shows.
    """
    assert e002.unit_eps("bfloat16") > e002.unit_eps("float16") > e002.unit_eps("float32")
    for dt in e002.SWEEP_DTYPES:
        assert e002.err_over_eps(e002.unit_eps(dt), dt) == pytest.approx(1.0), dt
    assert e002.err_over_eps(float("nan"), "bfloat16") is None
    assert e002.err_over_eps(None, "bfloat16") is None


@pytest.mark.parametrize(
    ("err_over_eps", "ratio", "want"),
    [
        (1.0, 8.0, "EXPECTED"),
        (64.0, 8.0, "BUG"),
        (8.0, 8.0, "EXPECTED"),  # the boundary is inclusive: "> ratio" is a bug, "= ratio" is not
        (1.0, None, "UNJUDGED"),
        (1.0, math.inf, "UNJUDGED"),
        (None, 8.0, "UNJUDGED"),
        (float("nan"), 8.0, "UNJUDGED"),
    ],
)
def test_classify(err_over_eps: float | None, ratio: float | None, want: str) -> None:
    assert e002.classify({"err_over_eps": err_over_eps}, ratio) == want


def test_unjudged_is_never_silently_expected() -> None:
    """The one invariant of the verdict: absence of evidence is not evidence of correctness.

    A cell that failed to run, and a run made before the tolerance exists, must both come out as
    UNJUDGED. Collapsing either into EXPECTED puts a green square on the map where there is
    nothing at all — and the stub tolerance (``LADDERS_STUB_HOLES=1`` returns infinity) is exactly
    the case that would otherwise do it for every cell at once.
    """
    errored = {"err_over_eps": None, "error": "RuntimeError: whatever"}
    assert e002.classify(errored, 8.0) == "UNJUDGED"
    assert e002.classify({"err_over_eps": 1e-9}, math.inf) == "UNJUDGED"


# =============================================================================================
# CPU — the row schema and resume
# =============================================================================================


def _row(**over) -> dict:
    cell = e002.Cell("cpu-proxy", "chunk", 0.0, "bfloat16", 64, "ieee", 32, 8, 0)
    # `dict[str, Any]`: these are make_row's kwargs, whose types differ per key. Inference would
    # join them into one union and then reject every keyword it is splatted into.
    kw: dict[str, Any] = dict(
        rel_err_vs_floor=1e-3,
        max_abs_err_vs_floor=1e-3,
        rel_err_vs_fp64=1e-3,
        floor_norm=1.0,
        provenance={
            "arch": "sm90",
            "device": "H100",
            "driver": "12.8",
            "torch": "2.9",
            "triton": "3.5",
        },
        ts="2026-09-07T00:00:00",
    )
    kw.update(over)
    return e002.make_row(cell, **kw)


def test_row_schema_is_checked_on_write_not_on_read(monkeypatch) -> None:
    """A resumable sweep never re-measures a cell it has already written, so a malformed row is
    permanent. Validating at write time is the difference between failing now and a hole in the
    map that nothing ever reports.

    The drift half is simulated by adding a field to ROW_FIELDS: that is what a future edit to
    :class:`Cell` or :func:`make_row` looks like from here, and the check has to catch it rather
    than write a row the figure cannot read.
    """
    row = _row()
    assert set(row) == set(e002.ROW_FIELDS)
    assert row["eps"] == e002.unit_eps("bfloat16")
    assert row["err_over_eps"] == pytest.approx(1e-3 / e002.unit_eps("bfloat16"))
    monkeypatch.setattr(e002, "ROW_FIELDS", (*e002.ROW_FIELDS, "a_field_nobody_populates"))
    with pytest.raises(ValueError, match="row schema drift"):
        _row()


def test_row_key_separates_the_same_cell_on_two_architectures() -> None:
    """E002's exit is "on 3 archs". If the arch were not part of a row's identity, the sm100 run
    would resume into the sm90 rows and record nothing at all."""
    a, b = _row(), _row(provenance={"arch": "sm100"})
    assert e002.row_key(a) != e002.row_key(b)
    assert e002.row_key(a).startswith(
        e002.Cell("cpu-proxy", "chunk", 0.0, "bfloat16", 64, "ieee", 32, 8, 0).key()
    )


def test_jsonl_round_trip_and_a_bad_line_is_loud(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    e002.append_row(p, _row())
    e002.append_row(p, _row(provenance={"arch": "sm100"}))
    assert len(e002.read_rows(p)) == 2
    assert e002.done_keys(p) == {
        e002.row_key(_row()),
        e002.row_key(_row(provenance={"arch": "sm100"})),
    }
    assert e002.read_rows(tmp_path / "absent.jsonl") == [], "a first run is empty, not an error"
    p.write_text(p.read_text() + "{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a JSON row"):
        e002.read_rows(p)


def test_an_errored_cell_counts_as_done(tmp_path: Path) -> None:
    """Otherwise a sweep containing one unsupported dtype re-attempts it on every resume, and the
    recorded rejection — itself a finding about fla — never settles."""
    p = tmp_path / "s.jsonl"
    e002.append_row(p, _row(rel_err_vs_floor=None, error="ValueError: unsupported"))
    assert len(e002.done_keys(p)) == 1


# =============================================================================================
# CPU — the runner, end to end, on E001's paths (the cpu-proxy backend)
# =============================================================================================

_TINY = [
    "--backend",
    "cpu-proxy",
    "--gates",
    "0.0,-0.1",
    "--dtypes",
    "float32,bfloat16",
    "--chunks",
    "16,64",
    "--seq-len",
    "32",
    "--d-head",
    "8",
]


def test_dry_run_prints_the_grid_size_and_runs_nothing(tmp_path, capsys) -> None:
    sweep = _load("sweep")
    out = tmp_path / "s.jsonl"
    assert sweep.main([*_TINY, "--out", str(out), "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "E002 grid" in text and "cells" in text
    assert "UNION" in text, "the grid's shape must be stated where it is counted"
    assert not out.exists(), "--dry-run wrote a sweep file"


def test_runner_is_resumable(tmp_path, capsys) -> None:
    """The property a rented hour depends on: a second run adds nothing and re-measures nothing."""
    sweep = _load("sweep")
    out = tmp_path / "s.jsonl"
    assert sweep.main([*_TINY, "--out", str(out)]) == 0
    first = e002.read_rows(out)
    assert first, "the cpu-proxy run produced no rows"
    assert sweep.main([*_TINY, "--out", str(out)]) == 0
    assert e002.read_rows(out) == first
    assert "0 to run" in capsys.readouterr().out


def test_runner_matches_e001s_paths_computed_directly(tmp_path) -> None:
    """R5's slow reference for this runner: the same divergence, recomputed by hand from E001.

    The runner does three things that could each be silently wrong — pick the floor cell, cast the
    operands, and take the ratio. This recomputes all three from ``chunked_wy`` directly and
    demands the same number, so a change to any of them fails here rather than on a rented box.
    """
    sweep = _load("sweep")
    out = tmp_path / "s.jsonl"
    args = [
        "--backend",
        "cpu-proxy",
        "--gates",
        "-0.1",
        "--dtypes",
        "bfloat16",
        "--chunks",
        "16",
        "--seq-len",
        "32",
        "--d-head",
        "8",
        "--out",
        str(out),
    ]
    assert sweep.main(args) == 0
    row = next(r for r in e002.read_rows(out) if r["path"] == "chunk" and r["chunk_size"] == 16)

    q, k, v, la, b = make_inputs(32, 8, 8, -0.1, 0, device="cpu")
    floor, _ = chunked_wy(
        q.float(),
        k.float(),
        v.float(),
        la.float(),
        b.float(),
        chunk_size=e002.FUSED_CHUNK_SIZE,
        solve_dtype=torch.float64,
    )
    got, _ = chunked_wy(
        q.bfloat16(),
        k.bfloat16(),
        v.bfloat16(),
        la.bfloat16(),
        b.bfloat16(),
        chunk_size=16,
        solve_dtype=torch.float64,
    )
    assert row["rel_err_vs_floor"] == pytest.approx(
        _rel(got.to(torch.float64), floor.to(torch.float64)), rel=1e-12
    )


def test_a_cpu_proxy_row_never_carries_an_fla_commit(tmp_path) -> None:
    """The proxy is E001's pure PyTorch. It never calls fla, so an fla revision on its row would be
    the one field a reader needs to quote a laptop number as an upstream measurement at a named
    commit — the ``backend`` label says "not fla", and the provenance must not contradict it."""
    sweep = _load("sweep")
    out = tmp_path / "s.jsonl"
    assert sweep.main([*_TINY, "--out", str(out)]) == 0
    rows = e002.read_rows(out)
    assert rows
    for r in rows:
        assert r["fla_commit"] is None, r


def test_a_missing_floor_aborts_the_arm_instead_of_poisoning_the_resume_set(
    tmp_path, capsys, monkeypatch
) -> None:
    """The failure that would cost the experiment rather than the hour.

    An error row counts as DONE (``done_keys``), which is right for a cell fla rejects and fatal
    for a floor that could not be computed: writing one per cell would mark every cell at that
    gate permanently measured, and no later resume — not even a correct one with the IEEE arm run
    first — would ever fill them in. So the arm aborts, having written nothing.
    """
    sweep = _load("sweep")
    out = tmp_path / "s.jsonl"

    def no_floor(*_a, **_k):
        raise RuntimeError("no floor tensor and this is the 'tf32' arm, which cannot compute one")

    monkeypatch.setattr(sweep, "get_floor", no_floor)
    assert sweep.main([*_TINY, "--out", str(out)]) == 2
    assert e002.read_rows(out) == [], "an unmeasurable floor was recorded as a measured cell"
    text = capsys.readouterr().out
    assert "ABORT" in text and "resume set is intact" in text
    assert "--solve-precision ieee" in text, "the abort must print the command that fixes it"


def test_fla_backend_self_skips_without_cuda_and_prints_the_box_command(
    tmp_path, capsys, monkeypatch
) -> None:
    """A CPU "result" for a Triton-only kernel is worse than no result: it would enter the map
    wearing an arch label and be quoted upstream. So the fla backend refuses, exits 0 (this is not
    a failure, it is the wrong host), and prints the commands that do produce the number."""
    sweep = _load("sweep")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "s.jsonl"
    assert sweep.main(["--backend", "fla", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "SKIPPED" in text and "no CUDA device" in text
    assert "infra/bench.sh -l K3 -r E002" in text
    assert "experiments/K3/E002/plot.py" in text
    assert not out.exists()


def test_the_tf32_arm_cannot_invent_a_floor(tmp_path) -> None:
    """The subtlest way this experiment could produce beautiful wrong numbers.

    ``SOLVE_TRIL_DOT_PRECISION`` is a process-global, so inside the tf32 child EVERY chunk call is
    tf32 — including one asking for the fp32/IEEE floor. Recomputing the floor there would measure
    each tf32 deviation against a tf32 baseline: small, consistent, and meaningless. So the tf32
    arm may only LOAD a floor the IEEE arm wrote.
    """
    sweep = _load("sweep")
    cell = e002.Cell("fla", "chunk", 0.0, "bfloat16", 64, "tf32", 32, 8, 0)
    with pytest.raises(RuntimeError, match="cannot compute one"):
        sweep.get_floor(tmp_path / "s.jsonl", cell, "sm90", "tf32", lambda c: (None, None))


def test_the_ieee_arm_runs_first(tmp_path) -> None:
    """...which is what makes the rule above satisfiable rather than a deadlock."""
    sweep = _load("sweep")
    parser = sweep.build_parser()
    args = parser.parse_args(["--solve-precision", "tf32,ieee"])
    assert (
        tuple(
            sorted(
                dict.fromkeys(args.solve_precision.split(",")),
                key=lambda p: p != e002.FLOOR_SOLVE_PRECISION,
            )
        )[0]
        == e002.FLOOR_SOLVE_PRECISION
    )


# =============================================================================================
# CPU — the tf32/ieee arm detector
# =============================================================================================


def test_identical_arms_are_reported_as_a_failed_experiment_not_a_finding() -> None:
    """Two bit-identical arms mean the override never reached the kernel — almost always a shared
    Triton cache. That looks exactly like "solve precision does not matter", which is a publishable
    and completely wrong conclusion, so it gets its own detector."""
    base = dict(
        backend="fla",
        path="chunk",
        log_gate=0.0,
        dtype="bfloat16",
        seed=0,
        chunk_size=64,
        arch="sm90",
    )
    same = [{**base, "solve_precision": p, "rel_err_vs_floor": 1e-3} for p in ("tf32", "ieee")]
    ok, why = e002.arms_are_distinguishable(same)
    assert not ok and "bit-identical" in why

    diff = [
        {**base, "solve_precision": "tf32", "rel_err_vs_floor": 2e-3},
        {**base, "solve_precision": "ieee", "rel_err_vs_floor": 1e-3},
    ]
    ok, why = e002.arms_are_distinguishable(diff)
    assert ok and "differ" in why


def test_no_comparable_pair_is_not_an_all_clear() -> None:
    """ "We could not check" and "we checked and it is fine" must not print the same word."""
    ok, why = e002.arms_are_distinguishable([])
    assert not ok and "cannot be compared" in why


# =============================================================================================
# CPU — the fixture and the figure
# =============================================================================================


def test_fixture_exists_and_matches_the_schema() -> None:
    """The figure's test data. Committed because a fixture regenerated by the code under test is
    not a fixture, and because ``plot.py --fixture`` has to work on a box with no sweep yet."""
    rows = e002.read_rows(FIXTURE)
    assert rows, f"{FIXTURE} is empty — regenerate: sweep.py --fixture --out {FIXTURE}"
    for r in rows:
        assert set(r) == set(e002.ROW_FIELDS), sorted(set(r) ^ set(e002.ROW_FIELDS))
        assert r["backend"] == "cpu-proxy", "the fixture must never claim to be an fla measurement"
    assert e002.arms_are_distinguishable(rows)[0], "the fixture's own arms must be distinguishable"


def test_plot_renders_the_fixture_without_a_gpu(tmp_path) -> None:
    plot = _load("plot")
    rows = e002.read_rows(FIXTURE)
    written = plot.render(rows, str(tmp_path / "fig"), ratio=None)
    if not written:
        pytest.skip("matplotlib not installed")
    for p in written:
        assert Path(p).stat().st_size > 1000, f"{p} rendered but is empty"


def test_plot_text_map_labels_every_configuration() -> None:
    """The text map is what gets pasted into fla RFC #1155, so it has to be readable without the
    figure: every row says which path, which dtype, which chunk size, which precision."""
    plot = _load("plot")
    rows = e002.read_rows(FIXTURE)
    text = plot.text_map(rows, ratio=None)
    assert "chunk · float32 · C=64 · ieee" in text
    assert "fused_recurrent · float32" in text
    assert "C=" not in text.split("fused_recurrent · float32")[1].split("\n")[0]
    assert "eps" in text, "the units of the cell values must travel with the table"


def test_plot_facets_by_arch(tmp_path) -> None:
    """E002's exit criterion is "on 3 archs" — a claim about reproducibility that one averaged
    panel would erase. Three archs must render as three panels, not as a mean."""
    plot = _load("plot")
    rows = []
    for i, arch in enumerate(("sm90", "sm100", "sm120")):
        for r in e002.read_rows(FIXTURE):
            rows.append(
                {**r, "arch": arch, "rel_err_vs_floor": (r["rel_err_vs_floor"] or 0) * (i + 1)}
            )
    assert len({r["arch"] for r in rows}) == 3
    written = plot.render(rows, str(tmp_path / "three"), ratio=None)
    if not written:
        pytest.skip("matplotlib not installed")
    assert all(Path(p).stat().st_size > 1000 for p in written)


def test_plot_draws_absent_and_perfect_cells_differently_from_small_ones() -> None:
    """The floor's deviation from itself is exactly zero and has no logarithm; a cell that errored
    has no value at all. Both are absent from the colour scale. Rendering either as "very small"
    would put the most reassuring colour on the two cells carrying the least information."""
    plot = _load("plot")
    assert plot._value({"err_over_eps": 0.0}) is None
    assert plot._value({"err_over_eps": 4.0, "error": "boom"}) is None
    assert plot._value(None) is None
    assert plot._value({"err_over_eps": 100.0}) == pytest.approx(2.0)


def test_plot_refuses_an_empty_sweep(tmp_path, capsys) -> None:
    plot = _load("plot")
    empty = tmp_path / "none.jsonl"
    empty.write_text("", encoding="utf-8")
    assert plot.main(["--in", str(empty)]) == 1
    assert "no rows" in capsys.readouterr().out


# =============================================================================================
# upstream — the citations this whole experiment rests on, read out of the pinned checkout
# =============================================================================================


def _fla_root() -> Path:
    root = workspace_root() / "oss" / "fla"
    if not root.is_dir():
        pytest.skip(f"no fla checkout at {root} — run infra/oss.sh")
    return root


def test_the_tf32_ieee_split_is_where_we_say_it_is() -> None:
    """Plan §05 names ``chunk_fwd.py:20-23`` and the sweep's whole design follows from it. An
    upstream reshuffle that moved those lines would leave every comment in E002 citing a line that
    says something else — the quietest way for a write-up to become wrong."""
    for key, (rel, lo, hi, needle) in e002.CITATIONS.items():
        p = _fla_root() / rel
        assert p.is_file(), f"{key}: {p} missing"
        lines = p.read_text(encoding="utf-8").splitlines()
        window = "\n".join(lines[lo - 1 : hi])
        assert needle in window, (
            f"{key}: {rel}:{lo}-{hi} no longer contains {needle!r}. Re-read the file, fix the "
            f"citation in mastery/e002_fla.py and in experiments/K3/E002/spec.md, then re-run."
        )


def test_the_fused_solve_really_is_tf32_on_every_arch_this_plan_targets() -> None:
    """Why the sweep has to reach in and set a module global to get an IEEE arm at all: there is
    no supported way to ask fla for it on sm90/sm100/sm120."""
    src = (_fla_root() / "fla/ops/gated_delta_rule/chunk_fwd.py").read_text(encoding="utf-8")
    assert "if IS_TF32_SUPPORTED:" in src and "tl.constexpr('tf32')" in src
    dev = (_fla_root() / "fla/utils/_device.py").read_text(encoding="utf-8")
    assert "get_device_capability(0)[0] >= 8" in dev, (
        "IS_TF32_SUPPORTED is no longer 'Ampere or newer' — the tf32 arm may not be the default "
        "on every target arch any more, and the grid's premise needs re-reading"
    )


def test_the_citations_hold_at_the_declared_pin_too() -> None:
    """The checkout moves; the pin does not, and the file:line numbers in this experiment were
    read at the pin.

    ``infra/oss.sh`` leaves ``oss/fla`` on a PR branch, so on 2026-09-07 the checkout was already
    23 commits past ``FLA_PIN``. The test above proves the citations hold at HEAD; this one proves
    they also hold at the pin, which is what makes "the checkout drifted and the numbers are still
    comparable" a checked statement rather than an assumption.
    """
    root = _fla_root()
    for key, (rel, lo, hi, needle) in e002.CITATIONS.items():
        got = subprocess.run(
            ["git", "-C", str(root), "show", f"{e002.FLA_PIN}:{rel}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if got.returncode != 0:
            pytest.skip(f"{e002.FLA_PIN} not reachable in this checkout: {got.stderr.strip()[:90]}")
        window = "\n".join(got.stdout.splitlines()[lo - 1 : hi])
        assert needle in window, (
            f"{key}: {rel}:{lo}-{hi} does not contain {needle!r} at the pinned {e002.FLA_PIN}. "
            "The pin and the citation disagree — one of them is wrong."
        )


def test_the_fla_commit_stamped_on_a_row_is_read_from_the_checkout() -> None:
    """Provenance is a reading, not a declaration (invariant 4).

    A hard-coded commit records the author's belief about which fla ran. On a box whose checkout
    had moved, every row would carry a revision that never produced it, and two boxes' rows would
    be silently uncomparable while looking identical.
    """
    root = _fla_root()
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert head, "the fla checkout has no HEAD"
    assert e002.fla_head() == head


def test_chunk_sizes_are_the_ones_upstream_accepts() -> None:
    src = (_fla_root() / "fla/ops/gated_delta_rule/chunk.py").read_text(encoding="utf-8")
    assert "chunk_size not in (16, 32, 64)" in src
    assert set(e002.CHUNK_SIZES) == {16, 32, 64}


# =============================================================================================
# gpu — one real fla cell. Needs CUDA + triton + fla.
# =============================================================================================


@pytest.mark.gpu
def test_one_fla_cell_runs_and_lands_near_the_floor() -> None:
    """The smoke test for the fla adapter: shapes, scale, and the l2norm flag.

    No tolerance is asserted — that is the hole. What is asserted is that the call returns the
    oracle's shape and a finite divergence, which is the difference between "the adapter is wired
    correctly" and "the number is right".
    """
    sweep = _load("sweep")
    ok, why = sweep.fla_available()
    if not ok:
        pytest.skip(why)
    cell = e002.Cell("fla", "chunk", -0.1, "bfloat16", 64, "tf32", 128, 64, 0)
    out, _ = sweep.run_cell_fla(cell)
    assert out.shape == (128, 64)
    ref = sweep.fp64_oracle(cell)
    rel = _rel(out, ref)
    assert math.isfinite(rel), "fla returned a non-finite output at a benign gate"


@pytest.mark.gpu
def test_fused_recurrent_and_chunk_agree_more_closely_in_fp32_than_in_bf16() -> None:
    """The ordering the whole map assumes. If it does not hold, the floor is not a floor and every
    percentage in the sweep is measured against the wrong thing — check this before the grid."""
    sweep = _load("sweep")
    ok, why = sweep.fla_available()
    if not ok:
        pytest.skip(why)
    ref = None
    errs = {}
    for dt in ("float32", "bfloat16"):
        cell = e002.Cell("fla", "fused_recurrent", -0.1, dt, None, e002.NO_SOLVE, 128, 64, 0)
        out, _ = sweep.run_cell_fla(cell)
        ref = ref if ref is not None else sweep.fp64_oracle(cell)
        errs[dt] = _rel(out, ref)
    assert errs["float32"] < errs["bfloat16"], errs


def test_fixture_is_json_lines_not_a_json_document() -> None:
    """One complete write per cell is what makes the sweep survive a pod whose hour ran out."""
    text = FIXTURE.read_text(encoding="utf-8")
    assert not text.lstrip().startswith("["), "a JSON array would have to be rewritten every cell"
    for line in text.splitlines():
        if line.strip():
            json.loads(line)
