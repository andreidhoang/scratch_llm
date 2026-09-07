"""E002 — the grid, the row schema, and the verdict for the fla gate × dtype × chunk sweep.

E001 asked whether the gated delta rule's chunked and recurrent paths agree *in principle*, in
pure PyTorch, on a CPU. The answer was a property of the algebra. E002 asks the same question of
the code people actually ship — ``fla``'s two Triton kernels — on the three architectures the
sixty-day plan routes through (sm90 · sm100 · sm120), and turns the answer into a divergence map
and an upstream regression test (plan §05, K3, D28–D33).

WHY THIS FILE IS HERE AND NOT UNDER ``experiments/K3/E002/``
    ``tests/conftest.py`` resolves a ``@pytest.mark.hole`` source path against *this* repo's root.
    ``scratch_llm`` is a symlink inside the ladders workspace, so a hole living in
    ``ladders/experiments/K3/E002/sweep.py`` would resolve to a path that does not exist, count as
    permanently open, and pin its guard test to a strict xfail that can never turn green. So the
    grid, the schema and the tolerance live here; ``experiments/K3/E002/sweep.py`` is the runner
    and ``plot.py`` the figure. E001's own files are untouched — this module *imports* them, which
    is the point: every E002 number is comparable to an E001 number by construction.

WHAT THE SWEEP MEASURES
    For one (gate, dtype, chunk size, solve precision) cell, the relative deviation of that cell's
    output from THE FLOOR — ``fla``'s own chunk kernel at float32 with the triangular solve forced
    to IEEE, chunk size 64. The floor is not "the fast one", it is the most accurate configuration
    fla can be asked for, which is the only thing the others can honestly be said to diverge *from*.

THE TWO PRECISION KNOBS, BOTH CITED
    They are not the same knob, and the difference is the sweep's headline:

    1. ``oss/fla/fla/ops/gated_delta_rule/chunk_fwd.py:20-23`` ::

           if IS_TF32_SUPPORTED:
               SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
           else:
               SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')

       ``IS_TF32_SUPPORTED`` is ``capability[0] >= 8`` (``oss/fla/fla/utils/_device.py:152``), so on
       every arch this plan touches the branch taken is **tf32**. It feeds the *fused* kkt+solve
       kernel, which ``chunk_fwd.py:381`` reaches only when ``BT == 64 and not IS_INTEL`` — the
       fused kernel keeps ten [BC, BC] fp32 accumulators live and spills on Intel, so there the
       unfused IEEE path is taken instead. The tf32 arm therefore exists only on NVIDIA, which is
       every box this plan rents, but it is why the arm is a *measurement* and not a given.

    2. ``oss/fla/fla/ops/utils/solve_tril.py:19-22`` ::

           FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')
           DOT_PRECISION_AUTOTUNE_LIST = ["ieee"] if not IS_TMA_SUPPORTED else list({...})

       This feeds the *unfused* solve used at chunk size 16 and 32, and ``IS_TMA_SUPPORTED`` is
       false unless ``FLA_USE_TMA=1``, so that path is **ieee** by default.

    Same mathematical operation, two different precisions, selected by chunk size alone. That is
    why ``chunk_size`` is a sweep axis and not a tuning detail: a divergence that appears only at
    64 is evidence about knob 1, and a divergence that survives at 16 and 32 is evidence about the
    algebra. Nothing downstream can tell those apart if the axis is collapsed.

THE GRID IS A UNION, NOT A PRODUCT
    ``fused_recurrent`` has no WY representation, therefore no triangular solve, therefore neither
    a chunk size nor a solve precision. Enumerating it against those axes would manufacture
    duplicate rows that differ only in a field the path ignores, and every one of them would land
    on the same measurement — a heatmap with four identical columns is a lie told with real data.
    :func:`grid` enumerates the union; ``sweep.py --dry-run`` prints the count it actually returns.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .divergence import DTYPES as _E001_DTYPES
from .divergence import sweep as _e001_sweep

#: E001's dtype table plus float16. E001 never needed fp16 — its question was fp64-vs-bf16 in pure
#: PyTorch — but fla ships fp16 and half the reports on RFC #1155 are fp16, so the sweep has to
#: cover it. Extended here rather than edited there: E001 is a live lane and its rows must keep
#: meaning exactly what they meant when they were written.
TORCH_DTYPE: dict[str, torch.dtype] = {**_E001_DTYPES, "float16": torch.float16}

#: The pin the plan declares for ``oss/fla``. Every file:line cited below was verified against it
#: *and* against the checkout's HEAD, and they agree — but a pin is a declaration, not a reading.
#: ``infra/oss.sh`` leaves the checkout on a PR branch that moves (on 2026-09-07 it was already 23
#: commits past this pin), so the pin says what a row was expected to be measured at and
#: :func:`fla_head` says what it actually was.
FLA_PIN = "c3db408f"

_HEAD_CACHE: dict[str, str | None] = {}


def fla_head() -> str | None:
    """The short commit ``oss/fla`` is actually on, or ``None`` if there is no checkout.

    Read, not declared. Invariant 4 wants the commit *recorded*, and a hard-coded constant records
    the author's belief rather than the run: a sweep launched on a box whose ``oss/fla`` had moved
    would stamp every row with a revision that never produced it, and rows from two boxes would be
    silently uncomparable while looking identical. This is the one provenance field that cannot be
    read off ``torch`` or the driver, so it is read off git.

    Cached: the sweep calls this once per row and the answer cannot change inside a run.
    """
    if "head" in _HEAD_CACHE:
        return _HEAD_CACHE["head"]
    head: str | None = None
    try:
        from scratch_llm._workspace import workspace_root

        root = workspace_root() / "oss" / "fla"
        if root.is_dir():
            out = subprocess.run(  # noqa: S603
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            head = out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — absence is the fact being recorded, never a run-ender
        head = None
    _HEAD_CACHE["head"] = head
    return head


#: The two upstream lines a reader of this sweep has to see before the numbers mean anything.
#: Asserted against the real checkout by ``tests/test_k3_e002.py`` so the citation cannot rot.
CITATIONS: dict[str, tuple[str, int, int, str]] = {
    "fused_solve_precision": (
        "fla/ops/gated_delta_rule/chunk_fwd.py",
        20,
        23,
        "SOLVE_TRIL_DOT_PRECISION",
    ),
    "unfused_solve_precision": ("fla/ops/utils/solve_tril.py", 19, 22, "FLA_TRIL_PRECISION"),
    "is_tf32_supported": ("fla/utils/_device.py", 152, 152, "IS_TF32_SUPPORTED"),
    "fused_only_at_64": ("fla/ops/gated_delta_rule/chunk_fwd.py", 381, 381, "BT == 64"),
}

#: The gate column, taken from E001's own default rather than retyped. Retyping it is how two
#: experiments end up plotting different x-axes and nobody notices for a month.
GATES: tuple[float, ...] = tuple(inspect.signature(_e001_sweep).parameters["log_gates"].default)

#: ``log_gate = 0`` — alpha = 1, "forget nothing". This is the reported case in fla #104 / #389 and
#: the reason the sweep exists; a grid that dropped it would be measuring the easy half.
GATE_AT_ONE = 0.0

#: Storage dtypes fla is asked for. float64 is absent on purpose: Triton has no fp64 tensor-core
#: path, so the fp64 oracle stays on the CPU (``rel_err_vs_fp64`` below) where it belongs.
SWEEP_DTYPES: tuple[str, ...] = ("float32", "float16", "bfloat16")

#: ``oss/fla/fla/ops/gated_delta_rule/chunk.py:539-540`` rejects anything else with a ValueError.
CHUNK_SIZES: tuple[int, ...] = (16, 32, 64)

#: The one chunk size whose triangular solve runs in the fused kernel, i.e. the only one where
#: ``SOLVE_TRIL_DOT_PRECISION`` is reachable at all (``chunk_fwd.py:381``).
FUSED_CHUNK_SIZE = 64

#: What the ``solve_precision`` axis may say. ``n/a`` is not a third precision — it is the honest
#: label for a path that performs no triangular solve.
SOLVE_PRECISIONS: tuple[str, ...] = ("tf32", "ieee")
NO_SOLVE = "n/a"

#: THE FLOOR. fla's own chunk kernel, most accurate configuration it has: float32 operands, the
#: triangular solve forced to IEEE, chunk size 64. Every ``rel_err_vs_floor`` in the sweep is a
#: deviation from this one cell at the same gate, so rows from different dtypes are comparable.
FLOOR_PATH = "chunk"
FLOOR_DTYPE = "float32"
FLOOR_SOLVE_PRECISION = "ieee"

PATHS: tuple[str, ...] = ("chunk", "fused_recurrent")

#: Fixed problem geometry. T=512, d_head=64 are E001's defaults, again so the two experiments'
#: rows describe the same problem. B=H=1: this sweep is about arithmetic, not about occupancy.
DEFAULT_SEQ_LEN = 512
DEFAULT_D_HEAD = 64
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Cell:
    """One point of the sweep. Frozen because :meth:`key` is used for resume — a cell that can be
    mutated after it has been written is a cell that can be written twice under one name."""

    backend: str  # "fla" (Triton, on silicon) | "cpu-proxy" (E001's paths, for the CPU tier)
    path: str  # "chunk" | "fused_recurrent"
    log_gate: float
    dtype: str
    chunk_size: int | None  # None for fused_recurrent — it has no chunking
    solve_precision: str  # "tf32" | "ieee" | NO_SOLVE
    seq_len: int = DEFAULT_SEQ_LEN
    d_head: int = DEFAULT_D_HEAD
    seed: int = DEFAULT_SEED

    def key(self) -> str:
        """The resume identity. Deliberately excludes every measured field and every provenance
        field except the arch, which the runner appends: re-running the same cell on a second
        architecture must add a row, not skip one."""
        return "|".join(
            str(x)
            for x in (
                self.backend,
                self.path,
                f"{self.log_gate:.10g}",
                self.dtype,
                self.chunk_size,
                self.solve_precision,
                self.seq_len,
                self.d_head,
                self.seed,
            )
        )

    def is_floor(self) -> bool:
        return (
            self.path == FLOOR_PATH
            and self.dtype == FLOOR_DTYPE
            and self.chunk_size == FUSED_CHUNK_SIZE
            and self.solve_precision == FLOOR_SOLVE_PRECISION
        )


def grid(
    backend: str = "fla",
    gates: tuple[float, ...] = GATES,
    dtypes: tuple[str, ...] = SWEEP_DTYPES,
    chunk_sizes: tuple[int, ...] = CHUNK_SIZES,
    solve_precisions: tuple[str, ...] = SOLVE_PRECISIONS,
    paths: tuple[str, ...] = PATHS,
    seq_len: int = DEFAULT_SEQ_LEN,
    d_head: int = DEFAULT_D_HEAD,
    seed: int = DEFAULT_SEED,
) -> list[Cell]:
    """Enumerate the sweep. See the module docstring: this is a union, not a cross product.

    ``chunk``            gate × dtype × {16, 32} × {the one precision that path can use}
                         plus gate × dtype × {64} × solve_precisions
    ``fused_recurrent``  gate × dtype, once

    The ``{16, 32}`` rows carry ``solve_precision="ieee"`` because ``solve_tril.py:19-22`` fixes it
    there; asking for tf32 at 16 or 32 does not produce a tf32 row, it produces a mislabelled ieee
    one, so the request is dropped rather than honoured.
    """
    cells: list[Cell] = []
    for path in paths:
        for gate in gates:
            for dt in dtypes:
                if path == "fused_recurrent":
                    cells.append(
                        Cell(backend, path, gate, dt, None, NO_SOLVE, seq_len, d_head, seed)
                    )
                    continue
                for c in chunk_sizes:
                    precisions = (
                        tuple(p for p in solve_precisions if p in SOLVE_PRECISIONS)
                        if c == FUSED_CHUNK_SIZE
                        else (("ieee",) if "ieee" in solve_precisions else ())
                    )
                    for prec in precisions:
                        cells.append(Cell(backend, path, gate, dt, c, prec, seq_len, d_head, seed))
    return cells


# ---------------------------------------------------------------------------------------------
# The metric, and what counts as "explained by the dtype"
# ---------------------------------------------------------------------------------------------


def unit_eps(dtype: str) -> float:
    """``torch.finfo(dt).eps`` — the same convention E001's figure draws as its floor line, so the
    two experiments' "is this just precision?" question is asked with one ruler.

    Read from torch rather than tabulated: a hard-coded 7.81e-3 for bfloat16 is right until the
    day someone sweeps a dtype that is not in the table, and then it is silently wrong.
    """
    return float(torch.finfo(TORCH_DTYPE[dtype]).eps)


def err_over_eps(rel_err: float | None, dtype: str) -> float | None:
    """The dimensionless quantity the verdict is about: how many unit roundoffs of divergence.

    Relative error alone cannot be judged — 1e-3 is catastrophic in fp32 and beneath notice in
    bf16. Dividing by the dtype's own eps is what makes one threshold apply to every row.
    """
    if rel_err is None or not math.isfinite(rel_err):
        return None
    return rel_err / unit_eps(dtype)


# ---------------------------------------------------------------------------------------------
# THE HOLE — the tolerance ratio. CLAUDE.md names tolerance design as never-delegated, and this
# is the whole of what E002 is for: a divergence map is a picture until someone draws the line
# that separates "bf16 is doing what bf16 does" from "this kernel is wrong".
# ---------------------------------------------------------------------------------------------


def tol_ratio() -> float:
    """How many multiples of the cell dtype's unit roundoff a relative divergence may reach before
    :func:`classify` calls it a BUG rather than expected numerics.

    Three things have to be decided together, and only the person who will defend the number
    upstream can decide them:

      * the error-growth model — does the bound go as ``sqrt(T)``, ``T``, ``T/C``, or with
        ``cond(I + tril(BKK^T,-1))`` from E001's H1 diagnostic? Each gives a different ratio, and
        the sweep's own ``cond_T`` column is the evidence for choosing;
      * the smallest drift worth shouting about, which is not the largest drift tolerable;
      * the false-positive budget: this ratio becomes the assertion in a regression test proposed
        to fla, and a test that cries wolf on a Triton version bump gets reverted, not merged.

    Under ``LADDERS_STUB_HOLES=1`` this returns infinity so the surrounding plumbing (runner,
    resume, JSONL, plot) can be exercised end to end on a CPU. Infinity does not mean "everything
    passes": :func:`classify` maps a non-finite ratio to ``UNJUDGED``, never to ``EXPECTED``, so a
    placeholder can never manufacture a green verdict. ``infra/bench.sh`` refuses to run at all
    while that flag is set.
    """
    if os.environ.get("LADDERS_STUB_HOLES") == "1":
        return math.inf
    # HUY: TOL_RATIO — multiples of the dtype's unit roundoff at which a divergence becomes a BUG — spec: experiments/K3/E002/spec.md — fill before E002
    raise NotImplementedError(
        "HUY: E002 TOL_RATIO is unset. Replace this raise with "
        "`return <ratio>` and write the one-line argument into "
        "experiments/K3/E002/spec.md's 'Correctness gate' line. Read the map "
        "first (`python experiments/K3/E002/plot.py --in <sweep.jsonl>`): the "
        "ratio is chosen from the evidence, not before it."
    )


VERDICTS = ("EXPECTED", "BUG", "UNJUDGED")


def classify(row: dict, ratio: float | None) -> str:
    """One row → one verdict.

    ``UNJUDGED`` is a first-class answer, not an error state. A cell that failed to run, a cell
    whose dtype has no eps, and a run made with an unset tolerance are all "no verdict available",
    and collapsing any of them into ``EXPECTED`` would put a green square on the heatmap where
    there is no evidence at all.
    """
    if ratio is None or not math.isfinite(ratio):
        return "UNJUDGED"
    e = row.get("err_over_eps")
    if e is None or not isinstance(e, int | float) or not math.isfinite(e):
        return "UNJUDGED"
    return "BUG" if e > ratio else "EXPECTED"


def verdict_counts(rows: list[dict], ratio: float | None) -> dict[str, int]:
    # Declared `dict[str, int]`, not left to inference: `dict.fromkeys(VERDICTS, 0)` keys the dict
    # by the three string LITERALS, and `classify` returns a plain `str`.
    counts: dict[str, int] = dict.fromkeys(VERDICTS, 0)
    for r in rows:
        counts[classify(r, ratio)] += 1
    return counts


# ---------------------------------------------------------------------------------------------
# The row schema. Rows are FACTS — config, provenance, and what was measured. No verdict is
# stored in a row: a verdict is a function of the tolerance, the tolerance is chosen after the
# map is read, and a stored verdict would silently outlive the ratio that produced it.
# ---------------------------------------------------------------------------------------------

#: Every field a row must carry. Asserted on write, so a row that reaches the JSONL is a row
#: plot.py and the regression test can read without defensive ``.get`` chains.
ROW_FIELDS: tuple[str, ...] = (
    # the cell
    "backend",
    "path",
    "log_gate",
    "dtype",
    "chunk_size",
    "solve_precision",
    "seq_len",
    "d_head",
    "seed",
    # the measurement
    "rel_err_vs_floor",
    "max_abs_err_vs_floor",
    "rel_err_vs_fp64",
    "floor_norm",
    "eps",
    "err_over_eps",
    "error",
    # provenance — invariant 4
    "arch",
    "device",
    "driver",
    "torch",
    "triton",
    "fla_commit",
    "autotune",
    "ts",
)


def make_row(
    cell: Cell,
    *,
    rel_err_vs_floor: float | None,
    max_abs_err_vs_floor: float | None,
    rel_err_vs_fp64: float | None,
    floor_norm: float | None,
    provenance: dict,
    error: str | None = None,
    autotune: str | None = None,
    ts: str,
) -> dict:
    """Build one JSONL row and check it is complete before it is written.

    A sweep is resumable, which means a malformed row written on Tuesday is skipped on Wednesday
    and never re-measured. Validating here rather than at read time is the difference between a
    loud failure now and a hole in the map that nobody sees.
    """
    row = {
        **asdict(cell),
        "rel_err_vs_floor": rel_err_vs_floor,
        "max_abs_err_vs_floor": max_abs_err_vs_floor,
        "rel_err_vs_fp64": rel_err_vs_fp64,
        "floor_norm": floor_norm,
        "eps": unit_eps(cell.dtype),
        "err_over_eps": err_over_eps(rel_err_vs_floor, cell.dtype),
        "error": error,
        "arch": provenance.get("arch"),
        "device": provenance.get("device"),
        "driver": provenance.get("driver"),
        "torch": provenance.get("torch"),
        "triton": provenance.get("triton"),
        "fla_commit": provenance.get("fla_commit", FLA_PIN),
        "autotune": autotune,
        "ts": ts,
    }
    missing = [f for f in ROW_FIELDS if f not in row]
    extra = [f for f in row if f not in ROW_FIELDS]
    if missing or extra:
        raise ValueError(f"row schema drift: missing={missing} extra={extra}")
    return row


def row_key(row: dict) -> str:
    """Resume identity of an already-written row, including the arch it was measured on.

    The arch is part of the identity here but not in :meth:`Cell.key` for a reason: the runner
    knows which box it is on and there is exactly one, so a *cell* is identified without it, while
    a *row* on disk may have come from any of the three and must not shadow the other two.
    """
    cell = Cell(
        backend=row["backend"],
        path=row["path"],
        log_gate=row["log_gate"],
        dtype=row["dtype"],
        chunk_size=row["chunk_size"],
        solve_precision=row["solve_precision"],
        seq_len=row["seq_len"],
        d_head=row["d_head"],
        seed=row["seed"],
    )
    return f"{cell.key()}|{row.get('arch')}"


# ---------------------------------------------------------------------------------------------
# JSONL, one row per cell. Line-delimited rather than one JSON document because the sweep is
# resumable: appending a line is a complete write, so a run killed when the pod's hour expires
# leaves every finished cell readable, and a re-run picks up exactly where it stopped. A single
# top-level JSON array would have to be rewritten in full each time, and a kill mid-rewrite would
# lose the whole file — the one failure mode a resumable sweep exists to prevent.
# ---------------------------------------------------------------------------------------------


def read_rows(path: str | Path) -> list[dict]:
    """Every row in a sweep file. A missing file is an empty sweep, not an error — that is the
    first run. A malformed line is an error: silently dropping it would make the resume set wrong
    and the missing cell would look measured."""
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{i} is not a JSON row ({e}). Fix or delete the line.") from e
    return rows


def append_row(path: str | Path, row: dict) -> None:
    """Append one row and flush it. Flushing per row is the price of resumability: a buffered
    write that never reaches the disk is a cell that gets measured twice and, worse, a cell whose
    absence from the file is indistinguishable from never having been attempted."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_keys(path: str | Path) -> set[str]:
    """Resume set: the row keys already on disk, errors included.

    A cell that raised is DONE, not pending. Re-attempting it every run would turn a sweep with
    one unsupported dtype into an infinite loop of the same traceback, and the recorded error is
    itself a finding — "fla rejects fp32 q/k/v here" is a result, and the map should show it.
    """
    return {row_key(r) for r in read_rows(path)}


def arms_are_distinguishable(rows: list[dict]) -> tuple[bool, str]:
    """Did forcing the triangular solve to IEEE actually change anything?

    Triton compiles a ``@triton.jit`` kernel once per specialization and caches it on disk. The
    tf32/ieee choice at ``chunk_fwd.py:20-23`` is a module-level global captured at compile time,
    NOT a kernel argument — so an ieee run that reuses a cache populated by a tf32 run silently
    measures tf32 twice and reports a beautiful null result: two identical arms, "no effect of
    solve precision", publishable and wrong. The runner defends against this structurally (a
    separate process and a separate ``TRITON_CACHE_DIR`` per arm); this is the detector that says
    the defence held.

    Returns ``(distinguishable, explanation)``. Undecidable — fewer than one comparable pair — is
    reported as not distinguishable, because "we could not check" and "we checked and it is fine"
    must not print the same word.
    """
    by_key: dict[tuple, dict[str, float | None]] = {}
    for r in rows:
        if r.get("chunk_size") != FUSED_CHUNK_SIZE:
            continue
        if r.get("solve_precision") not in SOLVE_PRECISIONS:
            continue
        k = (r["backend"], r["path"], r["log_gate"], r["dtype"], r["seed"], r.get("arch"))
        by_key.setdefault(k, {})[r["solve_precision"]] = r.get("rel_err_vs_floor")
    pairs = [v for v in by_key.values() if len(v) == 2]
    if not pairs:
        return False, (
            f"no cell at chunk_size={FUSED_CHUNK_SIZE} has both a tf32 and an ieee row — "
            "the two arms cannot be compared, so the override is unverified"
        )
    differing = [p for p in pairs if p["tf32"] != p["ieee"]]
    if not differing:
        return False, (
            f"all {len(pairs)} tf32/ieee pairs are bit-identical. The ieee override did not reach "
            f"the kernel — almost always a shared Triton cache. Delete the ieee rows and re-run "
            f"that arm with a fresh TRITON_CACHE_DIR; see {CITATIONS['fused_solve_precision'][0]}"
            f":{CITATIONS['fused_solve_precision'][1]}."
        )
    return (
        True,
        f"{len(differing)}/{len(pairs)} tf32/ieee pairs differ — the override reached the kernel",
    )
