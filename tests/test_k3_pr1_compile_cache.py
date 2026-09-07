"""K3/PR-1 — the compile-count instrument for FlashInfer #4110. Three tiers, ascending cost.

  CPU   the pure core: grid enumeration, distinct-key collapsing, the B-vs-T attribution, and the
        ``cute.compile`` counter's two call shapes. Hand-counted fakes, no flashinfer, milliseconds.
  hole  the proposed cache key. Strict-xfail until Huy writes it (tests/conftest.py).
  gpu   ``--mode live`` end to end. Needs a GPU *and* an installed flashinfer; self-skips.

The instrument itself lives outside this repo — ``experiments/K3/PR-1/count_compiles.py`` in the
ladders workspace, because it is a measurement of someone else's library, not a kernel of ours. It
is loaded here by path rather than imported: adding the workspace to ``sys.path`` would put
``experiments`` on the import path of every other test in this suite.

What the CPU tier is actually protecting. The script's headline claim is a ratio — N (B, T) cases
collapse to M compiles — and that ratio is arithmetic over a grid, not a measurement. If the
collapsing is wrong the PR quotes a wrong number at a maintainer, and no GPU run would catch it,
because the GPU run reports the same arithmetic with a compiler attached. So the collapsing is
tested against grids small enough to count by hand.

Spec: experiments/K3/PR-1/spec.md   ·   Upstream issue: flashinfer-ai/flashinfer#4110
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest

from scratch_llm.kernels.gemm.cuda._k1_loader import workspace_root

_SCRIPT = workspace_root() / "experiments" / "K3" / "PR-1" / "count_compiles.py"


def _load() -> ModuleType:
    """Load count_compiles.py by path, under a private name.

    ``spec_from_file_location`` rather than an import: the script is not in a package, has no
    ``__init__.py`` above it, and must stay runnable as a bare file on a rented box that has
    flashinfer but has never heard of scratch_llm.
    """
    name = "_k3_pr1_count_compiles"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: ``@dataclass`` resolves annotations through
    # ``sys.modules[cls.__module__].__dict__``, so a module that is not yet registered raises
    # AttributeError on the decorator rather than anywhere near the dataclass.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cc = _load()


def _flashinfer_present() -> bool:
    return importlib.util.find_spec("flashinfer") is not None


# =============================================================================================
# The hole guard — exactly one (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


# The source path is spelled relative to THIS repo because that is what tests/conftest.py resolves
# it against, and the hole lives in the sibling `ladders` workspace that symlinks to this repo.
# If that traversal ever stops resolving the failure is loud, not silent: a missing file reads as
# "hole open", the xfail stays strict, and the day Huy fills the hole this test XPASSes and goes
# red. `python3 infra/holes.py` joins on the rung id and is unaffected either way.
@pytest.mark.hole("K3/PR-1", "../ladders/experiments/K3/PR-1/count_compiles.py")
def test_proposed_cache_key_shares_the_seqlen_band_and_splits_the_tile_band() -> None:
    """Fails while ``proposed_cache_key`` raises; passes when it satisfies both halves of the claim.

    Both halves, because a key that only ever collapses is not a fix, it is a correctness bug. The
    WY path already learned this the expensive way — ``gdn_decode_bf16_wy_output_only.py:2282`` carries
    a comment about a process mixing HV values reusing the first compile, reading the state pool
    with the wrong strides and returning ~3e-01 garbage. Any key that drops a field the *cubin*
    bakes in does the same thing silently.

    The two bands, at the tests/gdn geometry (HV=32, V=128), from ``get_mtp_config``
    (gdn_decode_mtp.py:66) with work_units = B*HV = 256:

      B=8, T=4  and  B=8, T=8  -> (tile_v=32, ilp_rows=4, use_smem_v=False).  Identical tuning.
                                  Nothing but T itself distinguishes them, so they MUST share.
      B=8, T=2                 -> (tile_v=16, ilp_rows=2, use_smem_v=False).  Different tuning,
                                  therefore a different cubin, therefore MUST NOT share.
    """
    geom = cc.DEFAULT_GEOM
    shared_a = cc.proposed_cache_key(8, 4, geom, "mtp_fp32")
    shared_b = cc.proposed_cache_key(8, 8, geom, "mtp_fp32")
    split = cc.proposed_cache_key(8, 2, geom, "mtp_fp32")

    assert hash(shared_a) is not None, "a cache key must be hashable — it is a dict key upstream"
    assert shared_a == shared_b, (
        "T=4 and T=8 at B=8 select the identical (tile_v, ilp_rows, use_smem_v) and differ only in "
        "the literal T at gdn_decode_mtp.py:2642 — the proposed key must collapse them, that is "
        "the whole PR"
    )
    assert shared_a != split, (
        "T=2 at B=8 selects tile_v=16 / ilp_rows=2, a different cubin. A key that collapses it into "
        "the T>=3 band would launch a kernel compiled for the wrong tile shape"
    )


# =============================================================================================
# CPU — the grid
# =============================================================================================


def test_suite_grid_mirrors_the_upstream_sweep() -> None:
    """The headline ratio is only comparable to bkryu's ~800/~200 if the grid is the suite's grid."""
    assert cc.GRIDS["suite"]["B"] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert cc.GRIDS["suite"]["T"] == [1, 2, 3, 4, 8]
    points = cc.grid_points(cc.GRIDS["suite"])
    assert len(points) == 8 * 5
    assert points[0] == (1, 1) and points[-1] == (128, 8)
    # B-major: a sweep holds the model loaded and walks batch outermost, which is also the order
    # that makes a per-B compile burst visible in the live table.
    assert points[:5] == [(1, 1), (1, 2), (1, 3), (1, 4), (1, 8)]


def test_reachable_splits_the_t1_and_mtp_kernels() -> None:
    """T=1 and T>1 are different kernels behind one public API; counting them into one cache would
    inflate the exact number this script exists to report."""
    t1 = cc.Backend("x", "site", "sel", ("T=1",), lambda b, t, g: {})
    mtp = cc.Backend("y", "site", "sel", ("T>=2",), lambda b, t, g: {})
    assert cc.reachable(t1, 1) and not cc.reachable(t1, 2)
    assert cc.reachable(mtp, 2) and cc.reachable(mtp, 8) and not cc.reachable(mtp, 1)


# =============================================================================================
# CPU — the collapsing and the attribution
# =============================================================================================


def test_distinct_keys_collapses_points_that_share_a_key() -> None:
    """Four points, two keys, and every point accounted for exactly once."""
    points = [(1, 2), (1, 4), (8, 2), (8, 4)]
    buckets = cc.distinct_keys(points, lambda b, t: {"tile_v": 32 if b >= 8 else 16})
    assert len(buckets) == 2
    assert sorted(len(v) for v in buckets.values()) == [2, 2]
    assert sorted(p for pts in buckets.values() for p in pts) == points


def test_distinct_keys_ignores_field_order() -> None:
    """Two dicts with the same items in different order are the same key.

    Not pedantry: the key fields are built by hand per backend, and a dict literal reordered in a
    later edit would otherwise double the reported compile count with no code change behind it.
    """
    points = [(1, 2), (8, 2)]
    buckets = cc.distinct_keys(
        points, lambda b, t: {"a": 1, "b": 2} if b == 1 else {"b": 2, "a": 1}
    )
    assert len(buckets) == 1


def test_distinct_keys_drops_unreachable_points_instead_of_counting_them() -> None:
    """``None`` means "this backend is not the one this call lands in" — not "a key of None"."""
    points = [(1, 1), (1, 2), (8, 1), (8, 2)]
    buckets = cc.distinct_keys(points, lambda b, t: None if t == 1 else {"T": t})
    assert len(buckets) == 1
    assert sorted(p for pts in buckets.values() for p in pts) == [(1, 2), (8, 2)]


def test_field_drivers_attributes_each_axis_from_the_data() -> None:
    """B, T, both, neither — the four answers, measured by holding one axis and varying the other."""
    points = cc.grid_points({"B": [1, 8, 64], "T": [2, 4]})

    def fields(b: int, t: int) -> dict[str, object]:
        return {
            "constant": 4,
            "from_b": 32 if b >= 8 else 16,
            "from_t": t,
            "from_both": (b >= 8) and (t >= 4),
        }

    assert cc.field_drivers(points, fields, "constant") == set()
    assert cc.field_drivers(points, fields, "from_b") == {"B"}
    assert cc.field_drivers(points, fields, "from_t") == {"T"}
    assert cc.field_drivers(points, fields, "from_both") == {"B", "T"}


def test_field_drivers_reports_nothing_for_a_field_constant_on_this_grid() -> None:
    """A field that reads as B-derived in the source but never moves over the grid the suite runs
    costs no compiles, and a PR that "fixes" it buys nothing. The table has to say so."""
    points = cc.grid_points({"B": [1, 2], "T": [2, 4]})
    # `use_small_batch = B < 32` (gdn_decode_nontranspose.py:48) is genuinely B-derived, and dead
    # on a grid whose largest B is 2.
    assert cc.field_drivers(points, lambda b, t: {"small": b < 32}, "small") == set()


def test_field_values_lists_none_first_then_numbers_in_numeric_order() -> None:
    """``None`` is "backend declines this shape" (``_select_wide_vec_tile_v`` returns it at
    work_units < 128) and must be readable as such at the head of the row.

    Numeric order for the rest, because a lexicographic sort prints ``16, 32, 64, 8`` for tile_v and
    this row goes verbatim into a PR body, where it would read as a bug in the selector.
    """
    points = cc.grid_points({"B": [1, 2, 4, 8], "T": [2]})
    tiles = {1: None, 2: 8, 4: 64, 8: 16}
    vals = cc.field_values(points, lambda b, t: {"tile_v": tiles[b]}, "tile_v")
    assert vals == [None, 8, 16, 64]


def test_field_values_keeps_bools_out_of_the_numeric_run() -> None:
    """``use_smem_v`` is a bool, and ``bool`` is an ``int`` in Python. Sorted numerically it lands
    between 0 and 2 and prints as ``False, True`` scattered through tile sizes."""
    points = cc.grid_points({"B": [1, 8], "T": [2]})
    vals = cc.field_values(points, lambda b, t: {"smem": b == 8}, "smem")
    assert vals == [False, True]


def test_analytic_report_is_the_table_the_pr_quotes() -> None:
    """End to end over fakes: reachability, per-backend collapsing, totals, and the field table.

    The PR body quotes "N cases -> M compiles" straight out of this dict. Two fakes with known
    answers: a T=1 backend whose key is only T (8 B values, 1 key — the pretranspose shape) and an
    MTP backend keyed on T alone (4 T values above 1, so 4 keys).
    """
    points = cc.grid_points(cc.GRIDS["suite"])
    backends = [
        cc.Backend("t1_only", "fake:1", "-", ("T=1",), lambda b, t, g: {"T": t}),
        cc.Backend("mtp", "fake:2", "fake:3", ("T>=2",), lambda b, t, g: {"T": t}),
    ]
    report = cc.analytic_report(backends, points, cc.DEFAULT_GEOM)

    t1, mtp = report["backends"]
    assert (t1["reachable_points"], t1["distinct_keys"]) == (8, 1)
    assert (mtp["reachable_points"], mtp["distinct_keys"]) == (32, 4)
    assert report["total_reachable_points"] == 40
    assert report["total_distinct_keys"] == 5
    # Every reachable point is counted exactly once across the two backends: the T=1 and T>1
    # kernels partition the grid, they do not overlap. A backend list that double-counts would
    # inflate the "cases" column and deflate the ratio the PR claims.
    assert report["total_reachable_points"] == len(points)
    assert [f["field"] for f in mtp["fields"]] == ["T"]
    assert mtp["fields"][0]["drivers"] == ["T"]


# =============================================================================================
# CPU — the cute.compile hook
# =============================================================================================


def test_compile_counter_counts_both_call_shapes() -> None:
    """``cute.compile(...)`` and ``cute.compile[options](...)`` are both live upstream.

    ``gdn_decode_bf16_wy_output_only.py:2358`` uses the subscript form. A plain function
    substituted for ``cute.compile`` raises TypeError there, and the run dies inside the code path
    it is trying to measure — the failure looks like a broken kernel, not a broken hook.
    """

    class FakeCute:
        def __call__(self, kernel, *args):
            return ("compiled", kernel)

        def __getitem__(self, options):
            return lambda kernel, *args: ("compiled", kernel, options)

    counter = cc.CompileCounter(FakeCute())
    assert counter("k0") == ("compiled", "k0")
    assert counter[{"opt": 1}]("k1") == ("compiled", "k1", {"opt": 1})
    assert counter.count == 2
    assert counter.seconds >= 0.0


def test_compile_counter_counts_a_compile_that_raises() -> None:
    """A failed compile still cost wall time and still consumed a cache miss. Counting it only on
    success would make an unsupported-shape sweep look cheaper than it is."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("ptxas said no")

    counter = cc.CompileCounter(boom)
    with pytest.raises(RuntimeError, match="ptxas"):
        counter("k")
    assert counter.count == 1


# =============================================================================================
# CPU — the CLI's two GPU-free paths
# =============================================================================================


def test_dry_run_imports_no_flashinfer(capsys: pytest.CaptureFixture[str]) -> None:
    """``--dry-run`` is what runs on this Mac, so it must not import the library it describes."""
    assert cc.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "40 (B,T) points" in out
    assert "gdn_decode_mtp.py:2642" in out, "the dry run must name the cache-key sites it hooks"
    assert not any(m.startswith("flashinfer") for m in sys.modules)


@pytest.mark.skipif(_flashinfer_present(), reason="flashinfer installed — the skip path is moot")
def test_self_skip_prints_the_box_command(capsys: pytest.CaptureFixture[str]) -> None:
    """No flashinfer is the normal state of this machine, not an error. Exit 0 and say where to go.

    Returning non-zero here would make ``run.sh``/``floor.sh`` look broken on the laptop every time
    someone reads them, and the one thing a rung's scripts must never do is cry wolf.
    """
    assert cc.main([]) == 0
    out = capsys.readouterr().out
    assert "not importable here" in out
    assert "count_compiles.py --mode live" in out


def test_proposed_mode_reports_the_hole_and_needs_no_flashinfer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--proposed`` runs before the flashinfer import check, so the key can be checked where it is
    written — on the laptop, against the field table from an earlier box run.

    Written to hold on both sides of the fill: an open hole must refuse loudly and non-zero, a
    filled one must print the collapse it buys. A test that only knew the open state would go red
    the day Huy filled it, which is the one thing the hole convention exists to prevent.
    """
    rc = cc.main(["--proposed", "--grid", "smoke"])
    out = capsys.readouterr().out
    if cc.hole_is_open():
        assert rc == 1 and "open hole" in out
    else:
        assert rc == 0 and "distinct keys" in out


def test_provenance_survives_a_box_with_no_torch() -> None:
    """Invariant 4 wants provenance on every run; a missing field must degrade to a string, never
    to a traceback that eats the measurement that was already taken."""
    prov = cc.provenance()
    assert set(prov) >= {"host", "python", "flashinfer_commit", "torch", "device"}
    assert all(isinstance(v, str) for v in prov.values())


# =============================================================================================
# gpu — the live path, on a box that has both a device and flashinfer
# =============================================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not _flashinfer_present(), reason="flashinfer not installed")
def test_live_floor_prints_a_single_integer(capsys: pytest.CaptureFixture[str]) -> None:
    """``--floor`` is read by ``floor.sh`` with ``tail -1``. Anything else on that line is a floor
    of ``''`` in the ledger."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    assert cc.main(["--mode", "live", "--grid", "smoke", "--floor"]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.strip().isdigit(), f"floor.sh reads the last line as a number; got {last!r}"


@pytest.mark.gpu
@pytest.mark.skipif(not _flashinfer_present(), reason="flashinfer not installed")
def test_analytic_report_counts_no_more_keys_than_cases() -> None:
    """The claim is a collapse. A backend reporting more distinct keys than reachable cases would
    mean the key fields depend on something outside (B, T) — a bug in this script, not upstream.

    Analytic mode launches nothing and still needs a device: ``gdn_decode_bf16_state`` reads
    ``torch.cuda.get_device_properties(0)`` at import, so on a flashinfer-present CPU box this
    would ERROR inside ``_load_backends`` rather than skip.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device — analytic mode still imports gdn_decode_bf16_state")
    points = cc.grid_points(cc.GRIDS["suite"])
    report = cc.analytic_report(cc._load_backends(), points, cc.DEFAULT_GEOM)
    for backend in report["backends"]:
        assert backend["distinct_keys"] <= backend["reachable_points"], backend["name"]
    assert report["total_distinct_keys"] <= report["total_reachable_points"]
