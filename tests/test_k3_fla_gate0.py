"""K3/fla-test — the gate-at-zero regression test for fla, and the reproduction that argues for it.

Two tiers, ascending cost.

  CPU   the pure core of ``experiments/K3/fla-test/reproduce.py`` — the error metric against a slow
        reference, the input construction the control column depends on, the table, ``--dry-run`` —
        plus the citation guard that keeps every file:line this rung quotes pointing at the code it
        claims. Milliseconds, no fla, no GPU.
  gpu   ``measure()`` end to end. Needs a GPU *and* an installed fla; self-skips.

The deliverable itself is ``oss/fla/tests/ops/test_gdn_gate0.py``, which lives in fla's tree
because it is going to fla — it cannot import anything from here, and it cannot run here either
(fla is Triton). What can be checked here is everything the upstream file's argument rests on:
that the lines it cites still say what it says they say, and that the reproduction's numbers and
the test's threshold are in the same units.

WHY THE CONTROL COLUMN IS THE THING WORTH TESTING ON A LAPTOP
    The claim of the whole rung is "a gate of exactly 0 is not a special case", and it is made by
    comparing two runs that differ only in the gate. If ``make_inputs`` were to hand the two runs
    different q/k/v — one stray reseed, one moved ``torch.manual_seed`` — the comparison would
    still produce a plausible-looking number, on a rented GPU, that means nothing. No GPU run can
    catch that; a CPU test can, and does, below.

There is no ``# HUY:`` hole in this rung. The one number it needs — the error ratio the gate-at-0
output is held to — is the same act of judgement as E002's ``TOL_RATIO`` and is recorded as such
in ``experiments/K3/fla-test/spec.md``; opening a second hole for one number decided once would
put two rows in ``make holes`` that are filled or unfilled together.

Spec: experiments/K3/fla-test/spec.md · Upstream: fla-org/flash-linear-attention#104, #389
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from scratch_llm.kernels.gemm.cuda._k1_loader import workspace_root

_SCRIPT = workspace_root() / "experiments" / "K3" / "fla-test" / "reproduce.py"
_FLA = workspace_root() / "oss" / "fla"
_UPSTREAM_TEST = _FLA / "tests" / "ops" / "test_gdn_gate0.py"


def _load() -> ModuleType:
    """Load reproduce.py by path, under a private name.

    By path rather than by import: the script must stay a bare pasteable file with no package
    above it, runnable on a box that has fla and has never heard of scratch_llm.
    """
    name = "_k3_fla_gate0_reproduce"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rp = _load()


def _fla_present() -> bool:
    return importlib.util.find_spec("fla") is not None


def _slow_err_ratio(ref: torch.Tensor, tri: torch.Tensor) -> float:
    """RMS(ref - tri) / RMS(ref), written as loops over Python floats.

    Deliberately not a tensor expression: the point of a reference is to be wrong in different
    ways than the thing it checks, and a one-line torch reimplementation of a one-line torch
    function shares every mistake it could make.
    """
    a = ref.flatten().tolist()
    b = tri.flatten().tolist()
    assert len(a) == len(b)
    err = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / len(a))
    base = math.sqrt(sum(x * x for x in a) / len(a))
    return err / (base + 1e-8)


# =============================================================================================
# CPU — the error metric
# =============================================================================================


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 5), (7,)])
def test_err_ratio_matches_a_slow_python_reference(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    ref = torch.randn(*shape, dtype=torch.float64)
    tri = ref + 0.01 * torch.randn(*shape, dtype=torch.float64)
    assert rp.err_ratio(ref, tri) == pytest.approx(_slow_err_ratio(ref, tri), rel=1e-9, abs=1e-12)


def test_err_ratio_is_zero_on_identical_tensors_and_grows_with_the_perturbation() -> None:
    """Monotone in the error, and exactly zero when there is none — the two properties the table's
    ratio column silently assumes when it divides one of these by another."""
    torch.manual_seed(0)
    ref = torch.randn(64, dtype=torch.float64)
    assert rp.err_ratio(ref, ref.clone()) == 0.0
    small = rp.err_ratio(ref, ref + 1e-4 * torch.randn_like(ref))
    large = rp.err_ratio(ref, ref + 1e-2 * torch.randn_like(ref))
    assert 0.0 < small < large


def test_err_ratio_is_scale_free_up_to_flas_denominator_epsilon() -> None:
    """Scaling both tensors must not move the number — rows at different sequence lengths have
    different output magnitudes, and a metric that drifted with scale could not be read down a
    column.

    It is scale-free only up to the ``+ 1e-8`` fla adds to the denominator: scaling by s multiplies
    the result by ``(base + 1e-8) / (base + 1e-8/s)``. That is nothing at the magnitudes this table
    lives at, and everything if a row's reference RMS ever approaches 1e-8 — worth knowing before
    someone reads a near-zero row as a divergence rather than as a division.
    """
    torch.manual_seed(0)
    ref = torch.randn(128, dtype=torch.float64)
    tri = ref + 0.05 * torch.randn(128, dtype=torch.float64)
    base = ref.square().mean().sqrt().item()
    predicted = (base + 1e-8) / (base + 1e-8 / 2)
    assert rp.err_ratio(2 * ref, 2 * tri) / rp.err_ratio(ref, tri) == pytest.approx(
        predicted, rel=1e-9
    )


# =============================================================================================
# CPU — the inputs the control column stands on
# =============================================================================================


def test_make_inputs_differs_only_in_the_gate() -> None:
    """The load-bearing property of the whole rung.

    ``gated@0`` and ``gated@-e`` are only a control and its case if they are the same problem. Any
    difference in q, k, v, beta or h0 turns the comparison into two unrelated measurements that
    still print as a table.
    """
    at_zero = rp.make_inputs(16, log_gate=rp.LOG_GATE_ZERO, B=1, H=1, D=8)
    below = rp.make_inputs(16, log_gate=rp.LOG_GATE_BELOW_ZERO, B=1, H=1, D=8)
    for name in ("q", "k", "v", "beta", "h0"):
        assert torch.equal(at_zero[name], below[name]), f"{name} differs between the two gates"
    assert not torch.equal(at_zero["g"], below["g"])


def test_make_inputs_gate_is_exactly_the_requested_constant() -> None:
    """Exactly, not approximately: the case under test *is* the exact value 0, and a gate that
    arrived as 1e-45 instead would pass every assertion while testing nothing."""
    at_zero = rp.make_inputs(8, log_gate=rp.LOG_GATE_ZERO, B=1, H=2, D=4)
    assert torch.equal(at_zero["g"], torch.zeros_like(at_zero["g"]))
    below = rp.make_inputs(8, log_gate=rp.LOG_GATE_BELOW_ZERO, B=1, H=2, D=4)
    assert (below["g"] < 0).all()
    assert torch.equal(below["g"], torch.full_like(below["g"], rp.LOG_GATE_BELOW_ZERO))


def test_make_inputs_keys_are_unit_norm_and_shapes_are_the_documented_ones() -> None:
    got = rp.make_inputs(12, log_gate=rp.LOG_GATE_ZERO, B=2, H=3, D=8, dtype=torch.float32)
    assert got["q"].shape == (2, 12, 3, 8)
    assert got["v"].shape == (2, 12, 3, 8)
    assert got["beta"].shape == (2, 12, 3)
    assert got["g"].shape == (2, 12, 3)
    assert got["h0"].shape == (2, 3, 8, 8)
    assert torch.allclose(got["k"].norm(dim=-1), torch.ones(2, 12, 3), atol=1e-6)


def test_control_gate_is_inside_one_bfloat16_ulp_over_the_longest_sequence() -> None:
    """The argument written next to ``LOG_GATE_BELOW_ZERO``, checked rather than asserted in prose.

    The control is only a control while the recurrence it describes is the alpha = 1 recurrence.
    Two ways it could quietly stop being one: someone makes the constant larger, or someone adds a
    longer sequence to the sweep. Both are caught here.
    """
    longest = max(rp.DEFAULT_SEQ_LENS)
    total_decay = math.exp(longest * rp.LOG_GATE_BELOW_ZERO)
    ulp = 2.0**-8  # bfloat16 has 8 head-dim bits of mantissa; ulp at 1.0 is 2**-8
    assert 0.0 < 1.0 - total_decay < ulp, (
        f"total decay {total_decay} is not within one bf16 ulp of 1"
    )
    assert rp.LOG_GATE_BELOW_ZERO < 0.0, "a control gate of zero is not a control"


# =============================================================================================
# CPU — the table and the dry run
# =============================================================================================


def test_render_prints_every_row_and_keeps_a_failed_cell() -> None:
    """A cell that raised is a result — it is where the kernel stopped working. Dropping it would
    make a partially-failed sweep look like a shorter successful one."""
    rows = [
        {
            "T": 7,
            "gated_at_zero_vs_ref": 0.1,
            "gated_below_zero_vs_ref": 0.1,
            "ungated_vs_ref": 0.1,
            "ratio_gated_over_ungated": 1.0,
        },
        {"T": 50, "error": "RuntimeError: out of memory"},
    ]
    out = rp.render(rows)
    assert "7" in out and "50" in out
    assert "out of memory" in out
    assert len(out.splitlines()) == 2 + len(rows) + 1  # header, rule, rows, closing rule


def test_dry_run_names_every_citation_and_imports_nothing() -> None:
    """Every cited span reaches the printed page, and printing it pulls in no fla.

    The import half is measured as a delta rather than as `"fla" not in sys.modules`: on the box
    fla is installed and some earlier test will have imported it, and an assertion that is
    vacuously true on a laptop and spuriously false on the box is worse than none.
    """
    before = set(sys.modules)
    text = rp.dry_run(rp.DEFAULT_SEQ_LENS, "bfloat16")
    for name, (path, lo, hi, _) in rp.CITATIONS.items():
        assert name in text
        assert f"{path}:{lo}" in text, f"{name} span missing from --dry-run"
        if hi != lo:
            assert f"{path}:{lo}-{hi}" in text
    assert rp.FLA_COMMIT in text
    newly_imported = {m for m in set(sys.modules) - before if m == "fla" or m.startswith("fla.")}
    assert not newly_imported, f"--dry-run imported {sorted(newly_imported)}"


def test_main_dry_run_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert rp.main(["--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out


def test_main_dry_run_honours_a_custom_grid(capsys: pytest.CaptureFixture[str]) -> None:
    assert rp.main(["--dry-run", "--seq-lens", "64,128"]) == 0
    assert "[64, 128]" in capsys.readouterr().out


def test_floor_refuses_on_a_host_with_no_gpu(capsys: pytest.CaptureFixture[str]) -> None:
    """``floor.sh`` reads the last stdout line as the floor value. On a box with no device this
    must exit non-zero with an empty stdout, not print a helpful sentence that the ledger would
    then record as a number."""
    if torch.cuda.is_available():
        pytest.skip("this asserts the CPU-host refusal path")
    assert rp.main(["--floor"]) == 2
    assert capsys.readouterr().out == ""


# =============================================================================================
# CPU — the citations, against the real checkout
# =============================================================================================


def _fla_missing() -> bool:
    return not (_FLA / "fla" / "ops" / "gated_delta_rule" / "chunk_fwd.py").is_file()


@pytest.mark.skipif(_fla_missing(), reason="oss/fla checkout absent — infra/oss.sh has not run")
@pytest.mark.parametrize("name", sorted(rp.CITATIONS))
def test_citation_resolves_in_the_fla_checkout(name: str) -> None:
    """Every file:line this rung quotes still contains what it is quoted for.

    A rotted citation in a PR body is the cheapest way to lose a maintainer's trust, and line
    numbers move on every rebase. This is the guard that makes the quotes maintainable rather than
    a snapshot: it fails the day the checkout moves under them, which is the day to re-read.
    """
    path, lo, hi, marker = rp.CITATIONS[name]
    lines = (_FLA / path).read_text(encoding="utf-8").splitlines()
    span = "\n".join(lines[lo - 1 : hi])
    assert marker in span, f"{path}:{lo}-{hi} no longer contains {marker!r}\n---\n{span}\n---"


@pytest.mark.skipif(_fla_missing(), reason="oss/fla checkout absent — infra/oss.sh has not run")
def test_upstream_test_cites_the_same_lines_as_the_reproduction() -> None:
    """The test and the reproduction must be about the same code.

    They are two files in two repositories that will be read weeks apart, and the only thing
    stopping them from drifting into describing different kernels is this assertion.
    """
    text = _UPSTREAM_TEST.read_text(encoding="utf-8")
    for key in ("fused_gate_term", "fused_boundary_mask", "fused_only_at_64", "unfused_gate_term"):
        path, lo, hi, _ = rp.CITATIONS[key]
        span = f"{Path(path).name}:{lo}" + (f"-{hi}" if hi != lo else "")
        assert span in text, f"upstream test does not cite {span}"
    assert rp.FLA_COMMIT in text, (
        "upstream test does not name the commit its line numbers are against"
    )


@pytest.mark.skipif(_fla_missing(), reason="oss/fla checkout absent — infra/oss.sh has not run")
def test_upstream_test_is_a_valid_module_in_flas_conventions() -> None:
    """It cannot be imported here — fla needs triton — so it is checked as text and as an AST.

    The two things that would make it useless on the box: a syntax error found at 23:00, and a
    missing copyright header, which ``scripts/check_header.py --check`` fails the whole repo on.
    """
    text = _UPSTREAM_TEST.read_text(encoding="utf-8")
    assert text.startswith("# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li")
    tree = ast.parse(text)
    tests = {
        n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }
    assert tests == {
        "test_chunk_gate_at_zero_is_finite",
        "test_chunk_gate_matches_recurrent_reference",
        "test_chunk_gate_at_zero_matches_gate_just_below_zero",
        "test_chunk_gate_at_zero_matches_ungated_delta_rule",
        "test_fused_recurrent_gate_at_zero_matches_reference",
    }


def _module_constant(source: str, name: str) -> float:
    """The value of a module-level numeric constant, read out of the AST.

    Read rather than string-matched: the upstream file goes through fla's formatter and its
    pre-commit hooks, so ``-2.0 ** -20`` may legally come back as ``-2.0**-20``. A guard that
    pins the spelling of a constant goes red on a reformat and teaches everyone to ignore it.
    ``ast.literal_eval`` will not do either — it rejects the unary minus over a power — so this
    walks the small expression grammar the file is allowed to use and refuses anything else.
    """

    def value(node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            got = value(node.operand)
            return -got if isinstance(node.op, ast.USub) else got
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            return value(node.left) ** value(node.right)
        raise AssertionError(f"{name} is not a plain numeric literal expression: {ast.dump(node)}")

    for node in ast.parse(source).body:
        target = None
        # Carried out of the branch with the name: past the `if`, `node` is a bare `ast.stmt`
        # again, and only AnnAssign/Assign have a `.value` to read.
        assigned: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, assigned = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target, assigned = node.targets[0].id, node.value
        if target == name and assigned is not None:
            return value(assigned)
    raise AssertionError(f"{name} is not defined at module level in the upstream test")


@pytest.mark.skipif(_fla_missing(), reason="oss/fla checkout absent — infra/oss.sh has not run")
def test_upstream_test_and_the_reproduction_agree_on_the_two_gates() -> None:
    """Two files, one experiment. The control gate and the scale are the design, not magic numbers,
    and a copy of either that drifted would make the test and the table describe different runs."""
    source = _UPSTREAM_TEST.read_text(encoding="utf-8")
    assert _module_constant(source, "LOG_GATE_BELOW_ZERO") == rp.LOG_GATE_BELOW_ZERO
    assert _module_constant(source, "LOG_GATE_ZERO") == rp.LOG_GATE_ZERO
    assert _module_constant(source, "SCALE") == rp.SCALE
    assert rp.LOG_GATE_BELOW_ZERO == -(2.0**-20)
    assert rp.LOG_GATE_ZERO == 0.0


# =============================================================================================
# gpu — the live measurement, on a box with a device and an installed fla
# =============================================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not _fla_present(), reason="fla not installed")
def test_measure_returns_a_finite_row() -> None:
    """The whole row, at the smallest useful length. Every field the table and the JSON read."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device — fla's kernels are Triton")
    row = rp.measure(64, "cuda", torch.bfloat16)
    assert set(row) >= {
        "T",
        "gated_at_zero_vs_ref",
        "gated_below_zero_vs_ref",
        "ungated_vs_ref",
        "ratio_gated_over_ungated",
        "finite",
    }
    for key in ("gated_at_zero_vs_ref", "gated_below_zero_vs_ref", "ungated_vs_ref"):
        assert math.isfinite(row[key]) and row[key] >= 0.0, key
    assert row["finite"] is True, "the gate-at-zero output contained a non-finite value"


@pytest.mark.gpu
@pytest.mark.skipif(not _fla_present(), reason="fla not installed")
def test_provenance_records_the_device_and_the_commit() -> None:
    prov = rp.provenance("cuda", torch.bfloat16)
    assert set(prov) >= {"torch", "dtype", "device", "fla_commit", "fla"}
