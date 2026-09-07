"""K3/PR-2 — the analytic KDA peak check. Two tiers, because there is no third one to have.

  hole  the model itself. Strict-xfail while ``kda_chunk_scan_peak_bytes`` is a ``# HUY:`` hole;
        an ordinary test that must pass the moment it is filled. It asserts only properties every
        correct model has — sign, integrality, and the direction each of the six shape parameters
        moves the answer — never a value, because the value is exactly what is not delegated.
  CPU   the plumbing around the model: shape validation, budget arithmetic, the margin, the
        refusal and what its message has to say. Run against a stub model, so all of it is
        exercised today. Microseconds; no torch, no GPU, no vLLM.

There is deliberately no gpu tier. Nothing here touches a device: the whole value of direction 3
is that it decides before the first allocation. The device-side half of this rung is
``experiments/K3/PR-2/measure_peak_alloc.py``, which is a script and not a test because its
output is a measured curve for the ledger, not a pass/fail.

Spec: experiments/K3/PR-2/spec.md   ·   Upstream: vllm-project/vllm#54775
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from types import ModuleType

import pytest

from scratch_llm.kernels.gemm.cuda._k1_loader import workspace_root
from scratch_llm.kernels.linear_attn.kda_memory import (
    FLA_CHUNK_SIZE,
    MAX_HEAD_K,
    RUNG,
    SAFETY_MARGIN,
    KdaScanBudget,
    KdaScanShape,
    KdaScanWouldNotFit,
    check_kda_chunk_scan_headroom,
    kda_chunk_scan_peak_bytes,
    plan_kda_chunk_scan,
    scan_shape_for_profile_run,
)

#: The shape #54775 quotes: T = 81,920 tokens through one KDA layer at BT = 64. Used only where a
#: test needs a *realistic* magnitude; nothing here asserts a byte count at it.
ISSUE_SHAPE = KdaScanShape(
    num_seqs=256,
    num_tokens=81_920,
    num_v_heads=96,
    head_k=128,
    head_v=128,
)

#: A small shape, so the plumbing tests read as arithmetic rather than as a memory report.
SMALL_SHAPE = KdaScanShape(
    num_seqs=2,
    num_tokens=256,
    num_v_heads=4,
    head_k=64,
    head_v=64,
)


def _stub_model(shape: KdaScanShape) -> int:
    """A stand-in with the right *shape* of behaviour and none of the right numbers.

    Monotone in every parameter the real model is monotone in, and integral — enough to drive the
    plumbing — but deliberately not a memory model: no chunk-count bound, no cross-call live set,
    no dtype split. It exists so that a reader cannot mistake it for a draft of Huy's answer, and
    so that a stub accidentally left wired in as the default is caught by
    ``test_the_default_model_is_the_hole_and_not_a_stub``.
    """
    return (
        shape.num_tokens * shape.num_v_heads * shape.head_v * shape.elem_bytes
        + shape.num_seqs * shape.head_k
    )


# =============================================================================================
# The hole guard — exactly one (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K3/PR-2", "src/scratch_llm/kernels/linear_attn/kda_memory.py")
def test_memory_model_is_written_and_monotone_where_the_kernel_is() -> None:
    """Fails while the model raises; passes when it is filled and behaves like a memory model.

    Every assertion is a direction, not a value. Each one comes off an allocation line rather than
    off intuition, so a model that violates one is wrong about the kernel and not merely about
    taste:

      more tokens        h is (B, NT, H, V, K) with NT ~ T/BT, and v_new is (B, T, H, V)  -> up
      more heads         H indexes both                                                   -> up
      larger K, V        both index h; V indexes v_new                                     -> up
      wider elements     h and v_new are k.dtype                                           -> up
      more sequences     varlen NT is sum_i ceil(T_i/BT); splitting a fixed token budget
                         across more sequences can only add partial chunks                 -> up

    The last one is the reason direction 1 needs a multi-sequence profile batch, so a model that
    ignores ``num_seqs`` would make this PR's other half pointless.

    ``chunk_size`` is deliberately NOT asserted, in either direction. It looks monotone — NT ~ 1/BT
    so ``h`` shrinks as BT grows — but the cross-call live set this module's docstring tells Huy to
    include runs the other way: ``Aqk`` is ``(B, T, HV, BT)`` (chunk_intra.py:488), linear in BT.
    Which term wins turns on K*V against BT^2, so no small shape settles it and a correct model can
    land on either side. The BT dependence is a measurement, not an assertion:
    ``measure_peak_alloc.py --chunk 32`` against ``--chunk 64`` is where it gets decided.
    """
    base = kda_chunk_scan_peak_bytes(SMALL_SHAPE)
    assert isinstance(base, int) and not isinstance(base, bool), (
        f"the model must return an int of bytes, got {type(base).__name__}"
    )
    assert base > 0, "a KDA layer's chunked scan cannot be free"

    def bytes_for(**changes: object) -> int:
        return kda_chunk_scan_peak_bytes(replace(SMALL_SHAPE, **changes))  # type: ignore[arg-type]

    assert bytes_for(num_tokens=SMALL_SHAPE.num_tokens * 2) > base, "not linear in T"
    assert bytes_for(num_v_heads=SMALL_SHAPE.num_v_heads * 2) > base, "H indexes h and v_new"
    assert bytes_for(head_k=SMALL_SHAPE.head_k * 2) > base, "K indexes h"
    assert bytes_for(head_v=SMALL_SHAPE.head_v * 2) > base, "V indexes h and v_new"
    assert bytes_for(elem_bytes=4) > base, "h and v_new are k.dtype"
    assert bytes_for(num_seqs=SMALL_SHAPE.num_seqs * 2) >= base, (
        "varlen NT = sum_i ceil(T_i/BT); more sequences over the same token budget can only add "
        "partial chunks (chunk_delta_h.py:347)"
    )

    # The check is armed against a real config only once the margin has a measured basis. This is
    # not a second hole: SAFETY_MARGIN stays None until the pred-err table exists, and None means
    # an exact comparison, which is a defensible default. It is asserted here so that filling the
    # model without reading experiments/K3/PR-2/results/ is at least noticed.
    assert SAFETY_MARGIN is None or 0.0 <= SAFETY_MARGIN < 1.0, (
        "SAFETY_MARGIN must be None (exact) or a fraction in [0, 1) backed by measured pred err"
    )


# =============================================================================================
# CPU — the shape refuses batches the kernel cannot build
# =============================================================================================


def test_rung_identity_matches_the_experiment_directory() -> None:
    assert RUNG == "K3/PR-2"


def test_chunk_default_mirrors_upstreams() -> None:
    """``FLA_CHUNK_SIZE = 64`` at vllm/third_party/flash_linear_attention/ops/utils.py:31.

    Mirrored rather than imported because this module must import on a laptop with no vLLM. A
    mirror that drifts is worse than no mirror, so it is asserted where a reader will see it.
    """
    assert FLA_CHUNK_SIZE == 64
    assert (
        KdaScanShape(num_seqs=1, num_tokens=1, num_v_heads=1, head_k=1, head_v=1).chunk_size == 64
    )


@pytest.mark.parametrize(
    "field", ["num_seqs", "num_tokens", "num_v_heads", "head_k", "head_v", "chunk_size"]
)
@pytest.mark.parametrize("bad", [0, -1])
def test_shape_rejects_non_positive_dimensions(field: str, bad: int) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a positive int"):
        replace(SMALL_SHAPE, **{field: bad})


def test_shape_rejects_bools_masquerading_as_dimensions() -> None:
    """``True`` is an ``int`` in Python, and ``num_v_heads=True`` would silently mean one head.

    A config plumbed through a CLI flag can produce exactly that, and a check that then predicts
    one head's worth of bytes would pass a config that OOMs — the failure this whole rung exists
    to prevent, reintroduced by a type nobody looked at.
    """
    with pytest.raises(ValueError, match="num_v_heads must be a positive int"):
        replace(SMALL_SHAPE, num_v_heads=True)


def test_shape_rejects_more_sequences_than_tokens() -> None:
    """Every scheduled sequence carries at least one token; the NT bound leans on that."""
    with pytest.raises(ValueError, match="cannot be built"):
        replace(SMALL_SHAPE, num_seqs=SMALL_SHAPE.num_tokens + 1)


def test_shape_rejects_head_dim_the_kernel_asserts_against() -> None:
    """chunk_delta_h.py:350 — ``assert K <= 256``. Rejecting here names the number; the assert does not."""
    ok = replace(SMALL_SHAPE, head_k=MAX_HEAD_K)
    assert ok.head_k == MAX_HEAD_K
    with pytest.raises(ValueError, match="exceeds the kernel's ceiling"):
        replace(SMALL_SHAPE, head_k=MAX_HEAD_K + 1)


@pytest.mark.parametrize("bad", [0, 3, 8])
def test_shape_rejects_element_widths_no_dtype_has(bad: int) -> None:
    with pytest.raises(ValueError, match="elem_bytes must be"):
        replace(SMALL_SHAPE, elem_bytes=bad)


def test_shape_is_frozen() -> None:
    """A budget computed from a shape that was mutated afterwards describes no batch at all."""
    with pytest.raises(Exception):  # noqa: B017  (FrozenInstanceError is a dataclasses detail)
        SMALL_SHAPE.num_tokens = 1  # type: ignore[misc]


# =============================================================================================
# CPU — the profile-run adapter mirrors vLLM's own dummy-batch arithmetic
# =============================================================================================


def test_profile_shape_takes_the_engine_ceilings_not_a_model_config() -> None:
    """gpu_model_runner.py:6014 — ``num_reqs = min(num_tokens, max_num_reqs)``.

    Mirrored exactly, because the check's claim is "the worst batch this engine can schedule", and
    the engine's own builder is the only authority on what that is.
    """
    shape = scan_shape_for_profile_run(
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        num_v_heads=96,
        head_k=128,
        head_v=128,
    )
    assert shape.num_tokens == 8192
    assert shape.num_seqs == 256
    assert shape.chunk_size == FLA_CHUNK_SIZE
    assert shape.elem_bytes == 2


def test_profile_shape_clamps_sequences_to_the_token_budget() -> None:
    """A tiny token budget cannot host max_num_seqs sequences — and an unclamped shape would be
    rejected by ``KdaScanShape`` as unbuildable, turning a legal (if odd) config into a crash."""
    shape = scan_shape_for_profile_run(
        max_num_batched_tokens=16,
        max_num_seqs=256,
        num_v_heads=4,
        head_k=64,
        head_v=64,
    )
    assert shape.num_seqs == 16
    assert shape.num_tokens == 16


# =============================================================================================
# CPU — the budget arithmetic, against a stub model
# =============================================================================================


def test_plan_returns_the_models_answer_untouched() -> None:
    budget = plan_kda_chunk_scan(SMALL_SHAPE, 1 << 30, model=_stub_model)
    assert isinstance(budget, KdaScanBudget)
    assert budget.predicted_bytes == _stub_model(SMALL_SHAPE)
    assert budget.available_bytes == 1 << 30
    assert budget.shape is SMALL_SHAPE


def test_no_margin_means_an_exact_comparison_at_the_boundary() -> None:
    """``margin=None`` withholds nothing, so predicted == available still fits.

    The boundary matters: a model whose error bar has not been measured has no business
    withholding an invented fraction, and it has no business refusing the config it exactly
    predicts either.
    """
    exact = _stub_model(SMALL_SHAPE)
    assert plan_kda_chunk_scan(SMALL_SHAPE, exact, model=_stub_model).fits
    assert not plan_kda_chunk_scan(SMALL_SHAPE, exact - 1, model=_stub_model).fits


def test_margin_withholds_that_fraction_of_the_headroom() -> None:
    exact = _stub_model(SMALL_SHAPE)
    tight = plan_kda_chunk_scan(SMALL_SHAPE, exact, model=_stub_model, margin=0.1)
    assert tight.budget_bytes == int(exact * 0.9)
    assert not tight.fits
    assert tight.headroom_bytes < 0

    roomy = plan_kda_chunk_scan(SMALL_SHAPE, exact * 2, model=_stub_model, margin=0.1)
    assert roomy.fits
    assert roomy.headroom_bytes > 0


def test_margin_zero_is_not_the_same_object_as_no_margin_but_is_the_same_budget() -> None:
    """0.0 and None must agree numerically, or a caller reading a log cannot tell them apart."""
    exact = _stub_model(SMALL_SHAPE)
    zero = plan_kda_chunk_scan(SMALL_SHAPE, exact, model=_stub_model, margin=0.0)
    none = plan_kda_chunk_scan(SMALL_SHAPE, exact, model=_stub_model, margin=None)
    assert zero.budget_bytes == none.budget_bytes
    assert zero.margin == 0.0 and none.margin is None


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_plan_rejects_a_margin_outside_zero_to_one(bad: float) -> None:
    with pytest.raises(ValueError, match="margin must lie"):
        plan_kda_chunk_scan(SMALL_SHAPE, 1 << 30, model=_stub_model, margin=bad)


def test_plan_rejects_negative_or_non_integer_headroom() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        plan_kda_chunk_scan(SMALL_SHAPE, -1, model=_stub_model)
    with pytest.raises(TypeError, match="must be an int"):
        plan_kda_chunk_scan(SMALL_SHAPE, 1.5, model=_stub_model)  # type: ignore[arg-type]


@pytest.mark.parametrize("bogus", [-1, 1.5, "8 GiB", None, True])
def test_plan_rejects_a_model_that_does_not_return_bytes(bogus: object) -> None:
    """A model returning GiB-as-float, or ``None`` from a forgotten return, must not become a
    budget: the comparison would still "work" and would silently admit anything."""
    with pytest.raises((TypeError, ValueError)):
        plan_kda_chunk_scan(SMALL_SHAPE, 1 << 30, model=lambda _shape: bogus)  # type: ignore[arg-type,return-value]


def test_report_states_the_shape_that_produced_the_number() -> None:
    """A memory line without its batch shape is unactionable — #54775's whole difficulty."""
    line = plan_kda_chunk_scan(ISSUE_SHAPE, 1 << 34, model=_stub_model).report()
    assert "GiB" in line
    for token in ("num_seqs=256", "num_tokens=81920", "H=96", "K=128", "V=128", "BT=64"):
        assert token in line, f"report() dropped {token}: {line}"


# =============================================================================================
# CPU — the refusal
# =============================================================================================


def test_check_returns_the_budget_when_it_fits() -> None:
    budget = check_kda_chunk_scan_headroom(SMALL_SHAPE, 1 << 30, model=_stub_model)
    assert budget.fits
    assert budget.predicted_bytes == _stub_model(SMALL_SHAPE)


def test_check_refuses_and_names_every_knob_that_moves_the_number() -> None:
    """The error this replaces is a CUDA OOM inside a Triton launch, which names nothing.

    Each knob listed has to actually be in the message with its current value, because an operator
    reading it at 03:00 will turn exactly one of them and needs to know where it stands now.
    """
    with pytest.raises(KdaScanWouldNotFit) as excinfo:
        check_kda_chunk_scan_headroom(ISSUE_SHAPE, 1024, model=_stub_model)
    msg = str(excinfo.value)
    assert "--max-num-batched-tokens" in msg and "81920" in msg
    assert "--max-num-seqs" in msg and "256" in msg
    assert "--gpu-memory-utilization" in msg
    assert "chunk size" in msg and "64" in msg
    assert "54775" in msg, "the message must point at the issue that explains the buffer"
    assert "Short by" in msg


def test_refusal_is_a_runtime_error_so_it_is_not_swallowed_as_a_config_typo() -> None:
    assert issubclass(KdaScanWouldNotFit, RuntimeError)
    assert not issubclass(KdaScanWouldNotFit, ValueError)


def test_zero_headroom_refuses_rather_than_dividing_by_it() -> None:
    with pytest.raises(KdaScanWouldNotFit):
        check_kda_chunk_scan_headroom(SMALL_SHAPE, 0, model=_stub_model)


def test_the_default_model_is_the_hole_and_not_a_stub() -> None:
    """The one way this rung can ship broken: someone wires a placeholder in as the default.

    ``plan_kda_chunk_scan`` and ``check_kda_chunk_scan_headroom`` must both default to the real
    model, so that calling them without an explicit ``model=`` while the hole is open raises
    ``NotImplementedError`` instead of quietly admitting every config.
    """
    for fn in (plan_kda_chunk_scan, check_kda_chunk_scan_headroom):
        kwdefaults = fn.__kwdefaults__
        assert kwdefaults is not None, (
            f"{fn.__name__} has no keyword-only defaults at all — `model=` must stay keyword-only "
            f"with the real model as its default"
        )
        default = kwdefaults["model"]
        assert default is kda_chunk_scan_peak_bytes, (
            f"{fn.__name__} defaults to {default!r}, not the model — a stub default would make "
            f"the startup check pass every config"
        )


# =============================================================================================
# CPU — the measurement script's device-independent half
# =============================================================================================
#
# `measure_peak_alloc.py` cannot be imported: it lives in the sibling `ladders` workspace, is not
# in a package, and must stay runnable as a bare file on a rented box that has vLLM and has never
# heard of scratch_llm. So it is loaded by path, exactly as K3/PR-1's script is. Everything in it
# that does not touch a device is decidable here, and every one of those parts is a place a wrong
# grid or a wrong split would silently produce a curve about the wrong batch.


def _measure_script() -> ModuleType:
    script = workspace_root() / "experiments" / "K3" / "PR-2" / "measure_peak_alloc.py"
    if not script.is_file():
        pytest.skip(f"{script} not found — the ladders workspace is not resolvable from here")
    name = "_k3_pr2_measure_peak_alloc"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None, f"cannot load {script}"
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__].__dict__`, so an unregistered module raises AttributeError on
    # the decorator rather than anywhere near the dataclass.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("num_tokens", "num_seqs"), [(256, 2), (81_920, 256), (4096, 256), (7, 3)])
def test_token_split_conserves_the_budget_and_matches_vllms_own(
    num_tokens: int, num_seqs: int
) -> None:
    """gpu_model_runner.py:6014-6017 — even split, remainder onto the last sequence.

    The split is not cosmetic. It sets the varlen chunk count NT = sum_i ceil(T_i/BT), which is the
    leading dimension of `h`; a split that loses or invents tokens measures a batch the engine
    cannot schedule, and the curve would then falsify the model against fiction.
    """
    m = _measure_script()
    lengths = m._split_tokens(num_tokens, num_seqs)
    assert len(lengths) == num_seqs
    assert sum(lengths) == num_tokens
    assert all(n >= 1 for n in lengths), "a sequence with no tokens is not a sequence"
    assert lengths[:-1] == [num_tokens // num_seqs] * (num_seqs - 1)
    assert lengths[-1] == num_tokens // num_seqs + num_tokens % num_seqs


def test_grid_shapes_clamp_sequences_the_way_the_engine_does() -> None:
    """A 16-token point of the grid cannot host 256 sequences; the shape would be unbuildable."""
    m = _measure_script()
    args = m.argparse.Namespace(
        seqs=256, heads=4, head_k=64, head_v=64, chunk=m.FLA_CHUNK_SIZE, dtype="bfloat16"
    )
    assert m._shape(args, 16).num_seqs == 16
    assert m._shape(args, 81_920).num_seqs == 256
    assert m._shape(args, 81_920).elem_bytes == 2


def test_prediction_is_absent_rather_than_fatal_while_the_hole_is_open() -> None:
    """`--dry-run` has to stay useful on the day the grid is designed and the model is not written.

    If `_predict` let NotImplementedError escape, the one command that costs nothing would be the
    one command that cannot run — and the grid, the shapes and the box invocation are all decidable
    without the model.
    """
    m = _measure_script()
    shape = m.KdaScanShape(num_seqs=2, num_tokens=256, num_v_heads=4, head_k=64, head_v=64)
    assert m._predict(shape) is None or isinstance(m._predict(shape), int)


def test_dry_run_exits_clean_and_touches_no_device(capsys: pytest.CaptureFixture[str]) -> None:
    m = _measure_script()
    assert m.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "81920" in out, "the headline T from #54775 must be on the grid"
    assert "no device was touched" in out


def test_default_grid_spans_a_decade_so_a_non_linear_term_cannot_hide() -> None:
    """Two points fit any curve. The claim under test is that the buffers are linear in T, and a
    grid that does not span at least a decade cannot tell linear from mildly super-linear."""
    m = _measure_script()
    assert min(m.DEFAULT_T_GRID) * 10 <= max(m.DEFAULT_T_GRID)
    assert 81_920 in m.DEFAULT_T_GRID
    assert list(m.DEFAULT_T_GRID) == sorted(m.DEFAULT_T_GRID)
