"""K1/B-R5 — tcgen05 + TMEM accumulator in the CuTe DSL. Two tiers of gate, in ascending cost.

  CPU  the tile arithmetic (shared memory, TMEM columns, epilogue sub-tile), which spec shapes TMA
       can address at all, the oracle, and the two error paths that cost a rental hour if they are
       wrong: "this is not a B200" and "the DSL is not installed". Milliseconds, here, no GPU.
  gpu  correctness against the fp32 oracle at the spec shapes. Needs a B200.

There is no drydock tier: this rung is Python, JIT-compiled by nvidia-cutlass-dsl on the device, so
there is no nvcc invocation to run in a container and no ptxas report to assert on. Everything the
dry dock would have caught for a .cu — the shared-memory budget, the accumulator budget, the tile
divisibility — is caught by the CPU tier below instead, from published constants rather than from
a compiler's output. That is a weaker check than reading real SASS and is worth saying out loud.

Spec: experiments/K1/B-R5/spec.md   ·   Map: experiments/K1/B-R5/map.md
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import torch

from scratch_llm.kernels.common.hopper_contracts import (
    ARCH as ARCH_LIMITS,
)
from scratch_llm.kernels.common.hopper_contracts import (
    check_smem_budget,
    tile_covers,
    wave_quantization,
)
from scratch_llm.kernels.gemm.cute_dsl.b_r5 import (
    ARCH,
    DSL_REQUIREMENT,
    MMA_INST_K,
    MMA_INST_TILE_K,
    MMA_TILE_MN,
    RUNG,
    SOURCE,
    TILE_K,
    TMEM_COLUMNS,
    b_r5_gemm,
    mainloop_stages,
    reference_gemm,
    require_cute_dsl,
    tile_config,
    tma_alignment_violations,
)

_REPO = Path(__file__).resolve().parents[3]

#: B200 (GB100). A published die fact, not a measurement — re-check it against the device table
#: infra/bootstrap.sh writes on the box before any number that leans on it.
B200_SM_COUNT = 148

# ---------------------------------------------------------------------------------------------
# The tolerance. HUY SETS THIS, with its argument, before the first measured run.
#
# Same form as H-R1's:  max|C_kernel - C_fp32|  <=  TOL_CONST * sqrt(K) * max|A| * max|B|
# — a bf16 accumulation-order bound. It is NOT obviously the same constant: the operands are still
# bf16 (relative step 2^-8) and the accumulator is still fp32, but the reduction order is not the
# same shape as Hopper's, and the fp32 accumulator here lives in TMEM and is read back through
# tcgen05.ld rather than being the warp's own registers. Whether that changes the constant is a
# question to answer with the measured error distribution, not by assuming.
# ---------------------------------------------------------------------------------------------
TOL_CONST: float | None = None

#: The five shapes the spec measures. Kept identical to bench/kernels/gemm/k1_ladder.py's SHAPES.
SHAPES: dict[str, tuple[int, int, int]] = {
    "sq4096": (4096, 4096, 4096),  # the headline: the number quoted against cuBLASLt bf16
    "rect8192": (8192, 8192, 4096),  # secondary: more waves, same tile — isolates grid effects
    "skinny16": (16, 4096, 4096),  # M < cta tile M: one partial tile row, most of the machine idle
    "npot": (257, 1023, 512),  # non-power-of-two — and, at this rung, not TMA-addressable
    "untuned": (1536, 6144, 2560),  # divides nothing the tile likes; not a tuning target
}

#: The subset TMA can actually address with this wrapper's layouts. Derived, never hand-listed:
#: the arithmetic is in the module and the test below asserts which shape falls out.
TMA_LEGAL = [name for name, mnk in SHAPES.items() if not tma_alignment_violations(*mnk)]


def _hole_is_open() -> bool:
    """True while the kernel body still carries the hole sentinel (same rule as tests/conftest.py)."""
    return 'NotImplementedError("HUY:' in (_REPO / SOURCE).read_text(encoding="utf-8")


def _dsl_installed() -> bool:
    return importlib.util.find_spec("cutlass") is not None


# =============================================================================================
# The hole guard — exactly one per hole (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K1/B-R5", "src/scratch_llm/kernels/gemm/cute_dsl/b_r5.py")
def test_b_r5_kernel_body_is_filled_and_the_class_builds() -> None:
    """Fails while ``_mainloop_and_epilogue`` raises; passes once it is written.

    A DSL rung has no compile step to run on a laptop, so this guard can only assert the two things
    that do not need silicon: that the body exists, and — where the DSL is installed — that the
    kernel class still constructs with it. Correctness is ``test_matches_oracle``, on the B200.
    """
    assert not _hole_is_open(), (
        f"{RUNG} kernel body is still an open hole. Read experiments/K1/B-R5/spec.md and "
        f"experiments/K1/B-R5/map.md items 9-11, then write the mainloop and epilogue in {SOURCE}."
    )
    if not _dsl_installed():
        pytest.skip(
            f"the CuTe DSL is absent here (uv pip install {DSL_REQUIREMENT}); box-only check"
        )
    from scratch_llm.kernels.gemm.cute_dsl.b_r5 import _kernel_class

    assert _kernel_class() is not None


# =============================================================================================
# CPU — the tile arithmetic. Every one of these is a launch failure or a silent slowdown caught
# for $0 instead of at $6/h with a B200 clock running.
# =============================================================================================


def test_tile_constants_are_the_plan_row_and_the_instruction() -> None:
    """128x256x64 is not three tuning knobs: the 64 is the instruction's K times the K-tile count."""
    assert MMA_TILE_MN == (128, 256)
    assert TILE_K == MMA_INST_K * MMA_INST_TILE_K == 64
    cfg = tile_config(1)
    assert cfg.cta_tile_mnk == (128, 256, 64)
    assert cfg.cluster_shape_mn == (1, 1)
    # cta_group::2 divides M across the pair and divides B a second time for staging (:434-435).
    two = tile_config(2)
    assert two.cta_tile_mnk == (64, 256, 64)
    assert two.b_tile_nk == (128, 64)
    assert two.cluster_shape_mn == (2, 1)


def test_one_cta_tile_fits_shared_memory_at_four_stages_and_not_five() -> None:
    """48 KB per stage: four stages plus the epilogue buffer fit sm_100a's 227 KB, five do not.

    The pipeline depth is the first thing a Blackwell GEMM gets wrong, and it is wrong in the most
    expensive possible way — the launch succeeds at three stages and is simply slower, which reads
    in a profile as a mainloop problem rather than as a budget arithmetic mistake.
    """
    cfg = tile_config(1)
    arch = ARCH_LIMITS["sm_100a"]
    assert cfg.smem_bytes_per_stage() == (128 * 64 + 256 * 64) * 2 == 49152
    fits = check_smem_budget(
        bytes_per_stage=cfg.smem_bytes_per_stage(),
        stages=4,
        arch=arch,
        extra_bytes=cfg.epilogue_smem_bytes(),
    )
    # Legal, but only because the launch opts in — which it does, with the full sm_100 capacity.
    assert len(fits) == 1 and fits[0].startswith("OPT-IN REQUIRED"), fits
    over = check_smem_budget(
        bytes_per_stage=cfg.smem_bytes_per_stage(),
        stages=5,
        arch=arch,
        extra_bytes=cfg.epilogue_smem_bytes(),
    )
    assert any("exceeds" in e for e in over), over
    # ...and the wrapper launches with exactly that ceiling, which is upstream's fallback (:223-225).
    assert mainloop_stages(1) == 4


def test_two_cta_halves_the_stage_and_doubles_the_pipeline_depth() -> None:
    """cta_group::2 buys pipeline depth with the same shared memory — the second half of the rung."""
    two = tile_config(2)
    assert two.smem_bytes_per_stage() == (64 * 64 + 128 * 64) * 2 == 24576
    assert mainloop_stages(2) == 8
    fits = check_smem_budget(
        bytes_per_stage=two.smem_bytes_per_stage(),
        stages=8,
        arch=ARCH_LIMITS["sm_100a"],
        extra_bytes=two.epilogue_smem_bytes(),
    )
    assert len(fits) == 1 and fits[0].startswith("OPT-IN REQUIRED"), fits


def test_one_cta_accumulator_consumes_the_entire_tmem() -> None:
    """256 columns/stage x 2 stages = 512 = all of TMEM. This is the rung's binding constraint.

    It is the answer to "why not deepen acc_stage to overlap the epilogue with the next tile's
    MMA": there is nowhere to put a third stage. Only cta_group::2, which halves the column cost,
    creates the room — so the 1-CTA and 2-CTA measurements are not two tunings of one design, they
    are two different points on the one budget.
    """
    cfg = tile_config(1)
    assert cfg.acc_tmem_columns_per_stage == 256
    assert cfg.acc_tmem_columns == TMEM_COLUMNS == 512
    two = tile_config(2)
    assert two.acc_tmem_columns_per_stage == 128
    assert two.acc_tmem_columns == 256 < TMEM_COLUMNS


def test_epilogue_subtile_is_eight_trips_through_tmem_per_output_tile() -> None:
    """(128, 32) for the 1-CTA tile with an fp32 D — so eight TMEM->RMEM->SMEM->GMEM rounds.

    Worth pinning because it is the unit the epilogue's cost is counted in: any measurement of
    "epilogue overhead" on this rung is per-sub-tile times eight, not per output tile.
    """
    assert tile_config(1).epilogue_tile_mn == (128, 32)
    assert tile_config(2).epilogue_tile_mn == (64, 64)


def test_every_spec_shape_has_a_k_this_rung_can_run() -> None:
    """K % 64 == 0 for all five — B-R5 has no K-remainder path and the wrapper says so."""
    for name, (m, n, k) in SHAPES.items():
        assert k % TILE_K == 0, f"{name}: K={k} is not a multiple of TILE_K={TILE_K}"
        assert tile_covers(m=m, n=n, k=k, tile_m=128, tile_n=256, tile_k=TILE_K)["k"], name


def test_npot_is_the_one_spec_shape_tma_cannot_address() -> None:
    """N=1023 is not 16 B aligned for either the bf16 B load or the fp32 D store.

    A TMA descriptor is not negotiable about this (blackwell_helpers.py:106-132), and the honest
    response is to reject the shape rather than to insert a .contiguous() copy that would be timed
    as part of the GEMM. So this rung measures four of the five spec shapes, and the ledger row for
    npot at B-R5 does not exist rather than being a slower number with an asterisk.
    """
    assert TMA_LEGAL == ["sq4096", "rect8192", "skinny16", "untuned"]
    violations = tma_alignment_violations(*SHAPES["npot"])
    assert len(violations) == 2
    assert any("B TMA load" in v for v in violations)
    assert any("D TMA store" in v for v in violations)


def test_n_predication_is_uncovered_and_the_shape_that_would_cover_it_is_verified() -> None:
    """None of the four legal shapes leaves a partial N tile, so the N epilogue bound is untested.

    M predication IS covered — skinny16's M=16 is a partial 128-row tile. N is not: 4096, 8192 and
    6144 all divide 256, and the one non-dividing shape (npot) is TMA-illegal. The replacement is
    asserted here rather than suggested in prose, so the lead can paste it into k1_ladder.SHAPES
    knowing it is both TMA-legal and genuinely predicating. If such a shape is added, delete this
    test's first assertion — the gap it records will be closed.
    """
    covers = {
        name: tile_covers(
            m=SHAPES[name][0],
            n=SHAPES[name][1],
            k=SHAPES[name][2],
            tile_m=128,
            tile_n=256,
            tile_k=TILE_K,
        )
        for name in TMA_LEGAL
    }
    assert any(not c["m"] for c in covers.values()), (
        "M predication is untested by every legal shape"
    )
    assert all(c["n"] for c in covers.values()), (
        "a legal spec shape now exercises N predication — delete this assertion, the gap is closed"
    )
    m, n, k = 257, 1032, 512  # the smallest npot-shaped replacement that TMA can address
    assert tma_alignment_violations(m, n, k) == []
    replacement = tile_covers(m=m, n=n, k=k, tile_m=128, tile_n=256, tile_k=TILE_K)
    assert replacement == {"m": False, "n": False, "k": True}


def test_skinny_shape_is_grid_limited_and_the_suite_knows_it() -> None:
    """M=16 cannot fill a B200. Any %-of-cuBLASLt there measures occupancy, not the mainloop.

    Worse here than on Hopper: the tile is twice as wide in N, so one skinny row of tiles is 16
    CTAs against 148 SMs. Asserted rather than commented so a tile-shape change cannot quietly turn
    the skinny row into a number that reads like a mainloop regression.
    """
    m, n, _ = SHAPES["skinny16"]
    wq = wave_quantization(m=m, n=n, tile_m=128, tile_n=256, sm_count=B200_SM_COUNT)
    assert wq.full_waves == 0 and wq.tail_utilization < 0.25


def test_reference_is_fp32_over_the_same_bf16_operands() -> None:
    """The oracle must not be a different problem — it up-casts, it does not re-generate."""
    torch.manual_seed(0)
    a = torch.randn(64, 32, dtype=torch.bfloat16)
    b = torch.randn(32, 48, dtype=torch.bfloat16)
    ref = reference_gemm(a, b)
    assert ref.dtype is torch.float32
    assert torch.equal(ref, torch.matmul(a.float(), b.float()))


def test_missing_dsl_names_the_pinned_wheel_to_install() -> None:
    """ "You have not installed the compiler" must not arrive as a bare ModuleNotFoundError.

    This is the first thing that happens on a fresh B200 pod, and the difference between a message
    with the pinned install line in it and a traceback ending in ``No module named 'cutlass'`` is
    several minutes of rented time spent guessing at a package name.
    """
    if _dsl_installed():
        pytest.skip("the CuTe DSL is installed here — the missing-dependency path is unreachable")
    with pytest.raises(ImportError, match=re.escape(DSL_REQUIREMENT)):
        require_cute_dsl()


def test_arch_gate_fires_before_anything_else_and_names_the_arch() -> None:
    """Off sm_100 the wrapper must refuse by arch, not by whatever the DSL fails at later.

    sm_120 is the trap this exists for: client Blackwell has no tcgen05 and no TMEM, yet it is
    ``cc >= (10, 0)`` and passes every floor-style check. require_arch matches exactly.
    """
    if torch.cuda.is_available() and tuple(torch.cuda.get_device_capability(0)) == ARCH:
        pytest.skip("this IS an sm_100 device — the rejection path is unreachable here")
    with pytest.raises(RuntimeError, match="sm_100"):
        b_r5_gemm(
            torch.zeros(128, 64, dtype=torch.bfloat16),
            torch.zeros(64, 256, dtype=torch.bfloat16),
        )


# =============================================================================================
# gpu — correctness against the oracle. B200 only.
# =============================================================================================


def _require_tolerance() -> float:
    if TOL_CONST is None:
        pytest.fail(
            "TOL_CONST is unset. The correctness gate cannot run without a tolerance, and picking "
            "one is the rung's numerics lesson: set it in this file and write the one-line argument "
            "into experiments/K1/B-R5/spec.md's 'Correctness gate' line before the first measured run."
        )
    return TOL_CONST


def _assert_matches_oracle(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    ref = reference_gemm(a, b)
    assert out.shape == ref.shape, f"{out.shape} != {ref.shape}"
    assert out.dtype is torch.float32, (
        f"kernel must return fp32 (the accumulator's dtype), got {out.dtype}"
    )
    k = a.shape[1]
    bound = (
        _require_tolerance()
        * (k**0.5)
        * a.float().abs().max().item()
        * b.float().abs().max().item()
    )
    err = (out - ref).abs().max().item()
    assert err <= bound, (
        f"max|Δ| = {err:.4e} > {bound:.4e} = TOL_CONST * sqrt({k}) * max|A| * max|B|. "
        f"A wrong TMEM sub-tile mapping shows up here as a large error on a STRUCTURED subset of "
        f"the output — print (out - ref).abs().amax(dim=1) and look for a period of 32 columns "
        f"(the epilogue sub-tile's N) before widening this."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("shape_name", TMA_LEGAL)
def test_matches_oracle(shape_name: str) -> None:
    """Every TMA-addressable spec shape against the fp32 oracle at the spec's tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m, n, k = SHAPES[shape_name]
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches_oracle(b_r5_gemm(a, b), a, b)
    del a, b
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_two_cta_matches_the_same_oracle() -> None:
    """cta_group::2 is a different instruction and a different accumulator split, same answer."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(0)
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    _assert_matches_oracle(b_r5_gemm(a, b, cta_group=2), a, b)


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3], ids=["tiny", "unit", "large"])
def test_matches_oracle_across_magnitudes(scale: float) -> None:
    """bf16 has 8 mantissa bits and a huge exponent range; the tolerance must be scale-relative."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(1)
    a = (torch.randn(256, 128, device="cuda", dtype=torch.float32) * scale).bfloat16()
    b = (torch.randn(128, 256, device="cuda", dtype=torch.float32) * scale).bfloat16()
    _assert_matches_oracle(b_r5_gemm(a, b), a, b)


@pytest.mark.gpu
def test_rejects_bad_inputs() -> None:
    """The guard clauses, on the device where they can actually be reached."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    def z(r: int, c: int, dt: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(r, c, device="cuda", dtype=dt)

    with pytest.raises(TypeError, match="bfloat16"):
        b_r5_gemm(z(128, 64, torch.float16), z(64, 256))
    with pytest.raises(ValueError, match="inner dimensions"):
        b_r5_gemm(z(128, 64), z(128, 256))
    with pytest.raises(ValueError, match=f"K % {TILE_K}"):
        b_r5_gemm(z(128, 32), z(32, 256))
    with pytest.raises(ValueError, match="TMA"):
        b_r5_gemm(z(128, 64), z(64, 1023))
    with pytest.raises(ValueError, match="cta_group"):
        b_r5_gemm(z(128, 64), z(64, 256), cta_group=3)


@pytest.mark.gpu
def test_handles_nan_and_inf_without_masking_them() -> None:
    """A NaN in must produce a NaN out — a kernel that swallows them hides real training bugs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    a = torch.ones(128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(64, 256, device="cuda", dtype=torch.bfloat16)
    a[0, 0] = float("nan")
    b[0, 1] = float("inf")
    out = b_r5_gemm(a, b)
    assert torch.isnan(out[0, 0]), "NaN was swallowed"
    assert not torch.isfinite(out[:, 1]).all(), "Inf was swallowed"
