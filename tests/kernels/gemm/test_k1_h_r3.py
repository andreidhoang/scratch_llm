"""K1/H-R3 — warp-specialised multistage pipeline. Three tiers of gate, in ascending cost.

  CPU      the wrapper's contracts, the oracle, and — the point of this rung — the pipeline
           arithmetic: how many stages fit, that the launch must opt in to the shared memory it
           asks for, that both setmaxnreg immediates are legal and integrate to a CTA that is
           resident, and that the two shared-memory descriptors the .cu hard-codes are the ones
           CUTLASS's canonical layouts imply. Milliseconds, here.
  drydock  what nvcc actually generated: the warpgroup MMA is issued with the MN-major B this rung
           is forced into (`.tnspB`), the TMA copy is the 5-D form, nothing spills to local memory.
           Needs the CUDA container (infra/drydock.sh), still no GPU.
  gpu      correctness against the fp32 oracle at the spec shapes. Needs an H100.

Only the last of those can say the kernel is right, and only it costs money — so the first two are
built to catch everything they possibly can before it is spent. Nearly every H-R3 failure mode is
arithmetic (a stage too many, a missing opt-in, an offset from the wrong layout) and every one of
those is decidable on this laptop.

Spec: experiments/K1/H-R3/spec.md   ·   Map: experiments/K1/H-R3/map.md
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest
import torch

from scratch_llm._workspace import workspace_root
from scratch_llm.kernels.common.hopper_contracts import (
    ARCH,
    SwizzleMode,
    check_register_budget,
    check_setmaxnreg,
    check_smem_budget,
    check_tma_tensor_map,
    descriptor_fields,
    max_stages,
    tile_covers,
    wave_quantization,
    wgmma_smem_descriptor,
)
from scratch_llm.kernels.gemm.cuda._k1_loader import HoleOpenError, hole_is_open, source_for
from scratch_llm.kernels.gemm.cuda.h_r3 import (
    BARRIER_BYTES,
    BYTES_PER_STAGE,
    CONSUMER_REGS,
    NUM_CONSUMERS,
    PRODUCER_REGS,
    REGS_AFTER_SPLIT,
    REGS_PER_THREAD_UNIFORM,
    RUNG,
    SMEM_ALIGN,
    SMEM_BYTES,
    SOURCE,
    STAGES,
    THREADS_PER_CTA,
    TILE_K,
    TILE_M,
    TILE_N,
    TMA_INNER_ELEMS,
    h_r3_gemm,
    k_major_descriptor_offsets,
    mn_major_descriptor_offsets,
    reference_gemm,
    shape_violations,
    tma_tensor_map,
)

# workspace_root(), not a `parents[]` walk: scratch_llm is a SYMLINK inside the workspace, so
# `.resolve()` lands on the link target's parent and every path built from it points OUTSIDE the
# workspace. The drydock assertions below then find no artifacts, skip, and report green while
# checking nothing — which is exactly what they were doing: 4 skipped, 0 run.
_WORKSPACE = workspace_root()
_DRYDOCK = _WORKSPACE / "experiments" / "K1" / "H-R3" / "drydock"
#: infra/drydock.sh names its artifacts per SOURCE, not per rung — a rung may own several .cu
#: and per-rung names let the last one compiled overwrite the evidence for all the others.
_STEM = SOURCE.removesuffix(".cu")
_SM90 = ARCH["sm_90a"]

# ---------------------------------------------------------------------------------------------
# The tolerance. HUY SETS THIS, with its argument, before the first measured run.
#
# The spec's form is  max|C_kernel - C_fp32|  <=  TOL_CONST * sqrt(K) * max|A| * max|B|
# — a bf16 accumulation-order bound: each output element is a sum of K products, the tensor core
# accumulates in fp32 but the *operands* were rounded to bf16 (8 explicit mantissa bits, so a
# relative step of 2^-8), and independent rounding errors grow as sqrt(K) rather than K.
#
# H-R3 does not change the numerics of H-R1 — same bf16 operands, same fp32 accumulator, same
# m64nNk16 instruction — but it does change the *order*: four k-strips per stage across a
# four-deep ring, two warpgroups splitting the rows. If the constant that worked at H-R1 has to
# move here, that is a finding about the reduction order and belongs in the spec's Result block.
#
# Leaving it None is deliberate. Choosing the constant IS the rung's numerics lesson: too loose and
# the test cannot fail, too tight and a correct kernel whose reduction order differs from cuBLAS's
# is rejected. Nobody can pick it for him and have him learn anything.
# ---------------------------------------------------------------------------------------------
TOL_CONST: float | None = None

#: The shapes the spec measures. Each is here for a reason, and the reason is the comment. They are
#: NOT identical to bench/kernels/gemm/k1_ladder.py's registry: that file's `npot` is (257, 1023,
#: 512), and N=1023 is not expressible as a whole number of 64-element swizzle atoms, so B's tensor
#: map cannot be built for it. `npot64` is the same idea inside this rung's constraint.
SHAPES: dict[str, tuple[int, int, int]] = {
    "sq4096": (4096, 4096, 4096),  # the headline: the number quoted against cuBLAS bf16
    "rect8192": (8192, 8192, 4096),  # secondary: more waves, same tile — isolates grid effects
    "skinny16": (16, 4096, 4096),  # M < TILE_M: one partial tile row, most of the machine idle
    "npot64": (257, 1088, 512),  # M and N both predicated; N still a multiple of 64
    "untuned": (1536, 6144, 2560),  # divides nothing the tile likes; not a tuning target
}


def _src() -> str:
    return source_for(SOURCE).read_text(encoding="utf-8")


def _define(src: str, macro: str) -> int:
    m = re.search(rf"^#define {macro} (\d+)\s*(?://.*)?$", src, re.MULTILINE)
    assert m, f"#define {macro} not found in csrc/gemm/{SOURCE}"
    return int(m.group(1))


# =============================================================================================
# The hole guard — exactly one per hole (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K1/H-R3", "csrc/gemm/h_r3_ws_bf16_sm90.cu")
def test_h_r3_kernel_body_is_filled_and_compiles_clean() -> None:
    """Fails while the mainloop is a ``#error``; passes when it compiles clean for sm_90a.

    This is the CPU-side half of the gate and it is genuinely useful on its own: a kernel that does
    not compile, spills registers, or overflows shared memory is not worth renting an H100 to find
    out about. Correctness still needs the GPU — that is ``test_matches_oracle`` below.
    """
    assert not hole_is_open(SOURCE), (
        f"{RUNG} kernel body is still an open hole. Read experiments/K1/H-R3/spec.md and "
        f"experiments/K1/H-R3/map.md, then write the producer's TMA loop, the consumer's wgmma "
        f"loop and the epilogue in csrc/gemm/{SOURCE}."
    )
    if shutil.which("docker") is None:
        pytest.skip("docker absent — cannot run the dry-dock compile (infra/drydock.sh)")
    proc = subprocess.run(
        ["bash", str(_WORKSPACE / "infra" / "drydock.sh"), "compile", SOURCE],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"dry-dock compile failed:\n{proc.stdout}\n{proc.stderr}"


# =============================================================================================
# CPU — contracts that hold with no device at all
# =============================================================================================


def test_source_and_wrapper_agree_on_every_shared_constant() -> None:
    """The ``.cu``'s defines and the wrapper's constants must not drift.

    They are used for different things — one generates code, the other feeds the shared-memory,
    register and descriptor arithmetic below — so nothing else would notice if they diverged, and
    the arithmetic would then be checking a kernel that does not exist.
    """
    src = _src()
    for macro, expected in (
        ("BM", TILE_M),
        ("BN", TILE_N),
        ("BK", TILE_K),
        ("STAGES", STAGES),
        ("NUM_CONSUMERS", NUM_CONSUMERS),
        ("PRODUCER_REGS", PRODUCER_REGS),
        ("CONSUMER_REGS", CONSUMER_REGS),
        ("TMA_INNER_ELEMS", TMA_INNER_ELEMS),
        ("SMEM_ALIGN", SMEM_ALIGN),
    ):
        assert _define(src, macro) == expected, f"{macro} disagrees with h_r3.py"


def test_stage_count_is_the_most_that_fits_and_the_source_says_so() -> None:
    """4 stages, because 4 is what 227 KB holds — not because 4 looked like a nice number.

    49152 B per stage: 4 x 49152 = 196608 fits, 5 x 49152 = 245760 does not. fast.cu's M7, M8 and
    M9 all ship 3 with no comment saying why (map.md's one open question), so this rung takes the
    arithmetic's answer and leaves ``-DSTAGES=3`` as the A/B on the box.
    """
    assert BYTES_PER_STAGE == 49152
    fits = max_stages(bytes_per_stage=BYTES_PER_STAGE, arch=_SM90)
    assert fits == 4, f"the shared-memory arithmetic says {fits} stages, not 4"
    assert fits == STAGES, "the .cu's STAGES is not the number that fits"
    # ...and with the barriers and the alignment slack charged too, still 4 — the slack is cheap.
    charged = max_stages(
        bytes_per_stage=BYTES_PER_STAGE, arch=_SM90, extra_bytes=BARRIER_BYTES + SMEM_ALIGN
    )
    assert charged == STAGES
    one_more = (STAGES + 1) * BYTES_PER_STAGE
    assert one_more > _SM90.smem_per_cta, "a fifth stage would fit — STAGES is not the ceiling"


def test_the_launch_must_opt_in_to_its_shared_memory() -> None:
    """The single most expensive thing this rung can forget, asserted on a laptop.

    Above 48 KB a kernel's dynamic shared memory is granted only if the launch asks for it with
    ``cudaFuncSetAttribute``. ptxas compiles the kernel either way and the failure appears as a
    launch error on the box. H-R1 (32 KB) never met this; H-R3 does. So the test is not
    ``check_smem_budget(...) == []`` — the budget is *deliberately* over the threshold — it is that
    the finding is exactly the opt-in one, and that the ``.cu`` actually makes the call.
    """
    errs = check_smem_budget(
        bytes_per_stage=BYTES_PER_STAGE,
        stages=STAGES,
        arch=_SM90,
        extra_bytes=BARRIER_BYTES + SMEM_ALIGN,
    )
    assert len(errs) == 1 and errs[0].startswith("OPT-IN REQUIRED"), errs
    requested = SMEM_BYTES + BARRIER_BYTES
    assert requested <= _SM90.smem_per_cta, "the CTA asks for more shared memory than Hopper has"
    assert requested > _SM90.smem_opt_in_threshold, "no opt-in needed; this test has nothing to say"
    src = _src()
    assert "cudaFuncAttributeMaxDynamicSharedMemorySize" in src, (
        "the launcher never opts in to > 48 KB of shared memory; the kernel will fail to launch"
    )
    assert re.search(r"h_r3_ws_bf16_sm90<<<grid, block, SMEM_BYTES>>>", src), (
        "the launch must pass SMEM_BYTES as its dynamic shared-memory argument"
    )


def test_one_cta_per_sm_follows_from_the_tile() -> None:
    """The plan row says 1 CTA/SM. That is a consequence, not a setting: two of these tiles do not
    fit in one SM's shared memory, so the occupancy is decided before any tuning happens."""
    two_ctas = 2 * STAGES * BYTES_PER_STAGE
    assert two_ctas > _SM90.smem_per_sm


def test_setmaxnreg_immediates_are_legal_and_the_split_is_resident() -> None:
    """Warp specialisation lives or dies on these two numbers.

    Legality first (PTX: 24..256, multiple of 8), then the thing legality does not cover — that the
    CTA the split produces can actually be resident. 24x128 + 240x256 = 64512 of an SM's 65536.
    That it lands exactly on ``REGS_PER_THREAD_UNIFORM * THREADS_PER_CTA`` is the point:
    ``setmaxnreg`` does not create registers, it decides who holds them.
    """
    assert check_setmaxnreg(PRODUCER_REGS) == []
    assert check_setmaxnreg(CONSUMER_REGS) == []
    assert _SM90.regs_per_sm >= REGS_AFTER_SPLIT, "the split asks for more registers than an SM has"
    assert REGS_PER_THREAD_UNIFORM == 168
    assert REGS_AFTER_SPLIT == REGS_PER_THREAD_UNIFORM * THREADS_PER_CTA
    assert (
        check_register_budget(
            regs_per_thread=REGS_PER_THREAD_UNIFORM,
            threads_per_cta=THREADS_PER_CTA,
            arch=_SM90,
        )
        == []
    )
    # The consumer's accumulator alone is 128 fp32 registers (WGMMA_N/16 fragments x 8), which is
    # why the producer has to give up as much as it does.
    assert TILE_N // 2 == 128
    assert CONSUMER_REGS > TILE_N // 2


def test_both_tensor_maps_are_legal_for_every_measured_shape() -> None:
    """``cuTensorMapEncodeTiled`` returns one error code for every violated constraint, on the host,
    at startup — so a violation reaches you as a kernel that never launches. Name it here instead."""
    for name, (m, n, k) in SHAPES.items():
        a_map = tma_tensor_map(rows=m, cols=k, box_rows=TILE_M, box_cols=TILE_K)
        b_map = tma_tensor_map(rows=k, cols=n, box_rows=TILE_K, box_cols=TILE_N)
        assert check_tma_tensor_map(**a_map) == [], f"{name}: A map illegal"  # type: ignore[arg-type]
        assert check_tma_tensor_map(**b_map) == [], f"{name}: B map illegal"  # type: ignore[arg-type]
    # The box shapes themselves, spelled out once so a tile change has to face them.
    a_map = tma_tensor_map(rows=4096, cols=4096, box_rows=TILE_M, box_cols=TILE_K)
    b_map = tma_tensor_map(rows=4096, cols=4096, box_rows=TILE_K, box_cols=TILE_N)
    assert a_map["box_dims"] == [64, 128, 1, 1, 1]
    assert b_map["box_dims"] == [64, 64, 4, 1, 1]
    assert a_map["swizzle"] is SwizzleMode.B128


def test_a_tile_swizzle_needs_the_inner_box_to_be_exactly_one_atom() -> None:
    """The rule the whole N%64 constraint comes from, asserted as the failure it produces."""
    bad = tma_tensor_map(rows=4096, cols=4096, box_rows=TILE_M, box_cols=TILE_K)
    bad["box_dims"] = [32, 128, 1, 1, 1]  # half an atom
    errs = check_tma_tensor_map(**bad)  # type: ignore[arg-type]
    assert errs and "B128 swizzle requires exactly 128 B" in errs[0]


def test_the_two_descriptor_layouts_are_the_ones_cutlass_implies() -> None:
    """A is K-major, B is MN-major, and the ``.cu`` must hard-code exactly those offsets.

    This is the rung's one real departure from fast.cu M7 (whose B is stored [N,K], so both its
    operands are K-major and both trans flags are 0). Deriving the MN-major pair here, from the
    layout TMA produces, is what makes ``#define B_LBO (128 * BK)`` a claim rather than a constant.
    """
    src = _src()
    a_lbo, a_sbo = k_major_descriptor_offsets()
    b_lbo, b_sbo = mn_major_descriptor_offsets()
    assert (a_lbo, a_sbo) == (16, 1024)  # identical to matmul_7.cuh:13-14
    assert (b_lbo, b_sbo) == (8192, 1024)
    assert _define(src, "A_LBO") == a_lbo
    assert _define(src, "A_SBO") == a_sbo
    assert _define(src, "B_SBO") == b_sbo
    m = re.search(r"^#define B_LBO \((\d+) \* BK\)$", src, re.MULTILINE)
    assert m, "#define B_LBO not found, or no longer written as a multiple of BK"
    assert int(m.group(1)) * TILE_K == b_lbo
    # The stride offset is the same for both layouts and the leading offset is not: that asymmetry
    # is the whole content of "MN-major", and swapping the two is a kernel that runs and is wrong.
    assert a_sbo == b_sbo and a_lbo != b_lbo


def test_descriptors_round_trip_through_the_documented_bit_layout() -> None:
    """The ``.cu``'s packing and hopper_contracts' must be one contract, checked both ways."""
    for lbo, sbo in (k_major_descriptor_offsets(), mn_major_descriptor_offsets()):
        desc = wgmma_smem_descriptor(
            0x4000, leading_byte_offset=lbo, stride_byte_offset=sbo, swizzle=SwizzleMode.B128
        )
        f = descriptor_fields(desc)
        assert f["leading_byte_offset"] == lbo >> 4
        assert f["stride_byte_offset"] == sbo >> 4
        assert f["layout_type"] == int(SwizzleMode.B128)
        assert f["base_offset"] == 0, "base_offset is 0 only because the tile base is 1024-aligned"


def test_stage_offsets_land_on_a_swizzle_repeat() -> None:
    """Why the kernel pays 1 KB to round the dynamic shared-memory base up.

    Every descriptor in this kernel carries ``base_offset = 0``, as fast.cu and CUTLASS both do.
    That is correct only if each tile's origin sits on a swizzle repeat — 8 rows of 128 B = 1024 B.
    The driver aligns dynamic shared memory to 128 B, not 1024, hence SMEM_ALIGN.
    """
    a_bytes = TILE_M * TILE_K * 2
    b_bytes = TILE_K * TILE_N * 2
    for s in range(STAGES):
        assert (s * a_bytes) % SMEM_ALIGN == 0
        assert (STAGES * a_bytes + s * b_bytes) % SMEM_ALIGN == 0
    # And every advance a descriptor makes inside a stage keeps the phase.
    a_strip = 16 * 2  # A: one k-strip = 32 B, still inside its 128 B row
    assert a_strip < 128
    b_strip = 16 * TMA_INNER_ELEMS * 2  # B: 16 k-rows of 128 B = 2048 B, inside one n_outer chunk
    assert b_strip % SMEM_ALIGN == 0
    # ...and the stride the descriptor's LBO carries between n_outer chunks keeps it too. This is
    # the pair the hole comment warns about: 2048 B walks k, 8192 B walks n_outer, and using the
    # second as the start-address advance lands on chunk 1's k=0.
    b_lbo, _ = mn_major_descriptor_offsets()
    assert b_lbo % SMEM_ALIGN == 0
    assert b_lbo == (TILE_N // TMA_INNER_ELEMS - 1 + 1) * 2048 == 8192


def test_expect_tx_bytes_cover_both_copies_of_a_stage() -> None:
    """One mbarrier, two TMA copies, one byte count — so the count is the sum or nothing works."""
    assert BYTES_PER_STAGE == TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2
    assert "expect_bytes(&full[qidx], BYTES_PER_STAGE)" in _src()


def test_every_spec_shape_is_legal_for_this_rung() -> None:
    """N and K must be whole numbers of swizzle atoms; M must not be, at least once."""
    for name, (m, n, k) in SHAPES.items():
        assert shape_violations(m, n, k) == [], name
        assert tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["k"], name
    assert any(
        not tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["m"]
        for m, n, k in SHAPES.values()
    ), "no shape exercises M predication — the epilogue's bounds checks would be untested"
    assert any(
        not tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["n"]
        for m, n, k in SHAPES.values()
    ), "no shape exercises N predication — a 256-wide tile makes that the more likely bug"


def test_the_ladder_registry_npot_shape_is_rejected_with_its_reason() -> None:
    """(257, 1023, 512) is measurable at H-R1 and not here, and the wrapper must say why.

    Not a defect in either: H-R1 stages by hand and can predicate anything, H-R3 stages through TMA
    and a tensor map cannot express 1023 as whole 64-element atoms. Asserted so that the divergence
    from bench/kernels/gemm/k1_ladder.py's registry is a recorded decision rather than a surprise
    on the box.
    """
    errs = shape_violations(257, 1023, 512)
    assert errs and "swizzle atoms" in errs[0] and "N=1023" in errs[0]


def test_skinny_shape_is_grid_limited_and_the_suite_knows_it() -> None:
    """M=16 cannot fill an H100. Any %-of-cuBLAS there measures occupancy, not the mainloop.

    Asserted rather than commented so that a future tile-shape change cannot quietly turn the
    skinny row into a number that looks like a mainloop regression.
    """
    m, n, _ = SHAPES["skinny16"]
    wq = wave_quantization(m=m, n=n, tile_m=TILE_M, tile_n=TILE_N, sm_count=132)
    assert wq.full_waves == 0 and wq.tail_utilization < 0.25


def test_headline_shape_leaves_a_partial_wave_and_the_spec_owns_it() -> None:
    """4096^3 at a 128x256 tile is 512 CTAs on 132 SMs: 3.88 waves, so the last one is 88% full.

    That ceiling is not recoverable inside the mainloop; H-R4's persistent scheduler is what
    addresses it. Recorded here so the H-R3 number is read against the right ceiling.
    """
    wq = wave_quantization(m=4096, n=4096, tile_m=TILE_M, tile_n=TILE_N, sm_count=132)
    assert wq.ctas == 512
    assert 0.90 < wq.efficiency < 1.0


def test_reference_is_fp32_over_the_same_bf16_operands() -> None:
    """The oracle must not be a different problem — it up-casts, it does not re-generate.

    If the reference multiplied fp32 operands, the operands' own bf16 quantization error would be
    inside the tolerance, and a real kernel bug could hide under it.
    """
    torch.manual_seed(0)
    a = torch.randn(64, 64, dtype=torch.bfloat16)
    b = torch.randn(64, 128, dtype=torch.bfloat16)
    ref = reference_gemm(a, b)
    assert ref.dtype is torch.float32
    assert torch.equal(ref, torch.matmul(a.float(), b.float()))


def test_wrapper_rejects_bad_shapes_and_dtypes_without_a_device() -> None:
    """The shape contract runs before the arch check, so all of it is reachable from a CPU test."""

    def z(r: int, c: int, dt: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(r, c, dtype=dt)

    with pytest.raises(TypeError, match="bfloat16"):
        h_r3_gemm(z(128, 64, torch.float16), z(64, 256))
    with pytest.raises(ValueError, match="inner dimensions"):
        h_r3_gemm(z(128, 64), z(128, 256))
    with pytest.raises(ValueError, match="swizzle atoms"):
        h_r3_gemm(z(128, 32), z(32, 256))
    with pytest.raises(ValueError, match="N=100"):
        h_r3_gemm(z(128, 64), z(64, 100))


def test_wrapper_reports_an_open_hole_as_such() -> None:
    """ "You have not written the kernel" and "your kernel is broken" are different messages."""
    if not hole_is_open(SOURCE):
        pytest.skip("hole filled — the HoleOpenError path no longer applies")
    with pytest.raises((HoleOpenError, RuntimeError)) as exc:
        h_r3_gemm(
            torch.zeros(128, 64, dtype=torch.bfloat16),
            torch.zeros(64, 256, dtype=torch.bfloat16),
        )
    # On a CPU box require_arch fires first (RuntimeError); on a Hopper box the loader does.
    assert "h_r3_gemm" in str(exc.value) or "HUY hole" in str(exc.value)


# =============================================================================================
# drydock — what nvcc generated. No GPU, but the CUDA container must have run.
# =============================================================================================


def _sass() -> str:
    p = _DRYDOCK / f"{_STEM}.sm_90a.sass.txt"
    if not p.is_file():
        pytest.skip(f"no SASS at {p} — run: infra/drydock.sh compile {SOURCE}")
    if hole_is_open(SOURCE):
        pytest.skip(
            "hole open — the captured SASS is the stub's, and says nothing about the kernel"
        )
    return p.read_text(encoding="utf-8", errors="replace")


@pytest.mark.drydock
def test_sass_issues_the_warpgroup_mma_against_an_mn_major_b() -> None:
    """``HGMMA ... .tnspB`` must appear: the tensor core is used, and B is transposed at the MMA.

    Two failure modes in one assertion. No ``HGMMA`` at all is the signature of dropping the ``a``
    from ``-arch=sm_90a`` — everything compiles, the kernel runs, and it is silently a scalar loop.
    ``HGMMA`` without ``tnspB`` is the subtler one: the mainloop copied fast.cu's ``0, 0`` trans
    immediates, which are right for its [N,K] B and wrong for the [K,N] B this harness passes.
    """
    sass = _sass()
    assert re.search(r"\bHGMMA\b", sass), (
        "no HGMMA in the SASS — either the mainloop does not issue wgmma, or it was compiled for "
        "base sm_90 instead of sm_90a (the trailing 'a' selects the accelerated ISA)"
    )
    assert re.search(r"HGMMA[^\n]*\.tnspB", sass), (
        "HGMMA is issued without .tnspB — the B operand is being read K-major, but TMA lays a "
        "[K,N] row-major B down MN-major. The trans immediates must be 0, 1 (see the .cu header)."
    )
    assert not re.search(r"HGMMA[^\n]*\.tnspA", sass), (
        "A is K-major; .tnspA means trans_a was set and the A operand will be read transposed"
    )


@pytest.mark.drydock
def test_sass_uses_the_tma_copy_engine() -> None:
    """``UTMALDG.5D`` — the staged loads go through the copy engine, not through LDG/STS.

    A mainloop that hand-copies would still be correct and would still be measured; it would just
    not be this rung. The 5-D form is the one the 128 B swizzle forces (map.md, "Descriptors").
    """
    sass = _sass()
    assert re.search(r"\bUTMALDG\.5D\b", sass), (
        "no 5-D TMA load in the SASS — the producer is not using cp.async.bulk.tensor"
    )


@pytest.mark.drydock
def test_sass_has_no_local_memory_traffic() -> None:
    """``LDL``/``STL`` mean registers spilled to local memory — the accumulator did not fit."""
    sass = _sass()
    spills = re.findall(r"\b(LDL|STL)\b", sass)
    assert not spills, f"{len(spills)} local-memory accesses in the SASS: the accumulator spilled"


@pytest.mark.drydock
def test_ptxas_report_is_clean() -> None:
    """0 spill bytes, and shared memory within the arch limit — read from the ptxas report.

    The ``bytes smem`` line counts STATIC shared memory only, which here is just the 64 B of
    mbarriers; the 196608 B of staging is dynamic and never appears in this report. That is exactly
    why ``test_the_launch_must_opt_in_to_its_shared_memory`` exists on the CPU side.
    """
    p = _DRYDOCK / f"{_STEM}.sm_90a.ptxas.txt"
    if not p.is_file():
        pytest.skip(f"no ptxas report at {p} — run: infra/drydock.sh compile {SOURCE}")
    report = p.read_text(encoding="utf-8", errors="replace")
    spill = re.search(r"(\d+) bytes spill stores", report)
    assert spill and int(spill.group(1)) == 0, f"register spill in {p.name}: {report[-400:]}"
    smem = re.search(r"(\d+) bytes smem", report)
    if smem:
        assert int(smem.group(1)) <= _SM90.smem_per_cta
    regs = re.search(r"Used (\d+) registers", report)
    if regs:
        assert int(regs.group(1)) <= REGS_PER_THREAD_UNIFORM, (
            f"ptxas gave every thread {regs.group(1)} registers, above the "
            f"{REGS_PER_THREAD_UNIFORM} that 384 threads on one SM can have"
        )


# =============================================================================================
# gpu — correctness against the oracle. H100 only.
# =============================================================================================


def _require_tolerance() -> float:
    if TOL_CONST is None:
        pytest.fail(
            "TOL_CONST is unset. The correctness gate cannot run without a tolerance, and picking "
            "one is the rung's numerics lesson: set it in this file and write the one-line argument "
            "into experiments/K1/H-R3/spec.md's 'Correctness gate' line before the first measured run."
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
        f"At this rung the two structured failures to look for first are a stale ring stage (a "
        f"whole 64-column band of C wrong, from a mis-ordered barrier) and an MN-major descriptor "
        f"read as K-major (wrong everywhere, plausible magnitude) — print "
        f"(out - ref).abs().sum(dim=0) before widening this."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_matches_oracle(shape_name: str) -> None:
    """Every spec shape against the fp32 oracle at the spec's tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m, n, k = SHAPES[shape_name]
    if shape_name in ("sq4096", "rect8192") and os.environ.get("LADDERS_SMOKE") == "1":
        pytest.skip("smoke mode: the large shapes are the measurement, not the smoke test")
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches_oracle(h_r3_gemm(a, b), a, b)
    del a, b
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3], ids=["tiny", "unit", "large"])
def test_matches_oracle_across_magnitudes(scale: float) -> None:
    """bf16 has 8 mantissa bits and a huge exponent range; the tolerance must be scale-relative."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(1)
    a = (torch.randn(256, 128, device="cuda", dtype=torch.float32) * scale).bfloat16()
    b = (torch.randn(128, 256, device="cuda", dtype=torch.float32) * scale).bfloat16()
    _assert_matches_oracle(h_r3_gemm(a, b), a, b)


@pytest.mark.gpu
def test_k_shorter_than_the_ring_still_drains() -> None:
    """K = one stage's worth: the pipeline never wraps, so the drain path runs with nothing behind it.

    A four-deep ring whose consumer waits on ``full[1]`` before the producer has issued it, or whose
    prologue releases fewer than STAGES empty barriers, hangs or reads garbage exactly here and
    nowhere else. Cheap, and it is the shape a pipeline bug hides from.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(2)
    for k in (TILE_K, 2 * TILE_K, STAGES * TILE_K, (STAGES + 1) * TILE_K):
        a = torch.randn(TILE_M, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, TILE_N, device="cuda", dtype=torch.bfloat16)
        _assert_matches_oracle(h_r3_gemm(a, b), a, b)


@pytest.mark.gpu
def test_rejects_bad_inputs() -> None:
    """The guard clauses, on the device where the CUDA-tensor ones can actually be reached."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    def z(r: int, c: int, dt: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(r, c, device="cuda", dtype=dt)

    with pytest.raises(TypeError, match="bfloat16"):
        h_r3_gemm(z(128, 64, torch.float16), z(64, 256))
    with pytest.raises(ValueError, match="inner dimensions"):
        h_r3_gemm(z(128, 64), z(128, 256))
    with pytest.raises(ValueError, match="swizzle atoms"):
        h_r3_gemm(z(128, 64), z(64, 100))


@pytest.mark.gpu
def test_handles_nan_and_inf_without_masking_them() -> None:
    """A NaN in must produce a NaN out — a kernel that swallows them hides real training bugs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    a = torch.ones(128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(64, 256, device="cuda", dtype=torch.bfloat16)
    a[0, 0] = float("nan")
    b[0, 1] = float("inf")
    out = h_r3_gemm(a, b)
    assert torch.isnan(out[0, 0]), "NaN was swallowed"
    assert not torch.isfinite(out[:, 1]).all(), "Inf was swallowed"
