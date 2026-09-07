"""K1/B-R6 — NVFP4 block-scaled GEMM. Three tiers of gate, in ascending cost.

  CPU      the scale-factor layout (exhaustively, it is finite), the two number formats, the
           quantization oracle's two provable properties, and the wrapper's contracts. Milliseconds.
  drydock  what nvcc actually generated: the block-scaled MMA assembles for sm_120a, nothing spills.
           Needs the CUDA container (infra/drydock.sh), still no GPU.
  gpu      correctness against the quantize-dequantize oracle at the five spec shapes. Needs sm120.

This rung's CPU tier carries more weight than any other K1 rung's, and deliberately. A block-scaled
GEMM's characteristic bug is not a race and not a slow mainloop: it is a scale factor read from the
wrong byte. That produces no fault, no NaN and no slowdown — just a wrong C — and it is entirely
decidable from integer arithmetic, so it is decided here rather than on rented silicon.

Spec: experiments/K1/B-R6/spec.md   ·   Map: experiments/K1/B-R6/map.md
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from scratch_llm.kernels.common.hopper_contracts import (
    ARCH,
    check_smem_budget,
    tile_covers,
    wave_quantization,
)
from scratch_llm.kernels.gemm.cuda._k1_loader import HoleOpenError, hole_is_open, source_for
from scratch_llm.kernels.gemm.cuda.b_r6 import (
    RUNG,
    SOURCE,
    THREADS_PER_CTA,
    TILE_K,
    TILE_M,
    TILE_N,
    NVFP4Operand,
    _sf_offset_table,
    b_r6_gemm,
    b_r6_gemm_packed,
    dequantize_nvfp4,
    quantization_error,
    quantize_nvfp4,
    reference_gemm,
)
from scratch_llm.kernels.gemm.nvfp4_layout import (
    ATOM_BYTES,
    ATOM_K,
    ATOM_ROWS,
    BLK_MN,
    BLK_SF,
    E2M1_MAGNITUDES,
    E2M1_MAX,
    E2M1_UNIT_ROUNDOFF,
    FP4_E2M1,
    SF_UE4M3,
    SF_UE8M0,
    SF_VEC_SIZE,
    UE4M3_MAX,
    UE4M3_MIN_NORMAL,
    UE4M3_MIN_POSITIVE,
    atom_is_bijection,
    atom_offset,
    atom_offset_inverse,
    e2m1_round_code,
    scale_for_amax,
    sf_atom_grid,
    sf_byte_offset,
    sf_byte_offset_inverse,
    sf_tensor_bytes,
    ue4m3_ceil_code,
)


def _workspace() -> Path:
    """The ladders workspace root — the directory holding ``infra/`` and ``experiments/``.

    Walking up from this file does not find it in the layout that actually ships: ``ladders/
    scratch_llm`` is a symlink to a sibling directory, and ``Path.resolve()`` (like ``os.getcwd()``)
    hands back the physical path, which is outside the workspace. So the workspace is identified by
    the relationship instead — it is the sibling directory whose own ``scratch_llm`` symlink points
    back at this repo. The upward walk is tried first anyway, for a checkout where ``scratch_llm``
    is a real subdirectory.
    """
    repo = Path(__file__).resolve().parents[3]  # tests/kernels/gemm -> scratch_llm
    for cand in repo.parents:
        if (cand / "infra" / "drydock.sh").is_file() and (cand / "experiments").is_dir():
            return cand
    try:
        for sib in sorted(repo.parent.iterdir()):
            if (sib / "infra" / "drydock.sh").is_file() and (sib / "scratch_llm").resolve() == repo:
                return sib
    except OSError:
        pass
    return repo.parent


_WORKSPACE = _workspace()
_DRYDOCK = _WORKSPACE / "experiments" / "K1" / "B-R6" / "drydock"
_STEM = SOURCE.removesuffix(".cu")

# ---------------------------------------------------------------------------------------------
# The tolerance. HUY SETS THIS, with its argument, before the first measured run.
#
# The gate compares the kernel against reference_gemm — the SAME NVFP4 operands, dequantized and
# multiplied in fp32. So the format's own error is on BOTH sides and cancels; what is left is
# fp32 accumulation order over K plus anything wrong in the fragment mapping. The tolerance has to
# admit the first and reject the second, and the two are separated by measuring them: run
# quantization_error() to see what the format costs (it is large — order 10% relative on gaussian
# data) and convince yourself the constant chosen here is far below it. A tolerance that is
# comfortably bigger than the format residual is a tolerance no wrong kernel can fail.
#
# Leaving it None is deliberate. Deriving it IS this rung's numerics lesson — the plan's row says
# "tolerance derived from quantization error" — and nobody can derive it for him and have him learn
# anything. The gate below fails loudly rather than guessing.
# ---------------------------------------------------------------------------------------------
TOL_CONST: float | None = None

#: The five shapes the spec measures — identical to k1_ladder.py's SHAPES. Every K here is a
#: multiple of TILE_K = 64, which for this rung is also one scale atom's K extent.
SHAPES: dict[str, tuple[int, int, int]] = {
    "sq4096": (4096, 4096, 4096),  # the headline: the number quoted against NVFP4 SoL
    "rect8192": (8192, 8192, 4096),  # secondary: more waves, same tile — isolates grid effects
    "skinny16": (16, 4096, 4096),  # M < TILE_M: also M < one scale atom's 128 rows
    "npot": (257, 1023, 512),  # non-power-of-two: M and N predication, and SF padding
    "untuned": (1536, 6144, 2560),  # divides nothing the tile likes; not a tuning target
}

#: SM count for the wave arithmetic. sm_120 is a CAPABILITY, never a card: RTX PRO 4000 Blackwell
#: has ~70 SMs where a 5090-class part has far more (kernels/common/arch.py:is_sm120), so this is
#: the small end and any conclusion drawn from it must be one that gets worse, not better, on it.
SM_COUNT_SM120 = 70


# =============================================================================================
# The hole guard — exactly one per hole (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K1/B-R6", "csrc/gemm/b_r6_nvfp4_sm120.cu")
def test_b_r6_kernel_body_is_filled_and_compiles_clean() -> None:
    """Fails while the mainloop is a ``#error``; passes when it compiles clean for sm_120a.

    The CPU-side half of the gate, and useful on its own: a kernel that does not compile, spills
    registers, or overflows sm_120a's 99 KB of shared memory is not worth renting a Blackwell to
    discover. Correctness still needs the GPU — that is ``test_matches_oracle`` below.
    """
    assert not hole_is_open(SOURCE), (
        f"{RUNG} kernel body is still an open hole. Read experiments/K1/B-R6/spec.md and "
        f"experiments/K1/B-R6/map.md, then write the mainloop and epilogue in csrc/gemm/{SOURCE}."
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
# CPU — the scale-factor layout. Finite, so tested exhaustively rather than sampled.
# =============================================================================================


def test_scale_atom_is_a_bijection_onto_512_bytes() -> None:
    """128 rows x 4 K-blocks -> [0, 512), one byte each, no aliasing.

    If two (row, block) pairs shared a byte the kernel would scale one block with another's factor:
    a wrong C that does not fault, does not produce a NaN, and costs nothing in time. There are
    only 512 cases, so there is no reason to sample.
    """
    assert atom_is_bijection()
    assert all(
        atom_offset_inverse(atom_offset(r, b)) == (r, b)
        for r in range(ATOM_ROWS)
        for b in range(BLK_SF)
    )


def test_atom_offset_reproduces_the_cutlass_stride_formula() -> None:
    """``16*r0 + 4*r1 + k1``, read straight off ``sm100_blockscaled_layout.hpp:54-55``.

    The map derived this from the CuTe layout's strides ``Stride<Stride<_16,_4>, Stride<_0,_1>>``;
    this asserts the module implements that and not something that merely happens to be injective.
    """
    for r in range(ATOM_ROWS):
        for b in range(BLK_SF):
            assert atom_offset(r, b) == 16 * (r % 32) + 4 * (r // 32) + b
    # The four anchors worth knowing by heart: +16 per row within a 32-row group, +4 across groups,
    # +1 per K-block, and 511 at the far corner.
    assert (atom_offset(0, 0), atom_offset(1, 0), atom_offset(32, 0), atom_offset(0, 1)) == (
        0,
        16,
        4,
        1,
    )
    assert atom_offset(127, 3) == ATOM_BYTES - 1


def test_sixteen_k_elements_share_one_scale_byte() -> None:
    """The stride-0 mode, as behaviour: every k in a block maps to the same byte, and k+16 does not.

    This is the format itself. A kernel that advanced the scale pointer per k-element instead of per
    block would read 16x too far and still produce plausible numbers for the first block.
    """
    rows, k = 128, 64
    for base in range(0, k, SF_VEC_SIZE):
        offs = {sf_byte_offset(3, base + j, rows=rows, k_total=k) for j in range(SF_VEC_SIZE)}
        assert len(offs) == 1
    assert sf_byte_offset(3, 0, rows=rows, k_total=k) != sf_byte_offset(
        3, SF_VEC_SIZE, rows=rows, k_total=k
    )


def test_full_tensor_offsets_are_a_bijection_and_invertible() -> None:
    """Over several shapes: every (row, block) gets its own byte, and the tensor is exactly full.

    Includes a shape whose rows do not fill a whole atom, because that is where an allocation sized
    ``rows*k/16`` instead of by whole atoms stops being large enough.
    """
    for rows, k in ((128, 64), (256, 128), (129, 128), (16, 256)):
        total = sf_tensor_bytes(rows, k)
        pairs = [(r, j * SF_VEC_SIZE) for r in range(rows) for j in range(k // SF_VEC_SIZE)]
        offs = [sf_byte_offset(r, kk, rows=rows, k_total=k) for r, kk in pairs]
        assert len(set(offs)) == len(offs), f"{rows}x{k}: two blocks share a byte"
        assert max(offs) < total
        assert all(
            sf_byte_offset_inverse(o, rows=rows, k_total=k) == p
            for o, p in zip(offs, pairs, strict=True)
        )
    # A full atom grid is exactly saturated — no gaps to hide an off-by-one in.
    assert set(
        sf_byte_offset(r, j * SF_VEC_SIZE, rows=256, k_total=128)
        for r in range(256)
        for j in range(8)
    ) == set(range(sf_tensor_bytes(256, 128)))


def test_scale_tensor_is_sized_by_whole_atoms_not_by_elements() -> None:
    """``512 * ceil(rows/128) * ceil(k/64)``, and the padding for a partial operand is real."""
    assert sf_tensor_bytes(128, 64) == ATOM_BYTES
    assert sf_tensor_bytes(256, 128) == ATOM_BYTES * 2 * 2
    # 257 rows is three atom-rows, not 2.008 of them; 4096/64 = 64 atoms along K.
    assert sf_atom_grid(257, 4096) == (3, 64)
    assert sf_tensor_bytes(257, 4096) == ATOM_BYTES * 3 * 64
    assert sf_tensor_bytes(257, 4096) > 257 * 4096 // SF_VEC_SIZE  # padding, and it is not optional


def test_atom_grid_is_k_major() -> None:
    """``Step<_2,_1>`` means K-atoms are contiguous: +1 atom along K, +k_atoms along M.

    The other order is a plausible-looking transpose that reads every scale from the wrong atom
    once the operand is wider than 64 in K, which every spec shape is.
    """
    rows, k = 256, 256
    _, k_atoms = sf_atom_grid(rows, k)
    assert k_atoms == 4
    base = sf_byte_offset(0, 0, rows=rows, k_total=k)
    assert sf_byte_offset(0, ATOM_K, rows=rows, k_total=k) - base == ATOM_BYTES
    assert sf_byte_offset(ATOM_ROWS, 0, rows=rows, k_total=k) - base == ATOM_BYTES * k_atoms


def test_partial_scale_blocks_are_rejected_rather_than_rounded() -> None:
    """A K that is not a multiple of 16 has no encoding, so it is an error, not a pad-and-hope."""
    with pytest.raises(ValueError, match="SF_VEC_SIZE"):
        sf_atom_grid(128, 24)
    with pytest.raises(ValueError, match="outside"):
        atom_offset(128, 0)
    with pytest.raises(ValueError, match="outside"):
        atom_offset(0, BLK_SF)


def test_vectorized_offset_table_matches_the_scalar_reference() -> None:
    """The torch index table the quantizer uses vs. the pure-Python formula, element by element.

    Two implementations of one layout is two chances to be wrong; this is what makes the fast one
    trustworthy, and it is exhaustive over a shape big enough to exercise both atom strides.
    """
    rows, k = 256, 192
    table = _sf_offset_table(rows, k, "cpu")
    assert tuple(table.shape) == (rows, k // SF_VEC_SIZE)
    for r in range(rows):
        for j in range(k // SF_VEC_SIZE):
            assert int(table[r, j]) == sf_byte_offset(r, j * SF_VEC_SIZE, rows=rows, k_total=k)


# =============================================================================================
# CPU — the .cu agrees with the Python. The device code is the third implementation.
# =============================================================================================


def _cu_source() -> str:
    return source_for(SOURCE).read_text(encoding="utf-8")


def _cu_return_expr(fn_signature: str) -> str:
    """The single ``return`` expression of a device function in the ``.cu``, as text."""
    src = _cu_source()
    m = re.search(re.escape(fn_signature) + r"\s*\{\s*return ([^;]+);", src)
    assert m, (
        f"could not find `{fn_signature}` with a single return expression in csrc/gemm/{SOURCE}"
    )
    return m.group(1)


def test_cu_scale_addressing_is_the_same_arithmetic_as_the_python() -> None:
    """Evaluate the ``.cu``'s own offset expressions in Python and compare against the module.

    ``&``, ``>>``, ``*`` and ``+`` mean the same thing in both languages for non-negative ints, and
    the expressions are fully parenthesised, so the C text can simply be evaluated. That makes this
    a real contract rather than a string match: an edit to either side that changes the arithmetic
    fails here, on a laptop, instead of on a rented Blackwell as a wrong C.
    """
    atom_expr = _cu_return_expr("sf_atom_offset(int row_in_atom, int sf_block)")
    byte_expr = _cu_return_expr("sf_byte_offset(int row, int k, int k_atoms)")

    def cu_atom(row_in_atom: int, sf_block: int) -> int:
        return eval(
            atom_expr, {"__builtins__": {}}, {"row_in_atom": row_in_atom, "sf_block": sf_block}
        )

    for r in range(ATOM_ROWS):
        for b in range(BLK_SF):
            assert cu_atom(r, b) == atom_offset(r, b), (
                f"the .cu's sf_atom_offset differs at ({r},{b})"
            )

    rows, k = 384, 192
    _, k_atoms = sf_atom_grid(rows, k)
    env = {"SF_ATOM_BYTES": ATOM_BYTES, "sf_atom_offset": cu_atom}
    for r in range(0, rows, 7):
        for j in range(k // SF_VEC_SIZE):
            got = eval(  # noqa: S307 - the point of the test is to run the .cu's own expression
                byte_expr,
                {"__builtins__": {}},
                {**env, "row": r, "k": j * SF_VEC_SIZE, "k_atoms": k_atoms},
            )
            assert got == sf_byte_offset(r, j * SF_VEC_SIZE, rows=rows, k_total=k)


def test_source_and_wrapper_agree_on_the_tile_shape() -> None:
    """The ``.cu``'s defines and the wrapper's TILE_*/THREADS must not drift.

    They are used for different things — one generates code, the other feeds the shared-memory and
    wave arithmetic — so nothing else would notice if they diverged.
    """
    src = _cu_source()

    def macro(name: str) -> int:
        m = re.search(rf"^#define {name} (\d+)", src, re.MULTILINE)
        assert m, f"#define {name} not found in csrc/gemm/{SOURCE}"
        return int(m.group(1))

    for name, expected in (("BM", TILE_M), ("BN", TILE_N), ("BK", TILE_K)):
        assert macro(name) == expected, f"{name}={macro(name)} in the .cu but {expected} in b_r6.py"
    assert macro("WARPS_M") * macro("WARPS_N") * 32 == THREADS_PER_CTA


def test_source_and_layout_module_agree_on_the_scale_constants() -> None:
    """``SF_VEC_SIZE``/``SF_BLK_MN``/``SF_BLK_SF`` are one layout stated twice; they must match.

    And TILE_K must equal one atom's K extent: that equality is what makes the kernel's scale stage
    a single contiguous 512-byte copy instead of a per-element gather.
    """
    src = _cu_source()
    for name, expected in (
        ("SF_VEC_SIZE", SF_VEC_SIZE),
        ("SF_BLK_MN", BLK_MN),
        ("SF_BLK_SF", BLK_SF),
        ("MMA_K", 64),
    ):
        m = re.search(rf"^#define {name} (\d+)", src, re.MULTILINE)
        assert m and int(m.group(1)) == expected, f"{name} disagrees with nvfp4_layout"
    assert TILE_K == ATOM_K == BLK_SF * SF_VEC_SIZE


def test_tile_shape_fits_shared_memory_on_sm120() -> None:
    """One stage of A+B (packed fp4) plus the two 512-byte scale atoms, against sm_120a's cap.

    sm_120a has 99 KB per CTA, not Hopper's 227 — routing this rung to "the biggest card" would
    change the answer, which is why the arch is named.
    """
    operand_bytes = (TILE_M * TILE_K + TILE_N * TILE_K) // 2  # two e2m1 per byte
    scale_bytes = 2 * ATOM_BYTES
    assert (
        check_smem_budget(
            bytes_per_stage=operand_bytes, stages=1, arch=ARCH["sm_120a"], extra_bytes=scale_bytes
        )
        == []
    )
    # Headroom is the point: a later multistage version has room for many stages, so the single
    # stage this rung ships is a choice about legibility, not a shared-memory limit.
    assert operand_bytes + scale_bytes < ARCH["sm_120a"].smem_per_cta // 8


def test_every_spec_shape_is_legal_for_this_rung() -> None:
    """K must divide TILE_K for all five shapes, and something must exercise M/N predication."""
    for name, (m, n, k) in SHAPES.items():
        assert k % TILE_K == 0, f"{name}: K={k} is not a multiple of TILE_K={TILE_K}"
        assert k % SF_VEC_SIZE == 0, f"{name}: K={k} does not divide into whole scale blocks"
        assert tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["k"], name
    assert any(
        not tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["m"]
        for m, n, k in SHAPES.values()
    ), "no shape exercises M predication — the epilogue's bounds checks would be untested"
    # ...and at least one must leave the scale tensor padded, or nothing tests the atom rounding-up.
    assert any(sf_atom_grid(m, k)[0] * ATOM_ROWS > m for m, _, k in SHAPES.values())


def test_skinny_shape_cannot_fill_the_machine_and_the_suite_knows_it() -> None:
    """M=16 is one eighth of a single scale atom's 128 rows. Any %-of-SoL there measures occupancy.

    Asserted rather than commented so a future tile change cannot quietly turn the skinny row into
    a number that reads like a mainloop regression.
    """
    m, n, _ = SHAPES["skinny16"]
    wq = wave_quantization(m=m, n=n, tile_m=TILE_M, tile_n=TILE_N, sm_count=SM_COUNT_SM120)
    assert wq.full_waves == 0 and wq.tail_utilization < 1.0
    assert m < ATOM_ROWS  # the whole M range lives inside one padded scale atom


# =============================================================================================
# CPU — the two number formats
# =============================================================================================


def test_format_anchors_match_the_cutlass_encodings() -> None:
    """Spot values nobody should have to re-derive, from ``exmy_base.h:917-939``."""
    assert E2M1_MAGNITUDES == (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    assert E2M1_MAX == 6.0
    assert FP4_E2M1.decode(0x7) == 6.0 and FP4_E2M1.decode(0xF) == -6.0
    assert FP4_E2M1.nan_code is None  # NanInfEncoding::NONE — no NaN can survive this format
    assert SF_UE4M3.decode(0x38) == 1.0  # exponent 7 == bias, mantissa 0
    assert SF_UE4M3.decode(0x7E) == UE4M3_MAX == 448.0
    assert SF_UE4M3.decode(0x01) == UE4M3_MIN_POSITIVE == 2.0**-9
    assert SF_UE4M3.nan_code == 0x7F  # NOT 0xFF: the sign bit is not part of an unsigned encoding
    assert UE4M3_MIN_NORMAL == 2.0**-6
    # MXFP4's scale for contrast: all exponent, so it has no zero and no mantissa to round with.
    assert SF_UE8M0.decode(127) == 1.0 and SF_UE8M0.decode(0) == 2.0**-127
    with pytest.raises(ValueError, match="NaN"):
        SF_UE4M3.decode(0x7F)


def test_e2m1_rounding_is_nearest_with_ties_to_even() -> None:
    """The seven midpoints, each named. Ties-to-even is what keeps the quantizer unbiased."""
    cases = {
        0.25: 0.0,  # tie between 0 and 0.5 -> 0 (mantissa 0)
        0.75: 1.0,  # tie between 0.5 and 1.0 -> 1.0
        1.25: 1.0,
        1.75: 2.0,
        2.5: 2.0,
        3.5: 4.0,
        5.0: 4.0,  # the widest gap in the format, and the one a real block lands in
        5.5: 6.0,
        7.0: 6.0,  # saturates: E2M1 has no Inf
    }
    for value, expected in cases.items():
        assert FP4_E2M1.decode(e2m1_round_code(value)) == expected, value
        assert FP4_E2M1.decode(e2m1_round_code(-value)) == -expected, -value
    with pytest.raises(ValueError, match="NanInfEncoding"):
        e2m1_round_code(float("inf"))


def test_scale_rounds_up_so_the_largest_element_never_saturates() -> None:
    """``ceil_ue4m3(amax/6) >= amax/6`` for every amax, hence ``|x/scale| <= 6``. The theorem.

    Rounding the scale to NEAREST instead would push ``amax/scale`` above 6 whenever it rounded
    down — unbounded error on exactly the element that set the block's range.
    """
    assert ue4m3_ceil_code(1.0) == 0x38
    for amax in (1e-4, 0.017, 0.5, 1.0, 6.0, 100.0, 2000.0):
        code, scale = scale_for_amax(amax)
        assert scale >= amax / E2M1_MAX, amax
        assert SF_UE4M3.decode(code) == scale
        assert amax / scale <= E2M1_MAX + 1e-12, amax
    assert scale_for_amax(0.0) == (0, 0.0)
    with pytest.raises(ValueError, match="unrepresentable|exceeds"):
        scale_for_amax(UE4M3_MAX * E2M1_MAX * 2)


def test_scale_is_tight_when_it_is_normal() -> None:
    """``amax/scale >= 16/3``, so the block's largest element rounds to 6.0 and no binade is wasted.

    Only claimed above UE4M3's smallest normal: below it the grid is absolute rather than relative
    and the ratio argument does not hold. That boundary is exactly where the idempotence property
    below stops holding too, which is why both are stated with the same precondition.
    """
    for amax in (0.1, 1.0, 3.7, 60.0, 2000.0):
        _, scale = scale_for_amax(amax)
        assert scale >= UE4M3_MIN_NORMAL
        assert amax / scale >= E2M1_MAX / (1 + 0.125) - 1e-12, amax
        assert FP4_E2M1.decode(e2m1_round_code(amax / scale)) == E2M1_MAX, amax


# =============================================================================================
# CPU — the quantization oracle. This is what makes the tolerance derivable.
# =============================================================================================


def _block_scales(x: torch.Tensor) -> torch.Tensor:
    """The per-block scale, computed by the scalar reference rather than by the quantizer."""
    rows, k = x.shape
    amax = x.double().view(rows, k // SF_VEC_SIZE, SF_VEC_SIZE).abs().amax(-1)
    return torch.tensor(
        [[scale_for_amax(float(a))[1] for a in row] for row in amax], dtype=torch.float64
    )


@pytest.mark.parametrize("magnitude", [1e-3, 1e-2, 1.0, 1e2, 200.0], ids=lambda m: f"x{m:g}")
def test_round_trip_error_is_bounded_by_the_formats_own_step(magnitude: float) -> None:
    """``|x - dequant(quant(x))| <= 0.25 * max(|x|, scale)``, elementwise, at every magnitude.

    The 0.25 is E2M1's unit roundoff: two significand bits, so within a binade the step is half the
    smaller neighbour and nearest-rounding is off by at most a quarter of it. The ``max(|x|, scale)``
    is what covers the sub-scale elements, where the grid's spacing is absolute rather than
    relative. Everything computed in float64 so the ``<=`` is exact and not one ulp of luck.

    This bound is the ONLY thing the rung's tolerance can honestly be derived from, and it is the
    reason the derivation is possible on a laptop before any silicon is rented.
    """
    torch.manual_seed(0)
    x = torch.randn(128, 64) * magnitude
    dq = dequantize_nvfp4(quantize_nvfp4(x))
    scale = _block_scales(x)
    xb = x.double().view(128, 4, SF_VEC_SIZE)
    bound = E2M1_UNIT_ROUNDOFF * torch.maximum(xb.abs(), scale[..., None])
    err = (xb - dq.double().view(128, 4, SF_VEC_SIZE)).abs()
    assert bool((err <= bound).all()), (
        f"round-trip error exceeds E2M1's own step at magnitude {magnitude}: worst ratio "
        f"{float((err / bound.clamp_min(1e-300)).max()):.6f}"
    )
    # No element saturated — the ceiling-rounded scale's other guarantee, checked directly.
    assert float((xb / scale.clamp_min(1e-300)[..., None]).abs().max()) <= E2M1_MAX


@pytest.mark.parametrize("magnitude", [1e-1, 1.0, 10.0, 1e2], ids=lambda m: f"x{m:g}")
def test_quantize_dequantize_is_idempotent(magnitude: float) -> None:
    """Re-quantizing a dequantized tensor reproduces the same bytes — codes AND scales.

    Not a tautology: it holds only because the scale rounds UP and is therefore tight enough that
    the block's largest element lands on 6.0, which makes the second pass choose the same scale.
    The precondition (every scale normal) is asserted rather than assumed, because below
    UE4M3_MIN_NORMAL the property genuinely fails and a test that silently drifted into that regime
    would be reporting on a different claim.
    """
    torch.manual_seed(1)
    x = torch.randn(128, 64) * magnitude
    scale = _block_scales(x)
    assert bool((scale[scale > 0] >= UE4M3_MIN_NORMAL).all()), (
        "precondition: this test only claims idempotence where the UE4M3 scale is normal"
    )
    first = quantize_nvfp4(x)
    second = quantize_nvfp4(dequantize_nvfp4(first))
    assert torch.equal(second.packed, first.packed)
    assert torch.equal(second.scales, first.scales)


def test_zero_blocks_quantize_to_zero_without_a_nan() -> None:
    """An all-zero block has amax 0, hence scale 0, hence a 0/0 that must not reach the output."""
    x = torch.zeros(128, 64)
    x[0, :16] = 1.0  # one non-zero block, so the tensor is not uniformly degenerate
    dq = dequantize_nvfp4(quantize_nvfp4(x))
    assert torch.isfinite(dq).all()
    assert torch.equal(dq[1:], torch.zeros_like(dq[1:]))


def test_quantizer_refuses_a_non_finite_operand() -> None:
    """E2M1 encodes neither NaN nor Inf, so there is nothing honest to store.

    Every other K1 rung's suite asserts "NaN in, NaN out"; this rung cannot, and the difference is
    the format's, not the kernel's. Saturating a NaN to a finite 6.0 would let a poisoned activation
    reach the tensor core disguised as data.
    """
    for bad in (float("nan"), float("inf")):
        x = torch.zeros(64, 64)
        x[3, 7] = bad
        with pytest.raises(ValueError, match="NaN or Inf"):
            quantize_nvfp4(x)


def test_packing_puts_the_lower_k_in_the_low_nibble() -> None:
    """Two E2M1 per byte, lowest element in the lowest bits — the order a b32 hands to the MMA.

    Swapping the nibbles is a bug that transposes each operand pair along K: exactly correct on any
    symmetric test input, and wrong on everything else.
    """
    x = torch.zeros(128, 64)
    x[0, 0] = 6.0  # amax of block 0 -> scale is tight, so this element encodes as code 0x7
    x[0, 1] = -6.0  # ...and this one as 0xF
    op = quantize_nvfp4(x)
    assert int(op.packed[0, 0]) & 0xF == 0x7, "k=0 is not in the low nibble"
    assert int(op.packed[0, 0]) >> 4 == 0xF, "k=1 is not in the high nibble"
    dq = dequantize_nvfp4(op)
    assert (float(dq[0, 0]), float(dq[0, 1])) == (6.0, -6.0)


def test_scales_land_where_the_layout_says_they_do() -> None:
    """The quantizer writes each block's scale at ``sf_byte_offset``, not at ``row*nblk + block``.

    A row-major scale tensor is the single most natural wrong thing to build, it passes every test
    that only round-trips through the same wrong index, and the kernel then reads garbage. So this
    checks the bytes against the scalar formula directly.
    """
    rows, k = 256, 128
    torch.manual_seed(2)
    x = torch.randn(rows, k)
    op = quantize_nvfp4(x)
    expected = _block_scales(x)
    for r in range(0, rows, 11):
        for j in range(k // SF_VEC_SIZE):
            byte = int(op.scales[sf_byte_offset(r, j * SF_VEC_SIZE, rows=rows, k_total=k)])
            assert SF_UE4M3.decode(byte) == float(expected[r, j]), (r, j)


def test_operand_rejects_a_scale_tensor_of_the_wrong_size() -> None:
    """The invariant that turns a truncated allocation into an error instead of an illegal access."""
    op = quantize_nvfp4(torch.randn(128, 64))
    with pytest.raises(ValueError, match="swizzled layout"):
        NVFP4Operand(packed=op.packed, scales=op.scales[:-1], rows=128, k=64)
    with pytest.raises(ValueError, match="expected"):
        NVFP4Operand(packed=op.packed[:, :-1], scales=op.scales, rows=128, k=64)


def test_reference_is_the_same_problem_the_kernel_solves() -> None:
    """The oracle multiplies the DEQUANTIZED operands in fp32 — the exact values the MMA sees.

    If it multiplied the original operands instead, the format's error would be inside the
    tolerance and a wrong fragment mapping could hide under it. This asserts the oracle really is
    ``dequant(A) @ dequant(B)`` and not something else that happens to be close.
    """
    torch.manual_seed(3)
    a, b = torch.randn(64, 128), torch.randn(128, 96)
    ref = reference_gemm(a, b)
    assert ref.dtype is torch.float32
    da = dequantize_nvfp4(quantize_nvfp4(a))
    db = dequantize_nvfp4(quantize_nvfp4(b.t().contiguous()))
    assert torch.equal(ref, torch.matmul(da, db.t()))


def test_quantization_error_is_large_enough_that_it_cannot_be_the_tolerance() -> None:
    """The format residual, measured — and shown to be orders above fp32 noise.

    This is the number Huy's tolerance has to be argued against. It is reported here, not asserted
    tightly, because it is a property of the data; what IS asserted is that it is far too large to
    double as a correctness bound. A gate set at this size would pass a kernel that never ran.
    """
    torch.manual_seed(4)
    a, b = torch.randn(128, 256), torch.randn(256, 128)
    err = quantization_error(a, b)
    assert set(err) == {"max_abs", "rel_max", "rel_fro"}
    assert err["rel_max"] > 1e-2, "NVFP4 on gaussian data should cost percent-level relative error"
    assert err["rel_fro"] < 1.0, "a residual of order the signal itself means the oracle is broken"


def test_wrapper_reports_an_open_hole_as_such() -> None:
    """ "You have not written the kernel" and "your kernel is broken" are different messages."""
    if not hole_is_open(SOURCE):
        pytest.skip("hole filled — the HoleOpenError path no longer applies")
    with pytest.raises((HoleOpenError, RuntimeError)) as exc:
        b_r6_gemm(torch.zeros(128, 64), torch.zeros(64, 128))
    # On a CPU box require_arch fires first (RuntimeError); on an sm120 box the loader does.
    assert "b_r6_gemm" in str(exc.value) or "HUY hole" in str(exc.value)


# =============================================================================================
# drydock — what nvcc generated. No GPU, but the CUDA container must have run.
# =============================================================================================


def _drydock(kind: str) -> str:
    p = _DRYDOCK / f"{_STEM}.sm_120a.{kind}.txt"
    if not p.is_file():
        pytest.skip(f"no {kind} report at {p} — run: infra/drydock.sh compile {SOURCE}")
    return p.read_text(encoding="utf-8", errors="replace")


@pytest.mark.drydock
def test_the_block_scaled_mma_assembles_for_sm120a() -> None:
    """``OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`` must be in the SASS — from the ISA probe.

    This one is meaningful WHILE THE HOLE IS OPEN, which is the point of the probe kernel: it
    proves today, for free, that the instruction exists in this toolkit for this arch, that the
    operand constraint letters are right, and that ``-arch=sm_120a`` (not ``sm_120``) selects it —
    ``cute/arch/config.hpp:165-173`` gates the whole mxf4nvf4 family on sm120a AND CUDA >= 12.8.
    Discovering any of those on a rented Blackwell instead costs an hour and yields no number.
    """
    sass = _drydock("sass")
    assert re.search(r"\bOMMA\.SF\.16864\.F32\.E2M1\.E2M1\.UE4M3\.4X\b", sass), (
        "no block-scaled OMMA in the SASS — either the ISA probe was dropped, or this was compiled "
        "for base sm_120 instead of sm_120a (the trailing 'a' selects the accelerated ISA)"
    )


@pytest.mark.drydock
def test_the_mainloop_itself_issues_the_block_scaled_mma() -> None:
    """The GEMM kernel — not just the probe — must contain the tensor instruction.

    Skipped while the hole is open, because the captured SASS is then the stub's scalar loop and
    says nothing about a mainloop that does not exist yet.
    """
    if hole_is_open(SOURCE):
        pytest.skip(
            "hole open — the captured SASS is the stub's, and says nothing about the kernel"
        )
    sass = _drydock("sass")
    body = sass.split("Function : b_r6_nvfp4_isa_probe")[0]
    assert "b_r6_nvfp4_sm120" in body
    assert re.search(r"\bOMMA\.SF\.16864\b", body), (
        "the GEMM kernel's SASS has no block-scaled OMMA — the mainloop is not using the tensor "
        "core this rung is about"
    )


@pytest.mark.drydock
def test_ptxas_report_is_clean() -> None:
    """0 spill bytes, no stack frame, and shared memory within sm_120a's per-CTA limit.

    64 fp32 accumulators per thread plus the operand and scale fragments is the tight part of this
    rung's register budget; a spill here is what turns a correct kernel into a slow one.
    """
    report = _drydock("ptxas")
    spill = re.search(r"(\d+) bytes spill stores", report)
    assert spill and int(spill.group(1)) == 0, f"register spill:\n{report[-400:]}"
    stack = re.search(r"(\d+) bytes stack frame", report)
    assert stack and int(stack.group(1)) == 0, f"local-memory stack frame:\n{report[-400:]}"
    smem = re.search(r"(\d+) bytes smem", report)
    if smem:
        assert int(smem.group(1)) <= ARCH["sm_120a"].smem_per_cta


# =============================================================================================
# gpu — correctness against the oracle. sm120 only.
# =============================================================================================


def _require_tolerance() -> float:
    if TOL_CONST is None:
        pytest.fail(
            "TOL_CONST is unset. The correctness gate cannot run without a tolerance, and deriving "
            "one is this rung's numerics lesson (plan §05: 'tolerance derived from quantization "
            "error'). Run quantization_error() to see what the FORMAT costs, note that it cancels "
            "between kernel and oracle, and set the constant to what fp32 accumulation over K can "
            "explain — then write the one-line argument into experiments/K1/B-R6/spec.md."
        )
    return TOL_CONST


def _assert_matches_oracle(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    ref = reference_gemm(a, b)
    assert out.shape == ref.shape, f"{out.shape} != {ref.shape}"
    assert out.dtype is torch.float32, (
        f"kernel must return fp32 (the accumulator's dtype), got {out.dtype}"
    )
    k = a.shape[1]
    bound = _require_tolerance() * (k**0.5) * ref.abs().max().item()
    err = (out - ref).abs().max().item()
    assert err <= bound, (
        f"max|Δ| = {err:.4e} > {bound:.4e} = TOL_CONST * sqrt({k}) * max|C_ref|. Both sides use the "
        f"same NVFP4 operands, so the format's error has cancelled and this is a kernel bug: a "
        f"wrong scale byte or a wrong accumulator fragment shows up as a large error on a "
        f"structured subset — print (out - ref).abs().sum(dim=0) before widening this."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_matches_oracle(shape_name: str) -> None:
    """All five spec shapes against the quantize-dequantize oracle at the spec's tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m, n, k = SHAPES[shape_name]
    if shape_name in ("sq4096", "rect8192") and os.environ.get("LADDERS_SMOKE") == "1":
        pytest.skip("smoke mode: the large shapes are the measurement, not the smoke test")
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches_oracle(b_r6_gemm(a, b), a, b)
    del a, b
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [1e-2, 1.0, 1e2], ids=["tiny", "unit", "large"])
def test_matches_oracle_across_magnitudes(scale: float) -> None:
    """The per-block scale is supposed to make the kernel scale-invariant; this is that claim."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(1)
    a = torch.randn(256, 128, device="cuda", dtype=torch.float32) * scale
    b = torch.randn(128, 256, device="cuda", dtype=torch.float32) * scale
    _assert_matches_oracle(b_r6_gemm(a, b), a, b)


@pytest.mark.gpu
def test_packed_entry_is_the_one_a_number_comes_through() -> None:
    """Pre-quantized in, GEMM only. Same answer as b_r6_gemm, without the quantize pass inside."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(2)
    a = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    packed = b_r6_gemm_packed(quantize_nvfp4(a), quantize_nvfp4(b.t().contiguous()))
    assert torch.equal(packed, b_r6_gemm(a, b))


@pytest.mark.gpu
def test_rejects_bad_inputs() -> None:
    """The guard clauses, on the device where they can actually be reached."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    def z(r: int, c: int, dt: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(r, c, device="cuda", dtype=dt)

    with pytest.raises(TypeError, match="real dtype"):
        b_r6_gemm(z(128, 64, torch.int8), z(64, 128))
    with pytest.raises(ValueError, match="inner dimensions"):
        b_r6_gemm(z(128, 64), z(128, 128))
    with pytest.raises(ValueError, match=f"K % {TILE_K}"):
        b_r6_gemm_packed(quantize_nvfp4(z(128, 32)), quantize_nvfp4(z(128, 32)))


@pytest.mark.gpu
def test_refuses_a_non_finite_operand_rather_than_saturating_it() -> None:
    """E2M1 has no NaN encoding, so the only honest behaviour is to refuse — on the device too."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    a = torch.ones(128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(64, 128, device="cuda", dtype=torch.bfloat16)
    a[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        b_r6_gemm(a, b)
