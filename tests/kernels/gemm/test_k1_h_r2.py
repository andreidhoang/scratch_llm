"""K1/H-R2 — TMA + mbarrier expect-tx, 2 stages. Three tiers of gate, in ascending cost.

  CPU      the wrapper's contracts, the oracle, the tile arithmetic, and — new at this rung — every
           cuTensorMapEncodeTiled field and both shared-memory descriptor layouts. Milliseconds.
  drydock  what nvcc actually generated: the TMA copy and the warpgroup MMA are in the main loop,
           no Ampere-style cp.async survived, nothing spills. Needs the CUDA container, still no GPU.
  gpu      correctness against the fp32 oracle at the spec shapes. Needs an H100.

The CPU tier matters more here than it did at H-R1. A tensor map is built on the *host*, at startup,
and the driver answers every illegal field with CUDA_ERROR_INVALID_VALUE and names none of them; a
descriptor whose LBO and SBO do not describe the layout TMA actually wrote does not fault at all,
it returns a plausible wrong C. Both are pure integer arithmetic, so both are decidable here.

Spec: experiments/K1/H-R2/spec.md   ·   Map: experiments/K1/H-R2/map.md
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest
import torch

from scratch_llm.kernels.common.hopper_contracts import (
    ARCH,
    SwizzleMode,
    check_smem_budget,
    check_tma_tensor_map,
    tile_covers,
)
from scratch_llm.kernels.gemm.cuda._k1_loader import (
    HoleOpenError,
    hole_is_open,
    source_for,
    workspace_root,
)
from scratch_llm.kernels.gemm.cuda.h_r2 import (
    A_DESC_LBO,
    A_DESC_SBO,
    B_CHUNKS,
    B_DESC_LBO,
    B_DESC_SBO,
    BYTES_PER_STAGE,
    ELEM_BYTES,
    EXPECT_TX_BYTES,
    RUNG,
    SMEM_BARRIER_BYTES,
    SMEM_DYNAMIC_BYTES,
    SOURCE,
    STAGES,
    SWIZZLE_ATOM_ELEMS,
    TILE_K,
    TILE_M,
    TILE_N,
    h_r2_gemm,
    reference_gemm,
    tensor_map_params,
    unsupported_reason,
)

_WORKSPACE = workspace_root()
_DRYDOCK = _WORKSPACE / "experiments" / "K1" / "H-R2" / "drydock"
#: drydock names its artifacts per SOURCE, not per rung — a rung may own more than one .cu.
_STEM = SOURCE.removesuffix(".cu")

# ---------------------------------------------------------------------------------------------
# The tolerance. HUY SETS THIS, with its argument, before the first measured run.
#
# The spec's form is  max|C_kernel - C_fp32|  <=  TOL_CONST * sqrt(K) * max|A| * max|B|
# — the same bf16 accumulation-order bound H-R1 uses, and it must stay the same number: H-R2 changes
# how the operands reach shared memory, not what is multiplied or in what precision, so a tolerance
# that had to be loosened for this rung would be evidence of a bug, not of numerics.
#
# Leaving it None is deliberate. Choosing the constant IS the rung's numerics lesson.
# ---------------------------------------------------------------------------------------------
TOL_CONST: float | None = None

#: The five shapes the K1 bench registry measures — identical to H-R1's, so the two rungs' numbers
#: are numbers about the same problem.
SHAPES: dict[str, tuple[int, int, int]] = {
    "sq4096": (4096, 4096, 4096),  # the headline: the number quoted against cuBLAS bf16
    "rect8192": (8192, 8192, 4096),  # secondary: more waves, same tile — isolates grid effects
    "skinny16": (16, 4096, 4096),  # M < TILE_M: one partial tile row, most of the machine idle
    "npot": (257, 1023, 512),  # non-power-of-two: M and N predication, K still exact
    "untuned": (1536, 6144, 2560),  # divides nothing the tile likes; not a tuning target
}

#: What the correctness gate actually runs. ``npot`` drops out — N=1023 gives a 2046-byte row
#: stride and TMA requires a multiple of 16, so no tensor map for B exists at that shape. Dropping
#: it would also drop the only N-tail in the set, so one is put back at a legal alignment:
#: 1032 = 8*129 is TMA-legal and 1032 % 128 = 8, so the last N tile is 8 columns wide.
GATE_SHAPES: dict[str, tuple[int, int, int]] = {
    **{name: mnk for name, mnk in SHAPES.items() if unsupported_reason(*mnk) is None},
    "ntail": (257, 1032, 512),
}


# =============================================================================================
# The hole guard — exactly one per hole (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K1/H-R2", "csrc/gemm/h_r2_tma_bf16_sm90.cu")
def test_h_r2_kernel_body_is_filled_and_compiles_clean() -> None:
    """Fails while the mainloop is a ``#error``; passes when it compiles clean for sm_90a.

    The stdout assertion is not belt-and-braces: ``drydock.sh compile <name>`` filters its registry
    by substring, and a name with no matching row compiles zero files and exits 0. Without it this
    test would go green on an empty run the day the hole is filled but the registry row is not.
    """
    assert not hole_is_open(SOURCE), (
        f"{RUNG} kernel body is still an open hole. Read experiments/K1/H-R2/spec.md and "
        f"experiments/K1/H-R2/map.md, then write the 2-stage k-loop and epilogue in csrc/gemm/{SOURCE}."
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
    assert SOURCE in proc.stdout and "✔" in proc.stdout, (
        f"dry-dock compiled nothing for {SOURCE} — add its row to infra/drydock.sh's _REGISTRY "
        f"(lead-only), or the gate passes without compiling anything:\n{proc.stdout}"
    )


# =============================================================================================
# CPU — the source and the wrapper agree
# =============================================================================================


def _source() -> str:
    return source_for(SOURCE).read_text(encoding="utf-8")


def test_source_and_wrapper_agree_on_the_tile_and_stage_count() -> None:
    """The ``.cu``'s defines and the wrapper's constants must not drift.

    They are used for different things — one generates code, the other feeds the shared-memory,
    tensor-map and descriptor arithmetic below — so nothing else would notice if they diverged.
    """
    src = _source()
    for macro, expected in (
        ("BM", TILE_M),
        ("BN", TILE_N),
        ("BK", TILE_K),
        ("STAGES", STAGES),
        ("SWZ_ELEMS", SWIZZLE_ATOM_ELEMS),
        ("A_DESC_LBO", A_DESC_LBO),
        ("A_DESC_SBO", A_DESC_SBO),
        ("B_DESC_SBO", B_DESC_SBO),
    ):
        m = re.search(rf"^#define {macro} (\d+)", src, re.MULTILINE)
        assert m, f"#define {macro} not found in csrc/gemm/{SOURCE}"
        assert int(m.group(1)) == expected, (
            f"{macro}={m.group(1)} in the .cu but {expected} in h_r2.py"
        )


def test_tile_is_frozen_at_h_r1s() -> None:
    """The ladder's one-variable rule, asserted rather than trusted.

    H-R2 changes the staging path. If it also changed the tile, the measured delta against H-R1
    would be attributable to neither, and the rung would have taught nothing.
    """
    from scratch_llm.kernels.gemm.cuda.h_r1 import TILE_K as R1_K
    from scratch_llm.kernels.gemm.cuda.h_r1 import TILE_M as R1_M
    from scratch_llm.kernels.gemm.cuda.h_r1 import TILE_N as R1_N

    assert (TILE_M, TILE_N, TILE_K) == (R1_M, R1_N, R1_K)


def test_b_tile_needs_two_boxes_because_of_the_swizzle_atom() -> None:
    """The rule that shapes this rung's whole B path, as an assertion rather than a comment.

    Under 128B swizzle the innermost box extent must be exactly 128 B. TILE_N bf16 is 256 B, so the
    obvious single box ``{TILE_N, TILE_K}`` is illegal — and the driver would say only
    CUDA_ERROR_INVALID_VALUE. Two boxes of ``SWIZZLE_ATOM_ELEMS`` are the split that fixes it.
    """
    errs = check_tma_tensor_map(
        rank=2,
        elem_bytes=ELEM_BYTES,
        global_dims=[4096, 4096],
        global_strides_bytes=[4096 * ELEM_BYTES],
        box_dims=[TILE_N, TILE_K],
        swizzle=SwizzleMode.B128,
    )
    assert any("swizzle requires exactly 128 B" in e for e in errs), errs
    assert B_CHUNKS * SWIZZLE_ATOM_ELEMS == TILE_N
    assert SwizzleMode.B128.atom_bytes == SWIZZLE_ATOM_ELEMS * ELEM_BYTES


@pytest.mark.parametrize("shape_name", list(GATE_SHAPES))
def test_both_tensor_maps_are_legal_at_every_gate_shape(shape_name: str) -> None:
    """Every cuTensorMapEncodeTiled field, checked by name, before an hour is spent on the box."""
    m, n, k = GATE_SHAPES[shape_name]
    for operand, params in tensor_map_params(m, n, k).items():
        assert check_tma_tensor_map(**params) == [], f"{shape_name}/{operand}"  # type: ignore[arg-type]


def test_npot_is_rejected_by_the_wrapper_and_by_the_tensor_map_alike() -> None:
    """H-R1's ``npot`` shape is not measurable here, and both ends of the stack say why.

    N=1023 is a 2046-byte row stride and TMA requires a multiple of 16. That is a property of the
    hardware, not a corner this rung cut, so the wrapper must name it rather than launch and produce
    garbage — and the tensor-map checker must independently reach the same verdict.
    """
    m, n, k = SHAPES["npot"]
    why = unsupported_reason(m, n, k)
    assert why is not None and "16 B" in why
    errs = check_tma_tensor_map(**tensor_map_params(m, n, k)["b"])  # type: ignore[arg-type]
    assert any("not a multiple of 16 bytes" in e for e in errs), errs


def test_gate_shapes_still_exercise_both_tails() -> None:
    """Dropping ``npot`` must not quietly drop the predication coverage it was carrying."""
    assert any(
        not tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["m"]
        for m, n, k in GATE_SHAPES.values()
    ), "no gate shape exercises the M tail"
    assert any(
        not tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["n"]
        for m, n, k in GATE_SHAPES.values()
    ), "no gate shape exercises the N tail — TMA's out-of-bounds zero-fill would be untested"


# =============================================================================================
# CPU — shared memory and the pipeline arithmetic
# =============================================================================================


def test_two_stages_need_the_dynamic_smem_opt_in_and_the_launcher_makes_it() -> None:
    """64 KB is legal on Hopper and NOT free: past 48 KB the launch must opt in.

    ptxas emits the kernel either way and the failure appears only as a launch error at runtime.
    The dry dock cannot catch this one — its shared-memory gate reads the ptxas report, which counts
    static shared memory and says 0 for a dynamic allocation — so this assertion is the real gate.
    """
    errs = check_smem_budget(
        bytes_per_stage=BYTES_PER_STAGE,
        stages=STAGES,
        arch=ARCH["sm_90a"],
        extra_bytes=SMEM_BARRIER_BYTES,
    )
    assert len(errs) == 1 and errs[0].startswith("OPT-IN REQUIRED"), errs
    assert str(SMEM_DYNAMIC_BYTES + SMEM_BARRIER_BYTES) in errs[0], errs[0]
    src = _source()
    assert "cudaFuncAttributeMaxDynamicSharedMemorySize" in src, (
        "the launcher never opts in to > 48 KB of shared memory — the launch will fail at runtime"
    )
    assert re.search(r"<<<grid, block, SMEM_DYNAMIC_BYTES>>>", src), (
        "the launch does not pass the dynamic shared-memory size"
    )


def test_expect_tx_equals_what_the_three_tma_boxes_deliver() -> None:
    """The mbarrier counts bytes, not arrivals. Too many hangs; too few reads a half-written tile."""
    a_box = TILE_K * TILE_M * ELEM_BYTES
    b_boxes = B_CHUNKS * (SWIZZLE_ATOM_ELEMS * TILE_K * ELEM_BYTES)
    assert a_box + b_boxes == EXPECT_TX_BYTES == BYTES_PER_STAGE
    assert BYTES_PER_STAGE * STAGES == SMEM_DYNAMIC_BYTES


# =============================================================================================
# CPU — the two shared-memory descriptors describe the layout TMA actually writes
#
# This is the assertion that would otherwise cost a rented hour. A descriptor whose LBO/SBO do not
# match the smem layout does not fault: the MMA reads the wrong core matrix and returns a plausible
# C. The two canonical forms are CUTLASS make_gmma_desc's, quoted at
# oss/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:186-296, restated here in bytes.
# =============================================================================================


def _tma_offset_a(m: int, k: int) -> int:
    """Byte offset TMA writes for A element (m,k): box {TILE_K, TILE_M}, K innermost."""
    return (k + TILE_K * m) * ELEM_BYTES


def _tma_offset_b(n: int, k: int) -> int:
    """Byte offset TMA writes for B element (n,k): two boxes {SWIZZLE_ATOM_ELEMS, TILE_K}."""
    chunk, inner = divmod(n, SWIZZLE_ATOM_ELEMS)
    return (inner + SWIZZLE_ATOM_ELEMS * k) * ELEM_BYTES + chunk * (
        SWIZZLE_ATOM_ELEMS * TILE_K * ELEM_BYTES
    )


def test_a_descriptor_walk_matches_the_tma_layout() -> None:
    """Major::K, B128: ((8,m),(T,2)):((8T,SBO),(1,LBO)) — the pair H-R1 already uses, unchanged."""
    for m in range(TILE_M):
        for k in range(TILE_K):
            walk = (
                (m % 8) * 128 + (m // 8) * A_DESC_SBO + (k // 8) * A_DESC_LBO + (k % 8) * ELEM_BYTES
            )
            assert walk == _tma_offset_a(m, k), f"A descriptor disagrees with TMA at (m={m}, k={k})"


def test_b_descriptor_walk_matches_the_tma_layout() -> None:
    """Major::MN, B128: ((T,8,n),(8,k)):((1,T,LBO),(8T,SBO)).

    LBO is the stride between the two 64-wide N halves (8192 B, i.e. one whole box), SBO the stride
    between k-groups of 8 (1024 B). Swapping them is the single most common descriptor bug and it
    does not fault.
    """
    for n in range(TILE_N):
        for k in range(TILE_K):
            walk = (
                (n % 8) * ELEM_BYTES
                + ((n // 8) % 8) * 16
                + (n // 64) * B_DESC_LBO
                + (k % 8) * 128
                + (k // 8) * B_DESC_SBO
            )
            assert walk == _tma_offset_b(n, k), f"B descriptor disagrees with TMA at (n={n}, k={k})"


def test_the_two_operands_do_not_share_a_descriptor_layout() -> None:
    """The rung's one real asymmetry, asserted so a later "simplification" cannot erase it.

    TMA cannot transpose. A arrives K-contiguous and B arrives N-contiguous, so B is an MN-major
    operand — which is also why the wgmma's last immediate is 1 here and 0 in H-R1.
    """
    assert (A_DESC_LBO, A_DESC_SBO) != (B_DESC_LBO, B_DESC_SBO)
    assert B_DESC_LBO == SWIZZLE_ATOM_ELEMS * TILE_K * ELEM_BYTES
    asm = re.search(r'"%64, %65, %66, (\d), (\d), (\d), (\d);', _source())
    assert asm, "the wgmma immediates were not found in csrc/gemm/" + SOURCE
    scale_a, scale_b, trans_a, trans_b = (int(g) for g in asm.groups())
    assert (scale_a, scale_b) == (1, 1)
    assert trans_a == 0, "A is K-major (GMMA::Major::K = 0)"
    assert trans_b == 1, (
        "B reaches shared memory N-contiguous, so trans-b must be 1 (GMMA::Major::MN). Cloning "
        "H-R1's `1, 1, 0, 0` compiles, runs, and returns a plausible wrong C."
    )


# =============================================================================================
# CPU — the oracle and the open-hole message
# =============================================================================================


def test_reference_is_fp32_over_the_same_bf16_operands() -> None:
    """The oracle must not be a different problem — it up-casts, it does not re-generate.

    If the reference multiplied fp32 operands, the operands' own bf16 quantization error would be
    inside the tolerance, and a real kernel bug could hide under it.
    """
    torch.manual_seed(0)
    a = torch.randn(64, 32, dtype=torch.bfloat16)
    b = torch.randn(32, 48, dtype=torch.bfloat16)
    ref = reference_gemm(a, b)
    assert ref.dtype is torch.float32
    assert torch.equal(ref, torch.matmul(a.float(), b.float()))


def test_wrapper_reports_an_open_hole_as_such() -> None:
    """ "You have not written the kernel" and "your kernel is broken" are different messages."""
    if not hole_is_open(SOURCE):
        pytest.skip("hole filled — the HoleOpenError path no longer applies")
    with pytest.raises((HoleOpenError, RuntimeError)) as exc:
        h_r2_gemm(
            torch.zeros(128, 64, dtype=torch.bfloat16),
            torch.zeros(64, 128, dtype=torch.bfloat16),
        )
    # On a CPU box require_arch fires first (RuntimeError); on a Hopper box the loader does.
    assert "h_r2_gemm" in str(exc.value) or "HUY hole" in str(exc.value)


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
def test_sass_issues_the_tensor_copy_and_the_warpgroup_mma() -> None:
    """The two instructions this rung is about, in the code nvcc actually emitted."""
    sass = _sass()
    assert re.search(r"\b(UTMALDG|UBLKCP)\b", sass), (
        "no bulk-tensor copy in the SASS — the mainloop is not using TMA, which is the entire "
        "difference between this rung and H-R1"
    )
    assert re.search(r"\bHGMMA\b", sass), (
        "no HGMMA in the SASS — either the mainloop does not issue wgmma, or it was compiled for "
        "base sm_90 instead of sm_90a (the trailing 'a' selects the accelerated ISA)"
    )
    assert re.search(r"\bSYNCS\b", sass), (
        "no SYNCS in the SASS — the mbarrier arrive/wait pair is missing, so nothing is ordering "
        "the TMA against the MMA"
    )


@pytest.mark.drydock
def test_sass_has_no_ampere_style_async_copy() -> None:
    """``LDGSTS`` is cp.async — H-R1's staging path. Its presence means TMA did not replace it."""
    sass = _sass()
    assert not re.findall(r"\bLDGSTS\b", sass), (
        "cp.async (LDGSTS) survived in the mainloop: this rung's premise is that TMA replaces it"
    )


@pytest.mark.drydock
def test_sass_has_no_local_memory_traffic() -> None:
    """``LDL``/``STL`` mean registers spilled to local memory — the accumulator did not fit."""
    sass = _sass()
    spills = re.findall(r"\b(LDL|STL)\b", sass)
    assert not spills, f"{len(spills)} local-memory accesses in the SASS: the accumulator spilled"


@pytest.mark.drydock
def test_ptxas_report_is_clean() -> None:
    """0 spill bytes, and static shared memory within the arch limit.

    The static figure here is only the mbarriers; the 64 KB of staging is dynamic and does not
    appear in this report at all, which is why the opt-in has its own CPU test above.
    """
    p = _DRYDOCK / f"{_STEM}.sm_90a.ptxas.txt"
    if not p.is_file():
        pytest.skip(f"no ptxas report at {p} — run: infra/drydock.sh compile {SOURCE}")
    report = p.read_text(encoding="utf-8", errors="replace")
    spill = re.search(r"(\d+) bytes spill stores", report)
    assert spill and int(spill.group(1)) == 0, f"register spill in {p.name}: {report[-400:]}"
    smem = re.search(r"(\d+) bytes smem", report)
    if smem:
        assert int(smem.group(1)) <= ARCH["sm_90a"].smem_per_cta


# =============================================================================================
# gpu — correctness against the oracle. H100 only.
# =============================================================================================


def _require_tolerance() -> float:
    if TOL_CONST is None:
        pytest.fail(
            "TOL_CONST is unset. The correctness gate cannot run without a tolerance, and picking "
            "one is the rung's numerics lesson: set it in this file and write the one-line argument "
            "into experiments/K1/H-R2/spec.md's 'Correctness gate' line before the first measured run."
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
        f"At this rung the two structured failures to look for first are a wrong wgmma trans-b "
        f"(B read as K-major: error spread over every column) and a wrong stage/phase (error "
        f"confined to k-blocks 2, 4, 6...). Print (out - ref).abs().sum(dim=0) before widening this."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("shape_name", list(GATE_SHAPES))
def test_matches_oracle(shape_name: str) -> None:
    """Every TMA-legal spec shape against the fp32 oracle at the spec's tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m, n, k = GATE_SHAPES[shape_name]
    if shape_name in ("sq4096", "rect8192") and os.environ.get("LADDERS_SMOKE") == "1":
        pytest.skip("smoke mode: the large shapes are the measurement, not the smoke test")
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches_oracle(h_r2_gemm(a, b), a, b)
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
    _assert_matches_oracle(h_r2_gemm(a, b), a, b)


@pytest.mark.gpu
def test_multi_k_block_shape_exercises_the_pipeline_wrap() -> None:
    """K must span more than STAGES k-blocks or the phase bit is never flipped by the test.

    A 2-stage pipeline with a wrong phase parity is correct for the first STAGES iterations and
    hangs or reads stale data on the next one. A K of 2*TILE_K would pass a broken kernel.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    k = TILE_K * (2 * STAGES + 1)
    torch.manual_seed(2)
    a = torch.randn(TILE_M, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, TILE_N, device="cuda", dtype=torch.bfloat16)
    _assert_matches_oracle(h_r2_gemm(a, b), a, b)


@pytest.mark.gpu
def test_rejects_bad_inputs() -> None:
    """The guard clauses, on the device where they can actually be reached."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    def z(r: int, c: int, dt: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(r, c, device="cuda", dtype=dt)

    with pytest.raises(TypeError, match="bfloat16"):
        h_r2_gemm(z(128, 64, torch.float16), z(64, 128))
    with pytest.raises(ValueError, match="inner dimensions"):
        h_r2_gemm(z(128, 64), z(128, 128))
    with pytest.raises(ValueError, match=f"K % {TILE_K}"):
        h_r2_gemm(z(128, 32), z(32, 128))
    with pytest.raises(ValueError, match="16 B"):
        h_r2_gemm(z(128, 64), z(64, 1023))


@pytest.mark.gpu
def test_handles_nan_and_inf_without_masking_them() -> None:
    """A NaN in must produce a NaN out — a kernel that swallows them hides real training bugs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    a = torch.ones(128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(64, 128, device="cuda", dtype=torch.bfloat16)
    a[0, 0] = float("nan")
    b[0, 1] = float("inf")
    out = h_r2_gemm(a, b)
    assert torch.isnan(out[0, 0]), "NaN was swallowed"
    assert not torch.isfinite(out[:, 1]).all(), "Inf was swallowed"
