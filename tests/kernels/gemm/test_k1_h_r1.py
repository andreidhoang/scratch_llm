"""K1/H-R1 — warpgroup MMA, single stage. Three tiers of gate, in ascending cost.

  CPU      the wrapper's contracts, the oracle, and the tile-shape arithmetic. Milliseconds, here.
  drydock  what nvcc actually generated: the tensor instruction is in the main loop, nothing spills
           to local memory. Needs the CUDA container (infra/drydock.sh), still no GPU.
  gpu      correctness against the fp32 oracle at the five spec shapes. Needs an H100.

Only the last of those can say the kernel is right, and only it costs money — so the first two are
built to catch everything they possibly can before it is spent.

Spec: experiments/K1/H-R1/spec.md   ·   Map: experiments/K1/H-R1/map.md
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
from scratch_llm.kernels.gemm.cuda.h_r1 import (
    RUNG,
    SOURCE,
    TILE_K,
    TILE_M,
    TILE_N,
    h_r1_gemm,
    reference_gemm,
)

_WORKSPACE = Path(__file__).resolve().parents[3].parent  # scratch_llm/tests/kernels/gemm -> ladders
_DRYDOCK = _WORKSPACE / "experiments" / "K1" / "H-R1" / "drydock"

# ---------------------------------------------------------------------------------------------
# The tolerance. HUY SETS THIS, with its argument, before the first measured run.
#
# The spec's form is  max|C_kernel - C_fp32|  <=  TOL_CONST * sqrt(K) * max|A| * max|B|
# — a bf16 accumulation-order bound: each output element is a sum of K products, the tensor core
# accumulates in fp32 but the *operands* were rounded to bf16 (8 explicit mantissa bits, so a
# relative step of 2^-8), and independent rounding errors grow as sqrt(K) rather than K.
#
# Leaving it None is deliberate. Choosing the constant IS the rung's numerics lesson: too loose and
# the test cannot fail, too tight and a correct kernel whose reduction order differs from cuBLAS's
# is rejected. Nobody can pick it for him and have him learn anything.
# ---------------------------------------------------------------------------------------------
TOL_CONST: float | None = None

#: The five shapes the spec measures. Each is here for a reason, and the reason is the comment.
SHAPES: dict[str, tuple[int, int, int]] = {
    "sq4096": (4096, 4096, 4096),  # the headline: the number quoted against cuBLAS bf16
    "rect8192": (8192, 8192, 4096),  # secondary: more waves, same tile — isolates grid effects
    "skinny16": (16, 4096, 4096),  # M < TILE_M: one partial tile row, most of the machine idle
    "npot": (257, 1023, 512),  # non-power-of-two: M and N predication, K still exact
    "untuned": (1536, 6144, 2560),  # divides nothing the tile likes; not a tuning target
}


# =============================================================================================
# The hole guard — exactly one per hole (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K1/H-R1", "csrc/gemm/h_r1_wgmma_bf16_sm90.cu")
def test_h_r1_kernel_body_is_filled_and_compiles_clean() -> None:
    """Fails while the mainloop is a ``#error``; passes when it compiles clean for sm_90a.

    This is the CPU-side half of the gate and it is genuinely useful on its own: a kernel that does
    not compile, spills registers, or overflows shared memory is not worth renting an H100 to find
    out about. Correctness still needs the GPU — that is ``test_matches_oracle`` below.
    """
    assert not hole_is_open(SOURCE), (
        f"{RUNG} kernel body is still an open hole. Read experiments/K1/H-R1/spec.md and "
        f"experiments/K1/H-R1/map.md, then write the k-loop and epilogue in csrc/gemm/{SOURCE}."
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


def test_source_and_wrapper_agree_on_the_tile_shape() -> None:
    """The ``.cu``'s BM/BN/BK defines and the wrapper's TILE_* must not drift.

    They are used for different things — one generates code, the other feeds the shared-memory and
    wave-quantization arithmetic — so nothing else would notice if they diverged.
    """
    src = source_for(SOURCE).read_text(encoding="utf-8")
    for macro, expected in (("BM", TILE_M), ("BN", TILE_N), ("BK", TILE_K)):
        m = re.search(rf"^#define {macro} (\d+)", src, re.MULTILINE)
        assert m, f"#define {macro} not found in csrc/gemm/{SOURCE}"
        assert int(m.group(1)) == expected, (
            f"{macro}={m.group(1)} in the .cu but {expected} in h_r1.py"
        )


def test_tile_shape_fits_shared_memory_without_the_opt_in() -> None:
    """One stage of A+B in bf16 must stay under 48 KB, or the launch needs an opt-in it does not make."""
    bytes_per_stage = (TILE_M * TILE_K + TILE_N * TILE_K) * 2
    assert check_smem_budget(bytes_per_stage=bytes_per_stage, stages=1, arch=ARCH["sm_90a"]) == []


def test_every_spec_shape_is_legal_for_this_rung() -> None:
    """K must divide TILE_K for all five shapes — H-R1 has no K-remainder path and says so."""
    for name, (m, n, k) in SHAPES.items():
        assert k % TILE_K == 0, f"{name}: K={k} is not a multiple of TILE_K={TILE_K}"
        cover = tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)
        assert cover["k"], name
    # ...and at least one shape must NOT divide M or N, or nothing tests the epilogue's predication.
    assert any(
        not tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["m"]
        for m, n, k in SHAPES.values()
    ), "no shape exercises M predication — the epilogue's bounds checks would be untested"


def test_skinny_shape_is_grid_limited_and_the_suite_knows_it() -> None:
    """M=16 cannot fill an H100. Any %-of-cuBLAS there measures occupancy, not the mainloop.

    Asserted rather than commented so that a future tile-shape change cannot quietly turn the
    skinny row into a number that looks like a mainloop regression.
    """
    m, n, _ = SHAPES["skinny16"]
    wq = wave_quantization(m=m, n=n, tile_m=TILE_M, tile_n=TILE_N, sm_count=132)
    assert wq.full_waves == 0 and wq.tail_utilization < 0.25


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
        h_r1_gemm(
            torch.zeros(128, 64, dtype=torch.bfloat16),
            torch.zeros(64, 128, dtype=torch.bfloat16),
        )
    # On a CPU box require_arch fires first (RuntimeError); on a Hopper box the loader does.
    assert "h_r1_gemm" in str(exc.value) or "HUY hole" in str(exc.value)


# =============================================================================================
# drydock — what nvcc generated. No GPU, but the CUDA container must have run.
# =============================================================================================


def _sass() -> str:
    p = _DRYDOCK / "sass.sm_90a.txt"
    if not p.is_file():
        pytest.skip(f"no SASS at {p} — run: infra/drydock.sh compile {SOURCE}")
    if hole_is_open(SOURCE):
        pytest.skip(
            "hole open — the captured SASS is the stub's, and says nothing about the kernel"
        )
    return p.read_text(encoding="utf-8", errors="replace")


@pytest.mark.drydock
def test_sass_issues_the_warpgroup_mma() -> None:
    """``HGMMA`` must appear, or the kernel is not using the tensor core this rung is about.

    A wgmma that assembled to nothing is the specific failure mode of forgetting the ``a`` in
    ``-arch=sm_90a``: everything compiles, the kernel runs, and it is silently a scalar loop.
    """
    sass = _sass()
    assert re.search(r"\bHGMMA\b", sass), (
        "no HGMMA in the SASS — either the mainloop does not issue wgmma, or it was compiled for "
        "base sm_90 instead of sm_90a (the trailing 'a' selects the accelerated ISA)"
    )


@pytest.mark.drydock
def test_sass_has_no_local_memory_traffic() -> None:
    """``LDL``/``STL`` mean registers spilled to local memory — the accumulator did not fit."""
    sass = _sass()
    spills = re.findall(r"\b(LDL|STL)\b", sass)
    assert not spills, f"{len(spills)} local-memory accesses in the SASS: the accumulator spilled"


@pytest.mark.drydock
def test_ptxas_report_is_clean() -> None:
    """0 spill bytes, and shared memory within the arch limit — read from the ptxas report."""
    p = _DRYDOCK / "ptxas.sm_90a.txt"
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

pytestmark_gpu = pytest.mark.gpu


def _require_tolerance() -> float:
    if TOL_CONST is None:
        pytest.fail(
            "TOL_CONST is unset. The correctness gate cannot run without a tolerance, and picking "
            "one is the rung's numerics lesson: set it in this file and write the one-line argument "
            "into experiments/K1/H-R1/spec.md's 'Correctness gate' line before the first measured run."
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
        f"A wrong descriptor phase or a wrong accumulator mapping shows up here as a large error on "
        f"a structured subset of the output — print (out - ref).abs().sum(dim=0) before widening this."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_matches_oracle(shape_name: str) -> None:
    """All five spec shapes against the fp32 oracle at the spec's tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m, n, k = SHAPES[shape_name]
    if shape_name in ("sq4096", "rect8192") and os.environ.get("LADDERS_SMOKE") == "1":
        pytest.skip("smoke mode: the large shapes are the measurement, not the smoke test")
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches_oracle(h_r1_gemm(a, b), a, b)
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
    _assert_matches_oracle(h_r1_gemm(a, b), a, b)


@pytest.mark.gpu
def test_rejects_bad_inputs() -> None:
    """The guard clauses, on the device where they can actually be reached."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    def z(r: int, c: int, dt: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(r, c, device="cuda", dtype=dt)

    with pytest.raises(TypeError, match="bfloat16"):
        h_r1_gemm(z(128, 64, torch.float16), z(64, 128))
    with pytest.raises(ValueError, match="inner dimensions"):
        h_r1_gemm(z(128, 64), z(128, 128))
    with pytest.raises(ValueError, match=f"K % {TILE_K}"):
        h_r1_gemm(z(128, 32), z(32, 128))


@pytest.mark.gpu
def test_handles_nan_and_inf_without_masking_them() -> None:
    """A NaN in must produce a NaN out — a kernel that swallows them hides real training bugs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    a = torch.ones(128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(64, 128, device="cuda", dtype=torch.bfloat16)
    a[0, 0] = float("nan")
    b[0, 1] = float("inf")
    out = h_r1_gemm(a, b)
    assert torch.isnan(out[0, 0]), "NaN was swallowed"
    assert not torch.isfinite(out[:, 1]).all(), "Inf was swallowed"
