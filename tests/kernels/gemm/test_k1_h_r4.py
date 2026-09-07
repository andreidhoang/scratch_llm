"""K1/H-R4 — persistent + cluster + multicast. Three tiers of gate, in ascending cost.

  CPU      the wrapper's contracts, the oracle, both descriptor layouts, the TMA tensor maps, the
           shared-memory and register budgets, and the wave arithmetic that IS this rung's thesis.
  drydock  what nvcc actually generated: the tensor instruction and the TMA are in the main loop,
           nothing spills. Needs the CUDA container (infra/drydock.sh), still no GPU.
  gpu      correctness against the fp32 oracle at the spec shapes. Needs an H100.

Only the last of those can say the kernel is right, and only it costs money — so the first two are
built to catch everything they possibly can before it is spent. At this rung that is a lot: a
persistent kernel's grid, its cluster's multicast mask, its two descriptor layouts and its
shared-memory budget are all pure integer arithmetic, and every one of them is a bug that costs an
hour of H100 and returns a plausible wrong number rather than an error.

Spec: experiments/K1/H-R4/spec.md   ·   Map: experiments/K1/H-R4/map.md
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess

import pytest
import torch

from scratch_llm.kernels.common.hopper_contracts import (
    ARCH,
    SwizzleMode,
    check_setmaxnreg,
    check_smem_budget,
    check_tma_tensor_map,
    descriptor_fields,
    max_stages,
    tile_covers,
    wave_quantization,
    wgmma_smem_descriptor,
)
from scratch_llm.kernels.gemm.cuda._k1_loader import (
    HoleOpenError,
    hole_is_open,
    source_for,
    workspace_root,
)
from scratch_llm.kernels.gemm.cuda.h_r4 import (
    A_DESC_LBO_BYTES,
    A_DESC_SBO_BYTES,
    B_DESC_LBO_BYTES,
    B_DESC_SBO_BYTES,
    CLUSTER_M,
    CLUSTER_N,
    CLUSTER_SIZE,
    CONSUMER_REGS,
    GROUP_M,
    GROUP_N,
    NUM_CONSUMER_WG,
    PRODUCER_REGS,
    QSIZE,
    RUNG,
    SOURCE,
    THREADS_PER_CTA,
    TILE_K,
    TILE_M,
    TILE_N,
    TMA_ATOM_ELEMS,
    barrier_bytes,
    cluster_tiles,
    h_r4_gemm,
    persistent_grid,
    reference_gemm,
    shape_is_legal,
    stage_bytes,
)

_WORKSPACE = workspace_root()
_DRYDOCK = _WORKSPACE / "experiments" / "K1" / "H-R4" / "drydock"

#: H100 SXM. Hard-coded rather than queried because these are CPU tests and the whole point is that
#: the arithmetic is decidable without the device.
SM_COUNT = 132

# ---------------------------------------------------------------------------------------------
# The tolerance. HUY SETS THIS, with its argument, before the first measured run.
#
# The spec's form is  max|C_kernel - C_fp32|  <=  TOL_CONST * sqrt(K) * max|A| * max|B|
# — a bf16 accumulation-order bound: each output element is a sum of K products, the tensor core
# accumulates in fp32 but the *operands* were rounded to bf16 (8 explicit mantissa bits, so a
# relative step of 2^-8), and independent rounding errors grow as sqrt(K) rather than K.
#
# H-R4 does not change the arithmetic that bound describes — the reduction is still one fp32
# accumulator per output element, walked in the same k order — so the constant that fits H-R1 must
# fit here. If it does not, the difference is a bug in the scheduler or the epilogue, not numerics,
# and widening the tolerance would hide exactly the thing this rung can get wrong.
# ---------------------------------------------------------------------------------------------
TOL_CONST: float | None = None

#: The five shapes the bench can measure — identical to ``bench/kernels/gemm/k1_ladder.py``'s
#: SHAPES, which is what makes H-R1's number and H-R4's number comparable at all.
SHAPES: dict[str, tuple[int, int, int]] = {
    "sq4096": (4096, 4096, 4096),  # the headline: the number quoted against cuBLAS bf16
    "rect8192": (8192, 8192, 4096),  # secondary: more waves, same tile — isolates grid effects
    "skinny16": (16, 4096, 4096),  # M < TILE_M: one partial cluster-tile row, machine mostly idle
    "npot": (257, 1023, 512),  # non-power-of-two — NOT runnable here, see below
    "untuned": (1536, 6144, 2560),  # divides nothing the tile likes; not a tuning target
}

#: ``npot`` has N=1023. TMA's innermost box dimension is exactly one 128 B swizzle atom, so B's
#: tensor map cannot describe an N that is not a multiple of 64 — no predication saves this, the
#: descriptor simply cannot be encoded. H-R1 ran it (a cp.async gather has no such constraint);
#: H-R4 refuses it, loudly, and that regression in shape coverage is a real cost of adopting TMA
#: rather than something to paper over.
REJECTED_AT_THIS_RUNG = {"npot"}
RUNNABLE = {k: v for k, v in SHAPES.items() if k not in REJECTED_AT_THIS_RUNG}

#: A shape sitting just past one wave: ceil(3400/128) x ceil(1088/256) = 27 x 5 = 135 CTA-tiles
#: against 132 SMs. That is the worst case for wave quantization and the case the persistence
#: tests below are calibrated on. M and N are both deliberately ragged (3400 % 128 = 72,
#: 1088 % 256 = 64) while staying legal here (K % 64 == 0, N % 64 == 0), which makes this the only
#: shape that tests the epilogue's column predication at all. It is measurable on the box; it is
#: not in ``SHAPES`` only because that dict is the bench's registry and shared with H-R1.
WAVE_CLIFF: tuple[int, int, int] = (3400, 1088, 4096)


# =============================================================================================
# The hole guard — exactly one per hole (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K1/H-R4", "csrc/gemm/h_r4_persistent_bf16_sm90.cu")
def test_h_r4_kernel_body_is_filled_and_compiles_clean() -> None:
    """Fails while the persistent loop is a ``#error``; passes when it compiles clean for sm_90a.

    One hole covers two definitions — ``GroupedRaster::next_tile`` and ``h_r4_persistent_body`` —
    because the grouped raster's index arithmetic and the mainloop it feeds are one lesson.
    Compiling is the CPU-side half of the gate and is worth having on its own: a kernel that spills
    the 128-register accumulator or overflows shared memory is not worth renting an H100 to find
    out about. Correctness still needs the GPU — that is ``test_matches_oracle`` below.
    """
    assert not hole_is_open(SOURCE), (
        f"{RUNG} persistent loop is still an open hole. Read experiments/K1/H-R4/spec.md and "
        f"experiments/K1/H-R4/map.md, then write the grouped-raster mapping, the k-mainloop and "
        f"the epilogue in csrc/gemm/{SOURCE}."
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
# CPU — the source and the wrapper must not drift
# =============================================================================================


def _src() -> str:
    return source_for(SOURCE).read_text(encoding="utf-8")


def _define(src: str, macro: str) -> int:
    m = re.search(rf"^#define {macro} (\d+)", src, re.MULTILINE)
    assert m, f"#define {macro} not found in csrc/gemm/{SOURCE}"
    return int(m.group(1))


def test_source_and_wrapper_agree_on_the_tile_and_cluster_shape() -> None:
    """The ``.cu``'s defines and the wrapper's constants feed different machines — codegen on one
    side, the smem/wave/TMA arithmetic on the other — so nothing else would notice a divergence."""
    src = _src()
    for macro, expected in (
        ("BM", TILE_M),
        ("BN", TILE_N),
        ("BK", TILE_K),
        ("QSIZE", QSIZE),
        ("CLUSTER_M", CLUSTER_M),
        ("CLUSTER_N", CLUSTER_N),
        ("GROUP_M", GROUP_M),
        ("GROUP_N", GROUP_N),
        ("NUM_CONSUMER_WG", NUM_CONSUMER_WG),
        ("PRODUCER_REGS", PRODUCER_REGS),
        ("CONSUMER_REGS", CONSUMER_REGS),
        ("TMA_ATOM_ELEMS", TMA_ATOM_ELEMS),
    ):
        assert _define(src, macro) == expected, f"{macro} differs between the .cu and h_r4.py"


def test_the_grouped_raster_group_size_is_the_one_the_map_cites() -> None:
    """``GROUP_M x GROUP_N`` is a tuning knob, not a formality, and the map pins where it came from.

    Upstream instantiates ``Schedule<1, ..., 16/CLUSTER_M, 8/CLUSTER_N>`` at ``matmul_10.cuh:429``
    — an 8x8 supertile of cluster-tiles. At 4096^3 the cluster-tile grid is 16x16, so exactly 2x2
    supertiles: change the group and you change how much of A and B a group holds in L2, which is
    the only difference between upstream's kernel 10 and kernel 11.
    """
    assert (GROUP_M, GROUP_N) == (16 // CLUSTER_M, 8 // CLUSTER_N)
    tm, tn = cluster_tiles(4096, 4096)
    assert (tm, tn) == (16, 16)
    assert (tm % GROUP_M, tn % GROUP_N) == (0, 0), "4096^3 must tile the supertile exactly"


def test_the_two_descriptor_layouts_are_different_and_deliberately_so() -> None:
    """A is K-major, B is MN-major, and reusing A's offsets for B is the bug that does not fault.

    torch hands this kernel ``a`` [M,K] and ``b`` [K,N], both row-major, so A is K-contiguous and B
    is N-contiguous, and TMA cannot transpose. Derivation of B's pair (LBO = the stride between
    64-element N chunks, which TMA laid out TILE_K rows apart; SBO = 8 K-rows of 128 B) is in the
    ``.cu`` header, cross-checked against ``cute::make_gmma_desc`` for ``Major::MN``.
    """
    src = _src()
    assert _define(src, "A_DESC_LBO_BYTES") == A_DESC_LBO_BYTES
    assert _define(src, "A_DESC_SBO_BYTES") == A_DESC_SBO_BYTES
    assert _define(src, "B_DESC_SBO_BYTES") == B_DESC_SBO_BYTES
    m = re.search(r"^#define B_DESC_LBO_BYTES \(BK \* (\d+)\)", src, re.MULTILINE)
    assert m, "B_DESC_LBO_BYTES must stay expressed in terms of BK, or a tile change silently lies"
    assert int(m.group(1)) * TILE_K == B_DESC_LBO_BYTES
    # cute GMMA::Major::K == 0, Major::MN == 1 (mma_sm90_gmma.hpp:107-110). The trans immediate and
    # the offsets are one contract: honouring only one of them runs, is fast, and is wrong.
    assert _define(src, "A_DESC_TRANS") == 0
    assert _define(src, "B_DESC_TRANS") == 1
    assert (A_DESC_LBO_BYTES, A_DESC_SBO_BYTES) != (B_DESC_LBO_BYTES, B_DESC_SBO_BYTES)


def test_both_descriptors_pack_into_the_fields_the_isa_defines() -> None:
    """Every offset must survive the ``(x & 0x3FFFF) >> 4`` packing without losing bits."""
    for lbo, sbo in (
        (A_DESC_LBO_BYTES, A_DESC_SBO_BYTES),
        (B_DESC_LBO_BYTES, B_DESC_SBO_BYTES),
    ):
        desc = wgmma_smem_descriptor(
            0, leading_byte_offset=lbo, stride_byte_offset=sbo, swizzle=SwizzleMode.B128
        )
        f = descriptor_fields(desc)
        assert f["leading_byte_offset"] == lbo >> 4, f"LBO {lbo} does not round-trip"
        assert f["stride_byte_offset"] == sbo >> 4, f"SBO {sbo} does not round-trip"
        assert f["layout_type"] == int(SwizzleMode.B128)
        assert f["base_offset"] == 0, "the tile base must be 1024 B aligned, i.e. swizzle phase 0"


# =============================================================================================
# CPU — budgets: shared memory, registers, and the warp specialisation that makes them fit
# =============================================================================================


def test_the_pipeline_needs_the_shared_memory_opt_in_and_the_launcher_makes_it() -> None:
    """3 stages of 128x64 + 64x256 bf16 is 144 KB — legal on Hopper, but only if the launch asks.

    This is the inverse of H-R1's budget test. There, staying under 48 KB was the point; here,
    going over it is, and the failure mode is a kernel ptxas emits happily and the runtime refuses
    at launch. The stages are *dynamic* shared memory, so the dry-dock's ptxas report shows 0 bytes
    smem for this kernel and cannot catch it either — this assertion is the only gate there is.
    """
    errs = check_smem_budget(
        bytes_per_stage=stage_bytes(),
        stages=QSIZE,
        arch=ARCH["sm_90a"],
        extra_bytes=barrier_bytes(),
    )
    assert len(errs) == 1 and errs[0].startswith("OPT-IN REQUIRED"), errs
    assert "cudaFuncSetAttribute" in _src(), (
        "shared memory is over 48 KB but the launcher never calls cudaFuncSetAttribute"
    )


def test_qsize_is_chosen_against_the_ceiling_and_leaves_one_stage_on_the_table() -> None:
    """Four stages fit Hopper; this rung runs three, and that is a knob, not a constraint.

    ``max_stages`` is the number the tile shape has to be chosen against. Recording it here means
    the D9 question "would a deeper pipeline hide more TMA latency?" has an answer that does not
    need a compile — and that a later tile change which quietly makes QSIZE illegal fails a test
    instead of a launch.
    """
    fit = max_stages(
        bytes_per_stage=stage_bytes(), arch=ARCH["sm_90a"], extra_bytes=barrier_bytes()
    )
    assert fit == 4, "the smem ceiling moved; re-derive QSIZE before trusting any number"
    assert fit > QSIZE, "QSIZE has no headroom left — say so in the spec if that becomes deliberate"
    over = check_smem_budget(bytes_per_stage=stage_bytes(), stages=fit + 1, arch=ARCH["sm_90a"])
    assert any("exceeds" in e for e in over), f"{fit + 1} stages should not fit 227 KB"


def test_warp_specialisation_is_what_makes_the_register_budget_close() -> None:
    """240 registers x 384 threads does not fit an SM. 24 + 2x240 does, and that is the whole
    reason ``setmaxnreg`` exists — the producer is one thread issuing TMA descriptors and gives its
    registers to the consumers, which hold 128 fp32 accumulators each."""
    assert check_setmaxnreg(PRODUCER_REGS) == []
    assert check_setmaxnreg(CONSUMER_REGS) == []
    arch = ARCH["sm_90a"]
    uniform = CONSUMER_REGS * THREADS_PER_CTA
    assert uniform > arch.regs_per_sm, (
        "if a uniform allocation fit, this rung would not need setmaxnreg and the test is stale"
    )
    specialised = PRODUCER_REGS * 128 + CONSUMER_REGS * (NUM_CONSUMER_WG * 128)
    assert specialised <= arch.regs_per_sm, (
        f"{specialised} registers over the {arch.regs_per_sm} an SM has — one CTA cannot be resident"
    )


# =============================================================================================
# CPU — TMA legality and which shapes this rung can actually run
# =============================================================================================


def _tma_maps(m: int, n: int, k: int) -> dict[str, list[str]]:
    """The two rank-3 tensor maps the launcher builds, checked against the driver's constraints.

    Rank 3 with a 64-element innermost dimension is not a style choice: the 128 B swizzle requires
    the innermost box to be exactly one atom wide, so the real extent has to move into an outer
    dimension. ``check_tma_tensor_map`` says the same thing in Python.
    """
    elem = 2
    return {
        "A": check_tma_tensor_map(  # [M, K] row-major, K contiguous
            rank=3,
            elem_bytes=elem,
            global_dims=[TMA_ATOM_ELEMS, m, k // TMA_ATOM_ELEMS],
            global_strides_bytes=[k * elem, TMA_ATOM_ELEMS * elem],
            box_dims=[TMA_ATOM_ELEMS, TILE_M, TILE_K // TMA_ATOM_ELEMS],
            swizzle=SwizzleMode.B128,
        ),
        "B": check_tma_tensor_map(  # [K, N] row-major, N contiguous
            rank=3,
            elem_bytes=elem,
            global_dims=[TMA_ATOM_ELEMS, k, n // TMA_ATOM_ELEMS],
            global_strides_bytes=[n * elem, TMA_ATOM_ELEMS * elem],
            box_dims=[TMA_ATOM_ELEMS, TILE_K, TILE_N // TMA_ATOM_ELEMS],
            swizzle=SwizzleMode.B128,
        ),
    }


def test_every_runnable_shape_encodes_a_legal_pair_of_tensor_maps() -> None:
    """The driver returns INVALID_VALUE for any tensor-map violation with no hint which one, on the
    host, at startup — so a bad tile shape appears as a kernel that never launches."""
    for name, (m, n, k) in RUNNABLE.items():
        assert shape_is_legal(m, n, k) == [], name
        for operand, errs in _tma_maps(m, n, k).items():
            assert errs == [], f"{name} operand {operand}: {errs}"
    m, n, k = WAVE_CLIFF
    assert shape_is_legal(m, n, k) == [], "the wave-cliff shape must be measurable on the box"


def test_npot_is_rejected_and_the_reason_is_tma_not_predication() -> None:
    """N=1023 cannot be described by a 128 B-swizzled tensor map at all. Adopting TMA cost this
    rung a shape H-R1 could run; the refusal is explicit so the loss is visible in the ledger
    rather than discovered as a launch failure on rented silicon."""
    m, n, k = SHAPES["npot"]
    errs = shape_is_legal(m, n, k)
    assert len(errs) == 1 and str(TMA_ATOM_ELEMS) in errs[0], errs
    assert "swizzle atom" in errs[0]


def test_which_predication_the_runnable_shapes_actually_exercise() -> None:
    """Honesty about coverage: M predication is tested, N predication is not.

    Every runnable N (4096, 8192, 6144) is a multiple of TILE_N, and the shapes that would not be
    are exactly the ones TMA rejects. So the epilogue's column bounds check is untested until a
    shape with N % 256 != 0 but N % 64 == 0 is added — 1280 in WAVE_CLIFF is the cheapest one, and
    it is why that shape earns its place beyond the wave arithmetic.
    """
    assert any(
        not tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["m"]
        for m, n, k in RUNNABLE.values()
    ), "no runnable shape exercises M predication"
    assert all(
        tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["n"]
        for m, n, k in RUNNABLE.values()
    ), "a runnable shape now exercises N predication — update this test and the spec"
    m, n, k = WAVE_CLIFF
    assert not tile_covers(m=m, n=n, k=k, tile_m=TILE_M, tile_n=TILE_N, tile_k=TILE_K)["n"]


def test_the_skinny_shape_leaves_half_a_cluster_with_no_tile_row() -> None:
    """M=16 is one cluster-tile row, and CLUSTER_M=2 means rank_m=1 owns a row that does not exist.

    That CTA must still participate: it receives the multicast B tile and it still owes its
    arrivals on both CTAs' ``empty`` barriers, and only its epilogue is fully predicated away.
    A cluster kernel that lets it return early hangs its peer, which is the specific failure this
    assertion exists to keep in view.
    """
    m, n, _ = SHAPES["skinny16"]
    tm, tn = cluster_tiles(m, n)
    assert tm == 1 and tn == n // (TILE_N * CLUSTER_N)
    assert math.ceil(m / TILE_M) < tm * CLUSTER_M, "rank_m=1 must have no CTA tile row here"


# =============================================================================================
# CPU — the rung's thesis: what a persistent grid removes, and what it does not
# =============================================================================================


def _non_persistent(m: int, n: int):  # noqa: ANN202
    """The grid H-R3 launches: one CTA per output tile."""
    return wave_quantization(m=m, n=n, tile_m=TILE_M, tile_n=TILE_N, sm_count=SM_COUNT)


def test_the_non_persistent_grid_has_a_wave_cliff_and_it_is_where_theory_says() -> None:
    """135 CTA-tiles on 132 SMs: 1.02 waves, and half the machine idles through the second one.

    ``WaveQuantization.efficiency`` is ``waves / ceil(waves)`` — the fraction of peak available from
    grid shape alone, before any mainloop tuning. At 1.02 waves it is 0.51, and no amount of
    pipelining recovers it, which is why the tile shape and the grid are chosen first.
    """
    m, n, _ = WAVE_CLIFF
    wq = _non_persistent(m, n)
    assert wq.ctas == 135 and wq.full_waves == 1 and wq.tail_ctas == 3
    assert 1.02 < wq.waves < 1.03
    assert 0.50 < wq.efficiency < 0.52


def test_the_persistent_grid_does_not_depend_on_the_problem_shape() -> None:
    """This is the literal content of "persistent", and the one thing it removes outright.

    The launch asks for the same 132 CTAs at every shape while the tile count swings from 64 to
    2048. The grid stops being a function of M and N, which is the precondition for a tile
    scheduler existing at all — and it is what the launcher computes from
    ``cudaDevAttrMultiProcessorCount``, with no reference to the problem.
    """
    grid = persistent_grid(SM_COUNT)
    assert grid == 132 and grid % CLUSTER_SIZE == 0
    tile_counts = {name: _non_persistent(m, n).ctas for name, (m, n, _) in RUNNABLE.items()}
    assert min(tile_counts.values()) < 100 < max(tile_counts.values()), (
        f"the shapes must actually differ in tile count for this to say anything: {tile_counts}"
    )


def test_persistence_pays_the_pipeline_prologue_once_per_sm_not_once_per_tile() -> None:
    """The quantifiable win, in units of cold pipeline fills.

    Every CTA starts by filling QSIZE stages with TMA loads and computing nothing while it waits.
    A non-persistent launch pays that once per tile; a persistent one pays it once per CTA, ever.
    At 4096^3 that is 512 prologues against 132 — a factor of 3.9, which is the ratio the ledger
    should see move when H-R3's number becomes H-R4's.
    """
    grid = persistent_grid(SM_COUNT)
    m, n, _ = SHAPES["sq4096"]
    wq = _non_persistent(m, n)
    assert wq.ctas == 512
    assert wq.ctas / grid == pytest.approx(wq.waves, rel=1e-9)
    assert wq.ctas / grid > 3.8, "the prologue saving at the headline shape"
    # ...and at the cliff shape there is almost none: 135 tiles, 132 CTAs. Persistence is not what
    # fixes that shape, which is the next test.
    cm, cn, _ = WAVE_CLIFF
    assert _non_persistent(cm, cn).ctas / grid < 1.05


def test_persistence_does_not_remove_the_tail_and_stream_k_is_the_thing_that_would() -> None:
    """The claim this rung must NOT make, asserted so nobody makes it by accident.

    For equal-cost tiles a persistent schedule still finishes in ``ceil(tiles / clusters)``
    tile-times: 70 cluster-tiles over 66 clusters is 2 rounds, so ~53% of peak from grid fit alone
    at the cliff shape — better than the non-persistent 51% only because the cluster-tile is
    coarser, not because persistence fixed anything. The residual is at most one tile of imbalance
    per cluster and the only construction that removes it is stream-K, which splits the k dimension
    so the tail wave has partial tiles to work on (``get_num_sk_tiles``,
    ``oss/cutlass/.../tile_scheduler_params.h:1066-1090``; plan §05 K1 lists it as optional here).

    If the D9 measurement at this shape comes in near 50%, that is this arithmetic, not a broken
    kernel — check the headline 4096^3 row before touching the mainloop.
    """
    clusters = persistent_grid(SM_COUNT) // CLUSTER_SIZE
    m, n, _ = WAVE_CLIFF
    tm, tn = cluster_tiles(m, n)
    tiles = tm * tn
    rounds = math.ceil(tiles / clusters)
    assert (clusters, tiles, rounds) == (66, 70, 2)
    assert tiles / (clusters * rounds) < 0.60, "the tail survives persistence"
    per_cluster = [tiles // clusters + (1 if i < tiles % clusters else 0) for i in range(clusters)]
    assert max(per_cluster) - min(per_cluster) <= 1, "the residual imbalance is one tile, no more"


def test_multicast_removes_a_third_of_the_clusters_hbm_traffic() -> None:
    """Why a cluster of 2 is worth the launch constraint it imposes.

    Both CTAs of the cluster want the same B tile (they split M, they share the N strip). Without
    multicast each fetches it: 2 A tiles + 2 B tiles per cluster per k-step. With multicast the B
    tile is read from HBM once and landed in both shared-memory windows: 2 A + 1 B. At
    128x256x64 that is 64 KB per k-step instead of 96 KB — a third of the cluster's read traffic,
    and the reason CLUSTER_M rather than CLUSTER_N is the dimension that is 2.
    """
    a_tile = TILE_M * TILE_K * 2
    b_tile = TILE_K * TILE_N * 2
    without = CLUSTER_SIZE * a_tile + CLUSTER_SIZE * b_tile
    with_mc = CLUSTER_SIZE * a_tile + b_tile
    assert without == 96 * 1024 and with_mc == 64 * 1024
    assert with_mc / without == pytest.approx(2 / 3)
    # The mask the kernel builds: every CTA in this CTA's N column, i.e. all CLUSTER_M of them.
    mask = 0
    for i in range(CLUSTER_M):
        mask |= 1 << (i * CLUSTER_N)
    assert mask == 0b11 and bin(mask).count("1") == CLUSTER_M


# =============================================================================================
# CPU — the oracle and the open-hole contract
# =============================================================================================


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


def test_wrapper_reports_an_open_hole_as_such() -> None:
    """ "You have not written the kernel" and "your kernel is broken" are different messages."""
    if not hole_is_open(SOURCE):
        pytest.skip("hole filled — the HoleOpenError path no longer applies")
    with pytest.raises((HoleOpenError, RuntimeError)) as exc:
        h_r4_gemm(
            torch.zeros(TILE_M, TILE_K, dtype=torch.bfloat16),
            torch.zeros(TILE_K, TILE_N, dtype=torch.bfloat16),
        )
    # On a CPU box require_arch fires first (RuntimeError); on a Hopper box the loader does.
    assert "h_r4_gemm" in str(exc.value) or "HUY hole" in str(exc.value)


# =============================================================================================
# drydock — what nvcc generated. No GPU, but the CUDA container must have run.
# =============================================================================================


def _artifact(kind: str) -> str:
    """One dry-dock artifact, named the way infra/drydock.sh writes it: ``<stem>.<arch>.<kind>``.

    Per SOURCE rather than per rung, because a rung can own more than one ``.cu`` and per-rung
    names let the last file compiled silently overwrite the evidence for the others.
    """
    p = _DRYDOCK / f"{SOURCE.removesuffix('.cu')}.sm_90a.{kind}.txt"
    if not p.is_file():
        pytest.skip(f"no {kind} report at {p} — run: infra/drydock.sh compile {SOURCE}")
    if hole_is_open(SOURCE):
        pytest.skip(
            "hole open — the captured artifact is the stub's, and says nothing about the kernel"
        )
    return p.read_text(encoding="utf-8", errors="replace")


@pytest.mark.drydock
def test_sass_issues_the_warpgroup_mma() -> None:
    """``HGMMA`` must appear, or the kernel is not using the tensor core this rung is about.

    A wgmma that assembled to nothing is the specific failure mode of forgetting the ``a`` in
    ``-arch=sm_90a``: everything compiles, the kernel runs, and it is silently a scalar loop.
    """
    sass = _artifact("sass")
    assert re.search(r"\bHGMMA\b", sass), (
        "no HGMMA in the SASS — either the mainloop does not issue wgmma, or it was compiled for "
        "base sm_90 instead of sm_90a (the trailing 'a' selects the accelerated ISA)"
    )


@pytest.mark.drydock
def test_sass_loads_through_tma_and_not_through_cp_async() -> None:
    """``UTMALDG`` is the TMA bulk load. ``LDGSTS`` is ``cp.async`` — H-R1's mechanism, and its
    presence here would mean the pipeline quietly fell back to the previous rung's."""
    sass = _artifact("sass")
    assert re.search(r"\bUTMALDG\b", sass), "no TMA load reached SASS"
    assert not re.search(r"\bLDGSTS\b", sass), (
        "cp.async in a TMA kernel — a staging path from H-R1 survived into the mainloop"
    )


@pytest.mark.drydock
def test_sass_has_no_local_memory_traffic() -> None:
    """``LDL``/``STL`` mean registers spilled to local memory — 128 fp32 accumulators did not fit."""
    sass = _artifact("sass")
    spills = re.findall(r"\b(LDL|STL)\b", sass)
    assert not spills, f"{len(spills)} local-memory accesses in the SASS: the accumulator spilled"


@pytest.mark.drydock
def test_ptxas_report_is_clean() -> None:
    """0 spill bytes. Shared memory is dynamic at this rung, so the report says 0 bytes smem and
    the real budget is asserted on the CPU side instead — see the opt-in test above."""
    report = _artifact("ptxas")
    spill = re.search(r"(\d+) bytes spill stores", report)
    assert spill and int(spill.group(1)) == 0, f"register spill: {report[-400:]}"


# =============================================================================================
# gpu — correctness against the oracle. H100 only.
# =============================================================================================


def _require_tolerance() -> float:
    if TOL_CONST is None:
        pytest.fail(
            "TOL_CONST is unset. The correctness gate cannot run without a tolerance, and picking "
            "one is the rung's numerics lesson: set it in this file and write the one-line argument "
            "into experiments/K1/H-R4/spec.md's 'Correctness gate' line before the first measured run."
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
        f"At this rung the three structured ways to be wrong are: a scheduler that visits a tile "
        f"twice or not at all (error concentrated on whole 128x256 blocks), a B descriptor read as "
        f"K-major (error everywhere, but exact on the first 64 columns), and a stage released "
        f"before its wgmma retired (error on one k-strip, so ~1/{k // 16} of the magnitude). "
        f"Print (out - ref).abs().amax(dim=1) reshaped to tiles before widening this."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("shape_name", list(RUNNABLE))
def test_matches_oracle(shape_name: str) -> None:
    """Every runnable spec shape against the fp32 oracle at the spec's tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m, n, k = RUNNABLE[shape_name]
    if shape_name in ("sq4096", "rect8192") and os.environ.get("LADDERS_SMOKE") == "1":
        pytest.skip("smoke mode: the large shapes are the measurement, not the smoke test")
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches_oracle(h_r4_gemm(a, b), a, b)
    del a, b
    torch.cuda.empty_cache()


@pytest.mark.gpu
def test_matches_oracle_at_the_wave_cliff_shape() -> None:
    """The shape the persistence arithmetic is calibrated on — and the only one whose N is not a
    multiple of TILE_N, so it is also the only test of the epilogue's column predication."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m, n, k = WAVE_CLIFF
    torch.manual_seed(2)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches_oracle(h_r4_gemm(a, b), a, b)


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3], ids=["tiny", "unit", "large"])
def test_matches_oracle_across_magnitudes(scale: float) -> None:
    """bf16 has 8 mantissa bits and a huge exponent range; the tolerance must be scale-relative."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(1)
    a = (torch.randn(512, 128, device="cuda", dtype=torch.float32) * scale).bfloat16()
    b = (torch.randn(128, 512, device="cuda", dtype=torch.float32) * scale).bfloat16()
    _assert_matches_oracle(h_r4_gemm(a, b), a, b)


@pytest.mark.gpu
def test_is_run_to_run_deterministic() -> None:
    """Same inputs, bit-identical output. A persistent kernel assigns tiles to CTAs by a fixed rule,
    so anything that varies between runs is a race on a stage or a barrier — and a race that only
    sometimes corrupts a k-strip will pass a single oracle check."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(3)
    a = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16)
    first = h_r4_gemm(a, b)
    for _ in range(4):
        assert torch.equal(first, h_r4_gemm(a, b)), "output varies between identical runs"


@pytest.mark.gpu
def test_rejects_bad_inputs() -> None:
    """The guard clauses, on the device where they can actually be reached."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    def z(r: int, c: int, dt: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(r, c, device="cuda", dtype=dt)

    with pytest.raises(TypeError, match="bfloat16"):
        h_r4_gemm(z(TILE_M, TILE_K, torch.float16), z(TILE_K, TILE_N))
    with pytest.raises(ValueError, match="inner dimensions"):
        h_r4_gemm(z(TILE_M, TILE_K), z(TILE_M, TILE_N))
    with pytest.raises(ValueError, match=f"multiple of TILE_K={TILE_K}"):
        h_r4_gemm(z(TILE_M, 32), z(32, TILE_N))
    with pytest.raises(ValueError, match=f"multiple of {TMA_ATOM_ELEMS}"):
        h_r4_gemm(z(TILE_M, TILE_K), z(TILE_K, 1023))


@pytest.mark.gpu
def test_handles_nan_and_inf_without_masking_them() -> None:
    """A NaN in must produce a NaN out — a kernel that swallows them hides real training bugs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    a = torch.ones(TILE_M, TILE_K, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(TILE_K, TILE_N, device="cuda", dtype=torch.bfloat16)
    a[0, 0] = float("nan")
    b[0, 1] = float("inf")
    out = h_r4_gemm(a, b)
    assert torch.isnan(out[0, 0]), "NaN was swallowed"
    assert not torch.isfinite(out[:, 1]).all(), "Inf was swallowed"
