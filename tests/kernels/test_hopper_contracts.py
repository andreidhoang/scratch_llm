"""The Hopper/Blackwell hardware contracts, checked without a GPU.

Every assertion here is a bug that would otherwise be found on rented silicon, and most of them do
not announce themselves when they happen: a descriptor field packed one bit off does not fault, it
reads the wrong core matrix; a swizzle that is not a bijection does not fault, it reads one element
twice; a tile that leaves 40% of the SMs idle on the last wave does not fault, it just never gets
past 60% of the floor no matter how good the mainloop is.

These run in the ordinary CPU suite, in milliseconds, on a laptop.
"""

from __future__ import annotations

import pytest

from scratch_llm.kernels.common.hopper_contracts import (
    ARCH,
    SMEM_BANKS,
    SwizzleMode,
    arch_for,
    bank_conflict_degree,
    bank_of,
    check_register_budget,
    check_setmaxnreg,
    check_smem_budget,
    check_tma_tensor_map,
    descriptor_encode,
    descriptor_fields,
    is_bijection,
    max_stages,
    swizzle_128b,
    tile_covers,
    wave_quantization,
    wgmma_smem_descriptor,
)

# =============================================================================================
# The 64-bit shared-memory matrix descriptor
# =============================================================================================


def test_descriptor_encode_is_the_ptx_packing() -> None:
    """``(x & 0x3FFFF) >> 4`` — mask 18 bits, drop the low 4 (16-byte granularity)."""
    assert descriptor_encode(0) == 0
    assert descriptor_encode(16) == 1
    assert descriptor_encode(1024) == 64
    # The mask is 18 bits wide, so anything at or above 2^18 wraps rather than overflowing into
    # the neighbouring field. That wrap is the hardware's, not ours — reproduce it, do not "fix" it.
    assert descriptor_encode(1 << 18) == 0
    assert descriptor_encode((1 << 18) + 32) == 2


def test_descriptor_matches_the_canonical_128b_ktile() -> None:
    """LBO=16 B, SBO=1024 B, 128B swizzle — the geometry every upstream Hopper GEMM uses.

    These exact constants appear in ``oss/fast.cu/h100/matmul/matmul_12.cuh::make_smem_desc`` and in
    CUTLASS's ``GmmaDescriptor``, and they are what ``csrc/gemm/h_r1_wgmma_bf16_sm90.cu`` builds.
    Two independent implementations agreeing is the only reason to trust a bit layout.
    """
    desc = wgmma_smem_descriptor(
        0, leading_byte_offset=16, stride_byte_offset=1024, swizzle=SwizzleMode.B128
    )
    f = descriptor_fields(desc)
    assert f["leading_byte_offset"] == 1
    assert f["stride_byte_offset"] == 64
    assert f["layout_type"] == int(SwizzleMode.B128) == 1
    assert f["base_offset"] == 0


def test_descriptor_fields_do_not_collide() -> None:
    """Each field must round-trip with every other field at its maximum — proves the shifts."""
    desc = wgmma_smem_descriptor(
        0x3FFF0,  # 14 bits of start address, once encoded
        leading_byte_offset=0x3FFF0,
        stride_byte_offset=0x3FFF0,
        base_offset=7,
        swizzle=SwizzleMode.B32,
    )
    f = descriptor_fields(desc)
    assert f["start_address"] == 0x3FFF
    assert f["leading_byte_offset"] == 0x3FFF
    assert f["stride_byte_offset"] == 0x3FFF
    assert f["base_offset"] == 7
    assert f["layout_type"] == 3


@pytest.mark.parametrize("bad", [1, 8, 15, 17, 1000])
def test_descriptor_rejects_unaligned_input(bad: int) -> None:
    """The ``>>4`` would silently discard the low bits; refuse instead of computing garbage."""
    with pytest.raises(ValueError, match="16-byte aligned"):
        wgmma_smem_descriptor(bad, leading_byte_offset=16, stride_byte_offset=1024)


def test_descriptor_rejects_out_of_range_base_offset() -> None:
    with pytest.raises(ValueError, match="3-bit swizzle phase"):
        wgmma_smem_descriptor(0, leading_byte_offset=16, stride_byte_offset=1024, base_offset=8)


# =============================================================================================
# Swizzle
# =============================================================================================


def test_swizzle_128b_is_a_bijection_over_its_atom() -> None:
    """A swizzle relocates; it must never alias. Checked exhaustively over the 1024-byte atom.

    (128 B x 8 rows = 1024 B is the span the XOR of bits [7,10) into [4,7) acts on.)
    """
    assert is_bijection(swizzle_128b, 1024, granularity=16)


def test_swizzle_128b_is_an_involution() -> None:
    """XOR-based swizzles are their own inverse — which is why one function serves both directions."""
    for off in range(0, 1024, 16):
        assert swizzle_128b(swizzle_128b(off)) == off


def test_swizzle_removes_the_bank_conflict_it_exists_to_remove() -> None:
    """The point of the 128B swizzle, stated as a measurement rather than as a claim.

    A K-major bf16 tile is 64 elements = 128 bytes per row. Eight lanes reading one 16-byte chunk
    from eight consecutive rows hit byte offsets 0, 128, 256, ... — every one of which is bank 0.
    That is an 8-way conflict and it serialises the load. The swizzle is what turns it into eight
    distinct banks.
    """
    row_bytes = 128
    unswizzled = [r * row_bytes for r in range(8)]
    swizzled = [swizzle_128b(o) for o in unswizzled]

    assert bank_conflict_degree(unswizzled) == 8, "the conflict this swizzle exists to remove"
    assert bank_conflict_degree(swizzled) == 1, "swizzled accesses must be conflict-free"
    assert len({bank_of(o) for o in swizzled}) == 8


def test_bank_of_wraps_at_32_banks() -> None:
    assert bank_of(0) == 0
    assert bank_of(4) == 1
    assert bank_of(SMEM_BANKS * 4) == 0
    assert bank_of(SMEM_BANKS * 4 + 4) == 1


def test_bank_conflict_degree_treats_a_broadcast_as_free() -> None:
    """All lanes reading the identical word is a broadcast in hardware, not a conflict."""
    assert bank_conflict_degree([0] * 32) == 1


# =============================================================================================
# TMA tensor maps
# =============================================================================================


def test_tma_accepts_the_fast_cu_tensor_map() -> None:
    """The exact map ``create_tensor_map`` builds in ``oss/fast.cu/h100/matmul/matmul_12.cuh``.

    bf16, 128B swizzle, innermost box 64 elements = 128 bytes: the innermost dimension is split at
    64 precisely so the box's innermost extent equals the swizzle atom. That split is the reason
    the map is rank-3 for a 2-D matrix, and it is the thing most first attempts get wrong.
    """
    errs = check_tma_tensor_map(
        rank=3,
        elem_bytes=2,
        global_dims=[64, 4096, 4096 // 64],
        global_strides_bytes=[2 * 4096, 64 * 2],
        box_dims=[64, 128, 1],
        swizzle=SwizzleMode.B128,
    )
    assert errs == [], errs


def test_tma_rejects_an_innermost_box_that_disagrees_with_the_swizzle() -> None:
    """128 B swizzle with a 256-byte innermost box: the driver returns INVALID_VALUE, no detail."""
    errs = check_tma_tensor_map(
        rank=2,
        elem_bytes=2,
        global_dims=[4096, 4096],
        global_strides_bytes=[2 * 4096],
        box_dims=[128, 128],  # 128 elems x 2 B = 256 B != 128 B atom
        swizzle=SwizzleMode.B128,
    )
    assert any("swizzle requires exactly" in e for e in errs), errs


def test_tma_rejects_box_over_256_and_unaligned_strides() -> None:
    errs = check_tma_tensor_map(
        rank=2,
        elem_bytes=2,
        global_dims=[4096, 4096],
        global_strides_bytes=[4098],  # not a multiple of 16
        box_dims=[64, 257],  # over the 256 cap
        swizzle=SwizzleMode.NONE,
    )
    assert any("boxDim[1]=257" in e for e in errs), errs
    assert any("not a multiple of 16 bytes" in e for e in errs), errs


# =============================================================================================
# Shared memory, registers, occupancy
# =============================================================================================


def test_h_r1_single_stage_tile_fits_without_the_opt_in() -> None:
    """K1/H-R1: 128x64 A + 128x64 B in bf16 = 32 KB, one stage — under the 48 KB opt-in threshold.

    The rung's kernel therefore needs no ``cudaFuncSetAttribute`` at launch. Asserting it here
    means the day a tile grows past 48 KB, a CPU test says so rather than a launch failing on a
    rented box.
    """
    bytes_per_stage = 2 * (128 * 64 * 2)  # A tile + B tile, bf16
    assert bytes_per_stage == 32768
    assert check_smem_budget(bytes_per_stage=bytes_per_stage, stages=1, arch=ARCH["sm_90a"]) == []


def test_smem_budget_flags_the_opt_in_and_the_hard_cap() -> None:
    sm90 = ARCH["sm_90a"]
    # 3 stages of 32 KB = 96 KB: legal, but only with the explicit opt-in.
    opt_in = check_smem_budget(bytes_per_stage=32768, stages=3, arch=sm90)
    assert len(opt_in) == 1 and opt_in[0].startswith("OPT-IN REQUIRED")
    assert "cudaFuncSetAttribute" in opt_in[0]
    # 8 stages of 32 KB = 256 KB: over the 227 KB per-CTA cap, and the message says how far.
    over = check_smem_budget(bytes_per_stage=32768, stages=8, arch=sm90)
    assert over and "exceeds" in over[0]
    assert "7 stages" in over[0], "the error must name the stage count that would fit"


def test_max_stages_is_the_number_a_pipeline_may_actually_use() -> None:
    """H-R3 picks its stage count from this, not from a round number."""
    assert max_stages(bytes_per_stage=32768, arch=ARCH["sm_90a"]) == 7
    # 1 KB of mbarriers and epilogue scratch does not change the answer here, but it can.
    assert max_stages(bytes_per_stage=32768, arch=ARCH["sm_90a"], extra_bytes=1024) == 7
    # sm_120 has 99 KB per CTA, not 227: the same tile gets 3 stages, not 7.
    assert max_stages(bytes_per_stage=32768, arch=ARCH["sm_120a"]) == 3


def test_arch_limits_do_not_confuse_per_sm_with_per_cta() -> None:
    """227 KB per CTA, 228 KB per SM. Budgeting against the larger number overflows at launch."""
    for name in ("sm_90a", "sm_100a"):
        a = ARCH[name]
        assert a.smem_per_cta == 227 * 1024
        assert a.smem_per_sm == 228 * 1024
        assert a.smem_per_cta < a.smem_per_sm
    sm120 = ARCH["sm_120a"]
    assert sm120.smem_per_cta == 99 * 1024 < ARCH["sm_90a"].smem_per_cta


def test_arch_for_resolves_by_compute_capability() -> None:
    assert arch_for((9, 0)).name == "sm_90a"
    assert arch_for((12, 0)).name == "sm_120a"
    with pytest.raises(KeyError):
        arch_for((8, 6))


def test_h_r1_register_budget() -> None:
    """128 accumulators + addressing on 128 threads: fits, and is why occupancy is 1 CTA/SM.

    2 wgmma issues x 64 fp32 accumulators = 128 registers of accumulator alone. At 128 threads that
    is 16384 registers before anything else — a quarter of the SM's file for the accumulator.
    """
    assert (
        check_register_budget(regs_per_thread=168, threads_per_cta=128, arch=ARCH["sm_90a"]) == []
    )
    over = check_register_budget(regs_per_thread=255, threads_per_cta=1024, arch=ARCH["sm_90a"])
    assert over and "cannot be resident" in over[-1]


@pytest.mark.parametrize("n", [24, 32, 152, 240, 256])
def test_setmaxnreg_accepts_legal_immediates(n: int) -> None:
    assert check_setmaxnreg(n) == []


@pytest.mark.parametrize("n", [16, 23, 100, 257, 264])
def test_setmaxnreg_rejects_illegal_immediates(n: int) -> None:
    """H-R3's warp specialisation depends on this instruction; an illegal immediate is silent."""
    assert check_setmaxnreg(n) != []


# =============================================================================================
# Wave quantization — the ceiling the tile shape sets before the mainloop is even written
# =============================================================================================


def test_wave_quantization_for_every_k1_spec_shape() -> None:
    """The five shapes K1 measures, at H-R1's 128x128 tile, on an H100's 132 SMs.

    4096^2 / 128^2 = 1024 CTAs over 132 SMs = 7.76 waves: the last wave is 78% full, so grid shape
    alone caps this at ~97% of peak. The skinny M=16 shape is the interesting one — it produces 32
    CTAs for 132 SMs, so three quarters of the machine is idle no matter what the kernel does, and
    any "% of cuBLAS" measured there is a statement about occupancy, not about the mainloop.
    """
    sm_count = 132  # H100 SXM
    shapes = {
        "sq4096": (4096, 4096),
        "rect8192": (8192, 8192),
        "skinny16": (16, 4096),
        "npot": (257, 1023),
        "untuned": (1536, 6144),
    }
    wq = {
        k: wave_quantization(m=m, n=n, tile_m=128, tile_n=128, sm_count=sm_count)
        for k, (m, n) in shapes.items()
    }

    assert wq["sq4096"].ctas == 1024
    assert wq["sq4096"].efficiency > 0.95, "the headline shape must not be grid-limited"

    # M=16 fills one tile row; 32 CTAs cannot fill 132 SMs even once.
    assert wq["skinny16"].ctas == 32
    assert wq["skinny16"].full_waves == 0
    assert wq["skinny16"].tail_utilization < 0.25

    # The non-power-of-two shape rounds up in both dimensions — 3x8 tiles for a 257x1023 problem.
    assert wq["npot"].ctas == 3 * 8


def test_efficiency_exposes_the_one_point_oh_two_wave_cliff() -> None:
    """1.02 waves is 51% efficiency: the classic persistent-kernel motivation, as arithmetic."""
    wq = wave_quantization(m=132 * 128 + 128, n=128, tile_m=128, tile_n=128, sm_count=132)
    assert wq.waves == pytest.approx(133 / 132, rel=1e-6)
    assert wq.efficiency < 0.52


def test_tile_covers_names_which_dimensions_need_predication() -> None:
    """The non-power-of-two shape must require predication, or it is not testing anything."""
    assert tile_covers(m=4096, n=4096, k=4096, tile_m=128, tile_n=128, tile_k=64) == {
        "m": True,
        "n": True,
        "k": True,
    }
    npot = tile_covers(m=257, n=1023, k=512, tile_m=128, tile_n=128, tile_k=64)
    assert npot == {"m": False, "n": False, "k": True}, (
        "the npot shape must exercise M and N predication while keeping K exact — H-R1 has no "
        "K-remainder path, so an inexact K would test the guard clause, not the epilogue"
    )
