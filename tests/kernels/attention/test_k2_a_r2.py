"""K2/A-R2 — FA3-shaped Hopper forward. Three tiers of gate, in ascending cost.

  CPU      the wrapper's contracts, the oracle, the warp geometry, every cuTensorMapEncodeTiled
           field, all three shared-memory descriptor layouts, the wgmma accumulator fragment map,
           and the causal trip count. Milliseconds.
  drydock  what nvcc actually generated: TMA in the loads, setmaxnreg in the role split, mbarriers
           ordering them, both wgmma forms and the named barriers assembling, nothing in local
           memory, no Ampere-style cp.async, no fences ptxas had to inject. Needs the CUDA
           container, still no GPU.
  gpu      correctness against the fp32 oracle at the spec shapes. Needs an H100.

The CPU tier carries more here than it did for a GEMM rung, because attention adds two failure
modes that do not fault and do not show up in a benchmark:

  1. The accumulator fragment map. Each thread owns two ROWS of the 64 x N tile, selected by
     ``(reg >> 1) & 1``. Using ``reg & 1`` — which selects the column — produces a kernel that runs
     at full speed and takes the softmax over the wrong axis. The pre-ladder skeleton does exactly
     that (``csrc/attention/fa3_hopper.cu:289``), which is why v2 exists and why the map is proved
     to be a bijection here rather than checked by eye.
  2. TMA's out-of-bounds zero fill. For a GEMM a zero-filled tail is free — a zero contributes
     nothing to a sum. For attention a zero SCORE is not minus infinity: it enters the softmax as a
     real key with weight ``exp(0)``. So the sequence-length rule is a correctness rule, not a
     convenience, and both ends of the stack have to say so.

Spec: experiments/K2/A-R2/spec.md   ·   Map: experiments/K2/A-R2/map.md
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest
import torch

from scratch_llm.kernels.attention.prefill.fa3_v2 import (
    ARCH as RUNG_ARCH,
)
from scratch_llm.kernels.attention.prefill.fa3_v2 import (
    BLOCK_M,
    BLOCK_N,
    BYTES_PER_STAGE,
    CONSUMER_REGS,
    D_CHUNKS,
    ELEM_BYTES,
    EXPECT_TX_BYTES_KV,
    HEAD_DIM,
    K_DESC_LBO,
    K_DESC_SBO,
    LAYOUTS,
    NUM_CONSUMER_THREADS,
    NUM_CONSUMER_WG,
    NUM_THREADS,
    PRODUCER_REGS,
    Q_DESC_LBO,
    Q_DESC_SBO,
    Q_TILE_BYTES,
    ROWS_PER_THREAD,
    RUNG,
    SMEM_BARRIER_BYTES,
    SMEM_DYNAMIC_BYTES,
    SOURCE,
    STAGES,
    SWIZZLE_ATOM_ELEMS,
    SYMBOL,
    V_DESC_LBO,
    V_DESC_SBO,
    WG_THREADS,
    WGMMA_K,
    WGMMA_M,
    HoleOpenError,
    acc_frag_col,
    acc_frag_row,
    acc_frag_row_slot,
    causal_coverage_gap,
    dims_for,
    fa3_hopper_v2_fwd,
    floor_unavailable_reason,
    hole_is_open,
    kv_blocks_for_tile,
    reference_attention,
    softmax_scale_log2,
    source_path,
    strides_elems,
    tensor_map_params,
    unsupported_reason,
)
from scratch_llm.kernels.common.hopper_contracts import (
    ARCH,
    SwizzleMode,
    check_register_budget,
    check_setmaxnreg,
    check_smem_budget,
    check_tma_tensor_map,
)

# workspace_root() locates the `ladders` directory, which cannot be found by walking up from a file
# in this repo: scratch_llm is a SYMLINK inside the workspace pointing at a sibling. The K1 loader
# already solves that, correctly and with the failure mode documented; a second copy here would be
# a second thing to get wrong, so this is the one cross-family import in the file.
from scratch_llm.kernels.gemm.cuda._k1_loader import workspace_root  # noqa: E402

_WORKSPACE = workspace_root()
_DRYDOCK = _WORKSPACE / "experiments" / "K2" / "A-R2" / "drydock"
#: drydock names its artifacts per SOURCE, not per rung — K2/A-R2 owns two .cu (the pre-ladder
#: skeleton and this one), and per-rung names would let one silently overwrite the other's evidence.
_STEM = SOURCE.removesuffix(".cu")

# -------------------------------------------------------------------------------------------------
# The tolerances. HUY SETS THESE, with their argument, before the first measured run.
#
# The gate is on O *and* on LSE, because that is what upstream gates: FlashInfer checks both against
# its sm80 kernel (oss/flashinfer/tests/attention/test_hopper.py:55-56) and CUTLASS's FMHA example
# checks both against a max-diff and a mean-diff threshold (88_hopper_fmha.cu:327,:334). An O-only
# gate would be weaker than the thing this rung is measured against.
#
# The form is  max|O_kernel - O_fp32| <= TOL_O  and  max|LSE_kernel - LSE_fp32| <= TOL_LSE, both
# absolute, because the fp32 oracle already normalises: O is a convex combination of V rows so its
# scale is V's, and LSE is a logarithm so its scale is O(log S). A ratio bound would be the wrong
# shape for both.
#
# Leaving them None is deliberate. Choosing them IS the rung's numerics lesson, and the argument has
# to name which error it is bounding — bf16 rounding of P, fp32 accumulation order over S keys, or
# the exp2 basis change — because those three grow differently with sequence length and the tolerance
# has to hold at s2048 and s8192 alike.
# -------------------------------------------------------------------------------------------------
TOL_O: float | None = None
TOL_LSE: float | None = None

#: The K2 shapes this rung runs, taken from bench/kernels/attention/k2_ladder.py's registry. The
#: decode shape is A-R3's and is not here: one query token is a different kernel, not a smaller one.
#: (batch, heads, kv_heads, seqlen, causal)
SHAPES: dict[str, tuple[int, int, int, int, bool]] = {
    "s2048": (8, 32, 32, 2048, True),
    "s4096": (4, 32, 32, 4096, True),  # the headline: the number quoted against FA3
    "s8192": (2, 32, 32, 8192, True),  # where causal block-skipping pays most
    "gqa8k": (4, 32, 4, 8192, True),  # GQA 8:1, the serving-shaped prefill
    "noncausal": (4, 32, 32, 4096, False),  # the mask factor, isolated
}

#: Small shapes for the correctness gate. The spec shapes are the measurement; running the oracle
#: (a Python-level tiled loop in fp32) at 8192 would take longer than the rung.
GATE_SHAPES: dict[str, tuple[int, int, int, int, bool]] = {
    "tiny": (1, 2, 2, BLOCK_M, True),  # exactly one k-block: no pipeline wrap at all
    "wrap": (1, 2, 2, BLOCK_M * (2 * STAGES + 1), True),  # more k-blocks than stages
    "gqa": (2, 8, 2, BLOCK_M * 3, True),  # 4:1 GQA
    "noncausal": (1, 2, 2, BLOCK_M * 2, False),
}


def _source() -> str:
    return source_path().read_text(encoding="utf-8")


# =================================================================================================
# The hole guard — exactly one per hole (tests/conftest.py turns it into a strict xfail while open)
# =================================================================================================


@pytest.mark.hole("K2/A-R2", "csrc/attention/fa3_hopper_v2.cu")
def test_a_r2_kernel_body_is_filled_and_compiles_clean() -> None:
    """Fails while the consumer mainloop is a ``#error``; passes when it compiles clean for sm_90a.

    The stdout assertion is not belt-and-braces: ``drydock.sh compile <name>`` filters its registry
    by substring, and a name with no matching row compiles zero files and exits 0. Without it this
    test would go green on an empty run the day the hole is filled but the registry row is not.
    """
    assert not hole_is_open(), (
        f"{RUNG} consumer mainloop is still an open hole. Read experiments/K2/A-R2/spec.md and "
        f"experiments/K2/A-R2/map.md, then write the online softmax, the causal mask and the "
        f"ping-pong handoff in csrc/attention/{SOURCE}."
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


# =================================================================================================
# CPU — the source and the wrapper agree, and the warp geometry closes
# =================================================================================================


def test_source_and_wrapper_agree_on_the_tile_and_the_warp_split() -> None:
    """The ``.cu``'s defines and the wrapper's constants must not drift.

    They are used for different things — one generates code, the other feeds the shared-memory,
    tensor-map, descriptor and fragment arithmetic below — so nothing else would notice if they
    diverged.
    """
    src = _source()
    for macro, expected in (
        ("BLOCK_M", BLOCK_M),
        ("BLOCK_N", BLOCK_N),
        ("HEAD_DIM", HEAD_DIM),
        ("STAGES", STAGES),
        ("WGMMA_M", WGMMA_M),
        ("WGMMA_K", WGMMA_K),
        ("SWZ_ELEMS", SWIZZLE_ATOM_ELEMS),
        ("PRODUCER_REGS", PRODUCER_REGS),
        ("CONSUMER_REGS", CONSUMER_REGS),
        ("Q_DESC_LBO", Q_DESC_LBO),
        ("Q_DESC_SBO", Q_DESC_SBO),
        ("K_DESC_LBO", K_DESC_LBO),
        ("K_DESC_SBO", K_DESC_SBO),
        ("V_DESC_SBO", V_DESC_SBO),
    ):
        m = re.search(rf"^#define {macro} (\d+)", src, re.MULTILINE)
        assert m, f"#define {macro} not found in csrc/attention/{SOURCE}"
        assert int(m.group(1)) == expected, (
            f"{macro}={m.group(1)} in the .cu but {expected} in fa3_v2.py"
        )


def test_the_consumer_count_is_a_consequence_of_block_m_not_a_choice() -> None:
    """One producer warp + two consumer warpgroups is what BLOCK_M=128 and wgmma M=64 imply.

    FlashInfer states the same dependency as a single expression,
    ``NUM_WARPS = ((CTA_Q/64)+1)*4`` (oss/flashinfer .../hopper/kernel_traits.cuh:62). Asserting it
    stops a later "let's try BLOCK_M=192" from changing the launch geometry silently.
    """
    assert NUM_CONSUMER_WG == BLOCK_M // WGMMA_M == 2
    assert NUM_CONSUMER_THREADS == NUM_CONSUMER_WG * WG_THREADS
    assert NUM_THREADS == (NUM_CONSUMER_WG + 1) * WG_THREADS
    src = _source()
    assert "__launch_bounds__(NUM_THREADS, 1)" in src, (
        "the kernel does not declare its launch bounds; setmaxnreg's budget is only meaningful "
        "against a known block size"
    )


def test_the_setmaxnreg_pair_is_legal_and_the_register_file_holds_it() -> None:
    """24/240 is not a preference, it is the largest pair that fits an SM's 65536 registers.

    ``NUM_CONSUMER_THREADS*240 + WG_THREADS*24 = 64512``. Raising the consumer figure to the next
    legal multiple of 8 overflows the file and the CTA cannot be resident at all — a launch failure
    on the box, for a constant that is checkable here. FA3 uses this exact pair for two MMA
    warpgroups with TMA KV (oss/flash-attention/hopper/flash_fwd_kernel_sm90.h:82-83).
    """
    assert check_setmaxnreg(PRODUCER_REGS) == []
    assert check_setmaxnreg(CONSUMER_REGS) == []
    arch = ARCH["sm_90a"]
    total = NUM_CONSUMER_THREADS * CONSUMER_REGS + WG_THREADS * PRODUCER_REGS
    one_step_up = NUM_CONSUMER_THREADS * (CONSUMER_REGS + 8) + WG_THREADS * PRODUCER_REGS
    assert total <= arch.regs_per_sm, f"{total} registers > {arch.regs_per_sm} per SM"
    assert one_step_up > arch.regs_per_sm, (
        "CONSUMER_REGS is not at the ceiling — either the budget moved or this test is stale"
    )
    assert (
        check_register_budget(regs_per_thread=CONSUMER_REGS, threads_per_cta=WG_THREADS, arch=arch)
        == []
    )


# =================================================================================================
# CPU — the tensor maps
# =================================================================================================


def test_head_dim_needs_two_boxes_because_of_the_swizzle_atom() -> None:
    """The rule that shapes every tile in this kernel, as an assertion rather than a comment.

    Under 128B swizzle the innermost box extent must be exactly 128 B. HEAD_DIM bf16 is 256 B, so
    the obvious single box is illegal — and the driver would say only CUDA_ERROR_INVALID_VALUE, on
    the host, at startup, naming no field. Two boxes of SWIZZLE_ATOM_ELEMS are the split that fixes
    it, which is why every tile is stored as two 64-wide head-dim slabs.
    """
    errs = check_tma_tensor_map(
        rank=4,
        elem_bytes=ELEM_BYTES,
        global_dims=[HEAD_DIM, 4096, 32, 4],
        global_strides_bytes=[HEAD_DIM * ELEM_BYTES, 4096 * HEAD_DIM * ELEM_BYTES, 1 << 24],
        box_dims=[HEAD_DIM, BLOCK_N, 1, 1],
        swizzle=SwizzleMode.B128,
    )
    assert any("swizzle requires exactly 128 B" in e for e in errs), errs
    assert D_CHUNKS * SWIZZLE_ATOM_ELEMS == HEAD_DIM
    assert SwizzleMode.B128.atom_bytes == SWIZZLE_ATOM_ELEMS * ELEM_BYTES


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_all_three_tensor_maps_are_legal_at_every_spec_shape(shape_name: str, layout: str) -> None:
    """Every cuTensorMapEncodeTiled field, checked by name, for both layouts, before a rented hour.

    Both layouts matter: SDPA hands attention ``[B,H,S,D]`` and ``flash_attn_func`` hands it
    ``[B,S,H,D]``. If only one were legal the benchmark would have to transpose one side of the
    comparison inside the measured region, and the percentage would be against a memcpy.
    """
    b, h, kv_h, s, _ = SHAPES[shape_name]
    params = tensor_map_params(batch=b, heads=h, kv_heads=kv_h, seqlen=s, layout=layout)
    for operand, p in params.items():
        assert check_tma_tensor_map(**p) == [], f"{shape_name}/{layout}/{operand}"  # type: ignore[arg-type]


def test_the_map_is_rank_four_so_a_sequence_tail_is_out_of_bounds_not_a_neighbour() -> None:
    """Why the maps are not rank 2 over a flattened ``(B*H*S, D)``.

    With the sequence collapsed into the row index, the rows past the end of one head are the first
    rows of the NEXT head — in bounds, so TMA reads them happily instead of bounds-checking. The
    rank-4 map makes the sequence its own dimension, which is what makes the head and batch
    coordinates indices rather than arithmetic on a flat row.
    """
    p = tensor_map_params(batch=4, heads=32, kv_heads=32, seqlen=4096)["k"]
    assert p["rank"] == 4
    assert p["global_dims"] == [HEAD_DIM, 4096, 32, 4]
    assert len(p["global_strides_bytes"]) == 3  # type: ignore[arg-type]
    assert p["box_dims"] == [SWIZZLE_ATOM_ELEMS, BLOCK_N, 1, 1]
    assert "cp.async.bulk.tensor.4d" in _source()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_dims_for_recovers_the_shape_in_both_layouts(shape_name: str, layout: str) -> None:
    """The layout is a parameter, and this is the test that says it cannot be anything else.

    A contiguous ``[B,H,S,D]`` and a contiguous ``[B,S,H,D]`` have the SAME stride pattern — dim 3
    is 1, dim 2 is D, dims 1 and 0 decrease — so the tempting "read the head axis off the strides"
    predicate is true for both and silently swaps heads for sequence. That does not fault: it makes
    the shape checks reject a valid tensor, or (when S == H) accept it and compute nonsense. Both
    are invisible on this Mac, where ``require_arch`` raises before the layout code runs, which is
    exactly why this is a pure function with its own test.
    """
    b, h, kv_h, s, _ = SHAPES[shape_name]
    q_shape = (b, s, h, HEAD_DIM) if layout == "bshd" else (b, h, s, HEAD_DIM)
    k_shape = (b, s, kv_h, HEAD_DIM) if layout == "bshd" else (b, kv_h, s, HEAD_DIM)
    assert dims_for(q_shape, layout) == (b, h, s, HEAD_DIM)
    assert dims_for(k_shape, layout) == (b, kv_h, s, HEAD_DIM)
    # and the two contiguous layouts really are indistinguishable by stride pattern
    q = torch.empty(q_shape, dtype=torch.bfloat16)
    other = torch.empty(
        (b, h, s, HEAD_DIM) if layout == "bshd" else (b, s, h, HEAD_DIM), dtype=torch.bfloat16
    )
    assert q.stride(3) == other.stride(3) == 1
    assert q.stride(2) == other.stride(2) == HEAD_DIM


def test_both_layouts_give_the_head_dim_a_unit_stride_and_differ_only_in_two_strides() -> None:
    """The one property that lets a single tensor-map shape describe both layouts."""
    b, h, s = 4, 32, 4096
    bhsd = strides_elems("bhsd", batch=b, heads=h, seqlen=s)
    bshd = strides_elems("bshd", batch=b, heads=h, seqlen=s)
    assert bhsd == (h * s * HEAD_DIM, s * HEAD_DIM, HEAD_DIM)
    assert bshd == (s * h * HEAD_DIM, HEAD_DIM, h * HEAD_DIM)
    assert bhsd[0] == bshd[0], "the batch stride is the whole plane in both layouts"


# =================================================================================================
# CPU — shared memory and the pipeline arithmetic
# =================================================================================================


def test_the_pipeline_needs_the_dynamic_smem_opt_in_and_the_launcher_makes_it() -> None:
    """160 KB is legal on Hopper and NOT free: past 48 KB the launch must opt in.

    ptxas emits the kernel either way and the failure appears only as a launch error at runtime. The
    dry dock cannot catch this one — its shared-memory gate reads the ptxas report, which counts
    static shared memory and says nothing about a dynamic allocation — so this assertion is the real
    gate. The Q tile is ``extra_bytes`` rather than part of a stage because it is loaded once and
    stays resident for the whole k-loop.
    """
    errs = check_smem_budget(
        bytes_per_stage=BYTES_PER_STAGE,
        stages=STAGES,
        arch=ARCH["sm_90a"],
        extra_bytes=Q_TILE_BYTES + SMEM_BARRIER_BYTES,
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


def test_expect_tx_equals_what_the_two_tma_boxes_deliver() -> None:
    """The mbarrier counts bytes, not arrivals. Too many hangs; too few reads a half-written tile."""
    assert D_CHUNKS * (SWIZZLE_ATOM_ELEMS * BLOCK_N * ELEM_BYTES) == EXPECT_TX_BYTES_KV
    assert SMEM_DYNAMIC_BYTES == Q_TILE_BYTES + STAGES * BYTES_PER_STAGE
    assert BYTES_PER_STAGE == 2 * EXPECT_TX_BYTES_KV, "a stage is one K tile plus one V tile"


def test_k_and_v_have_separate_barriers_so_k_can_retire_early() -> None:
    """The reason there are four barrier arrays and not two.

    A single per-stage barrier would hold K hostage to the PV wgmma, which is still reading V long
    after QK has retired — halving the pipeline's effective depth for nothing. FA3 gives K and V
    their own pipeline objects for exactly this (oss/flash-attention/hopper/
    flash_fwd_kernel_sm90.h:226-263), and FlashInfer's steady-state loop releases K at
    mainloop_mma.cuh:243 and V only at :268.
    """
    src = _source()
    for name in ("bar_k_full", "bar_k_empty", "bar_v_full", "bar_v_empty", "bar_q"):
        assert re.search(rf"__shared__ __align__\(8\) uint64_t {name}", src), (
            f"{name} is not declared in csrc/attention/{SOURCE}"
        )
    assert SMEM_BARRIER_BYTES == 8 * (4 * STAGES + 1)


def test_the_producer_and_consumer_phase_parities_are_complements() -> None:
    """The one line of this kernel that hangs rather than fails.

    CUTLASS: "Producer starts with an opposite phase as the buffers are initially empty" —
    ``InitialProducerPhase = 1``, flipped on index wrap (oss/cutlass/include/cutlass/pipeline/
    sm90_pipeline.hpp:254-260, :204-213). So the producer waits on ``((kt/STAGES)&1)^1`` and the
    consumer on ``(kt/STAGES)&1``. Getting the initial value wrong deadlocks on iteration 0, which
    at least fails loudly; getting the FLIP wrong deadlocks on iteration STAGES, after the kernel
    has looked healthy for a while.
    """
    src = _source()
    assert "((kt / STAGES) & 1) ^ 1u" in src, (
        "the producer's empty-barrier parity is not the complement of the consumer's — see "
        "oss/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:254-260"
    )
    assert "(kt / STAGES) & 1u" in src, "the consumer's full-barrier parity is missing"


# =================================================================================================
# CPU — the three shared-memory descriptors describe the layout TMA actually writes
#
# This is the assertion that would otherwise cost a rented hour. A descriptor whose LBO/SBO do not
# match the smem layout does not fault: the MMA reads the wrong core matrix and returns a plausible
# O. The two canonical forms are CUTLASS make_gmma_desc's, quoted at
# oss/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:186-296, restated here in bytes.
# =================================================================================================


def _tma_offset_k_major(row: int, d: int, rows: int) -> int:
    """Byte offset TMA writes for element ``(row, d)`` of a Q or K tile.

    Two boxes ``{SWIZZLE_ATOM_ELEMS, rows}``, one per 64-wide head-dim slab; within a slab the head
    dim is contiguous.
    """
    slab, inner = divmod(d, SWIZZLE_ATOM_ELEMS)
    return (inner + SWIZZLE_ATOM_ELEMS * row) * ELEM_BYTES + slab * (
        SWIZZLE_ATOM_ELEMS * rows * ELEM_BYTES
    )


@pytest.mark.parametrize(
    ("name", "rows", "lbo", "sbo"),
    [("q", BLOCK_M, Q_DESC_LBO, Q_DESC_SBO), ("k", BLOCK_N, K_DESC_LBO, K_DESC_SBO)],
)
def test_q_and_k_descriptor_walks_match_the_tma_layout(
    name: str, rows: int, lbo: int, sbo: int
) -> None:
    """Major::K, B128: ((8,m),(T,2)):((8T,SBO),(1,LBO)).

    Both operands of ``S = Q·K^T`` are K-major, because the contracted dimension is the head dim
    and the head dim is contiguous in both Q and K. That is also why the QK wgmma's trans-a and
    trans-b immediates are both 0 — unlike K1/H-R2's GEMM, where B arrives MN-major and trans-b is 1.
    FA3 gets there through the type system: ``ss_op_selector`` with its default ``Major::K`` for both
    (oss/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:93).
    """
    for row in range(rows):
        for d in range(HEAD_DIM):
            slab, inner = divmod(d, SWIZZLE_ATOM_ELEMS)
            walk = (
                (row % 8) * 128
                + (row // 8) * sbo
                + (inner // 8) * lbo
                + (inner % 8) * ELEM_BYTES
                + slab * (SWIZZLE_ATOM_ELEMS * rows * ELEM_BYTES)
            )
            assert walk == _tma_offset_k_major(row, d, rows), (
                f"{name} descriptor disagrees with TMA at (row={row}, d={d})"
            )


def test_v_descriptor_walk_matches_the_tma_layout() -> None:
    """Major::MN, B128: ((T,8,n),(8,k)):((1,T,LBO),(8T,SBO)).

    V is the odd one out and the reason is worth stating: in ``O += P·V`` the CONTRACTED dimension
    is the key index and the free one is the head dim — and it is the head dim that is contiguous.
    So V is an MN-major B operand, its LBO is the stride between the two 64-wide head-dim slabs, and
    the PV wgmma's trans-b immediate is 1 while the QK one's is 0. FA3 says the same thing in one
    line: ``MmaMajorV`` is ``GMMA::Major::MN`` for 16-bit V
    (oss/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:65). Swapping LBO and SBO is the
    single most common descriptor bug and it does not fault.
    """
    for d in range(HEAD_DIM):
        for kv in range(BLOCK_N):
            walk = (
                (d % 8) * ELEM_BYTES
                + ((d // 8) % 8) * 16
                + (d // SWIZZLE_ATOM_ELEMS) * V_DESC_LBO
                + (kv % 8) * 128
                + (kv // 8) * V_DESC_SBO
            )
            assert walk == _tma_offset_k_major(kv, d, BLOCK_N), (
                f"V descriptor disagrees with TMA at (d={d}, kv={kv})"
            )


def test_v_does_not_share_a_descriptor_layout_with_q_and_k() -> None:
    """The kernel's one real asymmetry, asserted so a later "simplification" cannot erase it."""
    assert (Q_DESC_LBO, Q_DESC_SBO) == (K_DESC_LBO, K_DESC_SBO)
    assert (V_DESC_LBO, V_DESC_SBO) != (Q_DESC_LBO, Q_DESC_SBO)
    assert V_DESC_LBO == SWIZZLE_ATOM_ELEMS * BLOCK_N * ELEM_BYTES


def test_every_descriptor_base_this_kernel_builds_is_swizzle_phase_zero() -> None:
    """Why ``base_offset`` is 0 everywhere, checked rather than assumed.

    A tile base at a multiple of the 1024-byte swizzle repeat has phase 0. Every base here is one:
    the tile bases (Q, and each stage's K and V), the consumer warpgroup's 64-row offset into Q
    (8192 B), the head-dim slab stride (16384 B) and the PV k-strip stride (2048 B). The single
    exception is the QK k-strip advance along the contiguous head dim, which moves 32 B at a time —
    exactly the case K1/H-R2 is in, where fast.cu leaves base_offset at 0 and is verified against
    cuBLAS (oss/fast.cu/h100/matmul/matmul_2.cuh:9-17, matmul.cu:192).
    """
    repeat = 1024
    bases = {
        "Q tile": 0,
        "K stage 1": Q_TILE_BYTES + BYTES_PER_STAGE,
        "V stage 0": Q_TILE_BYTES + EXPECT_TX_BYTES_KV,
        "warpgroup 1 Q rows": WGMMA_M * SWIZZLE_ATOM_ELEMS * ELEM_BYTES,
        "head-dim slab": SWIZZLE_ATOM_ELEMS * BLOCK_N * ELEM_BYTES,
        "PV k-strip": WGMMA_K * SWIZZLE_ATOM_ELEMS * ELEM_BYTES,
    }
    for what, base in bases.items():
        assert base % repeat == 0, f"{what} is at {base}, not a multiple of the {repeat} B repeat"


def test_the_two_wgmma_forms_carry_the_immediates_their_operands_require() -> None:
    """SS for ``S = Q·K^T``, RS for ``O += P·V`` — and the RS form has one immediate FEWER.

    This is the structural difference between an attention mainloop and a GEMM mainloop. The SS
    form's tail is ``scale-a, scale-b, trans-a, trans-b``; the RS form's is
    ``scale-d(pred), scale-a, scale-b, trans-b`` — there is no trans-a, because a register A-operand
    has exactly one legal layout and CUTLASS enforces it with a static_assert rather than an option
    ("Register source operand A must have K major layout",
    oss/cutlass/include/cute/arch/mma_sm90_gmma.hpp:2896-2897). Cloning the SS asm and adding a
    register operand list compiles and is wrong.
    """
    src = _source()
    ss = re.search(r'"%64, %65, %66, (\d), (\d), (\d), (\d);', src)
    assert ss, "the SS-form wgmma immediates were not found"
    scale_a, scale_b, trans_a, trans_b = (int(g) for g in ss.groups())
    assert (scale_a, scale_b) == (1, 1)
    assert trans_a == 0, "Q is K-major (GMMA::Major::K = 0)"
    assert trans_b == 0, "K is K-major too — the head dim is contiguous in both operands"

    rs = re.search(r'"%68, p, (\d), (\d), (\d);', src)
    assert rs, "the RS-form wgmma immediates were not found"
    rs_scale_a, rs_scale_b, rs_trans_b = (int(g) for g in rs.groups())
    assert (rs_scale_a, rs_scale_b) == (1, 1)
    assert rs_trans_b == 1, (
        "V reaches shared memory head-dim-contiguous, so the PV wgmma's trans-b must be 1 "
        "(GMMA::Major::MN). Copying the QK gemm's 0 compiles, runs, and returns a plausible wrong O."
    )
    assert '"{%64,%65,%66,%67}, "' in src, (
        "the RS form does not take a four-register A operand — P is going through shared memory, "
        "which is the round trip this rung exists to remove"
    )


# =================================================================================================
# CPU — the wgmma accumulator fragment map
# =================================================================================================


@pytest.mark.parametrize("n", [BLOCK_N, HEAD_DIM])
def test_the_fragment_map_is_a_bijection_over_the_whole_tile(n: int) -> None:
    """Every (row, col) of the 64 x N accumulator is owned by exactly one (thread, register).

    Both accumulators are checked because they are different tiles: S is 64 x BLOCK_N and O is
    64 x HEAD_DIM. A map that is not a bijection means two registers alias one output element, which
    shows up as a wrong answer in some columns and no error anywhere.
    """
    seen: dict[tuple[int, int], tuple[int, int, int]] = {}
    for warp in range(4):
        for lane in range(32):
            for reg in range(n // 2):
                cell = (acc_frag_row(reg, lane, warp), acc_frag_col(reg, lane))
                assert cell not in seen, f"{cell} owned twice: {seen[cell]} and {(warp, lane, reg)}"
                seen[cell] = (warp, lane, reg)
    assert len(seen) == WGMMA_M * n
    assert {r for r, _ in seen} == set(range(WGMMA_M))
    assert {c for _, c in seen} == set(range(n))


def test_each_thread_owns_exactly_two_rows_and_the_selector_is_bit_one() -> None:
    """The bug that makes a fast, wrong attention kernel.

    ``(reg >> 1) & 1`` selects the ROW; ``reg & 1`` selects the column. The pre-ladder skeleton uses
    ``i & 1`` for the row (csrc/attention/fa3_hopper.cu:289, :310, :317), which takes the running
    max, the rescale and the denominator over the wrong axis. Nothing about that fails loudly.
    """
    for warp in range(4):
        for lane in range(32):
            rows = {acc_frag_row(reg, lane, warp) for reg in range(BLOCK_N // 2)}
            assert len(rows) == ROWS_PER_THREAD, f"warp {warp} lane {lane} owns {sorted(rows)}"
            for reg in range(BLOCK_N // 2):
                slot = acc_frag_row_slot(reg)
                assert acc_frag_row(reg, lane, warp) == sorted(rows)[slot]
            # the column selector must NOT be the row selector
            assert acc_frag_row(0, lane, warp) == acc_frag_row(1, lane, warp)
            assert acc_frag_col(0, lane) != acc_frag_col(1, lane)


def test_a_quad_shares_its_rows_which_is_why_the_reduction_is_shfl_xor_1_and_2() -> None:
    """The row reduction's width, derived rather than copied.

    The four lanes of a quad (``lane % 4``) hold the same two rows and different columns; no wider
    group does. So the row max reduces with ``__shfl_xor_sync`` at offsets 1 and 2 and nothing else
    — FlashInfer's ``quad_allreduce_`` (oss/flashinfer .../hopper/attention_updater.cuh:230). A
    full-warp reduction would silently mix four different rows.
    """
    for warp in range(4):
        for quad in range(8):
            rows = {
                acc_frag_row(reg, 4 * quad + lane, warp)
                for lane in range(4)
                for reg in range(BLOCK_N // 2)
            }
            assert len(rows) == ROWS_PER_THREAD, f"quad {quad} of warp {warp} spans {sorted(rows)}"
        # the next quad up holds different rows, so a shfl at offset 4 would mix them
        assert acc_frag_row(0, 0, warp) != acc_frag_row(0, 4, warp)


def test_p_relayout_is_a_pure_view_change_with_no_data_movement() -> None:
    """The A-fragment of PV k-strip ``j`` is exactly S accumulator registers ``[8j, 8j+8)``.

    ``ALayout_64x16`` (mma_traits_sm90_gmma.hpp:451-452) and ``CLayout_64xN`` (:432-434) share their
    thread mode byte for byte and their value modes agree once N is truncated to 16. So the register
    that holds ``S[row][16j + c]`` is the register that must hold ``A[row][c]`` for strip ``j``, in
    the same thread, in the same order. That is why FA3 can call ``convert_layout_acc_Aregs``
    (mainloop_fwd_sm90_tma_gmma_ws.hpp:1157) — a relayout of a view, not a copy — and it is what
    makes the RS form worth its extra instruction.

    Also asserted: registers ``2i`` and ``2i+1`` are adjacent along k, so they pack into one
    ``__nv_bfloat162`` with the even index in the low half.
    """
    strips = BLOCK_N // WGMMA_K
    for lane in range(32):
        for j in range(strips):
            cols = [acc_frag_col(8 * j + v, lane) for v in range(8)]
            assert all(WGMMA_K * j <= c < WGMMA_K * (j + 1) for c in cols), (
                f"strip {j} of lane {lane} reaches outside its 16-wide k window: {cols}"
            )
            for i in range(4):
                lo, hi = cols[2 * i], cols[2 * i + 1]
                assert hi == lo + 1, (
                    f"registers {8 * j + 2 * i}/{8 * j + 2 * i + 1} are not adjacent"
                )
    # A quad's eight registers per strip, across its four lanes, cover that strip's 16 columns —
    # each exactly ROWS_PER_THREAD times, once for each row a thread owns. That is what makes the
    # four uint32 a complete A-operand for both of the thread's rows and not a slice of one.
    for quad in range(8):
        for j in range(strips):
            cols = [acc_frag_col(8 * j + v, 4 * quad + lane) for lane in range(4) for v in range(8)]
            want = list(range(WGMMA_K * j, WGMMA_K * (j + 1))) * ROWS_PER_THREAD
            assert sorted(cols) == sorted(want), (
                f"strip {j} of quad {quad} does not tile its 16 columns: {sorted(cols)}"
            )
            for v in range(8):
                rows = {acc_frag_row(8 * j + v, 4 * quad + lane, 0) for lane in range(4)}
                assert len(rows) == 1, f"register {v} of strip {j} spans rows {rows} in one quad"


# =================================================================================================
# CPU — the causal trip count, the shape rules, and the floor
# =================================================================================================


@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_the_causal_k_loop_loads_every_key_the_mask_admits(shape_name: str) -> None:
    """An off-by-one that loads too few blocks does not fault and does not hang.

    It silently drops the tail of each row's attention and returns a plausible O — the one causal
    bug an oracle catches and a benchmark never does. Checked exhaustively over every query tile of
    every spec shape, on a laptop.
    """
    _, _, _, s, causal = SHAPES[shape_name]
    if not causal:
        assert kv_blocks_for_tile(0, s, causal=False) == s // BLOCK_N
        return
    assert causal_coverage_gap(s) is None
    # ... and it must not load blocks that are entirely in the future, which is where causal
    # block-skipping actually pays: the first tile reads one block, not S/BLOCK_N of them.
    assert kv_blocks_for_tile(0, s) == 1
    assert kv_blocks_for_tile(s // BLOCK_M - 1, s) == s // BLOCK_N


def test_the_trip_count_lives_in_one_place_in_the_cu() -> None:
    """The producer and both consumers must read the SAME ``n_kv_blocks``.

    Two derivations that disagree by one is not a wrong answer, it is a hang: the producer waits on
    an empty barrier nobody will signal, or a consumer on a full barrier nobody will fill. So the
    ``.cu`` computes it once, before the warp-role split, and :func:`kv_blocks_for_tile` mirrors that
    one expression rather than re-deriving it.
    """
    src = _source()
    assert "(q_row0 + BLOCK_M - 1) / BLOCK_N + 1" in src, (
        "the causal trip count in the .cu no longer matches kv_blocks_for_tile"
    )
    assert src.count("const int n_kv_blocks") == 1, (
        "n_kv_blocks is computed more than once — the producer and the consumers can now drift"
    )
    assert src.index("const int n_kv_blocks") < src.index("if (tid < WG_THREADS)"), (
        "n_kv_blocks is computed after the warp-role split, so the two roles derive it separately"
    )


def test_a_kv_tail_is_refused_rather_than_zero_filled() -> None:
    """Where attention differs from a GEMM, stated by the wrapper and not only by a comment.

    TMA zero-fills out-of-bounds reads. In K1/H-R2 that is a free tail — a zero contributes nothing
    to a sum. Here a zero-filled K row yields a SCORE of zero, and ``exp(0) = 1``: the phantom key
    gets more weight than most real ones. So the rung refuses the shape.
    """
    why = unsupported_reason(heads=32, kv_heads=32, seqlen=BLOCK_M + 8, head_dim=HEAD_DIM)
    assert why is not None and "-inf" in why and "softmax" in why


def test_the_other_shape_rules_name_themselves() -> None:
    """Each refusal must say which constraint it belongs to — hardware, mask, or this rung's scope."""
    assert unsupported_reason(heads=32, kv_heads=32, seqlen=4096, head_dim=HEAD_DIM) is None
    assert "head dim" in (unsupported_reason(heads=32, kv_heads=32, seqlen=4096, head_dim=64) or "")
    assert "GQA" in (unsupported_reason(heads=32, kv_heads=5, seqlen=4096, head_dim=HEAD_DIM) or "")
    assert "S_q must equal S_kv" in (
        unsupported_reason(heads=32, kv_heads=32, seqlen=4096, head_dim=HEAD_DIM, seqlen_kv=2048)
        or ""
    )


def test_the_floor_is_fa3_and_its_absence_is_named_not_silently_replaced() -> None:
    """The invariant that keeps this rung's percentage meaning what the plan says it means.

    FA3 has no CPU build and is Hopper-only, so on this Mac the floor is simply unavailable — and
    the rung must say so. Falling back to FA2 or SDPA would not be a weaker version of ">= 60% of
    FA3"; it would be a different, roughly 1.5x easier claim, and nothing downstream of the number
    would record which floor produced it.
    """
    why = floor_unavailable_reason()
    assert why is not None, (
        "FA3 appears importable on this box — then this test is on the wrong host"
    )
    assert "flash_attn_interface" in why or "sm_90a" in why
    assert "FA2" in why or "SDPA" in why or "sm_90a" in why


def test_the_softmax_scale_is_carried_in_the_log2_basis() -> None:
    """One multiply is folded into the host constant so every exponential is a bare ``exp2``.

    FA3 carries the scale the same way (``softmax_scale_log2``,
    oss/flash-attention/hopper/flash_fwd_kernel_sm90.h:416) and converts back out of the basis
    exactly once, when it writes LSE.
    """
    import math

    s = softmax_scale_log2(HEAD_DIM)
    assert abs(s - math.log2(math.e) / math.sqrt(HEAD_DIM)) < 1e-12
    src = _source()
    assert "1.4426950408889634f / std::sqrt" in src, "the host does not fold log2(e) into the scale"
    assert "exp2f" in src, "the kernel uses exp rather than exp2 somewhere"


# =================================================================================================
# CPU — the oracle and the open-hole message
# =================================================================================================


def test_reference_is_fp32_over_the_same_bf16_operands_and_returns_lse() -> None:
    """The oracle must not be a different problem — it up-casts, it does not re-generate.

    If the reference attended fp32 operands, the operands' own bf16 quantization error would be
    inside the tolerance and a real kernel bug could hide under it.
    """
    torch.manual_seed(0)
    q = torch.randn(1, 2, 64, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 64, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 64, HEAD_DIM, dtype=torch.bfloat16)
    o, lse = reference_attention(q, k, v, causal=True)
    assert o.dtype is torch.float32 and lse.dtype is torch.float32
    assert o.shape == q.shape and lse.shape == (1, 2, 64)

    # against a dense fp32 softmax, which is a different algorithm and therefore a real check
    s = (q.float() @ k.float().transpose(-1, -2)) / HEAD_DIM**0.5
    mask = torch.arange(64)[None, :] > torch.arange(64)[:, None]
    s = s.masked_fill(mask, float("-inf"))
    dense = torch.softmax(s, dim=-1) @ v.float()
    assert torch.allclose(o, dense, atol=1e-4), (o - dense).abs().max()
    assert torch.allclose(lse, torch.logsumexp(s, dim=-1), atol=1e-4)


def test_reference_handles_gqa_and_both_layouts() -> None:
    """GQA is head indexing, not a different algorithm; the oracle must agree with itself on both."""
    torch.manual_seed(1)
    q = torch.randn(1, 8, 64, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 64, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 64, HEAD_DIM, dtype=torch.bfloat16)
    o_bhsd, lse_bhsd = reference_attention(q, k, v, causal=True, layout="bhsd")
    o_bshd, lse_bshd = reference_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        causal=True,
        layout="bshd",
    )
    assert torch.allclose(o_bhsd, o_bshd.transpose(1, 2), atol=1e-5)
    assert torch.allclose(lse_bhsd, lse_bshd, atol=1e-5)


def test_wrapper_reports_an_open_hole_as_such() -> None:
    """ "You have not written the kernel" and "your kernel is broken" are different messages."""
    if not hole_is_open():
        pytest.skip("hole filled — the HoleOpenError path no longer applies")
    z = torch.zeros(1, 2, BLOCK_M, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises((HoleOpenError, RuntimeError)) as exc:
        fa3_hopper_v2_fwd(z, z, z)
    # On a CPU box require_arch fires first (RuntimeError); on a Hopper box the loader does.
    assert "fa3_hopper_v2_fwd" in str(exc.value) or "HUY hole" in str(exc.value)


# =================================================================================================
# drydock — what nvcc generated. No GPU, but the CUDA container must have run.
#
# Unlike K1/H-R2, part of this tier runs TODAY: the producer path, the barrier ring, the warp-role
# split and the epilogue are all real code that compiles under -DHUY_STUB_KERNEL_BODY=1. Only the
# assertions about the consumer mainloop have to wait for the hole.
# =================================================================================================


def _sass(*, needs_hole_filled: bool) -> str:
    p = _DRYDOCK / f"{_STEM}.sm_90a.sass.txt"
    if not p.is_file():
        pytest.skip(f"no SASS at {p} — run: infra/drydock.sh compile {SOURCE}")
    if needs_hole_filled and hole_is_open():
        pytest.skip("hole open — the captured SASS has no mainloop in it yet")
    return p.read_text(encoding="utf-8", errors="replace")


@pytest.mark.drydock
def test_sass_has_the_scaffolding_this_rung_is_built_out_of() -> None:
    """TMA, warp specialisation and mbarriers, in the code nvcc actually emitted.

    These three are outside the hole, so this assertion is live from day one — which is the point of
    building the scaffolding first. ``UTMALDG.4D`` in particular is the rank-4 tensor copy: a rank-2
    map would show as ``UTMALDG.2D`` and would mean the sequence dimension had been flattened away.
    """
    sass = _sass(needs_hole_filled=False)
    assert re.search(r"\bUTMALDG\.4D\b", sass), (
        "no rank-4 bulk-tensor copy in the SASS — either the loads are not using TMA, or the "
        "tensor maps lost the sequence dimension"
    )
    assert re.search(r"\bUSETMAXREG\b", sass), (
        "no setmaxnreg in the SASS — the warp-role split is not actually redistributing registers, "
        "so the consumer warpgroups cannot reach 240 and the accumulators will spill"
    )
    assert re.search(r"\bSYNCS\b", sass), (
        "no SYNCS in the SASS — the mbarrier arrive/wait pairs are missing, so nothing is ordering "
        "the TMA against the consumers"
    )


@pytest.mark.drydock
def test_sass_has_no_ampere_style_async_copy() -> None:
    """``LDGSTS`` is cp.async. Its presence means some load did not go through TMA."""
    sass = _sass(needs_hole_filled=False)
    assert not re.findall(r"\bLDGSTS\b", sass), (
        "cp.async (LDGSTS) survived: this rung's premise is that TMA moves every K and V tile"
    )


@pytest.mark.drydock
def test_sass_has_no_local_memory_traffic() -> None:
    """``LDL``/``STL`` mean something landed in the stack frame.

    Not only spills: an array of shared-memory pointers indexed by a runtime stage — the obvious way
    to write ``sK[stage]`` — is a local array, and ``-Xptxas -v`` reports it as a stack frame with
    ZERO spill bytes, so the drydock's spill gate does not see it. The mainloop then pays an LDL per
    k-block. This assertion is what catches it.
    """
    sass = _sass(needs_hole_filled=False)
    spills = re.findall(r"\b(LDL|STL)\b", sass)
    assert not spills, f"{len(spills)} local-memory accesses in the SASS"


@pytest.mark.drydock
def test_sass_assembles_both_wgmma_forms_and_the_named_barriers() -> None:
    """The two instructions this rung is built out of, in the code nvcc actually emitted.

    This one is live while the hole is open, and that is deliberate. The stub body uses no wgmma and
    no named barriers, so the wrappers would be ``__forceinline__`` and uninstantiated and their asm
    would never reach ptxas — "it compiles" would be a claim about the loads and the epilogue only.
    The ``fa3_v2_asm_probe`` kernel in the ``.cu`` (stub builds only, never launched, never in the
    AOT extension) calls every one of them so that the RS-form operand list, which is the single
    most likely thing in this file to be mistyped, is assembled today rather than on the day the
    hole is filled. When the hole IS filled the probe is compiled out and these same instructions
    come from the real mainloop.
    """
    sass = _sass(needs_hole_filled=False)
    assert re.search(r"\bHGMMA\b", sass), (
        "no HGMMA in the SASS — either the wgmma asm does not assemble, or the file was compiled "
        "for base sm_90 instead of sm_90a (the trailing 'a' selects the accelerated ISA)"
    )
    assert re.search(r"\bBAR\.ARV\b", sass), (
        "no BAR.ARV in the SASS — the named-barrier ping-pong between the two consumer warpgroups "
        "is missing, so they are not being kept out of each other's way"
    )


@pytest.mark.drydock
def test_ptxas_injected_no_fences_of_its_own() -> None:
    """C7519 means ptxas inserted a ``warpgroup.arrive`` the code should have carried itself.

    It is a rescue, not a free lunch: each injection is a serialisation point between a register
    write and the wgmma that reads it. Interleaving ``pack_p_fragment`` with the PV issues produces
    seven of them; converting the whole P fragment first — which is what FA3 does,
    ``convert_layout_acc_Aregs`` at mainloop_fwd_sm90_tma_gmma_ws.hpp:1157 and then the gemm call —
    produces none. Measured in the dry dock, 2026-09-07.
    """
    p = _DRYDOCK / f"{_STEM}.sm_90a.ptxas.txt"
    if not p.is_file():
        pytest.skip(f"no ptxas report at {p} — run: infra/drydock.sh compile {SOURCE}")
    report = p.read_text(encoding="utf-8", errors="replace")
    injected = re.findall(r"C7519.*warpgroup\.arrive is injected", report)
    assert not injected, (
        f"{len(injected)} fences injected by ptxas — convert the whole P fragment before issuing "
        f"any PV wgmma (step 5 of the hole comment):\n{report[:600]}"
    )


@pytest.mark.drydock
def test_ptxas_report_is_clean() -> None:
    """0 spill bytes, 0 stack frame, and static shared memory within the arch limit.

    The static figure is only the mbarriers; the 160 KB of staging is dynamic and does not appear in
    this report at all, which is why the opt-in has its own CPU test above.
    """
    p = _DRYDOCK / f"{_STEM}.sm_90a.ptxas.txt"
    if not p.is_file():
        pytest.skip(f"no ptxas report at {p} — run: infra/drydock.sh compile {SOURCE}")
    report = p.read_text(encoding="utf-8", errors="replace")
    # Every entry function, not the first: a stub build also carries fa3_v2_asm_probe, and a spill
    # in either one is a spill.
    spills = re.findall(r"(\d+) bytes spill stores", report)
    assert spills and all(int(x) == 0 for x in spills), f"register spill in {p.name}: {report}"
    frames = re.findall(r"(\d+) bytes stack frame", report)
    assert frames and all(int(x) == 0 for x in frames), (
        f"a stack frame with no spills means a local array — see the LDL/STL test: {report}"
    )
    smem = re.search(r"(\d+) bytes smem", report)
    if smem:
        assert int(smem.group(1)) <= ARCH["sm_90a"].smem_per_cta


# =================================================================================================
# gpu — correctness against the oracle. H100 only.
# =================================================================================================


def _require_tolerances() -> tuple[float, float]:
    if TOL_O is None or TOL_LSE is None:
        pytest.fail(
            "TOL_O/TOL_LSE are unset. The correctness gate cannot run without a tolerance, and "
            "picking one is the rung's numerics lesson: set them in this file and write the "
            "one-line argument into experiments/K2/A-R2/spec.md's 'Correctness gate' line before "
            "the first measured run."
        )
    return TOL_O, TOL_LSE


def _assert_matches_oracle(
    out: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    layout: str = "bhsd",
) -> None:
    tol_o, tol_lse = _require_tolerances()
    ref_o, ref_lse = reference_attention(q, k, v, causal=causal, layout=layout)
    assert out.shape == ref_o.shape, f"{out.shape} != {ref_o.shape}"
    assert out.dtype is torch.bfloat16, f"kernel must return bf16, got {out.dtype}"
    err_o = (out.float() - ref_o).abs().max().item()
    err_lse = (lse - ref_lse).abs().max().item()
    assert err_o <= tol_o, (
        f"max|dO| = {err_o:.4e} > {tol_o:.4e}. The three structured failures to look for first are "
        f"a wrong PV trans-b (V read as K-major: error in every head-dim column), a row/column swap "
        f"in the fragment map (error correlated with row % 8), and a missing O rescale (error only "
        f"in rows whose running max rose late). Print (out - ref).abs().amax(dim=-1) before "
        f"widening this."
    )
    assert err_lse <= tol_lse, (
        f"max|dLSE| = {err_lse:.4e} > {tol_lse:.4e}. LSE isolates the softmax from the PV gemm: if "
        f"O is right and LSE is wrong the epilogue's log2->ln conversion is the suspect, and if "
        f"both are wrong the online recurrence is."
    )


@pytest.mark.gpu
@pytest.mark.parametrize("shape_name", list(GATE_SHAPES))
def test_matches_oracle(shape_name: str) -> None:
    """Every gate shape against the fp32 oracle at the spec's tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    b, h, kv_h, s, causal = GATE_SHAPES[shape_name]
    torch.manual_seed(0)
    q = torch.randn(b, h, s, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, kv_h, s, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, kv_h, s, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    o, lse = fa3_hopper_v2_fwd(q, k, v, causal=causal, return_lse=True)
    _assert_matches_oracle(o, lse, q, k, v, causal=causal)


@pytest.mark.gpu
def test_multi_stage_shape_exercises_the_pipeline_wrap() -> None:
    """The sequence must span more than STAGES k-blocks or the phase bit is never flipped.

    A 2-stage pipeline with a wrong phase parity is correct for the first STAGES iterations and then
    hangs or reads a stale tile. A sequence of 2*BLOCK_N would pass a broken kernel.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    s = BLOCK_M * (2 * STAGES + 1)
    torch.manual_seed(2)
    q = torch.randn(1, 4, s, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 4, s, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 4, s, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    o, lse = fa3_hopper_v2_fwd(q, k, v, causal=True, return_lse=True)
    _assert_matches_oracle(o, lse, q, k, v, causal=True)


@pytest.mark.gpu
def test_both_layouts_give_the_same_answer() -> None:
    """[B,H,S,D] and [B,S,H,D] must agree, because the floor is timed in one of them.

    If they disagree the strides are being misread, and the benchmark against ``flash_attn_func``
    would be comparing two different computations.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(3)
    q = torch.randn(2, 4, BLOCK_M * 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 4, BLOCK_M * 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 4, BLOCK_M * 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    o_bhsd = fa3_hopper_v2_fwd(q, k, v, causal=True)
    o_bshd = fa3_hopper_v2_fwd(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        causal=True,
        layout="bshd",
    )
    # `return_lse` is False here, so each call returns the output tensor alone. Asserted rather
    # than assumed: if the signature ever returns the pair by default, this says so instead of
    # comparing a tuple against a tensor.
    assert isinstance(o_bhsd, torch.Tensor) and isinstance(o_bshd, torch.Tensor)
    assert torch.equal(o_bhsd, o_bshd.transpose(1, 2)), (
        "the two layouts disagree — the strides handed to cuTensorMapEncodeTiled are being derived "
        "wrongly for one of them"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3], ids=["tiny", "unit", "large"])
def test_matches_oracle_across_magnitudes(scale: float) -> None:
    """The running max makes attention scale-invariant; a tolerance that is not must be argued for."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(4)
    shape = (1, 2, BLOCK_M * 2, HEAD_DIM)
    q = (torch.randn(*shape, device="cuda") * scale).bfloat16()
    k = (torch.randn(*shape, device="cuda") * scale).bfloat16()
    v = (torch.randn(*shape, device="cuda") * scale).bfloat16()
    o, lse = fa3_hopper_v2_fwd(q, k, v, causal=True, return_lse=True)
    _assert_matches_oracle(o, lse, q, k, v, causal=True)


@pytest.mark.gpu
def test_rejects_bad_inputs() -> None:
    """The guard clauses, on the device where they can actually be reached."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    def z(s: int, h: int = 2, d: int = HEAD_DIM, dt: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(1, h, s, d, device="cuda", dtype=dt)

    with pytest.raises(TypeError, match="bfloat16"):
        fa3_hopper_v2_fwd(z(BLOCK_M, dt=torch.float16), z(BLOCK_M), z(BLOCK_M))
    with pytest.raises(ValueError, match="head dim"):
        fa3_hopper_v2_fwd(z(BLOCK_M, d=64), z(BLOCK_M, d=64), z(BLOCK_M, d=64))
    with pytest.raises(ValueError, match="softmax"):
        fa3_hopper_v2_fwd(z(BLOCK_M + 8), z(BLOCK_M + 8), z(BLOCK_M + 8))
    with pytest.raises(ValueError, match="GQA"):
        fa3_hopper_v2_fwd(z(BLOCK_M, h=8), z(BLOCK_M, h=3), z(BLOCK_M, h=3))


@pytest.mark.gpu
def test_handles_nan_without_masking_it() -> None:
    """A NaN in must produce a NaN out — a kernel that swallows them hides real training bugs."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    q = torch.ones(1, 2, BLOCK_M, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    q[0, 0, 0, 0] = float("nan")
    out = fa3_hopper_v2_fwd(q, k, v, causal=True)
    assert isinstance(out, torch.Tensor)  # return_lse=False: the output tensor alone
    assert torch.isnan(out[0, 0, 0]).any(), "NaN was swallowed"


@pytest.mark.gpu
def test_the_floor_is_present_before_a_number_is_quoted() -> None:
    """On the box, the floor must actually be importable — otherwise there is no number to quote."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if torch.cuda.get_device_capability(0) != RUNG_ARCH:
        pytest.skip("not Hopper — FA3 is not the floor for this device")
    assert floor_unavailable_reason() is None, (
        f"FA3 is missing on a Hopper box: {floor_unavailable_reason()}. Install "
        f"flash_attn_interface before measuring; do NOT substitute FA2."
    )
    assert SYMBOL == "fa3_hopper_v2_fwd"
