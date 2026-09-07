"""K2/A-R3 — paged decode, split-KV. Three tiers of gate, in ascending cost.

  CPU      the page table's layout contract, the log-sum-exp merge, the partition work list, and
           the arithmetic that says chunking-then-merging equals not chunking. Milliseconds.
  drydock  what nvcc generated: cp.async in the loader, the butterfly in the score reduction,
           base-2 exponentials, 16 KB of static shared memory, nothing spilled. Needs the CUDA
           container, still no GPU.
  gpu      correctness against the fp32 oracle, and against FlashInfer at the same page table.
           Needs an H100.

The CPU tier carries more weight here than it did on the K1 rungs, for a reason specific to decode:
every failure mode of a split-KV kernel produces a plausible number. A page table whose
``last_page_len`` is 0 for a full page attends 16 tokens too few per request and is 0.2% wrong. A
merge that forgets to subtract the max overflows to inf. A work list that interleaves requests
merges one request's chunks into another's output. None of those fault, none of them are slow, and
all of them are decidable on a laptop.

Spec: experiments/K2/A-R3/spec.md   ·   Map: experiments/K2/A-R3/map.md
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess

import pytest
import torch

from scratch_llm.kernels.attention.decode.page_table import (
    INDEX_DTYPE,
    PageTable,
    empty_kv_pool,
    gather_kv,
)
from scratch_llm.kernels.attention.decode.paged_split_kv import (
    ARCH_CC,
    BDX,
    BDY,
    BYTES_PER_STAGE,
    GROUP_SIZE,
    HEAD_DIM,
    LOG2E,
    MERGE_ELEMS_PER_THREAD,
    MERGE_THREADS,
    PAGE_SIZE,
    PARTIAL_DTYPE,
    REGS_PER_THREAD,
    RUNG,
    SMEM_BYTES,
    SOURCE,
    STAGES,
    THREADS_PER_CTA,
    TILE_TOKENS,
    VEC_ELEMS,
    PartitionPlan,
    a_r3_paged_decode,
    ctas_per_sm,
    default_sm_scale,
    merge_states_reference,
    partial_buffers,
    partial_bytes,
    plan_partition,
    reference_paged_decode,
    reference_partials,
    smem_budget_errors,
    unsupported_reason,
)
from scratch_llm.kernels.common.hopper_contracts import ARCH, check_register_budget
from scratch_llm.kernels.gemm.cuda._k1_loader import workspace_root

_WORKSPACE = workspace_root()
_CSRC = _WORKSPACE / "scratch_llm" / "csrc" / "attention" / SOURCE
_DRYDOCK = _WORKSPACE / "experiments" / "K2" / "A-R3" / "drydock"
_STEM = SOURCE.removesuffix(".cu")

# ---------------------------------------------------------------------------------------------
# The tolerance. HUY SETS THIS, with its argument, before the first measured run.
#
# The gate's form is  max|o_kernel - o_fp32|  <=  TOL_ABS, on a bf16 output whose values live in
# [-1, 1] by construction (a convex combination of V rows). Two error sources have to be separated
# before a constant can be defended: the bf16 output quantum (2^-8 relative), and the split's own
# contribution — a chunked softmax re-associates the sum, so the merged result is NOT bit-identical
# to the unchunked one even in exact arithmetic order. The second is the interesting one, it is
# what this rung adds over A-R1/A-R2, and its size is a function of the split count.
#
# Leaving it None is deliberate. Choosing the constant IS the rung's numerics lesson, and the
# argument belongs in spec.md's "Correctness gate" line.
# ---------------------------------------------------------------------------------------------
TOL_ABS: float | None = None

#: The plan's row, verbatim (SIXTY_DAYS_SIX_LADDERS.md §05 K2 A-R3), and the shape every number is
#: quoted at. ``kv_h`` is the k2_ladder `decode64` shape's 4, giving 32 query heads at GQA 8:1.
SPEC = {"batch": 64, "ctx": 8192, "page": PAGE_SIZE, "head_dim": HEAD_DIM, "kv_heads": 4}

#: An H100 SXM. Only used to make the partition's inputs concrete in the hole's guard test; the
#: real number comes from ``cudaDeviceGetAttribute`` on the box (scheduler.cuh:179).
H100_SM_COUNT = 132


def _source() -> str:
    return _CSRC.read_text(encoding="utf-8")


def _tiny_problem(seq_lens: list[int], *, kv_heads: int = 2, seed: int = 0, page: int = PAGE_SIZE):
    """A small CPU problem: page table, NHD pool filled with noise, and a query row per request."""
    torch.manual_seed(seed)
    table = PageTable.from_seq_lens(seq_lens, page_size=page, shuffle_seed=seed + 1)
    heads = kv_heads * GROUP_SIZE
    k, v = empty_kv_pool(
        num_pages=table.num_pages,
        page_size=page,
        num_kv_heads=kv_heads,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
    )
    k.normal_()
    v.normal_()
    q = torch.randn(len(seq_lens), heads, HEAD_DIM)
    return q, k, v, table


# =============================================================================================
# The hole guard — exactly one (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("K2/A-R3", "src/scratch_llm/kernels/attention/decode/paged_split_kv.py")
def test_partition_chooses_a_split_that_fills_the_machine() -> None:
    """Fails while :func:`plan_partition` is a ``NotImplementedError``; passes once it decides.

    The assertions are the two walls the decision sits between, not the decision itself:

    * the work list must tile every request's pages exactly — a chunk range attended twice is
      double-counted in the softmax denominator, and one skipped is silently dropped context;
    * the grid must be at least as large as the machine. At the plan's shape the *unsplit* answer
      already clears this (64 requests x 4 KV heads = 256 CTAs against 132 SMs), so this cannot
      punish a conservative choice — it only catches a split that made the grid *smaller*, which
      is the one arithmetic slip (dividing where you meant to multiply) that would otherwise show
      up as a mysteriously slow kernel.

    Where the number between those walls should land is the rung, and the box settles it: too few
    chunks and the 8192-token serial walk is the kernel; too many and the reduce's fan-in is.
    """
    pages = [SPEC["ctx"] // SPEC["page"]] * SPEC["batch"]
    plan = plan_partition(
        pages,
        num_kv_heads=SPEC["kv_heads"],
        max_grid_size=H100_SM_COUNT * ctas_per_sm(),
        page_size=SPEC["page"],
    )
    assert plan.validate() == [], plan.validate()
    assert 1 <= plan.chunk_size_pages <= max(pages)
    assert plan.num_chunks_total >= SPEC["batch"], "every request needs at least one chunk"
    for b in range(SPEC["batch"]):
        ranges = plan.covered_pages(b)
        assert ranges[0][0] == 0 and ranges[-1][1] == pages[b], ranges
        assert all(a[1] == c[0] for a, c in zip(ranges, ranges[1:], strict=False)), ranges
    assert plan.num_ctas(SPEC["kv_heads"]) >= H100_SM_COUNT, (
        f"{plan.num_ctas(SPEC['kv_heads'])} CTAs for {H100_SM_COUNT} SMs — the split made the grid "
        f"smaller than the machine, which no partition should ever do"
    )


def test_the_hole_sentinel_survives_the_formatter() -> None:
    """``ruff format`` broke this once, on the day the file was written, and it fails SILENTLY.

    ``tests/conftest.py`` decides a hole is open by grepping for the literal
    ``NotImplementedError("HUY:``. Wrap the string onto the next line — which the formatter does to
    any long message — and the sentinel disappears, the strict xfail is never applied, and the
    guard test above goes green while the partition is still unwritten. That is the one failure the
    whole hole convention exists to prevent, so it gets its own assertion.
    """
    import importlib.util  # noqa: PLC0415

    # Loaded by path rather than `from tests.conftest import ...`. The bare `pytest` console
    # script — what CI and the pre-push hook run — does not put the rootdir on sys.path, so the
    # dotted import resolves only under `python -m pytest`. A test that passes one way and errors
    # the other is worse than no test, and this one is guarding the hole convention itself.
    _cf = _WORKSPACE / "scratch_llm" / "tests" / "conftest.py"
    _spec = importlib.util.spec_from_file_location("_hole_contract", _cf)
    assert _spec is not None and _spec.loader is not None, f"cannot load {_cf}"
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    hole_is_open = _mod.hole_is_open

    hole_source = "src/scratch_llm/kernels/attention/decode/paged_split_kv.py"
    assert hole_is_open(hole_source) or "raise NotImplementedError" not in (
        _WORKSPACE / "scratch_llm" / hole_source
    ).read_text(encoding="utf-8"), (
        "the source still raises NotImplementedError but conftest.hole_is_open() says the hole is "
        'closed — the `NotImplementedError("HUY: ...")` sentinel has been split across lines'
    )


# =============================================================================================
# CPU — the page table
# =============================================================================================


def test_round_trip_from_seq_lens() -> None:
    """The one property everything else rests on: ``get_length`` returns what was asked for."""
    lens = [1, 15, 16, 17, 31, 32, 4096, 8192]
    table = PageTable.from_seq_lens(lens, page_size=PAGE_SIZE)
    assert table.seq_lens() == lens
    assert table.validate() == []


def test_a_full_last_page_is_page_size_not_zero() -> None:
    """``seq_len % page_size`` is the wrong expression, and at the plan's shape it is wrong for
    every request in the batch — 8192 % 16 == 0, so the whole batch would attend 16 tokens short.
    """
    table = PageTable.from_seq_lens([16, 32, 8192], page_size=16)
    assert table.last_page_len.tolist() == [16, 16, 16]
    assert table.seq_lens() == [16, 32, 8192]
    # And the checker must reject the naive version rather than pass it through.
    bad = PageTable(
        indptr=table.indptr,
        indices=table.indices,
        last_page_len=torch.zeros_like(table.last_page_len),
        page_size=16,
        num_pages=table.num_pages,
    )
    assert any("outside 1..16" in e for e in bad.validate()), bad.validate()


def test_index_tensors_are_int32() -> None:
    """int64 is refused by the floor (csrc/batch_decode.cu:45) and accepted by us — the worst
    possible asymmetry, so the type is part of the contract and is checked here, not there."""
    table = PageTable.from_seq_lens([100, 200], page_size=PAGE_SIZE)
    for t in (table.indptr, table.indices, table.last_page_len):
        assert t.dtype is INDEX_DTYPE
    bad = PageTable(
        indptr=table.indptr.long(),
        indices=table.indices,
        last_page_len=table.last_page_len,
        page_size=PAGE_SIZE,
        num_pages=table.num_pages,
    )
    assert any("must be torch.int32" in e for e in bad.validate()), bad.validate()


def test_the_spec_shape_matches_the_maps_arithmetic() -> None:
    """B=64 · ctx 8192 · page 16, computed independently of map.md and asserted against it.

    map.md derived these by hand from the upstream sources; if the builder disagrees, one of the
    two is wrong and the FlashInfer comparison is being set up on a table the floor would reject.
    """
    table = PageTable.from_seq_lens([SPEC["ctx"]] * SPEC["batch"], page_size=SPEC["page"])
    assert table.indptr.numel() == SPEC["batch"] + 1
    assert table.indptr.tolist() == [512 * i for i in range(SPEC["batch"] + 1)]
    assert int(table.indptr[-1]) == 32768
    assert table.indices.numel() == 32768
    assert table.last_page_len.tolist() == [16] * SPEC["batch"]
    assert table.pages_per_request() == [512] * SPEC["batch"]


def test_out_of_pool_page_ids_are_rejected() -> None:
    """An out-of-range id does not fault on the read path — ``protective_get_kv_offset``
    (page.cuh:228-234) hands back offset 0, i.e. page 0's contents, silently."""
    table = PageTable.from_seq_lens([64], page_size=PAGE_SIZE)
    bad = PageTable(
        indptr=table.indptr,
        indices=table.indices + table.num_pages,
        last_page_len=table.last_page_len,
        page_size=PAGE_SIZE,
        num_pages=table.num_pages,
    )
    assert any("out-of-range" in e or "pool holds" in e for e in bad.validate()), bad.validate()


def test_non_monotonic_indptr_is_rejected() -> None:
    """``indptr[i+1] < indptr[i]`` makes ``get_length`` negative and the page slice run backwards
    — request ``i`` would read request ``i-1``'s pages."""
    table = PageTable.from_seq_lens([64, 64], page_size=PAGE_SIZE)
    ind = table.indptr.clone()
    ind[1] = 6  # request 0 keeps 6 pages, so request 1 spans 6..4 — backwards
    ind[2] = 4
    bad = PageTable(
        indptr=ind,
        indices=table.indices,
        last_page_len=table.last_page_len,
        page_size=PAGE_SIZE,
        num_pages=table.num_pages,
    )
    assert any("non-decreasing" in e for e in bad.validate()), bad.validate()


def test_shuffled_pages_are_the_default_hard_case() -> None:
    """A sequential table hides a kernel that computes its address from the logical page index.

    Such a kernel is exactly right on ``indices = arange`` and wrong on every fragmented table a
    real server produces — and the floor reads ``indices`` too (page.cuh:222), so both sides must.
    """
    table = PageTable.from_seq_lens([64, 128], page_size=PAGE_SIZE, num_pages=64, shuffle_seed=7)
    assert table.indices.tolist() != sorted(table.indices.tolist())
    assert table.validate() == []
    assert table.seq_lens() == [64, 128]


def test_gather_kv_de_pages_in_the_pool_order() -> None:
    """The oracle's de-paging must follow ``indices``, not the logical order, or the reference and
    the kernel are reading different bytes and every tolerance is meaningless."""
    page, seq = 4, 10
    table = PageTable.from_seq_lens([seq], page_size=page, num_pages=8, shuffle_seed=3)
    k, v = empty_kv_pool(
        num_pages=8, page_size=page, num_kv_heads=1, head_dim=HEAD_DIM, dtype=torch.float32
    )
    for p in range(8):
        k[p] = float(p)
        v[p] = float(-p)
    kk, vv = gather_kv(table, k, v, 0)
    assert kk.shape == (seq, 1, HEAD_DIM)
    want = [float(p) for p in table.page_ids(0) for _ in range(page)][:seq]
    assert kk[:, 0, 0].tolist() == want
    assert vv[:, 0, 0].tolist() == [-x for x in want]


# =============================================================================================
# CPU — the partition work list (the mechanical half; the number itself is the hole)
# =============================================================================================


def test_work_list_tiles_every_request_exactly() -> None:
    pages = [1, 7, 8, 9, 100]
    for chunk in (1, 2, 3, 8, 100, 1000):
        plan = PartitionPlan.from_chunk_size(pages, chunk).require_valid()
        assert plan.num_chunks_total == sum(max(1, -(-n // chunk)) for n in pages)
        for b, n in enumerate(pages):
            covered: list[int] = []
            for lo, hi in plan.covered_pages(b):
                covered.extend(range(lo, hi))
            assert covered == list(range(n)), (chunk, b)


def test_work_list_is_request_major() -> None:
    """The reduce merges ``tmp_o[o_indptr[b]:o_indptr[b+1]]`` as one contiguous span. An
    interleaved list merges one request's chunks into another's output, at full speed."""
    plan = PartitionPlan.from_chunk_size([4, 4, 4], 2)
    assert list(plan.request_indices) == [0, 0, 1, 1, 2, 2]
    assert list(plan.o_indptr) == [0, 2, 4, 6]
    shuffled = PartitionPlan(
        chunk_size_pages=plan.chunk_size_pages,
        pages_per_request=plan.pages_per_request,
        request_indices=(0, 1, 0, 1, 2, 2),
        chunk_indices=plan.chunk_indices,
        o_indptr=plan.o_indptr,
    )
    assert any("non-decreasing" in e for e in shuffled.validate()), shuffled.validate()


def test_a_zero_page_request_still_gets_a_work_item() -> None:
    """Otherwise ``o_indptr[i] == o_indptr[i+1]`` and the reduce emits an uninitialised row."""
    plan = PartitionPlan.from_chunk_size([0, 4], 2).require_valid()
    assert plan.chunks_of(0) == 1
    assert plan.covered_pages(0) == [(0, 0)]


def test_plan_rejects_a_work_list_that_does_not_match_its_chunk_size() -> None:
    plan = PartitionPlan.from_chunk_size([8], 4)
    broken = PartitionPlan(
        chunk_size_pages=4,
        pages_per_request=(8,),
        request_indices=(0,),
        chunk_indices=(0,),
        o_indptr=(0, 1),
    )
    assert plan.validate() == []
    assert any("needs 2 chunks" in e for e in broken.validate()), broken.validate()


def test_partition_tensors_are_int32_and_on_the_asked_for_device() -> None:
    req, chunk, o_indptr = PartitionPlan.from_chunk_size([4, 4], 2).to_tensors("cpu")
    for t in (req, chunk, o_indptr):
        assert t.dtype is torch.int32


# =============================================================================================
# CPU — the reduce, against a slow reference and against the un-chunked answer
# =============================================================================================


def test_merge_of_one_chunk_is_the_identity() -> None:
    torch.manual_seed(0)
    tmp_o = torch.randn(2, 3, HEAD_DIM)
    tmp_lse = torch.randn(2, 3)
    o, lse = merge_states_reference(tmp_o, tmp_lse, [0, 1, 2])
    torch.testing.assert_close(o, tmp_o, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(lse, tmp_lse, rtol=1e-6, atol=1e-6)


def test_merge_survives_chunk_maxima_a_hundred_apart() -> None:
    """The reason the max is subtracted. ``2^100`` overflows fp32 on the second chunk; a merge
    that scales by ``2^lse`` instead of ``2^(lse - m)`` returns ``inf`` and then ``nan``."""
    tmp_o = torch.zeros(2, 1, HEAD_DIM)
    tmp_o[0, 0] = 1.0
    tmp_o[1, 0] = 2.0
    tmp_lse = torch.tensor([[-60.0], [60.0]])
    o, lse = merge_states_reference(tmp_o, tmp_lse, [0, 2])
    assert torch.isfinite(o).all() and torch.isfinite(lse).all()
    # The high-lse chunk carries weight 1 - 2^-120, i.e. everything representable in fp32.
    torch.testing.assert_close(o[0, 0], torch.full((HEAD_DIM,), 2.0))
    torch.testing.assert_close(lse[0, 0], torch.tensor(60.0))


def test_merge_weights_are_base_two() -> None:
    """Two equal-mass chunks one lse unit apart weight 1 : 2, not 1 : e. An ln-based partial fed
    to a base-2 merge is wrong, finite and plausible (cascade.cuh:641)."""
    tmp_o = torch.zeros(2, 1, HEAD_DIM)
    tmp_o[0, 0] = 0.0
    tmp_o[1, 0] = 3.0
    tmp_lse = torch.tensor([[0.0], [1.0]])
    o, _ = merge_states_reference(tmp_o, tmp_lse, [0, 2])
    torch.testing.assert_close(o[0, 0, 0], torch.tensor(3.0 * 2.0 / 3.0))


@pytest.mark.parametrize(
    "seq_lens",
    [[16], [17], [64, 16], [100, 33, 1], [8, 8, 8, 8]],
    ids=["one-page", "ragged-tail", "two", "three-ragged", "four-equal"],
)
@pytest.mark.parametrize("chunk_pages", [1, 2, 3, 64])
def test_chunking_then_merging_equals_not_chunking(seq_lens, chunk_pages) -> None:
    """The premise of split-KV, proved on a laptop.

    If this holds, the only thing the GPU can add is a bug in the *implementation* of the merge —
    which the drydock and gpu tiers cover. If it did not hold, no amount of kernel work would
    matter, and no benchmark would ever say so.
    """
    q, k, v, table = _tiny_problem(seq_lens)
    plan = PartitionPlan.from_chunk_size(table.pages_per_request(), chunk_pages).require_valid()
    tmp_o, tmp_lse = reference_partials(q, k, v, table, plan)
    merged_o, merged_lse = merge_states_reference(tmp_o, tmp_lse, plan.o_indptr)
    direct_o, direct_lse = reference_paged_decode(q, k, v, table)
    torch.testing.assert_close(merged_o, direct_o, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(merged_lse, direct_lse, rtol=2e-6, atol=2e-6)


def test_the_oracles_lse_is_base_two() -> None:
    """One key with one head: ``lse`` must be the base-2 logit, i.e. ``q.k * scale * log2(e)``."""
    table = PageTable.from_seq_lens([1], page_size=PAGE_SIZE)
    k, v = empty_kv_pool(
        num_pages=table.num_pages,
        page_size=PAGE_SIZE,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
    )
    torch.manual_seed(0)
    k[0, 0, 0].normal_()
    v[0, 0, 0].normal_()
    q = torch.randn(1, GROUP_SIZE, HEAD_DIM)
    _, lse = reference_paged_decode(q, k, v, table)
    want = (q[0, 0] @ k[0, 0, 0]) * default_sm_scale() * LOG2E
    torch.testing.assert_close(lse[0, 0], want, rtol=1e-5, atol=1e-5)


def test_reference_output_is_a_convex_combination_of_v() -> None:
    """A softmax-weighted average cannot leave the hull of its inputs. Catches a missing
    normalisation and a sign slip in the max subtraction in one assertion."""
    q, k, v, table = _tiny_problem([48, 33], seed=5)
    o, _ = reference_paged_decode(q, k, v, table)
    for b in range(2):
        kk, vv = gather_kv(table, k, v, b)
        for h in range(o.shape[1]):
            lo = vv[:, h // GROUP_SIZE, :].min(dim=0).values
            hi = vv[:, h // GROUP_SIZE, :].max(dim=0).values
            assert (o[b, h] >= lo - 1e-5).all() and (o[b, h] <= hi + 1e-5).all()


# =============================================================================================
# CPU — the source, the wrapper and the hardware contracts agree
# =============================================================================================


def test_source_and_wrapper_agree_on_the_frozen_shape() -> None:
    """The ``.cu``'s ``#define``s and the wrapper's constants are used for different things — one
    generates code, the other sizes the buffers and the occupancy estimate — so nothing else would
    notice if they drifted."""
    src = _source()
    for macro, expected in (
        ("HEAD_DIM", HEAD_DIM),
        ("PAGE", PAGE_SIZE),
        ("GROUP_SIZE", GROUP_SIZE),
        ("STAGES", STAGES),
        ("VEC", VEC_ELEMS),
        ("MERGE_THREADS", MERGE_THREADS),
    ):
        m = re.search(rf"^#define {macro} (\d+)", src, re.MULTILINE)
        assert m, f"#define {macro} not found in csrc/attention/{SOURCE}"
        assert int(m.group(1)) == expected, (
            f"{macro}={m.group(1)} in the .cu but {expected} in paged_split_kv.py"
        )
    assert BDX == HEAD_DIM // VEC_ELEMS == 16
    assert BDY == GROUP_SIZE
    assert THREADS_PER_CTA == BDX * BDY
    assert TILE_TOKENS == PAGE_SIZE
    assert MERGE_ELEMS_PER_THREAD * MERGE_THREADS == HEAD_DIM


def test_blockdim_y_is_the_query_head_so_kv_is_read_once_per_group() -> None:
    """The rung's whole bandwidth story, asserted rather than trusted.

    ``BDY == GROUP_SIZE`` is what makes one HBM read of a page serve all eight query heads. Set
    them apart and the kernel still produces the right answer — at 8x the traffic, which no
    correctness test can see and which is the difference between 85% of HBM and 11%.
    """
    assert BDY == GROUP_SIZE
    src = _source()
    assert re.search(r"qo_head\s*=\s*kv_head \* GROUP_SIZE \+ ty", src), (
        "the query head must be derived from threadIdx.y; if it came from blockIdx the GQA group "
        "would not share a CTA and the K/V tile would be re-read per head"
    )


def test_both_kernels_are_base_two_and_neither_uses_a_natural_exponential() -> None:
    """The exact silent bug the map flags: the merge assumes base 2 (cascade.cuh:641), so an
    ln-based partial produces a merged output that is wrong, finite and plausible."""
    src = _source()
    assert "ex2.approx.ftz.f32" in src, "the partial's exponential must be base 2"
    assert "lg2.approx.ftz.f32" in src, "lse = m + log2(d) must be base 2"
    stray = re.findall(r"(?<![_a-z2])(expf|__expf|logf|__logf)\s*\(", src)
    assert not stray, f"natural-base math in a base-2 kernel: {stray}"
    assert src.count("a_r3_exp2(") >= 4, (
        "both the partial's rescale/weight and the merge's two weights must go through the same "
        "base-2 wrapper — a second exponential spelled differently is how the two sides drift"
    )


def test_shared_memory_fits_without_an_opt_in() -> None:
    """16 KB is under the 48 KB a launch gets for free, so this kernel needs no
    ``cudaFuncSetAttribute`` — unlike K1/H-R2, whose 64 KB does. A stage count that crossed that
    line would launch-fail on the box and nowhere else."""
    assert BYTES_PER_STAGE * STAGES == SMEM_BYTES == 16384
    assert smem_budget_errors() == [], smem_budget_errors()
    assert ARCH["sm_90a"].smem_opt_in_threshold >= SMEM_BYTES
    assert not re.search(r"cudaFuncSetAttribute\s*\(", _source()), (
        "no dynamic shared memory is requested, so an opt-in call would be dead code hiding a "
        "future overflow"
    )
    assert not re.search(r"extern\s+__shared__", _source()), (
        "the staging buffers are static; an `extern __shared__` would need the opt-in above and a "
        "size argument at the launch"
    )


def test_register_budget_leaves_more_than_one_cta_per_sm() -> None:
    """A decode kernel resident one-CTA-per-SM cannot hide its own memory latency: there is no
    other warp to run while the cp.async lands. The dry-dock ptxas number is the input."""
    assert (
        check_register_budget(
            regs_per_thread=REGS_PER_THREAD, threads_per_cta=THREADS_PER_CTA, arch=ARCH["sm_90a"]
        )
        == []
    )
    assert ctas_per_sm() >= 2, (
        f"{ctas_per_sm()} CTA/SM at {REGS_PER_THREAD} registers — the cp.async arrival would be "
        f"fully exposed"
    )


def test_partial_buffers_are_fp32_and_sized_from_the_split() -> None:
    tmp_o, tmp_lse = partial_buffers(7, 32)
    assert tmp_o.shape == (7, 32, HEAD_DIM) and tmp_o.dtype is PARTIAL_DTYPE
    assert tmp_lse.shape == (7, 32) and tmp_lse.dtype is PARTIAL_DTYPE
    assert partial_bytes(7, 32) == 7 * 32 * (HEAD_DIM + 1) * 4
    # The trade the split makes explicit: at the plan's shape one extra chunk per request is 1 MiB
    # of workspace and the same again in reduce traffic, against 512 MiB of KV read.
    extra = partial_bytes(SPEC["batch"], SPEC["kv_heads"] * GROUP_SIZE)
    assert 1.0e6 < extra < 1.2e6


def test_unsupported_shapes_are_named_not_silently_accepted() -> None:
    """Every one of these is a frozen ``#define``; a kernel that quietly ran another shape would
    be measuring something the plan did not commit to."""
    ok = {"head_dim": HEAD_DIM, "page_size": PAGE_SIZE, "num_qo_heads": 32, "num_kv_heads": 4}
    assert unsupported_reason(**ok) is None
    assert "head_dim" in (unsupported_reason(**{**ok, "head_dim": 64}) or "")
    assert "page_size" in (unsupported_reason(**{**ok, "page_size": 32}) or "")
    assert "GQA" in (unsupported_reason(**{**ok, "num_qo_heads": 16}) or "")


def test_wrapper_reports_an_open_hole_as_such() -> None:
    """ "You have not chosen the split" and "your kernel is broken" are different messages."""
    q, k, v, table = _tiny_problem([32])
    with pytest.raises((NotImplementedError, RuntimeError)) as exc:
        a_r3_paged_decode(q.bfloat16(), k.bfloat16(), v.bfloat16(), table)
    # On a CPU box require_arch fires first (RuntimeError); on a Hopper box plan_partition does.
    assert "a_r3_paged_decode" in str(exc.value) or "HUY" in str(exc.value)


def test_rung_identity_is_the_plans() -> None:
    assert RUNG == "K2/A-R3"
    assert ARCH_CC == (9, 0)
    assert _CSRC.is_file(), f"kernel source missing at {_CSRC}"


# =============================================================================================
# drydock — what nvcc generated. No GPU, but the CUDA container must have run.
# =============================================================================================


def _sass() -> str:
    p = _DRYDOCK / f"{_STEM}.sm_90a.sass.txt"
    if not p.is_file():
        pytest.skip(f"no SASS at {p} — run: infra/drydock.sh compile {SOURCE}")
    return p.read_text(encoding="utf-8", errors="replace")


#: The row this rung needs in ``infra/drydock.sh``'s ``_REGISTRY``. That file is lead-only, so the
#: test below names the row and skips rather than editing it or failing on someone else's file.
DRYDOCK_ROW = f'"csrc/attention/{SOURCE}|K2/A-R3|sm_90a"'


@pytest.mark.drydock
def test_kernel_compiles_clean_for_sm90a() -> None:
    """The whole file, through the real dry dock. Its stdout is asserted because
    ``drydock.sh compile <name>`` filters by substring and a name with no matching row compiles
    zero files and exits 0 — the gate would go green having compiled nothing."""
    if shutil.which("docker") is None:
        pytest.skip("docker absent — cannot run the dry-dock compile (infra/drydock.sh)")
    registry = (_WORKSPACE / "infra" / "drydock.sh").read_text(encoding="utf-8")
    if SOURCE not in registry:
        pytest.skip(
            f"{SOURCE} is not in infra/drydock.sh's _REGISTRY (a lead-only file). Add the row "
            f"{DRYDOCK_ROW} and this test compiles for real; until then the captured artifacts in "
            f"experiments/K2/A-R3/drydock/ are what the other drydock tests read."
        )
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


@pytest.mark.drydock
def test_sass_stages_the_pages_with_cp_async() -> None:
    """``LDGSTS`` is cp.async. Without it the loader is a synchronous global read and the CTA
    stalls on every page instead of on every ``STAGES``-th one."""
    sass = _sass()
    assert re.search(r"\bLDGSTS\b", sass), (
        "no cp.async in the SASS — the page loader is synchronous"
    )
    assert re.search(r"LDGSTS\.E\.BYPASS\.128", sass), (
        "cp.async is not 128-bit / L1-bypassing: the KV cache is streamed once and never reused "
        "in a launch, so an L1 line of it evicts something that would have been"
    )


@pytest.mark.drydock
def test_sass_reads_shared_memory_128_bits_at_a_time() -> None:
    """GQA multiplies every shared read by ``GROUP_SIZE``: the CTA pulls 64 KB out of smem for
    every 8 KB it pulls from HBM. Element-wise ``LDS.U16`` would make the shared pipe the wall of
    a kernel whose exit number is stated in % of *HBM*."""
    sass = _sass()
    assert re.search(r"\bLDS\.128\b", sass), "the K/V tile is not read 16 bytes at a time"
    u16 = len(re.findall(r"\bLDS\.U16\b", sass))
    assert u16 == 0, f"{u16} scalar 2-byte shared loads survived in the inner loop"


@pytest.mark.drydock
def test_sass_reduces_the_score_with_a_butterfly() -> None:
    """The 16 lanes spanning one head must all-reduce the dot product; every lane needs the score
    to rescale its own slice of the accumulator, so a one-way reduction is not enough."""
    assert re.search(r"\bSHFL\.BFLY\b", _sass()), (
        "no butterfly shuffle — the qk reduction is missing"
    )


@pytest.mark.drydock
def test_sass_uses_the_hardware_base_two_exponential() -> None:
    """``MUFU.EX2`` is ``ex2.approx``. Its absence means the inline asm was optimised into a
    library call, i.e. the numerics now depend on ``--use_fast_math``."""
    sass = _sass()
    assert re.search(r"\bMUFU\.EX2\b", sass), "no ex2 in the SASS"
    assert re.search(r"\bMUFU\.LG2\b", sass), "no lg2 in the SASS — lse is not being computed"


@pytest.mark.drydock
def test_sass_has_no_local_memory_traffic() -> None:
    """``LDL``/``STL`` mean the per-tile score array or the accumulator spilled."""
    spills = re.findall(r"\b(LDL|STL)\b", _sass())
    assert not spills, f"{len(spills)} local-memory accesses in the SASS: something spilled"


@pytest.mark.drydock
def test_ptxas_report_is_clean_and_matches_the_python_constants() -> None:
    """0 spill bytes, 16 KB of static shared memory, and the register count the occupancy estimate
    is built on. The last one is the point: ``REGS_PER_THREAD`` feeds ``ctas_per_sm``, which feeds
    ``max_grid_size``, which is an input to the partition decision."""
    p = _DRYDOCK / f"{_STEM}.sm_90a.ptxas.txt"
    if not p.is_file():
        pytest.skip(f"no ptxas report at {p} — run: infra/drydock.sh compile {SOURCE}")
    report = p.read_text(encoding="utf-8", errors="replace")
    spill = re.search(r"(\d+) bytes spill stores", report)
    assert spill and int(spill.group(1)) == 0, f"register spill in {p.name}: {report[-400:]}"
    smem = re.search(r"(\d+) bytes smem", report)
    assert smem and int(smem.group(1)) == SMEM_BYTES, (
        f"{smem.group(1) if smem else 'no'} bytes smem in the report, {SMEM_BYTES} in Python"
    )
    regs = [int(m) for m in re.findall(r"Used (\d+) registers", report)]
    assert regs, "no register count in the ptxas report"
    # Hopper allocates registers in multiples of 8; REGS_PER_THREAD is that rounding.
    assert max(regs) <= REGS_PER_THREAD, (
        f"the kernel now uses {max(regs)} registers but ctas_per_sm() is estimated at "
        f"{REGS_PER_THREAD} — the occupancy the partition is planned against is stale"
    )


# =============================================================================================
# gpu — correctness against the oracle and against the floor. H100 only.
# =============================================================================================


def _require_tolerance() -> float:
    if TOL_ABS is None:
        pytest.fail(
            "TOL_ABS is unset. The correctness gate cannot run without a tolerance, and picking "
            "one is the rung's numerics lesson: the split re-associates the softmax sum, so the "
            "merged result is not bit-identical to the unchunked one even in exact arithmetic. "
            "Set it here and write the one-line argument into experiments/K2/A-R3/spec.md's "
            "'Correctness gate' line before the first measured run."
        )
    return TOL_ABS


def _gpu_problem(seq_lens, *, kv_heads=4, seed=0):
    q, k, v, table = _tiny_problem(seq_lens, kv_heads=kv_heads, seed=seed)
    return (
        q.bfloat16().cuda(),
        k.bfloat16().cuda(),
        v.bfloat16().cuda(),
        table.to("cuda"),
        (q, k, v, table),
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "seq_lens",
    [[16] * 4, [8192], [8192, 17, 1, 4095], [128] * 64],
    ids=["one-page", "spec-ctx", "ragged", "spec-batch"],
)
@pytest.mark.parametrize("chunk_pages", [1, 8, 4096], ids=["fine", "upstream-floor", "unsplit"])
def test_matches_oracle_at_an_explicit_split(seq_lens, chunk_pages) -> None:
    """Correctness at hand-chosen splits — deliberately NOT through :func:`plan_partition`.

    The kernel and the reduce are agent work and are gated here; the split count is Huy's and is
    gated by the hole test. Coupling the two would mean the kernel could not be verified until the
    partition existed, which is exactly backwards: on the box he wants a known-correct kernel
    before he starts choosing a number for it.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    q, k, v, table, cpu = _gpu_problem(seq_lens)
    plan = PartitionPlan.from_chunk_size(table.pages_per_request(), chunk_pages).require_valid()
    o, lse = a_r3_paged_decode(q, k, v, table, plan=plan)
    ref_o, ref_lse = reference_paged_decode(*cpu[:3], cpu[3])
    tol = _require_tolerance()
    assert o.dtype is torch.bfloat16 and lse.dtype is torch.float32
    err = (o.float().cpu() - ref_o).abs().max().item()
    assert err <= tol, (
        f"max|Δo| = {err:.4e} > {tol:.4e}. The two structured failures to look for first are a "
        f"wrong last-page mask (error confined to the requests whose length is not a multiple of "
        f"{PAGE_SIZE}) and a base mismatch between the partial and the merge (error grows with the "
        f"chunk count). Print the per-request error before widening this."
    )
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
def test_split_and_unsplit_agree_with_each_other() -> None:
    """The split's own contribution to the error, isolated from the kernel's.

    Same kernel, same data, two chunk counts. Any difference here is the re-association the merge
    introduces and nothing else — which is the number the tolerance argument needs.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    q, k, v, table, _ = _gpu_problem([8192] * 2)
    pages = table.pages_per_request()
    one, _ = a_r3_paged_decode(q, k, v, table, plan=PartitionPlan.from_chunk_size(pages, 4096))
    many, _ = a_r3_paged_decode(q, k, v, table, plan=PartitionPlan.from_chunk_size(pages, 8))
    torch.testing.assert_close(one.float(), many.float(), rtol=2e-2, atol=_require_tolerance())


@pytest.mark.gpu
def test_merge_kernel_matches_its_python_twin_on_adversarial_partials() -> None:
    """The reduce alone, on partials a real decode will not produce on demand: chunk maxima 100
    apart, an empty chunk, and a single-chunk request."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from scratch_llm.kernels.attention.decode.paged_split_kv import a_r3_merge_states

    torch.manual_seed(0)
    heads = 8
    tmp_o = torch.randn(5, heads, HEAD_DIM, dtype=torch.float32, device="cuda")
    tmp_lse = torch.tensor(
        [[-60.0] * heads, [60.0] * heads, [0.0] * heads, [-math.inf] * heads, [1.0] * heads],
        dtype=torch.float32,
        device="cuda",
    )
    o_indptr = torch.tensor([0, 2, 4, 5], dtype=torch.int32, device="cuda")
    o, lse = a_r3_merge_states(tmp_o, tmp_lse, o_indptr)
    want_o, want_lse = merge_states_reference(tmp_o.cpu(), tmp_lse.cpu(), [0, 2, 4, 5])
    torch.testing.assert_close(o.float().cpu(), want_o, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse.cpu(), want_lse, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
def test_matches_flashinfer_at_the_identical_page_table() -> None:
    """The floor, as a correctness check before it is a performance one.

    A floor measured on a different table is a comparison to nothing, so the same
    ``indptr``/``indices``/``last_page_len`` objects go into both. ``use_tensor_cores`` is passed
    explicitly because the wrapper fronts two different kernels (decode.py:1753 vs :1820) and the
    floor must name which one it measured.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    flashinfer = pytest.importorskip("flashinfer")
    from scratch_llm.kernels.attention.decode.paged_split_kv import flashinfer_reference

    assert flashinfer is not None
    q, k, v, table, _ = _gpu_problem([8192, 4096, 17, 128])
    wrapper, (kc, vc) = flashinfer_reference(q, k, v, table, use_tensor_cores=False)
    floor_o = wrapper.run(q, (kc, vc))
    plan = PartitionPlan.from_chunk_size(table.pages_per_request(), 8).require_valid()
    ours, _ = a_r3_paged_decode(q, k, v, table, plan=plan)
    err = (ours.float() - floor_o.float()).abs().max().item()
    assert err <= _require_tolerance(), (
        f"max|Δ| vs FlashInfer = {err:.4e}. Both read the same page table, so a difference here "
        f"is arithmetic, not indexing: check the softmax base and the last-page mask first."
    )


@pytest.mark.gpu
def test_rejects_bad_inputs() -> None:
    """The guard clauses, on the device where they can actually be reached."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    q, k, v, table, _ = _gpu_problem([64])
    plan = PartitionPlan.from_chunk_size(table.pages_per_request(), 2)
    with pytest.raises(TypeError, match="bfloat16"):
        a_r3_paged_decode(q.float(), k, v, table, plan=plan)
    with pytest.raises(ValueError, match="GQA"):
        a_r3_paged_decode(q[:, :4], k, v, table, plan=plan)
    other = PartitionPlan.from_chunk_size([999], 2)
    with pytest.raises(ValueError, match="different page counts"):
        a_r3_paged_decode(q, k, v, table, plan=other)
