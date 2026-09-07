"""K2/A-R3 — paged decode, split-KV: partial (m, l, o) per chunk + a log-sum-exp reduce.

At B=64, ctx 8192, one query token, there is no arithmetic intensity: the kernel reads 512 MiB of
KV cache and writes 512 KiB of output. It is a streaming problem wearing an attention costume, and
the exit is stated as a fraction of **measured HBM bandwidth** for exactly that reason. Two things
decide whether the bandwidth is reachable at all:

1. **Enough CTAs to fill the machine.** One CTA per (request, KV head) is ``64 x 4 = 256`` on the
   plan's shape — more than an H100 has SMs, but each of those CTAs then walks 8192 tokens
   serially, so the tail of the longest request is the kernel. Splitting the KV range into chunks
   turns depth into width. That decision — how many chunks — is the rung, and it is the hole below.
2. **K/V read once per GQA group.** With GQA 8:1 the eight query heads of a group attend the same
   keys. Reading them once into shared memory and letting all eight consume from there divides HBM
   traffic by 8; reading per query head is a kernel that is 8x off the roofline and looks correct.
   Here that is structural: ``threadIdx.y`` *is* the query head within the group, and every
   ``threadIdx.y`` reads back the same shared tile (``csrc/attention/a_r3_paged_decode_sm90.cu``).

**What is not a hole.** The reduce — combining the chunks' ``(m, l, o)`` into the final output — is
the log-sum-exp merge, ordinary arithmetic that upstream also treats as library code
(``include/flashinfer/attention/state.cuh:52-64``). It is written here, in CUDA and again in Python
as :func:`merge_states_reference`, and the CPU suite proves the two agree with a direct
un-chunked reference at small shapes.

**Numerics: base 2, on both sides of the merge.** Logits are pre-scaled by ``sm_scale * log2(e)``
and exponentiated with ``exp2``; a partial's stored ``lse`` is ``m + log2(d)``. This is FlashInfer's
convention (``variants.cuh:54``, ``state.cuh:45``) and it is load bearing: the merge kernel assumes
the same base and an ln-based partial produces a merged output that is wrong and finite
(``cascade.cuh:641`` says so in as many words). The CPU suite greps both kernels for ``exp2f`` and
for the absence of a bare ``expf(``.

**What crosses to global per chunk is two floats per (chunk, head), not three.** The partial kernel
writes ``o`` already normalised by ``1/d`` plus a single ``lse``; the merge rebuilds the pair by
treating the stored ``lse`` as the state's ``m`` with ``d = 1`` — upstream's
``st.merge(v, s, /*other_d=*/1)`` (``cascade.cuh:446``). Halves the partial buffer and halves the
traffic the reduce has to move, which matters precisely when the split count is high.

Page table: :mod:`scratch_llm.kernels.attention.decode.page_table`, NHD, FlashInfer's exact layout.
Not the HND ``(n_blocks, H_kv, 16, d)`` pool that ``decode/paged.py``'s Triton kernel uses.

Kernel source: ``csrc/attention/a_r3_paged_decode_sm90.cu``.
Spec, floor and kill rule: ``experiments/K2/A-R3/spec.md``   ·   Map: ``experiments/K2/A-R3/map.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scratch_llm.kernels.attention.decode.page_table import PageTable, gather_kv
from scratch_llm.kernels.common.arch import require_arch
from scratch_llm.kernels.common.hopper_contracts import ARCH, check_smem_budget

#: The rung identity, in one place, so wrapper, tests, bench and dry-dock registry cannot drift.
RUNG = "K2/A-R3"
SOURCE = "a_r3_paged_decode_sm90.cu"
SYMBOL = "a_r3_paged_decode"
MERGE_SYMBOL = "a_r3_merge_states"
ARCH_CC = (9, 0)

# ---------------------------------------------------------------------------------------------
# The shape, frozen at the plan's row (SIXTY_DAYS_SIX_LADDERS.md §05 K2 A-R3): B=64, ctx 8k,
# page 16, hdim 128, GQA 8:1. These are `#define`s in the .cu, not template parameters, and the
# launcher rejects anything else by name. That is a deliberate scope choice, not laziness: the
# exit number is stated at this shape, a templated kernel would tempt a retune between the
# correctness run and the measured run, and every constant below falls out of the four numbers.
# A CPU test regexes the .cu's defines against these.
# ---------------------------------------------------------------------------------------------
HEAD_DIM = 128
PAGE_SIZE = 16
GROUP_SIZE = 8  # query heads per KV head (GQA 8:1)
ELEM_BYTES = 2  # bf16 KV — the floor's dtype, so not a variant

#: 16 bytes per lane per load: the widest global access the ISA has, and what makes a 16-token
#: page row (256 B) exactly one coalesced transaction per 16 lanes.
VEC_ELEMS = 16 // ELEM_BYTES  # 8 bf16 elements

#: Thread block: ``(BDX, BDY)`` = (lanes across the head dim, query heads of one GQA group).
#: ``BDY == GROUP_SIZE`` is the whole GQA trick — the head index is a thread index, so the shared
#: K/V tile is read GROUP_SIZE times from smem and once from HBM.
BDX = HEAD_DIM // VEC_ELEMS  # 16
BDY = GROUP_SIZE  # 8
THREADS_PER_CTA = BDX * BDY  # 128

#: One pipeline stage stages exactly one page. Choosing tile == page is what removes the
#: page-boundary divmod from the inner loop entirely: a tile never straddles two pages, so the
#: page id is looked up once per tile instead of once per token.
TILE_TOKENS = PAGE_SIZE
STAGES = 2  # cp.async depth; NUM_STAGES_SMEM is also 2 upstream for sm >= 80 (utils.cuh:330-332)

BYTES_PER_STAGE = 2 * TILE_TOKENS * HEAD_DIM * ELEM_BYTES  # K tile + V tile = 8192
SMEM_BYTES = BYTES_PER_STAGE * STAGES  # 16384 — comfortably under the 48 KB no-opt-in ceiling

#: The reduce kernel's block: one warp per (request, query head). 32 lanes x 4 floats covers the
#: 128-wide head with fully coalesced 128 B loads, and the merge state (m, d) is scalar so no
#: cross-lane communication is needed at all.
MERGE_THREADS = 32
MERGE_ELEMS_PER_THREAD = HEAD_DIM // MERGE_THREADS  # 4

#: Partial buffers are **fp32**, both of them. The partial ``o`` is an intermediate that will be
#: rescaled by a factor as large as the ratio between two chunks' maxima before it is summed;
#: storing it in bf16 spends 8 mantissa bits on a value that is not the answer. FlashInfer sizes
#: this buffer in f32 and then writes it as ``DTypeO`` (``scheduler.cuh:486`` vs
#: ``csrc/batch_decode.cu:231``), which is a wart the map flags — this rung does not copy it.
#: The cost is 4 B/elem on ``num_chunks * num_qo_heads * head_dim``, ~4 MiB at 4 chunks/request on
#: the plan shape, against 512 MiB of KV read. Free.
PARTIAL_DTYPE = torch.float32

#: Hopper per-SM occupancy ceilings, CUDA C Programming Guide "Compute Capabilities" table.
#: Kept local rather than pushed into ``hopper_contracts`` (which carries the shared-memory and
#: register ceilings this module imports) because only the partition estimator needs them.
SM90_MAX_BLOCKS_PER_SM = 32
SM90_MAX_WARPS_PER_SM = 64

#: log2(e). Folded into the softmax scale so the kernel's exponential is ``exp2``, which is one
#: hardware instruction (``MUFU.EX2``) rather than a multiply plus one.
LOG2E = 1.4426950408889634


# =============================================================================================
# Partition — the plan, which is written here, and the decision, which is the hole
# =============================================================================================


@dataclass(frozen=True)
class PartitionPlan:
    """A split-KV work list: which (request, page-range) pair each CTA gets.

    One row per work item, in **request-major order**, plus the prefix ``o_indptr`` over
    chunks-per-request. Request-major is not cosmetic: the reduce kernel merges
    ``tmp_o[o_indptr[b] : o_indptr[b+1]]`` as a contiguous span, so a work list that interleaved
    requests would merge one request's chunks into another's output — a wrong answer with no
    fault, at full speed. :meth:`validate` asserts it.

    Mirrors the three device-side arrays FlashInfer materialises on the host in
    ``DecodeSplitKVIndptr`` (``scheduler.cuh:349-361``).
    """

    chunk_size_pages: int
    pages_per_request: tuple[int, ...]
    request_indices: tuple[int, ...]
    chunk_indices: tuple[int, ...]
    o_indptr: tuple[int, ...]

    @classmethod
    def from_chunk_size(
        cls, pages_per_request: Sequence[int], chunk_size_pages: int
    ) -> PartitionPlan:
        """Expand one chunk size into the full work list.

        This is the mechanical half of the partition and it is agent work; the number handed in is
        the hole. Chunk ``c`` of request ``i`` covers pages ``[c*C, min((c+1)*C, n_i))`` of that
        request's own page list — a *logical* range, resolved to physical page ids through
        ``indices`` inside the kernel.
        """
        if chunk_size_pages < 1:
            raise ValueError(f"chunk_size_pages must be >= 1, got {chunk_size_pages}")
        req: list[int] = []
        chunk: list[int] = []
        o_indptr = [0]
        for i, n_pages in enumerate(pages_per_request):
            # A request with zero pages still gets one work item, whose chunk is empty and whose
            # partial is the empty state. Dropping it instead would make o_indptr[i] == o_indptr[i+1]
            # and the reduce would emit an uninitialised row rather than a zero one.
            n_chunks = max(1, -(-int(n_pages) // chunk_size_pages))
            req.extend([i] * n_chunks)
            chunk.extend(range(n_chunks))
            o_indptr.append(o_indptr[-1] + n_chunks)
        return cls(
            chunk_size_pages=int(chunk_size_pages),
            pages_per_request=tuple(int(n) for n in pages_per_request),
            request_indices=tuple(req),
            chunk_indices=tuple(chunk),
            o_indptr=tuple(o_indptr),
        )

    @property
    def batch_size(self) -> int:
        return len(self.pages_per_request)

    @property
    def num_chunks_total(self) -> int:
        """Work items = CTAs along ``gridDim.x``. The reduce's fan-in is this over the batch."""
        return len(self.request_indices)

    def num_ctas(self, num_kv_heads: int) -> int:
        """The whole grid: one CTA per (work item, KV head). ``gdy = num_kv_heads``, as upstream."""
        return self.num_chunks_total * num_kv_heads

    def chunks_of(self, request: int) -> int:
        return self.o_indptr[request + 1] - self.o_indptr[request]

    def validate(self) -> list[str]:
        """Violations of the work list's contract (empty == usable). Returned, not raised."""
        errs: list[str] = []
        if self.chunk_size_pages < 1:
            errs.append(f"chunk_size_pages={self.chunk_size_pages} must be >= 1")
        n = len(self.request_indices)
        if len(self.chunk_indices) != n:
            errs.append(f"request_indices has {n} entries, chunk_indices {len(self.chunk_indices)}")
        if len(self.o_indptr) != self.batch_size + 1:
            errs.append(
                f"o_indptr has {len(self.o_indptr)} entries for batch {self.batch_size} "
                f"(expected batch + 1)"
            )
            return errs
        if self.o_indptr[0] != 0 or self.o_indptr[-1] != n:
            errs.append(f"o_indptr must run 0..{n}, got {self.o_indptr[0]}..{self.o_indptr[-1]}")
        if any(a > b for a, b in zip(self.request_indices, self.request_indices[1:], strict=False)):
            errs.append(
                "request_indices is not non-decreasing — the reduce merges "
                "tmp_o[o_indptr[b]:o_indptr[b+1]] as one contiguous span, so an interleaved work "
                "list mixes one request's chunks into another's output, silently"
            )
        for i, n_pages in enumerate(self.pages_per_request):
            got = self.chunks_of(i)
            want = max(1, -(-n_pages // max(1, self.chunk_size_pages)))
            if got != want:
                errs.append(
                    f"request {i} has {n_pages} pages and chunk size {self.chunk_size_pages}, "
                    f"so it needs {want} chunks; the work list gives it {got}"
                )
            span = self.request_indices[self.o_indptr[i] : self.o_indptr[i + 1]]
            if any(r != i for r in span):
                errs.append(f"o_indptr[{i}] does not point at request {i}'s own work items")
            ids = self.chunk_indices[self.o_indptr[i] : self.o_indptr[i + 1]]
            if list(ids) != list(range(got)):
                errs.append(f"request {i}'s chunk ids are {list(ids)}, expected 0..{got - 1}")
        return errs

    def require_valid(self) -> PartitionPlan:
        errs = self.validate()
        if errs:
            raise ValueError("invalid partition plan:\n  " + "\n  ".join(errs))
        return self

    def covered_pages(self, request: int) -> list[tuple[int, int]]:
        """The ``[start, end)`` logical page range of every chunk of ``request``.

        The property a test asserts: the ranges tile ``[0, n_pages)`` exactly — no page attended
        twice (which double-counts it in the softmax denominator) and none skipped.
        """
        n_pages = self.pages_per_request[request]
        c = self.chunk_size_pages
        return [(i * c, min((i + 1) * c, n_pages)) for i in range(self.chunks_of(request))]

    def to_tensors(self, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor, Tensor]:
        """``(request_indices, chunk_indices, o_indptr)`` as int32 — what the launcher takes."""
        mk = lambda xs: torch.tensor(xs, dtype=torch.int32, device=device)  # noqa: E731
        return mk(self.request_indices), mk(self.chunk_indices), mk(self.o_indptr)


#: Registers per thread the partial kernel actually uses, from the dry-dock ptxas report
#: (``experiments/K2/A-R3/drydock/a_r3_paged_decode_sm90.sm_90a.ptxas.txt``: "Used 71 registers",
#: 0 spill), rounded up to Hopper's allocation granularity of 8. Not a guess and not a target — if
#: the kernel changes, the report changes, and this constant is the CPU test's anchor to it.
REGS_PER_THREAD = 72


def ctas_per_sm(*, regs_per_thread: int = REGS_PER_THREAD, arch_name: str = "sm_90a") -> int:
    """How many of this kernel's CTAs can be resident on one SM — the occupancy the split needs.

    The analytic form of ``cudaOccupancyMaxActiveBlocksPerMultiprocessor``, which is what upstream
    calls (``scheduler.cuh:180``) and which does not exist on a laptop. It is an *estimate*: the
    real number also depends on how ptxas allocated registers on the day, which is why
    ``regs_per_thread`` is a parameter fed from the dry-dock report rather than a literal. That the
    threshold moves when one register does is precisely why the split is a decision and not a
    formula — see :func:`plan_partition`.
    """
    a = ARCH[arch_name]
    by_smem = a.smem_per_sm // SMEM_BYTES
    by_regs = a.regs_per_sm // (regs_per_thread * THREADS_PER_CTA)
    by_warps = SM90_MAX_WARPS_PER_SM // (THREADS_PER_CTA // 32)
    return max(1, min(by_smem, by_regs, by_warps, SM90_MAX_BLOCKS_PER_SM))


#: The hole's message, kept out of the ``raise`` so the sentinel fits on one line — see the
#: comment at the raise for why that matters.
_HOLE = (
    "choose chunk_size_pages from the per-request page counts, gdy=num_kv_heads and max_grid_size. "
    "Too few chunks leaves SMs idle behind an 8192-token serial walk; too many lets the reduce's "
    "fan-in dominate. Upstream's binary search is scheduler.cuh:73-98, with a floor of "
    "max(128/page_size, 1) pages at :199. Spec: experiments/K2/A-R3/spec.md"
)


def plan_partition(
    pages_per_request: Sequence[int],
    *,
    num_kv_heads: int,
    max_grid_size: int,
    page_size: int = PAGE_SIZE,
) -> PartitionPlan:
    """Choose the split: how many pages each chunk covers. **This is the rung's hole.**

    Arguments are deliberately the same three FlashInfer's own chooser takes
    (``PartitionPagedKVCacheBinarySearchMinNumPagePerBatch``, ``scheduler.cuh:73-98``), so the two
    can be read side by side: the per-request page counts, ``gdy = num_kv_heads``, and
    ``max_grid_size = num_sm * ctas_per_sm``. Return a :class:`PartitionPlan`; building the work
    list from a chunk size is already written (:meth:`PartitionPlan.from_chunk_size`).

    The two walls the number sits between, both real and both measurable:

    * **Too few chunks and SMs idle.** ``batch * num_kv_heads`` CTAs is 256 on the plan's shape.
      If that is already >= ``max_grid_size`` upstream does not split at all (``scheduler.cuh:183``)
      — but it is comparing against ``num_blocks_per_sm * num_sm``, which for a kernel this small
      is in the low thousands, so at B=64 the branch almost certainly *does* split. Read it on the
      box (``plan()``'s returned vector, element 9 is ``split_kv``, ``scheduler.cuh:420``); do not
      assume it.
    * **Too many and the reduce dominates.** Each extra chunk costs the reduce one more
      ``(head_dim + 1)`` fp32 read per query head, and costs the decode a re-read of nothing but
      buys shorter tails. Upstream refuses to cut finer than ``max(128 / page_size, 1)`` pages
      (``scheduler.cuh:199``) = 8 pages = 128 tokens at page 16. Whether that floor is right for
      *this* kernel — whose tile is one page, not 128 tokens — is the question the rung answers.

    Constraints the returned plan must satisfy (:meth:`PartitionPlan.validate` checks the shape of
    the work list; these are on the number itself):
      * ``1 <= chunk_size_pages <= max(pages_per_request)``;
      * every request gets at least one chunk, including a zero-page one;
      * the chunk ranges tile each request's page list exactly (:meth:`covered_pages`).
    """
    # HUY: the split count, chunk_size_pages from (pages per request, gdy, max_grid_size) — spec: experiments/K2/A-R3/spec.md — fill before A-R3
    #
    # The message must BEGIN on the raise's own line. tests/conftest.py decides whether a hole is
    # open by grepping the source for the sentinel spelled out in its own docstring, so a formatter
    # that wraps the string onto the next line makes the hole invisible and flips its guard test
    # green — which is why the detail lives in _HOLE above rather than inline, and why this comment
    # does not quote the sentinel (a copy of it here would keep the hole "open" after it is filled).
    # test_the_hole_sentinel_survives_the_formatter is the CPU gate on both halves.
    raise NotImplementedError("HUY: K2/A-R3 partition logic — " + _HOLE)


# =============================================================================================
# Partial buffers — sized from the split count
# =============================================================================================


def partial_buffers(
    num_chunks_total: int,
    num_qo_heads: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """``(tmp_o [T, H_q, D] fp32, tmp_lse [T, H_q] fp32)`` — what the chunks write and the reduce reads.

    Allocated from the split count, so the split decision is also a memory decision: at the plan's
    shape every extra chunk per request costs ``64 * 32 * 129 * 4 B`` = 1.0 MiB of workspace and
    the same again in reduce traffic.
    """
    return (
        torch.empty((num_chunks_total, num_qo_heads, HEAD_DIM), dtype=PARTIAL_DTYPE, device=device),
        torch.empty((num_chunks_total, num_qo_heads), dtype=PARTIAL_DTYPE, device=device),
    )


def partial_bytes(num_chunks_total: int, num_qo_heads: int) -> int:
    """Workspace footprint of the partials, in bytes — the number the split trades against."""
    per = PARTIAL_DTYPE.itemsize
    return num_chunks_total * num_qo_heads * (HEAD_DIM + 1) * per


# =============================================================================================
# The reduce, in Python — the oracle for the CUDA merge kernel
# =============================================================================================


def merge_states_reference(tmp_o: Tensor, tmp_lse: Tensor, o_indptr: Sequence[int]):
    """Merge per-chunk ``(o_normalised, lse)`` into ``(o, lse)`` per request. Base 2 throughout.

    The log-sum-exp merge, written the way the kernel writes it rather than the way a textbook
    does, so the two can be compared line by line:

        m   = max_c lse_c
        f_c = 2^(lse_c - m)
        o   = sum_c f_c * o_c / sum_c f_c
        lse = m + log2(sum_c f_c)

    ``o_c`` is already divided by its own chunk's ``d``, which is why the weight is ``f_c`` and not
    ``d_c * f_c`` — the ``d`` is inside ``lse_c = m_c + log2(d_c)``. This is upstream's
    ``merge(v, s, /*other_d=*/1)`` (``cascade.cuh:446``) written out.

    Subtracting the max is not a nicety. Chunk maxima differ by the spread of the logits across the
    whole context; at ctx 8192 a difference of 100 in log2 units is ordinary, and ``2^100``
    overflows fp32 at the second chunk.
    """
    o_idx = list(int(x) for x in o_indptr)
    batch = len(o_idx) - 1
    heads = tmp_o.shape[1]
    o = torch.zeros((batch, heads, tmp_o.shape[2]), dtype=torch.float32, device=tmp_o.device)
    lse = torch.full((batch, heads), -math.inf, dtype=torch.float32, device=tmp_o.device)
    for b in range(batch):
        lo, hi = o_idx[b], o_idx[b + 1]
        if hi <= lo:
            continue
        s = tmp_lse[lo:hi].float()  # [C, H]
        m = s.max(dim=0).values  # [H]
        # An all-empty request has m = -inf and s - m = nan; nan_to_num maps it to 0 weight, the
        # same thing the kernel's fmaxf(x, -inf) guard does (state.cuh:57-59).
        f = torch.exp2(torch.nan_to_num(s - m, nan=-math.inf))  # [C, H]
        d = f.sum(dim=0)  # [H]
        num = (f.unsqueeze(-1) * tmp_o[lo:hi].float()).sum(dim=0)  # [H, D]
        o[b] = num / d.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
        lse[b] = torch.where(d > 0, m + torch.log2(d), torch.full_like(d, -math.inf))
    return o, lse


# =============================================================================================
# The oracle — paged decode, de-paged, in fp32 on the CPU
# =============================================================================================


def default_sm_scale(head_dim: int = HEAD_DIM) -> float:
    return 1.0 / math.sqrt(head_dim)


def reference_paged_decode(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    table: PageTable,
    *,
    sm_scale: float | None = None,
):
    """``softmax(q K^T scale) V`` for one query token per request, read through the page table.

    The correctness ground truth: no chunking, no online softmax, no split. It de-pages through
    :func:`~scratch_llm.kernels.attention.decode.page_table.gather_kv` — so it *does* exercise the
    page table, which is the point — and then does the dense thing in fp32.

    ``q`` is ``[B, H_q, D]``. Returns ``(o [B, H_q, D] fp32, lse [B, H_q] fp32, base 2)``.
    """
    scale = default_sm_scale(q.shape[-1]) if sm_scale is None else sm_scale
    batch, heads, dim = q.shape
    kv_heads = k_cache.shape[2]
    group = heads // kv_heads
    o = torch.zeros((batch, heads, dim), dtype=torch.float32)
    lse = torch.full((batch, heads), -math.inf, dtype=torch.float32)
    for b in range(batch):
        k, v = gather_kv(table, k_cache, v_cache, b)
        if k.shape[0] == 0:
            continue
        for h in range(heads):
            kh = h // group
            # Base 2, to match the kernel: scale by sm_scale * log2(e) and exponentiate with exp2.
            s = (k[:, kh, :].float() @ q[b, h].float()) * (scale * LOG2E)  # [L]
            m = s.max()
            w = torch.exp2(s - m)
            d = w.sum()
            o[b, h] = (w.unsqueeze(-1) * v[:, kh, :].float()).sum(dim=0) / d
            lse[b, h] = m + torch.log2(d)
    return o, lse


def reference_partials(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    table: PageTable,
    plan: PartitionPlan,
    *,
    sm_scale: float | None = None,
):
    """The per-chunk ``(o_normalised, lse)`` the CUDA partial kernel is supposed to produce.

    Same arithmetic as :func:`reference_paged_decode`, restricted to one chunk's page range. Its
    only job is to let the CPU suite prove, without a GPU, that chunking-then-merging equals not
    chunking — which is the entire premise of split-KV and the one thing that cannot be discovered
    from a benchmark.
    """
    scale = default_sm_scale(q.shape[-1]) if sm_scale is None else sm_scale
    heads = q.shape[1]
    kv_heads = k_cache.shape[2]
    group = heads // kv_heads
    tmp_o, tmp_lse = partial_buffers(plan.num_chunks_total, heads)
    tmp_o.zero_()
    tmp_lse.fill_(-math.inf)
    page = table.page_size
    for b in range(plan.batch_size):
        k, v = gather_kv(table, k_cache, v_cache, b)
        seq = k.shape[0]
        for j, (p0, p1) in enumerate(plan.covered_pages(b)):
            w_idx = plan.o_indptr[b] + j
            t0, t1 = p0 * page, min(p1 * page, seq)
            if t1 <= t0:
                continue
            for h in range(heads):
                kh = h // group
                s = (k[t0:t1, kh, :].float() @ q[b, h].float()) * (scale * LOG2E)
                m = s.max()
                w = torch.exp2(s - m)
                d = w.sum()
                tmp_o[w_idx, h] = (w.unsqueeze(-1) * v[t0:t1, kh, :].float()).sum(dim=0) / d
                tmp_lse[w_idx, h] = m + torch.log2(d)
    return tmp_o, tmp_lse


# =============================================================================================
# The kernel — loader, guards, launcher
# =============================================================================================

_CU = Path(__file__).resolve().parents[5] / "csrc" / "attention" / SOURCE


@lru_cache(maxsize=1)
def _module() -> Any:
    """AOT first (what a rented box builds once at bootstrap), JIT second.

    A JIT compile inside a timing window is a measurement of nvcc, so the box path must be AOT;
    the JIT fallback exists so one rung can be iterated on without rebuilding the extension.

    ``Any`` for the same reason ``_k1_loader.load_rung`` uses it: what comes back is an opaque
    pybind handle whose symbols exist only after a build, so there is nothing for a static checker
    to resolve ``a_r3_paged_decode`` against.
    """
    try:
        # The AOT extension only exists after a `[gpu-aot]` build on a GPU box; there is no
        # stub for a static checker to see, which is what the ignore is for.
        from scratch_llm import (
            _scratch_llm_kernels as _ext,  # type: ignore[attr-defined]  # noqa: PLC0415
        )

        if hasattr(_ext, SYMBOL):
            return _ext
    except ImportError:
        pass

    from torch.utils.cpp_extension import load  # noqa: PLC0415

    # sm_90a, not sm_90: the trailing 'a' selects the accelerated ISA the rest of the K2 ladder
    # is built for, and mixing the two in one extension is a link-time surprise on the box.
    return load(
        name="k2_a_r3_paged_decode",
        sources=[str(_CU)],
        extra_cuda_cflags=["-O3", "-lineinfo", "-arch=sm_90a"],
        verbose=False,
    )


def unsupported_reason(
    *, head_dim: int, page_size: int, num_qo_heads: int, num_kv_heads: int
) -> str | None:
    """Why this configuration cannot run on this rung, or ``None`` if it can.

    Every one of these is a frozen ``#define`` in the ``.cu``, not a missing feature: the rung's
    exit number is quoted at one shape and a kernel that silently accepted another would be
    measuring something the plan did not commit to.
    """
    if head_dim != HEAD_DIM:
        return f"head_dim must be {HEAD_DIM} at this rung (frozen #define); got {head_dim}"
    if page_size != PAGE_SIZE:
        return (
            f"page_size must be {PAGE_SIZE} at this rung: the pipeline stage IS one page, which is "
            f"what removes the page-boundary divmod from the inner loop; got {page_size}"
        )
    if num_kv_heads < 1 or num_qo_heads != num_kv_heads * GROUP_SIZE:
        return (
            f"this rung is GQA {GROUP_SIZE}:1 — blockDim.y is the query head within the group, so "
            f"num_qo_heads must be {GROUP_SIZE} * num_kv_heads; got {num_qo_heads} and "
            f"{num_kv_heads}"
        )
    return None


def smem_budget_errors() -> list[str]:
    """The shared-memory verdict for this kernel's stage arithmetic, from ``hopper_contracts``.

    Empty is the answer we want and it carries a claim: 16 KB is under the 48 KB a launch gets for
    free, so this kernel needs **no** ``cudaFuncSetAttribute`` opt-in — unlike K1/H-R2, whose 64 KB
    does. A stage count that quietly crossed that line would launch-fail on the box and nowhere
    else.
    """
    return check_smem_budget(
        bytes_per_stage=BYTES_PER_STAGE, stages=STAGES, arch=ARCH["sm_90a"], extra_bytes=0
    )


def a_r3_paged_decode(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    table: PageTable,
    *,
    plan: PartitionPlan | None = None,
    sm_scale: float | None = None,
):
    """Split-KV paged decode. Returns ``(o [B, H_q, D] bf16, lse [B, H_q] fp32, base 2)``.

    ``q`` is ``[B, H_q, D]`` bf16, one query token per request. ``k_cache``/``v_cache`` are NHD
    ``[num_pages, page_size, H_kv, D]`` bf16. ``table`` is the page table both this kernel and the
    FlashInfer floor are given.

    ``plan`` is the split. Passing one explicitly bypasses :func:`plan_partition` entirely, which
    is what makes the kernel and the reduce testable against the oracle *before* the partition
    decision exists — the correctness gate does not wait on the hole. ``None`` asks for the
    default, which is the hole.

    Raises ``RuntimeError`` off Hopper and ``NotImplementedError`` while the partition is unfilled.
    """
    require_arch(ARCH_CC, fn_name="a_r3_paged_decode")
    if q.dim() != 3:
        raise ValueError(f"q must be [B, H_q, D], got {tuple(q.shape)}")
    if k_cache.shape != v_cache.shape:
        raise ValueError(f"k_cache {tuple(k_cache.shape)} != v_cache {tuple(v_cache.shape)}")
    batch, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    why = unsupported_reason(
        head_dim=head_dim,
        page_size=table.page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
    )
    if why is not None:
        raise ValueError(f"a_r3_paged_decode: {why}")
    if batch != table.batch_size:
        raise ValueError(f"q has batch {batch}, page table has {table.batch_size}")
    if q.dtype is not torch.bfloat16 or k_cache.dtype is not torch.bfloat16:
        raise TypeError(
            f"a_r3_paged_decode expects bfloat16 q and KV — the FlashInfer floor is measured at "
            f"bf16, and a different dtype changes the byte count the % of HBM is computed from; "
            f"got {q.dtype} and {k_cache.dtype}"
        )

    if plan is None:
        dev_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        plan = plan_partition(
            table.pages_per_request(),
            num_kv_heads=num_kv_heads,
            max_grid_size=dev_sms * ctas_per_sm(),
            page_size=table.page_size,
        )
    plan.require_valid()
    if list(plan.pages_per_request) != table.pages_per_request():
        raise ValueError(
            "the plan was built for different page counts than this table has — the work list "
            "would address pages the request does not own"
        )

    req_idx, chunk_idx, o_indptr = plan.to_tensors(q.device)
    scale = default_sm_scale(head_dim) if sm_scale is None else sm_scale
    out = _module().a_r3_paged_decode(
        q.contiguous(),
        k_cache.contiguous(),
        v_cache.contiguous(),
        table.indptr,
        table.indices,
        table.last_page_len,
        req_idx,
        chunk_idx,
        o_indptr,
        int(plan.chunk_size_pages),
        float(scale),
    )
    return out[0], out[1]


def a_r3_merge_states(tmp_o: Tensor, tmp_lse: Tensor, o_indptr: Tensor):
    """The reduce alone: chunk partials in, ``(o bf16, lse fp32)`` out.

    Bound separately from the decode so the merge can be gate-tested against
    :func:`merge_states_reference` on synthetic partials — including the adversarial ones (chunk
    maxima 100 apart) that a real decode will not produce on demand.
    """
    require_arch(ARCH_CC, fn_name="a_r3_merge_states")
    if tmp_o.dtype is not PARTIAL_DTYPE or tmp_lse.dtype is not PARTIAL_DTYPE:
        raise TypeError(
            f"partials must be {PARTIAL_DTYPE} — the merge rescales by 2^(lse_c - m) before it "
            f"sums, and a bf16 partial spends its 8 mantissa bits on a value that is not the "
            f"answer; got {tmp_o.dtype} and {tmp_lse.dtype}"
        )
    out = _module().a_r3_merge_states(
        tmp_o.contiguous(), tmp_lse.contiguous(), o_indptr.to(torch.int32)
    )
    return out[0], out[1]


# =============================================================================================
# The floor — FlashInfer at the identical page table
# =============================================================================================


def flashinfer_reference(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    table: PageTable,
    *,
    use_tensor_cores: bool,
    sm_scale: float | None = None,
    workspace_mb: int = 128,
):
    """``BatchDecodeWithPagedKVCacheWrapper`` fed this rung's own page table. GPU only.

    ``use_tensor_cores`` is exposed and **required**, not defaulted, because the wrapper fronts two
    different kernels and the floor must name which one it measured. ``False`` (the constructor's
    own default, ``decode.py:824``) reaches the CUDA-core FA2 decode via
    ``get_batch_decode_module`` (``decode.py:1824``); ``True`` reaches the tensor-core *prefill*
    kernel with its own scheduler (``decode.py:1753, :1780``) — which upstream's docstring says
    "will be faster for large group size in grouped query attention" (``decode.py:850-851``), i.e.
    exactly this rung's 8:1. Measuring one and quoting the other is how a floor becomes fiction.

    ``plan()`` synchronises to the host (``indptr.to("cpu")``, ``decode.py:1558-1559``), so it must
    sit outside the timed window; this helper deliberately returns the *wrapper* so a bench can
    plan once and run many times.

    Returns ``(wrapper, (k_cache, v_cache))``. The pair is handed straight to
    ``wrapper.run(q, (k, v))``: ``run``'s ``paged_kv_cache`` accepts either a tuple of 4-D
    ``[max_num_pages, page_size, num_kv_heads, head_dim]`` NHD tensors or one fused 5-D tensor
    (``decode.py:1938,:1964-1976``), and the tuple is what this rung's pool already is — so the
    floor and the kernel read the identical bytes with no restacking in between.
    """
    import flashinfer  # noqa: PLC0415

    batch, num_qo_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    workspace = torch.empty(workspace_mb * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=use_tensor_cores
    )
    wrapper.plan(
        table.indptr,
        table.indices,
        table.last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        table.page_size,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
        sm_scale=default_sm_scale(head_dim) if sm_scale is None else sm_scale,
    )
    return wrapper, (k_cache, v_cache)
