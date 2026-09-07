"""The paged-KV page table, in FlashInfer's exact layout — buildable and checkable on a laptop.

K2/A-R3 is measured against `flashinfer`'s ``BatchDecodeWithPagedKVCacheWrapper`` **at the identical
page table**. That sentence is the whole reason this module exists: a kernel and its floor that
disagree about which physical pages hold a request's history are not measuring the same problem, and
the disagreement is invisible — both produce plausible attention outputs, at plausible speeds.

So the table is a real data structure with a real contract rather than three tensors assembled
inline at the call site, and every rule it has to obey is asserted here, on CPU, in milliseconds.

The layout, byte for byte, from ``oss/flashinfer`` @ 03d8b2cd:

* ``indptr``         int32 ``[B + 1]``, ``indptr[0] == 0``, non-decreasing. Request ``i`` owns
  ``indices[indptr[i]:indptr[i+1]]``.  (``flashinfer/decode.py:1322-1324``)
* ``indices``        int32 ``[indptr[-1]]``, physical page ids into the pool's dimension 0.
  (``flashinfer/decode.py:1325-1326``)
* ``last_page_len``  int32 ``[B]``, entries used in the request's LAST page.
  (``flashinfer/decode.py:1327-1329``)

and the length contract that binds them, ``paged_kv_t::get_length``
(``include/flashinfer/page.cuh:185-190``)::

    len(i) = 0                                                     if indptr[i+1] == indptr[i]
    len(i) = (indptr[i+1] - indptr[i] - 1) * page_size + last_page_len[i]   otherwise

Three consequences that are easy to get wrong and that :func:`PageTable.validate` therefore asserts
rather than documents:

1. **A full last page is ``page_size``, never ``0``.** ``last_page_len`` lives in ``1..page_size``.
   The natural ``seq_len % page_size`` is 0 for a request whose length divides the page size — and
   at the plan's shape (ctx 8192, page 16) that is *every* request, so the off-by-one would corrupt
   the whole batch and only the whole batch, which reads as "the kernel is wrong" rather than "the
   table is wrong".
2. **int32, not int64.** ``torch.tensor([...])`` defaults to int64 and FlashInfer type-checks
   rather than coercing (``csrc/batch_decode.cu:45``, and again in run at ``:137-139``), so an
   int64 table fails at the floor and passes to our own kernel — the worst possible asymmetry.
3. **Every page id must be inside the pool.** Out-of-range ids do not fault on the read path:
   ``protective_get_kv_offset`` (``page.cuh:228-234``) returns offset 0 for a page past
   ``indptr[batch_size]``, i.e. it silently attends to page 0's contents.

Layout note — this is **NHD**: the pool is ``[num_pages, page_size, num_kv_heads, head_dim]``
(``flashinfer/decode.py:1967-1976``). ``kernels.attention.decode.paged``'s Triton kernel uses the
*other* order, ``(n_blocks, H_kv, 16, d)`` = HND, and its "block table" is a dense ``(B, max_blocks)``
matrix rather than a ragged indptr/indices pair. The two are not interchangeable; do not feed one's
pool to the other.

Spec: ``experiments/K2/A-R3/spec.md``   ·   Map: ``experiments/K2/A-R3/map.md``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

#: The dtype every index tensor must have. Not a preference — FlashInfer checks it
#: (``csrc/batch_decode.cu:45``) and our own kernel's launcher checks it too, so a table that is
#: right in Python and int64 on the wire is a table that only one side of the comparison accepts.
INDEX_DTYPE = torch.int32


@dataclass(frozen=True)
class PageTable:
    """One batch's paged-KV index structure, in FlashInfer's layout.

    Construct with :meth:`from_seq_lens` (which allocates page ids out of a pool) rather than by
    hand; the constructor is deliberately dumb so a hand-built adversarial table can still be fed
    to :meth:`validate` in a test.
    """

    indptr: Tensor  # int32 [B + 1]
    indices: Tensor  # int32 [indptr[-1]]
    last_page_len: Tensor  # int32 [B]
    page_size: int
    num_pages: int  # pool capacity along dim 0 — what `indices` must stay inside

    # -- shape helpers ---------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.indptr.numel() - 1

    @property
    def nnz_pages(self) -> int:
        """Pages actually referenced across the batch — ``indices``' length."""
        return int(self.indices.numel())

    def pages_per_request(self) -> list[int]:
        """``indptr`` differenced. The input the partition decision is made from."""
        ind = self.indptr.tolist()
        return [ind[i + 1] - ind[i] for i in range(self.batch_size)]

    def seq_lens(self) -> list[int]:
        """The lengths ``get_length`` (``page.cuh:185-190``) would compute — the round-trip target."""
        ind = self.indptr.tolist()
        last = self.last_page_len.tolist()
        out = []
        for i in range(self.batch_size):
            n_pages = ind[i + 1] - ind[i]
            out.append(0 if n_pages == 0 else (n_pages - 1) * self.page_size + last[i])
        return out

    def page_ids(self, request: int) -> list[int]:
        """The physical page ids request ``request`` owns, in logical order."""
        lo, hi = int(self.indptr[request]), int(self.indptr[request + 1])
        return self.indices[lo:hi].tolist()

    # -- the contract ----------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Every rule the layout has to obey, as a list of violations (empty == legal).

        Returned rather than raised so a test can assert on the *message*: each of these has a
        distinct failure signature on the box (a wrong ``last_page_len`` attends to garbage in the
        tail; a non-monotone ``indptr`` makes one request read another's pages) and a checker that
        collapses them into one exception cannot tell them apart.
        """
        errs: list[str] = []
        for name, t, ndim in (
            ("indptr", self.indptr, 1),
            ("indices", self.indices, 1),
            ("last_page_len", self.last_page_len, 1),
        ):
            if t.dtype is not INDEX_DTYPE:
                errs.append(
                    f"{name}.dtype is {t.dtype}, must be {INDEX_DTYPE} — FlashInfer type-checks "
                    f"rather than coerces (csrc/batch_decode.cu:45), so an int64 table is refused "
                    f"by the floor and accepted by us"
                )
            if t.dim() != ndim:
                errs.append(f"{name} must be {ndim}-D, got {t.dim()}-D")
        if self.page_size < 1:
            errs.append(f"page_size must be >= 1, got {self.page_size}")
        if self.indptr.numel() < 1:
            errs.append("indptr must have at least one entry (batch_size + 1)")
            return errs

        ind = self.indptr.tolist()
        if ind[0] != 0:
            errs.append(f"indptr[0] must be 0, got {ind[0]}")
        if any(ind[i + 1] < ind[i] for i in range(len(ind) - 1)):
            errs.append(f"indptr is not non-decreasing: {ind}")
        if ind[-1] != self.indices.numel():
            errs.append(
                f"indptr[-1]={ind[-1]} != len(indices)={self.indices.numel()} — indices is exactly "
                f"the concatenation of every request's page list, with no slack"
            )
        if self.last_page_len.numel() != self.batch_size:
            errs.append(
                f"last_page_len has {self.last_page_len.numel()} entries for batch "
                f"{self.batch_size}"
            )

        if self.indices.numel():
            lo, hi = int(self.indices.min()), int(self.indices.max())
            if lo < 0 or hi >= self.num_pages:
                errs.append(
                    f"page ids span [{lo}, {hi}] but the pool holds {self.num_pages} pages — an "
                    f"out-of-range id does not fault, protective_get_kv_offset (page.cuh:228-234) "
                    f"reads page 0 instead"
                )

        last = self.last_page_len.tolist()
        for i in range(min(self.batch_size, len(last))):
            n_pages = ind[i + 1] - ind[i]
            if n_pages == 0:
                # get_length short-circuits to 0 and never reads last_page_len (page.cuh:186-188).
                continue
            if not 1 <= last[i] <= self.page_size:
                errs.append(
                    f"last_page_len[{i}]={last[i]} outside 1..{self.page_size}. A FULL last page "
                    f"is {self.page_size}, not 0: seq_len % page_size is the wrong expression, and "
                    f"at ctx 8192 / page 16 it is wrong for every request in the batch"
                )
        return errs

    def require_valid(self) -> PageTable:
        """:meth:`validate`, but raising. Returns ``self`` so it chains onto a builder."""
        errs = self.validate()
        if errs:
            raise ValueError("invalid page table:\n  " + "\n  ".join(errs))
        return self

    def to(self, device: torch.device | str) -> PageTable:
        """The same table on another device. dtypes are preserved — int32 is part of the contract."""
        return PageTable(
            indptr=self.indptr.to(device),
            indices=self.indices.to(device),
            last_page_len=self.last_page_len.to(device),
            page_size=self.page_size,
            num_pages=self.num_pages,
        )

    # -- construction ----------------------------------------------------------------------

    @classmethod
    def from_seq_lens(
        cls,
        seq_lens: Sequence[int],
        *,
        page_size: int,
        num_pages: int | None = None,
        shuffle_seed: int | None = None,
        device: torch.device | str = "cpu",
    ) -> PageTable:
        """Allocate pages for ``seq_lens`` and return the table describing them.

        ``num_pages`` is the pool's capacity; ``None`` means "exactly what the batch needs", which
        is the tight case. Pass a larger pool to model a real allocator with free space.

        ``shuffle_seed`` permutes which *physical* page backs each logical position. Sequential ids
        are the easy case and hide a whole class of bug: a kernel that computes its page address
        from the logical index rather than reading ``indices`` is exactly correct on a sequential
        table and wrong on a fragmented one, which is the only kind a real server has. The
        FlashInfer floor reads ``indices`` too (``page.cuh:222``), so both sides see the same
        fragmentation.
        """
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        lens = list(seq_lens)
        if any(n < 1 for n in lens):
            raise ValueError(
                f"decode attends at least the token written this step, so every sequence length "
                f"must be >= 1; got {min(lens) if lens else None}"
            )

        n_pages = [(n + page_size - 1) // page_size for n in lens]
        total = sum(n_pages)
        pool = total if num_pages is None else num_pages
        if pool < total:
            raise ValueError(
                f"pool of {pool} pages cannot hold the {total} the batch needs "
                f"(sum ceil(seq_len / {page_size}))"
            )

        if shuffle_seed is None:
            ids = torch.arange(total, dtype=INDEX_DTYPE, device=device)
        else:
            g = torch.Generator().manual_seed(shuffle_seed)
            ids = torch.randperm(pool, generator=g)[:total].to(device=device, dtype=INDEX_DTYPE)

        indptr = torch.zeros(len(lens) + 1, dtype=INDEX_DTYPE, device=device)
        indptr[1:] = torch.tensor(n_pages, dtype=INDEX_DTYPE, device=device).cumsum(0)
        # A full last page is page_size, not 0 — see the module docstring's point (1).
        last = torch.tensor(
            [((n - 1) % page_size) + 1 for n in lens], dtype=INDEX_DTYPE, device=device
        )
        return cls(
            indptr=indptr,
            indices=ids,
            last_page_len=last,
            page_size=page_size,
            num_pages=pool,
        ).require_valid()


def gather_kv(
    table: PageTable,
    k_cache: Tensor,
    v_cache: Tensor,
    request: int,
) -> tuple[Tensor, Tensor]:
    """One request's K and V, de-paged into contiguous ``[len, num_kv_heads, head_dim]`` views.

    The de-paging oracle. Slow on purpose — it is what "the same problem" means when the kernel
    reads the pool through ``indices`` and the reference reads a dense tensor, and it is the only
    place the NHD element order is spelled out in Python:
    ``cache[page, entry, head, feat]`` (``page.cuh:200-205`` with NHD strides, ``page.cuh:152-154``).
    """
    page_ids = table.page_ids(request)
    seq_len = table.seq_lens()[request]
    if not page_ids:
        empty = k_cache.new_zeros((0, k_cache.shape[2], k_cache.shape[3]))
        return empty, empty.clone()
    k = k_cache[page_ids].reshape(-1, k_cache.shape[2], k_cache.shape[3])[:seq_len]
    v = v_cache[page_ids].reshape(-1, v_cache.shape[2], v_cache.shape[3])[:seq_len]
    return k, v


def empty_kv_pool(
    *,
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """A zeroed NHD pool pair ``[num_pages, page_size, num_kv_heads, head_dim]``.

    Separate K and V tensors, not FlashInfer's fused 5-D ``[P, 2, page, H, D]``: the fused form is
    accepted by the floor as a ``(k, v)`` tuple of the 4-D views anyway (``decode.py:1967-1976``),
    and keeping them apart means the kernel's two ``stride_page`` values are literally equal, which
    is what ``run`` requires (``csrc/batch_decode.cu:176-182``).
    """
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    return (
        torch.zeros(shape, dtype=dtype, device=device),
        torch.zeros(shape, dtype=dtype, device=device),
    )
