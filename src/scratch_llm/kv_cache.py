"""KV-cache substrate for incremental decoding — the inference-serving storage layer.

Extracted from ``model.py`` (2026-07-08): the LM architecture (``model.py``) and the KV
substrate change for different reasons and are owned by different fronts, so they live in
different modules. Dependency is one-way — this module imports only ``torch`` and never the
architecture, so ``model.py`` and ``serving/*`` both import *down* into it (no cycle, correct
layering).

Cache forms:
- :class:`KVCache` — single-request incremental decode.
- :class:`SlotKVCache` (+ :class:`BatchedKVCache` / :class:`PagedKVCache`) — ragged-batch
  continuous-batching substrate (A1 R3b / R4.1).
- :class:`PrefillView` / :class:`ChunkPrefillView` — slot-routed one-shot / chunked prefill
  (A1 R3b / R4.2).

All duck-type the ``length``/``append``/``get``/``advance`` interface the attention layer uses;
:data:`AnyKVCache` is the union the model forward accepts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


class KVCache:
    """Per-layer K/V cache for incremental decoding (see docs/design/L2_kv_cache_SPEC.md).

    Stores GQA-native K, V — shape (B, n_kv_heads, T_cached, head_dim) per layer — so the cache
    is n_heads/n_kv_heads× smaller; the repeat to n_heads happens at attention time. ``length`` is
    the number of positions cached so far (identical across layers); it is bumped once per forward
    via :meth:`advance`, after every layer has appended, so all layers see the same start position.
    """

    def __init__(self, n_layers: int) -> None:
        self._k: list[Tensor | None] = [None] * n_layers
        self._v: list[Tensor | None] = [None] * n_layers
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    def get(self, layer: int) -> tuple[Tensor, Tensor] | None:
        k, v = self._k[layer], self._v[layer]
        if k is None or v is None:
            return None
        return k, v

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        """Concatenate this step's K, V onto the layer's cache along the sequence axis."""
        existing = self.get(layer)
        if existing is None:
            self._k[layer], self._v[layer] = k_new, v_new
        else:
            past_k, past_v = existing
            self._k[layer] = torch.cat([past_k, k_new], dim=2)
            self._v[layer] = torch.cat([past_v, v_new], dim=2)

    def advance(self, n: int) -> None:
        self._length += n

    def truncate(self, length: int) -> None:
        """Drop cached positions past ``length`` — the KV-rollback primitive for speculative
        decoding (A1 R4.3). After a verification forward appends K+1 draft positions, the rejected
        suffix's K/V is discarded so it is never attended again; the retained prefix is bit-identical
        to plain decode. ``length`` must not exceed the current length (rollback only, never grow)."""
        if not 0 <= length <= self._length:
            raise ValueError(f"truncate length {length} outside [0, {self._length}]")
        for layer in range(len(self._k)):
            k, v = self._k[layer], self._v[layer]
            if k is not None and v is not None:
                self._k[layer] = k[:, :, :length].contiguous()
                self._v[layer] = v[:, :, :length].contiguous()
        self._length = length


class SlotKVCache:
    """Slot machinery shared by every ragged-batch KV store (A1 R3b/R4.1) — storage-agnostic.

    Slot ``b`` holds ``lengths[b]`` valid positions. A decode step writes each row's new K,V at
    its *own* offset (**write-then-mask**: the per-row mask admits keys ``j ≤ lengths[b]``, so
    every row — active or not — attends at least its just-written key: no all-masked softmax row,
    hence no NaN by construction). ``advance(1)`` bumps only *active* rows, once per forward,
    after all layers (the same contract as :class:`KVCache`).

    ``py_lengths``/``py_active`` mirror the device tensors so ``view_len`` and scheduler
    bookkeeping never pay a per-step ``.item()`` host sync (the R1 lesson). **Ownership
    contract:** device tensors are mutated *inside* the forward (graph-owned: ``write_decode``/
    ``advance`` trace cleanly under ``torch.compile``); the python mirror and **every allocation
    decision** are **scheduler-owned** — call :meth:`mirror_admit` after a prefill forward,
    :meth:`mirror_advance` after each decode forward, and :meth:`pre_decode_reserve` before it,
    all *outside* the compiled region. Python list/bool state read inside the graph bakes concrete
    Dynamo guards; ragged churn permutes them → recompile storm → eager fallback (measured
    2026-07-03).

    Storage is a subclass concern: :class:`BatchedKVCache` (dense contiguous slots — the R4.4
    CUDA-graph substrate) and :class:`PagedKVCache` (16-token blocks + block table — R4.1).
    Subclasses implement ``_write_decode_kv`` / ``decode_view`` / ``_write_prefill_kv`` and may
    override the reserve hooks.

    Interview question this answers: how does an inference engine decode a *ragged* batch in
    lockstep without one row leaking into another?
    """

    def __init__(
        self,
        n_layers: int,
        n_slots: int,
        n_kv_heads: int,
        max_ctx: int,
        head_dim: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if n_layers < 1 or n_slots < 1 or n_kv_heads < 1 or max_ctx < 1 or head_dim < 1:
            raise ValueError("all cache dimensions must be ≥ 1")
        self.n_layers = n_layers
        self.n_slots = n_slots
        self.n_kv_heads = n_kv_heads
        self.max_ctx = max_ctx
        self.head_dim = head_dim
        self.dtype = dtype
        self.lengths = torch.zeros(n_slots, dtype=torch.long, device=device)
        self.active = torch.zeros(n_slots, dtype=torch.bool, device=device)
        self.py_lengths: list[int] = [0] * n_slots
        self.py_active: list[bool] = [False] * n_slots
        self._slot_idx = torch.arange(n_slots, device=device)
        self._view_len = 1

    # ------------------------------------------------------------------ storage hooks
    def _write_decode_kv(self, layer: int, pos: Tensor, k_new: Tensor, v_new: Tensor) -> None:
        raise NotImplementedError

    def decode_view(self, layer: int) -> tuple[Tensor, Tensor]:
        """The dense ``(B, H_kv, view_len, d)`` K,V the generic SDPA path attends over."""
        raise NotImplementedError

    def _write_prefill_kv(self, layer: int, slots: Tensor, k_new: Tensor, v_new: Tensor) -> None:
        raise NotImplementedError

    def _write_prefill_kv_at(
        self, layer: int, slot: int, offset: int, k_new: Tensor, v_new: Tensor
    ) -> None:
        """Write ONE slot's chunk of prefill K,V at ``[offset, offset+s)`` — A1 R4.2 chunked
        prefill. ``k_new``/``v_new``: ``(1, n_kv_heads, s, head_dim)``. Unlike
        :meth:`_write_prefill_kv` (offset-0, batched over slots), this appends a chunk to a single
        slot at a running offset so a long prompt is prefilled incrementally."""
        raise NotImplementedError

    def slot_kv_view(self, layer: int, slot: int, upto: int) -> tuple[Tensor, Tensor]:
        """One slot's K,V over positions ``[0, upto)`` as ``(1, n_kv_heads, upto, head_dim)`` — the
        full prefix a chunked-prefill chunk attends over (A1 R4.2)."""
        raise NotImplementedError

    def _free_storage(self, slot: int) -> None:
        """Storage-specific eviction (e.g. return blocks to the pool). Default: nothing."""

    def reserve_prefill(self, slots: Sequence[int], true_lengths: Sequence[int]) -> None:
        """Allocate storage for an admission (scheduler-owned, eager path). Default: nothing."""

    def pre_decode_reserve(self) -> None:
        """Allocate storage the next decode write needs (scheduler-owned, called by the engine
        before each decode forward, outside the graph). Default: nothing."""

    # ------------------------------------------------------------------ shared machinery
    @property
    def view_len(self) -> int:
        """Key positions visible this step: covers every row's just-written key (offset
        ``lengths[b]``). A plain int attribute, recomputed only by the scheduler-owned mirror
        ops — no device sync, and safe to read inside a compiled forward (a ``max()`` over
        ``py_lengths`` traced in-graph bakes guards on the list's ORDERING; measured 2026-07-03)."""
        return self._view_len

    def _recompute_view_len(self) -> None:
        self._view_len = min(self.max_ctx, 1 + max(self.py_lengths))

    def free_slots(self) -> list[int]:
        """Slots available for admission (inactive)."""
        return [b for b, a in enumerate(self.py_active) if not a]

    def free_slot(self, slot: int) -> None:
        """Evict: deactivate + zero the slot's length. Stale K/V is *not* zeroed — it is
        unreachable (masked) and overwritten/reused by the next admission."""
        self._free_storage(slot)
        self.py_lengths[slot] = 0
        self.py_active[slot] = False
        self.lengths[slot] = 0
        self.active[slot] = False
        self._recompute_view_len()

    def write_decode(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        """Write one new K,V per row at that row's offset (graph-owned).

        ``k_new``/``v_new``: ``(n_slots, n_kv_heads, 1, head_dim)``. Row ``b`` lands at position
        ``lengths[b]`` (inactive rows harmlessly rewrite offset 0 / the trash block). The
        attention view is fetched separately via :meth:`decode_view` — the paged kernel path
        (R4.1b) reads storage directly and never materializes it.
        """
        if k_new.shape[0] != self.n_slots or k_new.shape[2] != 1:
            raise ValueError(f"expected ({self.n_slots}, H_kv, 1, d), got {tuple(k_new.shape)}")
        pos = self.lengths.clamp(max=self.max_ctx - 1)  # scheduler guarantees no active overflow
        self._write_decode_kv(layer, pos, k_new, v_new)

    def advance(self, n: int) -> None:
        """Bump only *active* rows, once per forward after all layers (the :class:`KVCache`
        contract). ``n`` must be 1 — batched decode processes exactly one token per row.
        Device-only (graph-owned); the scheduler follows with :meth:`mirror_advance`."""
        if n != 1:
            raise ValueError(f"SlotKVCache.advance expects n=1 (decode), got {n}")
        self.lengths += self.active.long()

    def mirror_advance(self) -> None:
        """Scheduler-owned python half of :meth:`advance` — call after each decode forward,
        outside the compiled region (see the ownership contract in the class docstring)."""
        for b, a in enumerate(self.py_active):
            if a:
                self.py_lengths[b] += 1
        if max(self.py_lengths) > self.max_ctx:
            raise ValueError("a slot exceeded max_ctx — the scheduler must evict at capacity")
        self._recompute_view_len()

    def mirror_admit(self, slots: Sequence[int], true_lengths: Sequence[int]) -> None:
        """Scheduler-owned python half of a :class:`PrefillView` admission — call after the
        prefill forward, outside the compiled region."""
        for b, ln in zip(slots, true_lengths, strict=True):
            self.py_lengths[b] = int(ln)
            self.py_active[b] = True
        self._recompute_view_len()


class BatchedKVCache(SlotKVCache):
    """Dense contiguous slot storage — A1 R3b.

    One preallocated ``(n_slots, n_kv_heads, max_ctx, head_dim)`` K and V buffer per layer: fixed
    addresses (what R4.4 CUDA-graph capture needs; R1's ``torch.cat`` cache broke capture), zero
    gather cost — and **reservation waste**: every slot holds ``max_ctx`` whether it uses it or
    not, and decode reads a view padded to ``view_len`` (the mixed-age padding tax measured at
    ~1.27× in R3b). :class:`PagedKVCache` trades exactly the other way.
    """

    def __init__(
        self,
        n_layers: int,
        n_slots: int,
        n_kv_heads: int,
        max_ctx: int,
        head_dim: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(n_layers, n_slots, n_kv_heads, max_ctx, head_dim, device, dtype)
        shape = (n_slots, n_kv_heads, max_ctx, head_dim)
        self._k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self._v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]

    def _write_decode_kv(self, layer: int, pos: Tensor, k_new: Tensor, v_new: Tensor) -> None:
        self._k[layer][self._slot_idx, :, pos] = k_new[:, :, 0]
        self._v[layer][self._slot_idx, :, pos] = v_new[:, :, 0]

    def decode_view(self, layer: int) -> tuple[Tensor, Tensor]:
        length = self.view_len
        return self._k[layer][:, :, :length], self._v[layer][:, :, :length]

    def _write_prefill_kv(self, layer: int, slots: Tensor, k_new: Tensor, v_new: Tensor) -> None:
        s = k_new.shape[2]
        self._k[layer][slots, :, :s] = k_new
        self._v[layer][slots, :, :s] = v_new

    def _write_prefill_kv_at(
        self, layer: int, slot: int, offset: int, k_new: Tensor, v_new: Tensor
    ) -> None:
        s = k_new.shape[2]  # (1, H_kv, s, d) → the slot's [offset, offset+s) rows (R4.2)
        self._k[layer][slot, :, offset : offset + s] = k_new[0]
        self._v[layer][slot, :, offset : offset + s] = v_new[0]

    def slot_kv_view(self, layer: int, slot: int, upto: int) -> tuple[Tensor, Tensor]:
        return self._k[layer][slot : slot + 1, :, :upto], self._v[layer][slot : slot + 1, :, :upto]


class PagedKVCache(SlotKVCache):
    """Paged slot storage — KV memory as virtual memory (A1 R4.1, vLLM's design).

    Per layer, a pool of ``n_blocks`` fixed 16-token blocks ``(n_blocks, H_kv, 16, d)``; slot
    ``b`` maps logical block ``i`` → physical block via ``block_table[b, i]``. Allocation is
    on-demand (a new block only when a row's length crosses a 16 boundary), so waste collapses
    from *reservation* (``max_ctx − ℓ`` per slot, ~95% on short traces) to *internal
    fragmentation* (≤15 tokens in the last block, E≈8 — the 10–20× capacity win, P4.1.1).

    **Block 0 is the reserved trash block**: freed/inactive rows' table entry 0 points at it, so
    the static-shape decode write (every row writes — R3b design) lands harmlessly and the
    write-then-mask NaN guarantee carries over verbatim. Unallocated table entries are 0 too:
    the gather view reads trash there, and the per-row mask hides it.

    **Prefix sharing + CoW (block granularity):** :meth:`share_prefix` points a fresh slot's
    leading table entries at another slot's physical blocks (refcounted). Sharing is full-block
    only, and a row's next write lands at its own length — which sits at a block boundary right
    after a shared prefix — so :meth:`pre_decode_reserve` allocates it a *private* block and no
    shared block is ever written (CoW degenerates to copy-never; the refcount invariant is
    asserted every step). Partial-block CoW (beam search) is explicitly out of scope.

    Allocator state (free list, refcounts, python table) is **scheduler-owned** (never read in
    the graph); the device ``block_table`` is a graph *input* updated between steps.
    """

    BLOCK = 16

    def __init__(
        self,
        n_layers: int,
        n_slots: int,
        n_kv_heads: int,
        max_ctx: int,
        head_dim: int,
        n_blocks: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(n_layers, n_slots, n_kv_heads, max_ctx, head_dim, device, dtype)
        if n_blocks < 2:
            raise ValueError("n_blocks must be ≥ 2 (block 0 is the reserved trash block)")
        self.n_blocks = n_blocks
        self.max_blocks = (max_ctx + self.BLOCK - 1) // self.BLOCK
        shape = (n_blocks, n_kv_heads, self.BLOCK, head_dim)
        self._pool_k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self._pool_v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        # 0 everywhere = "points at trash": safe for the always-writes decode and the gather view
        self.block_table = torch.zeros((n_slots, self.max_blocks), dtype=torch.long, device=device)
        self._py_table: list[list[int]] = [[] for _ in range(n_slots)]
        self._free: list[int] = list(range(n_blocks - 1, 0, -1))  # block 0 never allocated
        self._refcount: list[int] = [0] * n_blocks
        self.use_kernel = False  # R4.1b: route decode attention through the paged Triton kernel

    # ------------------------------------------------------------------ allocator (scheduler-owned)
    @property
    def n_free_blocks(self) -> int:
        return len(self._free)

    def _alloc_block(self) -> int:
        if not self._free:
            raise RuntimeError(
                "PagedKVCache pool exhausted — admission accounting must prevent this"
            )
        blk = self._free.pop()
        self._refcount[blk] = 1
        return blk

    def _release_block(self, blk: int) -> None:
        self._refcount[blk] -= 1
        if self._refcount[blk] == 0:
            self._free.append(blk)

    def _free_storage(self, slot: int) -> None:
        for blk in self._py_table[slot]:
            self._release_block(blk)
        self._py_table[slot] = []
        self.block_table[slot] = 0  # everything points back at trash

    def reserve_prefill(self, slots: Sequence[int], true_lengths: Sequence[int]) -> None:
        """Allocate ⌈ℓ/16⌉ blocks per admitted slot (eager admission path)."""
        for slot, ln in zip(slots, true_lengths, strict=True):
            if self._py_table[slot]:
                raise ValueError(f"slot {slot} still holds blocks — evict before re-admitting")
            n_needed = (int(ln) + self.BLOCK - 1) // self.BLOCK
            blocks = [self._alloc_block() for _ in range(n_needed)]
            self._py_table[slot] = blocks
            self.block_table[slot, :n_needed] = torch.tensor(
                blocks, dtype=torch.long, device=self.block_table.device
            )

    def pre_decode_reserve(self) -> None:
        """Before each decode forward: give boundary rows a fresh private block; assert no row
        is about to write into a shared block (full-block sharing makes that impossible)."""
        for b, (ln, a) in enumerate(zip(self.py_lengths, self.py_active, strict=True)):
            if not a:
                continue
            idx = ln // self.BLOCK
            if ln % self.BLOCK == 0:
                if idx >= self.max_blocks:
                    raise RuntimeError("slot at max_ctx — the scheduler must evict at capacity")
                if len(self._py_table[b]) != idx:
                    raise RuntimeError("block-table bookkeeping out of sync with lengths")
                blk = self._alloc_block()
                self._py_table[b].append(blk)
                self.block_table[b, idx] = blk
            elif self._refcount[self._py_table[b][idx]] != 1:
                raise RuntimeError("decode write aimed at a shared block — CoW invariant broken")

    def share_prefix(self, src_slot: int, dst_slot: int, n_tokens: int) -> None:
        """Point fresh ``dst_slot`` at ``src_slot``'s first ``n_tokens`` (full blocks only) —
        the prefix-cache primitive. ``dst`` becomes active at length ``n_tokens``; its next
        write allocates a private block (see class docstring)."""
        if n_tokens % self.BLOCK != 0 or n_tokens == 0:
            raise ValueError("prefix sharing is full-block only")
        if self.py_active[dst_slot] or self._py_table[dst_slot]:
            raise ValueError(f"slot {dst_slot} is not fresh")
        n_shared = n_tokens // self.BLOCK
        if len(self._py_table[src_slot]) < n_shared:
            raise ValueError("source slot holds fewer blocks than the requested prefix")
        shared = self._py_table[src_slot][:n_shared]
        for blk in shared:
            self._refcount[blk] += 1
        self._py_table[dst_slot] = list(shared)
        self.block_table[dst_slot, :n_shared] = torch.tensor(
            shared, dtype=torch.long, device=self.block_table.device
        )
        self.mirror_admit([dst_slot], [n_tokens])
        self.lengths[dst_slot] = n_tokens
        self.active[dst_slot] = True

    def waste_stats(self) -> tuple[int, int, float]:
        """(allocated_tokens, live_tokens, fragmentation) — the P4.1.1 accounting. Shared blocks
        count once (they occupy HBM once)."""
        allocated_blocks = self.n_blocks - 1 - len(self._free)
        allocated_tokens = allocated_blocks * self.BLOCK
        live = sum(ln for ln, a in zip(self.py_lengths, self.py_active, strict=True) if a)
        frag = 0.0 if allocated_tokens == 0 else 1.0 - live / allocated_tokens
        return allocated_tokens, live, frag

    # ------------------------------------------------------------------ storage hooks (graph-owned)
    def _write_decode_kv(self, layer: int, pos: Tensor, k_new: Tensor, v_new: Tensor) -> None:
        blk_idx = pos // self.BLOCK
        offset = pos % self.BLOCK
        phys = self.block_table[self._slot_idx, blk_idx]  # (B,)
        self._pool_k[layer][phys, :, offset] = k_new[:, :, 0]
        self._pool_v[layer][phys, :, offset] = v_new[:, :, 0]

    def decode_view(self, layer: int) -> tuple[Tensor, Tensor]:
        """Gather the dense padded view from the pool (the oracle path — bit-exact vs contiguous
        storage, and the P4.1.3 negative control: this copy is the price paged storage pays
        without a paged kernel)."""
        length = self.view_len
        n_blocks = (length + self.BLOCK - 1) // self.BLOCK
        phys = self.block_table[:, :n_blocks]  # (B, nb); unallocated → 0 → trash → masked
        k = self._pool_k[layer][phys]  # (B, nb, H_kv, 16, d)
        v = self._pool_v[layer][phys]
        b = self.n_slots
        k = k.permute(0, 2, 1, 3, 4).reshape(b, self.n_kv_heads, n_blocks * self.BLOCK, -1)
        v = v.permute(0, 2, 1, 3, 4).reshape(b, self.n_kv_heads, n_blocks * self.BLOCK, -1)
        return k[:, :, :length], v[:, :, :length]

    def _write_prefill_kv(self, layer: int, slots: Tensor, k_new: Tensor, v_new: Tensor) -> None:
        # Eager admission path: scatter the padded block into each slot's blocks, 16 tokens at a
        # time. Chunks beyond a row's own blocks index table entry 0 (trash) — harmless garbage.
        s = k_new.shape[2]
        n_chunks = (s + self.BLOCK - 1) // self.BLOCK
        for j in range(n_chunks):
            lo, hi = j * self.BLOCK, min((j + 1) * self.BLOCK, s)
            phys = self.block_table[slots, j]  # (n,)
            self._pool_k[layer][phys, :, : hi - lo] = k_new[:, :, lo:hi]
            self._pool_v[layer][phys, :, : hi - lo] = v_new[:, :, lo:hi]


class PrefillView:
    """Routes one right-padded batched prefill into fresh :class:`SlotKVCache` slots — A1 R3b.

    Duck-types the :class:`KVCache` interface the attention layer uses (``length``/``append``/
    ``get``/``advance``), so a batched admission prefill IS the ordinary uniform-causal forward:
    fresh slots start at offset 0 ⇒ standard causal mask + shared positions ``0..s−1`` — no new
    attention math. K,V are written through the parent's storage hook (dense rows or paged
    blocks). Rows are right-padded to the widest prompt; K/V beyond a row's true length is dead —
    masked by write-then-mask on later steps and overwritten as the row decodes. Construction
    reserves storage (``parent.reserve_prefill`` — a no-op for dense, block allocation for paged;
    the admission path is eager, so allocation here honors the ownership contract). ``advance(s)``
    (called once by the LM after all layers) activates the slots at their TRUE lengths, not the
    padded width — device tensors only (graph-owned); the scheduler follows with
    ``parent.mirror_admit(slots, true_lengths)``.

    All python-valued state (slot list, widths) is resolved to tensors/ints at construction —
    *outside* any compiled region — so a traced ``append``/``advance`` guards only on stable ints.
    """

    def __init__(
        self, parent: SlotKVCache, slots: Sequence[int], true_lengths: Sequence[int]
    ) -> None:
        if len(slots) == 0 or len(slots) != len(true_lengths):
            raise ValueError("slots and true_lengths must be non-empty and equal-length")
        if len(set(slots)) != len(slots):
            raise ValueError("slots must be distinct")
        for b, ln in zip(slots, true_lengths, strict=True):
            if parent.py_active[b] or parent.py_lengths[b] != 0:
                raise ValueError(f"slot {b} is not fresh — evict before re-admitting")
            if not 0 < ln <= parent.max_ctx:
                raise ValueError(f"true length {ln} outside (0, max_ctx={parent.max_ctx}]")
        parent.reserve_prefill(list(slots), [int(x) for x in true_lengths])
        self._parent = parent
        device = parent.lengths.device
        self._slots = torch.tensor(list(slots), dtype=torch.long, device=device)
        self._true_lengths_t = torch.tensor(
            [int(x) for x in true_lengths], dtype=torch.long, device=device
        )
        self._min_width = max(int(x) for x in true_lengths)
        self._block: list[tuple[Tensor, Tensor] | None] = [None] * parent.n_layers

    @property
    def length(self) -> int:
        return 0  # fresh slots by contract: prefill is an offset-0, uniform-causal forward

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        s = k_new.shape[2]
        if s < self._min_width:
            raise ValueError("padded width must cover every row's true length")
        self._parent._write_prefill_kv(layer, self._slots, k_new, v_new)
        self._block[layer] = (k_new, v_new)  # this prefill attends exactly its own block

    def get(self, layer: int) -> tuple[Tensor, Tensor] | None:
        return self._block[layer]

    def advance(self, n: int) -> None:
        """Activate the slots at their true (unpadded) lengths — called once, after all layers.
        Device tensors only; the scheduler mirrors via ``parent.mirror_admit``."""
        parent = self._parent
        parent.lengths[self._slots] = self._true_lengths_t
        parent.active[self._slots] = True


class ChunkPrefillView:
    """Routes ONE prefilling slot's chunk (s tokens at a running offset) through the single-request
    KVCache attention branch — A1 R4.2 chunked prefill.

    A long prompt is prefilled in ``⌈L/C⌉`` chunks interleaved with decode steps, so a big prefill
    no longer head-of-line-blocks the decode stream (the measured R3b/R4.1 ITL p99 admission spike).
    Duck-types the :class:`KVCache` interface the attention layer uses on the non-slot path
    (``length``/``append``/``get``/``advance``): ``length`` is the already-written prefix — the
    chunk's absolute RoPE offset, so a token at position t is rotated at t regardless of chunk
    boundaries ⇒ the chunked KV is **bit-identical** to a one-shot prefill (the token-exactness
    oracle). ``append`` writes the chunk into the slot's storage at ``[offset, offset+s)`` and
    ``get`` returns the slot's full ``[0, offset+s)`` K,V so the chunk attends its whole prefix
    under the ordinary ``past_len`` causal mask — no new attention math.

    The parent slot is **reserved but inactive** while prefilling (masked out of the concurrent
    decode batch by write-then-mask, exactly like any inactive slot); the scheduler flips it active
    (``parent.active[slot]=True`` + ``parent.mirror_admit``) after the LAST chunk, whose final
    position emits the request's first token (= TTFT). A fresh view is built per chunk with the
    current ``offset``; ``advance(s)`` bumps only the device length (the scheduler owns the mirror).

    Note (scope): implemented for :class:`BatchedKVCache` (dense). Chunked prefill over paged
    storage combines R4.1 + R4.2 and is deferred — the TTFT/ITL-vs-chunk-size physics R4.2 measures
    is fully delivered on the dense slot buffer.
    """

    def __init__(self, parent: SlotKVCache, slot: int, offset: int) -> None:
        if not 0 <= slot < parent.n_slots:
            raise ValueError(f"slot {slot} out of range [0, {parent.n_slots})")
        if not 0 <= offset < parent.max_ctx:
            raise ValueError(f"offset {offset} outside [0, max_ctx={parent.max_ctx})")
        self._parent = parent
        self._slot = slot
        self._offset = offset
        self._view: list[tuple[Tensor, Tensor] | None] = [None] * parent.n_layers

    @property
    def length(self) -> int:
        return self._offset

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        s = k_new.shape[2]
        self._parent._write_prefill_kv_at(layer, self._slot, self._offset, k_new, v_new)
        self._view[layer] = self._parent.slot_kv_view(layer, self._slot, self._offset + s)

    def get(self, layer: int) -> tuple[Tensor, Tensor] | None:
        return self._view[layer]

    def advance(self, n: int) -> None:
        """Bump the slot's device length by the chunk width — called once by the LM after all
        layers. Activation is deferred to the scheduler after the final chunk (see class docstring).
        """
        self._parent.lengths[self._slot] = self._offset + n


AnyKVCache = KVCache | SlotKVCache | PrefillView | ChunkPrefillView
"""Cache forms accepted by the model forward: single-request (:class:`KVCache`), ragged slot
decode (:class:`SlotKVCache`: dense or paged), slot-routed one-shot prefill (:class:`PrefillView`),
or one slot's chunked prefill (:class:`ChunkPrefillView`, A1 R4.2)."""
