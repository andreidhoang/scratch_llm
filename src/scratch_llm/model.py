"""Decoder-only Transformer LM, from scratch — the policy we own end-to-end.

A1 substrate. 2026-default decoder: pre-norm, RMSNorm, RoPE, SwiGLU, GQA-ready
multi-head attention, no biases. Owning the model end-to-end — every logit, every mask —
is what lets you inspect, debug, and trust everything built on top of it.

Correctness invariants (tested in tests/test_model.py):
- **Loss at init:** a fresh LM's cross-entropy on random data is ≈ ``log(vocab_size)``
  (uniform prediction). Off by more than a small band ⇒ head/embedding/norm bug.
- **Causal no-leak:** perturbing future tokens must not change the logits at earlier
  positions (the autoregressive contract).
- **RoPE relative:** the q·k score after RoPE depends only on the position offset i−j.

Design decisions:
- GQA via ``n_kv_heads`` (ADR-0002): full MHA is the special case ``n_kv_heads == n_heads``.
- Untied embeddings by default (ADR-0004): keeps the loss-at-init derivation clean.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from scratch_llm.moe import AuxOutput, MoEConfig, MoEStats


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters. ``n_kv_heads=None`` ⇒ full MHA; ``d_ff=None`` ⇒
    round (8/3)·d_model to a multiple of 64 (the SwiGLU convention)."""

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int | None = None
    d_ff: int | None = None
    context_length: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    qk_norm: bool = (
        False  # RMSNorm on q,k per head before RoPE — bounds attention logits (Qwen3/Gemma3/OLMo2)
    )
    moe: MoEConfig | None = None  # None ⇒ dense SwiGLU (the default); else opt-in MoE

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        if self.n_heads % self.kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} not divisible by n_kv_heads={self.kv_heads} "
                "(query heads must group evenly over kv heads)"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")

    @property
    def kv_heads(self) -> int:
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        raw = int(8 / 3 * self.d_model)
        return ((raw + 63) // 64) * 64  # round up to a multiple of 64


def _trunc_normal_linear_weight(out_features: int, in_features: int) -> nn.Parameter:
    """Weight stored as (out, in); init ~ N(0, 2/(in+out)) truncated at ±3σ."""
    weight = torch.empty(out_features, in_features)
    std = math.sqrt(2.0 / (in_features + out_features))
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return nn.Parameter(weight)


class Linear(nn.Module):
    """y = x @ Wᵀ, no bias. Weight is (out_features, in_features)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = _trunc_normal_linear_weight(out_features, in_features)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """Token-id → vector lookup. Weight is (vocab_size, d_model), init ~ N(0, 1) at ±3σ."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. Computed in fp32 then cast back (precision matters at
    long context / low precision); learnable per-channel gain, no centering, no bias."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms * self.weight.float()).to(dtype)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax (subtract the max before exp)."""
    x = x - x.amax(dim=dim, keepdim=True)
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
) -> Tensor:
    """softmax(QKᵀ/√d_k + mask) V. ``mask`` is boolean, True = attend (disallowed → −∞).
    Softmax is done in fp32 for stability."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores.float(), dim=-1).to(q.dtype)
    return attn @ v


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: rotate each 2-D coordinate pair (x_2k, x_2k+1) by angle pos·θ^(−2k/d).

    cos/sin are a precomputed, non-persistent buffer (recomputed on load, never
    checkpointed) sliced by the actual token positions — so a KV-cache that feeds one
    token at position t rotates it correctly.
    """

    cos: Tensor  # registered buffers; annotated so the type checker sees Tensor, not Module
    sin: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        # x: (..., seq, head_dim); positions: (seq,) shared across the batch, or (B, seq)
        # per-row (ragged batched decode: each slot's token sits at its own absolute position).
        cos = self.cos[positions]  # (seq, head_dim/2) or (B, seq, head_dim/2)
        sin = self.sin[positions]
        if positions.dim() == 2:  # per-row: align batch dim, broadcast over heads → (B,1,seq,d/2)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:  # shared: broadcast over all leading (batch, head) dims
            shape = (1,) * (x.dim() - 2) + cos.shape
            cos = cos.view(shape)
            sin = sin.view(shape)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(x.dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Gated FFN: W2( SiLU(W1 x) ⊙ W3 x ). No biases."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)  # gate
        self.w3 = Linear(d_model, d_ff)  # up
        self.w2 = Linear(d_ff, d_model)  # down

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


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


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE and GQA. Heads are a batch dim."""

    def __init__(self, cfg: ModelConfig, rope: RotaryPositionalEmbedding) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.rope = rope
        self.q_proj = Linear(cfg.d_model, self.n_heads * self.head_dim)
        self.k_proj = Linear(cfg.d_model, self.n_kv * self.head_dim)
        self.v_proj = Linear(cfg.d_model, self.n_kv * self.head_dim)
        self.o_proj = Linear(self.n_heads * self.head_dim, cfg.d_model)
        # QK-norm (Qwen3/Gemma3/OLMo2): RMSNorm each head's q,k over head_dim *before* RoPE, so the
        # attention logit q·kᵀ/√d cannot run away — the #1 large-model bf16 instability. When off,
        # these are nn.Identity ⇒ the path is byte-identical to the no-QK-norm model.
        self.q_norm: nn.Module = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm: nn.Module = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: AnyKVCache | None = None,
        layer_idx: int | None = None,
    ) -> Tensor:
        b, s, _ = x.shape
        # project, split into heads, make head a batch dim: (B, H, S, head_dim)
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)

        # QK-norm before RoPE (no-op unless cfg.qk_norm): unit-RMS each head's q,k over head_dim.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE rotates Q, K at their ABSOLUTE positions — so a cached token at position t is
        # rotated correctly even when only one new token is processed this step.
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        past_len = 0
        row_mask: Tensor | None = None  # per-row key mask for ragged batched decode
        kernel_out: Tensor | None = None  # paged Triton decode path bypasses SDPA entirely
        if isinstance(cache, SlotKVCache):
            # Iteration-level batched decode (R3b/R4.1): one token per slot, each at its offset.
            if layer_idx is None:
                raise ValueError("slot-cache decode requires layer_idx")
            if s != 1:
                raise ValueError("slot-cache decode processes exactly one token per row")
            lengths = cache.lengths  # (B,) pre-write lengths; advance() bumps after all layers
            cache.write_decode(layer_idx, k, v)
            if isinstance(cache, PagedKVCache) and cache.use_kernel:
                # R4.1b: fused paged decode — reads blocks via the table; no gathered padded
                # view, no materialized scores, no GQA repeat (the R3b padding tax removed).
                from scratch_llm.kernels.paged_decode_triton import paged_decode_attention

                kernel_out = paged_decode_attention(
                    q,
                    cache._pool_k[layer_idx],
                    cache._pool_v[layer_idx],
                    cache.block_table,
                    lengths,
                )
            else:
                # Gather path (dense: a free slice; paged: the P4.1.3 gather copy — the oracle).
                k, v = cache.decode_view(layer_idx)
                # Write-then-mask: row b attends keys j ≤ lengths[b] — its history plus the key
                # it just wrote. An inactive row (length 0) attends exactly its own garbage key
                # at j=0: no all-masked softmax row ⇒ no NaN; the scheduler discards its output.
                k_pos = torch.arange(k.shape[2], device=x.device)
                row_mask = (k_pos.unsqueeze(0) <= lengths.unsqueeze(1))[:, None, None, :]
        elif cache is not None and layer_idx is not None:
            past_len = cache.length  # constant across layers within one forward
            cache.append(layer_idx, k, v)
            full = cache.get(layer_idx)
            assert full is not None  # just appended
            k, v = full  # K, V now span [past ; new]

        if kernel_out is not None:
            out = kernel_out  # (B, H, 1, head_dim)
        else:
            if self.n_kv != self.n_heads:  # GQA: each kv head serves a group of query heads
                repeats = self.n_heads // self.n_kv
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)

            # Causal mask over absolute positions: query row i (abs past_len+i) may attend key j
            # iff j <= past_len+i. A single new token (s==1) attends all cached keys → no mask.
            if row_mask is not None:
                mask = row_mask  # (B, 1, 1, view_len) — ragged per-row visibility
            elif s == 1:
                mask = None
            else:
                total = past_len + s
                q_pos = torch.arange(past_len, total, device=x.device).unsqueeze(1)  # (s, 1)
                k_pos = torch.arange(total, device=x.device).unsqueeze(0)  # (1, total)
                mask = k_pos <= q_pos  # (s, total) bool, True = attend

            out = scaled_dot_product_attention(q, k, v, mask)  # (B, H, S, head_dim)
        out = out.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(RMSNorm(x)); x + FFN(RMSNorm(x)).

    The FFN is dense ``SwiGLU`` unless the model opts into MoE *and* this layer is past the
    leading dense layers (``cfg.moe.n_dense_layers``) — then it is a :class:`MoEFeedForward`.
    A MoE FFN returns its contribution plus per-layer stats; ``forward`` surfaces those stats
    (``None`` for a dense layer) so the LM can aggregate them. The residual add is identical
    either way — the FFN returns the *delta only*."""

    def __init__(self, cfg: ModelConfig, rope: RotaryPositionalEmbedding, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg, rope)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.is_moe = cfg.moe is not None and layer_idx >= cfg.moe.n_dense_layers
        self.ffn: nn.Module
        if self.is_moe:
            assert cfg.moe is not None
            from scratch_llm.moe import MoEFeedForward

            self.ffn = MoEFeedForward(cfg.d_model, cfg.moe)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_dim)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: AnyKVCache | None = None,
        layer_idx: int | None = None,
    ) -> tuple[Tensor, MoEStats | None]:
        x = x + self.attn(self.attn_norm(x), positions, cache, layer_idx)
        if self.is_moe:
            delta, stats = self.ffn(self.ffn_norm(x))
        else:
            delta, stats = self.ffn(self.ffn_norm(x)), None
        x = x + delta
        return x, stats


class TransformerLM(nn.Module):
    """Full decoder LM: embed → N pre-norm blocks → final RMSNorm → LM head.

    forward(token_ids: (B, S) long) → logits (B, S, vocab_size).

    Pass a :class:`KVCache` for incremental decoding: positions continue from the cache length,
    and K/V are appended per layer. With ``cache=None`` this is the full (training) forward,
    unchanged. The cached path must produce logits identical to the full recompute.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = Embedding(cfg.vocab_size, cfg.d_model)
        # One RoPE instance shared across blocks (buffers are identical, saves memory).
        rope = RotaryPositionalEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, rope, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = Linear(cfg.d_model, cfg.vocab_size)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    def forward(
        self, token_ids: Tensor, cache: AnyKVCache | None = None, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, AuxOutput]:
        """``return_aux=False`` (default, and the decode path) returns just logits — identical
        to the dense build. ``return_aux=True`` returns ``(logits, AuxOutput)`` with the summed
        MoE aux/z losses and per-layer routing diagnostics for the training objective."""
        s = token_ids.shape[1]
        if isinstance(cache, SlotKVCache):
            # Ragged batched decode: each slot's next token sits at its own absolute position.
            positions = cache.lengths.unsqueeze(1)  # (B, 1) per-row RoPE positions
        else:
            start = 0 if cache is None else cache.length
            positions = torch.arange(start, start + s, device=token_ids.device)
        x = self.token_emb(token_ids)
        layer_stats: list[MoEStats] = []
        for layer_idx, block in enumerate(self.blocks):
            x, stats = block(x, positions, cache, layer_idx)
            if stats is not None:
                layer_stats.append(stats)
        x = self.final_norm(x)
        if cache is not None:
            cache.advance(s)  # bump once, after all layers, so each layer saw the same start
        logits = self.lm_head(x)
        if return_aux:
            return logits, self._aggregate_aux(layer_stats)
        return logits

    def _aggregate_aux(self, layer_stats: list[MoEStats]) -> AuxOutput:
        """Sum the MoE aux/z losses across layers (zeros on a dense model)."""
        from scratch_llm.moe import AuxOutput

        if layer_stats:
            aux_loss = torch.stack([s.aux_loss for s in layer_stats]).sum()
            z_loss = torch.stack([s.z_loss for s in layer_stats]).sum()
        else:
            device = self.lm_head.weight.device
            aux_loss = torch.zeros((), device=device)
            z_loss = torch.zeros((), device=device)
        return AuxOutput(aux_loss=aux_loss, z_loss=z_loss, layers=layer_stats)

    def moe_update_biases(self) -> None:
        """Aux-loss-free balancing step for every MoE layer — call after ``optimizer.step()``."""
        from scratch_llm.moe import MoEFeedForward

        for block in self.blocks:
            ffn = block.ffn  # type: ignore[union-attr]
            if isinstance(ffn, MoEFeedForward):
                ffn.router.update_bias(ffn.cfg.bias_update_speed)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean token cross-entropy from logits, computed via logsumexp (cancels log∘exp).

    logits: (..., vocab_size); targets: (...) long. At init (≈uniform logits) this is
    ≈ log(vocab_size).
    """
    logits = logits.float()
    log_z = torch.logsumexp(logits, dim=-1)
    chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_z - chosen).mean()
