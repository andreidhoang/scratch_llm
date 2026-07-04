"""A4 R0 — NAIVE 3-kernel attention: the O(N²) baseline FlashAttention exists to kill.

This is the *why* of flash attention, made falsifiable. Standard attention is three separate ops,
each materializing a full tensor to HBM and reading it back:

    1. S = QKᵀ / √d          — the full N×N score matrix   (O(N²) bytes)
    2. P = softmax(S)        — causal-masked, still N×N     (O(N²) bytes)
    3. O = P @ V             — collapses back to N×d

The killer is step 1/2: the N×N matrix S (and P) is written to and re-read from HBM. Its size grows
**quadratically** in sequence length — at N=16K, fp32, that is 1 GiB *per head*, and it never fits in
the ~100s-of-KB of on-chip SRAM. FlashAttention's whole trick is to **never materialize S**: it fuses
the three ops into one tiled kernel whose on-chip working set is O(tile²)+O(d), *constant* in N. The
`flash_attention_forward` oracle in this package is that fused recurrence; this module is the strawman
it beats.

Falsifiable invariants (tests/test_attention_naive.py):
  * `naive_attention` matches `F.scaled_dot_product_attention` to <1e-3 (fp32), causal + non-causal.
  * The score-matrix footprint grows as N² and, at N=16K, dwarfs the fused kernel's constant SRAM —
    demonstrated analytically (CPU) and by *measured* peak CUDA memory scaling ∝ N² (GPU).
Kill criterion: mismatch ⇒ the softmax axis or causal mask is wrong; non-quadratic growth ⇒ the
memory model is wrong and the FA motivation is unfounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def naive_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    is_causal: bool = False,
) -> Tensor:
    """Textbook 3-op attention. ``q,k,v`` are ``(..., N, d)``. Scale = 1/√d.

    Deliberately materializes the full ``(..., N, N)`` score matrix ``S`` and probability matrix
    ``P`` — this is the O(N²)-memory strawman, NOT to be used in production (that is what
    ``flash_attention_forward`` is for). Computes in fp32 at minimum for a fair correctness oracle."""
    d = q.shape[-1]
    n_q, n_k = q.shape[-2], k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    acc_dtype = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float32
    qf, kf, vf = q.to(acc_dtype), k.to(acc_dtype), v.to(acc_dtype)

    # Kernel 1 — the full N×N score matrix hits HBM.
    s = torch.einsum("...qd,...kd->...qk", qf, kf) * scale  # (..., N, N)
    if is_causal:
        mask = _causal_mask(n_q, n_k, q.device)  # (N, N), True = future key (disallowed)
        s = s.masked_fill(mask, float("-inf"))

    # Kernel 2 — softmax over keys; another N×N tensor.
    p = torch.softmax(s, dim=-1)  # (..., N, N)

    # Kernel 3 — collapse back to (..., N, d).
    o = torch.einsum("...qk,...kd->...qd", p, vf)
    return o.to(q.dtype)


def _causal_mask(n_q: int, n_k: int, device: torch.device) -> Tensor:
    """``(n_q, n_k)`` bool, True where a key is in the query's future (must be masked)."""
    q_idx = torch.arange(n_q, device=device)
    k_idx = torch.arange(n_k, device=device)
    return k_idx[None, :] > q_idx[:, None]


@dataclass(frozen=True)
class MemoryFootprint:
    """The memory story of one attention call: naive's N×N score matrix vs a fused kernel's SRAM.

    ``naive_score_bytes`` is the HBM the strawman must allocate for S (and again for P); the fused
    flash kernel materializes neither — its resident working set is ``flash_working_bytes`` (the O(N·d)
    output/stats + O(tile²) on-chip scratch), which is *independent of N* for the score matrix term."""

    n_q: int
    n_k: int
    d: int
    batch_heads: int
    itemsize: int
    naive_score_bytes: int
    flash_working_bytes: int
    tile_scratch_bytes: int

    @property
    def blowup_ratio(self) -> float:
        """How many times larger the naive score matrix is than the fused kernel's total working set.

        The fused set is dominated by the O(N·d) output both kernels must write, so this stays modest
        (the honest story: the N×N *intermediate* is the waste, not the output)."""
        return self.naive_score_bytes / self.flash_working_bytes

    @property
    def onchip_blowup_ratio(self) -> float:
        """Score matrix vs the fused kernel's CONSTANT on-chip scratch (O(tile²), independent of N).

        This is the real motivation: the N×N matrix cannot live in the ~KBs of on-chip SRAM, and it
        never needs to — a tiled kernel only ever holds one tile² block at a time."""
        return self.naive_score_bytes / self.tile_scratch_bytes


def attention_memory_footprint(
    n_q: int,
    n_k: int,
    d: int,
    *,
    batch_heads: int = 1,
    dtype: torch.dtype = torch.float32,
    tile: int = 64,
) -> MemoryFootprint:
    """Analytic memory model — no allocation, pure arithmetic (runs anywhere, incl. the CPU gate).

    * naive: one ``(batch_heads, N_q, N_k)`` score matrix = ``batch_heads · N_q · N_k · itemsize``
      bytes (the softmax reuses the buffer, so we count one N×N tensor — the *quadratic* term).
    * fused flash: never forms N×N. Its persistent footprint is the O(N·d) output + per-row (m, ℓ)
      stats, plus an O(tile²) on-chip score block that does NOT grow with N. We report that."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    naive_score_bytes = batch_heads * n_q * n_k * itemsize
    # Constant on-chip scratch: one tile² score block (independent of N) — what SRAM actually holds.
    tile_scratch_bytes = tile * tile * itemsize
    # Fused working set: output O (N_q·d) + running max/denom (2·N_q) per row, + a tile² score block.
    flash_working_bytes = batch_heads * (n_q * d + 2 * n_q) * itemsize + tile_scratch_bytes
    return MemoryFootprint(
        n_q=n_q,
        n_k=n_k,
        d=d,
        batch_heads=batch_heads,
        itemsize=itemsize,
        naive_score_bytes=naive_score_bytes,
        flash_working_bytes=flash_working_bytes,
        tile_scratch_bytes=tile_scratch_bytes,
    )
