"""A4 R1 — single-row ONLINE (streaming) softmax: the numerical core FlashAttention fuses.

A textbook softmax is **3 passes** over the row: (1) find the max ``m``, (2) sum ``Σ exp(xᵢ−m)`` to
get the denominator ``d``, (3) divide. Three reads of the whole row. FlashAttention cannot afford
that — it sees the scores one *tile* at a time and must never store the whole row. Online softmax
fuses passes 1 and 2 into a **single streaming pass** that maintains a running max and a running
denominator, *rescaling the denominator whenever a new, larger max arrives*:

    mᵢ = max(mᵢ₋₁, xᵢ)
    dᵢ = dᵢ₋₁ · exp(mᵢ₋₁ − mᵢ)  +  exp(xᵢ − mᵢ)
         └── rescale the old sum to the new max ──┘   └── add this element ──┘

The ``exp(mᵢ₋₁ − mᵢ)`` correction is the whole trick: when a late outlier bumps the max, every
previously-accumulated term is retro-actively re-based to the new max in O(1), with no re-read. This
is *exactly* the ``corr = exp(m − m_new)`` line in ``flash_attention_forward`` — here it is isolated
over a 1-D stream so the recurrence can be tested on its own.

Falsifiable invariants (tests/test_online_softmax.py):
  * `online_softmax` matches a 3-pass reference softmax to <1e-6 (fp64), for any tile size.
  * ADVERSARIAL: a ``+50`` outlier arriving in the *last* tile (after the running max is small) must
    still match — the rescale of the already-accumulated denominator is what makes this correct.
Kill criterion: mismatch on the late-outlier case ⇒ the ``exp(m_old − m_new)`` rescale is missing or
mis-signed, which is the single most common online-softmax bug.
"""

from __future__ import annotations

import torch
from torch import Tensor


def three_pass_softmax(x: Tensor) -> Tensor:
    """The reference: literal 3-pass safe softmax over the last dim (max, sum, divide).

    This is the pedagogical oracle — deliberately *not* ``torch.softmax`` — so the test compares the
    online recurrence against the textbook algorithm it is meant to reproduce, term for term."""
    m = x.amax(dim=-1, keepdim=True)  # pass 1: max
    e = torch.exp(x - m)  # (uses the shifted values)
    d = e.sum(dim=-1, keepdim=True)  # pass 2: denominator
    return e / d  # pass 3: normalize


def online_softmax_normalizer(x: Tensor, *, tile: int = 64) -> tuple[Tensor, Tensor]:
    """Streaming ``(m, d)`` over the last dim in tiles — ONE pass, the fused max+denominator.

    Returns the final running max ``m`` and denominator ``d`` (each ``(..., 1)``), computed by the
    online recurrence tile-by-tile. No full-row max pass; the denominator is rescaled on every tile
    that raises the max. Runs in the input dtype (use fp64 for the high-precision oracle test)."""
    *lead, n = x.shape
    dtype, device = x.dtype, x.device
    m = torch.full((*lead, 1), float("-inf"), dtype=dtype, device=device)  # running max
    d = torch.zeros((*lead, 1), dtype=dtype, device=device)  # running denominator

    for start in range(0, n, tile):
        xi = x[..., start : start + tile]  # this tile of the stream
        m_tile = xi.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m, m_tile)
        corr = torch.exp(m - m_new)  # rescale factor for the old sum; 0 on the first tile (m=-inf)
        d = corr * d + torch.exp(xi - m_new).sum(dim=-1, keepdim=True)
        m = m_new
    return m, d


def online_softmax(x: Tensor, *, tile: int = 64) -> Tensor:
    """Full softmax over the last dim via the online normalizer, then a normalize pass.

    The normalizer ``(m, d)`` comes from a single streaming pass (no separate max pass — that is the
    FA-relevant part); the final ``exp(x − m)/d`` is the standard normalize. Bit-for-bit equivalent
    to a 3-pass safe softmax, but computed the way a tiled kernel must."""
    m, d = online_softmax_normalizer(x, tile=tile)
    return torch.exp(x - m) / d
