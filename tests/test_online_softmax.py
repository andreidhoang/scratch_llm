"""A4 R1 — online (streaming) softmax must reproduce a 3-pass reference to fp64 precision.

CPU-only (pure torch, no Triton/CUDA) — runs in the ``not gpu`` commit gate. The load-bearing case
is the LATE-OUTLIER adversarial test: a +50 spike in the final tile forces the running-max rescale of
the already-accumulated denominator. If the ``exp(m_old − m_new)`` correction is wrong, that test
fails while the easy cases pass.
"""

from __future__ import annotations

import pytest
import torch

from scratch_llm.kernels.online_softmax import (
    online_softmax,
    online_softmax_normalizer,
    three_pass_softmax,
)


@pytest.mark.parametrize("tile", [1, 8, 64, 128, 1000])
@pytest.mark.parametrize("n", [1, 7, 64, 200, 1024])
def test_online_matches_three_pass_fp64(n: int, tile: int) -> None:
    """Core invariant: online == 3-pass reference to <1e-6, any tile size (incl. tile ≥ n)."""
    torch.manual_seed(0)
    x = torch.randn(3, n, dtype=torch.float64)
    out = online_softmax(x, tile=tile)
    ref = three_pass_softmax(x)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


def test_online_matches_torch_softmax() -> None:
    """Also agree with the library softmax (sanity on the normalize step)."""
    torch.manual_seed(1)
    x = torch.randn(4, 513, dtype=torch.float64)
    out = online_softmax(x, tile=64)
    ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(out, ref, atol=1e-9, rtol=1e-9)


def test_late_outlier_rescale() -> None:
    """ADVERSARIAL: a +50 outlier in the LAST tile — the running max is small until the very end, so
    the denominator accumulated over all earlier tiles must be rescaled by exp(m_old−m_new). This is
    the one case that distinguishes correct online softmax from a naive running-sum."""
    torch.manual_seed(2)
    n, tile = 512, 64
    x = torch.randn(1, n, dtype=torch.float64) * 0.1  # small values → running max ≈ 0.x
    x[0, -1] = 50.0  # the spike arrives dead last, after every earlier tile is accumulated
    out = online_softmax(x, tile=tile)
    ref = three_pass_softmax(x)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
    # The spike must dominate the distribution and nothing may be NaN/Inf from the rescale.
    assert out[0, -1] > 0.99
    assert torch.isfinite(out).all()


def test_early_outlier_no_spurious_rescale() -> None:
    """Mirror case: the +50 lands in the FIRST tile. Here the max never changes again, so every later
    tile's correction factor is exp(0)=1 — a wrong-signed correction would corrupt this too."""
    n, tile = 256, 32
    x = torch.full((1, n), -1.0, dtype=torch.float64)
    x[0, 0] = 50.0
    out = online_softmax(x, tile=tile)
    ref = three_pass_softmax(x)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)


def test_normalizer_denominator_ge_one_after_max() -> None:
    """The denominator includes the max term exp(m−m)=1, so d ≥ 1 always (a cheap invariant that
    catches a denominator that failed to include the current element)."""
    torch.manual_seed(3)
    x = torch.randn(5, 300, dtype=torch.float64)
    _, d = online_softmax_normalizer(x, tile=64)
    assert (d >= 1.0 - 1e-9).all()


def test_all_equal_row_is_uniform() -> None:
    """A constant row → uniform 1/n softmax (no rescale ever fires; pure accumulation)."""
    n = 100
    x = torch.full((2, n), 3.14, dtype=torch.float64)
    out = online_softmax(x, tile=16)
    torch.testing.assert_close(out, torch.full_like(out, 1.0 / n), atol=1e-12, rtol=0)
