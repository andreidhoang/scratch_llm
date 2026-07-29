"""A4 R0 — naive 3-kernel attention: correctness oracle + the O(N²)-memory blowup that motivates FA.

Two kinds of test:
  * CPU (``not gpu`` gate): correctness vs ``F.scaled_dot_product_attention`` (<1e-3 fp32), and the
    *analytic* memory blowup — the N×N score matrix grows as N² and at N=16K dwarfs the fused kernel's
    constant working set. Pure arithmetic, no allocation.
  * GPU (``gpu`` marker): the blowup MEASURED — peak CUDA memory of ``naive_attention`` scales ∝ N²
    (quadratic fit), while the fused ``flash_attention_forward`` stays roughly flat.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from scratch_llm.kernels.attention.reference_naive import (
    attention_memory_footprint,
    naive_attention,
)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("n", [1, 16, 64, 200])
def test_naive_matches_sdpa_fp32(n: int, is_causal: bool) -> None:
    """Correctness: naive 3-op attention == SDPA to <1e-3 (fp32), causal + non-causal, incl. N=1."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, n, 32, dtype=torch.float32) for _ in range(3))
    out = naive_attention(q, k, v, is_causal=is_causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


def test_naive_matches_flash_oracle() -> None:
    """The strawman and the fused oracle must agree — same math, different memory profile."""
    from scratch_llm.kernels import flash_attention_forward

    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 128, 48, dtype=torch.float32) for _ in range(3))
    naive = naive_attention(q, k, v, is_causal=True)
    fused, _ = flash_attention_forward(q, k, v, is_causal=True)
    torch.testing.assert_close(naive, fused, atol=1e-4, rtol=1e-4)


def test_causal_no_future_leak() -> None:
    """Row 0 attends only to key 0, so its output must equal V[0] exactly (a hard causal check)."""
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 1, 8, 16, dtype=torch.float64) for _ in range(3))
    out = naive_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out[..., 0, :], v[..., 0, :], atol=1e-12, rtol=0)


# --- the WHY: O(N²) memory blowup, analytic (CPU) ---------------------------------------------


def test_score_matrix_is_quadratic_in_n() -> None:
    """Doubling N quadruples the score-matrix bytes (∝ N²); the fused working set does not."""
    f_n = attention_memory_footprint(1024, 1024, 64)
    f_2n = attention_memory_footprint(2048, 2048, 64)
    ratio = f_2n.naive_score_bytes / f_n.naive_score_bytes
    assert math.isclose(ratio, 4.0, rel_tol=1e-9)
    # Fused output/stats grow only linearly in N (the N×N term is gone entirely).
    lin = f_2n.flash_working_bytes / f_n.flash_working_bytes
    assert lin < 2.5  # ~2× (linear), never 4× (quadratic)


def test_16k_blowup_dwarfs_fused_sram() -> None:
    """At N=16K, fp32, one head: the naive score matrix is ~1 GiB and >1000× the fused working set —
    the concrete number that says 'you cannot just make the tile bigger', you must not materialize S."""
    f = attention_memory_footprint(16_384, 16_384, 64, batch_heads=1, dtype=torch.float32)
    gib = f.naive_score_bytes / 2**30
    assert gib == pytest.approx(1.0, rel=0.01)  # 16384² · 4 B = 1.0 GiB
    # vs the fused kernel's CONSTANT on-chip scratch (tile²·4 B = 16 KiB): the N×N matrix is >60000×
    # too big to ever live on-chip — and it never needs to (tiling holds one block at a time).
    assert f.onchip_blowup_ratio > 60_000.0
    # A typical GPU SRAM budget is ~100s of KB per SM; 1 GiB cannot live on-chip, full stop.
    assert f.naive_score_bytes > 200_000 * 5000  # ≫ (per-SM SRAM) × (SM count) order of magnitude


# --- the WHY: measured on the GPU -------------------------------------------------------------


@pytest.mark.gpu
def test_measured_peak_memory_is_quadratic() -> None:
    """MEASURED: naive attention's peak CUDA allocation scales ∝ N² (the score matrix dominates),
    while the fused flash oracle's peak stays near-flat. Small d and batch keep this well under the
    shared 24 GB card (largest alloc here ≈ 4096²·4 B ≈ 64 MiB per the N=4096 point)."""
    if not torch.cuda.is_available():  # pragma: no cover - environment guard
        pytest.skip("CUDA required")
    from scratch_llm.kernels import flash_attention_forward

    d = 16
    ns = [512, 1024, 2048, 4096]

    def peak_mb(fn, *args) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn(*args, is_causal=False)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20

    # Warm up cuBLAS/Triton so their one-time lazy workspace init doesn't pollute the first point.
    warm = torch.randn(1, 1, 128, d, device="cuda", dtype=torch.float32)
    naive_attention(warm, warm, warm, is_causal=False)
    flash_attention_forward(warm[:, 0], warm[:, 0], warm[:, 0], is_causal=False)
    del warm

    naive_peaks: list[float] = []
    fused_peaks: list[float] = []
    for n in ns:
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 1, n, d, device="cuda", dtype=torch.float32) for _ in range(3))
        naive_peaks.append(peak_mb(naive_attention, q, k, v))
        # flash oracle wants (..., N, d); drop the head dim to a single batch row.
        fused_peaks.append(peak_mb(flash_attention_forward, q[:, 0], k[:, 0], v[:, 0]))
        del q, k, v

    # Peaks carry a constant offset (cuBLAS/Triton workspace) that swamps the N² term at small N. The
    # clean quadratic signature is in the INCREMENTS: for a ∝N² curve sampled at doubling N, each
    # step's increase quadruples (Δ(N²) at 2N is 4× Δ(N²) at N) — and the constant cancels in Δ.
    incr = [naive_peaks[i + 1] - naive_peaks[i] for i in range(len(ns) - 1)]
    assert all(x > 0 for x in incr), f"naive peaks not monotone: {naive_peaks}"
    ratios = [incr[i + 1] / incr[i] for i in range(len(incr) - 1)]
    for r in ratios:
        assert 3.0 < r < 5.0, f"naive increments {incr} (ratios {ratios}) not ~4× → not quadratic"
    # Fused never materializes N×N: its total growth is far below naive's (only O(N·d) output grows).
    naive_growth = naive_peaks[-1] - naive_peaks[0]
    fused_growth = fused_peaks[-1] - fused_peaks[0]
    assert fused_growth < naive_growth / 5.0, f"fused {fused_peaks} vs naive {naive_peaks}"
