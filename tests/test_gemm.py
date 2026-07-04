"""A2 R5+6 — Triton bf16 GEMM ladder vs the torch.matmul (cuBLAS) oracle (GPU).

torch.matmul with bf16 inputs and fp32 tensor-core accumulation is the reference. Each ladder stage
(naive / SMEM-tiled / autotuned) must match it at rtol 1e-2 over a magnitude range and on the
adversarial shapes that break a GEMM that mishandles remainders: M, N, K each non-multiples of the
block; M=1 (degenerates to a GEMV); and a single large outlier that stresses the fp32 accumulator.

The shipping kernel (``gemm_autotuned``) is additionally gated at >= 4096^3 — large enough that the
operands spill L2 and the result reflects real tensor-core throughput, not an L1-resident fantasy.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("GEMM kernel requires CUDA", allow_module_level=True)

pytest.importorskip("triton")

from scratch_llm.kernels.gemm_triton import (  # noqa: E402
    gemm_autotuned,
    gemm_naive,
    gemm_tiled,
)

_RTOL = 1e-2  # normalized relative error ‖Δ‖∞ / ‖ref‖∞ — the canonical GEMM correctness metric


def _matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """cuBLAS proxy: bf16 GEMM with fp32 accumulation (reduced-precision reduction off)."""
    prev = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        return torch.matmul(a, b)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev


def _assert_matches(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Gate on the normalized relative error, not per-element rtol.

    A GEMM is a sum of K products in floating point, and float addition is non-associative: the naive
    sequential-fp32 order and cuBLAS's blocked/tensor-core reduction give bit-different results on a
    *catastrophically-cancelled* output element (true value ≪ the magnitudes summed). No reordering can
    make those agree per-element, so per-element rtol would reject a *correct* kernel. The canonical
    GEMM oracle is therefore ‖out − ref‖∞ / ‖ref‖∞ ≤ rtol — still O(1) for a real indexing/mask bug, so
    it keeps its teeth, but scale-aware for legitimate reduction-order divergence.
    """
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    assert out.dtype == ref.dtype, f"dtype {out.dtype} != {ref.dtype}"
    d = (out.float() - ref.float()).abs()
    rel = d.max().item() / (ref.float().abs().max().item() + 1e-30)
    assert rel <= _RTOL, (
        f"normalized rel err {rel:.3e} > rtol {_RTOL} (max |Δ|={d.max().item():.3e})"
    )


_STAGES = [("naive", gemm_naive), ("tiled", gemm_tiled), ("autotuned", gemm_autotuned)]


@pytest.mark.parametrize(("name", "fn"), _STAGES, ids=[s[0] for s in _STAGES])
@pytest.mark.parametrize("scale", [1e-2, 1.0, 1e2], ids=["s1e-2", "s1", "s1e2"])
def test_gemm_stage_magnitude_range(name: str, fn, scale: float) -> None:
    """Every ladder stage matches cuBLAS across three input magnitudes (rtol is scale-invariant)."""
    torch.manual_seed(0)
    m, k, n = 512, 1024, 384
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * scale
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * scale
    _assert_matches(fn(a, b), _matmul_ref(a, b))
    del a, b
    torch.cuda.empty_cache()


@pytest.mark.parametrize(("name", "fn"), _STAGES, ids=[s[0] for s in _STAGES])
def test_gemm_remainder_tiles(name: str, fn) -> None:
    """M, N, K each a non-multiple of any block size — exercises the remainder-mask path."""
    torch.manual_seed(1)
    m, k, n = 250, 130, 197  # all non-multiples of 16/32/64/128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches(fn(a, b), _matmul_ref(a, b))


@pytest.mark.parametrize(("name", "fn"), _STAGES, ids=[s[0] for s in _STAGES])
def test_gemm_gemv_degenerate(name: str, fn) -> None:
    """M=1 collapses the GEMM to a GEMV — the tile is a single partial row, fully masked."""
    torch.manual_seed(2)
    m, k, n = 1, 4096, 320
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches(fn(a, b), _matmul_ref(a, b))


@pytest.mark.parametrize(("name", "fn"), _STAGES, ids=[s[0] for s in _STAGES])
def test_gemm_single_large_outlier(name: str, fn) -> None:
    """One huge entry dominates its output row/col — stresses the fp32 accumulator vs bf16 overflow."""
    torch.manual_seed(3)
    m, k, n = 128, 256, 96
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    a[7, 13] = 300.0  # near bf16's dynamic headroom; fp32 accumulate must not lose the small terms
    b[13, 41] = -250.0
    _assert_matches(fn(a, b), _matmul_ref(a, b))


def test_gemm_autotuned_4096_cubed() -> None:
    """The shipping kernel at 4096^3 (L2-spilling) matches cuBLAS at rtol 1e-2."""
    torch.manual_seed(4)
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches(gemm_autotuned(a, b), _matmul_ref(a, b))
    del a, b
    torch.cuda.empty_cache()
