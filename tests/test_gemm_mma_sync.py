"""A3 Rung 2 — mma.sync + ldmatrix + XOR-swizzled-SMEM warp-MMA GEMM vs torch.matmul (GPU).

The oracle is ``torch.matmul`` in FP32 (``a.float() @ b.float()``) — the same value cuBLAS computes
with fp32 tensor-core accumulation, but kept in FP32 so the single-large-outlier case (product far
past float16's 65504 ceiling) is representable in the reference too. The kernel takes float16 operands
(the ``mma.f16.f16`` type), accumulates in FP32, and returns FP32.

Correctness is the normalized relative error ‖out-ref‖∞ / ‖ref‖∞ (the canonical GEMM metric — float
addition is non-associative, so a per-element rtol would reject a correct kernel whose reduction order
differs from cuBLAS on a catastrophically-cancelled element; the normalized error still has O(1) teeth
for a real indexing / fragment-layout / swizzle bug). The adversarial cases are the ones that break a
tensor-core GEMM that mishandles fragments or remainders: K not a multiple of the 16/32 tiles, M/N
remainder tiles, M=1 (a GEMV), and a single large outlier that stresses the FP32 accumulator.

GPU-marked exactly like tests/test_paged_kernel.py (module-level ``pytestmark`` + a CUDA skip guard).
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("mma.sync kernel requires CUDA", allow_module_level=True)

from scratch_llm.kernels.gemm_mma_sync import gemm_mma_sync  # noqa: E402

_RTOL = 1e-2  # normalized ‖Δ‖∞ / ‖ref‖∞ — scale-invariant, fp32-accumulate GEMM tolerance


def _ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.matmul in FP32 (holds the outlier product that would overflow a float16 output)."""
    return torch.matmul(a.float(), b.float())


def _assert_matches(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    assert out.dtype == torch.float32, f"kernel must return fp32, got {out.dtype}"
    d = (out.float() - ref.float()).abs()
    rel = d.max().item() / (ref.float().abs().max().item() + 1e-30)
    assert rel <= _RTOL, (
        f"normalized rel err {rel:.3e} > rtol {_RTOL} (max |Δ|={d.max().item():.3e})"
    )


@pytest.mark.parametrize("scale", [1e-2, 1.0, 1e2], ids=["s1e-2", "s1", "s1e2"])
def test_magnitude_range(scale: float) -> None:
    """Matches the oracle across three input magnitudes (the normalized metric is scale-invariant)."""
    torch.manual_seed(0)
    m, k, n = 512, 1024, 384
    a = torch.randn(m, k, device="cuda", dtype=torch.float16) * scale
    b = torch.randn(k, n, device="cuda", dtype=torch.float16) * scale
    _assert_matches(gemm_mma_sync(a, b), _ref(a, b))
    del a, b
    torch.cuda.empty_cache()


def test_remainder_tiles() -> None:
    """M, N, K each a non-multiple of the 16/32/64 tiles — the bounds-masked zero-fill path."""
    torch.manual_seed(1)
    m, k, n = 250, 130, 197
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    _assert_matches(gemm_mma_sync(a, b), _ref(a, b))


@pytest.mark.parametrize("k", [1, 15, 16, 17, 33, 48], ids=lambda x: f"k{x}")
def test_k_remainder_of_mma_tile(k: int) -> None:
    """K around and below the MMA-K=16 / BK=32 boundaries — the partial-k-tile zero-fill must be exact."""
    torch.manual_seed(2)
    m, n = 80, 48
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    _assert_matches(gemm_mma_sync(a, b), _ref(a, b))


def test_gemv_degenerate() -> None:
    """M=1 collapses the GEMM to a GEMV — a single fully-masked partial row tile."""
    torch.manual_seed(3)
    m, k, n = 1, 4096, 320
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    _assert_matches(gemm_mma_sync(a, b), _ref(a, b))


def test_single_large_outlier() -> None:
    """One huge entry (product ~7.5e4, past float16's ceiling) — the FP32 accumulator/output must hold it."""
    torch.manual_seed(4)
    m, k, n = 128, 256, 96
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    a[7, 13] = 300.0
    b[13, 41] = -250.0
    ref = _ref(a, b)
    assert ref.abs().max().item() > 65504.0  # would overflow a float16 output — proves fp32 output
    _assert_matches(gemm_mma_sync(a, b), ref)


def test_4096_cubed() -> None:
    """The shipping size: 4096^3 (L2-spilling) matches cuBLAS at the fp32-accumulate tolerance."""
    torch.manual_seed(5)
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    _assert_matches(gemm_mma_sync(a, b), _ref(a, b))
    del a, b
    torch.cuda.empty_cache()
