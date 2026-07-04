"""A3 R0 — naive SMEM-blocked bf16 GEMM (CUDA C++, CUDA cores) vs the torch.matmul (cuBLAS) oracle.

torch.matmul with bf16 inputs and fp32 tensor-core accumulation is the reference. The baseline kernel
must match it at the canonical GEMM tolerance (normalized ‖Δ‖∞/‖ref‖∞ ≤ 1e-2) — including the
adversarial shapes that break a GEMM mishandling remainders: M, N, K each non-multiples of the 32-tile;
K non-multiple alone; M=1 (degenerates to a GEMV, a single partial row); and a lone large outlier that
stresses the fp32 accumulator against bf16 overflow. This is the CUDA-core floor WMMA later beats.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("SMEM GEMM kernel requires CUDA", allow_module_level=True)

from scratch_llm.kernels.gemm_smem_cuda import gemm_smem  # noqa: E402

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
    """Gate on the normalized relative error, not per-element rtol: float addition is non-associative,
    so the naive sequential-fp32 reduction and cuBLAS's blocked order bit-differ on a catastrophically
    cancelled element. ‖out-ref‖∞/‖ref‖∞ stays O(1) for a real indexing/mask bug but tolerates
    legitimate reduction-order divergence."""
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    assert out.dtype == ref.dtype, f"dtype {out.dtype} != {ref.dtype}"
    d = (out.float() - ref.float()).abs()
    rel = d.max().item() / (ref.float().abs().max().item() + 1e-30)
    assert rel <= _RTOL, f"normalized rel err {rel:.3e} > rtol {_RTOL} (max |Δ|={d.max().item():.3e})"


@pytest.mark.parametrize("scale", [1e-2, 1.0, 1e2], ids=["s1e-2", "s1", "s1e2"])
def test_smem_gemm_magnitude_range(scale: float) -> None:
    """Matches cuBLAS across three input magnitudes (the rtol gate is scale-invariant)."""
    torch.manual_seed(0)
    m, k, n = 512, 1024, 384
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * scale
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * scale
    _assert_matches(gemm_smem(a, b), _matmul_ref(a, b))
    del a, b
    torch.cuda.empty_cache()


def test_smem_gemm_remainder_tiles() -> None:
    """M, N, K each a non-multiple of the 32-tile — exercises the masked-load remainder path."""
    torch.manual_seed(1)
    m, k, n = 250, 130, 197
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches(gemm_smem(a, b), _matmul_ref(a, b))


def test_smem_gemm_k_non_multiple() -> None:
    """K alone a non-multiple of 32 — the last K-tile is partially masked to zero."""
    torch.manual_seed(5)
    m, k, n = 256, 100, 256  # M,N tile-aligned; K is not
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches(gemm_smem(a, b), _matmul_ref(a, b))


def test_smem_gemm_gemv_degenerate() -> None:
    """M=1 collapses the GEMM to a GEMV — a single partial row, the rest of the tile masked."""
    torch.manual_seed(2)
    m, k, n = 1, 4096, 320
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches(gemm_smem(a, b), _matmul_ref(a, b))


def test_smem_gemm_single_large_outlier() -> None:
    """One huge entry dominates its row/col — stresses the fp32 accumulator vs bf16 overflow."""
    torch.manual_seed(3)
    m, k, n = 128, 256, 96
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    a[7, 13] = 300.0
    b[13, 41] = -250.0
    _assert_matches(gemm_smem(a, b), _matmul_ref(a, b))


def test_smem_gemm_2048_cubed() -> None:
    """At 2048^3 (spills L1, well past the ridge) the baseline still matches cuBLAS at rtol 1e-2."""
    torch.manual_seed(4)
    n = 2048
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    _assert_matches(gemm_smem(a, b), _matmul_ref(a, b))
    del a, b
    torch.cuda.empty_cache()
