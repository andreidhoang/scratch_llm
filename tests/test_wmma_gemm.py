"""A3 Rung 1 — WMMA tensor-core GEMM oracle (GPU).

The kernel issues ``nvcuda::wmma`` fragments by hand (load_matrix_sync -> mma_sync ->
store_matrix_sync) with an FP32 accumulator. The oracle is ``torch.matmul`` computed in FP32 over the
same FP16-rounded inputs: the tensor core also accumulates in FP32, so the two must agree to FP16-input
tolerance on random AND adversarial shapes.

Load-bearing numerics test: the FP16-accumulate variant rounds the running sum to FP16 on every
``mma_sync``, so its reduction error GROWS with K and, at large K, dwarfs the FP32-accumulate error —
this is *why* tensor cores keep an FP32 accumulator.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu


def _require_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32 matmul over the FP16-rounded operands — the FP32-accumulate oracle."""
    return a.float() @ b.float()


# Random square + adversarial shapes: K not a multiple of the 16-wide MMA-K tile, M/N remainder tiles
# that don't fill a 128-block, odd N (the store-alignment edge), GEMV-shaped, and a sub-tile shape.
_SHAPES = [
    (256, 256, 256),
    (2048, 2048, 2048),
    (512, 512, 512),
    (301, 517, 203),  # all three remainders, odd N
    (129, 130, 131),  # 1-past-a-block on every axis, odd N
    (133, 71, 255),  # odd N=255, K=71 (non-mult of 16)
    (128, 48, 128),  # K a non-multiple of BK=32
    (1, 4096, 4096),  # GEMV row
    (4096, 4096, 1),  # GEMV column
    (16, 16, 16),  # a single MMA atom
    (15, 17, 19),  # sub-tile, all odd
]


@pytest.mark.parametrize(("m", "k", "n"), _SHAPES)
def test_wmma_matches_torch_matmul(m: int, k: int, n: int) -> None:
    _require_gpu()
    from scratch_llm.kernels.wmma_gemm import wmma_gemm

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    out = wmma_gemm(a, b)
    ref = _ref(a, b)
    assert out.shape == (m, n)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2), (
        f"{m}x{k}x{n}: max |Δ| = {(out - ref).abs().max().item():.2e}"
    )


def test_wmma_single_large_outlier() -> None:
    """A single large outlier in A and B must not corrupt the tile (a common tensor-core bug)."""
    _require_gpu()
    from scratch_llm.kernels.wmma_gemm import wmma_gemm

    torch.manual_seed(1)
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    a[100, 100] = 200.0
    b[100, 300] = -150.0
    out = wmma_gemm(a, b)
    ref = _ref(a, b)
    # The outlier lands at (100, 300) ~ -3e4; tolerate FP16-input rounding on a large magnitude.
    assert torch.allclose(out, ref, rtol=1e-2, atol=3e-1), (
        f"max |Δ| = {(out - ref).abs().max().item():.2e}"
    )


def test_wmma_long_odd_k_remainder_tiles() -> None:
    """Independent adversarial: a long ODD K (non-multiple of WMMA_K=16 and BK=32) combined with
    M/N remainder tiles that don't fill a 128-block. Stresses the zero-padded K-tail staging AND the
    guarded edge-store on the same call; verified element-exact at a *tight* FP32-accumulate tolerance
    (rtol/atol 2e-3, far inside the 1e-2 gate) to prove the accumulator is genuinely FP32."""
    _require_gpu()
    from scratch_llm.kernels.wmma_gemm import wmma_gemm

    torch.manual_seed(7)
    m, k, n = 200, 4099, 200  # K=4099 odd; M,N remainder past the 128-block
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    out = wmma_gemm(a, b)
    ref = _ref(a, b)
    assert out.shape == (m, n)
    assert torch.allclose(out, ref, rtol=2e-3, atol=2e-3), (
        f"max |Δ| = {(out - ref).abs().max().item():.2e}"
    )


def test_fp16_accumulate_error_grows_with_k() -> None:
    """WHY FP32 accumulate: the FP16-accumulate variant's error grows with K and, at large K, far
    exceeds the FP32-accumulate error at the same K. Both are compared to the FP32 oracle."""
    _require_gpu()
    from scratch_llm.kernels.wmma_gemm import wmma_gemm, wmma_gemm_fp16acc

    torch.manual_seed(0)
    m = n = 256
    ks = [128, 512, 2048, 8192]

    def mean_rel(out: torch.Tensor, ref: torch.Tensor) -> float:
        return (out - ref).abs().mean().item() / ref.abs().mean().item()

    err16: list[float] = []
    err32: list[float] = []
    for k in ks:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.5
        b = torch.randn(k, n, device="cuda", dtype=torch.float16) * 0.5
        ref = _ref(a, b)
        err32.append(mean_rel(wmma_gemm(a, b), ref))
        err16.append(mean_rel(wmma_gemm_fp16acc(a, b), ref))
        del a, b, ref
    torch.cuda.empty_cache()

    # (1) FP16-accumulate error grows monotonically with K (rounding compounds over the reduction).
    assert err16[-1] > err16[0], f"fp16-accum error did not grow with K: {err16}"
    for lo, hi in zip(err16[:-1], err16[1:], strict=True):
        assert hi >= lo * 0.9, f"fp16-accum error not ~monotone in K: {err16}"
    # (2) At the largest K, FP16-accumulate is dramatically worse than FP32-accumulate — the point.
    assert err16[-1] > 20 * err32[-1], (
        f"fp16-accum ({err16[-1]:.2e}) not >> fp32-accum ({err32[-1]:.2e}) at K={ks[-1]}"
    )
    # (3) FP32-accumulate stays tight even at large K (the reduction it makes numerically stable).
    assert err32[-1] < 1e-3, f"fp32-accum error too large at K={ks[-1]}: {err32[-1]:.2e}"
