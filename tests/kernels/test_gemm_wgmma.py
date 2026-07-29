"""Hopper WGMMA GEMM (sm_90a) vs torch.matmul — oracle-first correctness (GPU, H100/H200).

Mirrors ``test_gemm_mma_sync.py``: the oracle is ``torch.matmul`` in FP32, the
metric is the normalized relative error ‖Δ‖∞ / ‖ref‖∞ (scale-invariant — float
addition is non-associative so a per-element rtol rejects a *correct* kernel
whose reduction order differs from cuBLAS).

The kernel takes float16 operands, accumulates in FP32, returns FP32 — the
warpgroup-MMA path. M and N must be multiples of 64 (one warpgroup tile; no
edge handling in the structural skeleton — the rental measurement sizes are
4096+, so this is the fast path only).

ARCH GATE: skips on any non-Hopper device (the dev box is sm_120 — WGMMA does
not assemble there). Runs on the H100/H200 rental day.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("wgmma_gemm requires CUDA", allow_module_level=True)

# Skip cleanly on non-Hopper: the WGMMA ISA does not exist on sm_120 (the dev box)
# or on Ampere/Ada. require_cc(9, 0) is the load-bearing gate.
from scratch_llm.kernels.common.arch import compute_capability  # noqa: E402

if compute_capability() != (9, 0):
    pytest.skip(
        f"wgmma_gemm requires Hopper (sm_90a); current device is {compute_capability()}",
        allow_module_level=True,
    )

from scratch_llm.kernels.gemm.cuda.wgmma import wgmma_gemm  # noqa: E402

_RTOL = 2e-2  # looser than mma.sync — the WGMMA descriptor path is the structural-skeleton rung


def _ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
    torch.manual_seed(0)
    m = k = n = 256
    a = torch.randn(m, k, device="cuda", dtype=torch.float16) * scale
    b = torch.randn(k, n, device="cuda", dtype=torch.float16) * scale
    _assert_matches(wgmma_gemm(a, b), _ref(a, b))
    del a, b
    torch.cuda.empty_cache()


def test_4096_cubed() -> None:
    """The shipping size: 4096^3 — the rental-day headliner measurement vs cuBLAS."""
    torch.manual_seed(1)
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    _assert_matches(wgmma_gemm(a, b), _ref(a, b))
    del a, b
    torch.cuda.empty_cache()
