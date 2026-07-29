"""Hopper FP8 GEMM (sm_90a) — oracle-first test scaffold for the STUB.

The kernel is unimplemented (``csrc/gemm/fp8_gemm_sm90.cu`` is the scaffold).
This test is **xfail** until you implement the mainloop; it exists so the test
infrastructure is in place the moment the kernel lands, and so the ladder is
visible in the test suite (one more frontier rung to climb).

ARCH GATE: skips on non-Hopper (WGMMA + FP8 MMA are sm_90a).
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("fp8_gemm requires CUDA", allow_module_level=True)

from scratch_llm.kernels.common.arch import compute_capability  # noqa: E402

if compute_capability() != (9, 0):
    pytest.skip(
        f"fp8_gemm requires Hopper (sm_90a); current device is {compute_capability()}",
        allow_module_level=True,
    )

from scratch_llm.kernels.gemm.cuda.fp8 import fp8_gemm  # noqa: E402


@pytest.mark.xfail(reason="fp8_gemm is a STUB — implement csrc/gemm/fp8_gemm_sm90.cu")
def test_fp8_gemm_matches_oracle() -> None:
    torch.manual_seed(0)
    n = 4096
    a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
    out = fp8_gemm(a, b)
    ref = torch.matmul(a.float(), b.float())
    rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    assert rel <= 5e-2, f"fp8 vs oracle rel err {rel:.3e}"
