"""Hopper stream-K GEMM (sm_90a) — oracle-first test scaffold for the STUB.

The kernel is unimplemented (``csrc/gemm/stream_k_sm90.cu`` is the scaffold).
xfail until you implement the scheduler + mainloop.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stream_k_gemm requires CUDA", allow_module_level=True)

from scratch_llm.kernels.common.arch import compute_capability  # noqa: E402

if compute_capability() != (9, 0):
    pytest.skip(
        f"stream_k_gemm requires Hopper (sm_90a); current device is {compute_capability()}",
        allow_module_level=True,
    )

from scratch_llm.kernels.gemm.cuda.stream_k import stream_k_gemm  # noqa: E402


@pytest.mark.xfail(reason="stream_k_gemm is a STUB — implement csrc/gemm/stream_k_sm90.cu")
def test_stream_k_matches_oracle() -> None:
    torch.manual_seed(0)
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    out = stream_k_gemm(a, b)
    ref = torch.matmul(a.float(), b.float())
    rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    assert rel <= 2e-2, f"stream_k vs oracle rel err {rel:.3e}"
