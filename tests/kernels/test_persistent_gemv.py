"""Hopper persistent GEMV (sm_90a) — oracle-first test scaffold for the STUB.

The kernel is unimplemented (``csrc/persistent/persistent_gemv_sm90.cu`` is the
scaffold). xfail until you implement the persistent mainloop + SM-count grid.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("persistent_gemv requires CUDA", allow_module_level=True)

from scratch_llm.kernels.common.arch import compute_capability  # noqa: E402

if compute_capability() != (9, 0):
    pytest.skip(
        f"persistent_gemv requires Hopper (sm_90a); current device is {compute_capability()}",
        allow_module_level=True,
    )

from scratch_llm.kernels.gemm.cuda.persistent import persistent_gemv  # noqa: E402


@pytest.mark.xfail(
    reason="persistent_gemv is a STUB — implement csrc/persistent/persistent_gemv_sm90.cu"
)
def test_persistent_gemv_matches_oracle() -> None:
    torch.manual_seed(0)
    k, n = 4096, 4096
    x = torch.randn(k, device="cuda", dtype=torch.float16)
    a = torch.randn(k, n, device="cuda", dtype=torch.float16)
    out = persistent_gemv(x, a)
    ref = torch.mv(a.float(), x.float())
    rel = (out.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    assert rel <= 2e-2, f"persistent_gemv vs oracle rel err {rel:.3e}"
