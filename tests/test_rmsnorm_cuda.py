"""RMSNorm CUDA kernel — oracle-parity test (the RED test for the reconstruct loop).

GPU-only: gated on the `gpu` marker + importorskip(torch+cuda). Stays RED until the kernel body
in cuda/rmsnorm.cu is filled, then turns green — the test-first invariant for the inference track.
Invariant: fused CUDA RMSNorm == pure-torch oracle within atol, across shapes and fp16/bf16/fp32.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA required for the RMSNorm kernel", allow_module_level=True)

from scratch_llm.kernels.inference import rmsnorm, rmsnorm_ref  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1024), (8, 4096), (128, 2048)])
def test_rmsnorm_matches_oracle(shape, dtype):
    torch.manual_seed(0)
    n, d = shape
    x = torch.randn(n, d, device="cuda", dtype=dtype)
    w = torch.randn(d, device="cuda", dtype=dtype)

    got = rmsnorm(x, w)
    want = rmsnorm_ref(x, w)

    atol = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(got, want, atol=atol, rtol=atol)


def test_rmsnorm_preserves_leading_dims():
    x = torch.randn(2, 3, 512, device="cuda")
    w = torch.randn(512, device="cuda")
    assert rmsnorm(x, w).shape == x.shape
