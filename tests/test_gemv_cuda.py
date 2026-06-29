"""GEMV CUDA kernel — oracle-parity test (the RED test for the reconstruct loop).

GPU-only: gated on the `gpu` marker + importorskip(torch+cuda). Stays RED until the kernel body
in csrc/gemv.cu is filled, then turns green — the test-first invariant for the inference track.
Invariant: CUDA GEMV (y = A@x) == pure-torch oracle within atol, across shapes and fp16/bf16/fp32.

Bench the climb (on the GPU box):
    python -c "from scratch_llm.kernels.inference import gemv_roofline; gemv_roofline()"
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA required for the GEMV kernel", allow_module_level=True)

from scratch_llm.kernels.inference import gemv, gemv_ref  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1024), (4096, 4096), (320, 2048), (200, 192)])
def test_gemv_matches_oracle(shape, dtype):
    torch.manual_seed(0)
    m, k = shape
    A = torch.randn(m, k, device="cuda", dtype=dtype)
    x = torch.randn(k, device="cuda", dtype=dtype)

    got = gemv(A, x)
    want = gemv_ref(A, x)

    atol = {torch.float32: 1e-4, torch.float16: 5e-2, torch.bfloat16: 2e-1}[dtype]
    torch.testing.assert_close(got.float(), want.float(), atol=atol, rtol=atol)


def test_gemv_output_shape():
    A = torch.randn(512, 256, device="cuda")
    x = torch.randn(256, device="cuda")
    assert gemv(A, x).shape == (512,)
