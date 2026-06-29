"""GEMM CUDA kernel — oracle-parity test (the RED test for the reconstruct loop).

GPU-only: gated on the `gpu` marker + importorskip(torch+cuda). Stays RED until the kernel body in
csrc/gemm/gemm.cu is filled, then turns green — the test-first invariant for the R4 compute-bound rung.
Invariant: CUDA GEMM (C = A@B) == pure-torch oracle within atol, across shapes and fp16/bf16/fp32.

Bench the climb (on the GPU box):
    python -c "from scratch_llm.kernels.inference import gemm_roofline; gemm_roofline()"
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA required for the GEMM kernel", allow_module_level=True)

from scratch_llm.kernels.inference import gemm, gemm_ref  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(256, 256, 256), (384, 512, 256), (200, 320, 192)])
def test_gemm_matches_oracle(shape, dtype) -> None:  # noqa: ANN001 (torch is a runtime importorskip var)
    """C = A@B must equal the fp32 oracle. Shapes include non-multiples of TILE (edge-guard test)."""
    torch.manual_seed(0)
    m, n, k = shape
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    got = gemm(a, b)
    want = gemm_ref(a, b)

    atol = {torch.float32: 1e-3, torch.float16: 5e-2, torch.bfloat16: 2e-1}[dtype]
    torch.testing.assert_close(got.float(), want.float(), atol=atol, rtol=atol)


def test_gemm_output_shape() -> None:
    a = torch.randn(512, 256, device="cuda")
    b = torch.randn(256, 128, device="cuda")
    assert gemm(a, b).shape == (512, 128)
