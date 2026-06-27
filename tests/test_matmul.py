"""matmul kernels must equal torch.matmul (cuBLAS) on CUDA — gpu-only.

Skips on CPU/CI (gpu marker + Triton/CUDA guards, same pattern as test_flash_attention_triton.py).
`matmul_naive` must pass; `matmul_tiled` is the reconstruct-from-blank rung (xfail until you build it).

Bench the climb (on the GPU box):
    python -c "from scratch_llm.kernels.bench import matmul_roofline; \
               from scratch_llm.kernels.matmul import matmul_naive; \
               matmul_roofline(matmul_naive, 4096, 4096, 4096)"
"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover - environment guard
    pytest.skip("CUDA required for the matmul kernels", allow_module_level=True)

from scratch_llm.kernels.matmul import matmul_naive, matmul_tiled, reference  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("shape", [(256, 256, 256), (384, 512, 256), (200, 320, 192)])
def test_naive_matches_cublas(shape: tuple[int, int, int]) -> None:
    m, n, k = shape
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        matmul_naive(a, b).float(), reference(a, b).float(), atol=2e-2, rtol=2e-2
    )


@pytest.mark.xfail(reason="reconstruct-from-blank: implement the matmul_tiled rung", strict=False)
def test_tiled_matches_cublas() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        matmul_tiled(a, b).float(), reference(a, b).float(), atol=2e-2, rtol=2e-2
    )
