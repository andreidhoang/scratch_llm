"""vector-add (SAXPY) Triton kernel must equal torch (z = x + y) on CUDA — gpu-only.

The R2 launch-model rung (GPU_FROM_ZERO §Rung 2): the "hello world" kernel that teaches the launch
grid, program_id indexing, and bounds masking. `z = x + y` is **memory-bound** — 3 arrays × itemsize
bytes moved per 1 add → AI ≈ 1/12 (fp32), deep under the ridge. The DoD is a *profile*, not this
test: target ~80–90% of HBM bandwidth (predict the number first, then `vector_add_roofline`).

Skips on CPU/CI (gpu marker + Triton/CUDA guards) and skips until you build the kernel (Mode-3:
reconstruct `kernels/vector_add_triton.py` from blank in your own editor).

Bench the climb (on the GPU box):
    python -c "from scratch_llm.kernels.bench import vector_add_roofline; \
               from scratch_llm.kernels.vector_add_triton import vector_add; \
               vector_add_roofline(vector_add)"
"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover - environment guard
    pytest.skip("CUDA required for the vector-add kernel", allow_module_level=True)

pytest.importorskip(
    "scratch_llm.kernels.vector_add_triton",
    reason="vector-add kernel not built (Mode-3: reconstruct from blank)",
)

from scratch_llm.kernels.vector_add_triton import vector_add  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("n", [1, 128, 1024, 4096, 100_000, 1_048_576])
def test_vector_add_matches_torch(n: int) -> None:
    """Correctness floor across sizes spanning many blocks (and n=1, the degenerate grid)."""
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(vector_add(x, y), x + y)


def test_vector_add_size_not_a_block_multiple() -> None:
    """The bounds mask must handle a size that is not a multiple of BLOCK — the off-by-one rep.

    If the mask is wrong, the last (partial) block reads/writes out of bounds → wrong tail or a fault.
    """
    torch.manual_seed(0)
    n = 1003  # deliberately not a power of two / block multiple
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(vector_add(x, y), x + y)
