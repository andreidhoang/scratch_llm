"""Triton FA2 forward must equal the pure-PyTorch oracle (and SDPA) on CUDA — gpu-only.

Skips cleanly on a CPU box: importorskip drops it when Triton is absent, and the gpu marker keeps
it out of the `-m "not gpu"` smoke gate. On the rented 4090 it runs against the oracle in fp32
(bit-faithful, allow_tf32=False) and bf16 (realistic precision)."""

import pytest
import torch
import torch.nn.functional as F

from reasoning_llm.kernels import flash_attention_forward  # CPU oracle, no Triton

pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover - environment guard
    pytest.skip("CUDA required for the Triton kernel", allow_module_level=True)

# Imported only after the Triton/CUDA guard (this module pulls `import triton`).
from reasoning_llm.kernels.flash_attention_triton import flash_attention_triton_forward  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("n", [128, 200, 512])  # 200 → ragged final tile
def test_triton_fp32_matches_oracle(n: int, is_causal: bool) -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, n, 64, device="cuda", dtype=torch.float32) for _ in range(3))
    o_tri, l_tri = flash_attention_triton_forward(q, k, v, is_causal=is_causal, allow_tf32=False)
    o_ref, l_ref = flash_attention_forward(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(o_tri, o_ref, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(l_tri, l_ref, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("is_causal", [False, True])
def test_triton_fp32_matches_sdpa(is_causal: bool) -> None:
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 4, 256, 64, device="cuda", dtype=torch.float32) for _ in range(3))
    o_tri, _ = flash_attention_triton_forward(q, k, v, is_causal=is_causal, allow_tf32=False)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(o_tri, ref, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("is_causal", [False, True])
def test_triton_bf16_matches_sdpa(is_causal: bool) -> None:
    # Realistic precision: bf16 in, fp32 accumulate; loose tol (bf16 has ~3 decimal digits).
    torch.manual_seed(2)
    q, k, v = (torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o_tri, _ = flash_attention_triton_forward(q, k, v, is_causal=is_causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(o_tri.float(), ref.float(), atol=2e-2, rtol=2e-2)
