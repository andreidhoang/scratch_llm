"""A2 R1 — GEMV ladder vs the ``torch.mv`` oracle (GPU).

Every stage of the ladder must equal ``A @ x`` computed by torch, at the dtype-appropriate
tolerance: **rtol 1e-2 for bf16**, **rtol 1e-5 for fp32** (the kernel accumulates in fp32 for both).
Random inputs plus four adversarial shapes that historically break a GEMV: N not a multiple of the
block (masking), M=1 (degenerate grid), a single large-outlier row (fp32-accumulation stress), and a
zero row (exact-zero output, no NaN).
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("GEMV kernel tests need CUDA", allow_module_level=True)
pytest.importorskip("triton")

from scratch_llm.kernels.gemm.triton.gemv import (  # noqa: E402  (after the gpu/triton guards)
    gemv_blockrow,
    gemv_naive,
    gemv_split,
)

_STAGES = [
    ("naive", lambda a, x: gemv_naive(a, x)),
    ("blockrow", lambda a, x: gemv_blockrow(a, x)),
    ("split", lambda a, x: gemv_split(a, x, n_splits=8, block_n=256)),
]

# bf16: rtol 1e-2 (the bf16 rounding of the final cast dominates). fp32: rtol 1e-5 (fp32 accumulation
# vs torch's reduction order). atol covers exact/near-zero entries where rtol*|ref| underflows.
_TOL = {torch.bfloat16: (1e-2, 1e-2), torch.float32: (1e-5, 1e-4)}


def _check(a: torch.Tensor, x: torch.Tensor, dtype: torch.dtype) -> None:
    ref = torch.mv(a, x)
    rtol, atol = _TOL[dtype]
    for name, fn in _STAGES:
        out = fn(a, x)
        assert out.shape == ref.shape, f"{name}: shape {out.shape} != {ref.shape}"
        assert torch.isfinite(out.float()).all(), f"{name}: non-finite output"
        maxrel = ((out.float() - ref.float()).abs() / (ref.float().abs() + atol)).max().item()
        assert torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol), (
            f"{name} [{dtype}]: max|Δ|={(out.float() - ref.float()).abs().max().item():.3e} "
            f"maxrel={maxrel:.3e} (rtol={rtol})"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    ("m", "n"), [(512, 1024), (1, 2048), (2048, 1)], ids=["square", "m1", "n1"]
)
def test_gemv_random(dtype: torch.dtype, m: int, n: int) -> None:
    torch.manual_seed(0)
    a = torch.randn(m, n, device="cuda", dtype=dtype)
    x = torch.randn(n, device="cuda", dtype=dtype)
    _check(a, x, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_gemv_n_not_multiple_of_block(dtype: torch.dtype) -> None:
    """N=1000 is a multiple of no block size in the ladder ⇒ exercises the tail mask on every stage."""
    torch.manual_seed(1)
    a = torch.randn(300, 1000, device="cuda", dtype=dtype)
    x = torch.randn(1000, device="cuda", dtype=dtype)
    _check(a, x, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_gemv_outlier_and_zero_rows(dtype: torch.dtype) -> None:
    """A single large-outlier row (stresses fp32 accumulation) and a zero row (must yield exact 0)."""
    torch.manual_seed(2)
    m, n = 256, 1024
    a = torch.randn(m, n, device="cuda", dtype=dtype)
    a[7] = 1e3  # outlier row: |y[7]| ~ 1e3 * sqrt(N) — far from the rest
    a[42] = 0.0  # zero row: y[42] must be exactly 0
    x = torch.randn(n, device="cuda", dtype=dtype)
    _check(a, x, dtype)
    # the zero row is a hard invariant, not just within-tolerance
    for _, fn in _STAGES:
        assert fn(a, x)[42].item() == 0.0
