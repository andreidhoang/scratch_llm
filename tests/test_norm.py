"""A2 Rung 3 — Triton RMSNorm / LayerNorm vs the torch oracle (GPU).

Oracle-first: the kernels must equal ``F.rms_norm`` / ``F.layer_norm`` at rtol 1e-3 in bf16, on
random inputs PLUS the three adversarial shapes the rung pre-registers — a fully-zero row (the
epsilon path: finite, no NaN), a single large outlier (fp32-accumulation faithfulness), and N not a
multiple of the block (masked tail). Tolerance is never weakened to pass; a miss means the kernel is
wrong.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("norm kernels require CUDA", allow_module_level=True)

pytest.importorskip("triton")

from scratch_llm.kernels.norm_triton import layernorm_triton, rmsnorm_triton  # noqa: E402

_SHAPES = [(8, 64), (32, 768), (4, 4096), (16, 8192), (7, 100), (3, 8193)]  # incl. non-pow2 N


@pytest.mark.parametrize(("m", "n"), _SHAPES)
def test_rmsnorm_matches_oracle(m: int, n: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6
    ref = F.rms_norm(x, (n,), w, eps=eps)
    out = rmsnorm_triton(x, w, eps=eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize(("m", "n"), _SHAPES)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
def test_layernorm_matches_oracle(m: int, n: int, with_bias: bool) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16) if with_bias else None
    eps = 1e-5
    ref = F.layer_norm(x, (n,), w, b, eps=eps)
    out = layernorm_triton(x, w, b, eps=eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-2)


def test_adversarial_zero_row_no_nan() -> None:
    """A fully-zero row must take the epsilon path (sqrt(eps) > 0), never produce NaN/Inf."""
    n = 512
    x = torch.randn(4, n, device="cuda", dtype=torch.bfloat16)
    x[1] = 0.0  # the epsilon row
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    r_out, r_ref = rmsnorm_triton(x, w, eps=1e-6), F.rms_norm(x, (n,), w, eps=1e-6)
    l_out, l_ref = layernorm_triton(x, w, b, eps=1e-5), F.layer_norm(x, (n,), w, b, eps=1e-5)
    for out, ref in ((r_out, r_ref), (l_out, l_ref)):
        assert torch.isfinite(out).all(), "zero row produced non-finite output"
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-2)


def test_adversarial_single_outlier() -> None:
    """One large element dominates the reduction — fp32 accumulation must track the oracle."""
    n = 1024
    x = torch.randn(4, n, device="cuda", dtype=torch.bfloat16)
    x[2, 500] = 300.0  # single outlier (within bf16 range, large vs the ~N(0,1) rest)
    w = torch.ones(n, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(n, device="cuda", dtype=torch.bfloat16)

    r_out, r_ref = rmsnorm_triton(x, w, eps=1e-6), F.rms_norm(x, (n,), w, eps=1e-6)
    l_out, l_ref = layernorm_triton(x, w, b, eps=1e-5), F.layer_norm(x, (n,), w, b, eps=1e-5)
    assert torch.isfinite(r_out).all() and torch.isfinite(l_out).all()
    torch.testing.assert_close(r_out.float(), r_ref.float(), rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(l_out.float(), l_ref.float(), rtol=1e-3, atol=1e-2)
