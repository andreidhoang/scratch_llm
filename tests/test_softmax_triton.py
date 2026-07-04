"""A2 Rung 2 — oracle-first correctness for the row-softmax Triton kernel (GPU).

Oracle = ``F.softmax(x, dim=-1)`` at rtol 1e-3 on random inputs, for all three ladder modes
(twopass / online / fused). The load-bearing part is the adversarial suite: a ``+1e4`` outlier row
(the running-max rescale must not overflow), an all-equal row (uniform output), and a wholly
``−inf`` masked row (must give a uniform ``1/N`` row and never NaN — where ``F.softmax`` itself emits
``0/0`` NaN, so the kernel is *more* robust than the reference on that row and is checked directly).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("softmax_triton kernel requires CUDA", allow_module_level=True)

pytest.importorskip("triton")

import torch.nn.functional as F  # noqa: E402

from scratch_llm.kernels.softmax_triton import softmax_triton  # noqa: E402

_MODES = ["twopass", "online", "fused"]


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    ("m_rows", "n_cols"),
    [(64, 128), (128, 500), (256, 1024), (32, 3000), (17, 4096)],  # incl. non-pow2 + multi-tile
)
def test_softmax_matches_torch(mode: str, m_rows: int, n_cols: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(m_rows, n_cols, device="cuda", dtype=torch.bfloat16)
    ref = F.softmax(x, dim=-1)
    out = softmax_triton(x, mode)
    assert out.shape == ref.shape
    assert out.dtype == x.dtype
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("mode", _MODES)
def test_softmax_rows_sum_to_one(mode: str) -> None:
    torch.manual_seed(1)
    x = torch.randn(48, 777, device="cuda", dtype=torch.bfloat16)
    out = softmax_triton(x, mode).float()
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0, atol=2e-2)
    assert (out >= 0).all()


@pytest.mark.parametrize("mode", _MODES)
def test_softmax_adversarial(mode: str) -> None:
    """Outlier (no overflow), all-equal (uniform), all-−inf (uniform, never NaN)."""
    torch.manual_seed(2)
    n = 512
    x = torch.randn(4, n, device="cuda", dtype=torch.bfloat16)
    x[0, 42] += 1e4  # row 0: a huge positive outlier — max-subtraction must keep exp ≤ 1
    x[1, :] = 3.14  # row 1: all-equal → uniform
    x[2, :] = float("-inf")  # row 2: wholly masked → uniform 1/N, NOT 0/0 NaN
    # row 3: ordinary random row (kept as-is)

    out = softmax_triton(x, mode)
    assert torch.isfinite(out.float()).all(), f"{mode}: NaN/Inf in output (masked-row guard failed)"

    of = out.float()
    # row 0 — outlier dominates: near-one at the spike, sums to one, no overflow
    assert of[0, 42].item() > 0.99
    torch.testing.assert_close(of[0].sum(), torch.ones((), device="cuda"), rtol=0, atol=2e-2)
    # row 1 — all-equal → uniform 1/N
    torch.testing.assert_close(
        of[1], torch.full((n,), 1.0 / n, device="cuda"), rtol=1e-3, atol=1e-3
    )
    # row 2 — wholly −inf → uniform 1/N (reference F.softmax would be NaN here, so checked directly)
    torch.testing.assert_close(
        of[2], torch.full((n,), 1.0 / n, device="cuda"), rtol=1e-3, atol=1e-3
    )

    # finite rows (0, 1, 3) must still match the torch oracle; row 2 excluded (oracle is NaN there)
    ref = F.softmax(x, dim=-1).float()
    for r in (0, 1, 3):
        torch.testing.assert_close(of[r], ref[r], rtol=1e-3, atol=1e-3)


def test_softmax_rejects_non_2d() -> None:
    x = torch.randn(2, 3, 4, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="2-D"):
        softmax_triton(x, "fused")
