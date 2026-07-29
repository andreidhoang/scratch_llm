"""Blackwell tcgen05 GEMM (sm_100a) vs torch.matmul — oracle-first correctness (GPU, B200).

Same oracle / metric as ``test_gemm_mma_sync.py`` and ``test_gemm_wgmma.py``:
the oracle is FP32 ``torch.matmul``, the metric is the normalized relative error.

The kernel is the 1-SM tcgen05/UMMA rung: accumulator in Tensor Memory (TMEM),
one warpgroup, MMA issued by a single thread. M must be a multiple of 128, N of
256 (the 1-SM atom); no edge handling.

ARCH GATE: skips on any non-Blackwell-datacenter device. The dev box is sm_120
(Blackwell *client* — no tcgen05, no TMEM, no cta_group). Runs on the B200
rental day.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("tcgen05_gemm requires CUDA", allow_module_level=True)

from scratch_llm.kernels.common.arch import compute_capability  # noqa: E402

_cc = compute_capability()
if _cc is None or _cc[0] < 10 or _cc == (12, 0):
    pytest.skip(
        f"tcgen05_gemm requires Blackwell datacenter (sm_100a, B200); current device is {_cc} "
        "(sm_120 client Blackwell has no tcgen05/TMEM)",
        allow_module_level=True,
    )

from scratch_llm.kernels.gemm.cuda.tcgen05 import tcgen05_gemm  # noqa: E402

_RTOL = 3e-2  # the tcgen05/TMEM path is the structural-skeleton rung; looser tolerance


def _ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a.float(), b.float())


def _assert_matches(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.shape == ref.shape
    assert out.dtype == torch.float32
    d = (out.float() - ref.float()).abs()
    rel = d.max().item() / (ref.float().abs().max().item() + 1e-30)
    assert rel <= _RTOL, f"normalized rel err {rel:.3e} > rtol {_RTOL}"


def test_basic_tile() -> None:
    """One atom: M=128, N=256 — the smallest tile the 1-SM kernel accepts."""
    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    _assert_matches(tcgen05_gemm(a, b), _ref(a, b))


def test_large() -> None:
    """The rental measurement size — 4096^3 / atom-tile-aligned."""
    torch.manual_seed(1)
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    _assert_matches(tcgen05_gemm(a, b), _ref(a, b))
    del a, b
    torch.cuda.empty_cache()
