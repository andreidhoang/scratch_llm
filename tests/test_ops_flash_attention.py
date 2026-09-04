"""``scratch_llm::flash_attention`` — the custom-op contract, checked by ``opcheck``.

Why this file exists: ``ops/attention.py`` is the documented model-facing seam (a
``torch.library.custom_op`` with a registered fake), and the kernel spec's ship-into-torch
checklist ends in ``opcheck``. Without it the fake's declared metadata is never compared
against the real implementation, and a fake that lies about dtype or shape is invisible until
someone traces the op under ``torch.compile``/``export`` — where the failure surfaces far from
its cause.

The invariant: for every dtype the CPU path accepts, the fake's (shape, dtype, device) must
match what the composite implementation actually returns. fp64 is the discriminating case —
the reference accumulates the logsumexp in fp64 for fp64 inputs, so a fake hardcoding fp32
disagrees exactly there.
"""

from __future__ import annotations

import pytest
import torch

import scratch_llm.ops.attention  # noqa: F401  (registers scratch_llm::flash_attention)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_opcheck_flash_attention(dtype: torch.dtype, is_causal: bool) -> None:
    torch.manual_seed(0)
    b, n, d = 2, 8, 4
    q, k, v = (torch.randn(b, n, d, dtype=dtype) for _ in range(3))
    torch.library.opcheck(torch.ops.scratch_llm.flash_attention, (q, k, v, is_causal, True))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_fake_matches_real_lse_dtype(dtype: torch.dtype) -> None:
    """The regression this file was written for: the fake declared lse fp32 unconditionally."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 8, 4, dtype=dtype) for _ in range(3))
    _, lse_real = torch.ops.scratch_llm.flash_attention(q, k, v, False, True)
    with torch._subclasses.FakeTensorMode() as mode:
        fq, fk, fv = (mode.from_tensor(x) for x in (q, k, v))
        _, lse_fake = torch.ops.scratch_llm.flash_attention(fq, fk, fv, False, True)
    assert lse_fake.dtype == lse_real.dtype, f"fake {lse_fake.dtype} vs real {lse_real.dtype}"
    assert tuple(lse_fake.shape) == tuple(lse_real.shape)
