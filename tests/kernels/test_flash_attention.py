"""The pure-PyTorch FA2 forward must equal F.scaled_dot_product_attention — the oracle the
Triton kernel is later checked against. Causal + non-causal, ragged tiles, and L=logsumexp."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from scratch_llm.kernels import FlashAttentionPyTorch, flash_attention_forward


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "shape", [(2, 4, 32, 16), (1, 2, 20, 64), (2, 3, 100, 32)]
)  # last two: seq not a multiple of the tile → exercise ragged tiles
def test_matches_sdpa(shape: tuple[int, int, int, int], is_causal: bool) -> None:
    b, h, n, d = shape
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, n, d, dtype=torch.float64) for _ in range(3))

    o, _ = flash_attention_forward(q, k, v, is_causal=is_causal, q_tile=8, k_tile=8)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    torch.testing.assert_close(o, ref, atol=1e-10, rtol=1e-8)


def test_lse_matches_logsumexp() -> None:
    # L (the FA2 log-normalizer) must equal logsumexp of the (masked) scaled scores.
    torch.manual_seed(1)
    n, d = 24, 16
    q, k, v = (torch.randn(1, 1, n, d, dtype=torch.float64) for _ in range(3))
    _, lse = flash_attention_forward(q, k, v, is_causal=True, q_tile=8, k_tile=8)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # (1,1,n,n)
    causal = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    expected = torch.logsumexp(scores, dim=-1)  # (1,1,n)
    torch.testing.assert_close(lse, expected, atol=1e-10, rtol=1e-8)


def test_tiling_is_invariant_to_tile_size() -> None:
    # The online-softmax recurrence must give the same answer for any tiling.
    torch.manual_seed(2)
    q, k, v = (torch.randn(2, 2, 48, 32, dtype=torch.float64) for _ in range(3))
    o_small, _ = flash_attention_forward(q, k, v, is_causal=True, q_tile=8, k_tile=8)
    o_big, _ = flash_attention_forward(q, k, v, is_causal=True, q_tile=64, k_tile=64)
    torch.testing.assert_close(o_small, o_big, atol=1e-12, rtol=1e-10)


def _ref_attention(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("shape", [(2, 4, 32, 16), (2, 3, 100, 32)])  # 100 → ragged tiles
def test_backward_matches_autograd(shape: tuple[int, int, int, int], is_causal: bool) -> None:
    # The recomputation backward must reproduce plain attention's autograd grads (the oracle).
    b, h, n, d = shape
    torch.manual_seed(0)
    base = [torch.randn(b, h, n, d, dtype=torch.float64) for _ in range(3)]
    do = torch.randn(b, h, n, d, dtype=torch.float64)

    q1, k1, v1 = (t.clone().requires_grad_(True) for t in base)
    _ref_attention(q1, k1, v1, is_causal).backward(do)

    q2, k2, v2 = (t.clone().requires_grad_(True) for t in base)
    FlashAttentionPyTorch.apply(q2, k2, v2, is_causal).backward(do)

    for got, want in ((q2.grad, q1.grad), (k2.grad, k1.grad), (v2.grad, v1.grad)):
        torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-8)


def test_backward_returns_none_for_is_causal_flag() -> None:
    # is_causal is a non-differentiable bool → its grad slot must be None (else autograd errors).
    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 8, 16, dtype=torch.float64, requires_grad=True) for _ in range(3))
    grads = FlashAttentionPyTorch.backward(
        _Ctx(q.detach(), k.detach(), v.detach(), is_causal=False),
        torch.randn(1, 8, 16, dtype=torch.float64),
    )
    assert grads[3] is None and len(grads) == 4


class _Ctx:
    """Minimal stand-in for the autograd ctx, to unit-test backward's return arity directly."""

    def __init__(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> None:
        o, lse = flash_attention_forward(q, k, v, is_causal=is_causal)
        self.saved_tensors = (q, k, v, o, lse)
        self.is_causal = is_causal
