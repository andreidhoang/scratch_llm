"""The pure-PyTorch FA2 forward must equal F.scaled_dot_product_attention — the oracle the
Triton kernel is later checked against. Causal + non-causal, ragged tiles, and L=logsumexp."""

import math

import pytest
import torch
import torch.nn.functional as F

from scratch_llm.kernels import flash_attention_forward


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
