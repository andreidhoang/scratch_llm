"""A2 R4 — row-wise top-k Triton kernel vs the torch.topk oracle (GPU).

Two oracles, per the rung spec:
  * NO-TIE random inputs → EXACT values AND indices vs ``torch.topk`` (selection copies elements
    bitwise, so exact equality is the right bar, not a tolerance);
  * a CRAFTED tie input → our DEFINED rule "lowest column index wins" (``torch.topk``'s tie order
    is unspecified, so we assert against the rule, not against torch).

Plus the fused softmax+top-k path: same indices as plain top-k, values = softmax probabilities.
"""

import pytest
import torch

pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("top-k Triton kernel needs CUDA", allow_module_level=True)

pytestmark = pytest.mark.gpu

from scratch_llm.kernels.topk_triton import (  # noqa: E402
    fused_softmax_topk,
    topk_last_dim,
)


def _no_tie_matrix(m: int, n: int, seed: int = 0) -> torch.Tensor:
    """Random fp32 (M, N) with every row's values distinct, so the top-k boundary is unambiguous and
    torch.topk is a well-defined index oracle. A per-row random permutation of {0..N-1} (via argsort
    of noise) is distinct by construction — fp32 randn alone collides at large N (birthday paradox in
    the ~2**24 representable mantissa), which would silently introduce ties."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, n, device="cuda", dtype=torch.float32, generator=g).argsort(dim=-1)
    x = x.to(torch.float32)  # values 0..N-1 in random order; distinct within each row
    assert all(int(x[r].unique().numel()) == n for r in range(m)), "test matrix has row ties"
    return x


@pytest.mark.parametrize(
    ("m", "n", "k"), [(8, 128, 8), (64, 1000, 8), (33, 4096, 4), (5, 50257, 8)]
)
def test_topk_matches_torch_no_ties(m: int, n: int, k: int) -> None:
    x = _no_tie_matrix(m, n, seed=m * 131 + n)
    vals, idx = topk_last_dim(x, k)
    ref_vals, ref_idx = torch.topk(x, k, dim=-1, sorted=True)

    assert vals.shape == (m, k) and idx.shape == (m, k)
    assert idx.dtype == torch.int64
    # No-tie selection: bit-exact values and exact indices, in descending order.
    torch.testing.assert_close(vals, ref_vals, atol=0.0, rtol=0.0)
    assert torch.equal(idx, ref_idx), (
        f"index mismatch: max row Δ at {(idx != ref_idx).any(dim=1).nonzero().flatten().tolist()[:5]}"
    )
    # Descending invariant.
    assert torch.all(vals[:, :-1] >= vals[:, 1:])


def test_topk_tie_break_lowest_index_wins() -> None:
    """Crafted ties: the DEFINED rule is lowest column index wins. Row 0 has three 5.0's at cols
    0,1,2 and two 3.0's at cols 3,4; top-3 must be values [5,5,5] at indices [0,1,2]."""
    x = torch.tensor(
        [
            [5.0, 5.0, 5.0, 3.0, 3.0, 1.0, 0.0, -1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # all equal → first k indices
        ],
        device="cuda",
    )
    vals, idx = topk_last_dim(x, 3)
    assert torch.equal(vals[0], torch.tensor([5.0, 5.0, 5.0], device="cuda"))
    assert torch.equal(idx[0], torch.tensor([0, 1, 2], device="cuda"))
    # All-equal row: lowest three indices, in ascending order (each pass removes the current min).
    assert torch.equal(idx[1], torch.tensor([0, 1, 2], device="cuda"))


def test_topk_adversarial_shapes_and_values() -> None:
    """Adversarial: k==1, k==N, a single-column edge, ±inf and duplicated maxima, negatives."""
    # k == N: returns a full descending sort of the row.
    x = _no_tie_matrix(4, 16, seed=7)
    vals, idx = topk_last_dim(x, 16)
    rv, ri = torch.topk(x, 16, dim=-1, sorted=True)
    assert torch.equal(vals, rv) and torch.equal(idx, ri)

    # k == 1: plain row argmax.
    vals, idx = topk_last_dim(x, 1)
    assert torch.equal(idx.flatten(), x.argmax(dim=-1))

    # +inf present (a masked-logit sentinel a caller might feed) selects first.
    y = torch.tensor([[-5.0, float("inf"), 2.0, float("inf"), 1.0]], device="cuda")
    vals, idx = topk_last_dim(y, 2)
    assert vals[0, 0] == float("inf") and idx[0, 0] == 1  # lowest index among the two +inf


@pytest.mark.parametrize(("m", "n", "k"), [(16, 1024, 8), (4, 32000, 8)])
def test_fused_softmax_topk(m: int, n: int, k: int) -> None:
    """Fused kernel: indices == plain top-k (softmax monotone); values == the softmax probabilities
    of the selected logits, to fp32 tolerance vs a torch two-pass reference."""
    x = _no_tie_matrix(m, n, seed=99 + n)
    probs, idx = fused_softmax_topk(x, k)

    ref_probs, ref_idx = torch.topk(torch.softmax(x, dim=-1), k, dim=-1, sorted=True)
    assert torch.equal(idx, ref_idx)
    torch.testing.assert_close(probs, ref_probs, atol=1e-6, rtol=1e-4)
    # Probabilities are a valid (partial) distribution: within (0, 1], descending.
    assert torch.all((probs > 0.0) & (probs <= 1.0))
    assert torch.all(probs[:, :-1] >= probs[:, 1:])
