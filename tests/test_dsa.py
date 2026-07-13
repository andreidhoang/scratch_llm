"""F8.1 — DeepSeek Sparse Attention (DSA) core-mechanism oracles.

The correctness anchor: top-k sparse attention with k >= L must equal dense causal
attention EXACTLY (same precision path) — the O(L^2) -> O(L*k) rewrite is lossless at the
limit. The training oracle: a lightning indexer KL-trained against the dense attention
distribution recovers >= 0.95 of held-out attention mass at k = L/3, while a random-init
indexer sits near the uniform floor (~1/3 at this k/L) — proving the test has teeth.

No GPU marker: everything here is a tiny CPU problem (the training test is ~2 s).
"""

import math
import time

import torch

from scratch_llm.dsa import (
    DSAConfig,
    LightningIndexer,
    attention_mass_recall,
    dense_attention_target,
    dsa_attention_flops,
    indexer_kl_loss,
    select_topk_mask,
    topk_sparse_attention,
)
from scratch_llm.model import scaled_dot_product_attention


def _causal(L: int) -> torch.Tensor:
    return torch.tril(torch.ones(L, L, dtype=torch.bool))


# ---------------------------------------------------------------------------
# DoD (a): top-k == dense at k >= L — the exact O(L^2) -> O(L*k) identity, causal.
# ---------------------------------------------------------------------------


def test_topk_equals_dense_at_k_geq_L() -> None:
    torch.manual_seed(0)
    L, H, d = 32, 2, 16
    q = torch.randn(H, L, d, dtype=torch.float64)
    k = torch.randn(H, L, d, dtype=torch.float64)
    v = torch.randn(H, L, d, dtype=torch.float64)
    scores = torch.randn(L, L, dtype=torch.float64)  # any scores: at k>=L all survive

    for kk in (L, L + 7):
        mask = select_topk_mask(scores, kk)
        assert torch.equal(mask, _causal(L)), "at k>=L the top-k mask must be full causal"
        sparse = topk_sparse_attention(q, k, v, mask)
        dense = scaled_dot_product_attention(q, k, v, _causal(L))
        # Same function, same mask, same dtype => bitwise identical.
        assert torch.equal(sparse, dense)


def test_topk_mask_causal_self_and_budget() -> None:
    torch.manual_seed(1)
    L, kk = 17, 5
    mask = select_topk_mask(torch.randn(L, L), kk)
    assert not mask[~_causal(L)].any(), "no key position s > t may be selected"
    assert bool(mask.diagonal().all()), "every query must keep its own position"
    counts = mask.sum(dim=-1)
    expected = torch.tensor([min(kk, t + 1) for t in range(L)])
    assert torch.equal(counts, expected)


def test_topk_mask_deterministic_tie_break() -> None:
    L, kk = 12, 4
    scores = torch.zeros(L, L)  # all ties: stable sort must break toward the lowest index
    m1 = select_topk_mask(scores, kk)
    m2 = select_topk_mask(scores, kk)
    assert torch.equal(m1, m2)
    expected = torch.zeros(L, L, dtype=torch.bool)
    for t in range(L):
        expected[t, t] = True  # forced self
        expected[t, : min(kk - 1, t)] = True  # then lowest-index keys fill the budget
    assert torch.equal(m1, expected)


# ---------------------------------------------------------------------------
# DoD (c): KL is zero iff the indexer distribution matches dense attention exactly.
# ---------------------------------------------------------------------------


def test_kl_zero_at_exact_match_and_positive_otherwise() -> None:
    torch.manual_seed(2)
    L, d = 24, 8
    q = torch.randn(L, d, dtype=torch.float64)
    k = torch.randn(L, d, dtype=torch.float64)
    v = torch.randn(L, d, dtype=torch.float64)
    probs = dense_attention_target(q, k, v)
    assert probs.dtype == torch.float64

    neg_inf = torch.tensor(float("-inf"), dtype=torch.float64)
    exact_scores = torch.where(_causal(L), probs.log(), neg_inf)
    assert indexer_kl_loss(exact_scores, probs).item() < 1e-12

    torch.manual_seed(3)
    perturbed = exact_scores + torch.randn(L, L, dtype=torch.float64).masked_fill(~_causal(L), 0.0)
    assert indexer_kl_loss(perturbed, probs).item() > 1e-3


def test_dense_attention_target_is_a_causal_distribution() -> None:
    torch.manual_seed(4)
    L, H, d = 20, 3, 8
    q = torch.randn(H, L, d)
    k = torch.randn(H, L, d)
    v = torch.randn(H, L, d)
    probs = dense_attention_target(q, k, v)
    assert probs.shape == (L, L), "multi-head target aggregates heads into one [L, L] matrix"
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(L))
    assert not probs[~_causal(L)].any(), "no mass on future positions"


# ---------------------------------------------------------------------------
# DoD (d): recall is monotone in k, and exactly 1 at k = L.
# ---------------------------------------------------------------------------


def test_recall_monotone_in_k() -> None:
    torch.manual_seed(5)
    L, d = 48, 16
    q = torch.randn(L, d)
    k = torch.randn(L, d)
    v = torch.randn(L, d)
    probs = dense_attention_target(q, k, v)
    scores = torch.randn(L, L)

    r1 = attention_mass_recall(probs, select_topk_mask(scores, 1)).item()
    r12 = attention_mass_recall(probs, select_topk_mask(scores, L // 4)).item()
    rL = attention_mass_recall(probs, select_topk_mask(scores, L)).item()
    assert r1 < r12 < rL, f"recall must grow with k: {r1:.3f}, {r12:.3f}, {rL:.3f}"
    assert abs(rL - 1.0) < 1e-6, "at k=L the selected set covers all mass"


# ---------------------------------------------------------------------------
# Indexer module: shapes, causal masking, batch dim.
# ---------------------------------------------------------------------------


def test_indexer_scores_shape_and_causality() -> None:
    torch.manual_seed(6)
    cfg = DSAConfig(d_model=16, n_heads=2, d_index=8, n_index_heads=4, top_k=4)
    idx = LightningIndexer(cfg)
    L = 10

    scores = idx(torch.randn(L, cfg.d_model))
    assert scores.shape == (L, L)
    assert bool(torch.isinf(scores[~_causal(L)]).all()) and (scores[~_causal(L)] < 0).all()
    assert bool(torch.isfinite(scores[_causal(L)]).all())

    batched = idx(torch.randn(3, L, cfg.d_model))
    assert batched.shape == (3, L, L)


# ---------------------------------------------------------------------------
# DoD (b): a KL-trained indexer recovers >= 0.95 held-out attention mass at k = L/3;
# a random-init indexer sits near the uniform floor (~1/3 here). KILL criterion:
# held-out recall < 0.95 after ~300 KL steps.
# ---------------------------------------------------------------------------


def test_kl_trained_indexer_recovers_heldout_mass() -> None:
    torch.manual_seed(0)
    L, d_model = 96, 8
    x = torch.randn(L, d_model)
    # A peaked dense-attention problem (logit std ~4/sqrt(2)*...) — at k = L/3 the oracle
    # top-k covers ~0.996 of the mass, so 0.95 is reachable but not trivial.
    wq = torch.randn(d_model, d_model) / math.sqrt(d_model)
    wk = torch.randn(d_model, d_model) / math.sqrt(d_model)
    q = (x @ wq) * 4.0
    k = x @ wk
    v = torch.randn(L, d_model)
    probs = dense_attention_target(q, k, v)

    kk = L // 3  # 32 of 96
    # Parity split: train on even queries, hold out late odd queries (long rows, so the
    # uniform floor is ~ kk / (t+1) ~ 1/3). Every key appears in SOME training pair, so
    # generalization is across held-out queries — the honest test for a pairwise scorer.
    train_rows = torch.arange(0, L, 2)
    held_rows = torch.arange(65, L, 2)

    cfg = DSAConfig(d_model=d_model, n_heads=1, d_index=8, n_index_heads=4, top_k=kk)
    torch.manual_seed(1)
    indexer = LightningIndexer(cfg)

    with torch.no_grad():
        random_recall = attention_mass_recall(
            probs[held_rows], select_topk_mask(indexer(x), kk)[held_rows]
        ).item()

    opt = torch.optim.Adam(indexer.parameters(), lr=1e-2)
    t0 = time.time()
    for _ in range(300):
        opt.zero_grad()
        loss = indexer_kl_loss(indexer(x)[train_rows], probs[train_rows])
        loss.backward()
        opt.step()
    wall = time.time() - t0

    with torch.no_grad():
        trained_recall = attention_mass_recall(
            probs[held_rows], select_topk_mask(indexer(x), kk)[held_rows]
        ).item()

    print(
        f"\n[dsa-train] trained={trained_recall:.4f} random={random_recall:.4f} "
        f"margin={trained_recall - random_recall:.3f} final_kl={loss.item():.4f} "
        f"wall={wall:.1f}s"
    )
    assert trained_recall >= 0.95, f"held-out recall {trained_recall:.4f} < 0.95 (KILL)"
    # Margin over the random arm (floor ~1/3 at this k/L) — proves the test has teeth
    # without a brittle absolute on the random side.
    assert trained_recall - random_recall >= 0.3, (
        f"margin {trained_recall - random_recall:.3f} < 0.3 "
        f"(random arm not near its floor, or training barely helped)"
    )
    assert wall < 60.0, "training oracle must stay a fast CPU test"


# ---------------------------------------------------------------------------
# FLOP model: sparse+indexer < dense at k << L; crossover exists (dense wins at k = L).
# ---------------------------------------------------------------------------


def test_flops_sparse_wins_at_small_k_and_crossover_exists() -> None:
    L, d, d_index, h = 4096, 128, 32, 4

    small = dsa_attention_flops(L=L, d=d, k=64, d_index=d_index, n_index_heads=h)
    assert small.sparse < small.dense
    assert small.sparse_total < small.dense, "at k << L sparse + indexer must beat dense"
    assert small.sparse_total == small.sparse + small.indexer

    full = dsa_attention_flops(L=L, d=d, k=L, d_index=d_index, n_index_heads=h)
    assert full.sparse == full.dense, "at k = L the attention FLOPs coincide"
    assert full.sparse_total > full.dense, "the indexer overhead makes dense win at k = L"

    # monotone in k => a crossover k* exists between 64 and L
    mid = dsa_attention_flops(L=L, d=d, k=1024, d_index=d_index, n_index_heads=h)
    assert small.sparse < mid.sparse < full.sparse


def test_dsa_config_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        DSAConfig(d_model=0, n_heads=1, d_index=8, n_index_heads=4, top_k=4)
    with pytest.raises(ValueError):
        DSAConfig(d_model=16, n_heads=1, d_index=8, n_index_heads=0, top_k=4)
    with pytest.raises(ValueError):
        DSAConfig(d_model=16, n_heads=1, d_index=8, n_index_heads=4, top_k=0)
