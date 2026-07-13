"""DeepSeek Sparse Attention (DSA) — the core mechanism, F8.1 (no model wiring).

Intent
------
DSA (DeepSeek-V3.2) replaces O(L^2) dense attention with O(L*k) sparse attention over the
top-k key positions per query, chosen by a *lightning indexer*: a tiny scorer

    I[t, s] = sum_j w[t, j] * relu(qI[t, j] . kI[s])        (j over a few low-dim heads)

that is still O(L^2) in *pairs* but with a constant (n_index_heads * d_index) tens of times
smaller than the main attention's (n_heads * head_dim). The indexer is trained by
distillation — KL(dense attention distribution || softmax(indexer scores)) — so its top-k
selection covers almost all of the dense attention mass, and the expensive O(L^2 * d)
attention is spent only where the mass is.

Invariant (the correctness anchor)
----------------------------------
`topk_sparse_attention` at k >= L equals dense causal attention EXACTLY — same mask, same
precision path (`model.scaled_dot_product_attention`). The sparsification is a pure
restriction of the softmax support: at the limit it is the identity, so any quality gap at
k < L is attributable to the indexer's selection, never to the attention rewrite.

Interview question this module answers
--------------------------------------
"How does DSA turn O(L^2) into O(L*k) losslessly-at-the-limit, and why train the indexer
with KL against dense attention?" — Because top-k masking only shrinks the softmax support
(k >= L recovers dense bit-for-bit), and because the selection target *is* the dense
distribution: KL distillation makes softmax(indexer scores) match p_dense row-wise, so the
indexer's top-k is (approximately) the dense top-k mass — the metric that decides quality
(`attention_mass_recall`), not logit MSE.

Scope: mechanism only. No model.py wiring (F8.2), no Triton kernel. Everything here is
shape-simple: a single optional leading batch dim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from scratch_llm.model import Linear, scaled_dot_product_attention, softmax


@dataclass(frozen=True)
class DSAConfig:
    """Dimensions of the DSA pair: the dense attention it serves and the indexer itself.

    d_model / n_heads describe the host attention; d_index / n_index_heads the (much
    cheaper) indexer; top_k the per-query key budget.
    """

    d_model: int
    n_heads: int
    d_index: int
    n_index_heads: int
    top_k: int

    def __post_init__(self) -> None:
        for name in ("d_model", "n_heads", "d_index", "n_index_heads", "top_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"DSAConfig.{name} must be positive, got {getattr(self, name)}")


def _causal_mask(L: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))


class LightningIndexer(nn.Module):
    """The DSA lightning indexer: I[t, s] = sum_j w[t, j] * relu(qI[t, j] . kI[s]).

    Indexer keys are a SINGLE d_index vector per position shared across the indexer heads
    (that is what makes the indexer cache tiny at decode); queries get n_index_heads
    low-dim heads; w mixes the heads per query. The w projection carries a bias so a
    query-independent +/- head pairing is representable — relu(a) - relu(-a) = a lets two
    heads reconstruct a *signed* bilinear score, which pure ReLU features cannot.

    forward(x: [L, d_model] or [B, L, d_model]) -> causal scores [L, L] or [B, L, L],
    future positions filled with -inf (so softmax(scores) is a distribution over s <= t).
    """

    def __init__(self, cfg: DSAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = Linear(cfg.d_model, cfg.n_index_heads * cfg.d_index)
        self.k_proj = Linear(cfg.d_model, cfg.d_index)
        self.w_proj = Linear(cfg.d_model, cfg.n_index_heads)
        self.w_bias = nn.Parameter(torch.zeros(cfg.n_index_heads))

    def forward(self, x: Tensor) -> Tensor:
        L = x.shape[-2]
        h, d_i = self.cfg.n_index_heads, self.cfg.d_index
        q_i = self.q_proj(x).reshape(*x.shape[:-1], h, d_i)  # (..., L, H_I, d_I)
        k_i = self.k_proj(x)  # (..., L, d_I)
        w = self.w_proj(x) + self.w_bias  # (..., L, H_I)
        # (..., Lq, H_I, d_I) x (..., Lk, d_I) -> (..., Lq, H_I, Lk); scaled like attention
        # so init-time score variance is O(1) regardless of d_index.
        dots = torch.einsum("...qhd,...kd->...qhk", q_i, k_i) / math.sqrt(d_i)
        scores = (torch.relu(dots) * w.unsqueeze(-1)).sum(dim=-2)  # (..., Lq, Lk)
        return scores.masked_fill(~_causal_mask(L, x.device), float("-inf"))


def dense_attention_target(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Dense causal attention distribution p[t, :] — the indexer's distillation target.

    q, k: (L, d) single-head or (H, L, d) multi-head; multi-head targets are averaged over
    heads (== the L1-normalized head-sum, since every per-head row sums to 1 — the
    DeepSeek-V3.2 warmup aggregate). v is accepted for call-site symmetry with the
    attention functions; the *target* is the probability matrix, which does not need it.

    Returns (L, L): rows sum to 1, zero mass on s > t. Kept in the input float precision
    (float64 in, float64 out) so exactness tests can pin KL == 0.
    """

    del v  # the target is the distribution, not the attention output
    if q.dim() not in (2, 3):
        raise ValueError(f"expected (L, d) or (H, L, d), got shape {tuple(q.shape)}")
    d = q.shape[-1]
    L = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if scores.dtype not in (torch.float32, torch.float64):
        scores = scores.float()
    scores = scores.masked_fill(~_causal_mask(L, q.device), float("-inf"))
    probs = softmax(scores, dim=-1)
    if probs.dim() == 3:
        probs = probs.mean(dim=0)
    return probs


def indexer_kl_loss(indexer_scores: Tensor, dense_probs: Tensor) -> Tensor:
    """Row-wise mean KL(dense_probs || softmax(indexer_scores)).

    Expects causally masked inputs (scores == -inf and probs == 0 on s > t, as produced by
    `LightningIndexer` / `dense_attention_target`); rows are treated independently, so
    slicing a row subset (train/held-out queries) is safe. Zero iff softmax(scores)
    matches dense_probs exactly on every row. p == 0 entries contribute exactly 0
    (lim p->0 of p log p); a selected position the indexer gives -inf yields +inf loss —
    the correct, loud signal.
    """

    log_q = torch.log_softmax(indexer_scores, dim=-1)
    p = dense_probs.to(log_q.dtype)
    terms = torch.where(p > 0, p * (torch.log(p.clamp_min(1e-45)) - log_q), torch.zeros_like(p))
    return terms.sum(dim=-1).mean()


def select_topk_mask(scores: Tensor, k: int, causal: bool = True) -> Tensor:
    """Boolean mask of the top-k key positions per query (True = attend).

    scores: (..., L, L). Guarantees, per query row t:
      * only positions s <= t are selectable when causal=True;
      * the query's own position is ALWAYS kept (its sort key is forced to +inf);
      * exactly min(k, #valid positions) entries are True;
      * ties break deterministically toward the lower key index (stable argsort).
    """

    if k < 1:
        raise ValueError(f"top-k budget must be >= 1, got {k}")
    L = scores.shape[-1]
    if scores.shape[-2] != L:
        raise ValueError(f"expected square (..., L, L) scores, got {tuple(scores.shape)}")
    keys = scores
    if causal:
        keys = keys.masked_fill(~_causal_mask(L, scores.device), float("-inf"))
    eye = torch.eye(L, dtype=torch.bool, device=scores.device)
    keys = keys.masked_fill(eye, float("inf"))  # forced self-inclusion outranks everything
    order = torch.argsort(keys, dim=-1, descending=True, stable=True)
    top = order[..., : min(k, L)]
    picked = keys.gather(-1, top)
    mask = torch.zeros_like(keys, dtype=torch.bool)
    mask.scatter_(-1, top, picked > float("-inf"))  # drop -inf picks on short rows
    return mask


def topk_sparse_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
    """Attention restricted to the selected positions: non-selected logits -> -inf.

    Delegates to `model.scaled_dot_product_attention` — the SAME precision path as the
    dense baseline, so with mask == full causal (k >= L) the result is bit-for-bit dense
    causal attention. mask broadcasts over leading (batch, head) dims like any SDPA mask.
    """

    return scaled_dot_product_attention(q, k, v, mask)


def attention_mass_recall(dense_probs: Tensor, topk_mask: Tensor) -> Tensor:
    """Mean over queries of the dense attention mass covered by the selected positions.

    THE quality metric of the selection (the >= 0.95 DoD bar): recall 1.0 means sparse
    attention renormalizes exactly the mass dense attention cared about.
    """

    return (dense_probs * topk_mask.to(dense_probs.dtype)).sum(dim=-1).mean()


@dataclass(frozen=True)
class DSAFlops:
    """Analytic FLOP counts (1 MAC = 2 FLOPs, causal-exact pair counts) for one layer."""

    dense: float  # QK^T + PV over all causal pairs
    sparse: float  # QK^T + PV over the selected pairs only
    indexer: float  # indexer scores over all causal pairs + its projections

    @property
    def sparse_total(self) -> float:
        return self.sparse + self.indexer


def dsa_attention_flops(L: int, d: int, k: int, d_index: int, n_index_heads: int) -> DSAFlops:
    """Dense vs sparse(+indexer) attention FLOPs so F8.2 can locate the crossover k*.

    Conventions: causal-exact pair counts (sum_t (t+1) dense; sum_t min(k, t+1) sparse);
    each attended pair costs 4d FLOPs (2d for the q.k dot + 2d for the p*v accumulate); the
    indexer pays (2*d_index + 2) FLOPs per head per causal pair plus its linear
    projections. Top-k selection (sorting) is not counted — it is O(L^2 log k) comparisons,
    not multiply-accumulates, and vanishes against the d-dim terms.

    The indexer keeps an O(L^2) term, so sparsity only wins while
    n_index_heads * d_index << d — exactly the "lightning" design constraint.
    """

    dense_pairs = L * (L + 1) / 2
    kk = min(k, L)
    # rows t < kk keep t+1 keys; rows t >= kk keep exactly kk
    sparse_pairs = kk * (kk - 1) / 2 + (L - kk + 1) * kk
    dense = 4.0 * d * dense_pairs
    sparse = 4.0 * d * sparse_pairs
    indexer_scores = dense_pairs * n_index_heads * (2.0 * d_index + 2.0)
    indexer_proj = 2.0 * L * d * (n_index_heads * d_index + d_index + n_index_heads)
    return DSAFlops(dense=dense, sparse=sparse, indexer=indexer_scores + indexer_proj)
