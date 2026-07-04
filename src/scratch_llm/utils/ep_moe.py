"""Expert-parallel MoE — dispatch → expert-GEMM → combine over an all-to-all, token-choice top-K.

A6 systems, the MoE half of expert parallelism (EP). A dense DDP replica runs *every* expert on
*every* device; that does not scale — 256 DeepSeek-V3 experts will not fit. EP instead **shards the
experts across ranks** (rank ``r`` owns ``E/W`` of them) and keeps the tokens where they are, then
routes each token to the rank that owns its chosen expert. The movement is a pair of all-to-alls:

    router (replicated) → top-K per token
      → DISPATCH  all-to-all: send each (token, chosen-expert) slot to the expert's owner rank
      → expert FFN runs locally on the tokens it received (grouped by local expert)
      → COMBINE   all-to-all: send each result back to the token's origin rank
      → un-permute + gate-weight + sum over K → local output

The correctness basis is a single identity: **which expert a token is processed by, and with what
gate weight, is independent of where that expert physically lives.** So the expert-parallel output
must be *numerically identical* to a single-process dense reference that holds all experts and all
tokens and routes every token to its expert directly. The all-to-all only relocates bytes; it never
changes the arithmetic. Two things must hold for that identity to bite:

1. **Routing determinism.** The router is replicated bit-identical on every rank, so every rank
   makes the *same* top-K decision the dense reference makes. A 1e-6 gate-logit perturbation can flip
   a top-K tie and send a token to a different expert — so the invariant asserts **zero
   token-divergence**: the (token → expert) map from the EP path equals the CPU routing reference
   exactly, not just approximately.
2. **Weight identity.** Rank ``r``'s expert shard holds the same weights the reference's experts
   ``[r·E/W, (r+1)·E/W)`` hold. :func:`build_experts` builds the full set deterministically from a
   seed so every rank slices an identical global expert bank — no cross-rank broadcast needed.

Variable-sized all-to-all: token load per destination rank is data-dependent, so the split sizes are
not known a priori. We pay a first, tiny ``all_to_all_single`` on the *counts* (W longs) to learn how
many tokens each rank will receive, then the real ``all_to_all_single`` on the features/results with
those splits. The COMBINE leg reuses the DISPATCH splits swapped (it is the exact inverse permutation).

Falsifiable invariant (``tests/test_ep_moe.py``, 2-4 rank gloo): the concatenated EP output over all
ranks equals the single-process dense-gather reference to fp tolerance, AND the routing decision
matches the reference token-for-token (zero divergence). Kill: any token-divergence ⇒ a router that
is not bit-replicated or a tie-break that differs; any output drift at zero divergence ⇒ a
dispatch/combine permutation bug (results landing on the wrong slot) or a missing gate re-weight.

Per-rank expert **load** (how many token-slots each rank / each local expert received) is returned so
the caller can log routing imbalance — the quantity EP throughput is bottlenecked on (the busiest
rank sets the step time; a hot expert starves the others).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from scratch_llm.model import SwiGLU
from scratch_llm.moe import Router


def build_experts(d_model: int, d_ff: int, n_experts: int, seed: int) -> nn.ModuleList:
    """Deterministically build the *full* bank of ``n_experts`` SwiGLU experts from ``seed``.

    Every rank calls this identically, then keeps only its shard (see :func:`shard_range`); because
    the RNG stream is fixed, expert ``e`` has the same weights on every rank and in the reference.
    """
    gen = torch.Generator().manual_seed(seed)
    experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(n_experts))
    # Re-init every parameter from the shared generator so the bank is seed-deterministic regardless
    # of module construction order / global RNG state.
    for p in experts.parameters():
        with torch.no_grad():
            p.copy_(torch.empty_like(p).uniform_(-0.05, 0.05, generator=gen))
    return experts


def build_router(d_model: int, n_experts: int, seed: int) -> Router:
    """Deterministically build a replicated :class:`~scratch_llm.moe.Router` from ``seed``."""
    gen = torch.Generator().manual_seed(seed)
    router = Router(d_model, n_experts)
    with torch.no_grad():
        router.gate.weight.copy_(
            torch.empty_like(router.gate.weight).uniform_(-0.1, 0.1, generator=gen)
        )
    return router


def shard_range(n_experts: int, world_size: int, rank: int) -> range:
    """The contiguous block of global expert ids owned by ``rank`` (experts split evenly across W).

    Requires ``n_experts % world_size == 0`` — the clean, common EP layout.
    """
    if n_experts % world_size != 0:
        raise ValueError(f"n_experts={n_experts} must be divisible by world_size={world_size}")
    per = n_experts // world_size
    return range(rank * per, (rank + 1) * per)


def route(logits: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
    """Token-choice top-K routing decision. ``logits`` (N, E) → (topk_idx (N, K), gate (N, K)).

    Mirrors :class:`~scratch_llm.moe.MoEFeedForward`: affinity ``s = sigmoid(logits)`` per expert
    (independent), select the top-K by ``s`` (bias-free here — the aux-loss bias is a training-time
    balancer, orthogonal to dispatch correctness), gate value = raw affinity normalized over the
    selected K so it sums to 1. Deterministic given ``logits`` — this is *the* routing oracle both
    the EP path and the dense reference consult, so their decisions cannot disagree.
    """
    affinity = torch.sigmoid(logits)
    topk_idx = affinity.topk(top_k, dim=-1).indices  # (N, K)
    gate_sel = affinity.gather(-1, topk_idx)  # (N, K)
    gate = gate_sel / gate_sel.sum(-1, keepdim=True)
    return topk_idx, gate


def dense_moe_reference(
    x: Tensor, router: Router, experts: nn.ModuleList, top_k: int
) -> tuple[Tensor, Tensor]:
    """Single-process dense-gather reference: all experts, all tokens, every token run on its expert.

    Returns ``(y (N, d), expert_of_slot (N, K))`` — the second tensor is the routing decision, the
    zero-divergence oracle the EP path is checked against token-for-token.
    """
    logits = router(x)  # (N, E)
    topk_idx, gate = route(logits, top_k)  # (N, K), (N, K)
    y = torch.zeros_like(x)
    n_experts = len(experts)
    for e in range(n_experts):
        # (token, k) slots that chose expert e.
        hit = topk_idx == e  # (N, K) bool
        tok = hit.any(dim=-1).nonzero(as_tuple=True)[0]  # tokens routing to e at least once
        if tok.numel() == 0:
            continue
        out = experts[e](x[tok])  # (M, d)
        # gate weight of expert e for each of those tokens (a token picks e at most once).
        w = (gate * hit).sum(-1)[tok].unsqueeze(-1)  # (M, 1)
        y = y.index_add(0, tok, w * out)
    return y, topk_idx


@dataclass
class EPLoad:
    """Per-step routing load, for imbalance logging. ``recv_total`` is how many token-slots this rank
    received (the work it must do); ``per_local_expert`` breaks that down by owned expert. A large
    spread across ranks is the EP throughput bottleneck (the busiest rank sets step time)."""

    rank: int
    recv_total: int
    per_local_expert: list[int]
    send_per_rank: list[int]  # slots this rank dispatched to each destination rank


class ExpertParallelMoE(nn.Module):
    """Expert-parallel MoE layer. Holds a replicated router and *only this rank's expert shard*;
    ``forward`` dispatches tokens to expert owners over all-to-all, runs the local experts, and
    combines the results back.

    Construct every rank identically via :func:`build_router` / :func:`build_experts` + a shard slice
    so the global expert bank matches a :func:`dense_moe_reference` built from the same seed.
    """

    def __init__(
        self, router: Router, expert_shard: nn.ModuleList, expert_ids: range, top_k: int
    ) -> None:
        super().__init__()
        self.router = router
        self.experts = expert_shard  # this rank's E/W experts, in global-id order
        self.expert_ids = expert_ids  # global ids of the shard (contiguous)
        self.top_k = top_k
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.per_rank = len(expert_ids)  # experts per rank (== E/W)

    def forward(self, x_local: Tensor) -> tuple[Tensor, Tensor, EPLoad]:
        """``x_local`` (N, d) → ``(y_local (N, d), expert_of_slot (N, K), load)``.

        ``expert_of_slot`` is the routing decision (for the zero-divergence oracle); ``load`` reports
        this rank's received-token counts.
        """
        n, d = x_local.shape
        k = self.top_k
        w = self.world_size
        per = self.per_rank

        # --- routing (replicated, identical on every rank) ---------------------------------------
        logits = self.router(x_local)  # (N, E)
        topk_idx, gate = route(logits, k)  # (N, K), (N, K)

        # Flatten (token, k) → slots. Slot s ↔ token s//K, k-position s%K (row-major reshape).
        s = n * k
        flat_expert = topk_idx.reshape(-1)  # (S,) global expert id per slot
        flat_token = (
            torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
        )  # (S,) origin token index
        feat = x_local[flat_token]  # (S, d) the token feature carried by each slot

        dest_rank = flat_expert // per  # (S,) owner rank of each slot's expert
        local_expert = flat_expert % per  # (S,) that expert's index within the owner's shard

        # Group slots by destination rank (stable sort → contiguous per-rank blocks for the split).
        perm = torch.argsort(dest_rank, stable=True)  # (S,)
        send_feat = feat[perm].contiguous()  # (S, d)
        send_lexpert = local_expert[perm].contiguous()  # (S,)
        in_splits = torch.bincount(dest_rank, minlength=w)  # (W,) slots to each dest

        # --- metadata all-to-all: learn how many slots each rank sends us -------------------------
        out_splits = torch.empty(w, dtype=in_splits.dtype)
        dist.all_to_all_single(out_splits, in_splits)
        recv_total = int(out_splits.sum())
        in_list, out_list = in_splits.tolist(), out_splits.tolist()

        # --- DISPATCH all-to-all: features + which local expert each received slot needs ----------
        recv_feat = torch.empty(recv_total, d, dtype=send_feat.dtype)
        dist.all_to_all_single(
            recv_feat, send_feat, output_split_sizes=out_list, input_split_sizes=in_list
        )
        recv_lexpert = torch.empty(recv_total, dtype=send_lexpert.dtype)
        dist.all_to_all_single(
            recv_lexpert, send_lexpert, output_split_sizes=out_list, input_split_sizes=in_list
        )

        # --- local expert-GEMM: run each owned expert on the tokens it received -------------------
        recv_out = torch.empty_like(recv_feat)
        per_local: list[int] = []
        for le in range(per):
            mask = recv_lexpert == le
            per_local.append(int(mask.sum()))
            if not bool(mask.any()):
                continue
            recv_out[mask] = self.experts[le](recv_feat[mask])

        # --- COMBINE all-to-all: send each result back to its origin (splits swapped = inverse) ---
        back = torch.empty(s, d, dtype=recv_out.dtype)
        dist.all_to_all_single(
            back, recv_out, output_split_sizes=in_list, input_split_sizes=out_list
        )

        # Un-permute back to slot order, gate-weight, sum over K.
        slot_out = torch.empty_like(back)
        slot_out[perm] = back  # inverse of the argsort permutation
        y = (slot_out.view(n, k, d) * gate.unsqueeze(-1)).sum(dim=1)  # (N, d)

        load = EPLoad(
            rank=self.rank,
            recv_total=recv_total,
            per_local_expert=per_local,
            send_per_rank=in_list,
        )
        return y, topk_idx, load


__all__ = [
    "EPLoad",
    "ExpertParallelMoE",
    "build_experts",
    "build_router",
    "dense_moe_reference",
    "route",
    "shard_range",
]
