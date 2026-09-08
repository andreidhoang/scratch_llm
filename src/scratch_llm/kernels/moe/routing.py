"""S1/S-R4 — top-k routing, the permutation, and the load accounting for a Qwen3-30B-A3B MoE layer.

Everything in this file is index arithmetic on the CPU's terms: no kernel, no device required, no
number that needs silicon to be true. That is deliberate. A fused MoE has exactly one interesting
kernel (the grouped GEMM) and four places where it is silently wrong:

  1. **the tie.** ``torch.topk`` does not promise which of two equal scores it returns, and the
     choice is backend-dependent — so the same prompt can route differently on CPU and GPU, and
     the "kernel is wrong" hunt that follows costs a day. :func:`topk_deterministic` sorts stably
     and takes the first ``k``, so equal affinities always break toward the LOWER expert index.
  2. **the permutation.** Sorting ``(token, slot) -> expert`` into per-expert runs is one
     ``argsort``; inverting it is not, and an inverse that is off by the expert-major stride
     returns every token's output to the wrong token. It shows up as a small loss increase, never
     as a crash.
  3. **the group offsets.** The grouped GEMM reads ``group_offsets[e]..group_offsets[e+1]``. An
     empty expert makes those two equal, and any code that assumes ``> 0`` rows per group either
     divides by zero or — worse — shifts every later group by one and drops one token's slot.
     At decode B=64 there are 512 routed slots over 128 experts, so ~2 experts are empty by
     Poisson alone (e^-4 = 1.8%) before any real skew. It is the common case, not the edge case.
  4. **the load accounting.** ``counts.sum()`` must equal ``n_tokens * top_k`` exactly. It is the
     one invariant that catches a dropped slot anywhere upstream of it, and it costs nothing.

Router recipe (Qwen3-MoE, ``oss/vllm/vllm/model_executor/models/qwen3_moe.py:147-205``):
``softmax`` over all experts, then top-k, then renormalize over the selected k
(``norm_topk_prob``). This is NOT ``src/scratch_llm/moe.py``'s router — that one is DeepSeek-V3's
per-expert ``sigmoid`` plus an aux-loss-free selection bias, and it also carries a shared expert.
Qwen3-30B-A3B has no shared expert. Same file, different model; do not reuse the gate.

Spec: ``experiments/S1/S-R4/spec.md``   ·   Map: ``experiments/S1/S-R4/map.md``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

#: The rung these objects belong to; quoted in error messages so a stack trace names its spec.
RUNG = "S1/S-R4"

#: What the grouped-GEMM kernel loads. int32 is deliberate: ``group_offsets`` is read once per
#: tile by every CTA, and at E=128 the whole array is 516 B — one L2 line's worth of difference,
#: but it also matches what vLLM hands its own kernel (``moe_align_block_size.py:85-88``).
INDEX_DTYPE = torch.int32

#: What ``index_select`` requires. Kept separate from INDEX_DTYPE so neither is silently widened.
GATHER_DTYPE = torch.int64


def topk_deterministic(scores: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
    """``(values, indices)`` of the ``top_k`` largest per row, ties broken toward the LOWER index.

    ``torch.sort(..., stable=True)`` keeps equal elements in their original order, so slicing the
    first ``k`` of a descending stable sort is exactly "highest score, then lowest expert id". A
    plain ``torch.topk`` is faster and its tie behaviour is unspecified — which is fine in training
    and not fine in a test that has to reproduce on two devices.
    """
    if scores.ndim < 1:
        raise ValueError(f"{RUNG}: scores must have at least one dimension, got {scores.shape}")
    n_experts = scores.shape[-1]
    if not 1 <= top_k <= n_experts:
        raise ValueError(f"{RUNG}: top_k={top_k} must be in [1, n_experts={n_experts}]")
    values, indices = torch.sort(scores, dim=-1, descending=True, stable=True)
    return values[..., :top_k].contiguous(), indices[..., :top_k].contiguous()


@dataclass(frozen=True)
class Routing:
    """One layer's routing decision: which experts each token goes to, and with what gate weight.

    ``topk_ids`` is ``INDEX_DTYPE`` and ``topk_weights`` is fp32 — the same pair vLLM's
    ``fused_moe`` takes (``fused_moe.py:1593``), so the floor can be fed the identical routing and
    a tie-break difference can never be mistaken for a numerics difference.
    """

    topk_ids: Tensor  # (N, k) expert index per routed slot
    topk_weights: Tensor  # (N, k) gate weight per routed slot, fp32
    n_experts: int

    @property
    def n_tokens(self) -> int:
        return int(self.topk_ids.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.topk_ids.shape[1])

    @property
    def n_slots(self) -> int:
        """Rows the grouped GEMM will see: one per (token, selected expert) pair."""
        return self.n_tokens * self.top_k

    def validate(self) -> list[str]:
        """Every broken invariant, as sentences. Empty list ⇒ the routing is well formed."""
        bad: list[str] = []
        if self.topk_ids.shape != self.topk_weights.shape:
            bad.append(
                f"topk_ids {tuple(self.topk_ids.shape)} != weights {tuple(self.topk_weights.shape)}"
            )
        if self.topk_ids.numel():
            lo, hi = int(self.topk_ids.min()), int(self.topk_ids.max())
            if lo < 0 or hi >= self.n_experts:
                bad.append(f"expert ids span [{lo}, {hi}], outside [0, {self.n_experts})")
        if self.topk_weights.numel() and not torch.isfinite(self.topk_weights).all():
            bad.append("topk_weights contains a non-finite gate")
        return bad


def route_topk(router_logits: Tensor, top_k: int, *, renormalize: bool = True) -> Routing:
    """Qwen3-MoE routing: softmax over all experts → deterministic top-k → renormalize over k.

    The softmax is taken in fp32 regardless of the logits' dtype (vLLM does the same,
    ``fused_moe.py`` ``fused_topk``); a bf16 softmax over 128 experts loses ~3 bits of the gate,
    and the gate multiplies the expert output, so that error lands directly in the layer's output.
    """
    if router_logits.ndim != 2:
        raise ValueError(
            f"{RUNG}: router_logits must be (n_tokens, n_experts), got {tuple(router_logits.shape)}"
        )
    scores = torch.softmax(router_logits.float(), dim=-1)
    weights, ids = topk_deterministic(scores, top_k)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    return Routing(ids.to(INDEX_DTYPE), weights.float(), n_experts=int(router_logits.shape[-1]))


@dataclass(frozen=True)
class Permutation:
    """The token→expert-major reordering, and its exact inverse.

    Indexing convention, fixed here once so the kernel, the test and the bench cannot disagree:

    * a **flat slot** is ``s = token * top_k + j`` for the ``j``-th expert of that token —
      the row index of ``topk_ids.reshape(-1)``;
    * a **permuted row** ``p`` is the row index the grouped GEMM sees, ordered expert-major;
    * ``perm[p] = s``   (which slot supplies permuted row ``p``);
    * ``inv_perm[s] = p`` (where slot ``s`` ended up);
    * ``token_of_row[p] = perm[p] // top_k`` (which token's activation to gather into row ``p``).

    ``group_offsets`` has ``n_experts + 1`` entries, is non-decreasing, starts at 0 and ends at
    ``n_rows``. An expert with no tokens has ``group_offsets[e] == group_offsets[e+1]``; that is a
    legal group and every consumer must survive it.
    """

    perm: Tensor  # (n_rows,) int64
    inv_perm: Tensor  # (n_rows,) int64
    group_offsets: Tensor  # (n_experts + 1,) int32
    counts: Tensor  # (n_experts,) int64 — rows per expert, == diff(group_offsets)
    top_k: int

    @property
    def n_rows(self) -> int:
        return int(self.perm.numel())

    @property
    def n_experts(self) -> int:
        return int(self.group_offsets.numel()) - 1

    @property
    def n_tokens(self) -> int:
        return self.n_rows // self.top_k

    @property
    def token_of_row(self) -> Tensor:
        """(n_rows,) int64 — the token index each permuted row gathers its activation from."""
        return torch.div(self.perm, self.top_k, rounding_mode="floor")

    def group_bounds(self, expert: int) -> tuple[int, int]:
        """``(start, stop)`` permuted rows of one expert. Equal when the expert got nothing."""
        return int(self.group_offsets[expert]), int(self.group_offsets[expert + 1])

    def validate(self) -> list[str]:
        """Every broken invariant, as sentences. Empty list ⇒ the permutation is usable.

        Checked here rather than asserted at the call site because these are exactly the four
        properties the grouped GEMM's correctness rests on, and they are cheap enough to check in
        a test at every shape the rung runs.
        """
        bad: list[str] = []
        off = self.group_offsets
        if off.numel() < 1 or int(off[0]) != 0:
            bad.append(f"group_offsets[0] = {int(off[0]) if off.numel() else 'missing'}, must be 0")
        if off.numel() and int(off[-1]) != self.n_rows:
            bad.append(f"group_offsets[-1] = {int(off[-1])}, must equal n_rows = {self.n_rows}")
        if off.numel() > 1 and bool((off[1:] < off[:-1]).any()):
            bad.append(
                "group_offsets is not monotone non-decreasing — a group starts before the previous one ends"
            )
        if int(self.counts.sum()) != self.n_rows:
            bad.append(f"counts sum to {int(self.counts.sum())}, not n_rows = {self.n_rows}")
        if self.n_rows:
            round_trip = self.inv_perm[self.perm]
            if not torch.equal(
                round_trip, torch.arange(self.n_rows, dtype=GATHER_DTYPE, device=self.perm.device)
            ):
                bad.append("inv_perm is not the inverse of perm")
        return bad


def build_permutation(topk_ids: Tensor, n_experts: int) -> Permutation:
    """Sort ``(token, slot)`` pairs into expert-major runs and record how to undo it.

    ``argsort(stable=True)`` is what makes the within-expert order *token order*: slot ids are
    already token-major, so a stable sort by expert leaves each expert's rows in increasing token
    index. vLLM's aligner sorts stably too (its golden torch twin is
    ``oss/vllm/tests/kernels/moe/test_moe_align_block_size.py:115``), so the two orders agree row
    for row inside every expert — which is what makes a row-wise diff against the floor readable.

    No padding. vLLM pads each expert's run up to a multiple of ``BLOCK_SIZE_M`` and stores an
    out-of-range token id in the slack (``moe_align_block_size.py:59-72``); here the groups are
    ragged and the kernel masks its tail. The difference is a whole design decision and it is
    named in map.md, not smuggled in here.
    """
    if topk_ids.ndim != 2:
        raise ValueError(f"{RUNG}: topk_ids must be (n_tokens, top_k), got {tuple(topk_ids.shape)}")
    if n_experts < 1:
        raise ValueError(f"{RUNG}: n_experts={n_experts} must be >= 1")
    flat = topk_ids.reshape(-1).to(GATHER_DTYPE)
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= n_experts):
        raise ValueError(
            f"{RUNG}: expert ids span [{int(flat.min())}, {int(flat.max())}], outside "
            f"[0, {n_experts}). An expert-parallel rank must filter to its local experts and "
            f"renumber BEFORE building a permutation — see kernels/moe/ep.py."
        )
    perm = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_experts).to(GATHER_DTYPE)
    offsets = torch.zeros(n_experts + 1, dtype=INDEX_DTYPE, device=topk_ids.device)
    # cumsum into [1:] and a hard 0 at [0] — the one place the off-by-one lives. Writing
    # `offsets = cumsum(...)` and slicing later is the version that drops the first group.
    offsets[1:] = counts.cumsum(0).to(INDEX_DTYPE)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(flat.numel(), dtype=GATHER_DTYPE, device=perm.device)
    return Permutation(perm, inv_perm, offsets, counts, top_k=int(topk_ids.shape[1]))


def permute_rows(rows: Tensor, p: Permutation) -> Tensor:
    """Reorder ``(n_rows, ...)`` from flat-slot order into expert-major order."""
    if rows.shape[0] != p.n_rows:
        raise ValueError(f"{RUNG}: got {rows.shape[0]} rows, permutation is for {p.n_rows}")
    return rows.index_select(0, p.perm)


def unpermute_rows(prows: Tensor, p: Permutation) -> Tensor:
    """Exact inverse of :func:`permute_rows` — expert-major order back to flat-slot order."""
    if prows.shape[0] != p.n_rows:
        raise ValueError(f"{RUNG}: got {prows.shape[0]} rows, permutation is for {p.n_rows}")
    return prows.index_select(0, p.inv_perm)


def gather_tokens(x: Tensor, p: Permutation) -> Tensor:
    """``(n_tokens, hidden)`` → ``(n_rows, hidden)``: expand by top_k AND permute, in one gather.

    This is the physical permutation vLLM does not do (it gathers rows inside the GEMM instead,
    ``fused_moe.py:407-428``). At prefill 4k it moves 4096·8·2048·2 B = 128 MiB of activations
    through HBM that the fused path never touches; at decode B=64 it is 2 MiB and irrelevant. The
    trade is real and it is what the rung measures — the cost is written down here so the profile
    can be checked against it.
    """
    if x.ndim != 2:
        raise ValueError(f"{RUNG}: x must be (n_tokens, hidden), got {tuple(x.shape)}")
    if x.shape[0] != p.n_tokens:
        raise ValueError(f"{RUNG}: x has {x.shape[0]} tokens, permutation is for {p.n_tokens}")
    return x.index_select(0, p.token_of_row)


def combine(y_rows: Tensor, p: Permutation, topk_weights: Tensor) -> Tensor:
    """Un-permute expert outputs, apply the gate, and sum the ``top_k`` contributions per token.

    Three steps that must happen in this order and are fused into one expression on purpose:
    inverse-permute to flat-slot order, scale by that slot's gate, reduce over ``top_k``. The
    reduction is fp32 whatever the input dtype — 8 bf16 addends of similar magnitude lose about
    1.5 bits summed in bf16, and this sum is the layer's output. vLLM's ``moe_sum``
    (``fused_moe.py:1855``) reduces in the compute dtype; that divergence is named in map.md.
    """
    if topk_weights.shape != (p.n_tokens, p.top_k):
        raise ValueError(
            f"{RUNG}: topk_weights {tuple(topk_weights.shape)} does not match the permutation's "
            f"({p.n_tokens}, {p.top_k})"
        )
    flat = unpermute_rows(y_rows, p).float()
    scaled = flat * topk_weights.reshape(-1, 1).float()
    return scaled.view(p.n_tokens, p.top_k, -1).sum(dim=1).to(y_rows.dtype)


@dataclass(frozen=True)
class ExpertLoad:
    """Per-expert slot counts and the three numbers the plan row asks for.

    ``max_over_mean`` is the imbalance that decides the grouped GEMM's tail: the kernel finishes
    when its busiest expert does, so a value of 3 means two thirds of the tile-time on the widest
    group is spent with the rest of the machine idle.
    """

    counts: Tensor  # (n_experts,) int64
    n_tokens: int
    top_k: int

    @property
    def n_experts(self) -> int:
        return int(self.counts.numel())

    @property
    def total_slots(self) -> int:
        return int(self.counts.sum())

    @property
    def expected_slots(self) -> int:
        return self.n_tokens * self.top_k

    @property
    def n_empty(self) -> int:
        """Experts that received nothing. Non-zero is normal at decode, not a bug."""
        return int((self.counts == 0).sum())

    @property
    def fraction(self) -> Tensor:
        return self.counts.double() / max(self.total_slots, 1)

    @property
    def max_over_mean(self) -> float:
        """1.0 is perfect balance; the grouped GEMM's tail is proportional to this."""
        if self.total_slots == 0:
            return 0.0
        return float(self.counts.max()) * self.n_experts / self.total_slots

    def validate(self) -> list[str]:
        bad: list[str] = []
        if self.total_slots != self.expected_slots:
            bad.append(
                f"per-expert counts sum to {self.total_slots}, but {self.n_tokens} tokens x "
                f"top_k {self.top_k} = {self.expected_slots} slots were routed — a slot was dropped"
            )
        if bool((self.counts < 0).any()):
            bad.append("a negative expert count")
        return bad

    def as_row(self) -> dict[str, float | int]:
        """The JSON row the bench records: the shape of the load, not the whole histogram."""
        return {
            "n_experts": self.n_experts,
            "total_slots": self.total_slots,
            "n_empty_experts": self.n_empty,
            "max_slots": int(self.counts.max()) if self.n_experts else 0,
            "min_slots": int(self.counts.min()) if self.n_experts else 0,
            "mean_slots": self.total_slots / max(self.n_experts, 1),
            "max_over_mean": self.max_over_mean,
        }


def expert_load(topk_ids: Tensor, n_experts: int) -> ExpertLoad:
    """Count routed slots per expert. ``counts.sum() == n_tokens * top_k`` is the invariant."""
    if topk_ids.ndim != 2:
        raise ValueError(f"{RUNG}: topk_ids must be (n_tokens, top_k), got {tuple(topk_ids.shape)}")
    counts = torch.bincount(topk_ids.reshape(-1).to(GATHER_DTYPE), minlength=n_experts).to(
        GATHER_DTYPE
    )
    return ExpertLoad(counts, n_tokens=int(topk_ids.shape[0]), top_k=int(topk_ids.shape[1]))


__all__ = [
    "GATHER_DTYPE",
    "INDEX_DTYPE",
    "RUNG",
    "ExpertLoad",
    "Permutation",
    "Routing",
    "build_permutation",
    "combine",
    "expert_load",
    "gather_tokens",
    "permute_rows",
    "route_topk",
    "topk_deterministic",
    "unpermute_rows",
]
