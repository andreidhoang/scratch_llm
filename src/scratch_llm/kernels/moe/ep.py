"""S1/S-R4 — expert parallelism: who owns which expert, what crosses the wire, and how much.

EP=2 on 2xH100. Each rank owns half the experts and half the tokens, so every routed slot whose
expert lives on the other rank has to travel: activations out (**dispatch**), expert outputs back
(**combine**). Two all-to-alls per MoE layer, wrapped around the grouped GEMM — never inside it
(vLLM: ``prepare`` → ``apply`` → ``finalize``, ``modular_kernel.py:1228/1349/1395``).

What is measured, and what it is measured against:

* **comm fraction** = time in the two all-to-alls / time in the whole layer. It is the number the
  plan row asks for and the only one that says whether EP bought anything: an expert-parallel
  layer that spends half its time in NCCL has doubled its arithmetic throughput and its latency at
  the same time.
* **per-expert load**, which is what decides the comm volume. With ``top_k = 8`` over 128 experts
  split evenly across 2 ranks, a token sends about half its slots to the peer — so the wire sees
  roughly ``n_tokens * top_k / 2`` rows in each direction, *if* the routing is balanced. It is
  not, always, and the skew is measured rather than assumed (:func:`traffic`).

Everything here is index arithmetic and byte counting; it runs and is tested on a laptop with no
GPU and no process group. :func:`all_to_all_rows` is the one function that needs a real
``torch.distributed`` group, and it refuses with the exact launch command when there is none.

Expert placement follows vLLM's ``linear`` strategy exactly — contiguous chunks, the remainder to
the low ranks (``expert_map_manager.py:69-77``) — so an expert id means the same thing on both
sides of the comparison.

Spec: ``experiments/S1/S-R4/spec.md``   ·   Map: ``experiments/S1/S-R4/map.md``
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from scratch_llm.kernels.moe.routing import (
    GATHER_DTYPE,
    INDEX_DTYPE,
    RUNG,
    Permutation,
    build_permutation,
)


@dataclass(frozen=True)
class ExpertParallel:
    """The expert→rank map. ``linear`` placement, matching vLLM's default."""

    n_experts: int
    ep_size: int

    def __post_init__(self) -> None:
        if self.ep_size < 1:
            raise ValueError(f"{RUNG}: ep_size={self.ep_size} must be >= 1")
        if self.n_experts < self.ep_size:
            raise ValueError(
                f"{RUNG}: {self.n_experts} experts cannot be split over {self.ep_size} ranks"
            )

    @property
    def base(self) -> int:
        return self.n_experts // self.ep_size

    @property
    def remainder(self) -> int:
        return self.n_experts % self.ep_size

    def n_local(self, rank: int) -> int:
        """Experts on ``rank``. The low ranks take the remainder (vLLM's rule, not ours)."""
        self._check_rank(rank)
        return self.base + 1 if rank < self.remainder else self.base

    def local_range(self, rank: int) -> tuple[int, int]:
        """``[start, stop)`` global expert ids owned by ``rank`` — contiguous, by construction."""
        self._check_rank(rank)
        start = rank * self.base + min(rank, self.remainder)
        return start, start + self.n_local(rank)

    def owner_of(self, expert: int) -> int:
        """Which rank computes ``expert``. Inverse of :meth:`local_range`, done by search rather
        than by division because the remainder makes the division wrong for the low ranks."""
        if not 0 <= expert < self.n_experts:
            raise ValueError(f"{RUNG}: expert {expert} outside [0, {self.n_experts})")
        for rank in range(self.ep_size):
            start, stop = self.local_range(rank)
            if start <= expert < stop:
                return rank
        raise AssertionError(
            f"{RUNG}: expert {expert} has no owner — the placement is not a partition"
        )

    def expert_map(self, rank: int) -> Tensor:
        """(n_experts,) int32: local index of each global expert on ``rank``, ``-1`` elsewhere.

        vLLM's exact convention (``expert_map_manager.py:73-77``), so this tensor can be handed
        straight to ``fused_moe(..., expert_map=...)`` when the floor is measured under EP.
        """
        start, stop = self.local_range(rank)
        out = torch.full((self.n_experts,), -1, dtype=INDEX_DTYPE)
        out[start:stop] = torch.arange(stop - start, dtype=INDEX_DTYPE)
        return out

    def owner_of_all(self) -> Tensor:
        """(n_experts,) int64 — the owning rank of every expert, vectorised for counting."""
        owners = torch.empty(self.n_experts, dtype=GATHER_DTYPE)
        for rank in range(self.ep_size):
            start, stop = self.local_range(rank)
            owners[start:stop] = rank
        return owners

    def _check_rank(self, rank: int) -> None:
        if not 0 <= rank < self.ep_size:
            raise ValueError(f"{RUNG}: rank {rank} outside [0, {self.ep_size})")


@dataclass(frozen=True)
class DispatchPlan:
    """One rank's half of the dispatch: what it sends, to whom, in what order.

    The send order is the permutation's own order — expert-major — and because ``linear``
    placement makes each rank's experts contiguous, expert-major is *also* rank-major. That is the
    whole trick: one ``argsort`` produces both the all-to-all's send buffer and the grouped GEMM's
    groups, and no second sort is needed on either side of the wire.

    ``send_counts[d]`` rows go to rank ``d``; ``recv_counts[s]`` rows arrive from rank ``s``. The
    combine is the exact transpose — every row goes back where it came from, in the same order —
    so the inverse permutation that un-permutes locally also un-does the dispatch.
    """

    rank: int
    ep: ExpertParallel
    permutation: Permutation
    send_counts: Tensor  # (ep_size,) int64
    recv_counts: Tensor  # (ep_size,) int64 — filled in by plan_all_ranks / an all-to-all of counts

    @property
    def send_rows(self) -> Tensor:
        """(n_rows,) int64 flat-slot ids in send order — i.e. ``permutation.perm``.

        Named separately because at the call site "the rows I am about to send" and "the
        permutation" are different ideas that happen to be the same array, and the day they stop
        being the same array (a placement that is not linear) this is where it breaks.
        """
        return self.permutation.perm

    @property
    def send_total(self) -> int:
        return int(self.send_counts.sum())

    @property
    def recv_total(self) -> int:
        return int(self.recv_counts.sum())

    @property
    def send_to_peers(self) -> int:
        """Rows that actually cross the wire — the rows destined for this rank never do."""
        return self.send_total - int(self.send_counts[self.rank])

    def validate(self) -> list[str]:
        bad: list[str] = []
        if (
            int(self.send_counts.numel()) != self.ep.ep_size
            or int(self.recv_counts.numel()) != self.ep.ep_size
        ):
            bad.append(f"counts must have one entry per rank ({self.ep.ep_size})")
            return bad
        if self.send_total != self.permutation.n_rows:
            bad.append(
                f"send_counts sum to {self.send_total}, but the permutation has {self.permutation.n_rows} rows"
            )
        if bool((self.send_counts < 0).any()) or bool((self.recv_counts < 0).any()):
            bad.append("a negative count")
        return bad


def plan_dispatch(topk_ids: Tensor, ep: ExpertParallel, rank: int) -> DispatchPlan:
    """Build one rank's dispatch from its own tokens' routing. ``recv_counts`` is left at zero.

    In a real run ``recv_counts`` comes from a tiny all-to-all of the counts themselves — every EP
    implementation does this, because a rank cannot size its receive buffer without it (vLLM's
    DeepEP path calls it the "metadata" exchange). :func:`plan_all_ranks` fills it in directly for
    the in-process simulation the CPU tests use.
    """
    p = build_permutation(topk_ids, ep.n_experts)
    counts_by_expert = p.counts.to(GATHER_DTYPE)
    owners = ep.owner_of_all().to(counts_by_expert.device)
    send = torch.zeros(ep.ep_size, dtype=GATHER_DTYPE)
    send.index_add_(0, owners, counts_by_expert)
    return DispatchPlan(rank, ep, p, send, torch.zeros(ep.ep_size, dtype=GATHER_DTYPE))


def plan_all_ranks(topk_ids_per_rank: list[Tensor], ep: ExpertParallel) -> list[DispatchPlan]:
    """The whole EP group, simulated in one process: every rank's plan with ``recv_counts`` filled.

    ``recv_counts[r][s] = send_counts[s][r]`` — the transpose. Building it here rather than mocking
    a process group is what lets the index math be tested on a laptop, and the property that
    matters (nothing is lost or duplicated across the group) is a statement about the whole matrix,
    which a single-rank test cannot make.
    """
    if len(topk_ids_per_rank) != ep.ep_size:
        raise ValueError(f"{RUNG}: {len(topk_ids_per_rank)} rank routings for ep_size {ep.ep_size}")
    plans = [plan_dispatch(ids, ep, r) for r, ids in enumerate(topk_ids_per_rank)]
    send_matrix = torch.stack([p.send_counts for p in plans])  # [src, dst]
    return [
        DispatchPlan(p.rank, ep, p.permutation, p.send_counts, send_matrix[:, p.rank].clone())
        for p in plans
    ]


def counts_by_expert(topk_ids: Tensor, ep: ExpertParallel) -> Tensor:
    """(n_experts,) int64 — how many of this rank's slots go to each GLOBAL expert.

    The metadata a real EP exchanges before the payload: a receiver cannot size its buffers, and
    cannot build its group offsets, without knowing how many rows each peer is sending for each of
    its local experts. It is a few hundred bytes and it is on the critical path, which is why
    DeepEP-style implementations fuse it into the same all-to-all as the payload.
    """
    return build_permutation(topk_ids, ep.n_experts).counts.to(GATHER_DTYPE)


@dataclass(frozen=True)
class ReceivePlan:
    """How a rank turns what arrived into groups its grouped GEMM can read.

    Received rows arrive source-major: everything from rank 0 (in ITS expert order), then
    everything from rank 1. The grouped GEMM needs them expert-major: all of local expert 0 from
    every source, then all of local expert 1. That is a merge of ``ep_size`` already-sorted runs,
    and — this is the point — it is computable from the counts matrix alone, with no look at the
    data and no sort. ``regroup`` is that permutation; ``group_offsets`` is what the kernel reads.

    ``counts[r][j]`` = rows sent by source rank ``r`` for this rank's local expert ``j``.
    """

    counts: Tensor  # (ep_size, n_local) int64
    regroup: Tensor  # (recv_total,) int64 — expert-major position -> received-row index
    inv_regroup: Tensor  # (recv_total,) int64 — the exact inverse, built once, not per call
    group_offsets: Tensor  # (n_local + 1,) int32

    @property
    def recv_total(self) -> int:
        return int(self.regroup.numel())

    @property
    def n_local(self) -> int:
        return int(self.group_offsets.numel()) - 1

    def to(self, device: torch.device | str) -> ReceivePlan:
        """The same plan with its index tensors on ``device``. Call it ONCE, outside the timed
        window: a per-call ``.to()`` inside :meth:`apply` puts a host->device copy inside a
        measurement of the layer, which is how a comm fraction ends up measuring the harness."""
        return ReceivePlan(
            self.counts,
            self.regroup.to(device),
            self.inv_regroup.to(device),
            self.group_offsets.to(device),
        )

    def apply(self, recv_rows: Tensor) -> Tensor:
        """Source-major rows in, expert-major rows out. The indices must already be on the right
        device (:meth:`to`) — no implicit copy hides in here."""
        if int(recv_rows.shape[0]) != self.recv_total:
            raise ValueError(
                f"{RUNG}: got {int(recv_rows.shape[0])} received rows, planned for {self.recv_total}"
            )
        return recv_rows.index_select(0, self.regroup)

    def undo(self, grouped_rows: Tensor) -> Tensor:
        """Expert-major rows back to source-major, so the combine all-to-all is the exact transpose
        of the dispatch. Inverse of :meth:`apply`, built once in :func:`plan_receive`."""
        if int(grouped_rows.shape[0]) != self.recv_total:
            raise ValueError(
                f"{RUNG}: got {int(grouped_rows.shape[0])} rows, planned for {self.recv_total}"
            )
        return grouped_rows.index_select(0, self.inv_regroup)

    def validate(self) -> list[str]:
        bad: list[str] = []
        if int(self.group_offsets[0]) != 0:
            bad.append(f"group_offsets[0] = {int(self.group_offsets[0])}, must be 0")
        if int(self.group_offsets[-1]) != self.recv_total:
            bad.append(
                f"group_offsets[-1] = {int(self.group_offsets[-1])}, must equal recv_total = {self.recv_total}"
            )
        if self.recv_total and int(self.regroup.max()) >= self.recv_total:
            bad.append("regroup indexes past the received rows")
        if self.recv_total != len(set(self.regroup.tolist())):
            bad.append("regroup is not a permutation — it repeats or drops a row")
        if self.recv_total and not torch.equal(
            self.inv_regroup[self.regroup],
            torch.arange(self.recv_total, dtype=GATHER_DTYPE, device=self.regroup.device),
        ):
            bad.append("inv_regroup is not the inverse of regroup")
        return bad


def plan_receive(counts: Tensor) -> ReceivePlan:
    """Build the receiver's regroup from the ``(ep_size, n_local)`` counts matrix. No data read.

    The arithmetic, spelled out because every index in it is an off-by-one waiting to happen:
    source block ``(r, j)`` starts at the row-major prefix sum of ``counts``; destination block
    ``(j, r)`` starts at the prefix sum of the TRANSPOSED counts. ``regroup`` walks the destination
    order and, for each position, names the source row — ``src_start[block] + (i - dst_start[block])``.
    An empty block contributes zero positions and must not shift the ones after it, which is the
    whole reason this is built by prefix sums rather than by advancing a cursor.
    """
    if counts.ndim != 2:
        raise ValueError(f"{RUNG}: counts must be (ep_size, n_local), got {tuple(counts.shape)}")
    c = counts.to(GATHER_DTYPE)
    ep_size, n_local = (int(v) for v in c.shape)
    src_start = torch.zeros(ep_size * n_local, dtype=GATHER_DTYPE)
    src_start[1:] = c.reshape(-1).cumsum(0)[:-1]  # row-major: (r, j)
    src_start = src_start.view(ep_size, n_local)

    lengths = c.transpose(0, 1).reshape(-1)  # (j, r) order — the destination order
    starts_dst = torch.zeros_like(lengths)
    starts_dst[1:] = lengths.cumsum(0)[:-1]
    total = int(lengths.sum())
    block = torch.repeat_interleave(torch.arange(lengths.numel(), dtype=GATHER_DTYPE), lengths)
    within = torch.arange(total, dtype=GATHER_DTYPE) - starts_dst[block]
    regroup = src_start.transpose(0, 1).reshape(-1)[block] + within

    inv_regroup = torch.empty_like(regroup)
    inv_regroup[regroup] = torch.arange(total, dtype=GATHER_DTYPE)

    per_expert = c.sum(dim=0)  # rows per local expert, over all sources
    offsets = torch.zeros(n_local + 1, dtype=INDEX_DTYPE)
    offsets[1:] = per_expert.cumsum(0).to(INDEX_DTYPE)
    return ReceivePlan(c, regroup, inv_regroup, offsets)


@dataclass(frozen=True)
class CommTraffic:
    """Bytes the two all-to-alls move for one MoE layer, counted — not modelled.

    ``on_wire`` excludes the rows a rank sends to itself: ``all_to_all_single`` still copies them,
    but they never touch NVLink, and counting them would flatter the interconnect by a factor of
    ``ep_size / (ep_size - 1)``. Both numbers are kept because the local copy is not free either.
    """

    dispatch_rows_on_wire: int
    dispatch_rows_total: int
    hidden: int
    elem_bytes: int
    ep_size: int

    @property
    def dispatch_bytes_on_wire(self) -> float:
        return float(self.dispatch_rows_on_wire * self.hidden * self.elem_bytes)

    @property
    def combine_bytes_on_wire(self) -> float:
        """Symmetric with dispatch: every row that went out comes back, same width, same dtype."""
        return self.dispatch_bytes_on_wire

    @property
    def total_bytes_on_wire(self) -> float:
        return self.dispatch_bytes_on_wire + self.combine_bytes_on_wire

    @property
    def peer_fraction(self) -> float:
        """Share of routed slots that leave the rank. ``1 - 1/ep_size`` under perfect balance."""
        return self.dispatch_rows_on_wire / max(self.dispatch_rows_total, 1)

    def as_row(self) -> dict[str, float | int]:
        return {
            "ep_size": self.ep_size,
            "dispatch_rows_on_wire": self.dispatch_rows_on_wire,
            "dispatch_rows_total": self.dispatch_rows_total,
            "peer_fraction": self.peer_fraction,
            "dispatch_bytes_on_wire": self.dispatch_bytes_on_wire,
            "combine_bytes_on_wire": self.combine_bytes_on_wire,
            "total_bytes_on_wire": self.total_bytes_on_wire,
        }


def traffic(plans: list[DispatchPlan], *, hidden: int, elem_bytes: int = 2) -> CommTraffic:
    """Sum the group's dispatch traffic. Takes every rank's plan because the interesting number is
    the group's, and a per-rank number under skewed routing is not ``1/ep_size`` of it."""
    if not plans:
        raise ValueError(f"{RUNG}: no plans")
    ep_size = plans[0].ep.ep_size
    return CommTraffic(
        dispatch_rows_on_wire=sum(p.send_to_peers for p in plans),
        dispatch_rows_total=sum(p.send_total for p in plans),
        hidden=hidden,
        elem_bytes=elem_bytes,
        ep_size=ep_size,
    )


def comm_fraction(comm_ms: float, layer_ms: float) -> float:
    """``comm_ms / layer_ms``, the plan row's number. Both are medians of the same measurement.

    Stated as a ratio of two measured times rather than derived from bytes and a link speed: a
    bytes/bandwidth estimate assumes the all-to-all is bandwidth-bound, and at decode (32 tokens
    per rank, ~128 rows on the wire, 512 KiB) it is latency-bound instead, which is exactly the
    regime where the estimate would be most wrong and most convincing.
    """
    if layer_ms <= 0:
        raise ValueError(f"{RUNG}: layer_ms={layer_ms} must be > 0")
    return comm_ms / layer_ms


def all_to_all_rows(send: Tensor, plan: DispatchPlan, group: object | None = None) -> Tensor:
    """One dispatch: ``(send_total, hidden)`` out, ``(recv_total, hidden)`` in, split by the counts.

    The rows must already be in send order (:attr:`DispatchPlan.send_rows`), which they are if they
    came out of ``routing.permute_rows`` — that is the point of matching the two orders.
    """
    import torch.distributed as dist  # noqa: PLC0415

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            f"{RUNG}: all_to_all_rows needs an initialised process group. On the box:\n"
            f"  torchrun --nproc_per_node=2 bench/kernels/moe/s1_moe.py --point ep2_decode64"
        )
    problems = plan.validate()
    if problems:
        raise ValueError(f"{RUNG}: bad dispatch plan — " + "; ".join(problems))
    out = torch.empty((plan.recv_total, int(send.shape[1])), dtype=send.dtype, device=send.device)
    dist.all_to_all_single(
        out,
        send.contiguous(),
        output_split_sizes=plan.recv_counts.tolist(),
        input_split_sizes=plan.send_counts.tolist(),
        group=group,
    )
    return out


__all__ = [
    "CommTraffic",
    "DispatchPlan",
    "ExpertParallel",
    "ReceivePlan",
    "all_to_all_rows",
    "comm_fraction",
    "counts_by_expert",
    "plan_all_ranks",
    "plan_dispatch",
    "plan_receive",
    "traffic",
]
