"""ZeRO stage 1 — shard the *optimizer state* across data-parallel ranks; params/grads replicated.

A2 systems. AdamW carries two fp32 moment buffers per parameter, so fp32 training costs
16 B/param (4 weight + 4 grad + 8 moments) and plain DDP replicates all of it on every rank.
ZeRO-1's observation: the update is elementwise, so the *parameters* can be partitioned — each
rank owns ~1/world_size of them (greedy least-loaded assignment by numel), keeps optimizer state
only for its shard, steps that shard, then broadcasts the updated values from each owner. Every
rank still ends the step holding the full identical model; only the m/v memory divides by
world_size, for roughly the all-gather half of a ring all-reduce in extra bytes on the wire.

Falsifiable invariant (tests/test_zero1.py, 2-rank gloo): given identical full-batch gradients,
the sharded optimizer's parameter trajectory equals the unsharded optimizer's over multiple steps
and seeds, while per-rank optimizer-state numel ≈ total/world_size (within the largest param's
numel — the greedy-partition bound).

Interview question this answers: "100B params, AdamW — where did the memory go, and what does
ZeRO-1 buy at world size W?" (1.6 TB of state at 16 B/param; the 8 B/param of moments becomes
8/W, at near-DDP communication cost.)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


def partition_by_numel(numels: Sequence[int], world_size: int) -> list[int]:
    """Greedy owner assignment: param i goes to the currently least-loaded rank (ties → lowest
    rank). Guarantees every index gets exactly one owner and max−min rank load ≤ max(numels)."""
    if world_size < 1:
        raise ValueError(f"invalid world_size: {world_size}")
    loads = [0] * world_size
    owners: list[int] = []
    for n in numels:
        rank = min(range(world_size), key=loads.__getitem__)
        owners.append(rank)
        loads[rank] += n
    return owners


def optimizer_state_numel(optimizer: torch.optim.Optimizer) -> int:
    """Total elements across every tensor in ``optimizer.state`` (m and v for AdamW) — the
    quantity ZeRO-1 shards. Bytes = this × element size (8 B/param for fp32 Adam moments)."""
    return sum(
        v.numel()
        for state in optimizer.state.values()
        for v in state.values()
        if isinstance(v, Tensor)
    )


class ShardedOptimizer(torch.optim.Optimizer):
    """ZeRO-1 wrapper: partition params across ranks, step only the local shard with an inner
    ``optimizer_cls`` instance, then broadcast updated params from their owners.

    Single-process (no initialized process group) it degenerates to a plain wrapper around
    ``optimizer_cls``. The partition is fixed at construction; ``add_param_group`` after
    construction is unsupported. Constructor mirrors the official A2 adapter
    ``get_sharded_optimizer(params, optimizer_cls, **kwargs)``.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> None:
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        super().__init__(params, dict(kwargs))

        flat = [p for group in self.param_groups for p in group["params"]]
        owners = partition_by_numel([p.numel() for p in flat], self.world_size)
        self._owner: dict[Tensor, int] = dict(zip(flat, owners, strict=True))

        local_groups: list[dict[str, Any]] = []
        for group in self.param_groups:
            mine = [p for p in group["params"] if self._owner[p] == self.rank]
            if mine:
                options = {k: v for k, v in group.items() if k != "params"}
                local_groups.append({"params": mine, **options})
        self.inner: torch.optim.Optimizer | None = (
            optimizer_cls(local_groups, **kwargs) if local_groups else None
        )

    def params_for_rank(self, rank: int) -> list[Tensor]:
        """The shard owned by ``rank``, in registration order (deterministic across ranks)."""
        return [p for p, owner in self._owner.items() if owner == rank]

    def state_numel_on_rank(self) -> int:
        """Optimizer-state elements materialized on *this* rank — the ZeRO-1 memory claim."""
        return 0 if self.inner is None else optimizer_state_numel(self.inner)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.inner is not None:
            self.inner.step()
        if self.world_size > 1:
            handles = [
                dist.broadcast(p.data, src=self._owner[p], async_op=True)
                for group in self.param_groups
                for p in group["params"]
            ]
            for handle in handles:
                assert handle is not None  # async_op=True always yields a Work handle
                handle.wait()
        return loss

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        """Only the local shard's state — each rank persists ~1/world_size of the total."""
        return {"inner": None if self.inner is None else self.inner.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        inner_state = state_dict["inner"]
        if self.inner is not None and inner_state is not None:
            self.inner.load_state_dict(inner_state)
