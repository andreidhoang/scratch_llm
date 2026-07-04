"""Expert-parallel MoE correctness: the dispatch → expert-GEMM → combine over all-to-all must be
numerically identical to a single-process dense-gather reference, and its routing decisions must
match the CPU routing oracle token-for-token (zero divergence). Pure CPU via gloo + mp.spawn.

The oracle is built once, deterministically from a seed, so every rank reconstructs the *same* global
expert bank and the *same* replicated router — the only thing the EP path adds over the reference is
the two all-to-alls, which move bytes but must not change arithmetic."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scratch_llm.utils.ep_moe import (
    ExpertParallelMoE,
    build_experts,
    build_router,
    dense_moe_reference,
    shard_range,
)

_D_MODEL = 16
_D_FF = 32
_SEED = 1234


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(
    rank: int, world_size: int, n_experts: int, top_k: int, n_local: int, port: int
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        # Every rank builds the identical full router + expert bank (seed-deterministic), so the EP
        # shard weights equal the reference's experts and the router decision is bit-replicated.
        router = build_router(_D_MODEL, n_experts, _SEED)
        full_experts = build_experts(_D_MODEL, _D_FF, n_experts, _SEED)

        ids = shard_range(n_experts, world_size, rank)
        shard = torch.nn.ModuleList(full_experts[e] for e in ids)
        ep = ExpertParallelMoE(router, shard, ids, top_k)

        # Deterministic per-rank token block; the full token set is the concatenation over ranks.
        gen = torch.Generator().manual_seed(100 + rank)
        x_local = torch.randn(n_local, _D_MODEL, generator=gen)

        # Gather the full token set on every rank to build the single-process reference locally.
        parts = [torch.empty(n_local, _D_MODEL) for _ in range(world_size)]
        dist.all_gather(parts, x_local.contiguous())
        x_full = torch.cat(parts, dim=0)  # (W*N, d), rank r owns rows [r*N:(r+1)*N]

        with torch.no_grad():
            y_ref_full, route_ref_full = dense_moe_reference(x_full, router, full_experts, top_k)
            y_ep, route_ep, load = ep(x_local)

        lo, hi = rank * n_local, (rank + 1) * n_local
        y_ref = y_ref_full[lo:hi]
        route_ref = route_ref_full[lo:hi]

        # (1) ZERO token-divergence: the EP routing decision matches the CPU routing oracle exactly.
        n_div = int((route_ep != route_ref).any(dim=-1).sum())
        assert n_div == 0, f"rank {rank}: {n_div} tokens diverged from the routing oracle"

        # (2) Numerical identity: EP output == dense-gather reference to fp tolerance.
        assert torch.allclose(y_ep, y_ref, atol=1e-6, rtol=1e-5), (
            f"rank {rank}: EP output drifted from dense reference "
            f"(max |Δ|={float((y_ep - y_ref).abs().max()):.2e})"
        )

        # (3) Load accounting is self-consistent: received slots = sum over local experts, and total
        # dispatched slots across all ranks == W*N*K (every (token, k) slot is delivered once).
        assert load.recv_total == sum(load.per_local_expert)
        sent = torch.tensor(load.send_per_rank)
        total_sent = torch.empty_like(sent)
        dist.all_reduce(sent)  # sum each rank's per-dest send vector across ranks
        total_sent.copy_(sent)
        assert int(total_sent.sum()) == world_size * n_local * top_k
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world_size", "n_experts", "top_k", "n_local"),
    [
        (2, 4, 2, 6),  # 2 experts/rank, top-2
        (2, 8, 1, 10),  # top-1 routing
        (4, 8, 2, 5),  # 4 ranks, 2 experts/rank
        (4, 4, 1, 7),  # 1 expert/rank, top-1 (every rank owns exactly one expert)
    ],
)
def test_ep_moe_matches_dense_reference(
    world_size: int, n_experts: int, top_k: int, n_local: int
) -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _worker,
        args=(world_size, n_experts, top_k, n_local, _free_port()),
        nprocs=world_size,
        join=True,
    )
