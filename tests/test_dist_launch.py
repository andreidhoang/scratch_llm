"""dist_launch — the torchrun shim (runbook §0.5 G2): no-op off torchrun, gloo-verified under it."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scratch_llm.utils.dist_launch import cleanup, maybe_init_from_torchrun


def test_single_process_is_a_noop(monkeypatch) -> None:
    """No torchrun env ⇒ single-process context on the requested device, nothing initialized."""
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    ctx = maybe_init_from_torchrun("cpu")
    assert (ctx.rank, ctx.world_size, ctx.local_rank) == (0, 1, 0)
    assert not ctx.distributed
    assert ctx.device == "cpu"
    assert not dist.is_initialized()
    cleanup(ctx)  # must be a no-op, not an error


def test_world_size_one_env_is_still_single_process(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    ctx = maybe_init_from_torchrun("cpu")
    assert not ctx.distributed
    assert not dist.is_initialized()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gloo_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    ctx = maybe_init_from_torchrun("cpu")
    assert ctx.distributed
    assert (ctx.rank, ctx.world_size, ctx.device) == (rank, world_size, "cpu")
    t = torch.ones(1)
    dist.all_reduce(t)  # every rank's one ⇒ the group is real, not world× replicas
    assert t.item() == world_size
    cleanup(ctx)
    assert not dist.is_initialized()


def test_two_rank_gloo_world() -> None:
    """The shim under a real (CPU gloo) torchrun-style env: ranks init, collectives work,
    cleanup tears down. This is the property the d20's 8×NCCL launch relies on."""
    mp.spawn(_gloo_worker, args=(2, _free_port()), nprocs=2, join=True)  # pyright: ignore[reportPrivateImportUsage]
