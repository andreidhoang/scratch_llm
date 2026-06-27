"""DDP equivalence: each mode (naive/flat/overlap), trained 2-rank on half-batches with averaged
gradients, must match a single-process full-batch model step-for-step. Pure CPU via gloo + mp.spawn.

This is the data-parallel correctness identity in executable form: averaging per-rank gradients of a
mean loss over an even split == the full-batch gradient, so the replicas never diverge."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from scratch_llm.utils.ddp import DDP


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank: int, world_size: int, mode: str, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)  # identical init on every rank
        ref = _MLP()
        inner = _MLP()
        inner.load_state_dict(ref.state_dict())  # ref and ddp start from the same weights
        ddp = DDP(inner, mode=mode)

        torch.manual_seed(123)  # identical full batch on every rank
        local_bs = 4
        x = torch.randn(world_size * local_bs, 4)
        y = torch.randn(world_size * local_bs, 2)
        shard = slice(rank * local_bs, (rank + 1) * local_bs)

        opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
        opt_ddp = torch.optim.SGD(ddp.module.parameters(), lr=0.1)

        for step in range(5):
            opt_ref.zero_grad()
            F.mse_loss(ref(x), y).backward()  # full batch, mean loss
            opt_ref.step()

            opt_ddp.zero_grad()
            F.mse_loss(ddp(x[shard]), y[shard]).backward()  # local half, mean loss
            ddp.finish_gradient_synchronization()  # sum grads across ranks, /world_size
            opt_ddp.step()

            for pr, pd in zip(ref.parameters(), ddp.module.parameters(), strict=True):
                assert torch.allclose(pr, pd, atol=1e-5), f"{mode} drifted at step {step}"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["naive", "flat", "overlap"])
def test_ddp_matches_full_batch(mode: str) -> None:
    world_size = 2
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _worker, args=(world_size, mode, _free_port()), nprocs=world_size, join=True
    )
