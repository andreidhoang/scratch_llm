"""Minimal torchrun launcher shim (d20 runbook §0.5 gap G2 — closed for the probe path).

``train()``'s ZeRO-2 path activates on ``dist.is_initialized()``, but nothing in ``src/``
initialized the process group from torchrun's env: under a bare ``torchrun --nproc_per_node=8``
every rank would run as an independent single-GPU replica — silent world× data loss, no
reduce_scatter/all_gather, and all ranks hammering ``cuda:0``. Entrypoints that opt in call
``maybe_init_from_torchrun()`` before building anything CUDA-bound, use the returned per-rank
device, and call ``cleanup()`` at the end. First customer: ``scripts/d20_probe.py`` (the P5.5
probe, ADR-0020); speedrun's own ``main`` gets the same wiring as a P5 deliverable.

Single-process launches (no ``WORLD_SIZE`` in env, or ``WORLD_SIZE=1``) are a no-op pass-through,
so the same entrypoint works on a laptop and under torchrun unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistContext:
    """How this process sits in the (possibly trivial) distributed world."""

    rank: int
    world_size: int
    local_rank: int
    distributed: bool  # True ⇒ a process group is initialized and ZeRO-2 is live in train()
    device: str  # the per-rank device to build the model on ("cuda:{local_rank}" or as given)


def maybe_init_from_torchrun(device: str = "cuda") -> DistContext:
    """Initialize the process group from torchrun's env (RANK/WORLD_SIZE/LOCAL_RANK), or no-op.

    Under torchrun: sets the per-rank CUDA device BEFORE any tensor is allocated (all ranks
    landing on cuda:0 is the classic silent failure), picks nccl for CUDA / gloo for CPU, and
    initializes the default group env:// style (MASTER_ADDR/MASTER_PORT come from torchrun).
    Without torchrun: returns a single-process context on the requested device, unchanged.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        return DistContext(rank=0, world_size=1, local_rank=0, distributed=False, device=device)

    import torch
    import torch.distributed as dist

    if "cuda" in str(device):
        torch.cuda.set_device(local_rank)
        backend, dev = "nccl", f"cuda:{local_rank}"
    else:
        backend, dev = "gloo", "cpu"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        distributed=True,
        device=dev,
    )


def cleanup(ctx: DistContext) -> None:
    """Barrier (so no rank exits while another still expects collectives) + destroy the group."""
    if not ctx.distributed:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
