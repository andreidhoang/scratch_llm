"""Distributed Data Parallel — replicate the model, shard the data, average the gradients.

A2 systems. The correctness basis is one algebraic identity: with a **mean** loss and an evenly
split batch, the full-batch gradient equals the average of the per-rank gradients —
``∇(mean over B) = (1/B)Σᵢ∇lᵢ = avg_ranks[(2/B)Σ_shard ∇lᵢ]``. So every replica steps identically
to a single process trained on the whole batch, as long as (a) all replicas start from the same
weights (we broadcast from rank 0 at init) and (b) gradients are all-reduced (summed) and divided by
world_size before the optimizer step.

The ladder — each rung removes idle byte-movement, same numerics:
- ``"naive"``   — after backward, all-reduce each grad tensor separately (one collective per param).
- ``"flat"``    — concat all grads into one buffer → a single all-reduce (amortizes per-call latency).
- ``"overlap"`` — fire each grad's all-reduce from a ``register_post_accumulate_grad_hook`` the moment
  it is ready, so communication hides under the still-running backward. This is production DDP and the
  graded deliverable (``get_ddp``); ``finish_gradient_synchronization`` waits on the in-flight handles.

Falsifiable invariant (``tests/test_ddp.py``, 2-rank gloo ×5): for every mode, the DDP-trained params
equal a single-process full-batch model's params to fp tolerance, over several optimizer steps. Kill:
any drift ⇒ a missing init broadcast, a forgotten ``/world_size``, or grads consumed before the
async all-reduce completed.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

DDPMode = str  # one of {"naive", "flat", "overlap"}


class DDP(nn.Module):
    """Wrap a module for data-parallel training. ``mode`` selects the comm strategy (same result)."""

    def __init__(self, module: nn.Module, mode: DDPMode = "overlap") -> None:
        super().__init__()
        self.module = module
        self.mode = mode
        self.world_size = dist.get_world_size()
        # All replicas must start identical — broadcast rank 0's weights (and buffers) to everyone.
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers():
            dist.broadcast(b.data, src=0)

        self._handles: list[tuple[object, Tensor]] = []
        if mode == "overlap":
            for p in self.module.parameters():
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(self._async_all_reduce)

    def _async_all_reduce(self, p: Tensor) -> None:
        """Hook: the instant a grad is ready, start summing it across ranks (non-blocking)."""
        handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append((handle, p))

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Call after ``loss.backward()``, before ``optimizer.step()``. Completes the all-reduce(s)
        and divides by world_size so each grad is the mean across replicas."""
        if self.mode == "overlap":
            for handle, p in self._handles:
                handle.wait()  # type: ignore[attr-defined]
                assert p.grad is not None
                p.grad /= self.world_size
            self._handles.clear()
        elif self.mode == "flat":
            grads = [p.grad for p in self.module.parameters() if p.grad is not None]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= self.world_size
            offset = 0
            for g in grads:
                n = g.numel()
                g.copy_(flat[offset : offset + n].view_as(g))
                offset += n
        else:  # naive
            for p in self.module.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= self.world_size
