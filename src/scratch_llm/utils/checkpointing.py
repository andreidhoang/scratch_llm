"""Activation (gradient) checkpointing — the recompute-vs-store memory lever for training.

A2 systems. The *same* trade as the FA2 backward (``kernels/flash_attention.py``), one level up:
instead of recomputing the attention probability matrix, recompute whole ``TransformerBlock``
internals in the backward pass so their activations need not be stored during the forward. Three
modes span the trade:

- ``"none"``    — store every activation (max memory, min compute); the default eager path.
- ``"full"``   — store only each block's *input*, recompute the entire block in backward. Activation
  memory falls from O(depth · per_block) toward O(depth · residual) at the cost of ≈ one extra
  forward. (This is the assignment's checkpointing deliverable; the O(√N) recursive variant is the
  same idea applied to *where* you place the checkpoints.)
- ``"selective"`` — the 2026 default (Korthikanti et al.; Megatron/TorchTitan). Save the activations
  that are cheap to store but expensive to recompute (the **matmul outputs**) and recompute the rest
  (the cheap softmax / elementwise). ~70% activation memory for ~2.7% extra FLOPs at GPT-3 scale.

Falsifiable invariants (``tests/test_checkpointing.py``), both CPU:
1. Gradients under ``"full"`` and ``"selective"`` equal eager (``"none"``) gradients to fp tolerance
   — recompute must be transparent to autograd. Kill: any mismatch ⇒ the recompute boundary is
   wrong (non-determinism inside the block, or a dropped saved tensor).
2. Saved-activation **bytes** are ordered ``full < selective < none`` — the recompute-vs-store
   accounting, measured on CPU via :func:`torch.autograd.graph.saved_tensors_hooks` (a
   device-agnostic surrogate for the GPU peak-activation memory the same code reduces on a real run).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

# Matmul-family ops: one cheap-to-store output, expensive-to-recompute GEMM FLOPs. Saving these and
# recomputing everything else IS the selective-activation-checkpointing (SAC) policy.
MATMUL_OPS = frozenset(
    {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.matmul.default,
    }
)
_SAVE_OPS = MATMUL_OPS

CheckpointMode = str  # one of {"none", "full", "selective"}


def _sac_policy(_ctx: Any, op: Any, *_args: Any, **_kwargs: Any) -> CheckpointPolicy:
    """Save matmul outputs, recompute the rest — the store-cheap / recompute-expensive split."""
    if op in _SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def run_block(block: Callable[..., Any], mode: CheckpointMode, *args: Any) -> Any:
    """Run one block under ``mode``. Uses non-reentrant checkpoint — the modern path that composes
    with ``saved_tensors_hooks`` and the selective policy."""
    if mode == "none":
        return block(*args)
    if mode == "full":
        return checkpoint(block, *args, use_reentrant=False)
    if mode == "selective":
        return checkpoint(
            block,
            *args,
            use_reentrant=False,
            context_fn=lambda: create_selective_checkpoint_contexts(_sac_policy),
        )
    raise ValueError(f"unknown checkpoint mode: {mode!r}")


@contextmanager
def count_saved_activation_bytes() -> Iterator[dict[str, int]]:
    """Sum the bytes of distinct tensors saved for backward across the checkpoint **boundary**.

    The memory axis of the recompute-vs-store trade. A CPU surrogate for peak activation memory:
    both ``"full"`` and ``"selective"`` cut this drastically vs ``"none"`` (they recompute internals
    instead of storing them). Note ``full ≈ selective`` here — SAC caches its saved matmul outputs
    *inside* the checkpoint region, invisible to these outer hooks; the selective-vs-full difference
    shows up on the **compute** axis (:func:`count_op_executions`), not here. Dedups by storage
    pointer so a tensor saved by several ops is counted once."""
    total = {"bytes": 0}
    seen: set[int] = set()

    def pack(t: Tensor) -> Tensor:
        ptr = t.untyped_storage().data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total["bytes"] += t.untyped_storage().nbytes()
        return t

    def unpack(t: Tensor) -> Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield total


class _OpCounter(TorchDispatchMode):
    def __init__(self, watch: frozenset[Any]) -> None:
        self.watch = watch
        self.watched = 0
        self.total = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.total += 1
        if func in self.watch:
            self.watched += 1
        return func(*args, **(kwargs or {}))


@contextmanager
def count_op_executions(watch: frozenset[Any] = MATMUL_OPS) -> Iterator[dict[str, int]]:
    """Count aten op executions inside the ``with`` block — the compute axis of the trade.

    Keep both the forward *and* the backward inside the block: recompute happens in backward, so the
    extra matmul executions ``"full"`` pays (and ``"selective"`` avoids) only appear once backward
    runs. Returns ``{"watched": n_matmuls, "total": n_ops}`` (filled on exit)."""
    counter = _OpCounter(watch)
    with counter:
        result = {"watched": 0, "total": 0}
        yield result
    result["watched"] = counter.watched
    result["total"] = counter.total
