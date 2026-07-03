"""FSDP (ZeRO-3, minimal correct) — shard the parameters themselves; the full model exists
only transiently.

A2 systems, the last rung of the sharding ladder. DDP replicates everything; ZeRO-1 shards
optimizer state; ZeRO-3/FSDP shards the **parameters** too: at wrap, each rank keeps only its
1/world_size flat slice of every parameter (the fp32 master shard — and because the optimizer
steps on that shard, its state is shard-sized as well, so params+grads+optimizer state all
scale as 1/W). ``forward()`` all-gathers the shards into full-shape compute copies (optionally
cast to ``compute_dtype`` — e.g. bf16 — before communication to halve bandwidth; masters stay
fp32); after backward, ``finish_gradient_synchronization()`` reduce-scatters each full gradient
(in master dtype, so the reduction is fp32 even under bf16 compute) so every rank ends with
exactly its shard's mean gradient, then re-points each parameter at its master shard — freeing
the full copies. The optimizer then steps the local shard only.

Falsifiable invariant (``tests/test_fsdp.py``, 2-rank gloo, 3 seeds × ≥3 steps, MLP and
TransformerLM): the sharded run's loss, full gradients (reassembled from shards), and parameter
trajectory match unsharded single-process full-batch training to fp32 tolerance, while the
resident parameter footprint *between steps* is ≈ full/world_size (asserted by numel
accounting). Kill: trajectory drift ⇒ a missing init broadcast, a forgotten /world_size, or a
shard/full ``.data`` swap at the wrong time.

Wrap granularity — the ADR decision (A2 guide §7.4): sharding **units are per-parameter** (each
tensor is flattened, padded to a multiple of world_size, and sliced independently — no cross-
parameter flat buffer), but this minimal container gathers *all* parameters at forward and
frees them only at gradient sync. Two consequences, both deliberate on CPU/gloo: (1) peak
resident = full model + shards during fwd/bwd — freeing after forward would be accounting
theater here, since autograd's saved tensors pin the gathered weights until backward anyway;
production FSDP resizes storage and re-gathers per *block* in backward to make the free real.
(2) No overlap/prefetch: per-block units exist precisely so the next block's all-gather can
hide under the current block's compute. **ADR trigger:** if this gloo correctness suite passes
but a GPU benchmark shows all-gathers not overlapping compute, wrap granularity (per-param →
per-block) is the lever to log.

Interview question: "DDP all-reduce moves ~2·(W−1)/W·S bytes per step. What does FSDP move,
and where did the memory go?" — This container moves the *same* bytes (all-gather + reduce-
scatter = the two halves of a ring all-reduce) because it keeps weights resident through
backward; true FSDP frees after forward and re-gathers for backward, paying 3 legs (1.5×) to
cut resident params from S to S/W. Memory is traded for one extra all-gather.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn


@dataclass
class _Shard:
    """One sharded parameter. The Parameter object's identity never changes — only its
    ``.data`` swaps between ``master`` (between steps) and a gathered full copy (fwd/bwd)."""

    param: nn.Parameter
    name: str
    shape: torch.Size
    numel: int
    master: Tensor  # this rank's flat master slice — the only persistent copy


class FSDP(nn.Module):
    """Wrap a module for fully-sharded data parallelism. ``compute_dtype`` selects the dtype
    of the transient compute copies (masters and grad reduction stay in the master dtype)."""

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        # All ranks must shard identical values — broadcast rank 0's weights and buffers first.
        for p in module.parameters():
            dist.broadcast(p.data, src=0)
        for b in module.buffers():
            dist.broadcast(b.data, src=0)
        self._shards: list[_Shard] = []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue  # frozen params stay replicated; gather_full_params returns them as-is
            shape, numel = p.shape, p.numel()
            shard_len = -(-numel // self.world_size)  # ceil ⇒ pad to a multiple of world_size
            flat = torch.zeros(shard_len * self.world_size, dtype=p.dtype, device=p.device)
            flat[:numel] = p.detach().reshape(-1)
            master = flat[self.rank * shard_len : (self.rank + 1) * shard_len].clone()
            p.data = master  # from here on, this rank persistently holds only the shard
            self._shards.append(_Shard(p, name, shape, numel, master))

    def _all_gather_full(self, s: _Shard, dtype: torch.dtype) -> Tensor:
        """All-gather one parameter's shards (communicated in ``dtype``) into its full shape."""
        shard = s.master.to(dtype)
        parts = [torch.empty_like(shard) for _ in range(self.world_size)]
        dist.all_gather(parts, shard)
        return torch.cat(parts)[: s.numel].view(s.shape)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Materialize full compute copies (all-gather each shard), then run the module.
        The copies stay resident through backward; ``finish_gradient_synchronization`` frees
        them."""
        for s in self._shards:
            s.param.data = self._all_gather_full(s, self.compute_dtype or s.master.dtype)
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Call after ``loss.backward()``, before ``optimizer.step()``. Reduce-scatters each
        full gradient (SUM, then /world_size ⇒ mean) so this rank holds exactly its shard's
        gradient in master dtype, and re-points every parameter at its master shard — the full
        compute copies are dropped here."""
        for s in self._shards:
            p = s.param
            shard_len = s.master.numel()
            flat = torch.zeros(
                shard_len * self.world_size, dtype=s.master.dtype, device=s.master.device
            )
            if p.grad is not None:
                flat[: s.numel] = p.grad.detach().reshape(-1).to(s.master.dtype)
            shard_grad = torch.empty_like(s.master)
            dist.reduce_scatter_tensor(shard_grad, flat, op=dist.ReduceOp.SUM)
            shard_grad /= self.world_size
            p.grad = None
            p.data = s.master  # shard-shaped data first, so the grad assignment shape-checks
            p.grad = shard_grad

    def gather_full_params(self) -> dict[str, Tensor]:
        """All-gather every shard into full (unsharded) tensors, in master dtype, without
        disturbing the module's sharded state. Replicated params are returned as-is."""
        out = {s.name: self._all_gather_full(s, s.master.dtype) for s in self._shards}
        for name, p in self.module.named_parameters():
            if name not in out:
                out[name] = p.data
        return out

    def gather_full_grads(self) -> dict[str, Tensor]:
        """Reassemble each parameter's full gradient from the per-rank shard gradients.
        Valid only after ``finish_gradient_synchronization`` (grads must be shard-shaped)."""
        out: dict[str, Tensor] = {}
        for s in self._shards:
            g = s.param.grad
            if g is None or g.numel() != s.master.numel():
                raise RuntimeError(f"{s.name}: gradient is not a synchronized shard")
            parts = [torch.empty_like(g) for _ in range(self.world_size)]
            dist.all_gather(parts, g)
            out[s.name] = torch.cat(parts)[: s.numel].view(s.shape)
        return out

    def resident_param_numel(self) -> int:
        """Elements this container keeps materialized right now (masters + whatever each
        param's ``.data`` currently points at), deduplicated by storage pointer. Between steps
        this is ≈ full/world_size; during fwd/bwd it is full + shards."""
        seen: set[int] = set()
        total = 0
        tensors = [p.data for p in self.module.parameters()]
        tensors += [s.master for s in self._shards]
        for t in tensors:
            key = t.data_ptr()
            if key not in seen:
                seen.add(key)
                total += t.numel()
        return total

    def resident_param_bytes(self) -> int:
        """``resident_param_numel`` weighted by per-tensor element size."""
        seen: set[int] = set()
        total = 0
        tensors = [p.data for p in self.module.parameters()]
        tensors += [s.master for s in self._shards]
        for t in tensors:
            key = t.data_ptr()
            if key not in seen:
                seen.add(key)
                total += t.numel() * t.element_size()
        return total
