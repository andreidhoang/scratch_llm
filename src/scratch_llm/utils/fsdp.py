"""FSDP (ZeRO-3, minimal correct) — shard the *matrices*, replicate the *norms/biases*; the
full model exists only transiently.

A2 systems, the last rung of the sharding ladder. DDP replicates everything; ZeRO-1 shards
optimizer state; ZeRO-3/FSDP shards the **parameters** too. But not *all* of them:

Sharding policy — the ADR decision (A2 guide §7.4, and the CS336 official gradient-sync
contract). Each parameter is classified once, at wrap time, by rank:
- **SHARDED** — ``ndim >= 2`` (the big weight matrices: Linear/Embedding weights). Each rank keeps
  only its 1/world_size flat slice (the fp32 master shard). ``forward()`` all-gathers the shards
  into a full compute copy; after backward the full gradient is **reduce-scattered** so each rank
  ends with its shard's mean gradient. Params + grads + optimizer state all scale as 1/W.
- **REPLICATED** — ``ndim <= 1`` (RMSNorm weights, biases — the tiny 1-D params). Every rank keeps
  a *full* fp32 copy (no shard, no gather); after backward the full gradient is **all-reduced**
  (SUM /world_size ⇒ mean) so it is *identical on every rank*. Sharding a 64-element norm buys
  almost no memory yet still pays a full comm; replicating it is the standard FSDP choice.

Why this split is mandatory, not cosmetic: the official ``test_fsdp_gradient_sync`` (both fp32 and
fp16) asserts that the gradient of every non-Linear/-Embedding parameter is *bit-identical across
ranks* after ``finish_gradient_synchronization``. A reduce-scattered shard gradient is, by
construction, different per rank (each rank owns a different slice), so it fails that check — the
replicate-and-all-reduce path is what makes the norm/bias grads match. The classification by
``ndim`` coincides exactly with the test's "parent module is not Linear/Embedding" predicate on
the tested model (all matrices are 2-D, all norms are 1-D), and is the general ZeRO-3 convention.

Dtype toggle (both classes): ``compute_dtype`` selects the dtype of the *transient* compute copies
(gathered shards for the sharded class, a cast full copy for the replicated class). The fp32
masters and the gradient reduction always stay fp32 — so the reduction is fp32 even under bf16/fp16
compute, and after sync every parameter's ``.grad`` is fp32 and shape-matches its ``.data`` (shard-
shaped for sharded, full-shaped for replicated), exactly as the contract requires.

Falsifiable invariant (``tests/test_fsdp.py``, 2-rank gloo, 3 seeds × ≥3 steps, MLP and
TransformerLM): the sharded run's loss, full gradients (reassembled from shards / read directly for
replicas), and parameter trajectory match unsharded single-process full-batch training to fp32
tolerance, while the resident parameter footprint *between steps* is ≈ (sharded)/world_size +
(replicated) full (numel accounting). Kill: trajectory drift ⇒ a missing init broadcast, a
forgotten /world_size, or a shard/full ``.data`` swap at the wrong time.

Wrap granularity — the ADR decision (A2 guide §7.4): sharding **units are per-parameter** (each
sharded tensor is flattened, padded to a multiple of world_size, and sliced independently — no
cross-parameter flat buffer), but this minimal container gathers *all* sharded parameters at
forward and frees them only at gradient sync. Two consequences, both deliberate on CPU/gloo: (1)
peak resident = full model + shards during fwd/bwd — freeing after forward would be accounting
theater here, since autograd's saved tensors pin the gathered weights until backward anyway;
production FSDP resizes storage and re-gathers per *block* in backward to make the free real. (2)
No overlap/prefetch: per-block units exist precisely so the next block's all-gather can hide under
the current block's compute. **ADR trigger:** if this gloo correctness suite passes but a GPU
benchmark shows all-gathers not overlapping compute, wrap granularity (per-param → per-block) is
the lever to log.

Interview question: "DDP all-reduce moves ~2·(W−1)/W·S bytes per step. What does FSDP move, and
where did the memory go?" — This container moves the *same* bytes for the sharded params (all-gather
+ reduce-scatter = the two halves of a ring all-reduce) because it keeps weights resident through
backward; true FSDP frees after forward and re-gathers for backward, paying 3 legs (1.5×) to cut
resident params from S to S/W. Memory is traded for one extra all-gather. (The replicated 1-D params
pay a DDP-style all-reduce and stay full — negligible bytes, no memory saved, by design.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn


@dataclass
class _Shard:
    """One sharded (``ndim >= 2``) parameter. The Parameter object's identity never changes — only
    its ``.data`` swaps between ``master`` (the flat shard, between steps) and a gathered full copy
    (fwd/bwd)."""

    param: nn.Parameter
    name: str
    shape: torch.Size
    numel: int
    master: Tensor  # this rank's flat master slice (fp32) — the only persistent copy


@dataclass
class _Replica:
    """One replicated (``ndim <= 1``) parameter. The full fp32 weight lives on every rank as
    ``master``; between steps ``param.data is master``. In fwd it swaps to a ``compute_dtype`` copy;
    at grad sync its gradient is all-reduced (mean) so it is identical across ranks."""

    param: nn.Parameter
    name: str
    master: Tensor  # the full fp32 weight (== param.data between steps)


class FSDP(nn.Module):
    """Wrap a module for fully-sharded data parallelism with the replicate-small-params policy
    (matrices sharded, 1-D norms/biases replicated). ``compute_dtype`` selects the dtype of the
    transient compute copies (masters and grad reduction stay fp32)."""

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        # All ranks must start from identical values — broadcast rank 0's weights and buffers first.
        for p in module.parameters():
            dist.broadcast(p.data, src=0)
        for b in module.buffers():
            dist.broadcast(b.data, src=0)
        self._shards: list[_Shard] = []
        self._replicas: list[_Replica] = []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue  # frozen params stay replicated; gather_full_params returns them as-is
            if p.ndim >= 2:
                # SHARDED: flatten, pad to a multiple of world_size, keep only this rank's slice.
                shape, numel = p.shape, p.numel()
                shard_len = -(-numel // self.world_size)  # ceil ⇒ pad to a multiple of world_size
                flat = torch.zeros(shard_len * self.world_size, dtype=p.dtype, device=p.device)
                flat[:numel] = p.detach().reshape(-1)
                master = flat[self.rank * shard_len : (self.rank + 1) * shard_len].clone()
                p.data = master  # from here on, this rank persistently holds only the shard
                self._shards.append(_Shard(p, name, shape, numel, master))
            else:
                # REPLICATED: every rank keeps the full fp32 weight; the param IS its own master.
                self._replicas.append(_Replica(p, name, p.data))

    def _all_gather_full(self, s: _Shard, dtype: torch.dtype) -> Tensor:
        """All-gather one sharded parameter's shards (communicated in ``dtype``) into full shape."""
        shard = s.master.to(dtype)
        parts = [torch.empty_like(shard) for _ in range(self.world_size)]
        dist.all_gather(parts, shard)
        return torch.cat(parts)[: s.numel].view(s.shape)

    def forward(self, *args: object, **kwargs: object) -> object:
        """Materialize full compute copies — all-gather each shard, cast each replica — then run the
        module. The copies stay resident through backward; ``finish_gradient_synchronization`` frees
        them. Casting a replica to its own dtype (``compute_dtype is None``) is a no-op returning the
        master, so no extra copy is made in the fp32 path."""
        dtype = self.compute_dtype
        for s in self._shards:
            s.param.data = self._all_gather_full(s, dtype or s.master.dtype)
        for r in self._replicas:
            r.param.data = r.master.to(dtype or r.master.dtype)
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Call after ``loss.backward()``, before ``optimizer.step()``. For each sharded param:
        reduce-scatter the full gradient (SUM, then /world_size ⇒ mean) so this rank holds exactly
        its shard's gradient. For each replicated param: all-reduce the full gradient (SUM, then
        /world_size ⇒ mean) so it is identical on every rank. Both land in fp32 master dtype and
        shape-match their ``.data``; the transient compute copies are dropped here."""
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
        for r in self._replicas:
            p = r.param
            if p.grad is not None:
                g = p.grad.detach().to(r.master.dtype)
            else:
                g = torch.zeros_like(r.master)
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g /= self.world_size
            p.grad = None
            p.data = r.master  # restore fp32 master (was a compute copy during fwd)
            p.grad = g

    def gather_full_params(self) -> dict[str, Tensor]:
        """All-gather every shard into full (unsharded) tensors, in master dtype, without disturbing
        the module's sharded state. Replicated and frozen params are returned as-is (already full)."""
        out = {s.name: self._all_gather_full(s, s.master.dtype) for s in self._shards}
        for name, p in self.module.named_parameters():
            if name not in out:
                out[name] = p.data
        return out

    def gather_full_grads(self) -> dict[str, Tensor]:
        """Reassemble each parameter's full gradient. Sharded grads are all-gathered from the per-
        rank shard gradients; replicated grads are already full (and identical across ranks). Valid
        only after ``finish_gradient_synchronization`` (sharded grads must be shard-shaped)."""
        out: dict[str, Tensor] = {}
        for s in self._shards:
            g = s.param.grad
            if g is None or g.numel() != s.master.numel():
                raise RuntimeError(f"{s.name}: gradient is not a synchronized shard")
            parts = [torch.empty_like(g) for _ in range(self.world_size)]
            dist.all_gather(parts, g)
            out[s.name] = torch.cat(parts)[: s.numel].view(s.shape)
        for r in self._replicas:
            g = r.param.grad
            if g is None:
                raise RuntimeError(f"{r.name}: gradient is not synchronized")
            out[r.name] = g
        return out

    def resident_param_numel(self) -> int:
        """Elements this container keeps materialized right now (masters + whatever each param's
        ``.data`` currently points at), deduplicated by storage pointer. Between steps this is
        ≈ (sharded)/world_size + (replicated) full; during fwd/bwd it is full + sharded shards."""
        seen: set[int] = set()
        total = 0
        tensors = [p.data for p in self.module.parameters()]
        tensors += [s.master for s in self._shards]
        tensors += [r.master for r in self._replicas]
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
        tensors += [r.master for r in self._replicas]
        for t in tensors:
            key = t.data_ptr()
            if key not in seen:
                seen.add(key)
                total += t.numel() * t.element_size()
        return total
