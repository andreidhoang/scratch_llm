"""Megatron tensor-parallel MLP — split one big FFN across ranks with exactly one all-reduce.

A6 systems, the core tensor-parallel (TP) primitive. Data-parallel (DDP/ZeRO/FSDP) shards the
*batch* or the *optimizer/parameter storage* but every rank still runs the whole layer; tensor
parallel instead shards a *single* layer's matmul across ranks so a matrix too big for one GPU is
computed cooperatively. The Megatron MLP is the canonical construction:

    Y = GeLU(X @ A) @ B                     (single-GPU reference; A: d_model×d_ff, B: d_ff×d_model)

Two conjugate ways to split a GEMM, chosen so the two halves compose with **no comm between them**:

- **Column-parallel GEMM-1** (``ColumnParallelLinear``): shard ``A`` along its *output* columns —
  rank r holds ``A[:, r·d_ff/W : (r+1)·d_ff/W]`` and computes ``X @ A_r`` → a ``d_ff/W``-wide slice
  of the pre-activation. GeLU is elementwise, so ``GeLU`` on rank r's slice equals that slice of the
  full ``GeLU(X @ A)`` — **no communication**, the activation is computed on the local shard.
- **Row-parallel GEMM-2** (``RowParallelLinear``): shard ``B`` along its *input* rows — rank r holds
  ``B[r·d_ff/W : (r+1)·d_ff/W, :]`` and multiplies its GeLU slice by ``B_r`` to get a **partial**
  ``d_model`` output. The full output is the sum of the partials: ``Σ_r (GeLU_r @ B_r)``. One
  **all-reduce (SUM)** collapses the partials into the identical full output on every rank. The
  (unsharded) output bias is added *after* the reduce, once.

The two conjugate communication ops — placed so autograd is symmetric between the fwd and bwd pass:

- ``f`` (``_CopyToModelParallel``): entry to the parallel region. **fwd = identity** (X is already
  replicated), **bwd = all-reduce**. In backward each rank produced a gradient w.r.t. the *shared*
  input X (through its own ``A_r``); those must be summed to reconstruct the true ``∂L/∂X``.
- ``g`` (``_ReduceFromModelParallel``): exit of the region. **fwd = all-reduce** (sum the partials),
  **bwd = identity** (the full output's gradient is broadcast unchanged to every rank's B_r branch).

So a forward pass through the MLP does **exactly one** collective — the ``g`` all-reduce — and a
backward pass does exactly one — the ``f`` all-reduce. f and g are conjugates: (identity, all-reduce)
and (all-reduce, identity).

Falsifiable invariant (``tests/test_tp_mlp.py``, gloo, 2 and 4 ranks): the sharded TP MLP output is
numerically identical (``rtol 1e-5``) to the single-GPU MLP built on the *same* full weights (we
shard the reference A/B to the ranks), and the forward issues **exactly one** all-reduce (counted via
a wrapped process group). Kill: any drift ⇒ a wrong shard axis (column vs row), the output bias added
in the parallel region instead of after the reduce, or a missing/extra collective.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function


class _CopyToModelParallel(Function):
    """``f``: identity forward, all-reduce (SUM) backward — the entry to the parallel region."""

    @staticmethod
    def forward(ctx: object, x: Tensor, group: object) -> Tensor:  # type: ignore[override]
        ctx.group = group  # type: ignore[attr-defined]
        return x

    @staticmethod
    def backward(ctx: object, grad: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        grad = grad.clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)  # type: ignore[attr-defined]
        return grad, None


class _ReduceFromModelParallel(Function):
    """``g``: all-reduce (SUM) forward, identity backward — the exit of the parallel region."""

    @staticmethod
    def forward(ctx: object, x: Tensor, group: object) -> Tensor:  # type: ignore[override]
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx: object, grad: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return grad, None


def copy_to_model_parallel(x: Tensor, group: object = None) -> Tensor:
    """``f`` — replicate on the way in, sum gradients on the way back."""
    return _CopyToModelParallel.apply(x, group)  # type: ignore[no-any-return]


def reduce_from_model_parallel(x: Tensor, group: object = None) -> Tensor:
    """``g`` — sum partials on the way out, pass gradients through unchanged."""
    return _ReduceFromModelParallel.apply(x, group)  # type: ignore[no-any-return]


def _shard_size(total: int, world_size: int, rank: int) -> tuple[int, int]:
    """Contiguous even split of ``total`` across ``world_size`` — returns this rank's [start, end)."""
    assert total % world_size == 0, f"dim {total} not divisible by world_size {world_size}"
    per = total // world_size
    return rank * per, (rank + 1) * per


class ColumnParallelLinear(nn.Module):
    """GEMM-1: ``y = f(x) @ A_r (+ bias_r)`` where ``A`` is split along its output (column) dim.

    Each rank owns ``out_features/world_size`` output columns and computes them from the *full*
    input, so the outputs concatenated across ranks equal the single-GPU linear's output. ``f`` (the
    identity-fwd/all-reduce-bwd op) wraps the input so gradients w.r.t. the shared input are summed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        rank: int,
        world_size: int,
        group: object = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.group = group
        start, end = _shard_size(out_features, world_size, rank)
        self.local_out = end - start
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.bias = nn.Parameter(torch.empty(self.local_out)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        x = copy_to_model_parallel(x, self.group)  # f: identity fwd, all-reduce bwd
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def load_full(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Copy this rank's column-shard out of a full ``[out_features, in_features]`` weight."""
        start, end = _shard_size(self.out_features, self.world_size, self.rank)
        self.weight.copy_(weight[start:end])
        if self.bias is not None:
            assert bias is not None
            self.bias.copy_(bias[start:end])


class RowParallelLinear(nn.Module):
    """GEMM-2: ``y = g(x @ B_r) (+ bias)`` where ``B`` is split along its input (row) dim.

    The input is already sharded (it is column-parallel GEMM-1's output), so each rank multiplies its
    slice by ``B_r`` to get a *partial* full-width output; ``g`` all-reduces the partials into the
    identical full output. The (unsharded) output bias is added **after** the reduce, exactly once.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        rank: int,
        world_size: int,
        group: object = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.group = group
        start, end = _shard_size(in_features, world_size, rank)
        self.local_in = end - start
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        # Bias is NOT sharded — it is added once, after the all-reduce, on the full output.
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        partial = F.linear(x, self.weight)  # per-rank partial, no bias yet
        out = reduce_from_model_parallel(partial, self.group)  # g: all-reduce fwd, identity bwd
        if self.bias is not None:
            out = out + self.bias
        return out

    @torch.no_grad()
    def load_full(self, weight: Tensor, bias: Tensor | None = None) -> None:
        """Copy this rank's row-shard out of a full ``[out_features, in_features]`` weight.

        ``bias`` is the full (unsharded) output bias — every rank stores the same copy; it is applied
        once after the reduce, so replicating it does not double-count.
        """
        start, end = _shard_size(self.in_features, self.world_size, self.rank)
        self.weight.copy_(weight[:, start:end])
        if self.bias is not None:
            assert bias is not None
            self.bias.copy_(bias)


class TensorParallelMLP(nn.Module):
    """The Megatron MLP: column-parallel GEMM-1 → GeLU → row-parallel GEMM-2, one all-reduce fwd.

    Built on the same ``(d_model, d_ff)`` shapes as the single-GPU reference ``GeLU(x @ A) @ B``.
    Use :meth:`load_reference` to shard a plain two-``nn.Linear`` reference onto this rank.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        bias: bool = True,
        rank: int,
        world_size: int,
        group: object = None,
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            d_model, d_ff, bias=bias, rank=rank, world_size=world_size, group=group
        )
        self.fc2 = RowParallelLinear(
            d_ff, d_model, bias=bias, rank=rank, world_size=world_size, group=group
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    @torch.no_grad()
    def load_reference(self, fc1: nn.Linear, fc2: nn.Linear) -> None:
        """Shard a reference ``fc1: Linear(d_model, d_ff)`` / ``fc2: Linear(d_ff, d_model)`` here."""
        self.fc1.load_full(fc1.weight, fc1.bias)
        self.fc2.load_full(fc2.weight, fc2.bias)


class ReferenceMLP(nn.Module):
    """Single-GPU oracle: ``GeLU(x @ A) @ B`` as two plain ``nn.Linear`` layers."""

    def __init__(self, d_model: int, d_ff: int, *, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))
