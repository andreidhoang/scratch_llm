"""Megatron tensor-parallel MLP equivalence, gloo on CPU (2 and 4 ranks).

The TP MLP splits one FFN across ranks — column-parallel GEMM-1, GeLU on the local shard, row-
parallel GEMM-2, one all-reduce. The falsifiable claims, in executable form:

1. **Forward numerics** — the sharded TP output is ``rtol 1e-5`` identical to the single-GPU MLP on
   the *same* full weights (we shard the reference A/B onto the ranks).
2. **Exactly one all-reduce in forward** — counted via an all-reduce wrapper. Backward adds exactly
   one more (the ``f`` conjugate), so f and g each cost one collective, in opposite passes.
3. **Backward numerics** — every sharded weight/bias gradient equals the corresponding shard of the
   single-GPU gradient, and the input gradient (which the ``f`` all-reduce reconstructs) matches.
"""

from __future__ import annotations

import os
import socket
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scratch_llm.utils import tp_mlp
from scratch_llm.utils.tp_mlp import ReferenceMLP, TensorParallelMLP

D_MODEL = 8
D_FF = 16  # divisible by 2 and 4
BATCH = 5


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _build(rank: int, world_size: int) -> tuple[ReferenceMLP, TensorParallelMLP]:
    """Same seed on every rank ⇒ identical reference weights; shard them onto the TP MLP."""
    torch.manual_seed(0)
    ref = ReferenceMLP(D_MODEL, D_FF)
    tp = TensorParallelMLP(D_MODEL, D_FF, rank=rank, world_size=world_size)
    tp.load_reference(ref.fc1, ref.fc2)
    return ref, tp


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        ref, tp = _build(rank, world_size)

        torch.manual_seed(
            123
        )  # identical input on every rank (X is replicated across the TP group)
        x = torch.randn(BATCH, D_MODEL)

        # --- forward: exactly ONE all-reduce, output identical to the single-GPU MLP ---
        x_ref = x.clone().requires_grad_(True)
        x_tp = x.clone().requires_grad_(True)

        real_all_reduce = dist.all_reduce
        counter = {"n": 0}

        def counting_all_reduce(*args: object, **kwargs: object) -> object:
            counter["n"] += 1
            return real_all_reduce(*args, **kwargs)  # type: ignore[arg-type]

        with mock.patch.object(tp_mlp.dist, "all_reduce", counting_all_reduce):
            out_tp = tp(x_tp)
            assert counter["n"] == 1, f"forward issued {counter['n']} all-reduces, expected 1"

            out_ref = ref(x_ref)
            assert torch.allclose(out_tp, out_ref, rtol=1e-5, atol=1e-6), "TP forward drifted"

            # --- backward: the f conjugate adds exactly one more all-reduce ---
            grad_out = torch.randn(BATCH, D_MODEL)
            out_ref.backward(grad_out)
            out_tp.backward(grad_out)
            assert counter["n"] == 2, f"fwd+bwd issued {counter['n']} all-reduces, expected 2"

        # input gradient: f's backward all-reduce must reconstruct the single-GPU ∂L/∂X
        assert x_tp.grad is not None and x_ref.grad is not None
        assert torch.allclose(x_tp.grad, x_ref.grad, rtol=1e-5, atol=1e-6), "input grad drifted"

        # weight/bias gradients: each shard equals the matching slice of the single-GPU gradient
        per_ff = D_FF // world_size
        c0, c1 = rank * per_ff, (rank + 1) * per_ff
        assert ref.fc1.weight.grad is not None and ref.fc2.weight.grad is not None
        assert ref.fc1.bias.grad is not None and ref.fc2.bias.grad is not None
        assert tp.fc1.weight.grad is not None and tp.fc2.weight.grad is not None
        assert tp.fc1.bias is not None and tp.fc1.bias.grad is not None
        assert tp.fc2.bias is not None and tp.fc2.bias.grad is not None

        assert torch.allclose(tp.fc1.weight.grad, ref.fc1.weight.grad[c0:c1], rtol=1e-5, atol=1e-6)
        assert torch.allclose(tp.fc1.bias.grad, ref.fc1.bias.grad[c0:c1], rtol=1e-5, atol=1e-6)
        assert torch.allclose(
            tp.fc2.weight.grad, ref.fc2.weight.grad[:, c0:c1], rtol=1e-5, atol=1e-6
        )
        # fc2 bias is replicated (added once after the reduce) ⇒ full grad on every rank
        assert torch.allclose(tp.fc2.bias.grad, ref.fc2.bias.grad, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_tp_mlp_matches_single_gpu(world_size: int) -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _worker, args=(world_size, _free_port()), nprocs=world_size, join=True
    )
