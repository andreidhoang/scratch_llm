"""FSDP (ZeRO-3) equivalence: 2-rank per-parameter-sharded training must match unsharded
single-process full-batch training — loss, reassembled gradients, and the parameter trajectory —
over ≥3 AdamW steps × 3 seeds, on a plain MLP and a small TransformerLM. Pure CPU via gloo +
mp.spawn.

Also asserted: reduce-scattered shard grads reassemble to the reference full gradient, and the
container's resident parameter footprint between steps is ≈ full/world_size (numel accounting) —
the ZeRO-3 memory claim in executable form."""

from __future__ import annotations

import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy
from scratch_llm.optim import AdamW
from scratch_llm.utils.fsdp import FSDP
from scratch_llm.utils.seeding import seed_everything

MASTER_PORT = 29513  # unique to this test file so parallel node runs never collide
WORLD_SIZE = 2
SEEDS = (0, 1, 2)
STEPS = 3


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Replicate-small-params policy: 2-D weights are sharded, 1-D biases replicated. The
        # second Linear's weight has numel 5*15 = 75 (odd) so the pad-to-world_size *shard* path
        # is exercised at W=2 (the 1-D biases no longer are — they are kept full and all-reduced).
        self.net = nn.Sequential(nn.Linear(8, 15), nn.ReLU(), nn.Linear(15, 5))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2, context_length=8)


def _init_pg(rank: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)


def _batch_and_loss(kind: str) -> tuple[Tensor, Tensor]:
    """Full-batch inputs/targets for ``kind`` (identical on every rank — RNG is pre-seeded)."""
    if kind == "mlp":
        return torch.randn(2 * WORLD_SIZE, 8), torch.randn(2 * WORLD_SIZE, 5)
    ids = torch.randint(0, 32, (2 * WORLD_SIZE, 8))
    targets = torch.randint(0, 32, (2 * WORLD_SIZE, 8))
    return ids, targets


def _loss(kind: str, model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    out = model(x)
    assert isinstance(out, Tensor)
    return F.mse_loss(out, y) if kind == "mlp" else cross_entropy(out, y)


def _build(kind: str) -> nn.Module:
    return _TinyMLP() if kind == "mlp" else TransformerLM(_tiny_cfg())


def _equivalence_worker(rank: int, kind: str) -> None:
    _init_pg(rank)
    try:
        for seed in SEEDS:
            seed_everything(seed)
            ref = _build(kind)
            fsdp = FSDP(copy.deepcopy(ref))
            opt_ref = AdamW(ref.parameters(), lr=1e-3)
            opt_fsdp = AdamW(fsdp.parameters(), lr=1e-3)

            x, y = _batch_and_loss(kind)
            local = slice(rank * 2, (rank + 1) * 2)

            for step in range(STEPS):
                opt_ref.zero_grad(set_to_none=True)
                ref_loss = _loss(kind, ref, x, y)
                ref_loss.backward()

                opt_fsdp.zero_grad(set_to_none=True)
                local_loss = _loss(kind, fsdp, x[local], y[local])
                local_loss.backward()
                fsdp.finish_gradient_synchronization()

                # Loss: mean of equal-sized per-rank means == the full-batch mean.
                avg_loss = local_loss.detach().clone()
                dist.all_reduce(avg_loss)
                avg_loss /= WORLD_SIZE
                assert torch.allclose(avg_loss, ref_loss.detach(), atol=1e-6), (
                    f"{kind} seed {seed} step {step}: loss drifted"
                )

                # Gradients: shards reassemble to the reference full gradient.
                full_grads = fsdp.gather_full_grads()
                for name, p in ref.named_parameters():
                    assert p.grad is not None
                    assert torch.allclose(full_grads[name], p.grad, atol=1e-6, rtol=1e-5), (
                        f"{kind} seed {seed} step {step}: grad {name} drifted"
                    )

                opt_ref.step()
                opt_fsdp.step()

                # Parameter trajectory: sharded AdamW == full-batch AdamW after every step.
                full_params = fsdp.gather_full_params()
                for name, p in ref.named_parameters():
                    assert torch.allclose(full_params[name], p.data, atol=1e-5, rtol=1e-5), (
                        f"{kind} seed {seed} step {step}: param {name} drifted"
                    )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_fsdp_matches_single_process(kind: str) -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _equivalence_worker, args=(kind,), nprocs=WORLD_SIZE, join=True
    )


def _gradient_sync_worker(rank: int) -> None:
    _init_pg(rank)
    try:
        seed_everything(7)
        ref = _TinyMLP()
        fsdp = FSDP(copy.deepcopy(ref))
        x, y = _batch_and_loss("mlp")
        local = slice(rank * 2, (rank + 1) * 2)

        _loss("mlp", ref, x, y).backward()
        _loss("mlp", fsdp, x[local], y[local]).backward()
        fsdp.finish_gradient_synchronization()

        # Each rank ends with exactly its shard's gradient: shard-shaped, master-dtype.
        for p in fsdp.module.parameters():
            assert p.grad is not None
            assert p.grad.shape == p.data.shape
            assert p.grad.dtype == p.data.dtype

        full_grads = fsdp.gather_full_grads()
        for name, p in ref.named_parameters():
            assert p.grad is not None
            assert full_grads[name].shape == p.grad.shape
            assert torch.allclose(full_grads[name], p.grad, atol=1e-6, rtol=1e-5), (
                f"reassembled grad {name} != reference full grad"
            )
    finally:
        dist.destroy_process_group()


def test_fsdp_gradient_sync() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _gradient_sync_worker, args=(), nprocs=WORLD_SIZE, join=True
    )


def _memory_worker(rank: int) -> None:
    _init_pg(rank)
    try:
        seed_everything(11)
        ref = TransformerLM(_tiny_cfg())
        full_numel = sum(p.numel() for p in ref.parameters())
        # Replicate-small-params policy: matrices (ndim>=2) sharded, 1-D norms/biases replicated.
        sharded = [p for p in ref.parameters() if p.ndim >= 2]
        sharded_full = sum(p.numel() for p in sharded)
        replicated_full = sum(p.numel() for p in ref.parameters() if p.ndim < 2)
        shard_only = sum(-(-p.numel() // WORLD_SIZE) for p in sharded)  # ceil per sharded tensor
        # Between steps: sharded matrices reduced to ≈ 1/W, small 1-D params kept full.
        expected_resident = shard_only + replicated_full

        fsdp = FSDP(copy.deepcopy(ref))
        resident = fsdp.resident_param_numel()
        assert resident == expected_resident
        assert resident < full_numel
        # The sharded matrices dominate memory and are genuinely ≈ 1/W (< 1 pad elem / tensor).
        assert shard_only <= sharded_full // WORLD_SIZE + len(sharded)
        assert fsdp.resident_param_bytes() == resident * 4  # fp32 masters + replicas

        opt = AdamW(fsdp.parameters(), lr=1e-3)
        x, y = _batch_and_loss("transformer")
        local = slice(rank * 2, (rank + 1) * 2)

        loss = _loss("transformer", fsdp, x[local], y[local])
        # During fwd/bwd the full compute copies of the *sharded* params coexist with the masters;
        # replicated params are already full (no extra copy) ⇒ full model + sharded shards.
        assert fsdp.resident_param_numel() == full_numel + shard_only
        loss.backward()
        fsdp.finish_gradient_synchronization()
        # Full copies are freed at sync: back to the between-steps footprint, and it stays there.
        assert fsdp.resident_param_numel() == expected_resident
        opt.step()
        assert fsdp.resident_param_numel() == expected_resident
    finally:
        dist.destroy_process_group()


def test_fsdp_resident_param_bytes() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _memory_worker, args=(), nprocs=WORLD_SIZE, join=True
    )


def _compute_dtype_worker(rank: int) -> None:
    _init_pg(rank)
    try:
        seed_everything(3)
        ref = _TinyMLP()
        fsdp = FSDP(copy.deepcopy(ref), compute_dtype=torch.bfloat16)
        x, y = _batch_and_loss("mlp")
        local = slice(rank * 2, (rank + 1) * 2)

        opt_ref = torch.optim.SGD(ref.parameters(), lr=0.1)
        opt_fsdp = torch.optim.SGD(fsdp.parameters(), lr=0.1)

        F.mse_loss(ref(x), y).backward()
        opt_ref.step()

        out = fsdp(x[local].to(torch.bfloat16))
        assert isinstance(out, Tensor)
        assert out.dtype == torch.bfloat16  # compute ran in the toggled dtype
        F.mse_loss(out, y[local].to(torch.bfloat16)).backward()
        fsdp.finish_gradient_synchronization()
        # Masters and reduced grads stay fp32 regardless of compute dtype.
        for p in fsdp.module.parameters():
            assert p.data.dtype == torch.float32
            assert p.grad is not None and p.grad.dtype == torch.float32
        opt_fsdp.step()

        full_params = fsdp.gather_full_params()
        for name, p in ref.named_parameters():
            assert full_params[name].dtype == torch.float32
            assert torch.allclose(full_params[name], p.data, atol=2e-2), (
                f"bf16-compute param {name} too far from the fp32 reference step"
            )
    finally:
        dist.destroy_process_group()


def test_fsdp_compute_dtype_bf16() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _compute_dtype_worker, args=(), nprocs=WORLD_SIZE, join=True
    )
