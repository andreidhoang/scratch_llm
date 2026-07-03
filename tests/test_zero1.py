"""ZeRO-1 invariants, 2-rank gloo on CPU: sharding only the optimizer state must not change the
parameter trajectory (vs the unsharded optimizer, same grads), the greedy partition must cover
every param exactly once with per-rank state numel ≈ total/world_size, and a per-rank state-dict
round-trip must resume the exact trajectory. World size 1 must behave as a plain wrapper."""

from __future__ import annotations

import os
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import AdamW
from scratch_llm.utils.seeding import seed_everything
from scratch_llm.utils.zero1 import ShardedOptimizer, optimizer_state_numel, partition_by_numel

MASTER_PORT = 29511


def _ref_opt(model: nn.Module) -> AdamW:
    return AdamW(model.parameters(), lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)


def _sharded_opt(model: nn.Module) -> ShardedOptimizer:
    return ShardedOptimizer(
        model.parameters(), AdamW, lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _init_pg(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _lm_logits(model: TransformerLM, tokens: Tensor) -> Tensor:
    out = model(tokens)
    assert isinstance(out, Tensor)
    return out


def test_partition_covers_every_param_once_and_balances() -> None:
    seed_everything(0)
    for world_size in (1, 2, 3, 4):
        for _ in range(5):
            numels = [int(n) for n in torch.randint(1, 500, (12,))]
            owners = partition_by_numel(numels, world_size)
            assert len(owners) == len(numels)  # every param owned exactly once
            assert all(0 <= r < world_size for r in owners)
            loads = [0] * world_size
            for n, r in zip(numels, owners, strict=True):
                loads[r] += n
            assert max(loads) - min(loads) <= max(numels)  # the greedy balance bound


def _mlp_equivalence_worker(rank: int, world_size: int, seed: int) -> None:
    _init_pg(rank, world_size)
    try:
        seed_everything(seed)
        ref = _MLP()
        sharded_model = deepcopy(ref)
        opt_ref = _ref_opt(ref)
        opt_sharded = _sharded_opt(sharded_model)

        for step in range(4):
            x = torch.randn(8, 6)
            y = torch.randn(8, 3)

            opt_ref.zero_grad()
            F.mse_loss(ref(x), y).backward()
            opt_ref.step()

            opt_sharded.zero_grad()
            F.mse_loss(sharded_model(x), y).backward()
            opt_sharded.step()

            for pr, ps in zip(ref.parameters(), sharded_model.parameters(), strict=True):
                assert torch.allclose(pr, ps, atol=1e-6), f"seed {seed} drifted at step {step}"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sharded_matches_unsharded_mlp(seed: int) -> None:
    world_size = 2
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _mlp_equivalence_worker, args=(world_size, seed), nprocs=world_size, join=True
    )


def _transformer_worker(rank: int, world_size: int) -> None:
    _init_pg(rank, world_size)
    try:
        seed_everything(0)
        cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2, context_length=16)
        ref = TransformerLM(cfg)
        sharded_model = deepcopy(ref)
        opt_ref = _ref_opt(ref)
        opt_sharded = _sharded_opt(sharded_model)

        all_params = list(sharded_model.parameters())
        shard_ids = [{id(p) for p in opt_sharded.params_for_rank(r)} for r in range(world_size)]
        assert set().union(*shard_ids) == {id(p) for p in all_params}  # shards cover all params
        for r in range(world_size):
            for s in range(r + 1, world_size):
                assert not shard_ids[r] & shard_ids[s]  # shards are disjoint

        for step in range(3):
            tokens = torch.randint(0, cfg.vocab_size, (2, 8))
            targets = torch.randint(0, cfg.vocab_size, (2, 8))

            opt_ref.zero_grad()
            logits = _lm_logits(ref, tokens)
            F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
            opt_ref.step()

            opt_sharded.zero_grad()
            logits = _lm_logits(sharded_model, tokens)
            F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
            opt_sharded.step()

            for pr, ps in zip(ref.parameters(), sharded_model.parameters(), strict=True):
                assert torch.allclose(pr, ps, atol=1e-6), f"LM drifted at step {step}"

        # ZeRO-1 memory claim: per-rank state numel ≈ total/world_size, within the greedy bound
        # (2× the largest param's numel — AdamW keeps two moment tensors per param).
        rank_state = opt_sharded.state_numel_on_rank()
        total_state = optimizer_state_numel(opt_ref)
        max_param = max(p.numel() for p in all_params)
        assert abs(rank_state - total_state / world_size) <= 2 * max_param
        assert rank_state < total_state  # strictly sharded, not replicated
    finally:
        dist.destroy_process_group()


def test_transformer_lm_with_repo_adamw_2rank() -> None:
    world_size = 2
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _transformer_worker, args=(world_size,), nprocs=world_size, join=True
    )


def _state_dict_worker(rank: int, world_size: int) -> None:
    _init_pg(rank, world_size)
    try:
        seed_everything(7)
        batches = [(torch.randn(8, 6), torch.randn(8, 3)) for _ in range(5)]
        model = _MLP()
        opt = _sharded_opt(model)

        def train(m: nn.Module, o: ShardedOptimizer, steps: slice) -> None:
            for x, y in batches[steps]:
                o.zero_grad()
                F.mse_loss(m(x), y).backward()
                o.step()

        train(model, opt, slice(0, 3))
        saved_params = deepcopy(model.state_dict())
        saved_opt = deepcopy(opt.state_dict())  # this rank's shard only
        train(model, opt, slice(3, 5))
        final = [p.detach().clone() for p in model.parameters()]

        restored = _MLP()
        restored.load_state_dict(saved_params)
        opt2 = _sharded_opt(restored)
        opt2.load_state_dict(saved_opt)
        train(restored, opt2, slice(3, 5))

        for pf, pr in zip(final, restored.parameters(), strict=True):
            assert torch.allclose(pf, pr, atol=1e-7), "state-dict round-trip changed the trajectory"
    finally:
        dist.destroy_process_group()


def test_state_dict_round_trip_2rank() -> None:
    world_size = 2
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _state_dict_worker, args=(world_size,), nprocs=world_size, join=True
    )


def test_world_size_one_is_plain_wrapper() -> None:
    seed_everything(3)
    ref = _MLP()
    wrapped_model = deepcopy(ref)
    opt_ref = _ref_opt(ref)
    opt_wrapped = _sharded_opt(wrapped_model)

    assert opt_wrapped.world_size == 1
    assert len(opt_wrapped.params_for_rank(0)) == len(list(wrapped_model.parameters()))

    for _ in range(3):
        x = torch.randn(8, 6)
        y = torch.randn(8, 3)
        opt_ref.zero_grad()
        F.mse_loss(ref(x), y).backward()
        opt_ref.step()
        opt_wrapped.zero_grad()
        F.mse_loss(wrapped_model(x), y).backward()
        opt_wrapped.step()

    for pr, pw in zip(ref.parameters(), wrapped_model.parameters(), strict=True):
        assert torch.equal(pr, pw)  # world size 1: bitwise-identical to the plain optimizer
    assert opt_wrapped.state_numel_on_rank() == optimizer_state_numel(opt_ref)
