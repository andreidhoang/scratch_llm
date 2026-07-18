"""A7 — DistMuonAdamW (optimizer-embedded ZeRO-2) invariants, 2-rank gloo on CPU: one
distributed step from identical replicas must equal one single-process CombinedOptimizer step
on the averaged grads (matrix-granular Muon shards, row-sharded AdamW, collective grad clip —
each padding edge gets a direct test); replicas must stay BITWISE synced across steps on
different per-rank batches; and the wired train() must draw different batches per rank, raise
collectively on a one-rank NaN (no hang), and leave the single-process path untouched
(tests/test_train.py runs unmodified as the byte-identity witness)."""

from __future__ import annotations

import os
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import (
    AdamW,
    Muon,
    build_optimizer,
    gradient_clipping,
    split_muon_adamw_params,
)
from scratch_llm.train import TrainConfig, train
from scratch_llm.utils.dist_train import DistMuonAdamW
from scratch_llm.utils.seeding import seed_everything

WORLD = 2


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _init_pg(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _model_cfg() -> ModelConfig:
    # vocab 64 × d_model 32 covers all three shard classes in one tiny model: embed/lm_head are
    # 2048-elem (row-sharded AdamW), norm gains are 32-elem (replicated small AdamW), and the
    # block matrices form Muon shape-groups K=8 (32,32) / K=4 (128,32) / K=2 (32,128).
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, context_length=16)


def _exact_grad_pair(p: Tensor, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    """(base, delta) drawn as small multiples of 0.125/0.25, so ``base + rank·delta`` AND the
    2-rank average ``base + delta/2`` are exactly representable in fp32: the distributed AVG is
    then bitwise equal to the hand-computed average, and the oracle comparison isolates the
    sharding logic from communication rounding."""
    base = torch.randint(-8, 9, p.shape, generator=gen).float() * 0.125
    delta = torch.randint(-8, 9, p.shape, generator=gen).float() * 0.25
    return base, delta


def _assert_replicas_bitwise_equal(model: nn.Module, world_size: int) -> None:
    for name, p in model.named_parameters():
        gathered = [torch.empty_like(p) for _ in range(world_size)]
        dist.all_gather(gathered, p.data.contiguous())
        assert torch.equal(gathered[0], gathered[1]), f"replicas diverged at {name}"


# --------------------------------------------------------------------------------------------
# (a) optimizer-equivalence oracle: one distributed step == one CombinedOptimizer step on the
#     averaged grads
# --------------------------------------------------------------------------------------------


def _oracle_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        seed_everything(0)
        model = TransformerLM(_model_cfg())
        seed_everything(0)
        ref = TransformerLM(_model_cfg())  # identical weights on every rank and in the oracle

        muon_p, adamw_p = split_muon_adamw_params(model)
        opt = DistMuonAdamW(
            muon_p, adamw_p, lr=0.02, betas=(0.9, 0.95), weight_decay=0.1, muon_momentum=0.95
        )
        ref_opt = build_optimizer(
            ref, kind="muon_adamw", lr=0.02, betas=(0.9, 0.95), weight_decay=0.1, muon_momentum=0.95
        )

        gen = torch.Generator().manual_seed(99)  # same stream on every rank
        for _ in range(2):  # 2 steps: momentum buffers / Adam moments persist across steps
            for p, pr in zip(model.parameters(), ref.parameters(), strict=True):
                base, delta = _exact_grad_pair(p, gen)
                p.grad = base + rank * delta  # per-rank grads whose average is known exactly
                pr.grad = base + delta / 2
            opt.step()
            ref_opt.step()

        for (name, p), (_, pr) in zip(
            model.named_parameters(), ref.named_parameters(), strict=True
        ):
            torch.testing.assert_close(p, pr, atol=1e-6, rtol=1e-6, msg=f"{name} drifted")
    finally:
        dist.destroy_process_group()


def test_one_distributed_step_equals_combined_optimizer_on_averaged_grads() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _oracle_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True
    )


# --------------------------------------------------------------------------------------------
# (e) grad-clip equivalence: distributed global norm == single-process norm on the same
#     averaged grads, and the clipped trajectories match
# --------------------------------------------------------------------------------------------


def _clip_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        seed_everything(0)
        model = TransformerLM(_model_cfg())
        seed_everything(0)
        ref = TransformerLM(_model_cfg())

        muon_p, adamw_p = split_muon_adamw_params(model)
        opt = DistMuonAdamW(muon_p, adamw_p, lr=0.02, weight_decay=0.1, max_l2_norm=1.0)
        ref_opt = build_optimizer(ref, kind="muon_adamw", lr=0.02, weight_decay=0.1)

        gen = torch.Generator().manual_seed(7)
        for _ in range(2):
            for p, pr in zip(model.parameters(), ref.parameters(), strict=True):
                base, delta = _exact_grad_pair(p, gen)
                p.grad = base + rank * delta
                pr.grad = base + delta / 2
            ref_norm = gradient_clipping(ref.parameters(), 1.0)  # well above 1.0 by design
            ref_opt.step()
            opt.step()
            assert opt.last_grad_norm is not None
            # Same norm up to summation order (shard partial sums vs per-tensor sums).
            torch.testing.assert_close(
                torch.tensor(opt.last_grad_norm), ref_norm, atol=1e-5, rtol=1e-5
            )

        # The clip scale differs in its last ulp between the two summation orders; through the
        # bf16 Newton–Schulz that can flip a rounding here and there, so the trajectory match
        # is tight-but-not-bitwise (the unclipped oracle above pins the exact path).
        for (name, p), (_, pr) in zip(
            model.named_parameters(), ref.named_parameters(), strict=True
        ):
            torch.testing.assert_close(p, pr, atol=5e-4, rtol=1e-3, msg=f"{name} drifted")
    finally:
        dist.destroy_process_group()


def test_distributed_grad_clip_matches_single_process_on_averaged_grads() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _clip_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True
    )


# --------------------------------------------------------------------------------------------
# (b) replicas stay BITWISE synced across steps with different per-rank batches
# --------------------------------------------------------------------------------------------


def _sync_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        seed_everything(1)
        model = TransformerLM(_model_cfg())
        muon_p, adamw_p = split_muon_adamw_params(model)
        opt = DistMuonAdamW(muon_p, adamw_p, lr=0.02, max_l2_norm=1.0)

        torch.manual_seed(500 + rank)  # DIFFERENT batches per rank from here on
        for _ in range(3):
            tokens = torch.randint(0, 64, (2, 12))
            targets = torch.randint(0, 64, (2, 12))
            opt.zero_grad()
            logits = model(tokens)
            assert isinstance(logits, Tensor)
            F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
            opt.step()

        # Every sharded element is computed by exactly one owner and all-gathered; the
        # replicated small params see identical averaged grads — so equality is bitwise.
        _assert_replicas_bitwise_equal(model, world_size)
    finally:
        dist.destroy_process_group()


def test_params_stay_bitwise_synced_across_steps() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _sync_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True
    )


# --------------------------------------------------------------------------------------------
# padding edge cases — each gets a direct unit test against the single-process optimizer
# --------------------------------------------------------------------------------------------


def _muon_pad_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        # K=3 same-shape matrices, world 2 → k_pad=4: rank 1 owns matrix 2 plus a zero-pad slot.
        torch.manual_seed(0)
        mats = [nn.Parameter(torch.randn(4, 6)) for _ in range(3)]
        refs = [nn.Parameter(m.detach().clone()) for m in mats]
        opt = DistMuonAdamW(mats, [], lr=0.02)
        ref_opt = Muon(refs, lr=0.02)  # identical defaults: momentum .95, nesterov, ns 5, wd .1

        gen = torch.Generator().manual_seed(3)
        for _ in range(2):
            for p, pr in zip(mats, refs, strict=True):
                base, delta = _exact_grad_pair(p, gen)
                p.grad = base + rank * delta
                pr.grad = base + delta / 2
            opt.step()
            ref_opt.step()
        for i, (p, pr) in enumerate(zip(mats, refs, strict=True)):
            torch.testing.assert_close(p, pr, atol=1e-6, rtol=1e-6, msg=f"matrix {i} drifted")
    finally:
        dist.destroy_process_group()


def test_muon_group_size_not_divisible_by_world_size() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _muon_pad_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True
    )


def _adamw_pad_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        # (33, 32) = 1056 ≥ SMALL_PARAM_NUMEL → row-sharded with dim0 odd (one zero-pad row on
        # rank 1); the (10,) param takes the replicated all_reduce path.
        torch.manual_seed(1)
        big = nn.Parameter(torch.randn(33, 32))
        small = nn.Parameter(torch.randn(10))
        ref_big = nn.Parameter(big.detach().clone())
        ref_small = nn.Parameter(small.detach().clone())
        opt = DistMuonAdamW([], [big, small], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
        ref_opt = AdamW([ref_big, ref_small], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

        gen = torch.Generator().manual_seed(5)
        for _ in range(2):
            for p, pr in zip([big, small], [ref_big, ref_small], strict=True):
                base, delta = _exact_grad_pair(p, gen)
                p.grad = base + rank * delta
                pr.grad = base + delta / 2
            opt.step()
            ref_opt.step()
        torch.testing.assert_close(big, ref_big, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(small, ref_small, atol=1e-7, rtol=1e-6)
    finally:
        dist.destroy_process_group()


def test_adamw_dim0_not_divisible_by_world_size() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _adamw_pad_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True
    )


# --------------------------------------------------------------------------------------------
# (c) the wired train() draws DIFFERENT batches on each rank
# --------------------------------------------------------------------------------------------


def _different_batches_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        import scratch_llm.train as train_mod

        captured: list[Tensor] = []
        real_get_batch = train_mod.get_batch

        def spy(
            data: np.ndarray, batch_size: int, context_length: int, device: str = "cpu"
        ) -> tuple[Tensor, Tensor]:
            batch = real_get_batch(data, batch_size, context_length, device)
            captured.append(batch[0].clone())
            return batch

        train_mod.get_batch = spy
        try:
            seed_everything(3)
            model = TransformerLM(_model_cfg())
            data = np.tile(np.arange(16, dtype=np.int64), 200)
            cfg = TrainConfig(
                max_steps=2,
                batch_size=4,
                context_length=8,
                max_lr=1e-3,
                log_every=1,
                seed=3,
                optimizer="muon_adamw",
            )
            train_mod.train(cfg, data, model)
        finally:
            train_mod.get_batch = real_get_batch

        assert captured, "spy never saw a batch"
        first = captured[0]
        gathered = [torch.empty_like(first) for _ in range(world_size)]
        dist.all_gather(gathered, first)
        assert not torch.equal(gathered[0], gathered[1]), (
            "both ranks sampled the IDENTICAL first batch — per-rank data sharding is broken "
            "(the silent world_size× data loss A7 exists to prevent)"
        )
    finally:
        dist.destroy_process_group()


def test_train_draws_different_batches_per_rank() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _different_batches_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True
    )


# --------------------------------------------------------------------------------------------
# (d) collective NaN guard: a non-finite loss on ONE rank makes ALL ranks raise — no hang
# --------------------------------------------------------------------------------------------


def _nan_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        import scratch_llm.train as train_mod

        if rank == 1:  # inject the divergence on rank 1 ONLY
            real_ce = train_mod.cross_entropy

            def nan_ce(logits: Tensor, targets: Tensor) -> Tensor:
                return real_ce(logits, targets) * float("nan")

            train_mod.cross_entropy = nan_ce

        seed_everything(4)
        model = TransformerLM(_model_cfg())
        data = np.tile(np.arange(16, dtype=np.int64), 200)
        cfg = TrainConfig(
            max_steps=4,
            batch_size=4,
            context_length=8,
            max_lr=1e-3,
            log_every=1,
            seed=4,
            optimizer="muon_adamw",
        )
        try:
            train_mod.train(cfg, data, model)
        except RuntimeError as err:
            # Must be the guard's OWN message on every rank — not a gloo comm error that
            # the surviving rank would raise if the guard were NOT collective (else this
            # test would false-pass on the exact bug it exists to catch).
            assert "non-finite loss" in str(err), f"rank {rank} raised a non-guard error: {err}"
            return
        raise AssertionError(f"rank {rank} did not raise on the collective NaN guard")
    finally:
        dist.destroy_process_group()


def test_collective_nan_guard_raises_on_all_ranks_without_hang() -> None:
    ctx = mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _nan_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=False
    )
    assert ctx is not None
    # join(timeout) returns as soon as ANY rank exits (False while others live), so poll until
    # every rank is done or the deadline passes — a deadline hit means one rank raised alone
    # and left the other blocked in a collective (the pre-A7 bug this guard exists to prevent).
    deadline = time.monotonic() + 300
    while not ctx.join(timeout=5):
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                if proc is not None:
                    proc.terminate()
            raise AssertionError("ranks hung instead of raising together on the one-rank NaN")


# --------------------------------------------------------------------------------------------
# (f) end-to-end 2-rank train() smoke — the loop learns and the replicas end identical
# --------------------------------------------------------------------------------------------


def _e2e_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        data = np.tile(np.arange(16, dtype=np.int64), 400)  # learnable cycle, as in test_train
        seed_everything(11)
        model = TransformerLM(
            ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, context_length=16)
        )
        cfg = TrainConfig(
            max_steps=60,
            batch_size=8,
            context_length=12,
            max_lr=3e-3,
            warmup_steps=5,
            seed=11,
            optimizer="muon_adamw",
            log_every=10,
        )
        history = train(cfg, data, model)
        first_loss, last_loss = history[0][1], history[-1][1]
        assert last_loss < first_loss - 0.5, (
            f"2-rank distributed train() did not learn: {first_loss:.3f} → {last_loss:.3f}"
        )
        _assert_replicas_bitwise_equal(model, world_size)
    finally:
        dist.destroy_process_group()


def test_two_rank_train_smoke_loss_decreases() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _e2e_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True
    )


# --------------------------------------------------------------------------------------------
# world size 1 (no process group): the wrapper degenerates to the plain single-process math —
# and the non-distributed train() path stays untouched (tests/test_train.py is the witness)
# --------------------------------------------------------------------------------------------


def test_world_size_one_matches_combined_optimizer() -> None:
    seed_everything(3)
    model = TransformerLM(_model_cfg())
    seed_everything(3)
    ref = TransformerLM(_model_cfg())
    muon_p, adamw_p = split_muon_adamw_params(model)
    opt = DistMuonAdamW(muon_p, adamw_p, lr=0.02)
    assert opt.world_size == 1 and opt.rank == 0
    ref_opt = build_optimizer(ref, kind="muon_adamw", lr=0.02)

    torch.manual_seed(5)
    for _ in range(3):
        for p, pr in zip(model.parameters(), ref.parameters(), strict=True):
            g = torch.randn_like(p)
            p.grad = g.clone()
            pr.grad = g.clone()
        opt.step()
        ref_opt.step()
    for (name, p), (_, pr) in zip(model.named_parameters(), ref.named_parameters(), strict=True):
        torch.testing.assert_close(p, pr, atol=1e-7, rtol=0.0, msg=f"{name} drifted")


def test_world_size_one_pure_adamw_matches() -> None:
    torch.manual_seed(9)
    big = nn.Parameter(torch.randn(40, 30))  # 1200 ≥ 1024 → the row-shard path at world 1
    small = nn.Parameter(torch.randn(6))
    ref_big = nn.Parameter(big.detach().clone())
    ref_small = nn.Parameter(small.detach().clone())
    opt = DistMuonAdamW([], [big, small], lr=1e-3, weight_decay=0.01)
    ref_opt = AdamW([ref_big, ref_small], lr=1e-3, weight_decay=0.01)
    for _ in range(3):
        for p, pr in zip([big, small], [ref_big, ref_small], strict=True):
            g = torch.randn_like(p)
            p.grad = g.clone()
            pr.grad = g.clone()
        opt.step()
        ref_opt.step()
    torch.testing.assert_close(big, ref_big, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(small, ref_small, atol=1e-7, rtol=0.0)


def test_lr_schedule_write_reaches_both_algo_groups() -> None:
    seed_everything(0)
    model = TransformerLM(_model_cfg())
    muon_p, adamw_p = split_muon_adamw_params(model)
    opt = DistMuonAdamW(muon_p, adamw_p, lr=1e-3)
    for group in opt.param_groups:  # what train()'s scheduler writes to
        group["lr"] = 0.042
    assert all(g["lr"] == 0.042 for g in opt.param_groups)
    assert {g["algo"] for g in opt.param_groups} == {"muon", "adamw"}
