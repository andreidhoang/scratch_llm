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
from scratch_llm.train import (
    TrainConfig,
    load_consolidated_checkpoint,
    save_consolidated_checkpoint,
    train,
)
from scratch_llm.utils.dist_train import DistMuonAdamW, assert_model_replicas_identical
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


# ============================================================================================
# A8 — consolidated distributed checkpoint: SAVE from N ranks → resume/reshard on any topology.
# The A7 state_dict is rank-local shards (same-topology resume only); A8 all-gathers them into a
# single un-sharded, topology-independent state you can resume at a different world size or
# extract the model from for serving.
# ============================================================================================

_WARM_SEED = 21  # grad stream for the warmup half (pre-checkpoint)
_CONT_SEED = 77  # grad stream for the continuation half (post-resume, shared by both trajectories)


def _set_per_rank_grads(model: nn.Module, gen: torch.Generator, rank: int) -> None:
    """Per-rank grads whose 2-rank AVERAGE (base + delta/2) is exactly representable (see
    ``_exact_grad_pair``): rank r gets ``base + r·delta``."""
    for p in model.parameters():
        base, delta = _exact_grad_pair(p, gen)
        p.grad = base + rank * delta


# --------------------------------------------------------------------------------------------
# (a) round-trip: consolidate → save bytes → load_consolidated → resharded continuation equals
#     the uninterrupted continuation, BITWISE, on every rank — SAME topology (world 2 AND a
#     non-power-of-2 world 3, which drives the Muon k_pad and AdamW row_pad un-pad paths).
# --------------------------------------------------------------------------------------------


def _roundtrip_worker(rank: int, world_size: int, port: int, ckpt_path: str) -> None:
    _init_pg(rank, world_size, port)
    try:

        def build() -> tuple[TransformerLM, DistMuonAdamW]:
            seed_everything(0)
            m = TransformerLM(_model_cfg())  # identical init on every rank
            a, b = split_muon_adamw_params(m)
            return m, DistMuonAdamW(a, b, lr=0.02, max_l2_norm=1.0)

        model, opt = build()
        gwarm = torch.Generator().manual_seed(_WARM_SEED)
        for _ in range(3):  # warmup so momentum / Adam moments are populated (not lazy-None)
            _set_per_rank_grads(model, gwarm, rank)
            opt.step()

        # consolidate (every rank gathers) + write ONE file on rank 0, then a barrier so the file
        # is on disk before any rank reads it — the production save/resume shape.
        save_consolidated_checkpoint(model, opt, 3, ckpt_path)
        dist.barrier()

        # reference: keep stepping the ORIGINAL optimizer — the uninterrupted trajectory.
        gcont = torch.Generator().manual_seed(_CONT_SEED)
        for _ in range(3):
            _set_per_rank_grads(model, gcont, rank)
            opt.step()

        # resumed: fresh model + optimizer, reshard the consolidated file back onto THIS rank,
        # replay the SAME continuation grads.
        model2, opt2 = build()
        step = load_consolidated_checkpoint(ckpt_path, model2, opt2)
        assert step == 3, f"rank {rank}: resumed step {step} != 3"
        gcont2 = torch.Generator().manual_seed(_CONT_SEED)
        for _ in range(3):
            _set_per_rank_grads(model2, gcont2, rank)
            opt2.step()

        # SAME topology → the resharded state is bitwise-identical to the shard state it came
        # from, and identical grads drive an identical collective schedule ⇒ bitwise trajectories.
        ref = dict(model.named_parameters())
        for name, p2 in model2.named_parameters():
            assert torch.equal(p2, ref[name]), (
                f"rank {rank}: resumed param {name} != uninterrupted trajectory (not bitwise)"
            )
    finally:
        dist.destroy_process_group()


def test_consolidated_roundtrip_reshards_bitwise_world2(tmp_path) -> None:
    ckpt = str(tmp_path / "consolidated_w2.pt")
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _roundtrip_worker, args=(2, _free_port(), ckpt), nprocs=2, join=True
    )


def test_consolidated_roundtrip_reshards_bitwise_world3_nonpow2(tmp_path) -> None:
    ckpt = str(tmp_path / "consolidated_w3.pt")
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _roundtrip_worker, args=(3, _free_port(), ckpt), nprocs=3, join=True
    )


# --------------------------------------------------------------------------------------------
# (b) topology-independence: a state saved on world 2 resumes on WORLD 1 and its continuation
#     matches a native world-1 trajectory — proving the consolidated format carries no sharding.
# --------------------------------------------------------------------------------------------


def _w2_save_worker(rank: int, world_size: int, port: int, ckpt_path: str, warmup: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        seed_everything(0)
        model = TransformerLM(_model_cfg())
        muon_p, adamw_p = split_muon_adamw_params(model)
        opt = DistMuonAdamW(muon_p, adamw_p, lr=0.02, max_l2_norm=0.0)  # no clip: clean cross-topo
        gwarm = torch.Generator().manual_seed(_WARM_SEED)
        for _ in range(warmup):
            _set_per_rank_grads(model, gwarm, rank)  # avg over 2 ranks == base + delta/2
            opt.step()
        save_consolidated_checkpoint(model, opt, warmup, ckpt_path)  # rank 0 writes
    finally:
        dist.destroy_process_group()


def test_world2_consolidated_resumes_at_world1(tmp_path) -> None:
    ckpt = str(tmp_path / "w2_to_w1.pt")
    warmup, cont = 3, 3
    # phase 1: 2-rank warmup on per-rank grads → consolidate → file.
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _w2_save_worker, args=(2, _free_port(), ckpt, warmup), nprocs=2, join=True
    )

    # phase 2 (main process, NO process group ⇒ world size 1): resume the consolidated file and
    # continue on the AVERAGED grad stream (base + delta/2 == the 2-rank average of the warmup).
    seed_everything(0)
    model_r = TransformerLM(_model_cfg())
    a, b = split_muon_adamw_params(model_r)
    opt_r = DistMuonAdamW(a, b, lr=0.02, max_l2_norm=0.0)
    assert opt_r.world_size == 1 and opt_r.rank == 0
    step = load_consolidated_checkpoint(ckpt, model_r, opt_r)
    assert step == warmup
    gcont = torch.Generator().manual_seed(_CONT_SEED)
    for _ in range(cont):
        for p in model_r.parameters():
            base, delta = _exact_grad_pair(p, gcont)
            p.grad = base + delta / 2
        opt_r.step()

    # native world-1 reference: the SAME warmup+continuation on averaged grads, uninterrupted.
    seed_everything(0)
    model_ref = TransformerLM(_model_cfg())
    a2, b2 = split_muon_adamw_params(model_ref)
    opt_ref = DistMuonAdamW(a2, b2, lr=0.02, max_l2_norm=0.0)
    gw = torch.Generator().manual_seed(_WARM_SEED)
    for _ in range(warmup):
        for p in model_ref.parameters():
            base, delta = _exact_grad_pair(p, gw)
            p.grad = base + delta / 2
        opt_ref.step()
    gc = torch.Generator().manual_seed(_CONT_SEED)
    for _ in range(cont):
        for p in model_ref.parameters():
            base, delta = _exact_grad_pair(p, gc)
            p.grad = base + delta / 2
        opt_ref.step()

    # cross-topology: the AVG reduction order differs from a native single-tensor mean, so the
    # match is tight-but-not-bitwise (the same 1e-6 band the A7 oracle uses).
    ref = dict(model_ref.named_parameters())
    for name, p in model_r.named_parameters():
        torch.testing.assert_close(p, ref[name], atol=1e-6, rtol=1e-6, msg=f"{name} drifted")


# --------------------------------------------------------------------------------------------
# (c) the replica-identity guard fires (on EVERY rank, no hang) on an injected desync — and is a
#     no-op on genuinely-identical replicas.
# --------------------------------------------------------------------------------------------


def _replica_guard_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        seed_everything(0)
        model = TransformerLM(_model_cfg())  # identical init on every rank
        assert_model_replicas_identical(model)  # positive: identical replicas ⇒ no raise

        if rank == 1:  # inject a divergence on ONE rank
            with torch.no_grad():
                next(iter(model.parameters())).add_(1.0)

        raised = False
        try:
            assert_model_replicas_identical(model)
        except RuntimeError as err:
            assert "diverged" in str(err), f"rank {rank}: wrong error {err}"
            raised = True
        assert raised, f"rank {rank}: replica-identity guard did NOT fire on the injected desync"
    finally:
        dist.destroy_process_group()


def test_replica_identity_guard_fires_on_desync() -> None:
    ctx = mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _replica_guard_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=False
    )
    assert ctx is not None
    # poll with a deadline: a collective guard must make BOTH ranks raise together — a hang here
    # would mean one rank raised alone and left the other blocked in the all_gather.
    deadline = time.monotonic() + 120
    while not ctx.join(timeout=5):
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                if proc is not None:
                    proc.terminate()
            raise AssertionError("replica-identity guard hung instead of raising on all ranks")


# --------------------------------------------------------------------------------------------
# (d) un-pad correctness: with K not divisible by world (Muon) AND dim0 not divisible by world
#     (AdamW), the consolidated FULL state has exactly K matrices / dim0 rows and its VALUES equal
#     the single-process optimizer's un-padded state.
# --------------------------------------------------------------------------------------------


def _unpad_worker(rank: int, world_size: int, port: int) -> None:
    _init_pg(rank, world_size, port)
    try:
        # world 3: K=5 Muon matrices → k_pad=6 (1 pad slot); big dim0=7 → rows_pad=9 (2 pad rows).
        torch.manual_seed(0)
        mats = [nn.Parameter(torch.randn(4, 6)) for _ in range(5)]
        big = nn.Parameter(torch.randn(7, 200))  # 1400 ≥ SMALL_PARAM_NUMEL ⇒ row-sharded
        ref_mats = [nn.Parameter(m.detach().clone()) for m in mats]
        ref_big = nn.Parameter(big.detach().clone())

        opt = DistMuonAdamW(mats, [big], lr=0.02, muon_momentum=0.95, weight_decay=0.1)
        ref_muon = Muon(ref_mats, lr=0.02, momentum=0.95, weight_decay=0.1)  # matched defaults
        ref_adamw = AdamW([ref_big], lr=0.02, betas=(0.9, 0.95), weight_decay=0.1)

        # per-rank grad base + r·delta ⇒ mean over W ranks = base + ((W-1)/2)·delta (for W=3 → +δ).
        avg_coeff = (world_size - 1) / 2
        gen = torch.Generator().manual_seed(9)
        for _ in range(2):
            for p, pr in zip(mats + [big], ref_mats + [ref_big], strict=True):
                base, delta = _exact_grad_pair(p, gen)
                p.grad = base + rank * delta
                pr.grad = (
                    base + avg_coeff * delta
                )  # the averaged grad the single-process oracle sees
            opt.step()
            ref_muon.step()
            ref_adamw.step()

        consolidated = opt.consolidated_state_dict()  # collective — every rank participates
        if rank == 0:
            assert consolidated is not None
            group = consolidated["muon_momentum"][0]
            assert group is not None and len(group) == 5, "un-pad dropped the wrong Muon count"
            for i, mref in enumerate(ref_mats):
                buf = ref_muon.state[mref]["momentum_buffer"]
                torch.testing.assert_close(
                    group[i], buf, atol=1e-6, rtol=1e-6, msg=f"muon momentum matrix {i}"
                )
            mv = consolidated["adamw_rows"][0]
            assert mv is not None and mv["m"].shape[0] == 7 and mv["v"].shape[0] == 7, (
                "un-pad kept pad rows in the consolidated AdamW moments"
            )
            st = ref_adamw.state[ref_big]
            torch.testing.assert_close(mv["m"], st["m"], atol=1e-6, rtol=1e-6, msg="adamw m")
            torch.testing.assert_close(mv["v"], st["v"], atol=1e-6, rtol=1e-6, msg="adamw v")
        else:
            assert consolidated is None, "non-zero rank must return None from consolidation"
    finally:
        dist.destroy_process_group()


def test_consolidated_unpad_matches_single_process_world3() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _unpad_worker, args=(3, _free_port()), nprocs=3, join=True
    )


# --------------------------------------------------------------------------------------------
# train() wiring: a distributed run with checkpoint_every writes ONE consolidated file (rank 0)
# that load_consolidated_checkpoint can resume — and the single-process save path is untouched.
# --------------------------------------------------------------------------------------------


def _train_consolidated_ckpt_worker(rank: int, world_size: int, port: int, ckpt_path: str) -> None:
    _init_pg(rank, world_size, port)
    try:
        seed_everything(7)
        model = TransformerLM(_model_cfg())
        data = np.tile(np.arange(16, dtype=np.int64), 400)
        cfg = TrainConfig(
            max_steps=5,
            batch_size=4,
            context_length=8,
            max_lr=1e-3,
            seed=7,
            optimizer="muon_adamw",
            checkpoint_every=2,
            checkpoint_path=ckpt_path,
            log_every=1,
        )
        train(cfg, data, model)  # checkpoints at steps 2 and 4 (rank 0 writes; all ranks gather)

        if rank == 0:
            state = torch.load(ckpt_path, weights_only=False)
            assert state["format"] == "consolidated"
            assert state["world_size"] == world_size
            assert state["optim_consolidated"] is not None
            assert state["model_config"] is not None
        dist.barrier()

        # resume the consolidated file onto a fresh same-topology optimizer, take one more step.
        seed_everything(7)
        model2 = TransformerLM(_model_cfg())
        a, b = split_muon_adamw_params(model2)
        opt2 = DistMuonAdamW(a, b, lr=1e-3, max_l2_norm=1.0)
        step = load_consolidated_checkpoint(ckpt_path, model2, opt2)
        assert step == 4, f"rank {rank}: resumed at step {step}, expected 4"
        for p in model2.parameters():
            p.grad = torch.zeros_like(p)
        opt2.step()  # must not raise — the resharded state is a valid live optimizer
    finally:
        dist.destroy_process_group()


def test_train_writes_resumable_consolidated_checkpoint(tmp_path) -> None:
    ckpt = str(tmp_path / "train_consolidated.pt")
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _train_consolidated_ckpt_worker, args=(WORLD, _free_port(), ckpt), nprocs=WORLD, join=True
    )
