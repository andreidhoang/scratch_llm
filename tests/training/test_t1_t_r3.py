"""T1/T-R3 — three induced failures, three detections, and the clean-run control for each.

Every detector here is tested in BOTH directions, because only one of them is interesting on its
own. A detector that fires on the fault proves nothing if it also fires on a clean run: it has no
false-negative rate because it has no discrimination, and the first engineer it wakes at 3am turns
it off. So each of the three gets a "fires on the fault" test and a "silent on the clean run" test,
and the resume verifier additionally gets a "fires on a resume that is merely allclose" test —
the failure mode that would let a careless check pass forever.

Nothing here has a tolerance. ``torch.equal`` on a byte view, a bitwise digest, and
``math.isfinite`` are the only comparisons used. The ``SCENARIO_*`` constants below are *inputs to
a scenario* (what stream to feed the spike detector, how many skips before escalating), not
correctness tolerances; the spike detector's production operating point is Huy's to set and no
default for it exists anywhere in ``scratch_llm.faults``.

Distribution is 2-rank gloo on CPU, which is enough to prove every claim except NCCL's own
rank-symmetry and a real GPU rank death; the one test that needs those skips with the exact box
command rather than passing quietly.
"""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException

from scratch_llm.faults.capsule import (
    TrainSnapshot,
    as_train_py_checkpoint,
    bitwise_equal,
    capture,
    restore,
    verify_bit_exact,
    without_rng,
)
from scratch_llm.faults.gradnorm import CrossRankGradCheck, min_detectable_mantissa_bit
from scratch_llm.faults.inject import (
    NaNBatch,
    RankKill,
    describe_bit,
    flip_bit_,
    flip_grad_bit_,
    largest_grad_index,
)
from scratch_llm.faults.loop import StepHooks, pin_determinism, run_steps
from scratch_llm.faults.spike import LossSpikeDetector, SpikeAction
from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy
from scratch_llm.optim import build_optimizer
from scratch_llm.train import TrainConfig, train
from scratch_llm.utils.seeding import seed_everything

WORLD = 2
SEED = 1234
FLIP_PARAM = "token_emb.weight"

# Scenario inputs, not tolerances. They describe the stream fed to the detector and how patient its
# policy is; the z at which a production run should call a batch a spike is Huy's number and is a
# required argument with no default (src/scratch_llm/faults/spike.py).
SCENARIO_SPIKE_Z = 4.0
SCENARIO_WINDOW = 12
SCENARIO_MIN_OBS = 6
SCENARIO_MAX_SKIPS = 3
SCENARIO_MAX_RESTARTS = 3

BOX_COMMAND = (
    "  cd ~/ladders && experiments/T1/T-R3/run.sh\n"
    "  # or directly:\n"
    "  PYTHONPATH=scratch_llm/src python scratch_llm/bench/t1_faults.py \\\n"
    "      --mode fault --spike-z inf --world-size 8 --device cuda"
)


# ---------------------------------------------------------------------------------------------
# shared fixtures — the same tiny model everywhere, so a test measures logic and not throughput
# ---------------------------------------------------------------------------------------------


def _model_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, context_length=16)


def _train_cfg(steps: int) -> TrainConfig:
    return TrainConfig(
        max_steps=steps, batch_size=4, context_length=16, warmup_steps=2, log_every=1, seed=SEED
    )


def _corpus() -> np.ndarray:
    return (np.arange(8192, dtype=np.uint16) * 7 + 3) % 64


def _fresh(cfg: TrainConfig, rank: int = 0) -> tuple[TransformerLM, Any]:
    """A brand-new process's starting state: seed, build, seed again, and — only when a process
    group exists — offset the data stream by rank exactly as ``train()`` does (train.py:340-344).
    The offset is conditional there too; applying it single-process would silently change the batch
    stream and the oracle comparison against ``train()`` would compare two different runs.
    """
    pin_determinism()
    seed_everything(cfg.seed)
    model = TransformerLM(_model_cfg())
    seed_everything(cfg.seed)
    if dist.is_available() and dist.is_initialized():
        np.random.seed(cfg.seed + 1000 * rank + 1)
    optimizer = build_optimizer(
        model, kind=cfg.optimizer, lr=cfg.max_lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )
    return model, optimizer


def _snap(step: int, lr: float, model: TransformerLM, opt: Any, drawn: int) -> TrainSnapshot:
    return capture(step=step, lr=lr, model=model, optimizer=opt, batches_drawn=drawn)


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


# ---------------------------------------------------------------------------------------------
# the injector itself — if it does not inject, every "detector stayed silent" below is a lie
# ---------------------------------------------------------------------------------------------


def test_a_bit_flip_moves_exactly_one_bit_and_undoes_itself() -> None:
    t = torch.tensor([1.0, -2.5, 0.125, 7.75], dtype=torch.float32)
    for bit in (0, 11, 22, 23, 30, 31):
        original = t.clone()
        flip = flip_bit_(t, 1, bit)
        words_a = original.view(torch.int32)
        words_b = t.view(torch.int32)
        changed = int(words_a[1].item()) ^ int(words_b[1].item())
        assert changed == (1 << bit) - (1 << 32 if bit == 31 else 0) or changed == (
            1 << bit
        ) - 0x100000000 * (bit == 31), f"bit {bit}: xor was {changed:#x}"
        assert torch.equal(words_a[[0, 2, 3]], words_b[[0, 2, 3]]), "a neighbouring element moved"
        assert flip.bit == bit and flip.field == describe_bit(torch.float32, bit)
        flip_bit_(t, 1, bit)  # XOR is an involution
        assert bitwise_equal(t, original), f"bit {bit} did not undo"


def test_the_bit_fields_are_named_the_ieee_way() -> None:
    assert describe_bit(torch.float32, 31) == "sign bit"
    assert describe_bit(torch.float32, 30).startswith("exponent bit 7")
    assert describe_bit(torch.float32, 23).startswith("exponent bit 0")
    assert describe_bit(torch.float32, 22) == "mantissa bit 22"
    assert describe_bit(torch.float32, 0) == "mantissa bit 0 (lsb)"
    assert describe_bit(torch.bfloat16, 15) == "sign bit"  # 7-bit mantissa, 8-bit exponent
    assert describe_bit(torch.bfloat16, 7).startswith("exponent bit 0")


def test_a_flip_on_a_strided_tensor_is_refused_rather_than_silently_dropped() -> None:
    strided = torch.zeros(4, 4)[:, 1]  # a view, not contiguous
    with pytest.raises(ValueError, match="contiguous"):
        flip_bit_(strided, 0, 0)


def test_a_nan_batch_keeps_the_autograd_graph_intact() -> None:
    """The injector must poison the value, not sever the graph.

    ``torch.full_like(out, nan)`` reads as the obvious implementation and is a trap: it returns a
    fresh leaf, so the poisoned rank ends ``backward()`` with ``token_emb.weight.grad is None``
    while its peers have one. The ranks then issue different numbers of all-reduces and the job
    hangs in a collective with no error and no traceback — which is exactly what this harness did
    on its first run, and exactly the shape of the classic NCCL hang.
    """
    model = TransformerLM(_model_cfg())
    poison = NaNBatch(steps={0})
    poison.current_step = 0
    handle = poison.attach(model.token_emb)
    inputs = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    loss = cross_entropy(model(inputs), targets)
    loss.backward()
    handle.remove()

    assert not math.isfinite(float(loss.item())), "the poison did not reach the loss"
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == [], f"the poison severed the graph for {missing} — collective counts desync"
    embedding_grad = model.token_emb.weight.grad
    assert embedding_grad is not None
    assert not bool(torch.isfinite(embedding_grad).all().item())


def test_a_detached_injector_leaves_the_run_bit_identical() -> None:
    """Silence has to be free: an injector attached but never triggered must not perturb a step."""
    cfg = _train_cfg(4)
    model_a, opt_a = _fresh(cfg)
    run_steps(cfg, _corpus(), model_a, opt_a, start_step=0, n_steps=4)

    model_b, opt_b = _fresh(cfg)
    poison = NaNBatch(steps=set())  # attached, never fires
    handle = poison.attach(model_b.token_emb)
    hooks = StepHooks(on_batch=lambda s, _i, _t: setattr(poison, "current_step", s))
    run_steps(cfg, _corpus(), model_b, opt_b, start_step=0, n_steps=4, hooks=hooks)
    handle.remove()

    for (name, a), (_, b) in zip(
        model_a.state_dict().items(), model_b.state_dict().items(), strict=True
    ):
        assert bitwise_equal(a, b), f"the idle injector moved {name}"


# ---------------------------------------------------------------------------------------------
# the instrument loop is pinned to train() — otherwise it measures a different algorithm
# ---------------------------------------------------------------------------------------------


def test_the_instrument_loop_reproduces_train_step_for_step() -> None:
    cfg = _train_cfg(6)
    seed_everything(cfg.seed)
    reference_model = TransformerLM(_model_cfg())
    history = train(cfg, _corpus(), reference_model)

    model, optimizer = _fresh(cfg)
    result = run_steps(cfg, _corpus(), model, optimizer, start_step=0, n_steps=cfg.max_steps)

    assert [v for _, v in history] == result.losses, "the instrument loop is not train()'s loop"
    for (name, a), (_, b) in zip(
        reference_model.state_dict().items(), model.state_dict().items(), strict=True
    ):
        assert bitwise_equal(a, b), f"{name} diverged between train() and faults.loop"


def test_the_loop_refuses_the_paths_it_cannot_mirror() -> None:
    model, optimizer = _fresh(_train_cfg(1))
    for field, value, match in (
        ("amp_dtype", "bf16", "amp_dtype"),
        ("compile", True, "compile"),
    ):
        cfg = _train_cfg(1)
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match=match):
            run_steps(cfg, _corpus(), model, optimizer, start_step=0, n_steps=1)


# ---------------------------------------------------------------------------------------------
# fault 1 detection — bit-exact resume verification
# ---------------------------------------------------------------------------------------------


def _reference_run(steps: int, checkpoint_at: int) -> tuple[TrainSnapshot, TrainSnapshot]:
    """Run uninterrupted; return (mid-run capsule, final state). The final state is the oracle:
    there is no analytic answer for "what should step N be", only "what it was when nothing
    went wrong"."""
    cfg = _train_cfg(steps)
    model, optimizer = _fresh(cfg)
    first = run_steps(cfg, _corpus(), model, optimizer, start_step=0, n_steps=checkpoint_at)
    ckpt = _snap(first.next_step, first.records[-1].lr, model, optimizer, first.batches_drawn)
    rest = run_steps(
        cfg,
        _corpus(),
        model,
        optimizer,
        start_step=first.next_step,
        n_steps=steps - checkpoint_at,
        batches_drawn=first.batches_drawn,
    )
    final = _snap(rest.next_step, rest.records[-1].lr, model, optimizer, rest.batches_drawn)
    return ckpt, final


def _resume_from(ckpt: TrainSnapshot, steps: int) -> TrainSnapshot:
    """What a relaunched process does: fresh interpreter state, restore, continue."""
    cfg = _train_cfg(steps)
    model, optimizer = _fresh(cfg)  # re-seeds, as a fresh process does
    start = restore(ckpt, model, optimizer)
    res = run_steps(
        cfg,
        _corpus(),
        model,
        optimizer,
        start_step=start,
        n_steps=steps - start,
        batches_drawn=ckpt.batches_drawn,
    )
    return _snap(res.next_step, res.records[-1].lr, model, optimizer, res.batches_drawn)


def test_the_resume_verifier_is_silent_when_the_capsule_is_complete() -> None:
    ckpt, reference = _reference_run(steps=12, checkpoint_at=5)
    report = verify_bit_exact(reference, _resume_from(ckpt, steps=12))
    assert report.bit_exact, report.summary()
    assert report.mismatches == ()


def test_the_resume_verifier_fires_when_the_capsule_drops_the_rng() -> None:
    """torchtitan's checkpoint shape: MODEL / OPTIMIZER / LR_SCHEDULER / DATALOADER / TRAIN_STATE
    and no RNG (``components/checkpointer/base.py:30-34``). Everything is restored except the
    position in the random stream, so the resumed run draws different batches and quietly becomes
    a different run with a plausible loss curve."""
    ckpt, reference = _reference_run(steps=12, checkpoint_at=5)
    report = verify_bit_exact(reference, _resume_from(without_rng(ckpt), steps=12))
    assert report.fired, "dropping the RNG left the run bit-exact — the control proves nothing"
    assert any(m.startswith("model.") for m in report.mismatches), report.summary()
    assert any(m.startswith("rng") for m in report.mismatches), report.summary()


def test_the_resume_verifier_fires_when_the_lr_schedule_position_is_lost() -> None:
    """``scratch_llm.train``'s shape: ``save_checkpoint`` writes model+optim+step and ``train()``
    always begins at step 0 with a fresh warmup (train.py:5-10). The weights are right and the
    place on the cosine curve is not."""
    ckpt, reference = _reference_run(steps=12, checkpoint_at=5)
    degraded = as_train_py_checkpoint(ckpt)
    assert degraded.step == 0
    report = verify_bit_exact(reference, _resume_from(degraded, steps=12))
    assert report.fired, "restarting the schedule at 0 left the run bit-exact"
    assert any(m.startswith("model.") for m in report.mismatches), report.summary()
    # The sharpest witness is Adam's own step counter: the resumed run kept the moments from step
    # 5 and then took a further `steps` steps, so it has taken 17 where the reference took 12.
    # A resume that replays the schedule from 0 is a longer run wearing the same step number.
    assert any(m.startswith("optim") for m in report.mismatches), report.summary()


def test_the_resume_verifier_is_not_fooled_by_a_resume_that_is_merely_allclose() -> None:
    """The failure mode this whole rung exists to prevent.

    Perturb one parameter by ONE ulp — a change ``torch.allclose`` and every ``rtol=1e-5`` test in
    the repo call equal. It is not equal: the next Adam step divides by a different denominator,
    and the two runs separate. Exact equality is the only comparison a resume check may use.
    """
    ckpt, reference = _reference_run(steps=8, checkpoint_at=3)
    resumed = _resume_from(ckpt, steps=8)
    assert verify_bit_exact(reference, resumed).bit_exact  # the honest baseline

    name = "token_emb.weight"
    nudged = resumed.model[name].clone()
    flip_bit_(nudged, 0, 0)  # one ulp, in the mantissa lsb
    resumed.model[name] = nudged

    assert torch.allclose(reference.model[name], nudged), "the nudge was not sub-tolerance"
    report = verify_bit_exact(reference, resumed)
    assert report.fired, "a one-ulp resume passed — the verifier is using a tolerance somewhere"
    assert report.first_mismatch == f"model.{name}"


def test_the_spike_window_rides_in_the_capsule() -> None:
    """Trajectory state, not just weights: a resumed detector with an empty baseline re-arms from
    scratch and makes different skip decisions than the run it claims to continue."""
    detector = LossSpikeDetector(
        spike_z=SCENARIO_SPIKE_Z,
        window=SCENARIO_WINDOW,
        min_observations=SCENARIO_MIN_OBS,
        max_consecutive_skips=SCENARIO_MAX_SKIPS,
        max_restarts=SCENARIO_MAX_RESTARTS,
    )
    for step, loss in enumerate(4.0 + 0.01 * np.sin(np.arange(10))):
        detector.observe(step, float(loss))
    saved = detector.state_dict()

    cfg = _train_cfg(2)
    model, optimizer = _fresh(cfg)
    snap = capture(
        step=10, lr=1e-4, model=model, optimizer=optimizer, batches_drawn=10, extra={"spike": saved}
    )
    reloaded = LossSpikeDetector(
        spike_z=SCENARIO_SPIKE_Z,
        window=SCENARIO_WINDOW,
        min_observations=SCENARIO_MIN_OBS,
        max_consecutive_skips=SCENARIO_MAX_SKIPS,
        max_restarts=SCENARIO_MAX_RESTARTS,
    )
    reloaded.load_state_dict(snap.extra["spike"])
    assert reloaded.baseline == detector.baseline
    assert reloaded.scale == detector.scale
    assert reloaded.state_dict() == saved


# ---------------------------------------------------------------------------------------------
# fault 2 detection — the loss-spike detector and its skip policy
# ---------------------------------------------------------------------------------------------


def _detector() -> LossSpikeDetector:
    return LossSpikeDetector(
        spike_z=SCENARIO_SPIKE_Z,
        window=SCENARIO_WINDOW,
        min_observations=SCENARIO_MIN_OBS,
        max_consecutive_skips=SCENARIO_MAX_SKIPS,
        max_restarts=SCENARIO_MAX_RESTARTS,
    )


def test_the_spike_detector_is_silent_across_a_clean_run() -> None:
    cfg = _train_cfg(14)
    model, optimizer = _fresh(cfg)
    detector = _detector()
    run_steps(
        cfg,
        _corpus(),
        model,
        optimizer,
        start_step=0,
        n_steps=cfg.max_steps,
        hooks=StepHooks(on_loss=detector.observe),
    )
    assert not detector.fired, [str(e) for e in detector.events]
    assert all(bool(torch.isfinite(p).all().item()) for p in model.parameters())


def test_the_spike_detector_fires_on_the_nan_batch_and_the_optimizer_never_steps() -> None:
    cfg = _train_cfg(14)
    fault_step = 8
    model, optimizer = _fresh(cfg)
    detector = _detector()
    poison = NaNBatch(steps={fault_step})
    handle = poison.attach(model.token_emb)
    hooks = StepHooks(
        on_batch=lambda s, _i, _t: setattr(poison, "current_step", s), on_loss=detector.observe
    )
    result = run_steps(
        cfg, _corpus(), model, optimizer, start_step=0, n_steps=cfg.max_steps, hooks=hooks
    )
    handle.remove()

    assert poison.fired_at == [fault_step], "the injector did not inject"
    assert detector.fired
    assert [e.step for e in detector.events] == [fault_step]
    assert [e.reason for e in detector.events] == ["nonfinite"]
    assert [e.action for e in detector.events] == [SpikeAction.SKIP]
    assert [r.action for r in result.records if r.action != "continue"] == ["skip"]
    assert result.records[fault_step].action == "skip"
    # The point of skipping: a NaN in AdamW's exp_avg poisons every future step, not just this one.
    assert all(bool(torch.isfinite(p).all().item()) for p in model.parameters())
    assert len(result.records) == cfg.max_steps, "the run did not survive the batch it skipped"


def test_the_skip_policy_does_not_livelock_on_a_legitimately_hard_phase() -> None:
    """A livelock is a failure mode of the FIX, and it deserves its own test.

    Feed an easy phase then a permanently harder one — a domain switch, a longer-context shard.
    A policy that only admits non-skipped losses into its baseline will fight the new regime
    forever: every batch is a spike, no gradient is ever applied, the step counter advances and the
    money burns. The escalation ladder must terminate: adapt, restart once, and resume training.
    """
    detector = _detector()
    easy = 4.0 + 0.01 * np.sin(np.arange(24))
    hard = 6.0 + 0.01 * np.sin(np.arange(40))
    actions = [detector.observe(i, float(v)) for i, v in enumerate(np.concatenate([easy, hard]))]

    assert SpikeAction.SKIP in actions, "the detector did not even notice the regime shift"
    assert SpikeAction.HALT not in actions, "a legitimately hard corpus killed the run"
    assert actions[-10:] == [SpikeAction.CONTINUE] * 10, "still skipping at the end — livelocked"
    assert detector.skip_count() <= SCENARIO_MAX_SKIPS * (SCENARIO_MAX_RESTARTS + 1)
    assert math.isfinite(detector.baseline) and detector.baseline > 4.0, "baseline never adapted"


def test_an_all_nan_stream_is_never_adapted_into_the_baseline() -> None:
    """The counterpart: NaN is never a legitimate regime. The ladder must reach HALT and stop,
    not decide that NaN is the new normal and continue."""
    detector = _detector()
    actions = [detector.observe(i, 4.0 + 0.01 * math.sin(i)) for i in range(12)]
    nan_actions = [detector.observe(12 + i, float("nan")) for i in range(40)]

    assert SpikeAction.CONTINUE not in nan_actions, "a NaN step was allowed to update the model"
    assert SpikeAction.HALT in nan_actions, "the ladder never terminated on an unrecoverable stream"
    halt_at = nan_actions.index(SpikeAction.HALT)
    assert halt_at <= (SCENARIO_MAX_RESTARTS + 1) * (SCENARIO_MAX_SKIPS + 1)
    assert math.isfinite(detector.baseline), "NaN was absorbed into the baseline"
    assert actions[-1] is SpikeAction.CONTINUE


# ---------------------------------------------------------------------------------------------
# fault 3 detection — cross-rank gradient divergence (2-rank gloo)
# ---------------------------------------------------------------------------------------------


def _dist_setup(rank: int, world_size: int, port: int, steps: int) -> tuple[Any, Any, TrainConfig]:
    _init_pg(rank, world_size, port)
    cfg = _train_cfg(steps)
    model, optimizer = _fresh(cfg, rank=rank)
    return model, optimizer, cfg


def _gradcheck_worker(rank: int, world_size: int, port: int, mode: str, fault_step: int) -> None:
    model, optimizer, cfg = _dist_setup(rank, world_size, port, steps=6)
    check = CrossRankGradCheck()
    victim = world_size - 1

    def flip(step: int) -> None:
        if rank == victim and step == fault_step:
            flip_grad_bit_(model, FLIP_PARAM, largest_grad_index(model, FLIP_PARAM), 0)

    def cross_rank_check(_step: int) -> None:
        check.check(model)  # collective; the report is kept on the detector

    hooks = StepHooks(after_reduce_check=cross_rank_check)
    if mode == "post":
        hooks.after_reduce = flip
    elif mode == "pre":
        hooks.after_backward = flip
    run_steps(cfg, _corpus(), model, optimizer, start_step=0, n_steps=cfg.max_steps, hooks=hooks)

    fired = [i for i, r in enumerate(check.reports) if r.fired]
    if mode == "post":
        assert fired == [fault_step], f"rank {rank}: fired at {fired}, expected [{fault_step}]"
        report = check.reports[fault_step]
        assert "digest" in report.channels, report.summary()
        assert report.first_param == FLIP_PARAM, report.summary()
        assert not report.named_rank, "with 2 ranks the culprit is not identifiable — see the map"
    elif mode == "pre":
        # The stated blind spot. The all-reduce spread the corruption to every rank, they agree
        # perfectly, and no cross-rank comparison can see it. Deterministic replay is the tool
        # for this one (oss/torchtitan/torchtitan/observability/sdc_replayer.py).
        assert fired == [], f"rank {rank}: a pre-reduction flip was seen at {fired} — recheck why"
    else:
        assert fired == [], f"rank {rank}: fired on a clean run at {fired}"
    dist.destroy_process_group()


def test_the_cross_rank_check_is_silent_across_a_clean_run() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _gradcheck_worker, args=(WORLD, _free_port(), "clean", -1), nprocs=WORLD
    )


def test_the_cross_rank_check_fires_on_a_post_reduction_mantissa_lsb_flip() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _gradcheck_worker, args=(WORLD, _free_port(), "post", 3), nprocs=WORLD
    )


def test_the_cross_rank_check_is_blind_to_a_pre_reduction_flip() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _gradcheck_worker, args=(WORLD, _free_port(), "pre", 3), nprocs=WORLD
    )


def _boundary_worker(rank: int, world_size: int, port: int) -> None:
    """Measure which mantissa bits the NORM channel can and cannot catch, and check the derivation.

    The norm is a lossy reduction: the change a low-bit flip makes to one element disappears into
    the rounding of a sum over P elements. ``min_detectable_mantissa_bit`` predicts where that
    happens from ``(g_i/||g||)**2`` and the accumulator's unit roundoff; this asserts the predicted
    boundary against the measured one instead of hard-coding either.
    """
    model, optimizer, cfg = _dist_setup(rank, world_size, port, steps=1)
    victim = world_size - 1
    run_steps(
        cfg,
        _corpus(),
        model,
        optimizer,
        start_step=0,
        n_steps=1,
        hooks=StepHooks(after_reduce_check=lambda _s: _measure_boundary(model, rank, victim)),
    )
    dist.destroy_process_group()


def _measure_boundary(model: TransformerLM, rank: int, victim: int) -> None:
    index = largest_grad_index(model, FLIP_PARAM)  # identical on every rank: grads are reduced
    grad = dict(model.named_parameters())[FLIP_PARAM].grad
    assert grad is not None
    element = float(grad.reshape(-1)[index].item())
    found: dict[str, int] = {}
    derived: dict[str, int] = {}

    for label, accum in (("fp32", torch.float32), ("fp64", torch.float64)):
        check = CrossRankGradCheck(accum_dtype=accum, channels=frozenset({"norm"}))
        derived[label] = min_detectable_mantissa_bit(
            element, check.local_norm(model), accum_dtype=accum
        )
        for bit in range(23):
            if rank == victim:
                flip_grad_bit_(model, FLIP_PARAM, index, bit)
            fired = check.check(model).fired  # collective: every rank calls it
            if rank == victim:
                flip_grad_bit_(model, FLIP_PARAM, index, bit)
            if fired:
                found[label] = bit
                break

    assert "fp64" in found and "fp32" in found, f"nothing was catchable at all: {found}"
    # An fp64 accumulator catches the mantissa LSB on a model this size — "not only a catastrophic
    # flip". The bound scales as 1/P, so a 7B model loses roughly log2(P/37e3) ~ 18 of those bits.
    assert found["fp64"] == 0, f"fp64 norm missed the lsb (first caught bit {found['fp64']})"
    # An fp32 accumulator is blind to the low bits. That boundary is the derived one, to a bit.
    assert found["fp32"] > 0, "fp32 accumulation caught the lsb — the roundoff model is wrong"
    assert abs(found["fp32"] - derived["fp32"]) <= 1, (
        f"measured fp32 boundary {found['fp32']} vs derived {derived['fp32']} — the "
        "single-rounding model in min_detectable_mantissa_bit does not describe this sum"
    )
    assert found["fp32"] > found["fp64"], "the accumulator dtype did not move the boundary"

    # The bit BELOW the fp32 boundary: the norm channel cannot see it, the digest channel can.
    # That pair is the whole argument for carrying a lossless channel next to the named one.
    blind_bit = found["fp32"] - 1
    norm_only = CrossRankGradCheck(accum_dtype=torch.float32, channels=frozenset({"norm"}))
    lossless = CrossRankGradCheck(channels=frozenset({"digest"}))
    if rank == victim:
        flip_grad_bit_(model, FLIP_PARAM, index, blind_bit)
    norm_fired = norm_only.check(model).fired
    digest_report = lossless.check(model)
    if rank == victim:
        flip_grad_bit_(model, FLIP_PARAM, index, blind_bit)
    assert not norm_fired, f"fp32 norm saw bit {blind_bit}; the boundary is not where it measured"
    assert digest_report.fired and digest_report.first_param == FLIP_PARAM, digest_report.summary()


def test_the_norm_channel_boundary_is_where_the_roundoff_model_says_it_is() -> None:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _boundary_worker, args=(WORLD, _free_port()), nprocs=WORLD
    )


# ---------------------------------------------------------------------------------------------
# fault 1, for real: kill a rank mid-collective and resume across the death
# ---------------------------------------------------------------------------------------------


def _kill_worker(rank: int, world_size: int, port: int, kill_step: int) -> None:
    model, optimizer, cfg = _dist_setup(rank, world_size, port, steps=6)
    kill = RankKill(rank=world_size - 1, step=kill_step)
    run_steps(
        cfg,
        _corpus(),
        model,
        optimizer,
        start_step=0,
        n_steps=cfg.max_steps,
        hooks=StepHooks(before_step=lambda step: kill.maybe_kill(step=step, rank=rank)),
    )
    dist.destroy_process_group()


def test_a_dead_rank_is_visible_to_the_launcher_and_does_not_hang_the_survivors() -> None:
    """No in-process detector can catch this one — the dying rank runs no more Python and the
    survivors are blocked in a collective. The LAUNCHER is the thing that notices, which is why
    this fault's detection is a resume check and not a monitor."""
    with pytest.raises(ProcessExitedException) as excinfo:
        mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
            _kill_worker, args=(WORLD, _free_port(), 3), nprocs=WORLD
        )
    assert excinfo.value.exit_code == 137, "the rank did not die the way hardware kills one"
    assert excinfo.value.error_index == WORLD - 1, "the launcher blamed the wrong rank"


def _dist_resume_worker(rank: int, world_size: int, port: int, phase: str, workdir: str) -> None:
    steps, checkpoint_at = 8, 3
    model, optimizer, cfg = _dist_setup(rank, world_size, port, steps=steps)
    out = Path(workdir)

    if phase in ("reference", "crash"):
        hooks = StepHooks()
        if phase == "crash":
            kill = RankKill(rank=world_size - 1, step=checkpoint_at + 2)
            hooks.before_step = lambda step: kill.maybe_kill(step=step, rank=rank)
        first = run_steps(
            cfg, _corpus(), model, optimizer, start_step=0, n_steps=checkpoint_at, hooks=hooks
        )
        _snap(first.next_step, first.records[-1].lr, model, optimizer, first.batches_drawn).save(
            out / f"ckpt_{rank}.pt"
        )
        rest = run_steps(
            cfg,
            _corpus(),
            model,
            optimizer,
            start_step=first.next_step,
            n_steps=steps - checkpoint_at,
            hooks=hooks,
            batches_drawn=first.batches_drawn,
        )
        _snap(rest.next_step, rest.records[-1].lr, model, optimizer, rest.batches_drawn).save(
            out / f"ref_{rank}.pt"
        )
    else:
        snap = TrainSnapshot.load(out / f"ckpt_{rank}.pt")
        if phase == "resume_degraded":
            snap = without_rng(snap)
        start = restore(snap, model, optimizer)
        res = run_steps(
            cfg,
            _corpus(),
            model,
            optimizer,
            start_step=start,
            n_steps=steps - start,
            batches_drawn=snap.batches_drawn,
        )
        _snap(res.next_step, res.records[-1].lr, model, optimizer, res.batches_drawn).save(
            out / f"{phase}_{rank}.pt"
        )
    dist.destroy_process_group()


def test_a_resume_across_a_real_rank_death_is_bit_exact_and_the_verifier_still_discriminates(
    tmp_path: Path,
) -> None:
    """The rung's first row, end to end: run, kill a rank for real, relaunch, and prove the
    continuation is the same run — while proving in the same breath that the verifier would have
    said so if it were not (the RNG-dropping control must fire)."""
    work = str(tmp_path)
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _dist_resume_worker, args=(WORLD, _free_port(), "reference", work), nprocs=WORLD
    )
    with pytest.raises(ProcessExitedException) as excinfo:
        mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
            _dist_resume_worker, args=(WORLD, _free_port(), "crash", work), nprocs=WORLD
        )
    assert excinfo.value.exit_code == 137 and excinfo.value.error_index == WORLD - 1

    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _dist_resume_worker, args=(WORLD, _free_port(), "resume", work), nprocs=WORLD
    )
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _dist_resume_worker, args=(WORLD, _free_port(), "resume_degraded", work), nprocs=WORLD
    )

    for rank in range(WORLD):
        reference = TrainSnapshot.load(tmp_path / f"ref_{rank}.pt")
        good = verify_bit_exact(reference, TrainSnapshot.load(tmp_path / f"resume_{rank}.pt"))
        bad = verify_bit_exact(
            reference, TrainSnapshot.load(tmp_path / f"resume_degraded_{rank}.pt")
        )
        assert good.bit_exact, f"rank {rank}: {good.summary()}"
        assert bad.fired, f"rank {rank}: the RNG-dropping control did not fire — no discrimination"


# ---------------------------------------------------------------------------------------------
# the box
# ---------------------------------------------------------------------------------------------


@pytest.mark.gpu
def test_the_three_detections_hold_on_real_gpu_ranks() -> None:
    """gloo on CPU proves the detection logic. It does not prove NCCL's ring is rank-symmetric
    (the whole basis of the threshold-free comparison), and it does not kill a rank that owns a
    CUDA context mid-kernel. Those need the box."""
    if torch.cuda.device_count() < 2:
        pytest.skip(
            "needs >= 2 CUDA devices — NCCL rank-symmetry and a real GPU rank death are not "
            "reproducible on CPU/gloo. On the box:\n" + BOX_COMMAND
        )
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "bench" / "t1_faults.py"),
            "--mode",
            "fault",
            "--spike-z",
            "inf",
            "--world-size",
            str(min(8, torch.cuda.device_count())),
            "--device",
            "cuda",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "3", proc.stdout
