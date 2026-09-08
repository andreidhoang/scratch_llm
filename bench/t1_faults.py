"""T1/T-R3 driver — three induced failures, three detections, and the clean-run control.

    python bench/t1_faults.py --mode fault --spike-z inf     # the rung's number: detections /3
    python bench/t1_faults.py --mode clean --spike-z inf     # the floor:  silences /3

The last line of stdout is one bare number, which is the harness convention and what
``experiments/T1/T-R3/{run,floor}.sh`` read.

**The floor here is not a library.** There is no cuBLAS for "did you notice the rank died". The
thing this rung is judged against is the CLEAN RUN: the same three detectors, the same shapes, the
same number of steps, no faults injected, and every detector silent. A detector that fires on the
clean run has a false-positive rate, and a detector with a false-positive rate gets switched off by
the first engineer it wakes at 3am — at which point it detects nothing. ``--mode clean`` measures
that, and it is a measurement, so it gets a prediction and a ledger row like any other.

Why ``--spike-z`` is required and why the scripts pass ``inf``: the spike detector's magnitude
channel is the one place in this rung with a free operating point, and an operating point chosen by
the harness is a tolerance smuggled into a measurement. ``inf`` disarms it, leaving only the exact
channels — ``math.isfinite``, ``torch.equal`` on a byte view, and a bitwise digest — so the three
counted detections have no threshold behind them at all. The magnitude channel and its
skip/restart/livelock policy are exercised in ``tests/training/test_t1_t_r3.py`` with explicit
scenario values; its production z is Huy's to set.

Scenario 1 needs three launches (reference, crash, resume) plus a fourth for the discriminating
control, so the model is deliberately tiny: this measures detection logic, not throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException

from scratch_llm.faults.capsule import (
    TrainSnapshot,
    capture,
    restore,
    verify_bit_exact,
    without_rng,
)
from scratch_llm.faults.gradnorm import CrossRankGradCheck
from scratch_llm.faults.inject import NaNBatch, RankKill, flip_grad_bit_, largest_grad_index
from scratch_llm.faults.loop import StepHooks, pin_determinism, run_steps
from scratch_llm.faults.spike import LossSpikeDetector, SpikeAction
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.optim import build_optimizer
from scratch_llm.train import TrainConfig
from scratch_llm.utils.seeding import seed_everything

FLIP_PARAM = "token_emb.weight"
FLIP_BIT = 0  # mantissa lsb: the hardest single-bit flip there is, and the one norms lose


@dataclass(frozen=True)
class Job:
    """Everything a worker needs; must be picklable (spawn start method)."""

    workdir: str
    phase: str
    steps: int
    checkpoint_step: int
    fault_step: int
    seed: int
    device: str
    spike_z: float
    inject: bool


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Workers report back through the filesystem: a spawned rank cannot return a value."""
    path.write_text(json.dumps(payload, default=str))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _model_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, context_length=16)


def _train_cfg(job: Job) -> TrainConfig:
    return TrainConfig(
        max_steps=job.steps,
        batch_size=4,
        context_length=16,
        warmup_steps=2,
        log_every=1,
        device=job.device,
        seed=job.seed,
    )


def _corpus() -> np.ndarray:
    return (np.arange(8192, dtype=np.uint16) * 7 + 3) % 64


def _init(job: Job, rank: int, world: int, port: int) -> tuple[TransformerLM, Any, TrainConfig]:
    """Process-group init + the exact construction order ``train()`` uses, so the replicas are
    identical and each rank draws its own batch stream (``train.py:340-344``).

    ``set_device`` BEFORE any allocation, and a per-rank device string: without both, every rank
    builds on ``cuda:0`` and the job is a slow single-GPU run wearing a world size — the silent
    failure ``utils/dist_launch.py:35-37`` exists to prevent.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    cuda = job.device.startswith("cuda")
    if cuda:
        torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world)
    pin_determinism()
    cfg = _train_cfg(job)
    if cuda:
        cfg.device = f"cuda:{rank}"
    seed_everything(cfg.seed)
    model = TransformerLM(_model_cfg())  # identical on every rank: seeded before construction
    seed_everything(cfg.seed)
    if dist.is_initialized():
        np.random.seed(cfg.seed + 1000 * rank + 1)  # per-rank data shard, as train() does
    optimizer = build_optimizer(
        model, kind=cfg.optimizer, lr=cfg.max_lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )
    return model, optimizer, cfg


def _snap_path(job: Job, tag: str, rank: int) -> Path:
    return Path(job.workdir) / f"{tag}_rank{rank}.pt"


# ---------------------------------------------------------------------------------------------
# scenario 1 — kill a rank, resume bit-exact
# ---------------------------------------------------------------------------------------------


def _resume_worker(rank: int, world: int, port: int, job: Job) -> None:
    model, optimizer, cfg = _init(job, rank, world, port)
    corpus = _corpus()

    if job.phase in ("reference", "crash"):
        hooks = StepHooks()
        if job.phase == "crash":
            kill = RankKill(rank=world - 1, step=job.fault_step)
            hooks.before_step = lambda step: kill.maybe_kill(step=step, rank=rank)
        res = run_steps(
            cfg, corpus, model, optimizer, start_step=0, n_steps=job.checkpoint_step, hooks=hooks
        )
        capture(
            step=res.next_step,
            lr=res.records[-1].lr,
            model=model,
            optimizer=optimizer,
            batches_drawn=res.batches_drawn,
        ).save(_snap_path(job, "ckpt", rank))
        res2 = run_steps(
            cfg,
            corpus,
            model,
            optimizer,
            start_step=res.next_step,
            n_steps=job.steps - job.checkpoint_step,
            hooks=hooks,
            batches_drawn=res.batches_drawn,
        )
        capture(
            step=res2.next_step,
            lr=res2.records[-1].lr,
            model=model,
            optimizer=optimizer,
            batches_drawn=res2.batches_drawn,
        ).save(_snap_path(job, "ref", rank))
        return

    snap = TrainSnapshot.load(_snap_path(job, "ckpt", rank))
    tag = "resume"
    if job.phase == "resume_degraded":
        snap = without_rng(snap)  # torchtitan's checkpoint: everything but the RNG
        tag = "degraded"
    step = restore(snap, model, optimizer)
    res = run_steps(
        cfg,
        corpus,
        model,
        optimizer,
        start_step=step,
        n_steps=job.steps - step,
        batches_drawn=snap.batches_drawn,
    )
    capture(
        step=res.next_step,
        lr=res.records[-1].lr,
        model=model,
        optimizer=optimizer,
        batches_drawn=res.batches_drawn,
    ).save(_snap_path(job, tag, rank))


def _launch(job: Job, world: int, phase: str) -> None:
    print(f"  [{phase}] launching {world} ranks", flush=True)
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _resume_worker, args=(world, _free_port(), _replace(job, phase=phase)), nprocs=world
    )
    print(f"  [{phase}] done", flush=True)


def _scenario_resume(job: Job, world: int) -> dict[str, Any]:
    out: dict[str, Any] = {"scenario": "kill_rank_resume_bit_exact"}
    _launch(job, world, "reference")

    if job.inject:
        try:
            _launch(job, world, "crash")
            out["rank_death_seen"] = False
            out["rank_death_note"] = "the killed rank returned — the injector did not inject"
        except ProcessExitedException as exc:
            out["rank_death_seen"] = exc.exit_code == 137 and exc.error_index == world - 1
            out["rank_death"] = f"exit_code={exc.exit_code} rank={exc.error_index}"
            print(f"  [crash] {out['rank_death']}", flush=True)

    _launch(job, world, "resume")
    reports = [
        verify_bit_exact(
            TrainSnapshot.load(_snap_path(job, "ref", r)),
            TrainSnapshot.load(_snap_path(job, "resume", r)),
        )
        for r in range(world)
    ]
    out["resume_bit_exact"] = all(r.bit_exact for r in reports)
    out["resume_detail"] = [r.summary() for r in reports]

    if job.inject:
        _launch(job, world, "resume_degraded")
        degraded = [
            verify_bit_exact(
                TrainSnapshot.load(_snap_path(job, "ref", r)),
                TrainSnapshot.load(_snap_path(job, "degraded", r)),
            )
            for r in range(world)
        ]
        out["verifier_discriminates"] = all(d.fired for d in degraded)
        out["degraded_detail"] = [d.summary() for d in degraded]
        out["passed"] = bool(
            out["rank_death_seen"] and out["resume_bit_exact"] and out["verifier_discriminates"]
        )
    else:
        out["passed"] = bool(out["resume_bit_exact"])  # clean mode: silence is the pass
    return out


# ---------------------------------------------------------------------------------------------
# scenario 2 — NaN batch, spike detector skips before the optimizer sees it
# ---------------------------------------------------------------------------------------------


def _nan_worker(rank: int, world: int, port: int, job: Job) -> None:
    model, optimizer, cfg = _init(job, rank, world, port)
    detector = LossSpikeDetector(
        spike_z=job.spike_z,
        window=8,
        min_observations=4,
        max_consecutive_skips=2,
        max_restarts=1,
    )
    poison = NaNBatch(steps={job.fault_step} if job.inject else set())
    handle = poison.attach(model.token_emb) if rank == world - 1 else None

    hooks = StepHooks()

    def on_batch(step: int, _i: torch.Tensor, _t: torch.Tensor) -> None:
        poison.current_step = step

    # The hook must know the step BEFORE the forward runs; on_batch fires between get_batch and
    # the forward, which is exactly that window.
    hooks.on_batch = on_batch
    hooks.on_loss = detector.observe
    res = run_steps(cfg, _corpus(), model, optimizer, start_step=0, n_steps=job.steps, hooks=hooks)
    if handle is not None:
        handle.remove()

    finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    if rank == 0:
        skipped = [e.step for e in detector.events if e.action is SpikeAction.SKIP]
        _write(
            Path(job.workdir) / "nan.json",
            {
                "fired": detector.fired,
                "reasons": sorted({e.reason for e in detector.events}),
                "skipped_steps": skipped,
                "steps_run": len(res.records),
                "params_finite": bool(finite),
            },
        )
    if rank == world - 1:
        # The victim reports its own injection: rank 0 never attaches the poison, so reading
        # "fired_at: []" off rank 0 would look like the injector failed when it did not.
        _write(Path(job.workdir) / "nan_inj.json", {"poisoned_steps": poison.fired_at})


def _scenario_nan(job: Job, world: int) -> dict[str, Any]:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _nan_worker, args=(world, _free_port(), job), nprocs=world
    )
    r = _read(Path(job.workdir) / "nan.json")
    out: dict[str, Any] = {"scenario": "nan_batch_spike_skip", **r}
    out["injected"] = _read(Path(job.workdir) / "nan_inj.json")["poisoned_steps"]
    if job.inject:
        # Fired, for the right reason, at the injected step, the step was NOT taken, the run
        # survived it, and no NaN reached a parameter.
        out["passed"] = bool(
            r["fired"]
            and r["reasons"] == ["nonfinite"]
            and r["skipped_steps"] == [job.fault_step]
            and r["steps_run"] == job.steps
            and r["params_finite"]
            and out["injected"] == [job.fault_step]
        )
    else:
        out["passed"] = bool(
            not r["fired"]
            and r["steps_run"] == job.steps
            and r["params_finite"]
            and out["injected"] == []
        )
    return out


# ---------------------------------------------------------------------------------------------
# scenario 3 — one bit of one gradient on one rank
# ---------------------------------------------------------------------------------------------


def _flip_worker(rank: int, world: int, port: int, job: Job) -> None:
    model, optimizer, cfg = _init(job, rank, world, port)
    check = CrossRankGradCheck()
    victim = world - 1
    flips: list[str] = []

    hooks = StepHooks()

    def after_reduce(step: int) -> None:
        # POST-reduction on purpose: before the all-reduce the corruption propagates to every rank
        # and no cross-rank comparison can see it (see faults/gradnorm.py, and torchtitan's
        # sdc_replayer for the technique that can).
        if job.inject and rank == victim and step == job.fault_step:
            idx = largest_grad_index(model, FLIP_PARAM)
            flips.append(str(flip_grad_bit_(model, FLIP_PARAM, idx, FLIP_BIT)))

    hooks.after_reduce = after_reduce
    hooks.after_reduce_check = lambda step: check.check(model)
    run_steps(cfg, _corpus(), model, optimizer, start_step=0, n_steps=job.steps, hooks=hooks)

    if rank == 0:
        fired = [i for i, rep in enumerate(check.reports) if rep.fired]
        _write(
            Path(job.workdir) / "flip.json",
            {
                "fired_steps": fired,
                "channels": sorted({c for rep in check.reports for c in rep.channels}),
                "first_param": next(
                    (rep.first_param for rep in check.reports if rep.fired),
                    None,
                ),
                "summary": [check.reports[i].summary() for i in fired],
                "steps_run": len(check.reports),
            },
        )
    if rank == victim and flips:
        _write(Path(job.workdir) / "flip_injected.json", {"flips": flips})


def _scenario_flip(job: Job, world: int) -> dict[str, Any]:
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _flip_worker, args=(world, _free_port(), job), nprocs=world
    )
    r = _read(Path(job.workdir) / "flip.json")
    out: dict[str, Any] = {"scenario": "grad_bit_flip_cross_rank", **r}
    injected = Path(job.workdir) / "flip_injected.json"
    if injected.exists():
        out["injected"] = _read(injected)["flips"]
    if job.inject:
        out["passed"] = bool(
            r["fired_steps"] == [job.fault_step]
            and "digest" in r["channels"]
            and r["first_param"] == FLIP_PARAM
        )
    else:
        out["passed"] = bool(not r["fired_steps"] and r["steps_run"] == job.steps)
    return out


# ---------------------------------------------------------------------------------------------


def _replace(job: Job, **kw: Any) -> Job:
    return Job(**{**job.__dict__, **kw})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("fault", "clean"),
        required=True,
        help="fault: inject and count detections. clean: inject nothing and count silences.",
    )
    ap.add_argument(
        "--spike-z",
        type=float,
        required=True,
        help="robust-z operating point of the spike detector's MAGNITUDE channel. No default on "
        "purpose; pass inf to disarm it and count only the exact channels.",
    )
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--checkpoint-step", type=int, default=5)
    ap.add_argument("--fault-step", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    world = args.world_size
    if world < 2:
        raise SystemExit(
            "T-R3 needs >= 2 ranks: a rank cannot die on itself and a single rank has nobody to "
            "disagree with. Re-run with --world-size 2 (gloo/CPU is enough)."
        )
    if args.device.startswith("cuda") and torch.cuda.device_count() < world:
        raise SystemExit(
            f"--device cuda --world-size {world} needs {world} visible GPUs "
            f"(saw {torch.cuda.device_count()}). On the box:\n"
            f"  experiments/T1/T-R3/run.sh          # picks world size from nvidia-smi"
        )

    with tempfile.TemporaryDirectory(prefix="t1_r3_") as tmp:
        job = Job(
            workdir=tmp,
            phase="",
            steps=args.steps,
            checkpoint_step=args.checkpoint_step,
            fault_step=args.fault_step,
            seed=args.seed,
            device=args.device,
            spike_z=args.spike_z,
            inject=args.mode == "fault",
        )
        results = [
            _scenario_resume(job, world),
            _scenario_nan(job, world),
            _scenario_flip(job, world),
        ]

    verb = "detections" if job.inject else "silences"
    print(f"# T1/T-R3 mode={args.mode} world={world} device={args.device} steps={args.steps}")
    print(f"# torch {torch.__version__} | spike_z={args.spike_z} | seed={args.seed}")
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['scenario']}")
        for key, value in r.items():
            if key not in ("scenario", "passed"):
                print(f"        {key}: {value}")
    passed = sum(1 for r in results if r["passed"])
    print(f"# {verb}: {passed}/3")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "metric": "detections_of_3" if job.inject else "clean_run_silent_of_3",
                    "value": passed,
                    "world_size": world,
                    "device": args.device,
                    "torch": torch.__version__,
                    "scenarios": results,
                },
                indent=1,
                default=str,
            )
        )
    print(passed)  # the bare number, last line — the harness convention
    return 0


if __name__ == "__main__":
    sys.exit(main())
