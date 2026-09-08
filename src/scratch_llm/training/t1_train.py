"""T1/T-R1 — the trainer whose tok/s is measured against T-R0's torchtitan floor.

Launched by ``experiments/T1/T-R1/run.sh`` under ``torchrun --nproc_per_node=8``. It is a sibling
of :mod:`scratch_llm.train`, not a replacement: that loop's distributed path is optimizer-embedded
ZeRO-2 (``DistMuonAdamW``, ``train.py:346-369``), which owns its own reduce-scatter and cannot
take DTensor parameters. FSDP2 + TP is a different substrate, so it gets a different entry point
and reuses everything that is not about parallelism —
:func:`~scratch_llm.train.get_batch`, :func:`~scratch_llm.optim.cosine_lr`,
:class:`~scratch_llm.model.TransformerLM`, :func:`~scratch_llm.model.cross_entropy`.

THE SHAPE OF A STEP, and where each named field of the logging spec comes from:

    forward   (autocast bf16)      -> loss
    backward                       -> grads, as DTensors on the (dp, tp) mesh
    grad_norms(model)              -> grad_norm_global + grad_norm_groups   [one collective]
    clip by that same norm         -> the logged norm IS the norm used to clip, by construction
    optimizer.step()               -> fused AdamW on fp32 master shards
    checkpointer.save() if due     -> staged synchronously, written in the background
    logger.log_step(...)           -> validated against REQUIRED_STEP_FIELDS

MEASUREMENT. ``cfg.warmup_steps`` steps are discarded (step 1 pays torch.compile and FSDP's
first all-gather; the allocator settles over the next few), then the median and IQR of the
remaining ``cfg.measure_steps`` per-step tok/s is the rung's number — the same window and the
same reduction ``titan_floor.summarize`` applies to torchtitan's own log, so the two medians are
comparable reductions of comparable quantities.

THE RUNG'S METRIC is ``pct_of_torchtitan_tps`` = 100 x ours / T-R0's ``fsdp4_tp2`` floor. It is a
ratio, so it needs the floor: ``--floor-tps`` is mandatory and the run refuses without it rather
than printing a bare tok/s that a reader would mistake for the rung's number.

This module cannot run until the hole in ``parallel_plan.py`` is filled. That is deliberate:
there is no default plan to fall back to, so no number can be produced by an unmade decision.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from scratch_llm.model import TransformerLM, cross_entropy
from scratch_llm.optim import cosine_lr
from scratch_llm.train import get_batch
from scratch_llm.training.dcp_async import AsyncCheckpointer, TrainState
from scratch_llm.training.flops import flop_breakdown, step_report
from scratch_llm.training.logging_spec import (
    PERIODIC_EVERY,
    ActivationProbe,
    CommWaitTimer,
    RunLogger,
    grad_norms,
    weight_norms,
)
from scratch_llm.training.mp_policy import autocast_dtype
from scratch_llm.training.parallel_plan import ParallelPlan, t1_parallel_plan
from scratch_llm.training.parallelize import parallelize
from scratch_llm.training.rung_config import H100_SXM_PEAK_BF16, T_R1, T1RungConfig
from scratch_llm.training.titan_floor import summarize
from scratch_llm.utils.seeding import seed_everything

RUNG = "T1/T-R1"
METRIC = "pct_of_torchtitan_tps"

# Synthetic tokens. torchtitan's floor reads c4_test (T-R0 spec, "the floor has no network in
# it"); ours draws uniform ids from the same vocab. Token *values* change neither the shapes nor
# the FLOPs — every GEMM is 128256-wide either way — and fixing them by seed makes the batch
# stream reproducible without a dataset on the box. It does mean the loss curve is meaningless
# here (uniform tokens are unlearnable); this rung measures throughput, and T-R2 is where loss
# comparability is the point. Sized from the config so the CPU twin does not allocate 32 MB of
# int64 per rank to draw 16-token windows out of.
SYNTHETIC_TOKEN_MULTIPLE = 64  # x seq_len


def _dist_env() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def _clip_grads_(model: nn.Module, max_norm: float, total_norm: float) -> None:
    """Scale every gradient by ``max_norm / total_norm`` when the norm exceeds the cap.

    ``total_norm`` is passed in rather than recomputed: it is the same number the logging spec
    reports, so "the norm we logged" and "the norm we clipped by" cannot drift. Works on DTensor
    grads because ``mul_`` by a Python float is elementwise and placement-preserving.
    """
    if total_norm <= max_norm or total_norm == 0.0:
        return
    scale = max_norm / (total_norm + 1e-6)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)


def build_optimizer(model: nn.Module, lr: float, device_type: str) -> torch.optim.Optimizer:
    """Fused AdamW on CUDA, foreach elsewhere.

    ``fused=True`` writes the whole Adam update in one kernel per parameter group instead of ~10
    elementwise kernels, which at 1.24 B parameters is the difference between the optimizer being
    invisible and being a few percent of the step. It supports DTensor parameters, which is why
    the ``replicate`` TP style exists (see ``tensor_parallel.py`` FIXUP 2): a fused optimizer over
    a mix of DTensor and plain tensors does not.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=(device_type == "cuda"),
        foreach=(device_type != "cuda"),
    )


def _peak_memory_bytes(device_type: str) -> int:
    return int(torch.cuda.max_memory_allocated()) if device_type == "cuda" else 0


def train_t_r1(
    cfg: T1RungConfig,
    *,
    device_type: str,
    out_dir: Path,
    floor_tps: float,
    checkpoint_every: int,
    compile_model: bool,
    lr: float = 3e-4,
    grad_clip: float = 1.0,
    plan: ParallelPlan | None = None,
) -> dict[str, Any]:
    """Run the rung. Returns the result dict ``run.sh`` writes to JSON.

    ``plan`` is a *test seam*, not a knob: ``None`` (the only value the CLI can produce) calls
    :func:`~scratch_llm.training.parallel_plan.t1_parallel_plan`, so no measurement can be made
    from an unmade decision. ``tests/training/test_t1_t_r1.py`` passes an explicit plumbing plan
    so that every line of this loop — the logger wiring, the probe arming, the tps formula, the
    result dict — executes on four gloo ranks before it executes on eight H100s.
    """
    rank, world_size, local_rank = _dist_env()
    if world_size != cfg.world_size:
        raise SystemExit(
            f"{RUNG}: launched with WORLD_SIZE={world_size} but the config pins "
            f"{cfg.world_size}. A tok/s at a different world size is a different number."
        )
    device = f"cuda:{local_rank}" if device_type == "cuda" else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device_type == "cuda" else "gloo")

    seed_everything(cfg.seed)
    model = TransformerLM(cfg.model_config()).to(device)

    # The hole. No default, no fallback: an unmade parallelism decision cannot produce a number.
    plan = plan if plan is not None else t1_parallel_plan(cfg)
    model, _mesh, report = parallelize(
        model, plan, cfg, device_type=device_type, compile_model=compile_model
    )

    optimizer = build_optimizer(model, lr, device_type)
    train_state = TrainState()
    checkpointer = AsyncCheckpointer(
        model, optimizer, train_state, out_dir / "ckpt", every=checkpoint_every
    )
    probe = ActivationProbe(model)
    comm = CommWaitTimer()

    active_params = sum(p.numel() for p in model.parameters())
    # torchtitan's numerator, computed once from the shape (it has no run-time dependence).
    flops_per_token = flop_breakdown(
        cfg.shape, cfg.seq_len, ac_policy=cfg.ac_policy
    ).model_per_token

    config_log = {
        **cfg.as_log_dict(),
        "plan": plan.as_log_dict(),
        "applied": report.as_log_dict(),
        "flops_per_token": flops_per_token,
        "peak_flops_per_device": H100_SXM_PEAK_BF16,
        "data": "synthetic-uniform-token-ids",
        "trainable_params": active_params,
        "compile_model": compile_model,
        "host": socket.gethostname(),
        "torch": torch.__version__,
    }
    logger = RunLogger(
        config=config_log,
        total_steps=cfg.total_steps,
        periodic_every=PERIODIC_EVERY,
        out_path=(out_dir / f"steps.rank{rank}.jsonl") if rank == 0 else None,
    )

    rng = np.random.default_rng(cfg.seed + 1000 * rank)
    corpus_len = SYNTHETIC_TOKEN_MULTIPLE * cfg.seq_len
    corpus = rng.integers(0, cfg.shape.vocab_size, size=corpus_len, dtype=np.int64)
    amp = autocast_dtype(cfg.param_dtype)

    tps_series: list[float] = []
    for step in range(cfg.total_steps):
        periodic = logger.is_periodic(step)
        probe.arm(periodic)
        lr_now = cosine_lr(step, lr, lr / 10, cfg.warmup_steps, cfg.total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        inputs, targets = get_batch(corpus, cfg.micro_batch, cfg.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.autocast(device_type=device_type, dtype=amp) if amp else nullcontext()
        with ctx:
            loss = cross_entropy(model(inputs), targets)
        loss.backward()

        total_norm, groups = grad_norms(model)
        _clip_grads_(model, grad_clip, total_norm)
        optimizer.step()
        train_state.tokens_seen += cfg.global_tokens_per_step

        with comm.measure():
            # SUM then divide, not ReduceOp.AVG: gloo has no AVG, and the CPU twin of this loop
            # is the only place the plumbing is tested before the box.
            loss_t = loss.detach().float()
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            loss_value = float(loss_t.item()) / cfg.world_size

        if checkpointer.due(step):
            checkpointer.save(step)

        if device_type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - t0
        tps = cfg.global_tokens_per_step / (step_time * cfg.world_size)
        if step >= cfg.warmup_steps:
            tps_series.append(tps)

        record = logger.log_step(
            step=step,
            loss=loss_value,
            lr=lr_now,
            grad_norm_global=total_norm,
            grad_norm_groups=groups,
            tokens_per_s_per_device=tps,
            mfu=flops_per_token * tps / H100_SXM_PEAK_BF16,
            nccl_wait_s=comm.take(),
            step_time_s=step_time,
            peak_memory_bytes=_peak_memory_bytes(device_type),
            activation_max_abs=probe.snapshot() if periodic else None,
            weight_norms_=weight_norms(model) if periodic else None,
        )
        if rank == 0 and (step % 10 == 0 or periodic):
            sys.stderr.write(
                f"step {record['step']:4d} loss {record['loss']:.4f} "
                f"gnorm {record['grad_norm_global']:.3f} tps {record['tokens_per_s_per_device']:.0f} "
                f"mfu {100 * record['mfu']:.2f}%\n"
            )

    checkpointer.close()
    probe_points = probe.points
    probe.remove()
    logger.close()

    tps_stats = summarize(tps_series)
    final = step_report(
        cfg.shape,
        seq_len=cfg.seq_len,
        tokens_per_s_per_device=tps_stats["median"],
        peak_flops_per_device=H100_SXM_PEAK_BF16,
        ac_policy=cfg.ac_policy,
    )
    result = {
        "rung": RUNG,
        "metric": METRIC,
        "config": config_log,
        "measured_steps": len(tps_series),
        "tps_per_device": tps_stats,
        "floor_metric": cfg.floor_metric,
        "floor_tps": floor_tps,
        "pct_of_torchtitan_tps": 100.0 * tps_stats["median"] / floor_tps,
        "accounting": final.as_dict(),
        "checkpoint_stage_seconds": checkpointer.last_stage_seconds,
        "checkpoint_write_seconds": checkpointer.last_write_seconds,
        "activation_probe_points": probe_points,
    }
    dist.barrier()
    return result


def torchrun_command(nproc: int, worker_args: list[str]) -> list[str]:
    """The exact argv, mirroring ``oss/torchtitan/run_train.sh:43-45`` and T-R0's launcher."""
    return [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint=localhost:0",
        "--local-ranks-filter",
        "0",
        "--role",
        "rank",
        "--tee",
        "3",
        "-m",
        "scratch_llm.training.t1_train",
        *worker_args,
    ]


def _launch(args: argparse.Namespace) -> int:
    """Run the workers under torchrun, then print the one bare number ourselves.

    Why this exists rather than letting ``run.sh`` invoke ``torchrun`` directly: with
    ``--tee 3`` torchrun prefixes every worker line with ``[rank0]:``, so the harness convention
    ("the last line of stdout is the number") would hand ``make measure`` the string
    ``[rank0]:85.23``. T-R0's floor has the same shape for the same reason
    (``titan_floor.measure``). The workers write ``--json``; this process reads it and prints.
    """
    worker_args = [
        "--floor-tps",
        str(args.floor_tps),
        "--out",
        str(args.out),
        "--json",
        str(args.json),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--device",
        args.device,
        *(["--no-compile"] if args.no_compile else []),
    ]
    cmd = torchrun_command(args.nproc, worker_args)
    env = dict(os.environ)
    env["PYTORCH_ALLOC_CONF"] = env.get("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src
    if args.dry_run:
        print("# " + " ".join(cmd))
        return 0
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        sys.stderr.write(f"{RUNG}: torchrun exited {proc.returncode}\n")
        return proc.returncode
    result = json.loads(Path(args.json).read_text(encoding="utf-8"))
    tps = result["tps_per_device"]
    acc = result["accounting"]
    sys.stderr.write(
        f"{METRIC}: {result['pct_of_torchtitan_tps']:.2f}%  "
        f"(ours {tps['median']:.1f} tok/s/GPU, IQR {tps['iqr_pct']:.2f}%, "
        f"floor {result['floor_tps']:.1f} = {result['floor_metric']})\n"
        f"  MFU model (torchtitan convention) {100 * acc['mfu_model_torchtitan']:.2f}%\n"
    )
    # The harness convention: the last line of stdout is the one bare number.
    print(f"{result['pct_of_torchtitan_tps']:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{RUNG} — FSDP2 + TP=2 trainer; prints {METRIC}.")
    ap.add_argument(
        "--floor-tps",
        type=float,
        default=os.environ.get("T1_FLOOR_TPS"),
        help="T-R0's titan_llama3_1b_fsdp4_tp2_tps. Required: this rung's metric is a ratio.",
    )
    ap.add_argument("--out", default=None, help="directory for logs and checkpoints")
    ap.add_argument("--json", default=None, help="write the result dict here")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="0 = never (the timed run)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--nproc", type=int, default=T_R1.world_size)
    ap.add_argument("--dry-run", action="store_true", help="print the torchrun line, run nothing")
    args = ap.parse_args(argv)

    if args.floor_tps is None:
        sys.stderr.write(
            f"{RUNG}: REFUSED — no floor. {METRIC} is a ratio; without T-R0's\n"
            f"  {T_R1.floor_metric} it is not a number (workspace invariant 2). Measure it:\n"
            "    PRESET=fsdp4_tp2 experiments/T1/T-R0/floor.sh\n"
            "  then re-run with --floor-tps <value> (or T1_FLOOR_TPS=<value>).\n"
        )
        return 2

    args.out = Path(args.out) if args.out else Path.cwd() / "t1_r1_out"
    args.json = Path(args.json) if args.json else Path(args.out) / "latest.json"
    if os.environ.get("WORLD_SIZE") is None:
        # Not under torchrun: we are the launcher.
        return _launch(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = train_t_r1(
        T_R1,
        device_type=args.device,
        out_dir=out_dir,
        floor_tps=float(args.floor_tps),
        checkpoint_every=args.checkpoint_every,
        compile_model=not args.no_compile,
    )
    rank, _, _ = _dist_env()
    if rank == 0:
        # Rank 0 writes; the LAUNCHER prints the bare number (see _launch). A worker printing it
        # under `torchrun --tee 3` would emit "[rank0]:85.23", which is not a number.
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
