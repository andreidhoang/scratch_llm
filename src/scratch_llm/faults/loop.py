"""The instrument's training loop — the smallest loop that can be interrupted and resumed.

``scratch_llm.train.train`` cannot express this rung's fault: its own docstring says a reload
"always begins at step 0 with a fresh warmup", so there is no ``start_step`` to resume at and no
place to hang a fault hook. Rather than change it — the production loop is Huy's — this is a loop
made of ``train()``'s own parts (``get_batch``, ``cross_entropy``, ``gradient_clipping``,
``cosine_lr``, ``build_optimizer``) in ``train()``'s own order, with two things added:

* ``start_step`` / ``batches_drawn``, so a run can continue where another stopped;
* :class:`StepHooks` at the five places a fault or a detector belongs — before the step (rank
  death), on the batch, after backward (a PRE-reduction flip), after the reduction (a POST-reduction
  flip, then the cross-rank check), and on the reduced loss (the spike decision).

``tests/training/test_t1_t_r3.py::test_the_instrument_loop_reproduces_train_step_for_step`` pins it
to ``train()`` bit for bit on the single-process dense path. Every feature it cannot mirror —
MoE, MTP, QK-clip, autocast, ``torch.compile`` — raises rather than silently differing, because an
instrument that is quietly a different algorithm proves nothing about the thing it instruments.

**It is DDP-shaped, not ZeRO-2.** Gradients all-reduce and every rank runs the whole optimizer, so
"the post-reduction gradient" exists identically on every rank and the cross-rank check has
something exact to compare. ``train()``'s distributed path is ``DistMuonAdamW`` (reduce-scatter,
sharded state); its analogue of this check is ``assert_model_replicas_identical``.

**It never seeds.** Seeding is the caller's job, because ``seed_everything`` in the middle of a
resume is precisely what turns a resume into a different run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from scratch_llm.faults.spike import SpikeAction
from scratch_llm.model import TransformerLM, cross_entropy
from scratch_llm.optim import cosine_lr, gradient_clipping
from scratch_llm.train import TrainConfig, get_batch

AnyOptimizer = torch.optim.Optimizer


def pin_determinism() -> None:
    """One thread. Cross-*process* bit-exactness is what the resume verifier measures, and a
    different intra-op thread count changes the reduction tree, which looks exactly like a resume
    bug. Call it in every worker before building anything."""
    torch.set_num_threads(1)


@dataclass
class StepHooks:
    """The five injection/inspection points. Every one defaults to nothing, and a loop with no
    hooks is byte-identical to one compiled without them (tested)."""

    before_step: Callable[[int], None] | None = None
    on_batch: Callable[[int, Tensor, Tensor], None] | None = None
    after_backward: Callable[[int], None] | None = None  # PRE-reduction: the SDC blind spot
    after_reduce: Callable[[int], None] | None = None  # POST-reduction: where a flip is visible
    after_reduce_check: Callable[[int], None] | None = None  # detectors run here, after the flip
    on_loss: Callable[[int, float], SpikeAction] | None = None
    on_step_end: Callable[[int, StepRecord], None] | None = None


@dataclass(frozen=True)
class StepRecord:
    step: int
    loss: float
    lr: float
    grad_norm: float
    action: str  # SpikeAction.value: "continue" | "skip" | "restart" | "halt"


@dataclass
class LoopResult:
    records: list[StepRecord] = field(default_factory=list)
    batches_drawn: int = 0
    next_step: int = 0
    stop_reason: str = "completed"

    @property
    def losses(self) -> list[float]:
        return [r.loss for r in self.records]


def _reject_unsupported(cfg: TrainConfig, model: TransformerLM) -> None:
    """Refuse anything the loop cannot mirror from ``train()`` exactly."""
    if cfg.amp_dtype is not None:
        raise ValueError("faults.loop mirrors train()'s fp32 dense path; amp_dtype must be None")
    if cfg.compile:
        raise ValueError("faults.loop does not compile — a resume must compare eager to eager")
    if cfg.qk_clip:
        raise ValueError("faults.loop does not implement QK-clip; it would diverge from train()")
    if model.cfg.moe is not None or model.cfg.mtp_depth > 0:
        raise ValueError("faults.loop mirrors the dense path only (no MoE aux, no MTP head)")


def _all_reduce_grads(model: TransformerLM) -> None:
    """DDP-shaped gradient averaging, in NAME order so every rank reduces in the same sequence.

    After this every rank holds bitwise-identical gradients — that is what makes the cross-rank
    check threshold-free, and it is measured, not assumed (see faults/gradnorm.py).

    A missing gradient is MATERIALISED as zeros rather than skipped, which is what DDP does and
    which matters more here than it looks. ``if p.grad is not None: all_reduce(...)`` makes the
    number of collectives a function of each rank's own graph, so one rank taking a different
    branch — a dead expert, a masked-out loss term, an injected fault that severs the graph —
    issues fewer all-reduces than its peers and the job hangs in a collective forever with no
    error. That hang is the first thing this loop reproduced by accident; symmetric participation
    by construction is the fix.
    """
    world = dist.get_world_size()
    for _, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)


def run_steps(
    cfg: TrainConfig,
    train_data: np.ndarray,
    model: TransformerLM,
    optimizer: AnyOptimizer,
    *,
    start_step: int,
    n_steps: int,
    hooks: StepHooks | None = None,
    batches_drawn: int = 0,
) -> LoopResult:
    """Run ``n_steps`` steps beginning at ``start_step``. Returns the per-step record.

    The LR comes from ``cosine_lr(step, ...)`` with the ABSOLUTE step, so the schedule position is
    carried by ``start_step`` alone — that is the whole reason a resumed run's LR can be verified
    against the uninterrupted one instead of hoped about.
    """
    _reject_unsupported(cfg, model)
    hooks = hooks or StepHooks()
    distributed = dist.is_available() and dist.is_initialized()
    result = LoopResult(batches_drawn=batches_drawn, next_step=start_step)
    model.to(cfg.device)

    for step in range(start_step, start_step + n_steps):
        if hooks.before_step is not None:
            hooks.before_step(step)  # RankKill lands here — this call may never return
        lr = cosine_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        result.batches_drawn += 1
        if hooks.on_batch is not None:
            hooks.on_batch(step, inputs, targets)

        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        if hooks.after_backward is not None:
            hooks.after_backward(step)

        if distributed:
            _all_reduce_grads(model)
        if hooks.after_reduce is not None:
            hooks.after_reduce(step)
        if hooks.after_reduce_check is not None:
            hooks.after_reduce_check(step)

        loss_value = float(loss.item())
        if distributed:
            # The skip decision MUST be made on the same number on every rank: one rank skipping
            # while the others step desyncs the replicas, which is worse than the NaN it dodged.
            # AVG propagates a single rank's NaN to all of them, which is the wanted behaviour.
            scalar = torch.tensor([loss_value], device=cfg.device)
            dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
            loss_value = float(scalar[0].item()) / dist.get_world_size()

        action = SpikeAction.CONTINUE
        if hooks.on_loss is not None:
            action = hooks.on_loss(step, loss_value)

        grad_norm = float("nan")
        if action is SpikeAction.CONTINUE:
            grad_norm = float(gradient_clipping(model.parameters(), cfg.grad_clip).item())
            optimizer.step()
        else:
            # Drop the poisoned gradients on the floor. Not stepping is the point: a NaN that
            # reaches AdamW's exp_avg poisons every future step, not just this one.
            optimizer.zero_grad()

        record = StepRecord(
            step=step, loss=loss_value, lr=lr, grad_norm=grad_norm, action=action.value
        )
        result.records.append(record)
        result.next_step = step + 1
        if hooks.on_step_end is not None:
            hooks.on_step_end(step, record)
        if action in (SpikeAction.RESTART, SpikeAction.HALT):
            result.stop_reason = action.value
            return result

    return result


__all__ = ["LoopResult", "StepHooks", "StepRecord", "pin_determinism", "run_steps"]
