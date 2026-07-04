"""Data loading, checkpointing, and the training loop — the plumbing that turns the
model + optimizer into a trained policy, and (crucially) the same plumbing an RL run
reuses to resume from a checkpoint.

L1 substrate (A1). ``get_batch`` reads (input, next-token) windows from a flat token
array (``np.memmap`` so a corpus larger than RAM never loads fully); ``save/load_checkpoint``
round-trips model + optimizer + step so a run resumes exactly.

Correctness invariants (tested in tests/test_train.py):
- **Next-token alignment:** ``targets`` is ``inputs`` shifted by one position.
- **Checkpoint round-trip:** save → load restores the step and every parameter.
- **Reproducibility:** two runs with the same seed produce an identical loss history.
- **It learns:** on a structured corpus the loss drops well below log(vocab_size).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from scratch_llm.model import TransformerLM, cross_entropy
from scratch_llm.optim import (
    CombinedOptimizer,
    build_optimizer,
    cosine_lr,
    gradient_clipping,
)
from scratch_llm.utils.seeding import seed_everything


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Sample ``batch_size`` random (input, next-token) windows of length ``context_length``.

    Returns ``(inputs, targets)`` as long tensors of shape (batch_size, context_length),
    where ``targets[b, t] == inputs[b, t+1]`` in the underlying stream.
    """
    max_start = len(data) - context_length - 1
    if max_start < 1:
        raise ValueError(
            f"corpus of {len(data)} tokens too short for context_length={context_length}"
        )
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    inputs = np.stack([data[s : s + context_length] for s in starts])
    targets = np.stack([data[s + 1 : s + 1 + context_length] for s in starts])
    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(targets).long().to(device),
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | CombinedOptimizer,
    step: int,
    out: str | Path,
) -> None:
    torch.save(
        {"model": model.state_dict(), "optim": optimizer.state_dict(), "step": step},
        out,
    )


def load_checkpoint(
    src: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | CombinedOptimizer | None = None,
    map_location: str = "cpu",
) -> int:
    """Restore model (and optionally optimizer) state; return the saved step."""
    # weights_only=False: we load our own trusted checkpoints, which include the
    # optimizer's param-group metadata (not just tensors).
    ckpt = torch.load(src, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim"])
    return int(ckpt["step"])


@dataclass
class TrainConfig:
    max_steps: int
    batch_size: int
    context_length: int
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 0
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    log_every: int = 10
    checkpoint_every: int = 0  # 0 = never
    checkpoint_path: str | None = None
    device: str = "cpu"
    seed: int = 0
    verbose: bool = False
    # F1/F4 — close-the-loop knobs (defaults reproduce the A1 AdamW/fp32 path exactly).
    optimizer: str = "adamw"  # "adamw" | "muon_adamw" (Muon on block matrices + AdamW on the rest)
    muon_momentum: float = 0.95
    amp_dtype: str | None = None  # None ⇒ fp32; "bf16" ⇒ bf16 autocast (no GradScaler needed)
    compile: bool = False  # torch.compile the forward (the cheap-MFU win on the GPU box)


def train(
    cfg: TrainConfig,
    train_data: np.ndarray,
    model: TransformerLM,
) -> list[tuple[int, float]]:
    """Run the training loop. Returns the loss history as (step, loss) pairs.

    Same plumbing an RL fine-tune resumes from: cosine LR per step, global-ℓ₂ grad clip,
    periodic checkpointing.

    ``cfg.seed`` reseeds the data-sampling RNG here so batches are deterministic. For
    *end-to-end* reproducibility, seed before constructing ``model`` too — weight init
    happens before this call and is not covered by the reseed.
    """
    if cfg.amp_dtype not in (None, "bf16"):
        raise ValueError(
            f"amp_dtype must be None or 'bf16' (fp16 needs a GradScaler); got {cfg.amp_dtype!r}"
        )
    seed_everything(cfg.seed)
    model.to(cfg.device)
    optimizer = build_optimizer(
        model,
        kind=cfg.optimizer,
        lr=cfg.max_lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
        muon_momentum=cfg.muon_momentum,
    )
    # torch.compile wraps the module; keep the ORIGINAL for .cfg / .moe_update_biases / checkpoint
    # (the compiled module's state_dict carries an `_orig_mod.` prefix, and the split for Muon must
    # see the real submodules).
    forward_model = torch.compile(model) if cfg.compile else model
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else None
    device_type = "cuda" if "cuda" in str(cfg.device) else "cpu"

    history: list[tuple[int, float]] = []
    for step in range(cfg.max_steps):
        lr = cosine_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        optimizer.zero_grad()
        amp_ctx = (
            torch.autocast(device_type=device_type, dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with amp_ctx:
            if model.cfg.moe is not None:
                # MoE: add the sparse-regularization terms (seq-wise balance + router z-loss) to CE.
                logits, aux = forward_model(inputs, return_aux=True)
                loss = cross_entropy(logits, targets) + aux.total
            else:
                loss = cross_entropy(forward_model(inputs), targets)
        loss.backward()
        gradient_clipping(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if model.cfg.moe is not None:
            # Aux-loss-free load balancing: nudge the router biases after the weight update.
            model.moe_update_biases()

        if cfg.log_every and step % cfg.log_every == 0:
            loss_value = loss.item()  # the one host sync per log interval (R1 lesson)
            if not math.isfinite(loss_value):
                # Fail LOUD on divergence — never burn compute on a silently-NaN run (FRONTIER
                # 'silent divergence' triage). Known trigger on this box: bf16 autocast + torch.compile
                # together NaN on sm120 / torch-2.12 inductor (reproduces with plain AdamW); use
                # bf16-eager or fp32-compile here — bf16+compile is the H100-rental path.
                raise RuntimeError(
                    f"non-finite loss ({loss_value}) at step {step}: training diverged. Check the LR, "
                    "or avoid bf16 + torch.compile together on sm120 (an inductor codegen bug)."
                )
            history.append((step, loss_value))
            if cfg.verbose:
                print(f"step {step:6d} | lr {lr:.2e} | loss {loss_value:.4f}")
        if (
            cfg.checkpoint_every
            and cfg.checkpoint_path
            and step > 0
            and step % cfg.checkpoint_every == 0
        ):
            save_checkpoint(model, optimizer, step, cfg.checkpoint_path)

    return history


__all__ = ["TrainConfig", "get_batch", "load_checkpoint", "save_checkpoint", "train"]
