"""Bit-exact resume — the capsule that makes it possible and the verifier that proves it happened.

This is the sharp one. A resume that is merely *close* passes a careless test forever: the losses
look right, the curve continues, and the run is a different run. Everything here compares with
EXACT equality — ``torch.equal`` on a ``uint8`` view of the storage, which is strictly stronger
than ``torch.equal`` on the values (it separates ``-0.0`` from ``0.0`` and distinguishes NaN
payloads) and immeasurably stronger than ``allclose``.

Four pieces of state decide whether a resume is the same run. Drop any one and the run continues
plausibly and wrongly:

1. **RNG** — Python, NumPy, torch CPU, torch CUDA. ``train.get_batch`` draws from the *global*
   NumPy generator, so the NumPy state IS the data-loader position for that loader.
2. **Data-loader position** — kept separately as ``batches_drawn`` so the mismatch has a name a
   postmortem can quote, and so an index-based loader (which does not touch NumPy) is covered too.
3. **Optimizer state** — Adam's moments and per-parameter step counts. Restoring weights without
   moments restarts the bias correction and puts an enormous first step through the model.
4. **LR-schedule position** — ``cosine_lr`` is a pure function of ``step``, so ``step`` is the
   schedule. Resuming at 0 replays the warmup at the wrong place on the curve.

The two named degradations are not inventions. :func:`without_rng` is what torchtitan's
checkpointer saves (MODEL / OPTIMIZER / LR_SCHEDULER / DATALOADER / TRAIN_STATE and no RNG —
``oss/torchtitan/torchtitan/components/checkpointer/base.py:30-34``), and
:func:`as_train_py_checkpoint` is what ``scratch_llm.train.save_checkpoint`` saves today plus what
``train()`` does with it (always begins at step 0 — ``src/scratch_llm/train.py:5-10``). Both must
make the verifier fire; that is the proof it discriminates rather than always agreeing.

**World size is part of the contract.** Each rank's NumPy stream is offset by rank
(``train.py:340-344``), so a capsule is per-rank and a resume at a different world size cannot be
bit-exact. That is a stated limitation of exact resume, not a defect to be fixed by loosening the
comparison.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from scratch_llm.optim import CombinedOptimizer

AnyOptimizer = torch.optim.Optimizer | CombinedOptimizer


def bitwise_equal(a: Tensor, b: Tensor) -> bool:
    """Exact storage equality: same dtype, same shape, same bytes.

    Stronger than ``torch.equal`` on the values (``0.0 == -0.0`` and ``nan != nan`` both lie about
    whether the bits round-tripped) and it is the only comparison a resume check may use.
    """
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    return torch.equal(
        a.detach().cpu().contiguous().view(torch.uint8),
        b.detach().cpu().contiguous().view(torch.uint8),
    )


def _clone(obj: Any) -> Any:
    """Deep-copy a state tree, cloning tensors — ``optimizer.state_dict()`` hands back LIVE
    tensors, so a snapshot that does not clone silently tracks the next optimizer step."""
    if isinstance(obj, Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone(v) for v in obj)
    return copy.deepcopy(obj)


def capture_rng() -> dict[str, Any]:
    """Every RNG a training step can consume, as a picklable blob."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        # This rank's current device only. `get_rng_state_all()` would open a context on every
        # visible GPU from every rank — 64 contexts on an 8-GPU box, for 7 generators nobody uses.
        state["torch_cuda"] = torch.cuda.get_rng_state().clone()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    """Inverse of :func:`capture_rng`. A CUDA state restored on a CPU-only box is dropped loudly
    rather than silently: resuming a CUDA run without its CUDA RNG is not a bit-exact resume."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "capsule carries CUDA RNG state but this host has no CUDA — the resume cannot be "
                "bit-exact; run it on the same device class that wrote it"
            )
        torch.cuda.set_rng_state(state["torch_cuda"])


@dataclass
class TrainSnapshot:
    """Everything needed to continue a run as if it had never stopped.

    ``rng is None`` and ``step``/``batches_drawn`` reset to 0 are the *degradations*, and they are
    representable on purpose: the point of this rung is that the verifier can tell them apart from
    a real resume.
    """

    step: int
    batches_drawn: int
    lr: float
    model: dict[str, Tensor]
    optim: dict[str, Any] | None
    rng: dict[str, Any] | None
    rank: int = 0
    world_size: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> TrainSnapshot:
        # weights_only=False: our own capsule carries RNG tuples and optimizer group metadata,
        # not just tensors. Same trust boundary as train.load_checkpoint.
        return cls(**torch.load(path, map_location=map_location, weights_only=False))


def capture(
    *,
    step: int,
    lr: float,
    model: nn.Module,
    optimizer: AnyOptimizer | None,
    batches_drawn: int,
    extra: dict[str, Any] | None = None,
) -> TrainSnapshot:
    """Snapshot the full run state. Collective-free — every rank captures its own."""
    return TrainSnapshot(
        step=step,
        batches_drawn=batches_drawn,
        lr=lr,
        model={k: v.detach().clone() for k, v in model.state_dict().items()},
        optim=_clone(optimizer.state_dict()) if optimizer is not None else None,
        rng=capture_rng(),
        rank=dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
        world_size=dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
        extra=_clone(extra or {}),
    )


def restore(snap: TrainSnapshot, model: nn.Module, optimizer: AnyOptimizer | None) -> int:
    """Put a snapshot back and return the step to resume AT (the step not yet taken).

    A capsule whose ``rng`` is ``None`` restores everything else and leaves the RNGs wherever the
    fresh process left them — which is what torchtitan and ``train.save_checkpoint`` do, and what
    makes their resumes plausible-but-different.
    """
    model.load_state_dict(_clone(snap.model))
    if optimizer is not None:
        if snap.optim is None:
            raise ValueError(
                "capsule carries no optimizer state — restoring weights without Adam's moments "
                "restarts bias correction and puts a huge first step through the model"
            )
        optimizer.load_state_dict(_clone(snap.optim))
    if snap.rng is not None:
        restore_rng(snap.rng)
    return snap.step


def without_rng(snap: TrainSnapshot) -> TrainSnapshot:
    """torchtitan's checkpoint, exactly: model + optimizer + LR-schedule position + dataloader
    position, and no RNG (``components/checkpointer/base.py:30-34``). The verifier must fire."""
    out = copy.copy(snap)
    out.rng = None
    return out


def as_train_py_checkpoint(snap: TrainSnapshot) -> TrainSnapshot:
    """``scratch_llm.train.save_checkpoint`` + ``train()``, exactly: model + optimizer + a step
    that is *written but never used*, since ``train()`` always begins at step 0 with a fresh warmup
    (``train.py:5-10``). No RNG, no loader position, no detector state. The verifier must fire."""
    out = copy.copy(snap)
    out.rng = None
    out.step = 0
    out.batches_drawn = 0
    out.extra = {}
    return out


# ---------------------------------------------------------------------------------------------
# the verifier
# ---------------------------------------------------------------------------------------------


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    """Append a dotted path for every place two state trees differ. Exact comparisons only."""
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        if not (isinstance(a, Tensor) and isinstance(b, Tensor)) or not bitwise_equal(a, b):
            out.append(path)
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        both = isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
        if not both or a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            out.append(path)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                out.append(f"{path}.{key}")
            else:
                _diff(a[key], b[key], f"{path}.{key}", out)
        return
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            out.append(f"{path}.len")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diff(x, y, f"{path}[{i}]", out)
        return
    if type(a) is not type(b) or a != b:
        out.append(path)


@dataclass(frozen=True)
class ResumeReport:
    """The detector's verdict. ``fired`` means the resume was NOT bit-exact."""

    mismatches: tuple[str, ...]

    @property
    def bit_exact(self) -> bool:
        return not self.mismatches

    @property
    def fired(self) -> bool:
        return bool(self.mismatches)

    @property
    def first_mismatch(self) -> str | None:
        return self.mismatches[0] if self.mismatches else None

    def summary(self) -> str:
        if self.bit_exact:
            return "resume is BIT-EXACT (0 differing keys)"
        shown = ", ".join(self.mismatches[:5])
        more = f" (+{len(self.mismatches) - 5} more)" if len(self.mismatches) > 5 else ""
        return f"resume DIVERGED at {len(self.mismatches)} key(s): {shown}{more}"


def verify_bit_exact(reference: TrainSnapshot, resumed: TrainSnapshot) -> ResumeReport:
    """Compare an uninterrupted run's end state against a resumed run's end state, exactly.

    The uninterrupted run IS the oracle: there is no analytic answer for "what should step N look
    like", only "what it looked like when nothing went wrong". Everything is compared — step,
    loader position, LR, every parameter and buffer, every optimizer moment, every RNG stream, and
    any detector state parked in ``extra`` — because each of them is separately capable of making
    the *next* step differ while this one looks fine.
    """
    out: list[str] = []
    _diff(reference.step, resumed.step, "step", out)
    _diff(reference.batches_drawn, resumed.batches_drawn, "batches_drawn", out)
    _diff(reference.lr, resumed.lr, "lr", out)
    _diff(reference.world_size, resumed.world_size, "world_size", out)
    _diff(reference.model, resumed.model, "model", out)
    _diff(reference.optim, resumed.optim, "optim", out)
    _diff(reference.rng, resumed.rng, "rng", out)
    _diff(reference.extra, resumed.extra, "extra", out)
    return ResumeReport(mismatches=tuple(out))


__all__ = [
    "AnyOptimizer",
    "ResumeReport",
    "TrainSnapshot",
    "as_train_py_checkpoint",
    "bitwise_equal",
    "capture",
    "capture_rng",
    "restore",
    "restore_rng",
    "verify_bit_exact",
    "without_rng",
]
