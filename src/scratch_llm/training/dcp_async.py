"""Async distributed checkpointing — save without stopping the step, and resume at any topology.

``scratch_llm.train`` already has two checkpoint formats: rank-local ``torch.save`` and the
consolidated ZeRO-2 format (``train.py:150-205``). Neither works here. FSDP2 parameters are
:class:`~torch.distributed.tensor.DTensor` shards, and under TP they are 2-D DTensors on a
``(dp, tp)`` mesh; ``model.state_dict()`` on that yields DTensors that ``torch.save`` will pickle
as opaque objects tied to *this* mesh. DCP is the format that understands them: it writes each
rank's shard with the global shape and offsets as metadata, so a load into a different world size
re-slices instead of failing.

THREE THINGS THIS MODULE IS
---------------------------
**1. A ``Stateful`` app-state.** ``dcp.save`` walks a dict of ``Stateful`` objects. Model and
optimizer go through ``get_state_dict``/``set_state_dict``
(``torch.distributed.checkpoint.state_dict``), which is the only correct way to get an optimizer
state dict whose parameter keys survive re-sharding: it maps optimizer state onto FQNs and
converts each tensor to a DTensor with the same placement as its parameter. Calling
``optimizer.state_dict()`` directly gives you integer parameter *indices*, which mean nothing
after a topology change.

**2. Asynchrony that is actually asynchronous.** ``dcp.async_save`` does two phases: a
synchronous *staging* copy (device -> pinned host memory) and then a background write. Only the
write overlaps training. So the cost on the critical path is one D2H copy of the sharded state,
not a disk write — that is the whole win, and it is why the returned future must be kept and
awaited before the *next* save is issued. Two overlapping saves would race on the staging buffer;
:meth:`AsyncCheckpointer.save` refuses rather than corrupting a checkpoint.

**3. A place the "did it actually overlap?" question has an answer.** ``last_stage_seconds`` and
``last_write_seconds`` are recorded and go into the run log. A checkpoint whose staging time
equals its wall time did not overlap anything, and that is invisible in tok/s if it happens every
500 steps and you average over 100.

Ordering note: the checkpoint is taken *after* ``optimizer.step()``, so the saved model and
optimizer are the same iterate. Saving between ``backward`` and ``step`` writes a model at step N
with an optimizer at step N-1 — resumable, and off by one update forever.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

# Full (unflattened) state dicts, sharded across ranks — the format that re-slices on load.
# cpu_offload=False: the staging copy inside async_save already moves it; doing it twice doubles
# the synchronous part, which is the only part that costs training time.
_SD_OPTIONS = StateDictOptions(full_state_dict=False, cpu_offload=False)


@dataclass
class TrainState(Stateful):
    """The scalars a resume needs that are not in the model or the optimizer.

    ``step`` is the obvious one. ``tokens_seen`` is the one people forget: an LR schedule keyed on
    step resumes correctly, but a *data* position keyed on tokens does not, and a run that
    silently replays the first N tokens after every restart looks like a slightly better model.
    """

    step: int = 0
    tokens_seen: int = 0

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "tokens_seen": self.tokens_seen}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.step = int(state_dict["step"])
        self.tokens_seen = int(state_dict["tokens_seen"])


class ModelOptimState(Stateful):
    """Model + optimizer as one ``Stateful``, through the DTensor-aware state-dict API."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer, options=_SD_OPTIONS)
        return {"model": model_sd, "optim": optim_sd}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=_SD_OPTIONS,
        )


class AsyncCheckpointer:
    """One in-flight ``dcp.async_save`` at a time, with the staging/write split measured."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_state: TrainState,
        out_dir: str | Path,
        *,
        every: int = 0,
    ) -> None:
        self.app = ModelOptimState(model, optimizer)
        self.train_state = train_state
        self.out_dir = Path(out_dir)
        self.every = every
        # `async_save` returns a Future under ASYNC and an AsyncSaveResponse under the
        # pinned-memory stager; typed Any so this does not have to track that union.
        self._future: Any = None
        self._pending_step: int | None = None
        self.last_stage_seconds: float | None = None
        self.last_write_seconds: float | None = None

    def checkpoint_id(self, step: int) -> str:
        return str(self.out_dir / f"step-{step:08d}")

    def due(self, step: int) -> bool:
        """``every`` steps, and never at step 0 (a checkpoint of the initialization)."""
        return bool(self.every) and step > 0 and step % self.every == 0

    def save(self, step: int) -> str:
        """Stage synchronously, write in the background. Returns the checkpoint id.

        Blocks on the previous save first. Deliberately: overlapping two ``async_save`` calls
        races on the staging buffer, and the resulting checkpoint is a mix of two steps that
        loads without complaint.
        """
        self.wait()
        self.train_state.step = step
        t0 = time.perf_counter()
        cid = self.checkpoint_id(step)
        self._future = dcp.async_save(  # pyright: ignore[reportPrivateImportUsage]
            {"app": self.app, "train_state": self.train_state},
            checkpoint_id=cid,
        )
        self.last_stage_seconds = time.perf_counter() - t0
        self._pending_step = step
        return cid

    def wait(self) -> None:
        """Block until the background write finishes. Idempotent."""
        if self._future is None:
            return
        t0 = time.perf_counter()
        self._future.result()
        self.last_write_seconds = time.perf_counter() - t0
        self._future = None
        self._pending_step = None

    @property
    def in_flight_step(self) -> int | None:
        return self._pending_step

    def load(self, checkpoint_id: str | Path) -> int:
        """Restore model, optimizer and train state in place. Returns the restored step.

        ``dcp.load`` is *collective and in-place*: it reads the current state dict to learn the
        shapes and placements it must fill, so the model and optimizer must already exist at the
        target topology. That is the property that makes a different world size work — and the
        reason the optimizer must have taken at least one step (or been primed) before loading,
        or its state dict is empty and there is nothing for the loader to fill.
        """
        state = {"app": self.app, "train_state": self.train_state}
        dcp.load(state, checkpoint_id=str(checkpoint_id))  # pyright: ignore[reportPrivateImportUsage]
        return self.train_state.step

    def close(self) -> None:
        self.wait()


__all__ = ["AsyncCheckpointer", "ModelOptimState", "TrainState"]
