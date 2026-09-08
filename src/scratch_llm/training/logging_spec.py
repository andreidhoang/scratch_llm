"""The logging spec — every field the plan's §05 T-R1 row names, and a validator that says so.

    "full logging spec (loss, LR, grad norms global/group, tok/s, MFU; activation max-abs per
     residual point + weight norms /100 steps; NCCL wait; git hash + config)"

A logging spec that is a docstring is a suggestion. This one is a :data:`REQUIRED_STEP_FIELDS`
frozenset and a :func:`validate_step_record` that raises, so "the field quietly stopped being
emitted" is a test failure rather than a gap noticed three weeks later when the run is over and
the box is returned. Every field below exists because a specific class of training failure is
invisible without it:

``loss``                 the run's only real output.
``lr``                   a schedule bug (wrong warmup, wrong total) looks exactly like a bad LR.
``grad_norm_global``     the divergence tell, one step before the loss shows it.
``grad_norm_groups``     *where* the divergence is. A global norm that doubles tells you the run
                         is sick; embeddings at 40x the MLPs tells you which parameter group,
                         which is the difference between a diagnosis and a restart. This is why a
                         single global number is not a substitute and why the guard test builds
                         groups with deliberately different norms.
``tokens_per_s_per_device``  torchtitan's ``tps`` (metrics.py:485), defined identically here so the
                         two numbers are comparable: rank-local tokens / (dt x non-DP degree),
                         which equals global tokens / (dt x world_size).
``mfu``                  tok/s alone cannot say whether a config change helped the model or just
                         made the step smaller. Denominator and FLOP convention from
                         ``scratch_llm.training.flops`` — torchtitan's, so the ratio means
                         something.
``nccl_wait_s``          see :data:`NCCL_WAIT_SCOPE`. Read the caveat before quoting the number.
``step_time_s``          the stopwatch the other three are derived from; logged raw so a reader
                         can recompute them.
``peak_memory_bytes``    an OOM at step 900 of 1000 is the expensive failure; the high-water mark
                         is the leading indicator.
``git_sha`` / ``config`` provenance. The sha is *read* from the repo, never passed in — a
                         declared commit is a commit somebody forgot to update.

And every 100 steps (:data:`PERIODIC_FIELDS`):

``activation_max_abs``   per residual point. bf16 saturates at 3.4e38 but *loses* precision long
                         before that; a residual stream whose max-abs climbs 2x per hundred steps
                         is the shape of an instability that ends in a NaN 5000 steps later. Per
                         point, not one global max, because "layer 14 onward" and "everywhere" are
                         different bugs.
``weight_norms``         the slow counterpart: weight decay too weak, an embedding table growing
                         without bound, a norm gain collapsing to zero.

WHY EVERY 100 AND NOT EVERY STEP: both need a device-to-host sync of a value nothing else reads.
At 8 GPUs and ~1 s/step that is a stall per step for a signal that moves on a scale of hundreds.
:func:`fires_periodic` fires at step 0 (the initialization, the only baseline you get) and at the
final step (the state you are about to checkpoint) as well as every ``every`` steps — both
endpoints, because a diagnostic series missing its ends is the series you cannot difference.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

# ---------------------------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------------------------

REQUIRED_STEP_FIELDS: frozenset[str] = frozenset(
    {
        "step",
        "loss",
        "lr",
        "grad_norm_global",
        "grad_norm_groups",
        "tokens_per_s_per_device",
        "mfu",
        "nccl_wait_s",
        "step_time_s",
        "peak_memory_bytes",
        "git_sha",
        "config",
    }
)

PERIODIC_FIELDS: frozenset[str] = frozenset({"activation_max_abs", "weight_norms"})

ALL_FIELDS: frozenset[str] = REQUIRED_STEP_FIELDS | PERIODIC_FIELDS

# How often the periodic fields fire. The plan says /100 steps.
PERIODIC_EVERY = 100

NCCL_WAIT_SCOPE = (
    "explicit collectives issued by the training loop (loss all-reduce, grad-norm all-reduce, "
    "end-of-step barrier) measured on this rank. FSDP2's parameter all-gathers and gradient "
    "reduce-scatters and TP's activation all-reduces are issued inside autograd and are NOT in "
    "this number — they show up as step_time_s. Use a profiler (nsys / torch profiler) for "
    "those; this field is the part a Python loop can honestly claim to have measured."
)


def validate_step_record(record: dict[str, Any], *, periodic: bool) -> None:
    """Raise unless ``record`` carries every field the spec names for this kind of step.

    Two directions, both checked: a missing required field raises, and an *unknown* field raises
    too. The second is not pedantry — a metric added under a new name while the old one keeps
    being written is how two dashboards start disagreeing about the same run.
    """
    expected = REQUIRED_STEP_FIELDS | (PERIODIC_FIELDS if periodic else frozenset())
    keys = set(record)
    missing = expected - keys
    if missing:
        raise ValueError(
            f"logging spec violated: missing {sorted(missing)} at step {record.get('step')!r} "
            f"(periodic={periodic})"
        )
    unknown = keys - ALL_FIELDS
    if unknown:
        raise ValueError(f"logging spec violated: unknown field(s) {sorted(unknown)}")
    if not periodic and (keys & PERIODIC_FIELDS):
        raise ValueError(
            f"logging spec violated: periodic field(s) {sorted(keys & PERIODIC_FIELDS)} emitted "
            f"on a non-periodic step ({record.get('step')!r})"
        )
    for name in ("loss", "grad_norm_global", "step_time_s"):
        value = record[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"logging spec violated: {name} must be a number, got {value!r}")
    if not isinstance(record["grad_norm_groups"], dict) or not record["grad_norm_groups"]:
        raise ValueError(
            "logging spec violated: grad_norm_groups must be a non-empty mapping — a single "
            "global norm is not a per-group norm"
        )


def fires_periodic(step: int, every: int, total_steps: int) -> bool:
    """True on step 0, every ``every`` steps, and on the final step (``total_steps - 1``).

    Both endpoints are deliberate. Step 0 is the only measurement of the *initialization*, which
    is the baseline every later activation/weight number is read against; the final step is the
    state that gets checkpointed. A series missing either end cannot be differenced.
    """
    if every <= 0:
        return False
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    return step == 0 or step % every == 0 or step == total_steps - 1


# ---------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------


def git_head_sha(repo: str | Path | None = None) -> str:
    """The real HEAD of the repo this module lives in — read, never declared.

    ``Path(__file__).resolve()`` matters: ``scratch_llm/`` in the workspace is a symlink, and an
    unresolved path would run ``git`` in the workspace repo and record the wrong sha.
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parent
    out = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


# ---------------------------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------------------------

# Ordered rules: first match wins. Written against scratch_llm.model's FQNs
# (token_emb / blocks.N.attn_norm / blocks.N.attn.{q,k,v,o}_proj / blocks.N.attn.{q,k}_norm /
# blocks.N.ffn_norm / blocks.N.ffn.{w1,w2,w3} / final_norm / lm_head).
_GROUP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("embed", ("token_emb",)),
    ("head", ("lm_head",)),
    ("norm", ("attn_norm", "ffn_norm", "final_norm", ".q_norm", ".k_norm")),
    ("attn", (".attn.",)),
    ("mlp", (".ffn.",)),
)

GROUP_NAMES: tuple[str, ...] = tuple(name for name, _ in _GROUP_RULES) + ("other",)


# Wrapper levels torch inserts into FQNs. ``checkpoint_wrapper`` adds
# ``_checkpoint_wrapped_module`` and ``torch.compile`` adds ``_orig_mod``; both appear in
# ``named_parameters()`` / ``named_modules()`` (though torch strips the first from
# ``state_dict()``). Left in, every logged name changes the day AC or compile is toggled, and two
# runs of the same model produce two different sets of log keys.
_WRAPPER_SEGMENTS = ("_checkpoint_wrapped_module", "_orig_mod", "_fsdp_wrapped_module")


def clean_fqn(fqn: str) -> str:
    """Drop wrapper segments so a logged name is the model's name, not the plumbing's."""
    return ".".join(seg for seg in fqn.split(".") if seg not in _WRAPPER_SEGMENTS)


def param_group(fqn: str) -> str:
    """Which logging group a parameter FQN belongs to. Total function: falls back to ``other``.

    ``other`` is not a dumping ground, it is an alarm: a new parameter kind appearing in the model
    lands there, the group shows up in the log, and somebody has to decide where it goes.
    """
    fqn = clean_fqn(fqn)
    for name, needles in _GROUP_RULES:
        if any(n in fqn for n in needles):
            return name
    return "other"


def _scalar(t: Tensor | float) -> float:
    """0-dim tensor (possibly a DTensor) -> float, doing the one collective a DTensor needs.

    ``full_tensor()`` on a *scalar* is an all-reduce of 4 bytes; on a parameter it would be a
    full gather. That is why every norm below is reduced to a scalar DTensor *before* this is
    called, never after.
    """
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, DTensor):
        t = t.full_tensor()
    return float(t.item())


def _sum_of_squares(tensors: Iterable[Tensor]) -> Tensor | None:
    """Σ‖t‖² as a single 0-dim tensor, keeping DTensor-ness (so one collective, not N)."""
    squares = [torch.linalg.vector_norm(t.detach().float()) ** 2 for t in tensors]
    if not squares:
        return None
    return reduce(add, squares)


def grad_norms(model: nn.Module) -> tuple[float, dict[str, float]]:
    """``(global L2 grad norm, per-group L2 grad norms)``.

    The identity the guard test asserts: ``global² == Σ_groups group²``. It holds because the L2
    norm over a partition of the parameters is the root of the sum of the partition's squared
    norms — which is exactly why a per-group breakdown is free once you compute the global one,
    and why there is no excuse for not logging it.

    Cost, honestly: ``vector_norm`` of a sharded DTensor yields a ``Partial`` scalar, and squaring
    a ``Partial`` forces a redistribute — so this is O(number of parameters) tiny all-reduces per
    step, not one. Tiny (4 bytes each) but latency-bound, and the first thing to batch if the
    profile shows the step waiting on them.

    Groups with no gradients are omitted rather than reported as 0.0. Under weight tying
    ``lm_head.weight is token_emb.weight``, ``named_parameters`` deduplicates, and the ``head``
    group is genuinely empty — reporting 0.0 there would read as "the head has no gradient".
    """
    by_group: dict[str, list[Tensor]] = {}
    for fqn, p in model.named_parameters():
        if p.grad is None:
            continue
        by_group.setdefault(param_group(fqn), []).append(p.grad)
    groups: dict[str, float] = {}
    total_sq: Tensor | None = None
    for name, grads in by_group.items():
        ss = _sum_of_squares(grads)
        if ss is None:
            continue
        groups[name] = _scalar(ss) ** 0.5
        total_sq = ss if total_sq is None else total_sq + ss
    return (0.0 if total_sq is None else _scalar(total_sq) ** 0.5), groups


def weight_norms(model: nn.Module) -> dict[str, float]:
    """Per-parameter L2 norms. Emitted every ``PERIODIC_EVERY`` steps.

    Per parameter, not per group: the signal this catches (one embedding table growing, one norm
    gain collapsing) is invisible once averaged over a group.
    """
    return {
        clean_fqn(fqn): _scalar(torch.linalg.vector_norm(p.detach().float()))
        for fqn, p in model.named_parameters()
    }


# ---------------------------------------------------------------------------------------------
# Activation probes
# ---------------------------------------------------------------------------------------------


class ActivationProbe:
    """Max-abs at every residual point, via forward hooks, only on the steps that ask for it.

    THE POINTS. ``scratch_llm.model.TransformerBlock.forward`` is
    ``x = x + attn(attn_norm(x)); x = x + ffn(ffn_norm(x))`` (``model.py:429-436``). Hooking the
    two sub-modules gives the *deltas*; hooking the block gives the residual stream after both
    adds. Both are recorded, under different prefixes, because they answer different questions:

      ``resid.*``  the residual stream itself — embed output, each block's output, final norm.
                   This is the series that grows when a run is going to diverge.
      ``delta.*``  what each sub-layer contributed. A residual stream that grows because one
                   layer's attention output exploded is a different bug from one that grows
                   because every layer contributes a little too much.

    ENABLE/DISABLE. Every hook does an ``.abs().max().item()``, which is a device-to-host sync.
    On the periodic steps that is the point; on the other 99 it is a stall per residual point per
    step. The probe is therefore *armed* per step by :meth:`arm`, and the hooks return
    immediately when not armed — cheaper and less error-prone than adding and removing 30 hooks
    a hundred times.
    """

    def __init__(self, model: nn.Module) -> None:
        self._values: dict[str, float] = {}
        self._armed = False
        self._handles: list[Any] = []
        self._register(model)

    def _register(self, model: nn.Module) -> None:
        def make(name: str):
            def hook(_m: nn.Module, _inp: Any, out: Any) -> None:
                if not self._armed:
                    return
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, Tensor):
                    self._values[name] = _scalar(t.detach().abs().amax())

            return hook

        # setdefault, not a dict comprehension: ``named_modules`` yields parents before
        # children, so the FIRST module whose cleaned name is ``blocks.0`` is the AC/compile
        # wrapper and the second is the block inside it. A comprehension keeps the last, which
        # hooks the inner module — whose forward runs TWICE per armed step (once forward, once
        # during the checkpoint recompute), overwriting the recorded value with the recomputed one.
        named: dict[str, nn.Module] = {}
        for _fqn, _m in model.named_modules():
            named.setdefault(clean_fqn(_fqn), _m)
        if "token_emb" in named:
            self._handles.append(named["token_emb"].register_forward_hook(make("resid.embed")))
        for fqn, module in named.items():
            if fqn.startswith("blocks.") and fqn.count(".") == 1:
                self._handles.append(module.register_forward_hook(make(f"resid.{fqn}")))
            elif fqn.endswith(".attn") or fqn.endswith(".ffn"):
                self._handles.append(module.register_forward_hook(make(f"delta.{fqn}")))
        if "final_norm" in named:
            self._handles.append(
                named["final_norm"].register_forward_hook(make("resid.final_norm"))
            )

    @property
    def points(self) -> int:
        return len(self._handles)

    def arm(self, on: bool) -> None:
        self._armed = on
        if on:
            self._values.clear()

    def snapshot(self) -> dict[str, float]:
        return dict(sorted(self._values.items()))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------------------------
# Comm-wait accounting
# ---------------------------------------------------------------------------------------------


@dataclass
class CommWaitTimer:
    """Accumulates the seconds this rank spends blocked in explicitly-issued collectives.

    Read :data:`NCCL_WAIT_SCOPE` before quoting the number. It is honest about a real thing — a
    straggler rank shows up here as *low* wait while everyone else's is high — and it does not
    pretend to see inside autograd.
    """

    seconds: float = 0.0

    @contextmanager
    def measure(self) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds += time.perf_counter() - t0

    def take(self) -> float:
        """Read and reset — the per-step value."""
        value = self.seconds
        self.seconds = 0.0
        return value


# ---------------------------------------------------------------------------------------------
# The logger
# ---------------------------------------------------------------------------------------------


@dataclass
class RunLogger:
    """Builds, validates and emits one record per step.

    ``config`` and ``git_sha`` are captured once at construction and repeated on every record.
    Repeating them is deliberate: a log line has to be interpretable on its own, because that is
    how log lines are actually read — grepped out of a 100k-line file, one at a time.
    """

    config: dict[str, Any]
    total_steps: int
    periodic_every: int = PERIODIC_EVERY
    out_path: Path | None = None
    git_sha: str = field(default_factory=git_head_sha)
    records: list[dict[str, Any]] = field(default_factory=list)
    _fh: Any = None

    def __post_init__(self) -> None:
        if self.out_path is not None:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.out_path.open("a", encoding="utf-8")
            header = {
                "kind": "run_header",
                "git_sha": self.git_sha,
                "config": self.config,
                "nccl_wait_scope": NCCL_WAIT_SCOPE,
                "required_step_fields": sorted(REQUIRED_STEP_FIELDS),
                "periodic_fields": sorted(PERIODIC_FIELDS),
                "periodic_every": self.periodic_every,
            }
            self._fh.write(json.dumps(header) + "\n")
            self._fh.flush()

    def is_periodic(self, step: int) -> bool:
        return fires_periodic(step, self.periodic_every, self.total_steps)

    def log_step(
        self,
        *,
        step: int,
        loss: float,
        lr: float,
        grad_norm_global: float,
        grad_norm_groups: dict[str, float],
        tokens_per_s_per_device: float,
        mfu: float,
        nccl_wait_s: float,
        step_time_s: float,
        peak_memory_bytes: int,
        activation_max_abs: dict[str, float] | None = None,
        weight_norms_: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Assemble one record, validate it against the spec, emit it, and return it."""
        periodic = self.is_periodic(step)
        record: dict[str, Any] = {
            "step": step,
            "loss": loss,
            "lr": lr,
            "grad_norm_global": grad_norm_global,
            "grad_norm_groups": grad_norm_groups,
            "tokens_per_s_per_device": tokens_per_s_per_device,
            "mfu": mfu,
            "nccl_wait_s": nccl_wait_s,
            "step_time_s": step_time_s,
            "peak_memory_bytes": peak_memory_bytes,
            "git_sha": self.git_sha,
            "config": self.config,
        }
        if periodic:
            if activation_max_abs is None or weight_norms_ is None:
                raise ValueError(
                    f"step {step} is a periodic step (every {self.periodic_every}, plus step 0 "
                    f"and step {self.total_steps - 1}) but the caller did not arm the probes — "
                    "the /100-steps fields would be silently absent"
                )
            record["activation_max_abs"] = activation_max_abs
            record["weight_norms"] = weight_norms_
        validate_step_record(record, periodic=periodic)
        self.records.append(record)
        if self._fh is not None:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


__all__ = [
    "ALL_FIELDS",
    "GROUP_NAMES",
    "NCCL_WAIT_SCOPE",
    "PERIODIC_EVERY",
    "PERIODIC_FIELDS",
    "REQUIRED_STEP_FIELDS",
    "ActivationProbe",
    "CommWaitTimer",
    "RunLogger",
    "clean_fqn",
    "fires_periodic",
    "git_head_sha",
    "grad_norms",
    "param_group",
    "validate_step_record",
    "weight_norms",
]
