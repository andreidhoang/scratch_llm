"""Compose TP, AC, compile and FSDP — in the one order that is correct, and why the others are not.

torchtitan's ``parallelize_llama`` applies them in exactly this sequence
(``torchtitan/models/llama3/parallelize.py:40-78``):

    1. tensor parallel   ``model.parallelize(parallel_dims)``            (:40-41)
    2. activation ckpt   ``ac_config.build(...).apply(model)``           (:46-47)
    3. compile           ``apply_compile(...)`` — "after AC wrapping and before FSDP" (:49-55)
    4. FSDP2             ``apply_fsdp_to_decoder(...)``                  (:68-78)

Every adjacent swap is wrong, and three of the four are wrong *silently*:

**TP before AC.** AC replaces a module with a ``CheckpointWrapper`` whose child is the original.
FQNs the TP patterns match (``blocks.0.attn.q_proj``) become
``blocks.0._checkpoint_wrapped_module.attn.q_proj``, so every TP glob misses and the model runs
data-parallel-only. No error: it trains, it converges, it is 2× the memory and the wrong number.

**AC before compile.** ``torch.compile`` traces what it is given. Compiling first and then
wrapping means the checkpoint boundary sits outside a compiled region and the recompute re-enters
the compiled graph, which is at best a second compile and at worst a graph break per block.
torchtitan's comment at ``:49`` says exactly this.

**Compile before FSDP.** FSDP2's hooks are what the compiled region must *not* swallow; compiling
per-block and then sharding gives inductor a stable per-block graph while FSDP's collectives stay
in eager, where they can overlap. Doing it the other way asks dynamo to trace the all-gather.

**FSDP last.** This is the load-bearing one for correctness rather than speed. ``fully_shard``
converts parameters to DTensor on the FSDP mesh; if TP has already made them DTensor on the TP
mesh, the result is a 2-D ``(fsdp, tp)`` DTensor, which is what 2-D parallelism *is*. Reversed,
FSDP shards a plain tensor into ``(fsdp,)`` and TP then tries to ``distribute_tensor`` an object
that is already a DTensor on a different mesh — that one at least raises.

There is a fifth ordering constraint the list does not show: the plan's FQNs are resolved against
the *unwrapped* model, so :func:`parallelize` resolves and validates everything up front, before
step 1. After AC has run, the FQN ``blocks.0`` no longer names what the plan meant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from scratch_llm.training.activation_ckpt import apply_selective_ac
from scratch_llm.training.fsdp_wrap import apply_fsdp
from scratch_llm.training.mp_policy import bf16_mp_policy
from scratch_llm.training.parallel_plan import ParallelPlan, validate_plan
from scratch_llm.training.rung_config import T1RungConfig
from scratch_llm.training.tensor_parallel import apply_tensor_parallel


@dataclass
class ParallelizeReport:
    """What was actually applied — logged verbatim, so a run's log answers "what ran".

    A report, not a plan: the plan is what was asked for, this is what the four passes did. They
    differ exactly when a glob matched nothing, which is the failure this rung is most likely to
    hit and least likely to notice.
    """

    mesh_shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...]
    tp_applied: dict[str, str] = field(default_factory=dict)
    ac_applied: dict[str, str] = field(default_factory=dict)
    compiled: tuple[str, ...] = ()
    fsdp_units: tuple[tuple[tuple[str, ...], bool], ...] = ()

    def as_log_dict(self) -> dict[str, object]:
        return {
            "mesh_shape": list(self.mesh_shape),
            "mesh_dim_names": list(self.mesh_dim_names),
            "tp_applied": dict(sorted(self.tp_applied.items())),
            "ac_applied": dict(sorted(self.ac_applied.items())),
            "compiled": list(self.compiled),
            "fsdp_units": [
                {"modules": list(m), "reshard_after_forward": r} for m, r in self.fsdp_units
            ],
        }


def build_mesh(plan: ParallelPlan, device_type: str) -> DeviceMesh:
    """``init_device_mesh`` from the plan's declared shape and names.

    The *order* of ``mesh_shape`` is the rank-assignment order: the last dim is the fastest
    varying, so with ``("dp", "tp")`` ranks 0,1 are one TP group, 2,3 the next. On a single node
    that keeps a TP pair on one NVSwitch hop; on multi-node it keeps TP off the network entirely.
    Reversing the two names is a legal mesh that trains correctly and slowly — the plan owns the
    choice, this function only executes it.
    """
    return init_device_mesh(device_type, plan.mesh_shape, mesh_dim_names=plan.mesh_dim_names)


def _apply_compile(model: nn.Module, plan: ParallelPlan) -> tuple[str, ...]:
    """Per-module ``torch.compile``, replacing each matched child in its parent.

    Per-block rather than whole-model, for torchtitan's reason (``distributed/compile.py:45-47``):
    a decoder is the same block N times, so compiling one block and reusing the artifact turns N
    compilations into one. Whole-model compile also fights FSDP's hooks.
    """
    if not plan.compile_modules:
        return ()
    named = dict(model.named_modules())
    compiled: list[str] = []
    for fqn in sorted(named, key=len, reverse=True):
        if not fqn or not plan.matches_compile(fqn):
            continue
        parent_fqn, _, child = fqn.rpartition(".")
        parent = named[parent_fqn] if parent_fqn else model
        target = torch.compile(named[fqn])
        if isinstance(parent, nn.ModuleList):
            parent[int(child)] = target  # pyright: ignore[reportArgumentType]
        else:
            setattr(parent, child, target)
        compiled.append(fqn)
    return tuple(compiled)


def parallelize(
    model: nn.Module,
    plan: ParallelPlan,
    cfg: T1RungConfig,
    *,
    device_type: str = "cuda",
    mesh: DeviceMesh | None = None,
    compile_model: bool = True,
) -> tuple[nn.Module, DeviceMesh, ParallelizeReport]:
    """Apply the plan to the model. Returns ``(model, mesh, report)``.

    ``model`` is mutated in place and returned for convenience — FSDP2 and ``distribute_module``
    are both in-place APIs, so there is no un-parallelized copy afterwards.

    ``compile_model=False`` is the escape hatch the CPU tests use: inductor on a 2-layer model on
    a laptop costs more than the whole test suite and tests nothing the GPU run does not.
    """
    validate_plan(plan, model, cfg)
    mesh = mesh if mesh is not None else build_mesh(plan, device_type)
    report = ParallelizeReport(mesh_shape=plan.mesh_shape, mesh_dim_names=plan.mesh_dim_names)

    # 1. TP — first, while the plan's FQNs still name the modules the plan meant.
    if plan.tp_mesh_dim is not None and plan.tp_styles:
        report.tp_applied = apply_tensor_parallel(model, plan, mesh[plan.tp_mesh_dim])

    # 2. AC — inserts CheckpointWrapper levels into the FQN tree, so nothing FQN-addressed may
    #    follow it except FSDP, whose targets we resolve from the plan through the *wrappers*
    #    (apply_fsdp re-reads named_modules, and a wrapper is transparent to attribute access on
    #    the parent because we replaced the child in place).
    report.ac_applied = apply_selective_ac(model, plan)

    # 3. compile — after AC, before FSDP.
    if compile_model:
        report.compiled = _apply_compile(model, plan)

    # 4. FSDP2 — last, so it sees TP's DTensors and produces 2-D (fsdp, tp) parameters.
    mp = bf16_mp_policy(cfg.param_dtype, cfg.reduce_dtype)
    report.fsdp_units = tuple(apply_fsdp(model, plan, mesh[plan.fsdp_mesh_dim], mp))
    return model, mesh, report


__all__ = ["ParallelizeReport", "build_mesh", "parallelize"]
