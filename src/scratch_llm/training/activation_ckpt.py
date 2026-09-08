"""Selective activation checkpointing — apply the plan's save-set to the plan's modules.

The trade is the one ``utils/checkpointing.py`` sets out: store an activation (bytes now) or
recompute it in the backward (FLOPs later). Selective AC decides it per *op* rather than per
block, because the two extremes are both bad at 1B/8k — no AC does not fit the activation
budget, full AC re-runs attention it did not have to.

This module executes a decision; it does not make one. :class:`~ParallelPlan` carries
``ac_modules`` (which module subtrees are wrapped), ``sac_save_ops`` (which aten outputs are
MUST_SAVE) and ``sac_save_every_other_mm``. Everything here is the mechanism:

- ``_resolve_op`` turns the plan's op *names* into the ``OpOverload`` objects the policy
  compares against. Names are used in the plan so a plan stays a printable, diffable value; an
  op that does not exist in this torch build is skipped with its name recorded, not silently
  dropped, because "the save set was empty" and "the save set had no effect" look identical in
  a throughput number.
- ``_make_policy`` builds the ``CheckpointPolicy`` callback. The every-other-mm counter is
  per-``context_fn`` invocation and is keyed on ``ctx.is_recompute`` so the forward pass and the
  recompute pass count independently — otherwise the recompute makes different save decisions
  than the forward did and the recomputed values do not match the saved ones. torchtitan does
  exactly this (``torchtitan/distributed/activation_checkpoint.py:249-251``); getting it wrong
  is a silent numerical divergence, not a crash.
- :func:`apply_selective_ac` replaces each matched submodule with
  ``ptd_checkpoint_wrapper(module, context_fn=...)`` **in its parent**, preserving the FQN. That
  last part matters: the wrapper inserts a ``_checkpoint_wrapped_module`` level into
  ``named_parameters()`` unless it is applied with ``CheckpointImpl.NO_REPARAM``… which it is
  not. So the wrapper is installed *by replacing the child in-place on the parent*, and the FSDP
  units the plan names are resolved **before** this runs. ``parallelize.py`` owns that ordering.

``early_stop=False`` matches torchtitan (``activation_checkpoint.py:178``): with early-stop on,
recomputation halts as soon as the needed tensor is produced, which interacts badly with a
policy that saves ops *after* the first needed one.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from scratch_llm.training.parallel_plan import ParallelPlan

# The mm family, by identity. A policy that counts "every second matmul" has to agree with
# itself about what a matmul is; aten decomposes ``x @ Wᵀ`` to ``aten.mm`` for 2-D and
# ``aten.bmm``/``aten.matmul`` for batched, and inductor may emit ``aten.linear`` on some
# backends (torchtitan handles all three, activation_checkpoint.py:241-246).
_MM_OP_NAMES: frozenset[str] = frozenset(
    {"aten.mm.default", "aten.bmm.default", "aten.matmul.default", "aten.addmm.default"}
)


def _resolve_op(name: str) -> Any | None:
    """``"aten.mm.default"`` -> the ``OpOverload``. ``None`` if this build has no such op."""
    obj: Any = torch.ops
    try:
        for part in name.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def resolve_save_ops(names: tuple[str, ...]) -> tuple[dict[Any, str], tuple[str, ...]]:
    """Return ``(op -> name, unresolved names)``. Unresolved is returned, never swallowed."""
    resolved: dict[Any, str] = {}
    missing: list[str] = []
    for name in names:
        op = _resolve_op(name)
        if op is None:
            missing.append(name)
        else:
            resolved[op] = name
    return resolved, tuple(missing)


def _make_policy(save_ops: dict[Any, str], save_every_other_mm: bool):
    """Build the ``(ctx, func, *args, **kwargs) -> CheckpointPolicy`` callback."""
    mm_ops = {op for op, name in save_ops.items() if name in _MM_OP_NAMES}

    def context_fn():
        # Counters live per context_fn call — i.e. per checkpointed forward — and are split by
        # forward vs recompute so the two passes make identical decisions in the same order.
        counts = {"forward": 0, "recompute": 0}

        def policy(ctx: Any, func: Any, *_args: Any, **_kwargs: Any) -> CheckpointPolicy:
            if func in save_ops:
                if save_every_other_mm and func in mm_ops:
                    key = "recompute" if getattr(ctx, "is_recompute", False) else "forward"
                    counts[key] += 1
                    if counts[key] % 2 == 0:
                        return CheckpointPolicy.PREFER_RECOMPUTE
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        return create_selective_checkpoint_contexts(policy)

    return context_fn


def apply_selective_ac(model: nn.Module, plan: ParallelPlan) -> dict[str, str]:
    """Wrap every module matching ``plan.ac_modules``. Returns FQN -> a short description.

    A no-op when ``plan.ac_modules`` is empty — the eager path, which is what the ``ac_policy =
    "none"`` arm of a memory ablation wants.
    """
    if not plan.ac_modules:
        return {}
    save_ops, missing = resolve_save_ops(plan.sac_save_ops)
    if missing:
        raise ValueError(
            f"selective AC: these ops do not exist in torch {torch.__version__}: {missing}. "
            "An op silently dropped from the save set is a throughput change with no diff."
        )
    context_fn = _make_policy(save_ops, plan.sac_save_every_other_mm)

    applied: dict[str, str] = {}
    named = dict(model.named_modules())
    for fqn in sorted(named, key=len, reverse=True):
        if not fqn or not plan.matches_ac(fqn):
            continue
        parent_fqn, _, child_name = fqn.rpartition(".")
        parent = named[parent_fqn] if parent_fqn else model
        wrapped = ptd_checkpoint_wrapper(
            named[fqn],
            context_fn=context_fn,
            preserve_rng_state=True,
            early_stop=False,
        )
        if isinstance(parent, nn.ModuleList):
            parent[int(child_name)] = wrapped
        else:
            setattr(parent, child_name, wrapped)
        applied[fqn] = (
            f"selective_ac(save={len(save_ops)} ops, every_other_mm={plan.sac_save_every_other_mm})"
        )
    if not applied:
        raise ValueError(f"ac_modules {plan.ac_modules} matched no module — check the globs")
    return applied


__all__ = ["apply_selective_ac", "resolve_save_ops"]
