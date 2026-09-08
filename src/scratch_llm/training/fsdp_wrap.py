"""The ``fully_shard`` traversal — execute the plan's FSDP unit list, in the only safe order.

FSDP2 (``torch.distributed.fsdp.fully_shard``) is a *composable* API: calling it on a module
registers that module's parameters as a communication group and installs pre/post-forward hooks.
Calling it on an ancestor afterwards picks up whatever no descendant already claimed. Two
consequences that this module exists to enforce:

**Inner before outer.** ``fully_shard(model)`` first and ``fully_shard(model.blocks[0])`` second
is not "one big unit plus one small one"; it is undefined — the root already flattened the
block's parameters into its own group. So :func:`apply_fsdp` sorts the plan's units by FQN depth,
deepest first, with the root (``""``) last. torchtitan hard-codes the same order
(``torchtitan/distributed/fsdp.py:246-368``: embeddings, then norm+head, then each block, then
the root at ``:368``).

**A multi-FQN unit is one call.** ``fully_shard([m1, m2], ...)`` makes one group out of two
modules — one all-gather covers both. That is how torchtitan avoids gathering a tied
embedding/head twice (``fsdp.py:238-250``). A plan expresses it by putting both FQNs in one
:class:`~scratch_llm.training.parallel_plan.FsdpUnit`.

**Tied parameters.** Our model ties ``lm_head.weight`` to ``token_emb.weight``
(``model.py:463-464``): one tensor, two FQNs. If the two land in different FSDP units, FSDP
registers the same storage in two groups and the second all-gather is pure waste (and the
reduce-scatter double-counts). :func:`apply_fsdp` detects it and raises, naming both FQNs,
because the symptom otherwise is "throughput is 6% low and I do not know why".

**Gradient division.** FSDP2 divides gradients by the shard degree inside the reduce-scatter.
Our loss is already a mean over the local microbatch, so mean-of-means is the correct global mean
*only when every rank has the same token count* — which it does here (fixed-length sequences, no
padding, no variable-length packing). ``expected_uniform_token_count`` records that assumption in
the run log so the day it stops holding is a diff and not a mystery. torchtitan takes the other
branch — it disables FSDP's division and scales by the global token count itself
(``fsdp.py:85-99``) — because its packing dataloader does produce ragged ranks.
"""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from scratch_llm.training.parallel_plan import ParallelPlan
from scratch_llm.training.tensor_parallel import tied_parameter_groups


def _depth(fqn: str) -> int:
    """Root ("") is depth 0; deeper FQNs sort first so inner units are applied first."""
    return 0 if fqn == "" else fqn.count(".") + 1


def _check_tied_params_share_a_unit(model: nn.Module, plan: ParallelPlan) -> None:
    """Refuse a plan that splits a tied tensor's two FQNs across two FSDP units.

    Ownership is resolved the way ``fully_shard`` resolves it — deepest unit first, first claim
    wins, an ancestor unit only manages what no descendant claimed — and keyed on the parameter
    *path*, not on ``id``. Keying on ``id`` would be useless here: a tied tensor is one object, so
    every path to it would trivially agree. ``remove_duplicate=False`` is what makes both paths
    visible at all.
    """
    unit_of: dict[str, int] = {
        fqn: i for i, unit in enumerate(plan.fsdp_units) for fqn in unit.modules
    }
    named = dict(model.named_modules())
    owner_of_path: dict[str, int] = {}
    for fqn in sorted(unit_of, key=lambda f: (-_depth(f), f)):
        for pname, _p in named[fqn].named_parameters(recurse=True, remove_duplicate=False):
            full = f"{fqn}.{pname}" if fqn else pname
            owner_of_path.setdefault(full, unit_of[fqn])

    for group in tied_parameter_groups(model):
        paths = [f"{m}.{n}" if m else n for m, n in group]
        owners = {owner_of_path[path] for path in paths if path in owner_of_path}
        if len(owners) > 1:
            raise ValueError(
                f"tied parameter reachable as {paths} sits in FSDP units {sorted(owners)}. One "
                "tensor gathered by two groups is one wasted all-gather per forward and a "
                "double-counted reduce-scatter — put every FQN of the tie in the same FsdpUnit."
            )


def apply_fsdp(
    model: nn.Module,
    plan: ParallelPlan,
    fsdp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
) -> list[tuple[tuple[str, ...], bool]]:
    """Apply ``plan.fsdp_units`` to ``model`` on ``fsdp_mesh``. Returns the units in applied order.

    Call *after* TP and AC and compile — see ``parallelize.py`` for why the order is not
    negotiable.
    """
    if fsdp_mesh.ndim != 1:
        raise ValueError(f"fsdp_mesh must be 1-D, got ndim={fsdp_mesh.ndim}")
    _check_tied_params_share_a_unit(model, plan)

    named = dict(model.named_modules())
    ordered = sorted(
        plan.fsdp_units,
        key=lambda u: (-max(_depth(f) for f in u.modules), u.modules),
    )
    applied: list[tuple[tuple[str, ...], bool]] = []
    for unit in ordered:
        missing = [f for f in unit.modules if f not in named]
        if missing:
            raise ValueError(f"FSDP unit names modules that do not exist: {missing}")
        targets = [named[f] for f in unit.modules]
        fully_shard(
            targets if len(targets) > 1 else targets[0],
            mesh=fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=unit.reshard_after_forward,
        )
        applied.append((unit.modules, unit.reshard_after_forward))
    return applied


def expected_uniform_token_count(micro_batch: int, seq_len: int) -> int:
    """Tokens each data-parallel rank contributes to one step.

    FSDP2's built-in ``/shard_degree`` on the gradient reduce-scatter is only the right global
    mean when this is the same on every rank. It is, here, by construction: fixed-length windows
    out of ``scratch_llm.train.get_batch``, no packing, no padding. Logged, so the assumption is
    visible.
    """
    return micro_batch * seq_len


__all__ = ["apply_fsdp", "expected_uniform_token_count"]
