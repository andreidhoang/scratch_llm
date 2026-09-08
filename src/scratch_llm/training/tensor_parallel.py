"""TP=2 via DTensor — the placement plumbing, and the two fixups nobody warns you about.

The model code in ``scratch_llm/model.py`` is not touched. Tensor parallelism is applied from
the outside with :func:`torch.distributed.tensor.distribute_module`, which does exactly two
things per module: replaces its parameters with :class:`~torch.distributed.tensor.DTensor`
shards, and installs forward hooks that convert the module's *boundary* tensors into and out of
DTensor. Between modules, activations stay plain local tensors — so ``.view``, ``.transpose``
and the attention math run unchanged on each rank's shard.

Why not ``ColwiseParallel``/``RowwiseParallel`` from ``torch.distributed.tensor.parallel``:
those dispatch on ``isinstance(module, nn.Linear)`` and raise ``NotImplementedError`` for
anything else. ``scratch_llm.model.Linear`` is our own module (``model.py:125-133``, ``y = x @
Wᵀ``, no bias), so the stock styles cannot see it. torchtitan @ d263ca0a does not use them
either — it declares placements through ``spmd_types`` and lets ``Module.parallelize`` apply
them (``torchtitan/models/common/decoder_sharding.py:122-152``). Same three placements, three
different spellings.

THE FOUR STYLES
---------------
Weights are stored ``(out_features, in_features)``, so the placement axis is the opposite of
what "colwise" suggests — read the shapes, not the name.

- ``colwise``      weight ``Shard(0)`` (split ``out``). Input arrives replicated; ``x @ Wᵀ``
                   yields ``Shard(-1)``; the hook returns the local last-dim shard. No collective
                   in the forward; the backward's input-gradient is Partial and all-reduces.
- ``rowwise``      weight ``Shard(1)`` (split ``in``). Input is the last-dim shard from the
                   preceding colwise module; ``x @ Wᵀ`` yields ``Partial``; the hook redistributes
                   to ``Replicate`` — that is the one all-reduce per attention and per FFN.
- ``replicate``    weight ``Replicate``. No sharding, no collective. Needed anyway: see FIXUP 2.
- ``embedding_colwise``  ``Embedding`` weight ``Shard(1)`` — the *feature* axis. The vocab axis
                   is deliberately not supported; see ``parallel_plan.validate_plan`` check 6.

Colwise-then-rowwise is the Megatron pairing: q/k/v and w1/w3 colwise, o_proj and w2 rowwise,
one all-reduce per sub-layer. The plan decides whether that is the right pairing here; this
module only knows how to execute each style.

FIXUP 1 — the head count is part of the model's *shape*, not just its weights.
``MultiHeadSelfAttention.forward`` does ``self.q_proj(x).view(b, s, self.n_heads,
self.head_dim)`` (``model.py:274``). After a colwise q_proj the local tensor has
``n_heads/tp × head_dim`` columns, so the view either raises (if the numbers do not multiply) or
— worse, and this is the silent one — succeeds with a different head/dim split and produces a
plausible loss curve for a plausible-looking model that is not the model. :func:`apply_tensor_parallel`
therefore divides ``n_heads`` and ``n_kv`` on every attention module whose projections it
sharded, and asserts divisibility first. torchtitan does the same thing declaratively, by making
the head axis part of the placement.

FIXUP 2 — a tied parameter is two FQNs and one tensor, and sharding un-ties it.
``register_parameter`` installs a *new* Parameter on the module it shards, so sharding
``token_emb`` leaves ``lm_head.weight`` pointing at the old, unsharded tensor: the model silently
stops being weight-tied, the head trains a second 128256x2048 table, and the parameter count
in the log goes up by 263 M while nothing raises. :func:`apply_tensor_parallel` refuses a plan
that gives two tied FQNs different styles, and re-establishes the tie afterwards.

Our tie is ``lm_head.weight is token_emb.weight`` (``model.py:463-464``), which under TP admits
only ``replicate``: torchtitan shards the tied pair on the *vocab* axis (``S(0)`` for both
``tok_embeddings`` and ``lm_head``, ``decoder_sharding.py:373,387``), and vocab-parallel embedding
is the one thing this module does not implement. That is a real limitation and it is named in
``experiments/T1/T-R1/map.md``, not hidden.

FIXUP 3 — every parameter must end up on the same mesh.
FSDP2 is applied *after* TP and turns each parameter into a DTensor on the FSDP axis. A
parameter that TP already sharded becomes 2-D ``(fsdp, tp)``; a parameter TP never touched
becomes 1-D ``(fsdp,)``. Global gradient-norm clipping then has to reduce over tensors living on
different meshes, and the fused optimizer over a mixed set. torchtitan solves this with a
``NoParallel`` style whose entire purpose is to put untouched modules on the TP mesh as
``Replicate`` (``torchtitan/distributed/tensor_parallel.py:19-30``). ``replicate`` here is that
style. A plan that TP-shards the blocks but leaves the norms alone is the classic version of
this bug.
"""

from __future__ import annotations

from typing import Any

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard, distribute_module
from torch.distributed.tensor import distribute_tensor as _distribute_tensor

from scratch_llm.model import Embedding, MultiHeadSelfAttention
from scratch_llm.training.parallel_plan import ParallelPlan

# style -> (parameter placement, input placement, output placement after redistribute)
_STYLE_PLACEMENTS: dict[str, tuple[Placement, Placement, Placement]] = {
    "colwise": (Shard(0), Replicate(), Shard(-1)),
    "rowwise": (Shard(1), Shard(-1), Replicate()),
    "replicate": (Replicate(), Replicate(), Replicate()),
    "embedding_colwise": (Shard(1), Replicate(), Shard(-1)),
}


def _make_partition_fn(param_placement: Placement):
    def partition_fn(_name: str, module: nn.Module, mesh: DeviceMesh) -> None:
        # recurse=False: distribute_module already walks the tree, and a leaf style must not
        # reach into a child that has its own style.
        for pname, param in list(module.named_parameters(recurse=False)):
            module.register_parameter(
                pname,
                nn.Parameter(
                    _distribute_tensor(param.data, mesh, [param_placement]),
                    requires_grad=param.requires_grad,
                ),
            )

    return partition_fn


def _make_input_fn(input_placement: Placement):
    def input_fn(_module: nn.Module, inputs: tuple[Any, ...], mesh: DeviceMesh) -> tuple[Any, ...]:
        x = inputs[0]
        if not isinstance(x, DTensor):
            # run_check=False: the caller guarantees the local shards line up. run_check=True
            # would all-gather on every forward to verify — correct, and unaffordable.
            x = DTensor.from_local(x, mesh, (input_placement,), run_check=False)
        elif x.placements != (input_placement,):
            x = x.redistribute(placements=(input_placement,))
        return (x, *inputs[1:])

    return input_fn


def _make_output_fn(output_placement: Placement):
    def output_fn(_module: nn.Module, outputs: Any, mesh: DeviceMesh) -> Any:
        if not isinstance(outputs, DTensor):
            return outputs
        if outputs.placements != (output_placement,):
            outputs = outputs.redistribute(placements=(output_placement,))
        return outputs.to_local()

    return output_fn


def tied_parameter_groups(model: nn.Module) -> list[list[tuple[str, str]]]:
    """Groups of ``(module_fqn, param_name)`` that are the *same tensor*.

    ``named_parameters(remove_duplicate=False)`` is the only way to see a tie: the default
    deduplicates and reports a tied weight under one name, which is exactly the information
    needed here.
    """
    by_id: dict[int, list[tuple[str, str]]] = {}
    for fqn, p in model.named_parameters(remove_duplicate=False):
        module_fqn, _, pname = fqn.rpartition(".")
        by_id.setdefault(id(p), []).append((module_fqn, pname))
    return [group for group in by_id.values() if len(group) > 1]


def apply_tensor_parallel(
    model: nn.Module, plan: ParallelPlan, tp_mesh: DeviceMesh
) -> dict[str, str]:
    """Apply ``plan.tp_styles`` to ``model`` on ``tp_mesh``. Returns FQN → style actually applied.

    Idempotent in the sense that it visits each module once; calling it twice would re-shard an
    already-sharded parameter and is a caller error, not something this guards against.
    """
    if tp_mesh.ndim != 1:
        raise ValueError(f"tp_mesh must be 1-D, got ndim={tp_mesh.ndim}")
    tp = tp_mesh.size()
    ties = tied_parameter_groups(model)
    for group in ties:
        styles = {plan.tp_style_for(m) for m, _ in group}
        if len(styles) > 1:
            raise ValueError(
                f"tied parameter {group} would get different TP styles {styles} — one tensor "
                "cannot have two placements, and sharding it under one FQN silently un-ties it"
            )
        style = next(iter(styles))
        if style is not None and style != "replicate":
            raise ValueError(
                f"tied parameter {group} cannot take style {style!r}: the tied embed/head pair "
                "would need the SAME placement on the vocab axis, which is the vocab-parallel "
                "embedding this module does not implement. Use 'replicate' for both, and see "
                "experiments/T1/T-R1/map.md for what that costs."
            )
    applied: dict[str, str] = {}
    for fqn, module in list(model.named_modules()):
        style = plan.tp_style_for(fqn)
        if style is None:
            continue
        if style in ("colwise", "rowwise") and isinstance(module, Embedding):
            raise ValueError(
                f"{fqn}: style {style!r} on an Embedding is vocab-parallel and unimplemented "
                "(parallel_plan.validate_plan check 6)"
            )
        if style == "embedding_colwise" and not isinstance(module, Embedding):
            raise ValueError(f"{fqn}: style 'embedding_colwise' on a {type(module).__name__}")
        if not any(True for _ in module.parameters(recurse=False)):
            raise ValueError(
                f"{fqn} ({type(module).__name__}) has no parameters of its own — a TP style on "
                "it would shard nothing and only add two hooks per forward"
            )
        param_pl, in_pl, out_pl = _STYLE_PLACEMENTS[style]
        for pname, param in module.named_parameters(recurse=False):
            if isinstance(param_pl, Shard) and param.shape[param_pl.dim] % tp:
                raise ValueError(
                    f"{fqn}.{pname} has shape {tuple(param.shape)}; dim {param_pl.dim} is not "
                    f"divisible by tp_degree={tp}"
                )
        # torch types input_fn/output_fn as returning None; they are documented (and used
        # throughout torchtitan) as returning the transformed inputs/outputs.
        distribute_module(
            module,
            tp_mesh,
            _make_partition_fn(param_pl),
            _make_input_fn(in_pl),  # pyright: ignore[reportArgumentType]
            _make_output_fn(out_pl),  # pyright: ignore[reportArgumentType]
        )
        applied[fqn] = style

    _retie_parameters(model, ties)
    _fix_attention_head_counts(model, applied, tp)
    return applied


def _retie_parameters(model: nn.Module, ties: list[list[tuple[str, str]]]) -> None:
    """FIXUP 2 — re-point every member of a tie group at the first member's Parameter."""
    named = dict(model.named_modules())
    for group in ties:
        head_fqn, head_pname = group[0]
        shared = getattr(named[head_fqn], head_pname)
        for module_fqn, pname in group[1:]:
            setattr(named[module_fqn], pname, shared)


def _fix_attention_head_counts(model: nn.Module, applied: dict[str, str], tp: int) -> None:
    """FIXUP 1 — see the module docstring. Divide the per-rank head counts of every attention
    module whose q/k/v projections were column-sharded.

    Deliberately keyed on ``q_proj``/``k_proj``/``v_proj`` having style ``colwise`` rather than
    on "TP is on": an attention module left replicated keeps its full head count, and mixing the
    two in one model is a legal (if odd) plan.
    """
    for fqn, module in model.named_modules():
        if not isinstance(module, MultiHeadSelfAttention):
            continue
        q, k, v = (applied.get(f"{fqn}.{p}_proj") for p in ("q", "k", "v"))
        if q is None and k is None and v is None:
            continue
        if not (q == k == v == "colwise"):
            raise ValueError(
                f"{fqn}: q/k/v must share the 'colwise' style (got {q!r}/{k!r}/{v!r}) — a "
                "partially sharded projection triple produces a head split that .view() "
                "reshapes into garbage instead of raising"
            )
        if module.n_heads % tp or module.n_kv % tp:
            raise ValueError(
                f"{fqn}: tp_degree={tp} divides neither n_heads={module.n_heads} nor "
                f"n_kv={module.n_kv}"
            )
        module.n_heads //= tp
        module.n_kv //= tp


__all__ = ["apply_tensor_parallel", "tied_parameter_groups"]
