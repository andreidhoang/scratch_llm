"""The parallelism decision, as data — and the one block of T1/T-R1 that is not delegated.

Everything mechanical about running an 8-GPU 2-D-parallel training step is written: the mesh,
the ``fully_shard`` traversal (``fsdp_wrap.py``), the DTensor placement plumbing
(``tensor_parallel.py``), the selective-AC application (``activation_ckpt.py``), the order the
three compose in (``parallelize.py``), async checkpointing (``dcp_async.py``) and the full
logging spec (``logging_spec.py``). All of it is driven by *one immutable value*: a
:class:`ParallelPlan`.

That value is the rung. Which modules get their own ``fully_shard`` call and at what
granularity; how TP=2 composes with the FSDP shard axis; what selective AC saves versus
recomputes — those are three memory-and-communication arithmetic problems, and copying
torchtitan's answers teaches nothing. So :func:`t1_parallel_plan` is a hole
(``python3 infra/holes.py``), guarded by exactly one test.

There is deliberately **no ``LADDERS_STUB_HOLES`` placeholder** here. A stub plan would be a
plan — it would wrap modules, shard parameters, and run — and no reader could tell it from a
draft of the answer. Instead the plumbing is tested against explicit ``ParallelPlan`` literals
written *in the tests*, so the mechanism is exercised today and the decision stays empty.

WHY THE SHAPE OF THIS VALUE IS WHAT IT IS
-----------------------------------------
A ``fully_shard`` call is the unit of communication. Everything inside one call is one
all-gather in the forward and one reduce-scatter in the backward. So ``fsdp_units`` — a list of
*groups* of module FQNs — is exactly the knob: one unit for the whole model is one enormous
all-gather that nothing can hide behind; one unit per Linear is a hundred tiny all-gathers that
cannot saturate the link. ``reshard_after_forward`` per unit is the second half of the same
trade: resharding frees the gathered parameters immediately and pays a second all-gather in the
backward; not resharding keeps them resident. torchtitan's answer is at
``torchtitan/distributed/fsdp.py:246-368`` — read it, then decide, then diff.

``tp_styles`` maps module-FQN globs to a placement style. It is ordered and first-match-wins, so
a specific rule can precede a general one.

``sac_save_ops`` / ``sac_save_every_other_mm`` are the recompute-versus-store trade at op
granularity: which aten outputs are worth their bytes.

Sources for the arithmetic (all local, no network):
  torchtitan/distributed/fsdp.py:168-374                 apply_fsdp_to_decoder: granularity, reshard
  torchtitan/models/common/decoder_sharding.py:122-152   colwise / rowwise placements
  torchtitan/distributed/activation_checkpoint.py:31-95  the default SAC save set
  torchtitan/distributed/activation_checkpoint.py:186-290  save-every-other-mm
  scratch_llm/src/scratch_llm/utils/memory_math.py       parameter / activation byte model
  scratch_llm/src/scratch_llm/utils/comms_calc.py        all-gather / reduce-scatter / all-reduce bytes
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from typing import Literal

from torch import nn

from scratch_llm.training.rung_config import T1RungConfig

# The placement styles the DTensor plumbing in ``tensor_parallel.py`` implements. Naming them
# here (rather than letting the plan carry arbitrary strings) is what makes a plan that asks for
# something unimplemented fail at validation on a laptop instead of at step 300 on a rented box.
TPStyle = Literal["colwise", "rowwise", "replicate", "embedding_colwise"]
TP_STYLES: tuple[str, ...] = ("colwise", "rowwise", "replicate", "embedding_colwise")

# aten op names a SAC policy may name. Strings, not op objects, so a plan stays a plain value
# that can be printed into a log and diffed across runs; ``activation_ckpt.py`` resolves them.
SAC_SAVEABLE_OPS: tuple[str, ...] = (
    "aten.mm.default",
    "aten.bmm.default",
    "aten.matmul.default",
    "aten.addmm.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_efficient_attention.default",
    "aten._scaled_dot_product_cudnn_attention.default",
    "aten._scaled_dot_product_attention_math.default",
    "aten.max.default",
    "_c10d_functional.reduce_scatter_tensor.default",
    "_c10d_functional.all_gather_into_tensor.default",
)


def fqn_match(fqn: str, pattern: str) -> bool:
    """Glob over FQN *segments*: ``*`` stays inside one dot-separated segment, ``**`` spans any.

    Plain ``fnmatch`` treats ``.`` as an ordinary character, so ``blocks.*`` would also match
    ``blocks.0.attn.q_norm`` — every rule in a plan silently wider than it reads, and an AC glob
    meant for whole blocks wrapping every leaf module inside them. Segment matching is what makes
    ``blocks.*`` mean "each block" and ``blocks.**`` mean "everything under blocks".
    """
    f = fqn.split(".") if fqn else []
    q = pattern.split(".") if pattern else []
    return _seg_match(tuple(f), tuple(q))


def _seg_match(f: tuple[str, ...], q: tuple[str, ...]) -> bool:
    if not q:
        return not f
    if q[0] == "**":
        return _seg_match(f, q[1:]) or (bool(f) and _seg_match(f[1:], q))
    if not f:
        return False
    return fnmatch.fnmatchcase(f[0], q[0]) and _seg_match(f[1:], q[1:])


@dataclass(frozen=True)
class FsdpUnit:
    """One ``fully_shard`` call: the modules it covers, and whether it reshards after forward.

    ``modules`` holds FQNs relative to the model root; ``""`` is the root itself. Listing more
    than one FQN in a unit means FSDP treats them as a *single* communication group — the reason
    torchtitan groups ``norm`` with ``lm_head`` (``torchtitan/distributed/fsdp.py:260-265``) and,
    under weight tying, the embedding with both (``:238-250``): a tied parameter gathered twice
    is a wasted all-gather.
    """

    modules: tuple[str, ...]
    reshard_after_forward: bool = True

    def __post_init__(self) -> None:
        if not self.modules:
            raise ValueError("an FSDP unit must cover at least one module")
        if len(set(self.modules)) != len(self.modules):
            raise ValueError(f"duplicate FQN inside one FSDP unit: {self.modules}")


@dataclass(frozen=True)
class ParallelPlan:
    """An immutable description of how one model is spread over one device mesh.

    Pure data — no modules, no tensors, no process group. It can be built before
    ``torch.distributed`` is initialized, printed into the run log verbatim (it *is* part of the
    logging spec's ``config`` field), diffed between two runs, and unit-tested on a laptop.
    """

    mesh_shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...]
    fsdp_mesh_dim: str
    fsdp_units: tuple[FsdpUnit, ...]
    tp_mesh_dim: str | None = None
    tp_styles: tuple[tuple[str, str], ...] = ()
    ac_modules: tuple[str, ...] = ()
    sac_save_ops: tuple[str, ...] = ()
    sac_save_every_other_mm: bool = False
    compile_modules: tuple[str, ...] = ()
    notes: str = ""

    # ---- shape-only checks, cheap enough to run in ``__post_init__`` ----------------------

    def __post_init__(self) -> None:
        if len(self.mesh_shape) != len(self.mesh_dim_names):
            raise ValueError(
                f"mesh_shape {self.mesh_shape} and mesh_dim_names {self.mesh_dim_names} "
                "must have the same length"
            )
        if len(set(self.mesh_dim_names)) != len(self.mesh_dim_names):
            raise ValueError(f"duplicate mesh dim name in {self.mesh_dim_names}")
        if any(d < 1 for d in self.mesh_shape):
            raise ValueError(f"mesh dims must be >= 1, got {self.mesh_shape}")
        if self.fsdp_mesh_dim not in self.mesh_dim_names:
            raise ValueError(
                f"fsdp_mesh_dim {self.fsdp_mesh_dim!r} is not a mesh dim {self.mesh_dim_names}"
            )
        if self.tp_mesh_dim is not None and self.tp_mesh_dim not in self.mesh_dim_names:
            raise ValueError(
                f"tp_mesh_dim {self.tp_mesh_dim!r} is not a mesh dim {self.mesh_dim_names}"
            )
        if self.tp_mesh_dim is None and self.tp_styles:
            raise ValueError("tp_styles given but tp_mesh_dim is None — nothing to shard onto")
        if not self.fsdp_units:
            raise ValueError("a plan must contain at least one FSDP unit")
        for _, style in self.tp_styles:
            if style not in TP_STYLES:
                raise ValueError(f"unknown TP style {style!r}; implemented: {TP_STYLES}")
        for op in self.sac_save_ops:
            if op not in SAC_SAVEABLE_OPS:
                raise ValueError(f"unknown SAC op {op!r}; resolvable: {SAC_SAVEABLE_OPS}")
        seen: set[str] = set()
        for unit in self.fsdp_units:
            for fqn in unit.modules:
                if fqn in seen:
                    raise ValueError(
                        f"module {fqn!r} appears in two FSDP units — a module can belong to "
                        "exactly one communication group"
                    )
                seen.add(fqn)

    # ---- derived quantities ---------------------------------------------------------------

    @property
    def world_size(self) -> int:
        return math.prod(self.mesh_shape)

    @property
    def fsdp_degree(self) -> int:
        return self.mesh_shape[self.mesh_dim_names.index(self.fsdp_mesh_dim)]

    @property
    def tp_degree(self) -> int:
        if self.tp_mesh_dim is None:
            return 1
        return self.mesh_shape[self.mesh_dim_names.index(self.tp_mesh_dim)]

    def tp_style_for(self, fqn: str) -> str | None:
        """First matching glob wins; ``None`` means "leave this module alone"."""
        for pattern, style in self.tp_styles:
            if fqn_match(fqn, pattern):
                return style
        return None

    def matches_ac(self, fqn: str) -> bool:
        return any(fqn_match(fqn, p) for p in self.ac_modules)

    def matches_compile(self, fqn: str) -> bool:
        return any(fqn_match(fqn, p) for p in self.compile_modules)

    def as_log_dict(self) -> dict[str, object]:
        """The plan as it appears in the run's ``config`` log field — the whole decision, so a
        reader of one log line can reconstruct which run this was."""
        return {
            "mesh_shape": list(self.mesh_shape),
            "mesh_dim_names": list(self.mesh_dim_names),
            "fsdp_mesh_dim": self.fsdp_mesh_dim,
            "tp_mesh_dim": self.tp_mesh_dim,
            "fsdp_units": [
                {"modules": list(u.modules), "reshard_after_forward": u.reshard_after_forward}
                for u in self.fsdp_units
            ],
            "tp_styles": [list(t) for t in self.tp_styles],
            "ac_modules": list(self.ac_modules),
            "sac_save_ops": list(self.sac_save_ops),
            "sac_save_every_other_mm": self.sac_save_every_other_mm,
            "compile_modules": list(self.compile_modules),
            "notes": self.notes,
        }


def validate_plan(plan: ParallelPlan, model: nn.Module, cfg: T1RungConfig) -> None:
    """Every way a plan can be wrong that a laptop can see, checked before a GPU is rented.

    Six checks, each one a real failure mode of 2-D parallel training:

    1. **The mesh factors the world.** ``prod(mesh_shape) == cfg.world_size``. A mesh that does
       not is an ``init_device_mesh`` error at rank 0 and a hang everywhere else.
    2. **The declared TP degree is the config's.** The plan may choose *how* TP composes; it may
       not quietly change the rung to TP=1.
    3. **Every FQN resolves.** A typo'd module name is otherwise a silently un-sharded module —
       memory and comms both wrong, nothing raises.
    4. **Every parameter lands in exactly one FSDP unit.** Coverage is by subtree, root last.
       A parameter in no unit is replicated (ZeRO-0 for that tensor, silently); a parameter in
       two is double-registered.
    5. **TP divides what TP shards.** ``n_heads``, ``n_kv_heads`` and ``ffn_hidden`` must each be
       divisible by ``tp_degree``, or a column-parallel projection produces an uneven head split
       and ``.view(b, s, n_heads_local, head_dim)`` reshapes into garbage rather than raising.
    6. **Vocab-parallel embedding is refused.** ``embedding_colwise`` shards the *feature* axis.
       Sharding an ``Embedding`` on the vocab axis needs an index mask plus a Partial all-reduce,
       which ``tensor_parallel.py`` does not implement; ``scratch_llm.model.Embedding`` does a
       raw ``self.weight[token_ids]`` that would silently return the wrong rows.
    """
    if math.prod(plan.mesh_shape) != cfg.world_size:
        raise ValueError(
            f"mesh {plan.mesh_shape} has {math.prod(plan.mesh_shape)} ranks but the rung runs on "
            f"{cfg.world_size}"
        )
    if plan.tp_degree != cfg.tp_degree:
        raise ValueError(
            f"plan TP degree {plan.tp_degree} != the rung's {cfg.tp_degree} "
            "(the plan chooses how TP composes, not whether it exists)"
        )
    named = dict(model.named_modules())
    for unit in plan.fsdp_units:
        for fqn in unit.modules:
            if fqn not in named:
                raise ValueError(f"FSDP unit names {fqn!r}, which is not a module of the model")
    for pattern, _ in plan.tp_styles:
        if not any(fqn_match(name, pattern) for name in named):
            raise ValueError(f"tp_styles pattern {pattern!r} matches no module")
    for pattern in plan.ac_modules:
        if not any(fqn_match(name, pattern) for name in named):
            raise ValueError(f"ac_modules pattern {pattern!r} matches no module")
    for pattern in plan.compile_modules:
        if not any(fqn_match(name, pattern) for name in named):
            raise ValueError(f"compile_modules pattern {pattern!r} matches no module")

    # 4. coverage: walk units in order, root ("") last, and assign each parameter to the FIRST
    # unit whose subtree contains it — which is how ``fully_shard`` itself behaves (an outer
    # call only manages what no inner call already claimed).
    owner: dict[int, str] = {}
    ordered = sorted(
        ((fqn, i) for i, u in enumerate(plan.fsdp_units) for fqn in u.modules),
        key=lambda t: (t[0] == "", -len(t[0])),
    )
    for fqn, _idx in ordered:
        for pname, p in named[fqn].named_parameters():
            key = id(p)
            if key not in owner:
                owner[key] = f"{fqn}.{pname}" if fqn else pname
    uncovered = [name for name, p in model.named_parameters() if id(p) not in owner]
    if uncovered:
        raise ValueError(
            f"{len(uncovered)} parameter(s) belong to no FSDP unit and would stay replicated: "
            f"{uncovered[:5]}{' …' if len(uncovered) > 5 else ''}"
        )

    # 5. divisibility
    tp = cfg.tp_degree
    if plan.tp_styles:
        for label, value in (
            ("n_heads", cfg.shape.n_heads),
            ("n_kv_heads", cfg.shape.n_kv_heads),
            ("ffn_hidden", cfg.shape.ffn_hidden),
        ):
            if value % tp:
                raise ValueError(f"tp_degree {tp} does not divide {label}={value}")

    # 6. vocab-parallel embedding
    for pattern, style in plan.tp_styles:
        if style in ("colwise", "rowwise"):
            for name, mod in named.items():
                if fqn_match(name, pattern) and mod.__class__.__name__ == "Embedding":
                    raise ValueError(
                        f"{name!r} is an Embedding and cannot take style {style!r}: "
                        "vocab-parallel embedding needs an index mask + Partial all-reduce that "
                        "tensor_parallel.py does not implement. Use 'embedding_colwise' "
                        "(feature-axis shard) or 'replicate'."
                    )


# =============================================================================================
# The hole — the decision itself
# =============================================================================================


def t1_parallel_plan(cfg: T1RungConfig) -> ParallelPlan:
    # HUY: the T1 parallelism decision — fully_shard granularity, TP=2 composition, SAC save set — spec: experiments/T1/T-R1/spec.md — fill before T-R1
    #
    # One line, deliberately: infra/holes.py matches `HUY:` and `spec:` within a single line, so
    # a marker wrapped for width drops out of `make holes` and the morning inventory loses this
    # rung. `ruff format` will not touch a comment; it *will* wrap the raise below, which is why
    # tests/conftest.py greps for the sentinel as a regex.
    #
    # THREE DECISIONS, each argued from arithmetic you can do before renting anything. The model
    # is 1.24 B parameters at the shape in rung_config.py; the world is 8×H100 SXM with NVLink
    # inside the node; the step is cfg.global_tokens_per_step tokens.
    #
    # (a) FSDP GRANULARITY — how many `fully_shard` calls, over which modules.
    #     The quantity to write down is, per unit: bytes all-gathered = unit_params ×
    #     sizeof(param_dtype) × (fsdp_degree − 1)/fsdp_degree, and the compute time the *next*
    #     unit's forward gives you to hide it behind. A unit whose all-gather takes longer than
    #     the previous unit's compute is exposed on the critical path; a unit small enough that
    #     the collective is launch-latency-bound wastes the link. Both ends are computable from
    #     utils/comms_calc.py and utils/memory_math.py — do that, then pick.
    #     Also decide `reshard_after_forward` per unit: resharding frees
    #     unit_params × (1 − 1/fsdp_degree) × sizeof(param_dtype) between forward and backward
    #     and buys a second all-gather. The last unit executed in the forward is the first needed
    #     in the backward, which is why torchtitan does not reshard the head
    #     (torchtitan/distributed/fsdp.py:258-265, :246-250 for the tied case). Our model ties
    #     `lm_head.weight` to `token_emb.weight` (model.py:463-464) — the same parameter under two
    #     FQNs. Decide once whether they share a unit and say why.
    #
    # (b) TP=2 COMPOSITION — which modules take which style, and what the mesh order is.
    #     The mesh is (fsdp, tp) or (tp, fsdp); rank ordering decides which pairs of GPUs are a TP
    #     group, and a TP group split across a slower link is the classic silent 2×. The styles
    #     available are in TP_STYLES; the standard Megatron pairing is colwise on q/k/v and
    #     w1/w3, rowwise on o_proj and w2, so exactly one all-reduce (or reduce-scatter under SP)
    #     lands per attention and per FFN. What to compute: the per-step TP collective bytes
    #     (2 per layer × batch × seq × d_model × sizeof, times the ring factor) against the
    #     activation bytes TP saves. Note the interaction — TP shrinks each FSDP unit's parameter
    #     shard by another factor of tp_degree, which moves answer (a).
    #     Note also: `tensor_parallel.py` refuses `colwise`/`rowwise` on an Embedding (see
    #     validate_plan check 6). If the arithmetic says vocab-parallel embedding is the right
    #     answer, that is a finding, and the plumbing gap is the next rung's work, not a reason to
    #     write a different plan.
    #
    # (c) SELECTIVE AC — which op outputs are worth their bytes.
    #     Per transformer block, for one microbatch, write down (bytes saved, FLOPs to recompute)
    #     for each candidate: the SDPA output, the four attention projections, the three FFN
    #     matmuls, the two RMSNorms, the SwiGLU elementwise product. Save the ones with the
    #     highest recompute-FLOPs-per-saved-byte until the activation budget fits; recompute the
    #     rest. `sac_save_every_other_mm` is torchtitan's blunt version of the same idea
    #     (activation_checkpoint.py:266-273) — decide whether you want it or a hand-picked set.
    #     `ac_modules` decides the *wrapping* granularity (whole block vs. attention only), which
    #     is a different question from which ops are saved inside it.
    #
    # Falsify before shipping: `pytest tests/training/test_t1_t_r1.py -k plan` checks the plan is
    # structurally sound and covers every parameter; `utils/memory_math.py` prints the predicted
    # resident bytes; the run's own log emits `activation_max_abs` per residual point and peak
    # memory per step, so the predicted and measured activation footprints can be diffed on the
    # first 20 warm-up steps before the measured 50 begin.
    raise NotImplementedError("HUY: T1/T-R1 parallelism plan — see spec.md")


__all__ = [
    "SAC_SAVEABLE_OPS",
    "TP_STYLES",
    "FsdpUnit",
    "ParallelPlan",
    "TPStyle",
    "fqn_match",
    "t1_parallel_plan",
    "validate_plan",
]
