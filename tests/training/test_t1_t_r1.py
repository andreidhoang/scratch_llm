"""T1/T-R1 — FSDP2 + TP=2 + selective AC + async DCP + the full logging spec. Three tiers.

  CPU, one process   the logging spec (every named field, and nothing silently missing), the
                     every-100-steps schedule at both endpoints, per-group gradient norms, the
                     git sha read from the repo, the activation probe's residual points, plan
                     validation, and selective AC's gradient equivalence. Seconds.
  CPU, gloo          the real distributed assertions: FSDP2 at dp=2 must match single-process
                     full-batch training; TP=2 must match the unsharded model; async DCP must
                     round-trip; and per-group norms must still be right when every gradient is a
                     DTensor. A minute per world size, because torch takes that long to import
                     four times.
  8xH100             throughput, MFU, torch.compile, and the 85%-of-torchtitan comparison. Not
                     decidable on a laptop; SKIPPED with the exact command, never silently passed.

The CPU tiers carry the weight here for a reason specific to distributed training: every failure
mode of a 2-D-parallel step produces a plausible loss curve. A TP glob that matches nothing runs
data-parallel-only. A tied embedding that TP silently un-ties trains two tables. An FSDP unit list
that misses a module leaves it replicated. A per-group norm that is secretly the global norm
repeated tells you nothing on the day the run diverges. None of those fault, none are slow, and
all are decidable here.

Spec: experiments/T1/T-R1/spec.md   ·   Map: experiments/T1/T-R1/map.md
"""

from __future__ import annotations

import copy
import json
import os
import re
import socket
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn

from scratch_llm.model import (
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerLM,
    cross_entropy,
)
from scratch_llm.training.activation_ckpt import apply_selective_ac, resolve_save_ops
from scratch_llm.training.dcp_async import AsyncCheckpointer, TrainState
from scratch_llm.training.flops import LLAMA3_1B, flop_breakdown
from scratch_llm.training.logging_spec import (
    ALL_FIELDS,
    NCCL_WAIT_SCOPE,
    PERIODIC_EVERY,
    PERIODIC_FIELDS,
    REQUIRED_STEP_FIELDS,
    ActivationProbe,
    CommWaitTimer,
    RunLogger,
    clean_fqn,
    fires_periodic,
    git_head_sha,
    grad_norms,
    param_group,
    validate_step_record,
    weight_norms,
)
from scratch_llm.training.mp_policy import autocast_dtype, bf16_mp_policy
from scratch_llm.training.parallel_plan import (
    FsdpUnit,
    ParallelPlan,
    t1_parallel_plan,
    validate_plan,
)
from scratch_llm.training.parallelize import build_mesh, parallelize
from scratch_llm.training.rung_config import H100_SXM_PEAK_BF16, T_R1, T_R1_CPU
from scratch_llm.training.t1_train import train_t_r1
from scratch_llm.training.tensor_parallel import apply_tensor_parallel, tied_parameter_groups
from scratch_llm.training.titan_floor import PRESETS as TITAN_PRESETS
from scratch_llm.training.titan_floor import (
    SEQ_LEN as TITAN_SEQ_LEN,
)
from scratch_llm.training.titan_floor import (
    STEPS as TITAN_STEPS,
)
from scratch_llm.training.titan_floor import (
    TOKENS_PER_MICROBATCH_PER_DP_RANK as TITAN_TOKENS,
)
from scratch_llm.training.titan_floor import (
    WARMUP_STEPS as TITAN_WARMUP,
)
from scratch_llm.utils.seeding import seed_everything

# The command this rung's un-decidable-on-a-laptop assertions need. Quoted verbatim in every
# skip reason, so a skipped test tells you what to run rather than that it was skipped.
BOX_COMMAND = (
    "8xH100 SXM, one node (no container needed, no ncu):\n"
    "  infra/rent.sh sync-up <user@host> && ssh <host>\n"
    "  cd ladders && pip install -r oss/torchtitan/requirements.txt\n"
    "  PRESET=fsdp4_tp2 bash experiments/T1/T-R0/floor.sh      # -> titan_llama3_1b_fsdp4_tp2_tps\n"
    "  make predict L=T1 R=T-R1 M=pct_of_torchtitan_tps V=<your number> NOTE='<mechanism>'\n"
    "  T1_FLOOR_TPS=<floor> bash experiments/T1/T-R1/run.sh"
)

# fp32 everywhere on CPU: the gloo tiers assert *equivalence*, and bf16 would make every
# comparison a tolerance argument about the wrong thing. The bf16 policy is unit-tested
# separately; the dtype it selects is exercised on the box.
_CPU_CFG = replace(T_R1_CPU, param_dtype="float32", reduce_dtype="float32")

RTOL = 1e-5  # fp32 CPU, few-op reductions: the associativity spread of a 32-element sum
ATOL = 1e-6


# =============================================================================================
# helpers
# =============================================================================================


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _init_pg(rank: int, world: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)


def _build_cpu_model(cfg=_CPU_CFG) -> TransformerLM:
    seed_everything(0)
    return TransformerLM(cfg.model_config())


def _fsdp_only_plan(n_layers: int, dp: int) -> ParallelPlan:
    return ParallelPlan(
        mesh_shape=(dp,),
        mesh_dim_names=("dp",),
        fsdp_mesh_dim="dp",
        fsdp_units=(
            *(FsdpUnit((f"blocks.{i}",)) for i in range(n_layers)),
            FsdpUnit(("token_emb", "final_norm", "lm_head"), reshard_after_forward=False),
            FsdpUnit(("",)),
        ),
        ac_modules=("blocks.*",),
        sac_save_ops=("aten.mm.default", "aten.bmm.default"),
        sac_save_every_other_mm=False,
        notes="test plumbing plan, not a decision",
    )


def _tp_styles(n_layers: int) -> tuple[tuple[str, str], ...]:
    del n_layers
    return (
        # The tied pair must share a style, and under this plumbing that style is 'replicate'.
        ("token_emb", "replicate"),
        ("lm_head", "replicate"),
        ("final_norm", "replicate"),
        ("blocks.*.attn_norm", "replicate"),
        ("blocks.*.ffn_norm", "replicate"),
        ("blocks.*.attn.q_proj", "colwise"),
        ("blocks.*.attn.k_proj", "colwise"),
        ("blocks.*.attn.v_proj", "colwise"),
        ("blocks.*.attn.o_proj", "rowwise"),
        ("blocks.*.ffn.w1", "colwise"),
        ("blocks.*.ffn.w3", "colwise"),
        ("blocks.*.ffn.w2", "rowwise"),
    )


def _two_d_plan(n_layers: int, dp: int, tp: int) -> ParallelPlan:
    return ParallelPlan(
        mesh_shape=(dp, tp),
        mesh_dim_names=("dp", "tp"),
        fsdp_mesh_dim="dp",
        tp_mesh_dim="tp",
        fsdp_units=(
            *(FsdpUnit((f"blocks.{i}",)) for i in range(n_layers)),
            FsdpUnit(("token_emb", "final_norm", "lm_head"), reshard_after_forward=False),
            FsdpUnit(("",)),
        ),
        tp_styles=_tp_styles(n_layers),
        ac_modules=("blocks.*",),
        sac_save_ops=("aten.mm.default", "aten.bmm.default"),
        notes="test plumbing plan, not a decision",
    )


def _batch(cfg, n_seqs: int) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, cfg.shape.vocab_size, (n_seqs, cfg.seq_len), generator=g)
    tgt = torch.randint(0, cfg.shape.vocab_size, (n_seqs, cfg.seq_len), generator=g)
    return ids, tgt


def _attn(model: TransformerLM, i: int) -> MultiHeadSelfAttention:
    """``model.blocks[i].attn``, typed. ``ModuleList.__getitem__`` returns ``Module``."""
    block = model.blocks[i]
    assert isinstance(block, TransformerBlock)
    return block.attn


def _full_grads(model: nn.Module) -> dict[str, Tensor]:
    from torch.distributed.tensor import DTensor

    out: dict[str, Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        out[clean_fqn(name)] = g.full_tensor() if isinstance(g, DTensor) else g
    return out


# =============================================================================================
# tier 1 — the config is the floor's config
# =============================================================================================


def test_the_rung_config_is_the_floor_config_field_by_field() -> None:
    """T-R1's number is a percentage of T-R0's, so a field that differs invalidates the ratio.

    The fields are *imported* from ``titan_floor`` so they cannot drift; this asserts the import
    is still doing that job rather than having been replaced by a copied literal during a
    refactor, which is exactly how matched configs stop being matched.
    """
    assert T_R1.seq_len == TITAN_SEQ_LEN == 8192
    assert T_R1.tokens_per_microbatch_per_dp_rank == TITAN_TOKENS == 16384
    assert T_R1.total_steps == TITAN_STEPS
    assert T_R1.warmup_steps == TITAN_WARMUP
    assert T_R1.measure_steps >= 50, "workspace invariant 4"
    assert T_R1.shape is LLAMA3_1B
    assert T_R1.world_size == 8 and T_R1.tp_degree == 2 and T_R1.dp_degree == 4
    assert T_R1.micro_batch == 2


def test_the_floor_named_is_the_tp2_member_not_the_fsdp_only_one() -> None:
    """T-R0 measures two torchtitan numbers and hands the choice to this rung.

    The plan's T1 exit line is "≥ 85% torchtitan tok/s (FSDP2+TP=2)". Pointing at the FSDP-only
    preset would compare a TP=2 run against a TP=1 floor — two configs whose collectives differ
    per block — and the 85% would mean nothing.
    """
    assert T_R1.floor_preset == "fsdp4_tp2"
    assert T_R1.floor_metric == TITAN_PRESETS["fsdp4_tp2"].metric
    assert TITAN_PRESETS["fsdp4_tp2"].tensor_parallel_degree == T_R1.tp_degree


def test_mfu_uses_torchtitans_flop_convention_not_plain_6nd() -> None:
    """At seq 8192 the attention term is ~30% of model FLOPs. Dropping it inflates MFU by ~43%."""
    b = flop_breakdown(T_R1.shape, T_R1.seq_len, ac_policy=T_R1.ac_policy)
    assert b.attention_per_token > 0
    assert 0.25 < b.attention_share < 0.35, b.attention_share
    assert b.model_per_token > b.dense_per_token
    # An MFU computed off dense-only would be this much higher for the same stopwatch reading.
    assert b.model_per_token / b.dense_per_token > 1.4
    assert H100_SXM_PEAK_BF16 == 989.5e12  # dense bf16, not the 2:4-sparse headline


# =============================================================================================
# tier 1 — the logging spec
# =============================================================================================


def _minimal_record(step: int = 0, **over) -> dict:
    rec = {
        "step": step,
        "loss": 1.0,
        "lr": 3e-4,
        "grad_norm_global": 1.0,
        "grad_norm_groups": {"attn": 0.6, "mlp": 0.8},
        "tokens_per_s_per_device": 1000.0,
        "mfu": 0.4,
        "nccl_wait_s": 0.001,
        "step_time_s": 0.5,
        "peak_memory_bytes": 1 << 30,
        "git_sha": "0" * 40,
        "config": {"name": "x"},
    }
    rec.update(over)
    return rec


def test_the_logger_emits_every_field_the_plan_names() -> None:
    """Every field of plan §05's T-R1 row appears, on the steps it is supposed to appear on."""
    logger = RunLogger(config={"name": "t"}, total_steps=250, periodic_every=100, out_path=None)
    plain = logger.log_step(
        step=7,
        loss=2.0,
        lr=1e-4,
        grad_norm_global=1.5,
        grad_norm_groups={"attn": 1.0, "mlp": 1.1},
        tokens_per_s_per_device=1234.0,
        mfu=0.42,
        nccl_wait_s=0.002,
        step_time_s=0.4,
        peak_memory_bytes=123,
    )
    assert set(plain) >= REQUIRED_STEP_FIELDS
    assert not (PERIODIC_FIELDS & set(plain))
    periodic = logger.log_step(
        step=100,
        loss=2.0,
        lr=1e-4,
        grad_norm_global=1.5,
        grad_norm_groups={"attn": 1.0, "mlp": 1.1},
        tokens_per_s_per_device=1234.0,
        mfu=0.42,
        nccl_wait_s=0.002,
        step_time_s=0.4,
        peak_memory_bytes=123,
        activation_max_abs={"resid.embed": 1.0},
        weight_norms_={"token_emb.weight": 2.0},
    )
    assert set(periodic) >= REQUIRED_STEP_FIELDS | PERIODIC_FIELDS
    # every named field of the plan row is reachable
    for named in (
        "loss",
        "lr",
        "grad_norm_global",
        "grad_norm_groups",
        "tokens_per_s_per_device",
        "mfu",
        "activation_max_abs",
        "weight_norms",
        "nccl_wait_s",
        "git_sha",
        "config",
    ):
        assert named in ALL_FIELDS


@pytest.mark.parametrize("dropped", sorted(REQUIRED_STEP_FIELDS))
def test_dropping_any_required_field_is_caught(dropped: str) -> None:
    """The other direction. A spec that only checks presence of what is present checks nothing."""
    rec = _minimal_record()
    del rec[dropped]
    with pytest.raises(ValueError, match="logging spec violated"):
        validate_step_record(rec, periodic=False)


def test_an_unknown_field_is_caught_too() -> None:
    """A metric renamed while the old name keeps being written is how two dashboards diverge."""
    with pytest.raises(ValueError, match="unknown field"):
        validate_step_record(_minimal_record(loss_v2=1.0), periodic=False)


def test_a_periodic_field_on_a_non_periodic_step_is_caught() -> None:
    with pytest.raises(ValueError, match="periodic field"):
        validate_step_record(_minimal_record(activation_max_abs={"a": 1.0}), periodic=False)


def test_grad_norm_groups_must_be_a_non_empty_mapping() -> None:
    """The whole point of the field: one number relabelled is not a per-group breakdown."""
    with pytest.raises(ValueError, match="not a per-group norm"):
        validate_step_record(_minimal_record(grad_norm_groups={}), periodic=False)


def test_a_periodic_step_with_unarmed_probes_refuses_rather_than_logging_nothing() -> None:
    logger = RunLogger(config={}, total_steps=250, periodic_every=100)
    with pytest.raises(ValueError, match="did not arm the probes"):
        logger.log_step(
            step=100,
            loss=1.0,
            lr=1e-4,
            grad_norm_global=1.0,
            grad_norm_groups={"attn": 1.0},
            tokens_per_s_per_device=1.0,
            mfu=0.1,
            nccl_wait_s=0.0,
            step_time_s=0.1,
            peak_memory_bytes=0,
        )


def test_nccl_wait_scope_is_stated_and_the_header_carries_it(tmp_path: Path) -> None:
    """The field is named as the plan names it; the log says what it does and does not include."""
    assert "FSDP2" in NCCL_WAIT_SCOPE and "NOT" in NCCL_WAIT_SCOPE
    out = tmp_path / "steps.jsonl"
    RunLogger(config={"a": 1}, total_steps=10, out_path=out).close()
    header = json.loads(out.read_text().splitlines()[0])
    assert header["kind"] == "run_header"
    assert header["nccl_wait_scope"] == NCCL_WAIT_SCOPE
    assert sorted(REQUIRED_STEP_FIELDS) == header["required_step_fields"]


def test_comm_wait_timer_accumulates_and_resets() -> None:
    t = CommWaitTimer()
    with t.measure():
        pass
    first = t.take()
    assert first >= 0.0
    assert t.take() == 0.0


# =============================================================================================
# tier 1 — the every-100-steps schedule
# =============================================================================================


def test_periodic_schedule_fires_on_exactly_the_right_steps() -> None:
    """Step 0 and the last step included, and nothing else.

    Both endpoints matter: step 0 is the only measurement of the initialization (the baseline
    every later activation number is read against) and ``total-1`` is the state about to be
    checkpointed. ``249`` is deliberately not a multiple of 100 — a schedule that only fires on
    multiples passes a test written with total=300 and silently drops the last measurement of
    every real run.
    """
    total, every = 250, 100
    fired = {s for s in range(total) if fires_periodic(s, every, total)}
    assert fired == {0, 100, 200, 249}


def test_periodic_schedule_when_the_last_step_is_a_multiple() -> None:
    total, every = 201, 100
    fired = {s for s in range(total) if fires_periodic(s, every, total)}
    assert fired == {0, 100, 200}  # 200 == total-1, no duplicate


def test_periodic_schedule_disabled() -> None:
    assert not any(fires_periodic(s, 0, 10) for s in range(10))
    with pytest.raises(ValueError):
        fires_periodic(-1, 100, 10)


def test_the_integrated_logger_fires_on_the_same_steps() -> None:
    """The predicate and the loop must agree — a schedule tested only in isolation is not tested."""
    logger = RunLogger(config={}, total_steps=250, periodic_every=100)
    for step in range(250):
        periodic = logger.is_periodic(step)
        logger.log_step(
            step=step,
            loss=1.0,
            lr=1e-4,
            grad_norm_global=1.0,
            grad_norm_groups={"attn": 1.0},
            tokens_per_s_per_device=1.0,
            mfu=0.1,
            nccl_wait_s=0.0,
            step_time_s=0.1,
            peak_memory_bytes=0,
            activation_max_abs={"resid.embed": 1.0} if periodic else None,
            weight_norms_={"w": 1.0} if periodic else None,
        )
    emitted = {r["step"] for r in logger.records if "activation_max_abs" in r}
    assert emitted == {0, 100, 200, 249}
    assert {r["step"] for r in logger.records if "weight_norms" in r} == emitted
    assert PERIODIC_EVERY == 100, "the plan says /100 steps"


# =============================================================================================
# tier 1 — per-group gradient norms
# =============================================================================================


def test_param_group_assigns_every_parameter_of_the_real_model() -> None:
    model = _build_cpu_model()
    groups = {param_group(n) for n, _ in model.named_parameters()}
    assert "other" not in groups, "a new parameter kind appeared and nobody classified it"
    assert {"embed", "norm", "attn", "mlp"} <= groups


def test_grad_norm_groups_actually_group() -> None:
    """Each group's norm is its own, and they compose to the global one.

    Constructed so that a "global norm reported five times" implementation cannot pass: every
    group is given a *different, exactly known* squared norm, and each is asserted individually.
    The composition identity ``global² == Σ group²`` is asserted on top, because a per-group
    breakdown that does not add up to the global norm is measuring something else.
    """
    model = _build_cpu_model()
    wanted = {"embed": 3.0, "norm": 5.0, "attn": 7.0, "mlp": 11.0}
    per_group_params: dict[str, list[nn.Parameter]] = {}
    for fqn, p in model.named_parameters():
        per_group_params.setdefault(param_group(fqn), []).append(p)
    assert set(per_group_params) == set(wanted), sorted(per_group_params)

    # Give every parameter in a group a gradient of a constant c, chosen so the group's L2 norm
    # is exactly `wanted[g]`: ‖g‖ = c·sqrt(numel_total).
    for g, params in per_group_params.items():
        n = sum(p.numel() for p in params)
        c = wanted[g] / (n**0.5)
        for p in params:
            p.grad = torch.full_like(p, c)

    total, groups = grad_norms(model)
    assert set(groups) == set(wanted)
    for g, want in wanted.items():
        assert groups[g] == pytest.approx(want, rel=1e-5), f"{g}: {groups[g]} != {want}"
    # they are genuinely different numbers — no implementation that returns one value passes
    assert len(set(round(v, 6) for v in groups.values())) == len(groups)
    expected_total = sum(v * v for v in wanted.values()) ** 0.5
    assert total == pytest.approx(expected_total, rel=1e-5)
    assert total == pytest.approx(sum(v * v for v in groups.values()) ** 0.5, rel=1e-6), (
        "global² must equal Σ group²"
    )


def test_weight_tying_leaves_the_head_group_empty_and_that_is_reported_not_faked() -> None:
    """``lm_head.weight is token_emb.weight``: ``named_parameters`` dedups, so there is no
    separate head gradient. Reporting ``head: 0.0`` would read as "the head has no gradient",
    which is the opposite of true."""
    model = _build_cpu_model()
    assert model.cfg.tie_embeddings
    assert model.lm_head.weight is model.token_emb.weight
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    _, groups = grad_norms(model)
    assert "head" not in groups
    assert "embed" in groups


def test_group_norms_ignore_parameters_without_gradients() -> None:
    model = _build_cpu_model()
    for fqn, p in model.named_parameters():
        p.grad = None if param_group(fqn) == "mlp" else torch.ones_like(p)
    _, groups = grad_norms(model)
    assert "mlp" not in groups and "attn" in groups


# =============================================================================================
# tier 1 — provenance
# =============================================================================================


def test_git_sha_is_the_real_head_read_not_declared() -> None:
    """Recomputed independently here, from the resolved source path.

    ``scratch_llm/`` is a symlink in the workspace; an unresolved ``__file__`` would run ``git``
    in the outer repo and record a sha that looks fine and belongs to a different commit.
    """
    sha = git_head_sha()
    assert re.fullmatch(r"[0-9a-f]{40}", sha), sha
    import scratch_llm.training.logging_spec as ls

    repo = Path(ls.__file__).resolve().parent
    expected = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert sha == expected
    assert RunLogger(config={}, total_steps=1).git_sha == expected


def test_the_config_recorded_is_the_config_that_ran() -> None:
    cfg_log = T_R1.as_log_dict()
    assert cfg_log["seq_len"] == T_R1.seq_len
    assert cfg_log["tp_degree"] == T_R1.tp_degree
    assert cfg_log["dp_degree"] == T_R1.dp_degree
    assert cfg_log["floor_metric"] == T_R1.floor_metric
    assert cfg_log["param_dtype"] == "bfloat16" and cfg_log["reduce_dtype"] == "float32"
    json.dumps(cfg_log)  # a config that cannot be serialized is a config that is not logged


# =============================================================================================
# tier 1 — activation probe
# =============================================================================================


def test_activation_probe_covers_every_residual_point_and_matches_a_manual_forward() -> None:
    """One point per residual-stream location plus one per sub-layer delta, values exact.

    ``TransformerBlock.forward`` is ``x = x + attn(...); x = x + ffn(...)``: the block's own
    output is the residual stream after both adds, and the two children are the deltas. Both are
    recorded, because "the stream grew" and "one layer's contribution grew" are different bugs.
    """
    model = _build_cpu_model()
    n = model.cfg.n_layers
    probe = ActivationProbe(model)
    assert probe.points == 1 + n + 2 * n + 1, probe.points

    ids, _ = _batch(_CPU_CFG, 2)
    probe.arm(True)
    model(ids)
    snap = probe.snapshot()
    assert set(snap) == {
        "resid.embed",
        "resid.final_norm",
        *(f"resid.blocks.{i}" for i in range(n)),
        *(f"delta.blocks.{i}.attn" for i in range(n)),
        *(f"delta.blocks.{i}.ffn" for i in range(n)),
    }
    manual = float(model.token_emb(ids).detach().abs().amax())
    assert snap["resid.embed"] == pytest.approx(manual, rel=1e-6)
    probe.remove()


def test_activation_probe_is_silent_when_not_armed() -> None:
    """The hooks stay installed and cost nothing: an ``.item()`` per point per step is a
    device-to-host sync, which is why the fields are /100 steps in the first place."""
    model = _build_cpu_model()
    probe = ActivationProbe(model)
    ids, _ = _batch(_CPU_CFG, 2)
    probe.arm(False)
    model(ids)
    assert probe.snapshot() == {}
    probe.remove()


def test_weight_norms_cover_every_parameter() -> None:
    model = _build_cpu_model()
    norms = weight_norms(model)
    assert set(norms) == {n for n, _ in model.named_parameters()}
    assert all(v > 0 for v in norms.values())


# =============================================================================================
# tier 1 — mixed-precision policy
# =============================================================================================


def test_bf16_policy_reduces_in_fp32() -> None:
    pol = bf16_mp_policy()
    assert pol.param_dtype is torch.bfloat16
    assert pol.reduce_dtype is torch.float32, "a bf16 reduce-scatter biases every gradient"
    assert pol.cast_forward_inputs is False, "the dataloader hands the model int64 token ids"


def test_a_lower_precision_reduction_than_the_parameters_is_refused() -> None:
    with pytest.raises(ValueError, match="lower precision"):
        bf16_mp_policy("float32", "bfloat16")


def test_autocast_dtype_is_none_in_fp32() -> None:
    assert autocast_dtype("float32") is None
    assert autocast_dtype("bfloat16") is torch.bfloat16


# =============================================================================================
# tier 1 — plan validation
# =============================================================================================


def test_a_plan_that_covers_every_parameter_validates() -> None:
    model = _build_cpu_model()
    validate_plan(_two_d_plan(model.cfg.n_layers, 2, 2), model, _CPU_CFG)


def test_a_mesh_that_does_not_factor_the_world_is_refused() -> None:
    model = _build_cpu_model()
    plan = replace(_two_d_plan(model.cfg.n_layers, 2, 2), mesh_shape=(3, 2))
    with pytest.raises(ValueError, match="ranks but the rung runs on"):
        validate_plan(plan, model, _CPU_CFG)


def test_a_plan_that_leaves_a_parameter_out_of_every_unit_is_refused() -> None:
    """An uncovered parameter is silently ZeRO-0: full copy on every rank, no reduce-scatter."""
    model = _build_cpu_model()
    plan = replace(
        _two_d_plan(model.cfg.n_layers, 2, 2),
        fsdp_units=(FsdpUnit(("blocks.0",)),),
    )
    with pytest.raises(ValueError, match="belong to no FSDP unit"):
        validate_plan(plan, model, _CPU_CFG)


def test_a_module_in_two_fsdp_units_is_refused() -> None:
    with pytest.raises(ValueError, match="appears in two FSDP units"):
        ParallelPlan(
            mesh_shape=(2,),
            mesh_dim_names=("dp",),
            fsdp_mesh_dim="dp",
            fsdp_units=(FsdpUnit(("blocks.0",)), FsdpUnit(("blocks.0", "blocks.1"))),
        )


def test_a_plan_whose_tp_degree_is_not_the_rungs_is_refused() -> None:
    """The plan chooses how TP composes, not whether the rung has TP."""
    model = _build_cpu_model()
    cfg = replace(_CPU_CFG, world_size=2, tp_degree=2)  # mesh factors, TP degree does not match
    plan = _fsdp_only_plan(model.cfg.n_layers, 2)
    with pytest.raises(ValueError, match="plan TP degree"):
        validate_plan(plan, model, cfg)


def test_fqn_globs_do_not_cross_a_dot() -> None:
    """``blocks.*`` is each block, not every leaf inside one — the difference between wrapping
    two modules in an AC boundary and wrapping thirty."""
    from scratch_llm.training.parallel_plan import fqn_match

    assert fqn_match("blocks.0", "blocks.*")
    assert not fqn_match("blocks.0.attn.q_norm", "blocks.*")
    assert fqn_match("blocks.0.attn.q_proj", "blocks.*.attn.q_proj")
    assert fqn_match("blocks.0.attn.q_norm", "blocks.**")
    assert fqn_match("", "")
    assert not fqn_match("blocks.0", "")


def test_vocab_parallel_embedding_is_refused_by_name() -> None:
    """``Embedding.forward`` is ``self.weight[token_ids]``; a vocab-sharded table would return
    the wrong rows on every rank but the first, with no error."""
    model = _build_cpu_model()
    plan = replace(
        _two_d_plan(model.cfg.n_layers, 2, 2),
        tp_styles=(("token_emb", "colwise"), ("lm_head", "colwise")),
    )
    with pytest.raises(ValueError, match="vocab-parallel embedding"):
        validate_plan(plan, model, _CPU_CFG)


def test_an_unknown_tp_style_or_sac_op_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown TP style"):
        ParallelPlan(
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp"),
            fsdp_mesh_dim="dp",
            tp_mesh_dim="tp",
            fsdp_units=(FsdpUnit(("",)),),
            tp_styles=(("blocks.*", "sequence"),),
        )
    with pytest.raises(ValueError, match="unknown SAC op"):
        ParallelPlan(
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp"),
            fsdp_mesh_dim="dp",
            tp_mesh_dim="tp",
            fsdp_units=(FsdpUnit(("",)),),
            sac_save_ops=("aten.frobnicate.default",),
        )


def test_the_model_really_has_a_tied_pair_the_plan_has_to_reckon_with() -> None:
    model = _build_cpu_model()
    ties = tied_parameter_groups(model)
    assert [sorted(g) for g in ties] == [[("lm_head", "weight"), ("token_emb", "weight")]]


# =============================================================================================
# tier 1 — selective activation checkpointing
# =============================================================================================


def test_selective_ac_gradients_match_eager() -> None:
    """Recompute must be transparent to autograd; anything else is a silent numerical change."""
    eager = _build_cpu_model()
    acked = copy.deepcopy(eager)
    plan = _fsdp_only_plan(eager.cfg.n_layers, 2)
    applied = apply_selective_ac(acked, plan)
    assert set(applied) == {f"blocks.{i}" for i in range(eager.cfg.n_layers)}

    ids, tgt = _batch(_CPU_CFG, 2)
    for m in (eager, acked):
        cross_entropy(m(ids), tgt).backward()
    ref = dict(eager.named_parameters())
    for name, p in acked.named_parameters():
        assert p.grad is not None
        torch.testing.assert_close(p.grad, ref[clean_fqn(name)].grad, rtol=1e-5, atol=1e-6)


def test_wrapping_does_not_change_the_names_the_log_uses() -> None:
    """``checkpoint_wrapper`` inserts ``_checkpoint_wrapped_module`` into every FQN beneath it.

    Left in, the weight-norm keys and the activation-probe point names change the day AC is
    toggled, and two runs of the same model produce two incomparable sets of log keys — which is
    the whole reason to log per-parameter norms in the first place.
    """
    eager = _build_cpu_model()
    acked = copy.deepcopy(eager)
    apply_selective_ac(acked, _fsdp_only_plan(eager.cfg.n_layers, 2))
    raw = {n for n, _ in acked.named_parameters()}
    assert any("_checkpoint_wrapped_module" in n for n in raw), "the wrapper did not wrap"
    assert set(weight_norms(acked)) == set(weight_norms(eager))
    assert {param_group(n) for n, _ in acked.named_parameters()} == {
        param_group(n) for n, _ in eager.named_parameters()
    }
    probe_eager, probe_acked = ActivationProbe(eager), ActivationProbe(acked)
    ids, _ = _batch(_CPU_CFG, 2)
    for m, probe in ((eager, probe_eager), (acked, probe_acked)):
        probe.arm(True)
        m(ids)
    assert set(probe_acked.snapshot()) == set(probe_eager.snapshot())
    probe_eager.remove()
    probe_acked.remove()


def test_selective_ac_saves_fewer_bytes_than_eager() -> None:
    """The whole point: recompute buys memory. Measured with saved-tensor hooks, device-agnostic."""
    from scratch_llm.utils.checkpointing import count_saved_activation_bytes

    eager = _build_cpu_model()
    acked = copy.deepcopy(eager)
    apply_selective_ac(acked, _fsdp_only_plan(eager.cfg.n_layers, 2))
    ids, tgt = _batch(_CPU_CFG, 2)
    with count_saved_activation_bytes() as eager_bytes:
        cross_entropy(eager(ids), tgt).backward()
    with count_saved_activation_bytes() as ac_bytes:
        cross_entropy(acked(ids), tgt).backward()
    assert ac_bytes["bytes"] < eager_bytes["bytes"], (ac_bytes, eager_bytes)


def test_every_named_sac_op_resolves_in_this_torch_build() -> None:
    """An op that silently does not exist is a throughput change with no diff."""
    from scratch_llm.training.parallel_plan import SAC_SAVEABLE_OPS

    resolved, missing = resolve_save_ops(SAC_SAVEABLE_OPS)
    assert not missing, f"unresolvable in torch {torch.__version__}: {missing}"
    assert len(resolved) == len(SAC_SAVEABLE_OPS)


# =============================================================================================
# tier 2 — gloo, world = 2: FSDP2 equivalence, TP equivalence, async DCP
# =============================================================================================


def _world2_worker(rank: int, world: int, port: int, tmpdir: str) -> None:
    from torch.distributed.tensor import DTensor

    _init_pg(rank, world, port)
    try:
        cfg = replace(_CPU_CFG, world_size=world, tp_degree=1)
        n_layers = cfg.shape.n_layers
        mb = cfg.micro_batch

        # --- reference: one process, the full batch ---
        ref = _build_cpu_model(cfg)
        ids, tgt = _batch(cfg, mb * world)
        ref_loss = cross_entropy(ref(ids), tgt)
        ref_loss.backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters() if p.grad is not None}

        # --- FSDP2 at dp=world, no TP, no compile ---
        model = _build_cpu_model(cfg)
        plan = _fsdp_only_plan(n_layers, world)
        model, mesh, report = parallelize(model, plan, cfg, device_type="cpu", compile_model=False)
        assert mesh.ndim == 1 and mesh.size() == world
        assert len(report.fsdp_units) == len(plan.fsdp_units)
        my_ids = ids[rank * mb : (rank + 1) * mb]
        my_tgt = tgt[rank * mb : (rank + 1) * mb]
        loss = cross_entropy(model(my_ids), my_tgt)
        loss.backward()

        assert any(isinstance(p, DTensor) for p in model.parameters()), "nothing got sharded"
        got = _full_grads(model)
        assert set(got) == set(ref_grads), (sorted(got), sorted(ref_grads))
        for name, g in got.items():
            torch.testing.assert_close(
                g, ref_grads[name], rtol=1e-4, atol=1e-6, msg=f"FSDP2 grad mismatch at {name}"
            )

        # per-group norms still work when every gradient is a DTensor — and, crucially, agree
        # with the SINGLE-PROCESS norm. The identity total² == Σ group² holds even if both sides
        # are wrong the same way; only the reference catches a DTensor reduction that
        # double-counts a shard or drops one.
        total, groups = grad_norms(model)
        assert len(groups) >= 3 and total > 0
        assert total == pytest.approx(sum(v * v for v in groups.values()) ** 0.5, rel=1e-5)
        ref_total = float(sum(g.double().pow(2).sum() for g in ref_grads.values()) ** 0.5)
        assert total == pytest.approx(ref_total, rel=1e-4), (total, ref_total)

        # --- async DCP round trip at this topology ---
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        opt.step()
        state = TrainState()
        ck = AsyncCheckpointer(model, opt, state, Path(tmpdir) / "dcp", every=1)
        cid = ck.save(step=17)
        ck.wait()
        assert ck.last_stage_seconds is not None and ck.last_write_seconds is not None
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.full_like(p, 0.5))
        state.step = 0
        restored = ck.load(cid)
        assert restored == 17, restored
        for n, p in model.named_parameters():
            torch.testing.assert_close(p.detach(), before[n], rtol=0, atol=0)

        # --- TP=2 equivalence on a fresh, unsharded pair ---
        tp_cfg = replace(_CPU_CFG, world_size=world, tp_degree=world)
        unsharded = _build_cpu_model(tp_cfg)
        tp_model = copy.deepcopy(unsharded)
        tp_plan = _two_d_plan(n_layers, 1, world)
        tp_mesh = build_mesh(tp_plan, "cpu")
        applied = apply_tensor_parallel(tp_model, tp_plan, tp_mesh["tp"])
        assert applied, "no TP style matched"
        assert _attn(tp_model, 0).n_heads == _attn(unsharded, 0).n_heads // world
        assert tp_model.lm_head.weight is tp_model.token_emb.weight, "TP un-tied the head"

        tp_ids, tp_tgt = _batch(tp_cfg, 2)
        ref_out = unsharded(tp_ids)
        tp_out = tp_model(tp_ids)
        torch.testing.assert_close(tp_out, ref_out, rtol=1e-4, atol=1e-5)
        cross_entropy(ref_out, tp_tgt).backward()
        cross_entropy(tp_out, tp_tgt).backward()
        ref_p = dict(unsharded.named_parameters())
        for name, p in tp_model.named_parameters():
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            torch.testing.assert_close(
                g, ref_p[name].grad, rtol=1e-4, atol=1e-5, msg=f"TP grad mismatch at {name}"
            )
    finally:
        dist.destroy_process_group()


def test_fsdp2_tp_and_async_dcp_on_two_gloo_ranks(tmp_path: Path) -> None:
    """The real distributed assertions, at the smallest world where they mean anything.

    dp=2 FSDP2 must reproduce single-process full-batch gradients (mean of means == global mean
    at equal token counts); TP=2 must reproduce the unsharded forward and backward; async DCP
    must round-trip; and the per-group norms must survive every gradient being a DTensor.
    """
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _world2_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True
    )


# =============================================================================================
# tier 2 — gloo, world = 4: the composition, dp=2 x tp=2
# =============================================================================================


def _world4_worker(rank: int, world: int, port: int, tmpdir: str) -> None:
    from torch.distributed.tensor import DTensor

    _init_pg(rank, world, port)
    try:
        cfg = replace(_CPU_CFG, world_size=4, tp_degree=2)
        model = _build_cpu_model(cfg)
        plan = _two_d_plan(cfg.shape.n_layers, cfg.dp_degree, cfg.tp_degree)
        model, mesh, report = parallelize(model, plan, cfg, device_type="cpu", compile_model=False)
        assert mesh.ndim == 2 and mesh.mesh_dim_names == ("dp", "tp")
        assert report.tp_applied and report.ac_applied and report.fsdp_units

        # The composition's signature: parameters TP touched are 2-D DTensors on (dp, tp).
        ndims = {p.device_mesh.ndim for p in model.parameters() if isinstance(p, DTensor)}
        assert ndims == {2}, f"expected every parameter on the 2-D mesh, got ndims={ndims}"

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        probe = ActivationProbe(model)
        logger = RunLogger(config=cfg.as_log_dict(), total_steps=3, periodic_every=2)
        ids, tgt = _batch(cfg, cfg.micro_batch)
        for step in range(3):
            periodic = logger.is_periodic(step)
            probe.arm(periodic)
            opt.zero_grad(set_to_none=True)
            loss = cross_entropy(model(ids), tgt)
            loss.backward()
            total, groups = grad_norms(model)
            opt.step()
            logger.log_step(
                step=step,
                loss=float(loss),
                lr=1e-3,
                grad_norm_global=total,
                grad_norm_groups=groups,
                tokens_per_s_per_device=1.0,
                mfu=0.0,
                nccl_wait_s=0.0,
                step_time_s=1.0,
                peak_memory_bytes=0,
                activation_max_abs=probe.snapshot() if periodic else None,
                weight_norms_=weight_norms(model) if periodic else None,
            )
        assert {r["step"] for r in logger.records if "activation_max_abs" in r} == {0, 2}
        assert len(logger.records[0]["grad_norm_groups"]) >= 3
        probe.remove()
    finally:
        dist.destroy_process_group()


def test_two_d_parallelism_dp2_x_tp2_steps_and_logs(tmp_path: Path) -> None:
    """FSDP2 x TP composed, on four gloo ranks — the shape the rung actually runs.

    What this pins that the 2-rank tests cannot: that ``fully_shard`` applied *after*
    ``distribute_module`` yields 2-D ``(dp, tp)`` DTensor parameters rather than two
    incompatible 1-D meshes, and that the whole logging spec still produces a valid record when
    every gradient is one of those.
    """
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _world4_worker, args=(4, _free_port(), str(tmp_path)), nprocs=4, join=True
    )


# =============================================================================================
# tier 2 — gloo, world = 4: the whole step loop, end to end
# =============================================================================================


def _end_to_end_worker(rank: int, world: int, port: int, tmpdir: str) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    _init_pg(rank, world, port)
    try:
        cfg = replace(_CPU_CFG, world_size=world, tp_degree=2, total_steps=70, warmup_steps=20)
        out = Path(tmpdir) / "run"
        result = train_t_r1(
            cfg,
            device_type="cpu",
            out_dir=out,
            floor_tps=1.0,  # a stand-in denominator: this test asserts plumbing, not throughput
            checkpoint_every=25,
            compile_model=False,
            plan=_two_d_plan(cfg.shape.n_layers, cfg.dp_degree, cfg.tp_degree),
        )
        if rank != 0:
            return
        assert result["metric"] == "pct_of_torchtitan_tps"
        assert result["floor_metric"] == cfg.floor_metric
        assert result["measured_steps"] == cfg.measure_steps == 50
        assert result["tps_per_device"]["median"] > 0
        assert result["pct_of_torchtitan_tps"] > 0
        assert "mfu_model_torchtitan" in result["accounting"]
        assert result["checkpoint_stage_seconds"] is not None, "no checkpoint was taken"
        assert result["checkpoint_write_seconds"] is not None
        assert result["activation_probe_points"] == 1 + 3 * cfg.shape.n_layers + 1

        lines = (out / "steps.rank0.jsonl").read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        assert header["kind"] == "run_header"
        assert header["nccl_wait_scope"] == NCCL_WAIT_SCOPE
        records = [json.loads(line) for line in lines[1:]]
        assert len(records) == cfg.total_steps == 70
        assert [r["step"] for r in records] == list(range(70))
        assert {r["step"] for r in records if "activation_max_abs" in r} == {0, 69}
        assert {r["step"] for r in records if "weight_norms" in r} == {0, 69}
        assert set(records[0]) >= REQUIRED_STEP_FIELDS | PERIODIC_FIELDS
        assert set(records[1]) == REQUIRED_STEP_FIELDS
        assert len(records[0]["grad_norm_groups"]) >= 3
        assert records[0]["git_sha"] == git_head_sha()
        assert records[0]["config"]["plan"]["tp_styles"]
        assert all(r["lr"] > 0 for r in records[1:])
        assert records[0]["lr"] < records[cfg.warmup_steps]["lr"], "warmup did not ramp"
    finally:
        dist.destroy_process_group()


def test_the_whole_step_loop_runs_end_to_end_on_four_gloo_ranks(tmp_path: Path) -> None:
    """Every line of ``train_t_r1`` executes here, so none of it first executes on a rented box.

    Not a throughput test — CPU tok/s means nothing. It is a test that the loop *closes*: the
    optimizer steps on DTensor parameters, the LR schedule ramps, the async checkpoint is taken
    and awaited inside the loop, the activation and weight probes fire on exactly the right
    steps, the record validates against the spec on every one of 70 steps, and the result dict
    the harness reads has every key ``run.sh`` and the ledger need.

    The plan is passed explicitly. ``train_t_r1(plan=None)`` — the only thing the CLI can
    produce — calls the hole, so no measurement can come from an unmade decision.
    """
    mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
        _end_to_end_worker, args=(4, _free_port(), str(tmp_path)), nprocs=4, join=True
    )


# =============================================================================================
# tier 3 — 8xH100. Not decidable here; SKIPPED with the command, never silently passed.
# =============================================================================================

_HAS_8_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 8


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_8_GPUS, reason=f"needs 8 GPUs. Run it on the box:\n{BOX_COMMAND}")
def test_throughput_against_the_torchtitan_floor() -> None:
    """The rung's number. Requires the floor, the plan, and eight H100s — none of them here."""
    raise AssertionError(
        "this test body runs only on the box; the measurement path is "
        "experiments/T1/T-R1/run.sh, whose last stdout line is pct_of_torchtitan_tps"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _HAS_8_GPUS, reason=f"needs 8 GPUs. Run it on the box:\n{BOX_COMMAND}")
def test_compile_and_bf16_together_do_not_nan() -> None:
    """``scratch_llm.train`` records a known sm120/torch-2.12 inductor bug where bf16 autocast +
    torch.compile NaN together (``train.py:448-454``). H100 is the stated safe path, and this
    rung compiles under bf16 — so it is asserted there, and cannot be asserted here."""
    raise AssertionError(f"run on the box:\n{BOX_COMMAND}")


# =============================================================================================
# The hole guard — exactly one (tests/conftest.py turns it into a strict xfail while open)
# =============================================================================================


@pytest.mark.hole("T1/T-R1", "src/scratch_llm/training/parallel_plan.py")
def test_the_parallelism_decision_is_made_and_is_structurally_sound() -> None:
    """Fails while ``t1_parallel_plan`` raises; passes when the decision is written and coherent.

    Every assertion is a *constraint the arithmetic must respect*, never a value — the granularity,
    the composition and the save set are Huy's, and a test that pinned them would be the answer
    key rather than the gate. What is checked:

      mesh factors the world      8 ranks, and TP=2 on the axis the rung's config names.
      every parameter is covered  exactly one FSDP unit owns each; none left replicated.
      the tied pair is coherent   ``lm_head.weight is token_emb.weight``; two units would gather
                                  the same tensor twice, and two TP styles would un-tie it.
      TP divides what TP shards   n_heads, n_kv_heads and d_ff, or ``.view`` reshapes garbage.
      the plan is a value         serializable, so it lands in the run log verbatim.
      AC actually saves something an empty save set with AC modules named is a recompute of
                                  everything, which is a decision — but then say so in `notes`.

    ``validate_plan`` against the *real* 1B model is the expensive half and is what makes this a
    gate rather than a lint: it instantiates the model on meta to walk its FQNs.
    """
    plan = t1_parallel_plan(T_R1)
    assert isinstance(plan, ParallelPlan)
    assert plan.world_size == T_R1.world_size == 8
    assert plan.tp_degree == T_R1.tp_degree == 2
    assert plan.fsdp_degree == T_R1.dp_degree == 4

    with torch.device("meta"):
        model = TransformerLM(T_R1.model_config())
    validate_plan(plan, model, T_R1)

    # The tied pair, both halves of it.
    tied = tied_parameter_groups(model)
    assert tied, "the 1B config ties lm_head to token_emb; the plan must reckon with it"
    tied_modules = {m for group in tied for m, _ in group}
    unit_of = {f: i for i, u in enumerate(plan.fsdp_units) for f in u.modules}
    owning = {unit_of[m] for m in tied_modules if m in unit_of}
    assert len(owning) <= 1, (
        f"tied modules {sorted(tied_modules)} sit in different FSDP units {owning}: one tensor, "
        "two all-gathers per forward"
    )
    tied_styles = {plan.tp_style_for(m) for m in tied_modules}
    assert len(tied_styles) == 1, f"tied modules got different TP styles: {tied_styles}"

    # Every sharded axis is divisible, and the plan is a value that can be logged.
    for label, value in (
        ("n_heads", T_R1.shape.n_heads),
        ("n_kv_heads", T_R1.shape.n_kv_heads),
        ("ffn_hidden", T_R1.shape.ffn_hidden),
    ):
        assert value % plan.tp_degree == 0, f"{label}={value} is not divisible by TP"
    json.dumps(plan.as_log_dict())

    if plan.ac_modules:
        assert plan.sac_save_ops or plan.sac_save_every_other_mm or plan.notes, (
            "AC modules named with an empty save set recomputes everything — a legitimate "
            "decision, but then say so in notes"
        )
