"""F6 MoE balancing ablation — definition-of-done tests.

These tests are deliberately tiny (CPU, seconds) and focus on the contract of
``build_moe_config``, the iso-parametric granularity mapping, and the
key discriminating balancer dynamic.
"""

import math
from dataclasses import replace

import numpy as np
import torch

from scratch_llm.eval.moe_ablation import (
    AblationArm,
    Granularity,
    build_moe_config,
    evaluate_val_loss,
    router_diagnostics,
    train_arm,
)
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.moe import MoEConfig, MoEFeedForward
from scratch_llm.train import TrainConfig


def _tiny_model_cfg(
    arm: AblationArm = AblationArm.BIAS_FREE,
    granularity: Granularity = Granularity.COARSE,
    moe_overrides: dict | None = None,
) -> ModelConfig:
    """A CPU-instantly-trainable MoE model."""
    moe = build_moe_config(
        arm,
        granularity,
        base_cfg=MoEConfig(n_routed_experts=8, n_experts_per_tok=2, n_shared_experts=1),
        d_model=32,
    )
    if moe_overrides:
        moe = replace(moe, **moe_overrides)
    return ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        context_length=16,
        moe=moe,
    )


def _synthetic_loader(vocab_size: int, batch_size: int, context_length: int, batches: int):
    """Deterministic repeated-pattern batches, tiled to always fill context_length."""
    pattern = np.arange(vocab_size, dtype=np.int64)
    # Pre-tile once so any contiguous slice of length context_length is valid.
    tiled = np.tile(pattern, (context_length // vocab_size) + 2)
    for _ in range(batches):
        ids = np.zeros((batch_size, context_length), dtype=np.int64)
        for b in range(batch_size):
            start = np.random.randint(0, vocab_size)
            ids[b] = tiled[start : start + context_length]
        yield ids, np.roll(ids, shift=-1, axis=1)


def _moe_fields(cfg: MoEConfig) -> set[str]:
    """Dataclass field names actually present on a ``MoEConfig``."""
    return set(cfg.__dataclass_fields__)


# ---------------------------------------------------------------------------
# Configuration mapping
# ---------------------------------------------------------------------------


def test_build_moe_config_maps_arms_correctly() -> None:
    """Each arm sets the right existing fields; no new fields appear."""
    base = MoEConfig(
        n_routed_experts=8,
        n_experts_per_tok=2,
        n_shared_experts=1,
        expert_d_ff=64,
        n_dense_layers=1,
        routed_scaling_factor=2.0,
    )
    allowed = _moe_fields(base)

    bias_free = build_moe_config(AblationArm.BIAS_FREE, Granularity.COARSE, base)
    seq_aux = build_moe_config(AblationArm.SEQ_AUX, Granularity.COARSE, base)
    none = build_moe_config(AblationArm.NONE, Granularity.COARSE, base)

    for cfg in (bias_free, seq_aux, none):
        assert _moe_fields(cfg) == allowed, "build_moe_config introduced a new field"
        assert cfg.z_loss_coef == 0.0
        assert cfg.n_shared_experts == 1
        assert cfg.n_dense_layers == 1
        assert cfg.routed_scaling_factor == 2.0

    assert bias_free.bias_update_speed > 0.0 and bias_free.aux_loss_alpha == 0.0
    assert seq_aux.aux_loss_alpha > 0.0 and seq_aux.bias_update_speed == 0.0
    assert none.aux_loss_alpha == 0.0 and none.bias_update_speed == 0.0


def test_build_moe_config_default_base() -> None:
    """``build_moe_config`` works with ``base_cfg=None`` and infers defaults."""
    cfg = build_moe_config(AblationArm.BIAS_FREE, Granularity.COARSE, d_model=64)
    assert cfg.n_routed_experts == 16
    assert cfg.n_experts_per_tok == 2
    assert cfg.expert_d_ff == 192  # _default_expert_d_ff(64)
    assert cfg.z_loss_coef == 0.0


def test_coarse_fine_iso_param_and_flop() -> None:
    """COARSE and FINE keep total routed params and active FLOPs equal."""
    d_model = 64
    coarse = build_moe_config(AblationArm.BIAS_FREE, Granularity.COARSE, d_model=d_model)
    fine = build_moe_config(AblationArm.BIAS_FREE, Granularity.FINE, d_model=d_model)

    # Routed expert SwiGLU params scale as d_model * d_ff per matrix, three matrices.
    # Total routed params are proportional to N_r * d_ff.
    assert coarse.expert_d_ff is not None and fine.expert_d_ff is not None
    assert coarse.n_routed_experts * coarse.expert_d_ff == fine.n_routed_experts * fine.expert_d_ff

    # Active routed FLOPs per token are proportional to K_r * d_ff.
    assert (
        coarse.n_experts_per_tok * coarse.expert_d_ff == fine.n_experts_per_tok * fine.expert_d_ff
    )

    # Sanity: fine uses more experts, smaller width.
    assert fine.n_routed_experts > coarse.n_routed_experts
    assert fine.expert_d_ff < coarse.expert_d_ff


# ---------------------------------------------------------------------------
# Diagnostics + validation
# ---------------------------------------------------------------------------


def test_router_diagnostics_on_model_and_layer() -> None:
    """``router_diagnostics`` accepts both ``TransformerLM`` and ``MoEFeedForward``."""
    cfg = _tiny_model_cfg()
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.context_length))

    assert cfg.moe is not None
    diag_model = router_diagnostics(model, ids)
    assert diag_model["n_routed_experts"] == [cfg.moe.n_routed_experts] * cfg.n_layers
    assert 0.0 < diag_model["entropy"] <= math.log(cfg.moe.n_routed_experts)
    assert 0.0 <= diag_model["balance_score"] <= 1.0

    moe_layer = model.blocks[0].ffn
    assert isinstance(moe_layer, MoEFeedForward)
    diag_layer = router_diagnostics(moe_layer, model.token_emb(ids))
    assert isinstance(diag_layer["entropy"], float)


def test_evaluate_val_loss_matches_shape() -> None:
    """Validation CE is finite and perplexity = exp(CE)."""
    cfg = _tiny_model_cfg()
    model = TransformerLM(cfg)
    loader = _synthetic_loader(
        cfg.vocab_size, batch_size=2, context_length=cfg.context_length, batches=3
    )
    result = evaluate_val_loss(model, loader)
    assert math.isfinite(result["ce"])
    assert result["perplexity"] == math.exp(result["ce"])
    assert result["n_tokens"] == 2 * cfg.context_length * 3


# ---------------------------------------------------------------------------
# Discriminating balancer dynamic
# ---------------------------------------------------------------------------


def _biased_init(model: TransformerLM) -> None:
    """Set every router to strongly prefer expert 0 at init and freeze gate weights.

    Freezing the gate keeps the affinity distribution fixed, so the only force that
    can rebalance selection is the aux-loss-free bias update (or the aux loss).
    """
    for block in model.blocks:
        ffn = block.ffn
        if isinstance(ffn, MoEFeedForward):
            ffn.router.gate.weight.requires_grad = False
            with torch.no_grad():
                ffn.router.bias.zero_()
                ffn.router.bias[0] = 5.0
                ffn.router.bias[1:] = -5.0 / (ffn.router.bias.numel() - 1)


def test_balancer_overcomes_induced_preference() -> None:
    """A fast bias-free balancer recovers entropy >0.9·log(N_r); the NONE control stays collapsed."""
    n_routed = 8
    k = 1  # top-1 makes the "stays collapsed" control unambiguous (entropy ≈ 0).

    def make_model(arm: AblationArm) -> tuple[TransformerLM, ModelConfig]:
        # Architecture overrides only; balancer fields come from build_moe_config.
        overrides: dict[str, object] = {
            "n_routed_experts": n_routed,
            "n_experts_per_tok": k,
            "expert_d_ff": 48,
        }
        if arm is AblationArm.BIAS_FREE:
            overrides["bias_update_speed"] = 0.12
        elif arm is AblationArm.SEQ_AUX:
            overrides["aux_loss_alpha"] = 1e-2
        cfg = _tiny_model_cfg(
            arm=arm,
            granularity=Granularity.COARSE,
            moe_overrides=overrides,
        )
        model = TransformerLM(cfg)
        _biased_init(model)
        return model, cfg

    torch.manual_seed(0)
    np.random.seed(0)
    bias_free, cfg = make_model(AblationArm.BIAS_FREE)
    train_cfg = TrainConfig(
        max_steps=500,
        batch_size=8,
        context_length=cfg.context_length,
        max_lr=3e-3,
        grad_clip=1.0,
        seed=0,
        device="cpu",
    )
    train_loader = _synthetic_loader(
        cfg.vocab_size, batch_size=8, context_length=cfg.context_length, batches=800
    )
    val_loader = _synthetic_loader(
        cfg.vocab_size, batch_size=32, context_length=cfg.context_length, batches=4
    )

    # BIAS_FREE should recover routing diversity.
    result_bias_free = train_arm(bias_free, train_cfg, train_loader, val_loader)
    entropy_bf = result_bias_free["diagnostics"]["entropy"]
    threshold = 0.9 * math.log(n_routed)
    assert entropy_bf > threshold, (
        f"BIAS_FREE entropy {entropy_bf:.4f} did not recover above {threshold:.4f}"
    )

    # NONE should remain collapsed (expert 0 always selected).
    torch.manual_seed(0)
    np.random.seed(0)
    none, _ = make_model(AblationArm.NONE)
    train_loader = _synthetic_loader(
        cfg.vocab_size, batch_size=8, context_length=cfg.context_length, batches=800
    )
    val_loader = _synthetic_loader(
        cfg.vocab_size, batch_size=32, context_length=cfg.context_length, batches=4
    )
    result_none = train_arm(none, train_cfg, train_loader, val_loader)
    entropy_none = result_none["diagnostics"]["entropy"]
    assert entropy_none < 0.1 * threshold, f"NONE entropy {entropy_none:.4f} unexpectedly recovered"


def test_seq_aux_arm_has_large_alpha() -> None:
    """The pinned design uses a large α, not DeepSeek-V3's 1e-4."""
    cfg = build_moe_config(AblationArm.SEQ_AUX, Granularity.COARSE, d_model=64)
    assert cfg.aux_loss_alpha in {1e-3, 1e-2}
    assert cfg.bias_update_speed == 0.0
    assert cfg.z_loss_coef == 0.0
