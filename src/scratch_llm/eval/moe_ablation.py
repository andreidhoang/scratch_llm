"""F6 MoE balancing ablation — measurement-only harness over the existing MoE stack.

This module constructs iso-parametric, iso-active-FLOP MoE configurations that differ only
through the existing ``MoEConfig`` fields ``bias_update_speed``, ``aux_loss_alpha``,
``expert_d_ff``, ``n_routed_experts``, and ``n_experts_per_tok``.  It runs a tiny training
loop, reports validation CE, and exports routing diagnostics so the three balancing arms can
be compared:

* ``BIAS_FREE`` — aux-loss-free load balancing via router-bias updates (DeepSeek-V3 style).
* ``SEQ_AUX``  — sequence-level auxiliary loss with a deliberately large α.
* ``NONE``     — no balancing at all (control).

The granularity axis (``COARSE`` / ``FINE``) holds total routed parameters and active routed
FLOPs constant while changing the expert count and the number of experts activated per token.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy
from scratch_llm.moe import MoEConfig, MoEFeedForward
from scratch_llm.optim import AdamW, gradient_clipping
from scratch_llm.train import TrainConfig
from scratch_llm.utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------


class AblationArm(Enum):
    """Which load-balancing mechanism is active."""

    BIAS_FREE = "bias_free"
    SEQ_AUX = "seq_aux"
    NONE = "none"


class Granularity(Enum):
    """Expert granularity: coarse (fewer, wider experts per token) vs fine
    (more, narrower experts per token).  The two settings are designed to be
    iso-parametric and iso-active-FLOP."""

    COARSE = "coarse"
    FINE = "fine"


# ---------------------------------------------------------------------------
# Configuration builder
# ---------------------------------------------------------------------------


def _default_expert_d_ff(d_model: int) -> int:
    """Mirror of ``moe._default_expert_ffn`` so this module can size experts
    without exposing that private helper in the public API."""
    raw = int(8 / 3 * d_model)
    return ((raw + 63) // 64) * 64


def _base_expert_d_ff(base_cfg: MoEConfig | None, d_model: int | None) -> int:
    """Resolve the coarse-grained expert width to use as the iso-param anchor."""
    if base_cfg is not None and base_cfg.expert_d_ff is not None:
        return base_cfg.expert_d_ff
    if d_model is not None:
        return _default_expert_d_ff(d_model)
    raise ValueError(
        "build_moe_config needs either base_cfg.expert_d_ff or d_model "
        "to resolve a default expert width."
    )


def build_moe_config(
    arm: AblationArm,
    granularity: Granularity,
    base_cfg: MoEConfig | None = None,
    *,
    d_model: int | None = None,
    seq_aux_alpha: float = 1e-3,
    bias_update_speed: float = 1e-3,
) -> MoEConfig:
    """Return a ``MoEConfig`` for one ablation arm.

    Args:
        arm: Which balancing mechanism to enable.
        granularity: ``COARSE`` or ``FINE``.  The mapping is designed to keep
            total routed parameters and active routed FLOPs equal:

            * ``COARSE``: ``n_routed_experts=16``, ``n_experts_per_tok=2``,
              ``expert_d_ff=base_d_ff``.
            * ``FINE``: ``n_routed_experts=64``, ``n_experts_per_tok=8``,
              ``expert_d_ff=base_d_ff // 4``.

            Total routed params are proportional to ``N_r * d_ff``;
            active routed FLOPs are proportional to ``K_r * d_ff``.
            Both are ``16*base_d_ff`` and ``2*base_d_ff`` respectively
            (iso-parametric and iso-active-FLOP).
        base_cfg: Optional starting ``MoEConfig``; its ``n_shared_experts``,
            ``n_dense_layers``, and ``routed_scaling_factor`` are preserved.
        d_model: Required only when ``base_cfg.expert_d_ff`` is ``None``,
            so the default SwiGLU width can be computed.
        seq_aux_alpha: α for ``SEQ_AUX`` (pinned large to stress-test the
            loss/aux trade-off; default ``1e-3``).
        bias_update_speed: γ for ``BIAS_FREE`` (default ``1e-3``).

    Returns:
        A fresh ``MoEConfig`` with ``z_loss_coef=0`` and the arm-specific
        balancer fields set.  No new dataclass fields are introduced.
    """
    if not isinstance(arm, AblationArm):
        raise TypeError(f"arm must be AblationArm, got {type(arm)}")
    if not isinstance(granularity, Granularity):
        raise TypeError(f"granularity must be Granularity, got {type(granularity)}")

    base_d_ff = _base_expert_d_ff(base_cfg, d_model)

    if granularity is Granularity.COARSE:
        n_routed_experts = 16
        n_experts_per_tok = 2
        expert_d_ff = base_d_ff
    else:  # FINE
        n_routed_experts = 64
        n_experts_per_tok = 8
        expert_d_ff = base_d_ff // 4
        if expert_d_ff < 16:
            raise ValueError(
                f"computed fine expert_d_ff={expert_d_ff} too small; "
                "increase d_model or set base_cfg.expert_d_ff."
            )

    # Arm semantics: only the existing balancer fields change.
    if arm is AblationArm.BIAS_FREE:
        aux_alpha = 0.0
        bias_speed = bias_update_speed
    elif arm is AblationArm.SEQ_AUX:
        aux_alpha = seq_aux_alpha
        bias_speed = 0.0
    else:  # NONE
        aux_alpha = 0.0
        bias_speed = 0.0

    # Start from base_cfg if provided, otherwise default, and override the ablation knobs.
    if base_cfg is not None:
        cfg = replace(
            base_cfg,
            n_routed_experts=n_routed_experts,
            n_experts_per_tok=n_experts_per_tok,
            expert_d_ff=expert_d_ff,
            aux_loss_alpha=aux_alpha,
            bias_update_speed=bias_speed,
            z_loss_coef=0.0,
        )
    else:
        cfg = MoEConfig(
            n_routed_experts=n_routed_experts,
            n_experts_per_tok=n_experts_per_tok,
            n_shared_experts=1,
            expert_d_ff=expert_d_ff,
            n_dense_layers=0,
            aux_loss_alpha=aux_alpha,
            z_loss_coef=0.0,
            bias_update_speed=bias_speed,
            routed_scaling_factor=1.0,
        )
    return cfg


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def router_diagnostics(
    model_or_moe_layer: TransformerLM | MoEFeedForward,
    ids_or_input: Tensor,
) -> dict[str, Any]:
    """Run a forward pass and return routing diagnostics.

    Args:
        model_or_moe_layer: Either a full ``TransformerLM`` or a single
            ``MoEFeedForward`` layer.
        ids_or_input: Token ids ``(B, S)`` for a ``TransformerLM`` or hidden
            states ``(B, S, d)`` for a ``MoEFeedForward``.

    Returns:
        Dictionary with:

        * ``expert_fractions``: per-layer list of ``(n_routed,)`` load fractions.
        * ``entropy``: mean router entropy (nats) across layers.
        * ``max_load``: maximum load fraction across experts (per layer).
        * ``min_load``: minimum non-zero load fraction across experts (per layer).
        * ``balance_score``: entropy normalized by ``log(n_routed)`` (1 = perfectly uniform).
        * ``n_routed_experts``: list of expert counts per layer.
    """
    was_training = model_or_moe_layer.training
    model_or_moe_layer.eval()
    with torch.no_grad():
        if isinstance(model_or_moe_layer, TransformerLM):
            _, aux = model_or_moe_layer(ids_or_input, return_aux=True)
            layers = aux.layers
        elif isinstance(model_or_moe_layer, MoEFeedForward):
            _, stats = model_or_moe_layer(ids_or_input)
            layers = [stats]
        else:
            raise TypeError(
                "model_or_moe_layer must be TransformerLM or MoEFeedForward, "
                f"got {type(model_or_moe_layer)}"
            )

    if not layers:
        if was_training:
            model_or_moe_layer.train()
        return {
            "expert_fractions": [],
            "entropy": float("nan"),
            "max_load": float("nan"),
            "min_load": float("nan"),
            "balance_score": float("nan"),
            "n_routed_experts": [],
        }

    fractions: list[Tensor] = []
    entropies: list[float] = []
    max_loads: list[float] = []
    min_loads: list[float] = []
    scores: list[float] = []
    n_routed_experts: list[int] = []

    for stat in layers:
        load_frac = stat.load_fraction.detach().cpu().float()
        entropy = float(stat.entropy.detach().cpu().item())
        fractions.append(load_frac)
        entropies.append(entropy)
        max_loads.append(float(load_frac.max().item()))
        # Min over experts that actually received tokens.
        nz = load_frac[load_frac > 0]
        min_loads.append(float(nz.min().item()) if nz.numel() else 0.0)
        n_r = load_frac.numel()
        n_routed_experts.append(n_r)
        scores.append(entropy / math.log(n_r) if n_r > 1 else 1.0)

    if was_training:
        model_or_moe_layer.train()

    return {
        "expert_fractions": fractions,
        "entropy": float(np.mean(entropies)),
        "max_load": float(np.mean(max_loads)),
        "min_load": float(np.mean(min_loads)),
        "balance_score": float(np.mean(scores)),
        "n_routed_experts": n_routed_experts,
    }


# ---------------------------------------------------------------------------
# Validation loss + tiny training loop
# ---------------------------------------------------------------------------


def _to_long_tensor(x: np.ndarray | Tensor, device: str | torch.device) -> Tensor:
    """Normalize a batch to a long tensor on the requested device."""
    if isinstance(x, Tensor):
        return x.long().to(device)
    return torch.from_numpy(np.asarray(x)).long().to(device)


def evaluate_val_loss(
    model: TransformerLM,
    val_loader: Iterable[tuple[np.ndarray | Tensor, np.ndarray | Tensor]],
) -> dict[str, float]:
    """Mean validation cross-entropy plus token count.

    ``val_loader`` yields ``(inputs, targets)`` pairs.  The model is put in eval
    mode temporarily and restored afterwards.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs_t = _to_long_tensor(inputs, device)
            targets_t = _to_long_tensor(targets, device)
            if model.cfg.moe is not None:
                logits, aux = model(inputs_t, return_aux=True)
                # Validation CE is reported without the MoE regularizer so the
                # comparison across arms is fair.
                ce = cross_entropy(logits, targets_t)
            else:
                logits = model(inputs_t)
                ce = cross_entropy(logits, targets_t)
            ntok = targets_t.numel()
            total_loss += float(ce.item()) * ntok
            total_tokens += ntok
    if was_training:
        model.train()

    mean_ce = total_loss / max(total_tokens, 1)
    return {
        "ce": mean_ce,
        "perplexity": math.exp(mean_ce),
        "n_tokens": total_tokens,
    }


def train_arm(
    model_cfg_or_model: ModelConfig | TransformerLM,
    train_cfg: TrainConfig | Mapping[str, Any],
    train_loader: Iterator[tuple[np.ndarray | Tensor, np.ndarray | Tensor]],
    val_loader: Iterable[tuple[np.ndarray | Tensor, np.ndarray | Tensor]],
) -> dict[str, Any]:
    """Run a small training loop for one ablation arm and return results.

    Args:
        model_cfg_or_model: Either a ``ModelConfig`` (built fresh) or an already
            constructed ``TransformerLM``.
        train_cfg: ``TrainConfig`` or dict with at least ``max_steps``,
            ``max_lr``, ``grad_clip``, ``seed``.
        train_loader: Iterator over ``(inputs, targets)`` training batches.
        val_loader: Iterable over ``(inputs, targets)`` validation batches.

    Returns:
        Dictionary with ``final_val_ce``, ``diagnostics``, ``history``
        (list of ``(step, ce_loss, total_loss)``), and ``train_config``.
    """
    if isinstance(train_cfg, Mapping):
        train_cfg = TrainConfig(**train_cfg)  # type: ignore[arg-type]

    if isinstance(model_cfg_or_model, ModelConfig):
        seed_everything(train_cfg.seed)
        model = TransformerLM(model_cfg_or_model)
    else:
        model = model_cfg_or_model

    # Materialize validation batches so we can evaluate loss and then run
    # diagnostics on the same data without requiring a reusable iterator.
    val_batches = list(val_loader)
    if not val_batches:
        raise ValueError("val_loader yielded no batches")

    device = train_cfg.device
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.max_lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )

    history: list[tuple[int, float, float]] = []
    for step in range(train_cfg.max_steps):
        try:
            inputs, targets = next(train_loader)
        except StopIteration:
            break
        model.train()
        optimizer.zero_grad()
        inputs_t = _to_long_tensor(inputs, device)
        targets_t = _to_long_tensor(targets, device)

        if model.cfg.moe is not None:
            logits, aux = model(inputs_t, return_aux=True)
            ce_loss = cross_entropy(logits, targets_t)
            total_loss = ce_loss + aux.total
        else:
            logits = model(inputs_t)
            ce_loss = cross_entropy(logits, targets_t)
            total_loss = ce_loss

        total_loss.backward()
        gradient_clipping(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        if model.cfg.moe is not None:
            model.moe_update_biases()

        history.append((step, float(ce_loss.item()), float(total_loss.item())))

    val_result = evaluate_val_loss(model, val_batches)
    # Diagnostics from a single validation batch to keep the API simple.
    first_val = val_batches[0]
    diag_input = _to_long_tensor(first_val[0], device)
    diagnostics = router_diagnostics(model, diag_input)

    return {
        "final_val_ce": val_result["ce"],
        "final_val_perplexity": val_result["perplexity"],
        "diagnostics": diagnostics,
        "history": history,
        "train_config": asdict(train_cfg),
    }


# ---------------------------------------------------------------------------
# High-level matrix runner
# ---------------------------------------------------------------------------


Batch = tuple[np.ndarray | Tensor, np.ndarray | Tensor]
LoaderFactory = Callable[[], Iterator[Batch]]
ValLoaderFactory = Callable[[], Iterable[Batch]]


@dataclass
class AblationSpec:
    """Lightweight spec for ``run_moe_ablation``.

    ``train_loader`` may be an iterator (single-use) or a callable returning a
    fresh iterator, which is required when running more than one cell.  The
    same applies to ``val_loader``.
    """

    model_cfg: ModelConfig
    train_cfg: TrainConfig
    train_loader: Iterator[Batch] | LoaderFactory
    val_loader: Iterable[Batch] | ValLoaderFactory
    arms: tuple[AblationArm, ...] = (AblationArm.BIAS_FREE, AblationArm.SEQ_AUX, AblationArm.NONE)
    granularities: tuple[Granularity, ...] = (Granularity.COARSE, Granularity.FINE)
    d_model: int | None = None
    seq_aux_alpha: float = 1e-3
    bias_update_speed: float = 1e-3


def _resolve_loader(
    loader: Iterator[Batch] | LoaderFactory,
) -> Iterator[Batch]:
    return loader() if callable(loader) else loader


def _resolve_val_loader(
    loader: Iterable[Batch] | ValLoaderFactory,
) -> Iterable[Batch]:
    return loader() if callable(loader) else loader


def run_moe_ablation(spec: AblationSpec) -> dict[str, Any]:
    """Run the full arms × granularities matrix and return a results table.

    Each cell trains from a fresh seeded init so the arms are comparable.
    The returned dictionary contains ``rows`` (one per cell) and a ``summary``
    keyed by ``(arm, granularity)``.
    """
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for arm in spec.arms:
        for granularity in spec.granularities:
            moe_cfg = build_moe_config(
                arm,
                granularity,
                base_cfg=spec.model_cfg.moe,
                d_model=spec.d_model or spec.model_cfg.d_model,
                seq_aux_alpha=spec.seq_aux_alpha,
                bias_update_speed=spec.bias_update_speed,
            )
            model_cfg = replace(spec.model_cfg, moe=moe_cfg)

            result = train_arm(
                model_cfg,
                spec.train_cfg,
                _resolve_loader(spec.train_loader),
                _resolve_val_loader(spec.val_loader),
            )
            row = {
                "arm": arm.value,
                "granularity": granularity.value,
                "moe_config": asdict(moe_cfg),
                "final_val_ce": result["final_val_ce"],
                "final_val_perplexity": result["final_val_perplexity"],
                "entropy": result["diagnostics"]["entropy"],
                "balance_score": result["diagnostics"]["balance_score"],
                "max_load": result["diagnostics"]["max_load"],
                "min_load": result["diagnostics"]["min_load"],
                "history": result["history"],
            }
            rows.append(row)
            summary[f"{arm.value}_{granularity.value}"] = row

    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def ablation_table(results: dict[str, Any]) -> str:
    """Render the matrix as a Markdown table."""
    rows = results["rows"]
    if not rows:
        return ""
    header = ["arm", "granularity", "val_ce", "perplexity", "entropy", "balance_score"]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(" --- " for _ in header) + "|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["arm"],
                    r["granularity"],
                    f"{r['final_val_ce']:.4f}",
                    f"{r['final_val_perplexity']:.2f}",
                    f"{r['entropy']:.4f}",
                    f"{r['balance_score']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def save_ablation_results(
    results: dict[str, Any], out_dir: str | Path, prefix: str = "f6_moe_ablation"
) -> Path:
    """Write JSON and Markdown tables to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    md_path = out_dir / f"{prefix}.md"

    # Make JSON serializable: numpy/tensors are already converted above.
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# F6 MoE balancing ablation results\n\n")
        f.write(ablation_table(results))
        f.write("\n")
    return json_path


__all__ = [
    "AblationArm",
    "AblationSpec",
    "Granularity",
    "ablation_table",
    "build_moe_config",
    "evaluate_val_loss",
    "router_diagnostics",
    "run_moe_ablation",
    "save_ablation_results",
    "train_arm",
]
