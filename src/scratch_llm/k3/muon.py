"""k3.muon — Per-Head Muon + QK-Clip, K3's optimizer (report §2.5; docs/k3/FACTS.md A8).

Per-Head Muon. ``optim.Muon`` orthogonalizes each 2-D momentum matrix as one piece. K3 first cuts
the per-head projections into their heads: a weight [R, C] whose rows are R / h consecutive heads
of h = ``head_rows`` rows each is updated as R / h independent [h, C] matrices,

    buf ← momentum·buf + g ;   g̃ ← g + momentum·buf               whole matrix, as optim.Muon
    O_i ← NewtonSchulz₅(g̃[i·h : (i+1)·h])                          one call per head block
    θ_i ← (1 − lr·wd)·θ_i − lr·rms_scale·√max(h, C)·O_i              the block's own shape

Momentum and decoupled weight decay are elementwise, so only Newton–Schulz and the RMS scale
see the blocks. Why per head: NS on the stacked matrix couples the heads through one Frobenius
pre-normalization and one polar factor. The quintic's slope at 0 is 3.4445, so 5 steps lift a
singular value by at most ~485×: a head whose momentum is 1e-3 of its neighbours' gets rows of
norm ~0.1 instead of ~0.9 (measured, [2·16, 64]). And a tall stack (R > C) has only C singular
directions for all heads to share, so small heads get less of them. Either way the large-momentum
heads take the update; per block, every head gets its own direction at the same RMS.

Invariants (tests/test_k3_muon.py):
1. ``head_rows`` None or == R ⇒ bitwise ``optim.Muon`` (same helpers in the same order).
2. ``head_rows`` = h < R ⇒ block i of the update is NS of block i alone, scaled by
   rms_scale·√max(h, C). A full-rank [h, C] orthogonal block has element RMS 1/√max(h, C), so
   every head's update has RMS ``rms_scale`` (0.2, AdamW's band) whatever the head count.
3. With more than one block this is NOT whole-matrix Muon — the scaffold's "head-uniform ⇒ equals
   optim.Muon" was false for H > 1. The polar factor of a stack is the stack of polar factors
   only when the heads' row spaces are mutually orthogonal, and even then the 5-step quintic
   differs because each block is pre-normalized by its own Frobenius norm. The scale also
   differs whenever R > C (full KDA q_proj [12288, 7168]: √7168 vs √12288); for R ≤ C (mini
   q_proj [1024, 1024]) it coincides.

Production precedent (torchtitan @ d263ca0, Kimi K2.7 recipe): ``kimi_k2_7/config_registry.py``
:302-331 gives MLA ``wq_b`` blocks of nope + rope rows and ``wkv_b`` blocks of nope + v rows;
``distributed/flex_shard/dist_muon.py``:66-74 makes each consecutive R rows one independent
matrix, and :215 + :1816-1820 take the RMS scale 0.2·√max(rows, cols) from that block shape.
torchtitan keeps momentum as an EMA (``lerp_``, :1837), i.e. (1 − momentum)× ours; NS's
Frobenius pre-normalization cancels that factor (up to its 1e-7 eps), so the directions agree.
K2.7 has no KDA: the KDA q/k/v split at head_dim rows is our reading of the report's
"partitioned Q/K/V", not a disclosed recipe.

Partition (:func:`k3_param_groups`, torchtitan config_registry.py:427-441): per-head Muon on
KDA ``q_proj/k_proj/v_proj`` and MLA ``q_b_proj/kv_b_proj``; whole-matrix Muon on every other
2-D weight, including the router ``gate.weight`` [E, H] (kept on Muon there, after Moonlight
Fig. 4); AdamW on ``embed_tokens``, ``lm_head``, the AttnRes pseudo-queries ``*res_proj`` [1, H],
the KDA short convs and every 1-D tensor. Parameters with ``requires_grad=False`` (the
Quantile-Balancing bias, set by assignment) join no group.

QK-Clip (MuonClip, Kimi K2 arXiv:2507.20534; torchtitan ``kimi_k2_7/qk_clip.py``:79-177), after
the optimizer step, per MLA head h whose forward recorded a max pre-softmax logit S_h:

    γ_h = τ / max(S_h, τ)                            1 for heads already at or under τ
    q_b_proj:  nope rows × γ^α,   rope rows × γ
    kv_b_proj: k_nope rows × γ^(1−α),   v rows × 1

The logit is scale·(q_nope·k_nope + q_rope·k_rope). The nope term scales by γ^α·γ^(1−α) = γ and
the rope term by γ·1 = γ: ``k_rope`` comes from ``kv_a_proj_with_mqa`` and is shared by every
head, so scaling it would clip all heads at once; q_rope takes the whole factor instead. For the
same input a re-forward then has max logit min(S_h, τ), to rounding. KDA is never clipped: its q
and k are L2-normalized, so |logit| ≤ 1/√head_dim.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from scratch_llm.k3.core.gated_mla import GatedMLA
from scratch_llm.k3.core.kda import KDALayer
from scratch_llm.optim import (
    AdamW,
    CombinedOptimizer,
    _zeropower_via_newtonschulz5,
    muon_apply_,
    muon_momentum_,
)

__all__ = ["PerHeadMuon", "k3_param_groups", "build_k3_optimizer", "apply_k3_qk_clip"]


class PerHeadMuon(Optimizer):
    """Muon whose param groups may carry ``head_rows``: NS and the RMS scale then run per
    consecutive ``head_rows``-row block (module docstring). ``head_rows=None`` is
    ``optim.Muon`` exactly; hyperparameters and their defaults are ``optim.Muon``'s.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.1,
        rms_scale: float = 0.2,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be ≥ 1, got {ns_steps}")
        if weight_decay < 0 or rms_scale < 0:
            raise ValueError("weight_decay and rms_scale must be ≥ 0")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            rms_scale=rms_scale,
            head_rows=None,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            head_rows = group["head_rows"]
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only optimizes 2-D matrices; got shape {tuple(p.shape)}. Route "
                        "embeddings/head/1-D params to AdamW (see k3_param_groups)."
                    )
                if head_rows is not None and (head_rows < 1 or p.shape[0] % head_rows != 0):
                    raise ValueError(
                        f"head_rows={head_rows} must be ≥ 1 and divide the row count of "
                        f"shape {tuple(p.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                g_eff = muon_momentum_(
                    p.grad, state["momentum_buffer"], group["momentum"], group["nesterov"]
                )
                rows = p.shape[0] if group["head_rows"] is None else group["head_rows"]
                for start in range(0, p.shape[0], rows):
                    block = slice(start, start + rows)  # a view: the update lands in p
                    ortho = _zeropower_via_newtonschulz5(g_eff[block], group["ns_steps"])
                    muon_apply_(
                        p[block],
                        ortho,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        rms_scale=group["rms_scale"],
                    )

        return loss


def _owner(name: str) -> str:
    """Attribute name of the module that owns a parameter: ``layers.3.mlp.gate.weight`` → ``gate``."""
    return name.rpartition(".")[0].rpartition(".")[2]


def _head_rows_by_param(model: nn.Module) -> dict[int, int]:
    """id(weight) → rows per head, for the projections whose rows are stacked heads."""
    head_rows: dict[int, int] = {}
    for module in model.modules():
        if isinstance(module, KDALayer):
            for proj in (module.q_proj, module.k_proj, module.v_proj):
                head_rows[id(proj.weight)] = module.cfg.head_dim
        elif isinstance(module, GatedMLA):
            cfg = module.cfg
            head_rows[id(module.q_b_proj.weight)] = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
            head_rows[id(module.kv_b_proj.weight)] = cfg.qk_nope_head_dim + cfg.v_head_dim
    return head_rows


def k3_param_groups(model: nn.Module) -> tuple[list[dict[str, Any]], list[nn.Parameter]]:
    """Split ``model`` into (PerHeadMuon groups, AdamW params). Rules, first match wins:

    - ``requires_grad=False`` → no group (the Quantile-Balancing bias is assigned, not stepped);
    - fewer than 2 dims, a ``*conv1d`` short-conv kernel [P, 1, kernel], ``embed_tokens``,
      ``lm_head`` or an AttnRes pseudo-query ``*res_proj`` [1, H] → AdamW;
    - any other 2-D weight → Muon, with ``head_rows`` = head_dim for KDA q/k/v, nope + rope for
      MLA ``q_b_proj``, nope + v for MLA ``kv_b_proj``, None (whole matrix) otherwise;
    - anything else raises: a stacked 3-D expert tensor needs an explicit decision.

    Muon groups come one per distinct ``head_rows``, in first-seen order. Tied tensors are
    visited once. Raises if the result is not an exact cover of the trainable parameters.
    """
    head_rows = _head_rows_by_param(model)
    muon: dict[int | None, list[nn.Parameter]] = {}
    adamw: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        owner = _owner(name)
        if (
            p.ndim < 2
            or owner.endswith("conv1d")
            or owner in ("embed_tokens", "lm_head")
            or owner.endswith("res_proj")
        ):
            adamw.append(p)
        elif p.ndim == 2:
            muon.setdefault(head_rows.get(id(p)), []).append(p)
        else:
            raise ValueError(f"{name}: no optimizer rule for a {p.ndim}-D parameter")
    groups = [{"params": params, "head_rows": rows} for rows, params in muon.items()]

    routed = [id(p) for group in groups for p in group["params"]] + [id(p) for p in adamw]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if len(routed) != len(set(routed)) or set(routed) != trainable:
        raise RuntimeError("k3_param_groups must place every trainable parameter exactly once")
    return groups, adamw


def build_k3_optimizer(
    model: nn.Module,
    *,
    lr: float = 2e-2,
    adamw_lr: float = 3e-3,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    momentum: float = 0.95,
) -> CombinedOptimizer:
    """PerHeadMuon (``lr``) + AdamW (``adamw_lr``) over :func:`k3_param_groups`, one decoupled
    ``weight_decay`` for both (report §2.5: 0.1). LR schedules belong to the trainer."""
    muon_groups, adamw_params = k3_param_groups(model)
    muon = PerHeadMuon(muon_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    adamw = AdamW(adamw_params, lr=adamw_lr, betas=betas, weight_decay=weight_decay)
    return CombinedOptimizer([muon, adamw])


@torch.no_grad()
def apply_k3_qk_clip(model: nn.Module, tau: float = 100.0, alpha: float = 0.5) -> dict[str, float]:
    """QK-Clip every :class:`GatedMLA` with a recorded ``last_max_logits`` (module docstring).

    Call after ``optimizer.step()``. Each observation is consumed (``last_max_logits`` is reset
    to None, as torchtitan clears its record, qk_clip.py:177), so a stale maximum never clips
    twice. Heads with S_h ≤ τ get γ = 1: their rows are multiplied by exactly 1.0, a bitwise
    no-op. A non-finite S_h raises: the forward already diverged, and γ would write NaN or 0
    into the weights. Returns {module name: min_h γ_h}; 1.0 means nothing was clipped.
    """
    if not tau > 0 or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"need tau > 0 and 0 ≤ alpha ≤ 1; got tau={tau}, alpha={alpha}")
    min_gamma: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, GatedMLA) or module.last_max_logits is None:
            continue
        cfg = module.cfg
        H, nope = cfg.num_heads, cfg.qk_nope_head_dim
        q_b, kv_b = module.q_b_proj.weight, module.kv_b_proj.weight
        s_max = module.last_max_logits.detach()
        if s_max.shape != (H,):
            raise ValueError(f"{name}: last_max_logits must be [{H}], got {tuple(s_max.shape)}")
        if not bool(torch.isfinite(s_max).all()):
            raise ValueError(f"{name}: non-finite max logit {s_max.tolist()}")
        dtype = torch.promote_types(torch.promote_types(s_max.dtype, q_b.dtype), torch.float32)
        gamma = (tau / s_max.to(q_b.device, dtype).clamp_min(tau)).view(H, 1, 1)

        q = q_b.view(H, cfg.q_head_dim, q_b.shape[1])  # per head [nope | rope] rows
        kv = kv_b.view(H, nope + cfg.v_head_dim, kv_b.shape[1])  # per head [k_nope | v] rows
        q[:, :nope].mul_(gamma.pow(alpha))
        q[:, nope:].mul_(gamma)  # rope: the whole factor, since the shared k_rope takes none
        kv[:, :nope].mul_(gamma.pow(1.0 - alpha))

        module.last_max_logits = None
        min_gamma[name] = gamma.min().item()
    return min_gamma
