"""DeepSeek-V3-style Mixture-of-Experts FFN — the sparse alternative to the dense SwiGLU.

L1 substrate add-on A1.1 (pulled forward per ADR-0007; ADR-0004 had sequenced it v0.2.0-B).
**Opt-in:** a model only uses this when ``ModelConfig.moe is not None``; the dense path is
byte-for-byte unchanged, so nothing is smuggled into the v0.1.0 ship.

Recipe (DeepSeek-V3 Technical Report, arXiv:2412.19437, Eq. 12-20):

- **Affinity** ``s_{i,t} = sigmoid(u_t · e_i)`` — per-expert, independent (V3 replaced V2's softmax).
- **Selection** ``TopK_i(s_{i,t} + b_i, K_r)`` — the aux-loss-free bias ``b_i`` enters the *selection*
  only; the gate *value* uses the raw ``s_{i,t}``.
- **Gate** ``g_{i,t} = g'_{i,t} / Σ_j g'_{j,t}`` — normalize over the selected experts.
- **Bias update** (trainer-driven, non-grad): ``b_i += γ·sign(mean_load − load_i)`` after each
  optimizer step (overloaded ⇒ decrease, underloaded ⇒ increase), ``γ = bias_update_speed``.
- Output = ``Σ shared_i(u_t) + scaling · Σ g_{i,t} routed_i(u_t)`` — the **FFN delta only**
  (``TransformerBlock`` owns the residual; returning the residual here would double-add it).

Repo add-ons composed on top of V3 (A1.1 — flagged, *not* part of V3):

- **Seq-wise balance loss** (V3 Eq. 17-20, tiny ``α``): ``α Σ_i f_i P_i``, computed batch-wise here
  (a documented simplification of V3's strict per-sequence statistic — single device, educational).
- **Router z-loss** (ST-MoE stabilizer, classically a *softmax* tool): a magnitude penalty on the
  raw pre-sigmoid logits ``c_z · mean_t(logsumexp_i z)²``. ``c_z = 0`` disables.
- **Diagnostics**: per-expert load fraction + router entropy (nats) — feeds the through-line.

Falsifiable prediction (A1.1): on a tiny K-of-N MoE the router entropy stays ``> 0.9·log(N_r)``
through training; collapse to a few experts ⇒ the balancer is broken (kill: >2 debug days ⇒ dense).

Out of scope (single device): node/device-limited routing + device/comm balance losses — pure
expert-parallel comm optimizations, irrelevant here; a GPU/EP follow-up.

CPU compute is an explicit per-expert loop (gather → expert → scatter via ``index_add``): correct
and clear. A vectorized batched-gather is a GPU/perf follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from reasoning_llm.model import Linear, SwiGLU


def _default_expert_ffn(d_model: int) -> int:
    """The SwiGLU convention: round (8/3)·d_model up to a multiple of 64 (mirrors
    ``ModelConfig.ffn_dim``). Fine-grained experts opt in to a *smaller* dim via
    ``MoEConfig.expert_d_ff``."""
    raw = int(8 / 3 * d_model)
    return ((raw + 63) // 64) * 64


@dataclass(frozen=True)
class MoEConfig:
    """Routed-FFN hyperparameters. ``expert_d_ff=None`` ⇒ a full-size SwiGLU expert (set a
    smaller value for DeepSeek-style fine-grained segmentation). DeepSeek-V3 ships
    ``n_routed_experts=256, n_shared_experts=1, n_experts_per_tok=8, routed_scaling_factor≈2.5``;
    we default ``routed_scaling_factor=1.0`` so the loss-at-init derivation stays clean."""

    n_routed_experts: int
    n_experts_per_tok: int = 8
    n_shared_experts: int = 1
    expert_d_ff: int | None = None
    n_dense_layers: int = 0  # leading layers stay dense (V3 keeps the first few dense)
    aux_loss_alpha: float = 1e-4  # seq-wise balance loss weight (V3 uses 1e-4)
    z_loss_coef: float = 1e-3  # router z-loss weight; 0.0 disables
    bias_update_speed: float = 1e-3  # γ for the aux-loss-free bias (V3 uses 1e-3)
    routed_scaling_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.n_routed_experts < 1:
            raise ValueError(f"n_routed_experts={self.n_routed_experts} must be ≥ 1")
        if not 1 <= self.n_experts_per_tok <= self.n_routed_experts:
            raise ValueError(
                f"n_experts_per_tok={self.n_experts_per_tok} must be in "
                f"[1, n_routed_experts={self.n_routed_experts}]"
            )
        if self.n_shared_experts < 0:
            raise ValueError(f"n_shared_experts={self.n_shared_experts} must be ≥ 0")


@dataclass
class MoEStats:
    """Per-MoE-layer outputs threaded out of the forward pass. ``aux_loss``/``z_loss`` are
    differentiable scalars added to the training objective; the rest are detached diagnostics."""

    aux_loss: Tensor  # seq-wise balance loss (scalar, differentiable)
    z_loss: Tensor  # router z-loss (scalar, differentiable)
    load_fraction: Tensor  # (n_routed,) fraction of routed slots per expert (detached)
    entropy: Tensor  # router entropy over the load distribution, nats (detached scalar)


@dataclass
class AuxOutput:
    """Model-level aggregate of every MoE layer's stats (``TransformerLM.forward(return_aux=True)``)."""

    aux_loss: Tensor  # summed across MoE layers
    z_loss: Tensor  # summed across MoE layers
    layers: list[MoEStats]

    @property
    def total(self) -> Tensor:
        """The full sparse-regularization term to add to cross-entropy."""
        return self.aux_loss + self.z_loss


class Router(nn.Module):
    """Top-K router with an aux-loss-free load-balancing bias.

    ``gate`` holds the expert centroids ``e_i`` as rows (a bias-free ``Linear``). ``bias`` is the
    non-grad ``b_i`` (checkpointed — it is learned balancing state); ``load_count`` accumulates
    selection counts between bias updates (transient, not checkpointed).
    """

    bias: Tensor
    load_count: Tensor

    def __init__(self, d_model: int, n_routed_experts: int) -> None:
        super().__init__()
        self.gate = Linear(d_model, n_routed_experts)
        self.register_buffer("bias", torch.zeros(n_routed_experts), persistent=True)
        self.register_buffer("load_count", torch.zeros(n_routed_experts), persistent=False)

    def forward(self, x_flat: Tensor) -> Tensor:
        """(N, d_model) → router logits (N, n_routed) = x · e_iᵀ (pre-sigmoid)."""
        return self.gate(x_flat)

    @torch.no_grad()
    def update_bias(self, speed: float) -> None:
        """One aux-loss-free balancing step: nudge each ``b_i`` by ±speed toward the mean load,
        then reset the accumulator. Underloaded experts get a higher bias (selected more often)."""
        if float(self.load_count.sum()) == 0.0:
            return
        violation = self.load_count.mean() - self.load_count  # >0 ⇒ underloaded ⇒ raise bias
        self.bias += speed * torch.sign(violation)
        self.load_count.zero_()


class MoEFeedForward(nn.Module):
    """DeepSeek-V3 MoE FFN: ``n_shared`` always-on experts + ``n_routed`` Top-K experts.

    ``forward`` returns ``(delta, stats)`` where ``delta`` is the FFN contribution only — same
    contract as :class:`reasoning_llm.model.SwiGLU`, so the block's ``x + ffn(x)`` is unchanged.
    """

    def __init__(self, d_model: int, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_routed = cfg.n_routed_experts
        self.k = cfg.n_experts_per_tok
        d_ff = cfg.expert_d_ff if cfg.expert_d_ff is not None else _default_expert_ffn(d_model)
        self.router = Router(d_model, self.n_routed)
        self.routed_experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(self.n_routed))
        self.shared_experts = nn.ModuleList(
            SwiGLU(d_model, d_ff) for _ in range(cfg.n_shared_experts)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, MoEStats]:
        b, s, d = x.shape
        xf = x.reshape(-1, d)  # (N, d); routing is per-token, position-independent
        n_tokens = xf.shape[0]

        logits = self.router(xf)  # (N, n_routed), pre-sigmoid
        affinity = torch.sigmoid(logits)  # s_{i,t} ∈ (0, 1), per-expert independent

        # Selection uses s + b (aux-loss-free bias); the gate VALUE uses raw s.
        sel_scores = affinity + self.router.bias  # broadcast (n_routed,)
        topk_idx = sel_scores.topk(self.k, dim=-1).indices  # (N, k)
        gate_sel = affinity.gather(-1, topk_idx)  # (N, k) — raw affinity, not s+b
        gate_norm = gate_sel / gate_sel.sum(-1, keepdim=True)  # Σ over selected = 1 (sigmoid>0)

        # Dense, full-width gate + selection masks (N, n_routed).
        gates = torch.zeros_like(affinity).scatter(-1, topk_idx, gate_norm)
        selected = torch.zeros_like(affinity).scatter(-1, topk_idx, torch.ones_like(gate_norm))
        counts = selected.sum(0)  # (n_routed,) tokens routed to each expert

        # Routed experts: gather tokens per expert, run, scatter-add weighted by the gate.
        y = torch.zeros_like(xf)
        for e in range(self.n_routed):
            idx = (selected[:, e] > 0).nonzero(as_tuple=True)[0]  # (M,) token indices
            if idx.numel() == 0:
                continue
            out = self.routed_experts[e](xf[idx])  # (M, d)
            weighted = gates[idx, e].unsqueeze(-1) * out
            y = y.index_add(0, idx, weighted)  # out-of-place: autograd-safe accumulate
        y = y * self.cfg.routed_scaling_factor

        # Shared experts: always-on, every token.
        for shared in self.shared_experts:
            y = y + shared(xf)

        stats = self._stats(logits, affinity, counts, n_tokens)
        if self.training:  # accumulate load between bias updates (non-grad)
            with torch.no_grad():
                self.router.load_count += counts.detach()
        return y.reshape(b, s, d), stats

    def _stats(self, logits: Tensor, affinity: Tensor, counts: Tensor, n_tokens: int) -> MoEStats:
        # Seq-wise balance loss (batch-wise simplification of V3 Eq. 17-20):
        #   f_i = (N_r / (K_r·T)) · count_i  (detached);  P_i = mean_t s'_{i,t}  (differentiable)
        f = (self.n_routed / (self.k * n_tokens)) * counts.detach()
        s_norm = affinity / affinity.sum(-1, keepdim=True)  # s'_{i,t}
        p_mean = s_norm.mean(0)  # P_i
        aux_loss = self.cfg.aux_loss_alpha * (f * p_mean).sum()

        # Router z-loss on the raw logits (magnitude penalty; c_z=0 disables).
        if self.cfg.z_loss_coef:
            z_loss = self.cfg.z_loss_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()
        else:
            z_loss = logits.new_zeros(())

        # Diagnostics: load distribution over experts and its entropy (max = log n_routed).
        with torch.no_grad():
            load_fraction = counts / counts.sum().clamp_min(1.0)
            p = load_fraction.clamp_min(1e-12)
            entropy = -(p * p.log()).sum()
        return MoEStats(aux_loss, z_loss, load_fraction, entropy)


__all__ = ["AuxOutput", "MoEConfig", "MoEFeedForward", "MoEStats", "Router"]
