"""core/latent_moe.py — Stable LatentMoE: sigmoid router, latent-width SiTU experts, Quantile
Balancing.

One MoE FFN (every layer after ``first_k_dense``), returning the delta the block adds:

    s  = sigmoid(x Wᵀ)                        router at FULL width, fp32 logits     [N, E]
    T  = top-k of (s + b)                     b = e_score_correction_bias: selection only
    p  = s[T] / (Σ s[T] + 1e-20) · scale      mixture weights from s alone (k > 1)
    u  = Σ_{e ∈ T} p_e · w2_e(SiTU-GLU(w1_e z, w3_e z)),   z = routed_expert_down_proj(x)
    y  = routed_expert_up_proj(RMSNorm(u)) + shared_experts(x)

No norm before the down projection; the latent RMSNorm sits between the weighted combine and the
up projection (report Eq. 11, "Normalized LatentMoE"). Routed experts run at latent width
ℓ = hidden/2, so their parameters and the dispatch traffic both halve; the router and the shared
experts read the full-width x. The bias moves which experts are chosen and never how much they
weigh: were it in p, a high-bias expert would be over-weighted and the router's gradient would
depend on the balancing state.

Quantile Balancing (report Eq. 14) sets the bias without an auxiliary loss or a step size. For
token i let α_i be the (k+1)-th largest entry of s_i + b, the best score the top-k leaves out:
under the bias in force, expert j is in token i's top-k exactly when s_ij + b_j > α_i. QB holds α
fixed and asks which bias b'_j would give expert j its share. Token i then goes to j when

    s_ij + b'_j > α_i   ⟺   s_ij − α_i > −b'_j        (the margin must clear −b'_j)

so j's count c_j(τ) = #{i : s_ij − α_i > τ} is nonincreasing in the threshold τ = −b'_j. Setting
c_j = m·k/n (m tokens, n experts: each expert gets its share of the m·k slots) puts τ where a
fraction 1 − k/n of the margins lie at or below it, i.e. at their (1 − k/n)-quantile:

    b̂_j = −quantile_{1−k/n}(s_{:,j} − α),   b ← b̂ − mean(b̂)       direct assignment

Adding a constant to every b_j shifts every s + b and every α_i by it and changes no selection,
so the mean removal only fixes that gauge. α uses the bias in force (flagged: the report does not
say s + b or s), so the update is a fixed-point iteration b ← F(b) whose fixed points give every
expert m·k/n; α from the raw s would ignore how far the biases have already moved the cutoffs.
Holding α fixed is exact for an expert inside the top-k (to stay, it must beat the best rejected
score, α) but optimistic for one outside (to enter, it must beat the k-th score, above α): each
token's runner-up sits at margin exactly −b_j, an atom at the current threshold. So an
underloaded expert is raised too little per step, and the iteration needs several steps (about
ten on the tests' skewed router), not one.

The quantile comes from a per-expert histogram of the margins over fixed bins, linearly
interpolated inside the bin that reaches the target rank. The bins need no data-dependent range:
s ∈ [0, 1] and α_i is one of the s_il + b_l, so α_i ∈ [min b, 1 + max b] and every margin lies in
[−1 − max b, 1 − min b]. That range depends only on the bias, which every data-parallel rank
holds identically, so all ranks bin identically and the counts of the global batch are the sum
of the per-rank counts: one all-reduce (see QBStats). With QB_NUM_BINS = 512 the bin width is
(2 + max b − min b)/512 ≈ 4-5e-3 for the checkpoint's biases (per-layer std 0.02-0.1, FACTS A19c).

Timing: a training forward records the statistics of the batch it routed (several forwards
before an update add up, as micro-batches of one step should); the trainer calls
``update_bias()`` after ``optimizer.step()``, so the bias used at step t+1 comes from step t and a
batch is never routed with a bias derived from itself. Eval and no-grad forwards record nothing,
and forward never writes the bias: it is frozen at inference.

k == n is degenerate: every expert takes every token, no α exists and no bias can change the
selection, so QB sets b̂ ≡ 0 (any constant is the same bias after the mean removal).

Sources (reviewed 2026-09-22): Moonshot's HF reference for the checkpoint,
``oss/kimi_k3_hf/modeling_kimi_linear.py`` — ``KimiMoEGate`` :666-759, ``KimiSparseMoeBlock``
:762-838, ``moe_infer`` :840-874, ``KimiBlockSparseMLP`` :242-270, ``KimiRMSNorm`` :226-239,
``SituAndMul`` :64-79, ``_init_weights`` :1063-1073; vLLM @ dedcfa4483
``models/kimi_k3/nvidia/model.py:614-626`` (fp32 router logits, fp32 bias) and
``fused_moe/router/grouped_topk_router.py:80-161`` (config.json has one expert group, and one
group reduces grouped top-k to plain top-k); the K3 report Eq. 11-14 via
docs/k3/READING_GUIDE_K3_REPORT.md:139-155. HF's ``moe_infer`` runs under no_grad and its block
raises in training mode; the dispatch here is the same sort/run/unsort, written to carry autograd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_llm.k3.config import K3Config, MoEConfig
from scratch_llm.k3.core.norm import RMSNorm
from scratch_llm.k3.core.situ import DenseSiTUMLP, SiTUConfig, SiTUGLU

__all__ = [
    "QB_NUM_BINS",
    "GateOutput",
    "LatentExpert",
    "LatentMoE",
    "MoEGate",
    "QBStats",
    "histogram_quantile",
    "margin_histogram",
    "margin_range",
]

#: Histogram bins per expert. Report: "a few hundred". On strongly skewed synthetic routers
#: (E=16, k=2, 16K tokens/step, 8 router draws) the worst per-expert load deviation from k/n
#: over rounds 16-20 was 0.008-0.013 at 512 bins, as with the exact quantile (0.007-0.010), but
#: 0.013-0.019 at 256 and 0.018-0.029 at 128, against a 0.0146 sampling band (tests, gate 5).
QB_NUM_BINS = 512


def _math_dtype(*xs: Tensor) -> torch.dtype:
    """fp32 for every low-precision input, fp64 if any input is fp64."""
    dtype = torch.float32
    for x in xs:
        dtype = torch.promote_types(dtype, x.dtype)
    return dtype


# ---------------------------------------------------------------------------
# Quantile Balancing — pure functions over the margins s_ij − α_i.
# ---------------------------------------------------------------------------


def margin_range(bias: Tensor) -> tuple[Tensor, Tensor]:
    """[lo, hi] = [−1 − max b, 1 − min b]: holds every margin s − α for this bias (0-d, fp64)."""
    b = bias.detach().double()
    return -1.0 - b.max(), 1.0 - b.min()


def margin_histogram(margins: Tensor, lo: Tensor, hi: Tensor, num_bins: int) -> Tensor:
    """Per-expert counts [E, num_bins] (int64) of ``margins`` [N, E] over num_bins equal bins
    spanning [lo, hi]. The ends are clamped in, so a margin sitting exactly on hi still counts."""
    E = margins.shape[1]
    width = (hi - lo) / num_bins
    idx = ((margins.double() - lo) / width).floor().long().clamp_(0, num_bins - 1)
    flat = idx + torch.arange(E, device=idx.device) * num_bins  # expert j owns [j·bins, (j+1)·bins)
    return torch.bincount(flat.flatten(), minlength=E * num_bins).view(E, num_bins)


def histogram_quantile(counts: Tensor, lo: Tensor, hi: Tensor, level: float) -> Tensor:
    """Per-row quantile [E] (fp64) at ``level`` ∈ (0, 1) of histograms ``counts`` [E, bins].

    The target rank is r = level · (row total): the value with r margins at or below it. The
    first bin whose cumulative count reaches r holds it, and the estimate interpolates linearly
    inside that bin, as if its margins were spread evenly. Since r > 0 and the previous bin's
    cumulative count is < r, the chosen bin is never empty.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    counts = counts.double()
    cum = counts.cumsum(1)
    if bool((cum[:, -1] == 0).any()):
        raise ValueError("every histogram row needs at least one margin")
    num_bins = counts.shape[1]
    width = (hi - lo) / num_bins
    rank = level * cum[:, -1:]  # [E, 1]
    at = torch.searchsorted(cum, rank).clamp_(max=num_bins - 1)  # first bin with cum >= rank
    in_bin = counts.gather(1, at)
    frac = (rank - (cum.gather(1, at) - in_bin)) / in_bin  # ∈ (0, 1]
    return (lo + (at + frac) * width).squeeze(1)


@dataclass
class QBStats:
    """One layer's Quantile-Balancing statistics for the batch(es) routed since the last update.

    Everything but the range is a count, so the statistics of a global batch are the elementwise
    sum over its shards. Across data-parallel ranks that is one collective,
    ``all_reduce(cat([margin_counts.flatten(), load, num_tokens.view(1)]), SUM)``, split back
    afterwards; lo/hi need none, because they come from the replicated bias. Gradient-accumulation
    micro-batches add the same way (``+``). At full scale the payload is 896 × 512 counts per
    layer; int32 would halve it and stays exact below 2^31 tokens per step.
    """

    margin_counts: Tensor  # [E, bins] int64: per expert, histogram of s_ij − α_i
    load: Tensor  # [E] int64: tokens each expert took under the bias in force
    num_tokens: Tensor  # 0-d int64
    lo: Tensor  # 0-d fp64: histogram range, from the bias in force
    hi: Tensor

    @property
    def load_fraction(self) -> Tensor:
        """load / num_tokens per expert; balanced routing puts every entry at k/n."""
        return self.load.double() / self.num_tokens.double()

    def __add__(self, other: QBStats) -> QBStats:
        if not (torch.equal(self.lo, other.lo) and torch.equal(self.hi, other.hi)):
            raise ValueError("QB stats were binned under different biases; update_bias() first")
        return QBStats(
            self.margin_counts + other.margin_counts,
            self.load + other.load,
            self.num_tokens + other.num_tokens,
            self.lo,
            self.hi,
        )


# ---------------------------------------------------------------------------
# Router and experts.
# ---------------------------------------------------------------------------


class GateOutput(NamedTuple):
    scores: Tensor  # [N, E] s = sigmoid(x Wᵀ), fp32 (fp64 for fp64 input); carries the router grad
    topk_idx: Tensor  # [N, k] int64, ordered by s + b descending, ties to the lower expert id
    topk_weight: Tensor  # [N, k] mixture weights p, same dtype as scores, bias-free
    cutoff: Tensor | None  # [N] α = (k+1)-th largest of s + b, detached; None when k == E


class MoEGate(nn.Module):
    """HF ``KimiMoEGate``: router weight [E, H] + selection bias [E] (checkpoint names).

    A plain Module, not an ``nn.Linear``, as in HF: the model-wide init (normal(0, 0.02) for every
    ``nn.Linear``, HF ``_init_weights``) must skip the router, which keeps
    ``kaiming_uniform_(a=√5)`` (HF ``reset_parameters``). The bias is an fp32, zero-init parameter
    with ``requires_grad=False``: the checkpoint stores it and ``param_count`` counts it, the
    optimizer never sees it, and only ``LatentMoE.update_bias`` writes it. Keep the module out of
    a blanket bf16 cast, which would round the bias.
    """

    def __init__(self, cfg: MoEConfig, hidden_size: int) -> None:
        super().__init__()
        self.top_k = cfg.top_k
        self.num_experts = cfg.num_experts
        self.renormalize = cfg.renormalize
        self.routed_scaling_factor = cfg.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(cfg.num_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(cfg.num_experts, dtype=torch.float32), requires_grad=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> GateOutput:
        """x [N, H] -> GateOutput. One stable descending sort of s + b gives both the top-k
        (``kernels/moe/routing.topk_deterministic``'s tie rule) and α, the (k+1)-th value."""
        k, E = self.top_k, self.num_experts
        with torch.autocast(x.device.type, enabled=False):
            dtype = _math_dtype(x, self.weight)
            scores = torch.sigmoid(F.linear(x.to(dtype), self.weight.to(dtype)))
            biased = scores.detach() + self.e_score_correction_bias.to(dtype)
            ranked, order = torch.sort(biased, dim=-1, descending=True, stable=True)
            topk_idx = order[:, :k]
            weight = scores.gather(1, topk_idx)
            if k > 1 and self.renormalize:  # HF: never at k = 1 (vLLM renormalizes to 1 there)
                weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-20)
            weight = weight * self.routed_scaling_factor
        return GateOutput(scores, topk_idx, weight, ranked[:, k] if k < E else None)

    def extra_repr(self) -> str:
        return f"hidden={self.weight.shape[1]}, experts={self.num_experts}, top_k={self.top_k}"


class LatentExpert(nn.Module):
    """One routed expert, HF ``KimiBlockSparseMLP`` at latent width: w2(SiTU-GLU(w1 z, w3 z))."""

    def __init__(self, latent: int, intermediate: int, cfg: SiTUConfig | K3Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(latent, intermediate, bias=False)  # gate
        self.w2 = nn.Linear(intermediate, latent, bias=False)  # down
        self.w3 = nn.Linear(latent, intermediate, bias=False)  # up
        self.act = SiTUGLU(cfg)

    def forward(self, z: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(z), self.w3(z)))


# ---------------------------------------------------------------------------
# The block.
# ---------------------------------------------------------------------------


class LatentMoE(nn.Module):
    """One Stable LatentMoE FFN: forward(x [..., H]) -> delta [..., H]; the block adds it.

    Parameter names are the checkpoint's (``block_sparse_moe.*``): ``gate.weight``,
    ``gate.e_score_correction_bias``, ``routed_expert_down_proj``, ``experts.{e}.{w1,w2,w3}``,
    ``routed_expert_norm``, ``routed_expert_up_proj``, ``shared_experts.{gate,up,down}_proj``.
    Totals equal ``param_count._moe_block_params``.
    """

    def __init__(
        self,
        cfg: MoEConfig,
        hidden_size: int,
        model_cfg: K3Config,
        *,
        qb_num_bins: int = QB_NUM_BINS,
    ) -> None:
        super().__init__()
        if not 1 <= cfg.top_k <= cfg.num_experts:
            raise ValueError(f"top_k={cfg.top_k} must lie in [1, num_experts={cfg.num_experts}]")
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.qb_num_bins = qb_num_bins
        latent = cfg.latent_size
        self.gate = MoEGate(cfg, hidden_size)
        self.routed_expert_down_proj = nn.Linear(hidden_size, latent, bias=False)
        self.experts = nn.ModuleList(
            LatentExpert(latent, cfg.expert_intermediate, model_cfg) for _ in range(cfg.num_experts)
        )
        self.routed_expert_norm = RMSNorm(latent, model_cfg.rms_norm_eps)
        self.routed_expert_up_proj = nn.Linear(latent, hidden_size, bias=False)
        self.shared_experts = DenseSiTUMLP(hidden_size, cfg.shared_intermediate, model_cfg)
        self._pending: QBStats | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Returns the FFN delta, shaped like x. A training forward with autograd on adds this
        batch's QBStats to the pending ones (summed, so micro-batches of one step accumulate);
        ``update_bias`` consumes them. Eval and no-grad forwards record nothing."""
        xf = x.reshape(-1, self.hidden_size)
        route = self.gate(xf)
        if self.training and torch.is_grad_enabled():
            self._record(route)
        u = self._experts(self.routed_expert_down_proj(xf), route.topk_idx, route.topk_weight)
        y = self.routed_expert_up_proj(self.routed_expert_norm(u))
        return (y + self.shared_experts(xf)).view(x.shape)

    def _experts(self, z: Tensor, topk_idx: Tensor, topk_weight: Tensor) -> Tensor:
        """u = Σ_slots p · expert(z), HF ``moe_infer``'s dataflow with autograd kept.

        The N·k (token, slot) pairs are sorted by expert so each expert runs once on a contiguous
        run of rows (empty experts are skipped: at small batch they are the common case), then
        scattered back to slot order and summed over the k slots in the weights' dtype, as HF does.
        """
        N, k = topk_idx.shape
        flat = topk_idx.reshape(-1)  # slot t·k + j -> expert id
        order = flat.argsort(stable=True)
        runs = torch.bincount(flat, minlength=self.cfg.num_experts).tolist()  # host sync, as HF
        grouped = z[order // k]  # each slot's token row, grouped by expert
        outs = [
            expert(rows)
            for expert, rows in zip(self.experts, grouped.split(runs), strict=True)
            if rows.shape[0] > 0
        ]
        by_expert = torch.cat(outs)
        # Back to slot order: HF's new_x[idxs] = outs, as an out-of-place (differentiable) copy.
        by_slot = torch.zeros_like(by_expert).index_copy(0, order, by_expert)
        mixed = (by_slot.view(N, k, -1).to(topk_weight.dtype) * topk_weight[..., None]).sum(1)
        return mixed.to(z.dtype)

    @torch.no_grad()
    def _record(self, route: GateOutput) -> None:
        E, bins = self.cfg.num_experts, self.qb_num_bins
        lo, hi = margin_range(self.gate.e_score_correction_bias)
        if route.cutoff is None:  # k == E: no cutoff, nothing for QB to estimate
            counts = torch.zeros(E, bins, dtype=torch.long, device=route.scores.device)
        else:
            margins = route.scores.detach() - route.cutoff[:, None]
            counts = margin_histogram(margins, lo, hi, bins)
        stats = QBStats(
            margin_counts=counts,
            load=torch.bincount(route.topk_idx.flatten(), minlength=E),
            num_tokens=torch.tensor(route.topk_idx.shape[0], device=counts.device),
            lo=lo,
            hi=hi,
        )
        self._pending = stats if self._pending is None else self._pending + stats

    @torch.no_grad()
    def update_bias(self) -> QBStats | None:
        """Quantile Balancing: b ← b̂ − mean(b̂), b̂_j = −quantile_{1−k/n}(s_{:,j} − α).

        Call after ``optimizer.step()``. Consumes the pending statistics and returns them (None,
        with the bias untouched, if no training forward ran since the last call).
        """
        stats, self._pending = self._pending, None
        if stats is None:
            return None
        k, E = self.cfg.top_k, self.cfg.num_experts
        bias = self.gate.e_score_correction_bias
        if k == E:
            bias.zero_()
        else:
            b_hat = -histogram_quantile(stats.margin_counts, stats.lo, stats.hi, 1.0 - k / E)
            bias.copy_(b_hat - b_hat.mean())
        return stats

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"hidden={self.hidden_size}, latent={c.latent_size}, experts={c.num_experts}, "
            f"top_k={c.top_k}, shared={c.num_shared_experts}, qb_bins={self.qb_num_bins}"
        )
