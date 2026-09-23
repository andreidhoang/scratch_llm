"""K3/LatentMoE + Quantile Balancing gates (spec §4), against in-file transcriptions of Moonshot's
HF reference (``modeling_kimi_linear.py``: ``KimiMoEGate`` :666-759, ``KimiSparseMoeBlock``
:762-838 + ``moe_infer`` :840-874, ``KimiBlockSparseMLP``, ``KimiMLP``, ``KimiRMSNorm``,
``SituAndMul``).

1. Routing ≡ HF gate (fp64, tie-free): same selected set, weights to 1e-12.
2. The bias moves the selection, never the weights: they stay renormalized raw s.
3. Forward ≡ HF sparse block + moe_infer (fp64, 1e-10), our state_dict loaded strictly (names).
4. Histogram quantile within one bin of torch.quantile; margins inside the derived range; sharded
   stats sum to the global stats (the one-all-reduce claim).
5. QB drives four skewed routers' per-expert load to k/n inside the sampling-noise band; no-QB
   stays skewed.
6. One-step delay: batch t is routed with the bias derived from batch t-1, never from itself.
7. Eval / no_grad / inference_mode forwards record nothing and never touch the bias.
8. |SiTU output| ≤ 100 inside routed and shared experts on extreme inputs.
9. Degenerate E = 1 (k = 1) and k = E: match HF; QB has nothing to do and leaves b = 0.
10. Finite grads everywhere; the router learns through the mixture weights; the bias gets none.
11. Parameter count == param_count._moe_block_params (mini + full on meta), checkpoint names,
    latent-norm eps from the config; router kaiming_uniform_(a=√5) and zero bias at init.
12. Router logits stay fp32 under bf16 autocast.

The transcriptions keep HF's algebra and op order. HF upcasts with ``.float()`` /
``.type(torch.float32)``; here those promote instead (``_up``), so fp64 inputs stay fp64 — the
one systematic edit. HF's gate asserts ``not self.training`` and its block raises in training;
both are dropped (the reference runs in eval anyway).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_llm.k3.config import K3Config, MoEConfig, k3_full, mini_k3_d12
from scratch_llm.k3.core.latent_moe import (
    QB_NUM_BINS,
    LatentExpert,
    LatentMoE,
    histogram_quantile,
    margin_histogram,
    margin_range,
)
from scratch_llm.k3.param_count import _moe_block_params

FP64_STRUCTURAL_REL = 1e-10  # same bar as test_kda_parity_fla.py (two identical algebras)
# Gate weights: HF's exact ops in fp64, except the order of the k-term renormalizing sum
# (topk sorted=False vs our sorted order): <= k * 2^-53 ≈ 2e-15 relative for k <= 16.
GATE_WEIGHT_REL = 1e-12
B1, B2 = 4.0, 25.0


def _rel(a: Tensor, b: Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _model_cfg(moe: MoEConfig, hidden: int) -> K3Config:
    # eps off the presets' 1e-5 and KimiRMSNorm's 1e-6 default, so a hard-coded eps cannot match.
    return replace(mini_k3_d12(), hidden_size=hidden, moe=moe, rms_norm_eps=2e-5)


def _moe(
    E: int,
    k: int,
    *,
    hidden: int = 16,
    latent: int = 8,
    inter: int = 12,
    renormalize: bool = True,
    scale: float = 1.0,
    bias_std: float = 0.05,
    seed: int = 0,
) -> LatentMoE:
    """fp64 block moved off-init: random selection bias, non-unit latent-norm gain."""
    torch.manual_seed(seed)
    cfg = MoEConfig(
        num_experts=E,
        top_k=k,
        num_shared_experts=2,
        expert_intermediate=inter,
        latent_size=latent,
        dense_intermediate=4 * hidden,
        renormalize=renormalize,
        routed_scaling_factor=scale,
    )
    moe = LatentMoE(cfg, hidden, _model_cfg(cfg, hidden)).double()
    with torch.no_grad():
        moe.gate.e_score_correction_bias.normal_(0.0, bias_std)
        moe.routed_expert_norm.weight.uniform_(0.5, 1.5)
    return moe


# ---------------------------------------------------------------------------
# HF transcription (Kimi-K3 release, modeling_kimi_linear.py).
# ---------------------------------------------------------------------------


def _up(x: Tensor) -> Tensor:
    """HF's ``.float()``, promoting instead of truncating: fp64 stays fp64."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def _hf_situ_and_mul(x: Tensor) -> Tensor:
    d = x.shape[-1] // 2
    gate, up = _up(x[..., :d]), _up(x[..., d:])
    situ_a = B1 * torch.tanh(gate / B1) * torch.sigmoid(gate)
    up = B2 * torch.tanh(up / B2)
    return (situ_a * up).to(x.dtype)


class _HFBlockSparseMLP(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden, inter, bias=False)  # gate
        self.w2 = nn.Linear(inter, hidden, bias=False)  # down
        self.w3 = nn.Linear(hidden, inter, bias=False)  # up

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate_up = torch.cat([self.w1(hidden_states), self.w3(hidden_states)], dim=-1)
        return self.w2(_hf_situ_and_mul(gate_up))


class _HFMLP(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(_hf_situ_and_mul(gate_up))


class _HFRMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        x = _up(hidden_states)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class _HFGate(nn.Module):
    def __init__(self, E: int, hidden: int, k: int, renormalize: bool, scale: float) -> None:
        super().__init__()
        self.top_k, self.moe_renormalize, self.routed_scaling_factor = k, renormalize, scale
        self.weight = nn.Parameter(torch.empty(E, hidden))
        self.e_score_correction_bias = nn.Parameter(torch.empty(E))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(_up(hidden_states), _up(self.weight), None)
        scores = logits.sigmoid()
        scores = scores.view(bsz * seq_len, -1)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        tmp_scores = scores_for_choice  # num_expert_group == 1 (config.json)
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class _HFSparseMoeBlock(nn.Module):
    """KimiSparseMoeBlock with use_latent_moe and latent_moe_use_norm (both on in config.json).
    As in HF, every size and the latent norm's eps come from the model config (HF :811-813,
    ``eps=config.rms_norm_eps``), never from the module under test."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        c, H = config.moe, config.hidden_size
        self.experts = nn.ModuleList(
            _HFBlockSparseMLP(c.latent_size, c.expert_intermediate) for _ in range(c.num_experts)
        )
        self.gate = _HFGate(c.num_experts, H, c.top_k, c.renormalize, c.routed_scaling_factor)
        self.shared_experts = _HFMLP(H, c.expert_intermediate * c.num_shared_experts)
        self.routed_expert_down_proj = nn.Linear(H, c.latent_size, bias=False)
        self.routed_expert_up_proj = nn.Linear(c.latent_size, H, bias=False)
        self.routed_expert_norm = _HFRMSNorm(c.latent_size, config.rms_norm_eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.routed_expert_down_proj(hidden_states)
        y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        y = self.routed_expert_norm(y)
        y = self.routed_expert_up_proj(y)
        y = y.view(*orig_shape)
        return y + self.shared_experts(identity)

    @torch.no_grad()
    def moe_infer(self, x: Tensor, topk_ids: Tensor, topk_weight: Tensor) -> Tensor:
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert.tolist()):  # HF: .cpu().numpy()
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            outputs.append(expert(sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )


def _hf_reference(moe: LatentMoE) -> _HFSparseMoeBlock:
    """The HF block built from the fixture's own model config, holding our weights."""
    ref = _HFSparseMoeBlock(_model_cfg(moe.cfg, moe.hidden_size)).double()
    ref.load_state_dict(moe.state_dict(), strict=True)  # every checkpoint name, both directions
    return ref.eval()


def _by_expert(idx: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
    """Sort each row's (expert, weight) pairs by expert id, so slot order stops mattering."""
    sorted_idx, perm = idx.sort(dim=-1)
    return sorted_idx, weight.gather(1, perm)


# ---------------------------------------------------------------------------
# 1-3. Routing and forward vs the reference.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("E", "k", "renormalize", "scale"),
    [(16, 4, True, 1.0), (16, 4, True, 2.5), (16, 4, False, 1.0), (8, 1, True, 1.0)],
)
def test_routing_matches_hf_gate_fp64(E: int, k: int, renormalize: bool, scale: float) -> None:
    moe = _moe(E, k, renormalize=renormalize, scale=scale)
    ref = _hf_reference(moe)
    x = torch.randn(3, 7, moe.hidden_size, dtype=torch.float64)
    ours = moe.gate(x.view(-1, moe.hidden_size))
    ref_idx, ref_w = ref.gate(x)
    idx, w = _by_expert(ours.topk_idx, ours.topk_weight)
    ref_idx, ref_w = _by_expert(ref_idx, ref_w)
    assert torch.equal(idx, ref_idx)
    torch.testing.assert_close(w, ref_w, rtol=GATE_WEIGHT_REL, atol=0)
    assert ours.topk_weight.dtype == torch.float64 and ours.scores.dtype == torch.float64
    # QB's cutoff α (spec §4; not in HF, which never trains): the (k+1)-th largest s + b.
    s = torch.sigmoid(x.view(-1, moe.hidden_size) @ moe.gate.weight.T)
    alpha = (s + moe.gate.e_score_correction_bias).sort(-1, descending=True).values[:, k]
    assert ours.cutoff is not None
    torch.testing.assert_close(ours.cutoff, alpha, rtol=GATE_WEIGHT_REL, atol=0)


def test_bias_moves_selection_never_weights() -> None:
    moe = _moe(16, 4, bias_std=0.0)
    x = torch.randn(256, moe.hidden_size, dtype=torch.float64)
    before = moe.gate(x)
    with torch.no_grad():
        moe.gate.e_score_correction_bias[3] = 0.15  # admits expert 3 for some tokens only
    after = moe.gate(x)

    s = torch.sigmoid(x @ moe.gate.weight.T)  # raw scores, computed independently
    idx, w = _by_expert(after.topk_idx, after.topk_weight)
    expected = s.gather(1, idx) / s.gather(1, idx).sum(-1, keepdim=True)
    torch.testing.assert_close(w, expected, rtol=GATE_WEIGHT_REL, atol=0)

    idx0, w0 = _by_expert(before.topk_idx, before.topk_weight)
    same = (idx == idx0).all(-1)
    admitted = (idx == 3).any(-1) & ~(idx0 == 3).any(-1)
    assert same.any() and admitted.any()  # the bias moved some tokens and left others alone
    torch.testing.assert_close(w[same], w0[same], rtol=GATE_WEIGHT_REL, atol=0)
    biased = s.gather(1, idx) + (idx == 3) * 0.15
    wrong = biased / biased.sum(-1, keepdim=True)  # what a bias inside the weights would give
    assert (w[admitted] - wrong[admitted]).abs().min() > 1e-3


@pytest.mark.parametrize(("E", "k", "scale"), [(8, 2, 1.0), (16, 4, 2.5)])
def test_forward_matches_hf_sparse_moe_block_fp64(E: int, k: int, scale: float) -> None:
    moe = _moe(E, k, scale=scale)
    ref = _hf_reference(moe)
    x = torch.randn(2, 9, moe.hidden_size, dtype=torch.float64)
    moe.eval()
    out = moe(x)
    assert out.shape == x.shape
    assert _rel(out, ref(x)) < FP64_STRUCTURAL_REL
    moe.train()
    assert torch.equal(moe(x), out)  # training only adds the (no-grad) statistics


# ---------------------------------------------------------------------------
# 4. Histogram estimator.
# ---------------------------------------------------------------------------


def _margins(moe: LatentMoE, x: Tensor) -> Tensor:
    route = moe.gate(x)
    assert route.cutoff is not None
    return (route.scores - route.cutoff[:, None]).detach()


def test_histogram_quantile_within_one_bin_of_torch_quantile() -> None:
    E, k, bins = 8, 2, QB_NUM_BINS
    moe = _moe(E, k, bias_std=0.1)
    margins = _margins(moe, torch.randn(8192, moe.hidden_size, dtype=torch.float64))
    lo, hi = margin_range(moe.gate.e_score_correction_bias)
    width = (hi - lo).item() / bins
    level = 1 - k / E
    est = histogram_quantile(margin_histogram(margins, lo, hi, bins), lo, hi, level)
    exact = torch.quantile(margins, level, dim=0)
    # Both read the same order statistics up to one rank (target rank level·m vs torch's
    # (m − 1)·level), so |est − exact| <= one bin + the gap between the neighbouring order
    # statistics; at 8192 margins that gap is a small fraction of a bin.
    ranked = margins.sort(0).values
    r = math.ceil(level * margins.shape[0])
    gap = ranked[r] - ranked[r - 2]
    assert (gap < 0.1 * width).all()
    assert ((est - exact).abs() <= width + gap).all()


def test_histogram_quantile_interpolates_inside_the_bin() -> None:
    """Margins spread evenly inside every bin (c points at e_b + (i + ½)·w/c): the estimate and
    torch.quantile read ranks at most one apart, so they differ by at most two in-bin spacings,
    2w/c. A bin-midpoint rule would be off by up to w/2."""
    lo = torch.tensor(-1.0, dtype=torch.float64)
    hi = torch.tensor(1.0, dtype=torch.float64)
    bins, E, level = 64, 8, 1 - 2 / 8
    w = 2.0 / bins
    per_bin = torch.randint(10, 30, (E, bins), generator=torch.Generator().manual_seed(0))
    cols = [
        torch.tensor(
            [-1.0 + (b + (i + 0.5) / c) * w for b, c in enumerate(row) for i in range(c)],
            dtype=torch.float64,
        )
        for row in per_bin.tolist()
    ]
    hist = torch.cat([margin_histogram(col[:, None], lo, hi, bins) for col in cols])
    assert torch.equal(hist, per_bin)  # every margin landed in the bin it was placed in
    est = histogram_quantile(hist, lo, hi, level)
    exact = torch.stack([torch.quantile(col, level) for col in cols])
    assert ((est - exact).abs() <= 2 * w / per_bin.min()).all()


def test_histogram_quantile_separates_the_top_share_exactly() -> None:
    """Report Fig. 5 shape (m = 8 tokens, n = 4, k = 1): margins a full bin apart, so the
    estimated threshold leaves exactly m·k/n = 2 margins above it in every column."""
    lo, hi, bins = (
        torch.tensor(-1.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        64,
    )
    gen = torch.Generator().manual_seed(0)
    grid = (torch.arange(-28, 28, 7, dtype=torch.float64) + 0.5) / 32  # 8 values, 7 bins apart
    margins = torch.stack([grid[torch.randperm(8, generator=gen)] for _ in range(4)], dim=1)
    t = histogram_quantile(margin_histogram(margins, lo, hi, bins), lo, hi, 1 - 1 / 4)
    assert torch.equal((margins > t).sum(0), torch.full((4,), 2))


def test_margins_stay_inside_the_derived_range() -> None:
    moe = _moe(8, 2, bias_std=0.3).float()
    route = moe.gate(torch.randn(512, moe.hidden_size) * 1e3)
    assert route.cutoff is not None
    assert (route.scores == 0).any() and (route.scores == 1).any()  # fp32 sigmoid saturated
    margins = route.scores - route.cutoff[:, None]
    lo, hi = margin_range(moe.gate.e_score_correction_bias)
    assert lo.item() <= margins.min().item() and margins.max().item() <= hi.item()


def test_sharded_stats_sum_to_the_global_stats() -> None:
    moe = _moe(8, 2).train()
    whole = copy.deepcopy(moe)
    xa, xb = torch.randn(2, 64, moe.hidden_size, dtype=torch.float64)
    moe(xa)
    moe(xb)  # a second shard (or micro-batch) before the update: counts add
    whole(torch.cat([xa, xb]))
    parts, glob = moe.update_bias(), whole.update_bias()
    assert parts is not None and glob is not None
    assert torch.equal(parts.margin_counts, glob.margin_counts)
    assert torch.equal(parts.load, glob.load) and int(parts.num_tokens) == 128
    assert torch.equal(moe.gate.e_score_correction_bias, whole.gate.e_score_correction_bias)
    with pytest.raises(ValueError, match="different biases"):
        _ = parts + replace(parts, lo=parts.lo - 1)


# ---------------------------------------------------------------------------
# 5-7. Balancing dynamics, delay, freezing.
# ---------------------------------------------------------------------------


def _skewed(E: int = 16, k: int = 2, hidden: int = 32, seed: int = 1) -> tuple[LatentMoE, Tensor]:
    """fp32 block whose router strongly prefers some experts: inputs share a mean μ, so expert
    j's logit carries a persistent offset W_j·μ ~ N(0, 1). Returns the block and μ."""
    moe = _moe(E, k, hidden=hidden, latent=8, inter=8, bias_std=0.0, seed=seed).float()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        moe.gate.weight.copy_(torch.randn(E, hidden, generator=gen) / math.sqrt(hidden))
    return moe, torch.randn(hidden, generator=gen)


def _load_under(moe: LatentMoE, x: Tensor, bias: Tensor) -> Tensor:
    """Per-expert token counts for x routed by a copy of the gate holding ``bias``."""
    gate = copy.deepcopy(moe.gate)
    with torch.no_grad():
        gate.e_score_correction_bias.copy_(bias)
    idx = gate(x.reshape(-1, moe.hidden_size)).topk_idx
    return torch.bincount(idx.flatten(), minlength=moe.cfg.num_experts)


@pytest.mark.parametrize("router_seed", [1, 2, 3, 4])
def test_quantile_balancing_drives_load_to_k_over_n(router_seed: int) -> None:
    E, k, m, rounds = 16, 2, 16384, 20
    moe, mu = _skewed(E, k, seed=router_seed)
    moe.train()
    zero = torch.zeros(E)
    p = k / E
    # Band, from sampling noise alone. Batch t's load is binomial around the true load of the
    # bias in force: sd σ = sqrt(p(1 − p)/m). That bias was fitted on batch t−1, whose empirical
    # quantile misses its target coverage by another independent σ ⇒ sd σ√2; 4 sd per expert
    # keeps all 16 inside with probability ≈ 0.999. Nothing is added for the histogram: at
    # QB_NUM_BINS its interpolation error sits below this noise, so the band pins the resolution
    # too (measured on these four routers: 256 bins breach it on three, 128 bins on all four).
    band = 4 * math.sqrt(2) * math.sqrt(p * (1 - p) / m)
    torch.manual_seed(2)
    for t in range(rounds):
        x = torch.randn(16, m // 16, moe.hidden_size) + mu
        moe(x)
        stats = moe.update_bias()
        assert stats is not None
        dev = (stats.load_fraction - p).abs().max().item()
        no_qb = (_load_under(moe, x, zero) / m - p).abs().max().item()  # same router, b = 0
        if t == 0:
            assert dev == no_qb and no_qb > 2 * p  # step 0 routes with b = 0: strongly skewed
        elif t >= rounds - 5:
            assert dev <= band, f"round {t}: max load deviation {dev:.4f} > band {band:.4f}"
            assert no_qb > 5 * band  # without QB the skew never goes away


def test_batch_t_is_routed_with_the_bias_from_batch_t_minus_1() -> None:
    moe, mu = _skewed()
    moe.train()
    k, E = moe.cfg.top_k, moe.cfg.num_experts
    x0, x1 = (torch.randn(4096, moe.hidden_size) + mu for _ in range(2))
    moe(x0)
    moe.update_bias()
    b_prev = moe.gate.e_score_correction_bias.detach().clone()  # derived from batch 0
    frozen = copy.deepcopy(moe).eval()  # routes with b_prev and records nothing

    out = moe(x1)
    assert torch.equal(moe.gate.e_score_correction_bias, b_prev)  # forward never writes b
    stats = moe.update_bias()
    assert stats is not None
    b_new = moe.gate.e_score_correction_bias.detach().clone()  # derived from batch 1

    assert torch.equal(out, frozen(x1))  # batch 1 went through the experts under b_prev
    assert torch.equal(stats.load, _load_under(moe, x1, b_prev))
    assert not torch.equal(stats.load, _load_under(moe, x1, b_new))  # b_new routes differently

    route = frozen.gate(x1)  # b_new is QB of batch 1's margins under b_prev, nothing else
    assert route.cutoff is not None
    lo, hi = margin_range(b_prev)
    counts = margin_histogram(route.scores - route.cutoff[:, None], lo, hi, moe.qb_num_bins)
    b_hat = -histogram_quantile(counts, lo, hi, 1 - k / E)
    assert torch.equal(b_new, (b_hat - b_hat.mean()).float())


def test_eval_and_no_grad_forwards_never_record_or_move_the_bias() -> None:
    moe = _moe(8, 2)
    b0 = moe.gate.e_score_correction_bias.detach().clone()
    x = torch.randn(2, 5, moe.hidden_size, dtype=torch.float64)
    moe.eval()(x)
    moe.train()
    with torch.no_grad():
        moe(x)
    with torch.inference_mode():
        moe(x)
    assert moe.update_bias() is None
    assert torch.equal(moe.gate.e_score_correction_bias, b0)
    moe(x)  # positive control: a training forward with autograd records, and QB moves b
    assert moe.update_bias() is not None
    assert not torch.equal(moe.gate.e_score_correction_bias, b0)


# ---------------------------------------------------------------------------
# 8-12. Bounds, degenerate shapes, gradients, accounting, precision.
# ---------------------------------------------------------------------------


def test_situ_output_bounded_inside_experts_on_extreme_inputs() -> None:
    moe = _moe(8, 2).float()
    peaks: list[float] = []

    def grab(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        peaks.append(args[0].abs().max().item())  # input of the down projection = SiTU output

    experts = [m for m in moe.modules() if isinstance(m, LatentExpert)]
    handles = [e.w2.register_forward_pre_hook(grab) for e in experts]
    handles.append(moe.shared_experts.down_proj.register_forward_pre_hook(grab))
    out = moe(torch.randn(4, 64, moe.hidden_size) * 1e4)
    for h in handles:
        h.remove()
    assert torch.isfinite(out).all() and len(peaks) >= 2
    assert max(peaks) <= B1 * B2
    assert max(peaks) > 0.9 * B1 * B2  # the inputs really did drive the cap into saturation


@pytest.mark.parametrize(("E", "k"), [(1, 1), (4, 4)])
def test_degenerate_single_expert_and_every_expert(E: int, k: int) -> None:
    moe = _moe(E, k)
    ref = _hf_reference(moe)
    x = torch.randn(2, 6, moe.hidden_size, dtype=torch.float64)
    route = moe.gate(x.view(-1, moe.hidden_size))
    assert route.cutoff is None  # no (k+1)-th score exists
    if k == 1:  # HF renormalizes only when k > 1, so the lone weight is s itself
        assert torch.equal(route.topk_weight, route.scores)
    out = moe.train()(x)
    assert _rel(out, ref(x)) < FP64_STRUCTURAL_REL
    stats = moe.update_bias()
    assert stats is not None
    assert torch.equal(stats.load, torch.full((E,), 12)) and int(stats.margin_counts.sum()) == 0
    assert torch.equal(moe.gate.e_score_correction_bias, torch.zeros(E, dtype=torch.float64))


def test_gradients_finite_router_learns_bias_frozen() -> None:
    moe = _moe(8, 2).train()
    x = torch.randn(4, 32, moe.hidden_size, dtype=torch.float64, requires_grad=True)
    (moe(x) * torch.randn(4, 32, moe.hidden_size, dtype=torch.float64)).sum().backward()
    bias = moe.gate.e_score_correction_bias
    assert not bias.requires_grad and bias.grad is None
    for name, param in moe.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
    router_grad = moe.gate.weight.grad
    assert router_grad is not None and router_grad.abs().sum() > 0

    # Selection is piecewise constant, so the router's gradient can only come through the
    # mixture weights; gradcheck confirms it (and the dispatch's) against finite differences.
    small = _moe(4, 2, hidden=6, latent=4, inter=4).eval()
    xs = torch.randn(1, 3, 6, dtype=torch.float64, requires_grad=True)
    w = small.gate.weight.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b: torch.func.functional_call(small, {"gate.weight": b}, (a,)), (xs, w)
    )


@pytest.mark.parametrize("make", [mini_k3_d12, k3_full])
def test_parameter_count_and_checkpoint_names_on_meta(make: Callable[[], K3Config]) -> None:
    cfg = make()
    with torch.device("meta"):
        moe = LatentMoE(cfg.moe, cfg.hidden_size, cfg)
    params = dict(moe.named_parameters())
    assert sum(p.numel() for p in params.values()) == _moe_block_params(cfg)

    H, E, L = cfg.hidden_size, cfg.moe.num_experts, cfg.moe.latent_size
    inter, S = cfg.moe.expert_intermediate, cfg.moe.shared_intermediate
    expected = {
        "gate.weight": (E, H),
        "gate.e_score_correction_bias": (E,),
        "routed_expert_down_proj.weight": (L, H),
        "routed_expert_norm.weight": (L,),
        "routed_expert_up_proj.weight": (H, L),
        "shared_experts.gate_proj.weight": (S, H),
        "shared_experts.up_proj.weight": (S, H),
        "shared_experts.down_proj.weight": (H, S),
    }
    for e in range(E):
        expected |= {
            f"experts.{e}.w1.weight": (inter, L),
            f"experts.{e}.w2.weight": (L, inter),
            f"experts.{e}.w3.weight": (inter, L),
        }
    assert {n: tuple(p.shape) for n, p in params.items()} == expected
    bias = params["gate.e_score_correction_bias"]
    assert bias.dtype == torch.float32 and not bias.requires_grad
    assert moe.routed_expert_norm.eps == cfg.rms_norm_eps  # HF :811-813, not KimiRMSNorm's 1e-6


def test_router_kaiming_init_and_zero_bias_on_mini() -> None:
    """HF ``reset_parameters``: kaiming_uniform_(a=√5) on [E, H] has gain sqrt(2/(1 + a²)) = 1/√3
    and bound √3 · gain/√H = 1/√H, so W ~ U(−1/√H, 1/√H) with std 1/√(3H). The bias starts at 0."""
    cfg = mini_k3_d12()
    torch.manual_seed(0)
    moe = LatentMoE(cfg.moe, cfg.hidden_size, cfg)
    bias = moe.gate.e_score_correction_bias
    assert torch.equal(bias, torch.zeros(cfg.moe.num_experts, dtype=torch.float32))
    w, H = moe.gate.weight.detach(), cfg.hidden_size
    assert w.abs().max().item() <= 1.0 / math.sqrt(H)
    # Sample std of n iid uniforms: relative sd sqrt((κ − 1)/(4n)), kurtosis κ = 9/5, i.e.
    # sqrt(0.2/n) ≈ 1.7e-3 at n = 64 · 1024; allow 5 of those.
    rel = 5 * math.sqrt(0.2 / w.numel())
    assert abs(w.std().item() * math.sqrt(3 * H) - 1.0) <= rel


def test_router_stays_fp32_under_bf16_autocast() -> None:
    moe = _moe(16, 4).float().train()
    x = torch.randn(2, 8, moe.hidden_size)
    plain = moe.gate(x.view(-1, moe.hidden_size))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = moe.gate(x.view(-1, moe.hidden_size))
        out = moe(x)
    assert cast.scores.dtype == torch.float32
    assert torch.equal(cast.topk_idx, plain.topk_idx)
    assert torch.equal(cast.topk_weight, plain.topk_weight)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
