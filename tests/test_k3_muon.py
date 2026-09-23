"""K3/muon gates: Per-Head Muon, the K3 parameter partition, QK-Clip (spec §5).

1. ``head_rows`` None or == rows ≡ ``optim.Muon``, bitwise, step for step (same helpers, same
   order), momentum buffers included.
2. Per-block update ≡ ``_zeropower_via_newtonschulz5`` on each row block separately, concatenated,
   scaled by the block's own shape (bitwise); and it is NOT whole-matrix Muon for > 1 block.
3. Each block's recovered direction sits in the singular-value band of tests/test_optim.py:131-162
   and each block's update has AdamW-band RMS — the block-shape scale, not the matrix's.
4. ``head_rows`` must divide the row count; Muon takes only 2-D params.
5. The partition places every trainable parameter of a K3-named tiny model exactly once, with the
   per-head rows of KDA q/k/v and MLA q_b/kv_b, and leaves the frozen QB bias out; on the MLA
   stand-in and, once it constructs, on the real ``GatedMLA``.
6. QK-Clip: hand-set S = [50, 400] at τ = 100 scales exactly the rows the rule names (powers of
   two, so bitwise), the per-head nope and rope logits both scale by γ; bitwise ≡ a transcription
   of torchtitan's ``_clip_mla_weights``; a re-forward lands on τ; KDA untouched; consumed once.
7. Hybrid overfit smoke: the combined optimizer plus the post-step clip memorize one batch; a
   checkpoint round-trip of the combined optimizer resumes bitwise.

The MLA stand-in subclasses ``GatedMLA`` so ``muon.py``'s isinstance rules reach it without
duck typing; ``nn.Module.__init__`` bypasses the real constructor so these gates never depend on
another module's internals.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from scratch_llm.k3.config import KDAConfig, MLAConfig
from scratch_llm.k3.core.gated_mla import GatedMLA
from scratch_llm.k3.core.kda import KDALayer
from scratch_llm.k3.core.situ import DenseSiTUMLP, SiTUConfig, situ_glu
from scratch_llm.k3.muon import (
    PerHeadMuon,
    apply_k3_qk_clip,
    build_k3_optimizer,
    k3_param_groups,
)
from scratch_llm.model import cross_entropy
from scratch_llm.optim import (
    AdamW,
    CombinedOptimizer,
    Muon,
    _zeropower_via_newtonschulz5,
    gradient_clipping,
    muon_momentum_,
)

FP64_STRUCTURAL_REL = 1e-10  # same bar as test_k3_kda.py (two identical algebras)

_HIDDEN, _VOCAB = 32, 64
_KDA = KDAConfig(num_heads=2, head_dim=8, decay_rank=4, a_log_size=4)
_MLA = MLAConfig(
    num_heads=2,
    q_lora_rank=16,
    kv_lora_rank=8,
    qk_nope_head_dim=8,
    qk_rope_head_dim=4,
    v_head_dim=8,
)
_Q_ROWS, _KV_ROWS = 8 + 4, 8 + 8  # per-head rows of q_b_proj (nope | rope), kv_b_proj (k_nope | v)


def _rel(a: Tensor, b: Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


# ---------------------------------------------------------------------------
# A K3-named tiny model: checkpoint parameter names, simplest forward that uses them all.
# ---------------------------------------------------------------------------


class _MLAStandIn(GatedMLA):
    """GatedMLA's frozen attribute names with a naive causal forward (HF KimiMLAAttention layout:
    per head [q_nope | q_rope] and [k_nope | v]; one unrotated k_rope shared by all heads)."""

    def __init__(self, cfg: MLAConfig, hidden: int) -> None:
        nn.Module.__init__(self)  # skip GatedMLA.__init__: this double owns its parameters
        self.cfg = cfg
        h, v_rows = cfg.num_heads, cfg.num_heads * cfg.v_head_dim
        self.q_a_proj = nn.Linear(hidden, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(cfg.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, h * cfg.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(cfg.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank, h * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(v_rows, hidden, bias=False)
        self.g_proj = nn.Linear(hidden, v_rows, bias=False)
        self.track_max_logits = False
        self.last_max_logits: Tensor | None = None

    def logit_parts(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Scaled nope and rope logit terms [B, h, T, T] (unmasked) and v [B, h, T, v]."""
        B, T, _ = x.shape
        c = self.cfg
        h, nope, rope = c.num_heads, c.qk_nope_head_dim, c.qk_rope_head_dim
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(B, T, h, nope + rope)
        q_nope, q_rope = q.transpose(1, 2).split((nope, rope), dim=-1)
        latent, k_rope = self.kv_a_proj_with_mqa(x).split((c.kv_lora_rank, rope), dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(latent)).view(B, T, h, nope + c.v_head_dim)
        k_nope, v = kv.transpose(1, 2).split((nope, c.v_head_dim), dim=-1)
        scale = 1.0 / math.sqrt(nope + rope)
        nope_logits = scale * (q_nope @ k_nope.transpose(-2, -1))
        rope_logits = scale * (q_rope @ k_rope[:, None].transpose(-2, -1))  # one k_rope, all heads
        return nope_logits, rope_logits, v

    def forward(self, x: Tensor, state: object = None, layer_id: int = 0) -> Tensor:
        B, T, _ = x.shape
        nope_logits, rope_logits, v = self.logit_parts(x)
        future = torch.ones(T, T, dtype=torch.bool).triu(1)
        logits = (nope_logits + rope_logits).masked_fill(future, -math.inf)
        if self.track_max_logits:
            self.last_max_logits = logits.detach().amax(dim=(0, 2, 3))
        o = (logits.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(o * torch.sigmoid(self.g_proj(x)))


def _real_mla_or_skip() -> GatedMLA:
    try:
        return GatedMLA(_MLA, _HIDDEN)
    except NotImplementedError:
        pytest.skip("core/gated_mla.py GatedMLA is still a scaffold")


_MLA_FACTORIES: dict[str, Callable[[], GatedMLA]] = {
    "stand_in": lambda: _MLAStandIn(_MLA, _HIDDEN),
    "real": _real_mla_or_skip,
}


class _Expert(nn.Module):
    """``experts[e]``: SiTU-GLU at latent width (w1 gate, w3 up, w2 down)."""

    def __init__(self, dim: int, inter: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter, bias=False)
        self.w3 = nn.Linear(dim, inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(situ_glu(self.w1(x), self.w3(x), 4.0, 25.0))


class _Router(nn.Module):
    """``gate``: router weight [E, H] and the frozen Quantile-Balancing bias."""

    def __init__(self, experts: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(experts, hidden) / math.sqrt(hidden))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(experts), requires_grad=False)


class _LatentMoEStandIn(nn.Module):
    """LatentMoE's parameter names (spec §4) with dense soft routing: every expert, weighted by
    its sigmoid score. Enough for the partition and the smoke; not the routing math."""

    def __init__(self, hidden: int, latent: int, inter: int, experts: int) -> None:
        super().__init__()
        self.gate = _Router(experts, hidden)
        self.routed_expert_down_proj = nn.Linear(hidden, latent, bias=False)
        self.experts = nn.ModuleList(_Expert(latent, inter) for _ in range(experts))
        self.routed_expert_norm = nn.RMSNorm(latent)
        self.routed_expert_up_proj = nn.Linear(latent, hidden, bias=False)
        self.shared_experts = DenseSiTUMLP(hidden, 2 * inter, SiTUConfig())

    def forward(self, x: Tensor) -> Tensor:
        scores = torch.sigmoid(x @ self.gate.weight.T)  # [B, T, E]
        z = self.routed_expert_down_proj(x)
        routed = torch.stack([expert(z) for expert in self.experts], dim=-2)  # [B, T, E, latent]
        mixed = (scores[..., None] * routed).sum(dim=-2)
        return self.routed_expert_up_proj(self.routed_expert_norm(mixed)) + self.shared_experts(x)


def _mix(proj: nn.Linear, norm: nn.RMSNorm, sources: list[Tensor]) -> Tensor:
    """Spec §1 AttnRes mix over depth: Σ_i softmax_i(⟨w, RMSNorm(v_i)⟩)·v_i."""
    v = torch.stack(sources)  # [S, B, T, H]
    return (proj(norm(v)).softmax(dim=0) * v).sum(dim=0)


class _Layer(nn.Module):
    """One decoder layer with K3Block's names; AttnRes sources = {embedding, running prefix}."""

    def __init__(self, attn: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        self.self_attn = attn
        self.mlp = mlp
        self.input_layernorm = nn.RMSNorm(_HIDDEN)
        self.post_attention_layernorm = nn.RMSNorm(_HIDDEN)
        self.self_attention_res_norm = nn.RMSNorm(_HIDDEN)
        self.self_attention_res_proj = nn.Linear(_HIDDEN, 1, bias=False)
        self.mlp_res_norm = nn.RMSNorm(_HIDDEN)
        self.mlp_res_proj = nn.Linear(_HIDDEN, 1, bias=False)

    def forward(self, emb: Tensor, h: Tensor) -> Tensor:
        attn_in = _mix(self.self_attention_res_proj, self.self_attention_res_norm, [emb, h])
        h = h + self.self_attn(self.input_layernorm(attn_in))
        mlp_in = _mix(self.mlp_res_proj, self.mlp_res_norm, [emb, h])
        return h + self.mlp(self.post_attention_layernorm(mlp_in))


class _TinyK3(nn.Module):
    """Layer 1: KDA + dense SiTU MLP. Layer 2: Gated MLA + LatentMoE (K3's first_k_dense = 1)."""

    def __init__(self, mla: GatedMLA) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.layers = nn.ModuleList(
            [
                _Layer(KDALayer(_KDA, _HIDDEN), DenseSiTUMLP(_HIDDEN, 64, SiTUConfig())),
                _Layer(mla, _LatentMoEStandIn(_HIDDEN, latent=16, inter=16, experts=4)),
            ]
        )
        self.output_attn_res_norm = nn.RMSNorm(_HIDDEN)
        self.output_attn_res_proj = nn.Linear(_HIDDEN, 1, bias=False)
        self.norm = nn.RMSNorm(_HIDDEN)
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)

    def forward(self, ids: Tensor) -> Tensor:
        emb = h = self.embed_tokens(ids)
        for layer in self.layers:
            h = layer(emb, h)
        out = _mix(self.output_attn_res_proj, self.output_attn_res_norm, [emb, h])
        return self.lm_head(self.norm(out))


# ---------------------------------------------------------------------------
# 1-4. PerHeadMuon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("head_rows", ["none", "all_rows"])
def test_whole_matrix_block_is_optim_muon_bitwise(head_rows: str, nesterov: bool) -> None:
    torch.manual_seed(0)
    shapes = [(8, 24), (24, 8), (16, 16)]  # wide, tall (NS transposes), square
    ref = [nn.Parameter(torch.randn(s)) for s in shapes]
    ours = [nn.Parameter(p.detach().clone()) for p in ref]
    groups = [
        {"params": [p], "head_rows": None if head_rows == "none" else p.shape[0]} for p in ours
    ]
    ref_opt = Muon(ref, lr=0.02, nesterov=nesterov, weight_decay=0.1)
    opt = PerHeadMuon(groups, lr=0.02, nesterov=nesterov, weight_decay=0.1)
    for _ in range(3):
        for a, b in zip(ref, ours, strict=True):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        ref_opt.step()
        opt.step()
        for a, b in zip(ref, ours, strict=True):
            assert torch.equal(a, b)
            assert torch.equal(ref_opt.state[a]["momentum_buffer"], opt.state[b]["momentum_buffer"])


@pytest.mark.parametrize(("heads", "head_rows", "cols"), [(4, 6, 10), (3, 12, 5)])
def test_update_is_newton_schulz_per_block(heads: int, head_rows: int, cols: int) -> None:
    """Wide blocks, then tall ones (each block takes NS's transpose path)."""
    torch.manual_seed(0)
    lr, wd = 0.02, 0.1
    p = nn.Parameter(torch.randn(heads * head_rows, cols))
    p0, g = p.detach().clone(), torch.randn(heads * head_rows, cols)
    p.grad = g.clone()
    PerHeadMuon([{"params": [p], "head_rows": head_rows}], lr=lr, weight_decay=wd).step()

    g_eff = muon_momentum_(g, torch.zeros_like(g), momentum=0.95, nesterov=True)
    ortho = torch.cat([_zeropower_via_newtonschulz5(b) for b in g_eff.split(head_rows)])
    scale = 0.2 * math.sqrt(max(head_rows, cols))
    assert torch.equal(p.detach(), p0.mul(1 - lr * wd).add(ortho, alpha=-lr * scale))


def test_per_head_is_not_whole_matrix_muon() -> None:
    """The scaffold claimed per-head ≡ Muon on a head-uniform input. [2·8, 32] keeps the scale
    equal (max(8, 32) = max(16, 32)), so the gap is the per-block Newton–Schulz alone."""
    torch.manual_seed(0)
    whole = nn.Parameter(torch.randn(16, 32))
    per_head = nn.Parameter(whole.detach().clone())
    whole.grad = torch.randn(16, 32)
    per_head.grad = whole.grad.clone()
    Muon([whole], lr=0.02).step()
    PerHeadMuon([{"params": [per_head], "head_rows": 8}], lr=0.02).step()
    assert not torch.allclose(whole, per_head)


def test_per_block_spectrum_band_and_rms() -> None:
    """Tall [64, 32] blocks of a [256, 32] weight: block scale √64, matrix scale √256. With the
    matrix's scale the block RMS would be 0.4 and fail the band; with the block's it is ≈ 0.2."""
    torch.manual_seed(0)
    heads, h, C = 4, 64, 32
    p = nn.Parameter(torch.zeros(heads * h, C))
    p.grad = torch.randn(heads * h, C)
    PerHeadMuon([{"params": [p], "head_rows": h}], lr=1.0, weight_decay=0.0).step()
    update = p.detach()  # θ₀ = 0 and lr = 1: θ₁ = −scale·O
    scale = 0.2 * math.sqrt(max(h, C))
    for block in update.split(h):
        s = torch.linalg.svdvals(-block / scale)
        # Band of tests/test_optim.py:131-162 (same NS, same bf16 iteration).
        assert s.max() < 1.35
        q10, q90 = torch.quantile(s, 0.10).item(), torch.quantile(s, 0.90).item()
        assert q10 > 0.6 and q90 < 1.25, f"bulk σ ∈ [{q10:.3f}, {q90:.3f}]"
        rms = block.pow(2).mean().sqrt().item()
        assert 0.15 < rms < 0.28, f"block RMS {rms:.3f}"  # band of test_optim.py:165-175


def test_constructor_rejects_bad_head_rows_and_non_matrices() -> None:
    p = nn.Parameter(torch.randn(10, 4))
    for bad in (3, 0):
        with pytest.raises(ValueError, match="head_rows"):
            PerHeadMuon([{"params": [p], "head_rows": bad}], lr=0.02)
    with pytest.raises(ValueError, match="2-D"):
        PerHeadMuon([nn.Parameter(torch.randn(4))], lr=0.02)


# ---------------------------------------------------------------------------
# 5. Partition
# ---------------------------------------------------------------------------


def _routes(model: nn.Module) -> dict[str, int | str | None]:
    """name → head_rows (int or None) for Muon, "adamw", or "none" for no group."""
    groups, adamw = k3_param_groups(model)
    route: dict[int, int | str | None] = {id(p): "adamw" for p in adamw}
    for group in groups:
        route.update({id(p): group["head_rows"] for p in group["params"]})
    return {name: route.get(id(p), "none") for name, p in model.named_parameters()}


@pytest.mark.parametrize("mla_kind", sorted(_MLA_FACTORIES))
def test_partition_covers_every_trainable_param_once(mla_kind: str) -> None:
    torch.manual_seed(0)
    model = _TinyK3(_MLA_FACTORIES[mla_kind]())
    groups, adamw = k3_param_groups(model)
    routed = [p for group in groups for p in group["params"]] + adamw
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(routed) == len({id(p) for p in routed}) == len(trainable)
    assert [group["head_rows"] for group in groups] == [8, None, _Q_ROWS, _KV_ROWS]

    r = _routes(model)
    kda, mla, moe = "layers.0.self_attn.", "layers.1.self_attn.", "layers.1.mlp."
    expected: dict[str, int | str | None] = {
        **{f"{kda}{n}_proj.weight": _KDA.head_dim for n in "qkv"},
        **{f"{kda}{n}.weight": None for n in ("f_a_proj", "f_b_proj", "b_proj", "g_proj")},
        **{f"{kda}{n}.weight": "adamw" for n in ("q_conv1d", "k_conv1d", "v_conv1d", "o_norm")},
        f"{kda}o_proj.weight": None,
        f"{kda}A_log": "adamw",
        f"{kda}dt_bias": "adamw",
        f"{mla}q_b_proj.weight": _Q_ROWS,
        f"{mla}kv_b_proj.weight": _KV_ROWS,
        **{f"{mla}{n}.weight": None for n in ("q_a_proj", "kv_a_proj_with_mqa", "o_proj")},
        f"{mla}g_proj.weight": None,
        f"{mla}q_a_layernorm.weight": "adamw",
        f"{mla}kv_a_layernorm.weight": "adamw",
        f"{moe}gate.weight": None,  # router [E, H] on Muon (torchtitan K2.7)
        f"{moe}gate.e_score_correction_bias": "none",  # QB bias: assigned, never stepped
        f"{moe}experts.3.w2.weight": None,
        f"{moe}routed_expert_down_proj.weight": None,
        f"{moe}routed_expert_norm.weight": "adamw",
        f"{moe}shared_experts.up_proj.weight": None,
        "layers.0.mlp.down_proj.weight": None,
        "layers.0.self_attention_res_proj.weight": "adamw",  # AttnRes pseudo-query [1, H]
        "layers.1.mlp_res_proj.weight": "adamw",
        "layers.1.input_layernorm.weight": "adamw",
        "output_attn_res_proj.weight": "adamw",
        "embed_tokens.weight": "adamw",
        "lm_head.weight": "adamw",
        "norm.weight": "adamw",
    }
    assert {name: r[name] for name in expected} == expected
    assert [name for name, route in r.items() if route == "none"] == [
        f"{moe}gate.e_score_correction_bias"
    ]


def test_partition_refuses_a_parameter_no_rule_covers() -> None:
    stacked = nn.Module()
    stacked.w1 = nn.Parameter(torch.randn(4, 8, 8))  # a stacked-experts tensor: decide explicitly
    with pytest.raises(ValueError, match="no optimizer rule"):
        k3_param_groups(stacked)


# ---------------------------------------------------------------------------
# 6. QK-Clip
# ---------------------------------------------------------------------------


def _head_views(mla: GatedMLA) -> tuple[Tensor, Tensor]:
    """q_b_proj as [h, nope + rope, q_lora] and kv_b_proj as [h, nope + v, kv_lora], copied."""
    q, kv = mla.q_b_proj.weight.detach(), mla.kv_b_proj.weight.detach()
    h = _MLA.num_heads
    return q.view(h, _Q_ROWS, -1).clone(), kv.view(h, _KV_ROWS, -1).clone()


def test_qk_clip_scales_the_named_rows_and_both_logit_terms_by_gamma() -> None:
    """S = [50, 400], τ = 100: γ = [1, 1/4], γ^½ = 1/2. Powers of two scale floats exactly, so
    every check is bitwise."""
    torch.manual_seed(0)
    mla = _MLAStandIn(_MLA, _HIDDEN).double()
    x = torch.randn(2, 5, _HIDDEN, dtype=torch.float64)
    q0, kv0 = _head_views(mla)
    nope0, rope0, v0 = mla.logit_parts(x)

    mla.last_max_logits = torch.tensor([50.0, 400.0], dtype=torch.float64)
    assert apply_k3_qk_clip(mla, tau=100.0, alpha=0.5) == {"": 0.25}
    q1, kv1 = _head_views(mla)
    nope1, rope1, v1 = mla.logit_parts(x)

    assert torch.equal(q1[0], q0[0]) and torch.equal(kv1[0], kv0[0])  # head 0 under τ
    assert torch.equal(q1[1, :8], q0[1, :8] * 0.5)  # q_nope × γ^α
    assert torch.equal(q1[1, 8:], q0[1, 8:] * 0.25)  # q_rope × γ
    assert torch.equal(kv1[1, :8], kv0[1, :8] * 0.5)  # k_nope × γ^(1−α)
    assert torch.equal(kv1[1, 8:], kv0[1, 8:])  # v rows untouched
    assert torch.equal(nope1[:, 0], nope0[:, 0]) and torch.equal(rope1[:, 0], rope0[:, 0])
    assert torch.equal(nope1[:, 1], nope0[:, 1] * 0.25)  # nope logit × γ
    assert torch.equal(rope1[:, 1], rope0[:, 1] * 0.25)  # rope logit × γ
    assert torch.equal(v1, v0)
    assert mla.last_max_logits is None  # consumed


def _torchtitan_clip(q_b: Tensor, kv_b: Tensor, max_logits: Tensor, alpha: float) -> None:
    """torchtitan kimi_k2_7/qk_clip.py:80-133 and :169 (threshold 100), DTensor plumbing removed."""
    scales_H = 100.0 / max_logits.clamp_min(100.0)

    def _scale_mla_heads(
        weight: Tensor,
        rows_per_head: int,
        nope_rows_per_head: int,
        nope_scale_exponent: float,
        remaining_scale_exponent: float | None,
    ) -> None:
        num_heads = scales_H.numel()
        scales_H11 = scales_H.view(-1, 1, 1)
        heads_HDI = weight.view(num_heads, rows_per_head, weight.shape[1])
        heads_HDI[:, :nope_rows_per_head].mul_(scales_H11.pow(nope_scale_exponent))
        if remaining_scale_exponent is not None:
            heads_HDI[:, nope_rows_per_head:].mul_(scales_H11.pow(remaining_scale_exponent))

    nope = _MLA.qk_nope_head_dim
    _scale_mla_heads(q_b, _Q_ROWS, nope, alpha, 1.0)
    _scale_mla_heads(kv_b, _KV_ROWS, nope, 1.0 - alpha, None)


@pytest.mark.parametrize("alpha", [0.5, 0.3])
def test_qk_clip_matches_torchtitan_transcription(alpha: float) -> None:
    torch.manual_seed(1)
    mla = _MLAStandIn(_MLA, _HIDDEN)
    q_b, kv_b = mla.q_b_proj.weight.detach().clone(), mla.kv_b_proj.weight.detach().clone()
    s_max = torch.tensor([37.5, 173.2])
    _torchtitan_clip(q_b, kv_b, s_max, alpha)
    mla.last_max_logits = s_max
    apply_k3_qk_clip(mla, tau=100.0, alpha=alpha)
    assert torch.equal(mla.q_b_proj.weight, q_b) and torch.equal(mla.kv_b_proj.weight, kv_b)


@pytest.mark.parametrize("mla_kind", sorted(_MLA_FACTORIES))
def test_qk_clip_reforward_lands_on_tau(mla_kind: str) -> None:
    """τ between the two heads' measured maxima: the clipped head's re-forward max is τ, the
    other head's is bitwise unchanged (its rows were multiplied by exactly 1.0)."""
    torch.manual_seed(2)
    mla = _MLA_FACTORIES[mla_kind]().double()
    mla.track_max_logits = True
    x = torch.randn(2, 7, _HIDDEN, dtype=torch.float64)
    mla(x)
    assert mla.last_max_logits is not None
    s0 = mla.last_max_logits.clone()
    tau = float(s0.min() + s0.max()) / 2
    clipped = s0 > tau
    assert clipped.any() and not clipped.all()

    apply_k3_qk_clip(mla, tau=tau)
    mla(x)
    assert mla.last_max_logits is not None
    s1 = mla.last_max_logits
    assert _rel(s1[clipped], torch.full_like(s1[clipped], tau)) < FP64_STRUCTURAL_REL
    assert torch.equal(s1[~clipped], s0[~clipped])


def test_qk_clip_scope_consumption_and_guards() -> None:
    """Only MLA is clipped (KDA's L2-normed q, k bound its logits); an observation clips once;
    a non-finite maximum refuses to write into the weights."""
    torch.manual_seed(3)
    mla = _MLAStandIn(_MLA, _HIDDEN)
    model = _TinyK3(mla)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    assert apply_k3_qk_clip(model) == {}  # nothing observed yet: a no-op
    mla.last_max_logits = torch.tensor([400.0, 20.0])
    assert apply_k3_qk_clip(model) == {"layers.1.self_attn": 0.25}
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed == {"layers.1.self_attn.q_b_proj.weight", "layers.1.self_attn.kv_b_proj.weight"}
    assert apply_k3_qk_clip(model) == {}  # consumed: the stale maximum never clips twice

    mla.last_max_logits = torch.tensor([math.nan, 20.0])
    with pytest.raises(ValueError, match="non-finite"):
        apply_k3_qk_clip(model)


# ---------------------------------------------------------------------------
# 7. The combined optimizer on the hybrid
# ---------------------------------------------------------------------------


def test_build_k3_optimizer_wiring_and_bitwise_resume() -> None:
    torch.manual_seed(4)
    model = _TinyK3(_MLAStandIn(_MLA, _HIDDEN))
    ids = torch.randint(0, _VOCAB, (2, 9))

    def train_step(m: nn.Module, opt: CombinedOptimizer) -> None:
        opt.zero_grad()
        cross_entropy(m(ids[:, :-1]), ids[:, 1:]).backward()
        opt.step()

    opt = build_k3_optimizer(model)
    muon, adamw = opt.optimizers
    assert isinstance(muon, PerHeadMuon) and isinstance(adamw, AdamW)
    assert {g["lr"] for g in muon.param_groups} == {2e-2} and adamw.param_groups[0]["lr"] == 3e-3
    assert {g["weight_decay"] for g in opt.param_groups} == {0.1}

    train_step(model, opt)
    resumed = copy.deepcopy(model)
    resumed_opt = build_k3_optimizer(resumed)
    resumed_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    train_step(model, opt)
    train_step(resumed, resumed_opt)
    for (name, p), q in zip(model.named_parameters(), resumed.parameters(), strict=True):
        assert torch.equal(p, q), name


def test_hybrid_overfit_one_batch_with_qk_clip() -> None:
    """Mirror of tests/test_optim.py:215 on the hybrid: PerHeadMuon + AdamW, with the post-step
    clip every step, memorize one fixed batch. Unclipped, this MLA layer's max logit climbs from
    1.5 to ~12 by step 100 (measured), so τ = 5 makes the clip bind on most steps."""
    tau = 5.0
    torch.manual_seed(0)
    mla = _MLAStandIn(_MLA, _HIDDEN)
    mla.track_max_logits = True
    model = _TinyK3(mla)
    opt = build_k3_optimizer(model, weight_decay=0.0)
    ids = torch.randint(0, _VOCAB, (4, 17))
    inputs, targets = ids[:, :-1], ids[:, 1:]

    initial = cross_entropy(model(inputs), targets).item()
    final, clipped_steps = initial, 0
    for _ in range(120):  # loss ≈ 0.01 by step 100 (measured), ~3 s on CPU
        opt.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        opt.step()
        clipped_steps += apply_k3_qk_clip(model, tau=tau)["layers.1.self_attn"] < 1.0
        final = loss.item()

    assert clipped_steps > 0, "the clip never engaged; the smoke does not exercise it"
    assert final < 0.05, f"hybrid failed to overfit: final {final:.4f} (started {initial:.3f})"
