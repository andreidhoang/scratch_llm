"""K3/model gates (spec §6): the assembled model against an HF transcription, and its whole-model
contracts.

1. ≡ HF ``KimiLinearForCausalLM`` transcribed below (``KimiLinearModel.forward`` :1188-1219,
   ``KimiDecoderLayer._forward_attn_residual`` :973-1046, ``_apply_attn_res`` :1075-1088,
   ``_apply_output_attn_res`` :1226-1233, ``norm``, ``lm_head``), loaded from OUR state dict with
   HF's ``model.`` prefix added, strict in both directions (so a checkpoint loads by prefix
   strip): logits and the gradient of every parameter, fp64 at the structural bar. The MLA, MoE,
   MLP and norm legs are HF transcriptions (copied from test_k3_gated_mla.py and
   test_k3_latent_moe.py). HF's ``KimiDeltaAttention`` calls FLA's Triton kernels, so its leg
   transcribes the layer and replaces each kernel by the reference semantics FLA documents for
   it, with the recurrent delta rule of ``naive_recurrent_kda``. It shares no code with
   core/kda.py, but it is a re-derivation from FLA's conventions, not FLA-code parity: that is
   tests/test_kda_parity_fla.py (the operator only).
2. The two plausible AttnRes misreads planted in the transcription — FLA's sublayer boundary
   (2ℓ) % B, and a model that skips the output mix — move the logits far above the bar; the
   output pseudo-query gets a gradient and the layer-0 attention query none.
3. Prefill ≡ token-by-token decode ≡ split prefill from ``HybridState.empty``, fp64, off-init;
   ``lengths`` advances once per forward. ``HybridState.empty`` puts the right cache kind in each
   slot, in the activation dtype (the KDA state in its fp32 / fp64 math dtype).
4. Causality: perturbing future tokens leaves earlier logits unchanged.
5. Loss at init against a derived prediction (derivation in the test).
6. Overfit one batch with ``build_k3_optimizer`` and Quantile Balancing after every step.
7. Parameters: ``build_mini_k3()`` / ``build_k3()`` on meta build their presets, and the live
   count == ``param_count`` minus the A_log storage tail (both, and a tied tiny config);
   checkpoint names and layer-type dispatch; spec §5's optimizer partition by name.
8. Init: HF ``_init_weights`` on every nn.Linear and the embedding, and nothing else.
9. ``return_aux`` and ``moe_update_biases``.
10. bf16 autocast over fp32 weights (train()'s amp path): a finite training step (smoke only).

Tiny config: 7 layers at period 4 (KDA 1-3, 5-6; MLA 4 and terminal 7), rope 4 (live), layer 1
dense, B = 2. B must be even: for odd B, (2ℓ) % B == 0 ⟺ ℓ % B == 0, and gate 2's FLA plant
would be the K3 rule itself. ``rms_norm_eps`` = 2e-5 sits off both 1e-5 and KimiRMSNorm's 1e-6,
so a hard-coded eps cannot pass gate 1.

The transcriptions keep HF's algebra and op order (the KDA kernels aside, item 1) with two
edits: ``.float()`` promotes instead of truncating (fp64 stays fp64), and ``moe_infer`` loses its
``@torch.no_grad()`` so gate 1 can compare gradients (HF never trains that block; its in-place ops
are differentiable as written). The embedding omits HF's ``padding_idx`` (K3Config carries no pad
id; model.py docstring).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_llm.k3.config import (
    K3Config,
    KDAConfig,
    MLAConfig,
    MoEConfig,
    build_layer_pattern,
    k3_full,
    mini_k3_d12,
)
from scratch_llm.k3.core.gated_mla import MLALatentKV
from scratch_llm.k3.core.kda import KDALayer, KDAState
from scratch_llm.k3.core.latent_moe import MoEGate
from scratch_llm.k3.model import HybridState, K3Model, build_k3, build_mini_k3
from scratch_llm.k3.muon import build_k3_optimizer, k3_param_groups
from scratch_llm.k3.param_count import count_params
from scratch_llm.model import cross_entropy
from scratch_llm.optim import gradient_clipping

FP64_STRUCTURAL_REL = 1e-10  # same bar as test_kda_parity_fla.py (two identical algebras)
B1, B2 = 4.0, 25.0  # SiTU betas (config.json activation_situ_beta / _linear_beta)


def _tiny_cfg(**overrides: object) -> K3Config:
    kda_layers, mla_layers = build_layer_pattern(7, period=4)
    assert kda_layers == (1, 2, 3, 5, 6) and mla_layers == (4, 7)
    cfg = K3Config(
        hidden_size=32,
        num_layers=7,
        kda_layers=kda_layers,
        mla_layers=mla_layers,
        vocab_size=64,
        kda=KDAConfig(num_heads=2, head_dim=8, decay_rank=4, a_log_size=4),
        mla=MLAConfig(
            num_heads=2,
            q_lora_rank=12,
            kv_lora_rank=10,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=6,
        ),
        moe=MoEConfig(
            num_experts=8,
            top_k=2,
            num_shared_experts=2,
            expert_intermediate=12,
            latent_size=16,
            dense_intermediate=48,
        ),
        attn_res_block_size=2,
        max_position_embeddings=64,
        rms_norm_eps=2e-5,
    )
    return replace(cfg, **overrides)


def _rel(a: Tensor, b: Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _ids(cfg: K3Config, batch: int, seq: int, seed: int) -> Tensor:
    return torch.randint(
        0, cfg.vocab_size, (batch, seq), generator=torch.Generator().manual_seed(seed)
    )


@torch.no_grad()
def _off_init(model: K3Model, seed: int) -> K3Model:
    """Move every parameter off its init and off any symmetric point, in place: 2-D weights
    N(0, 1/fan_in) (unit-scale activations, so softmaxes and SiTU caps are exercised), AttnRes
    pseudo-queries N(0, 4/H) (depth-logit std ≈ 2: peaked, not uniform), norm gains U(0.5, 1.5),
    KDA A_log N(0, 0.25) and dt_bias N(0, 4) (spread decays), QB bias N(0, 0.05²). The router
    keeps its kaiming draw and the short convs theirs; neither is a symmetric point."""
    gen = torch.Generator().manual_seed(seed)
    H = model.cfg.hidden_size

    def normal(p: Tensor, std: float) -> None:
        p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * std)

    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            p.copy_(0.5 + torch.rand(p.shape, generator=gen, dtype=p.dtype))
        elif name.endswith("res_proj.weight"):
            normal(p, 2 / math.sqrt(H))
        elif name.endswith("A_log"):
            normal(p, 0.5)
        elif name.endswith("dt_bias"):
            normal(p, 2.0)
        elif name.endswith("e_score_correction_bias"):
            normal(p, 0.05)
        elif p.ndim == 2 and not name.endswith("gate.weight"):
            normal(p, 1 / math.sqrt(p.shape[1]))
    return model


# ---------------------------------------------------------------------------
# HF reference, transcribed (never imported from ../ladders).
# ---------------------------------------------------------------------------


def _up(x: Tensor) -> Tensor:
    """HF's ``.float()``, promoting instead of truncating: fp64 stays fp64."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


class _HFKimiRMSNorm(nn.Module):
    """KimiRMSNorm (:226-236)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        x = _up(hidden_states)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


def _hf_situ_and_mul(x: Tensor) -> Tensor:
    """SituAndMul (:64-79) with beta = 4, linear_beta = 25."""
    d = x.shape[-1] // 2
    gate, up = _up(x[..., :d]), _up(x[..., d:])
    situ_a = B1 * torch.tanh(gate / B1) * torch.sigmoid(gate)
    up = B2 * torch.tanh(up / B2)
    return (situ_a * up).to(x.dtype)


class _HFMLP(nn.Module):
    """KimiMLP (:273-299): the layer-1 dense MLP and the shared experts."""

    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(_hf_situ_and_mul(gate_up))


class _HFBlockSparseMLP(nn.Module):
    """KimiBlockSparseMLP (:242-270): one routed expert."""

    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden, inter, bias=False)  # gate
        self.w2 = nn.Linear(inter, hidden, bias=False)  # down
        self.w3 = nn.Linear(hidden, inter, bias=False)  # up

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate_up = torch.cat([self.w1(hidden_states), self.w3(hidden_states)], dim=-1)
        return self.w2(_hf_situ_and_mul(gate_up))


class _HFGate(nn.Module):
    """KimiMoEGate (:666-759), one expert group (config.json), sigmoid scores."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        c = config.moe
        self.top_k, self.moe_renormalize = c.top_k, c.renormalize
        self.routed_scaling_factor = c.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(c.num_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.empty(c.num_experts))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(_up(hidden_states), _up(self.weight), None)
        scores = logits.sigmoid()
        scores = scores.view(bsz * seq_len, -1)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class _HFSparseMoeBlock(nn.Module):
    """KimiSparseMoeBlock (:762-838) + moe_infer (:840-874), latent MoE with its norm (both on in
    config.json); ``moe_infer`` without its no_grad (module docstring)."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        c, H = config.moe, config.hidden_size
        self.experts = nn.ModuleList(
            _HFBlockSparseMLP(c.latent_size, c.expert_intermediate) for _ in range(c.num_experts)
        )
        self.gate = _HFGate(config)
        self.shared_experts = _HFMLP(H, c.expert_intermediate * c.num_shared_experts)
        self.routed_expert_down_proj = nn.Linear(H, c.latent_size, bias=False)
        self.routed_expert_up_proj = nn.Linear(c.latent_size, H, bias=False)
        self.routed_expert_norm = _HFKimiRMSNorm(c.latent_size, eps=config.rms_norm_eps)

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


class _HFKimiMLAAttention(nn.Module):
    """KimiMLAAttention (:335-474) + eager_attention_forward (:311-332), K3 branch: q-LoRA, NoPE,
    output gate, kv heads == heads, no cache, no dropout. Latent norms at KimiRMSNorm's 1e-6."""

    def __init__(self, cfg: MLAConfig, hidden_size: int) -> None:
        super().__init__()
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.kv_lora_rank = cfg.kv_lora_rank
        self.v_head_dim = cfg.v_head_dim
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim ** (-0.5)
        h = cfg.num_heads
        self.q_a_proj = nn.Linear(hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = _HFKimiRMSNorm(cfg.q_lora_rank)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, h * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = _HFKimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, h * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(h * self.v_head_dim, hidden_size, bias=False)
        self.g_proj = nn.Linear(hidden_size, h * self.v_head_dim, bias=False)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        scores = torch.einsum("bhqd,bhkd->bhqk", query_states, key_states) * self.scaling
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
        probs = F.softmax(scores, dim=-1, dtype=_up(scores).dtype).to(query_states.dtype)
        attn_output = torch.einsum("bhqk,bhkd->bhqd", probs, value_states).transpose(1, 2)

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        g = self.g_proj(hidden_states).sigmoid()
        attn_output = attn_output * g
        return self.o_proj(attn_output)


class _HFFusedRMSNormGated(nn.Module):
    """FLA FusedRMSNormGated(activation="sigmoid") (fla/modules/fused_norm_gate.py:1007): RMSNorm
    over the head dim, times the weight, times sigmoid of the gate."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        x_hat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_hat * self.weight * g.sigmoid()


def _hf_short_conv(conv: nn.Conv1d, x: Tensor) -> Tensor:
    """FLA ShortConvolution (fla/modules/conv/short_conv.py), no cache: an nn.Conv1d that is
    depthwise with padding kernel − 1, cut to the first T outputs (causal), then SiLU."""
    return F.silu(conv(x.transpose(1, 2))[..., : x.shape[1]]).transpose(1, 2)


def _hf_l2norm(x: Tensor) -> Tensor:
    """FLA l2norm_fwd (fla/modules/l2norm.py:148-150): x · rsqrt(Σx² + 1e-6)."""
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)


def _naive_recurrent_kda(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor) -> Tensor:
    """FLA naive_recurrent_kda (fla/ops/kda/naive.py:12-) from a zero state, without its fp32
    cast: S ← diag(e^g_t) S; S ← S + β_t k_t (v_t − k_tᵀ S)ᵀ; o_t = (K^−½ q_t)ᵀ S."""
    B, T, H, K = q.shape
    q = q * K**-0.5
    S = q.new_zeros(B, H, K, v.shape[-1])
    out = []
    for t in range(T):
        q_t, k_t, v_t, g_t, b_t = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        S = S * g_t[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv", b_t[..., None] * k_t, v_t - (k_t[..., None] * S).sum(-2)
        )
        out.append(torch.einsum("bhk,bhkv->bhv", q_t, S))
    return torch.stack(out, dim=1)


class _HFKimiDeltaAttention(nn.Module):
    """KimiDeltaAttention (:477-663), chunk mode, no cache and no padding mask, full-rank output
    gate. ``chunk_kda`` with ``use_qk_l2norm_in_kernel``, ``use_gate_in_kernel``,
    ``use_beta_sigmoid_in_kernel`` and ``safe_gate`` becomes its documented semantics: l2norm on q
    and k, g = lower_bound · sigmoid(exp(A_log) · (g + dt_bias)) (``naive_kda_lowerbound_gate``,
    fla/ops/kda/gate.py:58-93), β = sigmoid(b), then the delta rule above. ``f_a_proj`` has
    ``decay_rank`` outputs where HF writes ``head_dim``; config.json's two are equal."""

    def __init__(self, cfg: KDAConfig, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.num_heads, self.head_dim = cfg.num_heads, cfg.head_dim
        self.gate_lower_bound = cfg.gate_lower_bound
        P, K = cfg.projection_size, cfg.conv_kernel_size
        self.q_proj = nn.Linear(hidden_size, P, bias=False)
        self.k_proj = nn.Linear(hidden_size, P, bias=False)
        self.v_proj = nn.Linear(hidden_size, P, bias=False)
        self.q_conv1d = nn.Conv1d(P, P, K, groups=P, padding=K - 1, bias=False)
        self.k_conv1d = nn.Conv1d(P, P, K, groups=P, padding=K - 1, bias=False)
        self.v_conv1d = nn.Conv1d(P, P, K, groups=P, padding=K - 1, bias=False)
        self.A_log = nn.Parameter(torch.empty(cfg.num_heads))
        self.f_a_proj = nn.Linear(hidden_size, cfg.decay_rank, bias=False)
        self.f_b_proj = nn.Linear(cfg.decay_rank, P, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(P))
        self.b_proj = nn.Linear(hidden_size, cfg.num_heads, bias=False)
        self.g_proj = nn.Linear(hidden_size, P, bias=False)
        self.o_norm = _HFFusedRMSNormGated(cfg.head_dim, eps=eps)
        self.o_proj = nn.Linear(P, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        B, T, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        q = _hf_short_conv(self.q_conv1d, self.q_proj(hidden_states)).view(B, T, H, D)
        k = _hf_short_conv(self.k_conv1d, self.k_proj(hidden_states)).view(B, T, H, D)
        v = _hf_short_conv(self.v_conv1d, self.v_proj(hidden_states)).view(B, T, H, D)
        g = self.f_b_proj(self.f_a_proj(hidden_states)).view(B, T, H, D)
        beta = _up(self.b_proj(hidden_states))

        q, k = _hf_l2norm(q), _hf_l2norm(k)
        g = self.gate_lower_bound * torch.sigmoid(
            self.A_log.view(H, 1).exp() * (g + self.dt_bias.view(H, D))
        )
        o = _naive_recurrent_kda(q, k, v, g, beta.sigmoid())

        o = self.o_norm(o, self.g_proj(hidden_states).view(B, T, H, D))
        return self.o_proj(o.reshape(B, T, H * D))


def _hf_causal_mask(T: int, dtype: torch.dtype) -> Tensor:
    """Additive [1, 1, T, T] mask as transformers' create_causal_mask: finfo.min above the diagonal."""
    future = torch.ones(T, T, dtype=torch.bool).triu(1)
    return torch.zeros(T, T, dtype=dtype).masked_fill(future, torch.finfo(dtype).min)[None, None]


def _hf_apply_attn_res(
    prefix_sum: Tensor, block_residual: Tensor, proj: nn.Linear, norm: _HFKimiRMSNorm
) -> Tensor:
    """_apply_attn_res (:1075-1088). prefix_sum [N, H], block_residual [N, num_blocks, H]."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = _up(v)
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = _up(norm.weight) * _up(proj.weight.squeeze(0))
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


def _k3_boundary(layer_idx: int, block_size: int) -> bool:
    return layer_idx % block_size == 0  # HF :995, vLLM is_block_write_layer :977


def _fla_sublayer_boundary(layer_idx: int, block_size: int) -> bool:
    return (2 * layer_idx) % block_size == 0  # FLA modeling_kda.py counts sublayers


class _HFDecoderLayer(nn.Module):
    """KimiDecoderLayer (:877-917) with ``_forward_attn_residual`` (:973-1046); ``opens_block``
    stands in for line 995 so gate 2 can plant a misread."""

    def __init__(self, config: K3Config, layer_idx: int) -> None:
        super().__init__()
        H, eps = config.hidden_size, config.rms_norm_eps
        self.layer_idx = layer_idx
        self.attn_res_block_size = config.attn_res_block_size
        self.is_linear_attn = (layer_idx + 1) in config.kda_layers  # is_kda_layer (config :152)
        self.self_attn = (
            _HFKimiDeltaAttention(config.kda, H, eps=eps)  # HF :539: o_norm eps = rms_norm_eps
            if self.is_linear_attn
            else _HFKimiMLAAttention(config.mla, H)
        )
        if layer_idx >= config.first_k_dense:  # first_k_dense_replace; moe_layer_freq = 1
            self.block_sparse_moe = _HFSparseMoeBlock(config)
        else:
            self.mlp = _HFMLP(H, config.moe.dense_intermediate)
        self.input_layernorm = _HFKimiRMSNorm(H, eps=eps)
        self.post_attention_layernorm = _HFKimiRMSNorm(H, eps=eps)
        self.self_attention_res_norm = _HFKimiRMSNorm(H, eps=eps)
        self.mlp_res_norm = _HFKimiRMSNorm(H, eps=eps)
        self.self_attention_res_proj = nn.Linear(H, 1, bias=False)
        self.mlp_res_proj = nn.Linear(H, 1, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        block_residual: Tensor,
        opens_block: Callable[[int, int], bool],
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum: Tensor | None = hidden_states

        if block_residual.shape[1] > 0:
            hidden_states = _hf_apply_attn_res(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)

        if opens_block(self.layer_idx, self.attn_res_block_size):
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1
            )
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear_attn:
            hidden_states = self.self_attn(hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states, attention_mask)

        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        hidden_states = _hf_apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)

        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        prefix_sum = prefix_sum + hidden_states  # :1040-1046; never None after the attention
        return prefix_sum, block_residual


class _HFKimiLinearModel(nn.Module):
    """KimiLinearModel (:1090-1233), AttnRes branch, no cache, all-ones attention mask."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        H, eps = config.hidden_size, config.rms_norm_eps
        self.embed_tokens = nn.Embedding(config.vocab_size, H)
        self.layers = nn.ModuleList(
            _HFDecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)
        )
        self.norm = _HFKimiRMSNorm(H, eps=eps)
        self.output_attn_res_norm = _HFKimiRMSNorm(H, eps=eps)
        self.output_attn_res_proj = nn.Linear(H, 1, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        opens_block: Callable[[int, int], bool] = _k3_boundary,
        output_mix: bool = True,
    ) -> Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        causal_mask = _hf_causal_mask(input_ids.shape[1], inputs_embeds.dtype)
        hidden_states = inputs_embeds
        block_residual = hidden_states.new_zeros(
            hidden_states.shape[0] * hidden_states.shape[1], 0, hidden_states.shape[2]
        )
        for decoder_layer in self.layers:
            hidden_states, block_residual = decoder_layer(
                hidden_states, causal_mask, block_residual, opens_block
            )
        if output_mix:  # _apply_output_attn_res (:1226-1233); False is gate 2's plant
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = _hf_apply_attn_res(
                hidden_states.view(-1, hidden_size),
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
            ).view(batch_size, seq_len, hidden_size)
        return self.norm(hidden_states)


class _HFKimiLinearForCausalLM(nn.Module):
    """KimiLinearForCausalLM (:1236-1310): ``model`` then ``lm_head``."""

    def __init__(self, config: K3Config) -> None:
        super().__init__()
        self.model = _HFKimiLinearModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        opens_block: Callable[[int, int], bool] = _k3_boundary,
        output_mix: bool = True,
    ) -> Tensor:
        return self.lm_head(self.model(input_ids, opens_block, output_mix))


def _hf_name(name: str) -> str:
    """Our name -> the HF text model's: everything but lm_head lives under ``model.``. (The
    released checkpoint adds ``language_model.`` in front of both.)"""
    return name if name.startswith("lm_head.") else f"model.{name}"


def _hf_twin(model: K3Model) -> _HFKimiLinearForCausalLM:
    """The HF transcription holding ``model``'s weights, loaded strictly by name."""
    ref = _HFKimiLinearForCausalLM(model.cfg).to(model.embed_tokens.weight.dtype)
    ref.load_state_dict({_hf_name(n): t for n, t in model.state_dict().items()}, strict=True)
    return ref.eval()


def _logits_and_grads(
    module: nn.Module, run: Callable[[], Tensor]
) -> tuple[Tensor, dict[str, Tensor]]:
    """logits = run(), and the gradient of <logits, probe> (a fixed random probe) for every
    parameter the graph reaches, by name. A detached source keeps every value and drops or moves
    gradients, so only this half of gate 1 can see it."""
    module.zero_grad(set_to_none=True)
    logits = run()
    probe = torch.randn(
        logits.shape, generator=torch.Generator().manual_seed(7), dtype=logits.dtype
    )
    (logits * probe).sum().backward()
    grads = {n: p.grad for n, p in module.named_parameters() if p.grad is not None}
    return logits.detach(), grads


# ---------------------------------------------------------------------------
# 1-2. Whole model vs the HF transcription; the planted AttnRes misreads.
# ---------------------------------------------------------------------------


def test_matches_hf_transcription_logits_and_grads_fp64() -> None:
    torch.manual_seed(0)
    ours = _off_init(K3Model(_tiny_cfg()).double(), seed=1)
    ref = _hf_twin(ours)
    ids = _ids(ours.cfg, 2, 9, seed=2)

    out, grads = _logits_and_grads(ours, lambda: ours(ids))
    ref_out, ref_grads = _logits_and_grads(ref, lambda: ref(ids))
    assert _rel(out, ref_out) < FP64_STRUCTURAL_REL
    assert {_hf_name(n) for n in grads} == set(ref_grads)
    for name, g in grads.items():
        assert _rel(g, ref_grads[_hf_name(name)]) < FP64_STRUCTURAL_REL, name
    # Coverage: every trainable parameter but the layer-0 attention pair (one source, softmax ≡ 1,
    # skipped by HF too) and the experts no token chose is in both graphs.
    trainable = {n for n, p in ours.named_parameters() if p.requires_grad}
    idle = {n for n in trainable - grads.keys() if ".experts." not in n}
    assert idle == {
        "layers.0.self_attention_res_proj.weight",
        "layers.0.self_attention_res_norm.weight",
    }


@pytest.mark.parametrize(
    "plant",
    [{"opens_block": _fla_sublayer_boundary}, {"output_mix": False}],
    ids=["fla_sublayer_boundary", "no_output_mix"],
)
def test_planted_attn_res_misreads_move_the_logits(plant: Mapping[str, object]) -> None:
    """Gate 1 has the power to see both misreads on this config (B = 2 is even; module doc)."""
    torch.manual_seed(0)
    ours = _off_init(K3Model(_tiny_cfg()).double(), seed=1)
    ref = _hf_twin(ours)
    ids = _ids(ours.cfg, 2, 9, seed=2)
    with torch.no_grad():
        out = ours(ids)
        assert _rel(ref(ids), out) < FP64_STRUCTURAL_REL
        assert _rel(ref(ids, **plant), out) > 1e-3  # O(1) moves, 7 decades above the bar


def test_attn_res_query_gradients() -> None:
    """The output pseudo-query and every multi-source query learn; the layer-0 attention query
    mixes the embedding alone and gets no gradient (weight decay then shrinks it: the checkpoint's
    is ~4e-6)."""
    torch.manual_seed(0)
    model = _off_init(K3Model(_tiny_cfg()).double(), seed=1)
    cross_entropy(model(_ids(model.cfg, 2, 9, seed=2)), _ids(model.cfg, 2, 9, seed=3)).backward()
    for name, p in model.named_parameters():
        if name.endswith("res_proj.weight"):
            if name == "layers.0.self_attention_res_proj.weight":
                assert p.grad is None
            else:
                assert p.grad is not None and p.grad.abs().max().item() > 0, name
    out_query = model.get_parameter("output_attn_res_proj.weight").grad
    assert out_query is not None and out_query.abs().max().item() > 0


# ---------------------------------------------------------------------------
# 3-4. Hybrid cache and causality.
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_prefill_equals_decode_equals_split_prefill_fp64() -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = _off_init(K3Model(cfg).double(), seed=4).eval()
    ids = _ids(cfg, 2, 11, seed=5)
    full = model(ids)

    def fresh() -> HybridState:
        return HybridState.empty(cfg, 2, torch.device("cpu"), torch.float64)

    state = fresh()
    decoded = torch.cat([model(ids[:, t : t + 1], state) for t in range(11)], dim=1)
    assert torch.equal(state.lengths, torch.full((2,), 11))  # bumped once per forward

    split_state = fresh()
    split = torch.cat([model(ids[:, :5], split_state), model(ids[:, 5:], split_state)], dim=1)
    assert torch.equal(split_state.lengths, torch.full((2,), 11))

    assert _rel(decoded, full) < FP64_STRUCTURAL_REL
    assert _rel(split, full) < FP64_STRUCTURAL_REL


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float64], ids=["fp32", "bf16", "fp64"]
)
def test_hybrid_state_empty_slots(dtype: torch.dtype) -> None:
    """Pinned without a forward, because nothing downstream would notice: both modules read a
    cache of any dtype and write it back in their own, so an fp32 model decoding from fp64 slots
    raises nothing and runs the whole stream in fp64 (mutation review)."""
    cfg = _tiny_cfg()
    state = HybridState.empty(cfg, 2, torch.device("cpu"), dtype)
    state_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    for i in range(cfg.num_layers):  # the right cache kind in every slot, nothing in the other
        kda, mla = state.kda_states[i], state.mla_caches[i]
        if i + 1 in cfg.kda_layers:
            assert isinstance(kda, KDAState) and mla is None
            assert not kda.s_t.any() and not kda.conv.any()
            assert kda.s_t.dtype == state_dtype and kda.conv.dtype == dtype
        else:
            assert isinstance(mla, MLALatentKV) and kda is None and mla.length == 0
            assert mla.c_kv.dtype == mla.k_rope.dtype == dtype
    assert state.lengths.dtype == torch.long and not state.lengths.any()


@torch.no_grad()
def test_causal() -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = _off_init(K3Model(cfg).double(), seed=6).eval()
    ids = _ids(cfg, 2, 13, seed=7)
    t0 = 6
    ids2 = ids.clone()
    ids2[:, t0 + 1 :] = (ids2[:, t0 + 1 :] + 1) % cfg.vocab_size  # every future token changes
    out, out2 = model(ids), model(ids2)
    torch.testing.assert_close(out2[:, : t0 + 1], out[:, : t0 + 1], rtol=0, atol=1e-12)
    # Positive control: every perturbed position moves by O(1) (unit-scale off-init weights), so a
    # model that ignored its input would fail here; 1e-3 is 9 decades above the atol.
    assert (out2[:, t0 + 1 :] - out[:, t0 + 1 :]).abs().amax(-1).min().item() > 1e-3


# ---------------------------------------------------------------------------
# 5-6. Loss at init and overfitting one batch.
# ---------------------------------------------------------------------------


def test_loss_at_init_matches_the_derived_prediction() -> None:
    """Derivation, over the draw of lm_head W ~ N(0, s²) (s = initializer_range) and conditional
    on the final hidden states h_t, which do not depend on W (K3 is untied):

    1. h_t = norm(x_t) with weight 1, so |h_t|² = H·ρ_t, ρ_t = ms_t / (ms_t + eps) ≤ 1 (ms_t the
       mean square of x_t). "Unit RMS after norm" is the premise ρ_t ≥ 0.99, asserted below.
    2. z_tv = W_v · h_t: V independent N(0, σ_t²), σ_t² = s²·|h_t|² = σ²·ρ_t, σ² := s²·H.
    3. CE_t = logsumexp_v z_tv − z_{t,y_t}. The target is drawn apart from W: E[z_{t,y_t}] = 0.
       logsumexp = log V + log Ȳ_t, Ȳ_t = mean_v e^{z_tv} → E e^z = e^{σ_t²/2}: Jensen gives
       E[CE] ≤ log V + σ²/2, and the delta method its finite-V correction,
       E[log Ȳ] = σ_t²/2 − (e^{σ_t²} − 1)/(2V) + O(V⁻²).
    4. Noise of the mean over N = B·T positions. Target term: variance ≤ σ²(1 + N/V)/N (two
       positions share a row of W when their targets coincide, ≤ N/V of the time). logsumexp
       term: sd ≤ √((e^{σ²} − 1)/V) even if all positions moved together (they share W).
       Band = 4 × (sum of the two sds) + σ²(1 − 0.99)/2 for the ρ premise.

    Why s = 0.1 and not the checkpoint's 0.02: at a CPU-sized H, s²H/2 falls under the sampling
    band, and the gate could not tell this model from logits ≡ 0. With s = 0.1, H = 128,
    V = 8192, N = 2048: σ²/2 = 0.64, band ≈ 0.19. Excluded: loss ≈ log V (missing final norm,
    s ignored or hard-coded to 0.02: σ²/2 = 0.03) and an lm_head left at nn.Linear's kaiming
    init (σ² = |h|²/(3H) ≈ 1/3). The scaffold's "±0.1 of log V" assumed another init.
    """
    s, H, V, B, T = 0.1, 128, 8192, 8, 256
    cfg = _tiny_cfg(hidden_size=H, vocab_size=V, initializer_range=s)
    torch.manual_seed(0)
    model = K3Model(cfg)
    ids, targets = _ids(cfg, B, T, seed=1), _ids(cfg, B, T, seed=2)
    seen: dict[str, Tensor] = {}
    handle = model.lm_head.register_forward_pre_hook(lambda _m, args: seen.update(h=args[0]))
    with torch.no_grad():
        loss = cross_entropy(model(ids), targets).item()
    handle.remove()

    rho = seen["h"].pow(2).mean(-1)  # ρ_t, the norm's weight being 1 at init
    # ρ = ms/(ms + eps) < 1 exactly (measured max 1 − 1.6e-5, the eps term); 1e-6 ≈ 8 fp32 ulps
    # of 1 is headroom for rsqrt/square/mean rounding when ms ≫ 1 shrinks that gap below ulps.
    assert rho.min().item() >= 0.99 and rho.max().item() <= 1.0 + 1e-6

    sigma2, N = s * s * H, B * T
    predicted = math.log(V) + sigma2 / 2 - math.expm1(sigma2) / (2 * V)
    sd = math.sqrt(sigma2 * (1 + N / V) / N) + math.sqrt(math.expm1(sigma2) / V)
    band = 4 * sd + sigma2 * (1 - 0.99) / 2
    assert abs(loss - predicted) <= band, f"loss {loss:.4f}, predicted {predicted:.4f} ± {band:.4f}"
    # Power: logits ≡ 0 give CE = log V exactly; the band's lower edge must clear it (σ²/2 − band
    # ≈ 0.45 here), else a model with no head signal could pass.
    assert predicted - band > math.log(V)


def test_overfit_one_batch() -> None:
    """ROADMAP K6: memorize one fixed batch to CE < 1e-2 through the real optimizer (Per-Head Muon
    + AdamW) with Quantile Balancing after every step, as train() will run it. adamw_lr = 1e-2:
    the AdamW side holds the embedding and lm_head, and at the default 3e-3 this takes ~135 steps
    (measured, ~18 s); at 1e-2 it takes 33-34 on seeds 0-4 (~5 s), so 60 is the budget."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = K3Model(cfg).train()
    opt = build_k3_optimizer(model, adamw_lr=1e-2)
    ids = _ids(cfg, 4, 17, seed=1)
    inputs, targets = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():  # records no QB statistics
        initial = cross_entropy(model(inputs), targets).item()
    loss = initial
    for _ in range(60):
        opt.zero_grad()
        step_loss = cross_entropy(model(inputs), targets)
        step_loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        opt.step()
        assert len(model.moe_update_biases()) == cfg.num_moe_layers
        loss = step_loss.item()
        if loss < 1e-2:
            break
    assert loss < 1e-2, f"failed to overfit: {loss:.4f} after 60 steps (started {initial:.3f})"


# ---------------------------------------------------------------------------
# 7-8. Parameters, names and init.
# ---------------------------------------------------------------------------


def _a_log_tail(cfg: K3Config) -> int:
    """The checkpoint stores a_log_size A_log entries per KDA layer; KDALayer keeps num_heads."""
    return cfg.num_kda_layers * (cfg.kda.a_log_size - cfg.kda.num_heads)


@pytest.mark.parametrize(
    ("build", "preset"), [(build_mini_k3, mini_k3_d12), (build_k3, k3_full)], ids=["mini", "full"]
)
def test_parameter_count_names_and_dispatch_on_meta(
    build: Callable[[], K3Model], preset: Callable[[], K3Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public builders on meta build their preset, and live == param_count's closed text total
    minus the A_log storage tail. Meta tensors hold no values, so nn.Linear's kaiming init is
    patched out here: it draws nothing on meta and was half of the k3_full build (10.9 s → 5.7 s
    measured; the rest constructs 92 × 896 experts)."""
    monkeypatch.setattr(nn.Linear, "reset_parameters", lambda _self: None)
    with torch.device("meta"):
        model = build()
    cfg = model.cfg
    assert cfg == preset()
    params = dict(model.named_parameters())
    live = sum(p.numel() for p in params.values())
    assert live == count_params(cfg).total_text - _a_log_tail(cfg)

    H, V, m, e = cfg.hidden_size, cfg.vocab_size, cfg.mla, cfg.moe
    expected = {
        "embed_tokens.weight": (V, H),
        "layers.0.self_attn.A_log": (cfg.kda.num_heads,),
        "layers.0.mlp.gate_proj.weight": (e.dense_intermediate, H),
        "layers.0.self_attention_res_proj.weight": (1, H),
        "layers.0.mlp_res_norm.weight": (H,),
        "layers.1.block_sparse_moe.gate.e_score_correction_bias": (e.num_experts,),
        "layers.1.block_sparse_moe.experts.0.w1.weight": (e.expert_intermediate, e.latent_size),
        "layers.3.self_attn.kv_a_proj_with_mqa.weight": (m.kv_lora_rank + m.qk_rope_head_dim, H),
        "norm.weight": (H,),
        "output_attn_res_norm.weight": (H,),
        "output_attn_res_proj.weight": (1, H),
        "lm_head.weight": (V, H),
    }
    for name, shape in expected.items():
        assert tuple(params[name].shape) == shape, name
    assert params["lm_head.weight"] is not params["embed_tokens.weight"]  # untied

    def layers_with(suffix: str) -> set[int]:
        return {
            int(n.split(".")[1]) for n in params if n.startswith("layers.") and n.endswith(suffix)
        }

    assert layers_with("self_attn.A_log") == {i - 1 for i in cfg.kda_layers}
    assert layers_with("self_attn.kv_b_proj.weight") == {i - 1 for i in cfg.mla_layers}
    assert layers_with("mlp.gate_proj.weight") == set(range(cfg.first_k_dense))
    assert layers_with("block_sparse_moe.gate.weight") == {i - 1 for i in cfg.moe_layers}
    bias = params["layers.1.block_sparse_moe.gate.e_score_correction_bias"]
    assert bias.dtype == torch.float32 and not bias.requires_grad


def test_tied_embeddings_share_one_matrix() -> None:
    """``tie_word_embeddings`` (false in both presets and config.json): lm_head holds the
    embedding matrix itself, so the live count is param_count's tied total, V·H below untied."""
    cfg = _tiny_cfg(tie_word_embeddings=True)
    model = K3Model(cfg)
    assert model.lm_head.weight is model.embed_tokens.weight
    live = sum(p.numel() for p in model.parameters())  # a shared Parameter counts once
    assert live == count_params(cfg).total_text - _a_log_tail(cfg)
    untied = count_params(replace(cfg, tie_word_embeddings=False)).total_text
    assert untied - count_params(cfg).total_text == cfg.vocab_size * cfg.hidden_size


def test_optimizer_partition_on_the_assembled_model() -> None:
    """Spec §5's partition, restated on checkpoint names and checked for every parameter of a real
    K3Model (test_k3_muon.py pins it on stand-in modules). ``k3_param_groups`` itself raises
    unless the groups cover the trainable parameters exactly once."""
    cfg = _tiny_cfg()
    m = cfg.mla
    q_rows, kv_rows = m.qk_nope_head_dim + m.qk_rope_head_dim, m.qk_nope_head_dim + m.v_head_dim
    model = K3Model(cfg)
    groups, adamw = k3_param_groups(model)
    assert [group["head_rows"] for group in groups] == [cfg.kda.head_dim, None, q_rows, kv_rows]
    route: dict[int, int | str | None] = {id(p): "adamw" for p in adamw}
    for group in groups:
        route.update({id(p): group["head_rows"] for p in group["params"]})

    def expected(name: str) -> int | str | None:
        """head_rows for PerHeadMuon (None: whole matrix), "adamw", or "none" (no group)."""
        if name.endswith("e_score_correction_bias"):  # Quantile Balancing assigns it
            return "none"
        if name in ("embed_tokens.weight", "lm_head.weight") or name.endswith(
            ("norm.weight", "A_log", "dt_bias", "conv1d.weight", "res_proj.weight")
        ):
            return "adamw"
        layer_id = int(name.split(".")[1]) + 1
        if layer_id in cfg.kda_layers and name.endswith(
            ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")
        ):
            return cfg.kda.head_dim
        if name.endswith("self_attn.q_b_proj.weight"):
            return q_rows
        if name.endswith("self_attn.kv_b_proj.weight"):
            return kv_rows
        return None  # every other 2-D weight, router and experts included

    got = {name: route.get(id(p), "none") for name, p in model.named_parameters()}
    assert got == {name: expected(name) for name in got}


def _assert_sample_std(t: Tensor, std: float, kurtosis: float, name: str) -> None:
    """Sample std of n iid draws: relative sd ≈ √((κ − 1)/(4n)) (κ = 3 normal, 9/5 uniform);
    5 of those. The mean: |mean| ≤ 5·std/√n."""
    n = t.numel()
    got = t.double().std().item()
    assert abs(got / std - 1) <= 5 * math.sqrt((kurtosis - 1) / (4 * n)), f"{name}: std {got:.4g}"
    assert abs(t.double().mean().item()) <= 5 * std / math.sqrt(n), name


def test_init_is_hf_init_weights() -> None:
    """nn.Linear's own kaiming init has std 1/√(3·fan_in) ≥ 0.083 here (fan_in ≤ 48), four
    times initializer_range or more, and nn.Embedding's is 1: every per-module band below
    separates HF's init from the constructors'."""
    cfg = _tiny_cfg()
    H, s = cfg.hidden_size, cfg.initializer_range
    torch.manual_seed(0)
    model = K3Model(cfg)

    queries: list[Tensor] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear | nn.Embedding):
            _assert_sample_std(module.weight.detach(), s, 3.0, name)
            if name.endswith("res_proj"):
                queries.append(module.weight.detach().flatten())
        elif isinstance(module, MoEGate):  # HF reset_parameters: kaiming_uniform_(a=√5)
            w = module.weight.detach()
            assert w.abs().max().item() <= 1 / math.sqrt(H)
            _assert_sample_std(w, 1 / math.sqrt(3 * H), 9 / 5, name)
            assert not module.e_score_correction_bias.any()
        elif isinstance(module, KDALayer):  # KDALayer's own gate and conv init, untouched
            assert not module.A_log.any()
            dt_lo, dt_hi = math.log(math.expm1(1e-3)), math.log(math.expm1(1e-1))
            # fp64 bounds vs fp32-rounded draw endpoints and an fp32 softplus⁻¹: measured 4.1e-7
            # outside at the endpoint, 8.9e-7 one ulp further (ulp 4.8e-7 at |dt| ≈ 6.9); 1e-6.
            assert (
                dt_lo - 1e-6
                <= module.dt_bias.min().item()
                <= module.dt_bias.max().item()
                <= dt_hi + 1e-6
            )
            kernel = cfg.kda.conv_kernel_size
            for conv in (module.q_conv1d, module.k_conv1d, module.v_conv1d):
                w = conv.weight.detach()
                assert w.abs().max().item() <= 1 / math.sqrt(kernel)
                _assert_sample_std(w, 1 / math.sqrt(3 * kernel), 9 / 5, name)
    assert len(queries) == 2 * cfg.num_layers + 1
    _assert_sample_std(torch.cat(queries), s, 3.0, "pseudo-queries")
    for name, p in model.named_parameters():
        if name.endswith("norm.weight"):
            assert torch.equal(p, torch.ones_like(p)), name

    torch.manual_seed(0)
    zero = K3Model(cfg, attn_res_query_init="zero")
    for name, p in zero.named_parameters():
        if name.endswith("res_proj.weight"):
            assert not p.any(), name
    _assert_sample_std(zero.lm_head.weight.detach(), s, 3.0, "lm_head (zero arm)")
    with pytest.raises(ValueError, match="attn_res_query_init"):
        K3Model(cfg, attn_res_query_init="uniform")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 9. return_aux and Quantile Balancing; the bf16 training path.
# ---------------------------------------------------------------------------


def test_return_aux_and_moe_update_biases() -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = K3Model(cfg).double()
    ids = _ids(cfg, 2, 9, seed=1)
    biases = {
        name: m.e_score_correction_bias
        for name, m in model.named_modules()
        if isinstance(m, MoEGate)
    }
    assert len(biases) == cfg.num_moe_layers
    start = {name: b.clone() for name, b in biases.items()}

    model.eval()
    logits, aux = model(ids, return_aux=True)
    assert torch.equal(logits, model(ids))
    assert aux.total.shape == () and aux.total.dtype == logits.dtype == torch.float64
    assert aux.total.item() == 0.0  # aux-loss-free: train()'s CE + aux.total is the CE
    assert model.moe_update_biases() == {}  # an eval forward records nothing
    assert all(torch.equal(b, start[n]) for n, b in biases.items())

    model.train()
    model(ids, return_aux=True)  # a training forward with autograd on records the batch
    stats = model.moe_update_biases()
    assert set(stats) == {f"layers.{i - 1}.block_sparse_moe" for i in cfg.moe_layers}
    for name, b in biases.items():
        assert not torch.equal(b, start[name]), name  # QB assigned a new bias
        layer_stats = stats[name.removesuffix(".gate")]
        assert int(layer_stats.num_tokens) == ids.numel()
    assert model.moe_update_biases() == {}  # consumed


def test_bf16_autocast_training_step_is_finite() -> None:
    """train()'s amp_dtype="bf16" path: fp32 weights under bf16 autocast. The embedding stays
    fp32 while every nn.Linear returns bf16; AttnResState casts sublayer outputs into the fp32
    stream (the mixes would raise on two dtypes), and the norms, KDA state and MLA softmax run in
    fp32. Smoke only: finite logits and gradients, no tolerance claimed."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = K3Model(cfg).train()
    ids = _ids(cfg, 2, 9, seed=1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, aux = model(ids, return_aux=True)
        loss = cross_entropy(logits, ids) + aux.total
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
