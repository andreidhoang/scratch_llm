"""core/attn_res.py — Block Attention Residuals (arXiv:2603.15031), K3's depth-wise residual.

A plain residual stack feeds every sublayer the running sum x_ℓ = emb + Σ(all earlier outputs).
AttnRes feeds it a learned, per-token *choice* over depth instead. Each sublayer owns a
pseudo-query w ∈ R^H and a key-norm weight γ ∈ R^H, and mixes a short list of sources v_i:

    k_i = v_i / sqrt(mean(v_i²) + eps)            # RMSNorm without its weight
    s_i = Σ_d k_i[d] · γ[d] · w[d]                  # = ⟨w, RMSNorm_γ(v_i)⟩, NO 1/√d
    out = Σ_i softmax(s)_i · v_i                    # values are the RAW sources

Keys are normalised so a source's magnitude cannot buy it attention; values are not, so the
chosen sources arrive at their true scale. There is no 1/√d: HF ``_apply_attn_res`` (:1075-1088),
the vLLM kernel (``amd/ops/attn_res.py``, ``logits = Σ values·(norm_w·qk_w) · rsqrt(...)``) and FLA
``naive_attnres(scale=1.0)`` all omit it. Everything is per token — no mixing across positions — so
decode needs no AttnRes cache: a token's sources are rebuilt from its own layer outputs.

Sources are *blocks*, not single sublayers (block size B counted in LAYERS: 12 full, 4 mini).
The bank holds committed block sums, bank[0] being the embedding; the prefix is the running sum of
the current block. For layer ℓ (0-based, HF ``layer_idx``; model.py's 1-based ``layer_id`` − 1):

    attn input  = mix(w_attn, γ_attn, bank + [prefix])      # ℓ = 0: the single source emb
    if ℓ % B == 0:  bank += [prefix]; prefix = h_attn        # the attention output opens a block
    else:           prefix = prefix + h_attn
    mlp input   = mix(w_mlp, γ_mlp, bank + [prefix])
    prefix      = prefix + h_mlp
    final       = mix(w_out, γ_out, bank + [prefix])         # then ``norm`` → ``lm_head``

(HF ``KimiDecoderLayer._forward_attn_residual`` :973-1046, ``KimiLinearModel.forward``
:1188-1219 and ``_apply_output_attn_res`` :1226-1233; vLLM ``nvidia/model.py`` :977-979 and
``_pre_attn_norm``/``_post_attn_norm`` :1017-1079.) Source counts follow: ⌈ℓ/B⌉ + 1 at the
attention input, one more at the mlp input when ℓ % B == 0, ⌈L/B⌉ + 1 at the output — 9 for
K3 (93 layers, B = 12), 4 for mini (12, 4). FLA's ``modeling_kda.py`` counts SUBLAYERS instead
((2ℓ) % B for attention, (2ℓ+1) % B for mlp): a different architecture, not K3.

Known-flat gradient: the layer-0 attention query mixes one source, and softmax over one source is
identically 1, so it gets no gradient at all (HF and vLLM skip the mix; so does this module, which
makes that ``.grad`` None). Weight decay then shrinks it towards 0 — the checkpoint's is ~4e-6.
Every other query mixes ≥ 2 sources and must get a gradient; a wrong source list (e.g. the
embedding dropped) is caught by the bookkeeping gate, not by shapes.

Parameters live in the caller under their checkpoint names (per layer
``self_attention_res_{norm,proj}``, ``mlp_res_{norm,proj}``; model level
``output_attn_res_{norm,proj}``; proj = ``nn.Linear(H, 1, bias=False)``) — 4H per layer + 2H,
``param_count._attn_res_per_layer``. This module is the math and the bookkeeping only.

Not implemented: vLLM's fused next-layer RMSNorm (the caller applies ``input_layernorm`` /
``post_attention_layernorm``) and its online-softmax tiling over sources (≤ 9 sources, so the
[..., n, H] stack is materialised). Gates: ``tests/test_k3_attn_res.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["AttnResState", "attn_res_mix"]


def _math_dtype(*xs: Tensor) -> torch.dtype:
    """fp32 for every low-precision input, fp64 if any input is fp64."""
    dtype = torch.float32
    for x in xs:
        dtype = torch.promote_types(dtype, x.dtype)
    return dtype


def attn_res_mix(
    sources: Sequence[Tensor], query: Tensor, norm_weight: Tensor, eps: float
) -> Tensor:
    """Σ_i softmax_i(⟨query, RMSNorm_norm_weight(v_i)⟩) · v_i over ``sources``, per position.

    sources      each [..., H], same shape and dtype (bank first, prefix last — order is free,
                 the mix is permutation-invariant)
    query        pseudo-query, [1, H] (an ``nn.Linear(H, 1).weight``) or [H]
    norm_weight  key RMSNorm weight γ, [H]; ``eps`` is its epsilon (``rms_norm_eps``)

    Returns [..., H] in the sources' dtype. Math runs in fp32 (fp64 when any input is fp64) with
    autocast off, in HF ``_apply_attn_res``'s op order — bitwise equal to it on fp32/bf16 CPU inputs.
    One source is returned as is: its softmax weight is exactly 1, and HF/vLLM skip the call.
    """
    if len(sources) == 0:
        raise ValueError("attn_res_mix needs at least one source")
    first = sources[0]
    hidden = first.shape[-1]
    if any(v.shape != first.shape or v.dtype != first.dtype for v in sources):
        raise ValueError("every AttnRes source must share one shape and dtype")
    if query.numel() != hidden or norm_weight.shape != (hidden,):
        raise ValueError(
            f"query must hold {hidden} values and norm_weight be [{hidden}]; "
            f"got {tuple(query.shape)}, {tuple(norm_weight.shape)}"
        )
    if len(sources) == 1:
        return first

    dtype = _math_dtype(first, query, norm_weight)
    with torch.autocast(first.device.type, enabled=False):
        v = torch.stack(tuple(sources), dim=-2).to(dtype)  # [..., n, H]
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        w = norm_weight.to(dtype) * query.reshape(hidden).to(dtype)  # γ ⊙ w, folded once
        p = (k * w).sum(-1).softmax(-1)  # [..., n], softmax over depth
        out = torch.matmul(p.unsqueeze(-2), v).squeeze(-2)  # [..., H]
    return out.to(first.dtype)


@dataclass
class AttnResState:
    """Depth sources of one forward pass: committed block sums + the running block.

    ``bank`` holds the committed blocks (``bank[0]`` is the embedding once layer 0 has run);
    ``prefix`` is the running sum of the current block (the embedding itself before layer 0).
    The per-layer calls, with ``layer_idx`` 0-based::

        state = AttnResState.start(embedding)
        for layer_idx, layer in enumerate(layers):
            h = attn(input_layernorm(attn_res_mix(state.sources(), w_attn, γ_attn, eps)))
            state.after_attn(h, layer_idx, block_size)
            h = mlp(post_attention_layernorm(attn_res_mix(state.sources(), w_mlp, γ_mlp, eps)))
            state.after_mlp(h)
        final = attn_res_mix(state.sources(), w_out, γ_out, eps)

    The stream keeps the embedding's dtype: each sublayer output is cast to it on the way in, so
    the bank never holds two dtypes (``attn_res_mix`` takes one). Under bf16 autocast
    ``nn.Embedding`` returns fp32 but ``nn.Linear`` returns bf16; a plain residual ``x + h``
    keeps that fp32 stream by promotion, and so does this (bf16 → fp32 is exact). In uniform
    precision (fp32, bf16 weights, fp64) the cast is a no-op that returns ``h`` itself.

    Updates rebind ``bank`` and ``prefix`` to new objects and never write into a tensor, so
    autograd sees every sublayer output, and a list returned by ``sources()`` is never changed
    by a later update. The state itself is mutable: under activation checkpointing, pass
    ``sources()`` (tensors) into the recomputed region, not this object.
    """

    bank: list[Tensor]
    prefix: Tensor

    @classmethod
    def start(cls, embedding: Tensor) -> AttnResState:
        """State before layer 0: no committed block, the embedding is the prefix."""
        return cls(bank=[], prefix=embedding)

    def sources(self) -> list[Tensor]:
        """``bank + [prefix]`` as a fresh list — the inputs of the next ``attn_res_mix``."""
        return [*self.bank, self.prefix]

    def after_attn(self, h_attn: Tensor, layer_idx: int, block_size: int) -> None:
        """Fold layer ``layer_idx``'s attention output in; ``layer_idx % block_size == 0`` first
        commits the prefix (HF :995-998, vLLM ``is_block_write_layer`` :977)."""
        opens_block = layer_idx % block_size == 0
        if not self.bank and not opens_block:
            raise ValueError(
                f"layer {layer_idx} is the first to run but does not open a block — "
                "layer_idx is 0-based (model.py layer_id - 1)"
            )
        h_attn = h_attn.to(self.prefix.dtype)
        if opens_block:
            self.bank = [*self.bank, self.prefix]
            self.prefix = h_attn
        else:
            self.prefix = self.prefix + h_attn

    def after_mlp(self, h_mlp: Tensor) -> None:
        """Fold the mlp output in; it always joins the block its attention output is in."""
        self.prefix = self.prefix + h_mlp.to(self.prefix.dtype)
