"""k3.model — Kimi K3 assembled: embedding, hybrid layers under Block AttnRes, norm, lm_head.

Architecture (config.json): 93 layers, three KDA then one Gated-MLA per group of four plus a
terminal MLA (69 KDA : 24 MLA, ``build_layer_pattern``). Layer 1 has a dense SiTU MLP
(``first_k_dense_replace = 1``), layers 2..93 a LatentMoE: attention type and FFN type dispatch
independently on the 1-based ``layer_id``. No positional encoding anywhere: KDA's data-dependent
decay carries position, and MLA's "rope" dims are live but never rotated (core/gated_mla.py).

Dataflow. There is no plain residual stream x ← x + f(x). Every sublayer reads a learned mix over
depth of the embedding, the committed block sums and the running sum of the current block
(core/attn_res.py). With ℓ = layer_id − 1, B = ``attn_res_block_size`` counted in layers, and
mix(w, γ) = ``attn_res_mix(res.sources(), w, γ, rms_norm_eps)``:

    res = AttnResState.start(embed_tokens(ids))              bank = [], prefix = embedding
    for ℓ in 0 .. L−1:
        h = self_attn(input_layernorm(mix(w_attn_ℓ, γ_attn_ℓ)))
        res.after_attn(h, ℓ, B)                               ℓ % B == 0: commit prefix; prefix = h
        h = ffn(post_attention_layernorm(mix(w_mlp_ℓ, γ_mlp_ℓ)))
        res.after_mlp(h)                                      prefix += h
    logits = lm_head(norm(mix(w_out, γ_out)))

w_* are the ``*_res_proj`` weights (nn.Linear(H, 1): one pseudo-query each), γ_* the
``*_res_norm`` weights. The layer-0 attention mix has one source (the embedding) and returns it
unchanged, so that query gets no gradient. Sources: HF ``KimiDecoderLayer._forward_attn_residual``
:973-1046, ``KimiLinearModel.forward`` :1188-1219, ``_apply_output_attn_res`` :1226-1233,
``KimiLinearForCausalLM`` lm_head after ``model.norm`` :1301; vLLM ``nvidia/model.py``
``_pre_attn_norm``/``_post_attn_norm`` :1017-1079, model tail :1402, ``compute_logits`` :1705-1712.

Decode. :class:`HybridState` holds one constant-size ``KDAState`` per KDA layer and one growing
``MLALatentKV`` per MLA layer, slot ``layer_id − 1``; each attention module reads and commits its
own slot. AttnRes needs no cache: it mixes over depth, one token at a time, so a decoded token's
sources are rebuilt from its own layer outputs. ``forward(ids, state)`` continues the streams in
``state`` and returns the same logits as one prefill over the whole sequence (fp64 gate).

Checkpoint names are HF's with the ``language_model.model.`` prefix (``language_model.`` for
``lm_head``) stripped: ``embed_tokens``, ``layers.{i}.{self_attn, mlp | block_sparse_moe,
input_layernorm, post_attention_layernorm, self_attention_res_{norm,proj}, mlp_res_{norm,proj}}``,
``norm``, ``output_attn_res_{norm,proj}``, ``lm_head``. ``lm_head`` is tied to ``embed_tokens``
only if ``tie_word_embeddings`` (false for K3).

Init = HF ``KimiPreTrainedModel._init_weights`` :1063-1073, run once on the finished tree:
normal(0, ``initializer_range``) for every ``nn.Linear`` weight, which includes the AttnRes
pseudo-queries, and for the embedding. Everything else keeps its constructor init, because none
of it is an ``nn.Linear``: the router ``gate.weight`` keeps kaiming_uniform(a=√5) (HF
``KimiMoEGate.reset_parameters``), the Quantile-Balancing bias stays 0, KDA keeps ``KDALayer``'s
A_log / dt_bias / short-conv init, and every norm weight stays 1. Those constructors are ours,
not HF's: HF draws A_log = log U(1, 16) (:520-521) and leaves dt_bias and the QB bias
uninitialized (``torch.empty``, :526-527 and :693-695); K3 takes FLA's safe_gate init for the
first two (A_log = 0, FLA's dt_bias) and a zero bias, by choice. The pseudo-query init is a
constructor choice (``attn_res_query_init``): "normal" is HF and matches the checkpoint census
(layer-0 query ≈ 4e-6, not 0); "zero" is the AttnRes paper / FLA arm, where every mix starts as
a uniform mean over its sources.

Not implemented: HF's ``padding_idx`` on the embedding (config.json ``pad_token_id`` 163839:
that row starts at 0 and gets no gradient; ``K3Config`` carries no pad id), padding masks and
packed sequences, MTP heads, the vision tower.

Gates: ``tests/test_k3_model.py`` — ≡ an HF-transcribed forward sharing one state dict by name
(fp64, logits and every gradient), prefill ≡ decode ≡ split prefill, causality, loss at init vs a
derived prediction, overfit one batch, parameter count vs ``param_count``, init, AttnRes wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from scratch_llm.k3.config import K3Config, k3_full, mini_k3_d12
from scratch_llm.k3.core.attn_res import AttnResState, attn_res_mix
from scratch_llm.k3.core.gated_mla import GatedMLA, MLALatentKV
from scratch_llm.k3.core.kda import KDALayer, KDAState
from scratch_llm.k3.core.latent_moe import LatentMoE, QBStats
from scratch_llm.k3.core.norm import RMSNorm
from scratch_llm.k3.core.situ import DenseSiTUMLP

__all__ = ["HybridState", "K3Aux", "K3Block", "K3Model", "build_k3", "build_mini_k3"]


@dataclass
class HybridState:
    """K3's decode cache, one slot per layer (index ``layer_id − 1``): a ``KDAState`` in every
    KDA slot, an ``MLALatentKV`` in every MLA slot, None in the other list. A None slot of the
    right kind reads as empty, which is also what :meth:`empty` allocates.

    ``lengths`` [B] counts the tokens each row has consumed, bumped by T per forward. It is
    bookkeeping for the caller: no layer reads it (K3 has no positions, and each attention module
    sizes its own slot), so where in the forward the bump happens does not matter.
    """

    kda_states: list[KDAState | None]
    mla_caches: list[MLALatentKV | None]
    lengths: Tensor

    @classmethod
    def empty(
        cls, cfg: K3Config, batch: int, device: torch.device, dtype: torch.dtype
    ) -> HybridState:
        """A fresh stream: zero KDA state and conv history, empty latent caches, lengths 0.
        ``dtype`` is the activation dtype (the KDA state itself is fp32, or fp64 for fp64)."""
        layer_ids = range(1, cfg.num_layers + 1)
        return cls(
            kda_states=[
                KDAState.zeros(cfg.kda, batch, device, dtype) if i in cfg.kda_layers else None
                for i in layer_ids
            ],
            mla_caches=[
                MLALatentKV.empty(cfg.mla, batch, device, dtype) if i in cfg.mla_layers else None
                for i in layer_ids
            ],
            lengths=torch.zeros(batch, dtype=torch.long, device=device),
        )


@dataclass
class K3Aux:
    """``forward(..., return_aux=True)``'s second output: the auxiliary loss, which K3 does not
    have. Quantile Balancing sets the router bias by assignment after the optimizer step
    (:meth:`K3Model.moe_update_biases`), not through a loss, so ``total`` is a 0-d zero on the
    logits' device and dtype and ``cross_entropy + aux.total`` is the cross-entropy exactly.

    The balancing diagnostics are not here: routing statistics accumulate on each LatentMoE
    between forward and update, and ``moe_update_biases`` returns them (``QBStats.load_fraction``).
    """

    total: Tensor


def _mix(sources: list[Tensor], proj: nn.Linear, norm: RMSNorm) -> Tensor:
    """One AttnRes site: pseudo-query ``proj.weight`` [1, H], key norm ``norm`` (weight, eps)."""
    return attn_res_mix(sources, proj.weight, norm.weight, norm.eps)


class K3Block(nn.Module):
    """Decoder layer ``layer_id`` (1-based): attention sublayer, FFN sublayer, and their four norms
    and two AttnRes sites. ``forward`` advances an ``AttnResState`` by one layer.

    ``self_attn`` is ``KDALayer`` on KDA layers, ``GatedMLA`` on MLA layers. The FFN is ``mlp``
    (``DenseSiTUMLP``) while ``layer_id − 1 < first_k_dense``, else ``block_sparse_moe``
    (``LatentMoE``) — HF's attribute names, so exactly one of the two exists.
    """

    def __init__(self, cfg: K3Config, layer_id: int) -> None:
        super().__init__()
        is_kda, is_mla = layer_id in cfg.kda_layers, layer_id in cfg.mla_layers
        if is_kda == is_mla:
            raise ValueError(
                f"layer {layer_id} must be exactly one of KDA / MLA "
                f"(kda={is_kda}, mla={is_mla}); check build_layer_pattern"
            )
        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.layer_id = layer_id
        self.attn_res_block_size = cfg.attn_res_block_size
        self.is_moe = layer_id in cfg.moe_layers  # layer_id − 1 >= first_k_dense (HF :893-897)

        # HF :539-540: KDA's gated output norm takes rms_norm_eps. MLA's latent norms keep
        # KimiRMSNorm's default 1e-6 (GatedMLA's own default).
        self.self_attn: KDALayer | GatedMLA = (
            KDALayer(cfg.kda, H, norm_eps=eps) if is_kda else GatedMLA(cfg.mla, H)
        )
        if self.is_moe:
            self.block_sparse_moe = LatentMoE(cfg.moe, H, cfg)
        else:
            self.mlp = DenseSiTUMLP(H, cfg.moe.dense_intermediate, cfg)
        self.input_layernorm = RMSNorm(H, eps)
        self.post_attention_layernorm = RMSNorm(H, eps)
        self.self_attention_res_norm = RMSNorm(H, eps)
        self.mlp_res_norm = RMSNorm(H, eps)
        self.self_attention_res_proj = nn.Linear(H, 1, bias=False)
        self.mlp_res_proj = nn.Linear(H, 1, bias=False)

    def forward(self, res: AttnResState, state: HybridState | None = None) -> None:
        """One layer of the module-docstring loop; advances ``res`` in place. The attention module
        reads and commits ``state``'s slot ``layer_id − 1`` (``state=None``: training prefill)."""
        x = _mix(res.sources(), self.self_attention_res_proj, self.self_attention_res_norm)
        h = self.self_attn(self.input_layernorm(x), state, self.layer_id)
        res.after_attn(h, self.layer_id - 1, self.attn_res_block_size)
        x = _mix(res.sources(), self.mlp_res_proj, self.mlp_res_norm)
        ffn = self.block_sparse_moe if self.is_moe else self.mlp
        res.after_mlp(ffn(self.post_attention_layernorm(x)))


class K3Model(nn.Module):
    """K3 / mini-K3: ``forward(token_ids [B, T] long, state=None) -> logits [B, T, vocab]``.

    ``state=None`` is the training prefill. With a :class:`HybridState` the forward continues
    each row's stream from the cache, updates every slot and bumps ``state.lengths`` by T.
    ``return_aux=True`` returns ``(logits, K3Aux)`` for ``train()``'s MoE branch.

    ``attn_res_query_init``: "normal" (HF, default) or "zero" (AttnRes paper / FLA) for the
    ``2 · num_layers + 1`` pseudo-queries; module docstring. Construction under
    ``torch.device("meta")`` builds shapes only and skips the value init.
    """

    def __init__(
        self, cfg: K3Config, *, attn_res_query_init: Literal["normal", "zero"] = "normal"
    ) -> None:
        super().__init__()
        if attn_res_query_init not in ("normal", "zero"):
            raise ValueError(
                f"attn_res_query_init must be 'normal' or 'zero', got {attn_res_query_init!r}"
            )
        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, H)
        blocks = [K3Block(cfg, i) for i in range(1, cfg.num_layers + 1)]
        self.layers = nn.ModuleList(blocks)
        self.norm = RMSNorm(H, eps)
        self.output_attn_res_norm = RMSNorm(H, eps)
        self.output_attn_res_proj = nn.Linear(H, 1, bias=False)
        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        if not self.embed_tokens.weight.is_meta:  # meta: shapes only, no values to draw
            self._init_weights()
        if attn_res_query_init == "zero":
            queries = [q for b in blocks for q in (b.self_attention_res_proj, b.mlp_res_proj)]
            for query in (*queries, self.output_attn_res_proj):
                nn.init.zeros_(query.weight)

    @torch.no_grad()
    def _init_weights(self) -> None:
        """HF ``_init_weights`` (:1063-1073) over the whole tree; see the module docstring for
        what it leaves alone and why."""
        std = self.cfg.initializer_range
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self, token_ids: Tensor, state: HybridState | None = None, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, K3Aux]:
        res = AttnResState.start(self.embed_tokens(token_ids))
        for block in self.layers:
            block(res, state)
        x = _mix(res.sources(), self.output_attn_res_proj, self.output_attn_res_norm)
        logits = self.lm_head(self.norm(x))
        if state is not None:
            state.lengths = state.lengths + token_ids.shape[1]
        if return_aux:
            return logits, K3Aux(total=logits.new_zeros(()))
        return logits

    def moe_update_biases(self) -> dict[str, QBStats]:
        """Quantile Balancing on every LatentMoE (``LatentMoE.update_bias``); call once after
        ``optimizer.step()``. Returns the consumed statistics by module name
        (``layers.1.block_sparse_moe``, …); a layer with no training forward since the last call
        is absent and keeps its bias."""
        stats: dict[str, QBStats] = {}
        for name, module in self.named_modules():
            if isinstance(module, LatentMoE) and (s := module.update_bias()) is not None:
                stats[name] = s
        return stats


def build_k3() -> K3Model:
    """The released checkpoint's text architecture (``k3_full()``); 2.78T parameters — build it
    under ``torch.device("meta")`` unless you mean it."""
    return K3Model(k3_full())


def build_mini_k3() -> K3Model:
    """The d12 miniature (``mini_k3_d12()``), 370M parameters."""
    return K3Model(mini_k3_d12())
