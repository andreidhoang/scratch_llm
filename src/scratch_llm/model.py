"""Decoder-only Transformer LM, from scratch — the policy we own end-to-end.

A1 substrate. 2026-default decoder: pre-norm, RMSNorm, RoPE, SwiGLU, GQA-ready
multi-head attention, no biases. Owning the model end-to-end — every logit, every mask —
is what lets you inspect, debug, and trust everything built on top of it.

Correctness invariants (tested in tests/test_model.py):
- **Loss at init:** a fresh LM's cross-entropy on random data is ≈ ``log(vocab_size)``
  (uniform prediction). Off by more than a small band ⇒ head/embedding/norm bug.
- **Causal no-leak:** perturbing future tokens must not change the logits at earlier
  positions (the autoregressive contract).
- **RoPE relative:** the q·k score after RoPE depends only on the position offset i−j.

Design decisions:
- GQA via ``n_kv_heads`` (ADR-0002): full MHA is the special case ``n_kv_heads == n_heads``.
- Untied embeddings by default (ADR-0004): keeps the loss-at-init derivation clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from scratch_llm.moe import AuxOutput, MoEConfig, MoEStats


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters. ``n_kv_heads=None`` ⇒ full MHA; ``d_ff=None`` ⇒
    round (8/3)·d_model to a multiple of 64 (the SwiGLU convention)."""

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int | None = None
    d_ff: int | None = None
    context_length: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    qk_norm: bool = (
        False  # RMSNorm on q,k per head before RoPE — bounds attention logits (Qwen3/Gemma3/OLMo2)
    )
    moe: MoEConfig | None = None  # None ⇒ dense SwiGLU (the default); else opt-in MoE

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        if self.n_heads % self.kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} not divisible by n_kv_heads={self.kv_heads} "
                "(query heads must group evenly over kv heads)"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")

    @property
    def kv_heads(self) -> int:
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        raw = int(8 / 3 * self.d_model)
        return ((raw + 63) // 64) * 64  # round up to a multiple of 64


def _trunc_normal_linear_weight(out_features: int, in_features: int) -> nn.Parameter:
    """Weight stored as (out, in); init ~ N(0, 2/(in+out)) truncated at ±3σ."""
    weight = torch.empty(out_features, in_features)
    std = math.sqrt(2.0 / (in_features + out_features))
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    return nn.Parameter(weight)


class Linear(nn.Module):
    """y = x @ Wᵀ, no bias. Weight is (out_features, in_features)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = _trunc_normal_linear_weight(out_features, in_features)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """Token-id → vector lookup. Weight is (vocab_size, d_model), init ~ N(0, 1) at ±3σ."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. Computed in fp32 then cast back (precision matters at
    long context / low precision); learnable per-channel gain, no centering, no bias."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 * rms * self.weight.float()).to(dtype)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax (subtract the max before exp)."""
    x = x - x.amax(dim=dim, keepdim=True)
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
) -> Tensor:
    """softmax(QKᵀ/√d_k + mask) V. ``mask`` is boolean, True = attend (disallowed → −∞).
    Softmax is done in fp32 for stability."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores.float(), dim=-1).to(q.dtype)
    return attn @ v


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: rotate each 2-D coordinate pair (x_2k, x_2k+1) by angle pos·θ^(−2k/d).

    cos/sin are a precomputed, non-persistent buffer (recomputed on load, never
    checkpointed) sliced by the actual token positions — so a KV-cache that feeds one
    token at position t rotates it correctly.
    """

    cos: Tensor  # registered buffers; annotated so the type checker sees Tensor, not Module
    sin: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        # x: (..., seq, head_dim); positions: (seq,)
        cos = self.cos[positions]  # (seq, head_dim/2)
        sin = self.sin[positions]
        shape = (1,) * (x.dim() - 2) + cos.shape  # broadcast over leading (batch, head) dims
        cos = cos.view(shape)
        sin = sin.view(shape)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(x.dtype)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Gated FFN: W2( SiLU(W1 x) ⊙ W3 x ). No biases."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)  # gate
        self.w3 = Linear(d_model, d_ff)  # up
        self.w2 = Linear(d_ff, d_model)  # down

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class KVCache:
    """Per-layer K/V cache for incremental decoding (see docs/design/L2_kv_cache_SPEC.md).

    Stores GQA-native K, V — shape (B, n_kv_heads, T_cached, head_dim) per layer — so the cache
    is n_heads/n_kv_heads× smaller; the repeat to n_heads happens at attention time. ``length`` is
    the number of positions cached so far (identical across layers); it is bumped once per forward
    via :meth:`advance`, after every layer has appended, so all layers see the same start position.
    """

    def __init__(self, n_layers: int) -> None:
        self._k: list[Tensor | None] = [None] * n_layers
        self._v: list[Tensor | None] = [None] * n_layers
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    def get(self, layer: int) -> tuple[Tensor, Tensor] | None:
        k, v = self._k[layer], self._v[layer]
        if k is None or v is None:
            return None
        return k, v

    def append(self, layer: int, k_new: Tensor, v_new: Tensor) -> None:
        """Concatenate this step's K, V onto the layer's cache along the sequence axis."""
        existing = self.get(layer)
        if existing is None:
            self._k[layer], self._v[layer] = k_new, v_new
        else:
            past_k, past_v = existing
            self._k[layer] = torch.cat([past_k, k_new], dim=2)
            self._v[layer] = torch.cat([past_v, v_new], dim=2)

    def advance(self, n: int) -> None:
        self._length += n


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE and GQA. Heads are a batch dim."""

    def __init__(self, cfg: ModelConfig, rope: RotaryPositionalEmbedding) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.rope = rope
        self.q_proj = Linear(cfg.d_model, self.n_heads * self.head_dim)
        self.k_proj = Linear(cfg.d_model, self.n_kv * self.head_dim)
        self.v_proj = Linear(cfg.d_model, self.n_kv * self.head_dim)
        self.o_proj = Linear(self.n_heads * self.head_dim, cfg.d_model)
        # QK-norm (Qwen3/Gemma3/OLMo2): RMSNorm each head's q,k over head_dim *before* RoPE, so the
        # attention logit q·kᵀ/√d cannot run away — the #1 large-model bf16 instability. When off,
        # these are nn.Identity ⇒ the path is byte-identical to the no-QK-norm model.
        self.q_norm: nn.Module = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm: nn.Module = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> Tensor:
        b, s, _ = x.shape
        # project, split into heads, make head a batch dim: (B, H, S, head_dim)
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)

        # QK-norm before RoPE (no-op unless cfg.qk_norm): unit-RMS each head's q,k over head_dim.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE rotates Q, K at their ABSOLUTE positions — so a cached token at position t is
        # rotated correctly even when only one new token is processed this step.
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        if cache is not None and layer_idx is not None:
            past_len = cache.length  # constant across layers within one forward
            cache.append(layer_idx, k, v)
            full = cache.get(layer_idx)
            assert full is not None  # just appended
            k, v = full  # K, V now span [past ; new]
        else:
            past_len = 0

        if self.n_kv != self.n_heads:  # GQA: each kv head serves a group of query heads
            repeats = self.n_heads // self.n_kv
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        # Causal mask over absolute positions: query row i (abs past_len+i) may attend key j iff
        # j <= past_len+i. A single new token (s==1) attends all cached keys → no mask needed.
        if s == 1:
            mask = None
        else:
            total = past_len + s
            q_pos = torch.arange(past_len, total, device=x.device).unsqueeze(1)  # (s, 1)
            k_pos = torch.arange(total, device=x.device).unsqueeze(0)  # (1, total)
            mask = k_pos <= q_pos  # (s, total) bool, True = attend

        out = scaled_dot_product_attention(q, k, v, mask)  # (B, H, S, head_dim)
        out = out.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(RMSNorm(x)); x + FFN(RMSNorm(x)).

    The FFN is dense ``SwiGLU`` unless the model opts into MoE *and* this layer is past the
    leading dense layers (``cfg.moe.n_dense_layers``) — then it is a :class:`MoEFeedForward`.
    A MoE FFN returns its contribution plus per-layer stats; ``forward`` surfaces those stats
    (``None`` for a dense layer) so the LM can aggregate them. The residual add is identical
    either way — the FFN returns the *delta only*."""

    def __init__(self, cfg: ModelConfig, rope: RotaryPositionalEmbedding, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg, rope)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.is_moe = cfg.moe is not None and layer_idx >= cfg.moe.n_dense_layers
        self.ffn: nn.Module
        if self.is_moe:
            assert cfg.moe is not None
            from scratch_llm.moe import MoEFeedForward

            self.ffn = MoEFeedForward(cfg.d_model, cfg.moe)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_dim)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> tuple[Tensor, MoEStats | None]:
        x = x + self.attn(self.attn_norm(x), positions, cache, layer_idx)
        if self.is_moe:
            delta, stats = self.ffn(self.ffn_norm(x))
        else:
            delta, stats = self.ffn(self.ffn_norm(x)), None
        x = x + delta
        return x, stats


class TransformerLM(nn.Module):
    """Full decoder LM: embed → N pre-norm blocks → final RMSNorm → LM head.

    forward(token_ids: (B, S) long) → logits (B, S, vocab_size).

    Pass a :class:`KVCache` for incremental decoding: positions continue from the cache length,
    and K/V are appended per layer. With ``cache=None`` this is the full (training) forward,
    unchanged. The cached path must produce logits identical to the full recompute.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = Embedding(cfg.vocab_size, cfg.d_model)
        # One RoPE instance shared across blocks (buffers are identical, saves memory).
        rope = RotaryPositionalEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, rope, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = Linear(cfg.d_model, cfg.vocab_size)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    def forward(
        self, token_ids: Tensor, cache: KVCache | None = None, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, AuxOutput]:
        """``return_aux=False`` (default, and the decode path) returns just logits — identical
        to the dense build. ``return_aux=True`` returns ``(logits, AuxOutput)`` with the summed
        MoE aux/z losses and per-layer routing diagnostics for the training objective."""
        s = token_ids.shape[1]
        start = 0 if cache is None else cache.length
        positions = torch.arange(start, start + s, device=token_ids.device)
        x = self.token_emb(token_ids)
        layer_stats: list[MoEStats] = []
        for layer_idx, block in enumerate(self.blocks):
            x, stats = block(x, positions, cache, layer_idx)
            if stats is not None:
                layer_stats.append(stats)
        x = self.final_norm(x)
        if cache is not None:
            cache.advance(s)  # bump once, after all layers, so each layer saw the same start
        logits = self.lm_head(x)
        if return_aux:
            return logits, self._aggregate_aux(layer_stats)
        return logits

    def _aggregate_aux(self, layer_stats: list[MoEStats]) -> AuxOutput:
        """Sum the MoE aux/z losses across layers (zeros on a dense model)."""
        from scratch_llm.moe import AuxOutput

        if layer_stats:
            aux_loss = torch.stack([s.aux_loss for s in layer_stats]).sum()
            z_loss = torch.stack([s.z_loss for s in layer_stats]).sum()
        else:
            device = self.lm_head.weight.device
            aux_loss = torch.zeros((), device=device)
            z_loss = torch.zeros((), device=device)
        return AuxOutput(aux_loss=aux_loss, z_loss=z_loss, layers=layer_stats)

    def moe_update_biases(self) -> None:
        """Aux-loss-free balancing step for every MoE layer — call after ``optimizer.step()``."""
        from scratch_llm.moe import MoEFeedForward

        for block in self.blocks:
            ffn = block.ffn  # type: ignore[union-attr]
            if isinstance(ffn, MoEFeedForward):
                ffn.router.update_bias(ffn.cfg.bias_update_speed)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean token cross-entropy from logits, computed via logsumexp (cancels log∘exp).

    logits: (..., vocab_size); targets: (...) long. At init (≈uniform logits) this is
    ≈ log(vocab_size).
    """
    logits = logits.float()
    log_z = torch.logsumexp(logits, dim=-1)
    chosen = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_z - chosen).mean()
