"""A1 Rung 4.5 — Multi-head Latent Attention (MLA), toy scale.

MLA (DeepSeek-V2/V3) is the 2026 KV story: instead of caching K,V per head, cache a single low-rank
**latent** ``c_KV`` (dim ``d_latent``, e.g. 512) per token plus a small **decoupled-RoPE** key
``k_R`` (dim ``d_rope``, e.g. 64) shared across heads. R1's cache is ``(512+64) × 61 layers × 2 B =
70.3 KB/token`` — 4.7× smaller than a Llama-70B GQA-8's 327.7 KB, so a 671B model has a *smaller*
per-token cache than a 70B (ADR-0012). Long-context serving economics fall out of the architecture.

**The weight-absorption identity (why decode is cheap):** the content score is
``q_c · K^C = q_c · (W_UK c_KV) = (W_UK^T q_c) · c_KV`` — so fold ``W_UK`` into the query and attend
directly in the ``d_latent`` space, never reconstructing per-head K. Likewise the output
``Σ a_j V_j = W_UV (Σ a_j c_KV_j)`` folds ``W_UV`` into the output path. Decode then reads only
``c_KV`` (+ ``k_R``) from cache — never the full K,V. This ONLY works because RoPE is **decoupled**
into the separate ``d_rope`` pathway: a position-dependent rotation on the content K could not be
folded into the static ``W_UK`` (that is the whole reason DeepSeek split it out).

Correctness (tests/test_mla.py): the absorbed latent-space attention is NUMERICALLY IDENTICAL to the
naive "reconstruct K,V per head then attend" path (float64) — the identity that lets an engine cache
the latent instead of K,V with zero quality change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from scratch_llm.model import Linear, RotaryPositionalEmbedding


@dataclass(frozen=True)
class MLAConfig:
    d_model: int
    n_heads: int
    d_head: int  # content head dim (no RoPE)
    d_latent: int  # c_KV latent dim (the cached content, d_c)
    d_rope: int  # decoupled-RoPE dim per token (the cached rotary key, d_r); must be even
    max_seq_len: int = 4096
    rope_theta: float = 10000.0

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """MLA caches the latent + the shared rotary key: ``(d_latent + d_rope)`` values/token."""
        return (self.d_latent + self.d_rope) * dtype_bytes

    def mha_kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """The full-MHA baseline: 2 × n_heads × d_head values/token."""
        return 2 * self.n_heads * self.d_head * dtype_bytes

    def gqa_kv_bytes_per_token(self, n_kv_heads: int, dtype_bytes: int = 2) -> int:
        return 2 * n_kv_heads * self.d_head * dtype_bytes


class MultiHeadLatentAttention(nn.Module):
    """Toy MLA: single (B, S, d_model) causal attention with both the naive-reconstruct and the
    weight-absorbed latent-space paths, so the identity can be checked directly."""

    def __init__(self, cfg: MLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H, dh, dc, dr = cfg.n_heads, cfg.d_head, cfg.d_latent, cfg.d_rope
        if dr % 2 != 0:
            raise ValueError("d_rope must be even (RoPE rotates coordinate pairs)")
        # content pathway (no RoPE): query, latent down-proj, and per-head K/V up-projs
        self.q_c = Linear(cfg.d_model, H * dh)
        self.down_kv = Linear(cfg.d_model, dc)  # h → c_KV (the cached latent)
        self.up_k = Linear(dc, H * dh)  # c_KV → K^C per head (W_UK)
        self.up_v = Linear(dc, H * dh)  # c_KV → V per head (W_UV)
        # decoupled-RoPE pathway: per-head query rope part + a per-token shared key rope part
        self.q_r = Linear(cfg.d_model, H * dr)
        self.k_r = Linear(cfg.d_model, dr)  # shared across heads (the cached rotary key)
        self.o_proj = Linear(H * dh, cfg.d_model)
        self.rope = RotaryPositionalEmbedding(dr, cfg.max_seq_len, cfg.rope_theta)
        self.scale = (dh + dr) ** -0.5

    def _project(self, h: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Shared projections. Returns q_c (B,H,S,dh), q_rope (B,H,S,dr rotated),
        c_kv (B,S,dc), k_rope (B,1,S,dr rotated)."""
        b, s, _ = h.shape
        H, dh, dr = self.cfg.n_heads, self.cfg.d_head, self.cfg.d_rope
        q_c = self.q_c(h).view(b, s, H, dh).transpose(1, 2)  # (B,H,S,dh)
        q_rope = self.q_r(h).view(b, s, H, dr).transpose(1, 2)  # (B,H,S,dr)
        q_rope = self.rope(q_rope, positions)
        c_kv = self.down_kv(h)  # (B,S,dc) — the cached latent
        k_rope = self.k_r(h).view(b, s, 1, dr).transpose(1, 2)  # (B,1,S,dr) shared over heads
        k_rope = self.rope(k_rope, positions)
        return q_c, q_rope, c_kv, k_rope

    def _causal_mask(self, s: int, device: torch.device) -> Tensor:
        return torch.tril(torch.ones(s, s, dtype=torch.bool, device=device))

    def forward_naive(self, h: Tensor, positions: Tensor) -> Tensor:
        """Reconstruct K,V per head from the latent, then standard attention (the oracle)."""
        b, s, _ = h.shape
        H, dh, dr = self.cfg.n_heads, self.cfg.d_head, self.cfg.d_rope
        q_c, q_rope, c_kv, k_rope = self._project(h, positions)
        k_c = self.up_k(c_kv).view(b, s, H, dh).transpose(1, 2)  # (B,H,S,dh) reconstructed K
        v = self.up_v(c_kv).view(b, s, H, dh).transpose(1, 2)  # (B,H,S,dh) reconstructed V
        k_rope_h = k_rope.expand(b, H, s, dr)  # share the rotary key across heads
        # score = q_c·K^C + q_rope·k_R, scaled; causal
        scores = (
            torch.matmul(q_c, k_c.transpose(-1, -2))
            + torch.matmul(q_rope, k_rope_h.transpose(-1, -2))
        ) * self.scale
        mask = self._causal_mask(s, h.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,S,dh)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, H * dh))

    def forward_absorbed(self, h: Tensor, positions: Tensor) -> Tensor:
        """Attend in latent space with W_UK folded into the query and W_UV into the output — reads
        only c_KV and k_R, never the reconstructed K,V (the decode-time cache saving)."""
        b, s, _ = h.shape
        H, dh, dc, dr = self.cfg.n_heads, self.cfg.d_head, self.cfg.d_latent, self.cfg.d_rope
        q_c, q_rope, c_kv, k_rope = self._project(h, positions)
        # absorb W_UK into q_c: q_absorbed_h = q_c_h · W_UK_h → attend c_KV directly in latent space
        wk = self.up_k.weight.view(H, dh, dc)  # (H, dh, dc)
        q_abs = torch.einsum("bhsd,hdc->bhsc", q_c, wk)  # (B,H,S,dc)
        k_rope_h = k_rope.expand(b, H, s, dr)
        scores = (
            torch.matmul(q_abs, c_kv.unsqueeze(1).transpose(-1, -2))
            + torch.matmul(q_rope, k_rope_h.transpose(-1, -2))
        ) * self.scale
        mask = self._causal_mask(s, h.device)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        latent_out = torch.matmul(attn, c_kv.unsqueeze(1))  # (B,H,S,dc) — weighted sum of latents
        # apply W_UV per head to the latent output, then o_proj
        wv = self.up_v.weight.view(H, dh, dc)  # (H, dh, dc)
        out = torch.einsum("bhsc,hdc->bhsd", latent_out, wv)  # (B,H,S,dh)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, H * dh))

    def forward(self, h: Tensor, positions: Tensor, absorb: bool = True) -> Tensor:
        return self.forward_absorbed(h, positions) if absorb else self.forward_naive(h, positions)
