"""core/gated_mla.py — Gated Multi-head Latent Attention, NoPE (K3 rung, TEMPLATE).

TEMPLATE NOTICE (agent-authored scaffold, 2026-08-04). HANDCRAFTED.md Class 1 hand-built;
user waived the boundary for scaffolding. Scaffold only — docstring + signatures + source
anchors are reference; the weight-absorption identity and the gate wiring raise
``NotImplementedError`` for the human to author. Delete-test applies once PROVEN.

WHAT (FACTS A11; report §2.1.2, Eqs. 6–7; Gated Attention arXiv:2505.06708):
  MLA with a compressed KV latent (``kv_lora_rank``), decoupled-RoPE-optional (K3 is NoPE so
  RoPE is OFF), plus a FULL-RANK sigmoid output gate applied to the attention output before
  W_o. K3 is NoPE on ALL MLA layers — positions are carried by the KDA layers' recurrence
  (Kimi Linear §6.1: the gated delta rule is a data-dependent positional encoding), so MLA
  needs no positional encoding. ``qk_rope_head_dim=64`` in config is vestigial (FACTS A11):
  keep the parameter for checkpoint parity, do not apply RoPE.

TWO K3 DIFFS vs our mla.py (A1-era MLA with decoupled RoPE + absorption identity):
  1. NoPE: ``rotary_emb = None``, ``assert cfg.use_nope``. The absorption identity still holds
     (it's algebra, not positional) — the cache stores the latent, not the per-head K/V.
  2. Full-rank output gate: ``Y' = Y ⊙ σ(x W_θ)``, where x is the LAYER INPUT (pre-W_o,
     pre-norm), NOT the attention output Y. (Gated Attention, arXiv:2505.06708; report Eqs. 6–7.)

MASTERY BAR (HANDCRAFTED.md):
  - delete-test;
  - absorbed ≡ naive: the weight-absorption identity must produce bit-identical output to the
    materialized-attention path in float64;
  - gate-source probe: the sigmoid gate MUST read the layer input x_t (pre-W_o). Moving it
    post-W_o still trains, silently — the test must catch this (K2_PROPOSAL_KDA §3).

SILENT-BUG SURFACES:
  - wrong absorption fold (matmul order): looks correct, wastes the latent cache;
  - gate placement post-W_o: trains fine, loses the gating benefit silently;
  - applying RoPE under use_nope: doubles positional signal, subtle long-ctx regression.

INTERVIEW QUESTION: why gate from x_t (layer input) and not from Y (attention output)? —
the gate decides how much of the layer's OWN input to admit into the output stream; gating on
Y would let the gate react to attention noise rather than to the clean residual signal. The
gate is a skip-decision, so it reads the thing being skipped-around.

UPGRADE OF: scratch_llm.mla (MLA + decoupled RoPE + absorption identity).
Assembly swap-in (k3/model.py K3Block, MLA branch):
    from scratch_llm.k3.core.gated_mla import GatedMLA
    self.attn = GatedMLA(cfg.mla, cfg.hidden_size)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from scratch_llm.k3.config import MLAConfig

if TYPE_CHECKING:
    from scratch_llm.k3.model import HybridState


class MLALatentKV:
    """Latent KV cache for ONE MLA layer: the compressed ``kv_lora_rank`` latent only.

    The weight-absorption identity means we cache c_kv (the latent), never the materialized
    per-head K/V. Grows linearly with sequence length: at 1M tokens, 24 MLA layers × 512 × 2 B
    per token per layer (FACTS A11; ROADMAP K8 reproduces this table from config).
    """

    k_latent: Tensor  # (B, kv_lora_rank, T) — cached compressed key latent
    v_latent: Tensor  # (B, kv_lora_rank, T) — cached compressed value latent

    @classmethod
    def empty(
        cls, cfg: MLAConfig, batch: int, device: torch.device, dtype: torch.dtype
    ) -> MLALatentKV:
        # TODO(hand-build): allocate empty latent KV (grows at decode; prefill uses full chunk).
        raise NotImplementedError(
            "MLALatentKV.empty — hand-built (the cache contract is the silent-bug surface)."
        )


class GatedMLA(nn.Module):
    """One Gated-MLA layer, NoPE.

    The assembly (k3/model.py K3Block) calls::
        a = self.attn(x, state, layer_id)
    so ``forward`` reads/writes its slot in ``state.mla_caches[layer_id-1]`` on decode.
    """

    def __init__(self, cfg: MLAConfig, hidden_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        assert cfg.use_nope, "K3 is NoPE on all MLA layers (FACTS A11). Set use_nope=True."
        # TODO(hand-build): parameters per config.json MLA block —
        #     - q-a down/up projections (q_lora_rank), kv-a down/up (kv_lora_rank),
        #     - per-head q/k/v projections at nope/rope/v head dims (rope head dim VESTIGIAL —
        #       allocate for checkpoint parity, do not apply RoPE),
        #     - the FULL-RANK output gate W_θ (hidden_size → hidden_size; reads x_t, pre-W_o),
        #     - W_o output projection, scalar softmax-scale.
        raise NotImplementedError(
            "GatedMLA.__init__ — parameter layout is hand-built. Gate W_θ must be full-rank "
            "and read x_t (pre-W_o); the gate-source probe test enforces the source."
        )

    def forward(
        self,
        x: Tensor,  # (B, S, hidden) — block-normed input
        state: HybridState | None = None,  # HybridState on decode; None on training
        layer_id: int = 0,
    ) -> Tensor:
        """Returns (B, S, hidden), to be residual-added by the block.

        Path:
          c_q   = W_dq(x);      q        = W_uq(c_q)                       # query via lora
          c_kv  = W_dkv(x);     k,v      = W_ukv(c_kv split)               # cache c_kv at decode
          attn  = softmax(q kᵀ / √d) v                                     # NoPE — no rotary
          Y     = W_o(attn)
          Y'    = Y ⊙ σ(x W_θ)                                              # FULL-RANK gate, x_t pre-W_o

        The absorbed (latent-cached) path must equal the naive (materialized) path in float64.
        """
        # TODO(hand-build): the absorption identity is the silent-bug surface — derive it, then
        #   the gate placement. Both the fold and the gate source have tests that must pass.
        raise NotImplementedError(
            "GatedMLA.forward — hand-build the absorption identity + full-rank gate "
            "(absorbed ≡ naive float64; gate reads x_t pre-W_o)."
        )
