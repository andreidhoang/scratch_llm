"""F2a — multi-token-prediction (MTP) auxiliary training head, depth D=1.

DeepSeek-V3 (arXiv:2412.19437, §Multi-Token Prediction): at each position the MTP module
combines the trunk's hidden state h_i with the embedding of the NEXT token t_{i+1},
``RMSNorm(h) ⊕ RMSNorm(emb_next) → Linear(2d→d) → 1 dense TransformerBlock → RMSNorm``,
and reads out through the base model's SHARED ``lm_head`` to predict t_{i+2} — one token
further ahead than the main head. TRAINING ONLY: the head sits off the ``forward()`` /
decode path entirely, so inference (KV cache, speculative serving) is untouched.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor, nn

from scratch_llm.model import (
    Linear,
    ModelConfig,
    RMSNorm,
    RotaryPositionalEmbedding,
    TransformerBlock,
)


class MTPHead(nn.Module):
    """Depth-1 MTP module. Shares the base model's ``token_emb`` (emb_next) and
    ``lm_head`` (output projection) — references, not copies, exactly like
    ``tie_embeddings`` does, so the head adds only the proj + one dense block of params.

    The internal block is forced dense (``moe=None``): the trunk's router / aux-loss
    machinery must not leak into the aux head.
    """

    def __init__(self, cfg: ModelConfig, token_emb: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.norm_h = RMSNorm(cfg.d_model)
        self.norm_emb = RMSNorm(cfg.d_model)
        self.eh_proj = Linear(2 * cfg.d_model, cfg.d_model)
        rope = RotaryPositionalEmbedding(cfg.head_dim, cfg.context_length, cfg.rope_theta)
        block_cfg = dataclasses.replace(cfg, moe=None)  # the MTP block is always dense
        self.block = TransformerBlock(block_cfg, rope, layer_idx=cfg.n_layers)
        self.final_norm = RMSNorm(cfg.d_model)
        # Shared with the base model (the tie_embeddings precedent: one Parameter, two
        # state_dict names — checkpoints round-trip through both keys with equal values).
        self.token_emb = token_emb
        self.lm_head = lm_head

    def forward(self, h: Tensor, next_ids: Tensor) -> Tensor:
        """h: (B, S, d) trunk hidden states (post trunk final_norm); next_ids: (B, S) long
        — the one-step-shifted ids (the training targets). Returns logits (B, S, V) where
        position i predicts the token one step past next_ids[:, i] (i.e. t_{i+2})."""
        emb = self.token_emb(next_ids)
        x = self.eh_proj(torch.cat([self.norm_h(h), self.norm_emb(emb)], dim=-1))
        positions = torch.arange(h.shape[1], device=h.device)
        x, _ = self.block(x, positions)
        return self.lm_head(self.final_norm(x))
