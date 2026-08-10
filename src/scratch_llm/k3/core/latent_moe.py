"""core/latent_moe.py — Stable LatentMoE: sigmoid router + Quantile Balancing (K5, TEMPLATE).

TEMPLATE NOTICE (agent-authored scaffold, 2026-08-04). HANDCRAFTED.md Class 1 hand-built;
user waived the boundary for scaffolding. Scaffold only — the router, the latent dispatch,
and Quantile Balancing raise ``NotImplementedError`` for the human to author. Delete-test
applies once PROVEN.

WHAT (FACTS A3, A14; report §2.3.3 Eq. 14; LatentMoE arXiv:2601.18089):
  Stable LatentMoE per layer (layers > cfg.first_k_dense):
    1. down-project the residual x: hidden → latent_size (0.5× at full scale: 7168 → 3584);
    2. sigmoid router + learned bias → routing scores; top-k=16 routed experts selected
       (weights RENORMALIZED, bias EXCLUDED from the renormalized weight);
    3. routed experts operate AT LATENT WIDTH with SiTU-GLU;
    4. RMSNorm at latent width (pre-up);
    5. up-project: latent → hidden;
    6. ADD 2 full-width shared experts (dense, always-on, at hidden width, SiTU-GLU).
  Auxiliary-loss-free balancing — **Quantile Balancing** (FACTS A14, Eq. 14):
    b̂ ← −quantile_{1−k/n}(s − α)   (mean-removed, one-step delay, histogram-estimated over
    a single all-reduce). Bias FROZEN at inference. Replaces our F6 sign-step bias.

MASTERY BAR (HANDCRAFTED.md):
  - delete-test;
  - load convergence: target load fraction q = m·k/n reached WITHOUT an aux loss — the QB
    bias update alone balances the experts;
  - activation-bound test: no |x| > 100 post-SiTU anywhere in the expert path.

SILENT-BUG SURFACES:
  - wrong bias SIGN in the QB update: still trains, slow expert collapse (an expert dies
    per ~10k steps);
  - wrong bias DELAY (zero-step vs one-step): still trains, oscillates;
  - including the bias in the renormalized top-k weight: silently over-weights high-bias
    experts;
  - router-frozen-at-inference not actually frozen: drift at decode.

INTERVIEW QUESTION: why down-project BEFORE routing (latent MoE) instead of routing at full
width? — the experts then operate in a ~2× narrower space, so expert params scale 0.5× and
the all-gather/all-to-all dispatch traffic halves. The cost is a down/up projection pair;
the net win is ~2× expert FLOPs/byte for the same quality. The RMSNorm at latent width is
critical — without it the down-projection's scale collapse starves the experts.

UPGRADE OF: scratch_llm.moe + the F6 balancing harness (sign-step → Quantile Balancing).
Depends on: core/situ.py (SiTU-GLU) — build situ.py first.
Assembly swap-in (k3/model.py K3Block, MoE branch):
    from scratch_llm.k3.core.latent_moe import LatentMoE
    self.ffn = LatentMoE(cfg.moe, cfg.hidden_size, cfg)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from scratch_llm.k3.config import K3Config, MoEConfig


@dataclass
class QBStats:
    """Per-layer Quantile-Balancing diagnostics, surfaced to the trainer (aux-loss-free).

    The K3Block returns these so a post-step hook can update biases outside the autograd
    graph. Fields are filled in by the hand-build — the histogram, the mean-removed
    correction, and the per-expert load estimate.
    """

    # TODO(hand-build): the QB signal fields (histogram, mean, quantile estimate).


class LatentMoE(nn.Module):
    """One Stable LatentMoE FFN.

    The assembly (k3/model.py K3Block) calls::
        delta = self.ffn(x)            # returns the residual delta; the block adds it
    LatentMoE returns the FFN delta only (the residual add is the block's job).
    """

    def __init__(self, cfg: MoEConfig, hidden_size: int, model_cfg: K3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.latent = cfg.latent_size  # 0.5× hidden at full scale
        del model_cfg  # passed for SiTU β access when wiring experts; unused in this scaffold
        # TODO(hand-build): parameters —
        #     - down/up projections (hidden ↔ latent),
        #     - router W_r (hidden → num_experts) + learned bias (size num_experts),
        #     - num_experts ROUTED experts at (latent, expert_intermediate, latent) with SiTU-GLU,
        #     - 2 SHARED experts at FULL width (hidden, shared_intermediate, hidden) with SiTU-GLU,
        #     - RMSNorm at latent width (pre-up).
        raise NotImplementedError(
            "LatentMoE.__init__ — parameter layout is hand-built. Shared experts are FULL "
            "width (FACTS A3); routed experts are LATENT width."
        )

    def forward(self, x: Tensor) -> Tensor:
        """Returns the FFN delta (B, S, hidden). The QB bias update happens POST-step (outside
        autograd), reading QBStats — not returned here yet (wire when QB lands)."""
        # TODO(hand-build):
        #   1. down:    z = W_down(x)                                # (B,S,latent)
        #   2. scores:  s = sigmoid(W_r(x)) + bias                   # (B,S,num_experts)
        #   3. top-k = top_k(s, k=cfg.top_k); renormalize weights EXCLUDING bias
        #   4. dispatch z to selected experts (latent width), apply SiTU-GLU experts
        #   5. RMSNorm at latent width
        #   6. combine: weighted sum of routed-expert outputs
        #   7. up:      W_up(combined)                               # (B,S,hidden)
        #   8. ADD the 2 shared experts at full width (always-on)
        raise NotImplementedError(
            "LatentMoE.forward — routing + latent dispatch + QB is hand-built (K5 rung). "
            "Load convergence q = m·k/n must hold with no aux loss."
        )

    @torch.no_grad()
    def update_bias(self, stats: QBStats) -> None:
        """Quantile Balancing: b̂ ← −quantile_{1−k/n}(s − α), mean-removed, one-step delay.

        Called by the trainer after ``optimizer.step()``. Bias is FROZEN at inference — the
        eval/decode path must not call this."""
        # TODO(hand-build): the histogram-estimated quantile over a single all-reduce.
        raise NotImplementedError("update_bias — Quantile Balancing math (FACTS A14, Eq. 14).")
