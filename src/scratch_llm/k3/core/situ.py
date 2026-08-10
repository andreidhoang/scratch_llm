"""core/situ.py — SiTU-GLU activation + the layer-1 dense MLP (K5 prep, TEMPLATE).

TEMPLATE NOTICE (agent-authored scaffold, 2026-08-04). HANDCRAFTED.md Class 1 makes this
hand-built territory; the user waived the boundary for scaffolding. This file is a scaffold
to build on: docstring, signatures, and source anchors are reference; the activation math
bodies raise ``NotImplementedError`` for the human to author from the derivation. The
delete-test still applies — once your hand-build is PROVEN, ``rm`` this and confirm you can
rewrite it from the docstring alone.

WHAT (FACTS A10; report §2.3.2):
  SiTU-GLU(x) = [ β1 · tanh(W_g x / β1) ⊙ σ(W_g x) ] ⊙ [ β2 · tanh(W_u x / β2) ]
  β1 = 4.0 (gate-branch softcap), β2 = 25.0 (up-branch softcap), bound |f| ≤ β1·β2 = 100.
  ``hidden_act: "situ"`` in config.json. Used by every LatentMoE expert AND by the layer-1
  dense MLP.

WHY SOFTCAPS: bf16 has ~3 significant digits; activations drifting past ~1e3 lose all
precision and create the outlier channels that wreck downstream attention logits. The tanh
softcaps bound each branch independently so the activation can NEVER exceed β1·β2, killing the
outlier-generation failure mode at the source. A wrong β (say β2=2.5) still trains fine on
short context — the model is just silently worse. That is why this is hand-built: the bound
must be PROVEN, not hoped.

MASTERY BAR (HANDCRAFTED.md):
  - delete-test (rewrite from this docstring);
  - bound proof: |SiTU-GLU(x)| ≤ β1·β2 = 100 for all x (hand-derive, then test);
  - bf16 saturation: fp32 tanh-saturation test on |x| → 1e4 (no precision loss in the caps).

SILENT-BUG SURFACES:
  - wrong β value: trains, worse model;
  - softcap applied to the wrong branch (gate vs up swapped): trains, asymmetric saturation;
  - computing tanh in bf16 instead of fp32: loses the cap's precision exactly when it matters.

INTERVIEW QUESTION: why TWO different β's (4 and 25), not one shared softcap? — the gate
branch multiplies the output directly (β1=4 keeps σ·tanh tightly scaled so the gate never
saturates the output); the up branch carries the signal magnitude (β2=25 lets the FFN express
a wide dynamic range while still capped). Asymmetric because the two branches play different
roles: one decides "how much", the other decides "what".

UPGRADE OF: nothing in repo (net-new). Wires into k3/model.py via ``DenseSiTUMLP`` (layer-1
FFN) and into core/latent_moe.py (every routed + shared expert).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from scratch_llm.k3.config import K3Config


@dataclass(frozen=True)
class SiTUConfig:
    """The two softcaps — frozen so a wrong value fails loudly at construction."""

    beta_gate: float = 4.0  # β1 on the gate branch (W_g)
    beta_up: float = 25.0  # β2 on the up branch (W_u)

    @property
    def bound(self) -> float:
        return self.beta_gate * self.beta_up  # |f| ≤ 100


class SiTUGLU(nn.Module):
    """SiTU-GLU gated activation, applied to two pre-activation tensors.

    f(x_gate, x_up) = [ β1 · tanh(x_gate / β1) ⊙ σ(x_gate) ] ⊙ [ β2 · tanh(x_up / β2) ]

    Stateless: the W_g / W_u projections live in the caller (the expert or the dense MLP);
    this module owns ONLY the gated activation. Computes the tanh branches in fp32 (the caps
    are where precision matters most) and casts back.
    """

    def __init__(self, cfg: SiTUConfig | K3Config) -> None:
        super().__init__()
        if isinstance(cfg, K3Config):
            self.beta_gate = cfg.situ_beta_gate
            self.beta_up = cfg.situ_beta_up
        else:
            self.beta_gate = cfg.beta_gate
            self.beta_up = cfg.beta_up

    def forward(self, x_gate: Tensor, x_up: Tensor) -> Tensor:
        # TODO(hand-build): author the activation. Derive the bound AS you write it:
        #   β1·tanh(z/β1) ≤ β1 pointwise; σ ∈ [0,1]; so bracket_g ≤ β1, bracket_u ≤ β2;
        #   product ≤ β1·β2. Compute the tanh branches in fp32, cast back to input dtype.
        raise NotImplementedError(
            "SiTU-GLU math — hand-build per the docstring derivation "
            "(bound |f| ≤ β1·β2 = 100 must be proven, not assumed)."
        )


class DenseSiTUMLP(nn.Module):
    """The layer-1 dense FFN: w_down( SiTU-GLU(w_gate x, w_up x) ). No biases.

    This is the FFN for every layer ≤ ``cfg.first_k_dense`` (layer 1 at full scale; FACTS A2).
    Same SiTU-GLU activation as the LatentMoE experts, just dense — no routing, no latent
    down-project. Returns the delta only; the K3Block adds the residual.

    Assembly swap-in (k3/model.py K3Block, the dense-FFN branch):
        from scratch_llm.k3.core.situ import DenseSiTUMLP
        self.ffn = DenseSiTUMLP(cfg.hidden_size, cfg.moe.dense_intermediate, cfg)
    """

    def __init__(self, hidden_size: int, intermediate: int, cfg: K3Config) -> None:
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, intermediate, bias=False)  # W_g
        self.w_up = nn.Linear(hidden_size, intermediate, bias=False)  # W_u
        self.w_down = nn.Linear(intermediate, hidden_size, bias=False)  # W_o
        self.act = SiTUGLU(cfg)

    def forward(self, x: Tensor) -> Tensor:
        # TODO(hand-build): project gate & up → SiTU-GLU → project down. Trivial once SiTUGLU
        #   lands; the whole point is exercising the activation's bound, not this plumbing.
        raise NotImplementedError(
            "DenseSiTUMLP.forward — depends on SiTUGLU.forward (build that first)."
        )
