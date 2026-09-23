"""core/situ.py — SiTU-GLU activation + the layer-1 dense MLP.

    SiTU-GLU(g, u) = [ β1 · tanh(g / β1) ⊙ σ(g) ] ⊙ [ β2 · tanh(u / β2) ],   g = W_g x, u = W_u x

β1 = 4 (``activation_situ_beta``), β2 = 25 (``activation_situ_linear_beta``), so |f| ≤ β1·β2 = 100
for every input. Moonshot's reference (HF ``modeling_kimi_linear.SituAndMul``, vLLM
``SituAndMul.forward_native``) computes both branches in fp32 and casts the product back; this
module does the same, except fp64 inputs stay fp64 so equivalence tests can run at 1e-10.

Why two soft caps: bf16 keeps ~3 significant digits, and SwiGLU's two factors are both unbounded,
so coincident large coordinates make activation outliers. The gate branch is capped tight (β1 = 4:
it decides how much passes); the up branch keeps a wider range (β2 = 25: it carries magnitude).
Near the origin β·tanh(z/β) = z − z³/(3β²) + …, so SiTU-GLU is SwiGLU to first order and recovers
it as β → ∞. The sigmoid reads the *uncapped* g — capping it too still trains, but changes the gate
away from the origin (g = 6 is the test point).

Used by the layer-1 dense MLP (``DenseSiTUMLP``) and by every routed and shared expert in
``core/latent_moe.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from scratch_llm.k3.config import K3Config

__all__ = ["SiTUConfig", "SiTUGLU", "DenseSiTUMLP", "situ_gate", "soft_cap", "situ_glu"]


@dataclass(frozen=True)
class SiTUConfig:
    """The two softcaps."""

    beta_gate: float = 4.0  # β1, gate branch (W_g)
    beta_up: float = 25.0  # β2, up branch (W_u)

    @property
    def bound(self) -> float:
        return self.beta_gate * self.beta_up  # |f| ≤ 100


def _math_dtype(*xs: Tensor) -> torch.dtype:
    """fp32 for every low-precision input, fp64 if any input is fp64."""
    dtype = torch.float32
    for x in xs:
        dtype = torch.promote_types(dtype, x.dtype)
    return dtype


def soft_cap(x: Tensor, beta: float) -> Tensor:
    """β·tanh(x/β): odd, monotone, |·| < β, and |x − β·tanh(x/β)| ≤ |x|³/(3β²)."""
    return beta * torch.tanh(x / beta)


def situ_gate(g: Tensor, beta: float) -> Tensor:
    """Gate branch β·tanh(g/β)·σ(g); the sigmoid reads the uncapped g. |·| ≤ β."""
    return soft_cap(g, beta) * torch.sigmoid(g)


def situ_glu(x_gate: Tensor, x_up: Tensor, beta_gate: float, beta_up: float) -> Tensor:
    """SiTU-GLU in fp32 (fp64 for fp64 inputs), cast back to ``x_gate.dtype``."""
    dtype = _math_dtype(x_gate, x_up)
    with torch.autocast(x_gate.device.type, enabled=False):
        f = situ_gate(x_gate.to(dtype), beta_gate) * soft_cap(x_up.to(dtype), beta_up)
    return f.to(x_gate.dtype)


class SiTUGLU(nn.Module):
    """SiTU-GLU gated activation over the two pre-activation tensors.

    Stateless: the W_g / W_u projections live in the caller; this module owns the activation.
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
        return situ_glu(x_gate, x_up, self.beta_gate, self.beta_up)

    def extra_repr(self) -> str:
        return f"beta_gate={self.beta_gate}, beta_up={self.beta_up}"


class DenseSiTUMLP(nn.Module):
    """``down_proj(SiTU-GLU(gate_proj x, up_proj x))``, no biases — HF ``KimiMLP``.

    The FFN of every layer before ``cfg.first_k_dense`` (layer 1 at full scale) and the shape of
    the LatentMoE shared experts. Parameter names match the checkpoint
    (``mlp.{gate,up,down}_proj``). Returns the delta; the block adds the residual.
    """

    def __init__(self, hidden_size: int, intermediate: int, cfg: SiTUConfig | K3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)  # W_g
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)  # W_u
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)  # W_o
        self.act = SiTUGLU(cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x), self.up_proj(x)))
