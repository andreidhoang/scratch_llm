"""core/norm.py — the one RMSNorm of K3: every block, final, AttnRes key, MLA latent and MoE latent
norm is this class.

    y = weight ⊙ round_to_input_dtype( x · rsqrt(mean(x²) + eps) )

This is HF ``KimiRMSNorm`` (modeling_kimi_linear.py :226-236) and its op order matters in low
precision: normalize in fp32, round back to the input dtype, and only then multiply by the weight
(``self.weight * x.to(dtype)``). A bf16 checkpoint therefore rounds exactly where Moonshot's does
(``tests/test_k3_gated_mla.py`` pins it bitwise in fp32 and bf16). Two deliberate differences:

* fp64 input stays fp64 (HF's ``.float()`` would truncate it), so equivalence tests run at 1e-10;
* the normalization runs with autocast off, so a bf16 region cannot pull it back down.

The output dtype is HF's too: ``promote(weight.dtype, input dtype)`` — a bf16 activation through
an fp32 weight (bf16 autocast over fp32 master weights) comes out fp32, and the next ``nn.Linear``
casts it back under autocast.

``eps`` has no default on purpose: K3 uses two values. The residual-stream norms, AttnRes key
norms and the MoE latent norm take ``rms_norm_eps`` (1e-5, config.json); the MLA latent norms keep
KimiRMSNorm's own default 1e-6 because HF constructs them without an eps (vLLM passes 1e-5 there).

Not ``scratch_llm.model.RMSNorm``: that one computes fp64 input in fp32 and applies the weight
before rounding, which is neither HF's bf16 rounding nor an fp64 reference.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["RMSNorm"]


class RMSNorm(nn.Module):
    """``weight ⊙ x / sqrt(mean(x²) + eps)`` over the last dim; ``weight`` starts at ones."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(x.device.type, enabled=False):
            z = x.to(torch.promote_types(x.dtype, torch.float32))  # fp32, or fp64 for fp64
            z = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * z.to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
