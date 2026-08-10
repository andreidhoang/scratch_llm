"""k3.qat — MXFP4 quantization-aware training (K7, PAIRED, TEMPLATE).

TEMPLATE NOTICE (agent-authored scaffold, 2026-08-04). PAIRED track (HANDCRAFTED.md Class 2):
the human owns the MXFP4 quantizer + the STE (straight-through estimator) derivation; agents
own the training-loop integration and the QAT arms. This template carries the
``torch.autograd.Function`` shell both sides build on; the quantizer forward and the STE
backward raise ``NotImplementedError`` for the human.

WHAT (FACTS A6; OCP MX spec v1.0; Jacob et al. 2018, arXiv:1712.05877):
  K3 trains with MXFP4 QAT "from SFT onward" (FACTS A6), scope:
    - MXFP4 (E2M1 elements, block 32, E8M0 shared scale) on ROUTED-EXPERT WEIGHTS ONLY;
    - MXFP8 (E4M3) activations on the expert path;
    - everything else (attention, latent projections, shared experts, routers) stays BF16.
  Effective 4.25 bit/param ⇒ 1.561 TB arithmetic from config (FACTS A6 — derivable, not cited;
  book capsule 29's number, reproduced by param_count.py).
  QAT = fake-quant + STE: forward quantizes (round-trips to the fp dtype), backward passes
  the gradient through unchanged.

HUMAN-OWNED (you write these):
  - the MXFP4 fake-quant function (block reshape → E8M0 shared scale → E2M1 grid round-trip);
  - the STE backward — identity in-range, ZERO gradient on out-of-range (clipped) blocks
    (Jacob et al. 2018). A wrong STE still trains; it silently learns wrong scales.

AGENT-OWNED (a later agent session):
  - wiring quantize/dequantize around the routed-expert forward in core/latent_moe.py;
  - the QAT arm toggle (QAT vs post-training-quant quality arm, ROADMAP K7).

MASTERY BAR (HANDCRAFTED.md Class 2):
  - delete-test on the quantizer + STE derivation;
  - bit/param accounting reproduces 4.25 bit/param ⇒ 1.561 TB (already closed by
    param_count.py; the quantizer must be consistent with it);
  - STE gradient test: in-range grad passes through; out-of-range grad is zeroed.

UPGRADE OF: scratch_llm.quant.nvfp4_mxfp4 (A5).
"""

from __future__ import annotations

import torch
from torch import Tensor


class MXFP4FakeQuant(torch.autograd.Function):
    """MXFP4 fake-quant with the straight-through estimator.

    Forward: quantize to E2M1 elements / block-32 / E8M0 shared scale, then dequantize back
    (fake quant — values round-trip to the fp dtype so downstream math is unchanged type-wise).
    Backward: STE — identity for in-range blocks, ZERO gradient for clipped (out-of-range) blocks.

    Apply to ROUTED-EXPERT WEIGHTS ONLY (FACTS A6 scope). Shared experts, attention, latent
    projections, and routers stay BF16. Activations on the expert path use MXFP8 (E4M3),
    a sibling quantizer not scaffolded here.
    """

    @staticmethod
    def forward(ctx, w: Tensor) -> Tensor:  # type: ignore[override]
        # TODO(human-owned math): the MXFP4 quantizer —
        #   1. reshape into blocks of 32 along the last dim,
        #   2. compute the E8M0 shared scale per block (max-abs → power-of-2 exponent),
        #   3. quantize each element to the nearest E2M1 grid point (signed: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}·scale),
        #   4. dequantize back to the input dtype (fake quant).
        #   Save the in-range mask for the STE backward.
        raise NotImplementedError("MXFP4FakeQuant.forward — human-owned quantizer (PAIRED).")

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        # TODO(human-owned math): STE — pass grad through unchanged for in-range elements;
        #   ZERO the gradient where the forward clipped (out-of-range blocks). Jacob et al. 2018.
        #   Get this wrong → silently learns wrong scales.
        raise NotImplementedError("MXFP4FakeQuant.backward — STE derivation is human-owned.")
