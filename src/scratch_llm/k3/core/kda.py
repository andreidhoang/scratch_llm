"""core/kda.py — Kimi Delta Attention (K2 rung, the critical path, TEMPLATE).

TEMPLATE NOTICE (agent-authored scaffold, 2026-08-04). HANDCRAFTED.md Class 1 hand-built;
user waived the boundary for scaffolding. Scaffold only — docstring + signatures + source
anchors are reference; the recurrence math (chunkwise, recurrent, per-channel decay) raises
``NotImplementedError`` for the human to author from the derivation. Delete-test applies once
PROVEN. **This is the rung where the mastery happens — do not rush it** (ROADMAP K2).

WHAT (FACTS A12; report §2.1.1; Kimi Linear arXiv:2510.26692):
  KDA is a gated delta-rule linear-attention layer. Per step:
    S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ
    y_t = S_t q_t        (then full-rank output gate → RMSNorm → W_o)
  where:
    α_t ∈ (0,1)^{d_k}   per-CHANNEL decay (K3 change vs Kimi Linear's per-head scalar);
    β_t = sigmoid(...)  input-modulated write gate;
    q, k = L2Norm(Swish(ShortConv(W x)))   — short causal conv k=4, then Swish, then L2 norm;
    v    = Swish(ShortConv(W x)).
  Output gate is FULL-RANK sigmoid (K3 change; Kimi Linear was low-rank).

THE THREE K3 DIFFS vs our linear_attn.py (Gated DeltaNet, F10.1):
  1. decay: scalar α → Diag(α) per-channel. Map is negative-softplus in GDN; K3 uses the
     SCALED SIGMOID with g_min = −5 (``gate_lower_bound: -5.0`` in config). Predict: the decay
     floor moves so log-decay ≥ −5 ⇒ all-Tensor-Core tiles, no vanishing-decay pathologies at
     long context.
  2. F10.2 pieces land here: short causal conv (k=4) + Swish on q/k/v, L2Norm on q/k.
  3. output gate: low-rank → full-rank sigmoid.

MASTERY BAR (HANDCRAFTED.md) — ALL must pass before K2 is PROVEN:
  - delete-test (rewrite from this docstring);
  - three-path equivalence: chunkwise ≡ recurrent ≡ float64-reference, now with per-channel α;
  - FLA parity: numerical match vs ``fla.ops.kda`` (fla-core ≥ 0.4.0) on random tensors, fwd+bwd;
  - checkpoint load: the real K3 checkpoint carries ``A_log`` as **[128]** (= head_dim) per KDA
    layer, NOT [num_heads=96] as the HF reference code frames it (FACTS A18 — the −2,208 residual
    in the pre-census accounting; 69 × (128−96) = 2,208 exactly). The A_log → Diag(α) mapping
    (broadcast? interpolation per channel group?) is OPEN — resolve by reading
    ``modeling_kimi_linear.py`` checkpoint loading BEFORE the hand-build. This is the #1 silent
    trap: a wrong A_log interpretation still trains but forgets badly at long context.

SILENT-BUG SURFACES (why this is hand-built):
  - wrong decay floor (g_min sign/value): trains fine, forgets at long context;
  - A_log per-head-vs-per-dim semantics: trains fine, wrong retention profile;
  - chunkwise bookkeeping (WY/UT forms): wrong but still trains;
  - conv ring-buffer off-by-one: shifts the receptive field, no crash.

INTERVIEW QUESTION: why per-CHANNEL decay instead of per-head? — different channels carry
different-frequency information; a single per-head α forces all channels to forget at one
rate, smearing high-frequency detail and retaining low-frequency noise together. Per-channel
α lets the layer learn a frequency-selective retention basis — the diagonal decay IS a
per-channel first-order IIR filter bank.

UPGRADE OF: scratch_llm.linear_attn (Gated DeltaNet, F10.1) — keep its three-path contract.
Assembly swap-in (k3/model.py K3Block, KDA branch):
    from scratch_llm.k3.core.kda import KDALayer
    self.attn = KDALayer(cfg.kda, cfg.hidden_size)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from scratch_llm.k3.config import KDAConfig

if TYPE_CHECKING:
    from scratch_llm.k3.model import HybridState


@dataclass
class KDAState:
    """Recurrent state for ONE KDA layer, ONE batch row. CONSTANT-SIZE in sequence length —
    the K3 1M-context claim rests on this (FACTS A12; ROADMAP K8).

    - ``s_t``: (num_heads, head_dim, head_dim) — the delta-rule matrix state S_t.
    - ``conv``: (num_heads, conv_kernel_size - 1, head_dim) — short-conv ring buffer (causal:
      holds the last k-1 tokens so a single-step decode produces the right conv output).

    Both pieces must be zero-initialized EXACTLY as the reference does — wrong zero-init is a
    silent bug (first-token behaviour drifts). The factory below is for the human to author.
    """

    s_t: Tensor
    conv: Tensor

    @classmethod
    def zeros(
        cls, cfg: KDAConfig, batch: int, device: torch.device, dtype: torch.dtype
    ) -> KDAState:
        # TODO(hand-build): allocate zero-state. Mind dtype — S_t math runs in fp32 for the
        #   recurrent path even under bf16 params (the three-path equivalence test enforces it).
        raise NotImplementedError(
            "KDAState.zeros — zero-init numerics are hand-built (wrong init = silent drift)."
        )


class KDALayer(nn.Module):
    """One KDA layer.

    Wiring: q/k/v projections → short conv (k=4) + Swish (+ L2Norm on q/k) → delta-rule
    recurrence with per-channel decay → full-rank sigmoid output gate → RMSNorm → W_o.

    The assembly (k3/model.py K3Block) calls::
        a = self.attn(x, state, layer_id)        # state is the shared HybridState
    so ``forward`` must read/write its own slot ``state.kda_states[layer_id - 1]`` on decode.
    """

    def __init__(self, cfg: KDAConfig, hidden_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        p = cfg.projection_size  # num_heads * head_dim
        del p  # parameter allocation is the hand-build below
        # TODO(hand-build): define the parameters the recurrence needs. From the reference
        #   (modeling_kimi_linear.py) + config.json, the layer carries at minimum:
        #     - q/k/v projections (fused or separate) into `cfg.projection_size`
        #     - a learned `a_log` parameter — SIZE IS THE OPEN QUESTION (FACTS A18): [128]
        #       per-dim in the checkpoint vs [num_heads] in the released code. Resolve BEFORE
        #       allocating; cfg.a_log_size (=128 full, =64 mini) is set to match the checkpoint.
        #     - a learned `dt_bias` (FACTS A19d: ≈ −4.63 ± 0.05 across all 69 KDA layers)
        #     - conv weight/bias for the short causal conv (kernel = cfg.conv_kernel_size = 4)
        #     - output gate params (FULL-RANK) + W_o + a RMSNorm before W_o
        raise NotImplementedError(
            "KDALayer.__init__ — parameter layout is hand-built. Resolve the A_log [128] vs "
            "[num_heads] semantics (FACTS A18) before allocating parameters."
        )

    def forward(
        self,
        x: Tensor,  # (B, S, hidden) — already RMSNorm-ed by the block
        state: HybridState | None = None,  # HybridState on decode; None on training prefill
        layer_id: int = 0,
    ) -> Tensor:
        """Returns the attention output (B, S, hidden), to be residual-added by the block.

        Two execution paths, BOTH required, BOTH numerically equivalent (the mastery gate):
          - training (state=None): CHUNKWISE form — WY/UT block decomposition, chunk=64;
            O(N) in sequence length, tensor-core-friendly. Upgraded from linear_attn.py.
          - decode (state given):  RECURRENT form — one step at a time, reads/writes its slot
            in HybridState.kda_states[layer_id-1]; constant memory.

        A third REFERENCE path (full fp64 sequential loop) lives in the test, not here.
        """
        # TODO(hand-build), in order:
        #   1. project x → q, k, v (per cfg.projection_size)
        #   2. short causal conv (k=4) + Swish on q/k/v; L2Norm on q/k
        #   3. per-channel α from a_log via scaled sigmoid (g_min = cfg.gate_lower_bound)
        #   4. β = sigmoid(...)
        #   5. the recurrence (chunkwise if training, recurrent if decode)
        #   6. full-rank sigmoid output gate + RMSNorm + W_o
        raise NotImplementedError(
            "KDALayer.forward — the recurrence is the hand-built critical path (K2 rung). "
            "Chunkwise ≡ recurrent ≡ float64-reference must hold (HANDCRAFTED.md mastery bar)."
        )
