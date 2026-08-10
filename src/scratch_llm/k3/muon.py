"""k3.muon — Per-Head Muon optimizer + K2 weight clipping (K6, PAIRED, TEMPLATE).

TEMPLATE NOTICE (agent-authored scaffold, 2026-08-04). PAIRED track (HANDCRAFTED.md Class 2):
the human owns the per-head Newton–Schulz partitioning on Q/K/V momentum blocks + the K2
weight-clipping derivation; agents own the optimizer plumbing (param-group registration, the
step loop, F1-race harness wiring). This template carries the structure both sides build on;
the NS iteration body and the clip rule raise ``NotImplementedError`` for the human, the
param-group plumbing raises for a later agent session.

WHAT (FACTS A8; report §2.5):
  K3's optimizer is **Per-Head Muon**: Newton–Schulz (NS) orthogonalization applied per
  attention head on the partitioned Q/K/V momentum blocks, plus K2's weight clipping
  (arXiv:2507.20534, the QK-Clip lineage). Cosine LR, 1% warmup, weight-decay 0.1.

WHY PER-HEAD: the Q/K/V projection weight is one (hidden × H·d_h) matrix, but its meaningful
2D sub-structure is per-head (each head's (hidden × d_h) slice). Applying NS to the whole
matrix flattens that structure; applying it per-head slice preserves it. The partition is the
silent-bug surface — wrong slice boundaries still optimize, just less well.

HUMAN-OWNED (you write these):
  - the per-head NS iteration (the Newton–Schulz polynomial for the matrix sign / sqrt-inverse);
  - the Q/K/V block partition boundaries (which axis, which head stride);
  - the K2 weight-clip threshold and WHICH weights it applies to.

AGENT-OWNED (a later agent session writes these — flagged below):
  - param-group registration (Muon params vs AdamW params — norms/embeddings use AdamW);
  - the .step() loop scaffolding; F1-race harness wiring against optim.py's Muon (F1).

MASTERY BAR (HANDCRAFTED.md Class 2):
  - delete-test on the NS polynomial + the partition;
  - F1-race parity: Per-Head Muon matches optim.py's Muon on a head-uniform test (the
    per-head partition is a no-op when all heads are treated identically).

UPGRADE OF: scratch_llm.optim (Muon, F1).
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class PerHeadMuon(Optimizer):
    """Per-Head Muon optimizer.

    Construct with the model's param groups already split:
      - muon groups: 2D matrices (Q/K/V/O projections, expert weights) → Newton–Schulz;
      - adamw groups: 1D params (norms, embeddings, biases) → AdamW fallback.

    The NS direction is computed PER ATTENTION HEAD on the Q/K/V slices — the partition
    boundaries are the human-owned silent-bug surface.
    """

    def __init__(  # noqa: PLR0913 — mirrors optim.py Muon's signature
        self,
        params,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        clip_threshold: float = 1.0,
    ) -> None:
        # TODO(agent-owned plumbing): register param groups + defaults. Split the incoming
        #   params into muon (2D) vs adamw (1D) groups; store ns_steps, clip_threshold,
        #   momentum, weight_decay in defaults. The human fills in the NS iteration below.
        raise NotImplementedError(
            "PerHeadMuon.__init__ — agent-owned plumbing (PAIRED). Wire param groups here."
        )

    @torch.no_grad()
    def step(self, closure=None) -> None:  # type: ignore[override]
        # TODO(human-owned math):
        #   1. for each muon param: gather momentum buffer,
        #   2. PARTITION per-head on the Q/K/V blocks (the human's partition rule),
        #   3. apply Newton–Schulz orthogonalization per slice (the human's NS polynomial),
        #   4. apply K2 weight clipping at `clip_threshold` (the human's threshold/rule),
        #   5. update with the NS'd direction + weight-decay;
        #   6. for adamw params: standard AdamW update.
        raise NotImplementedError(
            "PerHeadMuon.step — per-head NS + K2 clip is human-owned (PAIRED track). "
            "The partition boundaries and NS polynomial are the silent-bug surface."
        )
