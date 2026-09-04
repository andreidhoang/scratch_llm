"""Correctness oracle for the linear-attention family — a re-export, not a reimplementation.

Intent
------
Every other kernel family in this tree owns its oracle (``attention/reference.py`` is a
pure-PyTorch FlashAttention-2). This family deliberately does not. The gated delta rule's
oracle already exists, one layer up, as the E1 instrument: ``mastery/reference.py`` is the
fp64 sequential recurrence written by hand from the equation, and ``mastery/paths.py`` is the
chunked WY form that a training kernel must use. Those two ARE the ground truth this repo
publishes divergence maps against, so a fourth copy here would be a fifth place for the
recurrence to drift.

Invariant
---------
This module adds **zero mathematics**. It re-exports, so that:

  * a production caller reaches the oracle through the family seam like every other family
    (``kernels/CLAUDE.md`` § the dispatch contract), and
  * the number a kernel is scored against is, by construction, the same number the public
    divergence map was built from — there is no second oracle to reconcile.

The seal is untouched: ``mastery/reference.py``'s body is the human's (``oracle-guard.sh``),
and it raises ``NotImplementedError`` until L0.1 lands. That is the correct state to re-export —
a kernel cannot be scored before the oracle exists, and this module makes that failure loud and
in one place rather than letting a family invent a weaker reference to get green.

Relationship to the other two expressions of the recurrence (PLAN.md item 6 — one dependency
chain, three deliberate expressions):

  * ``mastery/{reference,paths}.py``  fp64 oracle + chunked path  -> E1's divergence map  (HERE)
  * ``linear_attn.py``                the CS336 teaching module   -> substrate, batched/heads
  * ``k3/core/kda.py``                scalar alpha -> ``Diag(alpha)``  -> K2, the KDA generalization

``k3/core/kda.py`` is the same five lines with per-channel decay; when it is hand-built
(HANDCRAFTED.md Class 1) its chunkwise path is scored against THIS oracle, which is what makes
the K3 lane and the kernel lane one lane instead of two.
"""

from __future__ import annotations

from scratch_llm.mastery.paths import chunked_wy
from scratch_llm.mastery.reference import recurrent_reference

__all__ = ["chunked_wy", "recurrent_reference"]
