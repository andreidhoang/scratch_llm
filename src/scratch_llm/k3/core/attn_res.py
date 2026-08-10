"""core/attn_res.py — Block Attention Residuals (K4 rung, TEMPLATE).

TEMPLATE NOTICE (agent-authored scaffold, 2026-08-04). HANDCRAFTED.md Class 1 hand-built;
user waived the boundary for scaffolding. Scaffold only — the pseudo-query attention and the
online-softmax merge raise ``NotImplementedError`` for the human to author. Delete-test
applies once PROVEN.

WHAT (FACTS A9; arXiv:2603.15031; github.com/MoonshotAI/Attention-Residuals):
  Each sublayer gets a LEARNED pseudo-query ``q_l = w_l``. AttnRes computes, per position:
    out_l = softmax( q_l · RMSNorm(k_i)ᵀ / √d ) · v_i     over sources i
  where the sources are: the embedding + each preceding BLOCK output. BLOCK AttnRes groups
  sources into blocks of size ``attn_res_block_size`` (12 at full scale ⇒ 8 blocks +
  embedding = 9 sources); an online-softmax merge combines the intra-block and inter-block
  results WITHOUT materializing the full source×source attention.

MASTERY BAR (HANDCRAFTED.md):
  - delete-test;
  - block ≡ full: Block AttnRes output equals Full AttnRes output (float64), for any L;
  - merge-bookkeeping probe: the online-softmax running-max / running-normalizer must be
    exact — a wrong merge still trains;
  - pseudo-query gradient-flow probe: EVERY source's pseudo-query must receive non-zero
    gradient. FACTS A19(a) measured the layer-0 self-attn pseudo-query ≈ 0 (single source →
    softmax constant → no gradient → weight decay shrinks it) — that's EXPECTED for a
    single-source, but the test must catch a multi-source query that wrongly gets zero grad.

SILENT-BUG SURFACES:
  - merge bookkeeping (running max m, running normalizer l): wrong values still train,
    subtly reweight sources;
  - a wrong SOURCE LIST (e.g. dropping the embedding source, or double-counting a block):
    block≡full catches shape errors but NOT a wrong-but-plausible source list — the
    pseudo-query gradient-flow probe is what catches it (FACTS A19a).

INTERVIEW QUESTION: why is AttnRes residual-flow better than a learned scalar per source
(a "weighted skip connection")? — a scalar can only SCALE a source; attention can SELECT
(subspace projection) which dims of which source to admit. The pseudo-query is a learned
"what am I looking for" that gates sources content-wise, not just magnitude-wise. The cost
is L·d extra params — cheap.

UPGRADE OF: nothing in repo (net-new). Wires into k3/model.py K3Model.forward via
``self.attn_res.merge(x, block_reps)`` after the block loop, before the final norm. Assembly
swap-in (k3/model.py K3Model):
    from scratch_llm.k3.core.attn_res import BlockAttnRes
    self.attn_res = BlockAttnRes(cfg.hidden_size, cfg.attn_res_block_size, cfg.num_layers)
"""

from __future__ import annotations

from torch import Tensor, nn


class BlockAttnRes(nn.Module):
    """Block Attention Residuals across the whole stack.

    The assembly (k3/model.py K3Model) builds this once, collects
    ``block_reps = [embedding] + [block_out_i for i in blocks]``, and calls::

        x = self.attn_res.merge(x, block_reps)

    after the block loop, BEFORE the final norm.
    """

    def __init__(self, hidden_size: int, block_size: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.block_size = block_size  # 12 full, 4 mini
        self.num_layers = num_layers
        # TODO(hand-build): parameters —
        #     - one learned pseudo-query w_l per sublayer per source (param overhead = L·d,
        #       FACTS A9); resolve the exact (sublayer, source) indexing from arXiv:2603.15031
        #       and verify against the checkpoint census (res_proj shapes, FACTS A19);
        #     - per-source RMSNorm (over hidden_size) applied to keys before the dot product;
        #     - scalar softmax scale 1/√d.
        raise NotImplementedError(
            "BlockAttnRes.__init__ — pseudo-query layout is hand-built. The exact sublayer × "
            "source indexing is in arXiv:2603.15031; cross-check res_proj shapes (FACTS A19)."
        )

    def merge(self, x: Tensor, block_reps: list[Tensor]) -> Tensor:
        """Online-softmax merge of block reps into the residual stream x.

        ``block_reps[0]`` = embedding; ``block_reps[1..N]`` = per-block outputs. The merge
        computes, per position, a softmax-weighted combination of sources (intra-block and
        inter-block) and ADDS it to x. Must equal Full AttnRes in float64 (the mastery gate).
        """
        # TODO(hand-build):
        #   1. group sources into blocks of `block_size`
        #   2. per block: intra-block softmax over sources with pseudo-queries → block summary
        #   3. inter-block: online-softmax merge (running max m, running normalizer l) so the
        #      full source×source matrix never materializes
        #   4. add the merged result to x
        raise NotImplementedError(
            "BlockAttnRes.merge — online-softmax merge is hand-built (K4 rung). "
            "block ≡ full float64 equivalence must hold."
        )
