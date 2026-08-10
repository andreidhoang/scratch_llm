"""k3.serve — hybrid-state serving for K3 (K9, DELEGATED, scaffold).

TEMPLATE NOTICE (agent-authored scaffold, 2026-08-04; DELEGATED track, HANDCRAFTED.md
Class 3). Rental-gated (K9, ROADMAP §3): this is the serving layer that runs the released
2.78T checkpoint on rented 8×B300 via vLLM/SGLang. Scaffold only — raises
``NotImplementedError`` until core/ lands and the K9 rental happens. No runnable server today.

WHAT (FACTS S1–S15; ROADMAP K9):
  K3's inference state is HYBRID — not a uniform KV cache:
    - KDA layers (69/93): CONSTANT-SIZE recurrent state (S_t + conv ring buffer). Does NOT
      grow with sequence length ⇒ the 1M-context memory claim (FACTS A12; ROADMAP K8).
    - MLA layers (24/93): GROWING latent KV in the compressed ``kv_lora_rank`` space (the
      weight-absorption identity means we cache the latent, not per-head K/V).
  ``serve.py`` paginates this hybrid state for continuous batching + prefix caching, the way
  vLLM/SGLang do for vanilla attention — but the two state types need different paging
  disciplines: recurrent state is per-stream constant (reclaim on stream end); latent KV is
  per-token growing (block-paged, prefix-shareable).

DEPENDS ON: ``k3/model.py :: HybridState`` (the container), ``core/kda.py :: KDAState``,
``core/gated_mla.py :: MLALatentKV`` (the per-layer state numerics). None runnable yet.

MASTERY BAR (when K9 lands):
  - prefix-cache equivalence: a cached prefix produces identical logits to a cold decode;
  - memory table reproduces K8's 1M-token accounting from config (FACTS A11);
  - measured TTFT / tok/s / $/M land in ``bench/RESULTS.md §K3`` with hardware + method
    (the honesty exercise: our measured vs the book's 0.93s/92.1/$190.13 vs vLLM's 111 —
    three-way ledger, FACTS S5).

NOT IN SCOPE HERE: the vLLM Docker flags, the Modal runbook
(``deploy/runbooks/k3_8xb300_modal.md``), the DSpark draft (K10.1). Those are ops artifacts;
this module is the state-paging abstraction the ops layer calls into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scratch_llm.k3.config import K3Config
    from scratch_llm.k3.model import HybridState


class HybridStateCache:
    """Paged hybrid-state cache for continuous batching over K3.

    Two paging disciplines in one structure:
      - KDA slots: fixed-size per stream (recurrent state + conv ring); reclaimed on stream
        end, NOT grown per token.
      - MLA slots: paged latent KV (block-allocated in ``kv_lora_rank`` space), grown per
        token, prefix-shareable across streams (the vLLM prefix-cache analogue).

    Rental-gated: real implementation lands in K9 against a measured 8×B300 session. Today
    this is the interface the ops layer (``deploy/runbooks/k3_8xb300_modal.md``) will call.
    """

    def __init__(self, cfg: K3Config, max_batch: int, max_seq: int) -> None:
        # TODO(K9): allocate the paged KDA-state pool + the block-paged MLA latent-KV pool.
        #   Block sizing for the MLA latent must match vLLM's token-block granularity so prefix
        #   caching composes (FACTS S12 — known vLLM bug at the 1536-token boundary).
        del cfg, max_batch, max_seq
        raise NotImplementedError(
            "HybridStateCache — K9 rental-gated. Lands against a measured 8×B300 session."
        )

    def allocate(self, stream_id: int) -> HybridState:
        """Allocate a fresh hybrid state for a new request stream."""
        del stream_id
        raise NotImplementedError("HybridStateCache.allocate — K9.")

    def append_mla_tokens(self, stream_id: int, k_latent: object, v_latent: object) -> None:
        """Append MLA latent KV tokens for a stream (the growing half of the hybrid state)."""
        del stream_id, k_latent, v_latent
        raise NotImplementedError("HybridStateCache.append_mla_tokens — K9.")

    def update_kda_state(self, stream_id: int, layer_id: int, new_state: object) -> None:
        """Overwrite a KDA layer's recurrent state for a stream (the constant-size half)."""
        del stream_id, layer_id, new_state
        raise NotImplementedError("HybridStateCache.update_kda_state — K9.")
