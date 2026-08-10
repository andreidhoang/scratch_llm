"""scratch_llm.k3 — build & host Kimi K3 from scratch (the K-track).

Chartered 2026-07-31. Spec of record: ``docs/k3/FACTS.md`` (claim ledger). Plan:
``docs/k3/ROADMAP.md`` (K0→K10). Binding rules + status table: ``HANDCRAFTED.md``.

THE TWO-CITIZENSHIP RULE (HANDCRAFTED.md)
-----------------------------------------
Silent-bug surfaces are HAND-BUILT by the human (a bug there runs, loss still drops, the
model is subtly worse — exactly what the human is learning to catch). Loud-bug surfaces are
DELEGATED to agents (shape errors crash, tests fail). Three classes:

  Class 1 — ``core/``  : hand-built. The 5 mechanisms (situ, kda, gated_mla, latent_moe, attn_res).
  Class 2 — paired     : human writes the math; agents write plumbing + tests (muon, qat).
  Class 3 — delegated  : agents own, human reviews (config, param_count, model, serve, docs).

BUILD ORDER (serial — dependency order; do NOT parallelize hand-builds)
----------------------------------------------------------------------
  Rung   Module                Class   Status        Depends on
  ────   ───────────────────   ─────   ──────────    ──────────────────────
  K0     config.py             3       ✅ DONE       —
  K0     param_count.py        3       ✅ DONE       config
  K5p    core/situ.py          1       🟡 scaffold   —                    ← START HERE (warmup)
  K2     core/kda.py           1       🟡 scaffold   linear_attn          ← CRITICAL PATH
  K3     core/gated_mla.py     1       🟡 scaffold   mla
  K5     core/latent_moe.py    1       🟡 scaffold   situ, moe
  K4     core/attn_res.py      1       🟡 scaffold   —
  K6     muon.py               2       🟡 scaffold   optim
  K6     model.py              3       ✅ scaffold   all of core/
  K7     qat.py                2       🟡 scaffold   quant/nvfp4_mxfp4
  K9     serve.py              3       🟡 scaffold   model + core/

Status legend:
  ✅ DONE     shipped + gate green.
  ✅ scaffold assembly exists; ``forward`` raises until core/ lands (model.py).
  🟡 scaffold template exists (docstring + signatures + ``raise NotImplementedError`` math
              bodies). The human fills in the math by hand → delete-test → ✅ PROVEN.
  ⬜ not started.

PUBLIC API
----------
    from scratch_llm.k3 import k3_full, mini_k3_d12            # config presets
    from scratch_llm.k3.model import build_k3, build_mini_k3   # builders (raise until core/ lands)
    from scratch_llm.k3.param_count import count_params        # 2.78T closure (FACTS A18)

START-HERE (next session)
-------------------------
K2 (``core/kda.py``) is the critical path — where the mastery happens. Proposal of record:
``docs/k3/K2_PROPOSAL_KDA.md``. **Pre-flight:** resolve the A_log ``[128]`` vs ``[num_heads]``
semantics (FACTS A18, OPEN) by reading ``modeling_kimi_linear.py`` checkpoint loading, BEFORE
the hand-build. ``core/situ.py`` is the lighter warmup if you want to begin with the bound proof
(|f| ≤ β1·β2 = 100).

BOUNDARY WAIVER LOGGED 2026-08-04: the user waived the core/ read-only rule to obtain
scaffolding. The 5 core modules exist as agent-authored templates (🟡). The delete-test still
applies — when a hand-build passes its mastery bar, ``rm`` the template and rewrite from the
docstring, then mark ✅ PROVEN in ``HANDCRAFTED.md``. Templates are scaffolds to build on,
NOT verified math.
"""

from scratch_llm.k3.config import K3Config, k3_full, mini_k3_d12

__all__ = ["K3Config", "k3_full", "mini_k3_d12"]
