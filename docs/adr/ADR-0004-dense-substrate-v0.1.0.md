# ADR-0004 — Dense (GQA-ready) substrate for v0.1.0; MoE is v0.2.0-B

- **Status:** Accepted (2026-06-05)
- **Layer:** L1 Substrate (A1)
- **Decides:** §8.1 + §8.5 of `A1_basics_BUILD_GUIDE.md`

## Context

Every open-weight 2026 flagship is sparse MoE, and the hottest 2025–26 failure
mode (MoE breaks GRPO via routing mismatch; R3 ~94% of tokens differ in ≥1 layer
between train/infer engines) is diagnosed by our `kl_train_infer` pillar. That
makes MoE the natural v0.2.0-B *headline result* — but it is a ship-cadence (G6)
risk inside the frozen 14-day v0.1.0 sprint.

## Decision

1. **v0.1.0 substrate is dense + GQA-ready** (see ADR-0002). MoE-from-scratch
   (aux-loss-free balancing + z-loss + router-entropy logging) is **v0.2.0-B**,
   gated by the build guide's kill criterion: *>2 debug days ⇒ stay dense.*
2. **Embeddings are untied** for the from-scratch build (A1 does not require
   tying; explicit shapes keep the loss-at-init derivation clean). Recorded so
   checkpoint shapes are stable across the smoke run.

## Consequences

- (+) The 14-day ship is de-risked; MoE×RL collapse becomes a *headline*, not
  plumbing risk.
- (+) Untied shapes make `cross_entropy ≈ log(vocab_size)` at init clean to reason
  about.
- (−) No sparse result in v0.1.0 — accepted; it is explicitly the v0.2.0 axis.
