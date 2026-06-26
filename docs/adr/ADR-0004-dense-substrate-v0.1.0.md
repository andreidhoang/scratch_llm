# ADR-0004 — Dense (GQA-ready) substrate by default; MoE as an opt-in extension

- **Status:** Accepted (2026-06-05)
- **Layer:** A1 Substrate
- **Decides:** §8.1 + §8.5 of `A1_basics_BUILD_GUIDE.md`

## Context

Every open-weight 2026 flagship is sparse MoE, so a from-scratch MoE FFN is worth building. But the
core A1 deliverable — a correct, debuggable dense decoder — comes first: it is the substrate every
later assignment builds on, and a clean dense path keeps the loss-at-init derivation simple.

## Decision

1. **The default substrate is dense + GQA-ready** (see ADR-0002). MoE-from-scratch (aux-loss-free
   balancing + z-loss + router-entropy logging) is built as a **separate opt-in module** (see
   ADR-0007), never as a change to the dense default.
2. **Embeddings are untied** for the from-scratch build (A1 does not require tying; explicit shapes
   keep the `cross_entropy ≈ log(vocab_size)` derivation at init clean to reason about).

## Consequences

- (+) The dense path is the simple, always-correct baseline; MoE is additive and reversible.
- (+) Untied shapes make loss-at-init clean to reason about.
- (−) The default carries no sparse result — accepted; MoE is available opt-in (ADR-0007).
