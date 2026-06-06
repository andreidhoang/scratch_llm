# ADR-0007 — MoE-from-scratch (A1.1) pulled forward as an opt-in module

- **Status:** Accepted (2026-06-06)
- **Layer:** L1 Substrate (A1), add-on A1.1
- **Supersedes:** the *sequencing* clause of [ADR-0004](ADR-0004-dense-substrate-v0.1.0.md)
  (MoE ⇒ v0.2.0-B). ADR-0004's dense-default and untied-embedding decisions stand.

## Context

ADR-0004 deferred MoE-from-scratch to v0.2.0-B to protect the 14-day dense ship (a G6
ship-cadence risk), while naming it the natural *headline* result: every open-weight 2026
flagship is sparse MoE, and the hottest failure mode — MoE breaks GRPO via train/infer routing
mismatch (R3: ~94% of tokens differ in ≥1 layer), diagnosed by our `kl_train_infer` pillar — is
exactly A1.1's artifact. The Navigator chose to build A1.1 now. The constitution's rule for this
is explicit: *don't relitigate — log an ADR.* This is that ADR.

## Decision

1. **Build A1.1 now, as a separate opt-in module** (`src/reasoning_llm/moe.py`). The dense path
   is the default and is **byte-for-byte unchanged**: a model is sparse only when
   `ModelConfig.moe is not None`. Nothing is smuggled into the v0.1.0 dense ship — the smoke run
   stays dense.
2. **Recipe = DeepSeek-V3** (arXiv:2412.19437, Eq. 12-20): sigmoid affinity `s_{i,t}=σ(u·e_i)`,
   Top-K with the **aux-loss-free bias** `b_i` entering *selection only* (gate value uses raw `s`),
   normalize-over-selected, shared + (optionally fine-grained) routed experts, FFN returns the
   **delta only**. Bias updated by the trainer post-step: `b_i += γ·sign(mean_load − load_i)`.
3. **Composed A1.1 add-ons** (flagged as non-V3): router **z-loss** on raw pre-sigmoid logits
   (ST-MoE stabilizer; classically a softmax tool), a tiny **seq-wise balance loss** (V3 Eq. 17-20,
   batch-wise simplification on single device), and **diagnostics** (per-expert load histogram +
   router entropy) feeding the through-line.
4. **Out of scope** (single-device, from-scratch): node/device-limited routing and the
   device/communication balance losses — pure expert-parallel comm optimizations. A GPU/EP
   follow-up; not a v0.1.0 gap.
5. **Kill criterion carried from A1.1:** `>2 debug days` ⇒ revert to dense + GQA, MoE back to a
   v0.2.0-B axis. The opt-in design makes revert trivial (dense default is never touched).

## Consequences

- (+) The MoE×RL collapse study (A5.3) now has a from-scratch MoE to break and fix (GSPO/R3),
  turning the headline failure mode into an owned, instrumented experiment.
- (+) Correctness is pinned by two anchors: **dense-equivalence** (a 1-expert top-1 MoE *is* a
  SwiGLU — guards the double-residual bug) and **decode/cache parity** (full-forward == incremental
  decode logits with MoE active — the `kl_train_infer` invariant). Plus loss-at-init≈logV,
  overfit-one-batch, aux-loss-free bias direction, and the falsifiable entropy harness
  (`router entropy > 0.9·log N_r`). 13 tests, all CPU.
- (+) `routed_scaling_factor` defaults to **1.0** (V3 uses ≈2.5) so loss-at-init stays clean;
  exposed as config for faithful-V3 runs.
- (−) The dense v0.1.0 surface grew by one optional config field and a `return_aux` path. Mitigated:
  both default to the prior behavior; the full dense suite is unchanged and green.
- (−) CPU expert compute is an explicit per-expert loop (clear, correct, O(N_r) matmuls); a
  vectorized batched-gather and real expert-parallelism are GPU/perf follow-ups.
