# ADR-0007 — MoE-from-scratch as an opt-in module

- **Status:** Accepted (2026-06-06)
- **Layer:** A1 Substrate (MoE extension)
- **Refines:** the sequencing of [ADR-0004](ADR-0004-dense-substrate-v0.1.0.md) — MoE is built now,
  as opt-in. ADR-0004's dense-default and untied-embedding decisions stand.

## Context

Every open-weight 2026 flagship is sparse MoE, and the MoE training failure modes (router
load-imbalance; train/inference routing mismatch) are worth understanding from the inside. Building a
from-scratch MoE FFN — as an opt-in module that never touches the dense default — is a high-value A1
extension.

## Decision

1. **Build the MoE FFN as a separate opt-in module** (`src/scratch_llm/moe.py`). The dense path is the
   default and is **byte-for-byte unchanged**: a model is sparse only when `ModelConfig.moe is not None`.
2. **Recipe = DeepSeek-V3** (arXiv:2412.19437, Eq. 12–20): sigmoid affinity `s_{i,t}=σ(u·e_i)`, Top-K
   with the **aux-loss-free bias** `b_i` entering *selection only* (the gate value uses raw `s`),
   normalize-over-selected, shared + (optionally fine-grained) routed experts, FFN returns the
   **delta only**. Bias updated by the trainer post-step: `b_i += γ·sign(mean_load − load_i)`.
3. **Composed add-ons** (flagged as non-V3): router **z-loss** on raw pre-sigmoid logits (ST-MoE
   stabilizer), a tiny **seq-wise balance loss** (V3 Eq. 17–20, batch-wise on single device), and
   **diagnostics** (per-expert load histogram + router entropy).
4. **Out of scope** (single-device, from-scratch): node/device-limited routing and the
   device/communication balance losses — pure expert-parallel comm optimizations, a GPU/EP follow-up.
5. **Kill criterion:** if MoE costs `>2 debug days`, revert to dense + GQA (the opt-in design makes
   revert trivial — the dense default is never touched).

## Consequences

- (+) A from-scratch sparse model you can whiteboard — and a substrate to study MoE training failure
  modes (routing load-imbalance, train/inference mismatch) later if desired.
- (+) Correctness pinned by two anchors: **dense-equivalence** (a 1-expert top-1 MoE *is* a SwiGLU —
  guards the double-residual bug) and **decode/cache parity** (full-forward == incremental-decode
  logits with MoE active). Plus loss-at-init ≈ logV, overfit-one-batch, aux-loss-free bias direction,
  and the falsifiable entropy harness (`router entropy > 0.9·log N_r`). 13 CPU tests.
- (+) `routed_scaling_factor` defaults to **1.0** (V3 uses ≈2.5) so loss-at-init stays clean; exposed
  as config for faithful-V3 runs.
- (−) The dense surface grew by one optional config field and a `return_aux` path; both default to the
  prior behavior, the dense suite is unchanged and green.
- (−) CPU expert compute is an explicit per-expert loop; a vectorized batched-gather and real
  expert-parallelism are GPU/perf follow-ups.
