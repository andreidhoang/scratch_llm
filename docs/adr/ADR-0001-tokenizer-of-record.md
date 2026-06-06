# ADR-0001 — Tokenizer of record for the served policy

- **Status:** Accepted (2026-06-05)
- **Layer:** L1 Substrate (A1)
- **Decides:** §8.2 of `A1_basics_BUILD_GUIDE.md`

## Context

`kl_train_infer = KL(P_train ‖ P_infer)` is a sum over the vocabulary of two
engines' next-token probabilities. The KL is only meaningful if both engines
index the *same* vocabulary. A1 builds a byte-level BPE tokenizer from scratch,
but the v0.1.0 served policy is a Qwen3-class HF model that ships its own
tokenizer.

## Decision

**The tokenizer of record is the served model's native tokenizer.** The
hand-rolled BPE is the mastery artifact and powers only the from-scratch
tiny-LM smoke. The RLVR engine encodes every `envs/*` task and every rollout
with the served model's tokenizer.

**Contract:** at engine startup, assert `vocab_size` and every special-token ID
(`<|endoftext|>` + any reasoning delimiters) are identical across the training
engine and the serving engine. A mismatch aborts the run — it is never silently
tolerated, because it makes `kl_train_infer` compare incommensurable axes.

## Consequences

- (+) `kl_train_infer` measures real engine drift, not a vocab artifact.
- (+) The from-scratch BPE still pays off: it is why we can read/patch token IDs
  and trust special-token stability.
- (−) The hand-rolled tokenizer is not on the metric's critical path; it must be
  time-boxed, not gold-plated.
