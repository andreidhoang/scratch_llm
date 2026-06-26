# ADR-0001 — Tokenizer of record (when fine-tuning a pretrained model)

- **Status:** Accepted (2026-06-05)
- **Layer:** A1 Substrate
- **Decides:** §8.2 of `A1_basics_BUILD_GUIDE.md`

## Context

A1 builds a byte-level BPE tokenizer from scratch. But A5's RL fine-tuning runs on a pretrained HF
model (e.g. a Qwen-class math model) that ships its **own** tokenizer. Two engines — the training
engine and, if used, a separate serving/inference engine — must index the **same** vocabulary, or
any per-token comparison between them (e.g. `kl_train_infer = KL(P_train ‖ P_infer)`, a sum over the
vocab) is meaningless.

## Decision

**The tokenizer of record is the fine-tuned model's native tokenizer.** The hand-rolled BPE is the
A1 mastery artifact and powers the from-scratch tiny-LM; the A5 RL pipeline encodes every task and
rollout with the pretrained model's tokenizer.

**Contract:** at startup, assert `vocab_size` and every special-token ID (`<|endoftext|>` + any
reasoning delimiters) are identical across the training and serving engines. A mismatch aborts the
run — never silently tolerated, because it compares incommensurable axes.

## Consequences

- (+) Per-token train/inference comparisons measure real engine drift, not a vocab artifact.
- (+) The from-scratch BPE still pays off: it is why you can read/patch token IDs and trust
  special-token stability.
- (−) The hand-rolled tokenizer is not on the A5 critical path; time-box it, don't gold-plate.
