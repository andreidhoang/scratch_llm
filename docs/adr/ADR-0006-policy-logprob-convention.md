# ADR-0006 — Rollout policy log-prob convention

- **Status:** Accepted (2026-06-06)
- **Layer:** L2 Systems (A2.3) `rollout/` → consumed by L2 `utils/monitors.py`, L5 `algos/`
- **Decides:** §3 of `docs/design/L2_rollout_seam_SPEC.md`; sibling of [ADR-0003](ADR-0003-sampler-parity.md)

## Context

A `Rollout` carries one log-probability per generated token, and that number defines
what "the policy" *is* for everything downstream: GRPO/Dr.GRPO advantages, the off-policy
importance ratios + ESS, and the `KL(current‖ref)` / `KL(current‖old)` terms. There are two
candidates for the value:

1. **Raw policy** `log π(a|s) = log_softmax(logits)[a]` at temperature 1 — the model's own
   next-token distribution, independent of how the token was sampled.
2. **Sampling distribution** — the log-prob under the post-temperature, post-top-p
   renormalized distribution actually drawn from.

If the rollout stored (2), advantages and IS-ratios would couple to the *exploration* knobs:
raising temperature or tightening top-p would silently rescale the policy-gradient signal, and
`kl_train_infer` would mix engine drift with a sampling-config artifact (the exact failure
[ADR-0003](ADR-0003-sampler-parity.md) exists to prevent, one level down).

## Decision

**A `Rollout` stores the raw policy log π(a|s) at temperature 1** — `log_softmax(logits)[a]` of
the *taken* token under the unscaled model distribution. Temperature and top-p remain pure
exploration knobs that shape *which* tokens are drawn, never the logged probability of them.

This is the standard PPO/GRPO convention and exactly the input
`monitors.importance_ratios(logp_current, logp_old)` and the KL terms expect.

**Implementation:** `sampling._logprob_of` reads the log-prob from the *same* logits the token
was sampled from; `LocalBackend.score` re-derives it by teacher forcing as an independent check
(the contract test uses `top_p < 1` precisely so a regression to convention (2) would diverge and
fail). The grad-bearing current-policy log-prob for the L5 policy loss is **recomputed in the
training step** — the rollout's stored value is a frozen, `no_grad` diagnostic, not a loss input.

## Consequences

- (+) Advantages / IS-ratios / the three KLs are invariant to temperature and top-p — exploration
  and credit assignment stay decoupled.
- (+) `monitors` consumes rollout log-probs directly; no per-call convention negotiation.
- (+) The SGLang backend must return the same raw-policy log π for `kl_train_infer` to be valid —
  a concrete parity requirement to verify when that backend lands (its quantized/fused path may
  report a different distribution; record its log-prob contract, cf. A2 guide §8 #5).
- (−) The stored log-prob is *not* the density of the realized sampling step; any future
  off-policy correction that needs the true behavior policy must recompute it explicitly.
