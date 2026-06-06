# ADR-0002 — GQA in the policy substrate

- **Status:** Accepted (2026-06-05)
- **Layer:** L1 Substrate (A1)
- **Decides:** §8.3 of `A1_basics_BUILD_GUIDE.md`

## Context

The 2026 default decoder is GQA (grouped-query attention) — its purpose is
KV-cache economics: the KV cache is 60–85% of wall-clock past ~1M context, and
GQA shrinks it by `n_heads / n_kv_heads`. CS336 A1 builds full MHA. GQA is a
small adaptation (K/V head-sharing) on top of MHA.

## Decision

**Build full MHA; expose GQA as a config flag.** `ModelConfig.n_kv_heads`
satisfies `n_kv_heads ≤ n_heads`; when fewer, K/V heads are shared across query-
head groups (`repeat_interleave` KV up to `n_heads` before SDPA). Full MHA is the
special case `n_kv_heads == n_heads`.

## Consequences

- (+) The from-scratch build is the GQA / KV-cache-economics interview artifact
  for ~10 extra lines.
- (+) The served-policy code path matches how 2026 models actually run.
- (−) Slightly more reshape complexity in the attention module — already the #1
  bug site. Mitigated by `test_rope_relative` + the MHA shape-contract tests.
