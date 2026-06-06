# ADR-0003 — Train/infer sampler parity contract

- **Status:** Accepted (2026-06-05)
- **Layer:** L1 Substrate (A1) → consumed by L2 `rollout/`, L2 `utils/monitors.py`
- **Decides:** §8.4 of `A1_basics_BUILD_GUIDE.md`

## Context

`kl_train_infer` must isolate *engine* drift (different kernels/precision/
batching between the training engine and the serving engine). If the training-
time sampler and the rollout sampler use different decode params (temperature,
top-p, stop tokens), the KL is dominated by that config delta — a self-inflicted
bug masquerading as the headline finding.

## Decision

**One `SamplingParams` object is the single source of truth**, consumed by both
the from-scratch decoder and `rollout/sglang_client.py`. Fields: `temperature`,
`top_p`, `max_tokens`, `stop_ids`, `seed`.

**Contract:** `utils/monitors.py` asserts both engines were invoked with
identical `SamplingParams` before it trusts any `kl_train_infer` value. A
mismatch invalidates the measurement.

## Consequences

- (+) `kl_train_infer` reflects real engine drift, the quantity HALT@0.10 guards.
- (+) Decode semantics (temperature → top-p → stop) are defined once.
- (−) Requires the from-scratch decoder and the serving client to share a config
  type — a deliberate coupling, enforced by a parity assertion.
