# ADR-0005 — Multimodal / VLM is a v0.2.0 axis (not v0.1.0)

- **Status:** Accepted (2026-06-05)
- **Layer:** L1/L5 (A1.x / A5.3 extensions)
- **Decides:** the multimodal half of the spec's *"v0.2.0 (additive): MoE×RL collapse, multimodal — ADR
  stubs only, never smuggle into v0.1.0."* (The MoE half is recorded in [ADR-0004](ADR-0004-dense-substrate-v0.1.0.md).)

## Context

The open frontier includes the VLM spurious-reward gap (VLM-R1 reward hacking; PEARL), which makes a
vision-language env (`envs/vlm_verifiable.py`) an attractive *additional* VERA axis. But v0.1.0 is a
frozen 14-day text-only reasoning-RL ship; adding a vision encoder, image pipeline, and a VLM policy
is a G6 ship-cadence violation and dilutes the one honest finding.

## Decision

**Multimodal / VLM stays out of v0.1.0.** It is a **v0.2.0** additive axis:
- `envs/vlm_verifiable.py` and any vision-encoder / VLM-policy code are **not** built in v0.1.0.
- The v0.1.0 `true_quality_gap` finding (VERA) is text/math only.
- If pursued, multimodal reuses the *same* engine (`algos/`, `rewards/`, `envs/exploitability.py`,
  `envs/true_quality.py`, `utils/monitors.py`) — it adds an env, not a new engine.

## Consequences

- (+) v0.1.0 scope and the single finding stay sharp; the ship date is protected.
- (+) The VLM spurious-reward gap remains a clean, pre-scoped v0.2.0 headline if the engine lands.
- (−) No multimodal result in v0.1.0 — accepted; it is explicitly the v0.2.0 axis.
