"""Cross-cutting infrastructure.

Mostly the A2/A6 distributed-training stack — DDP -> ZeRO-1 -> FSDP -> tensor / pipeline /
expert parallel, plus the analytic comms, memory and MFU cost models that predict a run
before it is launched — alongside RL run monitors (entropy, the three KL divergences,
IS-ratio / ESS), checkpointing, mixed precision and seeding.
"""
