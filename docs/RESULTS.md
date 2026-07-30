# Frontier ablation results — predicted vs measured

Pre-registered measurement ledger for the F-front ablations defined in
`docs/FRONTIER_2026_TASKSPEC.md`.  A row is `[FACT]` only once measured; until
then it records the falsifiable prediction and the kill criterion.

## F6 · MoE balancing ablation

Registered **before** running. Spec: `docs/FRONTIER_2026_TASKSPEC.md` §F6.
Thesis: aux-loss-free bias balancing (`BIAS_FREE`) achieves final validation CE
≤ sequence-level auxiliary loss (`SEQ_AUX`) while keeping router entropy above
`0.9·log(N_r)`.  The granularity axis (`COARSE` vs `FINE`) holds total routed
parameters and active routed FLOPs constant via the mapping in
`src/scratch_llm/eval/moe_ablation.py`.

| arm | granularity | predicted final val CE | measured final val CE | entropy > 0.9·log(N_r) | status |
|---|---|---|---|---|---|
| BIAS_FREE | COARSE | ≤ SEQ_AUX | — (pending) | yes | pending |
| BIAS_FREE | FINE | ≤ SEQ_AUX | — (pending) | yes | pending |
| SEQ_AUX | COARSE | baseline | — (pending) | yes | pending |
| SEQ_AUX | FINE | baseline | — (pending) | yes | pending |
| NONE | COARSE | worse by ≤ 0.02 nats or collapsed | — (pending) | no | pending |
| NONE | FINE | worse by ≤ 0.02 nats or collapsed | — (pending) | no | pending |

**Expected falsifier:** `BIAS_FREE` final val CE ≤ `SEQ_AUX` with entropy
> `0.9·log(N_r)` after training.

**Kill line:** `BIAS_FREE` worse than `SEQ_AUX` by > 0.02 nats, or entropy
< `0.9·log(N_r)` after training (balancer fails to prevent collapse).

**Artifacts:** `scripts/f6_moe_ablation.py`, `tests/test_moe_ablation.py`,
`artifacts/f6_moe_ablation/`.
