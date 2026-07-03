# Mastery-debt ledger — read-and-teach-back backlog (ADR-0013 + ADR-0014)

> Under delegate mode (ADR-0013) and the main-track sprint (ADR-0014) agents ship first; the Navigator masters from the shipped code.
> Every shipped main-track module adds one row here. A row is **cleared** when its Vietnamese
> Feynman lesson exists in `docs/learning/` and the Navigator has taught it back. Ordering is the
> suggested study order (dependencies first).

| # | Concept | Files (code + tests) | The interview question it answers | Lesson | Cleared |
|---|---------|----------------------|-----------------------------------|--------|---------|
| 1 | ZeRO-1 optimizer-state sharding (owner-broadcast) | `utils/zero1.py` · `tests/test_zero1.py` | How does ZeRO-1 cut memory, and what does it cost in comms vs DDP? | — | ☐ |
| 2 | Training-memory accounting (16–20 B/param → 100B ⇒ TBs) | `utils/memory_math.py` · `docs/design/A2_100B_MEMORY_ONEPAGER.md` | How would you train a 100B-parameter model? | — | ☐ |
| 3 | FSDP / ZeRO-3 mechanics (gather-fwd · reduce-scatter grads · fp32 master shards) | `utils/fsdp.py` · `tests/test_fsdp.py` | What moves on the wire in FSDP, and why ~1.5× DDP bytes? | — | ☐ |
| 4 | Parallelism comms algebra + ring all-reduce | `utils/comms_calc.py` · `docs/design/A2_COMMS_ALGEBRA.md` | When does scaling become comms-bound? | — | ☐ |
| 5 | IsoFLOP / Chinchilla fit (log-log, C=6ND, a+b≈1) | `scaling/isoflop.py` · `bench/a3_isoflop.png` | Derive and defend a scaling law — and when does it break? | — | ☐ |
| 6 | Budget-constrained query planning (reserve/refund semantics) | `scaling/planner.py` | Spend a fixed compute budget to fit a loss surface | — | ☐ |

*(rows appended as modules ship — see `docs/EXECUTION_SPEC_CS336_FINISH.md` for the build DAG)*
