# AGENTS.md — rules for AI agents working in this repo

## What this repo is

`scratch_llm` — the production vehicle for linear-attention numerics & determinism
(`src/scratch_llm/mastery/`, a live vLLM review), sitting on a from-scratch CS336 stack that
stays green. **Spec of record: `PLAN.md` (the ONLY plan — it wins every contradiction), then
`CLAUDE.md` (constitution) and `docs/KERNEL_MASTERY_SPEC.md` (§12.5 = the session ladder).**
K3 (build & host Kimi K3 from scratch): `docs/k3/ROADMAP.md` + `docs/k3/FACTS.md` — the claim
ledger; where a secondary source and the K3 tech report arXiv:2607.24653 disagree, the tech
report wins and the disagreement is logged in FACTS.md.

## THE SEAL — four things agents never write

`src/scratch_llm/mastery/reference.py` · the measurement harness · RL loss math (advantage,
KL estimators, IS ratio, clipping) · verifier logic and tolerances. Hook-enforced by
`.claude/hooks/oracle-guard.sh`. Execution mode is `learn` (`.claude/execution-mode`):
offer failing tests, a critique, or a post-hoc review instead. Everything else is delegable.

## Build / verify

```bash
uv pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests
pyright
pytest -m "not gpu"          # CPU gate, mirrors CI — must stay green
```

Engineering disciplines that are the point of the repo: loss-at-init ≈ log(vocab) ·
overfit-one-batch · fixed-seed reproducibility · predict-before-you-run · every measured
number lands in `bench/RESULTS.md` with hardware + method.

## THE BOUNDARY — `src/scratch_llm/k3/core/` is hand-built

**Agents never create, edit, move, or delete files under `src/scratch_llm/k3/core/`.**
Those modules encode the mechanisms the human is mastering; a silent bug there is exactly
what the human must learn to catch. For core modules agents may ONLY:
- write adversarial tests in `tests/` after the human authors a module (red team, no fixes);
- write proposals as markdown for the human to retype — never as diffs to apply.

Full rules + per-module mastery bars: `src/scratch_llm/k3/HANDCRAFTED.md`.
Everything outside `core/` follows the paired/delegated split defined there.

## Honesty ledger

Claims are labelled **measured by us / reported / not verified** (FACTS.md, bench/RESULTS.md).
Never dress an unverified number up as measured; never silently "fix" a failing gate —
report it.
