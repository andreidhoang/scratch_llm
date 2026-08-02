# AGENTS.md — rules for AI agents working in this repo

## What this repo is

`scratch_llm` — a from-scratch, production-grade CS336 stack + 2026 frontier-practice layer,
built by hand for mastery. Spec of record for process: `README.md`, `docs/STATUS.md`,
`docs/IMPLEMENTATION_PLAN.md`. The K3 track (build & host Kimi K3 from scratch):
`docs/k3/ROADMAP.md` + `docs/k3/FACTS.md` (claim ledger — where a secondary source and the
K3 tech report arXiv:2607.24653 disagree, the tech report wins, and the disagreement is
logged in FACTS.md).

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
