# Gemini onboarding — pointer + the Gemini-specific deltas only

> **The constitution is [`../CLAUDE.md`](../CLAUDE.md)** — mission, module map, engineering
> disciplines, the "Orient before you build" protocol, Mode boundaries, build/test commands.
> Read it first; **this file holds only what is Gemini-specific**, so it cannot drift from the
> single source of truth. (The previous long version of this file duplicated CLAUDE.md and rotted —
> it referenced files removed in the 2026-07-01 perf reset.)

## Orient (read state, don't assume it)

1. `git log --oneline -15` — what just shipped.
2. `performance/PERF_PLAN.md` §Current Node — the active front (the perf curriculum).
3. `docs/STATUS.md` — build state · `bench/RESULTS.md` — the measured ledger.
4. Harness manual (why every `.claude/` file exists): `docs/CONTEXT_ENGINEERING.md`.

## Gemini-specific rules

1. **Formatting:** render math/formulas/numbers in clean plain text — **no LaTeX dollar signs**
   (`$`/`$$`), they don't render in the Gemini UI.
2. **Kernel meat boundary (same as every agent — mode-switched, ADR-0013):** read
   `.claude/execution-mode` first. In `learn` mode: never write/edit kernel bodies — anything under
   `src/scratch_llm/kernels/` matching `*_triton.py` / `*_kernel.py` (enforced for Claude by
   `.claude/hooks/kernel-write-guard.sh`; honor it voluntarily); tutor Socratically, scaffold
   tests/benches, review — never the rep itself. In `delegate` mode (current since 2026-07-03):
   kernel implementation is delegated to agents; oracle tests + adversarial review still gate.
   Boundary details: `src/scratch_llm/kernels/CLAUDE.md`.
3. **k3/core boundary (same as every agent — root `AGENTS.md` §THE BOUNDARY, which binds every
   agent session):** agents never create, edit, move, or delete files under
   `src/scratch_llm/k3/core/` — those modules encode the mechanisms the human is mastering; a
   silent bug there is exactly what the human must learn to catch. For core modules agents may
   ONLY: (a) write adversarial tests in `tests/` after the human authors a module (red team, no
   fixes); (b) write proposals as markdown for the human to retype — never as diffs to apply.
   Full rules + per-module mastery bars: `src/scratch_llm/k3/HANDCRAFTED.md`; everything outside
   `core/` follows the paired/delegated split defined there.
4. **Green-CI before done:** `ruff check src tests` · `ruff format --check src tests` · `pyright` ·
   `pytest -m "not gpu"` (the pre-commit hook enforces this on `git commit`; don't bypass).
5. **Context hygiene:** reset between unrelated tasks; rebuild context from the durable state above,
   not from chat history; write conclusions/code to files before long-running commands; no
   placeholder/mock code.

## Fresh pod

`bash scripts/bootstrap-pod.sh` — full runbook: `docs/VASTAI_BOOTSTRAP.md` (env + hook + memory
restore + verify). Don't duplicate those steps here.
