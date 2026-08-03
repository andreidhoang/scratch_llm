# Project-Scoped Rules for Gemini

When working in the `scratch_llm` repository, you must always align with the core context engineering principles and harness guidelines defined in this project.

## Context Engineering & Harness Reference
- The constitution is `CLAUDE.md` at the repo root (mission, disciplines, Mode boundaries, build/test commands). The Gemini-specific deltas live in [docs/GEMINI_CONTEXT.md](../docs/GEMINI_CONTEXT.md) — read both before acting.
- Kernel implementations are **mode-switched** (read `.claude/execution-mode`, see `docs/adr/ADR-0013-execution-mode-full-delegation.md`): in `learn` mode do NOT write or edit GPU kernel implementations directly (files under `src/scratch_llm/kernels/` matching `*_triton.py` / `*_kernel.py`) — follow the Socratic tutor method to guide the user in implementing them. In `delegate` mode (current since 2026-07-03) agents implement kernels end-to-end; oracle-first tests and adversarial review still gate every commit.
- **THE BOUNDARY — `src/scratch_llm/k3/core/` is hand-built** (mirrors the root `AGENTS.md`, which binds every agent session): agents never create, edit, move, or delete files under `src/scratch_llm/k3/core/`. Those modules encode the mechanisms the human is mastering; a silent bug there is exactly what the human must learn to catch. For core modules agents may ONLY:
  - write adversarial tests in `tests/` after the human authors a module (red team, no fixes);
  - write proposals as markdown for the human to retype — never as diffs to apply.

  Full rules + per-module mastery bars: `src/scratch_llm/k3/HANDCRAFTED.md`. Everything outside `core/` follows the paired/delegated split defined there.
- Ensure all CI validation checks pass (`ruff`, `pyright`, `pytest -m "not gpu"`) before completing a task.
- **Formatting Rule:** Always render math, formulas, equations, and numbers in clean, human-readable plain text. Avoid using LaTeX math dollar signs (`$` or `$$`) to ensure correct rendering in the user interface.
