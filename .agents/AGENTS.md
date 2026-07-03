# Project-Scoped Rules for Gemini

When working in the `scratch_llm` repository, you must always align with the core context engineering principles and harness guidelines defined in this project.

## Context Engineering & Harness Reference
- The constitution is `CLAUDE.md` at the repo root (mission, disciplines, Mode boundaries, build/test commands). The Gemini-specific deltas live in [docs/GEMINI_CONTEXT.md](../docs/GEMINI_CONTEXT.md) — read both before acting.
- Kernel implementations are **mode-switched** (read `.claude/execution-mode`, see `docs/adr/ADR-0013-execution-mode-full-delegation.md`): in `learn` mode do NOT write or edit GPU kernel implementations directly (files under `src/scratch_llm/kernels/` matching `*_triton.py` / `*_kernel.py`) — follow the Socratic tutor method to guide the user in implementing them. In `delegate` mode (current since 2026-07-03) agents implement kernels end-to-end; oracle-first tests and adversarial review still gate every commit.
- Ensure all CI validation checks pass (`ruff`, `pyright`, `pytest -m "not gpu"`) before completing a task.
- **Formatting Rule:** Always render math, formulas, equations, and numbers in clean, human-readable plain text. Avoid using LaTeX math dollar signs (`$` or `$$`) to ensure correct rendering in the user interface.
