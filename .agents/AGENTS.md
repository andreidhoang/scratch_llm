# Project-Scoped Rules for Gemini

When working in the `scratch_llm` repository, you must always align with the core context engineering principles and harness guidelines defined in this project.

## Context Engineering & Harness Reference
- The constitution is `CLAUDE.md` at the repo root (mission, disciplines, Mode boundaries, build/test commands). The Gemini-specific deltas live in [docs/GEMINI_CONTEXT.md](../docs/GEMINI_CONTEXT.md) — read both before acting.
- Do NOT write or edit GPU kernel implementations directly (files under `src/scratch_llm/kernels/` matching `*_triton.py` / `*_kernel.py`). Follow the Socratic tutor method to guide the user in implementing them.
- Ensure all CI validation checks pass (`ruff`, `pyright`, `pytest -m "not gpu"`) before completing a task.
- **Formatting Rule:** Always render math, formulas, equations, and numbers in clean, human-readable plain text. Avoid using LaTeX math dollar signs (`$` or `$$`) to ensure correct rendering in the user interface.
