# Project-Scoped Rules for Gemini

When working in the `scratch_llm` repository, you must always align with the core context engineering principles and harness guidelines defined in this project.

## Context Engineering & Harness Reference
- Always refer to [GEMINI_CONTEXT.md](file:///Users/danghuyhoang/Desktop/cs336/scratch_llm/docs/GEMINI_CONTEXT.md) (or the relative path `docs/GEMINI_CONTEXT.md` on Vast.ai) for the project topology, test commands, and specific AI development rules.
- Do NOT write or edit GPU kernel implementations directly (files under `src/scratch_llm/kernels/matmul.py` or matching `*_triton.py` / `*_kernel.py`). Follow the Socratic tutor method to guide the user in implementing them.
- Ensure all CI validation checks pass (`ruff`, `pyright`, `pytest -m "not gpu"`) before completing a task.
