# Vast.ai bootstrap — running the kernel harness on a rented GPU

The Claude Code harness (`.claude/` + `src/scratch_llm/kernels/CLAUDE.md`) is committed, so it loads
automatically when you open Claude Code in this repo on the box. No setup beyond cloning.

## 1. Clone + environment
```bash
git clone https://github.com/andreidhoang/scratch_llm.git && cd scratch_llm
uv sync                                  # project deps
uv pip install torch triton              # GPU box: match the box's CUDA; verify: python -c "import triton"
```

## 2. Sanity check (the profile-DoD harness works)
```bash
nvidia-smi                               # confirm the GPU + driver
PYTHONPATH=src uv run python -c "from scratch_llm.kernels.bench import matmul_roofline; \
  from scratch_llm.kernels.matmul import matmul_naive; matmul_roofline(matmul_naive, 4096, 4096, 4096)"
uv run pytest -m gpu tests/test_matmul.py   # matmul_naive passes; matmul_tiled xfails until you build it
```

## 3. The loop (in Claude Code)
`/kernel-day` → scaffolds the test/bench, then **pauses for you** to reconstruct the kernel from blank
(the meat boundary; the `kernel-write-guard` hook blocks any agent from writing kernel files). Then
`/profile` (diagnose the roofline) → `/kreview` (gate before commit). `/tutor <concept>` when stuck.
**The DoD is the roofline line, not a green test.**

## Notes
- The day grid (`BOOK_SPRINT_2026.md`) lives in your private `interview_synthesis/` folder — not cloned
  here. Paste today's rung + % target into `/kernel-day`, or clone that folder separately if you want it
  on the box.
- Heavy/sensitive files (`.venv`, course PDFs, weights, `*.ncu-rep`/`*.nsys-rep`) are gitignored — they
  never ship. Re-run `nvidia-smi` + the sanity check after any box restart.
