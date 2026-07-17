---
description: Visualize a kernel mechanism — use an existing tool if one covers it, else build a single-file interactive HTML stepper into performance/viz/ (tiny numeric example, predict-before-advance). Viz code is Mode-1 delegate.
argument-hint: "<mechanism>, e.g. cute layout swizzle / paged KV / GDN recurrence / bank conflicts"
---
Visualize: $ARGUMENTS

> Roadmap §6.5 rule: a viz you watched passively is decoration. Every viz here forces prediction
> before each step, shows a tiny hand-checkable numeric example, and ends with `/feynman`.

1. **Use-first check.** If a maintained tool already covers $ARGUMENTS, point there and stop:
   Triton-Viz (Triton program visualization, SIGCSE'26) · Compiler Explorer CUDA (PTX/SASS,
   source-correlated) · Nsight Compute roofline · `cute::print_layout` / `print_latex` (CuTe layouts,
   textual/LaTeX) · Modal GPU Glossary (concepts) · Mojo GPU Puzzles roofline explorer · Boehm/Gordić
   annotated worklogs. Building a worse copy of an existing tool is waste.

2. **Build on gap** (verified gaps as of 2026-07-14: interactive CuTe layout/swizzle explorer,
   paged-KV block-table animator, GDN/delta-rule recurrence stepper, coalescing/bank-conflict grid).
   Requirements:
   - Single self-contained HTML file → `performance/viz/<snake_name>.html` (no external deps, or
     cdnjs only; dark-mode friendly; opens in any browser; git-versioned).
   - **Three lenses present** (the /master rule): the shapes/layout, the data-flow, and a tiny
     numeric worked example (4–8 elements) whose arithmetic the human can check by hand.
   - **Stepper, not movie**: Step/Run/Reset controls; state panel showing every running quantity;
     before each Step the UI shows a "predict:" prompt (what will m/l/the address/the bank be?).
   - One deliberate toggle that teaches the frontier variant where one exists (e.g., FA4 lazy-rescale
     threshold on the online-softmax stepper; swizzle on/off on the layout explorer).
   - ≤ ~300 lines. Round every displayed number.

3. **The learning gate.** After the human has stepped through it: ask for one-sentence answers —
   "what is the invariant?", "what breaks if we remove X?" — then hand off to `/feynman $ARGUMENTS`.
   Log the artifact in the viz-library list (roadmap §6.5) so it isn't rebuilt.

Mode note: viz code is a teaching aid, not the interview-tested skill → Claude may write it fully
(Mode-1), even in learn mode. The kernel it explains stays human-typed.
