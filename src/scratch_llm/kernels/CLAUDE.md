# Kernels — the meat boundary (loaded when working in `src/scratch_llm/kernels/`)

> **Why this file exists.** In the CUDA-for-Deep-Learning kernel sprint, **you reconstruct the kernel
> from blank; agents do everything *around* it.** This isn't a preference: copying a kernel — from the
> book OR from an agent — builds nothing (the "illusion of fluency"), and the live interview rounds are
> AI-free. Guardrailed AI (hints, not answers) is the only kind that doesn't atrophy the skill you're
> here to build (PNAS 2025; Lancet endoscopist study 2025).

## The meat — HUMAN-only. Claude and all agents MUST NOT write these.

- The `@triton.jit` kernel bodies and the rung implementations: `matmul_tiled`, the FlashAttention
  reconstruct, reductions, the GDN / NVFP4 decode kernel — anything that is the *learning rep*.
- The from-blank RL-math derivations (R5).

You write these in **your own editor**. If asked to implement one, Claude refuses and switches to
tutor mode ("write it yourself first; describe what you tried"). A hard hook
(`.claude/hooks/kernel-write-guard.sh`) blocks Edit/Write to kernel files as a backstop.

## What agents DO (everything around the meat)

- **Scaffold** the failing test + the benchmark *before* you implement — `bench-writer` (so you have a
  target + a DoD).
- **Profile & diagnose** — run the bench / `ncu`, read the roofline, hand you "bound by X, fix = Y" —
  `roofline-analyst`. Never the fixed kernel.
- **Review** the kernel *after* you wrote it — correctness, numerics, "is the speedup real?" —
  `kernel-ship-reviewer`.
- **Teach** the concept Socratically, citing the book — `kernel-tutor`. Will not paste code.
- Write docs / the `bench.py` harness / tests / non-kernel modules.

## The loop — every kernel day (`/kernel-day`)

1. **Predict** the % of cuBLAS/SDPA before any code (the rep starts here).
2. **Reconstruct** the kernel from blank — *you*, in your editor.
3. **Profile** — `/profile`; the DoD is the roofline line, **not** a green test.
4. **Break it** — remove tiling/coalescing, watch it degrade.
5. **Review** — `/kreview`; then **Variant** — re-implement from the algorithm.

## The switch (learning → shipping)

Learning mode (this sprint): agents advise, you implement. Once you can **predict a kernel's roofline
before running it**, you've earned shipping mode — then leverage agents fully per your Agentic
Engineering Playbook: *rent the model, engineer the loop, own the verification.*

## Reliance drill (weekly)

One session/week, work read-only (`claude --permission-mode plan`): read the profiler output and form
**your own** bottleneck hypothesis *before* asking `roofline-analyst`. That's the off-AI check that
catches quiet skill erosion.
