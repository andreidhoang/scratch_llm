# Kernels — the meat boundary (loaded when working in `src/scratch_llm/kernels/`)

## Location map (where everything lives)

The kernel surface is mirrored across three trees that share the `<family>/<backend>/` shape:

```
src/scratch_llm/kernels/<family>/<backend>/   ← the kernels themselves (this tree)
bench/kernels/<family>/                       ← one roofline bench per kernel (index: bench/kernels/README.md)
tests/kernels/test_*.py                       ← one oracle-first correctness test per kernel
bench/_harness.py                             ← shared roofline measurement spine (all benches import this)
bench/kernels/run.py                          ← `python -m bench.kernels.run [name|family]` — the ladder runner
csrc/                                         ← AOT CUDA C++ sources for the frontier rungs (sm_90a/sm_100a)
CMakeLists.txt / setup.py                     ← the AOT build → `_scratch_llm_kernels.so` (rental box; [gpu-aot] extra)
```

Run a kernel bench: `PYTHONPATH=src python -m bench.kernels.run gemm` (family) or
`... run gemm/triton_tiled` (one). `run.py` lists every bench, its rung label, the kernel module it
exercises, and auto-skips GPU benches on a CPU box. See `bench/kernels/README.md` for the full table.

## Build system (how kernels compile)

Two equivalent paths — the JIT dev path and the AOT rental-box path:

- **JIT (the default for dev iteration):** every CUDA backend loader has a `@lru_cache _module()`
  that calls `torch.utils.cpp_extension.load` / `load_inline` at first call. The target ISA is
  read at runtime from `common.arch.arch_name()`, so the same loader builds for sm_120 / sm_90 /
  sm_100 without an edit. Triton kernels JIT-compile on first launch (some autotuned). This is
  what runs on the sm_120 dev box — no build step, just `import` + call.
- **AOT (the rental-box path for the frontier rungs):** `CMakeLists.txt` + `setup.py` compile every
  `.cu` under `csrc/` into ONE torch extension (`_scratch_llm_kernels.so`), the CUTLASS/xformers/
  FlashAttention convention. Built via `TORCH_CUDA_ARCH_LIST="9.0a;10.0a" pip install -e ".[gpu-aot]"`
  or `cmake -B csrc/build -DTARGET_ARCHS="sm_90a;sm_100a"`. See `csrc/README.md` + `docs/design/
  CMAKE_AOT_SPEC.md` + ADR-0019.

**The JIT→AOT swap (every loader):** `_module()` tries the AOT extension first
(`from scratch_llm import _scratch_llm_kernels`), falls back to JIT on ImportError. The public
entry function is untouched; the swap is internal. See `gemm/cuda/mma_sync.py::_module()` for the
canonical pattern.

**What is AOT vs JIT today:** the frontier rungs (WGMMA/tcgen05/FA3/fp8/stream_k/persistent under
`csrc/`) are AOT-built (they need sm_90a/sm_100a). The shipped learning rungs (mma.sync /
smem_tiled / wmma) stay JIT — they run on the sm_120 dev box and their `.cu` sources are co-located
with the Python loader (wmma's is an inline string). AOT-building them is a documented TODO in
ADR-0019 (would need source duplication or wmma string extraction — no perf gain on the dev box).

**CPU-safety:** the AOT extension is NEVER built on a CPU-only box. `import scratch_llm` never
reaches `_module()` (PEP 562 lazy + the `gpu` pytest marker). The `try/except ImportError` keeps
the lazy chain intact whether or not the extension is present.

## The dispatch contract (the production boundary — holds in BOTH modes)

Every consumer **above** `kernels/` reaches a kernel backend **only through its
family's `dispatch.py`** — never through the backend subpackage. This is the load-
bearing structural rule; it is enforced by an executable test
(`tests/kernels/test_kernel_dispatch_boundary.py`), not by convention.

```
ops/                          →  what the model calls (stable torch.library custom_op)
kernels/<family>/dispatch.py  →  arch / dtype / shape routing (the seam)
kernels/<family>/reference.py →  correctness oracle (CPU-safe, eager)
kernels/<family>/{prefill,decode,triton,cuda,wmma}/  →  hot paths (PRIVATE)
kernels/common/               →  shared primitives (arch detection, online-softmax) — below the backend layer
```

**What this means in practice:**

- `model.py`, `ops/`, and anything else outside `kernels/` import
  `from scratch_llm.kernels.<family>.dispatch import <symbol>` — full stop. They
  never import `_fa2_fwd_kernel`, `paged_decode_attention`, `wmma_gemm`, etc.
  straight from a backend file.
- `dispatch.py` is the ONLY module that may name a backend (it does so via a PEP 562
  `__getattr__` lazy map, so importing dispatch is still CPU-safe).
- Tests and benches are exempt — they intentionally target ONE backend to verify a
  specific rung of a ladder, not to dispatch.
- CPU-safety (the `kernels/` package invariant): `import scratch_llm.kernels` pulls
  NEITHER triton NOR `torch.utils.cpp_extension`. The oracle is imported eagerly;
  every GPU backend loads lazily on first explicit attribute access.

**Adding a backend:** register it in the family's `_LAZY_GPU` map (re-export surface)
and, if it adds a routing *choice*, add the branch in the dispatch's routing function
(e.g. `gemm/dispatch.py::matmul` now routes on `(device, dtype, arch)` — Hopper fp16 →
`wgmma_gemm`, Blackwell-DC fp16 → `tcgen05_gemm`, Hopper fp8 → `fp8_gemm`). If the backend
is a CUDA C++ kernel, drop the `.cu` under `csrc/<family>/` + bind it in `csrc/pybind.cpp` +
add it to `_SOURCES` in `CMakeLists.txt`/`setup.py` (see `csrc/README.md` §"Adding a new C++
kernel"). Callers above `kernels/` do not change.

**When you can lift a kernel to "shipped":** a rental experiment
(`performance/rental/kernels/`) graduates into `kernels/<family>/<backend>/` once it
has (a) an oracle-first correctness test, (b) a measured roofline line in
`bench/RESULTS.md`, and (c) a dispatch entry. Then delete the rental copy.

---

> **MODE SWITCH ([ADR-0013](../../../docs/adr/ADR-0013-execution-mode-full-delegation.md)).** This
> file describes the **`learn`-mode** contract. When `.claude/execution-mode` is **`delegate`**
> (current since 2026-07-03), agents implement kernel bodies end-to-end; what survives is
> oracle-first tests written by an independent context (bench-writer), the adversarial
> kernel-ship-reviewer gate, and the profile-DoD. The sections below apply as written only in
> `learn` mode; in `delegate` mode read them as the *study syllabus* for the post-hoc mastery pass.

> **Why this file exists.** In the CUDA-for-Deep-Learning kernel sprint, **you reconstruct the kernel
> from blank; agents do everything *around* it.** This isn't a preference: copying a kernel — from the
> book OR from an agent — builds nothing (the "illusion of fluency"), and the live interview rounds are
> AI-free. Guardrailed AI (hints, not answers) is the only kind that doesn't atrophy the skill you're
> here to build (PNAS 2025; Lancet endoscopist study 2025).

## The meat — HUMAN-only in `learn` mode. (In `delegate` mode: agent-built, reviewer-gated.)

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
