# ADR-0019 — CMake AOT build + arch-based dispatch (the frontier-kernel infrastructure)

- **Status:** Accepted (2026-07-21)
- **Layer:** L2 Systems — `src/scratch_llm/kernels/`, `csrc/`, `CMakeLists.txt`, `setup.py`
- **Decides:** how the kernel stack COMPILES (AOT CMake + setup.py vs JIT) and how it ROUTES (arch × dtype × device, not dtype × device only)
- **Supersedes:** none. Extends ADR-0011 (kernel framework policy) and ADR-0013 (execution mode) — the structural contracts stay; this adds the production build + the arch-dispatch seam that ADR-0011 anticipated but never built.
- **Spec:** [`../design/CMAKE_AOT_SPEC.md`](../design/CMAKE_AOT_SPEC.md) · **Build doc:** [`../../csrc/README.md`](../../csrc/README.md)

## Context

ADR-0011 named the kernel-framework policy (Triton-primary, CUDA/CUTLASS second-tier) but left the
**build mechanism** implicit — every CUDA backend JIT-compiled at first call via
`torch.utils.cpp_extension.load` / `load_inline`, each its own extension. ADR-0013 named the
learn/delegate execution-mode boundary but left **arch-based dispatch** absent: `gemm/dispatch.py::
matmul` routed on `(device, dtype)` only; `common/arch.py` had `is_hopper()/is_blackwell()/
require_cc()` but no dispatch layer ever called them.

A 2026-07 frontier-infra audit identified four gaps that separate this stack from a CUTLASS-grade
production stack:

1. **No build system.** Pure-Python hatchling wheel; no `csrc/`, no `CMakeLists.txt`, no
   `CUDAExtension`, no `setup.py`. Every CUDA kernel re-linked torch headers at first call.
2. **No arch-based dispatch.** The `is_hopper()`/`is_blackwell()` seam existed but was wired only
   into the JIT loader's `-arch` flag — never into the routing layer.
3. **3 proven compile-gated skeletons stranded outside the package.**
   `performance/rental/kernels/{wgmma_gemm_sm90a,tcgen05_gemm_sm100a,fa3_attention_hopper}.cu` —
   real structural code with `nvcc -arch=sm_Xa -ptx` commands and PTX-grep gates — never imported,
   never dispatched, never tested from the package.
4. **No frontier stubs.** FP8 / stream-K / persistent — the named frontier gaps — had zero
   scaffolding (no loader, no test, no bench, no dispatch entry). A learner had nowhere to land.

## Decision

1. **Add a CMake + setup.py AOT build that compiles `csrc/*.cu` into ONE torch extension
   (`_scratch_llm_kernels`).** One extension, one symbol table, one torch.library registration —
   the CUTLASS/xformers/FlashAttention convention, not per-kernel JIT.

2. **The JIT loaders stay as the dev-iteration fast path.** Each loader's `_module()` now tries
   the AOT extension first (`try: from scratch_llm import _scratch_llm_kernels`), falls back to JIT
   on `ImportError`. The public entry functions are untouched; the swap is internal and transparent.

3. **Wire `common/arch.py` into the routing layer, not just the JIT flag.** `gemm/dispatch.py::
   matmul` now routes on `(device, dtype, arch)`: Hopper fp16 → WGMMA, Blackwell-DC fp16 → tcgen05,
   Hopper fp8 → fp8_gemm (stub), etc. `attention/dispatch.py` gains a new `flash_attention()`
   routing fn (Hopper → FA3, else → Triton FA2, else → CPU oracle).

4. **Promote the 3 rental skeletons into `csrc/` with torch launchers** so they're package-reachable
   (dispatch + bench + test). Runtime correctness stays "deferred to rental day" — honestly
   documented, same status as the rental originals. The rental copies stay as provenance references
   until a kernel graduates per the "lift to shipped" rule.

5. **Add 3 fresh stubs (FP8 / stream-K / persistent)** with the full scaffolding: compile-gated
   `.cu` skeleton with `// TODO(you)` at the learning boundary, Python loader raising clear
   `NotImplementedError`, xfail oracle test, STUB bench row, dispatch entry. The ladder is visible.

## Why AOT (the alternative considered: stay JIT-only)

JIT (`torch.utils.cpp_extension.load`) is excellent for dev iteration: edit a `.cu`, re-run, nvcc
fires. It is the wrong default for three reasons:

- **Startup cost.** First call of every CUDA backend blocks on nvcc (10-60s per kernel). On a
  rental box where you pay $2-10/hr, that's 5+ minutes of paid idle at the start of every bench
  run, every ncu session, every autotune sweep. AOT pays it once at install time.
- **Frontier ISA gating.** WGMMA / tcgen05 assemble only under `-arch=sm_90a` / `sm_100a`. The JIT
  loader on the rental H100 would need to pass `sm_90a` explicitly (not the auto-detected `sm_90`);
  an AOT build with `TORCH_CUDA_ARCH_LIST="9.0a;10.0a"` gets this right by construction.
- **Production convention.** CUTLASS, xformers, flash-attn, Mamba, vLLM, SGLang all ship AOT
  extensions. The reason is the one-extension model: one compile pass per arch, one symbol table,
  one `torch.library` registration point. Per-kernel extensions re-link torch headers N times and
  are a research-grade smell this stack was carrying.

## Why one extension, not N

One extension: `_scratch_llm_kernels.so` exposes every kernel symbol via `pybind.cpp`. The
alternative — one extension per kernel (`wmma_ext.so`, `mma_sync_ext.so`, ...) — would:

- re-link torch headers N times (slow builds),
- create N `pybind` registration points (more surface for ABI mismatch),
- fragment the symbol table (no single `dir(_scratch_llm_kernels)` overview).

The CUTLASS convention is one extension; this ADR adopts it.

## Why arch dispatch lives in the routing fn, not a separate table

`gemm/dispatch.py::matmul` already owns the `(device, dtype)` policy; adding the arch dimension
there keeps the policy in one auditable place. A separate "arch → backend" table (considered)
would split the routing logic across two files and require a join at call time. The routing fn
reads better: `if is_hopper(): ... elif is_blackwell() and not is_sm120(): ...` — the policy is
the code, not a lookup.

## Non-goals (what this ADR does NOT decide)

- **No CUTLASS vendoring yet.** The CMake `target_include_directories` references
  `third_party/cutlass/include` as a documented next step; the promoted rungs are hand-PTX/inline-
  asm and do not require CUTLASS. Vendoring CUTLASS is a 200MB+ header-only drop best done via
  `git submodule add` as its own decision.
- **No runtime perf claims for frontier kernels.** WGMMA / tcgen05 / FA3 are compile-gated with
  runtime correctness deferred to the rental day (FOP-3 predict-then-measure; FOP-4 implemented ≠
  measured). FP8 / stream-K / persistent are `NotImplementedError` stubs.
- **No shipped-learning-rung AOT yet.** `mma_sync.py` / `smem_tiled.py` / `wmma/gemm.py` stay JIT
  (their `.cu` sources are co-located for sibling lookup, and wmma's source is an inline Python
  string). AOT-building them would duplicate sources or extract the wmma string — behavior churn
  for no perf gain on the sm_120 dev box.
- **No changes to the dispatch contract** (`tests/kernels/test_kernel_dispatch_boundary.py`).
  The new backends live under `kernels/<family>/<backend>/` like the existing ones.

## Consequences

- One new shared library (`_scratch_llm_kernels.so`) lands next to the package on the rental box
  via `pip install -e ".[gpu-aot]"` or `cmake -B csrc/build`.
- CPU CI is byte-identical: the AOT extra is opt-in; the JIT fallback keeps dev iteration working;
  the `try/except ImportError` in every loader keeps the lazy chain intact.
- The frontier ladder is now discoverable: 6 new bench rows in `bench/kernels/`, 6 new tests in
  `tests/kernels/` (3 Hopper arch-gated, 3 xfail stubs), 6 new dispatch entries.
- The rental-day smoke test is documented (`csrc/README.md`): build → verify symbols → run benches.
