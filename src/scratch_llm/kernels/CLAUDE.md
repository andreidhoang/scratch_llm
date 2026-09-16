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

## Active ownership and measurement policy — v5

The authority is `../ladders/CLAUDE.md` and `../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md`
(paths from the repository root), followed by root `AGENTS.md` and `CLAUDE.md`.
The old ADR-0013 mode switch is historical. Neither `.claude/execution-mode` nor
`.claude/hooks/kernel-write-guard.sh` exists in the reviewed checkout. No write guard enforces
ownership; the existing lint/CI hooks have a different purpose.

Huy writes the first core kernel, loss math and memory model, derives the tolerance and makes the
prediction. Agents may build scaffolds, independent references, tests, benchmark adapters and maps.
They may review profiles and propose mechanisms; Huy owns the diagnosis, and agents implement a
selected fix after he names it. The stricter `src/scratch_llm/k3/core/` boundary is unaffected.

A kernel experiment starts with its active rung contract: supported shapes/strides, dtype and
accumulation behavior, numerical oracle, baseline, performance hypothesis and stop rule. Write tests
against the intended semantics, not a copied implementation. Tests collected or skipped on CPU
do not demonstrate that an empty or wrong kernel fails on GPU.

Measure a matched baseline on the target GPU and record the actual runner protocol. Profiling helps
explain a bottleneck; low arithmetic intensity alone does not exclude launch, synchronization,
dependency or occupancy limits. State a cache policy and distinguish p20–p80 spread from IQR.
The legacy shared harness is reusable infrastructure, not automatic compliance with v5.

Record results and raw artifacts in the ladders campaign, with local `bench/RESULTS.md` links
where useful. A correct but slower kernel or a falsified optimization hypothesis is an interpretable
result; it does not earn an unmet speed target. A source's promotion into `kernels/` or `csrc/`
does not certify its runtime correctness, architecture coverage or production suitability.

For a mastery rep: derive → predict → implement → test → profile → explain the failed prediction →
make one controlled variant. A cold defence and an independent reproduction strengthen the evidence.
No claim about Anthropic's private interview format or automatic transition to unrestricted
delegation follows from completing the drill.
