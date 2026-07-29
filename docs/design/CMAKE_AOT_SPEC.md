# Design spec — CMake AOT build for the frontier-kernel extension

> **Status:** ✅ built 2026-07-21 — landed as `CMakeLists.txt` (repo root) + `setup.py` +
> `csrc/` tree. Kept as the design record. See ADR-0019 for the decision, `csrc/README.md` for
> the operator guide.
> **Layer:** L2 Systems — the production build path for `src/scratch_llm/kernels/` frontier rungs.
> **Files:** `CMakeLists.txt` (CMake build), `setup.py` (pip-friendly torch CUDAExtension build),
> `csrc/pybind.cpp` (single registration), `csrc/<family>/*.cu` (the sources).

## 1. Why (the problem this solves)

The kernel stack was JIT-only: every CUDA backend compiled at first call via
`torch.utils.cpp_extension.load`, each its own extension. That's the research-grade default — it
costs 10-60s of nvcc idle per kernel at the start of every bench run, re-links torch headers N
times, and has no single symbol table. Production stacks (CUTLASS, xformers, flash-attn, Mamba,
vLLM) ship ONE AOT extension built via CMake or `setup.py + CUDAExtension`. This spec is that build.

## 2. The mechanics

### Two equivalent build paths (same output, different ergonomics)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Path A: pip (the torch-blessed CUDAExtension path)                          │
│   TORCH_CUDA_ARCH_LIST="9.0a;10.0a" pip install -e ".[gpu-aot]"             │
│   → builds via setup.py + BuildExtension + CUDAExtension                    │
│   → drops _scratch_llm_kernels.so next to src/scratch_llm/                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ Path B: CMake (the out-of-tree ninja-driven path)                           │
│   cmake -B csrc/build -DTARGET_ARCHS="sm_90a;sm_100a"                       │
│   cmake --build csrc/build -j                                               │
│   → same _scratch_llm_kernels.so; incremental builds via ninja              │
└──────────────────────────────────────────────────────────────────────────────┘
```

Path A is the standard torch pattern (what xformers, Mamba use). Path B is the developer-
experience path (incremental, `cmake --build` only recompiles what changed). Both compile the
exact same `csrc/*.cu` list into the same extension.

### Target archs (the rental frontier)

| Arch    | GPU            | ISA features                          | CUDA toolkit |
|---------|----------------|---------------------------------------|--------------|
| sm_90a  | H100 / H200    | WGMMA + TMA + mbarrier                | 12.4+        |
| sm_100a | B200           | tcgen05 + TMEM + cta_group            | 13.0+        |

The `a` suffix is REQUIRED: WGMMA and tcgen05 are in the *accelerated* ISA, not base sm_90/sm_100.
Compiling for `sm_90` (no `a`) silently emits zero WGMMA instructions.

The sm_120 dev box (RTX PRO 4000 Blackwell client) is NOT an AOT target: no WGMMA, no tcgen05.
It runs only the JIT learning rungs (mma.sync / smem_tiled / wmma / Triton).

### The JIT→AOT swap point (every loader)

```python
@lru_cache(maxsize=1)
def _module():
    # 1. AOT path — prefer the built extension (no JIT cost on first call)
    try:
        from scratch_llm import _scratch_llm_kernels as _ext
        if hasattr(_ext, "gemm_mma_sync"):     # symbol present in the build
            return _ext
    except ImportError:
        pass
    # 2. JIT fallback — the dev-iteration path (nvcc at first call)
    from torch.utils.cpp_extension import load
    return load(name="gemm_mma_sync", sources=[str(_CU)], ...)
```

The public entry function (`gemm_mma_sync`, `wgmma_gemm`, etc.) is untouched. The swap is internal.

## 3. What is AOT vs JIT today

| Kernel | Source location | Default path | Why |
|--------|-----------------|--------------|-----|
| WGMMA (sm_90a)   | `csrc/gemm/wgmma_sm90.cu`       | AOT | Promoted rental skeleton; sm_90a gated |
| tcgen05 (sm_100a)| `csrc/gemm/tcgen05_sm100.cu`    | AOT | Promoted rental skeleton; sm_100a gated |
| FA3 (sm_90a)     | `csrc/attention/fa3_hopper.cu`  | AOT | Promoted rental skeleton; sm_90a gated |
| FP8 (stub)       | `csrc/gemm/fp8_gemm_sm90.cu`    | AOT | Scaffold; sm_90a gated |
| stream-K (stub)  | `csrc/gemm/stream_k_sm90.cu`    | AOT | Scaffold; sm_90a gated |
| persistent (stub)| `csrc/persistent/persistent_gemv_sm90.cu` | AOT | Scaffold; sm_90a gated |
| mma.sync         | sibling `mma_sync.cu`           | JIT  | Learning rung; runs on dev box sm_120 |
| smem_tiled       | sibling `smem_tiled.cu`         | JIT  | Learning rung; runs on dev box sm_120 |
| wmma             | inline `_CUDA_SRC` string       | JIT  | Learning rung; source is Python literal |

AOT-building the shipped learning rungs (mma.sync, smem_tiled, wmma) is a documented TODO: would
require duplicating `.cu` into `csrc/` or extracting the wmma inline string. Left as-is because the
value of AOT is the frontier kernels, not the learning rungs (which run on the dev box, not the
rental frontier box).

## 4. Adding a new C++ kernel (the 3-step recipe)

1. Write the `.cu` under `csrc/<family>/`; declare its launcher in `csrc/pybind.cpp`.
2. Add the `.cu` to `_CUDA_SOURCES` in `setup.py` AND `_SOURCES` in `CMakeLists.txt`.
3. Bind it in `csrc/pybind.cpp`: `m.def("my_kernel", &my_kernel);`

Then the Python loader swaps its `_module()` body to prefer the AOT extension (the same swap
pattern every loader uses). Callers above `kernels/` are unchanged.

## 5. Verification (DoD on the rental box)

```bash
# 1. Build (Path A — pip)
TORCH_CUDA_ARCH_LIST="9.0a;10.0a" pip install -e ".[gpu-aot]"
# 1b. OR build (Path B — CMake)
cmake -B csrc/build -DTARGET_ARCHS="sm_90a;sm_100a" && cmake --build csrc/build -j

# 2. Verify the symbols landed
python -c "from scratch_llm import _scratch_llm_kernels as e; print([s for s in dir(e) if not s.startswith('_')])"
# Expected: ['flash_attention_fa3_forward', 'fp8_gemm_sm90', 'persistent_gemv_sm90',
#            'stream_k_gemm_sm90', 'tcgen05_gemm_sm100', 'wgmma_gemm_sm90']

# 3. Run the frontier benches (the rental measurement day)
python -m bench.kernels.run gemm/cuda_wgmma
python -m bench.kernels.run attention/fa3_hopper
python -m bench.kernels.run --all
```

## 6. Non-goals

- No CUTLASS vendoring (see ADR-0019 — flagged for a future `git submodule add`).
- No shipped-learning-rung AOT (JIT is the right default for dev iteration on sm_120).
- No GPU build verification on CPU-only machines — the CI `gpu-aot-build` job is opt-in via the
  `gpu-aot` PR label and `continue-on-error: true` so it documents the build without gating.
