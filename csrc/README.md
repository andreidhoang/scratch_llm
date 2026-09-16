# `csrc/` — the AOT CUDA C++ source tree

The AOT kernel-build substrate. The configured `.cu` sources build into ONE torch extension
(`_scratch_llm_kernels.so` / `.pyd`) via `CMakeLists.txt` (repo root) or `setup.py`
(the pip path). This is the CUTLASS / xformers / FlashAttention convention — one
extension, one symbol table, one `pybind.cpp` registration.

Active priorities, ownership and evidence gates come from
[`../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md`](../../ladders/plan/SIXTY_DAYS_SIX_LADDERS.md)
and repository [`AGENTS.md`](../AGENTS.md). The source inventory below retains historical
compile/stub reports; it does not prove a current complete build or GPU execution. Select only
the active v5 experiment. Architecture-specific compilation and a Python entry point are
prerequisites for runtime verification, not substitutes for it.

## What's here

```
csrc/
├── pybind.cpp                          ← the single PYBIND11_MODULE registration
├── gemm/
│   ├── wgmma_sm90.cu                   ← Hopper WGMMA GEMM (sm_90a, promoted from rental/)
│   ├── tcgen05_sm100.cu                ← Blackwell tcgen05 GEMM (sm_100a, promoted from rental/)
│   ├── fp8_gemm_sm90.cu                ← Hopper FP8 GEMM (STUB — learning rep)
│   └── stream_k_sm90.cu                ← Hopper stream-K GEMM (STUB — learning rep)
├── attention/
│   └── fa3_hopper.cu                   ← Hopper FlashAttention-3 (sm_90a, promoted from rental/)
└── persistent/
    └── persistent_gemv_sm90.cu         ← Hopper persistent GEMV (STUB — learning rep)
```

The Python loaders that bind these live under `src/scratch_llm/kernels/<family>/<backend>/`
and reach the AOT extension via `from scratch_llm import _scratch_llm_kernels` (falling back
to JIT `torch.utils.cpp_extension.load` if the AOT build is absent — see the JIT→AOT swap in
`src/scratch_llm/kernels/gemm/cuda/mma_sync.py::_module()`).

## Build (on the rental box — H100, H200, or B200)

### Path A: pip (the torch-blessed path)

```bash
TORCH_CUDA_ARCH_LIST="9.0a;10.0a" pip install -e ".[gpu-aot]"
```

### Path B: CMake (the out-of-tree developer path)

```bash
cmake -B csrc/build -DTARGET_ARCHS="sm_90a;sm_100a"
cmake --build csrc/build -j
```

Both produce `_scratch_llm_kernels.so` next to the package. Verify:

```bash
python -c "from scratch_llm import _scratch_llm_kernels as e; print(dir(e))"
```

## Target archs

| Arch    | GPU            | ISA                          | CUDA toolkit |
|---------|----------------|------------------------------|--------------|
| sm_90a  | H100 / H200    | WGMMA + TMA + mbarrier       | 12.4+        |
| sm_100a | B200           | tcgen05 + TMEM + cta_group   | 13.0+        |

These sources target the architecture-specific ISA (`sm_90a` / `sm_100a`). Check the actual
toolchain, architecture guards and emitted PTX/SASS before interpreting a compile result;
unsupported targets can fail or compile a guarded fallback, rather than proving the intended
instructions execute. The table records this repository's target configurations, not universal
minimum toolkit versions. The sm_120 development configuration is not an AOT target here.

## Adding a new C++ kernel

1. Write the `.cu` under `csrc/<family>/`; declare its launcher in `csrc/pybind.cpp`.
2. Add the `.cu` to `_CUDA_SOURCES` in `setup.py` AND `_SOURCES` in `CMakeLists.txt`.
3. Bind it in `csrc/pybind.cpp`: `m.def("my_kernel", &my_kernel);`
4. Add the Python loader under `src/scratch_llm/kernels/<family>/<backend>/` using the AOT-or-JIT
   swap pattern. Add the dispatch entry, the oracle test, and the bench row.

See `docs/design/CMAKE_AOT_SPEC.md` §4 and `docs/adr/ADR-0019` for the full recipe.

## Honesty note

Three of these files (wgmma, tcgen05, fa3) are **promoted rental skeletons**: they compile for
their target archs with documented `nvcc -arch=sm_Xa -ptx` commands and PTX-grep gates, but
runtime correctness is deferred to the selected GPU experiment (H100 / H200 / B200). The host
launchers expose Python entry points, but availability does not establish a working runtime.
Run the relevant oracle tests (for example `tests/kernels/test_gemm_wgmma.py`) with the declared
input/numerical contract and record actual results before timing or claiming correctness.

Three (fp8, stream_k, persistent) are **stubs**: the host launcher raises `runtime_error` and the
Python loader raises `NotImplementedError` pointing at the exact file + reference. The kernel body
is the learning rep — left for you to implement.

The rental originals under `performance/rental/kernels/` stay as provenance references until each
kernel graduates per the `kernels/CLAUDE.md` "lift to shipped" rule (oracle test + measured
roofline line + dispatch entry).
