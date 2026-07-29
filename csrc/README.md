# `csrc/` — the AOT CUDA C++ source tree

The production kernel build. Every `.cu` here compiles into ONE torch extension
(`_scratch_llm_kernels.so` / `.pyd`) via `CMakeLists.txt` (repo root) or `setup.py`
(the pip path). This is the CUTLASS / xformers / FlashAttention convention — one
extension, one symbol table, one `pybind.cpp` registration.

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

The `a` suffix is REQUIRED. WGMMA and tcgen05 assemble ONLY under the accelerated ISA
(`sm_90a` / `sm_100a`); base `sm_90` / `sm_100` silently emit zero `wgmma.*` / `tcgen05.*`
instructions. The sm_120 dev box (RTX PRO 4000 Blackwell client) has no WGMMA / tcgen05 —
it is NOT an AOT target.

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
runtime correctness is deferred to the rental measurement day (H100 / H200 / B200). The host
launchers appended here make them *runnable* from Python; the oracle tests
(`tests/kernels/test_gemm_wgmma.py` etc.) gate correctness on the rental box.

Three (fp8, stream_k, persistent) are **stubs**: the host launcher raises `runtime_error` and the
Python loader raises `NotImplementedError` pointing at the exact file + reference. The kernel body
is the learning rep — left for you to implement.

The rental originals under `performance/rental/kernels/` stay as provenance references until each
kernel graduates per the `kernels/CLAUDE.md` "lift to shipped" rule (oracle test + measured
roofline line + dispatch entry).
