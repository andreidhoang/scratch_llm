# `csrc/` — inference kernels (CUDA C++ bound to PyTorch)

C/C++/CUDA source root (the PyTorch-ecosystem convention: torch, vLLM, FlashInfer, sgl-kernel all
use `csrc/`). Kernels are grouped **by domain** (`norm/`, `gemm/`, … — add `attention/`,
`activation/`, `quant/`, `sampling/` as you go); one `.cu` per kernel. They build into **one
aggregated extension** (`scratch_llm_kernels`, the shippable `_C` shape) via `ops.h` (declarations)
+ `bindings.cpp` (the `PYBIND11_MODULE` ABI). The `__global__` **body is your from-blank rep** (the
meat boundary — `../CLAUDE.md`); everything around it (build, binding, oracle, test, roofline) is
scaffold. At decode (batch=1, seq=1) these ops are **HBM-bandwidth bound** → DoD is *% of the memory
roofline*, not TFLOP/s.

```
csrc/
├── norm/rmsnorm.cu      launcher + __global__ body (your rep)
├── gemm/gemv.cu         "
├── ops.h                forward declarations for every launcher (centralized)
├── bindings.cpp         #include "ops.h"; one m.def per kernel (the public ABI)
├── _template.cu         copy-to-start scaffold (not compiled)
└── README.md
```

## The three layers (kernel → binding → wrapper)

| Layer | File | Job |
|---|---|---|
| **Kernel** | `csrc/<domain>/<name>.cu` | `__global__` body (your rep) + host launcher (checks, dispatch, launch) |
| **Binding** | `csrc/ops.h` + `csrc/bindings.cpp` | forward-decl + `m.def` per kernel — the ABI (g++) |
| **Wrapper** | `../inference/<domain>.py` | Pythonic API: flatten dims, dispatch via `_ext()`, + oracle + roofline |

## Build model

`inference/_build.py::_ext()` calls `torch.utils.cpp_extension.load(name="scratch_llm_kernels",
sources=[csrc/**/*.cu (recursive, excluding _*), csrc/bindings.cpp], extra_cuda_cflags=_CUDA_CFLAGS)`.
Built with **ninja**, so:

- Editing one `.cu` recompiles **only that object** and relinks — incremental, no manual build step.
- Cached under `~/.cache/torch_extensions`; first build ~10s, then instant until a source changes.
- A new domain dir is picked up automatically (recursive glob); files starting with `_` are excluded.
- `SCRATCH_LLM_KERNEL_VERBOSE=1` surfaces the build log + `ptxas -v` (registers / shared mem / spills
  per kernel → occupancy reasoning).

### Compile flags (`_CUDA_CFLAGS` in `inference/_build.py`)
- `-O3` — optimize.
- `-lineinfo` — keep optimization but emit SASS↔source mapping so `ncu`/Nsight show your `.cu`
  lines. Essential for the profile DoD; ~free.
- `--ptxas-options=-v` — print per-kernel register/shared-mem usage at build.
- **No global `--use_fast_math`** — fast-math is a per-op call: use `rsqrtf` / `__expf` directly
  in-kernel where you want it, so a softmax kernel's numerics aren't silently degraded module-wide.

## Add a kernel — 4 small edits, no infra change

Adding `rope` (a `pos_encoding`/`norm`-adjacent op). `_ext()`, the glob, and pyright config are
untouched:

1. **`csrc/<domain>/rope.cu`** — `cp _template.cu <domain>/rope.cu`, rename `foo_*` → `rope_*`. Write
   the `__global__` body yourself; keep the launcher shape (incl. the `CUDAGuard` line — always).
2. **`csrc/ops.h`** — add the forward declaration; **`csrc/bindings.cpp`** — add the `m.def`.
3. **`../inference/<domain>.py`** — add `rope_ref` (oracle), `rope` (dispatch via `_ext().rope_cuda`),
   `rope_roofline`; export the three names from `../inference/__init__.py`.
4. **`../../tests/test_rope_cuda.py`** — copy `test_rmsnorm_cuda.py`, swap the oracle. It is
   `gpu`-marked + `importorskip`'d, so the CPU/CI gate skips it automatically.

## Launcher conventions (the scaffold every kernel keeps)

- **`CUDAGuard`** — `const c10::cuda::CUDAGuard device_guard(x.device());` first thing. Pins the
  current device to the input's device (multi-GPU correctness) so the kernel and
  `getCurrentCUDAStream()` target the right GPU; restored on return. **Non-negotiable.**
- **`.contiguous()`** — the kernel indexes raw offsets; a sliced/transposed tensor has other strides.
- **dtype dispatch** — `AT_DISPATCH_FLOATING_TYPES_AND2(Half, BFloat16, ...)` → one templated kernel
  serves fp16/bf16/fp32. (Don't hardcode `data_ptr<float>()` — real decode is bf16.)
- **stream** — launch `<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>`, not the default stream.
- **launch check** — `C10_CUDA_CHECK(cudaGetLastError());` after the launch.
- **accumulate reductions in fp32** even for fp16/bf16 inputs.
- **one block per output row** (`grid = N|M`, `block = 256`) is the default decode layout.

## The one gotcha: the ABI declaration

The launcher is *defined* in the `.cu` (nvcc); `ops.h` declares it and `bindings.cpp` (g++) registers
it — separate TUs. The declaration in `ops.h` must match the `.cu` launcher signature byte-for-byte,
or the link fails / you call the wrong ABI. Treat `ops.h` as the contract.

## Scope & future splits

Inference-only today (forward kernels; the training/backward kernels are the Triton FA2 in
`../flash_attention_triton.py`). If CUDA *training* kernels ever appear, split into
`csrc/inference/` + `csrc/training/` with a separate `bindings.cpp` per side and the `_fwd` / `_bwd`
suffix convention — until then a flat (inference-implied) `csrc/` avoids empty ceremony.

## Graduation: JIT → AOT → custom op (later, drop-in)

`load()` (JIT) is right for the iterate loop. Two graduations, both additive — `ops.h`,
`bindings.cpp`, and every `.cu` carry over unchanged:

1. **AOT** — `setup.py` with `CUDAExtension(sources=glob("csrc/**/*.cu") + ["csrc/bindings.cpp"])` +
   `BuildExtension`, then `pip install -e .` → a pre-built `.so` (no per-process compile).
2. **Ship-grade custom op** — register via `TORCH_LIBRARY` + `torch.library.register_fake` (a meta
   impl for shape inference) instead of raw `PYBIND11_MODULE`. Plain pybind ops are opaque to
   `torch.compile` (graph breaks) and autograd; `TORCH_LIBRARY` composes with both + CUDA graphs.
   Do this when a kernel goes into a served model, not while iterating on its body.
