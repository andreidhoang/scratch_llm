// bindings.cpp — the single public ABI for the scratch_llm inference-kernel suite.
//
// One aggregated extension (`scratch_llm_kernels`, the shippable `_C` shape). Launchers are DEFINED
// in csrc/<domain>/<name>.cu (nvcc); their forward declarations live in ops.h; this file just
// registers each in the pybind module. To wire a NEW kernel: add one decl to ops.h + one m.def here
// (nothing else — inference/_build.py globs csrc/**/*.cu automatically). This .cpp is plumbing, not
// meat — the `__global__` body stays in the .cu and is yours to reconstruct.
//
// Built host-side by g++ via torch.utils.cpp_extension.load; TORCH_EXTENSION_NAME is defined by the
// build to "scratch_llm_kernels".

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_cuda", &rmsnorm_cuda, "Fused RMSNorm over the last dim (decode, memory-bound)");
    m.def("gemv_cuda", &gemv_cuda, "Matrix-vector y = A@x (decode Linear, memory-bound)");
    m.def("gemm_cuda", &gemm_cuda, "Matrix-matrix C = A@B (tiled GEMM, compute-bound)");
}
