// ops.h — the single declaration header for the scratch_llm inference-kernel suite.
//
// Every kernel's host launcher is DEFINED in its own csrc/<domain>/<name>.cu (compiled by nvcc);
// this header forward-DECLARES them all in one place. `bindings.cpp` includes this and registers
// each via `m.def`. The declaration is the ABI contract — keep each signature byte-for-byte in
// lock-step with the launcher in the .cu. To add a kernel: one line here + one m.def in bindings.cpp.

#pragma once
#include <torch/extension.h>

// ---- norm/ ----
torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps);

// ---- gemm/ ----
torch::Tensor gemv_cuda(torch::Tensor A, torch::Tensor x);
torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B);
