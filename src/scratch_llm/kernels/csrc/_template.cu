// _template.cu — copy me to start a new inference kernel. Rename foo -> your kernel.
//
// This is the SCAFFOLD shape every kernel in this dir follows: a templated __global__ (the body is
// YOUR from-blank rep — leave the TODO until you write it in your editor) + a host launcher that
// does tensor checks, dtype dispatch, grid/block, and the <<<>>> launch (plumbing — already done).
//
// Steps to wire a new kernel (mirror RMSNorm; full checklist in this dir's README.md):
//   1. cp _template.cu <domain>/foo.cu ; rename foo_kernel / foo_cuda  (pick the domain dir:
//      norm/ gemm/ attention/ activation/ ...; files starting with _ are NOT compiled)
//   2. ops.h: add 1 forward decl ; bindings.cpp: add 1 m.def for foo_cuda (the public ABI)
//   3. inference/<domain>.py: add foo_ref (oracle), foo (dispatch via _ext().foo_cuda), foo_roofline
//   4. add tests/test_foo_cuda.py (copy test_rmsnorm_cuda.py, swap the oracle)
// The launcher signature here MUST match its forward declaration in ops.h (the ABI contract).
// _build.py globs csrc/**/*.cu recursively, so a new domain dir is picked up automatically.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// ---- THE MEAT (your rep) — reconstruct from blank; agents won't write this. ----
template <typename scalar_t>
__global__ void foo_kernel(
    const scalar_t* __restrict__ in,   // (N, D) row-major
    scalar_t* __restrict__ out,        // (N, D)
    const int N,
    const int D) {
    // TODO(human): the kernel algorithm. Accumulate reductions in fp32. One block per row is the
    // usual decode layout (grid=N, block=THREADS); thread `tid` strides d in [tid, D) by THREADS.
}

// ---- SCAFFOLD (agent-owned): launcher. Register foo_cuda in csrc/bindings.cpp (decl + m.def). ----
torch::Tensor foo_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D (N, D); flatten leading dims before calling");
    const c10::cuda::CUDAGuard device_guard(x.device());   // multi-GPU correctness (always do this)
    x = x.contiguous();

    const int N = x.size(0);
    const int D = x.size(1);
    auto out = torch::empty_like(x);

    const int threads = 256;            // tune later (256/512/1024)
    const dim3 grid(N);
    const dim3 block(threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "foo_cuda", [&] {
            foo_kernel<scalar_t><<<grid, block, 0, stream>>>(
                x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), N, D);
        });
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
