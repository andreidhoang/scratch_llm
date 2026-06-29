// rmsnorm.cu — fused RMSNorm decode kernel (L2 Systems / inference track).
//
// Intent: the canonical *memory-bound* decode primitive. y = x * rsqrt(mean(x^2) + eps) * w,
// reduction over the last (hidden) dim D. One CUDA block per row; the whole op is a single
// HBM read of x + a single write of y (+ the tiny w broadcast), so the DoD is "% of the
// ~1.0 TB/s HBM roofline", NOT TFLOP/s. Invariant: bitwise-close to the pure-torch oracle
// (rmsnorm_ref) in inference.py within the test's atol. Interview question this answers:
// "why is RMSNorm at decode bandwidth-bound, and what is its arithmetic intensity?"
//
// MEAT BOUNDARY: the __global__ kernel BODY below is YOUR from-blank reconstruct (the reduction
// is the learning rep). Everything around it — headers, tensor checks, dtype dispatch, the
// <<<grid,block>>> launch, the binding — is scaffolding. Fill the body in your own editor; until
// you do, the kernel writes nothing and tests/test_rmsnorm_cuda.py stays RED (that is the loop).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// THE MEAT — reconstruct from blank. Contract you must satisfy:
//   grid = N rows, block = THREADS lanes. Row r = blockIdx.x.
//   1. each thread strides over d ∈ [tid, D) accumulating x[r,d]^2 (read+accumulate in fp32)
//   2. block-reduce the partial sums (warp shuffle → shared mem → broadcast) to get sumsq
//   3. inv_rms = rsqrtf(sumsq / D + eps)
//   4. each thread strides over d again writing y[r,d] = x[r,d] * inv_rms * w[d]
// Accumulate in fp32 even for half/bf16 inputs (numerics discipline). Templated on scalar_t.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void rmsnorm_kernel(
    const scalar_t* __restrict__ x,   // (N, D) row-major
    const scalar_t* __restrict__ w,   // (D,)
    scalar_t* __restrict__ y,         // (N, D) out
    const int N,
    const int D,
    const float eps) {
    // TODO(human): the per-row reduction + normalized write. ~15 lines. Profile with
    // kernels.inference.rmsnorm_roofline() and drive it toward the HBM roofline.
}

// ---------------------------------------------------------------------------
// SCAFFOLD (agent-owned): launcher + binding. The pybind registration lives in csrc/bindings.cpp
// (one aggregated module); this file only DEFINES the launcher it forward-declares there.
// ---------------------------------------------------------------------------
torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2-D (N, D); flatten (B,T,D) before calling");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "weight must be (D,)");
    // Multi-GPU correctness: pin the current device to x's device for this scope (and restore on
    // return), so the kernel + getCurrentCUDAStream() target the right GPU even if x is on cuda:k.
    const c10::cuda::CUDAGuard device_guard(x.device());
    x = x.contiguous();
    weight = weight.contiguous();

    const int N = x.size(0);
    const int D = x.size(1);
    auto y = torch::empty_like(x);

    const int threads = 256;          // one block per row; tune later (256/512/1024)
    const dim3 grid(N);
    const dim3 block(threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "rmsnorm_cuda", [&] {
            rmsnorm_kernel<scalar_t><<<grid, block, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                N, D, static_cast<float>(eps));
        });
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}
