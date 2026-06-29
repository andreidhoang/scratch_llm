// gemv.cu — fused GEMV decode kernel (L2 Systems / inference track).
//
// Intent: the canonical *decode-time* primitive. y = A @ x, A:(M,K) row-major weight, x:(K,) the
// single activation vector, y:(M,). At batch=1/seq=1 every Linear in the model IS this op — the
// arithmetic intensity collapses (2·M·K flops over an M·K weight read → AI ≈ 0.5 FLOP/B), so the
// whole kernel is one streaming HBM read of A and the DoD is "% of the ~1.0 TB/s HBM roofline",
// NOT TFLOP/s. This is *why* LLM decode is bandwidth-bound and why weight-quantization (fewer bytes
// per A element) is the lever. Invariant: close to the pure-torch oracle (gemv_ref) within atol.
// Interview question this answers: "at decode, what bounds a Linear, and what is its AI?"
//
// MEAT BOUNDARY: the __global__ kernel BODY below is YOUR from-blank reconstruct (the per-row dot
// product + block reduction is the learning rep — same shape as rmsnorm.cu's reduction, one rung up).
// Everything around it — headers, tensor checks, dtype dispatch, the <<<grid,block>>> launch, the
// binding — is scaffolding. Fill the body in your own editor; until you do, the kernel writes
// nothing and tests/test_gemv_cuda.py stays RED (that is the loop).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// THE MEAT — reconstruct from blank. Contract you must satisfy:
//   grid = M output rows, block = THREADS lanes. Output row m = blockIdx.x.
//   1. each thread strides over k ∈ [tid, K) accumulating A[m,k] * x[k]  (read+accumulate in fp32)
//   2. block-reduce the partial dot products (warp shuffle → shared mem → broadcast) to get acc
//   3. thread 0 writes y[m] = acc           (cast back to scalar_t)
// A is row-major so A[m,k] = A[m*K + k] — consecutive threads read consecutive k → coalesced.
// Accumulate in fp32 even for half/bf16 inputs (numerics discipline). Templated on scalar_t.
//
// The rung ladder (reconstruct one at a time, re-profiling % of HBM roofline after each):
//   R0 one block/row, scalar loads        R1 vectorized loads (float4 / half2 over k)
//   R2 one warp/row (raise occupancy)      R3 multiple rows/block to reuse x from shared/regs
// ---------------------------------------------------------------------------
// Warp-level sum reduction
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level sum reduction for 256 threads
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // 1024 threads max / 32 = 32 warps
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write warp sum to shared memory

    __syncthreads();              // Wait for all warp reductions

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) val = warpReduceSum(val); // Final reduction within first warp

    return val;
}

template <typename scalar_t>
__global__ void gemv_kernel(
    const scalar_t* __restrict__ A,   // (M, K) row-major weight
    const scalar_t* __restrict__ x,   // (K,)   activation vector
    scalar_t* __restrict__ y,         // (M,)   out
    const int M,
    const int K) {
    
    int bid = blockIdx.x;
    if (bid >= M) return;

    int tid = threadIdx.x;
    
    // each thread calculates its own partial output.
    float acc = 0.0f;

    // stride-k loop: each thread reads A[bid, tid], A[bid, tid + blockDim.x], etc.
    for (int col = tid; col < K; col += blockDim.x) {
        acc += (float) A[bid * K + col] * (float) x[col];
    }

    // Block level sum reduction
    float sum = blockReduceSum(acc);
    if (tid == 0) {
        y[bid] = (scalar_t) sum;
    }
}

// ---------------------------------------------------------------------------
// SCAFFOLD (agent-owned): launcher + binding. The pybind registration lives in csrc/bindings.cpp
// (one aggregated module); this file only DEFINES the launcher it forward-declares there.
// ---------------------------------------------------------------------------
torch::Tensor gemv_cuda(torch::Tensor A, torch::Tensor x) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2-D (M, K)");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == A.size(1), "x must be (K,) matching A's last dim");
    // Multi-GPU correctness: pin the current device to A's device for this scope (and restore on
    // return), so the kernel + getCurrentCUDAStream() target the right GPU even if A is on cuda:k.
    const c10::cuda::CUDAGuard device_guard(A.device());
    A = A.contiguous();
    x = x.contiguous();

    const int M = A.size(0);
    const int K = A.size(1);
    auto y = torch::empty({M}, A.options());

    const int threads = 256;          // one block per output row; tune later (128/256/512)
    const dim3 grid(M);
    const dim3 block(threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, A.scalar_type(), "gemv_cuda", [&] {
            gemv_kernel<scalar_t><<<grid, block, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                M, K);
        });
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}
