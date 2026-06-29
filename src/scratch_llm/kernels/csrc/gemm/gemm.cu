// gemm.cu — tiled GEMM (the COMPUTE-bound archetype, R4 rung). C = A @ B.
//
// Intent: the compute-bound counterpart to gemv.cu. C:(M,N) = A:(M,K) @ B:(K,N). Unlike decode-GEMV
// (one streaming weight read, memory-bound), a square GEMM does 2·M·N·K flops over (MK+KN+MN)
// elements → arithmetic intensity GROWS with the tile size, so a big-enough tile crosses the ridge
// into COMPUTE-bound and the DoD is "% of cuBLAS peak TFLOP/s", NOT % of HBM bandwidth. This is *why*
// shared-memory tiling + register blocking exist (reuse each loaded element T times) and the on-ramp
// to tensor cores / WGMMA (the CUTLASS frontier — awareness per ADR-0011).
// Interview question this answers: "why does a bigger tile move a GEMM up the roofline, and what caps the tile?"
//
// MEAT BOUNDARY: the __global__ kernel BODY below is YOUR from-blank reconstruct (the tiling is the
// learning rep). Everything around it — headers, tensor checks, dtype dispatch, the <<<grid,block>>>
// launch, the binding — is scaffold. Fill the body in your own editor; until you do, C is
// uninitialized and tests/test_gemm_cuda.py stays RED (that is the loop).

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#ifndef GEMM_TILE
#define GEMM_TILE 16  // tile/block edge: one (TILE×TILE) thread block computes one (TILE×TILE) C tile
#endif

// ---------------------------------------------------------------------------
// THE MEAT — reconstruct from blank. Contract you must satisfy:
//   Launch is 2-D: block = (GEMM_TILE, GEMM_TILE); grid = (ceil(N/TILE), ceil(M/TILE)).
//   This thread owns output element  C[row, col]  where
//       row = blockIdx.y * GEMM_TILE + threadIdx.y   (in [0, M))
//       col = blockIdx.x * GEMM_TILE + threadIdx.x   (in [0, N))
//   Row-major:  A[m,k] = A[m*K + k] ,  B[k,n] = B[k*N + n] ,  C[m,n] = C[m*N + n].
//   Guard the M/N edges (when M or N is not a multiple of TILE). Accumulate in fp32 even for
//   half/bf16 inputs (numerics discipline); cast back to scalar_t on store.
//
// The rung ladder (reconstruct one at a time, re-profiling % of cuBLAS after each):
//   R0 naive: each thread loops k, acc += A[row,k]·B[k,col]                 (low AI → memory-bound)
//   R1 shared-memory tiling: cooperatively stage TILE×TILE blocks of A and B into __shared__,
//      __syncthreads(), accumulate the partial products, advance k by TILE  (AI ∝ TILE → toward ridge)
//   R2 register blocking / thread coarsening: each thread computes a small C×C micro-tile of C,
//      reusing the staged tiles from registers (fewer shared loads → higher AI)
//   R3 vectorized loads (float4 / half2) + double-buffered shared tiles to hide latency
//   R4 tensor cores (wmma / WGMMA) — the CUTLASS frontier; awareness (ADR-0011), not this rung
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void gemm_kernel(
    const scalar_t* __restrict__ A,   // (M, K) row-major
    const scalar_t* __restrict__ B,   // (K, N) row-major
    scalar_t* __restrict__ C,         // (M, N) row-major
    const int M,
    const int N,
    const int K) {
    // TODO(human): the tiled GEMM. Start at R0 (naive), profile % of cuBLAS, then climb the ladder
    // above. Map (row, col) from blockIdx/threadIdx; accumulate over k in fp32; guard the M/N edges;
    // write C[row*N + col]. Agents won't write this — it's your rep.
}

// ---------------------------------------------------------------------------
// SCAFFOLD (agent-owned): launcher. Forward-declared in csrc/ops.h, registered in csrc/bindings.cpp.
// ---------------------------------------------------------------------------
torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2-D (A:(M,K), B:(K,N))");
    TORCH_CHECK(A.size(1) == B.size(0), "inner dims must match: A:(M,K) @ B:(K,N)");
    // Multi-GPU correctness: pin the current device to A's device for this scope (restored on return).
    const c10::cuda::CUDAGuard device_guard(A.device());
    A = A.contiguous();
    B = B.contiguous();

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    const dim3 block(GEMM_TILE, GEMM_TILE);
    const dim3 grid((N + GEMM_TILE - 1) / GEMM_TILE, (M + GEMM_TILE - 1) / GEMM_TILE);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, A.scalar_type(), "gemm_cuda", [&] {
            gemm_kernel<scalar_t><<<grid, block, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, N, K);
        });
    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}
