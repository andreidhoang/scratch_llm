// A3 Rung 0 — naive shared-memory-blocked bf16 GEMM (CUDA CORES only, no tensor cores).
//
// This is the re-anchor floor the A3 tensor-core ladder climbs from: a classic 32x32
// SMEM-tiled sgemm/hgemm. bf16 inputs, fp32 accumulate, bf16 output. Every global tile
// load is bounds-masked to 0, so M/N/K remainder tiles (and M=1) are handled without a
// separate epilogue. Accumulation is a plain fp32 inner product per output element — the
// thing WMMA's m16n16k16 fragment MMA will later replace and beat.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define TILE 32

__global__ void sgemm_smem_kernel(const __nv_bfloat16* __restrict__ A,
                                  const __nv_bfloat16* __restrict__ B,
                                  __nv_bfloat16* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;  // output row this thread owns
  const int col = blockIdx.x * TILE + threadIdx.x;  // output col this thread owns

  float acc = 0.0f;
  const int numTiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < numTiles; ++t) {
    const int aCol = t * TILE + threadIdx.x;  // K-index of the A element loaded
    const int bRow = t * TILE + threadIdx.y;  // K-index of the B element loaded

    // Masked cooperative load of the A tile [row, t*TILE:.] and B tile [t*TILE:., col].
    // Out-of-range lanes deposit 0.0f so they contribute nothing to the inner product —
    // this is the entire remainder-tile / M=1 correctness story.
    As[threadIdx.y][threadIdx.x] =
        (row < M && aCol < K) ? __bfloat162float(A[row * K + aCol]) : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (bRow < K && col < N) ? __bfloat162float(B[bRow * N + col]) : 0.0f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];  // fp32 MAC — CUDA cores, no MMA
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = __float2bfloat16(acc);  // round the fp32 accumulator once, at the end
  }
}

torch::Tensor sgemm_smem(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2-D");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
              "A and B must be bfloat16");
  TORCH_CHECK(A.size(1) == B.size(0), "inner dims must match: A[M,K] @ B[K,N]");

  A = A.contiguous();
  B = B.contiguous();
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = torch::empty({M, N}, A.options());

  const dim3 block(TILE, TILE);
  const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  sgemm_smem_kernel<<<grid, block>>>(reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
                                     reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
                                     reinterpret_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sgemm_smem kernel launch failed");
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgemm_smem", &sgemm_smem, "SMEM-tiled bf16 GEMM, fp32 accumulate (A3 R0 baseline)");
}
