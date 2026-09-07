// =============================================================================
// persistent_gemv_sm90.cu  —  Hopper persistent GEMV  [STUB — the learning rep]
// =============================================================================
//
//   STUB. The kernel body is the learning rep.
//
//   WHY PERSISTENT (the load-bearing concept):
//     A standard kernel launches one CTA per output tile; each CTA runs once
//     and retires. The launch overhead + the cold-start (no register state
//     carried across tiles) costs ~5-15% on memory-bound kernels. A PERSISTENT
//     kernel launches ONE CTA per SM that lives for the whole kernel and
//     processes MANY tiles in a loop — reusing registers, keeping SMEM warm,
//     and amortizing launch cost. This is the structure that underlies every
//     frontier kernel of 2024-2026: FlashAttention 3, LeanAttention, CUTLASS
//     persistent GEMM, the FlashInfer decode kernels.
//
//   WHY GEMV (the choice of stub):
//     GEMV (M=1) is the *decode matmul* — the hottest kernel in LLM serving.
//     It's memory-bound (one read of the weight matrix per output element), so
//     the persistent + SMEM-warm + TMA-prefetch pattern matters more here than
//     on any compute-bound GEMM. And it's the simplest non-trivial persistent
//     kernel (no K-reduction loop, no double-buffer math) — the right size for
//     a first learning rep before climbing to persistent FA3.
//
//   THE THREE LOAD-BEARING TECHNIQUES (what you'll implement):
//     1. A grid of exactly `sm_count` CTAs, each running a `for (tile < tiles)`
//        loop. The grid dim encodes the SM count (queried via the device prop
//        in the host launcher below).
//     2. A software-pipelined load loop: prefetch tile N+1 while computing
//        tile N. On Hopper this is TMA (`cp.async.bulk.tensor`) + mbarrier.
//     3. Register-resident accumulator state carried across tiles (for a
//        multi-output-tile persistent decode — not needed for M=1 but the
//        pattern for M>1 persistent GEMV).
//
//   WHY THE DEV BOX CAN'T RUN IT:
//     The persistent + TMA pattern is Hopper+ (sm_90a). The sm_120 dev box has
//     no TMA. This file compiles under -arch=sm_90a and runs on the rental H100.
//
//   REFERENCES (study these first):
//     * Mark Saroufim, "Life of a CUDA Stream" (the persistent GEMM tutorial)
//     * FlashAttention 3 paper §3.1 (the warp-specialized persistent structure)
//     * LeanAttention (the 2024 state-of-art persistent decode attention)
//     * CUTLASS 3.x include/cutlass/gemm/kernel/gemm_universal.h (the generic
//       persistent kernel + the `.workspace` epilogue coordination)
//
//   TARGET (PERF_PLAN, do NOT claim measured):
//     Match or beat the Triton gemv_split (the current decode path) on H100 at
//     M=1, N=4096, K=4096. The persistent + TMA path should reach ~95% of H100
//     HBM bandwidth (3.35 TB/s on H100 SXM5).
//
//   =====================================================================
//   LEARNING BOUNDARY: implement `persistent_gemv_kernel` below + the SM-count
//     grid setup in the host launcher.
//   =====================================================================
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>   // __half — the kernel's operand type
#include <cstdint>

// GEMV: x [K] @ A [K, N] = y [N]. One CTA per SM, each processes N_PER_SM cols.
extern "C" __global__ void
__launch_bounds__(128)
persistent_gemv_kernel(const __half* __restrict__ A,   // [K, N] col-major (K contiguous)
                      const __half* __restrict__ x,   // [K]
                      __half* __restrict__ y,         // [N]
                      int K, int N, int tiles_per_sm) {
    // TODO(you): the persistent mainloop. Structure:
    //   for (int t = 0; t < tiles_per_sm; ++t) {
    //     int col = blockIdx.x + t * gridDim.x;   // this CTA's output column(s)
    //     // TMA-prefetch A[:, col..col+TILE] while computing the previous tile.
    //     // dot product x . A[:, col] -> y[col].
    //   }
    // See the references above; the LeanAttention decode kernel is the closest.
}

// =============================================================================
// HOST LAUNCHER — raises NotImplementedError until you fill in the kernel body.
// =============================================================================
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#include <stdexcept>

torch::Tensor persistent_gemv_sm90(torch::Tensor x, torch::Tensor A) {
    TORCH_CHECK(x.is_cuda() && A.is_cuda(), "persistent_gemv_sm90: x, A must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf && A.scalar_type() == at::kHalf,
                "persistent_gemv_sm90: x, A must be float16");
    TORCH_CHECK(x.dim() == 1 && A.dim() == 2 && x.size(0) == A.size(0),
                "persistent_gemv_sm90: x[K] @ A[K,N]");
    throw std::runtime_error(
        "persistent_gemv_sm90: STUB. Implement the persistent mainloop in "
        "csrc/persistent/persistent_gemv_sm90.cu (see LeanAttention / FlashAttention 3 §3.1). "
        "The host launcher is wired; fill in persistent_gemv_kernel + the SM-count grid "
        "setup (cudaDeviceGetAttribute sharedMemPerBlockOptin), then remove this throw.");
}
#endif  // TORCH_EXTENSION_NAME
