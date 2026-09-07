// =============================================================================
// stream_k_sm90.cu  —  Hopper stream-K GEMM  [STUB — the learning rep]
// =============================================================================
//
//   STUB. The kernel body is the learning rep.
//
//   WHY STREAM-K (the load-bearing concept):
//     A standard CTA-parallel GEMM assigns one output tile per CTA. When
//     M*N/tile isn't divisible by the SM count (e.g. a 4096^3 GEMM on a 132-SM
//     H100), the last wave is a fraction of the SMs — the "tail wave" idles
//     60-90% of the chip. Stream-K FIXES this: instead of one tile per CTA,
//     each CTA owns a *slice of the K reduction* and atomically accumulates
//     into a partial-output tile. The work is evenly partitioned across SMs;
//     no tail-wave idle. On H100 at 4096^3, stream-K is the "last 10%" — the
//     difference between 90% of cuBLAS and 97%.
//
//   THE THREE LOAD-BEARING TECHNIQUES (what you'll implement):
//     1. A scheduler that partitions the K dimension across SMs (one partition
//        per SM, not one per output tile). CUTLASS's StreamKScheduler is the
//        reference implementation.
//     2. Atomic accumulator tiles in GMEM (the partial sums from multiple CTAs
//        on the same output tile must be summed — and they're fp32, so
//        atomicAdd works but needs careful write-ordering).
//     3. A "coordinate the epilogue" step: only the LAST CTA to finish a tile
//        does the final write (it knows it's last because it atomically decrements
//        a per-tile counter to 0).
//
//   WHY THIS IS HARDER THAN IT LOOKS:
//     The K-partition must not cross the MMA-K boundary mid-tile (or you have
//     to flush + reload the accumulator), AND the atomic accumulation must be
//     deterministic (the same sum regardless of CTA finish order — which fp32
//     atomicAdd is NOT, so you need a fixed-point or fp32-accumulated epilogue).
//     This is why CUTLASS's StreamKScheduler is ~400 lines of C++.
//
//   REFERENCES (study these first):
//     * CUTLASS 3.x: include/cutlass/gemm/kernel/streamk.hpp
//       (the canonical scheduler + the wave-partition math)
//     * "Stream-K GEMM" (Hwu, Kirk, El Hajj 2022) — the original paper
//     * NVIDIA Hopper whitepaper §3.2 (SM count / wave math on H100)
//
//   TARGET (PERF_PLAN, do NOT claim measured):
//     Closes the tail-wave gap on the wgmma GEMM. A correct stream-K on 4096^3
//     H100 should reach ~95-97% of cuBLAS (vs ~90% for plain wave-parallel).
//
//   ISA GATE: this kernel uses wgmma (the Hopper warpgroup MMA), so -arch=sm_90a.
//
//   =====================================================================
//   LEARNING BOUNDARY: implement `stream_k_gemm_kernel` below + the host-side
//     scheduler (partition + atomic tiles + last-CTA epilogue).
//   =====================================================================
// =============================================================================

#include <cuda.h>
#include <cuda_fp16.h>   // __half — the kernel's operand type
#include <cstdint>

// Tile shape (match the wgmma GEMM so the K-partition math is consistent).
#define SK_BM 64
#define SK_BN 64
#define SK_BK 64

// TODO(you): the stream-K kernel. The signature below is a starting point —
// CUTLASS's StreamKScheduler would also pass a `WorkTileInfo` array.
extern "C" __global__ void
__launch_bounds__(128)
stream_k_gemm_kernel(const __half* __restrict__ A,   // [M, K]
                     const __half* __restrict__ B,   // [K, N]
                     float* __restrict__ C_partial,  // [num_tiles, M_tile, N_tile] atomic accum
                     int* __restrict__ tile_done,    // [num_tiles] per-tile "last CTA" counter
                     float* __restrict__ C,          // [M, N] final output (written by last CTA)
                     int M, int N, int K, int num_tiles, int sm_count) {
    // TODO(you): the stream-K mainloop. See CUTLASS streamk.hpp for the reference.
}

// =============================================================================
// HOST LAUNCHER — raises NotImplementedError until you fill in the scheduler +
// kernel body. Dispatch calls this with stream_k=True on matmul; see
// kernels/gemm/dispatch.py::matmul.
// =============================================================================
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#include <stdexcept>

torch::Tensor stream_k_gemm_sm90(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "stream_k_gemm_sm90: A, B must be CUDA");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf,
                "stream_k_gemm_sm90: A, B must be float16 (wgmma operands)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0), "K mismatch");
    throw std::runtime_error(
        "stream_k_gemm_sm90: STUB. Implement the K-partition scheduler + kernel in "
        "csrc/gemm/stream_k_sm90.cu (see CUTLASS streamk.hpp). The host launcher is wired; "
        "fill in stream_k_gemm_kernel + the atomic-accumulator setup, then remove this throw.");
}
#endif  // TORCH_EXTENSION_NAME
