// =============================================================================
// fp8_gemm_sm90.cu  —  Hopper FP8 (E4M3) GEMM  [STUB — the learning rep]
// =============================================================================
//
//   STUB. The kernel body is the learning rep — left for you to implement.
//   This file is the scaffold around it: the host launcher (so dispatch +
//   tests + benches can be wired NOW) and the structural spine (the right
//   headers, the right instruction family, the right reference list).
//
//   WHY FP8 (the load-bearing concept):
//     FP8 E4M3 has 4 exponent bits + 3 mantissa bits — 1.5 bytes of dynamic
//     range per byte of storage. On Hopper, mma.sync.aligned.m16n8k32.f32.e4m3
//     .e4m3.f32 does a 16x8x32 MMA in ONE issue (vs m16n8k16 for fp16) — so FP8
//     is 2x the FLOP throughput of fp16 at half the memory traffic. That's why
//     every frontier training run in 2026 uses FP8 for the forward/backward
//     GEMMs (Llama 3, DeepSeek V3, Mistral Large 2).
//
//   THE TWO LOAD-BEARING TECHNIQUES (what you'll implement):
//     1. The mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instruction
//        (PTX ISA §9.7.13.4.4.1). Operands are fp8 in registers (ldmatrix loads
//        them — same as the mma.sync fp16 path in kernels/gemm/cuda/mma_sync.cu).
//     2. The byte-permuted cp.async load (or ldmatrix.b32 x4) to feed fp8
//        pairs: each register holds 4 fp8 values packed as a u32, so the load
//        must pack two adjacent fp16 lanes into one register.
//     3. (Optional, the real win) per-tensor / per-block scaling factors —
//        amax of each tensor, dequant on the accumulator path. CUTLASS 3.2's
//        "scaled FP8" GEMM is the production pattern.
//
//   REFERENCES (study these first):
//     * NVIDIA CUDA C++ Programming Guide §7.25.2 (FP8 intrinsics)
//     * PTX ISA §9.7.13.4.4 (mma.sync fp8 shapes)
//     * CUTLASS 3.2: examples/55_hopper_fp8_gemm (the canonical reference impl)
//     * Tri Dao, flash-attention fp8 path (the attention-side application)
//     * DeepSeek-V3 technical report §3.3 (the production motivation — FP8
//       training on Hopper at 2x the throughput of bf16)
//
//   TARGET (PERF_PLAN, do NOT claim measured — you haven't implemented it):
//     H100 fp8 dense MMA peak ≈ 1979 TF/s (2x the fp16 989). A correct basic
//     FP8 GEMM should reach ~60-70% of that on 4096^3 — ~1200-1400 TF/s.
//
//   ISA GATE: mma.sync.aligned.*.e4m3 assembles only under -arch=sm_90a (the
//   'a' suffix). This file is built only for sm_90a (see CMakeLists.txt).
//
//   =====================================================================
//   LEARNING BOUNDARY (where you start writing):
//     The `fp8_gemm_kernel` __global__ below. Everything above it (headers,
//     defines, host launcher) is the scaffold. Implement the mainloop.
//   =====================================================================
// =============================================================================

#include <cuda_fp8.h>
#include <cuda.h>
#include <cstdint>

// Hopper FP8 MMA atom: m16n8k32 — 16 rows, 8 cols, K-step 32.
#define FP8_M 16
#define FP8_N 8
#define FP8_K 32

// Block tile (a reasonable starting shape — you'll tune this).
// 128x256 output, K streamed in BK=128 slabs (4 MMA-K steps per slab).
#define BM 128
#define BN 256
#define BK 128

// TODO(you): implement the FP8 GEMM mainloop. The structure mirrors the
// mma.sync fp16 GEMM in kernels/gemm/cuda/mma_sync.cu, but with:
//   - operands are __nv_fp8_e4m3 (4 per u32 register)
//   - the mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instruction
//   - cp.async or ldmatrix.b32 loads that pack fp8 pairs
//   - (optional) per-tensor scale factors loaded as kernel args
extern "C" __global__ void
__launch_bounds__(256)
fp8_gemm_kernel(const __nv_fp8_e4m3* __restrict__ A,   // [M, K] row-major
                const __nv_fp8_e4m3* __restrict__ B,   // [K, N] row-major
                float* __restrict__ C,                 // [M, N] fp32 accumulator out
                int M, int N, int K) {
    // TODO(you): the mainloop lives here. Until you write it, the host launcher
    // below raises a clear runtime_error so dispatch/tests/benches fail loudly
    // rather than silently launching an empty grid.
}

// =============================================================================
// HOST LAUNCHER (torch binding) — raises NotImplementedError until you fill in
// the kernel body above. Wired into pybind.cpp so dispatch can register it NOW.
// =============================================================================
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#include <stdexcept>

torch::Tensor fp8_gemm_sm90(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "fp8_gemm_sm90: A, B must be CUDA");
    TORCH_CHECK(A.scalar_type() == at::kFloat8_e4m3fn && B.scalar_type() == at::kFloat8_e4m3fn,
                "fp8_gemm_sm90: A, B must be torch.float8_e4m3fn");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0), "K mismatch");
    throw std::runtime_error(
        "fp8_gemm_sm90: STUB. Implement the mainloop in csrc/gemm/fp8_gemm_sm90.cu "
        "(see CUTLASS 3.2 examples/55_hopper_fp8_gemm). The host launcher is wired; "
        "fill in fp8_gemm_kernel and remove this throw.");
}
#endif  // TORCH_EXTENSION_NAME
