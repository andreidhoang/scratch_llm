// =============================================================================
// h_r1_wgmma_bf16_sm90.cu — K1/H-R1: warpgroup MMA from shared memory, single stage
// =============================================================================
//
// RUNG:   experiments/K1/H-R1/spec.md   ·   MAP: experiments/K1/H-R1/map.md
// FLOOR:  cuBLAS bf16, M=N=K=4096, H100 SXM.
// TARGET: 25-40% of cuBLAS (plan SIXTY_DAYS_SIX_LADDERS.md §05 K1).
//
// The plainest warpgroup-MMA GEMM there is: one warpgroup (128 threads) owns one 128x128 output
// tile, stages A and B tiles into shared memory synchronously, and issues wgmma.mma_async against
// hand-built shared-memory descriptors. No TMA (H-R2), no multistage pipeline (H-R3), no warp
// specialisation (H-R3), no persistence (H-R4). Every one of those is a later rung precisely so
// that the gap THIS kernel leaves is legible in a profile: with a single stage and no async copy
// engine, the tensor pipe waits on global memory and the top stall reason should say so.
//
// WHAT IS ALREADY HERE (agent-written; compiles, links, launches):
//   * the shared-memory matrix descriptor and the 128B swizzle, mirroring the PTX ISA bit layout
//     exactly as scratch_llm.kernels.common.hopper_contracts models it in Python — the CPU
//     contract tests assert these two agree, so a packing bug is caught on a laptop;
//   * the bf16 m64n128k16 wgmma asm and the fence/commit/wait wrappers;
//   * tile/grid arithmetic, the shared-memory declaration, the accumulator declaration;
//   * the host launcher torch binds, with shape and dtype validation.
//
// WHAT IS HUY'S (the hole, below): the k-loop body and the accumulator epilogue. That is where
// this rung's four lessons actually live — which address and swizzle phase each k-strip's
// descriptor must carry, where the fence goes relative to the accumulator's zero-init, when the
// group may be committed and waited on before the staging buffer is reused, and how the m64nN
// accumulator fragment maps back to (row, col) in C.
//
// BUILDING IT (no GPU required; infra/drydock.sh does this automatically):
//   nvcc -arch=sm_90a -cubin -O3 -Xptxas -v csrc/gemm/h_r1_wgmma_bf16_sm90.cu -o /tmp/o.cubin
// While the hole is open that fails at the #error, by design. To verify the SCAFFOLDING compiles
// -- the descriptors, the launcher, the build wiring -- add -DHUY_STUB_KERNEL_BODY=1, which
// substitutes a trivially-correct, catastrophically slow reference loop. infra/bench.sh refuses to
// measure anything while that define (or LADDERS_STUB_HOLES=1) is in play: a number from the stub
// would be a number about the stub.
// =============================================================================

#include <cstdint>
#include <cuda_bf16.h>

#ifndef BM
#define BM 128        // output tile rows per CTA
#endif
#ifndef BN
#define BN 128        // output tile cols per CTA == the wgmma N
#endif
#ifndef BK
#define BK 64         // K advanced per mainloop step = 4 wgmma k-strips of 16
#endif
#define WGMMA_M 64                 // architectural: wgmma.m64nNk16 always has M = 64
#define WGMMA_K 16                 // architectural: k-strip width for 16-bit operands
#define M_STEPS  (BM / WGMMA_M)    // wgmma issues per k-strip to cover BM rows
#define ACC_REGS (BN / 2)          // fp32 accumulators per thread per wgmma: 128 lanes x this = 64x128

// wgmma exists ONLY in the sm_90a accelerated ISA. Not base sm_90 (the trailing 'a' is load
// bearing), and NOT sm_100a/sm_120a: ptxas rejects it there outright --
//   "Instruction 'wgmma.fence' not supported on .target 'sm_100a'"
// -- which is why the AOT build compiles each source for the archs its rung targets rather than
// for every arch in TORCH_CUDA_ARCH_LIST. This guard makes the file inert, not broken, elsewhere.
#define SCRATCH_LLM_HAS_WGMMA (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900))

// -----------------------------------------------------------------------------
// 64-bit shared-memory matrix descriptor. PTX ISA, "Matrix Descriptor Format":
//   [ 0,14) start address   [16,30) leading byte offset   [32,46) stride byte offset
//   [49,52) base offset (swizzle phase)                   [62,64) layout type
// Each offset field is (x & 0x3FFFF) >> 4: mask 18 bits, drop the low 4. Dropping those 4 bits is
// the hardware asserting 16-byte alignment, not a rounding convenience.
// Cross-checked against hopper_contracts.wgmma_smem_descriptor (tests/kernels/test_hopper_contracts.py).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFFull) >> 4;
}

// One 64-wide (128-byte) K-major bf16 tile in shared memory, 128B swizzle. LBO = 16 B walks
// between the two 8x8 core matrices along K; SBO = 1024 B walks between core-matrix rows.
// Swapping those two is the most common descriptor bug and it does not fault -- it reads the
// wrong core matrix and produces a plausible, wrong C.
__device__ __forceinline__ uint64_t
make_smem_desc(const __nv_bfloat16* smem_ptr, int base_offset /* 0..7 */) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = matrix_descriptor_encode(static_cast<uint64_t>(addr));
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(16))   << 16;
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(1024)) << 32;
    desc |= (static_cast<uint64_t>(base_offset) & 0x7ull)         << 49;
    desc |= 1ull << 62;   // layout_type = 1 = 128B swizzle
    return desc;
}

// CuTe Swizzle<3,4,3> == the 128B GMMA swizzle atom: XOR bits [7,10) into bits [4,7).
// The shared-memory STORE must apply this and the descriptor above must DECLARE it. They are one
// contract; honouring only one of them yields a kernel that runs, is fast, and is wrong.
__device__ __forceinline__ uint32_t swz128(uint32_t byte_off) {
    return byte_off ^ (((byte_off >> 7) & 0x7u) << 4);
}

#if SCRATCH_LLM_HAS_WGMMA
// The async warpgroup MMA, SS form (both operands are shared-memory descriptors), bf16 in /
// fp32 accumulate. ScaleD is a template parameter because the PTX immediate must be a compile
// time constant: 0 overwrites the accumulator, 1 accumulates into it.
template <int ScaleD>
__device__ __forceinline__ void
wgmma_m64n128k16_bf16(float d[ACC_REGS], uint64_t a_desc, uint64_t b_desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,"
        "%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,"
        "%56,%57,%58,%59,%60,%61,%62,%63}, "
        "%64, %65, %66, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(a_desc), "l"(b_desc), "n"(ScaleD));
}

__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait()   { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
#endif  // SCRATCH_LLM_HAS_WGMMA

// =============================================================================
// Kernel: C[MxN] = A[MxK] . B[KxN],  bf16 operands, fp32 accumulate, fp32 out.
//   grid = (ceil(N/BN), ceil(M/BM))   block = 128 threads (one warpgroup)
// =============================================================================
extern "C" __global__ void __launch_bounds__(128)
h_r1_wgmma_bf16_sm90(const __nv_bfloat16* __restrict__ A,   // M x K row-major
                     const __nv_bfloat16* __restrict__ B,   // K x N row-major
                     float* __restrict__ C,                 // M x N row-major
                     int M, int N, int K) {
#if !SCRATCH_LLM_HAS_WGMMA
    // Inert on every arch but sm_90a, so a multi-arch AOT build links. The host launcher checks
    // the device's compute capability and refuses before it can ever reach an empty kernel.
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#else
    const int tid   = threadIdx.x;              // 0..127, one warpgroup
    const int tileM = blockIdx.y * BM;          // this CTA's output row origin
    const int tileN = blockIdx.x * BN;          // this CTA's output col origin

    // Single-stage staging buffers, K-major (K contiguous: 64 bf16 = 128 B = one 128B swizzle
    // atom). 1024-byte aligned so the tile base's swizzle phase is 0.
    //   A: BM x BK x 2 B = 16384 B      B: BN x BK x 2 B = 16384 B      total 32 KB
    // Under the 48 KB opt-in threshold, so no cudaFuncSetAttribute is needed at this rung.
    // hopper_contracts.check_smem_budget asserts this in the CPU suite.
    __shared__ __align__(1024) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(1024) __nv_bfloat16 Bs[BN * BK];

    // fp32 accumulators, in registers, for the whole mainloop. M_STEPS wgmma of ACC_REGS each.
    // 2 x 64 = 128 registers/thread here -- the reason this rung's occupancy is one CTA per SM,
    // and a number worth holding next to the ptxas report drydock writes.
    float d[M_STEPS][ACC_REGS];
    #pragma unroll
    for (int s = 0; s < M_STEPS; ++s)
        #pragma unroll
        for (int i = 0; i < ACC_REGS; ++i) d[s][i] = 0.0f;

#if HUY_STUB_KERNEL_BODY
    // ---- STUB (only under -DHUY_STUB_KERNEL_BODY=1) -------------------------------------------
    // Trivially correct, catastrophically slow: one output element per thread, fp32, straight
    // from global memory. It exists so the scaffolding around the hole -- descriptors, launcher,
    // registration, the AOT build -- can be compiled and smoke-tested while the hole is open.
    // It touches neither the shared-memory tiles nor wgmma, so it says nothing about either.
    // infra/bench.sh refuses to run while this is in play.
    for (int idx = tid; idx < BM * BN; idx += 128) {
        const int r = tileM + idx / BN, c = tileN + idx % BN;
        if (r >= M || c >= N) continue;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += __bfloat162float(A[r * K + k]) * __bfloat162float(B[k * N + c]);
        C[r * N + c] = acc;
    }
    (void)As; (void)Bs; (void)d;
#else
    // HUY: the single-stage k-loop and the accumulator epilogue — spec: experiments/K1/H-R1/spec.md — fill before H-R1
    //
    // What has to happen, and the four things this rung exists to teach:
    //   1. Stage A[tileM..+BM][k0..+BK] and B[k0..+BK][tileN..+BN] into As/Bs, applying swz128()
    //      to the byte offset of every store. B is K-major in shared memory but N-major in global,
    //      so its load is a gather -- that transpose-on-load is the rung's other cost.
    //   2. __syncthreads(), then wgmma_fence() -- the fence orders the accumulator's zero-init
    //      (and the previous iteration's reads) before the async MMA reads them.
    //   3. For each of the BK/WGMMA_K = 4 k-strips, and each of the M_STEPS row groups, build the
    //      A and B descriptors with make_smem_desc(ptr, phase) and issue
    //      wgmma_m64n128k16_bf16<1>(d[s], a_desc, b_desc). The strip advances the start address by
    //      WGMMA_K bf16 = 32 B, and the 128B-swizzle phase advances with it -- reconciling those
    //      two is the single subtlest line in the rung, and the one to byte-diff against
    //      cute::make_gmma_desc on the box.
    //   4. wgmma_commit(); wgmma_wait<0>(); __syncthreads(); before overwriting the staging
    //      buffers next iteration. Waiting on 0 is what makes this rung single-stage, and what a
    //      later rung relaxes.
    // Then the epilogue: map d[s][i] back to (row, col) in C using the m64nN accumulator fragment
    // layout, predicating on r < M and c < N. Getting this wrong is visible as exactly 3/4 of the
    // output being stale -- an oracle test catches it, a benchmark does not.
    //
    // The map (experiments/K1/H-R1/map.md) has the file:line for each of these upstream.
    #error "HUY: K1/H-R1 mainloop + epilogue — see the comment above, experiments/K1/H-R1/spec.md, and map.md. Compile the scaffolding with -DHUY_STUB_KERNEL_BODY=1."
#endif  // HUY_STUB_KERNEL_BODY
#endif  // SCRATCH_LLM_HAS_WGMMA
}

// =============================================================================
// HOST LAUNCHER — what pybind.cpp binds and the Python wrapper calls.
// =============================================================================
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>

torch::Tensor h_r1_wgmma_bf16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "h_r1_wgmma_bf16: A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
                "h_r1_wgmma_bf16: A and B must be bfloat16 (wgmma.f32.bf16.bf16 operands); "
                "the K1 floor is cuBLAS bf16, so a float16 input would be measured against the wrong floor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "h_r1_wgmma_bf16: A and B must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(0), "h_r1_wgmma_bf16: inner dimensions disagree: ",
                A.size(1), " vs ", B.size(0));
    A = A.contiguous();
    B = B.contiguous();
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(K % BK == 0, "h_r1_wgmma_bf16: K must be a multiple of ", BK,
                " at this rung (no K-remainder handling); got K=", K);

    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(128);
    h_r1_wgmma_bf16_sm90<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
        C.data_ptr<float>(), M, N, K);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "h_r1_wgmma_bf16 launch failed: ", cudaGetErrorString(err));
    return C;
}
#endif  // TORCH_EXTENSION_NAME
