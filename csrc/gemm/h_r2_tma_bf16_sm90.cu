// =============================================================================
// h_r2_tma_bf16_sm90.cu — K1/H-R2: TMA + 128B swizzle + mbarrier expect-tx, 2 stages
// =============================================================================
//
// RUNG:   experiments/K1/H-R2/spec.md   ·   MAP: experiments/K1/H-R2/map.md
// FLOOR:  cuBLAS bf16, M=N=K=4096, H100 SXM.
// TARGET: 50-60% of cuBLAS (plan SIXTY_DAYS_SIX_LADDERS.md §05 K1).
//
// H-R1 with exactly one thing changed: how the operand tiles reach shared memory. Same 128x128x64
// tile, same warpgroup, same m64n128k16 wgmma, same fp32 accumulator, same epilogue mapping. The
// staging path is now the tensor memory accelerator — a host-built CUtensorMap, a single thread
// issuing cp.async.bulk.tensor, an mbarrier that counts BYTES rather than arrivals — and there are
// two buffers instead of one, so the copy for k+1 is in flight while the MMA for k runs. Anything
// else changing between the two rungs would make the delta uninterpretable, which is the whole
// reason the tile shape is frozen here rather than tuned.
//
// WHAT IS ALREADY HERE (agent-written; compiles, links, launches):
//   * the host-side CUtensorMap construction for A and B, with every cuTensorMapEncodeTiled field
//     spelled out and the driver entry point resolved without linking libcuda;
//   * the descriptor passed BY VALUE as `const __grid_constant__ CUtensorMap` (fast.cu does this
//     first at matmul_6.cuh:315; matmul_2-5 cudaMalloc a device copy and pass a pointer,
//     matmul_2.cuh:55-61, which costs a dependent load per launch for nothing);
//   * the two shared-memory descriptor layouts — K-major for A, MN-major for B — and why they
//     differ, which is the one real asymmetry this rung introduces;
//   * the mbarrier PTX wrappers (init / arrive-expect-tx / try_wait.parity), the TMA load wrapper,
//     the 2-stage shared-memory arithmetic, and the dynamic-smem opt-in at launch;
//   * the wgmma asm and the fence/commit/wait wrappers, cloned from H-R1 except for one immediate.
//
// WHAT IS HUY'S (the hole, below): the 2-stage k-loop and the epilogue. Four things live there and
// nowhere else — the expect-tx byte count that must equal what the three TMAs actually deliver, the
// phase parity the try_wait is given, which stage each wgmma reads, and when a buffer may be
// overwritten.
//
// BUILDING IT (no GPU required; infra/drydock.sh does this automatically):
//   nvcc -arch=sm_90a -cubin -O3 -Xptxas -v csrc/gemm/h_r2_tma_bf16_sm90.cu -o /tmp/o.cubin
// While the hole is open that fails at the #error, by design. Add -DHUY_STUB_KERNEL_BODY=1 to
// compile the SCAFFOLDING — descriptors, tensor maps, launcher, build wiring — around a trivially
// correct, catastrophically slow body. infra/bench.sh refuses to measure while that define (or
// LADDERS_STUB_HOLES=1) is in play: a number from the stub would be a number about the stub.
// =============================================================================

#include <cstdint>
#include <cuda.h>          // CUtensorMap, CUresult, the CU_TENSOR_MAP_* enums
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled_v12000
#include <cuda_bf16.h>

#ifndef BM
#define BM 128        // output tile rows per CTA        — frozen at H-R1's value
#endif
#ifndef BN
#define BN 128        // output tile cols per CTA == the wgmma N
#endif
#ifndef BK
#define BK 64         // K advanced per mainloop step = 4 wgmma k-strips of 16
#endif
#ifndef STAGES
#define STAGES 2      // the rung's variable: two staging buffers, one in flight behind the MMA
#endif

#define WGMMA_M 64                 // architectural: wgmma.m64nNk16 always has M = 64
#define WGMMA_K 16                 // architectural: k-strip width for 16-bit operands
#define M_STEPS  (BM / WGMMA_M)    // wgmma issues per k-strip to cover BM rows
#define K_STRIPS (BK / WGMMA_K)    // wgmma issues per mainloop step along K
#define ACC_REGS (BN / 2)          // fp32 accumulators per thread per wgmma: 128 lanes x this = 64x128

// The 128B swizzle atom, in bf16 elements. Every TMA box's INNERMOST extent must be exactly this:
// cuTensorMapEncodeTiled rejects a 128B-swizzled descriptor whose contiguous box extent is not
// 128 B, and it rejects it with CUDA_ERROR_INVALID_VALUE and nothing else — no field named, on the
// host, at startup. hopper_contracts.check_tma_tensor_map turns that into a named assertion in the
// CPU suite, and tests/kernels/gemm/test_k1_h_r2.py asserts it for the maps this file builds.
#define SWZ_ELEMS 64
#define B_CHUNKS  (BN / SWZ_ELEMS)                 // 2 — see the B-map comment below

#define A_TILE_BYTES   (BM * BK * 2)               // 16384
#define B_CHUNK_BYTES  (SWZ_ELEMS * BK * 2)        // 8192, one 64-wide N slice of the B tile
#define B_TILE_BYTES   (B_CHUNKS * B_CHUNK_BYTES)  // 16384
#define STAGE_BYTES    (A_TILE_BYTES + B_TILE_BYTES)      // 32768 — also the stage's expect-tx count
#define SMEM_DYNAMIC_BYTES (STAGE_BYTES * STAGES)         // 65536: past 48 KB, so the launch opts in
#define SMEM_BARRIER_BYTES (STAGES * 8)                   // static, outside the dynamic allocation

// wgmma exists ONLY in the sm_90a accelerated ISA. Not base sm_90 (the trailing 'a' is load
// bearing), and NOT sm_100a/sm_120a: ptxas rejects it there outright --
//   "Instruction 'wgmma.fence' not supported on .target 'sm_100a'"
// -- which is why the AOT build compiles each source for the archs its rung targets. This guard
// makes the file inert, not broken, elsewhere. The kernel SIGNATURE stays arch-independent so a
// multi-arch build links: CUtensorMap and __grid_constant__ are not sm_90a-only.
#define SCRATCH_LLM_HAS_TMA_WGMMA (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900))

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

// LBO and SBO are template parameters, not arguments, because THE TWO OPERANDS OF THIS RUNG DO NOT
// SHARE THEM. H-R1 could hard-code 16/1024 for both because it gathered B into a K-major tile by
// hand; TMA cannot transpose, so B arrives N-contiguous and is consumed as an MN-major operand with
// a different pair. Making them template parameters means A_DESC/B_DESC below are named types at
// the call site and cannot be swapped by accident — the failure mode being a kernel that runs, is
// fast, and is wrong. Values derived under "Shared-memory layout" below.
template <int LBO, int SBO>
__device__ __forceinline__ uint64_t
make_smem_desc(const __nv_bfloat16* smem_ptr, int base_offset /* 0..7 */ = 0) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = matrix_descriptor_encode(static_cast<uint64_t>(addr));
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(LBO)) << 16;
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(SBO)) << 32;
    desc |= (static_cast<uint64_t>(base_offset) & 0x7ull)        << 49;
    desc |= 1ull << 62;   // layout_type = 1 = 128B swizzle
    return desc;
}

// -----------------------------------------------------------------------------
// Shared-memory layout, and where LBO/SBO come from.
//
// TMA writes a box into shared memory in box-dimension order, innermost dimension fastest, then
// applies the 128B swizzle to the resulting byte offset. So the box shape IS the smem layout, and
// the descriptor must declare the layout the box produced. Both are derived below; CUTLASS's
// make_gmma_desc (oss/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:186-296) states the two
// canonical forms this has to land on, in units of uint128_t (16 B):
//
//   Major::K  B128 : ((8,m),(T,2)) : ((8T,SBO),(1,LBO))   -- K contiguous
//   Major::MN B128 : ((T,8,n),(8,k)) : ((1,T,LBO),(8T,SBO)) -- M|N contiguous
//
// A tile — box {BK=64, BM=128} over an (M,K) row-major matrix, K innermost.
//   smem offset(m,k) = k + 64*m elements. K contiguous, 128 B per row of 8 rows per 1024 B.
//   In uint128_t units the tile is (128,8):(8,1), which is the Major::K form with
//   LBO = 1 x 16 B = 16 and SBO = 64 x 16 B = 1024. Identical to H-R1 and to fast.cu's
//   make_smem_desc (matmul_2.cuh:9-17) — the A path did not change, only how it is filled.
//
// B tile — TWO boxes of {SWZ_ELEMS=64, BK=64} over a (K,N) row-major matrix, N innermost.
//   BN=128 bf16 is 256 B, twice the 128 B swizzle atom, so it CANNOT be one box dimension. The
//   split is what the atom rule forces; putting the outer half as a third, OUTERMOST rank of one
//   map would work too, and is what CUTLASS emits, but a rank-3 gmem_prob_shape of {64, K, N/64}
//   asserts N % 64 == 0 about the tensor and reads past the last row when it is false. Two rank-2
//   boxes at N-coordinates tileN and tileN+64 produce the byte-identical smem tile and let TMA's
//   own out-of-bounds zero-fill handle an N tail. Cost: three TMAs per stage instead of two.
//   smem offset(n,k) = (n%64) + 64*k + B_CHUNK_BYTES/2*(n/64) elements. In uint128_t units
//   ((8,2),(8,8)):((1,512),(8,64)) = the Major::MN form with LBO = 512 x 16 B = 8192 (the stride
//   between the two 64-wide halves) and SBO = 64 x 16 B = 1024 (the stride between k-groups of 8).
//   This is bit-identical to CUTLASS's tile_to_shape(Layout_MN_SW128_Atom<bf16>, (128,64)) with
//   K-tiles ordered fastest, which is the layout its row-major-B Hopper GEMMs actually use.
// -----------------------------------------------------------------------------
#define A_DESC_LBO 16
#define A_DESC_SBO 1024
#define B_DESC_LBO B_CHUNK_BYTES   // 8192 — stride between the two 64-wide N halves
#define B_DESC_SBO 1024

#if SCRATCH_LLM_HAS_TMA_WGMMA

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// --- mbarrier ---------------------------------------------------------------------------------
// An mbarrier is a 64-bit shared-memory object holding {pending arrivals, expected transaction
// bytes, phase}. TMA is what makes the second field matter: cp.async.bulk.tensor completes
// asynchronously and decrements the byte count as data lands, so "the tile is here" is a byte
// count reaching zero, not a thread count. Getting the count wrong does not fault — it hangs
// (too many expected) or lets the MMA read a half-written tile (too few).
//
// PTX: mbarrier.init / mbarrier.arrive.expect_tx / mbarrier.try_wait.parity, CUTLASS's
// oss/cutlass/include/cutlass/arch/barrier.h:399, :590-595, :422 respectively.

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t arrive_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(smem_u32(bar)), "r"(arrive_count) : "memory");
}

// Orders the generic-proxy writes that initialised the barriers against the async proxy that the
// TMA engine reads them through. Without it the TMA unit may observe an uninitialised barrier.
// fast.cu: cde::fence_proxy_async_shared_cta(), matmul_2.cuh:107.
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// One TMA tile load, global -> shared, completion tracked by `bar`'s transaction count.
// `c0` is the coordinate along the tensor's INNERMOST dimension (K for the A map, N for the B
// map), in elements. Out-of-bounds coordinates are not an error: the descriptor's
// FLOAT_OOB_FILL_NONE makes TMA zero-fill them, which is how the M and N tails are handled here
// without a single predicate in the load path.
__device__ __forceinline__ void
tma_load_2d(void* dst_smem, const CUtensorMap* tmap, uint64_t* bar, int c0, int c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_u32(dst_smem)), "l"(tmap), "r"(c0), "r"(c1), "r"(smem_u32(bar))
        : "memory");
}

// Arrive once AND declare how many bytes this phase is waiting for. Issued by the same single
// thread that issued the TMAs, after them.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}

// Spin until `bar` has completed the phase whose parity is `phase` (0 or 1). The parity bit, not a
// token: an mbarrier reused across k-iterations flips its phase on every completion, so the caller
// must know which flip it is waiting for. fast.cu never handles this — cuda::barrier hides it in an
// arrival token (matmul_2.cuh:4) — but CUTLASS's PipelineState carries it explicitly and flips it
// on index wrap (sm90_pipeline.hpp:204-213). At STAGES buffers, stage s is refilled every STAGES
// iterations, so the parity is a function of the iteration index and the stage count and nothing
// else. Written label-free (predicate -> register -> C++ loop) so that inlining it twice in one
// function cannot collide on a PTX label.
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar, uint32_t phase) {
    uint32_t done = 0;
    while (done == 0) {
        asm volatile("{\n"
                     ".reg .pred P;\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n"
                     "selp.b32 %0, 1, 0, P;\n"
                     "}\n"
                     : "=r"(done) : "r"(smem_u32(bar)), "r"(phase) : "memory");
    }
}

// --- wgmma ------------------------------------------------------------------------------------
// The async warpgroup MMA, SS form (both operands are shared-memory descriptors), bf16 in /
// fp32 accumulate. ScaleD is a template parameter because the PTX immediate must be a compile
// time constant: 0 overwrites the accumulator, 1 accumulates into it.
//
// The trailing immediates are scale-a, scale-b, trans-a, trans-b. H-R1's are `1, 1, 0, 0`; the
// LAST ONE DIFFERS HERE. trans = 0 declares a K-major operand, 1 an M|N-major one (CUTLASS
// GMMA::Major{K = 0, MN = 1}, passed straight through as the immediate:
// oss/cutlass/include/cute/arch/mma_sm90_gmma.hpp:107-110, :167). TMA cannot transpose, so B
// reaches shared memory N-contiguous and trans-b must be 1. Leaving it 0 — the natural result of
// cloning H-R1 literally — costs nothing at compile time and produces a plausible, wrong C.
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
        "%64, %65, %66, 1, 1, 0, 1;\n"
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
#endif  // SCRATCH_LLM_HAS_TMA_WGMMA

// =============================================================================
// Kernel: C[MxN] = A[MxK] . B[KxN],  bf16 operands, fp32 accumulate, fp32 out.
//   grid = (ceil(N/BN), ceil(M/BM))   block = 128 threads (one warpgroup)
//
// The tensor maps come in BY VALUE as __grid_constant__ parameters. That is not a style choice:
// cp.async.bulk.tensor requires the descriptor in parameter or global space, and __grid_constant__
// is what lets the address of a by-value parameter be taken without the compiler making a local
// copy. Passing a pointer to a cudaMalloc'd descriptor (fast.cu matmul_2.cuh:55-61, :176-177) also
// works and costs a dependent global load on every launch; matmul_6.cuh:315 is where upstream
// switched.
//
// A and B arrive as raw pointers as well, and the real path does not read them. They exist for the
// stub body below, which has to compute a correct C without going through a tensor map.
// =============================================================================
extern "C" __global__ void __launch_bounds__(128)
h_r2_tma_bf16_sm90(const __grid_constant__ CUtensorMap tmap_a,
                   const __grid_constant__ CUtensorMap tmap_b,
                   const __nv_bfloat16* __restrict__ A,   // M x K row-major (stub path only)
                   const __nv_bfloat16* __restrict__ B,   // K x N row-major (stub path only)
                   float* __restrict__ C,                 // M x N row-major
                   int M, int N, int K) {
#if !SCRATCH_LLM_HAS_TMA_WGMMA
    // Inert on every arch but sm_90a, so a multi-arch AOT build links. The host launcher checks
    // the device's compute capability and refuses before it can ever reach an empty kernel.
    (void)tmap_a; (void)tmap_b; (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#else
    const int tid   = threadIdx.x;              // 0..127, one warpgroup
    const int tileM = blockIdx.y * BM;          // this CTA's output row origin
    const int tileN = blockIdx.x * BN;          // this CTA's output col origin

    // STAGES x (A tile + B tile) = 64 KB, past the 48 KB static limit, so the buffers are dynamic
    // and the launcher opts in with cudaFuncSetAttribute. 1024-byte aligned because that is the
    // 128B-swizzle repeat: a tile base at a 1024-byte boundary has swizzle phase 0, which is what
    // lets every descriptor below carry base_offset = 0. fast.cu keeps the same invariant with
    // alignas(128) on a 128-byte-atom tile (matmul_4.cuh:263-267).
    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* const As[STAGES] = {
        reinterpret_cast<__nv_bfloat16*>(smem + 0 * STAGE_BYTES),
        reinterpret_cast<__nv_bfloat16*>(smem + 1 * STAGE_BYTES),
    };
    __nv_bfloat16* const Bs[STAGES] = {
        reinterpret_cast<__nv_bfloat16*>(smem + 0 * STAGE_BYTES + A_TILE_BYTES),
        reinterpret_cast<__nv_bfloat16*>(smem + 1 * STAGE_BYTES + A_TILE_BYTES),
    };

    // Static, not carved out of the dynamic block: an mbarrier is 8-byte aligned and 8 bytes long,
    // and keeping the 16 bytes out of the pipeline arithmetic keeps STAGE_BYTES exactly the number
    // the expect-tx count has to match. fast.cu does the same (matmul_4.cuh:280-282).
    __shared__ __align__(8) uint64_t full[STAGES];

    // Arrive count 1, NOT blockDim.x. Exactly one thread issues the TMAs and the expect-tx arrive;
    // the other 127 only wait on the parity, and waiting is not arriving. fast.cu inits with
    // blockDim.x (matmul_2.cuh:104-106) because there every thread calls barrier::arrive; CUTLASS
    // inits its full barriers with 1 and lets the leader arrive (sm90_pipeline.hpp:512-517). Mixing
    // the two conventions is a hang, and a hang on a rented box is an hour.
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) mbarrier_init(&full[s], 1);
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

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
    // from global memory. It exists so the scaffolding around the hole -- tensor maps, descriptors,
    // launcher, registration, the AOT build -- can be compiled and smoke-tested while the hole is
    // open. It touches neither the staging buffers, nor the barriers, nor wgmma, so it says nothing
    // about any of them. infra/bench.sh refuses to run while this is in play.
    for (int idx = tid; idx < BM * BN; idx += 128) {
        const int r = tileM + idx / BN, c = tileN + idx % BN;
        if (r >= M || c >= N) continue;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += __bfloat162float(A[r * K + k]) * __bfloat162float(B[k * N + c]);
        C[r * N + c] = acc;
    }
    (void)tmap_a; (void)tmap_b; (void)As; (void)Bs; (void)full; (void)d;
#else
    // HUY: the 2-stage TMA k-loop and the accumulator epilogue — spec: experiments/K1/H-R2/spec.md — fill before H-R2
    //
    // What has to happen, and the four things this rung exists to teach:
    //   1. PROLOGUE. For stage s in 0..STAGES-1, have thread 0 issue the three TMAs for k-block s
    //      -- tma_load_2d(As[s], &tmap_a, &full[s], s*BK, tileM) and, for h in 0..B_CHUNKS-1,
    //      tma_load_2d(Bs[s] + h*SWZ_ELEMS*BK, &tmap_b, &full[s], tileN + h*SWZ_ELEMS, s*BK) --
    //      then ONE mbarrier_arrive_expect_tx(&full[s], STAGE_BYTES). One arrive per stage, and its
    //      byte count must be what all three copies deliver together (fast.cu does the same
    //      per-stage accounting at matmul_4.cuh:306-308: expect-tx = (BK*BN + BK*BM)*sizeof(bf16)).
    //      Count the A tile twice, or forget a B half, and the barrier never releases.
    //   2. WAIT. Every thread calls mbarrier_wait_parity(&full[cur], phase). The phase is the
    //      subtlest line in the rung: stage `cur` is refilled on iterations cur, cur+STAGES,
    //      cur+2*STAGES..., its mbarrier flips phase on each completion, and the first completion
    //      is phase 0. So the parity is (iteration / STAGES) & 1 -- derive it, do not guess it, and
    //      note that CUTLASS keeps exactly this bit in PipelineState and flips it on index wrap
    //      (sm90_pipeline.hpp:204-213). A wrong parity is a hang on iteration STAGES, not on 0.
    //   3. MMA. wgmma_fence(), then for each of the K_STRIPS = 4 k-strips and each of the M_STEPS
    //      row groups, build the descriptors and issue wgmma_m64n128k16_bf16<1>(d[m], a_desc,
    //      b_desc):
    //          a_desc = make_smem_desc<A_DESC_LBO, A_DESC_SBO>(As[cur] + m*WGMMA_M*BK + s*WGMMA_K)
    //          b_desc = make_smem_desc<B_DESC_LBO, B_DESC_SBO>(Bs[cur] + s*WGMMA_K*SWZ_ELEMS)
    //      The two operands advance ASYMMETRICALLY along k, because their layouts differ: A's start
    //      address moves WGMMA_K bf16 = 32 B per strip (K is contiguous), B's moves
    //      WGMMA_K*SWZ_ELEMS bf16 = 2048 B per strip (a whole 16-row band of the N-contiguous
    //      tile). B's advance is a multiple of the 1024 B swizzle repeat so its phase is unchanged;
    //      A's is not, and reconciling that with base_offset = 0 is the line to byte-diff against
    //      cute::make_gmma_desc on the box -- fast.cu leaves base_offset at 0 and is verified
    //      against cuBLAS (matmul_2.cuh:9-17, matmul.cu:192), so 0 is the answer to reproduce, not
    //      to assume.
    //   4. HAND BACK THE BUFFER. wgmma_commit(); wgmma_wait<0>(); __syncthreads(); and only then
    //      may thread 0 issue the next TMAs into stage `cur` (for k-block iter + STAGES, if there
    //      is one). With one warpgroup producing and consuming, the wgmma wait IS the empty half of
    //      the handshake -- there is no empty[] barrier here because there is no second warpgroup
    //      to signal. H-R3 splits producer from consumer with setmaxnreg and needs one; keeping
    //      STAGES=2 without warp specialisation is what this rung is measuring, and if the profile
    //      says the TMA arrival is still exposed, that IS the result (map.md's open question).
    // Then the epilogue, unchanged from H-R1: map d[m][i] back to (row, col) in C using the m64nN
    // accumulator fragment layout, predicating on r < M and c < N. Getting it wrong is visible as
    // exactly 3/4 of the output being stale -- an oracle test catches it, a benchmark does not.
    //
    // The map (experiments/K1/H-R2/map.md) has the file:line for each of these upstream.
    #error "HUY: K1/H-R2 2-stage TMA mainloop + epilogue — see the comment above, experiments/K1/H-R2/spec.md, and map.md. Compile the scaffolding with -DHUY_STUB_KERNEL_BODY=1."
#endif  // HUY_STUB_KERNEL_BODY
#endif  // SCRATCH_LLM_HAS_TMA_WGMMA
}

// =============================================================================
// HOST — tensor map construction.
//
// Deliberately OUTSIDE the TORCH_EXTENSION_NAME guard: `nvcc -cubin` still type-checks host code,
// so the dry dock on the Mac compiles the cuTensorMapEncodeTiled call, its argument types and its
// enum spellings. Everything that needs torch stays below, where the dry dock cannot see it.
// =============================================================================

// cuTensorMapEncodeTiled lives in libcuda, which a torch extension does not link. Resolving it
// through the runtime's driver-entry-point table is how CUTLASS avoids that link edge, and it also
// makes a driver too old to have the symbol a clean runtime error instead of a load failure.
PFN_cuTensorMapEncodeTiled_v12000 h_r2_tensor_map_encoder() {
    static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
    if (fn == nullptr) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q;
        if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) ==
                cudaSuccess &&
            q == cudaDriverEntryPointSuccess) {
            fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
        }
    }
    return fn;
}

// One rank-2, 128B-swizzled, bf16 tiled tensor map. Every field of fast.cu's create_tensor_map
// (matmul_2.cuh:35-47), spelled out rather than templated, because at this rung the fields ARE the
// lesson:
//   inner_elems / outer_elems  the tensor's extents, INNERMOST FIRST. For A that is (K, M); for B,
//                              (N, K). This is the driver's order, not the matrix's.
//   outer_stride_bytes         the stride of dimension 1. Dimension 0's stride is not passed at
//                              all: the descriptor assumes the innermost stride is one element,
//                              which is why fast.cu builds a 5-entry array and hands over
//                              `gmem_prob_stride + 1` (matmul_2.cuh:38, :44) and CUTLASS does the
//                              same (copy_traits_sm90_tma.hpp:969, :1053). It must be a multiple
//                              of 16 B -- the constraint that makes an odd leading dimension
//                              un-TMA-able and that the wrapper rejects up front.
//   box_inner / box_outer      the tile TMA copies per instruction. box_inner * 2 B must be exactly
//                              128, the 128B swizzle atom.
// L2_PROMOTION_NONE keeps the citation literal; CUTLASS uses L2_128B here
// (copy_traits_sm90_tma.hpp:1040) and whether that matters is an ncu question for the box, not a
// guess for the Mac. FLOAT_OOB_FILL_NONE means out-of-bounds reads return zero, which is what
// handles the M and N tails.
CUresult h_r2_encode_tensor_map(CUtensorMap* map, void* gmem,
                                uint64_t inner_elems, uint64_t outer_elems,
                                uint64_t outer_stride_bytes,
                                uint32_t box_inner, uint32_t box_outer) {
    PFN_cuTensorMapEncodeTiled_v12000 encode = h_r2_tensor_map_encoder();
    if (encode == nullptr) return CUDA_ERROR_NOT_SUPPORTED;

    const uint64_t gmem_prob_shape[2]  = {inner_elems, outer_elems};
    const uint64_t gmem_prob_stride[1] = {outer_stride_bytes};
    const uint32_t smem_box_shape[2]   = {box_inner, box_outer};
    const uint32_t smem_box_stride[2]  = {1, 1};

    return encode(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, /*tensorRank=*/2, gmem,
                  gmem_prob_shape, gmem_prob_stride, smem_box_shape, smem_box_stride,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// =============================================================================
// HOST LAUNCHER — what pybind.cpp binds and the Python wrapper calls.
// =============================================================================
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>

torch::Tensor h_r2_tma_bf16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "h_r2_tma_bf16: A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
                "h_r2_tma_bf16: A and B must be bfloat16 (wgmma.f32.bf16.bf16 operands); "
                "the K1 floor is cuBLAS bf16, so a float16 input would be measured against the wrong floor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "h_r2_tma_bf16: A and B must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(0), "h_r2_tma_bf16: inner dimensions disagree: ",
                A.size(1), " vs ", B.size(0));
    A = A.contiguous();
    B = B.contiguous();
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(K % BK == 0, "h_r2_tma_bf16: K must be a multiple of ", BK,
                " at this rung (no K-remainder handling); got K=", K);
    // cuTensorMapEncodeTiled rejects a row stride that is not a multiple of 16 B. In bf16 that is
    // 8 elements, and it applies to A's K and B's N — the leading dimensions. This is a TMA
    // constraint, not a shortcut: it is why H-R1's npot shape (N=1023) cannot be measured here.
    TORCH_CHECK(K % 8 == 0 && N % 8 == 0,
                "h_r2_tma_bf16: TMA requires the row stride to be a multiple of 16 B, i.e. K and N "
                "multiples of 8 bf16 elements; got K=", K, " N=", N);
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(A.data_ptr()) % 16 == 0 &&
                reinterpret_cast<std::uintptr_t>(B.data_ptr()) % 16 == 0,
                "h_r2_tma_bf16: TMA requires a 16-byte-aligned global base address");

    CUtensorMap tmap_a{}, tmap_b{};
    // A is (M,K) row-major, so its innermost extent is K and its box is {BK, BM}.
    CUresult ra = h_r2_encode_tensor_map(&tmap_a, A.data_ptr(),
                                         static_cast<uint64_t>(K), static_cast<uint64_t>(M),
                                         static_cast<uint64_t>(K) * 2, BK, BM);
    // B is (K,N) row-major, so its innermost extent is N and its box is {SWZ_ELEMS, BK} — one of
    // the two 64-wide halves the mainloop loads per stage.
    CUresult rb = h_r2_encode_tensor_map(&tmap_b, B.data_ptr(),
                                         static_cast<uint64_t>(N), static_cast<uint64_t>(K),
                                         static_cast<uint64_t>(N) * 2, SWZ_ELEMS, BK);
    TORCH_CHECK(ra == CUDA_SUCCESS && rb == CUDA_SUCCESS,
                "h_r2_tma_bf16: cuTensorMapEncodeTiled failed (A=", static_cast<int>(ra),
                " B=", static_cast<int>(rb), "). The driver reports CUDA_ERROR_INVALID_VALUE (1) "
                "for every illegal field without naming one — run the CPU contract test "
                "tests/kernels/gemm/test_k1_h_r2.py, which checks each field by name.");

    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
    // 64 KB of staging is past the 48 KB a launch gets for free. ptxas will happily emit the
    // kernel without this call and the failure appears only as a launch error at runtime.
    const cudaError_t attr_err =
        cudaFuncSetAttribute(reinterpret_cast<const void*>(h_r2_tma_bf16_sm90),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DYNAMIC_BYTES);
    TORCH_CHECK(attr_err == cudaSuccess, "h_r2_tma_bf16: shared-memory opt-in failed: ",
                cudaGetErrorString(attr_err));
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(128);
    h_r2_tma_bf16_sm90<<<grid, block, SMEM_DYNAMIC_BYTES>>>(
        tmap_a, tmap_b,
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
        C.data_ptr<float>(), M, N, K);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "h_r2_tma_bf16 launch failed: ", cudaGetErrorString(err));
    return C;
}
#endif  // TORCH_EXTENSION_NAME
