// =============================================================================
// wgmma_gemm_sm90a.cu  —  Basic warpgroup-MMA GEMM mainloop (Hopper, sm_90a)
// =============================================================================
//
//   RENTAL ARTIFACT.  RUNTIME CORRECTNESS DEFERRED TO THE RENTAL DAY.
//   Compiles here (sm_120 standing box, CUDA 13.0) → runs on H100 (sm_90a).
//   This box cannot execute wgmma; the gate here is: (1) compiles cleanly with
//   the exact nvcc command below, (2) the emitted PTX contains wgmma.mma_async,
//   (3) a structural self-review against the reference (see bottom of file).
//
//   EXACT COMPILE COMMAND (must exit 0):
//     nvcc -arch=sm_90a -ptx \
//        performance/rental/kernels/wgmma_gemm_sm90a.cu -o /tmp/wgmma.ptx
//     grep -c 'wgmma.mma_async' /tmp/wgmma.ptx      # must be > 0
//   (Also compiles with -cubin for SASS inspection: cuobjdump -sass → HGMMA.)
//
//   REFERENCES (structural ground truth):
//     * performance/artifacts/wgmma_descriptor_manual.md
//         - §5  compiler-emitted SS-form inline-asm (the wgmma_ss helper here is
//               that exact string: 32 f32 accumulators, A-desc, B-desc, imms
//               "1,1,1,0,0"), and the verbatim fence/commit/wait protocol.
//         - §2  the 64-bit SMEM descriptor bit-field map + the >>4 (16-byte)
//               encoding matrix_descriptor_encode(x) = (x & 0x3FFFF) >> 4.
//         - §2.3 worked example: a 64-wide f16 tile, 128B swizzle →
//               start=addr>>4, LBO=16→1, SBO=1024→64, swizzle=1.
//     * Colfax Research, "WGMMA on Hopper" — make_smem_desc(), the same
//       encode(addr) | (encode(16)<<16) | (encode(1024)<<32) | (1ull<<62).
//     * A3_tensor_cores.md §3 Rung 3.1 / PERF_PLAN Phase 2 A2 §4.2.
//
//   PRE-REGISTERED TARGET (do NOT claim measured — cannot run here):
//     A2 §4.2 / A3 R3.1 "Basic WGMMA": book §7.3.1 ≈ 318 TFLOPS on H100 @4096³
//     FP16 (4.5× over WMMA's 71). This kernel is the *basic* first cut — a plain
//     synchronous SMEM stage feeding wgmma; no TMA, no multistage pipeline, no
//     warp-specialization (those are R3.3–R3.5 / §4.1, higher rungs).
//
//   WHAT THIS KERNEL IS:
//     One warpgroup (128 threads) computes one 64×64 output tile of C = A·B.
//     A: MxK row-major f16.   B: KxN row-major f16.   C: MxN f32.
//     Mainloop steps K by BK=64; each step plain-stages a 64×64 A tile and a
//     64×64 B tile (B transposed on load so both are K-contiguous / "K-major",
//     64 halves = 128 bytes wide → the 128B GMMA swizzle atom of the worked
//     example) into SMEM, then issues 4× wgmma.mma_async.m64n64k16 over the four
//     K=16 strips. FP32 accumulator lives in 32 registers/thread the whole loop;
//     it is zero-initialised once and every wgmma accumulates (scale-d=1), so
//     the fence sits once before the loop (artifact §1.10).
// =============================================================================

#include <cuda_fp16.h>
#include <cstdint>

#ifndef BM
#define BM 64          // output tile rows  (== wgmma M, fixed at 64)
#endif
#ifndef BN
#define BN 64          // output tile cols  (== wgmma N here)
#endif
#ifndef BK
#define BK 64          // K per mainloop step = 4 wgmma K-strips of 16
#endif

// -----------------------------------------------------------------------------
// 64-bit SMEM matrix descriptor  (artifact §2.1 / CUTLASS GmmaDescriptor)
//
//   [ 0,14) start_address  = byte_addr >> 4   (16-byte granular)
//   [16,30) leading_byte_offset (LBO) = LBO_bytes >> 4
//   [32,46) stride_byte_offset  (SBO) = SBO_bytes >> 4
//   [49,52) base_offset  (swizzle phase; 0 for atom-aligned base)
//   [62,64) layout_type : 0=none 1=128B 2=64B 3=32B
//
//   CUTLASS encode: matrix_descriptor_encode(x) = (x & 0x3FFFF) >> 4
//   (mask 18 bits → drop low 4 → 14-bit field). Enforces 16-byte alignment.
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFFull) >> 4;
}

// Build the descriptor for one 64-wide (128-byte) K-major f16 tile in SMEM,
// 128B swizzle — the exact geometry of the artifact §2.3 worked example
// (LBO=16 bytes, SBO=1024 bytes, swizzle=1). base_offset carries the intra-atom
// phase of the tile base (0 when the tile base is 1024-byte-atom aligned).
__device__ __forceinline__ uint64_t
make_smem_desc(const half* smem_ptr, int base_offset /*0..7*/) {
    // cvta.to.shared: generic → 32-bit CTA-window SMEM byte address.
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(addr));       // [0,14)
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(16))  << 16;  // LBO=16B
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(1024)) << 32; // SBO=1024B
    desc |= (static_cast<uint64_t>(base_offset) & 0x7ull)        << 49;  // phase
    desc |= 1ull << 62;                                                  // 128B swizzle
    return desc;
}

// -----------------------------------------------------------------------------
// CuTe Swizzle<3,4,3> == 128B GMMA swizzle atom (artifact §3.2).
//   XOR bits [7,10) of the byte offset into bits [4,7):
//     swz(off) = off ^ (((off >> 7) & 0x7) << 4)
// Applied on the SMEM *store* so the physical layout matches the swizzle=1 the
// descriptor above declares (the two must agree or the core walks garbage, §3.3).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t swz128(uint32_t byte_off) {
    return byte_off ^ (((byte_off >> 7) & 0x7u) << 4);
}

// -----------------------------------------------------------------------------
// The async warpgroup MMA — SS form (both operands SMEM descriptors).
// This asm string is copied verbatim from the compiler ground-truth in the
// descriptor artifact §5 (wgmma_ss): 32 "+f" f32 accumulators, "l"(a_desc),
// "l"(b_desc), trailing immediates  scale-d=1, scale-a=1, scale-b=1,
// trans-a=0, trans-b=0.  D = A·B + D  (accumulate).
// -----------------------------------------------------------------------------
__device__ __forceinline__ void
wgmma_m64n64k16_ss(float d[32], uint64_t a_desc, uint64_t b_desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(a_desc), "l"(b_desc));
}

// The three async-glue instructions, verbatim from artifact §1.10 / §5.
__device__ __forceinline__ void wgmma_fence()  {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// =============================================================================
// Kernel: C[MxN] = A[MxK] · B[KxN]   (f16 in, f32 accumulate)
//   grid  = (N/BN, M/BM)   block = 128 threads (one warpgroup)
// =============================================================================
extern "C" __global__ void
__launch_bounds__(128)
wgmma_gemm_sm90a(const half* __restrict__ A,   // MxK row-major
                 const half* __restrict__ B,   // KxN row-major
                 float* __restrict__       C,   // MxN row-major
                 int M, int N, int K) {
    const int tid    = threadIdx.x;             // 0..127 (one warpgroup)
    const int tileM  = blockIdx.y * BM;         // this CTA's output row origin
    const int tileN  = blockIdx.x * BN;         // this CTA's output col origin

    // Double SMEM staging buffers, 128B-swizzle-aligned.
    // As[BM][BK], Bs[BN][BK]  both K-major (K contiguous, 64 halves = 128 B).
    // 1024-byte (128B-atom) aligned so the tile-base swizzle phase is 0.
    __shared__ __align__(1024) half As[BM * BK];
    __shared__ __align__(1024) half Bs[BN * BK];

    // FP32 accumulator: 32 regs/thread = one 64×64 D tile across the warpgroup
    // (artifact §1.9). Zero-init once; every wgmma below accumulates (scale-d=1).
    float d[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) d[i] = 0.0f;

    // ----- mainloop over K in steps of BK=64 ----------------------------------
    for (int k0 = 0; k0 < K; k0 += BK) {

        // --- simple synchronous stage: GMEM -> SMEM (128B-swizzled store) -----
        // 128 threads move BM*BK = 4096 halves for A and 4096 for B.
        // Vectorized 8-half (float4 = 16 B, one swizzle granule) copies.
        #pragma unroll
        for (int i = tid * 8; i < BM * BK; i += 128 * 8) {
            int r = i / BK, c = i % BK;                 // (row, K-col) in tile
            // A[tileM+r][k0+c] contiguous in K -> vector load of 8 halves.
            const half* gptr = &A[(tileM + r) * K + (k0 + c)];
            uint32_t off = swz128(static_cast<uint32_t>(i) * 2u);   // byte offset
            *reinterpret_cast<float4*>(reinterpret_cast<char*>(As) + off) =
                *reinterpret_cast<const float4*>(gptr);
        }
        #pragma unroll
        for (int i = tid * 8; i < BN * BK; i += 128 * 8) {
            int r = i / BK, c = i % BK;                 // (N-row, K-col) in tile
            // B is KxN row-major; we want Bs[n][k] = B[k0+k][tileN+n] (K-major).
            // Gather 8 K-consecutive elements (strided by N in GMEM).
            half tmp[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                tmp[j] = B[(k0 + c + j) * N + (tileN + r)];
            uint32_t off = swz128(static_cast<uint32_t>(i) * 2u);
            *reinterpret_cast<float4*>(reinterpret_cast<char*>(Bs) + off) =
                *reinterpret_cast<const float4*>(tmp);
        }
        __syncthreads();

        // --- issue the warpgroup MMA over the 4 K=16 strips of this K-tile ----
        // fence orders the register zero-init (and prior tile's reads) before
        // the async wgmma reads (artifact §1.10 (A)).
        wgmma_fence();
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            // start_address advances by one K=16 strip = 16 halves = 32 bytes;
            // base_offset carries the 128B-swizzle phase of that strip
            // (32B/16B = 2 units per strip -> 0,2,4,6).  [STRUCTURAL — the exact
            //  strip-phase vs swizzle interaction is the #1 byte-diff target
            //  against cute::make_gmma_desc on the rental, artifact §4 step 4.]
            const half* a_strip = As + kk * 16;
            const half* b_strip = Bs + kk * 16;
            int         phase   = (kk * 2) & 0x7;
            uint64_t a_desc = make_smem_desc(a_strip, phase);
            uint64_t b_desc = make_smem_desc(b_strip, phase);
            wgmma_m64n64k16_ss(d, a_desc, b_desc);
        }
        // Seal the 4 wgmma into one group and drain fully before we overwrite
        // the SMEM staging buffers next iteration (artifact §1.10 (C),(D)).
        wgmma_commit();
        wgmma_wait<0>();
        __syncthreads();
    }

    // ----- epilogue: 32 accumulator regs/thread -> C[64×64] -------------------
    // Hopper wgmma.m64nNk16 accumulator fragment layout (PTX ISA §9.7.16.4.5 /
    // CUTLASS SM90 C layout):  warp w = tid/32 owns rows [16w, 16w+16); within a
    // warp the lane splits row = 16w + lane/4 (+8 for the second row group) and
    // col = (lane%4)*2 (+1). For N=64 there are 8 column octets; each contributes
    // 4 regs (2 rows × 2 cols).  [STRUCTURAL — index math follows the documented
    // layout; exact reg↔coord mapping is a rental byte-diff (cuobjdump / oracle).]
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row0 = tileM + warp * 16 + (lane / 4);
    const int col0 = tileN + (lane % 4) * 2;
    #pragma unroll
    for (int n = 0; n < BN / 8; ++n) {              // 8 column octets
        int col = col0 + n * 8;
        int r0  = row0;         // first row group
        int r1  = row0 + 8;     // second row group
        // reg quad for this octet: [r0c0, r0c1, r1c0, r1c1]
        float v_r0c0 = d[n * 4 + 0];
        float v_r0c1 = d[n * 4 + 1];
        float v_r1c0 = d[n * 4 + 2];
        float v_r1c1 = d[n * 4 + 3];
        if (r0 < M) {
            if (col     < N) C[r0 * N + col    ] = v_r0c0;
            if (col + 1 < N) C[r0 * N + col + 1] = v_r0c1;
        }
        if (r1 < M) {
            if (col     < N) C[r1 * N + col    ] = v_r1c0;
            if (col + 1 < N) C[r1 * N + col + 1] = v_r1c1;
        }
    }
}

// =============================================================================
// STRUCTURAL SELF-REVIEW  (against performance/artifacts/wgmma_descriptor_manual.md)
//   [MATCHES] wgmma_ss inline-asm string is byte-identical to artifact §5's
//             compiler-emitted SS form: 32 f32 accumulators (%f65..%f96 count),
//             one A-desc + one B-desc, five trailing imms "1,1,1,0,0".
//   [MATCHES] Descriptor fields: start_address = addr>>4 via
//             matrix_descriptor_encode ((x&0x3FFFF)>>4); LBO=16B→1 at <<16;
//             SBO=1024B→64 at <<32; layout_type=1 (128B) at bit 62 — the exact
//             §2.3 worked example / Colfax make_smem_desc constants.
//   [MATCHES] Fence protocol & ordering: wgmma.fence BEFORE the first wgmma
//             (after the register zero-init), then per K-tile
//             commit_group → wait_group 0 before reusing SMEM (§1.10 A,C,D).
//   [MATCHES] FP32 accumulator in registers: 32 f32/thread, zero-init once,
//             scale-d=1 accumulate throughout (§1.9, §1.10).
//   [MATCHES] sm_90a-gated: uses wgmma.* which only assembles under -arch=sm_90a
//             (the mandatory trailing 'a', artifact §4).
//   [DEFERRED to rental — runtime, NOT claimed correct here]:
//     * Byte-exact descriptor value per K-strip: the start-advance (+32B/strip)
//       vs 128B-swizzle base_offset phase — diff against cute::make_gmma_desc
//       (artifact §4 step 4) to promote from [STRUCTURAL] to [FACT].
//     * Epilogue reg↔(row,col) mapping — verify with an element-exact oracle /
//       cuobjdump on the H100 (a 32-lane bug shows as 3/4 stale output).
//     * The B-transpose-on-load gather and the swizzled SMEM store agreeing with
//       the descriptor's declared swizzle=1 (§3.3 "the two are one contract").
//     * Numerical correctness end-to-end — CANNOT run on sm_120; deferred.
// =============================================================================
