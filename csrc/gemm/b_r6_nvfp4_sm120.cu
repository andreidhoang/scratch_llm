// =============================================================================================
// b_r6_nvfp4_sm120.cu — K1/B-R6: NVFP4 block-scaled GEMM, mma.sync path (Blackwell client)
// =============================================================================================
//
// RUNG:   experiments/K1/B-R6/spec.md   ·   MAP: experiments/K1/B-R6/map.md
// FLOOR:  cuBLASLt at matched shape; the rung's own metric is % of NVFP4 speed-of-light.
// TARGET: <= 2x SoL (plan SIXTY_DAYS_SIX_LADDERS.md §05 K1, B-R6 row).
//
// One rung, two silicon paths, one layout. On B200 the same NVFP4 GEMM is issued as
// `tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X` with the scale factors living in TMEM and
// arriving there by UTCCP (map: sm100_blockscaled_mma_warpspecialized.hpp:839, :1018-1019); that
// path is reached through B-R5's CuTe DSL module. On sm_120a — 5090 / RTX PRO Blackwell — there is
// no TMEM and no UTCCP, and the identical arithmetic is issued per warp as
// `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` with the scales arriving in
// ONE ordinary 32-bit register per operand, loaded by a plain LDS
// (map: sm120_blockscaled_mma_builder.inl:179, cute/arch/mma_sm120.hpp:3194-3216).
// This file is the sm120 one. It is the compilable artifact of the rung: it assembles on this Mac.
//
// THE SCALE-FACTOR LAYOUT IS THE RUNG. It is identical on both architectures in global memory
// (sm120 reuses `Sm1xxBlockScaledConfig`, sm120_blockscaled_mma_builder.inl:164), and it is not a
// row-major side tensor: it is a 512-byte indivisible block, 128 rows x 4 K-blocks, permuted so
// that four consecutive bytes are four K-blocks OF ONE ROW. Getting the permutation wrong does not
// fault and does not slow anything down — it multiplies each block by another block's scale. The
// arithmetic lives, once, in scratch_llm.kernels.gemm.nvfp4_layout; sf_atom_offset below is its
// device-side twin and the CPU suite evaluates this file's expression against that module.
//
// WHAT IS ALREADY HERE (agent-written; compiles for sm_120a today, links, launches):
//   * the scale-factor byte-offset arithmetic, mirrored from nvfp4_layout.py and drift-tested;
//   * the block-scaled mma.sync asm wrapper, with the operand shapes the PTX fixes;
//   * an always-compiled ISA probe kernel, so the dry dock proves TODAY that this instruction
//     assembles for sm_120a on CUDA 12.8 (SASS: OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X);
//   * tile / warp / grid arithmetic, the shared-memory declarations, the accumulator declaration;
//   * the E2M1 and UE4M3 decoders (used by the stub, and by any debug epilogue);
//   * the host launcher torch binds, with shape, dtype and scale-tensor-size validation.
//
// WHAT IS HUY'S (the hole, below): the mainloop body and the epilogue — the staging of A/B/SFA/SFB
// into shared memory, the LDS into the register fragments the mma fixes, the mma issue, and the
// mapping of the m16n8 accumulator back to (row, col) in C.
//
// BUILDING IT (no GPU required; infra/drydock.sh does this):
//   nvcc -arch=sm_120a -cubin -O3 -Xptxas -v csrc/gemm/b_r6_nvfp4_sm120.cu -o /tmp/o.cubin
// While the hole is open that fails at the #error, by design. Add -DHUY_STUB_KERNEL_BODY=1 to
// compile the SCAFFOLDING around the hole: it substitutes a trivially-correct, catastrophically
// slow reference body. infra/bench.sh refuses to measure while that define (or LADDERS_STUB_HOLES)
// is in play — a number from the stub would be a number about the stub.
// =============================================================================================

#include <cstdint>

// ---------------------------------------------------------------------------------------------
// Tile shape. Mirrored by TILE_M/TILE_N/TILE_K in kernels/gemm/cuda/b_r6.py; the CPU suite asserts
// the two agree, because they are used for different things (one generates code, the other feeds
// the shared-memory and wave arithmetic) and nothing else would notice a drift.
// ---------------------------------------------------------------------------------------------
#ifndef BM
#define BM 128        // output tile rows per CTA
#endif
#ifndef BN
#define BN 128        // output tile cols per CTA
#endif
#ifndef BK
#define BK 64         // K per mainloop step. NOT free: 64 = SF_BLK_SF * SF_VEC_SIZE is exactly one
#endif                // scale atom's K extent AND exactly the mma's k, so one step consumes one
                      // 512-byte SF block per operand and needs no partial-atom addressing.

// The block-scaled mma.sync is m16n8k64 (cute/arch/mma_sm120.hpp:3216). Architectural, not a knob.
#define MMA_M 16
#define MMA_N 8
#define MMA_K 64

// 4x2 warps == upstream's cooperative AtomLayoutMNK Layout<Shape<_4,_2,_1>>
// (sm120_blockscaled_mma_builder.inl:130-133) for an N-tile of at least 16. Each warp owns a
// 32x64 slice, i.e. 2 x 8 = 16 mma per K-step, so 64 fp32 accumulators per thread — the number to
// hold next to ptxas's register report, since 64 acc + operand + address registers is what decides
// whether two CTAs are resident per SM.
#define WARPS_M 4
#define WARPS_N 2
#define WARPS_PER_CTA (WARPS_M * WARPS_N)
#define THREADS (WARPS_PER_CTA * 32)
#define WARP_TILE_M (BM / WARPS_M)
#define WARP_TILE_N (BN / WARPS_N)
#define MMA_STEPS_M (WARP_TILE_M / MMA_M)
#define MMA_STEPS_N (WARP_TILE_N / MMA_N)

// ---------------------------------------------------------------------------------------------
// The scale-factor layout, as constants. cutlass/detail/sm100_blockscaled_layout.hpp:51-55 and
// gemm/collective/builders/sm1xx_common.inl:470-475. Kept as macros so the CPU suite can read them
// out of this file and compare against nvfp4_layout.py.
// ---------------------------------------------------------------------------------------------
#define SF_VEC_SIZE 16      // K-elements sharing one ue4m3 scale (dense nv_float4_t)
#define SF_BLK_MN 128       // Blk_MN: rows (SFA) or cols (SFB) in one indivisible block
#define SF_BLK_SF 4         // Blk_SF: scale factors along K in one block
#define SF_ATOM_BYTES (SF_BLK_MN * SF_BLK_SF)   // 512
#define SF_ATOM_K (SF_BLK_SF * SF_VEC_SIZE)     // 64

// The block-scaled mma.sync exists ONLY in the sm_120a accelerated ISA (CUTE_ARCH_MXF4NVF4_4X_
// UE4M3_MMA_ENABLED, cute/arch/config.hpp:165-173, and CUDA >= 12.8). Not base sm_120, and NOT
// sm_100a — there the same arithmetic is tcgen05, a different instruction with a different operand
// model. This guard makes the file inert, not broken, on every other arch, so a multi-arch AOT
// build links instead of failing.
#define SCRATCH_LLM_HAS_NVFP4_MMA (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200))

// ---------------------------------------------------------------------------------------------
// Scale-factor addressing. The device twin of nvfp4_layout.atom_offset / sf_byte_offset; the CPU
// test extracts these two return expressions verbatim and evaluates them against that module, so a
// change here that is not made there fails on a laptop rather than on a rented Blackwell.
// ---------------------------------------------------------------------------------------------

// Byte offset inside one 512-byte atom, from the CuTe atom's strides
// (sm100_blockscaled_layout.hpp:54-55): row r splits as r0 + 32*r1 contributing 16*r0 + 4*r1, and
// the K-block index contributes 1 each. Dense onto [0, 512).
__device__ __forceinline__ int sf_atom_offset(int row_in_atom, int sf_block) {
    return 16 * (row_in_atom & 31) + 4 * (row_in_atom >> 5) + sf_block;
}

// Byte offset of the scale for (row, k) in a full operand's scale tensor. The atom grid is K-major
// (tile_to_shape(..., Step<_2,_1>), sm100_blockscaled_layout.hpp:93 and cute/layout.hpp:1808-1811),
// so k_atoms is the M-atom stride. `k_atoms` is passed rather than derived because the launcher
// already computed it and a second ceil-division per element is pure loop overhead.
__device__ __forceinline__ int sf_byte_offset(int row, int k, int k_atoms) {
    return SF_ATOM_BYTES * ((row >> 7) * k_atoms + (k >> 6)) + sf_atom_offset(row & 127, (k >> 4) & 3);
}

// ---------------------------------------------------------------------------------------------
// The two encodings, decoded arithmetically (no lookup table: a dynamically indexed local array
// lands in local memory, which the dry dock reports as a stack frame and which costs an LDL in the
// inner loop). Derived from exmy_base.h:937-939 (E2M1) and :917-919 (UE4M3).
// ---------------------------------------------------------------------------------------------

// E2M1: sign, 2 exponent, 1 mantissa, bias 1, no NaN and no Inf. Grid {0,.5,1,1.5,2,3,4,6}.
__device__ __forceinline__ float e2m1_to_float(unsigned int code) {
    const unsigned int e = (code >> 1) & 0x3u, m = code & 0x1u;
    const float mag = (e == 0u) ? (0.5f * (float)m) : ldexpf((float)(2u + m), (int)e - 2);
    return (code & 0x8u) ? -mag : mag;
}

// UE4M3: unsigned, 4 exponent, 3 mantissa, bias 7. 0x7F is the canonical NaN (NAN_MASK, not 0xFF —
// the sign bit is not part of the encoding), and it is returned as a NaN rather than as 480 so a
// poisoned scale propagates instead of quietly becoming a large finite number.
__device__ __forceinline__ float ue4m3_to_float(unsigned int code) {
    const unsigned int e = (code >> 3) & 0xFu, m = code & 0x7u;
    if (code == 0x7Fu) return nanf("");
    return (e == 0u) ? ldexpf((float)m, -9) : ldexpf((float)(8u + m), (int)e - 10);
}

#if SCRATCH_LLM_HAS_NVFP4_MMA
// ---------------------------------------------------------------------------------------------
// The block-scaled warp MMA, register-register form: D = A*B*scales + C over a 16x8x64 tile.
// cute/arch/mma_sm120.hpp:3216. Operand widths are fixed by the instruction and are worth reading
// as a fragment description: 16x64 e2m1 over 32 lanes = 32 nibbles = 4 x b32 for A; 8x64 = 16
// nibbles = 2 x b32 for B; 16x8 fp32 over 32 lanes = 4 floats for D/C; and ONE b32 of scales per
// operand — four ue4m3 bytes, which is exactly why the layout atom packs four K-blocks of one row
// into four consecutive bytes.
//
// bid/tid select which byte and which thread-group of the scale operand this mma consumes. CUTLASS
// hardcodes all four to 0 (mma_sm120.hpp:3208-3211) and so does this wrapper: with scale_vec::4X
// the single b32 already covers the instruction's whole k=64, so there is nothing to select.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void
mma_m16n8k64_nvfp4(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}
#endif  // SCRATCH_LLM_HAS_NVFP4_MMA

// =============================================================================================
// ISA probe — not bound, not launched, never measured. One block-scaled mma, always compiled.
//
// It exists so that the dry dock proves on a laptop, while the hole is still open, that this
// instruction assembles for sm_120a with this toolkit: that the operand constraint letters are
// right, that -arch=sm_120a (not sm_120) selects it, and that CUDA 12.8 is new enough
// (cute/arch/config.hpp:168-172 gates the whole family on it). Discovering any of those on a
// rented Blackwell instead costs an hour and produces no number.
// =============================================================================================
extern "C" __global__ void b_r6_nvfp4_isa_probe(const uint32_t* __restrict__ src,
                                                float* __restrict__ dst) {
#if !SCRATCH_LLM_HAS_NVFP4_MMA
    (void)src; (void)dst;
#else
    uint32_t a[4] = {src[0], src[1], src[2], src[3]};
    uint32_t b[2] = {src[4], src[5]};
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_m16n8k64_nvfp4(d, a, b, src[6], src[7]);
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst[threadIdx.x * 4 + i] = d[i];
#endif
}

// =============================================================================================
// Kernel: C[MxN] = dequant(A)[MxK] . dequant(B)[KxN], NVFP4 operands, fp32 accumulate, fp32 out.
//
//   A_packed  M x K/2 uint8, row-major. Two e2m1 per byte, LOW nibble = the LOWER k.
//   B_packed  N x K/2 uint8, row-major — i.e. the COLUMN-major [K,N] operand the `.row.col` mma
//             wants, which is the same bytes. Upstream says it the other way round: A RowMajor,
//             B ColumnMajor (79a_blackwell_geforce_nvfp4_bf16_gemm.cu:97,102).
//   SFA, SFB  flat uint8, the swizzled scale tensors of A (M x K) and B (N x K).
//   grid = (ceil(N/BN), ceil(M/BM))   block = THREADS
// =============================================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
b_r6_nvfp4_sm120(const uint8_t* __restrict__ A_packed,   // M x K/2
                 const uint8_t* __restrict__ SFA,        // swizzled scales of A
                 const uint8_t* __restrict__ B_packed,   // N x K/2
                 const uint8_t* __restrict__ SFB,        // swizzled scales of B
                 float* __restrict__ C,                  // M x N row-major
                 int M, int N, int K,
                 int sfa_k_atoms, int sfb_k_atoms) {
#if !SCRATCH_LLM_HAS_NVFP4_MMA
    // Inert on every arch but sm_120a, so a multi-arch AOT build links. The host launcher checks
    // the device's compute capability and refuses before it can ever reach an empty kernel.
    (void)A_packed; (void)SFA; (void)B_packed; (void)SFB; (void)C;
    (void)M; (void)N; (void)K; (void)sfa_k_atoms; (void)sfb_k_atoms;
#else
    const int tid   = threadIdx.x;              // 0..THREADS-1
    const int tileM = blockIdx.y * BM;          // this CTA's output row origin
    const int tileN = blockIdx.x * BN;          // this CTA's output col origin

    // Single-stage staging, K-contiguous. The fp4 tiles are BM x BK nibbles = BM * BK/2 bytes; the
    // scale tiles are exactly ONE 512-byte gmem atom each, because BM = BN = SF_BLK_MN and
    // BK = SF_ATOM_K. That equality is the reason the SF stage is a flat contiguous copy with no
    // swizzle to reconstruct: sm120 composes no swizzle onto the SF shared-memory layout
    // (sm120_blockscaled_mma_builder.inl:189-214 builds it from bare shapes and strides), unlike
    // the A/B tiles, whose layout atom comes from sm120_rr_smem_selector (:164-165).
    //   A 4096 B + B 4096 B + SFA 512 B + SFB 512 B = 9216 B — far under sm_120a's 99 KB per-CTA
    // cap, which is the headroom a later multistage version spends. hopper_contracts.
    // check_smem_budget asserts it in the CPU suite.
    __shared__ __align__(128) uint8_t As[BM * BK / 2];
    __shared__ __align__(128) uint8_t Bs[BN * BK / 2];
    __shared__ __align__(16)  uint8_t SFAs[SF_ATOM_BYTES];
    __shared__ __align__(16)  uint8_t SFBs[SF_ATOM_BYTES];

    // fp32 accumulators, in registers, for the whole mainloop: MMA_STEPS_M x MMA_STEPS_N tiles of
    // 4 each = 64 per thread. The MMA is the only thing that ever writes them, and it accumulates
    // in fp32 regardless of the operands' width (mma_sm120.hpp:3216 names .f32) — so the
    // accumulation error here is fp32's, and every bit of the format error is the operands'. That
    // separation is what makes this rung's tolerance derivable at all.
    float d[MMA_STEPS_M][MMA_STEPS_N][4];
    #pragma unroll
    for (int i = 0; i < MMA_STEPS_M; ++i)
        #pragma unroll
        for (int j = 0; j < MMA_STEPS_N; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) d[i][j][e] = 0.0f;

#if HUY_STUB_KERNEL_BODY
    // ---- STUB (only under -DHUY_STUB_KERNEL_BODY=1) -------------------------------------------
    // Trivially correct, catastrophically slow: one output element per thread, every operand
    // unpacked and dequantized in fp32 straight from global memory, no tensor core and no shared
    // memory. It exists so the scaffolding around the hole — the scale addressing, the launcher,
    // the registration, the AOT build — can be compiled and smoke-tested while the hole is open,
    // and so LADDERS_STUB_HOLES=1 can exercise the Python path end to end on a device.
    // It says nothing about the mainloop. infra/bench.sh refuses to run while it is in play.
    for (int idx = tid; idx < BM * BN; idx += THREADS) {
        const int r = tileM + idx / BN, c = tileN + idx % BN;
        if (r >= M || c >= N) continue;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const uint8_t pa = A_packed[r * (K / 2) + k / 2];
            const uint8_t pb = B_packed[c * (K / 2) + k / 2];
            const unsigned int na = (k & 1) ? (pa >> 4) : (pa & 0xF);
            const unsigned int nb = (k & 1) ? (pb >> 4) : (pb & 0xF);
            const float sa = ue4m3_to_float(SFA[sf_byte_offset(r, k, sfa_k_atoms)]);
            const float sb = ue4m3_to_float(SFB[sf_byte_offset(c, k, sfb_k_atoms)]);
            // Scale the OPERAND, not the product — the same order as the upstream reference
            // (tools/util/.../reference/host/gett.hpp:555-560 multiplies by SfA in fp32 before the
            // product). With fp32 accumulation the two orders agree, but only one of them is what
            // the hardware does, and the oracle has to match the hardware.
            acc += (e2m1_to_float(na) * sa) * (e2m1_to_float(nb) * sb);
        }
        C[r * N + c] = acc;
    }
    (void)As; (void)Bs; (void)SFAs; (void)SFBs; (void)d;
#else
    // HUY: the NVFP4 block-scaled mma.sync mainloop and the accumulator epilogue — spec: experiments/K1/B-R6/spec.md — fill before B-R6
    //
    // What has to happen, and the four things this rung exists to teach:
    //   1. Stage one K-step: A[tileM..+BM][k0..+BK] and B[tileN..+BN][k0..+BK] as BM*BK/2 and
    //      BN*BK/2 bytes into As/Bs, and the two 512-byte scale atoms into SFAs/SFBs. The scale
    //      copy is `SFA + SF_ATOM_BYTES * ((tileM>>7) * sfa_k_atoms + (k0>>6))`, one contiguous
    //      512-byte block, because BM and BK were chosen to make it one — a tile that broke that
    //      equality would need sf_byte_offset per element instead.
    //   2. __syncthreads(), then LDS into the register fragments. This is the subtle half of the
    //      rung: the mma fixes A as 4 x b32 (32 nibbles per lane over a 16x64 tile), B as 2 x b32,
    //      and the scales as ONE b32 per operand — four ue4m3 bytes which, by the atom's design,
    //      are four consecutive bytes of one row (sm100_blockscaled_layout.hpp:134-135). Which
    //      lane holds which of them is the PTX's `.m16n8k64` fragment layout with bid/tid = 0
    //      (cute/arch/mma_sm120.hpp:3194-3216); read it there and byte-diff a single tile against
    //      a CPU dequantization before trusting a 4096^3 number.
    //   3. For each of MMA_STEPS_M x MMA_STEPS_N warp tiles, issue
    //      mma_m16n8k64_nvfp4(d[i][j], a_frag[i], b_frag[j], sfa_frag[i], sfb_frag[j]).
    //      One K-step is exactly one mma k, so there is no inner k-strip loop here — the loop that
    //      a later pipeline version adds is over STAGES, not over k within a step.
    //   4. __syncthreads() before overwriting the staging buffers on the next K-step. Single stage
    //      is what makes this rung's first number a latency number; the SF traffic is 512 B per
    //      128 rows per K-step against 4 KB of operand, and whether it hides is the question the
    //      map could not answer.
    // Then the epilogue: map d[i][j][e] back to (row, col) in C through the m16n8 accumulator
    // fragment layout, predicating on r < M and c < N. A wrong fragment mapping is visible as a
    // structured subset of the output being wrong — the oracle test catches it, a benchmark does
    // not.
    //
    // The map (experiments/K1/B-R6/map.md) has the file:line for each of these upstream, and
    // scratch_llm.kernels.gemm.nvfp4_layout has the scale addressing as tested Python.
    #error "HUY: K1/B-R6 mainloop + epilogue — see the comment above, experiments/K1/B-R6/spec.md, and map.md. Compile the scaffolding with -DHUY_STUB_KERNEL_BODY=1."
#endif  // HUY_STUB_KERNEL_BODY
#endif  // SCRATCH_LLM_HAS_NVFP4_MMA
}

// =============================================================================================
// HOST LAUNCHER — what pybind.cpp binds and the Python wrapper calls.
//
// It takes PRE-QUANTIZED operands. Quantization is O(MK + KN) and memory bound; folding it into
// the call would put a ~25%-of-runtime memory pass inside a window whose FLOP model is 2*M*N*K,
// and the resulting TFLOP/s would be a statement about neither the GEMM nor the quantizer.
// kernels/gemm/cuda/b_r6.py's b_r6_gemm() is the bf16-in convenience wrapper and says so.
// =============================================================================================
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>

namespace {

// Bytes the swizzled scale tensor of a `rows x k` operand must have: 512 * ceil(rows/128) *
// ceil(k/64). The Python side computes the same number in nvfp4_layout.sf_tensor_bytes; a mismatch
// here is a truncated allocation, which on the last tile of the last wave is an illegal access.
int64_t sf_bytes(int64_t rows, int64_t k) {
    const int64_t row_atoms = (rows + SF_BLK_MN - 1) / SF_BLK_MN;
    const int64_t k_atoms = (k + SF_ATOM_K - 1) / SF_ATOM_K;
    return int64_t(SF_ATOM_BYTES) * row_atoms * k_atoms;
}

}  // namespace

torch::Tensor b_r6_nvfp4_mma_sync(torch::Tensor A_packed, torch::Tensor SFA,
                                  torch::Tensor B_packed, torch::Tensor SFB) {
    TORCH_CHECK(A_packed.is_cuda() && SFA.is_cuda() && B_packed.is_cuda() && SFB.is_cuda(),
                "b_r6_nvfp4_mma_sync: all four operands must be CUDA tensors");
    TORCH_CHECK(A_packed.scalar_type() == at::kByte && B_packed.scalar_type() == at::kByte &&
                    SFA.scalar_type() == at::kByte && SFB.scalar_type() == at::kByte,
                "b_r6_nvfp4_mma_sync: operands are uint8 — e2m1 nibble pairs and ue4m3 scale "
                "bytes. Pass torch.bfloat16 to b_r6_gemm() instead if you want quantization done "
                "for you");
    TORCH_CHECK(A_packed.dim() == 2 && B_packed.dim() == 2,
                "b_r6_nvfp4_mma_sync: A_packed and B_packed must be 2-D [rows, K/2]");
    TORCH_CHECK(SFA.dim() == 1 && SFB.dim() == 1,
                "b_r6_nvfp4_mma_sync: SFA and SFB are flat byte tensors in the swizzled layout");
    A_packed = A_packed.contiguous();
    B_packed = B_packed.contiguous();
    SFA = SFA.contiguous();
    SFB = SFB.contiguous();

    const int64_t M = A_packed.size(0), N = B_packed.size(0), K = A_packed.size(1) * 2;
    TORCH_CHECK(B_packed.size(1) * 2 == K, "b_r6_nvfp4_mma_sync: inner dimensions disagree: A has "
                "K=", K, ", B has K=", B_packed.size(1) * 2);
    TORCH_CHECK(K % BK == 0, "b_r6_nvfp4_mma_sync: K must be a multiple of ", BK,
                " at this rung (one K-step is one scale atom and one mma k; no remainder path); "
                "got K=", K);
    TORCH_CHECK(SFA.numel() == sf_bytes(M, K), "b_r6_nvfp4_mma_sync: SFA has ", SFA.numel(),
                " bytes, the swizzled layout for ", M, "x", K, " needs ", sf_bytes(M, K));
    TORCH_CHECK(SFB.numel() == sf_bytes(N, K), "b_r6_nvfp4_mma_sync: SFB has ", SFB.numel(),
                " bytes, the swizzled layout for ", N, "x", K, " needs ", sf_bytes(N, K));

    auto C = torch::empty({M, N}, A_packed.options().dtype(torch::kFloat32));
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(THREADS);
    b_r6_nvfp4_sm120<<<grid, block>>>(
        A_packed.data_ptr<uint8_t>(), SFA.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(), SFB.data_ptr<uint8_t>(),
        C.data_ptr<float>(), int(M), int(N), int(K),
        int((K + SF_ATOM_K - 1) / SF_ATOM_K), int((K + SF_ATOM_K - 1) / SF_ATOM_K));
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "b_r6_nvfp4_mma_sync launch failed: ", cudaGetErrorString(err));
    return C;
}
#endif  // TORCH_EXTENSION_NAME
