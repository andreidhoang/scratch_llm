// =============================================================================================
// fa3_hopper_v2.cu — K2/A-R2: FA3-shaped Hopper attention forward (sm_90a)
// =============================================================================================
//
// RUNG:   experiments/K2/A-R2/spec.md   ·   MAP: experiments/K2/A-R2/map.md
// FLOOR:  FA3 forward (flash_attn_interface), bf16, matched shape and mask, H100 SXM.
// TARGET: >= 60% of FA3 fwd, kill < 40% (plan SIXTY_DAYS_SIX_LADDERS.md §05 K2, 5-day cap).
//
// HOW THIS DIFFERS FROM csrc/attention/fa3_hopper.cu (which stays untouched). That file is the
// pre-ladder skeleton: fp16, D=64, ONE consumer warpgroup, m64n64k16, S written to shared memory
// and read back as an SS-form A-operand, and five places marked DEFER — the fragment->(row,col)
// map, the tensor-map coordinates, the mbarrier phase parity, the causal mask, the epilogue
// mapping. Its own header says so. Every one of those is load bearing, so v2 rebuilds rather than
// patches:
//   * bf16, D=128, BLOCK_M=BLOCK_N=128 — the shapes bench/kernels/attention/k2_ladder.py measures.
//   * 1 producer WARP (not warpgroup) + 2 consumer warpgroups, the FA3 §3 geometry.
//   * P NEVER touches shared memory: it stays in registers and enters O += P·V as a wgmma
//     REGISTER A-operand (the RS form). That is the structural difference between an attention
//     mainloop and a GEMM mainloop, and the whole reason P must be cast to bf16 in registers.
//   * K and V get SEPARATE mbarrier pipelines so the K stage can retire while PV is still in
//     flight (flash-attention/hopper/flash_fwd_kernel_sm90.h:226-263 gives K and V their own
//     pipeline params for exactly this).
//   * rank-4 tensor maps built from the tensor's REAL strides, so [B,H,S,D] (SDPA's layout) and
//     [B,S,H,D] (flash_attn_func's native layout) both work with no hidden transpose — a
//     benchmark that transposes for one side and not the other is not a matched comparison.
//   * the fragment->(row,col) map is derived from CUTLASS's CLayout_64xN and written down, not
//     deferred: cute/atom/mma_traits_sm90_gmma.hpp:432-434.
//
// WHAT IS ALREADY HERE (agent-written; compiles, links, launches):
//   * the host-side CUtensorMap construction for Q, K and V, rank 4, 128B-swizzled;
//   * the mbarrier ring — four barrier arrays (K/V x full/empty) plus one for Q — with the
//     producer-side phase parity derived, not guessed;
//   * the warp-role split with setmaxnreg immediates, and the named-barrier ping-pong wrappers;
//   * both wgmma forms: SS for S = Q·K^T (both operands from smem) and RS for O += P·V (A from
//     registers), with the immediates that differ between them spelled out;
//   * the P fragment packing (S accumulator registers -> the RS A-operand's 4 uint32), which is a
//     pure relayout with no data movement, and why;
//   * the accumulator -> (row, col) map and the epilogue that uses it, including the LSE write;
//   * tile/grid arithmetic and the dynamic-smem opt-in.
//
// WHAT IS HUY'S (the hole, below): the consumer mainloop body — the online softmax and the O
// rescale in registers, the causal mask on the score fragment, and the ping-pong handoff.
//
// BUILDING IT (no GPU required; infra/drydock.sh does this automatically):
//   nvcc -arch=sm_90a -cubin -O3 -Xptxas -v csrc/attention/fa3_hopper_v2.cu -o /tmp/o.cubin
// While the hole is open that fails at the #error, by design. Add -DHUY_STUB_KERNEL_BODY=1 to
// compile the SCAFFOLDING — tensor maps, barriers, both wgmma forms, epilogue, launcher, build
// wiring — around a trivially correct, catastrophically slow body. infra/bench.sh refuses to
// measure while that define (or LADDERS_STUB_HOLES=1) is in play: a number from the stub would be
// a number about the stub.
// =============================================================================================

#include <cstdint>
#include <cuda.h>          // CUtensorMap, CUresult, the CU_TENSOR_MAP_* enums
#include <cudaTypedefs.h>  // PFN_cuTensorMapEncodeTiled_v12000
#include <cuda_bf16.h>
#include <math_constants.h>  // CUDART_INF_F

#ifndef BLOCK_M
#define BLOCK_M 128   // query rows per CTA = 2 consumer warpgroups x the wgmma M of 64
#endif
#ifndef BLOCK_N
#define BLOCK_N 128   // key/value rows streamed per pipeline stage
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 128  // every K2 shape in k2_ladder.py:71-80 is d=128; not a tuning knob here
#endif
#ifndef STAGES
#define STAGES 2      // K/V staging buffers. The rung's first tunable — see spec.md.
#endif

#define WGMMA_M 64                     // architectural: wgmma.m64nNk16 always has M = 64
#define WGMMA_K 16                     // architectural: k-strip width for 16-bit operands
#define WG_THREADS 128                 // one warpgroup
#define NUM_CONSUMER_WG (BLOCK_M / WGMMA_M)              // 2 — BLOCK_M is what sets this
#define NUM_CONSUMER_THREADS (NUM_CONSUMER_WG * WG_THREADS)   // 256
#define NUM_THREADS ((NUM_CONSUMER_WG + 1) * WG_THREADS)      // 384: +1 producer warpgroup
#define PRODUCER_WARP_THREADS 32       // ONE warp issues every TMA; the other 96 exit immediately

// setmaxnreg immediates. FA3 with 2 MMA warpgroups and TMA KV uses exactly this pair
// (flash-attention/hopper/flash_fwd_kernel_sm90.h:82-83). The budget is a hard wall, not a
// preference: 256*240 + 128*24 = 64512 <= 65536 registers per SM. Raising the consumer figure to
// 248 overflows it and the CTA cannot be resident at all.
#define PRODUCER_REGS 24
#define CONSUMER_REGS 240

#define QK_KSTRIPS (HEAD_DIM / WGMMA_K)   // 8 — contract the head dim for S = Q·K^T
#define PV_KSTRIPS (BLOCK_N / WGMMA_K)    // 8 — contract the key dim for O += P·V
#define S_ACC_REGS (BLOCK_N / 2)          // 64 fp32 per thread: 128 lanes x this = 64 x BLOCK_N
#define O_ACC_REGS (HEAD_DIM / 2)         // 64 fp32 per thread: 128 lanes x this = 64 x HEAD_DIM
#define ROWS_PER_THREAD 2                 // the accumulator fragment gives each thread 2 rows

// The 128B swizzle atom, in bf16 elements. Every TMA box's INNERMOST extent must be exactly this:
// cuTensorMapEncodeTiled rejects a 128B-swizzled descriptor whose contiguous box extent is not
// 128 B, with CUDA_ERROR_INVALID_VALUE and no field named, on the host, at startup. HEAD_DIM=128
// bf16 is 256 B, so the head dim is TWO atoms and every tile takes two boxes — the same split
// K1/H-R2 makes along N, here along D. hopper_contracts.check_tma_tensor_map turns the rule into a
// named assertion and tests/kernels/attention/test_k2_a_r2.py asserts it for these maps.
#define SWZ_ELEMS 64
#define D_CHUNKS (HEAD_DIM / SWZ_ELEMS)   // 2

#define Q_TILE_BYTES  (BLOCK_M * HEAD_DIM * 2)      // 32768
#define KV_TILE_BYTES (BLOCK_N * HEAD_DIM * 2)      // 32768 — also each K/V stage's expect-tx count
#define Q_CHUNK_ELEMS  (SWZ_ELEMS * BLOCK_M)        // one 64-wide D slice of the Q tile
#define KV_CHUNK_ELEMS (SWZ_ELEMS * BLOCK_N)        // one 64-wide D slice of a K or V tile
#define STAGE_BYTES (2 * KV_TILE_BYTES)             // 65536: one K tile + one V tile
#define STAGE_ELEMS (STAGE_BYTES / 2)               // the same stride, in bf16 elements
#define SMEM_DYNAMIC_BYTES (Q_TILE_BYTES + STAGES * STAGE_BYTES)   // 163840 = 160 KB
#define SMEM_BARRIER_BYTES (8 * (4 * STAGES + 1))   // static, outside the dynamic allocation

// Named-barrier ids for the ping-pong. 0 is what __syncthreads() uses, so consumer warpgroup w
// owns id 1+w. FA3 numbers them the same way from its own reserved base
// (flash-attention/hopper/named_barrier.hpp:52-54, WarpSchedulerWG1..WG3).
#define PINGPONG_BARRIER(w) (1 + (w))

// wgmma exists ONLY in the sm_90a accelerated ISA — not base sm_90 (the trailing 'a' is load
// bearing) and not sm_100a/sm_120a, where ptxas rejects it outright. This guard makes the file
// inert, not broken, elsewhere; the kernel SIGNATURE stays arch-independent so a multi-arch build
// links, and the host launcher refuses on the wrong device before it can reach an empty kernel.
#define SCRATCH_LLM_HAS_TMA_WGMMA (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900))

// ---------------------------------------------------------------------------------------------
// 64-bit shared-memory matrix descriptor. PTX ISA, "Matrix Descriptor Format":
//   [ 0,14) start address   [16,30) leading byte offset   [32,46) stride byte offset
//   [49,52) base offset (swizzle phase)                   [62,64) layout type
// Each offset field is (x & 0x3FFFF) >> 4: mask 18 bits, drop the low 4. Dropping those 4 bits is
// the hardware asserting 16-byte alignment, not a rounding convenience.
// Cross-checked against hopper_contracts.wgmma_smem_descriptor (tests/kernels/test_hopper_contracts.py).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFFull) >> 4;
}

// LBO and SBO are template parameters, not arguments, because THE THREE OPERANDS OF THIS KERNEL DO
// NOT SHARE THEM (see the layout derivation below). Making them template parameters means Q_DESC,
// K_DESC and V_DESC are named types at the call site and cannot be swapped by accident — the
// failure mode being a kernel that runs, is fast, and is wrong.
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

// ---------------------------------------------------------------------------------------------
// Shared-memory layout, and where the three LBO/SBO pairs come from.
//
// TMA writes a box into shared memory in box-dimension order, innermost fastest, then applies the
// 128B swizzle to the resulting byte offset. So the box shape IS the smem layout, and each
// descriptor must declare the layout the box produced. CUTLASS's make_gmma_desc
// (oss/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:186-296) states the two canonical forms,
// in units of uint128_t (16 B):
//
//   Major::K  B128 : ((8,m),(T,2)) : ((8T,SBO),(1,LBO))     -- the contracted dim is contiguous
//   Major::MN B128 : ((T,8,n),(8,k)) : ((1,T,LBO),(8T,SBO)) -- the M|N dim is contiguous
//
// Every tile here is stored as D_CHUNKS = 2 slabs of {64 head-dim elements x rows}, because the
// swizzle atom forces the innermost box extent to 64 bf16. Within a slab,
//   offset(row, d) = (d % 64) + 64*row  elements, and slab c starts at c * 64*rows elements.
//
// Q — A-operand of S = Q·K^T, contracted dim = head dim, which is contiguous => Major::K.
//   In uint128_t units a slab is (rows, 8):(8, 1), so LBO = 1 x 16 B = 16 and SBO = 64 x 16 B =
//   1024. Identical to K1/H-R2's A operand and to fast.cu's make_smem_desc (matmul_2.cuh:9-17).
//
// K — B-operand of S = Q·K^T. S[m][n] = sum_c Q[m][c]·K[n][c], so as a wgmma B operand B[k][n] =
//   K[n][k] with k = the head dim: contiguous again => Major::K, same pair 16/1024. FA3 reaches
//   the same conclusion through the type system: ss_op_selector with its default Major::K for both
//   operands (flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:93).
//   Both Q and K therefore carry trans-a = trans-b = 0 on the QK wgmma.
//
// V — B-operand of O += P·V. Here the contracted dim is the KEY index and n is the head dim, and
//   it is the head dim that is contiguous => Major::MN. The pair is LBO = the stride between the
//   two 64-wide head-dim slabs = 64*BLOCK_N*2 = 16384 B, SBO = 1024 B (8 key rows x 128 B). This
//   is the same shape of asymmetry K1/H-R2 hit on its B operand, for the same reason (TMA cannot
//   transpose), and it is why the PV wgmma's trans-b immediate is 1 while the QK one's is 0. FA3:
//   `MmaMajorV = ... ? GMMA::Major::MN : GMMA::Major::K` — MN for 16-bit V
//   (mainloop_fwd_sm90_tma_gmma_ws.hpp:65).
//
// base_offset stays 0 for all three. Every descriptor base this kernel builds is a multiple of
// 1024 B (the 128B-swizzle repeat) EXCEPT the k-strip advance along a contiguous dim, which moves
// 32 B at a time — exactly the situation K1/H-R2 is in, where fast.cu leaves base_offset at 0 and
// is verified against cuBLAS (matmul_2.cuh:9-17, matmul.cu:192). The CPU test asserts the
// 1024-alignment of every tile base so the one remaining case is the documented one.
// ---------------------------------------------------------------------------------------------
#define Q_DESC_LBO 16
#define Q_DESC_SBO 1024
#define K_DESC_LBO 16
#define K_DESC_SBO 1024
#define V_DESC_LBO (SWZ_ELEMS * BLOCK_N * 2)   // 16384 — stride between the two head-dim slabs
#define V_DESC_SBO 1024

#if SCRATCH_LLM_HAS_TMA_WGMMA

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// --- mbarrier ---------------------------------------------------------------------------------
// An mbarrier is a 64-bit shared-memory object holding {pending arrivals, expected transaction
// bytes, phase}. TMA is what makes the second field matter: cp.async.bulk.tensor completes
// asynchronously and decrements the byte count as data lands, so "the tile is here" is a byte count
// reaching zero, not a thread count. Getting the count wrong does not fault — it hangs (too many
// expected) or lets the MMA read a half-written tile (too few).
//
// PTX: mbarrier.init / arrive.expect_tx / arrive / try_wait.parity; CUTLASS's
// oss/cutlass/include/cutlass/arch/barrier.h:399, :590-595, :422.

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t arrive_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(smem_u32(bar)), "r"(arrive_count) : "memory");
}

// Orders the generic-proxy writes that initialised the barriers against the async proxy the TMA
// engine reads them through. Without it the TMA unit may observe an uninitialised barrier.
// fast.cu: cde::fence_proxy_async_shared_cta(), matmul_2.cuh:107.
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// One TMA tile load, global -> shared, completion tracked by `bar`'s transaction count. Rank 4:
// the coordinates are (head-dim element, row, head, batch), innermost first — the driver's order.
// The rank is what lets a tensor keep its real strides: with a rank-2 map over a flattened
// (B*H*S, D) the sequence tail of one head would read the first rows of the next head instead of
// being out of bounds. Here dimension 1 is the sequence and TMA's own bounds checking applies.
__device__ __forceinline__ void
tma_load_4d(void* dst_smem, const CUtensorMap* tmap, uint64_t* bar,
            int c0, int c1, int c2, int c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
        :: "r"(smem_u32(dst_smem)), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
           "r"(smem_u32(bar))
        : "memory");
}

// Arrive once AND declare how many bytes this phase is waiting for. Issued by the same single
// thread that issued the TMAs, after them.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}

// A plain arrival, no byte count: how a consumer says "I am done reading this stage".
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                 :: "r"(smem_u32(bar)) : "memory");
}

// Spin until `bar` has completed the phase whose parity is `phase` (0 or 1). The parity bit, not a
// token: an mbarrier reused across k-iterations flips its phase on every completion, so the caller
// must know which flip it is waiting for. CUTLASS keeps exactly this bit in PipelineState and flips
// it on index wrap (sm90_pipeline.hpp:204-213). Written label-free (predicate -> register -> C++
// loop) so that inlining it several times in one function cannot collide on a PTX label.
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

// --- warp specialisation ----------------------------------------------------------------------
// setmaxnreg is warpgroup-COLLECTIVE (.sync.aligned): all 128 threads of the warpgroup must
// execute it, uniformly, before any of them diverges or exits. That is why the producer's three
// idle warps return only AFTER the dealloc, and why the immediate must be a compile-time constant
// in [24,256] and a multiple of 8 (hopper_contracts.check_setmaxnreg asserts both).
template <int N>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N));
}
template <int N>
__device__ __forceinline__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N));
}

// --- named barriers (the ping-pong) -----------------------------------------------------------
// `bar.sync id, count` and `bar.arrive id, count`: a rendezvous among an EXPLICIT number of
// threads, unlike __syncthreads() which means every thread in the CTA and would therefore deadlock
// here — the producer warpgroup has already returned. bar.sync counts as an arrival as well as a
// wait, which is what makes the two-warpgroup handoff work with one arrival each.
__device__ __forceinline__ void named_barrier_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void named_barrier_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(count) : "memory");
}

// --- wgmma, form 1: SS (both operands from shared memory) — S = Q·K^T --------------------------
// bf16 in, fp32 accumulate. ScaleD is a template parameter because the PTX immediate must be a
// compile-time constant: 0 overwrites the accumulator, 1 accumulates into it. The trailing
// immediates are scale-a, scale-b, trans-a, trans-b = 1, 1, 0, 0: BOTH operands are Major::K here
// (the head dim is contiguous in both Q and K), unlike K1/H-R2's GEMM where B arrives MN-major.
template <int ScaleD>
__device__ __forceinline__ void
wgmma_m64n128k16_bf16_ss(float d[S_ACC_REGS], uint64_t a_desc, uint64_t b_desc) {
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

// --- wgmma, form 2: RS (A-operand from REGISTERS) — O += P·V ----------------------------------
// THIS IS THE INSTRUCTION THAT MAKES AN ATTENTION MAINLOOP DIFFERENT FROM A GEMM MAINLOOP.
//
// In a GEMM both operands come from shared memory, so both are 64-bit descriptors and the operand
// list reads `%64, %65` (K1/H-R2, above). Here A is P = softmax(S), which the previous wgmma just
// produced INTO REGISTERS. Sending it back through shared memory to hand it to the next wgmma
// would cost a full round trip per k-block — a store, a barrier, and a load of BLOCK_M x BLOCK_N
// bf16 — for data that never leaves the warpgroup. The RS form takes it directly:
//
//   {d0..d63}, {a0,a1,a2,a3}, b_desc, scale-d, scale-a, scale-b, trans-b
//
// Note what is MISSING: there is no trans-a immediate. A register A-operand has one legal layout,
// K-major, and CUTLASS states it as a static_assert rather than an option
// ("Register source operand A must have K major layout", mma_sm90_gmma.hpp:2896-2897). The asm
// below is that struct's, MMA_64x128x16_F32BF16BF16_RS::fma at mma_sm90_gmma.hpp:2922-2957, with
// tnspB = 1 (V is Major::MN) baked in. Four immediates after the accumulator, not five.
//
// The four uint32 are 8 bf16 values: this is why P must be CAST to bf16 in registers. The
// accumulator that produced it is fp32; the MMA's A-operand is 16-bit. That cast is not a
// precision shortcut chosen for speed, it is the operand type of the instruction.
template <int ScaleD>
__device__ __forceinline__ void
wgmma_m64n128k16_bf16_rs(float d[O_ACC_REGS], const uint32_t a[4], uint64_t b_desc) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,"
        "%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,"
        "%56,%57,%58,%59,%60,%61,%62,%63}, "
        "{%64,%65,%66,%67}, "
        "%68, p, 1, 1, 1;\n"
        "}\n"
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
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "l"(b_desc), "r"(ScaleD));
}

__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait()   { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }
#endif  // SCRATCH_LLM_HAS_TMA_WGMMA

// =============================================================================================
// The wgmma accumulator fragment, written down rather than deferred.
//
// CUTLASS gives the map as a CuTe layout (cute/atom/mma_traits_sm90_gmma.hpp:432-434):
//
//   CLayout_64xN = Layout<Shape <Shape < _4,_8, _4>, Shape <_2,_2,Int<N/8>>>,
//                         Stride<Stride<_128,_1,_16>, Stride<_64,_8,   _512>>>
//
// The codomain is the 64 x N tile indexed as (row + 64*col). Decompose the thread id as
// tid = (lane%4) + 4*(lane/4) + 32*warp and the value id as v = v0 + 2*v1 + 4*v2, then read the
// strides off: 1 and 16 land on the row, 128 = 2*64, 64 and 512 = 8*64 land on the column. So
//
//   row = 16*warp + lane/4 + 8*v1          col = 2*(lane%4) + v0 + 8*v2
//
// with v0 = reg & 1, v1 = (reg >> 1) & 1, v2 = reg >> 2. Two consequences the rest of the kernel
// depends on:
//
//   1. EACH THREAD OWNS EXACTLY TWO ROWS, selected by v1 = (reg >> 1) & 1 — NOT by reg & 1, which
//      selects the column. That is the bug in the pre-ladder skeleton (fa3_hopper.cu:289, 310,
//      317: `const int r = i & 1`), and it is invisible in a benchmark: the kernel runs at full
//      speed and returns a softmax taken over the wrong axis.
//   2. The four lanes of a quad (lane%4 = 0..3) hold the SAME two rows and different columns, so
//      the row max and row sum reduce with __shfl_xor over lanes 1 and 2 and nothing wider.
//      FlashInfer calls that reduction quad_allreduce_ (attention_updater.cuh:230).
// =============================================================================================
__device__ __forceinline__ int acc_frag_row(int reg, int lane, int warp) {
    return 16 * warp + (lane >> 2) + 8 * ((reg >> 1) & 1);
}
__device__ __forceinline__ int acc_frag_col(int reg, int lane) {
    return 2 * (lane & 3) + (reg & 1) + 8 * (reg >> 2);
}
__device__ __forceinline__ int acc_frag_row_slot(int reg) { return (reg >> 1) & 1; }

// The P relayout: S accumulator registers -> the RS A-operand's four uint32, for k-strip `strip`.
//
// This costs no data movement at all, and the reason is worth the three lines it takes to see.
// The A-operand fragment of a m64nNk16 RS wgmma is ALayout_64x16 (mma_traits_sm90_gmma.hpp:451-452)
// — Shape<Shape<_4,_8,_4>,Shape<_2,_2,_2>>, Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>. That
// is byte-for-byte CLayout_64xN's thread mode and CLayout's value mode truncated to N=16. So the
// A-fragment for k-strip j IS the C-fragment restricted to columns [16j, 16j+16), held by the same
// threads in the same order: value index v = reg - 8j. FA3 performs the identical move as a pure
// view change, `convert_layout_acc_Aregs` (mainloop_fwd_sm90_tma_gmma_ws.hpp:1157), and CUTLASS as
// `make_acc_into_op`. Nothing here is a copy that a better implementation would avoid.
//
// Pairing: value indices 2i and 2i+1 differ in v0, i.e. in the column by one, so they are adjacent
// along k and pack into one __nv_bfloat162 with the even index in the low half.
__device__ __forceinline__ void
pack_p_fragment(const float p[S_ACC_REGS], int strip, uint32_t a[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = 8 * strip + 2 * i;
        __nv_bfloat162 pair = __floats2bfloat162_rn(p[r], p[r + 1]);
        a[i] = reinterpret_cast<const uint32_t&>(pair);   // a reference cast stays in a register
    }
}

// =============================================================================================
// Epilogue: normalise O by the running denominator, write O and (optionally) the log-sum-exp.
//
// Kept as a real __device__ function rather than inlined into the hole so that it compiles, is
// read by the dry dock, and is exercised by the stub body — the fragment map above is the single
// most expensive thing in this file to get wrong, and a map that is never compiled is a map that
// is never checked.
//
// `row_m_log2` is the running max IN THE LOG2 BASIS: the softmax scale is folded into the exponent
// argument once, as sm_scale_log2 = log2(e)/sqrt(d), so every exponential in the mainloop is a
// single exp2 with no multiply (FA3 carries the scale the same way — `softmax_scale_log2`,
// flash_fwd_kernel_sm90.h:416). LSE is converted back out of that basis here, exactly once, which
// is the only place ln 2 appears.
//
// O is stored as bf16 pairs: registers 2i and 2i+1 hold adjacent columns of the same row, so one
// 4-byte store replaces two 2-byte ones. FA3 instead stages O through smem and issues a TMA store
// (epilogue_fwd.hpp); whether that is worth its shared memory at this shape is an ncu question for
// the box, not a guess for the Mac — it is named as an open question in spec.md.
// =============================================================================================
__device__ __forceinline__ void
fa3_v2_epilogue(const float acc_o[O_ACC_REGS],
                const float row_m_log2[ROWS_PER_THREAD],
                const float row_l[ROWS_PER_THREAD],
                __nv_bfloat16* __restrict__ o_plane, int64_t o_stride_s,
                float* __restrict__ lse_plane,
                int q_row0, int wg, int tid_in_wg) {
    const int lane = tid_in_wg & 31;
    const int warp = tid_in_wg >> 5;

    // A fully masked row (every key in the future) has l == 0 and m == -inf. Dividing would give
    // NaN and poison a whole training step; FA3 clamps for the same reason
    // (flashinfer attention_updater.cuh:116-117). Zero output, -inf LSE is what a zero-length
    // softmax means.
    float inv_l[ROWS_PER_THREAD];
    #pragma unroll
    for (int v = 0; v < ROWS_PER_THREAD; ++v) {
        inv_l[v] = (row_l[v] > 0.0f) ? (1.0f / row_l[v]) : 0.0f;
    }

    #pragma unroll
    for (int r = 0; r < O_ACC_REGS; r += 2) {
        const int v = acc_frag_row_slot(r);
        const int row = q_row0 + WGMMA_M * wg + acc_frag_row(r, lane, warp);
        const int col = acc_frag_col(r, lane);   // even, so the pair is 4-byte aligned
        const __nv_bfloat162 pair =
            __floats2bfloat162_rn(acc_o[r] * inv_l[v], acc_o[r + 1] * inv_l[v]);
        *reinterpret_cast<__nv_bfloat162*>(o_plane + static_cast<int64_t>(row) * o_stride_s + col) =
            pair;
    }

    if (lse_plane != nullptr && (lane & 3) == 0) {
        // The four lanes of a quad hold identical row state after the reduction; one writes.
        #pragma unroll
        for (int v = 0; v < ROWS_PER_THREAD; ++v) {
            const int row = q_row0 + WGMMA_M * wg + 16 * warp + (lane >> 2) + 8 * v;
            lse_plane[row] = (row_l[v] > 0.0f)
                                 ? (row_m_log2[v] + log2f(row_l[v])) * 0.693147180559945309f
                                 : -CUDART_INF_F;
        }
    }
}

// =============================================================================================
// Kernel: O[B,H,S,D] = softmax(Q·K^T · scale) · V,  bf16 in / bf16 out, fp32 accumulate.
//   grid  = (S/BLOCK_M, H, B)
//   block = NUM_THREADS = 384 = 1 producer warpgroup (of which one warp works) + 2 consumers
//
// The tensor maps come in BY VALUE as __grid_constant__ parameters. That is not a style choice:
// cp.async.bulk.tensor requires the descriptor in parameter or global space, and __grid_constant__
// is what lets the address of a by-value parameter be taken without the compiler making a local
// copy (fast.cu switched to it at matmul_6.cuh:315 after paying a dependent global load per launch
// for the pointer form, matmul_2.cuh:55-61).
//
// Q, K and V also arrive as raw pointers with their strides. The real path does not read them —
// they exist for the stub body, which must compute a correct O without going through a tensor map.
// =============================================================================================
extern "C" __global__ void __launch_bounds__(NUM_THREADS, 1)
fa3_hopper_v2_fwd_kernel(const __grid_constant__ CUtensorMap tmap_q,
                         const __grid_constant__ CUtensorMap tmap_k,
                         const __grid_constant__ CUtensorMap tmap_v,
                         const __nv_bfloat16* __restrict__ Q,   // stub path only
                         const __nv_bfloat16* __restrict__ K,   // stub path only
                         const __nv_bfloat16* __restrict__ V,   // stub path only
                         __nv_bfloat16* __restrict__ O,
                         float* __restrict__ LSE,               // may be nullptr
                         int seqlen, int kv_group,
                         int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_s,
                         int64_t kv_stride_b, int64_t kv_stride_h, int64_t kv_stride_s,
                         float softmax_scale_log2, int causal) {
#if !SCRATCH_LLM_HAS_TMA_WGMMA
    // Inert on every arch but sm_90a, so a multi-arch AOT build links. The host launcher checks the
    // device's compute capability and refuses before it can ever reach an empty kernel.
    (void)tmap_q; (void)tmap_k; (void)tmap_v; (void)Q; (void)K; (void)V; (void)O; (void)LSE;
    (void)seqlen; (void)kv_group; (void)q_stride_b; (void)q_stride_h; (void)q_stride_s;
    (void)kv_stride_b; (void)kv_stride_h; (void)kv_stride_s; (void)softmax_scale_log2; (void)causal;
#else
    const int tid    = threadIdx.x;
    const int q_tile = blockIdx.x;
    const int head   = blockIdx.y;
    const int batch  = blockIdx.z;
    const int q_row0 = q_tile * BLOCK_M;
    const int kv_head = head / kv_group;   // GQA: kv_group query heads share one KV head

    // The k-block trip count, computed ONCE and used by the producer and both consumers. Two
    // separately derived counts that disagree by one is not a wrong answer, it is a hang: the
    // producer waits on an empty barrier nobody will signal, or the consumer waits on a full
    // barrier nobody will fill. Causal masking makes the count depend on the query tile, which is
    // exactly where the two derivations would drift.
    //
    // S_q == S_kv is required at this rung (the launcher enforces it), so the diagonal for this
    // tile ends at row q_row0 + BLOCK_M - 1 and the last key block it can see is
    // (q_row0 + BLOCK_M - 1) / BLOCK_N.
    const int kv_blocks_total = seqlen / BLOCK_N;
    const int n_kv_blocks =
        causal ? min(kv_blocks_total, (q_row0 + BLOCK_M - 1) / BLOCK_N + 1) : kv_blocks_total;

    // 160 KB of staging, past the 48 KB static limit, so the buffers are dynamic and the launcher
    // opts in with cudaFuncSetAttribute. 1024-byte aligned because that is the 128B-swizzle repeat:
    // a tile base at a 1024-byte boundary has swizzle phase 0, which is what lets every descriptor
    // below carry base_offset = 0.
    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* const sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    // Stage bases are computed ARITHMETICALLY, not read out of a `sK[STAGES]` array. An array of
    // pointers indexed by a runtime `stage` is a local-memory array — nvcc puts it in the stack
    // frame and the mainloop pays an LDL per k-block. `-Xptxas -v` reports it as a stack frame with
    // ZERO spill bytes, so the spill gate does not catch it; the SASS LDL/STL assertion in
    // tests/kernels/attention/test_k2_a_r2.py is what does.
    __nv_bfloat16* const sK0 = reinterpret_cast<__nv_bfloat16*>(smem + Q_TILE_BYTES);
    __nv_bfloat16* const sV0 =
        reinterpret_cast<__nv_bfloat16*>(smem + Q_TILE_BYTES + KV_TILE_BYTES);

    // Static, not carved out of the dynamic block, so SMEM_DYNAMIC_BYTES stays exactly the number
    // the launcher opts into and the expect-tx counts have to match. K and V get SEPARATE full and
    // empty barriers: the point of the split is that a consumer can release the K stage as soon as
    // the QK wgmma retires, while the PV wgmma is still reading V (FA3 gives K and V their own
    // pipelines for this, flash_fwd_kernel_sm90.h:226-263). One shared barrier would hold K
    // hostage to PV and halve the depth of the pipeline for free.
    __shared__ __align__(8) uint64_t bar_k_full[STAGES];
    __shared__ __align__(8) uint64_t bar_k_empty[STAGES];
    __shared__ __align__(8) uint64_t bar_v_full[STAGES];
    __shared__ __align__(8) uint64_t bar_v_empty[STAGES];
    __shared__ __align__(8) uint64_t bar_q;

    // Arrive counts. The `full` barriers count ONE arrival — the single producer thread that issued
    // the TMAs and the expect-tx; waiting is not arriving. The `empty` barriers count every
    // consumer thread, which is what CUTLASS/FA3 do (`consumer_arv_count = NumMmaThreads`,
    // flash_fwd_kernel_sm90.h:228). Mixing the two conventions is a hang, and a hang on a rented
    // box is an hour.
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbarrier_init(&bar_k_full[s], 1);
            mbarrier_init(&bar_v_full[s], 1);
            mbarrier_init(&bar_k_empty[s], NUM_CONSUMER_THREADS);
            mbarrier_init(&bar_v_empty[s], NUM_CONSUMER_THREADS);
        }
        mbarrier_init(&bar_q, 1);
        fence_proxy_async_shared_cta();
    }
    // The ONLY CTA-wide barrier in this kernel, and it must be before the role split: after the
    // producer warps return, __syncthreads() can never be satisfied again. Everything downstream
    // uses named barriers with explicit counts for exactly that reason.
    __syncthreads();

    // =========================================================================================
    // PRODUCER warpgroup — 128 threads execute setmaxnreg (it is warpgroup-collective), then 96 of
    // them leave and one warp issues every TMA in the kernel.
    // =========================================================================================
    if (tid < WG_THREADS) {
        warpgroup_reg_dealloc<PRODUCER_REGS>();
        if (tid < PRODUCER_WARP_THREADS) {
            // Q, once: two boxes, one per 64-wide head-dim slab.
            if (tid == 0) {
                #pragma unroll
                for (int c = 0; c < D_CHUNKS; ++c) {
                    tma_load_4d(sQ + c * Q_CHUNK_ELEMS, &tmap_q, &bar_q,
                                c * SWZ_ELEMS, q_row0, head, batch);
                }
                mbarrier_arrive_expect_tx(&bar_q, Q_TILE_BYTES);
            }

            for (int kt = 0; kt < n_kv_blocks; ++kt) {
                const int stage = kt % STAGES;
                // The producer's phase parity, derived rather than guessed. CUTLASS states it in
                // one line: "Producer starts with an opposite phase as the buffers are initially
                // empty" — InitialProducerPhase = 1, flipped on every index wrap
                // (sm90_pipeline.hpp:254-260 and :204-213). So stage `s` is re-acquired on
                // iterations s, s+STAGES, ..., and the parity it waits for is
                // ((kt / STAGES) & 1) ^ 1: 1 for the first STAGES iterations, which a
                // freshly-initialised barrier satisfies immediately, then alternating. Starting at
                // 0 instead deadlocks on iteration 0, which at least fails loudly; getting the
                // FLIP wrong deadlocks on iteration STAGES, after the kernel has looked healthy.
                const uint32_t empty_phase = ((kt / STAGES) & 1) ^ 1u;
                const int kv_row0 = kt * BLOCK_N;

                mbarrier_wait_parity(&bar_k_empty[stage], empty_phase);
                if (tid == 0) {
                    #pragma unroll
                    for (int c = 0; c < D_CHUNKS; ++c) {
                        tma_load_4d(sK0 + stage * STAGE_ELEMS + c * KV_CHUNK_ELEMS, &tmap_k,
                                    &bar_k_full[stage],
                                    c * SWZ_ELEMS, kv_row0, kv_head, batch);
                    }
                    // One arrive per stage, and its byte count must be what BOTH boxes deliver
                    // together. Count one box, or count the tile twice, and the barrier never
                    // releases.
                    mbarrier_arrive_expect_tx(&bar_k_full[stage], KV_TILE_BYTES);
                }

                mbarrier_wait_parity(&bar_v_empty[stage], empty_phase);
                if (tid == 0) {
                    #pragma unroll
                    for (int c = 0; c < D_CHUNKS; ++c) {
                        tma_load_4d(sV0 + stage * STAGE_ELEMS + c * KV_CHUNK_ELEMS, &tmap_v,
                                    &bar_v_full[stage],
                                    c * SWZ_ELEMS, kv_row0, kv_head, batch);
                    }
                    mbarrier_arrive_expect_tx(&bar_v_full[stage], KV_TILE_BYTES);
                }
            }
        }
        return;
    }

    // =========================================================================================
    // CONSUMER warpgroups — one per 64 query rows of the BLOCK_M tile.
    // =========================================================================================
    warpgroup_reg_alloc<CONSUMER_REGS>();
    const int wg        = tid / WG_THREADS - 1;   // 0 or 1
    const int tid_in_wg = tid % WG_THREADS;

    // The running attention state, all of it in registers for the whole mainloop:
    //   acc_o        the unnormalised output O~, an m64 x HEAD_DIM wgmma accumulator
    //   row_m_log2   the running row max, in the log2 basis (the scale is folded into it)
    //   row_l        the running denominator
    // 64 + 64 fp32 accumulators plus the P fragment is why CONSUMER_REGS is 240 and why exactly one
    // CTA is resident per SM. The ptxas report the dry dock writes is the number to check this
    // against — and while the hole is open, that report describes the stub.
    float acc_o[O_ACC_REGS];
    float row_m_log2[ROWS_PER_THREAD];
    float row_l[ROWS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < O_ACC_REGS; ++i) acc_o[i] = 0.0f;
    #pragma unroll
    for (int v = 0; v < ROWS_PER_THREAD; ++v) {
        row_m_log2[v] = -CUDART_INF_F;
        row_l[v] = 0.0f;
    }

    __nv_bfloat16* const o_plane = O + batch * q_stride_b + head * q_stride_h;
    float* const lse_plane =
        (LSE == nullptr) ? nullptr
                         : LSE + (static_cast<int64_t>(batch) * gridDim.y + head) * seqlen;

#if HUY_STUB_KERNEL_BODY
    // ---- STUB (only under -DHUY_STUB_KERNEL_BODY=1) ------------------------------------------
    // Trivially correct, catastrophically slow: a literal three-pass safe softmax, read straight
    // from global memory in fp32, over exactly the (row, col) pairs this thread's accumulator
    // fragment owns. It exists so the scaffolding around the hole — tensor maps, the barrier ring,
    // the warp-role split, the fragment map, the epilogue, the launcher, the AOT build — can be
    // compiled and smoke-tested while the hole is open. It uses NO online recurrence, NO wgmma and
    // NO shared memory, so it says nothing whatsoever about any of them. infra/bench.sh refuses to
    // run while this is in play.
    //
    // It does still DRAIN the pipeline. It must: the producer warp is real code and is issuing TMAs
    // regardless, so a consumer that never released a stage would hang the CTA on iteration
    // STAGES, and a CTA that exits with a TMA in flight is undefined behaviour.
    for (int kt = 0; kt < n_kv_blocks; ++kt) {
        const int stage = kt % STAGES;
        const uint32_t full_phase = (kt / STAGES) & 1u;
        mbarrier_wait_parity(&bar_k_full[stage], full_phase);
        mbarrier_arrive(&bar_k_empty[stage]);
        mbarrier_wait_parity(&bar_v_full[stage], full_phase);
        mbarrier_arrive(&bar_v_empty[stage]);
    }
    mbarrier_wait_parity(&bar_q, 0);

    {
        const int lane = tid_in_wg & 31;
        const int warp = tid_in_wg >> 5;
        const __nv_bfloat16* const q_plane = Q + batch * q_stride_b + head * q_stride_h;
        const __nv_bfloat16* const k_plane = K + batch * kv_stride_b + kv_head * kv_stride_h;
        const __nv_bfloat16* const v_plane = V + batch * kv_stride_b + kv_head * kv_stride_h;

        #pragma unroll
        for (int v = 0; v < ROWS_PER_THREAD; ++v) {
            const int q_abs = q_row0 + WGMMA_M * wg + 16 * warp + (lane >> 2) + 8 * v;
            const int kv_end = causal ? (q_abs + 1) : seqlen;
            // pass 1: the row max, in the log2 basis the epilogue expects.
            float mx = -CUDART_INF_F;
            for (int j = 0; j < kv_end; ++j) {
                float s = 0.0f;
                for (int c = 0; c < HEAD_DIM; ++c) {
                    s += __bfloat162float(q_plane[static_cast<int64_t>(q_abs) * q_stride_s + c]) *
                         __bfloat162float(k_plane[static_cast<int64_t>(j) * kv_stride_s + c]);
                }
                mx = fmaxf(mx, s * softmax_scale_log2);
            }
            row_m_log2[v] = mx;
            // passes 2 and 3: the denominator, and P·V over this thread's columns.
            float l = 0.0f;
            for (int j = 0; j < kv_end; ++j) {
                float s = 0.0f;
                for (int c = 0; c < HEAD_DIM; ++c) {
                    s += __bfloat162float(q_plane[static_cast<int64_t>(q_abs) * q_stride_s + c]) *
                         __bfloat162float(k_plane[static_cast<int64_t>(j) * kv_stride_s + c]);
                }
                const float p = exp2f(s * softmax_scale_log2 - mx);
                l += p;
                #pragma unroll
                for (int r = 0; r < O_ACC_REGS; ++r) {
                    if (acc_frag_row_slot(r) != v) continue;
                    acc_o[r] += p * __bfloat162float(
                        v_plane[static_cast<int64_t>(j) * kv_stride_s + acc_frag_col(r, lane)]);
                }
            }
            row_l[v] = l;
        }
    }
    fa3_v2_epilogue(acc_o, row_m_log2, row_l, o_plane, q_stride_s, lse_plane, q_row0, wg,
                    tid_in_wg);
    (void)sQ; (void)sK0; (void)sV0;
#else
    // HUY: the consumer mainloop — online softmax and O rescale in registers, causal mask on the score fragment, ping-pong handoff — spec: experiments/K2/A-R2/spec.md — fill before A-R2
    //
    // Everything around this block is written: the barrier ring and its parities, both wgmma forms,
    // the P relayout (pack_p_fragment), the fragment map (acc_frag_row / acc_frag_col), and the
    // epilogue. What lives here and nowhere else is the ORDER of the two asynchronous MMAs against
    // the softmax that sits between them, and the state machine that keeps two consumer warpgroups
    // out of each other's way.
    //
    //  0. PRIME THE PING-PONG. Warpgroup 0 alone calls
    //     named_barrier_arrive(PINGPONG_BARRIER(0), NUM_CONSUMER_THREADS) once, before the loop.
    //     Barrier w is a rendezvous of all 256 consumer threads; warpgroup w waits on its OWN
    //     (named_barrier_sync(PINGPONG_BARRIER(w), NUM_CONSUMER_THREADS)) and releases the OTHER
    //     (named_barrier_arrive(PINGPONG_BARRIER(1 - w), ...)). Without the prime, both warpgroups
    //     wait on barriers holding 128 of the required 256 arrivals and the CTA is dead on
    //     iteration 0. FA3 primes exactly this way, from `mma_init`, and only for WG1
    //     (mainloop_fwd_sm90_tma_gmma_ws.hpp:944-947; the sync/arrive pair is :915-930).
    //     Then wait for Q: mbarrier_wait_parity(&bar_q, 0).
    //
    //  1. THE k-LOOP, kt = 0 .. n_kv_blocks-1, stage = kt % STAGES, full-barrier parity
    //     (kt / STAGES) & 1 — the CONSUMER side of the parity the producer above derives, and the
    //     complement of it. A wrong parity here hangs on iteration STAGES, not on 0.
    //
    //  2. S = Q·K^T. mbarrier_wait_parity(&bar_k_full[stage], phase), then wgmma_fence() and, for
    //     each of the QK_KSTRIPS = 8 strips j:
    //         a_desc = make_smem_desc<Q_DESC_LBO, Q_DESC_SBO>(
    //                      sQ + (j / 4) * Q_CHUNK_ELEMS + wg * WGMMA_M * SWZ_ELEMS + (j % 4) * WGMMA_K)
    //         b_desc = make_smem_desc<K_DESC_LBO, K_DESC_SBO>(
    //                      sK0 + stage * STAGE_ELEMS + (j / 4) * KV_CHUNK_ELEMS + (j % 4) * WGMMA_K)
    //         wgmma_m64n128k16_bf16_ss<j == 0 ? 0 : 1>(acc_s, a_desc, b_desc);
    //     The j/4 and j%4 are the head dim's two 64-wide slabs: the strip advances 32 B inside a
    //     slab and jumps a whole slab every four strips. wg * WGMMA_M * SWZ_ELEMS is this
    //     warpgroup's 64 query rows, 8192 B — a multiple of the 1024 B swizzle repeat, so
    //     base_offset stays 0. Then wgmma_commit().
    //
    //  3. MASK. Causal, on the score FRAGMENT, before any exponential: for register r, the key
    //     index is kt * BLOCK_N + acc_frag_col(r, lane) and the query index is
    //     q_row0 + WGMMA_M * wg + acc_frag_row(r, lane, warp); set acc_s[r] to -inf where key >
    //     query. Only the diagonal block needs it — for kt * BLOCK_N + BLOCK_N - 1 <= q_row0 the
    //     whole block is in the past and the branch is skippable, which is where causal block
    //     skipping actually pays. -CUDART_INF_F, not a large negative: exp2 of it is exactly 0.
    //
    //  4. THE ONLINE SOFTMAX, in registers, on the fragment:
    //       m_tile[v] = max over the registers with acc_frag_row_slot(r) == v, then reduced across
    //                   the quad with __shfl_xor_sync at offsets 1 and 2 and NOTHING WIDER — the
    //                   four lanes of a quad share a row, other lanes do not (see the CLayout
    //                   derivation above; FlashInfer's quad_allreduce_, attention_updater.cuh:230).
    //       m_new[v]  = fmaxf(row_m_log2[v], m_tile[v])
    //       alpha[v]  = exp2f(row_m_log2[v] - m_new[v])      the rebasing factor
    //       row_l[v] *= alpha[v];  row_m_log2[v] = m_new[v]
    //       acc_o[r] *= alpha[acc_frag_row_slot(r)]           RESCALE O, NOT JUST l
    //       acc_s[r]  = exp2f(acc_s[r] - m_new[slot])         P, still fp32, still in registers
    //       row_l[v] += sum of the P registers for row slot v
    //     Two things to get right that a benchmark cannot see. First, forgetting to rescale acc_o
    //     leaves an output that is correct only when the max never rises — which is most rows, most
    //     of the time. Second, DEFER the cross-lane sum of row_l to the epilogue: the max must be
    //     reduced every tile (it gates the rescale), the denominator need not be, and FA3 passes
    //     warp_reduce=false for exactly this, doing one reduction per row for the whole loop
    //     instead of one per tile (FlashInfer attention_updater.cuh:194, :219, :230). If you defer
    //     it, the epilogue is where the quad reduction on row_l has to happen.
    //
    //  5. O += P·V. mbarrier_wait_parity(&bar_v_full[stage], phase), then CONVERT THE WHOLE P
    //     FRAGMENT FIRST and only then issue:
    //         uint32_t p_frag[PV_KSTRIPS][4];
    //         for j: pack_p_fragment(acc_s, j, p_frag[j]);
    //         wgmma_fence();
    //         for j: b_desc = make_smem_desc<V_DESC_LBO, V_DESC_SBO>(
    //                             sV0 + stage * STAGE_ELEMS + j * WGMMA_K * SWZ_ELEMS);
    //                wgmma_m64n128k16_bf16_rs<1>(acc_o, p_frag[j], b_desc);
    //     ScaleD is 1 on every strip and every iteration — O accumulates across the whole k-loop,
    //     which is the entire point of the rescale in step 4. The wgmma_fence() is NOT optional and
    //     NOT the same one as step 2's: P was just written by cvt instructions and acc_o by FMAs,
    //     both ordinary register writes, and PTX requires the fence between those and a wgmma that
    //     reads the registers. FA3 converts the whole fragment once and then calls gemm
    //     (mainloop_fwd_sm90_tma_gmma_ws.hpp:1157 then :1160), which is what the ORDER above is
    //     copying — and the order is measurable, not cosmetic. Interleaving the pack with the issue
    //     makes ptxas insert its own fence per strip and say so:
    //         ptxas info : (C7519) warpgroup.arrive is injected ... to allow use of registers in GMMA
    //     Seven of those in the ptxas report is seven serialisation points ptxas added because the
    //     code asked for them; converting first removes all seven (verified in the dry dock,
    //     2026-09-07). C7519 in experiments/K2/A-R2/drydock/*.ptxas.txt is the signal to look for.
    //     Then wgmma_commit().
    //
    //  6. RETIRE, IN THIS ORDER. wgmma_wait<1>() retires the QK group while PV is still running:
    //     that is what lets K be released early — every consumer thread calls
    //     mbarrier_arrive(&bar_k_empty[stage]) here, so the producer can start refilling K for
    //     kt+STAGES while this iteration's PV is still reading V. Then wgmma_wait<0>() and
    //     mbarrier_arrive(&bar_v_empty[stage]). FlashInfer's steady-state loop is exactly this
    //     sequence (mainloop_mma.cuh:231-271, wait<1> at :242, release K at :243, wait<0> at :267,
    //     release V at :268).
    //
    //  7. THE PING-PONG. named_barrier_sync(PINGPONG_BARRIER(wg), NUM_CONSUMER_THREADS) before the
    //     wgmma issues, named_barrier_arrive(PINGPONG_BARRIER(1 - wg), NUM_CONSUMER_THREADS) after
    //     them, so that while one warpgroup holds the tensor cores the other is doing its softmax
    //     on the multiply-add pipes. Where exactly the pair goes is the measurement this rung
    //     exists to make, and it is a compile-time GUESS upstream, not a derivation: FlashInfer
    //     switches it on `head_dim <= 128` for 16-bit (mainloop.cuh:90-91). Build it both ways and
    //     let the tensor-pipe utilisation and the issue-stall reasons decide — that is the open
    //     question in spec.md, and answering it with a number is the rung.
    //
    //  8. EPILOGUE. Reduce row_l across the quad if step 4 deferred it, then
    //     fa3_v2_epilogue(acc_o, row_m_log2, row_l, o_plane, q_stride_s, lse_plane, q_row0, wg,
    //                     tid_in_wg);
    //
    // The map (experiments/K2/A-R2/map.md) has the file:line for each of these upstream, and
    // oss/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1150-1300 is FA3's own version
    // of steps 2-6 in one function.
    #error "HUY: K2/A-R2 consumer mainloop — online softmax + O rescale in registers, causal mask, ping-pong handoff. See the comment above, experiments/K2/A-R2/spec.md, and map.md. Compile the scaffolding with -DHUY_STUB_KERNEL_BODY=1."
#endif  // HUY_STUB_KERNEL_BODY
#endif  // SCRATCH_LLM_HAS_TMA_WGMMA
}


// =============================================================================================
// ASSEMBLER PROBE — stub builds only. Never in the AOT extension, never launched.
//
// The stub body deliberately uses no wgmma and no named barriers, which means that while the hole
// is open the dry dock proves nothing about the two instructions this rung is actually built out
// of: the wrappers are __forceinline__ and uninstantiated, so their asm is never handed to ptxas.
// "It compiles" would then be a claim about the loads and the epilogue only.
//
// This kernel closes that gap. It calls every wrapper the mainloop will call — both wgmma forms,
// the P relayout, all three descriptor pairs, the ping-pong barriers, the epilogue — in the order
// the hole's comment prescribes, so ptxas assembles all of them TODAY. It computes nothing; it is
// not registered anywhere and nothing takes its address. `#if HUY_STUB_KERNEL_BODY` keeps it out of
// every real build, and the tests/kernels/attention/test_k2_a_r2.py drydock tier reads the HGMMA
// and BAR lines it produces.
// =============================================================================================
#if HUY_STUB_KERNEL_BODY && SCRATCH_LLM_HAS_TMA_WGMMA
extern "C" __global__ void __launch_bounds__(NUM_THREADS, 1)
fa3_v2_asm_probe(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, int wg) {
    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* const sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* const sK0 = reinterpret_cast<__nv_bfloat16*>(smem + Q_TILE_BYTES);
    __nv_bfloat16* const sV0 =
        reinterpret_cast<__nv_bfloat16*>(smem + Q_TILE_BYTES + KV_TILE_BYTES);

    float acc_s[S_ACC_REGS], acc_o[O_ACC_REGS];
    #pragma unroll
    for (int i = 0; i < S_ACC_REGS; ++i) acc_s[i] = __bfloat162float(src[i]);
    #pragma unroll
    for (int i = 0; i < O_ACC_REGS; ++i) acc_o[i] = 0.0f;

    named_barrier_sync(PINGPONG_BARRIER(wg), NUM_CONSUMER_THREADS);

    wgmma_fence();
    #pragma unroll
    for (int j = 0; j < QK_KSTRIPS; ++j) {
        const uint64_t a = make_smem_desc<Q_DESC_LBO, Q_DESC_SBO>(
            sQ + (j / 4) * Q_CHUNK_ELEMS + wg * WGMMA_M * SWZ_ELEMS + (j % 4) * WGMMA_K);
        const uint64_t b = make_smem_desc<K_DESC_LBO, K_DESC_SBO>(
            sK0 + (j / 4) * KV_CHUNK_ELEMS + (j % 4) * WGMMA_K);
        if (j == 0) wgmma_m64n128k16_bf16_ss<0>(acc_s, a, b);
        else        wgmma_m64n128k16_bf16_ss<1>(acc_s, a, b);
    }
    wgmma_commit();
    wgmma_wait<0>();

    // Convert the whole P fragment BEFORE issuing any PV wgmma — see step 5 of the hole comment.
    uint32_t p_frag[PV_KSTRIPS][4];
    #pragma unroll
    for (int j = 0; j < PV_KSTRIPS; ++j) pack_p_fragment(acc_s, j, p_frag[j]);
    wgmma_fence();
    #pragma unroll
    for (int j = 0; j < PV_KSTRIPS; ++j) {
        const uint64_t b = make_smem_desc<V_DESC_LBO, V_DESC_SBO>(
            sV0 + j * WGMMA_K * SWZ_ELEMS);
        wgmma_m64n128k16_bf16_rs<1>(acc_o, p_frag[j], b);
    }
    wgmma_commit();
    wgmma_wait<1>();   // retire QK only — the shape of the early-K release in step 6
    wgmma_wait<0>();
    named_barrier_arrive(PINGPONG_BARRIER(1 - wg), NUM_CONSUMER_THREADS);

    const float row_m[ROWS_PER_THREAD] = {acc_s[0], acc_s[2]};
    const float row_l[ROWS_PER_THREAD] = {1.0f, 1.0f};
    fa3_v2_epilogue(acc_o, row_m, row_l, dst, HEAD_DIM, nullptr, 0, wg,
                    static_cast<int>(threadIdx.x) % WG_THREADS);
}
#endif  // HUY_STUB_KERNEL_BODY && SCRATCH_LLM_HAS_TMA_WGMMA

// =============================================================================================
// HOST — tensor map construction.
//
// Deliberately OUTSIDE the TORCH_EXTENSION_NAME guard: `nvcc -cubin` still type-checks host code,
// so the dry dock on the Mac compiles the cuTensorMapEncodeTiled call, its argument types and its
// enum spellings. Everything that needs torch stays below, where the dry dock cannot see it.
// =============================================================================================

// cuTensorMapEncodeTiled lives in libcuda, which a torch extension does not link. Resolving it
// through the runtime's driver-entry-point table is how CUTLASS avoids that link edge, and it also
// makes a driver too old to have the symbol a clean runtime error instead of a load failure.
PFN_cuTensorMapEncodeTiled_v12000 fa3_v2_tensor_map_encoder() {
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

// One rank-4, 128B-swizzled, bf16 tiled tensor map over a [batch, head, seq, dim] logical tensor
// whose strides are given (so a [B,S,H,D] tensor is described by swapping two of them, not by
// transposing the data).
//   dims, innermost first: (HEAD_DIM, seqlen, heads, batch) — the driver's order, not the matrix's.
//   strides: dimensions 1.. only. Dimension 0's stride is not passed at all — the descriptor
//     assumes the innermost stride is one element, which is why fast.cu hands over
//     `gmem_prob_stride + 1` (matmul_2.cuh:38,:44) and CUTLASS does the same
//     (copy_traits_sm90_tma.hpp:969,:1053). Each must be a multiple of 16 B.
//   box: {SWZ_ELEMS, rows, 1, 1} — 64 bf16 is exactly the 128 B swizzle atom, and the head and
//     batch dimensions are indexed, not tiled.
// FLOAT_OOB_FILL_NONE zero-fills out-of-bounds reads. For a GEMM that is a free tail (a zero
// contributes nothing to a sum); for attention it is NOT, because a score of zero is not -inf and
// would enter the softmax as a real key. That is why the launcher requires seqlen % BLOCK_N == 0
// rather than leaning on it.
CUresult fa3_v2_encode_tensor_map(CUtensorMap* map, void* gmem,
                                  uint64_t head_dim, uint64_t seqlen, uint64_t heads,
                                  uint64_t batch,
                                  uint64_t stride_s_bytes, uint64_t stride_h_bytes,
                                  uint64_t stride_b_bytes,
                                  uint32_t box_rows) {
    PFN_cuTensorMapEncodeTiled_v12000 encode = fa3_v2_tensor_map_encoder();
    if (encode == nullptr) return CUDA_ERROR_NOT_SUPPORTED;

    const uint64_t gmem_prob_shape[4]  = {head_dim, seqlen, heads, batch};
    const uint64_t gmem_prob_stride[3] = {stride_s_bytes, stride_h_bytes, stride_b_bytes};
    const uint32_t smem_box_shape[4]   = {SWZ_ELEMS, box_rows, 1, 1};
    const uint32_t smem_box_stride[4]  = {1, 1, 1, 1};

    return encode(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, /*tensorRank=*/4, gmem,
                  gmem_prob_shape, gmem_prob_stride, smem_box_shape, smem_box_stride,
                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// =============================================================================================
// HOST LAUNCHER — what pybind.cpp binds and the Python wrapper calls.
// =============================================================================================
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>
#include <cmath>
#include <vector>

// Q, K, V are 4-D [batch, heads, seq, dim] (heads_last = false) or [batch, seq, heads, dim]
// (heads_last = true). Whichever it is, the last dimension must be contiguous and the real strides
// are read off the tensor and handed to TMA, so neither layout needs a transpose. Returns
// {O, LSE}: O like Q, LSE [batch, heads, seq] fp32. Upstream gates BOTH (flashinfer
// tests/attention/test_hopper.py:55-56, CUTLASS 88_hopper_fmha.cu:327,:334) and an O-only gate is
// weaker than the thing this rung is measured against.
std::vector<torch::Tensor> fa3_hopper_v2_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                             bool is_causal, bool heads_last) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "fa3_hopper_v2_fwd: Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == at::kBFloat16 && K.scalar_type() == at::kBFloat16 &&
                    V.scalar_type() == at::kBFloat16,
                "fa3_hopper_v2_fwd: Q, K, V must be bfloat16 (wgmma.f32.bf16.bf16 operands); the "
                "K2 floor is FA3 bf16, so a float16 input would be measured against the wrong floor");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "fa3_hopper_v2_fwd: Q, K, V must be 4-D (batch, heads, seq, dim) or "
                "(batch, seq, heads, dim)");
    TORCH_CHECK(K.sizes() == V.sizes() && K.strides() == V.strides(),
                "fa3_hopper_v2_fwd: K and V must share shape and layout (one tensor map shape is "
                "built for both)");

    // Which of dims 1 and 2 is the head axis is TOLD to this function, not inferred from the
    // strides. Inferring is the obvious idea and it cannot work: a contiguous [B,H,S,D] and a
    // contiguous [B,S,H,D] have the same stride PATTERN — dim 3 is 1, dim 2 is D, dims 1 and 0
    // decrease — so no predicate over strides separates them, and any that appears to is really
    // testing for a transposed view. Getting it wrong does not fault; it swaps heads for sequence
    // and the shape checks below then reject a perfectly good tensor (or, at S == H, accept it and
    // compute nonsense). Both layouts exist because the floor and the reference disagree —
    // flash_attn_func takes [B,S,H,D], SDPA takes [B,H,S,D] — and a silent .transpose() on one side
    // of that comparison is a memcpy inside the measured region.
    const bool q_heads_last = heads_last;
    const int64_t batch = Q.size(0);
    const int64_t heads = q_heads_last ? Q.size(2) : Q.size(1);
    const int64_t seq_q = q_heads_last ? Q.size(1) : Q.size(2);
    const int64_t kv_heads = q_heads_last ? K.size(2) : K.size(1);
    const int64_t seq_kv = q_heads_last ? K.size(1) : K.size(2);

    TORCH_CHECK(Q.size(3) == HEAD_DIM && K.size(3) == HEAD_DIM,
                "fa3_hopper_v2_fwd: head dim must be ", HEAD_DIM, " at this rung; got ", Q.size(3));
    TORCH_CHECK(Q.stride(3) == 1 && K.stride(3) == 1 && V.stride(3) == 1,
                "fa3_hopper_v2_fwd: the head dim must be contiguous (TMA's innermost stride is "
                "one element and is not passed to the descriptor at all)");
    TORCH_CHECK(seq_q == seq_kv,
                "fa3_hopper_v2_fwd: this rung is self-attention prefill, so S_q must equal S_kv "
                "(the causal diagonal's alignment is otherwise ambiguous); got ", seq_q, " vs ",
                seq_kv);
    TORCH_CHECK(seq_q % BLOCK_M == 0 && seq_kv % BLOCK_N == 0,
                "fa3_hopper_v2_fwd: seqlen must be a multiple of ", BLOCK_M,
                " at this rung. TMA zero-fills out-of-bounds rows, and a zero SCORE is not -inf — "
                "a KV tail would silently enter the softmax as a real key. got ", seq_q);
    TORCH_CHECK(kv_heads > 0 && heads % kv_heads == 0,
                "fa3_hopper_v2_fwd: query heads must be a multiple of KV heads (GQA); got ", heads,
                " and ", kv_heads);
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(Q.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(K.data_ptr()) % 16 == 0 &&
                    reinterpret_cast<std::uintptr_t>(V.data_ptr()) % 16 == 0,
                "fa3_hopper_v2_fwd: TMA requires a 16-byte-aligned global base address");

    const int64_t q_sb = Q.stride(0);
    const int64_t q_sh = q_heads_last ? Q.stride(2) : Q.stride(1);
    const int64_t q_ss = q_heads_last ? Q.stride(1) : Q.stride(2);
    const int64_t kv_sb = K.stride(0);
    const int64_t kv_sh = q_heads_last ? K.stride(2) : K.stride(1);
    const int64_t kv_ss = q_heads_last ? K.stride(1) : K.stride(2);
    TORCH_CHECK((q_ss * 2) % 16 == 0 && (q_sh * 2) % 16 == 0 && (q_sb * 2) % 16 == 0 &&
                    (kv_ss * 2) % 16 == 0 && (kv_sh * 2) % 16 == 0 && (kv_sb * 2) % 16 == 0,
                "fa3_hopper_v2_fwd: cuTensorMapEncodeTiled requires every non-innermost stride to "
                "be a multiple of 16 B");

    CUtensorMap tmap_q{}, tmap_k{}, tmap_v{};
    const CUresult rq = fa3_v2_encode_tensor_map(&tmap_q, Q.data_ptr(), HEAD_DIM, seq_q, heads,
                                                 batch, q_ss * 2, q_sh * 2, q_sb * 2, BLOCK_M);
    const CUresult rk = fa3_v2_encode_tensor_map(&tmap_k, K.data_ptr(), HEAD_DIM, seq_kv, kv_heads,
                                                 batch, kv_ss * 2, kv_sh * 2, kv_sb * 2, BLOCK_N);
    const CUresult rv = fa3_v2_encode_tensor_map(&tmap_v, V.data_ptr(), HEAD_DIM, seq_kv, kv_heads,
                                                 batch, kv_ss * 2, kv_sh * 2, kv_sb * 2, BLOCK_N);
    TORCH_CHECK(rq == CUDA_SUCCESS && rk == CUDA_SUCCESS && rv == CUDA_SUCCESS,
                "fa3_hopper_v2_fwd: cuTensorMapEncodeTiled failed (Q=", static_cast<int>(rq),
                " K=", static_cast<int>(rk), " V=", static_cast<int>(rv), "). The driver reports "
                "CUDA_ERROR_INVALID_VALUE (1) for every illegal field without naming one — run the "
                "CPU contract test tests/kernels/attention/test_k2_a_r2.py, which checks each "
                "field by name.");

    auto O = torch::empty_like(Q);
    auto LSE = torch::empty({batch, heads, seq_q}, Q.options().dtype(torch::kFloat32));
    // The epilogue writes O with Q's row stride, so the two must agree. empty_like preserves the
    // strides of a dense input, but "must" is cheaper to assert than to debug: a mismatch is a
    // silently scrambled output, not a fault.
    TORCH_CHECK(O.strides() == Q.strides(),
                "fa3_hopper_v2_fwd: O did not inherit Q's strides, so the epilogue's row stride is "
                "wrong for it");

    // 160 KB of staging is past the 48 KB a launch gets for free. ptxas will happily emit the
    // kernel without this call and the failure appears only as a launch error at runtime.
    const cudaError_t attr_err =
        cudaFuncSetAttribute(reinterpret_cast<const void*>(fa3_hopper_v2_fwd_kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_DYNAMIC_BYTES);
    TORCH_CHECK(attr_err == cudaSuccess, "fa3_hopper_v2_fwd: shared-memory opt-in failed: ",
                cudaGetErrorString(attr_err));

    // log2(e)/sqrt(d), folded once on the host: every exponential in the mainloop is then a bare
    // exp2 with no multiply in front of it, and the running max lives in the same basis.
    const float softmax_scale_log2 =
        1.4426950408889634f / std::sqrt(static_cast<float>(HEAD_DIM));

    const dim3 grid(static_cast<unsigned>(seq_q / BLOCK_M), static_cast<unsigned>(heads),
                    static_cast<unsigned>(batch));
    const dim3 block(NUM_THREADS);
    fa3_hopper_v2_fwd_kernel<<<grid, block, SMEM_DYNAMIC_BYTES>>>(
        tmap_q, tmap_k, tmap_v,
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr<at::BFloat16>()), LSE.data_ptr<float>(),
        static_cast<int>(seq_q), static_cast<int>(heads / kv_heads), q_sb, q_sh, q_ss, kv_sb, kv_sh,
        kv_ss, softmax_scale_log2, is_causal ? 1 : 0);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fa3_hopper_v2_fwd launch failed: ", cudaGetErrorString(err));
    return {O, LSE};
}
#endif  // TORCH_EXTENSION_NAME
