// =============================================================================
// h_r4_persistent_bf16_sm90.cu — K1/H-R4: persistent, cluster-of-2, TMA-multicast wgmma GEMM
// =============================================================================
//
// RUNG:   experiments/K1/H-R4/spec.md   ·   MAP: experiments/K1/H-R4/map.md
// FLOOR:  cuBLAS bf16, M=N=K=4096, H100 SXM.
// TARGET: >= 80% of cuBLAS, 90%+ wanted (plan SIXTY_DAYS_SIX_LADDERS.md §05 K1).
//
// The top Hopper rung. Four things arrive at once and each one is a different lesson:
//
//   PERSISTENT   grid = the SM count, not the problem shape. A CTA is launched once and walks a
//                strided sequence of output tiles, so the pipeline prologue (QSIZE stages of TMA
//                latency, ~2 us of nothing) is paid once per SM instead of once per tile: 132
//                prologues at 4096^3 instead of 512. The grid stops depending on M and N, which is
//                the only reason a tile scheduler can exist at all.
//   SCHEDULER    a struct with a next_tile() interface, so the ORDER tiles are visited in becomes
//                a variable. Grouped raster (GROUP_M x GROUP_N supertiles) walks a column of the
//                output before moving right, so the A-rows and B-cols a group touches stay in L2.
//   CLUSTER OF 2 two CTAs, launched together on adjacent SMs, sharing one distributed shared-memory
//                window. They split BM (rank_m 0 and 1) and share the same BN column strip.
//   MULTICAST    which is what makes the cluster pay: the B tile both CTAs need is fetched from
//                HBM ONCE (cp.async.bulk.tensor...multicast::cluster, issued by rank_m == 0 only)
//                and landed in both CTAs' shared memory. Halves B's HBM traffic; A is unshared and
//                still per-CTA.
//
// WHAT IS ALREADY HERE (agent-written; compiles, links, launches):
//   * both shared-memory matrix descriptors, including the MN-major one B needs (see below);
//   * the mbarrier / TMA / multicast / wgmma / setmaxnreg PTX wrappers;
//   * the host launcher: TMA tensor maps, the persistent grid, the smem opt-in, shape validation;
//   * the tile scheduler TYPE — its state, its constructor, and its next_tile() contract;
//   * the cluster launch, the cluster-wide barrier init, and the multicast mask arithmetic.
//
// WHAT IS HUY'S (the one hole): the persistent loop. That is the grouped-raster linear -> (m, n)
// mapping, the k-mainloop with wgmma kept in flight across k, and the epilogue. It is out-of-line
// in a __device__ function rather than inline in the kernel the way H-R1's is, for one reason:
// the scheduler's index arithmetic and the mainloop are one lesson and get ONE hole marker.
//
// WHY B's DESCRIPTOR IS NOT A's. torch hands this kernel A [M,K] and B [K,N], both row-major, and
// the ladder's floor is torch.matmul on exactly those. So A is K-contiguous (what wgmma wants: the
// K-major operand form, trans = 0) and B is N-contiguous (the MN-major form, trans = 1). TMA cannot
// transpose — its innermost box dimension must be the contiguous one — and transposing B on the
// host inside the timing loop would be a 67 MB round trip charged to this kernel's number. So B is
// loaded MN-major and read MN-major, which changes the descriptor's two offsets and nothing else.
// Derived from cute::make_gmma_desc (oss/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:225-260
// for Major::MN, :262-290 for Major::K) and cross-checked by re-deriving fast.cu's known-good
// K-major 16/1024 pair with the same method. hopper_contracts.wgmma_smem_descriptor holds the same
// numbers, and the CPU suite asserts the two agree — a swapped offset does not fault, it reads the
// wrong core matrix and produces a plausible, wrong C.
//
// BUILDING IT (no GPU required; infra/drydock.sh does this automatically):
//   nvcc -arch=sm_90a -cubin -O3 -Xptxas -v csrc/gemm/h_r4_persistent_bf16_sm90.cu -o /tmp/o.cubin
// While the hole is open that fails at the #error, by design. Add -DHUY_STUB_KERNEL_BODY=1 to
// verify the SCAFFOLDING compiles: it substitutes a trivially-correct, catastrophically slow
// reference body that walks the same scheduler and reads global memory directly. infra/bench.sh
// refuses to measure anything while that define (or LADDERS_STUB_HOLES=1) is in play.
// =============================================================================

#include <cstdint>
#include <cuda.h>        // CUtensorMap — the driver-API type the TMA parameters carry
#include <cuda_bf16.h>

#ifndef BM
#define BM 128        // output tile rows per CTA
#endif
#ifndef BN
#define BN 256        // output tile cols per CTA == the wgmma N (one wgmma.m64n256k16 covers it)
#endif
#ifndef BK
#define BK 64         // K advanced per mainloop step; also 128 B of bf16 = one 128B swizzle atom
#endif
#ifndef QSIZE
#define QSIZE 3       // pipeline stages. 3 x 48 KB of A+B = 144 KB, inside Hopper's 227 KB/CTA
#endif

// Cluster of 2 along M: the two CTAs own adjacent BM strips of the SAME BN column, so B is the
// tile they share and therefore the tile worth multicasting. CLUSTER_N is kept as a named 1 rather
// than folded away so the mask arithmetic below stays readable when a later variant flips it.
#define CLUSTER_M 2
#define CLUSTER_N 1
#define CLUSTER_SIZE (CLUSTER_M * CLUSTER_N)

// Grouped-raster supertile, in CLUSTER-tiles. Upstream instantiates
// Schedule<1, NUM_SM/CLUSTERS, BM*CLUSTER_M, BN*CLUSTER_N, 16/CLUSTER_M, 8/CLUSTER_N> at
// matmul_10.cuh:429 — an 8x8 supertile of 256x256 cluster-tiles. At 4096^3 the cluster-tile grid
// is 16x16, so exactly 2x2 supertiles: the group size is a real tuning knob at this shape, not a
// formality, and it is the only difference between upstream's kernel 10 and kernel 11.
#define GROUP_M 8
#define GROUP_N 8

#define NUM_CONSUMER_WG 2                              // warp-specialised: 1 producer + 2 consumers
#define NUM_THREADS ((NUM_CONSUMER_WG + 1) * 128)      // 384
#define B_WG_M (BM / NUM_CONSUMER_WG)                  // 64 output rows per consumer warpgroup

#define WGMMA_M 64                 // architectural: wgmma.m64nNk16 always has M = 64
#define WGMMA_N BN                 // one instruction spans the whole BN
#define WGMMA_K 16                 // architectural: k-strip width for 16-bit operands
#define ACC_M_STEPS (B_WG_M / WGMMA_M)   // 1 — each consumer warpgroup issues one wgmma per k-strip
#define ACC_REGS (WGMMA_N / 2)           // 128 fp32 accumulators per thread == d[WGMMA_N/16][8]

// setmaxnreg split. The producer is one thread issuing TMA descriptors and needs almost nothing;
// the consumers hold 128 fp32 accumulators each and need everything. 24 + 2x240 fits the 64 K
// registers per SM with room for the rest of the frame. hopper_contracts.check_setmaxnreg asserts
// both are in [24, 256] and multiples of 8 — an illegal immediate is a silently unspecialised
// kernel, which profiles as "mysteriously 60% of cuBLAS".
#define PRODUCER_REGS 24
#define CONSUMER_REGS 240

// Shared-memory matrix descriptor offsets, in bytes, per operand layout.
//   A is K-major   (torch [M,K] row-major, K contiguous)  -> LBO walks the two 8x8 core matrices
//                                                            along K (16 B), SBO walks 8 rows.
//   B is MN-major  (torch [K,N] row-major, N contiguous)  -> LBO walks between 64-element N chunks,
//                                                            which TMA laid out BK rows apart;
//                                                            SBO walks 8 K-rows (8 x 128 B).
// See the header note for the derivation and the cross-check.
#define A_DESC_LBO_BYTES 16
#define A_DESC_SBO_BYTES 1024
#define B_DESC_LBO_BYTES (BK * 128)
#define B_DESC_SBO_BYTES 1024
#define A_DESC_TRANS 0             // cute GMMA::Major::K  == 0
#define B_DESC_TRANS 1             // cute GMMA::Major::MN == 1 (mma_sm90_gmma.hpp:107-110)

// 128 B / sizeof(bf16). The TMA box's innermost dimension must be exactly one swizzle atom wide,
// so every tensor map here is 3-D with a 64-element inner dimension and the real extent carried in
// the outer one. check_tma_tensor_map() states the same rule in Python.
#define TMA_ATOM_ELEMS 64

// wgmma exists ONLY in the sm_90a accelerated ISA. Not base sm_90 (the trailing 'a' is load
// bearing), and NOT sm_100a/sm_120a: ptxas rejects it there outright. This guard makes the file
// inert, not broken, elsewhere — a multi-arch AOT build links, and the host launcher refuses.
#define SCRATCH_LLM_HAS_WGMMA (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900))

// -----------------------------------------------------------------------------
// Shared memory: QSIZE stages of A and B, plus the two mbarrier arrays.
// 1024-byte alignment, not 128: the 128B swizzle XORs address bits [7,10), so a tile base that is
// only 128-aligned starts at a nonzero swizzle phase and the descriptor (which declares phase 0)
// then disagrees with what TMA wrote. Every stage is a whole number of 1024-byte blocks so the
// per-stage bases inherit the alignment.
// -----------------------------------------------------------------------------
struct Stages {
    alignas(1024) __nv_bfloat16 A[QSIZE * BM * BK];
    alignas(1024) __nv_bfloat16 B[QSIZE * BK * BN];
    alignas(8) uint64_t full[QSIZE];    // producer -> consumers: this stage's bytes have landed
    alignas(8) uint64_t empty[QSIZE];   // consumers -> producer: this stage may be overwritten
};
static_assert((BM * BK * sizeof(__nv_bfloat16)) % 1024 == 0, "A stage breaks 1024 B alignment");
static_assert((BK * BN * sizeof(__nv_bfloat16)) % 1024 == 0, "B stage breaks 1024 B alignment");
static_assert(sizeof(Stages) < 227 * 1024, "smem over Hopper's per-CTA cap; drop QSIZE or BN");

// -----------------------------------------------------------------------------
// Tile scheduler. The TYPE is scaffolding; the MAPPING is the hole.
//
// State is deliberately tiny: a persistent CTA re-reads this every tile and it lives in the
// registers the consumers are already short of. `linear` is this cluster's next linear tile index
// and `stride` is the number of clusters in the grid, so cluster c visits c, c+stride, c+2*stride,
// ... — the same strided walk CUTLASS's PersistentTileSchedulerSm90 does by adding total_grid_size_
// (static_tile_scheduler.hpp:192-194, per the map).
//
// Contract next_tile() must honour, and the reason the ragged cases matter:
//   * a bijection onto [0, tiles_m) x [0, tiles_n) across all clusters — every tile visited once;
//   * return false once this cluster's indices run past tiles_m * tiles_n;
//   * tiles_m and tiles_n need NOT be multiples of GROUP_M/GROUP_N. Upstream asserts they are
//     (matmul_10.cuh:373) and so cannot run the skinny or untuned shapes at all; CUTLASS instead
//     rounds the problem up to the group and skips the overhang (tile_scheduler_params.h:143-144).
//     Take CUTLASS's side: at skinny16 the cluster-tile grid is 1x16, at untuned it is 6x24, and
//     neither divides 8.
// -----------------------------------------------------------------------------
struct GroupedRaster {
    int tiles_m;    // cluster-tiles down M  = ceil(M / (BM * CLUSTER_M))
    int tiles_n;    // cluster-tiles across N = ceil(N / (BN * CLUSTER_N))
    int linear;     // this cluster's next linear tile index
    int stride;     // clusters in the grid

    __device__ __forceinline__ GroupedRaster(int tm, int tn, int cluster_id, int num_clusters)
        : tiles_m(tm), tiles_n(tn), linear(cluster_id), stride(num_clusters) {}

    /// Next CLUSTER-tile for this cluster, or false when it is done. Coordinates are in
    /// cluster-tiles; the caller adds rank_m/rank_n to get its own CTA tile.
    __device__ bool next_tile(int &tile_m, int &tile_n);
};

#if SCRATCH_LLM_HAS_WGMMA

// -----------------------------------------------------------------------------
// 64-bit shared-memory matrix descriptor. PTX ISA, "Matrix Descriptor Format":
//   [ 0,14) start address   [16,30) leading byte offset   [32,46) stride byte offset
//   [49,52) base offset (swizzle phase)                   [62,64) layout type
// Each offset field is (x & 0x3FFFF) >> 4: mask 18 bits, drop the low 4. Dropping those 4 bits is
// the hardware asserting 16-byte alignment, not a rounding convenience.
// Cross-checked against hopper_contracts.wgmma_smem_descriptor (tests/kernels/gemm/test_k1_h_r4.py).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFFull) >> 4;
}

__device__ __forceinline__ uint64_t
make_smem_desc(const __nv_bfloat16 *smem_ptr, uint64_t lbo_bytes, uint64_t sbo_bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = matrix_descriptor_encode(static_cast<uint64_t>(addr));
    desc |= matrix_descriptor_encode(lbo_bytes) << 16;
    desc |= matrix_descriptor_encode(sbo_bytes) << 32;
    desc |= 1ull << 62;   // layout_type = 1 = 128B swizzle
    return desc;
}

/// A operand: torch [M,K] row-major, so K-contiguous, so the K-major descriptor form.
__device__ __forceinline__ uint64_t make_desc_a(const __nv_bfloat16 *p) {
    return make_smem_desc(p, A_DESC_LBO_BYTES, A_DESC_SBO_BYTES);
}

/// B operand: torch [K,N] row-major, so N-contiguous, so the MN-major descriptor form. Pair it
/// with B_DESC_TRANS in the wgmma or the instruction reads the tile as if it were K-major and
/// returns a wrong C without faulting.
__device__ __forceinline__ uint64_t make_desc_b(const __nv_bfloat16 *p) {
    return make_smem_desc(p, B_DESC_LBO_BYTES, B_DESC_SBO_BYTES);
}

// -----------------------------------------------------------------------------
// mbarrier: the producer/consumer handshake, cluster-wide.
// -----------------------------------------------------------------------------

/// `full` barriers are completed by TMA's byte count (thread_count 0, transaction_count 1);
/// `empty` barriers are completed by arrivals — NUM_CONSUMER_WG per CTA times CLUSTER_SIZE CTAs,
/// because with multicast a stage is only free once BOTH CTAs' consumers are done with it.
__device__ __forceinline__ void init_barrier(uint64_t *bar, int thread_count, int tx_count) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(p), "r"(thread_count + tx_count));
}

__device__ __forceinline__ void expect_bytes(uint64_t *bar, uint32_t bytes) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(p), "r"(bytes));
}

/// Spin on the parity bit. `phase` flips every time the queue index wraps QSIZE — that flip is the
/// whole of the pipeline's memory, and getting it out of step with the wrap is the classic
/// multistage hang.
__device__ __forceinline__ void barrier_wait(uint64_t *bar, int phase) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_H_R4:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra.uni DONE_H_R4;\n"
        "bra.uni WAIT_H_R4;\n"
        "DONE_H_R4:\n"
        "}\n"
        :: "r"(p), "r"(phase));
}

/// Arrive on a barrier that lives in ANOTHER CTA of the cluster. `mapa` translates this CTA's
/// shared address into the peer's window; without it a consumer would free only its own stage and
/// the producer on the peer would stall forever.
__device__ __forceinline__ void arrive_cluster(uint64_t *bar, uint32_t cta_rank, uint32_t count = 1) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .b32 remote;\n"
        "mapa.shared::cluster.u32 remote, %0, %1;\n"
        "mbarrier.arrive.shared::cluster.b64 _, [remote], %2;\n"
        "}\n"
        :: "r"(p), "r"(cta_rank), "r"(count));
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive;\n" ::: "memory");
    asm volatile("barrier.cluster.wait;\n" ::: "memory");
}

// -----------------------------------------------------------------------------
// TMA. Every map here is 3-D: {atom, rows, chunks} with a 64-element inner dimension, because a
// 2-D box caps the inner dimension at 256 elements and the 128B swizzle requires it to be exactly
// 64 bf16 wide. `chunk64` is the major coordinate already divided by TMA_ATOM_ELEMS.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void
tma_load_tile(__nv_bfloat16 *dst, const CUtensorMap *map, uint64_t *bar, int row, int chunk64) {
    uint64_t tma = reinterpret_cast<uint64_t>(map);
    uint32_t mb = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5}], [%2];\n"
        :: "r"(sd), "l"(tma), "r"(mb), "n"(0), "r"(row), "r"(chunk64)
        : "memory");
}

/// One HBM read, landed in every CTA whose bit is set in `mask`. Issued by exactly ONE CTA of the
/// cluster (the others must not issue it, or the tile is fetched CLUSTER_SIZE times and the whole
/// point is lost), but every CTA still runs expect_bytes on its own `full` barrier.
__device__ __forceinline__ void
tma_load_tile_multicast(__nv_bfloat16 *dst, const CUtensorMap *map, uint64_t *bar, int row,
                        int chunk64, uint16_t mask) {
    uint64_t tma = reinterpret_cast<uint64_t>(map);
    uint32_t mb = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t sd = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        ".multicast::cluster"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;\n"
        :: "r"(sd), "l"(tma), "r"(mb), "n"(0), "r"(row), "r"(chunk64), "h"(mask)
        : "memory");
}

// -----------------------------------------------------------------------------
// Warpgroup MMA and the register split.
// -----------------------------------------------------------------------------
template <uint32_t N>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N));
}
template <uint32_t N>
__device__ __forceinline__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N));
}

__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }

/// Wait until at most N committed groups are still in flight. N == 0 drains everything, which is
/// what every kernel in oss/fast.cu/h100 does (matmul_10.cuh:517/:540, matmul_11/12 hardcode 0).
/// N > 0 is this rung's own move and is not in the map — see the hole.
template <int N>
__device__ __forceinline__ void wgmma_wait() {
    static_assert(N >= 0 && N <= 7, "wgmma.wait_group takes 0..7");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

/// D = A x B + (ScaleD ? D : 0), bf16 operands from shared memory, fp32 accumulate.
/// ScaleD is a template parameter because the PTX immediate must be a compile-time constant: 0 on
/// the first k-strip of a tile overwrites the accumulator, which is how a persistent kernel starts
/// a NEW output tile without spending 128 registers' worth of stores zeroing it.
/// The trans immediates are A_DESC_TRANS / B_DESC_TRANS — see the header note on why they differ.
template <int ScaleD>
__device__ __forceinline__ void
wgmma_m64n256k16_bf16(float d[WGMMA_N / 16][8], uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7,"
        " %8, %9, %10, %11, %12, %13, %14, %15,"
        " %16, %17, %18, %19, %20, %21, %22, %23,"
        " %24, %25, %26, %27, %28, %29, %30, %31,"
        " %32, %33, %34, %35, %36, %37, %38, %39,"
        " %40, %41, %42, %43, %44, %45, %46, %47,"
        " %48, %49, %50, %51, %52, %53, %54, %55,"
        " %56, %57, %58, %59, %60, %61, %62, %63,"
        " %64, %65, %66, %67, %68, %69, %70, %71,"
        " %72, %73, %74, %75, %76, %77, %78, %79,"
        " %80, %81, %82, %83, %84, %85, %86, %87,"
        " %88, %89, %90, %91, %92, %93, %94, %95,"
        " %96, %97, %98, %99, %100, %101, %102, %103,"
        " %104, %105, %106, %107, %108, %109, %110, %111,"
        " %112, %113, %114, %115, %116, %117, %118, %119,"
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128, %129, %130, %131, %132, %133, %134;\n"
        :
          "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]),
          "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]), "+f"(d[5][0]), "+f"(d[5][1]),
          "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]),
          "+f"(d[6][6]), "+f"(d[6][7]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]),
          "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]), "+f"(d[8][0]), "+f"(d[8][1]),
          "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
          "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]),
          "+f"(d[9][6]), "+f"(d[9][7]), "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]),
          "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]), "+f"(d[11][0]), "+f"(d[11][1]),
          "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
          "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]),
          "+f"(d[12][6]), "+f"(d[12][7]), "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]),
          "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]), "+f"(d[14][0]), "+f"(d[14][1]),
          "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
          "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]),
          "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b), "n"(ScaleD), "n"(1), "n"(1),
          "n"(A_DESC_TRANS), "n"(B_DESC_TRANS));
}

// -----------------------------------------------------------------------------
// Everything the persistent body needs, in one struct so the hole's signature does not become an
// argument list nobody can read.
// -----------------------------------------------------------------------------
struct MainloopArgs {
    Stages *s;
    const CUtensorMap *tma_a;
    const CUtensorMap *tma_b;
    // Raw operands. The TMA path never dereferences these — they exist so the stub below can be
    // trivially correct without a pipeline, and so an M-remainder fallback has somewhere to read.
    const __nv_bfloat16 *A;
    const __nv_bfloat16 *B;
    float *C;
    int M, N, K;
    int wg_idx;          // 0 = producer, 1..NUM_CONSUMER_WG = consumers
    int tid;             // 0..127 within the warpgroup
    uint32_t rank_m;     // this CTA's row within the cluster
    uint32_t rank_n;     // this CTA's column within the cluster
    uint16_t mcast_b;    // multicast mask for the shared B tile
};

__device__ void h_r4_persistent_body(const MainloopArgs &a, GroupedRaster sched);

#if HUY_STUB_KERNEL_BODY
// ---- STUB (only under -DHUY_STUB_KERNEL_BODY=1) ---------------------------------------------
// Trivially correct, catastrophically slow. It exists so the scaffolding around the hole — both
// descriptors, the TMA maps, the cluster launch, the barriers, the launcher, the registration and
// the AOT build — can be compiled and smoke-tested while the hole is open. It touches neither
// shared memory nor wgmma nor the multicast, so it says nothing about any of them, and
// infra/bench.sh refuses to run while it is in play.

/// Plain row-major order over the ragged grid: a bijection, which is all the stub owes. It has no
/// L2 locality whatsoever — that is exactly what the real mapping is for.
__device__ bool GroupedRaster::next_tile(int &tile_m, int &tile_n) {
    const int total = tiles_m * tiles_n;
    if (linear >= total) return false;
    tile_m = linear / tiles_n;
    tile_n = linear % tiles_n;
    linear += stride;
    return true;
}

__device__ void h_r4_persistent_body(const MainloopArgs &a, GroupedRaster sched) {
    // One warpgroup does all the work; the other two fall through to the kernel's closing cluster
    // barrier. No setmaxnreg here: the stub's frame is nothing like the real one's, and a producer
    // warpgroup that dealloc'd to PRODUCER_REGS and then ran this would be undefined behaviour.
    if (a.wg_idx != 0) return;
    int tm, tn;
    while (sched.next_tile(tm, tn)) {
        const int tile_m = (tm * CLUSTER_M + static_cast<int>(a.rank_m)) * BM;
        const int tile_n = (tn * CLUSTER_N + static_cast<int>(a.rank_n)) * BN;
        for (int idx = a.tid; idx < BM * BN; idx += 128) {
            const int r = tile_m + idx / BN, c = tile_n + idx % BN;
            if (r >= a.M || c >= a.N) continue;
            float acc = 0.0f;
            for (int k = 0; k < a.K; ++k)
                acc += __bfloat162float(a.A[r * a.K + k]) * __bfloat162float(a.B[k * a.N + c]);
            a.C[r * a.N + c] = acc;
        }
    }
}
#else
// HUY: the persistent loop — grouped-raster tile mapping, k-mainloop with wgmma in flight, epilogue — spec: experiments/K1/H-R4/spec.md — fill before H-R4
//
// Two definitions go here, and they are one lesson, which is why they are one hole.
//
// 1. GroupedRaster::next_tile — the linear -> (tile_m, tile_n) mapping.
//    Upstream's arithmetic is matmul_10.cuh:376-386 (per the map): with a GROUP_M x GROUP_N
//    supertile, `cur_tile = linear / (GROUP_M*GROUP_N)` picks the supertile, `cur_tile_pos =
//    linear % (GROUP_M*GROUP_N)` picks the position inside it, and the supertile's origin is
//    (GROUP_M * (cur_tile / (tiles_n/GROUP_N)), GROUP_N * (cur_tile % (tiles_n/GROUP_N))).
//    Two things it does NOT handle and you must: tiles_n not divisible by GROUP_N (upstream
//    asserts it away at :373 — CUTLASS rounds up and skips the overhang, tile_scheduler_params.h
//    :143-144), and a partial last supertile. Advance `linear` by `stride`, return false past the
//    end. Whatever you write, it must stay a bijection: a mapping that visits a tile twice writes
//    a plausible C and only a full-shape oracle catches it.
//    Why it is worth the trouble: a plain row-major walk re-reads the whole B panel for every tile
//    row; an 8x8 group of 256x256 cluster-tiles touches 2048 rows of A and 2048 cols of B — about
//    8 MB of bf16 for 64 tiles, which is the size of an H100's L2. That number is the hypothesis.
//
// 2. h_r4_persistent_body — the producer and consumer loops.
//    Producer (wg_idx == 0): warpgroup_reg_dealloc<PRODUCER_REGS>(), then ONE thread walks the
//    scheduler; per tile, per k-step: barrier_wait(&empty[q], p) -> expect_bytes(&full[q],
//    (BM*BK + BK*BN)*2) -> tma_load_tile(A stage, tma_a, row = tile_m, chunk64 = k0/64) and, only
//    when rank_m == 0, tma_load_tile_multicast(B stage, tma_b, row = k0, chunk64 = tile_n/64,
//    a.mcast_b). Both loads complete the SAME `full` barrier: one barrier, two transactions, which
//    is why expect_bytes counts both tiles.
//    Consumers (wg_idx >= 1): warpgroup_reg_alloc<CONSUMER_REGS>(), seed all QSIZE `empty`
//    barriers once (tid < CLUSTER_SIZE arrive_cluster on each peer, or the producer's first wait
//    never clears), then per tile: float d[ACC_M_STEPS][WGMMA_N/16][8], first k-strip with
//    ScaleD = 0 (that is what starts a new tile without zeroing 128 registers), the rest with 1.
//    A's descriptor advances by WGMMA_K bf16 within a 64-K chunk and by BM*64 elements between
//    chunks; B's advances by WGMMA_K * TMA_ATOM_ELEMS elements per k-strip because its tile is
//    MN-major — reconciling those two against the 128B swizzle phase is the subtlest line here and
//    the one to byte-diff against cute::make_gmma_desc on the box.
//    ASYNC ACROSS K — the thing the plan row asks for and the map cannot give you: NOTHING in
//    oss/fast.cu/h100 keeps a wgmma group in flight (every call site passes wait_group 0;
//    matmul_11/12 hardcode it). So this line is yours to invent. The constraint is exactly one
//    sentence: a stage may be released to the producer only once the group that READ it has
//    retired, so with wgmma_wait<1>() you must release stage q-1, not stage q. Get that backwards
//    and the producer overwrites a stage a wgmma is still reading — which does not hang and does
//    not fault; it silently corrupts one k-strip out of K/16.
//    Epilogue: the m64nN fragment. Lane `tid` of a warpgroup owns rows warp*16 + lane/4 and +8,
//    columns 2*(lane%4) + {0,1} within each 16-wide group w, so d[m][w/16][{0,1,2,3}] maps to
//    (row, col+w), (row, col+w+1), (row+8, col+w), (row+8, col+w+1) and [4..7] to the same at
//    col+w+8 (upstream's ST macro, matmul_10.cuh:554-576). Consumer wg owns rows
//    wg_idx*B_WG_M .. +B_WG_M of the CTA tile. Write fp32 straight to C, predicated on r < M and
//    c < N — no smem staging, no TMA store; that is matmul_12's epilogue and a different rung's
//    lesson. Getting the fragment mapping wrong is visible as a structured 3/4-stale output: an
//    oracle test catches it, a benchmark does not.
//    Finally: every CTA of the cluster must leave this function. A cluster where one CTA returns
//    early while its peer still owes an arrive_cluster on the peer's smem is undefined behaviour,
//    and the kernel's closing cluster barrier only helps if you get there.
//
// The map (experiments/K1/H-R4/map.md) has the file:line for each of these upstream.
#error "HUY: K1/H-R4 grouped-raster mapping + persistent mainloop + epilogue — see the comment above, experiments/K1/H-R4/spec.md, and map.md. Compile the scaffolding with -DHUY_STUB_KERNEL_BODY=1."
#endif  // HUY_STUB_KERNEL_BODY

#endif  // SCRATCH_LLM_HAS_WGMMA

// =============================================================================
// Kernel: C[MxN] = A[MxK] . B[KxN],  bf16 operands, fp32 accumulate, fp32 out.
//   grid  = the SM count rounded down to a multiple of CLUSTER_SIZE — NOT the tile count
//   block = NUM_THREADS (1 producer + NUM_CONSUMER_WG consumer warpgroups)
//   cluster = CLUSTER_SIZE CTAs, declared at compile time so a plain <<<>>> launch carries it
// =============================================================================
extern "C" __global__ void __launch_bounds__(NUM_THREADS) __cluster_dims__(CLUSTER_SIZE, 1, 1)
h_r4_persistent_bf16_sm90(const __grid_constant__ CUtensorMap tma_a,
                          const __grid_constant__ CUtensorMap tma_b,
                          const __nv_bfloat16 *__restrict__ A,   // M x K row-major
                          const __nv_bfloat16 *__restrict__ B,   // K x N row-major
                          float *__restrict__ C,                 // M x N row-major
                          int M, int N, int K) {
#if !SCRATCH_LLM_HAS_WGMMA
    // Inert on every arch but sm_90a, so a multi-arch AOT build links. The host launcher checks
    // the device's compute capability and refuses before it can ever reach an empty kernel.
    (void)tma_a; (void)tma_b; (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#else
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    Stages &s = *reinterpret_cast<Stages *>(smem_raw);

    const int wg_idx = threadIdx.x / 128;
    const int tid = threadIdx.x % 128;

    if (threadIdx.x == 0) {
        for (int i = 0; i < QSIZE; ++i) {
            init_barrier(&s.full[i], 0, 1);
            init_barrier(&s.empty[i], 0, NUM_CONSUMER_WG * CLUSTER_SIZE);
        }
    }
    // Cluster-wide, not __syncthreads: the peer CTA will arrive on THESE barriers, so they must be
    // initialised before any CTA in the cluster is allowed past this point.
    cluster_sync();

    uint32_t cluster_id, cta_rank;
    asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(cluster_id));
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(cta_rank));
    const uint32_t rank_m = cta_rank / CLUSTER_N;
    const uint32_t rank_n = cta_rank % CLUSTER_N;

    // Multicast mask for B: every CTA sharing this CTA's N column, i.e. all CLUSTER_M ranks in
    // column rank_n. With CLUSTER_M=2, CLUSTER_N=1 this is 0b11 — both CTAs, one HBM read.
    uint16_t mcast_b = 0;
    #pragma unroll
    for (int i = 0; i < CLUSTER_M; ++i) mcast_b |= static_cast<uint16_t>(1u << (i * CLUSTER_N));
    mcast_b = static_cast<uint16_t>(mcast_b << rank_n);

    GroupedRaster sched((M + BM * CLUSTER_M - 1) / (BM * CLUSTER_M),
                        (N + BN * CLUSTER_N - 1) / (BN * CLUSTER_N),
                        static_cast<int>(cluster_id),
                        static_cast<int>(gridDim.x / CLUSTER_SIZE));

    MainloopArgs args{&s,     &tma_a, &tma_b, A,      B,      C,       M,
                      N,      K,      wg_idx, tid,    rank_m, rank_n,  mcast_b};
    h_r4_persistent_body(args, sched);

    // A CTA that retires while its peer can still mapa into its shared window is undefined
    // behaviour — the peer's mbarrier.arrive lands in memory that no longer belongs to it. Upstream
    // omits this; CUTLASS does not.
    cluster_sync();
#endif  // SCRATCH_LLM_HAS_WGMMA
}

// =============================================================================
// HOST LAUNCHER — what pybind.cpp binds and the Python wrapper calls.
// =============================================================================
#ifdef TORCH_EXTENSION_NAME
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream
#include <torch/extension.h>

namespace {

/// Every CUDA runtime call in the launcher goes through this. A silently-ignored
/// cudaFuncSetAttribute is the specific way a 144 KB kernel turns into a launch failure four
/// frames later, with a message about an invalid configuration and no mention of shared memory.
void cuda_check(cudaError_t e, const char *what) {
    TORCH_CHECK(e == cudaSuccess, "h_r4_persistent_bf16: ", what, " failed: ", cudaGetErrorString(e));
}

// One 3-D tensor map over a row-major [height, width] bf16 array: {64, height, width/64} with the
// inner dimension exactly one 128 B swizzle atom. `cuTensorMapEncodeTiled` takes the strides for
// dimensions 1..rank-1 only (the innermost is contiguous by definition), which is the argument
// convention hopper_contracts.check_tma_tensor_map models.
CUtensorMap make_map_3d(const void *ptr, int height, int width, int box_rows, int box_cols) {
    CUtensorMap map{};
    const uint64_t dims[3] = {TMA_ATOM_ELEMS, static_cast<uint64_t>(height),
                              static_cast<uint64_t>(width) / TMA_ATOM_ELEMS};
    const uint64_t strides[2] = {static_cast<uint64_t>(width) * sizeof(__nv_bfloat16),
                                 TMA_ATOM_ELEMS * sizeof(__nv_bfloat16)};
    const uint32_t box[3] = {TMA_ATOM_ELEMS, static_cast<uint32_t>(box_rows),
                             static_cast<uint32_t>(box_cols) / TMA_ATOM_ELEMS};
    const uint32_t elem_stride[3] = {1, 1, 1};
    CUresult r = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void *>(ptr), dims, strides, box,
        elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "h_r4: cuTensorMapEncodeTiled failed (", static_cast<int>(r),
                ") — check that the inner dimension is a multiple of 64 and the base is 16 B aligned");
    return map;
}

// A tensor map is a pure function of (base pointer, dims, box), so caching on those is exact, not
// a heuristic. It is here for measurement hygiene: building two maps costs a few microseconds of
// driver call per launch, and at 4096^3 the kernel is ~0.5 ms, so an uncached rebuild would spend
// roughly a percent of this rung's headline number encoding descriptors. Upstream caches for the
// same reason (matmul_12.cuh:604-606). Single-threaded, like every caller in this repo.
struct MapCache {
    const void *ptr = nullptr;
    int h = 0, w = 0, br = 0, bc = 0;
    CUtensorMap map{};

    const CUtensorMap &get(const void *p, int height, int width, int box_rows, int box_cols) {
        if (p != ptr || height != h || width != w || box_rows != br || box_cols != bc) {
            map = make_map_3d(p, height, width, box_rows, box_cols);
            ptr = p; h = height; w = width; br = box_rows; bc = box_cols;
        }
        return map;
    }
};

}  // namespace

torch::Tensor h_r4_persistent_bf16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "h_r4_persistent_bf16: A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
                "h_r4_persistent_bf16: A and B must be bfloat16 (wgmma.f32.bf16.bf16 operands); "
                "the K1 floor is cuBLAS bf16, so a float16 input would be measured against the wrong floor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "h_r4_persistent_bf16: A and B must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(0), "h_r4_persistent_bf16: inner dimensions disagree: ",
                A.size(1), " vs ", B.size(0));
    A = A.contiguous();
    B = B.contiguous();
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(K % BK == 0, "h_r4_persistent_bf16: K must be a multiple of ", BK,
                " at this rung (no K-remainder handling); got K=", K);
    // The TMA rank-3 trick, not a tile-shape preference: the innermost box dimension IS the 128 B
    // swizzle atom, so the array's contiguous extent must be a whole number of them. A is
    // K-contiguous (covered by the K % BK check above); B is N-contiguous, so N is the constraint.
    TORCH_CHECK(N % TMA_ATOM_ELEMS == 0,
                "h_r4_persistent_bf16: N must be a multiple of ", TMA_ATOM_ELEMS,
                " — the B tensor map's innermost box is one 128 B swizzle atom wide; got N=", N);

    const int dev = static_cast<int>(A.device().index());
    int cc_major = 0, cc_minor = 0, sm_count = 0;
    cuda_check(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev), "cc major");
    cuda_check(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev), "cc minor");
    TORCH_CHECK(cc_major == 9 && cc_minor == 0,
                "h_r4_persistent_bf16 needs sm_90a (wgmma + TMA multicast + clusters); this device "
                "is sm_", cc_major * 10 + cc_minor);
    cuda_check(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev), "SM count");

    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));

    static MapCache cache_a, cache_b;
    const CUtensorMap &map_a = cache_a.get(A.data_ptr(), M, K, BM, BK);
    const CUtensorMap &map_b = cache_b.get(B.data_ptr(), K, N, BK, BN);

    // 144 KB of stages is over the 48 KB that a launch gets without asking, so this opt-in is not
    // optional: without it the launch fails at runtime and ptxas never says a word (the stages are
    // dynamic shared memory, so its report shows 0 bytes smem for this kernel).
    constexpr int smem_bytes = static_cast<int>(sizeof(Stages));
    cuda_check(cudaFuncSetAttribute(h_r4_persistent_bf16_sm90,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes),
               "shared-memory opt-in");

    // THE PERSISTENT GRID. One CTA per SM, rounded down to a whole number of clusters, and
    // independent of M and N — that independence is the rung's thesis, not an implementation
    // detail: the launch stops quantizing against the machine and the tile scheduler becomes the
    // only thing that can. Extra clusters (a problem with fewer tiles than clusters) find
    // next_tile() false immediately and fall through to the closing cluster barrier.
    // If the measured occupancy on the box is not one CTA per SM, cudaOccupancyMaxActiveClusters
    // is the tighter answer and the one CUTLASS uses; check it before blaming the mainloop.
    const int grid = (sm_count / CLUSTER_SIZE) * CLUSTER_SIZE;
    TORCH_CHECK(grid > 0, "h_r4_persistent_bf16: device reports ", sm_count,
                " SMs, fewer than one cluster of ", CLUSTER_SIZE);

    h_r4_persistent_bf16_sm90<<<grid, NUM_THREADS, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        map_a, map_b,
        reinterpret_cast<const __nv_bfloat16 *>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(B.data_ptr<at::BFloat16>()),
        C.data_ptr<float>(), M, N, K);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "h_r4_persistent_bf16 launch failed: ", cudaGetErrorString(err));
    return C;
}
#endif  // TORCH_EXTENSION_NAME
