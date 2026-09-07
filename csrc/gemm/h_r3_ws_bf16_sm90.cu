// =============================================================================
// h_r3_ws_bf16_sm90.cu — K1/H-R3: warp-specialised, multistage TMA + wgmma pipeline (sm_90a)
// =============================================================================
//
// RUNG:   experiments/K1/H-R3/spec.md   ·   MAP: experiments/K1/H-R3/map.md
// FLOOR:  cuBLAS bf16, M=N=K=4096, H100 SXM.
// TARGET: 70-80% of cuBLAS (plan SIXTY_DAYS_SIX_LADDERS.md §05 K1).
//
// H-R1 was one warpgroup doing everything, single stage. H-R2 added TMA and two stages but kept
// every warp doing both jobs. This rung splits the CTA in two: warpgroup 0 does nothing but issue
// TMA copies into a 4-deep ring, warpgroups 1 and 2 do nothing but issue wgmma out of it, and the
// two sides meet only at a pair of mbarriers per stage. That is the whole idea — the copy engine
// and the tensor pipe stop taking turns inside one instruction stream — and `setmaxnreg` is what
// makes it pay: the producer hands back the registers it does not need (24) so each consumer can
// hold a 128-register fp32 accumulator (240).
//
// THE ONE DEPARTURE FROM fast.cu M7 (map.md, "Mechanism, in launch order"): M7's B lives in global
// memory as N x K (K-contiguous), so both its operands are K-major and both wgmma trans flags are
// 0 (m7:414). The K1 harness hands us B as [K, N] row-major — N-contiguous — and TMA cannot
// transpose. TMA's innermost box must be one 128 B swizzle atom, which for this B is 64
// *N*-elements, so the tile lands in shared memory MN-major and the B operand is issued with
// trans_b = 1 (cute::GMMA::Major::MN == 1, oss/cutlass/include/cute/arch/mma_sm90_gmma.hpp:107-110).
// That changes the descriptor's leading byte offset (B_LBO below) and nothing else. It is not a
// preference: for a row-major x row-major GEMM on Hopper exactly one operand is MN-major whichever
// way the two roles are assigned, because only one of A and B is K-contiguous.
//
// WHAT IS ALREADY HERE (agent-written; compiles, links, launches):
//   * both tensor maps, built host-side through the driver entry point (torch links cudart, not
//     libcuda), one-entry cached so an encode never sits inside the timing loop;
//   * the ring of 4 mbarrier pairs, their init, and the arrive / expect_tx / try_wait wrappers;
//   * the warpgroup role split and the two setmaxnreg immediates;
//   * the K-major and MN-major shared-memory descriptor builders, cross-checked against
//     scratch_llm.kernels.common.hopper_contracts by the CPU suite;
//   * the m64n256k16 wgmma asm with trans_b = 1, and fence / commit / wait;
//   * the host launcher, including the cudaFuncSetAttribute opt-in that 197632 B of dynamic shared
//     memory REQUIRES. H-R1 did not need it (32 KB, under the 48 KB threshold) and this rung does:
//     ptxas emits the kernel happily either way and the omission surfaces only as a launch failure
//     on the box, hours after the tile shape that caused it was chosen.
//
// WHAT IS HUY'S (the hole, below): the producer's TMA issue loop and the consumer's wgmma loop —
// the warp-specialised mainloop and its epilogue. That is where this rung's lessons live: which
// barrier each role waits on and which it releases, how the ring index and the phase bit walk
// together, how far ahead the producer is allowed to run, and how the m64n256 accumulator fragment
// maps back to a row-major fp32 C.
//
// BUILDING IT (no GPU required; infra/drydock.sh does this automatically):
//   nvcc -arch=sm_90a -cubin -O3 -Xptxas -v csrc/gemm/h_r3_ws_bf16_sm90.cu -o /tmp/o.cubin
// While the hole is open that fails at the #error, by design. Add -DHUY_STUB_KERNEL_BODY=1 to
// verify the SCAFFOLDING — tensor maps, barriers, descriptors, launcher, build wiring — against a
// trivially-correct, catastrophically slow reference body. infra/bench.sh refuses to measure
// anything while that define (or LADDERS_STUB_HOLES=1) is in play: a number from the stub would be
// a number about the stub.
//
// One thing the stub build does NOT prove: `wgmma_m64n256k16_bf16` is a template, and an
// uninstantiated template is never assembled, so the 135-operand asm block below goes unexercised
// until the hole is filled. It was checked once against a probe translation unit that instantiates
// it, which produced `HGMMA.64x256x16.F32.BF16 R24, gdesc[UR4].tnspB, ...` — note `.tnspB` and the
// absence of `.tnspA`, i.e. the trans immediates really landed as 0 and 1 — alongside `UTMALDG.5D`
// for the copy. The @pytest.mark.drydock tests re-assert both from the captured SASS the moment
// the hole closes.
// =============================================================================

#include <cstdint>
#include <cuda.h>          // CUtensorMap — the kernel's TMA descriptor parameter type
#include <cuda_bf16.h>

#ifndef BM
#define BM 128        // output tile rows per CTA, split 64/64 across the two consumer warpgroups
#endif
#ifndef BN
#define BN 256        // output tile cols per CTA == the wgmma N
#endif
#ifndef BK
#define BK 64         // K advanced per pipeline stage = 4 wgmma k-strips of 16
#endif
#ifndef STAGES
// The ring depth. 49152 B per stage: 4 fit under Hopper's 227 KB per-CTA cap (196608 B), 5 do not
// (245760 B). M7, M8 and M9 all ship 3 with no comment saying why (map.md, "One question the map
// cannot answer"), so this is a #define rather than a constant and `-DSTAGES=3` is the A/B.
#define STAGES 4
#endif
#ifndef PRODUCER_REGS
#define PRODUCER_REGS 24    // setmaxnreg.dec immediate for the producer warpgroup (m7:373)
#endif
#ifndef CONSUMER_REGS
#define CONSUMER_REGS 240   // setmaxnreg.inc immediate for each consumer warpgroup (m7:390)
#endif

#define NUM_CONSUMERS 2                            // 1 producer + 2 consumer warpgroups = 384 thr
#define NUM_THREADS   (128 * (NUM_CONSUMERS + 1))
#define WGMMA_M 64                                 // architectural: wgmma.m64nNk16 always has M=64
#define WGMMA_K 16                                 // architectural: k-strip width for 16-bit operands
#define WGMMA_N BN
#define WG_M    (BM / NUM_CONSUMERS)               // output rows one consumer warpgroup owns
#define M_STEPS (WG_M / WGMMA_M)                   // wgmma issues per k-strip, per consumer
#define ACC_TILES (WGMMA_N / 16)                   // accumulator fragments per thread: 16
#define ACC_REGS  (ACC_TILES * 8)                  // 128 fp32 registers of accumulator per thread

// TMA's innermost box must be exactly one 128 B swizzle atom, which in bf16 is 64 elements. Every
// shape constraint this rung puts on its caller comes from that one number — see the host
// launcher's TORCH_CHECK and hopper_contracts.check_tma_tensor_map.
#define TMA_INNER_ELEMS 64

#define BYTES_PER_STAGE ((BM * BK + BK * BN) * 2)  // 16384 B of A + 32768 B of B = 49152 B
// Dynamic shared memory is 128 B aligned by the driver, not 1024 B. Both fast.cu (m7:9-17) and
// CUTLASS (make_gmma_desc, mma_traits_sm90_gmma.hpp:218) hard-code the descriptor's base_offset to
// 0, which is only correct when the tile origin sits on a swizzle repeat — 8 rows of 128 B =
// 1024 B. So ask for 1024 B of slack and round the base up in the kernel. The cost is one KB; the
// alternative is a kernel that runs, is fast, and reads the wrong core matrix.
#define SMEM_ALIGN 1024
#define SMEM_BYTES (STAGES * BYTES_PER_STAGE + SMEM_ALIGN)

// Descriptor offsets, in BYTES, for the two shared-memory layouts TMA produces here. Both are
// re-derived in Python by scratch_llm.kernels.gemm.cuda.h_r3 and asserted equal to these #defines
// by the CPU suite, so a change to BK cannot silently desynchronise them.
//
// A tile, K-major: rows of BK = 64 bf16 = 128 B. LBO walks the two 8x8 core matrices along K
// (1 x uint128 = 16 B); SBO walks 8 core-matrix rows (8 x 128 B = 1024 B). Same pair as m7:13-14.
#define A_LBO 16
#define A_SBO 1024
// B tile, MN-major: TMA lays it down as [n_outer][k][n_inner] with n_inner = 64 elements = 128 B.
// In uint128 units the canonical MN layout is ((8,n),(8,k)):((1,LBO),(8,SBO))
// (mma_traits_sm90_gmma.hpp:229-234), so LBO is the stride between n_outer chunks — 64*BK elements
// = 128*BK bytes — and SBO is the stride across 8 k-rows — 8 x 128 B = 1024 B.
#define B_LBO (128 * BK)
#define B_SBO 1024

// wgmma exists ONLY in the sm_90a accelerated ISA. Not base sm_90 (the trailing 'a' is load
// bearing), and NOT sm_100a/sm_120a: ptxas rejects it there outright --
//   "Instruction 'wgmma.fence' not supported on .target 'sm_100a'"
// -- which is why the AOT build compiles each source for the archs its rung targets rather than
// for every arch in TORCH_CUDA_ARCH_LIST. This guard makes the file inert, not broken, elsewhere.
#define SCRATCH_LLM_HAS_WGMMA (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900))

#if SCRATCH_LLM_HAS_WGMMA

// -----------------------------------------------------------------------------
// 64-bit shared-memory matrix descriptor. PTX ISA, "Matrix Descriptor Format":
//   [ 0,14) start address   [16,30) leading byte offset   [32,46) stride byte offset
//   [49,52) base offset (swizzle phase)                   [62,64) layout type
// Each offset field is (x & 0x3FFFF) >> 4: mask 18 bits, drop the low 4. Dropping those 4 bits is
// the hardware asserting 16-byte alignment, not a rounding convenience.
// Cross-checked against hopper_contracts.wgmma_smem_descriptor (tests/kernels/gemm/test_k1_h_r3.py).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return (x & 0x3FFFFull) >> 4;
}

__device__ __forceinline__ uint64_t
make_smem_desc(const __nv_bfloat16* smem_ptr, uint64_t lbo, uint64_t sbo, int base_offset) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = matrix_descriptor_encode(static_cast<uint64_t>(addr));
    desc |= matrix_descriptor_encode(lbo)                 << 16;
    desc |= matrix_descriptor_encode(sbo)                 << 32;
    desc |= (static_cast<uint64_t>(base_offset) & 0x7ull) << 49;
    desc |= 1ull << 62;   // layout_type = 1 = 128B swizzle
    return desc;
}

// The only thing separating a K-major from an MN-major operand descriptor is the pair of offsets;
// the swizzle bit, the phase and the address field are identical. Two named wrappers rather than a
// bool parameter, because at the call site the question is "which layout is this operand in", and
// a bare `true` there is how the two get swapped without anything faulting.
__device__ __forceinline__ uint64_t make_a_desc(const __nv_bfloat16* p, int base_offset = 0) {
    return make_smem_desc(p, A_LBO, A_SBO, base_offset);
}
__device__ __forceinline__ uint64_t make_b_desc(const __nv_bfloat16* p, int base_offset = 0) {
    return make_smem_desc(p, B_LBO, B_SBO, base_offset);
}

// The async warpgroup MMA, SS form (both operands are shared-memory descriptors), bf16 in /
// fp32 accumulate. The trailing immediates are scale-a = 1, scale-b = 1, trans-a = 0, trans-b = 1.
// trans_b = 1 is THE departure from m7:414's `0, 0` and the reason this kernel is correct against a
// [K,N] row-major B; setting it back to 0 gives a kernel that runs, is fast, and is wrong.
// ScaleD is a template parameter because the PTX immediate must be a compile-time constant:
// 0 overwrites the accumulator, 1 accumulates into it.
template <int ScaleD>
__device__ __forceinline__ void
wgmma_m64n256k16_bf16(float d[ACC_TILES][8], uint64_t a_desc, uint64_t b_desc) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        "%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,"
        "%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,"
        "%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,"
        "%56,%57,%58,%59,%60,%61,%62,%63,"
        "%64,%65,%66,%67,%68,%69,%70,%71,"
        "%72,%73,%74,%75,%76,%77,%78,%79,"
        "%80,%81,%82,%83,%84,%85,%86,%87,"
        "%88,%89,%90,%91,%92,%93,%94,%95,"
        "%96,%97,%98,%99,%100,%101,%102,%103,"
        "%104,%105,%106,%107,%108,%109,%110,%111,"
        "%112,%113,%114,%115,%116,%117,%118,%119,"
        "%120,%121,%122,%123,%124,%125,%126,%127}, "
        "%128, %129, %130, 1, 1, 0, 1;\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
          "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
          "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
          "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
          "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
          "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
          "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
          "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
          "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(a_desc), "l"(b_desc), "n"(ScaleD));
}

__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait()   { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); }

// -----------------------------------------------------------------------------
// setmaxnreg — the instruction warp specialisation lives or dies on. The producer gives registers
// back so the consumers can take them; after the split the CTA holds
// PRODUCER_REGS*128 + CONSUMER_REGS*256 = 64512 of an SM's 65536, which is what makes one CTA of
// 384 threads resident at all. Both immediates are checked by hopper_contracts.check_setmaxnreg in
// the CPU suite (legal range 24..256, multiple of 8).
// -----------------------------------------------------------------------------
template <uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(RegCount));
}
template <uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(RegCount));
}

// -----------------------------------------------------------------------------
// mbarrier — the only thing the producer and the consumers ever say to each other.
//   full[s]   producer -> consumers: "stage s holds a complete A and B tile"
//   empty[s]  consumers -> producer: "stage s may be overwritten"
// The arrive count is what one phase flip costs: full[] flips when the TMA's expect_tx bytes have
// landed (1 arrival, m7:363), empty[] when every consumer has released it (NUM_CONSUMERS
// arrivals, m7:364).
// -----------------------------------------------------------------------------
__device__ __forceinline__ void init_barrier(uint64_t* bar, int arrive_count) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(p), "r"(arrive_count));
}

__device__ __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(p), "r"(bytes));
}

__device__ __forceinline__ void arrive_barrier(uint64_t* bar, uint32_t count = 1) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(p), "r"(count) : "memory");
}

// Spin until the barrier's parity matches `phase`. m7:285-299 spells this with PTX labels
// (LAB_WAIT / DONE) inside the asm block; this form pulls the predicate out through `selp` and
// leaves the loop to C++. It is CUTLASS's spelling, it emits no asm-local labels at all, and this
// kernel inlines the helper at both call sites — producer and consumer — so nothing has to stay
// unique across instantiations.
__device__ __forceinline__ void wait_barrier(uint64_t* bar, int phase) {
    uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t done = 0;
    do {
        asm volatile("{\n"
                     ".reg .pred P;\n"
                     "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2;\n"
                     "selp.b32 %0, 1, 0, P;\n"
                     "}\n"
                     : "=r"(done) : "r"(p), "r"(phase) : "memory");
    } while (done == 0);
}

// One TMA tile copy. The 5-D coordinate order follows the tensor map built on the host:
//   A map dims {64 k-elems, M, K/64}  ->  coords {0, m_origin, k_origin/64}
//   B map dims {64 n-elems, K, N/64}  ->  coords {0, k_origin, n_origin/64}
// They are NOT in the same order as m7:384-385, whose B is stored [N,K] and so takes {0, n0,
// k0/64}. Copying m7's call sites verbatim onto this B produces a plausible, wrong C.
// `shared::cluster` is the only destination-space spelling this instruction has even with no
// cluster; m7:274 notes the narrower `.cta` scope needs PTX 8.6.
__device__ __forceinline__ void
load_async_tile(__nv_bfloat16* dst, const CUtensorMap* map, uint64_t* bar, int coord1, int coord2) {
    uint64_t map_ptr  = reinterpret_cast<uint64_t>(map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, 0, 0}], [%2];"
        :: "r"(dst_ptr), "l"(map_ptr), "r"(mbar_ptr), "n"(0), "r"(coord1), "r"(coord2)
        : "memory");
}
#endif  // SCRATCH_LLM_HAS_WGMMA

// =============================================================================
// Kernel: C[MxN] = A[MxK] . B[KxN],  bf16 operands, fp32 accumulate, fp32 out.
//   grid = (ceil(N/BN), ceil(M/BM))   block = 384 threads (1 producer + 2 consumer warpgroups)
//
// A and B arrive twice: as tensor maps, which is what the mainloop uses, and as raw pointers,
// which only the -DHUY_STUB_KERNEL_BODY reference body reads (it does not go through TMA at all).
// The real mainloop must not touch the raw pointers — every byte it reads comes from shared memory.
// =============================================================================
extern "C" __global__ void __launch_bounds__(NUM_THREADS, 1)
h_r3_ws_bf16_sm90(const __nv_bfloat16* __restrict__ A,   // M x K row-major (stub path only)
                  const __nv_bfloat16* __restrict__ B,   // K x N row-major (stub path only)
                  float* __restrict__ C,                 // M x N row-major
                  int M, int N, int K,
                  const __grid_constant__ CUtensorMap tensorMapA,
                  const __grid_constant__ CUtensorMap tensorMapB) {
#if !SCRATCH_LLM_HAS_WGMMA
    // Inert on every arch but sm_90a, so a multi-arch AOT build links. The host launcher checks
    // the device's compute capability and refuses before it can ever reach an empty kernel.
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K; (void)tensorMapA; (void)tensorMapB;
#else
    const int tileM = blockIdx.y * BM;          // this CTA's output row origin
    const int tileN = blockIdx.x * BN;          // this CTA's output col origin
    const int wg_idx = threadIdx.x / 128;       // 0 = producer, 1..NUM_CONSUMERS = consumers
    const int tid    = threadIdx.x % 128;       // lane within the warpgroup
    const bool is_producer = (wg_idx == 0);

    // Round the dynamic shared-memory base up to a swizzle repeat (see SMEM_ALIGN). Adding a byte
    // offset to the generic pointer moves the shared address by the same amount, so this is the
    // whole fix. Every stage offset below is then a multiple of 1024 from here, which is what lets
    // every descriptor carry base_offset = 0.
    //
    // The declared alignment is 16, NOT SMEM_ALIGN, and that is load bearing: declaring
    // `__align__(1024)` here is a promise to the compiler, and nvcc believes it — with 1024 the
    // mask below constant-folds to zero and disappears from the PTX (verified 2026-09-07: `and.b32
    // %r3, %r2, 1023` is emitted at __align__(16) and absent at __align__(1024)). The declaration
    // does not make the driver align anything; it only makes the check that would have caught the
    // driver vanish.
    extern __shared__ __align__(16) uint8_t smem_raw[];
    const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    __nv_bfloat16* const sA = reinterpret_cast<__nv_bfloat16*>(
        smem_raw + ((SMEM_ALIGN - (smem_base & (SMEM_ALIGN - 1))) & (SMEM_ALIGN - 1)));
    __nv_bfloat16* const sB = sA + STAGES * BM * BK;

    // The barriers are static shared memory, deliberately outside the dynamic SMEM_BYTES: they are
    // 8-byte objects and folding them into the staged region would push every stage boundary off
    // the 1024 B swizzle repeat. 2 x STAGES x 8 B = 64 B, far under the 48 KB static cap that
    // forces the staging buffers themselves to be dynamic in the first place.
    __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];

    if (threadIdx.x == 0) {
        for (int i = 0; i < STAGES; ++i) {
            init_barrier(&full[i], 1);              // one arrival: the TMA's expect_tx completion
            init_barrier(&empty[i], NUM_CONSUMERS); // one arrival per consumer warpgroup
        }
    }
    __syncthreads();

    // The register split. Hoisted above the mainloop, and above the accumulator, on purpose:
    // `setmaxnreg` is warpgroup-uniform and must be reached by all 128 threads of its warpgroup,
    // and nothing register-hungry may be live across the `.dec`. The accumulator is therefore
    // declared INSIDE the consumer branch (m7:392) rather than hoisted the way H-R1 hoists it —
    // 128 live fp32 across a `.dec 24` is exactly what the instruction forbids.
    if (is_producer) warpgroup_reg_dealloc<PRODUCER_REGS>();
    else             warpgroup_reg_alloc<CONSUMER_REGS>();

#if HUY_STUB_KERNEL_BODY
    // ---- STUB (only under -DHUY_STUB_KERNEL_BODY=1) -------------------------------------------
    // Trivially correct, catastrophically slow: one output element per consumer thread, fp32,
    // straight from global memory. It exists so the scaffolding around the hole -- tensor maps,
    // barriers, descriptors, launcher, registration, the AOT build -- can be compiled and
    // smoke-tested while the hole is open. It runs in the consumer warpgroups only, because the
    // producer has just handed its registers away and a scalar k-loop does not fit in 24 of them.
    // It touches neither the ring nor wgmma, so it says nothing about either.
    // infra/bench.sh refuses to run while this is in play.
    if (!is_producer) {
        for (int idx = threadIdx.x - 128; idx < BM * BN; idx += 128 * NUM_CONSUMERS) {
            const int r = tileM + idx / BN, c = tileN + idx % BN;
            if (r >= M || c >= N) continue;
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += __bfloat162float(A[r * K + k]) * __bfloat162float(B[k * N + c]);
            C[r * N + c] = acc;
        }
    }
    (void)sA; (void)sB; (void)tid; (void)tensorMapA; (void)tensorMapB;
#else
    // HUY: the warp-specialised mainloop — producer TMA issue loop, consumer wgmma loop, epilogue — spec: experiments/K1/H-R3/spec.md — fill before H-R3
    //
    // PRODUCER (wg_idx == 0). Only lane 0 issues; the other 127 threads of the warpgroup have
    // nothing to do, and that is the design rather than waste (m7:375). For each of the K/BK steps,
    // with a ring index `qidx` walking 0..STAGES-1 and a phase bit `p` flipped on every wrap
    // (m7:380-381):
    //   1. wait_barrier(&empty[qidx], p)                  — the stage is free again (m7:382)
    //   2. expect_bytes(&full[qidx], BYTES_PER_STAGE)     — 49152 B are on their way (m7:383)
    //   3. load_async_tile(&sA[qidx*BM*BK], &tensorMapA, &full[qidx], tileM, k0/TMA_INNER_ELEMS)
    //      load_async_tile(&sB[qidx*BK*BN], &tensorMapB, &full[qidx], k0, tileN/TMA_INNER_ELEMS)
    //      — the coordinate orders differ between the two maps; see load_async_tile's comment.
    //      Both copies complete into the SAME full[qidx], which is why one expect_bytes covers the
    //      pair and why its byte count is the sum.
    //
    // CONSUMER (wg_idx 1..NUM_CONSUMERS). The accumulator is declared here, after the .inc:
    //     float d[ACC_TILES][8];                          // 128 fp32 = the register budget
    // and this warpgroup owns rows [(wg_idx-1)*WG_M, +WG_M) of the tile.
    //   1. Before the k-loop, release all STAGES empty barriers once from lane 0 (m7:394-396).
    //      That is what lets the producer run a full ring ahead instead of lock-stepping with the
    //      first MMA, and it is the single line that turns 4 stages into 4 stages of latency hiding.
    //   2. Zero the accumulator, then per k-step: wait_barrier(&full[qidx], p) (m7:404),
    //      wgmma_fence() (m7:405) — the fence is what orders the zero-init and the previous
    //      iteration's reads before the async MMA reads them — then for each of the
    //      BK/WGMMA_K = 4 k-strips issue
    //      wgmma_m64n256k16_bf16<1>(d, make_a_desc(...), make_b_desc(...)).
    //      The A descriptor advances WGMMA_K bf16 = 32 B per strip, inside one 128 B row, so its
    //      swizzle phase does not move and base_offset stays 0.
    //      The B tile is [n_outer][k][n_inner] — that layout is exactly what B_LBO encodes — so one
    //      k-row is TMA_INNER_ELEMS = 64 n-elements = 128 B, and a k-strip advance is 16 rows x
    //      128 B = 2048 B *inside n_outer chunk 0*. It is NOT WGMMA_K * BN * 2 = 8192 B: that is
    //      the stride to chunk 1, and reaching chunks 1..3 is the descriptor's LBO's job, not the
    //      start address's. Advancing the start address by 8192 lands on chunk 1's k=0 and
    //      computes a plausible, wrong C. Both 2048 and 8192 are whole numbers of the 1024 B
    //      swizzle repeat, so neither moves the phase. Byte-diff both descriptors against
    //      cute::make_gmma_desc on the box before trusting them.
    //   3. wgmma_commit(); wgmma_wait<0>(); then lane 0 arrive_barrier(&empty[qidx]) (m7:420-422).
    //      Waiting on 0 releases the stage as late as it can be released; CUTLASS instead keeps one
    //      MMA in flight and frees a lagging stage (coll:264, coll:547, coll:477) — that is H-R4's
    //      move, not this rung's.
    //
    // EPILOGUE. m7:425-448 stores to a COLUMN-major bf16 C; this rung's C is ROW-major fp32, so
    // the index arithmetic there is not copyable. The m64nN fragment itself is the same:
    //     lane = tid % 32, warp = tid / 32, row = warp*16 + lane/4
    //     for w in 0..WGMMA_N step 16:  col = w + 2*(tid % 4)
    //       d[w/16][0..7] -> (row, col) (row, col+1) (row+8, col) (row+8, col+1)
    //                        (row, col+8) (row, col+9) (row+8, col+8) (row+8, col+9)
    //   plus (wg_idx-1)*WG_M on the row, plus the tile origins, predicated on r < M and c < N.
    //   Getting this wrong leaves a structured subset of C stale — the oracle test catches that,
    //   the benchmark does not.
    //
    // The map (experiments/K1/H-R3/map.md) has the file:line for each of these upstream.
    #error "HUY: K1/H-R3 warp-specialised mainloop + epilogue — see the comment above, experiments/K1/H-R3/spec.md, and map.md. Compile the scaffolding with -DHUY_STUB_KERNEL_BODY=1."
#endif  // HUY_STUB_KERNEL_BODY
#endif  // SCRATCH_LLM_HAS_WGMMA
}

// =============================================================================
// HOST LAUNCHER — what pybind.cpp binds and the Python wrapper calls.
// =============================================================================
#ifdef TORCH_EXTENSION_NAME
#include <cudaTypedefs.h>
#include <torch/extension.h>

namespace {

// cuTensorMapEncodeTiled is a CUDA *driver* entry point. torch's JIT and AOT builds link cudart,
// not libcuda, so calling it directly is an `undefined symbol` when the extension loads — on the
// box, at the first import of this rung. Resolve it through cudart instead, exactly as CUTLASS
// does (oss/cutlass/include/cutlass/cuda_host_adapter.hpp:109-147, whose ifdef is here because the
// query gained a version argument after CUDA 12).
PFN_cuTensorMapEncodeTiled tensor_map_encoder() {
    void* pfn = nullptr;
    cudaDriverEntryPointQueryResult q;
#if (__CUDACC_VER_MAJOR__ > 12)
    const cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &pfn, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t err = cudaGetDriverEntryPoint(
        "cuTensorMapEncodeTiled", &pfn, cudaEnableDefault, &q);
#endif
    TORCH_CHECK(err == cudaSuccess && q == cudaDriverEntryPointSuccess,
                "h_r3_ws_bf16: could not resolve cuTensorMapEncodeTiled through cudart "
                "(driver older than the CUDA 12 TMA API?)");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(pfn);
}

// One tensor map for a `rows x cols` row-major bf16 matrix, tiled `box_rows x box_cols`. The 5-D
// shape is forced by the 128 B swizzle: the innermost box must be exactly one atom (64 bf16), so
// the contiguous extent is split into 64-element chunks and the chunk index becomes a third rank
// (m7:38-48). globalStrides holds dimensions 1..rank-1 only — the innermost is implicitly
// contiguous — which is why the array below starts at the row stride.
//   A: rows=M, cols=K, box BM x BK  -> dims {64, M, K/64}, box {64, 128, 1}
//   B: rows=K, cols=N, box BK x BN  -> dims {64, K, N/64}, box {64,  64, 4}
CUtensorMap make_operand_map(const __nv_bfloat16* ptr, int rows, int cols,
                             int box_rows, int box_cols) {
    CUtensorMap map{};
    uint64_t gdim[5]    = {TMA_INNER_ELEMS, static_cast<uint64_t>(rows),
                           static_cast<uint64_t>(cols) / TMA_INNER_ELEMS, 1, 1};
    uint64_t gstride[4] = {sizeof(__nv_bfloat16) * static_cast<uint64_t>(cols),
                           sizeof(__nv_bfloat16) * TMA_INNER_ELEMS, 0, 0};
    uint32_t bdim[5]    = {TMA_INNER_ELEMS, static_cast<uint32_t>(box_rows),
                           static_cast<uint32_t>(box_cols) / TMA_INNER_ELEMS, 1, 1};
    uint32_t bstride[5] = {1, 1, 1, 1, 1};
    const CUresult r = tensor_map_encoder()(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5,
        const_cast<void*>(static_cast<const void*>(ptr)),
        gdim, gstride, bdim, bstride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS,
                "h_r3_ws_bf16: cuTensorMapEncodeTiled failed (", static_cast<int>(r),
                ") for a ", rows, "x", cols, " matrix with a ", box_rows, "x", box_cols,
                " box. The driver returns one code for every violated constraint; "
                "hopper_contracts.check_tma_tensor_map names which one, on the CPU.");
    return map;
}

// Encoding a tensor map is a driver call, and k1_ladder.py times `kernel(a, b)` in a loop — an
// unconditional encode here would be measured as part of the kernel. One cached entry per operand,
// keyed on the pointer and the shape, is all this rung needs (m7:462-470 caches on M alone).
struct MapCache {
    const void* ptr = nullptr;
    int rows = 0, cols = 0, box_rows = 0, box_cols = 0;
    CUtensorMap map{};

    const CUtensorMap& get(const __nv_bfloat16* p, int r, int c, int br, int bc) {
        if (p != ptr || r != rows || c != cols || br != box_rows || bc != box_cols) {
            map = make_operand_map(p, r, c, br, bc);
            ptr = p; rows = r; cols = c; box_rows = br; box_cols = bc;
        }
        return map;
    }
};

}  // namespace

torch::Tensor h_r3_ws_bf16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "h_r3_ws_bf16: A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16,
                "h_r3_ws_bf16: A and B must be bfloat16 (wgmma.f32.bf16.bf16 operands); "
                "the K1 floor is cuBLAS bf16, so a float16 input would be measured against the wrong floor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "h_r3_ws_bf16: A and B must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(0), "h_r3_ws_bf16: inner dimensions disagree: ",
                A.size(1), " vs ", B.size(0));
    A = A.contiguous();
    B = B.contiguous();
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(K % TMA_INNER_ELEMS == 0 && N % TMA_INNER_ELEMS == 0,
                "h_r3_ws_bf16: the 128 B swizzle forces each tensor map's innermost box to be ",
                TMA_INNER_ELEMS, " bf16, so the contiguous extent of each operand must be a "
                "multiple of ", TMA_INNER_ELEMS, "; got K=", K, ", N=", N,
                ". M is free — TMA zero-fills out-of-bounds rows.");
    TORCH_CHECK(K % BK == 0, "h_r3_ws_bf16: K must be a multiple of ", BK,
                " at this rung (no K-remainder handling); got K=", K);

    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
    const auto* a_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>());
    const auto* b_ptr = reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>());

    static MapCache a_cache, b_cache;
    const CUtensorMap& map_a = a_cache.get(a_ptr, M, K, BM, BK);
    const CUtensorMap& map_b = b_cache.get(b_ptr, K, N, BK, BN);

    // The opt-in H-R1 did not need. Above 48 KB a kernel's dynamic shared memory must be granted
    // explicitly or the LAUNCH fails — ptxas compiles it either way, so the omission is a runtime
    // error hours downstream of the tile choice that caused it. Once per process, not per call:
    // this function body sits inside k1_ladder.py's timing loop.
    static const bool opted_in = [] {
        const cudaError_t e = cudaFuncSetAttribute(h_r3_ws_bf16_sm90,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   SMEM_BYTES);
        TORCH_CHECK(e == cudaSuccess, "h_r3_ws_bf16: cudaFuncSetAttribute(", SMEM_BYTES,
                    " B dynamic smem) failed: ", cudaGetErrorString(e));
        return true;
    }();
    (void)opted_in;

    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(NUM_THREADS);
    h_r3_ws_bf16_sm90<<<grid, block, SMEM_BYTES>>>(
        a_ptr, b_ptr, C.data_ptr<float>(), M, N, K, map_a, map_b);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "h_r3_ws_bf16 launch failed: ", cudaGetErrorString(err));
    return C;
}
#endif  // TORCH_EXTENSION_NAME
