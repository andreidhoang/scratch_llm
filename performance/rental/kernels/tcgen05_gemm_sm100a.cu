// ============================================================================
// A3 §4.2 — tcgen05 / UMMA GEMM, accumulator in TENSOR MEMORY (TMEM)
// Datacenter Blackwell, sm_100a (B200). 1-SM (cta_group::1) path.
// ============================================================================
//
// EXACT COMPILE COMMAND (the gate for this artifact):
//   nvcc -arch=sm_100a -ptx \
//     performance/rental/kernels/tcgen05_gemm_sm100a.cu -o /tmp/tc.ptx
//   # exit 0 ; then:  grep -c tcgen05 /tmp/tc.ptx   (must be > 0)
// A stronger check (ptxas actually encodes the tcgen05 ops), also run here:
//   nvcc -arch=sm_100a -cubin performance/rental/kernels/tcgen05_gemm_sm100a.cu -o /tmp/tc.cubin
//
// REFERENCE FOLLOWED (structural, not a copy):
//   - gau-nernst, "tcgen05 GEMM to ~98% of B200 cuBLAS"  (gau-nernst.github.io/tcgen05/)
//   - Colfax CUTLASS tutorials Part 3–4 (UMMA + TMEM + block-scaling)
//   - NVIDIA PTX ISA §9.7.15/§9.7.16 — tcgen05.{alloc,mma,commit,ld,wait,fence}
//     and Tensor Memory; matrix-descriptor encoding cross-checked against this
//     repo's artifact  performance/artifacts/wgmma_descriptor_manual.md
//     (the 64-bit SMEM descriptor — LBO/SBO >>4, swizzle in bits[62:64) — is
//      shared by wgmma.mma_async and tcgen05.mma A/B operands).
//
// PRE-REGISTERED TARGET (PERF_ENGINEERING_SPEC.md §4 · A3, B200 dense BF16 = 2,250 TF/s):
//   ~1,209 TF/s (~54% of dense) for the 1-SM path EARLY; warp-spec climbs to
//   ~1,300–1,476 TF/s; the 2-SM cta_group::2 win is an HONEST ~8% (1209->1302,
//   SMEM-bandwidth relief, not raw math). This file is the 1-SM rung only.
//
//   !!!  RUNTIME CORRECTNESS DEFERRED TO THE RENTAL DAY  !!!
//   This COMPILES here on the sm_120 dev box (nvcc targets sm_100a) but CANNOT
//   RUN here: sm_120 consumer Blackwell has NO tcgen05, NO TMEM, NO cta_group.
//   It runs on a B200 (sm_100a). No claim of numerical correctness or TF/s is
//   made from this box — only "compiles for sm_100a + PTX contains tcgen05 +
//   structurally matches the gau-nernst / PTX-ISA sequence."
//
// STRUCTURAL SPINE (register -> TMEM ; single-thread issue ; async-by-mbarrier):
//   1. tcgen05.alloc  a TMEM accumulator (NCOL cols, pow2 >=32, ONE warp).
//   2. TMA (cp.async.bulk.tensor) stages A/B K-strips into SMEM, mbarrier-tracked.
//   3. ONE thread issues tcgen05.mma.cta_group::1.kind::f16 [d-tmem],a-desc,b-desc,
//      idesc, enable-input-d  — accumulates in TMEM across the K loop.
//   4. tcgen05.commit arrives on an mbarrier (completion is NOT wait_group).
//   5. FULL-WARPGROUP (128-thread) drain: tcgen05.ld -> regs -> epilogue. The
//      32-lane-per-warp rule (§2.3): one warp reads only 32 of TMEM's 128 lanes,
//      so a single-warp drain leaves 3/4 of the 128-row tile STALE. All 4 warps
//      must drain, each addressing its own lane-quadrant. (See drain_tmem below.)
// ============================================================================

#include <cuda.h>          // CUtensorMap for the TMA descriptor param
#include <cstdint>
#include <cuda_fp16.h>

// ---- tile geometry: the largest tcgen05 f16 atom, 128 x 256 x 16 -----------
#define MMA_M 128
#define MMA_N 256
#define MMA_K 16
#define NCOL  256          // f32 accumulator: 128 lanes x 256 columns of TMEM
                           // (pow2, >=32 — satisfies the tcgen05.alloc contract)

// ---- small helpers: generic->shared address, mbarrier ----------------------
__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count));
}

// Arm the mbarrier for a TMA of `bytes` bytes, then launch the bulk-tensor copy.
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
  asm volatile(
    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(bar), "r"(bytes));
}

// Spin on the mbarrier phase (producer/consumer glue: releases only when the
// byte-transaction count AND arrivals are both satisfied — book §7.3.4).
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t phase) {
  asm volatile(
    "{\n"
    ".reg .pred P;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
    "@P bra DONE_WAIT;\n"
    "bra LAB_WAIT;\n"
    "DONE_WAIT:\n"
    "}\n"
    :: "r"(bar), "r"(phase));
}

// ---- TMA: 2D bulk-tensor global->SMEM, completion tracked on the mbarrier ---
// (cp.async.bulk.tensor is the Hopper/Blackwell TMA engine — replaces cp.async
//  for large tiles; the tensor-map is built host-side via cuTensorMapEncode.)
__device__ __forceinline__ void tma_load_2d(
    void *smem_dst, const CUtensorMap *tmap, uint32_t bar, int c0, int c1) {
  uint32_t dst = smem_u32(smem_dst);
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3}], [%4];\n"
    :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

// ---- 64-bit SMEM matrix descriptor (shared with WGMMA; see repo artifact) ---
// encode(x) = (x >> 4) & 0x3FFF   (the 16-byte-unit ">>4" encoding, §2 of the
// artifact). Fields: [0:14) start>>4, [16:30) LBO>>4, [32:46) SBO>>4,
// [62:64) swizzle {0 none,1 128B,2 64B,3 32B}. Worked 64-wide f16/128B tile in
// the artifact: LBO=16B->1, SBO=1024B->64, swizzle=1.
__device__ __forceinline__ uint64_t make_smem_desc(
    const void *smem_ptr, uint32_t lbo_bytes, uint32_t sbo_bytes, uint32_t swizzle) {
  uint32_t addr = smem_u32(smem_ptr);
  uint64_t d = 0;
  d |= (uint64_t)((addr      >> 4) & 0x3FFF) << 0;
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)(swizzle & 0x3)             << 62;
  return d;
}

// ---- 32-bit tcgen05 instruction descriptor (idesc) -------------------------
// Packs A/B dtype, transpose, sparsity, and (for block-scaled kinds) scale-IDs.
// [INFERENCE, reference-annotated] the exact bit layout of idesc is documented
// in the PTX ISA tcgen05.mma section / built by CUTLASS UMMA::MMA traits; here
// we pass a kind::f16, dense, non-transposed, no-scale descriptor. The precise
// field encoding is VALIDATED ON THE B200 against CUTLASS make_umma_desc — this
// constant is the placeholder the rental day replaces with the CuTe-built value.
__device__ __forceinline__ uint32_t make_idesc_f16_dense() {
  // dense, kind=f16, no transpose, no block-scaling. Structure-only constant;
  // rental day swaps for cutlass::gemm::collective UMMA idesc.
  return 0u;
}

// ---- (1) TMEM allocation: ONE warp, pow2 >=32 columns -----------------------
// tcgen05.alloc writes the allocated TMEM base address into SMEM at [dst_smem].
// Must be issued from a single (whole) warp; relinquish the alloc permit after.
__device__ __forceinline__ void tmem_alloc(uint32_t dst_smem, uint32_t ncols) {
  asm volatile(
    "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
    :: "r"(dst_smem), "r"(ncols));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
}

__device__ __forceinline__ void tmem_dealloc(uint32_t tmem_addr, uint32_t ncols) {
  asm volatile(
    "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
    :: "r"(tmem_addr), "r"(ncols));
}

// ---- (3) the MMA: single-thread issue, accumulator in TMEM ------------------
// tcgen05.mma.cta_group::1.kind::f16  [d-tmem], a-desc, b-desc, idesc, enable-d
// enable-input-d is a PREDICATE (accumulate onto the running D or overwrite it):
//   false on the first K-atom (D := A*B), true thereafter (D += A*B).
__device__ __forceinline__ void tcgen05_mma_f16(
    uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, bool accum) {
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %4, 0;\n"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
    "}\n"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"((int)accum));
}

// ---- (4) commit: MMA completion arrives on an mbarrier ----------------------
__device__ __forceinline__ void tcgen05_commit(uint32_t bar) {
  asm volatile(
    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
    :: "r"(bar) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;\n");
}
__device__ __forceinline__ void tcgen05_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;\n");
}

// ---- (5) FULL-WARPGROUP TMEM drain — the 32-lane-per-warp rule --------------
// TMEM is 128 lanes x 512 cols. A TMEM address packs [31:16]=lane, [15:0]=col.
// One warp's tcgen05.ld touches ONLY its 32 lanes; so all 4 warps of the
// warpgroup must run, warp w draining lanes [32*w, 32*w+32). If you drain from
// a single warp you read 32/128 lanes and 3/4 of the 128-row output tile is
// STALE — the classic tcgen05 correctness bug (A3 §pitfalls; verify per-quadrant
// against the reference, not just the top-left tile).
//   tcgen05.ld.sync.aligned.32x32b.x4.b32  {r0..r3}, [taddr]  reads 32 lanes x 4
//   columns into 4 regs/thread; loop the columns. [INFERENCE] the exact
//   fragment->(row,col) mapping of the ld result is per PTX-ISA/CUTLASS and is
//   VALIDATED ON THE B200 in the epilogue; here we drain every lane-quadrant and
//   store, proving the 128-lane coverage (the structural point of this rung).
__device__ __forceinline__ void drain_tmem(
    uint32_t tmem_base, float *__restrict__ D, int ld_D) {
  const int tid  = threadIdx.x;              // 0..127 : a full warpgroup
  const int warp = tid >> 5;                 // 0..3   : lane-quadrant owner
  const int lane = tid & 31;
  // warp w owns TMEM lanes [32*w, 32*w+32): OR the lane-quadrant into addr[31:16].
  const uint32_t warp_lane_base = tmem_base | (uint32_t)(32 * warp) << 16;

  #pragma unroll 1
  for (int col = 0; col < NCOL; col += 4) {
    uint32_t taddr = warp_lane_base | (uint32_t)col;   // addr[15:0] = column
    float r0, r1, r2, r3;
    asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n"
      : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3) : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");   // ld must retire before use
    // Epilogue store. Row = this warp's lane-quadrant base + intra-warp lane;
    // the 4 regs map to columns col..col+3 for this (lane,quadrant). The precise
    // ld fragment layout is B200-validated (see note above); the store below is
    // the structural placeholder that makes the 128-lane coverage explicit.
    const int row = 32 * warp + lane;
    if (row < MMA_M) {
      if (col + 0 < MMA_N) D[row * ld_D + col + 0] = r0;
      if (col + 1 < MMA_N) D[row * ld_D + col + 1] = r1;
      if (col + 2 < MMA_N) D[row * ld_D + col + 2] = r2;
      if (col + 3 < MMA_N) D[row * ld_D + col + 3] = r3;
    }
  }
}

// ============================================================================
// The 1-SM tcgen05 GEMM kernel.  D[MMA_M x MMA_N] = A[MMA_M x K] * B[K x MMA_N]
// A/B are staged by TMA from global into SMEM K-strips; the accumulator lives in
// TMEM for the whole K reduction and is drained once at the end.
// Launch: <<< grid, 128 >>>  (one warpgroup = 4 warps = 128 threads, 1 SM/CTA).
// ============================================================================
extern "C" __global__ void tcgen05_gemm_1sm(
    const __grid_constant__ CUtensorMap tma_a,   // A tensor-map (row-major MxK)
    const __grid_constant__ CUtensorMap tma_b,   // B tensor-map (col-major KxN)
    float *__restrict__ D, int K, int ld_D) {

  // --- SMEM layout: double-buffered A/B K-strips + mbarriers + TMEM-ptr slot --
  extern __shared__ uint8_t smem[];
  __half   *sA   = reinterpret_cast<__half*>(smem);                     // MMA_M*MMA_K
  __half   *sB   = sA + MMA_M * MMA_K;                                   // MMA_K*MMA_N
  uint64_t *barp = reinterpret_cast<uint64_t*>(sB + MMA_K * MMA_N);
  uint32_t *tmem_slot = reinterpret_cast<uint32_t*>(barp + 2);          // alloc dst

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;

  uint32_t bar_full = smem_u32(&barp[0]);   // TMA-arrival mbarrier
  uint32_t bar_mma  = smem_u32(&barp[1]);   // MMA-commit mbarrier
  uint32_t tslot    = smem_u32(tmem_slot);

  // --- (1) allocate the TMEM accumulator from ONE warp (warp 0) --------------
  if (warp == 0) tmem_alloc(tslot, NCOL);
  if (tid == 0) { mbar_init(bar_full, 1); mbar_init(bar_mma, 1); }
  __syncthreads();
  uint32_t d_tmem = *tmem_slot;             // TMEM base addr the alloc wrote back

  const uint32_t idesc = make_idesc_f16_dense();
  // 128B-swizzled, 128x16 / 16x256 f16 core-matrix tiling (artifact §2 worked
  // encoding): LBO=16B, SBO=1024B, swizzle=1. Rental day rebuilds these from the
  // CuTe layout of the actual SMEM tile; the encoder itself is byte-checked here.
  const uint32_t LBO = 16, SBO = 1024, SWZ = 1;

  const int n_kiter = K / MMA_K;
  uint32_t phase = 0;

  // ------------------------- the K mainloop --------------------------------
  #pragma unroll 1
  for (int k = 0; k < n_kiter; ++k) {
    // (2) TMA: stage this K-strip of A and B into SMEM, tracked on bar_full.
    if (tid == 0) {
      mbar_expect(bar_full,
                  MMA_M * MMA_K * (uint32_t)sizeof(__half)
                + MMA_K * MMA_N * (uint32_t)sizeof(__half));
      tma_load_2d(sA, &tma_a, bar_full, /*col*/ k * MMA_K, /*row*/ 0);
      tma_load_2d(sB, &tma_b, bar_full, /*col*/ 0, /*row*/ k * MMA_K);
    }
    mbar_wait(bar_full, phase & 1);

    // (3) ONE thread issues the async MMA; accumulate in TMEM.
    //     enable-input-d = false on the first K-atom (D := A*B), true after.
    tcgen05_fence_before();
    if (tid == 0) {
      uint64_t a_desc = make_smem_desc(sA, LBO, SBO, SWZ);
      uint64_t b_desc = make_smem_desc(sB, LBO, SBO, SWZ);
      tcgen05_mma_f16(d_tmem, a_desc, b_desc, idesc, /*accum=*/ k != 0);
      // (4) commit — completion arrives on bar_mma (async, not wait_group).
      tcgen05_commit(bar_mma);
    }
    // Wait for this atom's MMA to retire before overwriting the SMEM strip.
    mbar_wait(bar_mma, phase & 1);
    tcgen05_fence_after();
    __syncthreads();
    ++phase;
  }

  // (5) FULL-WARPGROUP drain of the TMEM accumulator -> epilogue -> global D.
  drain_tmem(d_tmem, D, ld_D);

  __syncthreads();
  if (warp == 0) tmem_dealloc(d_tmem, NCOL);   // free TMEM from one warp
}
