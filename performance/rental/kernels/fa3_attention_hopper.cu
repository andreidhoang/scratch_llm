// =============================================================================================
// A4 R4 — FA3-class fused attention forward SKELETON (Hopper, sm_90a), COMPILE-VERIFIED.
//
//   nvcc -arch=sm_90a -ptx  fa3_attention_hopper.cu -o fa3_attention_hopper.ptx        (exit 0)
//   # grep the emitted PTX to prove the intended ISA was generated:
//   grep -E 'wgmma\.mma_async|cp\.async\.bulk\.tensor|mbarrier|setmaxnreg' fa3_attention_hopper.ptx
//
// Reference: Shah/Bikshandi/Ye/Thakkar/Ramani/Dao, "FlashAttention-3: Fast and Accurate
//   Attention with Asynchrony and Low-precision" (arXiv:2407.08608) + Tri Dao's flash-attn v3
//   (hopper/ CUTLASS/CuTe mainloop) + CUTLASS 3.x GmmaDescriptor.  WGMMA descriptor encoding
//   follows the repo's PTX artifact  performance/artifacts/wgmma_descriptor_manual.md  (the
//   64-bit SMEM descriptor bit-field map §2.1 and the wgmma.fence/commit/wait protocol §1.10).
//
// Pre-registered target (performance/PERF_ENGINEERING_SPEC.md §4 · A4, R4 / A4_flash_attention.md §4.1):
//   ~75% of H100 FP16 peak  ≈  740 TFLOP/s  (FA3 paper number).  FP8 (E4M3) path ~1.2 PFLOP/s.
//   These are the *paper* numbers — cited as such, NOT measured here.
//
// ---------------------------------------------------------------------------------------------
// RUNTIME CORRECTNESS DEFERRED TO THE RENTAL DAY — compiles here (sm_90a), runs on H100/B200.
// The standing box is sm_120 consumer Blackwell; this kernel CANNOT run here.  nvcc *emits* the
// sm_90a PTX host-side (arch-checked text only, no Hopper GPU needed), which is the compile gate;
// the online-softmax oracle (vs torch SDPA) and the TF/s roofline run on the rented H100.  This
// is a STRUCTURAL skeleton: it contains the full FA3 pipeline shape (warp-specialized producer /
// consumer warpgroups, TMA loads via cp.async.bulk.tensor + mbarrier, WGMMA QK^T -> online-softmax
// rescale -> WGMMA PV, ping-pong SMEM double-buffering) but the exact WGMMA-fragment thread->element
// mapping in the softmax reduction and the TMA tensor-map coordinates are marked DEFER and are the
// first things to byte-check on the H100 against CUTLASS/flash-attn v3.
// =============================================================================================

#include <cuda.h>            // CUtensorMap (driver-API type; header-only for the struct)
#include <cuda_fp16.h>
#include <math_constants.h>   // CUDART_INF_F
#include <cstdint>

// ---- Tile geometry (correctness-first; FA3 production uses D=128, larger Bc, 2-CTA clusters) --
//   Br  = 64  query rows per consumer warpgroup   (WGMMA M is fixed at 64)
//   Bc  = 64  key/value rows streamed per stage
//   D   = 64  head dim   (production: 128 -> two n=64 WGMMA atoms; noted at each GEMM)
//   Both GEMMs are built from the artifact's verified atom  wgmma.m64n64k16.f32.f16.f16,
//   accumulating the K/contraction dim (=D for QK^T, =Bc for PV) in 16-wide steps.
#define Br 64
#define Bc 64
#define D  64
#define STAGES 2                    // ping-pong SMEM double-buffer for K,V

#define WG_SIZE 128                 // one warpgroup = 4 warps = 128 threads (wgmma-collective)
#define CONSUMER_WGS 1              // math warpgroups (FP16 skeleton; FA3 ping-pongs 2)
#define PRODUCER_WG  1              // one TMA producer warpgroup
#define NUM_WARPGROUPS (CONSUMER_WGS + PRODUCER_WG)
#define NTHREADS (NUM_WARPGROUPS * WG_SIZE)   // 256

#define KSTEPS_QK (D  / 16)         // 4  : contract head dim for S = Q K^T
#define KSTEPS_PV (Bc / 16)         // 4  : contract key dim  for O = P V
#define ACC_REGS 32                 // m64n64 accumulator = 64*64/128 regs/thread (artifact §1.9)

// =============================================================================================
// 1.  SMEM matrix descriptor  (performance/artifacts/wgmma_descriptor_manual.md §2.1)
//     64-bit descriptor = value>>4 packed fields; layout_type in bits [62,64).
// =============================================================================================
__device__ __forceinline__ uint64_t descriptor_encode(uint64_t x) {
  // artifact §2.1: matrix_descriptor_encode(x) = (x & 0x3FFFF) >> 4   (mask 18 bits, drop low 4)
  return (x & 0x3FFFFull) >> 4;
}

// Build a K-major SMEM descriptor for a 64-wide f16 tile with 128B swizzle.
// LBO=16B, SBO=1024B, swizzle=1 — the exact constants derived + Colfax-corroborated in §2.3.
__device__ __forceinline__ uint64_t make_smem_desc(const void* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint64_t desc = 0;
  desc |= descriptor_encode(addr);              // [0,14)  start_address = addr>>4
  desc |= descriptor_encode(16)   << 16;        // [16,30) LBO = 16 bytes  -> 1
  desc |= descriptor_encode(1024) << 32;        // [32,46) SBO = 1024 bytes -> 64
  desc |= 1ull << 62;                           // [62,64) layout_type = 1 (128B swizzle)
  return desc;
}

// =============================================================================================
// 2.  WGMMA async-MMA glue  (artifact §1.10 fence/commit/wait; §5 verified operand strings)
// =============================================================================================
__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// One m64n64k16 warpgroup MMA, SS form (A-desc, B-desc), FP32 accumulator (32 regs/thread).
// scale_d = 0 on the first K-step (D = A*B, free zero-init), 1 thereafter (D += A*B).
// Operand string is verbatim from wgmma_descriptor_manual.md §5 (compiler-emitted ground truth).
__device__ __forceinline__ void wgmma_m64n64k16(float d[ACC_REGS], uint64_t a_desc,
                                                uint64_t b_desc, int scale_d) {
  asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "  setp.ne.b32 p, %34, 0;\n"
    "  wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
    "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "   %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
    "  %32, %33, p, 1, 1, 0, 0;\n"
    "}\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
      "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
      "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
      "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
    : "l"(a_desc), "l"(b_desc), "r"(scale_d));
}
// NOTE (FP8 path, DEFER): the E4M3 variant is wgmma.mma_async...m64n64k32.f32.e4m3.e4m3 — same
// glue, K=32 per step, KSTEPS halve.  Add per-block scales + incoherent-processing Hadamard on
// Q,K before the quantize (FA3 §3.3).  Accumulate FP32; softmax exp always FP32.

// =============================================================================================
// 3.  mbarrier + TMA (producer/consumer handshake).  Async-barrier primitives, PTX ISA §9.7.13;
//     TMA bulk-tensor copy cp.async.bulk.tensor.2d, PTX ISA §9.7.9.  Hopper-only (sm_90a).
// =============================================================================================
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count));
}
// producer: announce this phase will deliver `bytes` via TMA, then arrive.
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
// consumer: spin on the phase bit until the TMA completion flips it.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
    "{\n"
    "  .reg .pred p;\n"
    "LAB_WAIT:\n"
    "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
    "  @p bra DONE_WAIT;\n"
    "  bra LAB_WAIT;\n"
    "DONE_WAIT:\n"
    "}\n"
    :: "r"(smem_u32(bar)), "r"(phase) : "memory");
}
// TMA 2-D tile load  HBM -> SMEM, single-thread-issued, completion signalled on `bar`.
__device__ __forceinline__ void tma_load_2d(void* smem_dst, const CUtensorMap* tmap,
                                            int coord_x, int coord_y, uint64_t* bar) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3}], [%4];\n"
    :: "r"(smem_u32(smem_dst)), "l"(reinterpret_cast<uint64_t>(tmap)),
       "r"(coord_x), "r"(coord_y), "r"(smem_u32(bar)) : "memory");
}

// register-reallocation: give math warpgroups more regs, the producer fewer (FA3 §4.1 warp-spec).
__device__ __forceinline__ void warpgroup_reg_alloc()   { asm volatile("setmaxnreg.inc.sync.aligned.u32 240;\n"); }
__device__ __forceinline__ void warpgroup_reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n"); }

// =============================================================================================
// 4.  The kernel — warp-specialized FA3 forward.
//     grid = (num_q_tiles, num_heads, batch);  block = NTHREADS (producer WG + consumer WG(s)).
//     Q,K,V,O are fronted by TMA tensor maps built host-side (cuTensorMapEncodeTiled).
// =============================================================================================
extern "C" __global__ __launch_bounds__(NTHREADS) void fa3_attention_fwd(
    const __grid_constant__ CUtensorMap tmap_Q,
    const __grid_constant__ CUtensorMap tmap_K,
    const __grid_constant__ CUtensorMap tmap_V,
    __half* __restrict__ O,          // [.. , Br, D] tile store (skeleton: direct store)
    int seqlen_kv,                   // number of key/value rows to stream
    float softmax_scale,             // 1/sqrt(D)
    bool causal) {

  // ---- SMEM layout: Q tile (resident) + ping-pong K,V stages + mbarriers -------------------
  extern __shared__ __align__(128) uint8_t smem_raw[];
  __half* sQ = reinterpret_cast<__half*>(smem_raw);                       // [Br, D]
  __half* sK = sQ + Br * D;                                               // [STAGES][Bc, D]
  __half* sV = sK + STAGES * Bc * D;                                      // [STAGES][Bc, D]
  uint64_t* bar_full  = reinterpret_cast<uint64_t*>(sV + STAGES * Bc * D); // [STAGES] K,V ready
  uint64_t* bar_empty = bar_full + STAGES;                                // [STAGES] stage free
  uint64_t* bar_q     = bar_empty + STAGES;                               // [1]      Q ready

  const int tid   = threadIdx.x;
  const int wg    = tid / WG_SIZE;            // 0 = producer, 1.. = consumer warpgroup(s)
  const int lane  = tid % 32;
  const bool is_producer = (wg == 0);

  // Batch/head/q-tile addressing for the TMA coordinates (row-major [B,H,S,D] tensor maps).
  const int q_tile = blockIdx.x;             // which Br-row block of queries
  const int head   = blockIdx.y;
  const int batch  = blockIdx.z;
  const int q_row0 = q_tile * Br;            // first query row this block owns
  // DEFER: the exact (x,y) tensor-map coordinate convention (element vs box units, head/batch
  // folded into y) is byte-checked on the H100 against cuTensorMapEncodeTiled + flash-attn v3.
  const int kv_head_off = (batch * gridDim.y + head) * seqlen_kv;

  const int num_kv_tiles = (seqlen_kv + Bc - 1) / Bc;

  // ---- barrier init (one thread) ----
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&bar_full[s], 1);            // filled by the single TMA-issuing producer thread
      mbar_init(&bar_empty[s], WG_SIZE * CONSUMER_WGS);
    }
    mbar_init(bar_q, 1);
  }
  __syncthreads();

  // =========================================================================================
  // PRODUCER warpgroup — issues TMA loads of Q once, then K,V tiles into ping-pong stages.
  // =========================================================================================
  if (is_producer) {
    warpgroup_reg_dealloc();                 // release registers to the math warpgroups
    const bool tma_thread = (tid == 0);      // one elected thread issues the bulk copies

    // Q tile: load once, resident for the whole tile loop.
    if (tma_thread) {
      mbar_expect_tx(bar_q, Br * D * sizeof(__half));
      tma_load_2d(sQ, &tmap_Q, /*x=*/0, /*y=*/q_row0, bar_q);
    }

    uint32_t empty_phase = 0;
    for (int kt = 0; kt < num_kv_tiles; ++kt) {
      const int stage = kt % STAGES;
      // causal skip: if the whole K tile is above the diagonal, the producer never loads it.
      if (causal && (kt * Bc) > (q_row0 + Br - 1)) break;

      // wait until the consumer has drained this stage (ping-pong back-pressure).
      mbar_wait(&bar_empty[stage], (empty_phase >> stage) & 1u);
      if (((kt / STAGES) & 1) && stage == STAGES - 1) empty_phase ^= (1u << 0); // toggle bookkeeping (DEFER: exact parity)

      if (tma_thread) {
        const int kv_row0 = kt * Bc;
        mbar_expect_tx(&bar_full[stage], 2 * Bc * D * sizeof(__half)); // K + V bytes
        tma_load_2d(&sK[stage * Bc * D], &tmap_K, 0, kv_head_off + kv_row0, &bar_full[stage]);
        tma_load_2d(&sV[stage * Bc * D], &tmap_V, 0, kv_head_off + kv_row0, &bar_full[stage]);
      }
    }
    return;                                  // producer done
  }

  // =========================================================================================
  // CONSUMER warpgroup(s) — WGMMA QK^T -> online softmax -> WGMMA PV, over streamed K,V tiles.
  // =========================================================================================
  warpgroup_reg_alloc();

  // Per-row online-softmax state.  m64n64 accumulator: 32 regs/thread hold a 64x64 tile; each
  // thread owns 2 of the 64 query rows (WGMMA D-fragment layout).  m,l are per-owned-row.
  // DEFER: the precise register<->(row,col) map of the wgmma D-fragment is byte-checked against
  // CUTLASS cute::partition_fragment on the H100; here it is the FA3 recurrence in structure.
  float acc_o[ACC_REGS];                     // running unnormalized output O~ (Br x D fragment)
  float row_m[2];                            // running max per owned query row
  float row_l[2];                            // running denom  per owned query row
#pragma unroll
  for (int i = 0; i < ACC_REGS; ++i) acc_o[i] = 0.0f;
  row_m[0] = row_m[1] = -CUDART_INF_F;
  row_l[0] = row_l[1] = 0.0f;

  // wait for the resident Q tile.
  mbar_wait(bar_q, 0);

  uint32_t full_phase = 0;
  for (int kt = 0; kt < num_kv_tiles; ++kt) {
    const int stage = kt % STAGES;
    if (causal && (kt * Bc) > (q_row0 + Br - 1)) break;

    // ---- wait for producer to fill this stage (K,V ready) ----
    mbar_wait(&bar_full[stage], (full_phase >> stage) & 1u);

    // ---- WGMMA #1:  S = Q @ K^T   (m64n64, contract D in KSTEPS_QK steps of 16) ------------
    float acc_s[ACC_REGS];
#pragma unroll
    for (int i = 0; i < ACC_REGS; ++i) acc_s[i] = 0.0f;
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < KSTEPS_QK; ++ks) {
      uint64_t a_desc = make_smem_desc(&sQ[ks * 16]);                 // Q[:, ks*16 : +16]
      uint64_t b_desc = make_smem_desc(&sK[stage * Bc * D + ks * 16]);// K[:, ks*16 : +16] (K^T walk)
      wgmma_m64n64k16(acc_s, a_desc, b_desc, /*scale_d=*/ks == 0 ? 0 : 1);
    }
    wgmma_commit();
    wgmma_wait<0>();                          // S valid.  (FA3 leaves prior PV in flight: wait<1>)

    // ---- scale + causal mask on the score fragment ----
#pragma unroll
    for (int i = 0; i < ACC_REGS; ++i) acc_s[i] *= softmax_scale;
    // DEFER: apply the j>i causal mask per fragment element (needs the byte-checked (row,col) map).

    // ---- online-softmax update (the engine — exact, not an approximation) ------------------
    // 1) tile row-max over acc_s -> m_tile[2] (per owned row); 2) m_new = max(m_old, m_tile);
    // 3) rescale factor alpha = exp2((m_old - m_new)*log2e) applied to acc_o and row_l;
    // 4) p = exp2((acc_s - m_new)*log2e);  row_l += sum(p);  cast p->f16 for the PV GEMM.
    const float LOG2E = 1.4426950408889634f;
    float m_tile[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
    for (int i = 0; i < ACC_REGS; ++i) {
      const int r = i & 1;                    // owned-row selector (DEFER: real frag map)
      m_tile[r] = fmaxf(m_tile[r], acc_s[i]);
    }
    // reduce m_tile across the 4 lanes that share a row's columns (WGMMA quad layout).
#pragma unroll
    for (int off = 2; off >= 1; off >>= 1) {
#pragma unroll
      for (int r = 0; r < 2; ++r)
        m_tile[r] = fmaxf(m_tile[r], __shfl_xor_sync(0xffffffff, m_tile[r], off));
    }
    float alpha[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float m_new = fmaxf(row_m[r], m_tile[r]);
      alpha[r]    = exp2f((row_m[r] - m_new) * LOG2E);   // rebasing factor e^{m_old - m_new}
      row_m[r]    = m_new;
      row_l[r]   *= alpha[r];                            // rescale running denom
    }
    // rescale the running output fragment by alpha (per owned row) -- the classic bug is
    // forgetting THIS (rescale O, not just l); see A4 §critique.
#pragma unroll
    for (int i = 0; i < ACC_REGS; ++i) acc_o[i] *= alpha[i & 1];

    // p = exp2((s - m_new)); accumulate denom; pack to f16 for PV.  Store P into SMEM (reuse sK
    // stage region conceptually; here a dedicated scratch would be used) -- DEFER exact P layout.
    __shared__ __half sP[Br * Bc];
#pragma unroll
    for (int i = 0; i < ACC_REGS; ++i) {
      const int r = i & 1;
      float p = exp2f((acc_s[i] - row_m[r]) * LOG2E);
      row_l[r] += p;
      acc_s[i] = p;                            // reuse acc_s to hold P fragment (f32)
    }
    // reduce row_l across the column-lanes (same quad reduction as the max).
#pragma unroll
    for (int off = 2; off >= 1; off >>= 1) {
#pragma unroll
      for (int r = 0; r < 2; ++r)
        row_l[r] += __shfl_xor_sync(0xffffffff, row_l[r], off);
    }
    // Write P fragment -> SMEM as f16 for the second GEMM (DEFER: fragment->SMEM store map;
    // FA3 keeps P in registers as the WGMMA A-operand (RS form) to skip this round-trip).
    if (lane < 1) {  // structural placeholder store to keep sP live in the PTX
#pragma unroll
      for (int i = 0; i < ACC_REGS && i < Br * Bc; ++i) sP[i] = __float2half(acc_s[i]);
    }
    asm volatile("bar.sync 8, %0;\n" :: "r"(WG_SIZE * CONSUMER_WGS)); // consumer-only barrier

    // ---- WGMMA #2:  O~ += P @ V   (m64n64, contract Bc in KSTEPS_PV steps of 16) ------------
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < KSTEPS_PV; ++ks) {
      uint64_t a_desc = make_smem_desc(&sP[ks * 16]);
      uint64_t b_desc = make_smem_desc(&sV[stage * Bc * D + ks * 16]);
      wgmma_m64n64k16(acc_o, a_desc, b_desc, /*scale_d=*/1);  // always accumulate into O~
    }
    wgmma_commit();
    wgmma_wait<0>();

    // ---- release the stage back to the producer (ping-pong) ----
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                 :: "r"(smem_u32(&bar_empty[stage])) : "memory");
    if (stage == STAGES - 1) full_phase ^= ((1u << 0) | (1u << 1));   // flip both stage phases
  }

  // ---- epilogue: normalize O = O~ / l, then store the Br x D output tile ---------------------
#pragma unroll
  for (int i = 0; i < ACC_REGS; ++i) {
    const int r = i & 1;
    float inv_l = (row_l[r] > 0.0f) ? (1.0f / row_l[r]) : 0.0f;       // deferred divide (FA2 trick)
    acc_o[i] *= inv_l;
  }
  // DEFER: real WGMMA D-fragment -> global (row,col) mapping (CUTLASS epilogue) or a TMA store.
  // Structural store so acc_o and O stay live in the PTX and the write-back path is present.
  const int base = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x * Br * D + q_row0 * D;
#pragma unroll
  for (int i = 0; i < ACC_REGS; ++i) {
    int idx = base + tid * ACC_REGS + i;      // placeholder linearization (NOT the FA layout)
    O[idx] = __float2half(acc_o[i]);
  }
}

// ---- host-side note (not compiled into the kernel PTX) --------------------------------------
// The three CUtensorMap arguments are built once on the host with the driver API
//   cuTensorMapEncodeTiled(&tmap_Q, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank=2, gmem_ptr, dims,
//                          strides, box={Br,D}/{Bc,D}, ..., CU_TENSOR_MAP_SWIZZLE_128B, ...)
// and passed by value as __grid_constant__ (they land in constant/param space).  The 128B swizzle
// in the tensor map MUST match the make_smem_desc layout_type=1 above (artifact §3.3: the TMA
// swizzle and the wgmma descriptor swizzle are one contract split across two descriptors).
