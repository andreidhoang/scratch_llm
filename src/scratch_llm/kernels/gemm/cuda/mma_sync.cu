// A3 Rung 2 — mma.sync + ldmatrix + XOR-swizzled SMEM GEMM (warp-level tensor cores).
//
// This is Rung 1's WMMA kernel with the lid off: instead of nvcuda::wmma hiding the
// thread->value fragment layout, we issue the PTX warp-MMA
//     mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// by hand, load the operand fragments straight into the tensor-core register layout with
// ldmatrix.sync (a warp-collective SMEM load), and lay SMEM out with an XOR swizzle so the
// ldmatrix rows land in distinct banks.  Inputs are f16, the accumulator is FP32 (mixed
// precision — the only thing that keeps a long-K reduction numerically stable), output FP32.
//
// Tile geometry (single-buffered, correctness-first):
//   block  : BM=64 x BN=64 output tile, stepped over K in BK=32 chunks
//   warps  : 4 warps (128 threads) in a 2x2 grid; each warp owns a 32x32 output sub-tile
//   atoms  : per warp, MMA_M=16 -> 2 m-atoms, MMA_N=8 -> 4 n-atoms, MMA_K=16 -> 2 k-atoms/BK
//   accum  : acc[2][4][4] FP32 registers, live across the whole K loop
//
// Remainders (K not a multiple of BK/16, M/N remainder tiles, M=1): global->SMEM loads are
// bounds-masked to 0, so out-of-range lanes contribute nothing to the reduction and the SMEM
// tile is always fully populated; the epilogue only writes in-range (row,col).  No separate
// remainder kernel — the zero-fill is the whole story, same as Rung 0.
//
// SWIZZLE (the bank-conflict argument; ncu blocked here -> recorded as ncu-debt):
//   SMEM is permuted at 16-byte (8xf16) chunk granularity so each ldmatrix "row" (8 contiguous
//   f16) stays intact.  Logical chunk c at row r is stored at physical chunk  c XOR (r & (C-1)):
//     As: 64x32 f16, C=4 chunks/row -> mask 3;  Bs: 32x64 f16, C=8 chunks/row -> mask 7.
//   XOR-with-a-per-row-constant is a bijection on each row (so it is trivially invertible and
//   correctness-preserving) and maps consecutive rows of the SAME logical chunk to DISTINCT
//   physical chunks -> distinct 4-byte-bank sets.  ldmatrix reads one 8x8 tile as 8 row-pointers;
//   the 8 rows differ in (r & mask) at the low bits, so their target chunks fan across banks
//   instead of colliding.  ncu metric to close on the H100 day:
//     l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum  (target ~0).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BM 64
#define BN 64
#define BK 32
#define WARPS 4              // 2x2 warp grid
#define THREADS (WARPS * 32) // 128

// ---- PTX wrappers ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ldmatrix.x4: load four 8x8 f16 tiles into the mma A-operand register layout (a0..a3).
__device__ __forceinline__ void ldmatrix_x4(uint32_t& d0, uint32_t& d1, uint32_t& d2,
                                            uint32_t& d3, const void* ptr) {
  uint32_t a = smem_u32(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
               : "r"(a));
}

// ldmatrix.x2.trans: load two 8x8 f16 tiles, transposing each on the way in -> the mma
// B-operand (col) register layout (b0,b1).  .trans is the canonical idiom for a K-major B.
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& d0, uint32_t& d1, const void* ptr) {
  uint32_t a = smem_u32(ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d0), "=r"(d1)
               : "r"(a));
}

// D = A*B + C for one m16n8k16 atom; A row-major, B col-major, FP32 accumulate (in place).
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1, float& c2, float& c3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---- kernel ---------------------------------------------------------------------------------

__global__ void gemm_mma_sync_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                                     float* __restrict__ C, int M, int N, int K) {
  __shared__ __half As[BM * BK];  // 64 x 32, swizzled by 16B chunk (C=4)
  __shared__ __half Bs[BK * BN];  // 32 x 64, swizzled by 16B chunk (C=8)

  const int tid = threadIdx.x;             // 0..127
  const int warp = tid >> 5;               // 0..3
  const int lane = tid & 31;               // 0..31
  const int warpRow = warp >> 1;           // 0..1  -> block rows [warpRow*32, +32)
  const int warpCol = warp & 1;            // 0..1  -> block cols [warpCol*32, +32)

  const int blockRow = blockIdx.y * BM;
  const int blockCol = blockIdx.x * BN;

  float acc[2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mi][ni][e] = 0.0f;

  const int nKtiles = (K + BK - 1) / BK;
  for (int kt = 0; kt < nKtiles; ++kt) {
    const int k0 = kt * BK;

    // --- global -> SMEM, 16B (8xf16) chunks, bounds-masked to 0, stored swizzled ---
    // A tile: 64x32 = 256 chunks; 128 threads x 2 chunks.  chunk ci: row=ci/4, chunkInRow=ci%4.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int ci = tid + i * THREADS;  // 0..255
      const int row = ci >> 2;           // 0..63   (BK/8 = 4 chunks per row)
      const int cInRow = ci & 3;         // 0..3
      const int gRow = blockRow + row;
      const int gK = k0 + cInRow * 8;
      const int pchunk = cInRow ^ (row & 3);
      __half* dst = &As[row * BK + pchunk * 8];
      const long soff = (long)gRow * K + gK;  // source offset (elements)
      if (gRow < M && gK + 8 <= K && (soff & 7) == 0) {
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(&A[soff]);  // 128-bit
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int kk = gK + j;
          dst[j] = (gRow < M && kk < K) ? A[gRow * K + kk] : __float2half(0.0f);
        }
      }
    }
    // B tile: 32x64 = 256 chunks.  chunk ci: row(k)=ci/8, chunkInRow=ci%8.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int ci = tid + i * THREADS;  // 0..255
      const int row = ci >> 3;           // 0..31   (BN/8 = 8 chunks per row)
      const int cInRow = ci & 7;         // 0..7
      const int gK = k0 + row;
      const int gN = blockCol + cInRow * 8;
      const int pchunk = cInRow ^ (row & 7);
      __half* dst = &Bs[row * BN + pchunk * 8];
      const long soff = (long)gK * N + gN;  // source offset (elements)
      if (gK < K && gN + 8 <= N && (soff & 7) == 0) {
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(&B[soff]);  // 128-bit
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int nn = gN + j;
          dst[j] = (gK < K && nn < N) ? B[gK * N + nn] : __float2half(0.0f);
        }
      }
    }
    __syncthreads();

    // --- ldmatrix operand fragments from swizzled SMEM ---
    // A frags: a_frag[mi][ki][0..3]; B frags: b_frag[ni][ki][0..1].
    uint32_t a_frag[2][2][4];
    uint32_t b_frag[4][2][2];

    const int m = lane >> 3;        // 0..3  (ldmatrix.x4 group)
    const int r = lane & 7;         // 0..7  row within the 8x8
    const int rowblock = m & 1;     // which 16-row half
    const int colblock = m >> 1;    // which 8-col half within the 16-col A tile
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int ki = 0; ki < 2; ++ki) {
        const int aRow = warpRow * 32 + mi * 16 + rowblock * 8 + r;
        const int aChunk = ki * 2 + colblock;             // logical chunk 0..3
        const int pchunk = aChunk ^ (aRow & 3);
        ldmatrix_x4(a_frag[mi][ki][0], a_frag[mi][ki][1], a_frag[mi][ki][2], a_frag[mi][ki][3],
                    &As[aRow * BK + pchunk * 8]);
      }
    }
    const int bg = (lane & 15) >> 3;   // 0..1  matrix within x2 (lanes 16..31 alias to 0..15)
    const int br = lane & 7;           // 0..7  source row within the 8x8
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
      for (int ki = 0; ki < 2; ++ki) {
        const int bRow = ki * 16 + bg * 8 + br;
        const int bChunk = warpCol * 4 + ni;              // logical chunk 0..7
        const int pchunk = bChunk ^ (bRow & 7);
        ldmatrix_x2_trans(b_frag[ni][ki][0], b_frag[ni][ki][1], &Bs[bRow * BN + pchunk * 8]);
      }
    }

    // --- 2x4x2 = 16 warp-MMAs, accumulating both k-atoms into acc ---
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
          mma_m16n8k16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                       a_frag[mi][ki][0], a_frag[mi][ki][1], a_frag[mi][ki][2], a_frag[mi][ki][3],
                       b_frag[ni][ki][0], b_frag[ni][ki][1]);
    __syncthreads();
  }

  // --- epilogue: acc -> C (row-major FP32), bounds-checked ---
  const int groupID = lane >> 2;      // 0..7
  const int tInGroup = lane & 3;      // 0..3
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int baseRow = blockRow + warpRow * 32 + mi * 16;
      const int baseCol = blockCol + warpCol * 32 + ni * 8;
      const int r0 = baseRow + groupID;
      const int r1 = baseRow + groupID + 8;
      const int c0 = baseCol + tInGroup * 2;
      const int c1 = c0 + 1;
      if (r0 < M) {
        if (c0 < N) C[r0 * N + c0] = acc[mi][ni][0];
        if (c1 < N) C[r0 * N + c1] = acc[mi][ni][1];
      }
      if (r1 < M) {
        if (c0 < N) C[r1 * N + c0] = acc[mi][ni][2];
        if (c1 < N) C[r1 * N + c1] = acc[mi][ni][3];
      }
    }
  }
}

torch::Tensor gemm_mma_sync(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2-D");
  TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf,
              "A and B must be float16 (the mma.f16.f16 operand type)");
  TORCH_CHECK(A.size(1) == B.size(0), "inner dims must match: A[M,K] @ B[K,N]");

  A = A.contiguous();
  B = B.contiguous();
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = torch::empty({M, N}, A.options().dtype(at::kFloat));

  const dim3 block(THREADS);
  const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_mma_sync_kernel<<<grid, block>>>(reinterpret_cast<const __half*>(A.data_ptr()),
                                        reinterpret_cast<const __half*>(B.data_ptr()),
                                        C.data_ptr<float>(), M, N, K);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "gemm_mma_sync kernel launch failed");
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_mma_sync", &gemm_mma_sync,
        "A3 R2: warp-level mma.sync m16n8k16 + ldmatrix + XOR-swizzled SMEM GEMM (f16->fp32)");
}
