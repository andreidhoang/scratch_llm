"""A3 Rung 1 — WMMA tensor-core GEMM (raw CUDA C++, sm_120), the explicit-fragment kernel.

A3's whole point is the tensor-core *fragment* machinery that Triton's ``tl.dot`` abstracts away.
Here we issue it by hand with ``nvcuda::wmma``: a 128x128 block tile owned by 8 warps, each warp
holding a 4x2 grid of 16x16x16 accumulator fragments, driven by the canonical
``load_matrix_sync -> mma_sync -> store_matrix_sync`` sequence. Inputs are FP16; the accumulator is
FP32 — *mixed precision is what keeps the long-K reduction numerically stable*, which is the load-
bearing lesson of the rung.

Two kernels are compiled from one templated body:

  * ``wmma_gemm``          — FP32 accumulator (the shipping kernel; the correctness gate).
  * ``wmma_gemm_fp16acc``  — FP16 accumulator (the anti-example): every ``mma_sync`` rounds the running
    sum back to FP16, so the reduction error *grows with K*. The test drives K up and shows the FP16-
    accumulate error overtaking the FP32-accumulate error — the empirical "why FP32 accumulate".

Correctness (tests/test_wmma_gemm.py, gpu): element-wise vs ``torch.matmul`` at FP32-accumulate
tolerance (rtol ~1e-2 for FP16 inputs) on random AND adversarial shapes — K a non-multiple of the
16-wide MMA-K tile, M/N remainder tiles that don't fill a 128-block, and a single large outlier.

Fragment layout is opaque *by design*: WMMA hands each of the 32 threads an "assignment slip" and you
program tiles, not elements — which is exactly why WMMA cannot reach Hopper's WGMMA / Blackwell's
tcgen05 (those need hand-encoded descriptors). That wall is the point of Rung 2+.

Build: JIT via torch.utils.cpp_extension.load_inline; ``-arch`` is read from :func:`common.arch.arch_name`
at runtime so the loader is ISA-agnostic (sm_120/sm_90/sm_100).
"""

from __future__ import annotations

import functools

from torch import Tensor
from torch.utils.cpp_extension import load_inline

# ==================================================================================================
# CUDA source — one templated WMMA kernel, instantiated for float and half accumulators.
# ==================================================================================================
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <type_traits>

using namespace nvcuda;

// Explicit half<->float conversions (torch builds with __CUDA_NO_HALF_CONVERSIONS__).
template <typename AccT> __device__ __forceinline__ AccT from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ half  from_float<half>(float v)  { return __float2half(v); }
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(half v)  { return __half2float(v); }

// --- MMA atom (the hardware tile) ---
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// --- Block tile: a 128x128 output tile, K streamed in BK-deep slabs through SMEM. ---
#define BM 128
#define BN 128
#define BK 32
#define WARPS_M 2
#define WARPS_N 4
#define WARP_SIZE 32
#define NUM_WARPS (WARPS_M * WARPS_N)        // 8
#define NUM_THREADS (NUM_WARPS * WARP_SIZE)  // 256

// Per-warp fragment grid: each warp owns (BM/WARPS_M) x (BN/WARPS_N) = 64x32 => 4x2 of 16x16.
#define WM_FRAGS (BM / (WARPS_M * WMMA_M))   // 4
#define WN_FRAGS (BN / (WARPS_N * WMMA_N))   // 2

// SMEM leading-dim padding (halfs) — 16-byte aligned, ldm stays a multiple of 8, kills bank conflicts.
#define APAD 8
#define BPAD 8

template <typename AccT>
__global__ void wmma_gemm_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ __align__(16) half As[BM][BK + APAD];
    __shared__ __align__(16) half Bs[BK][BN + BPAD];
    // Per-warp staging for the guarded (edge-tile) store; sized for the widest accumulator.
    __shared__ AccT stage[NUM_WARPS][WMMA_M][WMMA_N];

    const int warpId = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warpRow = warpId / WARPS_N;   // 0..WARPS_M-1
    const int warpCol = warpId % WARPS_N;   // 0..WARPS_N-1

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, AccT> acc[WM_FRAGS][WN_FRAGS];
    #pragma unroll
    for (int i = 0; i < WM_FRAGS; ++i)
        #pragma unroll
        for (int j = 0; j < WN_FRAGS; ++j)
            wmma::fill_fragment(acc[i][j], from_float<AccT>(0.0f));

    for (int kk = 0; kk < K; kk += BK) {
        // --- stage A tile (BM x BK), zero-filled out of bounds so the MMA reads valid data ---
        #pragma unroll
        for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS) {
            const int r = idx / BK, c = idx % BK;
            const int gr = rowBase + r, gc = kk + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        // --- stage B tile (BK x BN) ---
        #pragma unroll
        for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS) {
            const int r = idx / BN, c = idx % BN;
            const int gr = kk + r, gc = colBase + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int kf = 0; kf < BK / WMMA_K; ++kf) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[WM_FRAGS];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[WN_FRAGS];
            #pragma unroll
            for (int i = 0; i < WM_FRAGS; ++i) {
                const int aRow = warpRow * (WM_FRAGS * WMMA_M) + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[aRow][kf * WMMA_K], BK + APAD);
            }
            #pragma unroll
            for (int j = 0; j < WN_FRAGS; ++j) {
                const int bCol = warpCol * (WN_FRAGS * WMMA_N) + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[kf * WMMA_K][bCol], BN + BPAD);
            }
            #pragma unroll
            for (int i = 0; i < WM_FRAGS; ++i)
                #pragma unroll
                for (int j = 0; j < WN_FRAGS; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // --- epilogue: store each 16x16 fragment, guarding remainder tiles at the M/N edges ---
    #pragma unroll
    for (int i = 0; i < WM_FRAGS; ++i) {
        #pragma unroll
        for (int j = 0; j < WN_FRAGS; ++j) {
            const int cRow = rowBase + warpRow * (WM_FRAGS * WMMA_M) + i * WMMA_M;
            const int cCol = colBase + warpCol * (WN_FRAGS * WMMA_N) + j * WMMA_N;
            // Fast path needs: whole tile in bounds, FP32 acc == FP32 output, AND an even ldm —
            // wmma's vectorized accumulator store requires >=8-byte-aligned row starts, i.e. N even
            // (float ldm even). An odd N falls through to the scalar-copy staging path (still exact).
            const bool full = (cRow + WMMA_M <= M) && (cCol + WMMA_N <= N) && (N % 2 == 0);
            if (full && std::is_same<AccT, float>::value) {
                // Fast path: whole tile in bounds and FP32 acc matches the FP32 output type.
                wmma::store_matrix_sync(reinterpret_cast<AccT*>(C) + cRow * N + cCol,
                                        acc[i][j], N, wmma::mem_row_major);
            } else {
                // Guarded path: drain to per-warp SMEM, then bounds-checked element copy + convert.
                wmma::store_matrix_sync(&stage[warpId][0][0], acc[i][j], WMMA_N, wmma::mem_row_major);
                __syncwarp();
                #pragma unroll
                for (int e = lane; e < WMMA_M * WMMA_N; e += WARP_SIZE) {
                    const int rr = e / WMMA_N, cc = e % WMMA_N;
                    const int gr = cRow + rr, gc = cCol + cc;
                    if (gr < M && gc < N)
                        C[gr * N + gc] = to_float(stage[warpId][rr][cc]);
                }
                __syncwarp();
            }
        }
    }
}

static torch::Tensor launch(torch::Tensor A, torch::Tensor B, bool fp16_acc) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A, B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf, "A, B must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K mismatch");
    A = A.contiguous();
    B = B.contiguous();
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NUM_THREADS);
    const half* pa = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half* pb = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    float* pc = C.data_ptr<float>();
    if (fp16_acc)
        wmma_gemm_kernel<half><<<grid, block>>>(pa, pb, pc, M, N, K);
    else
        wmma_gemm_kernel<float><<<grid, block>>>(pa, pb, pc, M, N, K);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "wmma_gemm kernel launch failed");
    return C;
}

torch::Tensor wmma_gemm(torch::Tensor A, torch::Tensor B) { return launch(A, B, /*fp16_acc=*/false); }
torch::Tensor wmma_gemm_fp16acc(torch::Tensor A, torch::Tensor B) { return launch(A, B, /*fp16_acc=*/true); }
"""

_CPP_DECL = (
    "torch::Tensor wmma_gemm(torch::Tensor A, torch::Tensor B);\n"
    "torch::Tensor wmma_gemm_fp16acc(torch::Tensor A, torch::Tensor B);\n"
)


@functools.lru_cache(maxsize=1)
def _mod():  # noqa: ANN202 - opaque JIT module handle
    """JIT-compile (and cache) the extension. WMMA fp16 in / fp32|fp16 accumulate.

    The target ISA is read at runtime from :func:`scratch_llm.kernels.common.arch.arch_name` so the
    same loader builds for sm_120 / sm_90 / sm_100 without an edit; it is only reached on the CUDA
    path (dispatch.py::matmul gates on a.is_cuda).
    """
    from scratch_llm.kernels.common.arch import arch_name

    flags = ["-O3", "--use_fast_math"]
    sm = arch_name()
    if sm is not None:
        flags.append(f"-arch={sm}")
    return load_inline(
        name="wmma_gemm_ext",
        cpp_sources=_CPP_DECL,
        cuda_sources=_CUDA_SRC,
        functions=["wmma_gemm", "wmma_gemm_fp16acc"],
        extra_cuda_cflags=flags,
        verbose=False,
    )


def wmma_gemm(a: Tensor, b: Tensor) -> Tensor:
    """C = A @ B with FP16 inputs and an **FP32** WMMA accumulator; returns an ``(M, N)`` fp32 tensor.

    Element-exact vs ``torch.matmul`` at FP32-accumulate tolerance. Handles remainder M/N/K.
    """
    return _mod().wmma_gemm(a, b)


def wmma_gemm_fp16acc(a: Tensor, b: Tensor) -> Tensor:
    """C = A @ B with an **FP16** WMMA accumulator (the anti-example) — error grows with K.

    Same fragment machinery as :func:`wmma_gemm`, but every ``mma_sync`` rounds the running sum back
    to FP16, so a long-K reduction loses precision. Returns fp32 (the accumulated fp16 values, widened).
    """
    return _mod().wmma_gemm_fp16acc(a, b)
