// =============================================================================
// a_r3_paged_decode_sm90.cu — K2/A-R3: paged decode, split-KV, + the log-sum-exp reduce
// =============================================================================
//
// RUNG:   experiments/K2/A-R3/spec.md   ·   MAP: experiments/K2/A-R3/map.md
// FLOOR:  flashinfer BatchDecodeWithPagedKVCacheWrapper at the identical page table.
// TARGET: <= 15% behind FlashInfer and >= 85% of measured HBM (plan §05 K2 A-R3).
//
// At B=64, ctx 8192, one query token, this kernel reads 512 MiB of KV cache and writes 512 KiB.
// There is no arithmetic intensity to speak of; the exit is a fraction of HBM bandwidth for that
// reason, and every structural decision below is a bandwidth decision.
//
// TWO KERNELS LIVE HERE, and the split between them is the rung:
//
//   1. a_r3_paged_decode_partial_sm90 — one CTA per (work item, KV head). A work item is a
//      (request, page-range) pair produced by the host-side partition. Each CTA runs an online
//      softmax over its own page range and writes a PARTIAL: the normalised o, plus one lse.
//   2. a_r3_merge_states_sm90 — one warp per (request, query head). Combines that request's
//      partials with the log-sum-exp merge. This is NOT a hole: it is ordinary arithmetic that
//      upstream also treats as library code (include/flashinfer/attention/state.cuh:52-64), and
//      kernels/attention/decode/paged_split_kv.py carries a Python twin the CPU suite checks it
//      against.
//
// THE HOLE IS NOT IN THIS FILE. Deciding how many chunks each sequence is split into — too few
// and SMs idle behind an 8192-token serial walk, too many and the reduce dominates — is the
// rung's real decision and it lives in Python, where it is a pure function of (pages per request,
// num_kv_heads, max_grid_size) and is testable without silicon:
//     kernels/attention/decode/paged_split_kv.py :: plan_partition
// This file therefore compiles clean, with no -DHUY_STUB_KERNEL_BODY and no #error. What it
// cannot do until the hole is filled is pick its own grid.
//
// THE THREE THINGS THIS FILE GETS RIGHT ON PURPOSE:
//
//   * K/V READ ONCE PER GQA GROUP. blockDim.y IS the query head within the group (bdy ==
//     GROUP_SIZE == 8), so one page reaches shared memory once from HBM and is read eight times
//     from smem. Read per query head instead and the kernel is 8x off the roofline while every
//     output element stays correct — the failure mode a correctness test cannot see. Upstream
//     does the same thing with the same trick (decode.cuh:422 qo_head_idx = kv_head*bdy + ty,
//     load indexed by token at :503,:512, compute reading the whole tile back at :541).
//   * ONE PIPELINE STAGE IS ONE PAGE. TILE_TOKENS == PAGE == 16, so a staged tile never straddles
//     a page boundary: the page id is resolved through kv_indices once per tile instead of once
//     per token, and the inner loop has no divmod in it at all. Costs generality (the launcher
//     rejects page_size != 16 by name) and buys the whole address computation.
//   * BASE 2, ON BOTH SIDES OF THE MERGE. Logits are pre-scaled by sm_scale*log2(e) on the host,
//     exponentiated with ex2.approx, and a partial's lse is m + log2(d) (flashinfer
//     variants.cuh:54, state.cuh:45). The merge assumes the same base — cascade.cuh:641 says so —
//     and an ln-based partial produces a merged output that is wrong, finite, and plausible.
//
// BUILDING IT (no GPU required; infra/drydock.sh does this automatically):
//   nvcc -arch=sm_90a -cubin -O3 -Xptxas -v csrc/attention/a_r3_paged_decode_sm90.cu -o /tmp/o.cubin
// =============================================================================

#include <cstdint>
#include <cuda_bf16.h>

#ifndef HEAD_DIM
#define HEAD_DIM 128        // the plan's row; frozen, not templated
#endif
#ifndef PAGE
#define PAGE 16             // tokens per page — the plan's row
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 8        // query heads per KV head — GQA 8:1, the plan's row
#endif
#ifndef STAGES
#define STAGES 2            // cp.async depth; upstream also uses 2 for sm >= 80 (utils.cuh:330-332)
#endif

#define VEC 8                        // bf16 elements per 16-byte lane load — the widest access
#define BDX (HEAD_DIM / VEC)         // 16 lanes span one head's 128 features
#define BDY GROUP_SIZE               // 8: threadIdx.y IS the query head inside the GQA group
#define THREADS (BDX * BDY)          // 128
#define TILE_TOKENS PAGE             // one stage stages exactly one page
#define TILE_VECS (TILE_TOKENS * HEAD_DIM / VEC)   // 256 sixteen-byte chunks per K (or V) tile
#define VECS_PER_THREAD (TILE_VECS / THREADS)      // 2
#define TILE_BYTES (TILE_TOKENS * HEAD_DIM * 2)    // 4096 B, one K (or V) tile
#define SMEM_BYTES (2 * STAGES * TILE_BYTES)       // 16384 — under 48 KB, so NO opt-in needed

#define MERGE_THREADS 32                                 // one warp per (request, query head)
#define MERGE_ELEMS (HEAD_DIM / MERGE_THREADS)           // 4 floats per lane, 128 B coalesced

// cp.async and the 16-lane butterfly are sm_80+, but this rung is routed to H100 and the rest of
// the K2 ladder is built -arch=sm_90a. Guarding on the exact arch keeps a multi-arch AOT build
// LINKING (the signature stays arch-independent) while making the body inert off Hopper; the host
// launcher checks the device's capability and refuses long before an inert kernel could run.
#define A_R3_HAS_ISA (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900))

#if A_R3_HAS_ISA

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// --- 8 bf16 in one 16-byte word ---------------------------------------------------------
// A named union, not a cast at the use site, so ptxas keeps it in registers. Its reason to exist
// is a SASS observation, not tidiness: reading the shared K/V tile element by element compiles to
// 256 `LDS.U16` per tile — 2 bytes per lane per instruction. That would be a rounding error in a
// dense kernel, but here GQA multiplies every shared read by GROUP_SIZE = 8 (each of the 8 query
// heads re-reads the same tile), so the CTA pulls 64 KB out of smem for every 8 KB it pulled from
// HBM. At 2 B per instruction the shared pipe, not HBM, becomes the wall — and this rung's exit
// is stated in % of HBM. One `LDS.128` per lane instead of eight `LDS.U16` is the fix.
union bf16x8 {
    uint4 raw;
    __nv_bfloat162 h[4];
};

// --- base-2 exponential and logarithm, as PTX ---------------------------------------------
// Written as inline asm rather than exp2f/log2f so the instruction does not depend on whether the
// translation unit was compiled with --use_fast_math: the AOT build passes it (CMakeLists.txt) and
// the dry dock does not, and a rung whose numerics change with a build flag is a rung whose
// tolerance means nothing. Identical to flashinfer math.cuh:47-50 and :172-175.
// .ftz matters at the edge: ex2.approx.ftz.f32(-inf) is exactly 0, which is what makes a fully
// masked score contribute nothing instead of a denormal.
__device__ __forceinline__ float a_r3_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float a_r3_log2(float x) {
    float y;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// --- cp.async ------------------------------------------------------------------------------
// 16 bytes, global -> shared, .cg = bypass L1 and land in L2. The KV cache is streamed exactly
// once per decode and never reused within a launch, so an L1 line of it is a line evicted from
// something that would have been reused. Upstream reaches the same instruction through
// cp_async::pred_load<128> (decode.cuh:501-516).
__device__ __forceinline__ void cp_async_16B(void* dst_smem, const void* src_gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_u32(dst_smem)), "l"(src_gmem) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait until at most N groups are still in flight. N is an immediate, hence the template.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
#endif  // A_R3_HAS_ISA

// =============================================================================
// Kernel 1 — the split-KV partial.
//   grid  = (num_work_items, num_kv_heads)        block = (BDX, BDY) = 128 threads
//   out   = tmp_o [T, H_q, D] fp32 (normalised)   tmp_lse [T, H_q] fp32 (base 2)
//
// Why the partial is (o/d, lse) and not (o, m, d): two floats per (chunk, head) cross to global
// instead of three, halving both the workspace and the traffic the reduce has to move — and the
// reduce can rebuild the pair exactly, because lse = m + log2(d) carries both. Upstream's merge
// does the same reconstruction with `st.merge(v, s, /*other_d=*/1)` (cascade.cuh:446).
// =============================================================================
extern "C" __global__ void __launch_bounds__(THREADS)
a_r3_paged_decode_partial_sm90(const __nv_bfloat16* __restrict__ q,        // [B, H_q, D]
                               const __nv_bfloat16* __restrict__ k_cache,  // [P, PAGE, H_kv, D]
                               const __nv_bfloat16* __restrict__ v_cache,  // [P, PAGE, H_kv, D]
                               const int* __restrict__ kv_indptr,          // [B+1]
                               const int* __restrict__ kv_indices,         // [nnz]
                               const int* __restrict__ last_page_len,      // [B]
                               const int* __restrict__ request_indices,    // [T]
                               const int* __restrict__ chunk_indices,      // [T]
                               float* __restrict__ tmp_o,                  // [T, H_q, D]
                               float* __restrict__ tmp_lse,                // [T, H_q]
                               int num_kv_heads,
                               int chunk_size_pages,
                               float scale_log2e) {
#if !A_R3_HAS_ISA
    // Inert off sm_90a so a multi-arch AOT build links; the launcher refuses before this can run.
    (void)q; (void)k_cache; (void)v_cache; (void)kv_indptr; (void)kv_indices; (void)last_page_len;
    (void)request_indices; (void)chunk_indices; (void)tmp_o; (void)tmp_lse; (void)num_kv_heads;
    (void)chunk_size_pages; (void)scale_log2e;
#else
    const int tx = threadIdx.x;                 // 0..BDX-1: which 8 features of the head
    const int ty = threadIdx.y;                 // 0..BDY-1: WHICH QUERY HEAD of the GQA group
    const int tid = tx + BDX * ty;              // 0..127, used only by the cooperative loader
    const int work = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int num_qo_heads = num_kv_heads * GROUP_SIZE;
    const int qo_head = kv_head * GROUP_SIZE + ty;

    const int b = request_indices[work];
    const int chunk = chunk_indices[work];
    const int page_lo = kv_indptr[b];
    const int n_pages = kv_indptr[b + 1] - page_lo;
    const int c0 = chunk * chunk_size_pages;
    const int c1 = min(c0 + chunk_size_pages, n_pages);

    // K and V, double buffered. STATIC shared memory: 16 KB is under the 48 KB a launch gets for
    // free, so there is no cudaFuncSetAttribute opt-in to forget, and the figure shows up in the
    // ptxas report the dry dock captures instead of being invisible as a dynamic allocation.
    __shared__ __align__(16) __nv_bfloat16 k_smem[STAGES][TILE_TOKENS * HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 v_smem[STAGES][TILE_TOKENS * HEAD_DIM];

    // The query row, in registers, ALREADY in log2 units. Folding sm_scale*log2(e) into q rather
    // than into each score is HEAD_DIM multiplies against TILE_TOKENS*chunk multiplies, and it is
    // what makes the exponential below a single ex2 (flashinfer does the same, variants.cuh:54).
    float qr[VEC];
    {
        bf16x8 qw;
        qw.raw = *reinterpret_cast<const uint4*>(
            q + (static_cast<size_t>(b) * num_qo_heads + qo_head) * HEAD_DIM + tx * VEC);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float2 f = __bfloat1622float2(qw.h[t]);
            qr[2 * t] = f.x * scale_log2e;
            qr[2 * t + 1] = f.y * scale_log2e;
        }
    }

    float acc[VEC];
#pragma unroll
    for (int e = 0; e < VEC; ++e) acc[e] = 0.0f;
    float m = -INFINITY;
    float d = 0.0f;

    // --- the cooperative page loader ------------------------------------------------------
    // 128 threads x 2 chunks of 16 B covers one 16x128 bf16 page tile. Consecutive tid walk the
    // feature axis first, so lanes 0..15 issue 16 contiguous 16-byte requests = one 256 B
    // transaction per token. At H_kv = 4 the next token is 1024 B away; that stride is inherent
    // to NHD (page.cuh:200-205) and is why the load is issued per token rather than per page.
    auto issue_page = [&](int logical_page, int stage) {
        if (logical_page >= c1) return;                       // past the chunk: nothing to stage
        const int phys = kv_indices[page_lo + logical_page];  // the block table read; never skip it
#pragma unroll
        for (int r = 0; r < VECS_PER_THREAD; ++r) {
            const int c = tid + r * THREADS;                  // 0..TILE_VECS-1
            const int token = c / (HEAD_DIM / VEC);
            const int feat = (c % (HEAD_DIM / VEC)) * VEC;
            const size_t goff =
                ((static_cast<size_t>(phys) * PAGE + token) * num_kv_heads + kv_head) * HEAD_DIM +
                feat;
            const int soff = token * HEAD_DIM + feat;
            cp_async_16B(&k_smem[stage][soff], k_cache + goff);
            cp_async_16B(&v_smem[stage][soff], v_cache + goff);
        }
    };

    // Prologue: exactly STAGES groups, some possibly empty. Committing an empty group rather than
    // skipping the commit is what keeps `cp.async.wait_group<STAGES-1>` below meaning "the tile I
    // am about to read has landed" even for a chunk shorter than STAGES pages. Skip the commit
    // instead and a one-page chunk reads an unwritten tile.
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        issue_page(c0 + s, s);
        cp_async_commit();
    }

    for (int p = c0; p < c1; ++p) {
        const int stage = (p - c0) % STAGES;
        cp_async_wait<STAGES - 1>();
        __syncthreads();  // cp.async.wait_group is per-thread; the tile is a block-wide object

        // Valid tokens in THIS page: full, unless it is the request's last page.
        const int valid = (p == n_pages - 1) ? last_page_len[b] : TILE_TOKENS;

        // Pass 1 — all TILE_TOKENS scores, so the tile costs ONE accumulator rescale instead of
        // TILE_TOKENS of them. The alternative (rescale per token) is the textbook online softmax
        // and is 16x the epilogue work for identical output.
        float s[TILE_TOKENS];
#pragma unroll
        for (int j = 0; j < TILE_TOKENS; ++j) {
            bf16x8 kw;
            kw.raw = *reinterpret_cast<const uint4*>(&k_smem[stage][j * HEAD_DIM + tx * VEC]);
            float dot = 0.0f;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float2 f = __bfloat1622float2(kw.h[t]);
                dot = fmaf(qr[2 * t], f.x, dot);
                dot = fmaf(qr[2 * t + 1], f.y, dot);
            }
            // Butterfly over the BDX=16 lanes that span one head. Lane id is tx + 16*ty mod 32, so
            // the 16 lanes of a given ty are a contiguous half-warp and masks 1..8 never cross
            // into the other query head's half. An all-reduce, not a reduce: every lane needs the
            // score to rescale its own slice of the accumulator.
#pragma unroll
            for (int off = BDX >> 1; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, off);
            s[j] = (j < valid) ? dot : -INFINITY;
        }

        float m_tile = -INFINITY;
#pragma unroll
        for (int j = 0; j < TILE_TOKENS; ++j) m_tile = fmaxf(m_tile, s[j]);
        const float m_new = fmaxf(m, m_tile);
        // fmaxf(x, -inf) is the NaN guard, not a clamp: (-inf) - (-inf) is NaN, and fmaxf returns
        // the non-NaN operand, so a fully masked tile contributes exactly 0 instead of poisoning
        // the accumulator. flashinfer relies on the identical trick (state.cuh:57-59).
        const float alpha = a_r3_exp2(fmaxf(m - m_new, -INFINITY));
        d *= alpha;
#pragma unroll
        for (int e = 0; e < VEC; ++e) acc[e] *= alpha;

#pragma unroll
        for (int j = 0; j < TILE_TOKENS; ++j) {
            const float pw = a_r3_exp2(fmaxf(s[j] - m_new, -INFINITY));
            d += pw;
            bf16x8 vw;
            vw.raw = *reinterpret_cast<const uint4*>(&v_smem[stage][j * HEAD_DIM + tx * VEC]);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float2 f = __bfloat1622float2(vw.h[t]);
                acc[2 * t] = fmaf(pw, f.x, acc[2 * t]);
                acc[2 * t + 1] = fmaf(pw, f.y, acc[2 * t + 1]);
            }
        }
        m = m_new;

        __syncthreads();          // nobody may still be reading `stage` when it is refilled
        issue_page(p + STAGES, stage);
        cp_async_commit();        // exactly one group per iteration, empty or not
    }

    // Epilogue: normalise here, so the reduce moves two floats per (chunk, head) instead of three.
    const float d_rcp = (d > 0.0f) ? __fdividef(1.0f, d) : 0.0f;
    float* op = tmp_o + (static_cast<size_t>(work) * num_qo_heads + qo_head) * HEAD_DIM + tx * VEC;
    // Two 16-byte stores: the offset is tx*32 B inside a 128-float row, so a scalar loop would
    // issue eight 4-byte strided stores per lane. The partials are ~1% of this kernel's traffic,
    // but a strided store is also a wrong-looking ncu row on the day the profile is read.
    float4 lo, hi;
    lo.x = acc[0]; lo.y = acc[1]; lo.z = acc[2]; lo.w = acc[3];
    hi.x = acc[4]; hi.y = acc[5]; hi.z = acc[6]; hi.w = acc[7];
    lo.x *= d_rcp; lo.y *= d_rcp; lo.z *= d_rcp; lo.w *= d_rcp;
    hi.x *= d_rcp; hi.y *= d_rcp; hi.z *= d_rcp; hi.w *= d_rcp;
    *reinterpret_cast<float4*>(op) = lo;
    *reinterpret_cast<float4*>(op + 4) = hi;
    if (tx == 0) {
        tmp_lse[static_cast<size_t>(work) * num_qo_heads + qo_head] =
            (d > 0.0f) ? (m + a_r3_log2(d)) : -INFINITY;
    }
#endif  // A_R3_HAS_ISA
}

// =============================================================================
// Kernel 2 — the reduce. NOT a hole: the log-sum-exp merge.
//   grid = (batch, num_qo_heads)   block = MERGE_THREADS (one warp)
//
//   m   = max_c lse_c ;  f_c = 2^(lse_c - m)
//   o   = sum_c f_c o_c / sum_c f_c        (o_c is already divided by its own chunk's d)
//   lse = m + log2(sum_c f_c)
//
// Subtracting the max is not hygiene. Chunk maxima differ by the spread of the logits over the
// whole context; at ctx 8192 a gap of 100 in log2 units is ordinary and 2^100 overflows fp32 on
// the second chunk. The merge state (m, d) is scalar and every lane computes it from the same
// lse values, so there is no cross-lane communication in this kernel at all.
// =============================================================================
extern "C" __global__ void __launch_bounds__(MERGE_THREADS)
a_r3_merge_states_sm90(const float* __restrict__ tmp_o,     // [T, H_q, D]
                       const float* __restrict__ tmp_lse,   // [T, H_q]
                       const int* __restrict__ o_indptr,    // [B+1]
                       __nv_bfloat16* __restrict__ o,       // [B, H_q, D]
                       float* __restrict__ lse,             // [B, H_q]
                       int num_qo_heads) {
#if !A_R3_HAS_ISA
    (void)tmp_o; (void)tmp_lse; (void)o_indptr; (void)o; (void)lse; (void)num_qo_heads;
#else
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    const int lo = o_indptr[b], hi = o_indptr[b + 1];

    float acc[MERGE_ELEMS];
#pragma unroll
    for (int e = 0; e < MERGE_ELEMS; ++e) acc[e] = 0.0f;
    float m = -INFINITY, d = 0.0f;

    for (int c = lo; c < hi; ++c) {
        const float s = tmp_lse[static_cast<size_t>(c) * num_qo_heads + h];
        const float m_new = fmaxf(m, s);
        const float f_prev = a_r3_exp2(fmaxf(m - m_new, -INFINITY));
        const float f_new = a_r3_exp2(fmaxf(s - m_new, -INFINITY));
        // other_d = 1: the chunk's own d is already folded into its lse and out of its o
        // (cascade.cuh:446). Using d_c here as well would divide by it twice.
        d = d * f_prev + f_new;
        const float* src =
            tmp_o + (static_cast<size_t>(c) * num_qo_heads + h) * HEAD_DIM + lane;
#pragma unroll
        for (int e = 0; e < MERGE_ELEMS; ++e) {
            // lane strides by MERGE_THREADS, so one instruction reads 32 consecutive floats.
            acc[e] = acc[e] * f_prev + src[e * MERGE_THREADS] * f_new;
        }
        m = m_new;
    }

    const float d_rcp = (d > 0.0f) ? __fdividef(1.0f, d) : 0.0f;
    __nv_bfloat16* dst = o + (static_cast<size_t>(b) * num_qo_heads + h) * HEAD_DIM + lane;
#pragma unroll
    for (int e = 0; e < MERGE_ELEMS; ++e)
        dst[e * MERGE_THREADS] = __float2bfloat16(acc[e] * d_rcp);
    if (lane == 0) {
        lse[static_cast<size_t>(b) * num_qo_heads + h] =
            (d > 0.0f) ? (m + a_r3_log2(d)) : -INFINITY;
    }
#endif  // A_R3_HAS_ISA
}

// =============================================================================
// HOST LAUNCHERS — what pybind.cpp binds and the Python wrapper calls.
// =============================================================================
#ifdef TORCH_EXTENSION_NAME
#include <torch/extension.h>

#include <vector>

namespace {

void check_index(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == at::kInt, "a_r3_paged_decode: ", name,
                " must be int32 — flashinfer type-checks its index tensors rather than coercing "
                "them (csrc/batch_decode.cu:45), so an int64 table is refused by the floor and "
                "accepted here, which is the worst possible asymmetry; got ", t.scalar_type());
    TORCH_CHECK(t.is_cuda(), "a_r3_paged_decode: ", name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), "a_r3_paged_decode: ", name, " must be contiguous");
}

// The reduce, as a host function, so a_r3_paged_decode and the separately-bound merge share one
// launch site. Splitting them would let the two drift on grid shape — and the merge is exactly
// the kind of kernel whose grid is "obviously" right in two different ways.
std::vector<torch::Tensor> launch_merge(const torch::Tensor& tmp_o, const torch::Tensor& tmp_lse,
                                        const torch::Tensor& o_indptr) {
    const int64_t num_qo_heads = tmp_o.size(1);
    const int64_t batch = o_indptr.numel() - 1;
    auto o = torch::empty({batch, num_qo_heads, HEAD_DIM},
                          tmp_o.options().dtype(torch::kBFloat16));
    auto lse = torch::empty({batch, num_qo_heads}, tmp_o.options().dtype(torch::kFloat32));
    const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(num_qo_heads));
    a_r3_merge_states_sm90<<<grid, MERGE_THREADS>>>(
        tmp_o.data_ptr<float>(), tmp_lse.data_ptr<float>(), o_indptr.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>()), lse.data_ptr<float>(),
        static_cast<int>(num_qo_heads));
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "a_r3_merge_states launch failed: ", cudaGetErrorString(err));
    return {o, lse};
}

}  // namespace

// tmp_o [T, H_q, D] fp32 + tmp_lse [T, H_q] fp32 + o_indptr [B+1] int32 -> {o bf16, lse fp32}.
std::vector<torch::Tensor> a_r3_merge_states(torch::Tensor tmp_o, torch::Tensor tmp_lse,
                                             torch::Tensor o_indptr) {
    TORCH_CHECK(tmp_o.is_cuda() && tmp_lse.is_cuda(), "a_r3_merge_states: partials must be CUDA");
    TORCH_CHECK(tmp_o.scalar_type() == at::kFloat && tmp_lse.scalar_type() == at::kFloat,
                "a_r3_merge_states: partials must be fp32 — the merge rescales by 2^(lse_c - m) "
                "before it sums, and a bf16 partial spends its 8 mantissa bits on a value that is "
                "not the answer");
    TORCH_CHECK(tmp_o.dim() == 3 && tmp_o.size(2) == HEAD_DIM,
                "a_r3_merge_states: tmp_o must be [T, H_q, ", HEAD_DIM, "]");
    TORCH_CHECK(tmp_lse.dim() == 2 && tmp_lse.size(0) == tmp_o.size(0) &&
                    tmp_lse.size(1) == tmp_o.size(1),
                "a_r3_merge_states: tmp_lse must be [T, H_q] matching tmp_o");
    check_index(o_indptr, "o_indptr");
    return launch_merge(tmp_o.contiguous(), tmp_lse.contiguous(), o_indptr);
}

// The rung's entry point. Returns {o [B, H_q, D] bf16, lse [B, H_q] fp32 (base 2)}.
std::vector<torch::Tensor> a_r3_paged_decode(torch::Tensor q, torch::Tensor k_cache,
                                             torch::Tensor v_cache, torch::Tensor kv_indptr,
                                             torch::Tensor kv_indices,
                                             torch::Tensor last_page_len,
                                             torch::Tensor request_indices,
                                             torch::Tensor chunk_indices, torch::Tensor o_indptr,
                                             int64_t chunk_size_pages, double sm_scale) {
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
                "a_r3_paged_decode: q and the KV pool must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k_cache.scalar_type() == at::kBFloat16 &&
                    v_cache.scalar_type() == at::kBFloat16,
                "a_r3_paged_decode: bf16 only — the FlashInfer floor is measured at bf16, and a "
                "different dtype changes the byte count the % of HBM is computed from");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == HEAD_DIM,
                "a_r3_paged_decode: q must be [B, H_q, ", HEAD_DIM, "]; got ", q.sizes());
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "a_r3_paged_decode: K and V pools must be NHD [P, PAGE, H_kv, D] and identical in "
                "shape — flashinfer requires matched strides too (csrc/batch_decode.cu:176-182)");
    TORCH_CHECK(k_cache.size(1) == PAGE,
                "a_r3_paged_decode: page_size must be ", PAGE,
                " at this rung — one pipeline stage IS one page, which is what removes the "
                "page-boundary divmod from the inner loop; got ", k_cache.size(1));
    TORCH_CHECK(k_cache.size(3) == HEAD_DIM, "a_r3_paged_decode: head_dim must be ", HEAD_DIM);
    const int64_t batch = q.size(0);
    const int64_t num_qo_heads = q.size(1);
    const int64_t num_kv_heads = k_cache.size(2);
    TORCH_CHECK(num_qo_heads == num_kv_heads * GROUP_SIZE,
                "a_r3_paged_decode: this rung is GQA ", GROUP_SIZE,
                ":1 — blockDim.y is the query head within the group, so num_qo_heads must be ",
                GROUP_SIZE, " * num_kv_heads; got ", num_qo_heads, " and ", num_kv_heads);
    for (const auto& p : {std::make_pair(&kv_indptr, "kv_indptr"),
                          std::make_pair(&kv_indices, "kv_indices"),
                          std::make_pair(&last_page_len, "last_page_len"),
                          std::make_pair(&request_indices, "request_indices"),
                          std::make_pair(&chunk_indices, "chunk_indices"),
                          std::make_pair(&o_indptr, "o_indptr")}) {
        check_index(*p.first, p.second);
    }
    TORCH_CHECK(kv_indptr.numel() == batch + 1 && o_indptr.numel() == batch + 1,
                "a_r3_paged_decode: kv_indptr and o_indptr must both hold batch + 1 = ", batch + 1,
                " entries");
    TORCH_CHECK(last_page_len.numel() == batch,
                "a_r3_paged_decode: last_page_len must hold one entry per request");
    const int64_t work_items = request_indices.numel();
    TORCH_CHECK(work_items == chunk_indices.numel() && work_items > 0,
                "a_r3_paged_decode: request_indices and chunk_indices must be the same non-empty "
                "length — they are one work list, one row per (request, page range)");
    TORCH_CHECK(chunk_size_pages >= 1,
                "a_r3_paged_decode: chunk_size_pages must be >= 1; got ", chunk_size_pages);

    q = q.contiguous();
    k_cache = k_cache.contiguous();
    v_cache = v_cache.contiguous();

    // The partials. fp32 both, and sized from the split count — so the partition decision is also
    // a workspace decision: each extra chunk per request costs num_qo_heads*(HEAD_DIM+1) floats.
    auto tmp_o = torch::empty({work_items, num_qo_heads, HEAD_DIM},
                              q.options().dtype(torch::kFloat32));
    auto tmp_lse = torch::empty({work_items, num_qo_heads}, q.options().dtype(torch::kFloat32));

    const dim3 grid(static_cast<unsigned>(work_items), static_cast<unsigned>(num_kv_heads));
    const dim3 block(BDX, BDY);
    // log2(e) folded in on the host: the kernel's exponential is then one ex2.approx, and the
    // merge's base-2 assumption holds by construction rather than by convention.
    const float scale_log2e = static_cast<float>(sm_scale) * 1.4426950408889634f;
    a_r3_paged_decode_partial_sm90<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>()),
        kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(), last_page_len.data_ptr<int>(),
        request_indices.data_ptr<int>(), chunk_indices.data_ptr<int>(), tmp_o.data_ptr<float>(),
        tmp_lse.data_ptr<float>(), static_cast<int>(num_kv_heads),
        static_cast<int>(chunk_size_pages), scale_log2e);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "a_r3_paged_decode launch failed: ", cudaGetErrorString(err));

    // The merge runs unconditionally, even for a one-chunk request where it degenerates to a
    // normalise-and-cast. Skipping it would need a host read of o_indptr — a sync inside the
    // timed window — to save 4 MiB of traffic against 512 MiB of KV. Not a trade worth a branch.
    return launch_merge(tmp_o, tmp_lse, o_indptr);
}
#endif  // TORCH_EXTENSION_NAME
