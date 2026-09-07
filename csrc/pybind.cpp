// =============================================================================
// pybind.cpp — the single AOT extension entry point for scratch_llm kernels.
// =============================================================================
//
// Production kernel stacks (CUTLASS, xformers, FlashAttention, Mamba) ship ONE
// torch extension that exposes every C++ kernel symbol, rather than per-kernel
// JIT loads. This file is that single registration: the CMakeLists.txt /
// setup.py builds it into `_scratch_llm_kernels.{so,pyd}`, and the Python
// loaders under src/scratch_llm/kernels/ reach it via:
//
//     from scratch_llm import _scratch_llm_kernels as _ext
//
// (wrapped in try/except so the JIT path still works on a dev box without AOT).
//
// WHY ONE EXTENSION, not N: one compile pass per arch, one symbol table, one
// torch.library registration point. Per-kernel extensions re-link torch headers
// N times — a research-grade smell this file replaces.
//
// CPU-SAFETY (the kernels/ package invariant): this translation unit is NEVER
// compiled on a CPU-only box. The build is gated behind `[gpu-aot]` (setup.py)
// and the CMake path is opt-in (`-DTARGET_ARCHS=sm_90a;sm_100a`). `import
// scratch_llm` on a CPU box never reaches this code.
//
// WHAT IS AOT vs JIT TODAY:
//   * Frontier kernels (below) — AOT from csrc/. These are the value of the
//     AOT build: WGMMA/tcgen05/FA3 need sm_90a/sm_100a and the rental box.
//   * Shipped learning rungs (mma_sync, smem_tiled, wmma) — stay JIT. Their
//     .cu sources live co-located with the .py loader (for the JIT sibling
//     lookup), and wmma's source is an inline Python string. AOT-building them
//     would mean either duplicating the .cu into csrc/ or extracting the wmma
//     string — both are behavior churn for no perf gain (they run on the sm_120
//     dev box, not the rental frontier box). Left as a documented TODO.
//
// ADDING A NEW C++ KERNEL (the 3-step recipe):
//   1. Write the .cu under csrc/<family>/; declare its launcher below.
//   2. Add the .cu to _SOURCES in ../CMakeLists.txt (and setup.py if pip-built).
//   3. Bind it here: `m.def("my_kernel", &my_kernel);`
// The Python loader in src/scratch_llm/kernels/<family>/<backend>/ then swaps
// its _module() to prefer the AOT extension (see the swap pattern in
// kernels/gemm/cuda/mma_sync.py — the same pattern every frontier loader uses).
// =============================================================================

#include <torch/extension.h>

// --- Frontier rungs promoted from performance/rental/kernels/ (WS-B) -------
// sm_90a (Hopper) warpgroup-MMA GEMM. Compile-gated; runtime correctness
// deferred to the H100/H200 rental day (see csrc/gemm/wgmma_sm90.cu header).
torch::Tensor wgmma_gemm_sm90(torch::Tensor A, torch::Tensor B);
// sm_100a (Blackwell datacenter) tcgen05 / UMMA GEMM. Compile-gated for B200.
torch::Tensor tcgen05_gemm_sm100(torch::Tensor A, torch::Tensor B);
// sm_90a (Hopper) FlashAttention-3 forward. Compile-gated for H100/H200.
torch::Tensor flash_attention_fa3_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool is_causal);

// --- K1 ladder rungs (ladders plan §05; one .cu per rung, one hole per rung) ----
// bf16 in, fp32 accumulate, fp32 out — the K1 floor is cuBLAS bf16, so bf16 is not a variant.
// Each is arch-guarded internally: off its own architecture the device body compiles to nothing
// and the host launcher refuses, so a multi-arch build links instead of failing.
torch::Tensor h_r1_wgmma_bf16(torch::Tensor A, torch::Tensor B);
torch::Tensor h_r2_tma_bf16(torch::Tensor A, torch::Tensor B);
torch::Tensor h_r3_ws_bf16(torch::Tensor A, torch::Tensor B);
torch::Tensor h_r4_persistent_bf16(torch::Tensor A, torch::Tensor B);
// B-R6 takes packed e2m1 operands plus their swizzled e4m3 scale tensors (kernels/gemm/nvfp4_layout.py).
torch::Tensor b_r6_nvfp4_mma_sync(torch::Tensor A, torch::Tensor SFA, torch::Tensor B, torch::Tensor SFB);

// --- K2 ladder rungs (ladders plan §05) ------------------------------------
// Both return {O, LSE}: upstream gates the log-sum-exp as well as the output (flashinfer
// tests/attention/test_hopper.py:55-56), and an O-only comparison is weaker than the floor's.
// <vector> comes in with torch/extension.h; named here because these two depend on it directly.
#include <vector>
std::vector<torch::Tensor> fa3_hopper_v2_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                             bool is_causal, bool heads_last);
std::vector<torch::Tensor> a_r3_paged_decode(torch::Tensor q, torch::Tensor k_cache,
                                             torch::Tensor v_cache, torch::Tensor kv_indptr,
                                             torch::Tensor kv_indices, torch::Tensor last_page_len,
                                             torch::Tensor request_indices,
                                             torch::Tensor chunk_indices, torch::Tensor o_indptr,
                                             int64_t chunk_size_pages, double sm_scale);
std::vector<torch::Tensor> a_r3_merge_states(torch::Tensor tmp_o, torch::Tensor tmp_lse,
                                             torch::Tensor o_indptr);

// --- Frontier stubs (WS-C) — bodies are the learning rep, unimplemented ----
// Each throws a C++ runtime_error until you implement the mainloop. Declared
// here so dispatch + Python loaders wire NOW; bodies land as learning reps.
torch::Tensor fp8_gemm_sm90(torch::Tensor A, torch::Tensor B);
torch::Tensor stream_k_gemm_sm90(torch::Tensor A, torch::Tensor B);
torch::Tensor persistent_gemv_sm90(torch::Tensor x, torch::Tensor A);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "scratch_llm AOT kernel extension (frontier rungs)";

    // --- Promoted rungs (WS-B) — compile-gated, runtime deferred to rental ---
    m.def("wgmma_gemm_sm90", &wgmma_gemm_sm90,
          "Hopper WGMMA GEMM (sm_90a, compile-gated; rental-day runtime)");
    m.def("tcgen05_gemm_sm100", &tcgen05_gemm_sm100,
          "Blackwell tcgen05 GEMM (sm_100a, compile-gated; rental-day runtime)");
    m.def("flash_attention_fa3_forward", &flash_attention_fa3_forward,
          "Hopper FlashAttention-3 fwd (sm_90a, compile-gated; rental-day runtime)");

    // --- K1 ladder rungs (experiments/K1/<rung>/spec.md) ---
    m.def("h_r1_wgmma_bf16", &h_r1_wgmma_bf16,
          "K1/H-R1: wgmma from smem, single stage, 128x128 tile (sm_90a, bf16->fp32)");
    m.def("h_r2_tma_bf16", &h_r2_tma_bf16,
          "K1/H-R2: TMA + mbarrier expect-tx, 2 stages (sm_90a, bf16->fp32)");
    m.def("h_r3_ws_bf16", &h_r3_ws_bf16,
          "K1/H-R3: warp-specialised multistage, 128x256 tile (sm_90a, bf16->fp32)");
    m.def("h_r4_persistent_bf16", &h_r4_persistent_bf16,
          "K1/H-R4: persistent + tile scheduler, cluster of 2 (sm_90a, bf16->fp32)");
    m.def("b_r6_nvfp4_mma_sync", &b_r6_nvfp4_mma_sync,
          "K1/B-R6: NVFP4 block-scaled mma.sync, swizzled e4m3 scales (sm_120a, e2m1->fp32)");

    // --- K2 ladder rungs (experiments/K2/<rung>/spec.md) ---
    m.def("fa3_hopper_v2_fwd", &fa3_hopper_v2_fwd,
          "K2/A-R2: TMA + warp-specialised ping-pong FA3-shaped fwd (sm_90a, bf16) -> {O, LSE}",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("is_causal"),
          py::arg("heads_last") = false);
    m.def("a_r3_paged_decode", &a_r3_paged_decode,
          "K2/A-R3: paged split-KV decode partials + merge (sm_90a, bf16) -> {O, LSE}");
    m.def("a_r3_merge_states", &a_r3_merge_states,
          "K2/A-R3: the log-sum-exp merge alone, so the reduce can be tested without the partials");

    // --- Stubs (WS-C) — raise at runtime until the mainloop is implemented ---
    m.def("fp8_gemm_sm90", &fp8_gemm_sm90, "Hopper FP8 GEMM (STUB — learning rep)");
    m.def("stream_k_gemm_sm90", &stream_k_gemm_sm90, "Hopper stream-K GEMM (STUB — learning rep)");
    m.def("persistent_gemv_sm90", &persistent_gemv_sm90,
          "Hopper persistent GEMV (STUB — learning rep)");
}
