# =============================================================================
# CMakeLists.txt — the AOT build for the scratch_llm frontier-kernel extension.
# =============================================================================
#
# This is the production kernel build: it compiles every frontier .cu under
# csrc/ into ONE torch extension (_scratch_llm_kernels.so / .pyd), the way
# CUTLASS / xformers / FlashAttention ship.
#
# The dev-iteration path (JIT via torch.utils.cpp_extension) stays the default
# for the learning rungs; this AOT build is for the rental box where you want
# the frontier kernels (WGMMA / tcgen05 / FA3) built for sm_90a / sm_100a.
#
# ------------------------------------------------------------------------------
# BUILD (on the rental box — H100, H200, or B200):
#
#   cmake -B csrc/build -DTARGET_ARCHS="sm_90a;sm_100a"
#   cmake --build csrc/build -j
#
# Or the pip path (see setup.py):
#
#   TORCH_CUDA_ARCH_LIST="9.0a;10.0a" pip install -e ".[gpu-aot]"
#
# ------------------------------------------------------------------------------
# TARGET ARCHITECTURES (the rental frontier):
#
#   sm_90a   Hopper (H100 / H200) — WGMMA + TMA. Needs CUDA 12.4+.
#   sm_100a  Blackwell datacenter (B200) — tcgen05 + TMEM. Needs CUDA 13.0+.
#
# The "a" suffix is REQUIRED: WGMMA and tcgen05 live in the *accelerated* ISA,
# not base sm_90 / sm_100. Compiling for sm_90 (no a) silently omits WGMMA.
#
# The sm_120 dev box (RTX PRO 4000 Blackwell client) is NOT an AOT target:
# it has no WGMMA and no tcgen05. It runs only the JIT learning rungs.
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.27)

# torch's CMake helpers (CUDAExtension, torch_extension) live in the torch package;
# find it via the active Python so the same torch that imports the ext builds it.
# This is the torch-blessed pattern; do not hardcode a CUDA toolkit path.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.__version__)"
    OUTPUT_VARIABLE TORCH_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_VERSION_RC)
if(NOT TORCH_VERSION_RC EQUAL 0)
    message(FATAL_ERROR "torch not importable from ${Python3_EXECUTABLE}; "
            "activate the GPU venv first (`source .venv/bin/activate`).")
endif()
message(STATUS "Found torch ${TORCH_VERSION}")

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_CMAKE_PREFIX_RC)
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

project(scratch_llm_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Target archs (default = the rental frontier) ----------------------------
# Override with -DTARGET_ARCHS="sm_90a" to build only one arch (faster compile).
if(NOT DEFINED TARGET_ARCHS)
    set(TARGET_ARCHS "sm_90a;sm_100a")
endif()
foreach(sm IN LISTS TARGET_ARCHS)
    message(STATUS "  AOT target arch: ${sm}")
    list(APPEND _CUDA_ARCH_FLAGS "-arch=${sm}")
endforeach()

# --- Sources (the frontier .cu under csrc/) -----------------------------------
# Each promoted rung and each stub lives under csrc/<family>/. Adding a new
# C++ kernel = drop the .cu here + bind it in csrc/pybind.cpp.
set(_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/wgmma_sm90.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/tcgen05_sm100.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/attention/fa3_hopper.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/fp8_gemm_sm90.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/stream_k_sm90.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/persistent/persistent_gemv_sm90.cu
)

find_package(Torch REQUIRED)

add_library(_scratch_llm_kernels SHARED ${_SOURCES})
target_link_libraries(_scratch_llm_kernels PRIVATE ${TORCH_LIBRARIES})

# torch's ABI setting (must match torch's own build or the link fails at import).
set_target_properties(_scratch_llm_kernels PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON)

target_include_directories(_scratch_llm_kernels PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    # CUTLASS 3.x header-only include (future — see WS-E ADR-0019 / the "not
    # vendored yet" note in the plan). The promoted rungs use hand PTX/inline
    # asm and do NOT require CUTLASS. Uncomment after `git submodule add`:
    # "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include"
)

# Per-arch SASS, fast-math, verbose ptxas (so the rental-day build log shows
# register / SMEM usage per kernel — the load-bearing profiling signal).
target_compile_options(_scratch_llm_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        ${_CUDA_ARCH_FLAGS}
        -O3
        --use_fast_math
        -Xptxas=-v
        -lineinfo
    >
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Standard torch-extension install: land next to the package so
# `from scratch_llm import _scratch_llm_kernels` resolves.
install(TARGETS _scratch_llm_kernels LIBRARY DESTINATION scratch_llm)
