"""setup.py — the hybrid build backend for scratch_llm.

This file exists ONLY for the GPU AOT extension (`[gpu-aot]` extra). The
pure-Python wheel is still built by hatchling (`[build-system]` in pyproject);
this script is invoked directly on the rental box to compile `csrc/*.cu` into
the `_scratch_llm_kernels` torch extension::

    # 1. Pure-Python install (hatchling — the normal path, what CI uses)
    pip install -e ".[dev]"

    # 2. AOT frontier-kernel extension (rental box only — H100/H200/B200)
    TORCH_CUDA_ARCH_LIST="9.0a;10.0a" pip install -e ".[gpu-aot]"

Step 2 runs this script, which uses ``torch.utils.cpp_extension.CUDAExtension``
+ ``BuildExtension`` — the torch-blessed AOT path (the same one xformers,
flash-attn, and Mamba use). It builds ONE shared library from every .cu listed
in ``_CUDA_SOURCES`` and drops it next to the package so
``from scratch_llm import _scratch_llm_kernels`` resolves.

CPU-safety: this file is inert on a CPU-only box. It only does work when run
explicitly (``python setup.py build_ext --inplace`` or the ``[gpu-aot]`` pip
extra). The normal ``pip install -e .`` never touches it.

The CMake build (see CMakeLists.txt) is the *alternative* out-of-tree path for
users who want ninja-driven incremental builds. Both produce the same extension;
setup.py is the pip-friendly one, CMake is the developer-experience one.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_REPO_ROOT = Path(__file__).resolve().parent
_CSRC = _REPO_ROOT / "csrc"

# Every frontier .cu under csrc/, WITH the archs whose ISA it uses. The arch column is load
# bearing: WGMMA is sm_90a-only and ptxas rejects it elsewhere ("Instruction 'wgmma.fence' not
# supported on .target 'sm_100a'", verified 2026-09-07), so building every source for every arch in
# TORCH_CUDA_ARCH_LIST cannot work. infra/drydock.sh carries the same table; keep the two in step.
# `None` = every requested arch: host code, or a device body guarded by `#if __CUDA_ARCH__`.
_CUDA_SOURCES: list[tuple[str, set[str] | None]] = [
    (str(_CSRC / "pybind.cpp"), None),
    # K1 rungs — all arch-guarded with `#if __CUDA_ARCH__`, so inert rather than broken off-arch.
    (str(_CSRC / "gemm" / "h_r1_wgmma_bf16_sm90.cu"), None),
    (str(_CSRC / "gemm" / "h_r2_tma_bf16_sm90.cu"), None),
    (str(_CSRC / "gemm" / "h_r3_ws_bf16_sm90.cu"), None),
    (str(_CSRC / "gemm" / "h_r4_persistent_bf16_sm90.cu"), None),
    (str(_CSRC / "gemm" / "b_r6_nvfp4_sm120.cu"), None),
    (str(_CSRC / "gemm" / "wgmma_sm90.cu"), {"90"}),
    (str(_CSRC / "gemm" / "tcgen05_sm100.cu"), {"100"}),
    (str(_CSRC / "attention" / "fa3_hopper.cu"), {"90"}),
    # K2 rungs. `None`, like the K1 rows: both are arch-guarded internally
    # (`#if __CUDA_ARCH__ == 900`), so on another arch the Hopper ISA is preprocessed away and the
    # file still links. Verified in the dry dock on 2026-09-07 — both compile clean for sm_100a and
    # sm_120a — with A-R2's hole open, i.e. against its stub. When that hole is filled, the
    # mainloop must stay inside the SCRATCH_LLM_HAS_TMA_WGMMA guard or this row becomes {"90"}.
    (str(_CSRC / "attention" / "fa3_hopper_v2.cu"), None),
    (str(_CSRC / "attention" / "a_r3_paged_decode_sm90.cu"), None),
    (str(_CSRC / "gemm" / "fp8_gemm_sm90.cu"), {"90"}),
    (str(_CSRC / "gemm" / "stream_k_sm90.cu"), {"90"}),
    (str(_CSRC / "persistent" / "persistent_gemv_sm90.cu"), {"90"}),
]


def _sources_for(arch_flags: list[str]) -> list[str]:
    """The sources buildable for the requested archs; others are dropped, not built and hoped for.

    A dropped source's symbol is absent from the extension, which the loaders handle by JIT-
    compiling that one rung. Building it anyway fails the extension and takes every rung down.
    """
    requested = {f.rsplit("sm_", 1)[-1].rstrip("a") for f in arch_flags}
    restricted = [Path(s).name for s, a in _CUDA_SOURCES if a is not None]
    if len(requested) > 1 and restricted:
        raise SystemExit(
            f"setup.py: one CUDAExtension applies one -arch to every source, and {restricted[0]} "
            f"does not assemble off its own arch. Build one arch per pass — a rented box has "
            f"exactly one:\n    TORCH_CUDA_ARCH_LIST=\"9.0a\" pip install -e '.[gpu-aot]'\n"
            f"  (infra/bootstrap.sh derives this from nvidia-smi.)"
        )
    out = []
    for src, archs in _CUDA_SOURCES:
        if archs is None or (archs & requested):
            out.append(src)
        else:
            print(f"setup.py: skipping {Path(src).name} — needs sm_{'/'.join(sorted(archs))}a, "
                  f"requested sm_{'/'.join(sorted(requested))}a")
    return out


def _cuda_arch_flags() -> list[str]:
    """Translate ``TORCH_CUDA_ARCH_LIST`` (the torch convention) to nvcc ``-arch``.

    torch's ``CUDAExtension`` does this internally when TORCH_CUDA_ARCH_LIST is
    set; we expose it explicitly here so the CMake path and setup.py path share
    the exact same default ("9.0a;10.0a" — the rental frontier).
    """
    env = os.environ.get("TORCH_CUDA_ARCH_LIST", "9.0a;10.0a")
    out: list[str] = []
    for tok in env.replace(" ", "").split(";"):
        if not tok:
            continue
        # torch accepts "9.0a" / "9.0+a" / "sm_90a"; normalize to nvcc's "-arch=sm_90a"
        sm = tok.replace("sm_", "").replace("+a", "a").replace(".", "")
        if sm.isdigit():
            sm = sm  # "90" stays "90" — base arch, no WGMMA; user almost certainly meant 9.0a
            sm = sm[:-1] + "0" if len(sm) > 2 else sm
        out.append(f"-arch=sm_{sm}")
    return out


setup(
    name="scratch_llm_kernels_aot",  # the ext_modules name; the .so is _scratch_llm_kernels
    ext_modules=[
        CUDAExtension(
            name="_scratch_llm_kernels",
            sources=_sources_for(_cuda_arch_flags()),
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    *_cuda_arch_flags(),
                    "-O3",
                    "--use_fast_math",
                    "-Xptxas=-v",
                    "-lineinfo",
                ],
            },
            include_dirs=[str(_CSRC)],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
