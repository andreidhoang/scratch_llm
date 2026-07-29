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

# Every frontier .cu under csrc/ — the same list CMakeLists.txt::_SOURCES uses.
# Adding a new C++ kernel = drop the .cu here AND in CMakeLists.txt.
_CUDA_SOURCES = [
    str(_CSRC / "pybind.cpp"),
    str(_CSRC / "gemm" / "wgmma_sm90.cu"),
    str(_CSRC / "gemm" / "tcgen05_sm100.cu"),
    str(_CSRC / "attention" / "fa3_hopper.cu"),
    str(_CSRC / "gemm" / "fp8_gemm_sm90.cu"),
    str(_CSRC / "gemm" / "stream_k_sm90.cu"),
    str(_CSRC / "persistent" / "persistent_gemv_sm90.cu"),
]


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
            sources=_CUDA_SOURCES,
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
