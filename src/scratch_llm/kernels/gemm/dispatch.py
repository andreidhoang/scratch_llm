"""Stable public surface for GEMM / GEMV kernels -- the dispatch layer.

This is the contract every consumer above ``kernels/`` imports from. Backend
subpackages (``triton/``, ``cuda/``, ``wmma/``) are PRIVATE; production code
reaches them only through this module. Tests and benches may target a backend
directly (they are measuring one specific rung of the GEMM ladder, not dispatching).

The GEMM family is where dispatch earns its keep: there are FIVE backends, each a
different point on the precision / tensor-core / autotune trade-off, and the right
one depends on ``dtype`` x ``shape`` x ``device``. Routing that choice out of the
model and into a single policy here means the model calls ``matmul(a, b)`` and gets
the best kernel for its inputs; adding a sixth backend (e.g. a Hopper WGMMA GEMM)
lands as a new branch in :func:`matmul`, not as a rewrite of every caller.

Routing policy (:func:`matmul` -- the drop-in for ``torch.matmul`` on the model's
hot path; output dtype always matches input dtype):

  * CPU / non-CUDA ............ ``torch.matmul`` (no custom kernel; CPU is not the target)
  * CUDA fp8_e4m3 × fp8_e4m3, Hopper . ``fp8_gemm``  (STUB; 2x FLOP density of fp16)
  * CUDA fp16 × fp16, Hopper .. ``wgmma_gemm`` (warpgroup-MMA; ~4.5x WMMA — the rental frontier)
  * CUDA fp16 × fp16, Blackwell DC . ``tcgen05_gemm`` (TMEM accumulator — the B200 frontier)
  * CUDA fp16 × fp16, other ... ``wmma_gemm`` (WMMA tensor cores, fp32-accumulate) -> fp16
  * CUDA bf16 × bf16 .......... ``gemm_autotuned`` (Triton ``tl.dot``, the shipping bf16 path)
  * CUDA, other dtypes ........ ``torch.matmul`` (no fp32 rung in the ladder -- tf32 is the
                                training precision, owned by Triton's ``allow_tf32``)

The frontier paths (fp8 / wgmma / tcgen05) are **arch-gated**: they fire only on
Hopper (sm_90a) or Blackwell-DC (sm_100a). On an sm_120 client card they are
unreachable from ``matmul`` (the dtype branch falls through to WMMA / Triton), so
sm_120 keeps the exact pre-frontier behavior. The frontier rungs are reached
directly by their benches/tests (the exempt path) and, on the rental box,
automatically by this routing. ``stream_k=True`` opts into the stream-K backend
(wave-quantization fixer; Hopper only).

Per-backend entry points (lazy) are ALSO exposed for rung-specific use -- the
bench compares them head-to-head and the adversarial tests target one rung each:
``gemm_naive``, ``gemm_tiled``, ``gemm_autotuned`` (Triton); ``gemv_naive``,
``gemv_blockrow``, ``gemv_split`` (Triton GEMV); ``gemm_smem`` (CUDA cores);
``gemm_mma_sync`` (PTX mma.sync); ``wmma_gemm``, ``wmma_gemm_fp16acc`` (WMMA);
``wgmma_gemm`` (Hopper WGMMA); ``tcgen05_gemm`` (Blackwell tcgen05);
``fp8_gemm``, ``stream_k_gemm``, ``persistent_gemv`` (stubs — the learning ladder).

CPU-safety: the oracle is absent here (GEMM has no pure-torch oracle -- ``torch.matmul``
IS the oracle), so nothing is imported eagerly. Every backend loads lazily via PEP 562
``__getattr__`` on first explicit access; ``import scratch_llm.kernels.gemm.dispatch``
stays CPU-safe (no Triton, no nvcc JIT).
"""

from __future__ import annotations

import torch
from torch import Tensor

# ``__all__`` is the EAGER, CPU-safe surface (empty -- GEMM has no CPU oracle).
# ``from dispatch import *`` therefore pulls nothing; ``__dir__()`` advertises the
# full lazy surface, and explicit ``from dispatch import <name>`` resolves lazily.
__all__: list[str] = []

# Lazy GPU backend loader (PEP 562). Each entry maps a public name to
# (backend module, attribute). The import runs on first attribute access only, so
# importing this module is CPU-safe; accessing a name is the explicit act that
# pulls Triton / triggers nvcc JIT.
_LAZY_GPU: dict[str, tuple[str, str]] = {
    # Triton tiled GEMM ladder
    "gemm_naive": ("scratch_llm.kernels.gemm.triton.tiled", "gemm_naive"),
    "gemm_tiled": ("scratch_llm.kernels.gemm.triton.tiled", "gemm_tiled"),
    "gemm_autotuned": ("scratch_llm.kernels.gemm.triton.tiled", "gemm_autotuned"),
    "gemm": ("scratch_llm.kernels.gemm.triton.tiled", "gemm"),
    # Triton GEMV ladder
    "gemv_naive": ("scratch_llm.kernels.gemm.triton.gemv", "gemv_naive"),
    "gemv_blockrow": ("scratch_llm.kernels.gemm.triton.gemv", "gemv_blockrow"),
    "gemv_split": ("scratch_llm.kernels.gemm.triton.gemv", "gemv_split"),
    # CUDA-core (no tensor cores) -- the re-anchor baseline
    "gemm_smem": ("scratch_llm.kernels.gemm.cuda.smem_tiled", "gemm_smem"),
    # PTX mma.sync warp-MMA (ldmatrix + XOR-swizzled SMEM)
    "gemm_mma_sync": ("scratch_llm.kernels.gemm.cuda.mma_sync", "gemm_mma_sync"),
    # WMMA fragment MMA (the shipping fp16 tensor-core path; fp32 + fp16acc variants)
    "wmma_gemm": ("scratch_llm.kernels.gemm.wmma.gemm", "wmma_gemm"),
    "wmma_gemm_fp16acc": ("scratch_llm.kernels.gemm.wmma.gemm", "wmma_gemm_fp16acc"),
    # --- Frontier rungs (arch-gated; reached only via the is_hopper/is_blackwell
    #     branches in matmul() below, or by explicit bench/test access). Runtime
    #     correctness is deferred to the rental day (wgmma, tcgen05) or to the
    #     learning rep (fp8, stream_k, persistent). ---
    "wgmma_gemm": ("scratch_llm.kernels.gemm.cuda.wgmma", "wgmma_gemm"),
    "tcgen05_gemm": ("scratch_llm.kernels.gemm.cuda.tcgen05", "tcgen05_gemm"),
    "fp8_gemm": ("scratch_llm.kernels.gemm.cuda.fp8", "fp8_gemm"),
    "stream_k_gemm": ("scratch_llm.kernels.gemm.cuda.stream_k", "stream_k_gemm"),
    "persistent_gemv": ("scratch_llm.kernels.gemm.cuda.persistent", "persistent_gemv"),
}


def __getattr__(name: str):
    if name in _LAZY_GPU:
        import importlib

        mod_name, attr = _LAZY_GPU[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()) | set(_LAZY_GPU))


def matmul(a: Tensor, b: Tensor, *, stream_k: bool = False) -> Tensor:
    """``C = A @ B`` routed to the best GEMM backend for ``(device, dtype, arch)``.

    Drop-in for ``torch.matmul`` on the model's hot path. Output dtype always
    matches ``a.dtype`` (cast back if the backend accumulates wider, e.g. WMMA's
    fp32 accumulator on fp16 inputs).

    Policy (see module docstring for the why behind each branch):
      * CPU, or non-CUDA tensors ........ ``torch.matmul`` (the framework path)
      * CUDA fp8_e4m3 × fp8_e4m3, Hopper  ``fp8_gemm``  (STUB; 2x FLOP density of fp16)
      * CUDA fp16 × fp16, Hopper ........ ``wgmma_gemm`` (warpgroup-MMA; ~4.5x WMMA)
      * CUDA fp16 × fp16, Blackwell DC .. ``tcgen05_gemm`` (TMEM accumulator)
      * CUDA fp16 × fp16, other ......... ``wmma_gemm`` (tensor cores, fp32-acc) -> fp16
      * CUDA bf16 × bf16 ................ ``gemm_autotuned`` (Triton ``tl.dot``, autotuned)
      * CUDA, other dtypes .............. ``torch.matmul`` (no fp32 rung in the ladder)

    The frontier paths (fp8/wgmma/tcgen05) are arch-gated: they fire only on the
    Hopper (sm_90a) or Blackwell-DC (sm_100a) rental box, where the ISA exists.
    On an sm_120 client card they are unreachable from ``matmul`` — the dtype branch
    falls through to the shipping rungs (wmma / triton), so the dispatch layer
    still works on sm_120 exactly as before. The frontier rungs are reached
    directly by their benches/tests (the exempt path) and, on the rental box,
    automatically by this routing.

    ``stream_k=True`` overrides to the stream-K partitioned backend (sm_90a only,
    the wave-quantization fixer — wins only on specific shapes where the tail
    wave idles; opt-in, not the default path).

    ``M == 1`` (true GEMV) is NOT special-cased here -- the GEMV ladder takes a 1-D
    ``x`` (``A @ x``), not a 2-D ``(1, K) @ (K, N)`` matmul, so it is a different op
    surface. Callers with a 1-D vector use ``gemv_blockrow`` directly; ``matmul`` is
    strictly the 2-D GEMM entry.
    """
    if not (a.is_cuda and b.is_cuda):
        return torch.matmul(a, b)

    # stream-K opt-in override (sm_90a only; takes precedence over dtype routing).
    if stream_k:
        from scratch_llm.kernels.common.arch import is_hopper

        if is_hopper():
            from scratch_llm.kernels.gemm.cuda.stream_k import stream_k_gemm

            return stream_k_gemm(a, b)
        # Not Hopper — stream-K is not implemented elsewhere; fall through silently.

    if a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn:
        from scratch_llm.kernels.common.arch import is_hopper

        if is_hopper():
            # FP8 path: 2x the FLOP density of fp16 on Hopper (mma m16n8k32 vs m16n8k16).
            # STUB until csrc/gemm/fp8_gemm_sm90.cu is implemented; raises NotImplementedError.
            from scratch_llm.kernels.gemm.cuda.fp8 import fp8_gemm

            return fp8_gemm(a, b)
        # Not Hopper — no FP8 rung; fall through to torch.matmul below.

    if a.dtype == torch.float16 and b.dtype == torch.float16:
        from scratch_llm.kernels.common.arch import is_blackwell, is_hopper, is_sm120

        if is_hopper():
            # WGMMA: warpgroup-MMA against SMEM descriptors, ~4.5x WMMA on H100.
            # Runtime correctness deferred to rental day (compile-gated skeleton).
            from scratch_llm.kernels.gemm.cuda.wgmma import wgmma_gemm

            return wgmma_gemm(a, b).to(torch.float16)
        if is_blackwell() and not is_sm120():
            # tcgen05: TMEM-accumulator MMA, the B200 frontier rung.
            # Excludes sm_120 (client Blackwell — no tcgen05/TMEM).
            from scratch_llm.kernels.gemm.cuda.tcgen05 import tcgen05_gemm

            return tcgen05_gemm(a, b).to(torch.float16)
        # Default fp16 path: WMMA tensor cores, fp32 accumulate -> cast back to fp16.
        from scratch_llm.kernels.gemm.wmma.gemm import wmma_gemm

        return wmma_gemm(a, b).to(torch.float16)
    if a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16:
        # Triton autotuned tl.dot: the shipping bf16 path; output is already bf16.
        from scratch_llm.kernels.gemm.triton.tiled import gemm_autotuned

        return gemm_autotuned(a, b)

    # fp32 / fp64 / tf32: no rung in the ladder (tf32 is a Triton allow_tf32 flag on
    # the bf16 path, not a standalone kernel). Fall through to the framework.
    return torch.matmul(a, b)
