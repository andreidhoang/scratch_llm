"""Hopper persistent GEMV (sm_90a) — STUB, the learning rep.

A persistent kernel launches ONE CTA per SM that lives for the whole kernel and
processes MANY tiles in a loop — reusing registers, keeping SMEM warm, and
amortizing launch cost. This is the structure behind every frontier kernel of
2024-2026: FlashAttention 3, LeanAttention, CUTLASS persistent GEMM, FlashInfer
decode.

The stub is a **persistent GEMV** (``x @ A`` where ``x`` is 1-D) — the decode
matmul, the hottest kernel in LLM serving, and the simplest non-trivial
persistent case (no K-reduction loop). It's the right first learning rep before
climbing to a persistent FA3.

**This is a STUB.** The persistent mainloop + SM-count grid setup in
``csrc/persistent/persistent_gemv_sm90.cu`` are the learning rep.

See ``csrc/persistent/persistent_gemv_sm90.cu`` for the full reference list
(LeanAttention, FA3 §3.1, Mark Saroufim's persistent GEMM tutorial, CUTLASS
``gemm_universal.h``).
"""

from __future__ import annotations

from torch import Tensor

from scratch_llm.kernels.common.arch import require_cc

_NOT_IMPL = (
    "persistent_gemv: STUB. Implement the persistent mainloop + SM-count grid in "
    "csrc/persistent/persistent_gemv_sm90.cu (see LeanAttention + FlashAttention 3 §3.1 "
    "+ Mark Saroufim's CUDA persistent GEMM tutorial). The host launcher + dispatch + "
    "test + bench are wired; fill the body and remove the runtime_error in the C++ launcher."
)


def persistent_gemv(x: Tensor, a: Tensor) -> Tensor:
    """y = x @ A for float16 ``x`` [K], ``A`` [K,N]; returns ``y`` [N] float16.

    STUB — raises ``NotImplementedError`` until the mainloop in
    ``csrc/persistent/persistent_gemv_sm90.cu`` is implemented (the learning rep).
    """
    require_cc(9, 0, fn_name="persistent_gemv")
    raise NotImplementedError(_NOT_IMPL)
