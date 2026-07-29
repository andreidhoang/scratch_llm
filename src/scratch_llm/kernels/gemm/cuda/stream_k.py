"""Hopper stream-K GEMM (sm_90a) — STUB, the learning rep.

Stream-K is the *wave-quantization fixer*: when ``M*N/tile`` isn't divisible by
the SM count (e.g. 4096^3 on a 132-SM H100), the last wave idles 60-90% of the
chip. Stream-K partitions the K reduction across SMs — each CTA owns a slice of
K, atomically accumulating into partial-output tiles. No tail-wave idle; this is
the "last 10%" that takes a wgmma GEMM from 90% to 97% of cuBLAS.

**This is a STUB.** The K-partition scheduler + the atomic-accumulator kernel in
``csrc/gemm/stream_k_sm90.cu`` are the learning rep. CUTLASS's
``StreamKScheduler`` is the canonical reference (~400 lines of C++).

See ``csrc/gemm/stream_k_sm90.cu`` for the full reference list (CUTLASS
``streamk.hpp``, the Stream-K paper, the Hopper whitepaper §3.2).
"""

from __future__ import annotations

from torch import Tensor

from scratch_llm.kernels.common.arch import require_cc

_NOT_IMPL = (
    "stream_k_gemm: STUB. Implement the K-partition scheduler + kernel in "
    "csrc/gemm/stream_k_sm90.cu (see CUTLASS include/cutlass/gemm/kernel/streamk.hpp "
    "and the Stream-K paper). The host launcher + dispatch + test + bench are wired; "
    "fill the body and remove the runtime_error in the C++ launcher."
)


def stream_k_gemm(a: Tensor, b: Tensor) -> Tensor:
    """C = A @ B, stream-K partitioned (the wave-quantization fixer).

    STUB — raises ``NotImplementedError`` until the mainloop in
    ``csrc/gemm/stream_k_sm90.cu`` is implemented (the learning rep).
    """
    require_cc(9, 0, fn_name="stream_k_gemm")
    raise NotImplementedError(_NOT_IMPL)
