"""Hopper FP8 (E4M3) GEMM (sm_90a) — STUB, the learning rep.

FP8 E4M3 is the 2026 frontier training precision: 4 exponent bits + 3 mantissa
bits give 1.5 bytes of dynamic range per byte of storage, and Hopper's
``mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32`` does a 16x8x32 MMA in one issue
(vs 16x8k16 for fp16) — so FP8 is 2x the FLOP throughput at half the memory
traffic. That's why Llama 3, DeepSeek V3, and Mistral Large 2 train in FP8.

**This is a STUB.** The kernel body in ``csrc/gemm/fp8_gemm_sm90.cu`` is a
scaffold (headers, defines, host launcher, references). The mainloop is the
learning rep — left for you to implement per the CUTLASS 3.2 FP8 GEMM example.
Until then, calling this raises a clear ``NotImplementedError`` pointing at the
exact file + reference.

See ``csrc/gemm/fp8_gemm_sm90.cu`` for the full reference list (CUDA C++
Programming Guide §7.25.2, PTX ISA §9.7.13.4.4, CUTLASS 3.2 example 55,
DeepSeek V3 §3.3).
"""

from __future__ import annotations

from torch import Tensor

from scratch_llm.kernels.common.arch import require_cc

_NOT_IMPL = (
    "fp8_gemm: STUB. Implement the mainloop in csrc/gemm/fp8_gemm_sm90.cu "
    "(see CUTLASS 3.2 examples/55_hopper_fp8_gemm and PTX ISA §9.7.13.4.4 — "
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32). "
    "The host launcher + dispatch + test + bench are already wired; fill the "
    "kernel body and remove the runtime_error in the C++ launcher."
)


def fp8_gemm(a: Tensor, b: Tensor) -> Tensor:
    """C = A @ B for float8_e4m3fn inputs, fp32-accumulated, **fp32 output**.

    STUB — raises ``NotImplementedError`` until the mainloop in
    ``csrc/gemm/fp8_gemm_sm90.cu`` is implemented (the learning rep).
    """
    require_cc(9, 0, fn_name="fp8_gemm")
    raise NotImplementedError(_NOT_IMPL)
