"""scratch_llm kernels — operation-family layout with CPU-safe public exports.

Layout (each module's docstring states its intent + invariant):

  attention/dispatch.py               STABLE public surface — the routing layer callers use
  attention/reference.py              FlashAttention-2 oracle — CPU-testable ground truth
  attention/prefill/fa2.py            FlashAttention-2 fwd + causal (autotuned) — GPU (private)
  attention/decode/paged.py           Fused paged decode attention — GPU (private)
  gemm/dispatch.py                    STABLE public surface — routes by (device, dtype, shape)
  gemm/triton/                        Triton GEMM/GEMV ladder (private)
  gemm/cuda/                          CUDA-core + mma.sync GEMM, JIT nvcc (private)
  gemm/wmma/                          WMMA fragment GEMM (private)
  norm/dispatch.py                    STABLE public surface — CPU/CUDA routing
  norm/normalize.py                   RMSNorm / LayerNorm (private)
  reduce/dispatch.py                  STABLE public surface — CPU/CUDA routing
  reduce/                             Softmax + top-k (private)
  common/arch.py                      GPU compute-capability detection + ISA gating
  common/online_softmax.py            Shared online-softmax recurrence (attention building block)

Export policy: only the CPU-safe pure-PyTorch oracle is re-exported here, so
``import scratch_llm.kernels`` never pulls Triton or compiles CUDA (CI stays green on a CPU box).
GPU kernels are reached through their operation family's ``dispatch.py`` (the
stable seam): ``from scratch_llm.kernels.attention.dispatch import
TritonFlashAttention`` loads Triton lazily via PEP 562 on first access. Production
code above ``kernels/`` never imports a ``@triton.jit`` symbol or a backend module
directly — that boundary is enforced by routing through ``dispatch``.
"""

from scratch_llm.kernels.attention import (  # routes through attention/dispatch.py
    FlashAttentionPyTorch,
    flash_attention_forward,
)

__all__ = ["FlashAttentionPyTorch", "flash_attention_forward"]
