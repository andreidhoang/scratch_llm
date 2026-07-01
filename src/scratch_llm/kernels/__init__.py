"""scratch_llm kernels — the CS336 A2 FlashAttention-2 artifact + the meat boundary.

The `@triton.jit` / `__global__` kernel bodies are HUMAN reconstruct-from-blank reps; agents do
everything around them (build, oracle, test, bench). See `CLAUDE.md` in this dir for the boundary.

Layout (each module's docstring states its intent + invariant):

  flash_attention_triton.py   FlashAttention-2 fwd + causal (autotuned) — the A2.1 roofline kernel
  flash_attention.py          pure-PyTorch FA2 oracle — the CPU-testable ground truth

Export policy: only the CPU-safe pure-PyTorch oracle is re-exported here, so
`import scratch_llm.kernels` never pulls Triton or compiles CUDA (CI stays green on a CPU box). The
Triton kernel is imported directly on a GPU box (`.flash_attention_triton`).

> **Perf-curriculum reset (2026-07-01).** The GEMV/GEMM/RMSNorm CUDA suite (`inference/`, `csrc/`,
> `bench.py`, `gemv_triton.py`) was removed to rebuild the `performance/` A1–A7 track from scratch
> against its own spec (`performance/PERF_ENGINEERING_SPEC.md`). The prior work is preserved at the
> git tag `pre-perf-kernel-reset`. FA2 above is CS336 A2 substrate and was intentionally kept.
"""

from scratch_llm.kernels.flash_attention import (
    FlashAttentionPyTorch,
    flash_attention_forward,
)

__all__ = ["FlashAttentionPyTorch", "flash_attention_forward"]
