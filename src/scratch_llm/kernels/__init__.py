"""scratch_llm kernels — two backends, one roofline harness, one meat boundary.

The `@triton.jit` / `__global__` kernel bodies are HUMAN reconstruct-from-blank reps; agents do
everything around them (build, oracle, test, bench). See `CLAUDE.md` in this dir for the boundary.

Layout (organized by backend — each module's docstring states its intent + invariant):

  Triton  (A2 Systems · GPU-only · NOT imported on CPU):
    flash_attention_triton.py   FlashAttention-2 fwd + causal (autotuned) — the A2.1 roofline kernel
    matmul.py                   naive→tiled matmul rungs (% of cuBLAS)
  CUDA C++  (inference / decode · JIT-built via torch.utils.cpp_extension.load):
    inference/                  Python wrappers, one module per domain (norm.py, gemm.py, ...);
                                each owns a kernel's oracle (*_ref) · dispatch · roofline (*_roofline)
    csrc/                       .cu launchers grouped by domain (norm/, gemm/, ...) + ops.h (decls)
                                + bindings.cpp (the aggregated `_C` ABI) + _template.cu + README
  Reference / shared:
    flash_attention.py          pure-PyTorch FA2 oracle — the CPU-testable ground truth
    bench.py                    backend-agnostic roofline (% of a measured ref · AI · mem/compute-bound)

Export policy: only the CPU-safe pure-PyTorch oracle is re-exported here, so
`import scratch_llm.kernels` never pulls Triton or compiles CUDA (CI stays green on a CPU box). GPU
kernels are imported directly from their module on a GPU box (`.flash_attention_triton`, `.matmul`,
`.inference`).

Asymmetry is deliberate: the CUDA track is a *growing* multi-kernel suite (norm → gemm → ...) so it
gets the real-world csrc-by-domain + per-domain-wrapper structure (mirrors vLLM). The Triton files
are a *finished* A2 artifact welded into the CS336 docs (LECTURE_MAP / A2 guide / design specs), the
`bench/` script, and the meat-boundary hook's `*_triton.py` pattern — so they stay flat (churn ≫
value). If the Triton track ever grows into a suite, mirror the split with a `triton/` package then.
"""

from scratch_llm.kernels.flash_attention import (
    FlashAttentionPyTorch,
    flash_attention_forward,
)

__all__ = ["FlashAttentionPyTorch", "flash_attention_forward"]
