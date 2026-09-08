"""S1/S-R4 — fused MoE: routing, permutation, grouped GEMM, un-permute, expert parallelism.

The Triton kernel lives in ``grouped_gemm_triton`` and is NOT imported here: importing it would
drag ``triton`` into every CPU import of this package, and the whole point of the split is that
everything except the kernel's mainloop is decidable, and tested, on a laptop. Use
``fused.resolve_gemm(device)``.
"""

from scratch_llm.kernels.moe.fused import (
    MoETrace,
    reference_moe,
    resolve_gemm,
    s_r4_fused_moe,
    silu_and_mul,
    torch_grouped_gemm,
    trace_moe,
)
from scratch_llm.kernels.moe.grouped_gemm import (
    GroupedGEMMConfig,
    default_config,
    exact_m_tiles,
    max_m_tiles,
    reference_grouped_gemm,
    validate_group_offsets,
)
from scratch_llm.kernels.moe.routing import (
    ExpertLoad,
    Permutation,
    Routing,
    build_permutation,
    combine,
    expert_load,
    gather_tokens,
    permute_rows,
    route_topk,
    topk_deterministic,
    unpermute_rows,
)

__all__ = [
    "ExpertLoad",
    "GroupedGEMMConfig",
    "MoETrace",
    "Permutation",
    "Routing",
    "build_permutation",
    "combine",
    "default_config",
    "exact_m_tiles",
    "expert_load",
    "gather_tokens",
    "max_m_tiles",
    "permute_rows",
    "reference_grouped_gemm",
    "reference_moe",
    "resolve_gemm",
    "route_topk",
    "s_r4_fused_moe",
    "silu_and_mul",
    "topk_deterministic",
    "torch_grouped_gemm",
    "trace_moe",
    "unpermute_rows",
    "validate_group_offsets",
]
