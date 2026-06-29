"""The roofline engine — predict the bound and the number *before* the run.

``attainable = min(peak_compute, AI x peak_bandwidth)`` where ``AI = useful_FLOPs / bytes_moved``.
Feed an op's FLOP and byte counts (the op-counters below, or your own) + a :class:`GpuSpec`; get back
the arithmetic intensity, the predicted bound (compute / memory), and the predicted runtime. After the
run, :func:`pct_of_roof` scores the measured time against that prediction — the DoD is a profile that
lands near the roof you wrote down first (FOP-3), not a green test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scratch_llm.bench.gpu_specs import GpuSpec

Bound = Literal["compute", "memory"]


@dataclass(frozen=True)
class Roofline:
    """A predicted operating point on a GPU's roofline."""

    arithmetic_intensity: float  # FLOP/byte
    ridge_point: float  # FLOP/byte; AI below this = memory-bound
    bound: Bound
    attainable_flops: float  # the roofline ceiling at this AI
    predicted_seconds: float  # FLOPs / attainable_flops
    flops: float
    bytes_moved: float

    @property
    def predicted_tokens_per_s(self) -> float:
        """For a single decode step priced as one forward: 1 / predicted_seconds."""
        return 1.0 / self.predicted_seconds if self.predicted_seconds > 0 else float("inf")


def roofline(flops: float, bytes_moved: float, spec: GpuSpec, dtype: str) -> Roofline:
    """Place an op (given its FLOP + byte counts) on ``spec``'s roofline for ``dtype``."""
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be > 0")
    peak = spec.peak_flops.get(dtype)
    if peak is None:
        raise KeyError(f"{spec.name} has no dense {dtype} path")
    ai = flops / bytes_moved
    ridge = spec.ridge_point(dtype)
    attainable = min(peak, ai * spec.hbm_bandwidth)
    return Roofline(
        arithmetic_intensity=ai,
        ridge_point=ridge,
        bound="compute" if ai >= ridge else "memory",
        attainable_flops=attainable,
        predicted_seconds=flops / attainable if attainable > 0 else float("inf"),
        flops=flops,
        bytes_moved=bytes_moved,
    )


def pct_of_roof(predicted_seconds: float, measured_seconds: float) -> float:
    """Fraction of the predicted roofline the measured run achieved, in [0, 1+]. 1.0 = on the roof;
    <1 = the run is slower than the ceiling (kernel/launch/occupancy overhead — name which)."""
    if measured_seconds <= 0:
        raise ValueError("measured_seconds must be > 0")
    return predicted_seconds / measured_seconds


# --- op-counters: (FLOPs, bytes_moved) for common ops -----------------------------------------------
# Predict-before-run uses these to get AI; the *kernel* that realizes them is the Mode-3 rep.


def gemm_flops_bytes(m: int, n: int, k: int, dtype_bytes: int) -> tuple[float, float]:
    """Dense GEMM C[m,n] = A[m,k] @ B[k,n]: 2*m*n*k FLOPs; reads A,B + writes C bytes.

    The compute-bound archetype — AI grows with the shared dimension, so a big GEMM sits far right of
    the ridge.
    """
    flops = 2.0 * m * n * k
    bytes_moved = float((m * k + k * n + m * n) * dtype_bytes)
    return flops, bytes_moved


def elementwise_flops_bytes(
    n_elements: int, dtype_bytes: int, flops_per_elem: float = 1.0
) -> tuple[float, float]:
    """A memory-bound elementwise/norm op: O(n) FLOPs against one read + one write. AI ~ O(1) -> deep
    in the memory roof; the fix is fusion (cut the byte traffic, not the FLOPs)."""
    flops = flops_per_elem * n_elements
    bytes_moved = float(2 * n_elements * dtype_bytes)
    return flops, bytes_moved


def decode_step_flops_bytes(
    n_params: int,
    *,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_len: int,
    batch: int,
    weight_bytes: int,
    kv_bytes: int,
) -> tuple[float, float]:
    """One autoregressive decode step (query length 1): the canonical memory-bound workload.

    FLOPs ~ 2 * n_params per token (one MAC per weight); bytes = every weight read once + the whole KV
    cache read once. At batch 1 this gives AI ~ 1 FLOP/byte — ~300x below the H100 dense ridge — which
    is *why* decode tok/s is set by HBM bandwidth, not FLOPs, and why every serving lever (batching, KV
    quant, paged KV, MLA/GQA) exists to raise this AI back toward the ceiling.
    """
    flops = 2.0 * n_params * batch
    weight_traffic = n_params * weight_bytes
    kv_traffic = 2 * n_layers * n_kv_heads * head_dim * context_len * batch * kv_bytes
    return flops, float(weight_traffic + kv_traffic)
