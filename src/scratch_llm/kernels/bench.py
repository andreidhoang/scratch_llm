"""Kernel roofline bench — the *profile-DoD engine* for the CUDA-for-Deep-Learning sprint.

Every kernel day closes on ONE line this module prints (CADENCE rule 4 / BOOK_SPRINT §4):

    matmul_naive       1234.5 us |   2.1 TF/s |   4.8% of ref | AI=42.7 FLOP/B | memory-bound | fix=___

The loop: **predict the % before you run → reconstruct the kernel from blank → run this → log the
line → break/optimize → re-run.** `% of a measured reference` (cuBLAS for matmul, SDPA for attention)
is the robust signal and needs no peak constants; the `AI / {mem|compute}-bound` verdict is a teaching
aid whose peak numbers are *nominal* — replace `_PEAKS` with YOUR measured peak before trusting it.

Triton-first (your stack), but `load_cuda` benchmarks the book's CUDA C++ kernels with the same
`roofline()` so the CUDA-for-DL reps and your Triton kernels share one number. Run on the GPU box;
import is CPU-safe (Triton is imported lazily; nothing here needs a GPU at import time).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Nominal dense (no-sparsity) peaks — APPROX, verify per your card: (HBM TB/s, bf16 TFLOP/s).
# The "RTX PRO 4000" row is *measured-achieved* on this Blackwell instance (do_bench: 8192³ bf16
# matmul + 1 GB copy), per the module's own "replace with YOUR measured peak" rule — note Blackwell's
# real headroom is FP4/FP8, not bf16, so this bf16 figure understates the card for low-precision work.
_PEAKS: dict[str, tuple[float, float]] = {
    "RTX PRO 4000": (
        0.55,
        72.0,
    ),  # measured-achieved (sm120 Blackwell, 25 GB); ridge ≈ 130 FLOP/byte
    "4090": (1.01, 165.0),
    "H100": (3.35, 989.0),
    "A100": (2.04, 312.0),
    "L40": (0.86, 181.0),
}


def gpu_peaks(name: str | None = None) -> tuple[float, float] | None:
    """(HBM TB/s, bf16 TFLOP/s) for the current card, or None if unknown. Nominal — verify."""
    if name is None:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    for key, val in _PEAKS.items():
        if key in name:
            return val
    return None


def _bench_ms(fn) -> float:
    """Median runtime in ms. Prefer triton.testing.do_bench; fall back to CUDA events."""
    try:
        import triton  # noqa: PLC0415

        return float(triton.testing.do_bench(fn))
    except Exception:  # noqa: BLE001 - no Triton (e.g. pure-CUDA load_inline kernel): time by hand
        torch.cuda.synchronize()
        for _ in range(5):  # warmup
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 50
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters


@dataclass
class Roofline:
    label: str
    ms: float
    tflops: float
    pct_ref: float | None
    ai: float | None
    bound: str | None

    def line(self) -> str:
        parts = [f"{self.label:<18} {self.ms * 1e3:8.1f} us", f"{self.tflops:6.1f} TF/s"]
        if self.pct_ref is not None:
            parts.append(f"{self.pct_ref:5.1f}% of ref")
        if self.ai is not None:
            parts.append(f"AI={self.ai:.1f} FLOP/B")
        if self.bound is not None:
            parts.append(f"{self.bound}-bound")
        parts.append("fix=___")
        return " | ".join(parts)


def roofline(
    fn,
    *,
    flops: float,
    rw_bytes: float,
    label: str,
    ref=None,
    peaks: tuple[float, float] | None = None,
) -> Roofline:
    """Benchmark `fn`, print the one-line DoD, and return it. `ref` (e.g. cuBLAS) sets the %."""
    ms = _bench_ms(fn)
    tflops = flops / (ms * 1e-3) / 1e12
    pct = None
    if ref is not None:
        ms_ref = _bench_ms(ref)
        pct = 100.0 * ms_ref / ms  # faster kernel => higher % of the reference's throughput
    ai = (flops / rw_bytes) if rw_bytes else None
    bound = None
    peaks = peaks or gpu_peaks()
    if ai is not None and peaks is not None:
        bw_tbps, tf = peaks
        ridge = (tf * 1e12) / (bw_tbps * 1e12)  # FLOP/byte knee
        bound = "memory" if ai < ridge else "compute"
    out = Roofline(label, ms, tflops, pct, ai, bound)
    print(out.line())
    return out


def matmul_roofline(
    fn,
    m: int = 4096,
    n: int = 4096,
    k: int = 4096,
    *,
    dtype: torch.dtype = torch.bfloat16,
    label: str | None = None,
    ref_to_cublas: bool = True,
) -> Roofline:
    """C = A@B roofline vs cuBLAS (torch.matmul). flops=2·M·N·K; bytes=(MK+KN+MN)·itemsize."""
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    flops = 2.0 * m * n * k
    rw_bytes = (m * k + k * n + m * n) * a.element_size()
    ref = (lambda: torch.matmul(a, b)) if ref_to_cublas else None
    return roofline(
        lambda: fn(a, b),
        flops=flops,
        rw_bytes=rw_bytes,
        label=label or getattr(fn, "__name__", "matmul"),
        ref=ref,
    )


def vector_add_roofline(
    fn,
    n: int = 1 << 24,
    *,
    dtype: torch.dtype = torch.float32,
    label: str | None = None,
    ref_to_torch: bool = True,
) -> Roofline:
    """z = x + y roofline vs torch. flops=N (one add/elem); bytes=3·N·itemsize (read x,y; write z).

    Memory-bound: AI = N / (3·N·itemsize) = 1/(3·itemsize) ≈ 1/12 for fp32 → grade on % of HBM BW,
    not % of FLOP/s. A correct kernel should land near the bandwidth roof (~80–90% of `x + y`).
    """
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    flops = float(n)
    rw_bytes = 3.0 * n * x.element_size()
    ref = (lambda: x + y) if ref_to_torch else None
    return roofline(
        lambda: fn(x, y),
        flops=flops,
        rw_bytes=rw_bytes,
        label=label or getattr(fn, "__name__", "vector_add"),
        ref=ref,
    )


def load_cuda(name: str, cuda_source: str, functions: list[str], **kwargs):
    """GPU MODE-style CUDA C++ build: benchmark the *book's* kernels with the same roofline().

    Example (the book's naive matmul, reconstructed from blank):
        mod = load_cuda("mm_naive", CUDA_SRC, ["mm_naive"])
        matmul_roofline(mod.mm_naive, 4096, 4096, 4096)
    """
    from torch.utils.cpp_extension import load_inline  # noqa: PLC0415

    return load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=functions,
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        **kwargs,
    )
