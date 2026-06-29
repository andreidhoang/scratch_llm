"""Shared roofline-bench primitives — one measurement methodology for every kernel.

flash_roofline, and the matmul/rmsnorm/gemv benches, must all measure the same way or their numbers
are not comparable. This module is that single source of truth: CUDA-event timing with per-rep L2
flush and quantile reporting, machine peaks measured on *this* GPU (not a datasheet), the roofline
arithmetic, wave-quantization accounting, and a provenance line. Each kernel bench keeps only what is
specific to it (its FLOP/byte model, its correctness gate, its grid shape) and imports the rest here.

Nothing here is kernel-aware: it speaks in FLOPs, bytes, and CTAs.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import torch
import triton


def bench_ms(fn, *, warmup: int = 50, rep: int = 200) -> tuple[float, float, float]:
    """Median, p20, p80 milliseconds. do_bench times with CUDA events and flushes the L2 between
    reps, so this is cold-L2 steady state — not a cache-resident fantasy. Report the median (typical),
    flag the p20–p80 spread (measurement trust); never the mean (interrupt-contaminated)."""
    med, lo, hi = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
    return float(med), float(lo), float(hi)


def spread_pct(med: float, lo: float, hi: float) -> float:
    """(p80 − p20) / median, in percent — the row's measurement noise. >5% ⇒ distrust (unstable clocks)."""
    return 100.0 * (hi - lo) / med


def smi(fields: str) -> list[float]:
    """Query nvidia-smi for comma-listed fields; return floats (NaN per field on any failure — a
    benchmark must never die because a clock probe hiccuped)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.splitlines()[0]
        return [float(x) for x in out.split(",")]
    except Exception:  # pragma: no cover - smi may be unavailable
        return [float("nan")] * len(fields.split(","))


def measure_mem_bw_bytes_s() -> float:
    """Achieved HBM bandwidth via a 256 MB device-to-device copy (read + write = 2 bytes/elem)."""
    a = torch.empty(1 << 26, device="cuda", dtype=torch.float32)  # 64 Mi elems = 256 MB
    b = torch.empty_like(a)
    ms = bench_ms(lambda: b.copy_(a))[0]
    return 2 * a.numel() * a.element_size() / (ms * 1e-3)


def measure_compute_peak_flops_s(dtype: torch.dtype, n: int = 8192) -> float:
    """Achievable tensor-core peak via a large square cuBLAS GEMM — a fairer compute roof than the
    datasheet: it is what well-tuned code on THIS card actually reaches."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    ms = bench_ms(lambda: torch.matmul(a, b))[0]
    return 2.0 * n**3 / (ms * 1e-3)


@dataclass(frozen=True)
class Roofs:
    """The two roofline ceilings, measured on this GPU."""

    compute_flops_s: float
    bw_bytes_s: float

    @classmethod
    def measure(cls, dtype: torch.dtype) -> Roofs:
        return cls(measure_compute_peak_flops_s(dtype), measure_mem_bw_bytes_s())

    @property
    def ridge(self) -> float:
        """Arithmetic intensity (FLOP/byte) where the memory and compute roofs meet."""
        return self.compute_flops_s / self.bw_bytes_s

    def attainable(self, flops: float, bytes_: float) -> tuple[float, str]:
        """Roofline-attainable FLOP/s for a kernel doing `flops` over `bytes_` of HBM traffic, plus
        which roof binds ('mem' | 'cmp'). %roof = achieved / attainable = absolute headroom."""
        mem_bound = (flops / bytes_) * self.bw_bytes_s
        return (
            (mem_bound, "mem")
            if mem_bound < self.compute_flops_s
            else (self.compute_flops_s, "cmp")
        )


def waves(ctas: int, sm_count: int) -> tuple[float, bool]:
    """CTA waves over the SMs, and whether the grid fails to fill the GPU even once (under-utilized)."""
    return ctas / sm_count, ctas < sm_count


def provenance_line(title: str) -> str:
    """A number without its environment is unreproducible — print this above every result table."""
    props = torch.cuda.get_device_properties(0)
    cuda = getattr(getattr(torch, "version", None), "cuda", "?")  # stubs omit torch.version
    return (
        f"# {title} | {props.name} | {props.multi_processor_count} SMs | "
        f"torch {torch.__version__} triton {triton.__version__} cuda {cuda}"
    )
