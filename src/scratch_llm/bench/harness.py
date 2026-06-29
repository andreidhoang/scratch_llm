"""Timing harness — turn a callable into a trustworthy measurement.

CUDA is asynchronous: a wall-clock timer around a kernel launch measures *enqueue* time, not compute.
So this synchronizes (CUDA events on device, ``perf_counter`` + ``synchronize`` on the host path) and
reports a distribution (median + p10/p90 + min), never a single noisy sample. The median is what you
score against the roofline; the spread tells you whether the number is stable enough to trust.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchStats:
    """The distribution of per-iteration times (seconds) — report the median, not a lone sample."""

    median: float
    mean: float
    std: float
    p10: float
    p90: float
    min: float
    iters: int

    @property
    def median_ms(self) -> float:
        return self.median * 1e3


def _summarize(times: list[float], iters: int) -> BenchStats:
    ordered = sorted(times)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
        return ordered[idx]

    return BenchStats(
        median=statistics.median(ordered),
        mean=statistics.fmean(ordered),
        std=statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        p10=pct(0.10),
        p90=pct(0.90),
        min=ordered[0],
        iters=iters,
    )


def benchmark(
    fn: Callable[[], object], *, warmup: int = 10, iters: int = 50, use_cuda: bool | None = None
) -> BenchStats:
    """Time ``fn`` ``iters`` times after ``warmup`` untimed calls, with correct synchronization.

    ``use_cuda`` defaults to autodetect (CUDA events when a device is present, else host timing).
    Each iteration is timed individually so the spread is real; warmup hides one-time JIT / autotune /
    allocator costs (and CUDA-graph capture, if any).
    """
    if use_cuda is None:
        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except ImportError:
            use_cuda = False

    for _ in range(warmup):
        fn()

    if use_cuda:
        import torch

        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends, strict=True)]
        return _summarize(times, iters)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return _summarize(times, iters)
