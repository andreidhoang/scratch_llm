"""GPU peak-performance table — the denominators every roofline is scored against.

The numbers are **dense** tensor-core peaks (not the 2:4-sparse datasheet headlines — halve those for
real LLM weights) and HBM bandwidths, each with a dated source. Getting dense-vs-sparse right is a
silent credibility tell: the widely-quoted H100 "590 FLOP/byte" ridge uses the sparse 1979 TFLOP/s; the
honest dense ridge is ~295. ``FP4 = 2x FP8`` is the durable cross-Blackwell invariant.

A GPU's real roof is often below the datasheet (clocks, power); ``measure_hbm_bandwidth`` reads the
*achieved* bandwidth on the live device — the honest number to place a memory-bound kernel against.
"""

from __future__ import annotations

from dataclasses import dataclass

# Dtype keys used across the table and the roofline engine.
FP32 = "fp32"
TF32 = "tf32"
BF16 = "bf16"
FP16 = "fp16"
FP8 = "fp8"
FP4 = "fp4"


@dataclass(frozen=True)
class GpuSpec:
    """One GPU's dense tensor-core peaks (FLOP/s, per dtype) + HBM bandwidth (bytes/s).

    ``peak_flops`` omits a dtype the silicon lacks (e.g. Hopper has no FP4) so a roofline against an
    unsupported precision raises rather than silently inventing a number.
    """

    name: str
    hbm_bandwidth: float  # bytes/s (HBM, dense achievable headline)
    peak_flops: dict[str, float]  # dtype -> dense FLOP/s
    source: str

    def ridge_point(self, dtype: str) -> float:
        """Arithmetic intensity (FLOP/byte) where the compute roof meets the memory roof.

        Below this AI a kernel is memory-bandwidth-bound; above it, compute-bound.
        """
        if dtype not in self.peak_flops:
            raise KeyError(f"{self.name} has no dense {dtype} tensor-core path")
        return self.peak_flops[dtype] / self.hbm_bandwidth


# Dense peaks. Sources: NVIDIA H100/H200/B200/GB200 datasheets; FA3 paper (989 TFLOP/s H100 dense
# matmul, 1979 FP8); the repo's own RTX 4090 FA2 roofline (~165 fp16 TFLOP/s dense, ~1 TB/s). FP4 only
# exists on Blackwell (2x FP8). All values dense; do not substitute the 2:4-sparse headlines.
GPUS: dict[str, GpuSpec] = {
    "h100-sxm": GpuSpec(
        name="H100 SXM",
        hbm_bandwidth=3.35e12,
        peak_flops={BF16: 989e12, FP16: 989e12, FP8: 1979e12, TF32: 494e12, FP32: 67e12},
        source="NVIDIA H100 datasheet; FA3 arXiv:2407.08608 (dense matmul peak)",
    ),
    "h200": GpuSpec(
        name="H200 SXM",
        hbm_bandwidth=4.8e12,  # the decode-memory-wall upgrade over H100, same compute die
        peak_flops={BF16: 989e12, FP16: 989e12, FP8: 1979e12, TF32: 494e12, FP32: 67e12},
        source="NVIDIA H200 datasheet (HBM3e 4.8 TB/s; H100 compute)",
    ),
    "b200": GpuSpec(
        name="B200 (DGX)",
        hbm_bandwidth=8.0e12,
        peak_flops={BF16: 2.25e15, FP16: 2.25e15, FP8: 4.5e15, FP4: 9.0e15},
        source="NVIDIA Blackwell/DGX B200 datasheet (dense; FP4 = 2x FP8)",
    ),
    "gb200": GpuSpec(
        name="GB200 (per GPU)",
        hbm_bandwidth=8.0e12,
        peak_flops={BF16: 2.5e15, FP16: 2.5e15, FP8: 5.0e15, FP4: 10.0e15},
        source="NVIDIA GB200 NVL72 datasheet (per-GPU dense; the marketing 20 PF FP4 is sparse)",
    ),
    "rtx4090": GpuSpec(
        name="RTX 4090 (Ada)",
        hbm_bandwidth=1.008e12,
        peak_flops={BF16: 165e12, FP16: 165e12, TF32: 82.6e12, FP32: 82.6e12},
        source="NVIDIA Ada datasheet (fp16 w/ fp32-accumulate dense); repo FA2 roofline",
    ),
}


def measure_hbm_bandwidth(n_bytes: int = 1 << 28, iters: int = 50) -> float:
    """Empirically measure achieved HBM bandwidth (bytes/s) on the live CUDA device via a large
    device-to-device copy. The *honest* memory roof — usually 70-90% of the datasheet headline — and
    the right denominator for a memory-bound kernel on a card not in :data:`GPUS`.

    Raises if CUDA is unavailable (the measurement is meaningless on CPU).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("measure_hbm_bandwidth needs a CUDA device")
    device = torch.device("cuda")
    n = n_bytes // 4  # fp32 elements
    src = torch.empty(n, dtype=torch.float32, device=device)
    dst = torch.empty(n, dtype=torch.float32, device=device)

    for _ in range(5):  # warmup
        dst.copy_(src)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1e3 / iters
    moved = 2 * n_bytes  # one read + one write per copy
    return moved / seconds
