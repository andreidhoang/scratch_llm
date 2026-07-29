"""A3 R0 — naive SMEM-blocked bf16 GEMM (CUDA C++, CUDA cores) roofline vs cuBLAS.

The DoD of this rung is the *number*: what fraction of cuBLAS (torch.matmul proxy) and of the
bf16-tensor-core roof a pure-CUDA-core, shared-memory-tiled hgemm reaches at >=2048^3. Expect a low
single-digit %-of-cuBLAS — this is the floor with NO tensor cores; the whole point of A3 is that WMMA
and mma.sync beat it. Measured the same way as every other kernel bench (CUDA-event timing, L2 flush).

Run on the GPU box:  PYTHONPATH=../src python gemm_smem_cuda.py   (or --n 4096)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # bench/ for the R0 harness
from _harness import Roofs, bench_ms, provenance_line, spread_pct  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scratch_llm.kernels.gemm_smem_cuda import gemm_smem  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="A3 R0 — SMEM-blocked bf16 GEMM roofline")
    ap.add_argument("--n", type=int, default=2048, help="square GEMM dim (>=2048 exceeds L1)")
    args = ap.parse_args()
    n = args.n

    print(provenance_line(f"A3 R0 — SMEM-blocked bf16 GEMM {n}^3 (CUDA cores, no tensor cores)"))
    roofs = Roofs.measure(torch.bfloat16)
    print(
        f"# measured peaks: {roofs.bw_bytes_s / 1e12:.3f} TB/s HBM · "
        f"{roofs.compute_flops_s / 1e12:.1f} TF/s bf16 · ridge {roofs.ridge:.0f} FLOP/byte"
    )

    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    flops = 2.0 * n**3
    bytes_ = 3.0 * n * n * 2  # read A,B + write C, bf16 (minimal-traffic model)
    ai = flops / bytes_

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cublas_med = bench_ms(lambda a=a, b=b: torch.matmul(a, b))[0]
    cublas_tflops = flops / (cublas_med * 1e-3) / 1e12
    print(f"\n# cuBLAS (torch.matmul) baseline: {cublas_med:.3f} ms · {cublas_tflops:.1f} TF/s\n")

    header = f"{'stage':<14}{'ms':>10}{'spread%':>9}{'TF/s':>9}{'%roof':>8}{'%cuBLAS':>9}{'%72TF':>8}"
    print(header)
    print(
        f"{'cuBLAS':<14}{cublas_med:>10.3f}{'—':>9}{cublas_tflops:>9.1f}"
        f"{'—':>8}{100.0:>9.1f}{100.0 * cublas_tflops / 72.0:>8.1f}"
    )

    med, lo, hi = bench_ms(lambda a=a, b=b: gemm_smem(a, b), warmup=5, rep=30)
    sec = med * 1e-3
    tflops = flops / sec / 1e12
    attainable, bound = roofs.attainable(flops, bytes_)
    p_roof = 100.0 * (flops / sec) / attainable
    pct_cublas = 100.0 * cublas_med / med
    pct_72 = 100.0 * tflops / 72.0
    print(
        f"{'smem_tiled':<14}{med:>10.3f}{spread_pct(med, lo, hi):>9.1f}{tflops:>9.1f}"
        f"{p_roof:>8.1f}{pct_cublas:>9.1f}{pct_72:>8.1f}"
    )

    print(
        f"\n# AI = {ai:.0f} FLOP/byte  (ridge {roofs.ridge:.0f}) → compute-bound ({bound}): "
        f"%roof == %-of-bf16-tensor-roof"
    )
    print(
        f"# baseline (no tensor cores): {tflops:.1f} TF/s · {pct_cublas:.1f}% of cuBLAS · "
        f"{pct_72:.1f}% of 72-TF/s bf16 peak — the CUDA-core floor WMMA climbs from"
    )

    del a, b
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
