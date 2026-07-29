"""A2 R5+6 — bf16 GEMM ladder roofline: naive -> SMEM-tiled -> autotuned, measured vs cuBLAS.

The DoD of this rung is the *number*, not the kernel: what fraction of cuBLAS (torch.matmul, the
cuBLAS proxy) each ladder stage reaches at 4096^3, and where each sits on the measured roofline. The
shape to reproduce is the siboehm ladder — naive ~1% -> tiled -> autotuned toward ~70-80% — on THIS
sm120 card (do not cross-compare absolute numbers with siboehm's A6000/fp32 run).

At 4096^3 the arithmetic intensity (~1365 FLOP/byte minimal-traffic, ~512 under the A2 heavier model)
is far past this card's ridge (~131 FLOP/byte): the GEMM is compute-bound, so %-of-peak here means
%-of-the-bf16-tensor-core-roof, and %-of-cuBLAS is the tighter, fairer number (cuBLAS ≈ that roof).

Run on the GPU box:  PYTHONPATH=../../src python -m bench.kernels.gemm.triton_tiled   (or --n 8192)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

# bench/ on sys.path for the shared _harness + kernel_roofline (resolve upward to the dir holding
# _harness.py — move-proof: this script lives at bench/kernels/<family>/, two levels below bench/).
_BENCH_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_harness.py").exists())
sys.path.insert(0, str(_BENCH_ROOT))
from _harness import Roofs, bench_ms, provenance_line, spread_pct  # noqa: E402
from kernel_roofline import profile  # noqa: E402

sys.path.insert(0, str(_BENCH_ROOT.parent / "src"))
from scratch_llm.kernels.gemm.triton.tiled import (  # noqa: E402
    gemm_autotuned,
    gemm_naive,
    gemm_tiled,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="A2 R5+6 — GEMM ladder roofline")
    ap.add_argument("--n", type=int, default=4096, help="square GEMM dim (>=4096 spills L2)")
    ap.add_argument("--skip-naive", action="store_true", help="naive is ~1s/call; skip for speed")
    args = ap.parse_args()
    n = args.n

    print(provenance_line(f"A2 R5+6 — GEMM ladder {n}^3 bf16"))
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

    # cuBLAS baseline (the proxy) — measured the same way as the ladder; fp32 accumulate.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cublas_med = bench_ms(lambda a=a, b=b: torch.matmul(a, b))[0]
    cublas_tflops = flops / (cublas_med * 1e-3) / 1e12
    print(f"\n# cuBLAS (torch.matmul) baseline: {cublas_med:.3f} ms · {cublas_tflops:.1f} TF/s\n")

    stages: list[tuple[str, Callable[[Tensor, Tensor], Tensor]]] = [
        ("tiled", gemm_tiled),
        ("autotuned", gemm_autotuned),
    ]
    if not args.skip_naive:
        stages.insert(0, ("naive", gemm_naive))

    header = (
        f"{'stage':<12}{'ms':>10}{'spread%':>9}{'TF/s':>9}{'%roof':>8}{'bound':>7}{'%cuBLAS':>9}"
    )
    print(header)
    print(
        f"{'cuBLAS':<12}{cublas_med:>10.3f}{'—':>9}{cublas_tflops:>9.1f}{'—':>8}{'cmp':>7}{100.0:>9.1f}"
    )
    rows: list[tuple[str, float, float]] = []
    for name, fn in stages:
        # naive is ~1s/call — measure it with few reps so the shared GPU isn't tied up for minutes.
        if name == "naive":
            med, lo, hi = bench_ms(lambda fn=fn, a=a, b=b: fn(a, b), warmup=2, rep=6)
            sec = med * 1e-3
            tflops = flops / sec / 1e12
            attainable, bound = roofs.attainable(flops, bytes_)
            p_roof = 100.0 * (flops / sec) / attainable
            sp = spread_pct(med, lo, hi)
        else:
            p = profile(name, lambda fn=fn, a=a, b=b: fn(a, b), flops, bytes_, roofs)
            med, sp, tflops, p_roof, bound = p.ms, p.spread_pct, p.gflops / 1e3, p.pct_roof, p.bound
        pct_cublas = 100.0 * cublas_med / med
        rows.append((name, tflops, pct_cublas))
        print(
            f"{name:<12}{med:>10.3f}{sp:>9.1f}{tflops:>9.1f}{p_roof:>8.1f}{bound:>7}{pct_cublas:>9.1f}"
        )

    print(
        f"\n# AI = {ai:.0f} FLOP/byte  (ridge {roofs.ridge:.0f}) → compute-bound: %roof == %-of-bf16-tensor-roof"
    )
    print("# ladder shape (siboehm analog, this card):")
    for name, tflops, pct in rows:
        print(f"#   {name:<10} {tflops:>7.1f} TF/s  {pct:>6.1f}% of cuBLAS")

    del a, b
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
