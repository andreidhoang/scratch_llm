"""A3 Rung 1 — WMMA tensor-core GEMM roofline: hand-issued fragments vs cuBLAS at 4096^3.

The DoD of this rung is the *number*: what fraction of the FP16 tensor-core peak (and of cuBLAS, the
proxy for that peak) a by-hand ``nvcuda::wmma`` GEMM reaches on THIS sm120 card. At 4096^3 the
arithmetic intensity (~1365 FLOP/byte minimal-traffic) is far past the card's ridge, so the GEMM is
compute-bound and %-of-peak means %-of-the-FP16-tensor-core-roof.

This is a *synchronous* WMMA kernel (128x128 block tile, 8 warps x 4x2 fragments, FP32 accumulate) with
no cp.async double-buffering — the book's "basic WMMA" cut. Expect a meaningful-but-sub-cuBLAS fraction;
the identified path to the tuned 40-60% band is cp.async GMEM->SMEM overlap (see the module docstring).

Run on the GPU box:  PYTHONPATH=../../src python -m bench.kernels.gemm.wmma   (or --n 8192)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# bench/ on sys.path for the shared _harness (resolve upward to the dir holding _harness.py).
_BENCH_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_harness.py").exists())
sys.path.insert(0, str(_BENCH_ROOT))
from _harness import Roofs, bench_ms, provenance_line, spread_pct  # noqa: E402

sys.path.insert(0, str(_BENCH_ROOT.parent / "src"))
from scratch_llm.kernels.gemm.wmma.gemm import wmma_gemm  # noqa: E402

# Nominal FP16/BF16 tensor-core peak quoted for this RTX PRO 4000 Blackwell (sm120), for the
# %-of-peak column. cuBLAS on this card measures a hair above it, so %-of-cuBLAS is the tighter number.
PEAK_TF = 72.0


def main() -> None:
    ap = argparse.ArgumentParser(description="A3 R1 — WMMA GEMM roofline")
    ap.add_argument("--n", type=int, default=4096, help="square GEMM dim (>=4096 spills L2)")
    args = ap.parse_args()
    n = args.n

    print(provenance_line(f"A3 R1 — WMMA GEMM {n}^3 fp16-in/fp32-acc"))
    roofs = Roofs.measure(torch.float16)
    print(
        f"# measured peaks: {roofs.bw_bytes_s / 1e12:.3f} TB/s HBM · "
        f"{roofs.compute_flops_s / 1e12:.1f} TF/s fp16(cuBLAS) · ridge {roofs.ridge:.0f} FLOP/byte "
        f"· nominal peak {PEAK_TF:.0f} TF/s"
    )

    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    flops = 2.0 * n**3
    bytes_ = (
        3.0 * n * n * 2
    )  # read A,B (fp16) + write C (fp32 counted as 2B fp16-equiv; minimal model)
    ai = flops / bytes_

    cublas_med, clo, chi = bench_ms(lambda a=a, b=b: torch.matmul(a, b))
    cublas_tf = flops / (cublas_med * 1e-3) / 1e12
    print(f"\n# cuBLAS (torch.matmul) baseline: {cublas_med:.3f} ms · {cublas_tf:.1f} TF/s\n")

    header = f"{'kernel':<12}{'ms':>10}{'spread%':>9}{'TF/s':>9}{'%peak':>8}{'%cuBLAS':>9}"
    print(header)
    print(
        f"{'cuBLAS':<12}{cublas_med:>10.3f}{spread_pct(cublas_med, clo, chi):>9.1f}"
        f"{cublas_tf:>9.1f}{100 * cublas_tf / PEAK_TF:>8.1f}{100.0:>9.1f}"
    )

    med, lo, hi = bench_ms(lambda a=a, b=b: wmma_gemm(a, b))
    tf = flops / (med * 1e-3) / 1e12
    print(
        f"{'wmma':<12}{med:>10.3f}{spread_pct(med, lo, hi):>9.1f}"
        f"{tf:>9.1f}{100 * tf / PEAK_TF:>8.1f}{100 * cublas_med / med:>9.1f}"
    )

    print(
        f"\n# AI = {ai:.0f} FLOP/byte (ridge {roofs.ridge:.0f}) → compute-bound: "
        f"%peak == %-of-fp16-tensor-roof"
    )
    print(
        "# synchronous WMMA (no cp.async double-buffer); next optimization = GMEM->SMEM overlap "
        "to hide the load behind the MMA (the book's climb into the 40-60% tuned band)."
    )
    print(
        "# ncu-debt: l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ~= 0 target "
        "(SMEM padded APAD/BPAD=8) — UNMEASURABLE here (ERR_NVGPUCTRPERM); verify on sm_120 + "
        "counters (KVM 5090), NOT H100: this kernel is -arch=sm_120 and cannot load on sm_90."
    )

    del a, b
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
