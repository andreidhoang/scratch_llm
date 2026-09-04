"""Hopper WGMMA GEMM (sm_90a) roofline vs cuBLAS — the rental-day headliner bench.

Pre-registered target (performance/PERF_ENGINEERING_SPEC.md §4 · A2 §4.2 / A3 R3.1, book §7.3.1):
  ≈318 TFLOPS on H100 at 4096³ FP16 (~4.5× over WMMA's 71 TF/s). This bench
  measures the *fraction of that* the promoted skeleton reaches.

The kernel body (``csrc/gemm/wgmma_sm90.cu``) is the compile-gated structural
skeleton; correctness is gated before timing by the same normalized-rel-err
metric as the other GEMM benches. This bench is the DoD of the Hopper rung.

Run on the rental box (H100/H200):
  PYTHONPATH=src python -m bench.kernels.gemm.cuda_wgmma   (or --n 8192)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from scratch_llm.kernels.common.arch import compute_capability, require_arch  # noqa: E402
from scratch_llm.kernels.gemm.cuda.wgmma import wgmma_gemm  # noqa: E402

# H100 fp16 dense peak (the datasheet roof). The bench reports % of this.
_H100_FP16_PEAK_TF = 989.0  # H100 SXM5 / 80GB dense fp16


def _bench_ms(fn) -> float:
    return float(triton.testing.do_bench(fn, warmup=50, rep=200, quantiles=[0.5]))


def main() -> None:
    require_arch((9, 0), fn_name="cuda_wgmma bench")
    ap = argparse.ArgumentParser(description="Hopper WGMMA GEMM roofline")
    ap.add_argument("--n", type=int, default=4096, help="square GEMM dim (multiples of 64)")
    args = ap.parse_args()
    n = args.n
    if n % 64 != 0:
        raise SystemExit(f"--n must be a multiple of 64 (wgmma tile), got {n}")

    dev = torch.cuda.get_device_name()
    print(f"# WGMMA GEMM {n}^3 f16->fp32 · {dev} · cc={compute_capability()}")

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    flops = 2.0 * n**3

    out = wgmma_gemm(a, b)
    ref = torch.matmul(a.float(), b.float())
    rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    print(f"# correctness: normalized rel err vs fp32 torch.matmul = {rel:.2e}  (rental-day gate)")
    # NOTE: no hard assert — the skeleton's runtime correctness is deferred to rental day.
    # The gate is the test_gemm_wgmma.py oracle, not the bench. Document the number honestly.

    ms = _bench_ms(lambda a=a, b=b: wgmma_gemm(a, b))
    tf = flops / (ms * 1e-3) / 1e12
    ms_c = _bench_ms(lambda a=a, b=b: torch.matmul(a, b))
    tf_c = flops / (ms_c * 1e-3) / 1e12

    print(f"# wgmma    : {tf:6.1f} TF/s  ({ms:8.3f} ms)")
    print(f"# cuBLAS   : {tf_c:6.1f} TF/s  ({ms_c:8.3f} ms)  [proxy]")
    print(f"# %-of-cuBLAS    : {100 * tf / tf_c:5.1f}%")
    print(f"# %-of-H100-peak : {100 * tf / _H100_FP16_PEAK_TF:5.1f}%")
    print("# PRE-REGISTERED TARGET (book §7.3.1): ~318 TF/s basic WGMMA on H100 @4096³")

    del a, b, out, ref
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
