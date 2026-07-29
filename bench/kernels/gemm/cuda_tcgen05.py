"""Blackwell tcgen05 GEMM (sm_100a) roofline vs cuBLAS — the B200 rental-day bench.

Pre-registered target (PERF_PLAN Phase 3, B200 dense BF16 = 2,250 TF/s):
  ~1,209 TF/s (~54% of dense) for the 1-SM path EARLY; warp-spec climbs to
  ~1,300–1,476 TF/s. This bench measures the fraction the promoted skeleton
  reaches on a B200.

Run on the rental box (B200):
  PYTHONPATH=src python -m bench.kernels.gemm.cuda_tcgen05   (or --n 8192)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from scratch_llm.kernels.common.arch import compute_capability, require_cc  # noqa: E402
from scratch_llm.kernels.gemm.cuda.tcgen05 import tcgen05_gemm  # noqa: E402

_B200_FP16_PEAK_TF = 2250.0  # B200 SXM dense fp16 (conservative; BF16 number)


def _bench_ms(fn) -> float:
    return float(triton.testing.do_bench(fn, warmup=50, rep=200, quantiles=[0.5]))


def main() -> None:
    require_cc(10, 0, fn_name="cuda_tcgen05 bench")
    ap = argparse.ArgumentParser(description="Blackwell tcgen05 GEMM roofline")
    ap.add_argument("--n", type=int, default=4096, help="square GEMM dim (M mult of 128, N of 256)")
    args = ap.parse_args()
    n = args.n
    if n % 256 != 0:
        raise SystemExit(f"--n must be a multiple of 256 (tcgen05 atom), got {n}")

    dev = torch.cuda.get_device_name()
    print(f"# tcgen05 GEMM {n}^3 f16->fp32 · {dev} · cc={compute_capability()}")

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    flops = 2.0 * n**3

    out = tcgen05_gemm(a, b)
    ref = torch.matmul(a.float(), b.float())
    rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    print(f"# correctness: normalized rel err vs fp32 torch.matmul = {rel:.2e}  (rental-day gate)")

    ms = _bench_ms(lambda a=a, b=b: tcgen05_gemm(a, b))
    tf = flops / (ms * 1e-3) / 1e12
    ms_c = _bench_ms(lambda a=a, b=b: torch.matmul(a, b))
    tf_c = flops / (ms_c * 1e-3) / 1e12

    print(f"# tcgen05  : {tf:6.1f} TF/s  ({ms:8.3f} ms)")
    print(f"# cuBLAS   : {tf_c:6.1f} TF/s  ({ms_c:8.3f} ms)  [proxy]")
    print(f"# %-of-cuBLAS   : {100 * tf / tf_c:5.1f}%")
    print(f"# %-of-B200-peak: {100 * tf / _B200_FP16_PEAK_TF:5.1f}%")
    print("# PRE-REGISTERED: ~1,209 TF/s 1-SM path (PERF_PLAN Phase 3)")

    del a, b, out, ref
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
