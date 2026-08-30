"""A3 Rung 2 — mma.sync + ldmatrix + XOR-swizzled-SMEM warp-MMA GEMM roofline vs cuBLAS.

The DoD of this rung is the *number*: what fraction of cuBLAS (torch.matmul, the cuBLAS proxy) the
hand-issued warp-MMA kernel reaches at >=4096^3, and its %-of-the-measured-72-TF/s-bf16-peak on THIS
sm120 card (RTX PRO 4000 Blackwell). At 4096^3 the operands spill L2, so the number reflects real
tensor-core throughput, not an L1-resident fantasy. The kernel takes float16 operands (mma.f16.f16),
accumulates and outputs FP32; cuBLAS is measured on the same float16 inputs with fp32-accumulate and
reduced-precision reduction OFF, the fairest apples-to-apples proxy.

  Run on the GPU box:  PYTHONPATH=../../src python -m bench.kernels.gemm.cuda_mma_sync   (or --n 8192)

ncu-debt (blocked here — ERR_NVGPUCTRPERM, unprivileged): the shared-load bank-conflict count
(l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum, target ~0) is UNMEASURABLE on this box.
The XOR swizzle's correctness is argued structurally (see the kernel header + the printout below);
the metric is recorded, not claimed as measured.

Discharge on **sm_120 silicon with counters enabled** (a `vms_enabled` KVM 5090, ~$0.33/hr) — NOT
the H100 day, which this was misfiled to on 2026-07-04 and which cannot discharge it even in
principle: this kernel is built `-arch=sm_120` and a compute_120 cubin does not load on sm_90 (PTX
compat is forward-only). The bank-conflict claim is about THIS compilation on THIS arch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from scratch_llm.kernels.gemm.cuda.mma_sync import gemm_mma_sync  # noqa: E402

_BF16_PEAK_TF = 72.0  # measured bf16 tensor-core roof on this sm120 card (bench/RESULTS.md A2 R0)


def _bench_ms(fn) -> float:
    return float(triton.testing.do_bench(fn, warmup=50, rep=200, quantiles=[0.5]))


def _cublas_ref(a: Tensor, b: Tensor) -> Tensor:
    return torch.matmul(a, b)


def main() -> None:
    ap = argparse.ArgumentParser(description="A3 R2 — mma.sync warp-MMA GEMM roofline")
    ap.add_argument("--n", type=int, default=4096, help="square GEMM dim (>=4096 spills L2)")
    args = ap.parse_args()
    n = args.n

    dev = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    print(f"# A3 R2 mma.sync GEMM {n}^3 f16->fp32 · {dev} sm_{cap[0]}{cap[1]}")

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    flops = 2.0 * n**3

    # correctness gate before any timing (a fast-but-wrong kernel is not a result)
    out = gemm_mma_sync(a, b)
    ref = torch.matmul(a.float(), b.float())
    rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    assert rel <= 1e-2, f"correctness gate failed: normalized rel err {rel:.3e}"
    print(f"# correctness: normalized rel err vs fp32 torch.matmul = {rel:.2e}  (<= 1e-2 OK)")

    ms = _bench_ms(lambda a=a, b=b: gemm_mma_sync(a, b))
    tf = flops / (ms * 1e-3) / 1e12

    ms_c = _bench_ms(lambda a=a, b=b: _cublas_ref(a, b))
    tf_c = flops / (ms_c * 1e-3) / 1e12

    print(f"# mma.sync : {tf:6.1f} TF/s  ({ms:8.3f} ms)")
    print(f"# cuBLAS   : {tf_c:6.1f} TF/s  ({ms_c:8.3f} ms)  [proxy]")
    print(f"# %-of-cuBLAS      : {100 * tf / tf_c:5.1f}%")
    print(f"# %-of-72-TF-bf16  : {100 * tf / _BF16_PEAK_TF:5.1f}%")
    print(
        "# ncu-debt: l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ~0 "
        "(UNMEASURABLE here — ERR_NVGPUCTRPERM; swizzle correctness is structural; discharge on "
        "sm_120 + counters, NOT H100 — compute_120 does not load on sm_90)"
    )

    del a, b, out, ref
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
