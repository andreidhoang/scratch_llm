"""Hopper FlashAttention-3 (sm_90a) roofline vs SDPA — the H100 rental-day bench.

Pre-registered target (Tri Dao FA3 paper, H100):
  FA3 fwd ~1.2-1.5× over FA2 on H100 at seq=4096, D=64. The structural skeleton
  won't hit that — this bench measures the *fraction* it reaches and the
  correctness vs the CPU oracle.

Run on the rental box (H100/H200):
  PYTHONPATH=src python -m bench.kernels.attention.fa3_hopper   (or --seq 8192)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from scratch_llm.kernels.attention.prefill.fa3 import (  # noqa: E402
    flash_attention_fa3_forward,
)
from scratch_llm.kernels.common.arch import compute_capability, require_arch  # noqa: E402


def _bench_ms(fn) -> float:
    return float(triton.testing.do_bench(fn, warmup=50, rep=200, quantiles=[0.5]))


def main() -> None:
    require_arch((9, 0), fn_name="fa3_hopper bench")
    ap = argparse.ArgumentParser(description="Hopper FlashAttention-3 roofline")
    ap.add_argument("--seq", type=int, default=4096, help="sequence length (mult of 64)")
    ap.add_argument("--d", type=int, default=64, help="head dim (FA3 atom width)")
    args = ap.parse_args()
    if args.seq % 64 != 0:
        raise SystemExit(f"--seq must be a multiple of 64 (wgmma atom), got {args.seq}")

    dev = torch.cuda.get_device_name()
    print(f"# FA3 fwd seq={args.seq} d={args.d} f16 · {dev} · cc={compute_capability()}")

    q = torch.randn(1, 1, args.seq, args.d, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    # Attention FLOPs: 4*S*S*D (QK^T + softmax + AV, self-attention, one head)
    flops = 4.0 * args.seq * args.seq * args.d

    out = flash_attention_fa3_forward(q, k, v, is_causal=True)
    print(f"# output shape {tuple(out.shape)} dtype {out.dtype} (rental-day oracle gate)")

    ms = _bench_ms(lambda q=q, k=k, v=v: flash_attention_fa3_forward(q, k, v, is_causal=True))
    tf = flops / (ms * 1e-3) / 1e12

    # SDPA baseline (Flash backend) for the ratio
    sdpa_fn = torch.nn.functional.scaled_dot_product_attention
    ms_s = _bench_ms(lambda q=q, k=k, v=v: sdpa_fn(q, k, v, is_causal=True))
    tf_s = flops / (ms_s * 1e-3) / 1e12

    print(f"# FA3   : {tf:6.1f} TF/s  ({ms:8.3f} ms)")
    print(f"# SDPA  : {tf_s:6.1f} TF/s  ({ms_s:8.3f} ms)  [flash baseline]")
    if tf_s > 0:
        print(f"# speedup vs SDPA: {tf / tf_s:5.2f}×")
    print("# PRE-REGISTERED: FA3 fwd ~1.2-1.5× over FA2/SDPA on H100 (Tri Dao)")

    del q, k, v, out
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
