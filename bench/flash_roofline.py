"""A2.1 roofline — Triton FA2 forward vs F.scaled_dot_product_attention.

The deliverable of the FA2 work is THIS number, not the kernel (A2 guide §6/§8). Predict-before-run
(docs/design/L2_flash_attention_SPEC.md §4): the Triton forward should reach ≥ 60% of SDPA at
seq 4096, d=64, causal, bf16 on the 4090 (predicted ~65%); below 60% → ship the honest gap.

Run on the GPU box:  python bench/flash_roofline.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from reasoning_llm.kernels.flash_attention_triton import flash_attention_triton_forward


def _tflops(b: int, h: int, n: int, d: int, ms: float, causal: bool) -> float:
    # Attention fwd FLOPs ≈ 4·B·H·N²·d (QKᵀ then PV, 2 FLOP/MAC); causal attends ~half the pairs.
    flops = 4.0 * b * h * n * n * d * (0.5 if causal else 1.0)
    return flops / (ms * 1e-3) / 1e12


def run(
    seqs: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384),
    b: int = 2,
    h: int = 8,
    d: int = 64,
    causal: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    dev = torch.cuda.get_device_name(0)
    print(f"# FA2-Triton vs SDPA | {dev} | B={b} H={h} d={d} causal={causal} dtype={dtype}")
    print(f"{'seq':>6} {'triton_ms':>10} {'sdpa_ms':>9} {'tri_TFLOPs':>11} {'sdpa_TFLOPs':>12} {'%SDPA':>7}")
    for n in seqs:
        q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))
        t_tri = triton.testing.do_bench(lambda: flash_attention_triton_forward(q, k, v, is_causal=causal))
        t_sdpa = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
        tf_tri = _tflops(b, h, n, d, t_tri, causal)
        tf_sdpa = _tflops(b, h, n, d, t_sdpa, causal)
        pct = 100.0 * tf_tri / tf_sdpa
        print(f"{n:>6} {t_tri:>10.3f} {t_sdpa:>9.3f} {tf_tri:>11.1f} {tf_sdpa:>12.1f} {pct:>6.1f}%")


if __name__ == "__main__":
    run()
