"""A5 Rung 3 bench — NVFP4 vs MXFP4 per-block MSE histogram + the mechanism attribution.

Run: ``PYTHONPATH=src .venv/bin/python bench/a5_nvfp4_mxfp4.py``
Emits ``bench/a5_nvfp4_mxfp4.png`` (per-block MSE histogram) and prints the SQNR/MSE table plus the
one-variable ablations that attribute the NVFP4 win to (a) the E4M3 non-power-of-two block scale and
(b) the finer k=16 block. This is a measurement harness, not a test — the assertions live in
``tests/test_quant_nvfp4_mxfp4.py``.
"""

from __future__ import annotations

import pathlib

import torch

from scratch_llm.quant.nvfp4_mxfp4 import (
    block_mse,
    mxfp4_dequantize,
    mxfp4_quantize,
    nvfp4_dequantize,
    nvfp4_quantize,
    quant_linear_nvfp4,
    sqnr_db,
)


def _mse(x: torch.Tensor, xh: torch.Tensor) -> float:
    return float(((x - xh) ** 2).mean())


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(8192, 256)  # i.i.d. N(0,1); 2.1M elements

    nv = nvfp4_dequantize(nvfp4_quantize(x, block=16))
    mx = mxfp4_dequantize(mxfp4_quantize(x, block=32))

    mse_nv, mse_mx = _mse(x, nv), _mse(x, mx)
    print("=== NVFP4 (k=16, E4M3 block scale) vs MXFP4 (k=32, E8M0 block scale) ===")
    print(f"NVFP4 : MSE {mse_nv:.4e}   SQNR {sqnr_db(x, nv):6.2f} dB")
    print(f"MXFP4 : MSE {mse_mx:.4e}   SQNR {sqnr_db(x, mx):6.2f} dB")
    print(f"MSE(MXFP4)/MSE(NVFP4) = {mse_mx / mse_nv:.3f}x   (>1 ⇒ NVFP4 wins, load-bearing)\n")

    # --- mechanism attribution (one variable at a time) ---
    nv16 = sqnr_db(x, nvfp4_dequantize(nvfp4_quantize(x, block=16)))
    nv32 = sqnr_db(x, nvfp4_dequantize(nvfp4_quantize(x, block=32)))
    mx16 = sqnr_db(x, mxfp4_dequantize(mxfp4_quantize(x, block=16)))
    mx32 = sqnr_db(x, mxfp4_dequantize(mxfp4_quantize(x, block=32)))
    print("mechanism 1 — scale format @ k=32 (E4M3 vs E8M0):"
          f" NVFP4 {nv32:.2f} vs MXFP4 {mx32:.2f} dB  → +{nv32 - mx32:.2f} dB")
    print("mechanism 2 — block size (k=16 vs k=32), per codec:"
          f" E4M3 {nv16:.2f}>{nv32:.2f} (+{nv16 - nv32:.2f}),"
          f" E8M0 {mx16:.2f}>{mx32:.2f} (+{mx16 - mx32:.2f}) dB\n")

    # --- software block-scaled GEMM ---
    w = torch.randn(256, 512)
    xin = torch.randn(64, 512)
    y = quant_linear_nvfp4(xin, w, block=16)
    w_hat = nvfp4_dequantize(nvfp4_quantize(w, block=16))
    y_dq = xin @ w_hat.t()
    y_bf16_same = (xin.to(torch.bfloat16) @ w_hat.to(torch.bfloat16).t()).float()
    y_bf16_orig = (xin.to(torch.bfloat16) @ w.to(torch.bfloat16).t()).float()
    print("=== software block-scaled GEMM (NVFP4 weights, scales in accumulation) ===")
    print(f"vs fp32 dequant-matmul (same W)  : {(y - y_dq).norm() / y_dq.norm():.2e}  (accumulation exact)")
    print(f"vs BF16 matmul (same W)          : {100 * (y - y_bf16_same).norm() / y_bf16_same.norm():.3f}%  (<1% oracle)")
    print(f"vs BF16 matmul (original W)      : {100 * (y - y_bf16_orig).norm() / y_bf16_orig.norm():.3f}%  (honest 4-bit cost = the ~20 dB SQNR)\n")

    # --- per-block MSE histogram ---
    bm_nv = block_mse(x, nv, block=16).log10().numpy()
    bm_mx = block_mse(x, mx, block=32).log10().numpy()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(bm_nv, bins=60, alpha=0.6, label=f"NVFP4 k=16 (mean MSE {mse_nv:.2e})", color="#2a9d8f")
        ax.hist(bm_mx, bins=60, alpha=0.6, label=f"MXFP4 k=32 (mean MSE {mse_mx:.2e})", color="#e76f51")
        ax.set_xlabel("log10(per-block MSE)")
        ax.set_ylabel("block count")
        ax.set_title("Per-block MSE: NVFP4 (E4M3 scale, k=16) shifts left of MXFP4 (E8M0 scale, k=32)")
        ax.legend()
        fig.tight_layout()
        out = pathlib.Path(__file__).with_suffix(".png")
        fig.savefig(out, dpi=120)
        print(f"histogram → {out}")
    except ImportError:
        print("matplotlib not available; skipped the PNG (numbers above are the record)")


if __name__ == "__main__":
    main()
