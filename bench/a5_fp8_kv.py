"""A5 Rung 4 — FP8 E4M3 KV cache: measured summary (CPU toy demo).

Run: PYTHONPATH=src .venv/bin/python bench/a5_fp8_kv.py

Prints the rung's measured numbers: E4M3 vs INT4 tensor SQNR, the per-channel-K law under
channel outliers, end-to-end teacher-forced logit fidelity vs a BF16 KV cache, the INT4 stress
degradation, and the measured KV-byte halving.
"""

from __future__ import annotations

import torch

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.quant.fp8_kv import (
    QuantizedKVCache,
    dequantize_fp8_e4m3,
    dequantize_int4,
    greedy_generate_with_cache,
    quantize_fp8_e4m3,
    quantize_int4,
    sqnr_db,
)


@torch.no_grad()
def _tf_logits(
    model: TransformerLM, ids: list[int], prompt_len: int, cache: object
) -> torch.Tensor:
    x = torch.tensor([ids[:prompt_len]])
    logits = [model(x, cache)[0, -1]]  # type: ignore[arg-type]
    for t in range(prompt_len, len(ids)):
        x = torch.tensor([[ids[t]]])
        logits.append(model(x, cache)[0, -1])  # type: ignore[arg-type]
    return torch.stack(logits[:-1])


def main() -> None:
    torch.manual_seed(3)
    g = torch.randn(4, 64, 128)
    fp8 = dequantize_fp8_e4m3(*quantize_fp8_e4m3(g, -1))
    int4 = dequantize_int4(*quantize_int4(g, -1))
    print(f"tensor SQNR  FP8(E4M3)={sqnr_db(g, fp8):.2f} dB  INT4={sqnr_db(g, int4):.2f} dB")

    torch.manual_seed(1)
    k = torch.randn(1, 4, 256, 32)
    k[..., torch.randperm(32)[:2]] *= 10.0
    pc = dequantize_int4(*quantize_int4(k, 2))
    pt = dequantize_int4(*quantize_int4(k, 3))
    mse_pc = torch.mean((k - pc) ** 2).item()
    mse_pt = torch.mean((k - pt) ** 2).item()
    print(f"INT4 per-channel-K law   MSE(per-token)/MSE(per-channel) = {mse_pt / mse_pc:.2f}x")

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=256, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, context_length=256
    )
    model = TransformerLM(cfg)
    model.eval()
    ids = list(range(1, 49))
    ref = _tf_logits(model, ids, 16, QuantizedKVCache(cfg.n_layers, "bf16"))
    f8 = _tf_logits(model, ids, 16, QuantizedKVCache(cfg.n_layers, "fp8"))
    i4 = _tf_logits(model, ids, 16, QuantizedKVCache(cfg.n_layers, "int4"))
    f8_mse = torch.mean((ref - f8) ** 2).item()
    i4_mse = torch.mean((ref - i4) ** 2).item()
    print(
        f"E2E logit vs BF16-KV  FP8: SQNR={sqnr_db(ref, f8):.2f} dB MSE={f8_mse:.2e} | "
        f"INT4: SQNR={sqnr_db(ref, i4):.2f} dB MSE={i4_mse:.2e} ({i4_mse / f8_mse:.2f}x worse)"
    )

    bf16c = QuantizedKVCache(cfg.n_layers, "bf16")
    fp8c = QuantizedKVCache(cfg.n_layers, "fp8")
    int4c = QuantizedKVCache(cfg.n_layers, "int4")
    for c in (bf16c, fp8c, int4c):
        greedy_generate_with_cache(model, list(range(1, 17)), 32, c)
    print(
        f"KV bytes  BF16={bf16c.kv_bytes}  "
        f"FP8={fp8c.kv_bytes} ({fp8c.kv_bytes / bf16c.kv_bytes:.3f}x)  "
        f"INT4={int4c.kv_bytes} ({int4c.kv_bytes / bf16c.kv_bytes:.3f}x)"
    )


if __name__ == "__main__":
    main()
