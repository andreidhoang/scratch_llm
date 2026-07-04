"""A1 Rung 4.6 — prefill/decode disaggregation (demonstrate, toy scale).

Prefill (compute-bound, one big GEMM over the whole prompt) and decode (memory-bound GEMV, one token)
have opposite profiles. **Co-locating** them means a prefill burst steals the GPU from in-flight
decodes — the measured ITL admission spike (R4.1/R4.2: p99 ~29–200 ms vs ~6 ms p50). **Disaggregation**
runs prefill on a separate worker so the decode worker's stream is never interrupted; the cost is a
one-time **KV-cache transfer** from the prefill worker to the decode worker.

On a single GPU we can't place two real workers, so we demonstrate the two measurable halves:
1. the KV-transfer cost (a device-to-device copy of a prefilled request's paged blocks) vs its
   analytic bound (bytes ÷ HBM bandwidth), and vs a decode step — the disagg *tax*;
2. the ITL that a disaggregated decode worker sees (a clean decode stream, no prefill) vs a
   co-located engine (decode with interleaved prefill bursts) — the spike disagg *removes*.
Goodput under an ITL SLO then favours disagg: it trades a large recurring ITL spike for a small
one-time transfer.

Run:  python bench/disagg.py --device cuda
"""

from __future__ import annotations

import argparse
import random
import time

import torch
from decode_roofline import RUNG1_CONFIG, count_params  # noqa: E402

from scratch_llm.model import PagedKVCache, PrefillView, TransformerLM
from scratch_llm.serving.continuous import Request, serve_continuous
from scratch_llm.serving.metrics import SLO, summarize

N_SLOTS = 32
LONG_PROMPT = 512


def _prefill_one(model: TransformerLM, plen: int, device: str) -> PagedKVCache:
    """Prefill a single request into a fresh paged cache (the 'prefill worker' output)."""
    blocks = (plen + PagedKVCache.BLOCK - 1) // PagedKVCache.BLOCK + 1
    cache = PagedKVCache(
        n_layers=model.cfg.n_layers,
        n_slots=1,
        n_kv_heads=model.cfg.kv_heads,
        max_ctx=model.cfg.context_length,
        head_dim=model.cfg.head_dim,
        n_blocks=blocks + 1,
        device=device,
        dtype=torch.bfloat16,
    )
    rng = random.Random(0)
    x = torch.tensor([[rng.randrange(RUNG1_CONFIG.vocab_size) for _ in range(plen)]], device=device)
    with torch.no_grad():
        model(x, PrefillView(cache, [0], [plen]))
    cache.mirror_admit([0], [plen])
    return cache


def measure_kv_transfer(model: TransformerLM, device: str) -> None:
    """The disagg tax: copy a prefilled request's KV pool to a second cache (D2D), timed + analytic."""
    src = _prefill_one(model, LONG_PROMPT, device)
    dst = _prefill_one(model, LONG_PROMPT, device)
    # KV bytes actually transferred = the live blocks of the prompt across all layers.
    live_blocks = len(src._py_table[0])
    kv_bytes = (
        2  # K and V
        * model.cfg.n_layers
        * live_blocks
        * PagedKVCache.BLOCK
        * model.cfg.kv_heads
        * model.cfg.head_dim
        * 2  # bf16
    )

    def _copy() -> None:
        for layer in range(model.cfg.n_layers):
            dst._pool_k[layer].copy_(src._pool_k[layer])
            dst._pool_v[layer].copy_(src._pool_v[layer])

    _copy()
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        _copy()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    ms = ts[len(ts) // 2]
    # measured D2D bandwidth for the pool copy (2× the bytes: read src + write dst)
    gbps = (2 * kv_bytes) / (ms / 1e3) / 1e9
    print(
        f"\n# KV-transfer tax (disagg): {LONG_PROMPT}-tok prompt, {live_blocks} blocks/layer × "
        f"{model.cfg.n_layers} layers = {kv_bytes / 1e6:.1f} MB KV"
    )
    print(
        f"  D2D copy {ms:.3f} ms | {gbps:.0f} GB/s (of ~0.55 TB/s HBM) | "
        f"analytic {2 * kv_bytes / 0.55e12 * 1e3:.3f} ms — a ONE-TIME per-request cost"
    )


def measure_itl_contrast(model: TransformerLM, prefill_model: TransformerLM, device: str) -> None:
    """Decode-only (disaggregated decode worker) vs decode+interleaved-prefill (co-located)."""
    rng = random.Random(0)

    def short(i: int) -> Request:
        return Request(i, tuple(rng.randrange(RUNG1_CONFIG.vocab_size) for _ in range(32)), 96)

    def long(i: int) -> Request:
        return Request(
            i, tuple(rng.randrange(RUNG1_CONFIG.vocab_size) for _ in range(LONG_PROMPT)), 48
        )

    # disaggregated decode worker: only short-prompt decode jobs arrive (prefill happened elsewhere)
    decode_only = [short(i) for i in range(96)]
    # co-located: the same decode stream with periodic long prefills interleaved in-stream
    co_located = [long(i) if i % 6 == 5 else short(i) for i in range(96)]

    slo = SLO(ttft_s=1e9, itl_s=1e9)
    for name, trace in (("disagg decode-only", decode_only), ("co-located +prefill", co_located)):
        serve_continuous(model, trace[:24], N_SLOTS, device, prefill_model=prefill_model)  # warm
        res = serve_continuous(model, trace, N_SLOTS, device, prefill_model=prefill_model)
        rep = summarize([c.record for c in res.completed], slo)
        print(
            f"  {name:<20} | ITL p50/p95/p99 {rep.itl_ms.p50:5.1f}/{rep.itl_ms.p95:6.1f}/"
            f"{rep.itl_ms.p99:6.1f} ms | agg {rep.throughput_tok_s:6.0f} tok/s"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 R4.6 — prefill/decode disaggregation demo")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(0)
    model = TransformerLM(RUNG1_CONFIG).to(device=args.device, dtype=torch.bfloat16).eval()
    print(
        f"# A1 R4.6 PD-disaggregation | GQA-4 {count_params(model) / 1e9:.2f}B bf16 | {torch.cuda.get_device_name(0)}"
    )
    measure_kv_transfer(model, args.device)
    print("\n# ITL contrast: what the decode worker sees, disaggregated vs co-located")
    measure_itl_contrast(model, model, args.device)


if __name__ == "__main__":
    main()
