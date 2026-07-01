"""A1 Rung 3a — static batched decode roofline: aggregate tok/s vs batch B (weight amortization).

R1 proved B=1 decode is memory- and overhead-bound (AI≈1). The fix that beats the wall is batching:
read each weight once, apply it to B rows → AI≈B, aggregate tok/s ≈ B × single-stream ceiling, until
either KV traffic (B·KV) or the compute ridge (sm120 ≈ 130 FLOP/byte, i.e. B≈130 at short ctx) binds.

This sweeps B and places each on the roofline (AI, achieved BW, %HBM), under ``torch.compile`` (the R1
fused path that actually reaches the wall — eager would be overhead-bound and hide the amortization).
Pre-registered (bench/RESULTS.md R3.2): aggregate tok/s at B=32 is **≥10×** B=1; the curve is ~linear
in the memory regime and bends toward the compute ceiling as AI crosses the ridge.

Run:  python bench/batched_decode.py                 # compiled sweep
      python bench/batched_decode.py --no-compile    # eager (expect the amortization to be overhead-masked)
"""

from __future__ import annotations

import argparse

import torch

from scratch_llm.bench.gpu_specs import GPUS
from scratch_llm.bench.harness import benchmark
from scratch_llm.bench.roofline import decode_step_flops_bytes
from scratch_llm.model import KVCache, TransformerLM

from decode_roofline import RUNG1_CONFIG, STANDING_GPU, count_params  # noqa: E402

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
PROMPT_LEN = 32  # short ctx → weights dominate, so AI≈B (isolate the weight-amortization lever)


@torch.no_grad()
def sweep(device: str, *, compile_mode: str | None, warmup: int, iters: int) -> None:
    spec = GPUS[STANDING_GPU]
    peak_bw, ridge = spec.hbm_bandwidth, spec.ridge_point("bf16")
    model = TransformerLM(RUNG1_CONFIG).to(device=device, dtype=torch.bfloat16).eval()
    n_params = count_params(model)
    run = model
    tag = "eager"
    if compile_mode:
        run = torch.compile(model, dynamic=True, mode=compile_mode)  # type: ignore[assignment]
        tag = f"compiled[{compile_mode}]"

    print(f"# A1 R3a batched decode ({tag}) | GQA-4 {n_params / 1e9:.2f}B | ctx={PROMPT_LEN} | ridge≈{ridge:.0f}")
    print(f"{'B':>4} {'AI':>6} {'agg tok/s':>10} {'per-stream':>11} {'ms/step':>8} {'GB/s':>7} {'%HBM':>6} {'bound':>8}")

    agg: dict[int, float] = {}
    for b in BATCHES:
        cache = KVCache(len(model.blocks))
        prompt = torch.randint(0, RUNG1_CONFIG.vocab_size, (b, PROMPT_LEN), device=device)
        run(prompt, cache)  # prefill B equal-length rows (untimed)
        tok = torch.randint(0, RUNG1_CONFIG.vocab_size, (b, 1), device=device)

        def step() -> None:
            run(tok, cache)  # one batched decode step; cache grows 1/call (negligible)  # noqa: B023

        per_step = benchmark(step, warmup=warmup, iters=iters).median
        flops, bytes_step = decode_step_flops_bytes(
            n_params,
            n_layers=RUNG1_CONFIG.n_layers,
            n_kv_heads=RUNG1_CONFIG.kv_heads,
            head_dim=RUNG1_CONFIG.head_dim,
            context_len=PROMPT_LEN,
            batch=b,
            weight_bytes=2,
            kv_bytes=2,
        )
        ai = flops / bytes_step
        agg_tok_s = b / per_step
        bw = bytes_step / per_step
        agg[b] = agg_tok_s
        print(
            f"{b:>4} {ai:>6.1f} {agg_tok_s:>10.0f} {agg_tok_s / b:>11.0f} {per_step * 1e3:>7.2f} "
            f"{bw / 1e9:>7.0f} {100 * bw / peak_bw:>5.0f}% {'memory' if ai < ridge else 'compute':>8}"
        )
        del cache
        torch.cuda.empty_cache()

    print()
    if 1 in agg and 32 in agg:
        print(f"  R3.2 falsifier: agg(B=32)/agg(B=1) = {agg[32] / agg[1]:.1f}×  [{'PASS ≥10×' if agg[32] / agg[1] >= 10 else 'BELOW 10× — not amortizing'}]")
    if 1 in agg:
        peak_b = max(agg, key=lambda k: agg[k])
        print(f"  peak aggregate = {agg[peak_b]:.0f} tok/s @ B={peak_b} ({agg[peak_b] / agg[1]:.0f}× the B=1 rate)")


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 R3a — batched decode weight-amortization roofline")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(0)
    sweep(
        args.device,
        compile_mode=None if args.no_compile else "default",
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
