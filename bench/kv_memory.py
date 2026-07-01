"""A1 Rung 2 — GQA/MQA KV-memory bench: capacity + crossover table, and the tok/s falsifier.

Spec: ``performance/notes/A1_R2_gqa.md``. Two parts:

  1. **Capacity table** (arithmetic, cheap): for MHA / GQA-4 / MQA on the Rung-1 config, print KV
     bytes/token, max context in the 25 GB budget, and the crossover ctx (where KV read == weight
     read). These are footprint facts — the lever that makes R3 batching fit.
  2. **The tok/s falsifier** (measured, compiled): decode-step throughput at ctx ∈ {2K, 8K, 16K} for
     each variant, under ``torch.compile`` (the R1 fused path that actually reaches the memory wall —
     eager is overhead-bound and hides the KV-size difference). Pre-registered prediction R2.5: at
     ctx=16K, MQA is **>20% faster** than MHA, because MHA reads ~2.1 GB of KV/step on top of the
     1.68 GB of weights while MQA reads ~67 MB.

Run:  python bench/kv_memory.py                 # capacity table + compiled decode sweep
      python bench/kv_memory.py --no-compile    # eager (expect MHA≈MQA — overhead hides KV traffic)
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from scratch_llm.bench.gpu_specs import GPUS
from scratch_llm.bench.harness import benchmark
from scratch_llm.bench.roofline import decode_step_flops_bytes
from scratch_llm.model import KVCache, TransformerLM

from decode_roofline import RUNG1_CONFIG, STANDING_GPU, count_params  # noqa: E402

# n_kv_heads on the Rung-1 config (n_heads=32): full MHA, GQA group-4, and MQA.
VARIANTS = {"MHA": 32, "GQA-4": 8, "MQA": 1}
BUDGET_BYTES = 24 * 1024**3  # usable VRAM for KV after weights (25 GB card, headroom)


def kv_bytes_per_token(n_kv_heads: int, *, kv_bytes: int = 2) -> int:
    """KV stored per token (B=1) = 2·L·H_kv·d_head·dtype — the GQA-native footprint."""
    _, total = decode_step_flops_bytes(
        1,  # n_params irrelevant to the KV term
        n_layers=RUNG1_CONFIG.n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=RUNG1_CONFIG.head_dim,
        context_len=1,
        batch=1,
        weight_bytes=0,  # strip the weight term; keep only KV
        kv_bytes=kv_bytes,
    )
    return int(total)


@torch.no_grad()
def seed_cache(cache: KVCache, n_layers: int, ctx: int, n_kv: int, head_dim: int, device: str) -> None:
    """Directly allocate a length-``ctx`` KV cache of random K,V — bypassing the O(ctx²) attention
    prefill that OOMs at 16K (that quadratic blowup is *why* flash-attention exists; not what R2
    measures). Correctness of the cached values is irrelevant to a **throughput** bench: the decode
    step reads the whole cache regardless of contents, so the measured bytes/step are exact. Pokes
    KVCache internals — it has no bulk-seed API, and adding one to the A1 substrate for a bench hack
    isn't worth it."""
    for layer in range(n_layers):
        cache._k[layer] = torch.randn(1, n_kv, ctx, head_dim, device=device, dtype=torch.bfloat16)
        cache._v[layer] = torch.randn(1, n_kv, ctx, head_dim, device=device, dtype=torch.bfloat16)
    cache._length = ctx


def capacity_table() -> None:
    print("## Capacity (arithmetic) — Rung-1 config, bf16, 24 GB KV budget")
    print(f"{'variant':<8} {'H_kv':>5} {'KV/token':>12} {'vs MHA':>7} {'max ctx (B=1)':>14} {'crossover ctx':>14}")
    mha_kv = kv_bytes_per_token(VARIANTS["MHA"])
    # weight bytes from an actual build (param count depends slightly on H_kv via k/v proj sizes).
    for name, n_kv in VARIANTS.items():
        model = TransformerLM(replace(RUNG1_CONFIG, n_kv_heads=n_kv))
        weight_bytes = count_params(model) * 2
        del model
        kv_tok = kv_bytes_per_token(n_kv)
        max_ctx = BUDGET_BYTES // kv_tok
        crossover = weight_bytes / kv_tok
        print(
            f"{name:<8} {n_kv:>5} {kv_tok / 1024:>10.0f} KB {mha_kv / kv_tok:>6.0f}× "
            f"{max_ctx:>14,} {crossover:>14,.0f}"
        )
    print()


@torch.no_grad()
def decode_sweep(device: str, contexts: list[int], *, compile_mode: str | None, warmup: int, iters: int) -> None:
    spec = GPUS[STANDING_GPU]
    peak_bw = spec.hbm_bandwidth
    tag = f"compiled[{compile_mode}]" if compile_mode else "eager"
    print(f"## Decode-step throughput ({tag}) — the R2.5 falsifier (MQA >20% faster than MHA @ 16K?)")
    print(f"{'variant':<8} {'ctx':>7} {'tok/s':>8} {'ms/step':>9} {'bytes/step':>11} {'GB/s':>8} {'%HBM':>6}")

    # The RoPE table must cover the longest prefill ctx PLUS the decode steps that grow the cache
    # (the Rung-1 config's context_length=2048 would send cos[pos] out of bounds at ctx≥2048).
    max_pos = max(contexts) + 256

    results: dict[tuple[str, int], float] = {}
    for name, n_kv in VARIANTS.items():
        cfg = replace(RUNG1_CONFIG, n_kv_heads=n_kv, context_length=max_pos)
        model = TransformerLM(cfg).to(device=device, dtype=torch.bfloat16)
        model.eval()
        n_params = count_params(model)
        run = model
        if compile_mode:
            try:
                run = torch.compile(model, dynamic=True, mode=compile_mode)  # type: ignore[assignment]
            except Exception as e:  # noqa: BLE001
                print(f"  compile({compile_mode}) failed for {name}: {type(e).__name__}; falling back to eager")
                run = model

        for ctx in contexts:
            cache = KVCache(len(model.blocks))
            seed_cache(cache, len(model.blocks), ctx, n_kv, RUNG1_CONFIG.head_dim, device)
            tok = torch.randint(0, RUNG1_CONFIG.vocab_size, (1, 1), device=device)

            def step() -> None:
                run(tok, cache)  # grows cache by 1/call — negligible vs ctx  # noqa: B023

            per_step = benchmark(step, warmup=warmup, iters=iters).median
            _, bytes_step = decode_step_flops_bytes(
                n_params,
                n_layers=RUNG1_CONFIG.n_layers,
                n_kv_heads=n_kv,
                head_dim=RUNG1_CONFIG.head_dim,
                context_len=ctx,
                batch=1,
                weight_bytes=2,
                kv_bytes=2,
            )
            tok_s = 1.0 / per_step
            bw = bytes_step / per_step
            results[(name, ctx)] = tok_s
            print(
                f"{name:<8} {ctx:>7} {tok_s:>8.0f} {per_step * 1e3:>8.2f} "
                f"{bytes_step / 1e9:>9.2f} GB {bw / 1e9:>8.0f} {100 * bw / peak_bw:>5.0f}%"
            )
            del cache
        del model, run
        torch.cuda.empty_cache()
        if compile_mode:
            torch._dynamo.reset()  # type: ignore[attr-defined]

    # The falsifier verdict.
    print()
    for ctx in contexts:
        mha, mqa = results.get(("MHA", ctx)), results.get(("MQA", ctx))
        if mha and mqa:
            speedup = mqa / mha
            verdict = "PASS (>1.2×)" if speedup > 1.2 else "below 1.2× — weight-bound still"
            print(f"  ctx={ctx:>6}: MQA/MHA decode speedup = {speedup:.2f}×  [{verdict}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 Rung 2 — GQA/MQA KV-memory bench")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 8192, 16384])
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"# A1 R2 GQA/MQA KV-memory | Rung-1 config | device={args.device}\n")
    capacity_table()
    decode_sweep(
        args.device,
        args.contexts,
        compile_mode=None if args.no_compile else "default",
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
