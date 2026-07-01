"""A1 Rung 1 (continued) — strip the launch overhead and *prove* decode is memory-bound.

Rung 1's first measurement found the honest negative: eager batch-1 decode reaches only 16% of the
315 tok/s memory ceiling (89 GB/s = 16% of the card's 550 GB/s), because ~955 kernel launches/token +
a `.item()` host-sync/token dominate the step — the *workload* is memory-bound (AI≈1) but the *run* is
overhead-bound and never touches the memory wall. This script closes that thread: it removes the
overhead one variable at a time (D4) and re-measures, testing the pre-registered prediction in
``bench/RESULTS.md`` (achieved BW climbs 89 → toward 550 as the launches collapse).

Three variants, each timed with the two-length slope ``(t(n2) − t(n1)) / (n2 − n1)`` so the constant
prefill + one-time overhead cancel and only the steady-state decode *step* is compared:

  baseline : the Rung-1 path — ``.item()`` per token + a fresh host→device tensor per step.
  nosync   : greedy argmax stays on GPU (no ``.item()``, no per-token host tensor); ids stacked once.
  compiled : the nosync loop with ``torch.compile`` collapsing the pointwise launches (fusion; and,
             if the dynamic KV-cat allows it, cudagraphs via ``mode="reduce-overhead"``).

Every variant is asserted **token-exact** vs ``sampling.generate`` (temp=0) before its number is
trusted (D1 oracle discipline) — an overhead strip may not change what is generated.

Mode boundary: this is plumbing (sync removal + ``torch.compile`` wiring), Mode-1 — no kernel body is
written here. The Mode-3 kernel rep is A2 (GEMV/fusion) and R4.4 (a static-buffer CUDA-graph decoder).

Run:  python bench/decode_overhead_strip.py            # all three strips + ledger rows
      python bench/decode_overhead_strip.py --no-compile
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date as _date

import torch

from scratch_llm.bench.gpu_specs import GPUS
from scratch_llm.bench.harness import benchmark
from scratch_llm.bench.roofline import decode_step_flops_bytes
from scratch_llm.model import KVCache, TransformerLM
from scratch_llm.sampling import SamplingParams, generate

# Reuse the pristine Rung-1 harness config + predictors — same 0.84B model, so the number is comparable.
# Sibling import (run as `python bench/decode_overhead_strip.py`, so bench/ is sys.path[0]), matching
# flash_roofline.py's `from _harness import ...` convention.
from decode_roofline import (  # noqa: E402
    RUNG1_CONFIG,
    STANDING_GPU,
    count_params,
    decode_n,
    predict_decode,
)


@torch.no_grad()
def decode_n_nosync(
    model: TransformerLM, prompt_ids: Sequence[int], n: int, device: str
) -> torch.Tensor:
    """Greedy decode of ``n`` tokens with **zero host syncs in the loop**.

    The only difference from the baseline ``decode_n`` is the strip: the argmax token id stays a 0-dim
    GPU tensor (never ``.item()``-ed) and is fed straight back as the next input, so nothing forces a
    device→host round-trip until the caller stacks the ids once at the end. Token-identical to
    ``sampling.generate`` at temperature 0 (both take the argmax).
    """
    model.eval()
    ctx = model.cfg.context_length
    cache = KVCache(len(model.blocks))
    x = torch.tensor([list(prompt_ids)[-ctx:]], dtype=torch.long, device=device)
    logits = model(x, cache)[0, -1]
    ids: list[torch.Tensor] = []
    for _ in range(n):
        nid = torch.argmax(logits)  # 0-dim long, on GPU — no sync
        ids.append(nid)
        x = nid.view(1, 1)  # (1, 1) long, on GPU — no host→device copy
        logits = model(x, cache)[0, -1]
    return torch.stack(ids)  # (n,) on GPU; caller reads it once


def _assert_token_exact(model: TransformerLM, prompt: Sequence[int], n: int, device: str) -> None:
    """Oracle gate (D1): the nosync strip must generate exactly what the reference sampler does."""
    ref = generate(model, prompt, SamplingParams(temperature=0.0, max_tokens=n), device)
    got = decode_n_nosync(model, prompt, n, device).tolist()
    if got != ref:
        raise AssertionError(f"nosync decode diverged from sampling.generate: {got} != {ref}")


def measure_slope(fn_n1, fn_n2, n1: int, n2: int, *, warmup: int, iters: int) -> float:
    """Per-token seconds via the two-length slope (cancels prefill + one-time launch overhead)."""
    t1 = benchmark(fn_n1, warmup=warmup, iters=iters).median
    t2 = benchmark(fn_n2, warmup=warmup, iters=iters).median
    return (t2 - t1) / (n2 - n1)


def _row(
    label: str, per_tok_s: float, bytes_per_tok: float, ceiling_tok_s: float, peak_bw: float
) -> dict[str, float | str]:
    tok_s = 1.0 / per_tok_s
    bw = bytes_per_tok / per_tok_s
    return {
        "label": label,
        "tok_s": tok_s,
        "ms": per_tok_s * 1e3,
        "pct_ceiling": 100.0 * tok_s / ceiling_tok_s,
        "bw_gbs": bw / 1e9,
        "pct_bw": 100.0 * bw / peak_bw,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 Rung 1 — decode overhead strip (prove the memory wall)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--n1", type=int, default=8)
    ap.add_argument("--n2", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-compile", action="store_true", help="skip the torch.compile strip")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = TransformerLM(RUNG1_CONFIG).to(device=args.device, dtype=torch.bfloat16)
    prompt = list(range(1, args.prompt_len + 1))
    n_params = count_params(model)

    spec = GPUS[STANDING_GPU]
    peak_bw = spec.hbm_bandwidth
    pred = predict_decode(n_params, RUNG1_CONFIG, context_len=args.prompt_len + args.n2)
    ceiling = pred.predicted_tokens_per_s
    _, bytes_per_tok = decode_step_flops_bytes(
        n_params,
        n_layers=RUNG1_CONFIG.n_layers,
        n_kv_heads=RUNG1_CONFIG.kv_heads,
        head_dim=RUNG1_CONFIG.head_dim,
        context_len=args.prompt_len + args.n2,
        batch=1,
        weight_bytes=2,
        kv_bytes=2,
    )

    print(
        f"# A1 R1 overhead-strip | {n_params / 1e9:.2f}B bf16 | ceiling≈{ceiling:.0f} tok/s "
        f"| peak HBM {peak_bw / 1e12:.2f} TB/s | AI={pred.arithmetic_intensity:.2f} → {pred.bound}-bound workload"
    )
    print(f"  bytes/token ≈ {bytes_per_tok / 1e9:.2f} GB (weights dominate; KV traffic negligible at short ctx)")
    print()

    # Oracle gate before any timing (D1).
    _assert_token_exact(model, prompt, args.n2, args.device)
    print("oracle: nosync decode is token-exact vs sampling.generate (temp=0) ✓")
    print()

    rows: list[dict[str, float | str]] = []

    # --- baseline (Rung-1 path): .item()/token + host tensor/token ------------------------------------
    per_tok = measure_slope(
        lambda: decode_n(model, prompt, args.n1, args.device),
        lambda: decode_n(model, prompt, args.n2, args.device),
        args.n1,
        args.n2,
        warmup=args.warmup,
        iters=args.iters,
    )
    rows.append(_row("baseline (.item/token)", per_tok, bytes_per_tok, ceiling, peak_bw))

    # --- strip A: kill the host sync ------------------------------------------------------------------
    per_tok = measure_slope(
        lambda: decode_n_nosync(model, prompt, args.n1, args.device),
        lambda: decode_n_nosync(model, prompt, args.n2, args.device),
        args.n1,
        args.n2,
        warmup=args.warmup,
        iters=args.iters,
    )
    rows.append(_row("nosync (argmax on GPU)", per_tok, bytes_per_tok, ceiling, peak_bw))

    # --- strip B: torch.compile (fusion; cudagraphs if the dynamic KV-cat allows) ---------------------
    if not args.no_compile:
        for mode in ("default", "reduce-overhead"):
            try:
                cmodel = torch.compile(model, dynamic=True, mode=mode)
                per_tok = measure_slope(
                    lambda: decode_n_nosync(cmodel, prompt, args.n1, args.device),  # type: ignore[arg-type]
                    lambda: decode_n_nosync(cmodel, prompt, args.n2, args.device),  # type: ignore[arg-type]
                    args.n1,
                    args.n2,
                    warmup=max(args.warmup, 12),  # absorb compile / recompile / graph capture
                    iters=args.iters,
                )
                rows.append(_row(f"compiled[{mode}]", per_tok, bytes_per_tok, ceiling, peak_bw))
            except Exception as e:  # noqa: BLE001 — a compile failure IS a result (scopes R4.4)
                print(f"  torch.compile(mode={mode!r}) failed: {type(e).__name__}: {str(e)[:200]}")
                torch._dynamo.reset()  # type: ignore[attr-defined]

    # --- report ---------------------------------------------------------------------------------------
    print(f"{'variant':<26} {'tok/s':>8} {'ms/tok':>8} {'%ceil':>7} {'GB/s':>8} {'%HBM':>7}")
    for r in rows:
        print(
            f"{r['label']:<26} {r['tok_s']:>8.0f} {r['ms']:>8.2f} {r['pct_ceiling']:>6.0f}% "
            f"{r['bw_gbs']:>8.0f} {r['pct_bw']:>6.0f}%"
        )
    print()

    print("Paste measured rows into bench/RESULTS.md (fill root-cause from nsys/ncu if a strip surprises):")
    today = _date.today().isoformat()
    for r in rows[1:]:  # baseline already logged; log the strips
        verdict = "memory" if r["pct_bw"] >= 60 else "overhead"  # ≥60% of HBM ⇒ the wall is visible
        print(
            f"| {today} | A1 R1 · {r['label']} (0.84B bf16) | RTX PRO 4000 Blackwell (sm120) "
            f"| decode tok/s @B=1 | {ceiling:.0f} (mem ceiling) "
            f"| **{r['tok_s']:.0f}** ({r['pct_ceiling']:.0f}% of roof, {r['bw_gbs']:.0f} GB/s = {r['pct_bw']:.0f}% HBM) "
            f"| {verdict} | <FILL: strip effect> | <FILL: next> |"
        )


if __name__ == "__main__":
    main()
