"""A1 Rung 1 — measure decode tok/s and prove decode is memory-bound (three ways).

The from-scratch decoder + KV cache already exist (``sampling.generate`` over ``model.KVCache``); this
script does not rebuild them — it *measures* the one fact that drives all of inference: at batch 1,
decode reads every weight once per token, so arithmetic intensity ≈ 1 FLOP/byte and throughput is set
by HBM bandwidth, not FLOPs.

PREDICT-BEFORE-RUN (your rep, Mode-3): write your predicted decode tok/s in ``bench/RESULTS.md`` FIRST.
The hand ceiling is ``HBM_BW / (2 · n_params)`` for bf16 at batch 1. Then run this and compare — a
surprise is as suspicious as a regression.

What this prints — the three-way memory-bound proof:
  1. hand roofline : AI ≈ 1 FLOP/byte → ~130× below the sm120 ridge → memory-bound
  2. measured      : decode tok/s (marginal, prefill-cancelled) vs the ceiling
  3. Nsight SoL    : the ``ncu`` command to run — you confirm Memory% ≫ Compute% and read the gap

The honest pre-registration: eager batch-1 lands *below* the ceiling (launch overhead + unfused ops);
the bound is still memory. Naming that gap from nsys is the deliverable — and the thread into CUDA
graphs (R4.4) and A2. Fill the ``<FILL: ...>`` cells of the printed row from your profile.

Run:  python bench/decode_roofline.py                 # measure on the standing GPU
      python bench/decode_roofline.py --profile       # one short decode, as the ncu target
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date as _date

import torch

from scratch_llm.bench.gpu_specs import GPUS
from scratch_llm.bench.harness import benchmark
from scratch_llm.bench.roofline import Roofline, decode_step_flops_bytes, roofline
from scratch_llm.model import KVCache, ModelConfig, TransformerLM
from scratch_llm.sampling import SamplingParams, _sample_next

STANDING_GPU = "rtx4000-blackwell"

# A ~1B-parameter bf16 decoder — sized so weight traffic (2·P bytes/token) dominates launch overhead,
# making the HBM-bandwidth ceiling the binding constraint (the whole point of the proof). Fits 25 GB:
# ~1.7 GB weights in bf16 + a small KV cache.
RUNG1_CONFIG = ModelConfig(
    vocab_size=32000, d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8, context_length=2048
)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def decode_n(
    model: TransformerLM,
    prompt_ids: Sequence[int],
    n: int,
    device: str,
    params: SamplingParams | None = None,
) -> list[int]:
    """Decode ``n`` tokens from a fresh KV cache with **no per-token sync** (the timer syncs once).

    Reuses the existing cache decode (``model`` over ``KVCache``, ``_sample_next``); token-identical to
    ``sampling.generate`` for the same params. Returns the generated ids so callers can assert exactness.
    """
    params = params or SamplingParams(temperature=0.0, max_tokens=n)
    model.eval()
    ctx = model.cfg.context_length
    cache = KVCache(len(model.blocks))
    x = torch.tensor([list(prompt_ids)[-ctx:]], dtype=torch.long, device=device)
    logits = model(x, cache)[0, -1]
    out: list[int] = []
    for _ in range(n):
        nid = _sample_next(logits, params)
        out.append(nid)
        if nid in params.stop_ids:
            break
        x = torch.tensor([[nid]], dtype=torch.long, device=device)
        logits = model(x, cache)[0, -1]
    return out


def measure_decode_tok_s(
    model: TransformerLM,
    prompt_ids: Sequence[int],
    device: str,
    *,
    n1: int = 8,
    n2: int = 40,
    warmup: int = 5,
    iters: int = 20,
) -> tuple[float, float]:
    """Marginal decode throughput via the two-length slope: time an ``n1``- and an ``n2``-token decode
    and take ``(t2 − t1) / (n2 − n1)`` as the per-token time. The subtraction cancels the (constant)
    prefill and the one-time launch overhead, leaving the steady-state decode step — the number the
    roofline ceiling is about. Returns ``(tok_s, per_token_s)``.
    """
    t1 = benchmark(
        lambda: decode_n(model, prompt_ids, n1, device), warmup=warmup, iters=iters
    ).median
    t2 = benchmark(
        lambda: decode_n(model, prompt_ids, n2, device), warmup=warmup, iters=iters
    ).median
    per_token_s = (t2 - t1) / (n2 - n1)
    return 1.0 / per_token_s, per_token_s


def predict_decode(
    n_params: int,
    cfg: ModelConfig,
    *,
    context_len: int | None = None,
    batch: int = 1,
    dtype_bytes: int = 2,
) -> Roofline:
    """The hand-roofline ceiling for one bf16 decode step on the standing GPU (via the measured spec)."""
    flops, bytes_ = decode_step_flops_bytes(
        n_params,
        n_layers=cfg.n_layers,
        n_kv_heads=cfg.kv_heads,
        head_dim=cfg.head_dim,
        context_len=context_len or cfg.context_length,
        batch=batch,
        weight_bytes=dtype_bytes,
        kv_bytes=dtype_bytes,
    )
    return roofline(flops, bytes_, GPUS[STANDING_GPU], "bf16")


def results_row(measured_tok_s: float, predicted: Roofline, n_params: int) -> str:
    """The ``bench/RESULTS.md`` row to paste — measured filled, root-cause left for your profile read."""
    ceiling = predicted.predicted_tokens_per_s
    pct = 100.0 * measured_tok_s / ceiling
    return (
        f"| {_date.today().isoformat()} | A1 R1 · decode tok/s ({n_params / 1e9:.2f}B bf16) "
        f"| RTX PRO 4000 Blackwell (sm120) | decode tok/s @B=1 "
        f"| {ceiling:.0f} (mem ceiling) | **{measured_tok_s:.0f}** ({pct:.0f}% of roof) "
        f"| {predicted.bound} | <FILL: name the gap from nsys — launch overhead? unfused eager op?> "
        f"| <FILL: next — CUDA graphs (R4.4)? batch sweep (R3)?> |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 Rung 1 — decode roofline measurement")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument(
        "--profile", action="store_true", help="run one short decode as the ncu target, then exit"
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    model = TransformerLM(RUNG1_CONFIG).to(device=args.device, dtype=torch.bfloat16)
    prompt = list(range(1, args.prompt_len + 1))

    if args.profile:  # bounded target for `ncu` — no bench loop
        decode_n(model, prompt, 20, args.device)
        return

    n_params = count_params(model)
    cfg = RUNG1_CONFIG
    pred = predict_decode(n_params, cfg)
    below = pred.ridge_point / max(pred.arithmetic_intensity, 1e-9)

    print(
        f"# A1 Rung 1 · decode roofline | {cfg.d_model}d × {cfg.n_layers}L × {cfg.n_heads}h/{cfg.kv_heads}kv "
        f"| {n_params / 1e9:.2f}B params bf16 | device={args.device}"
    )
    print()
    print(
        "PREDICT-BEFORE-RUN (your rep): write your predicted decode tok/s in bench/RESULTS.md FIRST."
    )
    print(
        f"  tool roofline: AI={pred.arithmetic_intensity:.3f} FLOP/byte | ridge={pred.ridge_point:.0f} "
        f"| bound={pred.bound} | ceiling≈{pred.predicted_tokens_per_s:.0f} tok/s"
    )
    print()

    tok_s, per_tok_s = measure_decode_tok_s(model, prompt, args.device)
    pct = 100.0 * tok_s / pred.predicted_tokens_per_s
    print("MEASURED — three-way memory-bound proof:")
    print(
        f"  1. hand roofline : AI≈{pred.arithmetic_intensity:.2f} → {below:.0f}× below ridge → {pred.bound}-bound"
    )
    print(
        f"  2. measured      : {tok_s:.0f} tok/s ({per_tok_s * 1e3:.2f} ms/token) "
        f"= {pct:.0f}% of the {pred.predicted_tokens_per_s:.0f} tok/s ceiling"
    )
    print("  3. Nsight SoL    : run this and confirm Memory% ≫ Compute%:")
    print(
        f"       ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --launch-skip 4 "
        f"--launch-count 20 -f -o decode_sol python bench/decode_roofline.py --profile --device {args.device}"
    )
    print()
    print("Paste into bench/RESULTS.md (fill the <FILL: ...> cells from your nsys/ncu read):")
    print(results_row(tok_s, pred, n_params))


if __name__ == "__main__":
    main()
