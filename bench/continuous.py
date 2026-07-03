"""A1 Rung 3b — continuous vs static-wave batching on ragged traces (R3.4/R3.4s/R3.5/R3.6).

Static-wave batching idles slots while the longest row of a wave finishes; iteration-level
(continuous) scheduling refills freed slots every step. Pre-registered (bench/RESULTS.md,
corrected 2026-07-03): speedup ≈ max_len / (mean_len + admit_tax), so

  - heavy-tail trace (24×64 + 6×256 + 2×512 per 32-slot wave): static util ~25%, ceiling 4.0×,
    predict ~2.7×, gate **≥2×** (R3.4)
  - 16×128 + 16×512 sensitivity trace: static util ~62%, ceiling 1.6×, predict ~1.4–1.5× (R3.4s)
  - TTFT p95 wave/continuous **≥4×** (R3.6); ITL cost at B=32 vs B=1 reported (R3.5)

Both policies run the SAME engine (serving/continuous.py) — the delta is pure scheduling. Per
(trace, policy): a short warm pass primes the torch.compile caches (admission widths vary → a few
dynamic-shape recompiles), then the long pass is measured. The measured trace is long enough
(WAVES_MEASURE wave-mixes) that the saturated-queue steady state the analytic model assumes
dominates the fixed-size drain tail (the 2-wave run measured 2026-07-03 was drain-dominated:
continuous util 44%, ratio 1.7×). Continuous is measured before wave so heat-soak biases against
the claim. Per-step metric clocking (sync + host read) costs both arms equally; ratios are the
result.

Run:  python bench/continuous.py                # compiled, both traces + B=1 ITL reference
      python bench/continuous.py --no-compile   # eager control
"""

from __future__ import annotations

import argparse
import random

import torch

from scratch_llm.model import TransformerLM
from scratch_llm.serving.continuous import (
    Request,
    ServeResult,
    serve_continuous,
    serve_static_wave,
)
from scratch_llm.serving.metrics import SLO, ServingReport, summarize

from decode_roofline import RUNG1_CONFIG, count_params  # noqa: E402

PROMPT_LEN = 32
N_SLOTS = 32
# The analytic model (and the R3.4 gate) assumes a SATURATED queue — measured 2026-07-03: at 2
# wave-mixes the post-queue drain tail dominates (continuous util 44% ≪ steady-state) and the
# ratio reads 1.7×. The measured trace therefore uses enough waves that steady state dominates;
# the warm pass (compile-cache priming) uses a short one.
WAVES_MEASURE = 8
WAVES_WARM = 2

# trace kind → (per-wave output-length mix, analytic static util over decode steps, ceiling)
TRACES: dict[str, tuple[list[int], float, float]] = {
    "heavy": ([64] * 24 + [256] * 6 + [512] * 2, 0.249, 4.0),
    "16x16": ([128] * 16 + [512] * 16, 0.624, 1.6),
}

_NO_SLO = SLO(ttft_s=1e9, itl_s=1e9)  # goodput/SLO is not under test in R3b


def make_trace(kind: str, waves: int, seed: int = 0) -> list[Request]:
    mix, _, _ = TRACES[kind]
    rng = random.Random(seed)
    lens: list[int] = []
    for _ in range(waves):
        wave = mix[:]
        rng.shuffle(wave)
        lens.extend(wave)
    return [
        Request(
            request_id=i,
            prompt_ids=tuple(rng.randrange(RUNG1_CONFIG.vocab_size) for _ in range(PROMPT_LEN)),
            max_new_tokens=m,
        )
        for i, m in enumerate(lens)
    ]


def report(name: str, res: ServeResult) -> ServingReport:
    rep = summarize([c.record for c in res.completed], _NO_SLO)
    print(
        f"  {name:<12} agg {rep.throughput_tok_s:>7.0f} tok/s | util {res.mean_utilization * 100:>5.1f}% | "
        f"steps {res.n_decode_steps:>4} | prefills {res.n_prefill_forwards:>3} "
        f"({res.prefill_s:>5.2f}s vs decode {res.decode_s:>6.2f}s) | "
        f"TTFT p50/p95 {rep.ttft_ms.p50:>6.0f}/{rep.ttft_ms.p95:>6.0f} ms | "
        f"ITL p50/p95/p99 {rep.itl_ms.p50:>5.1f}/{rep.itl_ms.p95:>5.1f}/{rep.itl_ms.p99:>6.1f} ms"
    )
    return rep


def _sm_clock_mhz() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "?"


def _measure(
    model: TransformerLM,
    prefill_model: TransformerLM,
    kind: str,
    waves: int,
    device: str,
) -> dict[str, tuple[ServeResult, ServingReport]]:
    """Warm both policies on a short trace, then measure them on a ``waves``-deep one.
    Continuous is measured FIRST: heat-soak across the sequence then penalizes the wave arm —
    biasing AGAINST the ≥2× claim, never for it."""
    warm_reqs = make_trace(kind, WAVES_WARM)
    reqs = make_trace(kind, waves)
    results: dict[str, tuple[ServeResult, ServingReport]] = {}
    for name, fn in (("continuous", serve_continuous), ("static-wave", serve_static_wave)):
        fn(model, warm_reqs, N_SLOTS, device, prefill_model=prefill_model)  # warm (discarded)
        res = fn(model, reqs, N_SLOTS, device, prefill_model=prefill_model)
        results[name] = (res, report(f"{name}", res))
        print(f"               [SM clock after pass: {_sm_clock_mhz()} MHz]")
        torch.cuda.empty_cache()
    return results


def run_trace(
    model: TransformerLM, prefill_model: TransformerLM, kind: str, device: str, waves: int
) -> None:
    _, util_pred, ceiling = TRACES[kind]
    n_wave_tokens = sum(TRACES[kind][0])
    print(
        f"\n# trace={kind}: {len(TRACES[kind][0])} req/wave, {n_wave_tokens} tok/wave, B={N_SLOTS}, "
        f"prompt={PROMPT_LEN} | analytic (steady state): static util {util_pred:.1%}, ceiling {ceiling:.1f}×"
    )

    # Load 1 — SATURATED queue (waves deep): the throughput gate R3.4. TTFT here is queue-wait-
    # dominated for both policies, so it is NOT the R3.6 surface.
    print(f"  -- saturated ({waves} wave-mixes): throughput gate --")
    sat = _measure(model, prefill_model, kind, waves, device)
    wave_res, wave_rep = sat["static-wave"]
    cont_res, cont_rep = sat["continuous"]
    ratio = cont_rep.throughput_tok_s / wave_rep.throughput_tok_s
    step_ratio = wave_res.n_decode_steps / max(cont_res.n_decode_steps, 1)
    gate = (
        f"R3.4 gate ≥2×: {'PASS' if ratio >= 2.0 else 'FAIL'}"
        if kind == "heavy"
        else "R3.4s report-only (pred 1.4–1.5×)"
    )
    print(f"  → continuous/static-wave = {ratio:.2f}× wall ({step_ratio:.2f}× by step count; ceiling {ceiling:.1f}×) [{gate}]")
    print(f"  → measured static util {wave_res.mean_utilization:.1%} vs analytic {util_pred:.1%}; continuous util {cont_res.mean_utilization:.1%}")

    # Load 2 — SHALLOW queue (2 wave-mixes): the latency surface R3.6 (admit-on-slot-free vs
    # wait-for-wave-end shows in TTFT only when queue wait doesn't dominate both policies).
    print(f"  -- shallow ({WAVES_WARM} wave-mixes): TTFT gate --")
    shallow = _measure(model, prefill_model, kind, WAVES_WARM, device)
    ttft_ratio = shallow["static-wave"][1].ttft_ms.p95 / shallow["continuous"][1].ttft_ms.p95
    print(f"  → TTFT p95 wave/continuous = {ttft_ratio:.1f}×  [R3.6 gate ≥4×: {'PASS' if ttft_ratio >= 4.0 else 'FAIL'}]")


def itl_reference_b1(model: TransformerLM, prefill_model: TransformerLM, device: str) -> None:
    """R3.5 reference point: single-slot ITL through the same engine (the B=1 latency floor)."""
    reqs = [
        Request(request_id=i, prompt_ids=tuple(range(1, PROMPT_LEN + 1)), max_new_tokens=64)
        for i in range(2)
    ]
    serve_continuous(model, reqs, 1, device, prefill_model=prefill_model)  # warm
    res = serve_continuous(model, reqs, 1, device, prefill_model=prefill_model)
    rep = summarize([c.record for c in res.completed], _NO_SLO)
    print(f"\n# R3.5 reference — B=1 (same engine): ITL p50 {rep.itl_ms.p50:.1f} ms "
          f"({1000.0 / rep.itl_ms.p50:.0f} tok/s single-stream)")


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 R3b — continuous vs static-wave batching")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--trace", choices=[*TRACES, "all"], default="all")
    ap.add_argument("--waves", type=int, default=WAVES_MEASURE)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = TransformerLM(RUNG1_CONFIG).to(device=args.device, dtype=torch.bfloat16).eval()
    n_params = count_params(model)
    run = model
    prefill_model = model  # ALWAYS eager: admission widths (n_admit = 1, 2, 3, …) are unbounded
    # shape churn — compiling them measured 47 graphs + ~35 s in-run compile + a per-call guard
    # scan taxing every step (2026-07-03). Prefill is ~1% of wall; decode owns the compile budget.
    tag = "eager"
    if not args.no_compile:
        run = torch.compile(model, dynamic=True, mode="default")  # type: ignore[assignment]
        tag = "compiled[default] decode + eager prefill"
    print(f"# A1 R3b continuous batching ({tag}) | GQA-4 {n_params / 1e9:.2f}B bf16 | {torch.cuda.get_device_name(0)}")

    itl_reference_b1(run, prefill_model, args.device)
    for kind in TRACES if args.trace == "all" else [args.trace]:
        run_trace(run, prefill_model, kind, args.device, args.waves)

    if not args.no_compile:
        try:
            from torch._dynamo.utils import counters

            print(f"\n# dynamo: unique_graphs={counters['stats'].get('unique_graphs', '?')} "
                  f"(a recompile storm here would invalidate the compiled comparison)")
        except Exception:  # counter layout is torch-version-dependent; best-effort telemetry
            pass


if __name__ == "__main__":
    main()
