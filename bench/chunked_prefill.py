"""A1 Rung 4.2 — chunked prefill: the TTFT/ITL-vs-chunk-size curve (P4.2.2–P4.2.5).

R3b/R4.1 admit a request with ONE prefill forward over its whole prompt, inserted into the decode
stream — a long prompt (here 512 tokens) is ~512× a decode step, so the decode step sharing that
iteration stalls: a few in-flight requests see a huge inter-token gap (the measured ITL p99 ~29 ms
admission spike vs ~6 ms p50). Chunked prefill (Sarathi-Serve / vLLM-V1) caps the prefill work done
between two decode steps at C tokens, so the spike flattens toward p50 — at the cost of a higher
p50 (every gap now carries a chunk) and higher TTFT (⌈L/C⌉ chunk-gaps to first token).

The trace interleaves short-prompt/moderate-decode requests (the decode stream to protect) with
periodic 512-token-prompt requests (the spike source), so long prefills land while decodes are in
flight — the condition chunking targets. We sweep C ∈ {∞, 512, 256, 128, 64, 32} and report the
TTFT/ITL curve + goodput under an ITL SLO with vs without chunking.

Run:  python bench/chunked_prefill.py                 # compiled decode + eager prefill
      python bench/chunked_prefill.py --no-compile    # eager control
"""

from __future__ import annotations

import argparse
import random

import torch
from decode_roofline import RUNG1_CONFIG, count_params  # noqa: E402

from scratch_llm.model import TransformerLM
from scratch_llm.serving.continuous import Request, ServeResult, serve_continuous
from scratch_llm.serving.metrics import SLO, ServingReport, summarize

N_SLOTS = 32
SHORT_PROMPT = 32
LONG_PROMPT = 512  # the admission-spike source: ~512× a decode step in one forward at C=∞
SHORT_DECODE = 96
LONG_DECODE = 48
LONG_EVERY = 6  # one long-prompt request per 6 short ones (interspersed, admitted mid-decode)
N_REQUESTS = 160
CHUNK_SIZES: list[int | None] = [None, 512, 256, 128, 64, 32]
_NO_SLO = SLO(ttft_s=1e9, itl_s=1e9)


def make_r42_trace(seed: int = 0) -> list[Request]:
    """Interleave short-prompt decoders with periodic long-prompt requests (the spike source)."""
    rng = random.Random(seed)
    reqs: list[Request] = []
    for i in range(N_REQUESTS):
        if i % LONG_EVERY == LONG_EVERY - 1:
            plen, budget = LONG_PROMPT, LONG_DECODE
        else:
            plen, budget = SHORT_PROMPT, SHORT_DECODE
        reqs.append(
            Request(
                request_id=i,
                prompt_ids=tuple(rng.randrange(RUNG1_CONFIG.vocab_size) for _ in range(plen)),
                max_new_tokens=budget,
            )
        )
    return reqs


def _report(chunk: int | None, res: ServeResult, rep: ServingReport) -> None:
    label = "∞ (one-shot)" if chunk is None else f"C={chunk}"
    step_ms = 1e3 * res.decode_s / max(res.n_decode_steps, 1)
    print(
        f"  {label:<13} | agg {rep.throughput_tok_s:>6.0f} tok/s | steps {res.n_decode_steps:>4} | "
        f"prefills {res.n_prefill_forwards:>4} ({res.prefill_s:>5.2f}s) | {step_ms:>5.2f} ms/step | "
        f"TTFT p50/p95 {rep.ttft_ms.p50:>6.0f}/{rep.ttft_ms.p95:>6.0f} ms | "
        f"ITL p50/p95/p99 {rep.itl_ms.p50:>5.1f}/{rep.itl_ms.p95:>5.1f}/{rep.itl_ms.p99:>6.1f} ms"
    )


def run_r42(model: TransformerLM, prefill_model: TransformerLM, device: str) -> None:
    warm = make_r42_trace(seed=1)[:48]
    reqs = make_r42_trace(seed=0)
    n_long = sum(1 for r in reqs if len(r.prompt_ids) == LONG_PROMPT)
    print(
        f"\n# R4.2 chunked prefill | {N_REQUESTS} reqs ({n_long} × {LONG_PROMPT}-tok prompts "
        f"interspersed), B={N_SLOTS} | pre-reg: ITL p99 falls ≥2× as C↓, p50 rises modestly, "
        f"TTFT rises, goodput@SLO ≥1.3× vs one-shot"
    )
    reports: dict[int | None, tuple[ServeResult, ServingReport]] = {}
    for chunk in CHUNK_SIZES:
        serve_continuous(
            model, warm, N_SLOTS, device, prefill_model=prefill_model, prefill_chunk_size=chunk
        )
        res = serve_continuous(
            model, reqs, N_SLOTS, device, prefill_model=prefill_model, prefill_chunk_size=chunk
        )
        rep = summarize([c.record for c in res.completed], _NO_SLO)
        reports[chunk] = (res, rep)
        _report(chunk, res, rep)
        torch.cuda.empty_cache()

    base_res, base_rep = reports[None]
    # P4.2.2/3/4: the curve, relative to the one-shot baseline.
    print("\n  # curve vs one-shot (∞):")
    for chunk in CHUNK_SIZES[1:]:
        _, rep = reports[chunk]
        print(
            f"    C={chunk:<4}: ITL p99 ×{rep.itl_ms.p99 / base_rep.itl_ms.p99:.2f} "
            f"({base_rep.itl_ms.p99:.1f}→{rep.itl_ms.p99:.1f} ms) | "
            f"ITL p50 ×{rep.itl_ms.p50 / base_rep.itl_ms.p50:.2f} | "
            f"TTFT p50 ×{rep.ttft_ms.p50 / max(base_rep.ttft_ms.p50, 1e-9):.2f}"
        )

    # P4.2.5: goodput under an ITL-p99 SLO. SLO = 2× the one-shot p50 (protect the decode stream;
    # the one-shot p99 spike blows past it, chunking should keep more tokens under it).
    slo_itl_ms = 2.0 * base_rep.itl_ms.p50
    slo = SLO(ttft_s=1e9, itl_s=slo_itl_ms / 1e3)
    print(f"\n  # goodput @ ITL SLO (per-token ≤ {slo_itl_ms:.1f} ms = 2× one-shot p50):")
    base_gp = summarize([c.record for c in base_res.completed], slo).goodput_tok_s
    for chunk in CHUNK_SIZES:
        res, _ = reports[chunk]
        gp = summarize([c.record for c in res.completed], slo).goodput_tok_s
        label = "∞ (one-shot)" if chunk is None else f"C={chunk}"
        ratio = f" ×{gp / base_gp:.2f} vs one-shot" if base_gp > 0 and chunk is not None else ""
        print(f"    {label:<13}: goodput {gp:>6.0f} tok/s{ratio}")


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 R4.2 — chunked prefill TTFT/ITL curve")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    model = TransformerLM(RUNG1_CONFIG).to(device=args.device, dtype=torch.bfloat16).eval()
    n_params = count_params(model)
    run = model
    prefill_model = model  # eager prefill (admission/chunk widths churn; decode owns the compile)
    tag = "eager"
    if not args.no_compile:
        run = torch.compile(model, dynamic=True, mode="default")  # type: ignore[assignment]
        tag = "compiled[default] decode + eager prefill"
    print(
        f"# A1 R4.2 chunked prefill ({tag}) | GQA-4 {n_params / 1e9:.2f}B bf16 | {torch.cuda.get_device_name(0)}"
    )

    run_r42(run, prefill_model, args.device)

    if not args.no_compile:
        try:
            from torch._dynamo.utils import counters

            print(f"\n# dynamo: unique_graphs={counters['stats'].get('unique_graphs', '?')}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
