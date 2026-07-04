"""A1 Rung 4.3 — speculative decoding: acceptance rate + end-to-end speedup (P4.3.2–P4.3.5).

Plain greedy decode does one target forward per token (memory-bound: one weight read/token).
Speculative decode does one target forward per ROUND, committing (1 + accepted) tokens — so if the
n-gram drafter's guesses land, fewer forwards produce the same tokens. On the standing GPU's eager
decode (launch/overhead-bound, R1) AND on the memory-bound wall, the per-forward cost is ~constant in
the K+1 verify width (weights/launches dominate), so wall speedup ≈ tokens-per-forward. We measure
both, on a repetitive prompt (n-gram hits) vs a random one (misses → net loss), against the SAME
eager target forward (apples-to-apples: the ratio isolates speculation, not compilation).

Run:  python bench/speculative.py --device cuda
"""

from __future__ import annotations

import argparse
import time

import torch
from decode_roofline import RUNG1_CONFIG, count_params  # noqa: E402

from scratch_llm.model import TransformerLM
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.serving.speculative import NGramDrafter, speculative_generate

MAX_NEW = 128
KS = [2, 4, 8]


def _repetitive_prompt(period: int = 16, length: int = 64) -> list[int]:
    base = [(i * 7 + 3) % RUNG1_CONFIG.vocab_size for i in range(period)]
    return (base * ((length // period) + 1))[:length]


def _random_prompt(length: int = 64, seed: int = 0) -> list[int]:
    import random

    rng = random.Random(seed)
    return [rng.randrange(RUNG1_CONFIG.vocab_size) for _ in range(length)]


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _median_tok_s(fn, iters: int, device: str) -> float:
    """Median per-iteration tok/s — robust to the eager-decode / unlocked-clock timing noise on this
    box (mean over a few iters is dominated by clock-boost transients; median is the honest number)."""
    fn()  # warm (discarded)
    _sync(device)
    rates: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        rates.append(MAX_NEW / (time.perf_counter() - t0))
    rates.sort()
    return rates[len(rates) // 2]


def _time_plain(model: TransformerLM, prompt: list[int], device: str, iters: int = 7) -> float:
    return _median_tok_s(
        lambda: generate(
            model, prompt, SamplingParams(temperature=0.0, max_tokens=MAX_NEW), device
        ),
        iters,
        device,
    )


def _time_spec(
    model: TransformerLM, prompt: list[int], device: str, k: int, iters: int = 7
) -> tuple[float, float, float]:
    drafter = NGramDrafter(n=3)
    holder: dict[str, object] = {}

    def _one() -> None:
        _, holder["stats"] = speculative_generate(model, drafter, prompt, MAX_NEW, device, k=k)

    tok_s = _median_tok_s(_one, iters, device)
    stats = holder["stats"]
    assert stats is not None
    return tok_s, stats.acceptance_rate, stats.mean_tokens_per_forward  # type: ignore[union-attr]


def run(model: TransformerLM, device: str) -> None:
    print(
        f"\n# R4.3 speculative decoding | max_new={MAX_NEW}, drafter=n-gram(n=3), eager target | "
        f"pre-reg: accept 40–75% (repetitive) / ~0 (random); speedup 1.3–2.0× (rep) / <1× (rand)"
    )
    for name, prompt in (("repetitive", _repetitive_prompt()), ("random", _random_prompt())):
        base = _time_plain(model, prompt, device)
        print(f"\n  {name} prompt | plain greedy: {base:6.1f} tok/s (baseline)")
        for k in KS:
            spec_tok_s, accept, tpf = _time_spec(model, prompt, device, k)
            print(
                f"    K={k}: {spec_tok_s:6.1f} tok/s | ×{spec_tok_s / base:.2f} wall | "
                f"accept {accept * 100:5.1f}% | {tpf:.2f} tok/target-forward"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 R4.3 — speculative decoding acceptance/speedup")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(0)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = TransformerLM(RUNG1_CONFIG).to(device=args.device, dtype=dtype).eval()
    dev_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    print(f"# A1 R4.3 spec decode | GQA-4 {count_params(model) / 1e9:.2f}B {dtype} | {dev_name}")
    run(model, args.device)


if __name__ == "__main__":
    main()
