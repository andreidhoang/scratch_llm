"""A1 Rung 4.4 — CUDA-graph decode: launch-overhead elimination (closes the R1 eager→wall gap).

Both arms use the SAME paged Triton kernel decode (`use_kernel=True`), so the delta isolates the
CUDA graph, not paged-vs-dense: eager replays the step by re-issuing every kernel launch from the
host each token; the graph submits one `cudaGraphLaunch`. Correctness first (graph tokens ==
eager tokens, token-exact), then step-time + aggregate tok/s at B ∈ {1, 8, 32}.

Pre-registered (RESULTS.md): B=1 step-time reduction ~20–28% (vLLM-V1), bound = launch overhead;
the eager paged path is overhead-bound at small B, so the graph should recover a real fraction.

Run:  python bench/cudagraph_decode.py --device cuda
"""

from __future__ import annotations

import argparse
import random
import time

import torch
from decode_roofline import RUNG1_CONFIG, count_params  # noqa: E402

from scratch_llm.model import PagedKVCache, PrefillView, TransformerLM
from scratch_llm.serving.cudagraph import CudaGraphDecoder

PROMPT = 32
DECODE = 48


def _prefilled_cache(
    model: TransformerLM, b: int, device: str
) -> tuple[PagedKVCache, torch.Tensor]:
    block = PagedKVCache.BLOCK
    max_blocks = (PROMPT + DECODE + block - 1) // block
    cache = PagedKVCache(
        n_layers=model.cfg.n_layers,
        n_slots=b,
        n_kv_heads=model.cfg.kv_heads,
        max_ctx=model.cfg.context_length,
        head_dim=model.cfg.head_dim,
        n_blocks=b * max_blocks + 1,
        device=device,
        dtype=torch.bfloat16,
    )
    cache.use_kernel = True
    rng = random.Random(1)
    prompts = [[rng.randrange(RUNG1_CONFIG.vocab_size) for _ in range(PROMPT)] for _ in range(b)]
    x = torch.tensor(prompts, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x, PrefillView(cache, list(range(b)), [PROMPT] * b))
    cache.mirror_admit(list(range(b)), [PROMPT] * b)
    first = logits[torch.arange(b, device=device), PROMPT - 1].argmax(dim=-1)
    return cache, first


@torch.no_grad()
def _eager_decode(model: TransformerLM, cache: PagedKVCache, first: torch.Tensor) -> torch.Tensor:
    last, out = first, []
    for _ in range(DECODE):
        cache.pre_decode_reserve()
        logits = model(last.unsqueeze(1), cache)
        last = logits[:, -1].argmax(dim=-1)
        cache.mirror_advance()
        out.append(last)
    return torch.stack(out)


def _time_eager(model: TransformerLM, b: int, device: str, iters: int = 5) -> float:
    """Median ms/step over `iters` runs, each from a FRESH prefilled cache (decode mutates the pool's
    block allocation, so a run cannot be replayed on the same cache without a full reset)."""
    ms: list[float] = []
    for _ in range(iters + 1):  # first run warms (JIT/autotune), discarded
        cache, first = _prefilled_cache(model, b, device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _eager_decode(model, cache, first)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3 / DECODE)
    return sorted(ms[1:])[iters // 2]


def _time_graph(model: TransformerLM, b: int, device: str, iters: int = 5) -> float:
    ms: list[float] = []
    for _ in range(iters + 1):
        cache, first = _prefilled_cache(model, b, device)
        dec = CudaGraphDecoder(model, cache)
        dec.capture()  # untimed setup (warmup + capture)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dec.decode(first, DECODE)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1e3 / DECODE)
    return sorted(ms[1:])[iters // 2]


def run(model: TransformerLM, device: str) -> None:
    print(
        f"\n# R4.4 CUDA-graph decode | paged kernel both arms, prompt={PROMPT}, decode={DECODE} | "
        f"pre-reg: B=1 step-time −20–28% (launch overhead)"
    )
    for b in (1, 8, 32):
        # correctness: graph tokens must equal eager tokens (token-exact)
        ce, fe = _prefilled_cache(model, b, device)
        eager_tok = _eager_decode(model, ce, fe)
        cg, fg = _prefilled_cache(model, b, device)
        dec = CudaGraphDecoder(model, cg)
        dec.capture()
        exact = torch.equal(eager_tok, dec.decode(fg, DECODE))

        eager_ms = _time_eager(model, b, device)
        graph_ms = _time_graph(model, b, device)
        red = 100.0 * (1 - graph_ms / eager_ms)
        print(
            f"  B={b:>2}: token-exact={exact} | eager {eager_ms:5.2f} ms/step | "
            f"graph {graph_ms:5.2f} ms/step | −{red:4.1f}% | "
            f"agg {b * 1e3 / graph_ms:6.0f} tok/s (eager {b * 1e3 / eager_ms:6.0f})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="A1 R4.4 — CUDA-graph decode")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(0)
    model = TransformerLM(RUNG1_CONFIG).to(device=args.device, dtype=torch.bfloat16).eval()
    print(
        f"# A1 R4.4 CUDA-graph decode | GQA-4 {count_params(model) / 1e9:.2f}B bf16 | "
        f"{torch.cuda.get_device_name(0)}"
    )
    run(model, args.device)


if __name__ == "__main__":
    main()
