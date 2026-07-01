"""A1 Rung-0 baseline — instrument the existing KV-cache decoder for per-token timing.

Rung 0's GPU half. The from-scratch decoder already exists (``sampling.generate`` over
``model.KVCache``) and is token-exact; this module **instruments** it — it does not reimplement it.
Each request is decoded **sequentially** (static, one at a time — the honest "naive" baseline the
book shows is no faster than batched PyTorch; continuous batching is Rung 3), timing every token on
one monotonic clock so :mod:`~scratch_llm.serving.metrics` can compute TTFT/ITL/throughput/goodput.

Timing correctness (CUDA is async): synchronize the device before reading each timestamp; lock clocks
at the session level (``nvidia-smi -lgc``) and discard warmup before trusting any number — see
``performance/00_foundations.md`` §4.

**Mode-3 boundary.** The signatures + the timing contract are given here; the decode-and-time loop is
the Navigator's to implement. ``tests/test_serving_baseline.py`` pins token-exactness (vs
``sampling.generate``) and fixed-seed reproducibility.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from scratch_llm.model import KVCache, TransformerLM
from scratch_llm.sampling import SamplingParams, _sample_next
from scratch_llm.serving.metrics import RequestRecord


@dataclass(frozen=True)
class DecodeResult:
    """One instrumented decode: the generated tokens plus their timing trace.

    ``token_ids`` MUST equal ``sampling.generate(model, prompt, params, device)`` for the same
    inputs — the instrumentation may not change what is generated.
    """

    token_ids: list[int]
    record: RequestRecord


@torch.no_grad()
def decode_record(
    model: TransformerLM,
    prompt_ids: Sequence[int],
    params: SamplingParams,
    device: str = "cuda",
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> DecodeResult:
    """Greedy/sampled decode instrumented for per-token wall-clock timing.

    Contract:
    - Reuse the existing KV-cache decode (``model`` over a ``KVCache``): prefill the prompt once, then
      decode one token at a time appending to the cache. Do not reimplement attention.
    - ``record.start_s`` is read (via ``clock``) immediately before prefill; ``record.token_times_s[i]``
      immediately after output token ``i`` is available. On CUDA, ``torch.cuda.synchronize()`` before
      each ``clock()`` read so the timestamp reflects device completion, not kernel enqueue.
      - ``record.token_times_s`` has one entry per output token; ``token_times_s[0]`` is the first token
      (so ``record.ttft_s`` is the prefill time). ``record.prompt_len == len(prompt_ids)``.
    - ``token_ids`` is token-identical to ``sampling.generate(model, prompt_ids, params, device)``.
    """

    def get_time() -> float:
        if torch.device(device).type == "cuda":  # also matches "cuda:N", not just bare "cuda"
            torch.cuda.synchronize()
        return clock()

    if params.seed is not None:
        torch.manual_seed(params.seed)

    model.eval()
    context_length = model.cfg.context_length
    ids = list(prompt_ids)

    start_s = get_time()

    cache = KVCache(len(model.blocks))
    x = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)
    logits = model(x, cache)[0, -1]

    generated: list[int] = []
    token_times: list[float] = []

    for _ in range(params.max_tokens):
        next_id = _sample_next(logits, params)
        generated.append(next_id)
        token_times.append(get_time())
        if next_id in params.stop_ids:
            break
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits = model(x, cache)[0, -1]

    record = RequestRecord(
        prompt_len=len(prompt_ids),
        start_s=start_s,
        token_times_s=tuple(token_times),
    )
    return DecodeResult(token_ids=generated, record=record)


def run_baseline(
    model: TransformerLM,
    prompts: Sequence[Sequence[int]],
    params: SamplingParams,
    device: str = "cuda",
) -> list[DecodeResult]:
    """Decode ``prompts`` **sequentially** (static, one at a time — Rung 0), returning a
    :class:`DecodeResult` per prompt in order. Feed ``[r.record for r in run_baseline(...)]`` to
    :func:`scratch_llm.serving.metrics.summarize` for the TTFT/ITL/throughput/goodput report.
    """
    return [decode_record(model, prompt, params, device) for prompt in prompts]
