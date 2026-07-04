"""A1 Rung 4.2 — chunked-prefill scheduler oracle (D1/D5/D6).

Chunked prefill slices a long prompt into ``⌈L/C⌉`` chunks interleaved with decode steps so a big
prefill no longer head-of-line-blocks the decode stream (the measured R3b/R4.1 ITL p99 admission
spike). Correctness is exact by construction — RoPE rotates each token at its absolute position, so
the chunked KV is *algebraically* identical to a one-shot prefill. The tests pin that at three
strengths, honestly separating algebra from float rounding:

1. **Mechanism (float32, exact):** a single chunk (C ≥ L) writes KV **bit-identical** to the
   one-shot ``PrefillView`` (``torch.equal``), and decodes token-exactly to the one-shot continuous
   engine — the offset/RoPE/storage logic is exact.
2. **Algebra (float64, exact):** for **every** chunk size, chunked output is token-exact to
   single-stream greedy — the algorithm is correct independent of float precision.
3. **float32 multi-chunk:** chunked agrees with one-shot continuous on the vast majority of tokens;
   the residual divergences are greedy-argmax **tie flips** from float32 matmul reduction-order
   non-associativity across sequence-length shapes (a 7-token chunk vs a 40-token prompt give
   ~5e-7 different KV) — the SAME batched-vs-unbatched fp nondeterminism the R3b path already has
   vs ``generate``, not a chunked-prefill artifact (float64 removes it entirely). We assert high
   agreement + identical completion set, not brittle bit-exactness.

CPU + fixed seed; asserted on tokens/KV, never wall time.
"""

import random

import pytest
import torch

from scratch_llm.model import (
    BatchedKVCache,
    ChunkPrefillView,
    ModelConfig,
    PrefillView,
    TransformerLM,
)
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.serving.continuous import Request, serve, serve_continuous


def _model(seed: int = 0, dtype: torch.dtype = torch.float32) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=128
    )
    return TransformerLM(cfg).eval().to(dtype)


# Ragged prompts spanning block/chunk boundaries + ragged budgets + more requests than slots.
_PLENS = [1, 5, 15, 16, 17, 31, 33, 48, 64]
_MAX_PLEN = max(_PLENS)


def _mixed(rng_seed: int = 1) -> list[Request]:
    rng = random.Random(rng_seed)
    return [
        Request(i, tuple(rng.randrange(256) for _ in range(pl)), max_new_tokens=8 + (i % 5))
        for i, pl in enumerate(_PLENS)
    ]


# ---------------------------------------------------------------- 1. mechanism (float32, exact)
def test_single_chunk_writes_bit_identical_kv() -> None:
    """A chunk covering the whole prompt (C ≥ L) writes KV bit-identical to the one-shot
    ``PrefillView`` — the offset/RoPE/storage logic is exact (``torch.equal``, no tolerance)."""
    model = _model()
    prompt = tuple(range(40))

    def _cache() -> BatchedKVCache:
        return BatchedKVCache(n_layers=2, n_slots=2, n_kv_heads=2, max_ctx=128, head_dim=8)

    with torch.no_grad():
        ref = _cache()
        model(torch.tensor([prompt]), PrefillView(ref, [0], [len(prompt)]))
        got = _cache()
        model(torch.tensor([prompt]), ChunkPrefillView(got, 0, 0))
    for layer in range(2):
        assert torch.equal(ref._k[layer][0, :, :40], got._k[layer][0, :, :40])
        assert torch.equal(ref._v[layer][0, :, :40], got._v[layer][0, :, :40])


def test_single_chunk_token_exact_to_oneshot_serial() -> None:
    """n_slots=1, C ≥ every prompt: one request at a time, one chunk each ⇒ bit-identical prefill
    (single row in BOTH paths) and identical decode ⇒ token-exact to the one-shot engine, in
    float32 — the fp-safe exactness gate. (For n_slots>1 the one-shot engine BATCHES concurrent
    admits into one padded prefill while chunked prefills per row; that batched-vs-single-row
    matmul is fp-incomparable at the token level — see the float64 test and the high-agreement test
    for the multi-slot correctness statement.)"""
    model = _model()
    reqs = _mixed()
    base = {
        c.request.request_id: list(c.token_ids) for c in serve_continuous(model, reqs, 1).completed
    }
    res = serve_continuous(model, reqs, 1, prefill_chunk_size=_MAX_PLEN)
    got = {c.request.request_id: list(c.token_ids) for c in res.completed}
    assert got == base


# ---------------------------------------------------------------- 2. algebra (float64, exact)
@pytest.mark.parametrize("chunk", [1, 2, 4, 7, 8, 16, 32, 128])
@pytest.mark.parametrize("n_slots", [1, 2, 4, 8])
def test_chunked_token_exact_float64(chunk: int, n_slots: int) -> None:
    """The load-bearing correctness gate: in float64 (no reduction-order noise) chunked prefill is
    token-exact to single-stream greedy for EVERY chunk size and slot count (kill line: any
    divergence here is a real offset/mask bug)."""
    model = _model(dtype=torch.float64)
    reqs = _mixed()
    oracle = {
        r.request_id: generate(
            model, list(r.prompt_ids), SamplingParams(temperature=0.0, max_tokens=r.max_new_tokens)
        )
        for r in reqs
    }
    res = serve_continuous(model, reqs, n_slots, prefill_chunk_size=chunk)
    assert len(res.completed) == len(reqs)
    for c in res.completed:
        assert list(c.token_ids) == oracle[c.request.request_id], (
            f"req {c.request.request_id} (plen={len(c.request.prompt_ids)}, chunk={chunk})"
        )


@pytest.mark.parametrize("chunk", [1, 8, 16])
def test_adversarial_lengths_float64(chunk: int) -> None:
    """A1 §5 adversarial cases (float64, exact): prompt < C (one chunk), prompt == C exactly,
    prompt = C+1 (two chunks), and a not-a-multiple-of-C ragged last chunk."""
    model = _model(dtype=torch.float64)
    reqs = [
        Request(0, tuple(range(chunk - 1 if chunk > 1 else 1)), 6),
        Request(1, tuple(range(chunk)), 6),
        Request(2, tuple(range(chunk + 1)), 6),
        Request(3, tuple(range(3 * chunk + 1)), 6),  # ragged last chunk
    ]
    oracle = {
        r.request_id: generate(
            model, list(r.prompt_ids), SamplingParams(temperature=0.0, max_tokens=r.max_new_tokens)
        )
        for r in reqs
    }
    res = serve_continuous(model, reqs, n_slots=2, prefill_chunk_size=chunk)
    for c in res.completed:
        assert list(c.token_ids) == oracle[c.request.request_id]


# ---------------------------------------------------------------- 3. float32 multi-chunk (fp-tie)
@pytest.mark.parametrize("chunk", [1, 4, 8, 16])
def test_multichunk_float32_high_agreement_same_completions(chunk: int) -> None:
    """float32 multi-chunk: same completion set + total tokens, and ≥85% token agreement with the
    one-shot continuous engine. Residual divergences are greedy-argmax tie flips from float32
    reduction-order non-associativity (float64 → 100%, asserted above), not a logic error."""
    model = _model()
    reqs = _mixed()
    base = {
        c.request.request_id: list(c.token_ids) for c in serve_continuous(model, reqs, 4).completed
    }
    res = serve_continuous(model, reqs, 4, prefill_chunk_size=chunk)
    got = {c.request.request_id: list(c.token_ids) for c in res.completed}
    assert set(got) == set(base)
    assert sum(len(v) for v in got.values()) == sum(len(v) for v in base.values())
    tot = agree = 0
    for rid in base:
        for a, b in zip(base[rid], got[rid], strict=True):
            tot += 1
            agree += a == b
    assert agree / tot >= 0.85, f"chunk={chunk}: token agreement {agree}/{tot} too low"


# ---------------------------------------------------------------- 4. scheduling signature / guards
def test_none_is_byte_identical_to_one_shot() -> None:
    """prefill_chunk_size=None takes the unchanged one-shot code path: identical tokens, step
    counts, and utilization (chunking must not perturb the R3b engine)."""
    model = _model()
    reqs = _mixed()
    base = serve_continuous(model, reqs, 4)
    none = serve_continuous(model, reqs, 4, prefill_chunk_size=None)
    assert none.n_decode_steps == base.n_decode_steps
    assert none.n_prefill_forwards == base.n_prefill_forwards
    assert none.utilization == base.utilization
    for a, b in zip(base.completed, none.completed, strict=True):
        assert a.token_ids == b.token_ids and a.admitted_at_step == b.admitted_at_step


def test_chunked_slices_prefill_into_more_forwards() -> None:
    """The scheduling signature: chunking does strictly more prefill forwards (⌈L/C⌉ per request)
    than the one-shot path, for the same total decode work."""
    model = _model()
    reqs = _mixed()
    one_shot = serve_continuous(model, reqs, 4)
    chunked = serve_continuous(model, reqs, 4, prefill_chunk_size=8)
    assert chunked.n_prefill_forwards > one_shot.n_prefill_forwards
    assert sum(len(c.token_ids) for c in chunked.completed) == sum(r.max_new_tokens for r in reqs)


def test_chunked_rejects_wave_paged_and_bad_size() -> None:
    """Chunked prefill is a continuous+dense feature; other combos raise (documented scope)."""
    model = _model()
    reqs = _mixed()
    with pytest.raises(ValueError, match="continuous"):
        serve(model, reqs, 4, policy="wave", prefill_chunk_size=8)
    with pytest.raises(ValueError, match="dense"):
        serve(model, reqs, 4, policy="continuous", cache_kind="paged", prefill_chunk_size=8)
    with pytest.raises(ValueError, match="≥ 1"):
        serve_continuous(model, reqs, 4, prefill_chunk_size=0)
