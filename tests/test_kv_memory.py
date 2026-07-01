"""A1 Rung 2 — GQA/MQA KV-memory correctness rep (the capacity math R2 is about).

Decode reads every weight once + the entire KV cache once per token. GQA/MQA shrink the cache by
``n_heads / n_kv_heads`` because K,V are stored **GQA-native** — shape ``(B, n_kv_heads, T, head_dim)``
per layer (``model.KVCache``) — with the repeat-to-``n_heads`` deferred to attention time. These tests
pin the two facts the Rung-2 bench (``bench/kv_memory.py``) measures and scores against:

  1. KV footprint = ``2 · L · H_kv · d_head · T · dtype`` (linear in ``H_kv`` → the capacity lever), and
  2. the grouping maps query head ``h`` → kv head ``h // G`` (contiguous query blocks share one kv head).

They verify existing-correct behavior (GQA + KVCache already ship), so they pass on first run — the
value is the executable spec + regression guard on the memory formula the roofline reasoning depends on.
Pure CPU: the footprint is a property of the tensors, not the hardware.
"""

import pytest
import torch

from scratch_llm.bench.roofline import decode_step_flops_bytes
from scratch_llm.model import KVCache, ModelConfig, TransformerLM


def _cfg(n_kv_heads: int) -> ModelConfig:
    # n_heads=8 so {8,4,2,1} all divide it: MHA(8) : GQA-2(4) : GQA-4(2) : MQA(1).
    return ModelConfig(
        vocab_size=256, d_model=32, n_layers=3, n_heads=8, n_kv_heads=n_kv_heads, context_length=64
    )


def _cache_bytes(n_kv_heads: int, n_tokens: int = 16) -> int:
    """Total K+V cache bytes after prefilling ``n_tokens`` through a fresh model (B=1)."""
    torch.manual_seed(0)
    cfg = _cfg(n_kv_heads)
    model = TransformerLM(cfg).eval()
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, n_tokens)), cache)
    total = 0
    for i in range(len(model.blocks)):
        got = cache.get(i)
        assert got is not None  # every layer appended during the forward
        k, v = got
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


@pytest.mark.parametrize("n_kv_heads", [8, 4, 2, 1])
def test_kv_cache_bytes_match_formula(n_kv_heads: int) -> None:
    """Measured KVCache footprint == the analytic ``2·L·H_kv·d_head·T·dtype`` (GQA-native storage)."""
    torch.manual_seed(0)
    cfg = _cfg(n_kv_heads)
    model = TransformerLM(cfg).eval()
    n = 16
    cache = KVCache(len(model.blocks))
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, n)), cache)

    measured = 0
    itemsize = 0
    for i in range(len(model.blocks)):
        got = cache.get(i)
        assert got is not None
        k, v = got
        # GQA-native: cache stores n_kv_heads, NOT n_heads — that IS the memory win.
        assert k.shape == (1, n_kv_heads, n, cfg.head_dim)
        assert v.shape == (1, n_kv_heads, n, cfg.head_dim)
        itemsize = k.element_size()
        measured += k.numel() * itemsize + v.numel() * itemsize

    expected = 2 * cfg.n_layers * n_kv_heads * cfg.head_dim * n * itemsize
    assert measured == expected


def test_kv_bytes_scale_linearly_with_kv_heads() -> None:
    """Halving H_kv halves the cache — the GQA/MQA capacity lever (what lets a bigger batch fit in VRAM).

    MHA(8) : GQA-2(4) : GQA-4(2) : MQA(1) footprints must be 8:4:2:1.
    """
    b8, b4, b2, b1 = (_cache_bytes(h) for h in (8, 4, 2, 1))
    assert b8 == 2 * b4 == 4 * b2 == 8 * b1


def test_gqa_grouping_maps_query_head_to_kv_head() -> None:
    """The grouping oracle (DoD): ``repeat_interleave(K, G, dim=heads)`` maps query head h → kv head
    ``h // G`` — contiguous query-head blocks share one kv head, exactly as ``model.py`` expands K,V."""
    torch.manual_seed(0)
    b, n_kv, s, d = 1, 2, 3, 4
    n_heads = 8
    g = n_heads // n_kv  # 4 query heads per kv head
    k = torch.randn(b, n_kv, s, d)
    expanded = k.repeat_interleave(g, dim=1)  # the model's GQA expansion (model.py:286)
    assert expanded.shape == (b, n_heads, s, d)
    for h in range(n_heads):
        assert torch.equal(expanded[:, h], k[:, h // g])


def test_decode_step_kv_traffic_and_crossover() -> None:
    """Guard the roofline op-counter the R2 bench scores against, and the crossover identity: the
    context where KV read == weight read (below it decode is weight-bound; above it KV-bound → where
    GQA/MQA finally moves decode tok/s)."""
    n_params, n_layers, n_kv_heads, head_dim, kv_bytes = 1_000_000, 16, 8, 64, 2
    weight_traffic = n_params * 2
    kv_per_ctx = 2 * n_layers * n_kv_heads * head_dim * kv_bytes

    _, total = decode_step_flops_bytes(
        n_params,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        context_len=2048,
        batch=1,
        weight_bytes=2,
        kv_bytes=kv_bytes,
    )
    assert total == weight_traffic + kv_per_ctx * 2048

    crossover = round(weight_traffic / kv_per_ctx)  # ctx where KV read == weight read
    _, total_at_crossover = decode_step_flops_bytes(
        n_params,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        context_len=crossover,
        batch=1,
        weight_bytes=2,
        kv_bytes=kv_bytes,
    )
    assert total_at_crossover == pytest.approx(2 * weight_traffic, rel=0.01)
