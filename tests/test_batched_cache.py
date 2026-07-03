"""A1 Rung 3b — BatchedKVCache / PrefillView / per-row attention oracle (D1–D3).

The load-bearing correctness surface of continuous batching is the per-row length mask + per-row
RoPE positions: a ragged batched decode must be *only* a scheduling change — row ``b`` is
token-identical to a single-stream greedy decode of prompt ``b``, and a row's logits are invariant
to whatever garbage sits in other slots (batch independence through the slot buffer). CPU + fixed
seed; these gates precede any throughput number (R3.3 before R3.4).
"""

import pytest
import torch

from scratch_llm.model import (
    BatchedKVCache,
    ModelConfig,
    PrefillView,
    RotaryPositionalEmbedding,
    TransformerLM,
)
from scratch_llm.sampling import SamplingParams, generate


def _model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=64
    )
    return TransformerLM(cfg).eval()


def _cache_for(model: TransformerLM, n_slots: int) -> BatchedKVCache:
    cfg = model.cfg
    return BatchedKVCache(
        n_layers=cfg.n_layers,
        n_slots=n_slots,
        n_kv_heads=cfg.kv_heads,
        max_ctx=cfg.context_length,
        head_dim=cfg.head_dim,
    )


@torch.no_grad()
def _prefill(
    model: TransformerLM,
    cache: BatchedKVCache,
    prompts: list[list[int]],
    slots: list[int],
) -> torch.Tensor:
    """Right-padded batched prefill into ``slots``; returns each row's greedy first token (n,)."""
    lens = [len(p) for p in prompts]
    x = torch.zeros((len(prompts), max(lens)), dtype=torch.long)
    for i, p in enumerate(prompts):
        x[i, : lens[i]] = torch.tensor(p, dtype=torch.long)
    view = PrefillView(cache, slots, lens)
    logits = model(x, view)
    cache.mirror_admit(slots, lens)  # scheduler-owned python half of the admission
    last = logits[torch.arange(len(prompts)), torch.tensor(lens) - 1]  # (n, V) last VALID position
    return last.argmax(dim=-1)


# ---------------------------------------------------------------------------- cache mechanics


def test_write_decode_lands_at_per_row_offsets() -> None:
    """Each row's new K/V lands at that row's own length; other positions are untouched."""
    cache = BatchedKVCache(n_layers=1, n_slots=2, n_kv_heads=1, max_ctx=8, head_dim=4)
    # slot 0 at length 3, slot 1 at length 0 (fresh)
    cache.lengths[0] = 3
    cache.py_lengths[0] = 3
    cache.py_active = [True, True]
    cache.active[:] = True
    cache._recompute_view_len()  # direct pokes bypass the mirror ops that normally maintain it
    k_new = torch.arange(8, dtype=torch.float32).reshape(2, 1, 1, 4)  # row 0: 0..3, row 1: 4..7
    v_new = -k_new
    cache.write_decode(0, k_new, v_new)
    k_view, v_view = cache.decode_view(0)
    assert k_view.shape == (2, 1, cache.view_len, 4)
    assert torch.equal(cache._k[0][0, 0, 3], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(cache._k[0][1, 0, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.equal(cache._v[0][0, 0, 3], -torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert cache._k[0][0, 0, 0].abs().sum() == 0  # row 0 position 0 untouched


def test_view_len_covers_every_written_key() -> None:
    cache = BatchedKVCache(n_layers=1, n_slots=3, n_kv_heads=1, max_ctx=16, head_dim=2)
    assert cache.view_len == 1  # all fresh: only the about-to-be-written offset 0
    cache.mirror_admit([0, 1], [5, 2])  # view_len is maintained by the mirror ops (not in-graph)
    assert cache.view_len == 6  # row 0 writes at 5 → keys [0,6) must be visible


def test_advance_bumps_only_active_rows() -> None:
    """advance() is the graph-owned device half; mirror_advance() the scheduler-owned python
    half — each bumps only active rows, and together they stay in lockstep."""
    cache = BatchedKVCache(n_layers=1, n_slots=3, n_kv_heads=1, max_ctx=8, head_dim=2)
    cache.py_lengths = [4, 2, 0]
    cache.py_active = [True, False, True]
    cache.lengths[:] = torch.tensor([4, 2, 0])
    cache.active[:] = torch.tensor([True, False, True])
    cache.advance(1)
    assert cache.lengths.tolist() == [5, 2, 1]
    assert cache.py_lengths == [4, 2, 0]  # device-only: the python half is the scheduler's call
    cache.mirror_advance()
    assert cache.py_lengths == [5, 2, 1]
    assert cache.lengths.tolist() == cache.py_lengths
    with pytest.raises(ValueError, match="n=1"):
        cache.advance(2)


def test_free_slot_resets_length_and_active() -> None:
    cache = BatchedKVCache(n_layers=1, n_slots=2, n_kv_heads=1, max_ctx=8, head_dim=2)
    cache.py_lengths = [4, 3]
    cache.py_active = [True, True]
    cache.lengths[:] = torch.tensor([4, 3])
    cache.active[:] = True
    cache.free_slot(0)
    assert cache.free_slots() == [0]
    assert cache.py_lengths == [0, 3]
    assert cache.lengths.tolist() == [0, 3]


def test_prefill_view_guards() -> None:
    cache = BatchedKVCache(n_layers=1, n_slots=2, n_kv_heads=1, max_ctx=8, head_dim=2)
    cache.py_active[1] = True
    with pytest.raises(ValueError, match="not fresh"):
        PrefillView(cache, [1], [3])
    with pytest.raises(ValueError, match="distinct"):
        PrefillView(cache, [0, 0], [3, 3])
    with pytest.raises(ValueError, match="max_ctx"):
        PrefillView(cache, [0], [9])


def test_prefill_activates_true_lengths_not_padded_width() -> None:
    """Rows are right-padded to the widest prompt; activation must use TRUE lengths."""
    model = _model()
    cache = _cache_for(model, n_slots=3)
    _prefill(model, cache, [[1, 2, 3], [7, 7, 7, 7, 7]], slots=[0, 2])
    assert cache.py_lengths == [3, 0, 5]
    assert cache.py_active == [True, False, True]
    assert cache.lengths.tolist() == [3, 0, 5]


# ---------------------------------------------------------------------------- RoPE per-row


def test_rope_per_row_positions_match_shared() -> None:
    """rope(x, (B,s) positions) row b == rope(x[b], (s,) positions[b]) — the ragged-decode path."""
    torch.manual_seed(0)
    rope = RotaryPositionalEmbedding(head_dim=8, max_seq_len=32)
    x = torch.randn(2, 3, 4, 8)  # (B, H, s, d)
    positions = torch.tensor([[5, 6, 7, 8], [0, 1, 2, 3]])
    out = rope(x, positions)
    for b in range(2):
        expected = rope(x[b : b + 1], positions[b])
        assert torch.equal(out[b : b + 1], expected), f"row {b} diverges from shared-path RoPE"


# ---------------------------------------------------------------------------- the R3.3 oracle


@torch.no_grad()
def test_ragged_batched_decode_matches_single_stream() -> None:
    """R3.3 (ragged): batched slot decode of different-length prompts == per-prompt greedy."""
    model = _model()
    prompts = [[1, 2, 3], [10, 20, 30, 40, 50], [7, 7]]
    n_steps = 6
    cache = _cache_for(model, n_slots=3)
    first = _prefill(model, cache, prompts, slots=[0, 1, 2])
    outs: list[list[int]] = [[int(first[b])] for b in range(3)]
    x = first.unsqueeze(1)
    for _ in range(n_steps):
        logits = model(x, cache)
        cache.mirror_advance()
        nid = logits[:, -1].argmax(dim=-1)
        for b in range(3):
            outs[b].append(int(nid[b]))
        x = nid.unsqueeze(1)
    assert cache.lengths.tolist() == cache.py_lengths  # mirror stayed in lockstep with device
    for b, p in enumerate(prompts):
        single = generate(
            model, p, SamplingParams(temperature=0.0, max_tokens=n_steps + 1), device="cpu"
        )
        assert outs[b] == single, f"row {b}: {outs[b]} != {single}"


@torch.no_grad()
def test_padding_invariance_under_poisoned_slots() -> None:
    """A row's logits are bit-identical whatever garbage sits in inactive slots (no row leaks)."""
    model = _model()
    prompts = [[1, 2, 3, 4], [9, 8, 7]]
    caches = [_cache_for(model, n_slots=4), _cache_for(model, n_slots=4)]
    for cache in caches:
        _prefill(model, cache, prompts, slots=[0, 2])
    # poison the INACTIVE slots (1, 3) of the second cache with finite garbage
    torch.manual_seed(123)
    for layer in range(model.cfg.n_layers):
        for slot in (1, 3):
            caches[1]._k[layer][slot].normal_()
            caches[1]._v[layer][slot].normal_()
    x = torch.tensor([[5], [0], [6], [0]], dtype=torch.long)
    for _ in range(3):
        logits_clean = model(x, caches[0])
        caches[0].mirror_advance()
        logits_poisoned = model(x, caches[1])
        caches[1].mirror_advance()
        assert isinstance(logits_clean, torch.Tensor)
        assert isinstance(logits_poisoned, torch.Tensor)
        for active_row in (0, 2):
            assert torch.equal(logits_clean[active_row], logits_poisoned[active_row]), (
                f"row {active_row} leaked from a poisoned inactive slot"
            )
        nid = logits_clean[:, -1].argmax(dim=-1)
        x = nid.unsqueeze(1)


@torch.no_grad()
def test_uniform_lengths_match_r3a_static_batch() -> None:
    """Degenerate ragged case (equal lengths) == the R3a equal-length batched decode."""
    from scratch_llm.serving.batched import batched_greedy_decode

    model = _model()
    prompts = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]
    n = 5
    r3a = batched_greedy_decode(model, prompts, n, device="cpu")
    cache = _cache_for(model, n_slots=2)
    first = _prefill(model, cache, prompts, slots=[0, 1])
    outs: list[list[int]] = [[int(first[b])] for b in range(2)]
    x = first.unsqueeze(1)
    for _ in range(n - 1):
        logits = model(x, cache)
        cache.mirror_advance()
        nid = logits[:, -1].argmax(dim=-1)
        for b in range(2):
            outs[b].append(int(nid[b]))
        x = nid.unsqueeze(1)
    assert outs == r3a


def test_batched_cache_validation() -> None:
    with pytest.raises(ValueError, match="≥ 1"):
        BatchedKVCache(n_layers=0, n_slots=1, n_kv_heads=1, max_ctx=4, head_dim=2)
    cache = BatchedKVCache(n_layers=1, n_slots=2, n_kv_heads=2, max_ctx=4, head_dim=2)
    with pytest.raises(ValueError, match="expected"):
        cache.write_decode(0, torch.zeros(3, 2, 1, 2), torch.zeros(3, 2, 1, 2))
