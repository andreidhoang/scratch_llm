"""A1 R4.1 — PagedKVCache oracle: paged storage must be invisible to the math (P4.1.2).

The contiguous-match gate: with blocks DELIBERATELY scattered across the pool, ragged batched
decode through the paged cache is bit-identical to the dense `BatchedKVCache` (same SDPA, storage
is the only variable) and token-identical to single-stream greedy. Plus the vLLM adversarial set:
block-boundary crossings, a request finishing mid-batch (churn/reuse, no leaked blocks), poisoned
free blocks, prefix sharing with refcounts, and the waste accounting the P4.1.1 capacity headline
rests on. CPU + fixed seed; these gates precede any capacity or speed number.
"""

import random

import pytest
import torch

from scratch_llm.model import (
    BatchedKVCache,
    ModelConfig,
    PagedKVCache,
    PrefillView,
    SlotKVCache,
    TransformerLM,
)
from scratch_llm.sampling import SamplingParams, generate
from scratch_llm.serving.continuous import Request, serve_continuous


def _model(seed: int = 0) -> TransformerLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=256, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, context_length=64
    )
    return TransformerLM(cfg).eval()


def _dense(model: TransformerLM, n_slots: int) -> BatchedKVCache:
    cfg = model.cfg
    return BatchedKVCache(cfg.n_layers, n_slots, cfg.kv_heads, cfg.context_length, cfg.head_dim)


def _paged(
    model: TransformerLM, n_slots: int, n_blocks: int = 24, scatter: bool = True
) -> PagedKVCache:
    cfg = model.cfg
    cache = PagedKVCache(
        cfg.n_layers, n_slots, cfg.kv_heads, cfg.context_length, cfg.head_dim, n_blocks=n_blocks
    )
    if scatter:  # deliberately scattered physical blocks (the assignment's adversarial case)
        random.Random(7).shuffle(cache._free)
    return cache


@torch.no_grad()
def _prefill(
    model: TransformerLM, cache: SlotKVCache, prompts: list[list[int]], slots: list[int]
) -> torch.Tensor:
    lens = [len(p) for p in prompts]
    x = torch.zeros((len(prompts), max(lens)), dtype=torch.long)
    for i, p in enumerate(prompts):
        x[i, : lens[i]] = torch.tensor(p, dtype=torch.long)
    view = PrefillView(cache, slots, lens)
    logits = model(x, view)
    cache.mirror_admit(slots, lens)
    last = logits[torch.arange(len(prompts)), torch.tensor(lens) - 1]
    return last.argmax(dim=-1)


@torch.no_grad()
def _decode_steps(
    model: TransformerLM, cache: SlotKVCache, x: torch.Tensor, n_steps: int
) -> list[torch.Tensor]:
    """Run n lockstep decode steps (with the scheduler-side reserve + mirror calls); return the
    per-step logits."""
    outs = []
    for _ in range(n_steps):
        cache.pre_decode_reserve()
        logits = model(x, cache)
        assert isinstance(logits, torch.Tensor)
        cache.mirror_advance()
        outs.append(logits)
        x = logits[:, -1].argmax(dim=-1).unsqueeze(1)
    return outs


# --------------------------------------------------------------------- allocator mechanics


def test_allocator_alloc_free_reuse_and_trash() -> None:
    model = _model()
    cache = _paged(model, n_slots=2, n_blocks=6, scatter=False)
    assert cache.n_free_blocks == 5  # block 0 is trash, never allocatable
    cache.reserve_prefill([0], [20])  # 2 blocks
    assert cache.n_free_blocks == 3
    assert 0 not in cache._py_table[0]
    cache.free_slot(0)
    assert cache.n_free_blocks == 5
    assert cache._py_table[0] == []
    assert cache.block_table[0].tolist() == [0, 0, 0, 0]  # back to trash everywhere


def test_allocator_oom_raises() -> None:
    model = _model()
    cache = _paged(model, n_slots=2, n_blocks=2, scatter=False)  # 1 usable block
    with pytest.raises(RuntimeError, match="exhausted"):
        cache.reserve_prefill([0], [17])  # needs 2 blocks


def test_waste_accounting_matches_hand_analytic() -> None:
    """P4.1.1 instrument: frag == 1 − live/allocated on a constructed scenario."""
    model = _model()
    cache = _paged(model, n_slots=3, n_blocks=16, scatter=False)
    cache.reserve_prefill([0, 1, 2], [5, 16, 20])  # blocks: 1 + 1 + 2 = 4 → 64 allocated tokens
    cache.mirror_admit([0, 1, 2], [5, 16, 20])
    allocated, live, frag = cache.waste_stats()
    assert allocated == 64
    assert live == 41
    assert frag == 1.0 - 41 / 64


# --------------------------------------------------------------------- the contiguous-match oracle


@torch.no_grad()
def test_paged_matches_dense_bit_exact_scattered() -> None:
    """P4.1.2: with scattered physical blocks, paged logits == dense logits, bit for bit."""
    model = _model()
    prompts = [[1, 2, 3], [10, 20, 30, 40, 50], [7, 7]]
    dense = _dense(model, 3)
    paged = _paged(model, 3, scatter=True)
    f_dense = _prefill(model, dense, prompts, [0, 1, 2])
    f_paged = _prefill(model, paged, prompts, [0, 1, 2])
    assert torch.equal(f_dense, f_paged)
    x = f_dense.unsqueeze(1)
    outs_dense = _decode_steps(model, dense, x, 8)
    outs_paged = _decode_steps(model, paged, x, 8)
    for step, (a, b) in enumerate(zip(outs_dense, outs_paged, strict=True)):
        assert torch.equal(a, b), f"step {step}: paged logits diverge from dense"


@torch.no_grad()
def test_block_boundary_crossings_match_single_stream() -> None:
    """Adversarial lengths: prompts of 15/16/17 tokens decode across the 16-token boundary and
    must stay token-exact vs single-stream greedy (off-by-one hotspot)."""
    model = _model()
    prompts = [list(range(1, 16)), list(range(1, 17)), list(range(1, 18))]  # len 15, 16, 17
    paged = _paged(model, 3, n_blocks=24)
    first = _prefill(model, paged, prompts, [0, 1, 2])
    outs: list[list[int]] = [[int(first[b])] for b in range(3)]
    x = first.unsqueeze(1)
    for _ in range(20):  # crosses 16 and 32 for every row
        paged.pre_decode_reserve()
        logits = model(x, paged)
        assert isinstance(logits, torch.Tensor)
        paged.mirror_advance()
        nid = logits[:, -1].argmax(dim=-1)
        for b in range(3):
            outs[b].append(int(nid[b]))
        x = nid.unsqueeze(1)
    for b, p in enumerate(prompts):
        single = generate(model, p, SamplingParams(temperature=0.0, max_tokens=21), device="cpu")
        assert outs[b] == single, f"row {b} diverged crossing a block boundary"


@torch.no_grad()
def test_poisoned_free_blocks_do_not_leak() -> None:
    """Garbage in unallocated blocks (and the trash block) must not change active rows."""
    model = _model()
    prompts = [[1, 2, 3, 4], [9, 8, 7]]
    caches = [_paged(model, 4, scatter=True), _paged(model, 4, scatter=True)]
    for cache in caches:
        _prefill(model, cache, prompts, [0, 2])
    torch.manual_seed(99)
    for layer in range(model.cfg.n_layers):
        used = {blk for table in caches[1]._py_table for blk in table}
        for blk in range(caches[1].n_blocks):
            if blk not in used:  # includes trash block 0
                caches[1]._pool_k[layer][blk].normal_()
                caches[1]._pool_v[layer][blk].normal_()
    x = torch.tensor([[5], [0], [6], [0]], dtype=torch.long)
    outs_clean = _decode_steps(model, caches[0], x, 3)
    outs_poisoned = _decode_steps(model, caches[1], x, 3)
    for a, b in zip(outs_clean, outs_poisoned, strict=True):
        for active_row in (0, 2):
            assert torch.equal(a[active_row], b[active_row]), "poisoned block leaked into a row"


@torch.no_grad()
def test_churn_reuses_blocks_without_leak() -> None:
    """Finish-mid-batch: freed blocks are reused by later admissions; after everything frees,
    the pool is whole again (refcount/leak gate)."""
    model = _model()
    paged = _paged(model, 2, n_blocks=8)
    for round_idx in range(3):  # 3 admission waves through the same 2 slots
        prompts = [[round_idx + 1, 2, 3], [round_idx + 5, 6, 7, 8]]
        first = _prefill(model, paged, prompts, [0, 1])
        _decode_steps(model, paged, first.unsqueeze(1), 15)  # crosses a boundary → extra blocks
        paged.free_slot(0)
        paged.free_slot(1)
    assert paged.n_free_blocks == paged.n_blocks - 1  # nothing leaked
    assert all(rc == 0 for rc in paged._refcount[1:])


# --------------------------------------------------------------------- prefix sharing (CoW scope)


@torch.no_grad()
def test_shared_prefix_matches_private_copy_and_refcounts() -> None:
    """share_prefix: a slot pointing at another's blocks decodes identically to owning a private
    copy; the next write lands in a private block (CoW-by-construction); refcounts balance."""
    model = _model()
    prompt = list(range(1, 33))  # 32 tokens = 2 full blocks (sharing is full-block only)
    shared = _paged(model, 2, n_blocks=16, scatter=True)
    control = _paged(model, 2, n_blocks=16, scatter=True)
    _prefill(model, shared, [prompt], [0])
    shared.share_prefix(0, 1, 32)
    assert shared._refcount[shared._py_table[0][0]] == 2
    _prefill(model, control, [prompt], [0])  # control: two PRIVATE copies of the same prompt
    control_view = PrefillView(control, [1], [32])
    x_p = torch.tensor([prompt], dtype=torch.long)
    model(x_p, control_view)
    control.mirror_admit([1], [32])
    x = torch.tensor([[42], [42]], dtype=torch.long)
    outs_shared = _decode_steps(model, shared, x, 6)
    outs_control = _decode_steps(model, control, x, 6)
    for a, b in zip(outs_shared, outs_control, strict=True):
        assert torch.equal(a[1], b[1]), "shared-prefix row diverged from private-copy row"
    # divergence wrote into a PRIVATE block, not the shared prefix
    assert shared._py_table[1][2] != shared._py_table[0][2]
    shared.free_slot(1)
    assert shared._refcount[shared._py_table[0][0]] == 1  # refcount released, prefix intact
    with pytest.raises(ValueError, match="full-block"):
        shared.share_prefix(0, 1, 17)


# --------------------------------------------------------------------- engine integration


_CHURN = [
    Request(0, (1, 2, 3), 4),
    Request(1, (10, 20, 30, 40, 50), 9),
    Request(2, (7, 7), 2),
    Request(3, (5, 4, 3, 2), 6),
    Request(4, (100, 3, 55, 9), 3),
    Request(5, (42,), 5),
]


def test_serve_paged_matches_dense_and_single_stream() -> None:
    """End-to-end R4.1 oracle: the full scheduler on paged storage is token-identical to dense
    storage and to per-request greedy decode."""
    model = _model()
    dense = serve_continuous(model, _CHURN, n_slots=2, device="cpu")
    paged = serve_continuous(model, _CHURN, n_slots=2, device="cpu", cache_kind="paged")
    for d, p in zip(dense.completed, paged.completed, strict=True):
        assert d.request.request_id == p.request.request_id
        assert d.token_ids == p.token_ids, f"request {d.request.request_id}: paged != dense"
    for c in paged.completed:
        expected = generate(
            model,
            list(c.request.prompt_ids),
            SamplingParams(temperature=0.0, max_tokens=c.request.max_new_tokens),
            device="cpu",
        )
        assert list(c.token_ids) == expected
    assert paged.paged_frag_mean is not None and 0.0 <= paged.paged_frag_mean < 1.0
    assert paged.paged_alloc_peak_tokens is not None and paged.paged_alloc_peak_tokens > 0
    assert dense.paged_frag_mean is None


def test_admission_guard_serializes_when_pool_is_tight() -> None:
    """Committed-blocks guard: a pool that fits one request to budget serializes admissions —
    and the outputs stay correct (capacity pressure must never corrupt, only delay)."""
    model = _model()
    reqs = [Request(0, (1, 2), 8), Request(1, (3, 4), 8)]  # each needs ⌈10/16⌉ = 1 block
    result = serve_continuous(
        model, reqs, n_slots=2, device="cpu", cache_kind="paged", paged_n_blocks=2
    )  # 1 usable block ⇒ strictly one active request at a time
    assert result.n_prefill_forwards == 2  # serialized admissions
    assert max(result.utilization) <= 0.5  # never both slots active
    for c in result.completed:
        expected = generate(
            model,
            list(c.request.prompt_ids),
            SamplingParams(temperature=0.0, max_tokens=c.request.max_new_tokens),
            device="cpu",
        )
        assert list(c.token_ids) == expected
