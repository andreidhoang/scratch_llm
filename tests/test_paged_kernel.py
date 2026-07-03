"""A1 R4.1b — fused paged decode kernel vs the gather-path oracle (GPU).

The gather path is bit-exact vs dense storage (test_paged_cache), so it is the reference here:
the Triton kernel must match it to dtype tolerance on scattered ragged blocks, and be greedy
token-exact end-to-end (the argmax gate that actually matters for serving).
"""

import pytest
import torch

from scratch_llm.model import (
    ModelConfig,
    PagedKVCache,
    PrefillView,
    TransformerLM,
    scaled_dot_product_attention,
)

pytestmark = pytest.mark.gpu


def _require_gpu_triton() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("triton")


def _gather_reference(cache: PagedKVCache, q: torch.Tensor, layer: int) -> torch.Tensor:
    """The R4.1a oracle path: gathered dense view + per-row mask + GQA repeat + SDPA."""
    k, v = cache.decode_view(layer)
    n_heads = q.shape[1]
    repeats = n_heads // cache.n_kv_heads
    if repeats > 1:
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    k_pos = torch.arange(k.shape[2], device=q.device)
    mask = (k_pos.unsqueeze(0) <= cache.lengths.unsqueeze(1))[:, None, None, :]
    return scaled_dot_product_attention(q, k, v, mask)


@pytest.mark.parametrize(
    ("dtype", "atol"), [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)], ids=["fp32", "bf16"]
)
def test_kernel_matches_gather_path(dtype: torch.dtype, atol: float) -> None:
    _require_gpu_triton()
    from scratch_llm.kernels.paged_decode_triton import paged_decode_attention

    torch.manual_seed(0)
    n_slots, n_kv, head_dim, n_heads = 5, 2, 64, 8
    cache = PagedKVCache(
        n_layers=1,
        n_slots=n_slots,
        n_kv_heads=n_kv,
        max_ctx=128,
        head_dim=head_dim,
        n_blocks=48,
        device="cuda",
        dtype=dtype,
    )
    import random

    random.Random(3).shuffle(cache._free)  # deliberately scattered physical blocks
    lengths = [1, 15, 16, 33, 70]  # boundary-adjacent + a fresh row
    slots = list(range(n_slots))
    cache.reserve_prefill(slots, lengths)
    cache.mirror_admit(slots, lengths)
    cache.lengths[:] = torch.tensor(lengths, device="cuda")
    cache.active[:] = True
    for layer in range(cache.n_layers):  # real content in every allocated block
        cache._pool_k[layer].normal_()
        cache._pool_v[layer].normal_()

    k_new = torch.randn(n_slots, n_kv, 1, head_dim, device="cuda", dtype=dtype)
    v_new = torch.randn(n_slots, n_kv, 1, head_dim, device="cuda", dtype=dtype)
    cache.write_decode(0, k_new, v_new)
    q = torch.randn(n_slots, n_heads, 1, head_dim, device="cuda", dtype=dtype)

    ref = _gather_reference(cache, q, layer=0)
    out = paged_decode_attention(
        q, cache._pool_k[0], cache._pool_v[0], cache.block_table, cache.lengths
    )
    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=1e-3), (
        f"max |Δ| = {(out.float() - ref.float()).abs().max().item():.2e}"
    )


def test_kernel_e2e_greedy_token_exact() -> None:
    """use_kernel=True must reproduce the gather path's greedy tokens through a real model."""
    _require_gpu_triton()
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=256, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, context_length=128
    )
    model = TransformerLM(cfg).to("cuda").eval()
    prompts = [[1, 2, 3], [10, 20, 30, 40, 50], list(range(1, 18))]

    outs: dict[bool, list[list[int]]] = {}
    for use_kernel in (False, True):
        cache = PagedKVCache(
            cfg.n_layers,
            3,
            cfg.kv_heads,
            cfg.context_length,
            cfg.head_dim,
            n_blocks=32,
            device="cuda",
        )
        cache.use_kernel = use_kernel
        lens = [len(p) for p in prompts]
        x = torch.zeros((3, max(lens)), dtype=torch.long, device="cuda")
        for i, p in enumerate(prompts):
            x[i, : lens[i]] = torch.tensor(p, dtype=torch.long, device="cuda")
        with torch.no_grad():
            view = PrefillView(cache, [0, 1, 2], lens)
            logits = model(x, view)
            assert isinstance(logits, torch.Tensor)
            cache.mirror_admit([0, 1, 2], lens)
            nid = logits[
                torch.arange(3, device="cuda"), torch.tensor(lens, device="cuda") - 1
            ].argmax(-1)
            tokens: list[list[int]] = [[int(nid[b])] for b in range(3)]
            xd = nid.unsqueeze(1)
            for _ in range(24):  # crosses block boundaries for every row
                cache.pre_decode_reserve()
                logits = model(xd, cache)
                assert isinstance(logits, torch.Tensor)
                cache.mirror_advance()
                nid = logits[:, -1].argmax(dim=-1)
                for b in range(3):
                    tokens[b].append(int(nid[b]))
                xd = nid.unsqueeze(1)
        outs[use_kernel] = tokens
    assert outs[True] == outs[False], "paged kernel diverged from the gather path under greedy"
