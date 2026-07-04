"""A1 Rung 4.4 — CUDA-graph decode oracle (gpu-marked; runs on the standing sm120 box, not in CI).

Capture changes only HOW the decode step is launched (one cudaGraphLaunch instead of ~54–68 kernel
launches), never WHAT it computes: graph-replayed greedy tokens must be IDENTICAL to the eager
paged-kernel decode, for every batch size and across block boundaries.
"""

import random

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():  # pragma: no cover - gpu gate
    pytest.skip("CUDA required for CUDA-graph decode", allow_module_level=True)
pytest.importorskip("triton")

from scratch_llm.model import ModelConfig, PagedKVCache, PrefillView, TransformerLM  # noqa: E402
from scratch_llm.serving.cudagraph import CudaGraphDecoder  # noqa: E402

_PROMPT = (
    20  # not a multiple of BLOCK=16 → decode crosses a 16-boundary (exercises pre_decode_reserve)
)
_DECODE = 40


def _model() -> TransformerLM:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=512, d_model=64, n_layers=3, n_heads=8, n_kv_heads=2, context_length=256
    )
    return TransformerLM(cfg).to("cuda", torch.bfloat16).eval()


def _prefilled(model: TransformerLM, b: int) -> tuple[PagedKVCache, torch.Tensor]:
    blk = PagedKVCache.BLOCK
    max_blocks = (_PROMPT + _DECODE + blk - 1) // blk
    cache = PagedKVCache(
        n_layers=model.cfg.n_layers,
        n_slots=b,
        n_kv_heads=model.cfg.kv_heads,
        max_ctx=model.cfg.context_length,
        head_dim=model.cfg.head_dim,
        n_blocks=b * max_blocks + 1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cache.use_kernel = True
    rng = random.Random(1)
    prompts = [[rng.randrange(512) for _ in range(_PROMPT)] for _ in range(b)]
    x = torch.tensor(prompts, dtype=torch.long, device="cuda")
    with torch.no_grad():
        logits = model(x, PrefillView(cache, list(range(b)), [_PROMPT] * b))
    cache.mirror_admit(list(range(b)), [_PROMPT] * b)
    first = logits[torch.arange(b, device="cuda"), _PROMPT - 1].argmax(dim=-1)
    return cache, first


@torch.no_grad()
def _eager(model: TransformerLM, cache: PagedKVCache, first: torch.Tensor) -> torch.Tensor:
    last, out = first, []
    for _ in range(_DECODE):
        cache.pre_decode_reserve()
        last = model(last.unsqueeze(1), cache)[:, -1].argmax(dim=-1)
        cache.mirror_advance()
        out.append(last)
    return torch.stack(out)


@pytest.mark.parametrize("b", [1, 4, 8])
def test_graph_decode_token_exact(b: int) -> None:
    """The load-bearing gate: CUDA-graph decode is token-identical to the eager paged decode."""
    model = _model()
    ce, fe = _prefilled(model, b)
    eager_tokens = _eager(model, ce, fe)
    cg, fg = _prefilled(model, b)
    dec = CudaGraphDecoder(model, cg)
    dec.capture()
    graph_tokens = dec.decode(fg, _DECODE)
    assert torch.equal(eager_tokens, graph_tokens), f"B={b}: graph decode diverged from eager"


def test_requires_paged_kernel_cache() -> None:
    """CudaGraphDecoder rejects a cache that is not a paged kernel cache (fixed-shape substrate)."""
    from scratch_llm.model import BatchedKVCache

    model = _model()
    dense = BatchedKVCache(
        model.cfg.n_layers, 1, model.cfg.kv_heads, 64, model.cfg.head_dim, "cuda"
    )
    with pytest.raises(ValueError):
        CudaGraphDecoder(model, dense)  # type: ignore[arg-type]
    paged = PagedKVCache(
        model.cfg.n_layers, 1, model.cfg.kv_heads, 64, model.cfg.head_dim, 8, "cuda"
    )
    paged.use_kernel = False
    with pytest.raises(ValueError):
        CudaGraphDecoder(model, paged)  # use_kernel must be True
