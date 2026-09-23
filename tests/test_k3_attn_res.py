"""K3/AttnRes gates (spec §1): the depth mix against two oracles, and the block bookkeeping.

1. Mix ≡ HF ``_apply_attn_res`` (in-file transcription) and ≡ FLA ``naive_attnres`` (loaded by
   path), fp64 at the structural bar, 1..9 sources, random queries and key-norm weights; bitwise
   equal to the verbatim HF transcription on fp32 and bf16 inputs.
2. Bookkeeping: a toy stack whose sublayers are fixed tanh(W x) maps, run three ways —
   ``AttnResState``, a naive rebuild that keeps every sublayer output and regroups it by block,
   and the HF decoder-layer loop transcribed — agrees in fp64 for B = 1, 2, 3, L, L+5 (L = 7).
   No update writes into a tensor already handed out, at any of the three update sites.
3. The two plausible misreads of the boundary — FLA's sublayer count (2ℓ) % B and a 1-based
   layer index — planted in the HF loop, move the output far above the bar.
4. A zero query mixes the exact uniform mean.
5. Gradients: every multi-source query and key-norm weight gets one; the layer-0 attention pair
   (one source, softmax ≡ 1) gets none; and every gradient of the toy stack (embedding, sublayer
   weights, queries, key-norm weights) equals the HF loop's — a detach is invisible to gate 2.
6. Source counts match vLLM's block arithmetic at every sublayer: 9 at K3's output, 4 at mini's,
   {emb, prefix} when B ≥ L, and the embedding is always source 0.
7. bf16 sources mix in fp32 and come back bf16; autocast cannot pull the math down; the stream
   keeps the embedding's dtype both under train.py's bf16 autocast and when fed wider outputs.
"""

from __future__ import annotations

import importlib.util
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from scratch_llm.k3.config import k3_full, mini_k3_d12
from scratch_llm.k3.core.attn_res import AttnResState, attn_res_mix

FP64_STRUCTURAL_REL = 1e-10  # same bar as test_k3_kda.py (two identical algebras in fp64)

# Huy review. FLA as shipped computes in fp32 whatever the input, so vs our fp32 path it is two
# fp32 evaluations of one algebra in different reduction orders: each lands ~2.5·2⁻²⁴ ≈ 1.5e-7
# from exact (measured, H = 64, 9 sources), so the gap stays below 5·2⁻²⁴ ≈ 3e-7; 1e-6 is headroom.
FLA_FP32_REL = 1e-6

EPS = 1e-5  # rms_norm_eps (config.json), the key-norm epsilon of every AttnRes site


def _rel(a: Tensor, b: Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _mix_inputs(n: int, dtype: torch.dtype, seed: int) -> tuple[list[Tensor], Tensor, Tensor]:
    """n sources [2, 5, 64] at spread magnitudes (the keys are normalised, the values are not),
    a [1, 64] pseudo-query with logit std ≈ 2, key-norm weight 1 ± 0.3. Drawn in fp64, then cast."""
    gen = torch.Generator().manual_seed(seed)
    h = 64
    scales = torch.randn(n, generator=gen, dtype=torch.float64).exp()
    sources = [torch.randn(2, 5, h, generator=gen, dtype=torch.float64) * s for s in scales]
    query = torch.randn(1, h, generator=gen, dtype=torch.float64) * (2 / math.sqrt(h))
    norm_weight = 1 + 0.3 * torch.randn(h, generator=gen, dtype=torch.float64)
    return [s.to(dtype) for s in sources], query.to(dtype), norm_weight.to(dtype)


# ---------------------------------------------------------------------------
# Oracles, transcribed (never imported from ../ladders).
# ---------------------------------------------------------------------------


def _hf_apply_attn_res(
    prefix_sum: Tensor,
    block_residual: Tensor,
    proj_weight: Tensor,
    norm_weight: Tensor,
    eps: float,
    up: torch.dtype,
) -> Tensor:
    """HF modeling_kimi_linear._apply_attn_res (:1075-1088), transcribed. The modules become their
    tensors (``proj.weight``, ``norm.weight``, ``norm.variance_epsilon``) and each ``.float()``
    becomes ``.to(up)``: ``up=float32`` is verbatim, ``float64`` lifts the same algebra.

    prefix_sum [N, H], block_residual [N, num_blocks, H].
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.to(up)
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + eps)
    score_weight = norm_weight.to(up) * proj_weight.squeeze(0).to(up)
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


def _hf_mix(
    sources: Sequence[Tensor], query: Tensor, norm_weight: Tensor, up: torch.dtype
) -> Tensor:
    """HF's calling convention on a source list: the last source is ``prefix_sum``, the rest the
    ``block_residual`` stack (empty [N, 0, H] for one source, as ``KimiLinearModel`` starts it)."""
    shape, h = sources[0].shape, sources[0].shape[-1]
    prefix = sources[-1].reshape(-1, h)
    if len(sources) > 1:
        blocks = torch.stack(tuple(sources[:-1]), dim=-2).reshape(-1, len(sources) - 1, h)
    else:
        blocks = prefix.new_zeros(prefix.shape[0], 0, h)
    return _hf_apply_attn_res(prefix, blocks, query, norm_weight, EPS, up).reshape(shape)


def _fla_attnres_naive() -> ModuleType:
    """Load oss/fla/fla/ops/attnres/naive.py by path; skip cleanly if the checkout or einops is
    absent (the tests/test_kda_parity_fla.py pattern)."""
    root = os.environ.get("LADDERS_OSS")
    oss = Path(root) if root else Path(__file__).resolve().parents[1].parent / "ladders" / "oss"
    path = oss / "fla" / "fla" / "ops" / "attnres" / "naive.py"
    if not path.exists():
        pytest.skip(f"no fla checkout at {path} (set LADDERS_OSS or run infra/oss.sh)")
    if importlib.util.find_spec("einops") is None:
        pytest.skip("einops missing: uv pip install --python .venv/bin/python einops")
    spec = importlib.util.spec_from_file_location("_fla_attnres_naive", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. The mix against HF and FLA.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 10))
def test_mix_matches_hf_transcription_fp64(n: int) -> None:
    sources, query, norm_weight = _mix_inputs(n, torch.float64, seed=n)
    ours = attn_res_mix(sources, query, norm_weight, EPS)
    assert _rel(ours, _hf_mix(sources, query, norm_weight, torch.float64)) < FP64_STRUCTURAL_REL


@pytest.mark.parametrize("n", range(1, 10))
def test_mix_matches_fla_naive_fp64(n: int, monkeypatch: pytest.MonkeyPatch) -> None:
    fla = _fla_attnres_naive()
    sources, query, norm_weight = _mix_inputs(n, torch.float64, seed=n)
    ours = attn_res_mix(sources, query, norm_weight, EPS)
    with monkeypatch.context() as m:
        # naive_attnres reaches fp32 only through ``.float()`` on each input; lifting that one
        # cast runs FLA's own algebra (F.rms_norm, einsum, softmax, scale=1.0) in fp64.
        m.setattr(torch.Tensor, "float", torch.Tensor.double)
        theirs = fla.naive_attnres(query, sources, norm_weight, rms_eps=EPS)
    assert theirs.dtype == torch.float64
    assert _rel(ours, theirs) < FP64_STRUCTURAL_REL


def test_mix_matches_fla_naive_as_shipped_fp32() -> None:
    fla = _fla_attnres_naive()
    sources, query, norm_weight = _mix_inputs(9, torch.float32, seed=11)
    ours = attn_res_mix(sources, query, norm_weight, EPS)
    theirs = fla.naive_attnres(query, sources, norm_weight, rms_eps=EPS)
    assert _rel(ours, theirs) < FLA_FP32_REL


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mix_bitwise_equal_to_hf_verbatim(dtype: torch.dtype) -> None:
    sources, query, norm_weight = _mix_inputs(9, dtype, seed=4)
    ours = attn_res_mix(sources, query, norm_weight, EPS)
    assert ours.dtype == dtype
    assert torch.equal(ours, _hf_mix(sources, query, norm_weight, torch.float32))


def test_mix_contract() -> None:
    sources, query, norm_weight = _mix_inputs(3, torch.float64, seed=0)
    both = attn_res_mix(sources, query, norm_weight, EPS)
    assert torch.equal(attn_res_mix(sources, query.reshape(-1), norm_weight, EPS), both)
    assert attn_res_mix(sources[:1], query, norm_weight, EPS) is sources[0]  # HF/vLLM bypass
    with pytest.raises(ValueError, match="at least one"):
        attn_res_mix([], query, norm_weight, EPS)
    with pytest.raises(ValueError, match="shape and dtype"):
        attn_res_mix([sources[0], sources[1][:1]], query, norm_weight, EPS)
    with pytest.raises(ValueError, match="shape and dtype"):
        attn_res_mix([sources[0], sources[1].float()], query, norm_weight, EPS)
    with pytest.raises(ValueError, match="query"):
        attn_res_mix(sources, query[:, :-1], norm_weight, EPS)


# ---------------------------------------------------------------------------
# 2–3. Bookkeeping: three runs of one toy stack, and two planted misreads.
# ---------------------------------------------------------------------------

_L, _H = 7, 8


class _Toy(NamedTuple):
    """fp64 toy stack: sublayer ℓ is tanh(W x); every tensor is a leaf that requires grad, and
    the queries and key-norm weights are off-init."""

    emb: Tensor  # [2, 3, H]
    w_attn: list[Tensor]  # [H, H] per layer
    w_mlp: list[Tensor]
    q_attn: list[Tensor]  # pseudo-queries [1, H] per layer
    q_mlp: list[Tensor]
    g_attn: list[Tensor]  # key-norm weights [H] per layer
    g_mlp: list[Tensor]
    q_out: Tensor
    g_out: Tensor


def _toy(seed: int = 0) -> _Toy:
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    def query() -> Tensor:
        return (randn(1, _H) * (2 / math.sqrt(_H))).requires_grad_()

    def norm() -> Tensor:
        return (1 + 0.3 * randn(_H)).requires_grad_()

    def weight() -> Tensor:
        return (randn(_H, _H) * (2 / math.sqrt(_H))).requires_grad_()

    return _Toy(
        emb=randn(2, 3, _H).requires_grad_(),
        w_attn=[weight() for _ in range(_L)],
        w_mlp=[weight() for _ in range(_L)],
        q_attn=[query() for _ in range(_L)],
        q_mlp=[query() for _ in range(_L)],
        g_attn=[norm() for _ in range(_L)],
        g_mlp=[norm() for _ in range(_L)],
        q_out=query(),
        g_out=norm(),
    )


def _sublayer(w: Tensor, x: Tensor) -> Tensor:
    return torch.tanh(x @ w.T)


def _run_state(toy: _Toy, block_size: int) -> Tensor:
    """(a) The module under test: ``AttnResState`` + ``attn_res_mix``."""
    state = AttnResState.start(toy.emb)
    for i in range(_L):
        x = attn_res_mix(state.sources(), toy.q_attn[i], toy.g_attn[i], EPS)
        state.after_attn(_sublayer(toy.w_attn[i], x), i, block_size)
        x = attn_res_mix(state.sources(), toy.q_mlp[i], toy.g_mlp[i], EPS)
        state.after_mlp(_sublayer(toy.w_mlp[i], x))
    return attn_res_mix(state.sources(), toy.q_out, toy.g_out, EPS)


def _run_naive(toy: _Toy, block_size: int) -> Tensor:
    """(b) No bank, no prefix, no boundary test: keep every sublayer output with its layer index
    and rebuild each source list from the definition — [embedding] + one source per started
    block (layer // B), holding the sum of that block's outputs so far."""
    outputs: list[tuple[int, Tensor]] = []

    def sources() -> list[Tensor]:
        blocks: dict[int, Tensor] = {}
        for layer_idx, h in outputs:
            n = layer_idx // block_size
            blocks[n] = blocks[n] + h if n in blocks else h
        return [toy.emb, *blocks.values()]

    for i in range(_L):
        x = attn_res_mix(sources(), toy.q_attn[i], toy.g_attn[i], EPS)
        outputs.append((i, _sublayer(toy.w_attn[i], x)))
        x = attn_res_mix(sources(), toy.q_mlp[i], toy.g_mlp[i], EPS)
        outputs.append((i, _sublayer(toy.w_mlp[i], x)))
    return attn_res_mix(sources(), toy.q_out, toy.g_out, EPS)


def _k3_boundary(layer_idx: int, block_size: int) -> bool:
    return layer_idx % block_size == 0  # HF :995, vLLM is_block_write_layer :977


def _fla_sublayer_boundary(layer_idx: int, block_size: int) -> bool:
    # FLA modeling_kda.py counts sublayers: attention opens a block at (2ℓ) % B == 0. Its mlp test
    # (2ℓ+1) % B never fires for even B, so for B = 4 this attention plant IS the FLA convention.
    return (2 * layer_idx) % block_size == 0


def _one_based_boundary(layer_idx: int, block_size: int) -> bool:
    return (layer_idx + 1) % block_size == 0  # model.py's 1-based layer_id fed in unshifted


def _hf_forward(
    toy: _Toy, block_size: int, opens_block: Callable[[int, int], bool] = _k3_boundary
) -> Tensor:
    """(c) HF ``KimiLinearModel.forward`` (:1188-1219) → ``KimiDecoderLayer._forward_attn_residual``
    (:973-1046) → ``_apply_output_attn_res`` (:1226-1233), transcribed. ``self_attn`` / ``mlp`` are
    the toy sublayers, ``input_layernorm`` / ``post_attention_layernorm`` are identity in the toy,
    the mix is ``_hf_apply_attn_res`` lifted to fp64, and ``opens_block`` stands in for line 995
    so a misread can be planted. Independent of ``attn_res.py`` entirely."""
    batch_size, seq_len, hidden_size = toy.emb.shape
    hidden_states = toy.emb
    block_residual = hidden_states.new_zeros(batch_size * seq_len, 0, hidden_size)
    for layer_idx in range(_L):
        prefix_sum: Tensor | None = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = _hf_apply_attn_res(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                toy.q_attn[layer_idx],
                toy.g_attn[layer_idx],
                EPS,
                torch.float64,
            ).view(batch_size, seq_len, hidden_size)
        if opens_block(layer_idx, block_size):
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1
            )
            prefix_sum = None
        hidden_states = _sublayer(toy.w_attn[layer_idx], hidden_states)
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        hidden_states = _hf_apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            toy.q_mlp[layer_idx],
            toy.g_mlp[layer_idx],
            EPS,
            torch.float64,
        ).view(batch_size, seq_len, hidden_size)
        hidden_states = _sublayer(toy.w_mlp[layer_idx], hidden_states)
        hidden_states = prefix_sum + hidden_states  # the layer returns prefix_sum (:1040-1046)
    return _hf_apply_attn_res(
        hidden_states.view(-1, hidden_size),
        block_residual,
        toy.q_out,
        toy.g_out,
        EPS,
        torch.float64,
    ).view(batch_size, seq_len, hidden_size)


@pytest.mark.parametrize("block_size", [1, 2, 3, _L, _L + 5])
def test_state_matches_naive_rebuild_and_hf_loop_fp64(block_size: int) -> None:
    toy = _toy(seed=block_size)
    ours = _run_state(toy, block_size)
    assert _rel(ours, _run_naive(toy, block_size)) < FP64_STRUCTURAL_REL
    assert _rel(ours, _hf_forward(toy, block_size)) < FP64_STRUCTURAL_REL


@pytest.mark.parametrize("misread", [_fla_sublayer_boundary, _one_based_boundary])
def test_planted_boundary_misreads_are_visible(misread: Callable[[int, int], bool]) -> None:
    toy = _toy(seed=0)
    ours = _run_state(toy, 4)
    assert _rel(_hf_forward(toy, 4), ours) < FP64_STRUCTURAL_REL
    assert _rel(_hf_forward(toy, 4, misread), ours) > 1e-3  # O(1) moves, 7 decades above the bar


def test_state_rejects_a_one_based_layer_index() -> None:
    state = AttnResState.start(torch.zeros(1))
    with pytest.raises(ValueError, match="0-based"):
        state.after_attn(torch.ones(1), 1, 4)


def test_state_never_writes_into_tensors_or_returned_lists() -> None:
    """Layers 0-2 at B = 2 hit every update site with both a leaf and a derived prefix. An
    in-place ``prefix += h`` would rewrite a source the previous mix already consumed — the
    autograd hazard under activation checkpointing — while every output value stays the same,
    so only this snapshot can see it."""
    gen = torch.Generator().manual_seed(6)
    emb, *h = torch.randn(7, 2, 3, _H, generator=gen, dtype=torch.float64).unbind()
    inputs = [emb, *h]
    inputs_snapshot = [t.clone() for t in inputs]
    state = AttnResState.start(emb)

    def leaves_handed_out_tensors_alone(update: Callable[[], None]) -> None:
        prefix, handed, bank = state.prefix, state.sources(), state.bank
        snapshot, n_committed = [t.clone() for t in handed], len(bank)
        update()
        assert state.prefix is not prefix  # rebound to a new tensor, never written through
        assert len(bank) == n_committed  # the old bank list is never appended to
        assert all(torch.equal(t, s) for t, s in zip(handed, snapshot, strict=True))

    leaves_handed_out_tensors_alone(lambda: state.after_attn(h[0], 0, 2))  # opens; prefix emb
    assert len(state.sources()) == 2 and state.sources()[0] is emb and state.sources()[1] is h[0]
    leaves_handed_out_tensors_alone(lambda: state.after_mlp(h[1]))  # prefix h[0], a leaf
    leaves_handed_out_tensors_alone(lambda: state.after_attn(h[2], 1, 2))  # in-block; derived
    leaves_handed_out_tensors_alone(lambda: state.after_mlp(h[3]))  # derived prefix
    leaves_handed_out_tensors_alone(lambda: state.after_attn(h[4], 2, 2))  # commits a derived one
    leaves_handed_out_tensors_alone(lambda: state.after_mlp(h[5]))
    assert len(state.sources()) == 3 and state.sources()[0] is emb
    assert all(torch.equal(t, s) for t, s in zip(inputs, inputs_snapshot, strict=True))


# ---------------------------------------------------------------------------
# 4–7. Uniform limit, gradients, source counts, precision.
# ---------------------------------------------------------------------------


def test_zero_query_is_the_uniform_mean() -> None:
    sources, query, norm_weight = _mix_inputs(5, torch.float64, seed=7)
    mean = torch.stack(sources).mean(0)
    uniform = attn_res_mix(sources, torch.zeros_like(query), norm_weight, EPS)
    assert _rel(uniform, mean) < FP64_STRUCTURAL_REL
    assert _rel(attn_res_mix(sources, query, norm_weight, EPS), mean) > 1e-3  # a real query is not


def test_every_multi_source_query_gets_a_gradient() -> None:
    toy = _toy(seed=5)
    target = torch.randn(
        toy.emb.shape, generator=torch.Generator().manual_seed(5), dtype=torch.float64
    )
    (_run_state(toy, 3) * target).sum().backward()
    for t in (toy.q_attn[0], toy.g_attn[0]):  # one source: softmax ≡ 1 (None under the bypass)
        assert t.grad is None or not t.grad.any()
    multi_source = [*toy.q_attn[1:], *toy.g_attn[1:], *toy.q_mlp, *toy.g_mlp, toy.q_out, toy.g_out]
    for t in multi_source:
        assert t.grad is not None and t.grad.abs().max().item() > 0


@pytest.mark.parametrize("block_size", [1, 2, 3, _L, _L + 5])
def test_state_gradients_match_hf_loop_fp64(block_size: int) -> None:
    """A detached source (values, keys or a committed block) leaves every output value as it is
    and moves gradients by O(1), so gate 2 cannot see it; this gate can. The naive rebuild shares
    ``attn_res_mix``, so the HF transcription is the independent leg. The layer-0 attention pair
    is in neither graph (HF skips its mix), so it is left out rather than allowed unused."""
    toy = _toy(seed=block_size)
    target = torch.randn(
        toy.emb.shape, generator=torch.Generator().manual_seed(block_size), dtype=torch.float64
    )
    leaves = [
        toy.emb,
        *toy.w_attn,
        *toy.w_mlp,
        *toy.q_attn[1:],
        *toy.g_attn[1:],
        *toy.q_mlp,
        *toy.g_mlp,
        toy.q_out,
        toy.g_out,
    ]
    ours = torch.autograd.grad((_run_state(toy, block_size) * target).sum(), leaves)
    theirs = torch.autograd.grad((_hf_forward(toy, block_size) * target).sum(), leaves)
    for a, b in zip(ours, theirs, strict=True):
        assert _rel(a, b) < FP64_STRUCTURAL_REL


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@pytest.mark.parametrize(
    ("num_layers", "block_size", "n_final"),
    [
        pytest.param(k3_full().num_layers, k3_full().attn_res_block_size, 9, id="k3_full"),
        pytest.param(mini_k3_d12().num_layers, mini_k3_d12().attn_res_block_size, 4, id="mini"),
        pytest.param(_L, _L, 2, id="B=L"),
        pytest.param(_L, _L + 5, 2, id="B>L"),
    ],
)
def test_source_counts_match_vllm_block_arithmetic(
    num_layers: int, block_size: int, n_final: int
) -> None:
    emb = torch.zeros(1)
    state = AttnResState.start(emb)
    for i in range(num_layers):
        assert len(state.sources()) == _cdiv(i, block_size) + 1  # vLLM prev_valid_blocks :979
        state.after_attn(torch.zeros(1), i, block_size)
        opened = int(i % block_size == 0)
        assert len(state.sources()) == _cdiv(i, block_size) + opened + 1  # mlp_valid_blocks :1066
        state.after_mlp(torch.zeros(1))
    final = state.sources()
    assert len(final) == _cdiv(num_layers, block_size) + 1 == n_final  # num_attn_res_blocks :1179
    assert final[0] is emb


def test_bf16_sources_mix_in_fp32() -> None:
    sources, query, norm_weight = _mix_inputs(9, torch.float32, seed=5)
    sources_bf16 = [s.bfloat16() for s in sources]
    out = attn_res_mix(sources_bf16, query, norm_weight, EPS)
    assert out.dtype == torch.bfloat16
    # bf16 → fp32 is exact, so fp32 math on the upcast sources is the same computation.
    fp32_math = attn_res_mix([s.float() for s in sources_bf16], query, norm_weight, EPS)
    assert torch.equal(out, fp32_math.bfloat16())
    bf16_math = _hf_mix(sources_bf16, query.bfloat16(), norm_weight.bfloat16(), torch.bfloat16)
    assert not torch.equal(out, bf16_math)  # the check can tell the two precisions apart

    plain = attn_res_mix(sources, query, norm_weight, EPS)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        under_autocast = attn_res_mix(sources, query, norm_weight, EPS)
        hf_under_autocast = _hf_mix(sources, query, norm_weight, torch.float32)
    assert under_autocast.dtype == torch.float32 and torch.equal(under_autocast, plain)
    assert not torch.equal(hf_under_autocast, plain)  # autocast would have lowered the matmul


def test_stream_keeps_the_embedding_dtype_under_bf16_autocast() -> None:
    """train.py's ``amp_dtype="bf16"`` path: under autocast the embedding lookup stays fp32 while
    every linear returns bf16. The state casts each sublayer output into the embedding's dtype,
    so the mixes see one dtype (they would raise on two) and the residual stream stays fp32."""
    gen = torch.Generator().manual_seed(3)
    table = torch.randn(11, _H, generator=gen).requires_grad_()
    weights = [torch.randn(_H, _H, generator=gen).requires_grad_() for _ in range(4)]
    query, norm_weight = torch.randn(1, _H, generator=gen), torch.ones(_H)
    ids = torch.randint(0, 11, (2, 3), generator=gen)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        state = AttnResState.start(F.embedding(ids, table))
        for i in range(2):  # B = 1: both layers commit a block and restart the prefix
            h = F.linear(attn_res_mix(state.sources(), query, norm_weight, EPS), weights[2 * i])
            assert h.dtype == torch.bfloat16  # autocast is live
            state.after_attn(h, i, 1)
            h = F.linear(attn_res_mix(state.sources(), query, norm_weight, EPS), weights[2 * i + 1])
            state.after_mlp(h)
        out = attn_res_mix(state.sources(), query, norm_weight, EPS)
    assert all(v.dtype == torch.float32 for v in state.sources()) and out.dtype == torch.float32
    out.sum().backward()
    assert table.grad is not None and all(w.grad is not None for w in weights)


def test_a_wider_sublayer_output_is_rounded_into_the_stream() -> None:
    """The other direction: a bf16 stream fed fp32 outputs stays bf16 at all three update sites
    (a promoted prefix beside the bf16 embedding would make the next mix raise)."""
    emb, *h = torch.randn(5, 2, 3, _H, generator=torch.Generator().manual_seed(4)).unbind()
    state = AttnResState.start(emb.bfloat16())
    state.after_attn(h[0], 0, 2)  # opens a block
    state.after_mlp(h[1])
    state.after_attn(h[2], 1, 2)  # in-block
    state.after_mlp(h[3])
    assert all(v.dtype == torch.bfloat16 for v in state.sources())
