"""K3/kda gates beyond FLA parity (tests/test_kda_parity_fla.py owns operator-vs-FLA).

1. Strong decay stays finite in fp32, forward AND backward. Regression: the FLA-naive port
   exponentiated future pairs, exp(G_i - G_j) > 1, and returned NaN grads from g <= -2 at
   chunk 64 — K3's own gate range is (-5, 0).
2. Chunk-size independence: chunk 1, ragged, and chunk > T all equal the recurrence (fp64).
3. Hybrid-cache equivalence: prefill ≡ token-by-token decode ≡ split prefill, fp64, off-init.
4. Conv/recurrence causality: perturbing future tokens leaves past outputs unchanged.
5. Parameters match param_count's closed accounting, minus the A_log storage tail.
6. A_log loader: padded storage with a zero tail loads; a nonzero tail is refused.
7. Gate floor: extreme logits pin g to [lower_bound, 0] without NaN.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from scratch_llm.k3.config import KDAConfig, k3_full, mini_k3_d12
from scratch_llm.k3.core.kda import KDALayer, kda_chunk, kda_recurrent
from scratch_llm.k3.model import HybridState
from scratch_llm.k3.param_count import _kda_attn_params

FP64_STRUCTURAL_REL = 1e-10  # same bar as test_kda_parity_fla.py (two identical algebras)

_TINY = KDAConfig(num_heads=2, head_dim=8, decay_rank=4, a_log_size=4)
_HIDDEN = 16


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _operator_inputs(
    B: int, T: int, H: int, K: int, V: int, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    """K3-range gates: g = -5 * sigmoid(3 * randn) spans nearly all of (-5, 0)."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=gen, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, K, generator=gen, dtype=dtype), dim=-1)
    v = torch.randn(B, T, H, V, generator=gen, dtype=dtype)
    g = -5 * torch.sigmoid(3 * torch.randn(B, T, H, K, generator=gen, dtype=dtype))
    beta = torch.sigmoid(torch.randn(B, T, H, generator=gen, dtype=dtype))
    return q, k, v, g, beta


@pytest.mark.parametrize("g_value", [-2.0, -4.99])
def test_chunk_strong_decay_grads_finite_fp32(g_value: float) -> None:
    q, k, v, _, beta = _operator_inputs(1, 64, 2, 16, 16, torch.float32)
    g = torch.full_like(q, g_value)
    leaves = [t.requires_grad_() for t in (q, k, v, g, beta)]
    o, S = kda_chunk(q, k, v, g, beta, chunk_size=64)
    (o.square().sum() + S.square().sum()).backward()
    for name, t in zip("qkvgb", leaves, strict=True):
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"non-finite grad on {name}"


@pytest.mark.parametrize("chunk", [1, 5, 64])
def test_chunk_size_independence_fp64(chunk: int) -> None:
    q, k, v, g, beta = _operator_inputs(2, 37, 2, 8, 8, torch.float64, seed=1)
    h0 = torch.randn(2, 2, 8, 8, dtype=torch.float64)
    o_r, S_r = kda_recurrent(q, k, v, g, beta, initial_state=h0)
    o_c, S_c = kda_chunk(q, k, v, g, beta, initial_state=h0, chunk_size=chunk)
    assert _rel(o_c, o_r) < FP64_STRUCTURAL_REL and _rel(S_c, S_r) < FP64_STRUCTURAL_REL


def test_operator_dtypes_and_no_mutation() -> None:
    q, k, v, g, beta = _operator_inputs(1, 9, 1, 8, 8, torch.bfloat16)
    h0 = torch.randn(1, 1, 8, 8)
    before = h0.clone()
    for fn in (kda_recurrent, kda_chunk):
        o, S = fn(q, k, v, g, beta, initial_state=h0)
        assert o.dtype == torch.bfloat16 and S.dtype == torch.float32  # state never in bf16
        assert S.data_ptr() != h0.data_ptr()
    assert torch.equal(h0, before)


def _layer(chunk_size: int = 64, seed: int = 0) -> KDALayer:
    """fp64 layer moved off-init, so decays and gates are spread (K2_PROPOSAL_KDA §6.5)."""
    torch.manual_seed(seed)
    layer = KDALayer(_TINY, _HIDDEN, chunk_size=chunk_size).double()
    with torch.no_grad():
        layer.dt_bias.normal_(0.0, 2.0)
        layer.A_log.normal_(0.0, 0.5)
    return layer


def _fresh_state(batch: int) -> HybridState:
    return HybridState(kda_states=[None], mla_caches=[None], lengths=torch.zeros(batch))


@pytest.mark.parametrize("chunk", [4, 64])
def test_layer_prefill_equals_decode_equals_split_prefill(chunk: int) -> None:
    layer = _layer(chunk)
    x = torch.randn(2, 11, _HIDDEN, dtype=torch.float64)
    full = layer(x)

    decode_state = _fresh_state(2)
    decoded = torch.cat([layer(x[:, t : t + 1], decode_state, 1) for t in range(11)], dim=1)

    split_state = _fresh_state(2)
    split = torch.cat([layer(x[:, :5], split_state, 1), layer(x[:, 5:], split_state, 1)], dim=1)

    assert _rel(decoded, full) < FP64_STRUCTURAL_REL
    assert _rel(split, full) < FP64_STRUCTURAL_REL
    a, b = decode_state.kda_states[0], split_state.kda_states[0]
    assert a is not None and b is not None
    # Not bitwise: the raw projections come from different-shaped matmuls (1 vs 6 tokens).
    assert _rel(a.s_t, b.s_t) < FP64_STRUCTURAL_REL and _rel(a.conv, b.conv) < FP64_STRUCTURAL_REL


@pytest.mark.parametrize("chunk", [4, 64])
def test_layer_is_causal(chunk: int) -> None:
    layer = _layer(chunk, seed=1)
    x = torch.randn(1, 13, _HIDDEN, dtype=torch.float64)
    t0 = 6
    x2 = x.clone()
    x2[:, t0 + 1 :] = torch.randn_like(x2[:, t0 + 1 :])
    out, out2 = layer(x), layer(x2)
    torch.testing.assert_close(out2[:, : t0 + 1], out[:, : t0 + 1], rtol=0, atol=1e-12)
    assert not torch.allclose(out2[:, t0 + 1 :], out[:, t0 + 1 :])


def test_layer_parameters_match_param_count() -> None:
    for cfg in (mini_k3_d12(), k3_full()):
        with torch.device("meta"):
            layer = KDALayer(cfg.kda, cfg.hidden_size)
        live = sum(p.numel() for p in layer.parameters())
        assert live == _kda_attn_params(cfg) - (cfg.kda.a_log_size - cfg.kda.num_heads)


def test_a_log_loads_padded_storage_and_refuses_nonzero_tail() -> None:
    layer = KDALayer(_TINY, _HIDDEN)
    live = torch.tensor([0.3, -0.2])
    sd = layer.state_dict()
    sd["A_log"] = torch.cat([live, torch.zeros(2)])
    layer.load_state_dict(sd)
    assert torch.equal(layer.A_log.detach(), live)
    assert sd["A_log"].numel() == 4  # the caller's dict is not rewritten

    sd["A_log"] = torch.tensor([0.3, -0.2, 0.0, 1e-3])
    with pytest.raises(RuntimeError, match="nonzero padding"):
        layer.load_state_dict(sd)


def test_gate_floor_and_ceiling() -> None:
    layer = KDALayer(_TINY, _HIDDEN)
    x = torch.randn(1, 3, _HIDDEN)
    with torch.no_grad():
        layer.dt_bias.fill_(1e4)
        g_floor, _ = layer._gates(x)
        layer.dt_bias.fill_(-1e4)
        g_ceiling, _ = layer._gates(x)
    assert torch.isfinite(g_floor).all() and torch.isfinite(g_ceiling).all()
    torch.testing.assert_close(g_floor, torch.full_like(g_floor, _TINY.gate_lower_bound))
    torch.testing.assert_close(g_ceiling, torch.zeros_like(g_ceiling))
