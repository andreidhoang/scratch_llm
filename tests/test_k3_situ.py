"""K3/SiTU-GLU gates (docs/k3/K5_PROPOSAL_SITU.md §6) plus the Moonshot-reference transcription.

1. Bound: |f| ≤ β1·β2 = 100, gate branch ≤ β1, up branch ≤ β2, on sweeps out to ±1e4 (exact).
2. SwiGLU closeness: |f − SiLU(g)·u| ≤ σ(g)·(|g|³|u|/(3β1²) + |g||u|³/(3β2²)) for ALL inputs
   (exact inequality from |z − β·tanh(z/β)| ≤ |z|³/(3β²)), plus the g = 6 wiring point that tells
   the uncapped sigmoid from a capped one.
3. fp32 saturation: β·tanh(±1e3/β) == ±β exactly; f(1e3, 1e3) == 100 exactly.
4. Negative tail: g → −∞ sends the output to 0 whatever u is.
5. β → ∞ recovers SwiGLU.
6. Derivative: autograd ≡ the hand product rule; gradcheck in fp64; finite at saturation.
7. Reference: bitwise equal to HF ``SituAndMul`` on fp32 inputs; ``DenseSiTUMLP`` ≡ HF ``KimiMLP``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from scratch_llm.k3.config import k3_full, mini_k3_d12
from scratch_llm.k3.core.situ import (
    DenseSiTUMLP,
    SiTUConfig,
    SiTUGLU,
    situ_gate,
    situ_glu,
    soft_cap,
)
from scratch_llm.k3.param_count import _glu_mlp_params

B1, B2 = 4.0, 25.0


def _hf_situ_and_mul(x: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """Transcribed from HF modeling_kimi_linear.SituAndMul.forward (Kimi-K3 release)."""
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


def _sweep(dtype: torch.dtype, n: int = 20_000, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Magnitudes spread log-uniformly over 1e-3..1e4, both signs, plus the exact extremes."""
    gen = torch.Generator().manual_seed(seed)
    mag = 10 ** (torch.rand(2, n, generator=gen, dtype=torch.float64) * 7 - 3)
    sign = torch.where(torch.rand(2, n, generator=gen) < 0.5, -1.0, 1.0).double()
    g, u = (mag * sign).to(dtype)
    edges = torch.tensor([1e4, -1e4, 0.0, 1e4, -1e4], dtype=dtype)
    return torch.cat([g, edges]), torch.cat([u, edges.flip(0)])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_bound_is_exact(dtype: torch.dtype) -> None:
    g, u = _sweep(dtype)
    f = situ_glu(g, u, B1, B2)
    assert f.dtype == dtype and torch.isfinite(f).all()
    assert f.abs().max().item() <= B1 * B2
    assert situ_gate(g.float(), B1).abs().max().item() <= B1
    assert soft_cap(u.float(), B2).abs().max().item() <= B2


def test_swiglu_closeness_bound_holds_everywhere() -> None:
    g, u = _sweep(torch.float64, seed=1)
    g, u = g / 100, u / 100  # keep plenty of points where the bound is tight
    swiglu = F.silu(g) * u
    err = (situ_glu(g, u, B1, B2) - swiglu).abs()
    bound = torch.sigmoid(g) * (
        g.abs() ** 3 * u.abs() / (3 * B1**2) + g.abs() * u.abs() ** 3 / (3 * B2**2)
    )
    rounding = 8 * torch.finfo(torch.float64).eps * swiglu.abs()  # the subtraction's own noise
    assert (err <= bound * (1 + 1e-12) + rounding).all()
    small = (g.abs() < 1e-2) & (u.abs() < 1e-2)
    assert small.any() and (err[small] <= 1e-9).all()  # fourth order in the joint scale


def test_sigmoid_reads_the_uncapped_gate() -> None:
    g = torch.tensor(6.0, dtype=torch.float64)
    expected = B1 * math.tanh(1.5) / (1 + math.exp(-6.0))
    capped_sigmoid = B1 * math.tanh(1.5) / (1 + math.exp(-B1 * math.tanh(1.5)))
    assert situ_gate(g, B1).item() == pytest.approx(expected, rel=1e-15)
    assert abs(expected - capped_sigmoid) > 1e-2  # the wrong wiring is visible here


def test_fp32_saturation_is_exact() -> None:
    big = torch.tensor([1e3, -1e3], dtype=torch.float32)
    assert torch.equal(soft_cap(big, B1), torch.tensor([B1, -B1]))
    assert torch.equal(soft_cap(big, B2), torch.tensor([B2, -B2]))
    f = situ_glu(big[:1], big[:1], B1, B2)
    assert f.item() == B1 * B2


@pytest.mark.parametrize("g_value", [-50.0, -100.0, -1e4])
def test_negative_tail_goes_to_zero(g_value: float) -> None:
    u = torch.tensor([-1e4, -3.0, 0.5, 1e4], dtype=torch.float64)
    g = torch.full_like(u, g_value)
    f = situ_glu(g, u, B1, B2)
    assert (f.abs() <= B1 * B2 * math.exp(g_value)).all()


def test_large_beta_recovers_swiglu() -> None:
    g, u = torch.randn(2, 4096, dtype=torch.float64) * 5
    torch.testing.assert_close(situ_glu(g, u, 1e9, 1e9), F.silu(g) * u, rtol=1e-12, atol=0)
    g32, u32 = g.float(), u.float()
    torch.testing.assert_close(situ_glu(g32, u32, 1e9, 1e9), F.silu(g32) * u32)


def test_autograd_matches_hand_product_rule() -> None:
    g = (torch.randn(512, dtype=torch.float64) * 6).requires_grad_()
    u = (torch.randn(512, dtype=torch.float64) * 40).requires_grad_()
    situ_glu(g, u, B1, B2).sum().backward()
    with torch.no_grad():
        t, s = torch.tanh(g / B1), torch.sigmoid(g)
        gate = B1 * t * s
        d_gate = (1 - t**2) * s + B1 * t * s * (1 - s)
        up = B2 * torch.tanh(u / B2)
        d_up = 1 - torch.tanh(u / B2) ** 2
    assert g.grad is not None and u.grad is not None
    torch.testing.assert_close(g.grad, d_gate * up, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(u.grad, gate * d_up, rtol=1e-12, atol=1e-12)


def test_gradcheck_and_finite_saturated_grads() -> None:
    g = torch.randn(16, dtype=torch.float64, requires_grad=True)
    u = torch.randn(16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: situ_glu(a, b, B1, B2), (g, u))
    g32, u32 = _sweep(torch.float32, n=1000, seed=2)
    g32.requires_grad_()
    u32.requires_grad_()
    situ_glu(g32, u32, B1, B2).sum().backward()
    assert g32.grad is not None and u32.grad is not None
    assert torch.isfinite(g32.grad).all() and torch.isfinite(u32.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bitwise_equal_to_hf_situ_and_mul(dtype: torch.dtype) -> None:
    x = (torch.randn(64, 2 * 96) * 20).to(dtype)
    d = x.shape[-1] // 2
    ours = SiTUGLU(SiTUConfig())(x[..., :d], x[..., d:])
    assert torch.equal(ours, _hf_situ_and_mul(x, B1, B2))


def test_config_wiring() -> None:
    cfg = mini_k3_d12()
    act = SiTUGLU(cfg)
    assert (act.beta_gate, act.beta_up) == (cfg.situ_beta_gate, cfg.situ_beta_up) == (B1, B2)
    assert SiTUConfig().bound == 100.0


def test_dense_mlp_matches_hf_kimi_mlp_and_param_count() -> None:
    torch.manual_seed(0)
    mlp = DenseSiTUMLP(32, 48, SiTUConfig()).double()
    x = torch.randn(2, 5, 32, dtype=torch.float64)
    gate_up = torch.cat([mlp.gate_proj(x), mlp.up_proj(x)], dim=-1)
    d = gate_up.shape[-1] // 2
    act = situ_gate(gate_up[..., :d], B1) * soft_cap(gate_up[..., d:], B2)  # HF order, fp64
    torch.testing.assert_close(mlp(x), mlp.down_proj(act), rtol=1e-12, atol=1e-12)
    assert [n for n, _ in mlp.named_parameters()] == [
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    ]
    for cfg in (mini_k3_d12(), k3_full()):
        with torch.device("meta"):
            dense = DenseSiTUMLP(cfg.hidden_size, cfg.moe.dense_intermediate, cfg)
        live = sum(p.numel() for p in dense.parameters())
        assert live == _glu_mlp_params(cfg.hidden_size, cfg.moe.dense_intermediate)
