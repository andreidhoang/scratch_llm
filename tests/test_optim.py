"""Tests for AdamW, gradient clipping, the cosine schedule, and the overfit-one-batch
discipline (the single highest-leverage check that the training wiring is correct)."""

import math

import pytest
import torch

from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy
from scratch_llm.optim import (
    AdamW,
    Muon,
    _zeropower_via_newtonschulz5,
    cosine_lr,
    gradient_clipping,
    split_muon_adamw_params,
)


def test_adamw_minimizes_quadratic() -> None:
    torch.manual_seed(0)
    x = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    target = torch.tensor([5.0, -3.0])
    opt = AdamW([x], lr=0.1, weight_decay=0.0)
    for _ in range(500):
        opt.zero_grad()
        loss = ((x - target) ** 2).sum()
        loss.backward()
        opt.step()
    torch.testing.assert_close(x.detach(), target, atol=1e-2, rtol=0)


def test_adamw_decoupled_weight_decay_shrinks_param() -> None:
    # With zero gradient, decoupled wd alone should shrink the parameter by (1 - lr*wd)/step.
    x = torch.nn.Parameter(torch.tensor([1.0]))
    opt = AdamW([x], lr=0.1, weight_decay=0.5)
    x.grad = torch.zeros_like(x)
    opt.step()
    # m=v=0 so the adam term is 0; only wd applies: 1 - lr*wd = 1 - 0.05 = 0.95
    torch.testing.assert_close(x.detach(), torch.tensor([0.95]), atol=1e-6, rtol=0)


def test_gradient_clipping_scales_when_over_and_noop_when_under() -> None:
    p_big = torch.nn.Parameter(torch.zeros(3))
    p_big.grad = torch.tensor([3.0, 4.0, 0.0])  # norm = 5
    norm = gradient_clipping([p_big], max_l2_norm=1.0)
    assert math.isclose(norm.item(), 5.0, rel_tol=1e-5)
    torch.testing.assert_close(p_big.grad.norm(), torch.tensor(1.0), atol=1e-4, rtol=0)

    p_small = torch.nn.Parameter(torch.zeros(2))
    p_small.grad = torch.tensor([0.3, 0.4])  # norm = 0.5 < 1.0 → unchanged
    gradient_clipping([p_small], max_l2_norm=1.0)
    torch.testing.assert_close(p_small.grad, torch.tensor([0.3, 0.4]))


@pytest.mark.parametrize(
    ("dtype", "grad", "expected"),
    [
        # fp16: squaring in the grad dtype overflows (3e3^2 = 9e6 > fp16 max 65504) -> the
        # global norm becomes inf, the scale becomes 0, and the step is silently ZEROED.
        (torch.float16, [3e3, 4e3], 5e3),
        # fp16 the other way: 1e-3^2 = 1e-6 < fp16 min normal (6.1e-5) -> underflows to a
        # subnormal and the norm collapses.
        (torch.float16, [3e-3, 4e-3], 5e-3),
        # fp64 is the guard against "fix it by casting to fp32": .float() would send this to
        # inf. The norm must be computed in AT LEAST fp32, never in LESS than the grad dtype.
        (torch.float64, [3e30, 4e30], 5e30),
        (torch.float64, [3e-25, 4e-25], 5e-25),
        (torch.bfloat16, [3.0, 4.0], 5.0),
    ],
)
def test_gradient_clipping_norm_is_computed_in_at_least_fp32(
    dtype: torch.dtype, grad: list[float], expected: float
) -> None:
    """The global norm must survive dtypes whose square overflows or underflows.

    A wrong norm here is the worst class of bug this repo screens for: nothing raises, the
    loss curve merely stops improving because every step was scaled by 0 (norm=inf) or left
    unclipped (norm=0). bf16 is included as the control -- it has fp32's exponent range, so
    it never triggered the bug and must not regress.
    """
    p = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
    p.grad = torch.tensor(grad, dtype=dtype)
    norm = gradient_clipping([p], max_l2_norm=float("inf"))
    assert torch.isfinite(norm), f"{dtype} norm is {norm.item()}"
    assert math.isclose(norm.item(), expected, rel_tol=1e-2), (
        f"{dtype}: {norm.item()} != {expected}"
    )


def test_cosine_schedule_phases() -> None:
    max_lr, min_lr, warmup, cosine = 1.0, 0.1, 10, 110
    assert cosine_lr(0, max_lr, min_lr, warmup, cosine) == 0.0
    assert math.isclose(cosine_lr(5, max_lr, min_lr, warmup, cosine), 0.5)  # mid-warmup
    assert math.isclose(cosine_lr(warmup, max_lr, min_lr, warmup, cosine), max_lr)
    # midpoint of the cosine phase → halfway between max and min
    mid = cosine_lr(warmup + (cosine - warmup) // 2, max_lr, min_lr, warmup, cosine)
    assert math.isclose(mid, (max_lr + min_lr) / 2, abs_tol=1e-6)
    assert math.isclose(cosine_lr(cosine, max_lr, min_lr, warmup, cosine), min_lr)
    assert cosine_lr(cosine + 50, max_lr, min_lr, warmup, cosine) == min_lr


def test_overfit_one_batch() -> None:
    """The wiring check: model + AdamW + cross_entropy must memorize a single batch."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, context_length=32)
    model = TransformerLM(cfg)
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    ids = torch.randint(0, cfg.vocab_size, (4, 17))
    inputs, targets = ids[:, :-1], ids[:, 1:]

    initial = cross_entropy(model(inputs), targets).item()
    final = initial
    for _ in range(300):
        opt.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        opt.step()
        final = loss.item()

    assert final < 0.05, f"failed to overfit: final loss {final:.4f} (started {initial:.3f})"


# --------------------------------------------------------------------------------------------
# Muon (F1) — Newton–Schulz orthogonality, the corrected RMS-match, the param split, overfit
# --------------------------------------------------------------------------------------------


def test_newton_schulz_orthogonalizes_singular_values() -> None:
    """F1 DoD (corrected by measurement — see bench/RESULTS.md §Frontier ablations).

    5-step Newton–Schulz *compresses* the singular-value spectrum into a bounded band (~[0.68,
    1.14]) — a BAND, not a delta at 1 (convergence to exactly 1 is asymptotic in the step count).
    Two pre-registered over-claims were FALSIFIED and are recorded honestly: (i) "*all* σ ∈ [0.7,1.3]
    for a 256×256 update" — a worst-case square Gaussian has near-zero σ (κ ~ n) that 5 steps cannot
    lift (measured min σ ≈ 0.08); (ii) "median σ ≈ 1" — measured median ≈ 0.77 at 5 steps. Both are
    inherent to few-step NS and immaterial to Muon: real momentum-gradients aren't worst-case, and
    Muon needs only *approximate* orthogonality (the update *direction*, uniform spectrum). The
    honest invariants below: never inflates; the bulk collapses into a tight band; the spread
    collapses vs the raw (ill-conditioned) input — the whole point of orthogonalization.
    """
    torch.manual_seed(0)
    for shape in [(128, 384), (256, 256), (384, 128)]:  # wide, square, tall (transpose path)
        g = torch.randn(*shape)
        s_in = torch.linalg.svdvals(
            g / (g.norm() + 1e-7)
        )  # normalized input spectrum (wide spread)
        o = _zeropower_via_newtonschulz5(g, steps=5)
        assert o.shape == shape
        s = torch.linalg.svdvals(o.float())
        # (1) never inflates past the quintic's fixed point.
        assert s.max() < 1.35, f"{shape}: NS inflated σ_max to {s.max():.3f} (>1.35)"
        # (2) the bulk (10th–90th pct) is compressed into a tight band ~[0.68,1.14]; the
        # ill-conditioned tail is excluded by construction and is immaterial to the update direction.
        q10 = torch.quantile(s, 0.10).item()
        q90 = torch.quantile(s, 0.90).item()
        assert q10 > 0.6 and q90 < 1.25, f"{shape}: bulk σ∈[{q10:.3f},{q90:.3f}] not compressed"
        # (3) the spread collapses vs the raw input (κ≫1) — the reason to orthogonalize at all.
        spread_in = torch.quantile(s_in, 0.90).item() / (torch.quantile(s_in, 0.10).item() + 1e-9)
        assert q90 / q10 < 2.0 < spread_in, f"{shape}: spread {q90 / q10:.2f} not collapsed"


def test_muon_update_rms_matches_adamw_band() -> None:
    """The corrected Moonlight identity (F1 honesty-ledger, [REFUTED→fixed]): an orthogonalized
    update on [A,B] has RMS 1/√max(A,B), so scaling by 0.2·√max(A,B) lands its RMS at ~0.2 —
    AdamW's band — *independent of shape*. (If the draft's 1/max(A,B) were right, this would be
    off by a √max factor.)"""
    torch.manual_seed(0)
    for shape in [(256, 256), (128, 512), (1024, 256)]:
        g = torch.randn(*shape)
        o = _zeropower_via_newtonschulz5(g, steps=5)
        scale = 0.2 * math.sqrt(max(shape))
        rms = (scale * o).pow(2).mean().sqrt().item()
        assert 0.15 < rms < 0.28, f"{shape}: scaled update RMS {rms:.3f} not ≈0.2"


def test_split_muon_adamw_params_routes_the_tied_tensor_to_adamw() -> None:
    """The repo-specific trap: with tie_embeddings the shared 2-D embed/head tensor MUST go to
    AdamW, not Muon. Also: disjoint cover, and RMSNorm(1-D)/head → AdamW, block matrices → Muon."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=48, d_model=32, n_layers=2, n_heads=4, tie_embeddings=True)
    model = TransformerLM(cfg)
    muon, adamw = split_muon_adamw_params(model)

    muon_ids, adamw_ids = {id(p) for p in muon}, {id(p) for p in adamw}
    assert muon_ids.isdisjoint(adamw_ids)  # no overlap
    # disjoint cover of every UNIQUE parameter (tied tensor counted once)
    unique = {id(p): p for p in model.parameters()}
    assert sum(p.numel() for p in muon) + sum(p.numel() for p in adamw) == sum(
        p.numel() for p in unique.values()
    )
    # the tied embed/head tensor is one object, and it is in the AdamW group (not Muon)
    assert model.token_emb.weight is model.lm_head.weight
    assert id(model.token_emb.weight) in adamw_ids and id(model.token_emb.weight) not in muon_ids
    # every Muon param is a 2-D block matrix; every 1-D param is on AdamW
    assert all(p.ndim == 2 for p in muon)
    assert id(model.final_norm.weight) in adamw_ids  # RMSNorm gain (1-D)
    assert id(model.blocks[0].attn.q_proj.weight) in muon_ids  # type: ignore[attr-defined]  # a block projection (2-D); ModuleList[i] loses the type


def test_split_muon_adamw_params_untied_keeps_both_embed_and_head_on_adamw() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=48, d_model=32, n_layers=1, n_heads=4, tie_embeddings=False)
    model = TransformerLM(cfg)
    muon, adamw = split_muon_adamw_params(model)
    adamw_ids = {id(p) for p in adamw}
    assert model.token_emb.weight is not model.lm_head.weight  # distinct tensors when untied
    assert id(model.token_emb.weight) in adamw_ids
    assert id(model.lm_head.weight) in adamw_ids  # head excluded from Muon even when 2-D & untied
    assert all(p.ndim == 2 for p in muon)


def test_overfit_one_batch_muon_hybrid() -> None:
    """Wiring check for the hybrid: Muon on the block matrices + AdamW on embed/head/norms must
    memorize a single batch, same discipline as the AdamW-only overfit test."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=4, context_length=32)
    model = TransformerLM(cfg)
    muon_params, adamw_params = split_muon_adamw_params(model)
    muon = Muon(muon_params, lr=2e-2, weight_decay=0.0)
    adamw = AdamW(adamw_params, lr=3e-3, weight_decay=0.0)

    ids = torch.randint(0, cfg.vocab_size, (4, 17))
    inputs, targets = ids[:, :-1], ids[:, 1:]

    initial = cross_entropy(model(inputs), targets).item()
    final = initial
    for _ in range(300):
        muon.zero_grad()
        adamw.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        muon.step()
        adamw.step()
        final = loss.item()

    assert final < 0.05, f"hybrid failed to overfit: final {final:.4f} (started {initial:.3f})"
