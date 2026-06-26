"""Discipline tests for the DeepSeek-V3-style MoE FFN (A1.1).

The two lead anchors are correctness oracles for the through-line:

- **Dense-equivalence** — a 1-expert top-1 MoE *is* a SwiGLU. Guards the double-residual bug
  (the FFN must return the delta only) and proves gate-normalization sums to 1.
- **Decode/cache parity** — full-forward logits == incremental KV-cache decode logits with MoE
  active. This is the repo's headline failure mode (``kl_train_infer``); routing is per-token so
  it must hold exactly.

The rest pin the routing math (sigmoid + Top-K + normalize-over-selected), the aux-loss-free
bias (selection-only), the balancer's update direction, and the falsifiable entropy prediction.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from scratch_llm.model import KVCache, ModelConfig, SwiGLU, TransformerLM, cross_entropy
from scratch_llm.moe import MoEConfig, MoEFeedForward, Router, _default_expert_ffn
from scratch_llm.optim import AdamW
from scratch_llm.train import TrainConfig, load_checkpoint, save_checkpoint, train


def _moe_routers(model: TransformerLM) -> list[Router]:
    """The Router of every MoE layer (skips dense layers)."""
    return [b.ffn.router for b in model.blocks if getattr(b, "is_moe", False)]  # type: ignore[union-attr]


def _moe_cfg(**moe_overrides: object) -> ModelConfig:
    """A tiny GQA model whose every layer is MoE (n_dense_layers=0 by default)."""
    moe_base: dict[str, object] = dict(
        n_routed_experts=8,
        n_experts_per_tok=2,
        n_shared_experts=1,
    )
    moe_base.update(moe_overrides)
    return ModelConfig(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        context_length=64,
        moe=MoEConfig(**moe_base),  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- anchors


def test_dense_equivalence_single_expert() -> None:
    """1 routed expert, top-1, no shared, scaling 1 ⇒ MoE FFN == its single SwiGLU expert.
    The normalized gate over one selected expert is exactly 1, and the module returns the
    delta only — so `x + moe(x)` would match `x + swiglu(x)`."""
    torch.manual_seed(0)
    d = 64
    moe = MoEFeedForward(d, MoEConfig(n_routed_experts=1, n_experts_per_tok=1, n_shared_experts=0))
    ref = SwiGLU(d, _default_expert_ffn(d))
    ref.load_state_dict(moe.routed_experts[0].state_dict())

    x = torch.randn(2, 5, d)
    out, _ = moe(x)
    torch.testing.assert_close(out, ref(x))


def test_decode_cache_parity_with_moe() -> None:
    """Full forward over S tokens == incremental one-token-at-a-time decode with a KVCache.
    The MoE through-line invariant: train/infer must produce identical logits."""
    torch.manual_seed(0)
    cfg = _moe_cfg()
    model = TransformerLM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        full = model(ids)
        cache = KVCache(cfg.n_layers)
        steps = [model(ids[:, t : t + 1], cache=cache) for t in range(ids.shape[1])]
        incremental = torch.cat(steps, dim=1)  # type: ignore[arg-type]

    torch.testing.assert_close(full, incremental)


# --------------------------------------------------------------- disciplines carried over


def test_loss_at_init_is_log_vocab_moe() -> None:
    """Predicted first: shared+routed, normalized gates, scaling 1 ⇒ loss stays in the same
    ±0.3 band as dense (RMSNorm before the head renormalizes regardless of FFN sub-count)."""
    torch.manual_seed(0)
    cfg = _moe_cfg()
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (8, 32))
    targets = torch.randint(0, cfg.vocab_size, (8, 32))
    logits, _ = model(ids, return_aux=True)
    loss = cross_entropy(logits, targets).item()  # type: ignore[arg-type]
    expected = math.log(cfg.vocab_size)
    assert abs(loss - expected) < 0.3, f"loss {loss:.3f} vs log V {expected:.3f}"


def test_overfit_one_batch_moe() -> None:
    """The sparse FFN + router must drive a single batch's CE to ~0 (optimizer/data/loss wiring).
    Regularizers off (z=α=0) to isolate the wiring; bias balancing runs each step as in training."""
    torch.manual_seed(0)
    cfg = _moe_cfg(z_loss_coef=0.0, aux_loss_alpha=0.0)
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (4, 16))
    targets = torch.randint(0, cfg.vocab_size, (4, 16))
    opt = AdamW(model.parameters(), lr=2e-3)

    final = float("inf")
    for _ in range(400):
        opt.zero_grad()
        logits, aux = model(ids, return_aux=True)
        loss = cross_entropy(logits, targets) + aux.total
        loss.backward()
        opt.step()
        model.moe_update_biases()
        final = loss.item()
    assert final < 0.05, f"failed to overfit: final loss {final:.4f}"


def test_causal_no_leak_with_moe() -> None:
    """The autoregressive contract still holds with MoE (routing is per-token)."""
    torch.manual_seed(0)
    cfg = _moe_cfg()
    model = TransformerLM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 20))
    cut = 10
    with torch.no_grad():
        base = model(ids)
        perturbed = ids.clone()
        perturbed[:, cut + 1 :] = (perturbed[:, cut + 1 :] + 7) % cfg.vocab_size
        after = model(perturbed)
    torch.testing.assert_close(base[:, : cut + 1], after[:, : cut + 1])  # type: ignore[index]
    assert not torch.allclose(base[:, cut + 1 :], after[:, cut + 1 :])  # type: ignore[index]


# ----------------------------------------------------------------------- routing math


class _ConstExpert(nn.Module):
    """An 'expert' that ignores its input and returns a fixed vector — lets a test read off the
    exact gate weights the router applied."""

    def __init__(self, vec: Tensor) -> None:
        super().__init__()
        self.vec = vec

    def forward(self, x: Tensor) -> Tensor:
        return self.vec.expand(x.shape[0], -1)


def test_gates_sum_to_one_identical_experts() -> None:
    """If every routed expert computes the same f, the output is f(x) for any K — because the
    normalized gates over the selected experts sum to 1."""
    torch.manual_seed(0)
    d = 32
    moe = MoEFeedForward(d, MoEConfig(n_routed_experts=4, n_experts_per_tok=2, n_shared_experts=0))
    shared = moe.routed_experts[0]
    for e in moe.routed_experts:  # make all experts identical
        e.load_state_dict(shared.state_dict())
    x = torch.randn(3, 1, d)
    out, _ = moe(x)
    torch.testing.assert_close(out, shared(x))


def test_bias_steers_selection_not_value() -> None:
    """The aux-loss-free bias enters Top-K *selection* only; the gate *value* uses raw sigmoid."""
    torch.manual_seed(0)
    d = 16
    # (a) value ignores bias: n_routed=2, k=2 ⇒ both always selected; biasing must not move output.
    moe2 = MoEFeedForward(d, MoEConfig(n_routed_experts=2, n_experts_per_tok=2, n_shared_experts=0))
    e0, e1 = torch.randn(d), torch.randn(d)
    moe2.routed_experts = nn.ModuleList([_ConstExpert(e0), _ConstExpert(e1)])
    x = torch.randn(1, 1, d)
    with torch.no_grad():
        out_zero, _ = moe2(x)
        moe2.router.bias.copy_(torch.tensor([10.0, -10.0]))
        out_biased, _ = moe2(x)
    torch.testing.assert_close(out_zero, out_biased)  # value used raw s, not s+b

    # (b) selection obeys bias: n_routed=3, k=1 ⇒ a huge bias forces that expert's output.
    moe1 = MoEFeedForward(d, MoEConfig(n_routed_experts=3, n_experts_per_tok=1, n_shared_experts=0))
    consts = [torch.full((d,), float(i + 1)) for i in range(3)]
    moe1.routed_experts = nn.ModuleList(_ConstExpert(c) for c in consts)
    with torch.no_grad():
        moe1.router.bias.copy_(torch.tensor([0.0, 0.0, 50.0]))  # force expert 2
        out, _ = moe1(x)
    torch.testing.assert_close(out[0, 0], consts[2])  # top-1 normalized gate = 1


def test_bias_update_direction() -> None:
    """Overloaded experts get a lower bias, underloaded a higher one; the accumulator resets."""
    router = Router(d_model=8, n_routed_experts=4)
    router.load_count.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))  # expert 0 overloaded
    router.update_bias(speed=0.1)
    assert router.bias[0].item() < 0.0  # overloaded ⇒ decreased
    assert (router.bias[1:] > 0.0).all()  # underloaded ⇒ increased
    torch.testing.assert_close(router.load_count, torch.zeros(4))  # reset


# ------------------------------------------------------------------ diagnostics / losses


def test_router_entropy_balanced_at_init() -> None:
    """Falsifiable prediction (A1.1): a fresh router stays > 0.9·log(N_r) — i.e. it does not
    collapse onto a few experts. Also checks the entropy matches -Σ p·log p of the load."""
    torch.manual_seed(0)
    cfg = _moe_cfg(n_routed_experts=32, n_experts_per_tok=8)
    model = TransformerLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (8, 32))
    _, aux = model(ids, return_aux=True)
    stat = aux.layers[0]

    p = stat.load_fraction.clamp_min(1e-12)
    expected_entropy = -(p * p.log()).sum()
    torch.testing.assert_close(stat.entropy, expected_entropy)
    assert stat.entropy.item() > 0.9 * math.log(32), (
        f"router collapsed: H={stat.entropy.item():.3f} ≤ 0.9·log32={0.9 * math.log(32):.3f}"
    )


def test_aux_and_z_losses_disable_cleanly() -> None:
    """z=α=0 ⇒ both terms are exactly zero; positive coefs ⇒ finite non-negative scalars."""
    torch.manual_seed(0)
    ids = torch.randint(0, 512, (4, 16))

    off = TransformerLM(_moe_cfg(z_loss_coef=0.0, aux_loss_alpha=0.0))
    _, aux_off = off(ids, return_aux=True)
    assert aux_off.aux_loss.item() == 0.0 and aux_off.z_loss.item() == 0.0

    on = TransformerLM(_moe_cfg(z_loss_coef=1e-3, aux_loss_alpha=1e-4))
    _, aux_on = on(ids, return_aux=True)
    assert aux_on.z_loss.item() > 0.0
    assert aux_on.aux_loss.item() >= 0.0
    assert math.isfinite(aux_on.total.item())


def test_dense_default_path_unchanged() -> None:
    """No `moe` field ⇒ a plain dense model; `return_aux` yields empty, zero aux."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = TransformerLM(cfg)
    assert not any(getattr(b, "is_moe", False) for b in model.blocks)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, aux = model(ids, return_aux=True)
    assert logits.shape == (1, 8, cfg.vocab_size)  # type: ignore[union-attr]
    assert aux.aux_loss.item() == 0.0 and aux.layers == []


def test_n_dense_layers_keeps_leading_layers_dense() -> None:
    """`n_dense_layers=1` ⇒ layer 0 dense, layer 1 MoE."""
    cfg = _moe_cfg(n_dense_layers=1)
    model = TransformerLM(cfg)
    assert model.blocks[0].is_moe is False  # type: ignore[union-attr]
    assert model.blocks[1].is_moe is True  # type: ignore[union-attr]


def test_moe_config_rejects_bad_shapes() -> None:
    for bad in (
        dict(n_routed_experts=4, n_experts_per_tok=5),  # k > N_r
        dict(n_routed_experts=4, n_experts_per_tok=0),  # k < 1
        dict(n_routed_experts=0, n_experts_per_tok=1),  # N_r < 1
    ):
        try:
            MoEConfig(**bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# -------------------------------------------------------------- balancer dynamics + plumbing


def test_balancer_overcomes_any_routing_preference() -> None:
    """The discriminating balancer test (the entropy-at-init test cannot distinguish a working
    balancer from a disabled one). A persistently overloaded expert is pushed *below* the starved
    ones, and the bias gap grows past 1.0 — and since sigmoid affinities live in (0, 1), a gap > 1
    is guaranteed to flip Top-K selection. With ``bias_update_speed=0`` this gap would stay 0."""
    router = Router(d_model=8, n_routed_experts=4)
    for _ in range(40):
        router.load_count.copy_(torch.tensor([30.0, 0.0, 0.0, 0.0]))  # expert 0 always overloaded
        router.update_bias(speed=0.1)
    assert router.bias[0] < router.bias[1:].min()  # overloaded expert demoted below the rest
    gap = (router.bias[1:] - router.bias[0]).min().item()
    assert gap > 1.0, f"bias gap {gap:.3f} ≤ 1 — cannot overcome a sigmoid-affinity gap (<1)"


def test_train_loop_runs_with_moe() -> None:
    """Covers `train()`'s MoE branch (ce + aux.total) and the post-step `moe_update_biases()` —
    proven by the router biases moving off their zero init through the real training loop."""
    torch.manual_seed(0)
    cfg = _moe_cfg()
    model = TransformerLM(cfg)
    data = np.arange(2000, dtype=np.int64) % cfg.vocab_size  # structured, learnable ramp
    before = [r.bias.clone() for r in _moe_routers(model)]

    tcfg = TrainConfig(max_steps=20, batch_size=4, context_length=16, max_lr=2e-3, seed=0)
    history = train(tcfg, data, model)

    assert history and math.isfinite(history[-1][1])
    after = [r.bias for r in _moe_routers(model)]
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=False)), (
        "moe_update_biases never ran"
    )


def test_checkpoint_roundtrip_preserves_router_bias(tmp_path: Path) -> None:
    """The aux-loss-free bias is learned balancing state (persistent buffer) — it must survive a
    save/load round-trip, like every other parameter."""
    torch.manual_seed(0)
    cfg = _moe_cfg()
    model = TransformerLM(cfg)
    with torch.no_grad():  # distinctive, non-zero biases to round-trip
        for r in _moe_routers(model):
            r.bias.copy_(torch.randn_like(r.bias))
    opt = AdamW(model.parameters(), lr=1e-3)

    path = tmp_path / "moe.pt"
    save_checkpoint(model, opt, step=5, out=path)
    restored = TransformerLM(cfg)
    assert load_checkpoint(path, restored, optimizer=None) == 5
    for r0, r1 in zip(_moe_routers(model), _moe_routers(restored), strict=False):
        torch.testing.assert_close(r0.bias, r1.bias)
