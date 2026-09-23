"""K3 trainer-wiring gates (spec §7): a K3Model through the real ``train()``, its checkpoints, and
the speedrun family switch.

1. A tiny K3 learns a structured corpus through ``train()`` with ``optimizer="k3_muon"``: every
   logged loss is finite, the loss falls tenfold, and Quantile Balancing moves every router bias.
2. One learning rate for both optimizer halves: at every step, warmup and cosine, every Per-Head
   Muon group and the AdamW group carry exactly the scheduled lr; ``TrainConfig``'s weight decay,
   betas and momentum reach their groups.
3. Two ``train()`` steps ≡ the same two steps composed by hand from the pieces ``train()``'s
   docstring names (scheduled lr, forward with aux, CE + aux, backward, global grad clip, step,
   QK-Clip, Quantile Balancing), bitwise: every loss, weight and router bias.
4. QK-Clip through ``train()`` on two MLA layers: one step with the clip against the same step
   without it. Heads whose max logit is at or under τ, and everything outside the MLA q/kv
   up-projections, are bitwise equal; a head over τ is exactly the unclipped step rescaled (so
   the clip runs after the step, with ``TrainConfig.qk_clip_tau``), in every MLA layer.
5. The optimizer kinds that do not fit the model are refused before any work.
6. save → ``build_k3_from_checkpoint``: the frozen config comes back equal field for field (nested
   configs, tuples, a vision config), every parameter and the logits bitwise; the payload loads
   with ``weights_only=True``; a missing or an unexpected key raises (the strict load); each
   builder refuses the other family's checkpoint.
7. ``SpeedrunConfig.model_family="k3_mini"``: ``stage_pretrain`` builds the preset (patched to the
   tiny config here) with the call's vocab and context, trains it with ``k3_muon`` by default,
   writes a boundary checkpoint that ``resume`` rebuilds; ``k3_config`` replaces the preset; an
   unknown family, or a ``k3_config`` without the K3 family, is refused. The dense default still
   trains with ``muon_adamw``. The whole ``run_speedrun`` chain on K3 is tests/test_k3_e2e.py.

Tiny config: 4 layers at period 4 (KDA 1-3, MLA 4; 8 layers give MLA 4 and 8), layer 1 dense and
the rest LatentMoE, B = 2, MLA rope 4 (live) and 4 heads so that τ can split them. ~80 ms per
training step on CPU at 4 layers.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

import scratch_llm.speedrun as speedrun
from scratch_llm.k3.config import (
    K3Config,
    KDAConfig,
    MLAConfig,
    MoEConfig,
    VisionConfig,
    build_layer_pattern,
)
from scratch_llm.k3.core.gated_mla import GatedMLA
from scratch_llm.k3.core.latent_moe import MoEGate
from scratch_llm.k3.model import K3Model
from scratch_llm.k3.muon import PerHeadMuon, apply_k3_qk_clip, build_k3_optimizer
from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy
from scratch_llm.optim import AdamW, CombinedOptimizer, cosine_lr, gradient_clipping
from scratch_llm.speedrun import SpeedrunConfig, stage_pretrain
from scratch_llm.train import (
    TrainConfig,
    build_k3_from_checkpoint,
    build_model_from_checkpoint,
    save_checkpoint,
    train,
)

VOCAB = 32
CORPUS = np.tile(np.arange(16, dtype=np.int64), 400)  # a period-16 cycle over 16 of the 32 ids


def _tiny_cfg(num_layers: int = 4, **overrides: object) -> K3Config:
    kda_layers, mla_layers = build_layer_pattern(num_layers, period=4)
    cfg = K3Config(
        hidden_size=32,
        num_layers=num_layers,
        kda_layers=kda_layers,
        mla_layers=mla_layers,
        vocab_size=VOCAB,
        kda=KDAConfig(num_heads=2, head_dim=8, decay_rank=4, a_log_size=4),
        mla=MLAConfig(
            num_heads=4,
            q_lora_rank=12,
            kv_lora_rank=10,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=6,
        ),
        moe=MoEConfig(
            num_experts=8,
            top_k=2,
            num_shared_experts=1,
            expert_intermediate=12,
            latent_size=16,
            dense_intermediate=48,
        ),
        attn_res_block_size=2,
        max_position_embeddings=16,
        rms_norm_eps=2e-5,
    )
    return replace(cfg, **overrides)


def _mlas(model: K3Model) -> list[GatedMLA]:
    return [m for m in model.modules() if isinstance(m, GatedMLA)]


def _router_biases(model: K3Model) -> dict[str, Tensor]:
    return {
        n: m.e_score_correction_bias for n, m in model.named_modules() if isinstance(m, MoEGate)
    }


def _fixed_batch(seed: int, batch: int = 4, seq: int = 12) -> tuple[Tensor, Tensor]:
    starts = np.random.default_rng(seed).integers(0, len(CORPUS) - seq - 1, size=batch)
    ids = torch.from_numpy(np.stack([CORPUS[s : s + seq + 1] for s in starts]))
    return ids[:, :-1], ids[:, 1:]


# ---------------------------------------------------------------------------
# 1-2. Training through train().
# ---------------------------------------------------------------------------


def test_k3_learns_through_train() -> None:
    """The corpus is a cycle, so the next token is a function of the current one and the Bayes
    loss is 0. The gate is last ≤ first / 10: the first logged loss is the init's ≈ log 32 = 3.47,
    so the bar is ≈ 0.35 nats, far under log 16 = 2.77, the best a model that ignores context can
    reach here (the unigram marginal is uniform over 16 ids). Passing it takes the previous token.
    Measured on seeds 0-4: 3.44-3.48 → 0.040-0.047 (74-85×), under the bar from step 18-19 of
    40; ~3.5 s."""
    torch.manual_seed(0)
    model = K3Model(_tiny_cfg())
    start = {name: b.clone() for name, b in _router_biases(model).items()}
    history = train(
        TrainConfig(
            max_steps=40,
            batch_size=8,
            context_length=12,
            max_lr=1e-2,
            min_lr=1e-3,
            warmup_steps=4,
            optimizer="k3_muon",
            log_every=1,  # the non-finite guard runs on every step
        ),
        CORPUS,
        model,
    )
    losses = [loss for _, loss in history]
    assert len(losses) == 40 and all(math.isfinite(loss) for loss in losses)
    assert losses[-1] <= losses[0] / 10, f"{losses[0]:.3f} -> {losses[-1]:.3f}"
    for name, bias in _router_biases(model).items():  # QB ran after every step
        assert not torch.equal(bias, start[name]), name


def test_one_learning_rate_for_both_optimizer_halves() -> None:
    """build_k3_optimizer defaults to Muon 2e-2 / AdamW 3e-3; train() writes the one scheduled lr
    into every group each step, before the step, so neither default ever updates a weight. The
    eval hook runs after each step and reads the lr that step used. Weight decay, betas and
    momentum are off build_k3_optimizer's defaults, so a dropped argument shows."""
    tcfg = TrainConfig(
        max_steps=8,
        batch_size=4,
        context_length=12,
        max_lr=1e-2,
        min_lr=1e-3,
        warmup_steps=3,
        weight_decay=0.05,
        betas=(0.8, 0.99),
        muon_momentum=0.9,
        optimizer="k3_muon",
        eval_every=1,
        eval_batches=1,
    )
    torch.manual_seed(0)
    model = K3Model(_tiny_cfg())
    optimizers: list[torch.optim.Optimizer | CombinedOptimizer] = []
    seen: list[tuple[int, set[object]]] = []

    def record(step: int, _val_ce: float) -> None:
        seen.append((step, {g["lr"] for g in optimizers[0].param_groups}))

    train(tcfg, CORPUS, model, val_data=CORPUS, eval_hook=record, optimizer_out=optimizers)

    opt = optimizers[0]
    assert isinstance(opt, CombinedOptimizer)
    muon, adamw = opt.optimizers
    assert isinstance(muon, PerHeadMuon) and isinstance(adamw, AdamW)
    assert len(muon.param_groups) > 1  # per-head blocks and whole matrices
    assert {g["weight_decay"] for g in opt.param_groups} == {0.05}
    assert {g["momentum"] for g in muon.param_groups} == {0.9}
    assert {g["betas"] for g in adamw.param_groups} == {(0.8, 0.99)}
    assert [step for step, _ in seen] == list(range(tcfg.max_steps))
    for step, lrs in seen:
        expected = cosine_lr(step, tcfg.max_lr, tcfg.min_lr, tcfg.warmup_steps, tcfg.max_steps)
        assert lrs == {expected}, (step, lrs)


# ---------------------------------------------------------------------------
# 3. train() ≡ the step composed by hand.
# ---------------------------------------------------------------------------


def test_train_steps_equal_the_hand_composed_steps() -> None:
    """The hand loop asserts that each optional piece acts on both steps, else the comparison
    could not see it go missing: the global grad norm is over ``grad_clip`` (≈ 4 against 1.0 on
    this batch), and QK-Clip rescales some head (τ = 0.015 sits inside the init's S_h range
    0.010-0.022 here). No warmup, so step 0's lr is max_lr and not 0; two steps, because
    AdamW's bias correction makes its first update independent of the betas up to eps."""
    tcfg = TrainConfig(
        max_steps=2,
        batch_size=4,
        context_length=12,
        max_lr=1e-2,
        min_lr=1e-3,
        weight_decay=0.05,
        betas=(0.8, 0.99),
        muon_momentum=0.9,
        optimizer="k3_muon",
        qk_clip=True,
        qk_clip_tau=0.015,
        log_every=1,
    )
    inputs, targets = _fixed_batch(seed=1)
    torch.manual_seed(0)
    model = K3Model(_tiny_cfg())
    by_hand = copy.deepcopy(model)
    history = train(tcfg, CORPUS, model, batch_fn=lambda _step: (inputs, targets))

    for mla in _mlas(by_hand):
        mla.track_max_logits = True
    opt = build_k3_optimizer(  # its lr defaults never act: the schedule overwrites them first
        by_hand, weight_decay=tcfg.weight_decay, betas=tcfg.betas, momentum=tcfg.muon_momentum
    )
    losses: list[float] = []
    for step in range(tcfg.max_steps):
        lr = cosine_lr(step, tcfg.max_lr, tcfg.min_lr, tcfg.warmup_steps, tcfg.max_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad()
        logits, aux = by_hand(inputs, return_aux=True)
        loss = cross_entropy(logits, targets) + aux.total
        loss.backward()
        assert gradient_clipping(by_hand.parameters(), tcfg.grad_clip) > tcfg.grad_clip
        opt.step()
        assert min(apply_k3_qk_clip(by_hand, tcfg.qk_clip_tau).values()) < 1.0
        by_hand.moe_update_biases()
        losses.append(loss.item())

    assert history == list(enumerate(losses))
    for (name, p), q in zip(model.named_parameters(), by_hand.parameters(), strict=True):
        assert torch.equal(p, q), name  # the router biases included: they are parameters


# ---------------------------------------------------------------------------
# 4. QK-Clip through train().
# ---------------------------------------------------------------------------


def test_qk_clip_through_train_rescales_only_heads_over_tau() -> None:
    """Two MLA layers, so the observer wiring has to reach more than the first. Run the plain
    step first with every observer on by hand (it changes no numerics, and without the clip
    nothing consumes the record): those are the S_h the clipped run's identical forward will
    see. τ then goes between the largest per-layer minimum and the smallest per-layer maximum of
    S_h, so every layer has heads on both sides of it."""
    cfg = _tiny_cfg(num_layers=8)
    assert cfg.mla_layers == (4, 8)
    torch.manual_seed(0)
    base = K3Model(cfg)
    batch = _fixed_batch(seed=1)

    def one_step(model: K3Model, qk_clip: bool, tau: float) -> None:
        tcfg = TrainConfig(
            max_steps=1,
            batch_size=4,
            context_length=12,
            max_lr=1e-2,
            optimizer="k3_muon",
            qk_clip=qk_clip,
            qk_clip_tau=tau,
        )
        train(tcfg, CORPUS, model, batch_fn=lambda _step: batch)

    plain = copy.deepcopy(base)
    for mla in _mlas(plain):
        mla.track_max_logits = True
    one_step(plain, qk_clip=False, tau=100.0)
    records = [mla.last_max_logits for mla in _mlas(plain)]
    s_max = torch.stack([s for s in records if s is not None])  # [MLA layers, heads]
    assert s_max.shape[0] == len(cfg.mla_layers) and bool((s_max > 0).all())
    lo, hi = s_max.amin(1).max().item(), s_max.amax(1).min().item()
    assert lo < hi, (lo, hi)
    tau = (lo + hi) / 2
    clip = copy.deepcopy(base)
    one_step(clip, qk_clip=True, tau=tau)

    # train() turned every observer on, and the clip consumed every record.
    assert all(mla.track_max_logits and mla.last_max_logits is None for mla in _mlas(clip))
    touched = {
        f"layers.{i - 1}.self_attn.{proj}.weight"
        for i in cfg.mla_layers
        for proj in ("q_b_proj", "kv_b_proj")
    }
    reference = dict(plain.named_parameters())
    for name, p in clip.named_parameters():  # the QB biases included: same routing statistics
        if name not in touched:
            assert torch.equal(p, reference[name]), name

    for mla, mla_ref, s in zip(_mlas(clip), _mlas(plain), s_max, strict=True):
        clipped = s > tau
        assert clipped.any() and not clipped.all()  # τ's choice: both sides in every layer
        m = mla.cfg
        H, nope = m.num_heads, m.qk_nope_head_dim
        q = mla.q_b_proj.weight.detach().view(H, m.q_head_dim, -1)
        q_ref = mla_ref.q_b_proj.weight.detach().view(H, m.q_head_dim, -1)
        kv = mla.kv_b_proj.weight.detach().view(H, nope + m.v_head_dim, -1)
        kv_ref = mla_ref.kv_b_proj.weight.detach().view(H, nope + m.v_head_dim, -1)
        assert torch.equal(q[~clipped], q_ref[~clipped])
        assert torch.equal(kv[~clipped], kv_ref[~clipped])
        assert not torch.equal(q[clipped], q_ref[clipped])
        # A clipped head is the plain step's weights times apply_k3_qk_clip's factors (α = 0.5),
        # in its fp32 op order: γ = τ / max(S, τ), which is exactly 1.0 for the heads under τ.
        gamma = (tau / s.clamp_min(tau)).view(H, 1, 1)
        assert torch.equal(q[:, :nope], q_ref[:, :nope] * gamma.pow(0.5))
        assert torch.equal(q[:, nope:], q_ref[:, nope:] * gamma)
        assert torch.equal(kv[:, :nope], kv_ref[:, :nope] * gamma.pow(0.5))
        assert torch.equal(kv[:, nope:], kv_ref[:, nope:])  # v rows never scale


# ---------------------------------------------------------------------------
# 5. Refusals.
# ---------------------------------------------------------------------------


def test_optimizer_kinds_that_do_not_fit_the_model_are_refused() -> None:
    tcfg = TrainConfig(max_steps=1, batch_size=2, context_length=8)
    dense = TransformerLM(ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=1, n_heads=2))
    with pytest.raises(ValueError, match="needs a K3Model"):
        train(replace(tcfg, optimizer="k3_muon"), CORPUS, dense)
    k3 = K3Model(_tiny_cfg())
    with pytest.raises(ValueError, match="muon_adamw"):
        train(replace(tcfg, optimizer="muon_adamw"), CORPUS, k3)
    with pytest.raises(ValueError, match="muon_profile_ns"):
        train(replace(tcfg, optimizer="k3_muon", muon_profile_ns=True), CORPUS, k3)


# ---------------------------------------------------------------------------
# 6. Checkpoints.
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_rebuilds_the_same_k3(tmp_path: Path) -> None:
    """Off-init (three training steps, so the QB biases are nonzero too) and with the fields a
    shallow rebuild would drop: a nested VisionConfig with a tuple, a non-default eps, ``extra``."""
    cfg = _tiny_cfg(vision=VisionConfig(merge_kernel=(2, 3)), extra={"note": 0.5})
    torch.manual_seed(0)
    model = K3Model(cfg)
    train(
        TrainConfig(max_steps=3, batch_size=4, context_length=12, optimizer="k3_muon"),
        CORPUS,
        model,
    )
    assert all(b.any() for b in _router_biases(model).values())
    path = tmp_path / "k3.pt"
    save_checkpoint(model, None, step=3, out=path)

    rebuilt, step = build_k3_from_checkpoint(path)
    assert step == 3
    assert rebuilt.cfg == cfg  # frozen-dataclass equality, nested: a (2, 3) is no [2, 3]
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), rebuilt.named_parameters(), strict=True
    ):
        assert n1 == n2 and torch.equal(p1, p2), n1
    ids = _fixed_batch(seed=2)[0]
    model.eval()
    rebuilt.eval()
    with torch.no_grad():
        assert torch.equal(model(ids), rebuilt(ids))

    payload = torch.load(path, weights_only=True)["model_config"]
    assert payload["family"] == "k3"
    assert payload == torch.load(path, weights_only=False)["model_config"]

    # The strict load is the kill-switch: a renamed or dropped module never loads as fresh init.
    ckpt = torch.load(path, weights_only=False)
    weights = ckpt["model"]
    edits = {
        "missing": ({k: v for k, v in weights.items() if k != "norm.weight"}, r"Missing key.*norm"),
        "extra": ({**weights, "layers.0.bogus.weight": torch.zeros(1)}, r"Unexpected key.*bogus"),
    }
    for kind, (edited, match) in edits.items():
        torch.save({**ckpt, "model": edited}, tmp_path / f"{kind}.pt")
        with pytest.raises(RuntimeError, match=match):
            build_k3_from_checkpoint(tmp_path / f"{kind}.pt")

    with pytest.raises(ValueError, match="build_k3_from_checkpoint"):
        build_model_from_checkpoint(path)
    dense_path = tmp_path / "dense.pt"
    save_checkpoint(
        TransformerLM(ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=1, n_heads=2)),
        None,
        0,
        dense_path,
    )
    with pytest.raises(ValueError, match="build_model_from_checkpoint"):
        build_k3_from_checkpoint(dense_path)


# ---------------------------------------------------------------------------
# 7. The speedrun family switch.
# ---------------------------------------------------------------------------


def test_speedrun_k3_mini_family_pretrains_a_k3model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stage_pretrain looks the preset up by name, so the patch swaps the 370M mini_k3_d12 for
    the tiny config; vocab and context still come from the call, as in a real run."""
    monkeypatch.setattr(speedrun, "mini_k3_d12", _tiny_cfg)
    cfg = SpeedrunConfig(
        model_family="k3_mini",
        context_length=12,
        train_steps=4,
        batch_size=4,
        lr=1e-2,
        work_dir=str(tmp_path),
        checkpoint_every=2,
    )
    model, name = stage_pretrain(cfg, CORPUS, vocab_size=24)
    assert name == "pretrain" and isinstance(model, K3Model)
    assert model.cfg == replace(_tiny_cfg(), vocab_size=24, max_position_embeddings=12)

    snapshot = torch.load(tmp_path / "pretrain_ckpt.pt", weights_only=True)  # k3_muon by default
    assert snapshot["step"] == 2
    muon_groups = snapshot["optim"]["optimizers"][0]["param_groups"]
    assert all("head_rows" in group for group in muon_groups)  # PerHeadMuon's state, not Muon's

    boundary, step = build_k3_from_checkpoint(tmp_path / "pretrain.pt")
    resumed, resumed_name = stage_pretrain(replace(cfg, resume=True), CORPUS, vocab_size=24)
    assert step == cfg.train_steps and resumed_name == "pretrain[resumed]"
    for (n, p), q, r in zip(
        model.named_parameters(), boundary.parameters(), resumed.parameters(), strict=True
    ):
        assert torch.equal(p, q) and torch.equal(p, r), n

    # k3_config wins over the (patched) preset; vocab and context still come from the call.
    eight = _tiny_cfg(num_layers=8)
    own_cfg = replace(cfg, k3_config=eight, train_steps=1, work_dir=None, checkpoint_every=0)
    own, _ = stage_pretrain(own_cfg, CORPUS, vocab_size=24)
    assert isinstance(own, K3Model)
    assert own.cfg == replace(eight, vocab_size=24, max_position_embeddings=12)

    with pytest.raises(ValueError, match="model_family"):
        stage_pretrain(replace(cfg, model_family="k3"), CORPUS, vocab_size=24)
    with pytest.raises(ValueError, match="k3_config"):
        replace(cfg, model_family="dense", k3_config=eight)


def test_speedrun_dense_default_still_trains_with_muon_adamw(tmp_path: Path) -> None:
    cfg = SpeedrunConfig(
        depth=1,
        context_length=12,
        train_steps=2,
        batch_size=2,
        work_dir=str(tmp_path),
        checkpoint_every=1,
    )
    assert cfg.model_family == "dense" and cfg.optimizer is None
    model, _ = stage_pretrain(cfg, CORPUS, vocab_size=VOCAB)
    assert isinstance(model, TransformerLM)
    snapshot = torch.load(tmp_path / "pretrain_ckpt.pt", weights_only=False)
    muon_state, adamw_state = snapshot["optim"]["optimizers"]  # a CombinedOptimizer of two
    assert all("rms_scale" in g and "head_rows" not in g for g in muon_state["param_groups"])
    assert "betas" in adamw_state["param_groups"][0]
