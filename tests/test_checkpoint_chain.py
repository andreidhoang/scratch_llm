"""A2 checkpoint chaining — config-carrying save/rebuild + the speedrun stage spine.

Pre-registered falsifiers (bench/RESULTS.md §Frontier, 2026-07-09):
- save → ``build_model_from_checkpoint`` rebuilds the exact model from the file alone
  (every param ``torch.equal``, step restored, config — incl. nested MoEConfig — round-trips).
- A rebuilt trained model is WARM (CE ≪ log V) while a fresh same-config model is COLD (≈ log V).
- ``Tokenizer.save/load`` round-trips a specials-bearing string; each special stays 1 id.
- ``run_speedrun(work_dir)`` re-run with ``resume=True`` rebuilds instead of retraining and
  reproduces the eval + sample byte-identically.
- Stage-boundary checkpoints carry NO optimizer state (the pinned stage-transition policy);
  loading one with an optimizer fails loud.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy
from scratch_llm.moe import MoEConfig
from scratch_llm.optim import AdamW
from scratch_llm.speedrun import (
    SpeedrunConfig,
    model_config_for_depth,
    run_speedrun,
    stage_midtrain,
    stage_sft,
)
from scratch_llm.tokenizer import Tokenizer, train_bpe
from scratch_llm.train import (
    TrainConfig,
    build_model_from_checkpoint,
    get_batch,
    load_checkpoint,
    save_checkpoint,
    train,
)

# ---------------------------------------------------------------------------------------------
# Config-carrying checkpoints (train.py)
# ---------------------------------------------------------------------------------------------


def test_checkpoint_carries_config_and_rebuilds_exactly(tmp_path: Path) -> None:
    """The kill criterion: the rebuilt model must be the SAME architecture, not a near-miss —
    non-default fields (GQA, qk_norm, rope_theta) must survive the round-trip."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        context_length=16,
        rope_theta=50000.0,
        qk_norm=True,
    )
    model = TransformerLM(cfg)
    opt = AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, step=42, out=path)

    rebuilt, step = build_model_from_checkpoint(path)
    assert step == 42
    assert rebuilt.cfg == cfg  # config round-trip, field for field
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), rebuilt.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_checkpoint_roundtrips_nested_moe_config(tmp_path: Path) -> None:
    """MoEConfig nests inside ModelConfig — the reconstruction trap asdict() sets up."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        context_length=16,
        moe=MoEConfig(n_routed_experts=4, n_experts_per_tok=2, n_dense_layers=1),
    )
    model = TransformerLM(cfg)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, None, step=3, out=path)

    rebuilt, step = build_model_from_checkpoint(path)
    assert step == 3
    assert rebuilt.cfg == cfg and rebuilt.cfg.moe == cfg.moe
    for (_, p1), (_, p2) in zip(model.named_parameters(), rebuilt.named_parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_rebuilt_model_is_warm_fresh_model_is_cold(tmp_path: Path) -> None:
    """The point of the safety-net: what comes back is the TRAINED model. Rebuilt CE ≪ log V;
    a fresh same-config model sits at ≈ log V (loss-at-init discipline)."""
    data = np.tile(np.arange(16, dtype=np.int64), 400)
    cfg = ModelConfig(vocab_size=32, d_model=64, n_layers=2, n_heads=4, context_length=16)
    model = TransformerLM(cfg)
    train(
        TrainConfig(
            max_steps=300, batch_size=16, context_length=12, max_lr=3e-3, warmup_steps=20, seed=0
        ),
        data,
        model,
    )
    path = tmp_path / "pretrain.pt"
    save_checkpoint(model, None, step=300, out=path)

    rebuilt, _ = build_model_from_checkpoint(path)
    fresh = TransformerLM(cfg)
    np.random.seed(0)
    inputs, targets = get_batch(data, batch_size=16, context_length=12)
    with torch.no_grad():
        warm = cross_entropy(rebuilt(inputs), targets).item()
        cold = cross_entropy(fresh(inputs), targets).item()
    log_v = math.log(cfg.vocab_size)
    # Pre-reg correction (bench/RESULTS.md): on a structured half-vocab batch the init CE sits
    # ABOVE log V (init-logit non-uniformity adds a positive excess — measured log V + 0.55), so
    # "≈ log V ± 15%" was falsified; the true invariant is one-sided — the fresh model has
    # learned NOTHING (CE ≥ log V − ε) while the rebuilt one is trained (≪ log V). The two-sided
    # ±0.3 check on uniform-random ids lives in test_model.py::test_loss_at_init_is_log_vocab.
    assert warm < 1.0, f"rebuilt model is cold: CE {warm:.3f} (trained to ≪ log V)"
    assert cold > log_v - 0.3, f"fresh model CE {cold:.3f} below log V {log_v:.3f} — learned?"
    assert warm < cold / 2, f"warm {warm:.3f} not ≪ cold {cold:.3f}"


def test_stage_boundary_checkpoint_has_no_optimizer_and_fails_loud(tmp_path: Path) -> None:
    """The pinned stage-transition policy: boundary snapshots are model+config+step only.
    Loading one WITH an optimizer must raise, never silently skip."""
    model = TransformerLM(ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2))
    path = tmp_path / "boundary.pt"
    save_checkpoint(model, None, step=10, out=path)

    fresh = TransformerLM(ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2))
    assert load_checkpoint(path, fresh) == 10  # optimizer-less load works
    with pytest.raises(ValueError, match="stage-boundary"):
        load_checkpoint(path, fresh, AdamW(fresh.parameters(), lr=1e-3))


def test_build_model_from_legacy_checkpoint_fails_loud(tmp_path: Path) -> None:
    """Pre-A2 checkpoints carry no config — rebuilding from one must be a clear error,
    not a guessed architecture."""
    model = TransformerLM(ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2))
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "optim": None, "step": 1}, path)
    with pytest.raises(ValueError, match="model_config"):
        build_model_from_checkpoint(path)


# ---------------------------------------------------------------------------------------------
# Tokenizer one-file serialization (tokenizer.py)
# ---------------------------------------------------------------------------------------------


def test_tokenizer_save_load_roundtrips_specials(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog. " * 20, encoding="utf-8")
    vocab, merges = train_bpe(str(corpus), vocab_size=300, special_tokens=["<|eot|>"])
    tok = Tokenizer(vocab, merges, special_tokens=["<|eot|>"])

    path = tok.save(tmp_path / "tokenizer.json")
    loaded = Tokenizer.load(path)

    assert loaded.vocab == tok.vocab
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    text = "the quick fox<|eot|>the lazy dog"
    ids = tok.encode(text)
    assert loaded.encode(text) == ids
    assert loaded.decode(ids) == text
    assert len(loaded.encode("<|eot|>")) == 1  # a special is exactly one id


# ---------------------------------------------------------------------------------------------
# Speedrun stage chaining (speedrun.py)
# ---------------------------------------------------------------------------------------------


def _nano_cfg(**overrides: object) -> SpeedrunConfig:
    base = SpeedrunConfig(
        depth=2,
        vocab_size=300,
        context_length=48,
        train_steps=40,
        batch_size=8,
        device="cpu",
        seed=0,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_speedrun_work_dir_writes_stage_artifacts(tmp_path: Path) -> None:
    cfg = _nano_cfg(work_dir=str(tmp_path))
    res = run_speedrun(cfg)
    assert res.stages == ["tokenizer", "pretrain", "eval", "sample"]
    assert (tmp_path / "tokenizer.json").exists()
    assert (tmp_path / "pretrain.pt").exists()

    rebuilt, step = build_model_from_checkpoint(tmp_path / "pretrain.pt")
    assert step == cfg.train_steps
    # stage_pretrain's device policy: qk_norm always on (F9 regime); SDPA only on cuda —
    # this CPU run trains with use_sdpa=False, qk_norm=True.
    expected = replace(model_config_for_depth(cfg.depth, 300, cfg.context_length), qk_norm=True)
    assert rebuilt.cfg == expected


def test_speedrun_resume_rebuilds_and_reproduces(tmp_path: Path) -> None:
    """The rental safety-net end to end: the resumed run must chain from the artifacts —
    no retrain — and land on the SAME model (identical eval + sample)."""
    cfg = _nano_cfg(work_dir=str(tmp_path))
    first = run_speedrun(cfg)
    resumed = run_speedrun(_nano_cfg(work_dir=str(tmp_path), resume=True))
    assert resumed.stages == ["tokenizer[resumed]", "pretrain[resumed]", "eval", "sample"]
    assert resumed.report_card.val_bpb == pytest.approx(first.report_card.val_bpb, rel=1e-6)
    assert resumed.sample == first.sample


def test_speedrun_midtrain_sft_slots_skip_at_zero_and_fail_loud_otherwise() -> None:
    from typing import cast

    model = TransformerLM(ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2))
    cfg = _nano_cfg()
    # Both slots skip at 0 steps (the SFT skip short-circuits before touching the tokenizer).
    assert stage_midtrain(cfg, model) is model
    assert stage_sft(cfg, model, cast(Tokenizer, None)) is model
    # midtrain is still the unbuilt A4 slot; SFT (A5) is now implemented and instead fails loud
    # when asked to run without the chat specials (the mask is defined by the special positions).
    with pytest.raises(NotImplementedError, match="A4"):
        stage_midtrain(_nano_cfg(midtrain_steps=5), model)
    with pytest.raises(ValueError, match="chat=True"):
        stage_sft(_nano_cfg(sft_steps=5), model, cast(Tokenizer, None))
