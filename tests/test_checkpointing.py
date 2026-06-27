"""Activation checkpointing must be transparent to autograd and must reduce saved activations.

Both invariants run on CPU. (1) "full" and "selective" gradients equal eager ("none") gradients to
fp tolerance. (2) saved-activation bytes are ordered full < selective < none — the recompute-vs-store
accounting, the CPU surrogate for the GPU peak-memory win."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.utils.checkpointing import (
    count_op_executions,
    count_saved_activation_bytes,
    run_block,
)

_MODES = ("none", "full", "selective")


def _toy() -> tuple[TransformerLM, ModelConfig]:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=3, n_heads=4, context_length=64)
    return TransformerLM(cfg).double(), cfg


def _dense(block: torch.nn.Module, idx: int) -> Callable[[Tensor, Tensor], Tensor]:
    """A dense block as a single-tensor-in / single-tensor-out fn (drops the None MoE stats)."""

    def fn(x: Tensor, positions: Tensor) -> Tensor:
        out, _stats = block(x, positions, None, idx)
        return out

    return fn


def _run_stack(lm: TransformerLM, mode: str, x: Tensor, positions: Tensor) -> Tensor:
    for i, block in enumerate(lm.blocks):
        x = run_block(_dense(block, i), mode, x, positions)
    return x


def test_checkpoint_grads_match_eager() -> None:
    lm, _cfg = _toy()
    positions = torch.arange(8)
    base = torch.randn(2, 8, lm.cfg.d_model, dtype=torch.float64)

    grads: dict[str, dict[str, Tensor]] = {}
    for mode in _MODES:
        lm.zero_grad(set_to_none=True)
        x = base.clone().requires_grad_(True)
        _run_stack(lm, mode, x, positions).sum().backward()
        g = {n: p.grad.clone() for n, p in lm.named_parameters() if p.grad is not None}
        assert x.grad is not None
        g["__input__"] = x.grad.clone()
        grads[mode] = g

    for mode in ("full", "selective"):
        assert grads[mode].keys() == grads["none"].keys()
        for name, want in grads["none"].items():
            torch.testing.assert_close(grads[mode][name], want, atol=1e-10, rtol=1e-8)


def test_checkpointing_cuts_boundary_memory() -> None:
    # Memory axis: both checkpoint modes store far less across the fwd→bwd boundary than eager.
    lm, _cfg = _toy()
    positions = torch.arange(8)
    saved: dict[str, int] = {}
    for mode in _MODES:
        x = torch.randn(2, 8, lm.cfg.d_model, dtype=torch.float64, requires_grad=True)
        with count_saved_activation_bytes() as counter:
            _run_stack(lm, mode, x, positions)
        saved[mode] = counter["bytes"]

    assert saved["full"] < saved["none"], saved
    assert saved["selective"] < saved["none"], saved


def test_selective_avoids_matmul_recompute() -> None:
    # Compute axis: "full" re-runs the expensive matmuls in backward; "selective" saves them (so it
    # matches eager's matmul count), recomputing only the cheap ops. This is why selective is the
    # 2026 default — full's memory win without full's GEMM-recompute cost.
    lm, _cfg = _toy()
    positions = torch.arange(8)
    matmuls: dict[str, int] = {}
    total: dict[str, int] = {}
    for mode in _MODES:
        x = torch.randn(2, 8, lm.cfg.d_model, dtype=torch.float64, requires_grad=True)
        with count_op_executions() as counts:
            _run_stack(lm, mode, x, positions).sum().backward()
        matmuls[mode] = counts["watched"]
        total[mode] = counts["total"]

    assert matmuls["full"] > matmuls["none"], matmuls  # full recomputes GEMMs in backward
    assert matmuls["selective"] == matmuls["none"], matmuls  # selective saved them
    assert total["none"] < total["full"], total  # both checkpoint modes recompute cheap ops
    assert total["none"] < total["selective"], total
