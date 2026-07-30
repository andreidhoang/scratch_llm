"""F6 MoE balancing ablation CLI.

Runs the arms × granularities matrix on a small standing-box-friendly model and
saves JSON + Markdown tables.  Default settings target CPU/sm120 and finish in
well under a minute.

Usage:
    python scripts/f6_moe_ablation.py
    python scripts/f6_moe_ablation.py --device cuda --max-steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow `python scripts/f6_moe_ablation.py` without an editable install.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scratch_llm.eval.moe_ablation import (
    AblationArm,
    AblationSpec,
    Granularity,
    ablation_table,
    run_moe_ablation,
    save_ablation_results,
)
from scratch_llm.model import ModelConfig
from scratch_llm.train import TrainConfig

DEFAULT_OUT_DIR = REPO / "artifacts" / "f6_moe_ablation"


def _synthetic_loader(vocab_size: int, batch_size: int, context_length: int, batches: int):
    """Deterministic repeated-pattern batches, tiled to always fill context_length."""
    pattern = np.arange(vocab_size, dtype=np.int64)
    tiled = np.tile(pattern, (context_length // vocab_size) + 2)
    rng = np.random.default_rng(0)
    for _ in range(batches):
        ids = np.zeros((batch_size, context_length), dtype=np.int64)
        for b in range(batch_size):
            start = rng.integers(0, vocab_size)
            ids[b] = tiled[int(start) : int(start) + context_length]
        yield ids, np.roll(ids, shift=-1, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="F6 MoE balancing ablation")
    parser.add_argument("--device", default="cpu", help="torch device (default cpu)")
    parser.add_argument("--max-steps", type=int, default=60, help="training steps per cell")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seq-aux-alpha", type=float, default=1e-3)
    parser.add_argument("--bias-update-speed", type=float, default=1e-3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for JSON/Markdown outputs",
    )
    args = parser.parse_args()

    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=4,
        n_kv_heads=2,
        context_length=args.context_length,
        moe=None,  # filled in per-arm by run_moe_ablation
    )
    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        max_lr=3e-3,
        grad_clip=1.0,
        seed=0,
        device=args.device,
    )

    # Pass loader factories so each matrix cell gets a fresh stream.
    def train_loader_factory():
        return _synthetic_loader(
            args.vocab_size,
            args.batch_size,
            args.context_length,
            batches=args.max_steps * 6,
        )

    def val_loader_factory():
        return _synthetic_loader(
            args.vocab_size,
            args.batch_size,
            args.context_length,
            batches=8,
        )

    spec = AblationSpec(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_loader=train_loader_factory,
        val_loader=val_loader_factory,
        arms=(AblationArm.BIAS_FREE, AblationArm.SEQ_AUX, AblationArm.NONE),
        granularities=(Granularity.COARSE, Granularity.FINE),
        d_model=args.d_model,
        seq_aux_alpha=args.seq_aux_alpha,
        bias_update_speed=args.bias_update_speed,
    )

    results = run_moe_ablation(spec)
    print(ablation_table(results))
    print()

    json_path = save_ablation_results(results, args.out_dir)
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
