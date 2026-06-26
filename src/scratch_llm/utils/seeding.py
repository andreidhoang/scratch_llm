"""Fixed-seed reproducibility (engineering discipline #3).

Seed every RNG a training/eval run touches so a re-run reproduces the metric. Without
this, no ablation is trustworthy — you can't tell a real effect from RNG drift.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
