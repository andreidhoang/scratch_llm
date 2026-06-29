"""Rollout seam — the backend-agnostic train↔infer comparison harness (A2.3)."""

from scratch_llm.rollout.local import LocalBackend
from scratch_llm.rollout.types import Rollout, RolloutClient

__all__ = ["LocalBackend", "Rollout", "RolloutClient"]
