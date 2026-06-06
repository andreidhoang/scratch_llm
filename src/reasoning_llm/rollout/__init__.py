"""L2 — rollout/serving seam: the contract a policy is served behind, and the CPU backend.

``LocalBackend`` is the in-process reference engine now; a GPU SGLang backend slots in behind the
same ``RolloutClient`` Protocol later. See docs/design/L2_rollout_seam_SPEC.md.
"""

from reasoning_llm.rollout.local import LocalBackend
from reasoning_llm.rollout.types import Rollout, RolloutClient, StopReason

__all__ = ["LocalBackend", "Rollout", "RolloutClient", "StopReason"]
