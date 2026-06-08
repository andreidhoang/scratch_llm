"""HardeningLevel — the verifier-hardening dial, the single source of truth.

L5 envs · the keystone leaf (``docs/IMPLEMENTATION_PLAN.md`` §3.1; ADR-0010). This module imports
**nothing**: it owns the one ``HardeningLevel`` enum so ``rewards/`` and ``envs/`` both import it
from here, which is what breaks the ``reward ↔ exploitability`` import cycle. Every other L5 module
depends on this leaf; this leaf depends on no other L5 module.

The dial treats verifier exploitability as a controlled independent variable (A5.2,
UNIFIED_FRONTIER_PROJECT_SPEC §3 L5): L1 is maximally gameable, L5 is oracle-grade.

Falsifiable prediction (VERA H2): ``hack_rate`` falls **monotonically** as the level rises.
Kill criterion: if no matcher ordering yields a monotone ``hack_rate``, the dial is not a valid
independent variable — report the null (CAPSTONE H2), do not bury it.
"""

from __future__ import annotations

from enum import IntEnum


class HardeningLevel(IntEnum):
    """How hard the verifier is to game, low → high. Exact L2–L4 matcher semantics are pinned in
    ADR-0010; the names and ordering are the load-bearing contract every L5 module imports."""

    L1_EXTENSIONAL = 1  # answer-set / label match — maximally gameable
    L2_FORMAT = 2  # requires a well-formed <answer>…</answer> + exact-string match
    L3_NORMALIZED = 3  # match after normalization (LaTeX boxed, whitespace, sign, trailing zero)
    L4_HARDENED = 4  # + consistency / distractor checks (unit, sign, range, majority)
    L5_PERTURBATION = 5  # isomorphic-perturbation hardened (oracle-grade matcher)


#: The 3-of-5 levels the v0.1.0 smoke run logs (min · middle · max span), per the Day-14 ship test.
SMOKE_LEVELS: tuple[HardeningLevel, HardeningLevel, HardeningLevel] = (
    HardeningLevel.L1_EXTENSIONAL,
    HardeningLevel.L3_NORMALIZED,
    HardeningLevel.L5_PERTURBATION,
)
