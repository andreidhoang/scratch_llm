"""``kernels/common/arch.py`` gating — the arch-routing rule, made executable.

Why this file exists: PLAN.md's arch-routing rule is that a rung must be gated by the KERNEL'S
TARGET ARCH, never by "the biggest card you could rent" — a bigger card is not a superset,
because ``-arch=sm_120`` PTX will not load on sm_90 (PTX compatibility is forward-only) and
sm_120 client Blackwell has no tcgen05/TMEM despite ``cc >= (10, 0)``. Violating it is what
left four sm_120 ``ncu``-debt metrics filed against "the H100 day" for 57 days.

The rule was previously documentation. These tests are the enforcement: the single most
valuable assertion here is that ``require_cc(9, 0)`` and ``require_cc(10, 0)`` both PASS on a
simulated sm_120 while ``require_arch`` blocks it — that difference is the whole bug class, and
it costs rented GPU-hours when it is wrong.

CPU-only: ``compute_capability`` is monkeypatched, so nothing here needs a GPU.
"""

from __future__ import annotations

import pytest

from scratch_llm.kernels.common import arch


def test_require_arch_on_cpu_names_the_requirement_and_the_absence() -> None:
    """No CUDA device: the error must name the target arch, not just "no GPU"."""
    with pytest.raises(RuntimeError, match=r"exactly sm_90.*no CUDA device"):
        arch.require_arch((9, 0), fn_name="cuda_wgmma bench")


def test_require_arch_renders_every_allowed_arch() -> None:
    with pytest.raises(RuntimeError, match=r"sm_100, sm_103"):
        arch.require_arch((10, 0), (10, 3), fn_name="x")


def test_floor_admits_a_newer_arch_but_exact_match_blocks_it(monkeypatch) -> None:
    """THE regression this module exists for.

    On sm_120, a forward-compatible floor passes for both Hopper (sm_90a WGMMA) and
    Blackwell-DC (sm_100a tcgen05) — neither of which sm_120 implements. ``require_arch``
    is the primitive that says no.
    """
    monkeypatch.setattr(arch, "compute_capability", lambda: (12, 0))
    monkeypatch.setattr(arch, "arch_name", lambda: "sm_120")

    # The floor is permissive by design — this is not a bug in require_cc, it is why
    # require_cc must not be used to gate an arch-exclusive ISA.
    arch.require_cc(9, 0, fn_name="wgmma")
    arch.require_cc(10, 0, fn_name="tcgen05")

    for allowed, want in [((9, 0), "sm_90"), ((10, 0), "sm_100")]:
        with pytest.raises(RuntimeError, match=rf"{want} EXACTLY.*sm_120"):
            arch.require_arch(allowed, fn_name="frontier rung")

    # ...and it must still let the kernel that genuinely targets this arch through.
    arch.require_arch((12, 0), fn_name="sm_120 rung")
