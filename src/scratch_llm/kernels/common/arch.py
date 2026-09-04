"""GPU architecture detection + capability gating — the arch primitive.

A 2026 frontier kernel stack MUST know which ISA it is running on, because the
kernel body changes at every arch boundary:

  * sm_90  (Hopper, H100/H200)    -- WGMMA (warp-group MMA) + TMA (tensor-memory accel)
  * sm_100 (Blackwell datacenter) -- tcgen05 (gen-5 tensor cores, tensor-memory)
  * sm_120 (Blackwell client)     -- consumer Blackwell (RTX PRO 4000 / 5090 class):
                                     NO tcgen05, NO TMEM, NO cta_group

Routing a GEMM/attention/decode kernel to the wrong arch's ISA is a compile or
silently-wrong-result bug, so this module is the single source of truth for "what
device am I on". Two uses:

  1. Dispatch layers (``kernels/*/dispatch.py``) query it to pick the backend whose
     ISA matches the current GPU.
  2. Arch-gated kernel entry points call :func:`require_cc` at the top to fail
     loudly when run on the wrong arch (no silent fallback to a slow / wrong path).

CPU-safety (the ``kernels/`` package invariant): importing this module never
initializes CUDA. The detectors query the device lazily on first call and cache
the result; on a CPU box they return ``None`` / raise a clear ``RuntimeError``
(never a silent wrong-arch guess).
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def compute_capability() -> tuple[int, int] | None:
    """The ``(major, minor)`` compute capability of CUDA device 0, or ``None`` on CPU.

    Cached after first call. ``None`` (not a raise) when CUDA is unavailable so
    CPU-gated callers can fall through to a framework path without a try/except.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability(0))  # type: ignore[return-value]


@functools.lru_cache(maxsize=1)
def arch_name() -> str | None:
    """A human arch label (``"sm_90"``, ``"sm_120"``, ...) or ``None`` on CPU.

    Encodes the PyTorch ``(major, minor)`` tuple the way nvcc names ISAs:
    ``(9, 0) -> sm_90``, ``(12, 0) -> sm_120``. ``None`` on CPU (no arch)."""
    cc = compute_capability()
    if cc is None:
        return None
    return f"sm_{cc[0] * 10 + cc[1]}"


def is_hopper() -> bool:
    """sm_90 (H100 / H200) -- the WGMMA + TMA ISA generation."""
    cc = compute_capability()
    return cc is not None and cc == (9, 0)


def is_blackwell() -> bool:
    """``cc >= (10, 0)`` -- TRUE on sm_120 client Blackwell, which has NO tcgen05/TMEM.

    This predicate is deliberately over-broad. A tcgen05-gated call site MUST write
    ``is_blackwell() and not is_sm120()`` (see ``gemm/dispatch.py``); ``require_cc(10, 0)``
    is a lower bound and does NOT exclude sm_120.
    """
    cc = compute_capability()
    return cc is not None and cc[0] >= 10


def is_sm120() -> bool:
    """sm_120 -- Blackwell *client* (RTX PRO 4000, RTX 5090, ...).

    CAPABILITY ONLY, never a card identity: one sm_120 spans very different SM counts
    and bandwidths (70 SMs / 0.55 TB/s vs ~170 SMs / ~1.8 TB/s), so never reuse another
    sm_120's absolute peaks (CLAUDE.md: same compute capability != same card).
    """
    cc = compute_capability()
    return cc is not None and cc == (12, 0)


def require_cc(major: int, minor: int = 0, *, fn_name: str = "this kernel") -> None:
    """Raise ``RuntimeError`` unless the current device is ``>= (major, minor)``.

    This is a FORWARD-COMPATIBLE FLOOR, and only a floor. It is the right primitive for a
    genuine minimum (an ISA every later arch also implements) and the WRONG one for an
    arch-EXCLUSIVE ISA. ``sm_90a`` WGMMA and ``sm_100a`` tcgen05 are exclusive -- neither
    exists on client Blackwell -- yet ``(12, 0) >= (9, 0)`` and ``>= (10, 0)`` both hold, so
    a floor gate PASSES on sm_120 and the call proceeds to a JIT that cannot work.
    For those, use :func:`require_arch`, which matches exactly::

        require_cc(8, 0, fn_name="bf16 path")            # floor: Ampere and everything after
        require_arch((9, 0), fn_name="wgmma_gemm")       # exclusive: Hopper ONLY

    On CPU the error names both the requirement and the absence of a GPU.
    """
    cc = compute_capability()
    if cc is None:
        raise RuntimeError(
            f"{fn_name} requires CUDA with compute capability >= ({major}, {minor}); "
            "no CUDA device is available."
        )
    if cc < (major, minor):
        raise RuntimeError(
            f"{fn_name} requires sm_{major * 10 + minor}+ (current device is {arch_name()}); "
            "route through the family dispatch layer for a backend that matches this GPU."
        )


def require_arch(*allowed: tuple[int, int], fn_name: str = "this kernel") -> None:
    """Raise ``RuntimeError`` unless the current device's CC is EXACTLY one of ``allowed``.

    The counterpart to :func:`require_cc` for arch-exclusive ISAs. A bigger card is not a
    superset: ``-arch=sm_120`` PTX does not load or JIT on sm_90 (PTX compatibility is
    forward-only), and sm_120 client Blackwell has no tcgen05/TMEM despite ``cc >= (10, 0)``.
    Routing by "the biggest card available" instead of by the kernel's target arch is what
    left four sm_120 ``ncu``-debt metrics filed against "the H100 day" for 57 days
    (PLAN.md § arch-routing rule).

    The cost of getting this wrong is measured in rented GPU-hours, which is why it fails
    loudly and names both sides::

        require_arch((9, 0), fn_name="cuda_wgmma bench")             # Hopper only
        require_arch((10, 0), (10, 3), fn_name="cuda_tcgen05 bench")  # Blackwell-DC only
    """
    want = ", ".join(f"sm_{a * 10 + b}" for a, b in allowed)
    cc = compute_capability()
    if cc is None:
        raise RuntimeError(
            f"{fn_name} requires CUDA on exactly {want}; no CUDA device is available."
        )
    if cc not in allowed:
        raise RuntimeError(
            f"{fn_name} targets {want} EXACTLY (current device is {arch_name()}). "
            "A newer arch is not a superset — its ISA differs and a recompile on it is a "
            "different kernel instance, whose counters discharge nothing about this one. "
            "Route by the kernel's target arch, not by the largest card available."
        )
