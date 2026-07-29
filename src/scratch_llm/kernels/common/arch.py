"""GPU architecture detection + capability gating — the arch primitive.

A 2026 frontier kernel stack MUST know which ISA it is running on, because the
kernel body changes at every arch boundary:

  * sm_90  (Hopper, H100/H200)    -- WGMMA (warp-group MMA) + TMA (tensor-memory accel)
  * sm_100 (Blackwell datacenter) -- tcgen05 (gen-5 tensor cores, tensor-memory)
  * sm_120 (Blackwell client)     -- the RTX PRO 4000 in this box (tcgen05 ISA family)

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
    """sm_100+ (B100 / B200 / RTX PRO 4000 Blackwell) -- the tcgen05 ISA family."""
    cc = compute_capability()
    return cc is not None and cc[0] >= 10


def is_sm120() -> bool:
    """sm_120 -- the RTX PRO 4000 Blackwell (sm_120) in this development box."""
    cc = compute_capability()
    return cc is not None and cc == (12, 0)


def require_cc(major: int, minor: int = 0, *, fn_name: str = "this kernel") -> None:
    """Raise ``RuntimeError`` unless the current device is ``>= (major, minor)``.

    Use at the top of an arch-gated kernel entry point so a wrong-arch call fails
    loudly instead of silently dispatching to a slow or incorrect path::

        from scratch_llm.kernels.common.arch import require_cc
        require_cc(9, 0, fn_name="wgmma_gemm")  # the Hopper WGMMA path

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
