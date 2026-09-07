"""One loader for every K1 rung's CUDA extension — AOT first, JIT fallback, arch-gated.

Each K1 rung is a separate ``.cu`` under ``csrc/gemm/`` compiled for exactly the architecture whose
ISA it uses. That per-rung-per-arch split is not tidiness: WGMMA does not assemble for ``sm_100a``
and tcgen05 does not assemble for ``sm_90a``, so a single "build everything for every arch" pass
cannot work, and the loader has to know which arch each rung wants.

Two ways the kernel gets here, in order:

1. **AOT** — ``csrc/`` built into the ``_scratch_llm_kernels`` extension by ``setup.py`` /
   ``CMakeLists.txt``. This is the path a rented box uses: one compile at bootstrap, none at
   measurement time. A JIT compile inside a timing window is a measurement of nvcc.
2. **JIT** — ``torch.utils.cpp_extension.load`` on first call, for iterating on one rung without
   rebuilding the whole extension.

Both compile the same source with the same flags, so a number from one is a number from the other.

While a rung's ``# HUY:`` hole is open its source carries ``#error "HUY: ..."`` and neither path can
compile it — deliberately. :func:`load_rung` detects that first and raises with the spec path
instead of letting nvcc's error scroll past, because "you have not written the kernel yet" and "your
kernel does not compile" deserve different messages.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

# `workspace_root` lives in scratch_llm._workspace: it is a workspace-path helper, not a kernel
# one, and production code outside kernels/ may not import a private backend to borrow it
# (tests/kernels/test_kernel_dispatch_boundary.py). Re-exported here so the callers that already
# reach for it — all of them tests, which the boundary rule does not scan — keep working.
from scratch_llm._workspace import workspace_root

__all__ = ["HoleOpenError", "hole_is_open", "load_rung", "workspace_root"]

# kernels/gemm/cuda/_k1_loader.py -> src/scratch_llm/kernels/gemm/cuda -> repo root is parents[5]
_REPO_ROOT = Path(__file__).resolve().parents[5]
_CSRC = _REPO_ROOT / "csrc"


#: ``-arch`` for the JIT compile. The trailing ``a`` selects the *accelerated* ISA and is
#: mandatory: base ``sm_90`` silently omits ``wgmma.*``, so dropping it yields a kernel that
#: compiles, launches, and computes nothing.
_ARCH_FLAG = {(9, 0): "sm_90a", (10, 0): "sm_100a", (12, 0): "sm_120a"}


class HoleOpenError(NotImplementedError):
    """The rung's kernel body is still an unfilled ``# HUY:`` hole.

    A subclass of ``NotImplementedError`` so a caller that means to tolerate an unwritten rung can,
    while ``pytest.raises(NotImplementedError)`` in a hole's guard test still reads naturally.
    """


def source_for(rung_file: str) -> Path:
    """Absolute path to a rung's ``.cu`` under ``csrc/gemm/``."""
    return _CSRC / "gemm" / rung_file


def hole_is_open(rung_file: str) -> bool:
    """True iff the rung's source still carries the ``#error "HUY:`` sentinel.

    A missing source counts as open — the rung has not been written, so its kernel certainly
    cannot run, and reporting "filled" for a path typo would be the worst possible answer.
    """
    p = source_for(rung_file)
    if not p.is_file():
        return True
    return '#error "HUY:' in p.read_text(encoding="utf-8", errors="replace")


@cache
def load_rung(name: str, rung_file: str, symbol: str, arch: tuple[int, int]) -> Any:
    """Return the extension module exposing ``symbol`` for one rung.

    ``name``      a unique JIT build name, e.g. ``"k1_h_r1"``.
    ``rung_file`` the source basename under ``csrc/gemm/``, e.g. ``"h_r1_wgmma_bf16_sm90.cu"``.
    ``symbol``    the bound host launcher, e.g. ``"h_r1_wgmma_bf16"``.
    ``arch``      the compute capability whose ISA the rung uses, e.g. ``(9, 0)``.

    Raises :class:`HoleOpenError` while the hole is open, ``KeyError`` for an unmapped arch.
    """
    if hole_is_open(rung_file):
        rung = name.replace("k1_", "K1/").replace("_", "-").upper().replace("K1/", "K1/")
        raise HoleOpenError(
            f"{rung_file} still has an open HUY hole — the kernel body is unwritten, so there is "
            f"nothing to compile. Fill it (spec: experiments/{rung}/spec.md), or compile the "
            f"scaffolding only with nvcc -DHUY_STUB_KERNEL_BODY=1. `make holes` lists every hole."
        )

    # 1. AOT — the extension the rented box built once, at bootstrap.
    try:
        # The AOT extension only exists after a `[gpu-aot]` build on a GPU box; there is no
        # stub for a static checker to see, which is what the ignore is for.
        from scratch_llm import (
            _scratch_llm_kernels as _ext,  # type: ignore[attr-defined]  # noqa: PLC0415
        )

        if hasattr(_ext, symbol):
            return _ext
    except ImportError:
        pass

    # 2. JIT — compile this one source for its arch.
    from torch.utils.cpp_extension import load  # noqa: PLC0415

    if arch not in _ARCH_FLAG:
        raise KeyError(f"no -arch flag mapped for compute capability {arch}; add it to _ARCH_FLAG")
    return load(
        name=name,
        sources=[str(source_for(rung_file))],
        extra_cuda_cflags=["-O3", "-lineinfo", f"-arch={_ARCH_FLAG[arch]}"],
        verbose=False,
    )
