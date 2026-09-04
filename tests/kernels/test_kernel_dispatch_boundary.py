"""Architecture-invariant tests for the kernels/ dispatch boundary.

These are NOT correctness tests of any kernel -- they are tests of the CODEBASE
STRUCTURE. A frontier kernel stack has a load-bearing rule: production code
(model.py, ops/, anything outside ``kernels/``) reaches a backend ONLY through its
family's ``dispatch.py``, never through the backend subpackage. That rule keeps
callers decoupled from a backend's file layout and kernel signature, so retuning an
autotune table, renaming a backend file, or adding a second backend lands entirely
inside ``kernels/``.

Hand-enforced, that rule silently rots (the next agent re-opens ``_fa2_fwd_kernel``
in ``ops/`` and nothing complains). These tests make it CI-checkable:

  * :func:`test_no_production_code_imports_kernel_backends` -- AST-scans every
    ``.py`` file under ``src/scratch_llm/`` outside ``kernels/`` and fails on any
    import of ``scratch_llm.kernels.<family>.<backend>...`` where ``<family>`` is a
    family with backends (attention / gemm / norm / reduce) and ``<backend>`` is not
    one of the sanctioned paths (``dispatch``, ``reference``, ``reference_naive``).
    ``kernels.common.*`` is allowed (shared primitives, below the backend layer).
    Tests and benches are NOT scanned -- they intentionally target one backend.

  * :func:`test_kernels_package_import_is_cpu_safe` -- asserts the second kernels/
    invariant in a fresh subprocess: ``import scratch_llm.kernels`` (and every
    family dispatch module) pulls NEITHER triton NOR ``torch.utils.cpp_extension``,
    so CI on a CPU box never compiles a GPU kernel. The dispatch lazy-loader (PEP
    562 ``__getattr__``) is what makes this true; this test catches a regression
    that would re-introduce an eager GPU import at the module level.

Both tests are CPU-safe, hermetic, and fast (AST parse + one subprocess import).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

# tests/kernels/test_*.py -> parents[2] is the repo root (two levels up, not one --
# kernel tests live one directory deeper than the flat tests/test_*.py layout).
SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "scratch_llm"

# Families that HAVE private backends behind a dispatch.py. Adding a new family
# with backends means adding it here so this test covers it.
_FAMILIES_WITH_BACKENDS = {"attention", "gemm", "norm", "reduce"}

# The sanctioned sub-paths a production caller may reach below
# `scratch_llm.kernels.<family>`: the dispatch seam, plus the CPU-safe oracle
# modules (reference / reference_naive). Everything else in a family is a backend.
_SANCTIONED_SUBMODULES = {"dispatch", "reference", "reference_naive"}


# -------------------------------------------------------------------------------------------------
# Helpers: resolve imports (incl. relative) to absolute dotted paths, AST-walk a file.
# -------------------------------------------------------------------------------------------------


def _importer_package(py_file: Path) -> str:
    """The dotted package of a ``.py`` file under ``src/scratch_llm/``.

    ``src/scratch_llm/model.py``       -> ``scratch_llm``
    ``src/scratch_llm/ops/attention.py`` -> ``scratch_llm.ops``
    ``src/scratch_llm/ops/__init__.py``  -> ``scratch_llm.ops`` (an __init__'s package is its dir)
    """
    parts = list(py_file.relative_to(SRC_ROOT).with_suffix("").parts)[:-1]
    return "scratch_llm." + ".".join(parts) if parts else "scratch_llm"


def _resolve_import(module: str | None, level: int, importer_pkg: str) -> str | None:
    """Resolve a (possibly relative) import to an absolute dotted path.

    ``level`` is the number of leading dots: ``from .foo import bar`` has level 1
    (current package); ``from ..foo import bar`` has level 2 (parent). ``level == 0``
    is an absolute import."""
    if level == 0:
        return module
    parts = importer_pkg.split(".")
    up = level - 1  # dots beyond the first go up
    base = ".".join(parts[: len(parts) - up]) if up <= len(parts) else ""
    if not module:
        return base or None
    return f"{base}.{module}" if base else module


def _imports_in(py_file: Path) -> Iterator[tuple[str, int]]:
    """Yield ``(absolute_module_path, lineno)`` for every import in ``py_file``."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    importer_pkg = _importer_package(py_file)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import(node.module, node.level, importer_pkg)
            if resolved:
                yield resolved, node.lineno
                # `from scratch_llm.kernels.attention import prefill` binds a SUBMODULE, not a
                # symbol — the dotted form never appears, so scanning `node.module` alone would
                # miss it and the boundary could be crossed by that spelling. Yield each alias
                # as a candidate submodule too; a plain symbol import just resolves to a path
                # that is not a family submodule and is ignored downstream.
                for alias in node.names:
                    if alias.name != "*":
                        yield f"{resolved}.{alias.name}", node.lineno


def _is_forbidden_backend_import(module_path: str) -> bool:
    """True iff ``module_path`` reaches a private kernel backend from outside.

    Structural rule: ``scratch_llm.kernels.<family>.<sub>...`` where ``<family>`` is
    a family WITH backends and ``<sub>`` is NOT a sanctioned path (dispatch /
    reference / reference_naive). ``scratch_llm.kernels.common.*`` is allowed -- it
    is shared primitives, below the backend layer."""
    parts = module_path.split(".")
    if len(parts) < 4:
        return False  # importing the family package itself, or a top-level path
    if parts[:2] != ["scratch_llm", "kernels"]:
        return False
    if parts[2] not in _FAMILIES_WITH_BACKENDS:
        return False  # common/ etc. -- not a family with private backends
    return parts[3] not in _SANCTIONED_SUBMODULES


def _production_py_files() -> Iterator[Path]:
    """Every ``.py`` under ``src/scratch_llm/`` EXCEPT the ``kernels/`` package.

    ``kernels/`` internals are allowed to import their own backends (the dispatch
    layer IS the sanctioned importer of backends). Only production callers are bound."""
    for py_file in SRC_ROOT.rglob("*.py"):
        rel_parts = py_file.relative_to(SRC_ROOT).parts
        if rel_parts and rel_parts[0] == "kernels":
            continue
        yield py_file


# -------------------------------------------------------------------------------------------------
# Invariant 1: the dispatch boundary (no production code reaches a backend directly).
# -------------------------------------------------------------------------------------------------


def test_no_production_code_imports_kernel_backends() -> None:
    """Production code must reach kernel backends only through a family's dispatch.py.

    A violation means a caller is re-coupled to a backend's file layout / kernel
    signature -- the exact coupling the dispatch layer exists to kill. Route the
    import through ``scratch_llm.kernels.<family>.dispatch`` instead."""
    scanned = list(_production_py_files())
    assert scanned, f"SRC_ROOT misresolved ({SRC_ROOT}) -- the AST scan is vacuous"
    violations: list[tuple[Path, int, str]] = []
    for py_file in scanned:
        for module_path, lineno in _imports_in(py_file):
            if _is_forbidden_backend_import(module_path):
                violations.append((py_file.relative_to(SRC_ROOT.parent), lineno, module_path))

    assert not violations, (
        "Dispatch-boundary violation -- production code imported a private kernel "
        "backend. Route through the family's dispatch.py instead:\n  "
        + "\n  ".join(f"{f}:{n} imports {m}" for f, n, m in violations)
    )


# -------------------------------------------------------------------------------------------------
# Invariant 2: CPU-safety (importing kernels/ never pulls Triton / nvcc JIT).
# -------------------------------------------------------------------------------------------------


def test_kernels_package_import_is_cpu_safe() -> None:
    """``import scratch_llm.kernels`` (+ every family dispatch) must not pull Triton
    or ``torch.utils.cpp_extension`` -- the invariant that keeps CPU CI green.

    Runs in a fresh subprocess so ``sys.modules`` is unpolluted by pytest's own
    imports. The dispatch PEP-562 lazy-loader is what makes this pass; an eager
    module-level GPU import anywhere in the kernels/ import chain would fail it."""
    probe = (
        "import sys; "
        "import scratch_llm.kernels; "
        "import scratch_llm.kernels.attention.dispatch; "
        "import scratch_llm.kernels.gemm.dispatch; "
        "import scratch_llm.kernels.norm.dispatch; "
        "import scratch_llm.kernels.reduce.dispatch; "
        "import scratch_llm.kernels.linear_attn.dispatch; "
        "import scratch_llm.kernels.common; "
        "forbidden = {'triton', 'triton.language', 'torch.utils.cpp_extension'}; "
        "leaked = forbidden & set(sys.modules); "
        "assert not leaked, f'kernels import pulled GPU deps: {sorted(leaked)}'; "
        "print('CPU-safe OK')"
    )
    # Resolve scratch_llm the way pytest does (pyproject pythonpath=["src"]) so the
    # probe also works where the package is not pip-installed into sys.executable.
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(SRC_ROOT.parent), *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
        ),
    }
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert result.returncode == 0, (
        "CPU-safety invariant broken -- `import scratch_llm.kernels` pulled a GPU "
        f"dependency.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "CPU-safe OK" in result.stdout
